//! One admitted vision encoder, replicated immutable spans, and content-keyed
//! reuse. CPU code handles metadata and input bytes; all model work is on GPU.
use crate::{checkpoint::Checkpoint, config::ModelConfig, control_channel::ControlChannel,
    cuda::{Cuda, Stream}, image_cache::{Descriptor, SpanCache}, layout::RankPlan,
    prompt_media::{Bindings, Identity}, roce::Roce, vision::{Geometry, ImageSpan, Vision}};
use anyhow::{Result, ensure};
use half::{bf16, slice::HalfBitsSliceExt};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::PathBuf, rc::Rc, sync::Arc, time::Instant};

/// All peers vote before entering the next GPU collective. Allocation refusal
/// must not strand receivers in a broadcast, while execution/identity errors
/// must never be reported as retryable capacity failures.
fn coordinate_input_stage<T>(local:Result<T>, required:usize, phase:&'static str,
    vote:&mut impl FnMut(bool)->Result<bool>)->Result<T> {
    let recoverable=local.as_ref().err().is_none_or(|e|
        e.downcast_ref::<crate::memory::InputCapacityError>().is_some());
    if !vote(recoverable)? {
        return match local {
            Err(e) if !recoverable=>Err(e),
            _=>Err(anyhow::anyhow!("peer failed during {phase}")),
        };
    }
    if !vote(local.is_ok())? {
        drop(local);
        return Err(crate::memory::InputCapacityError(phase,required).into());
    }
    local
}

#[cfg(feature="storage-fault-injection")]
struct ImageCapacityFault {
    rank:usize,
    phase:String,
    remaining:std::cell::Cell<usize>,
}
#[cfg(feature="storage-fault-injection")]
impl ImageCapacityFault {
    fn parse(value:&str)->Result<Self> {
        let parts=value.split(':').collect::<Vec<_>>();
        ensure!(parts.len()==3,"image capacity fault must be rank:patches|span:count");
        let rank=parts[0].parse()?;let count=parts[2].parse()?;
        ensure!(rank<4 && ["patches","span"].contains(&parts[1]) && (1..=2).contains(&count),
            "bounded image capacity fault");
        Ok(Self{rank,phase:parts[1].into(),remaining:std::cell::Cell::new(count)})
    }
    fn check(&self,rank:usize,phase:&str,required:usize)->Result<()> {
        if self.rank==rank && self.phase==phase && self.remaining.get()>0 {
            self.remaining.set(self.remaining.get()-1);
            return Err(crate::memory::InputCapacityError("injected image allocation refusal",required).into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub checkpoint: PathBuf,
    pub stage: PathBuf,
    pub kernel: PathBuf,
    pub attention: PathBuf,
    pub weights_sha256: String,
    pub cache_bytes: usize,
}
impl Config {
    pub(crate) fn fingerprint(&self) -> Result<[u8;32]> {
        ensure!(self.weights_sha256.len()==64 && self.weights_sha256.bytes().all(|b|b.is_ascii_hexdigit()),
            "vision weight identity");
        let mut hash=Sha256::new();
        hash.update(b"canada-v41-vision-bf16-fp32-reduction-v1\0");
        hash.update(self.weights_sha256.as_bytes());
        for path in [&self.checkpoint,&self.kernel,&self.attention.join("manifest.json"),
            &self.attention.join("vision_full_attention.cubin")] {
            hash.update(Sha256::digest(std::fs::read(path)?));
        }
        Ok(hash.finalize().into())
    }
}

/// Prepared input bytes may cross HTTP threads. They never contain model
/// outputs. The identity includes the encoded source and input-processor ABI.
pub struct PatchInput {
    pub identity: [u8;32],
    pub geometry: Geometry,
    pub patches: Vec<u16>,
}
/// Trusted finite-run input. Serving never accepts filesystem paths from a
/// request; this fixture interface also verifies the complete patch-file hash.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Fixture {
    pub descriptor: Descriptor,
    pub patches: PathBuf,
    pub sha256: String,
}
impl Fixture {
    pub fn load(&self) -> Result<Arc<PatchInput>> {
        let bytes=self.descriptor.geometry.validate()?.rows()*588*2;
        ensure!(std::fs::metadata(&self.patches)?.len()==bytes as u64,"image fixture file extent");
        let raw=std::fs::read(&self.patches)?;
        ensure!(format!("{:x}",Sha256::digest(&raw))==self.sha256,"image fixture bytes differ");
        let input=PatchInput {identity:self.descriptor.input,geometry:self.descriptor.geometry,
            patches:raw.chunks_exact(2).map(|b|u16::from_le_bytes(b.try_into().unwrap())).collect()};
        input.validate()?;Ok(Arc::new(input))
    }
}
impl PatchInput {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.patches.len()==self.geometry.validate()?.rows()*588,
            "prepared image patch extent");
        Ok(())
    }
    pub fn bytes(&self) -> usize { self.patches.capacity()*2 }
}

pub(crate) struct Media {
    cuda: Cuda,
    rank: usize,
    encoder_rank: usize,
    encoder_key: [u8;32],
    encoder: Option<Vision>,
    compact_workspace:bool,
    cache: SpanCache,
    inputs: HashMap<[u8;32],Arc<PatchInput>>,
    encoded: u64,
    skipped: u64,
    transmitted: u64,
    #[cfg(feature="storage-fault-injection")]
    capacity_fault:Option<ImageCapacityFault>,
}
impl Media {
    pub(crate) fn load(cuda:&Cuda, rank:usize, model:&ModelConfig, config:&Config) -> Result<Self> {
        // This placement flag is included in the worker group's agreed
        // CANADA_SCRATCH identity. The atlas and arithmetic stay canonical;
        // rank zero remains the HTTP/input owner.
        let encoder_rank=encoder_rank(std::env::var("CANADA_SCRATCH_VISION_ENCODER_RANK").ok().as_deref())?;
        Self::load_at(cuda,rank,model,config,encoder_rank)
    }
    fn load_at(cuda:&Cuda,rank:usize,model:&ModelConfig,config:&Config,encoder_rank:usize)->Result<Self> {
        ensure!(rank<4 && encoder_rank<4,"vision physical placement extent");
        let encoder_key=config.fingerprint()?;
        let encoder=if rank==encoder_rank {
            let cp=Checkpoint::from_manifest(&config.checkpoint)?;
            // The vision-only atlas is the original logical rank-zero atlas.
            // Its tensors are replicated, independently of physical placement.
            let plan=RankPlan::build(&cp,model,0,4)?;
            let encoder=Vision::new_staged(cuda,&std::fs::read(&config.kernel)?,&config.attention,
                &cp,&plan,&config.stage)?;
            ensure!(encoder.source_sha256()==config.weights_sha256,"loaded vision identity differs");
            Some(encoder)
        }else{None};
        Ok(Self {cuda:cuda.clone(),rank,encoder_rank,encoder_key,encoder,compact_workspace:crate::vision::compact_workspace()?,cache:SpanCache::new(config.cache_bytes)?,
            inputs:HashMap::new(),encoded:0,skipped:0,transmitted:0,
            #[cfg(feature="storage-fault-injection")]
            capacity_fault:std::env::var("CANADA_SCRATCH_IMAGE_CAPACITY_FAULT").ok()
                .map(|v|ImageCapacityFault::parse(&v)).transpose()?,
        })
    }
    pub(crate) fn identity(&self, images:&[Descriptor], tokens:&[u32]) -> Result<Identity> {
        Descriptor::identity(images,self.encoder_key,tokens)
    }
    pub(crate) fn set_inputs(&mut self, inputs:&[Arc<PatchInput>]) -> Result<()> {
        ensure!(self.rank==0,"only the ingress rank receives host patch input");
        self.inputs.clear();
        for input in inputs {
            input.validate()?;
            if let Some(previous)=self.inputs.insert(input.identity,input.clone()) {
                ensure!(previous.geometry==input.geometry,"one image identity has two geometries");
            }
        }
        Ok(())
    }
    pub(crate) fn prepare(&mut self, identity:Identity, images:&[Descriptor], frontier:usize,
        control:&mut ControlChannel, comm:&Roce, stream:&Stream,
        reclaim:&mut impl FnMut(&mut ControlChannel,usize)->Result<Value>) -> Result<(Rc<Bindings>,Value)> {
        ensure!(comm.rank()==self.rank && identity.placements().len()==images.len(),"image worker identity");
        let began=Instant::now();let before=(self.encoded,self.skipped,self.transmitted);
        let mut spans=Vec::with_capacity(images.len());
        let mut hashes=Vec::new();
        let mut reclamations=Vec::new();
        for (image,placement) in images.iter().zip(identity.placements()) {
            ensure!(image.key(self.encoder_key)==placement.key && image.start==placement.start
                && image.geometry.validate()?.tokens()==placement.tokens,"image binding identity differs");
            if image.start+image.geometry.tokens()<=frontier {
                self.skipped+=1;spans.push(None);continue;
            }
            let key=placement.key;
            let hit=self.cache.get(key);
            // Cache ownership and eviction are driven by the same ordered
            // commands on all ranks. Detect a lifecycle divergence before any
            // participant can enter a different NCCL collective sequence.
            control.agree(&[u8::from(hit.is_some())])?;
            let span=if let Some(span)=hit {
                ensure!(span.geometry==image.geometry,"cached image geometry differs");
                span
            }else{
                let g=image.geometry;
                let patch_bytes=g.rows()*588*2;
                let needed=if self.rank==self.encoder_rank {
                    g.workspace_bytes(true,self.compact_workspace)?
                }else{g.tokens()*5120*2+(16<<20)
                    +if self.encoder_rank!=0{patch_bytes.next_multiple_of(2<<20)}else{0}};
                let base_required=self.cuda.allocation_reserve_bytes()+needed;
                let mut attempt=0;
                let span=loop {
                    // One bounded retry can reclaim optional completed caches;
                    // live/paused state and the physical reserve remain intact.
                    // workspace_bytes already includes the transient driver
                    // allowance. Do not double that allowance on retry: doing
                    // so needlessly evicts valid prefixes and can refuse an
                    // allocation that just succeeded before a local refusal.
                    let required=base_required;
                    let enough=|cuda:&Cuda|->Result<bool>{
                        Ok(cuda.admission_memory()?.admitted_available > required)
                    };
                    if !control.all_admit(enough(&self.cuda)?)? || attempt>0 {
                        stream.synchronize()?;self.cache.release_warm();
                        if !control.all_admit(enough(&self.cuda)?)? {
                            reclamations.push(reclaim(control,required)?);
                        }
                        if !control.all_admit(enough(&self.cuda)?)? {
                            eprintln!("image admission refused: {}",json!({"rank":self.rank,
                                "needed_bytes":needed,"required_available_bytes":required,"reclamations":reclamations,
                                "memory":self.cuda.admission_memory()?}));
                            return Err(crate::memory::InputCapacityError(
                                "image encoding cannot retain the UMA reserve",required).into());
                        }
                    }
                    match self.encode_attempt(image,required,control,comm,stream) {
                        Ok(span)=>break span,
                        Err(error) if attempt==0 && error.downcast_ref::<crate::memory::InputCapacityError>().is_some()=>{
                            eprintln!("image allocation retry: {}",json!({"rank":self.rank,
                                "geometry":g,"reason":error.to_string(),"memory":self.cuda.admission_memory()?}));
                            attempt+=1;
                        },
                        Err(error)=>return Err(error),
                    }
                };
                // No participant exposes this allocation before completion.
                // The root is read-only; all other ranks own fresh destinations.
                unsafe {comm.broadcast_image_span(&span.data,self.encoder_rank,stream)?;}
                stream.synchronize()?;comm.health()?;
                // Keep the original ingress receipt meaningful when another
                // physical rank performs the encoding.
                self.encoded+=1;
                self.transmitted+=1;
                let span=Rc::new(span);self.cache.insert(key,span.clone())?;span
            };
            if std::env::var("CANADA_SCRATCH_IMAGE_SPAN_HASH").as_deref()==Ok("1") {
                stream.synchronize()?;
                let bytes=span.data.download()?;
                let raw=unsafe {std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(),bytes.len()*2)};
                hashes.push(json!({"start":image.start,"sha256":format!("{:x}",Sha256::digest(raw))}));
            }
            spans.push(Some(span));
        }
        self.inputs.clear();
        let bindings=Rc::new(Bindings::suffix(&self.cuda,identity,spans,frontier)?);
        let receipt=json!({"seconds":began.elapsed().as_secs_f64(),"encoder_rank":self.encoder_rank,"encoded_images":self.encoded-before.0,
            "prefix_images_skipped":self.skipped-before.1,"broadcast_images":self.transmitted-before.2,
            "cache":self.cache.stats(),"live_request_bytes":bindings.owned_bytes(),
            "cache_reclamations":reclamations,"diagnostic_span_hashes":hashes});
        Ok((bindings,receipt))
    }
    fn encode_attempt(&self,image:&Descriptor,required:usize,control:&mut ControlChannel,
        comm:&Roce,stream:&Stream)->Result<ImageSpan> {
        let g=image.geometry;
        let patches=(||->Result<_>{
            let patches=if self.encoder_rank!=0 || self.rank==0 {
                let mut patches=self.cuda.allocate_input::<bf16>(g.rows()*588)?;
                if self.rank==0 {
                    let input=self.inputs.get(&image.input)
                        .ok_or_else(||anyhow::anyhow!("missing image input on the ingress rank"))?;
                    ensure!(input.geometry==g,"request image input geometry differs");
                    patches.upload(input.patches.reinterpret_cast())?;
                }
                stream.join_default()?;
                Some(patches)
            }else{None};
            #[cfg(feature="storage-fault-injection")]
            if let Some(fault)=&self.capacity_fault {fault.check(self.rank,"patches",required)?;}
            ensure!(self.cuda.admission_memory()?.admitted_available>self.cuda.allocation_reserve_bytes(),
                crate::memory::InputCapacityError("image patch allocation consumed the UMA reserve",required));
            Ok(patches)
        })();
        let patches=coordinate_input_stage(patches,required,"image patch allocation unavailable",
            &mut |v|control.all_admit(v))?;
        if self.encoder_rank!=0 {
            unsafe {comm.broadcast_image_patches(patches.as_ref().unwrap(),0,stream)?;}
        }
        // Complete transport before any rank can free input after a refusal.
        stream.synchronize()?;comm.health()?;
        let span=(||->Result<_>{
            let span=if self.rank==self.encoder_rank {
                self.encoder.as_ref().unwrap().encode_with_workspace(patches.as_ref().unwrap(),g,stream,self.compact_workspace)?
            }else{ImageSpan {data:self.cuda.allocate_input(g.tokens()*5120)?,geometry:g}};
            #[cfg(feature="storage-fault-injection")]
            if let Some(fault)=&self.capacity_fault {fault.check(self.rank,"span",required)?;}
            ensure!(self.cuda.admission_memory()?.admitted_available>self.cuda.allocation_reserve_bytes(),
                crate::memory::InputCapacityError("image span allocation consumed the UMA reserve",required));
            Ok(span)
        })();
        coordinate_input_stage(span,required,"image encoding allocation unavailable",
            &mut |v|control.all_admit(v))
    }
    pub(crate) fn release_warm(&mut self) { self.cache.release_warm(); }
    pub(crate) fn clear_inputs(&mut self) { self.inputs.clear(); }
}

fn encoder_rank(value:Option<&str>)->Result<usize> {
    let rank=value.unwrap_or("0").parse::<usize>()?;
    ensure!(rank<4,"vision encoder rank must be in0..4");
    Ok(rank)
}

/// Finite four-device placement check. Uses real canonical patch fixtures and
/// the same Media preparation/cache/transport path as serving, without loading
/// the language model. Every peer compares complete final span hashes.
pub fn placement_probe(path:&std::path::Path)->Result<Value> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Probe {
        rank:usize, endpoint:std::net::SocketAddr, rank_ips:[std::net::IpAddr;4],
        ticket:PathBuf, config:PathBuf, vision:Config, primitives:PathBuf,
        fixtures:Vec<Fixture>, output:PathBuf,
        #[serde(default)] capacity_checks:bool,
        #[serde(default)] workspace_compare:bool,
    }
    let p:Probe=serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(!p.capacity_checks || cfg!(feature="storage-fault-injection"),"capacity probe requires diagnostic build");
    ensure!(p.rank<4 && p.fixtures.len()==2 && !p.output.exists()
        && std::env::var("CANADA_SCRATCH_IMAGE_SPAN_HASH").as_deref()==Ok("1"),"finite vision placement admission");
    let model=ModelConfig::parse(&std::fs::read(&p.config)?)?;
    let nonce=std::fs::read(&p.ticket)?;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let identity=p.vision.fingerprint()?;
    let mut control=ControlChannel::connect(p.rank,p.endpoint,p.rank_ips,&nonce,&identity,std::time::Duration::from_secs(120))?;
    let mut address=p.endpoint;address.set_port(address.port().checked_add(1).ok_or_else(||anyhow::anyhow!("probe port overflow"))?);
    let mut comm=Roce::local(&cuda,&p.primitives.join("roce-v0"),p.rank)?;
    let blobs=crate::rendezvous::exchange_blob(p.rank,address,&nonce,&comm.blob()?)?;comm.connect(&blobs)?;
    address.set_port(address.port().checked_add(1).ok_or_else(||anyhow::anyhow!("probe port overflow"))?);
    let nccl=crate::collective::Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&nccl,p.rank,address,&nonce)?;
    comm.attach_prefill_sum(nccl.join(&cuda,id,p.rank,4)?)?;
    let mut reference=Vec::new();let mut records=Vec::new();
    let mut capacity_records:Vec<Value>=Vec::new();
    let compact=crate::vision::compact_workspace()?;
    let placements=if p.workspace_compare {vec![(0,false),(3,false),(3,true)]}else{vec![(0,compact),(3,compact)]};
    let mut workspace_records=Vec::new();
    for (location,compact) in placements {
        let before=cuda.admission_memory()?;
        let mut media=Media::load_at(&cuda,p.rank,&model,&p.vision,location)?;
        media.compact_workspace=compact;
        let after=cuda.admission_memory()?;
        for (step,case,skip) in [(0,0,false),(1,1,false),(2,0,false),(3,1,true)] {
            let f=&p.fixtures[case];
            let descriptor=Descriptor{start:1,..f.descriptor.clone()};
            let mut tokens=vec![crate::prompt_media::IMAGE_TOKEN;descriptor.geometry.tokens()+2];
            tokens[0]=2;*tokens.last_mut().unwrap()=2;
            if p.rank==0 {media.set_inputs(&[f.load()?])?;}
            let descriptors=[descriptor];let identity=media.identity(&descriptors,&tokens)?;
            let started=Instant::now();
            let (bindings,receipt)=media.prepare(identity,&descriptors,if skip{tokens.len()}else{0},
                &mut control,&comm,&stream,&mut |_,_|anyhow::bail!("placement probe unexpectedly needed reclamation"))?;
            let hashes=&receipt["diagnostic_span_hashes"];
            ensure!(hashes.as_array().unwrap().len()==usize::from(!skip),"missing diagnostic span hash");
            if location==0 {reference.push(hashes.clone());}
            else {ensure!(*hashes==reference[step],"vision placement changed span bits at step{step}");}
            control.agree(&serde_json::to_vec(hashes)?)?;
            ensure!(receipt["encoded_images"]==if step<2{1}else{0}
                && receipt["prefix_images_skipped"]==usize::from(skip),"vision placement cache/skip contract");
            records.push(json!({"encoder_rank":location,"compact_workspace":compact,"step":step,"case":case,"seconds":started.elapsed().as_secs_f64(),
                "receipt":receipt,"load_memory_before":before,"load_memory_after":after,
                "encoder_weight_bytes":media.encoder.as_ref().map_or(0,Vision::weight_bytes),"all_peer_span_hashes_equal":true}));
            drop(bindings);
        }
        #[cfg(feature="storage-fault-injection")]
        if p.capacity_checks && location==3 && (!p.workspace_compare || compact) {
            for fault_spec in ["3:span:1","3:span:2","1:patches:1","1:span:2"] {
                media.release_warm();media.capacity_fault=Some(ImageCapacityFault::parse(fault_spec)?);
                let f=&p.fixtures[0];let descriptor=Descriptor{start:1,..f.descriptor.clone()};
                let mut tokens=vec![crate::prompt_media::IMAGE_TOKEN;descriptor.geometry.tokens()+2];
                tokens[0]=2;*tokens.last_mut().unwrap()=2;
                let descriptors=[descriptor];let identity=media.identity(&descriptors,&tokens)?;
                if p.rank==0 {media.set_inputs(&[f.load()?])?;}
                let result=media.prepare(identity.clone(),&descriptors,0,&mut control,&comm,&stream,
                    &mut |_,_|anyhow::bail!("finite capacity probe unexpectedly needs cache reclamation"));
                let refused=fault_spec.ends_with(":2");
                if refused {
                    let error=result.err().ok_or_else(||anyhow::anyhow!("persistent refusal unexpectedly succeeded"))?;
                    ensure!(error.downcast_ref::<crate::memory::InputCapacityError>().is_some(),"refusal became fatal");
                }else{
                    let (_,receipt)=result?;
                    ensure!(receipt["diagnostic_span_hashes"]==reference[0],"retried image changed canonical span");
                }
                control.agree(fault_spec.as_bytes())?;
                media.capacity_fault=None;media.release_warm();
                if p.rank==0 {media.set_inputs(&[f.load()?])?;}
                let (_,receipt)=media.prepare(identity,&descriptors,0,&mut control,&comm,&stream,
                    &mut |_,_|anyhow::bail!("recovery probe unexpectedly needs cache reclamation"))?;
                ensure!(receipt["diagnostic_span_hashes"]==reference[0],"post-refusal recovery changed span");
                control.agree(&serde_json::to_vec(&receipt["diagnostic_span_hashes"])?)?;
                capacity_records.push(json!({"fault":fault_spec,"refused":refused,"recovery_exact":true}));
            }
        }
        if p.workspace_compare && location==3 && compact {
            // Synthetic BF16 input patterns use public fixture bytes, not CPU
            // model arithmetic. Compare complete canonical GPU spans across
            // real incident shapes, tiny grids, and legal aspect-ratio limits.
            let input=p.fixtures[0].load()?;
            for g in [Geometry{height:1,width:1},Geometry{height:1,width:3063},
                Geometry{height:1533,width:1},Geometry{height:3,width:3063},
                Geometry{height:58,width:86},Geometry{height:65,width:103},Geometry{height:90,width:90}] {
                g.validate()?;let mut reference=None;let mut timings=Vec::new();
                let id: [u8;32]=Sha256::digest(format!("vision-workspace-{g:?}").as_bytes()).into();
                let patches=Arc::new(PatchInput{identity:id,geometry:g,
                    patches:input.patches.iter().copied().cycle().take(g.rows()*588).collect()});
                for compact in [false,true] {
                    media.compact_workspace=compact;media.release_warm();
                    let descriptors=[Descriptor{start:1,geometry:g,input:id}];
                    let mut tokens=vec![crate::prompt_media::IMAGE_TOKEN;g.tokens()+2];
                    tokens[0]=2;*tokens.last_mut().unwrap()=2;
                    if p.rank==0 {media.set_inputs(&[patches.clone()])?;}
                    let identity=media.identity(&descriptors,&tokens)?;let began=Instant::now();
                    let (_,receipt)=media.prepare(identity,&descriptors,0,&mut control,&comm,&stream,
                        &mut |_,_|anyhow::bail!("workspace comparison unexpectedly needs reclamation"))?;
                    let hashes=receipt["diagnostic_span_hashes"].clone();
                    control.agree(&serde_json::to_vec(&hashes)?)?;
                    if compact {ensure!(Some(hashes)==reference,"compact workspace changed image span at{g:?}");}
                    else {reference=Some(hashes);}
                    timings.push(began.elapsed().as_secs_f64());
                }
                workspace_records.push(json!({"geometry":g,"exact":true,"seconds":timings,
                    "admission_before":g.workspace_bytes(true,false)?,"admission_after":g.workspace_bytes(true,true)?}));
            }
        }
        stream.synchronize()?;control.agree(&[location as u8])?;drop(media);
    }
    comm.health()?;stream.synchronize()?;
    let report=json!({"rank":p.rank,"records":records,"placement_exact":true,"capacity_checks":capacity_records,"workspace_checks":workspace_records,
        "scope":"Two canonical patch fixtures encoded on ingress0 and peer3, byte-exact completed spans on all4devices; repeated image cache hit and retained-prefix skip. Same BF16 weights/inputs, FP32 GEMM reduction and native vision kernels. No language-model or serving throughput claim."});
    std::fs::write(p.output,serde_json::to_vec_pretty(&report)?)?;Ok(report)
}

#[cfg(test)]
mod placement_tests {
    use super::*;
    #[test]
    fn encoder_placement_is_bounded_and_defaults_to_ingress() {
        assert_eq!(encoder_rank(None).unwrap(),0);
        for rank in 0..4 {assert_eq!(encoder_rank(Some(&rank.to_string())).unwrap(),rank);}
        for bad in ["", "4", "-1", "3.0", "peer"] {assert!(encoder_rank(Some(bad)).is_err());}
    }
    #[test]
    fn capacity_and_fatal_failures_are_agreed_before_gpu_transport() {
        use crate::memory::InputCapacityError;
        let mut votes=Vec::new();
        let value=coordinate_input_stage(Ok(7),19,"patch",&mut |v|{votes.push(v);Ok(v)}).unwrap();
        assert_eq!(value,7);assert_eq!(votes,vec![true,true]);
        // Local capacity: every rank stays usable, even when only one fails.
        let mut votes=Vec::new();
        let e=coordinate_input_stage::<()>(Err(InputCapacityError("local",19).into()),19,"patch",
            &mut |v|{votes.push(v);Ok(v)}).unwrap_err();
        assert!(e.downcast_ref::<InputCapacityError>().is_some());assert_eq!(votes,vec![true,false]);
        // Peer capacity releases this rank's allocation before returning.
        let owner=Rc::new(());let weak=Rc::downgrade(&owner);let mut call=0;
        let e=coordinate_input_stage(Ok(owner),19,"span",&mut |_|{call+=1;Ok(call==1)}).unwrap_err();
        assert!(weak.upgrade().is_none());assert!(e.downcast_ref::<InputCapacityError>().is_some());
        for local in [Ok(()),Err(anyhow::anyhow!("execution failure"))] {
            let mut calls=0;
            let e=coordinate_input_stage(local,19,"span",&mut |_|{calls+=1;Ok(false)}).unwrap_err();
            assert!(e.downcast_ref::<InputCapacityError>().is_none());assert_eq!(calls,1);
        }
    }
}
