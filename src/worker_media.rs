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
    cache: SpanCache,
    inputs: HashMap<[u8;32],Arc<PatchInput>>,
    encoded: u64,
    skipped: u64,
    transmitted: u64,
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
        Ok(Self {cuda:cuda.clone(),rank,encoder_rank,encoder_key,encoder,cache:SpanCache::new(config.cache_bytes)?,
            inputs:HashMap::new(),encoded:0,skipped:0,transmitted:0})
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
                    g.workspace_bytes(true)?
                }else{g.tokens()*5120*2+(16<<20)
                    +if self.encoder_rank!=0{patch_bytes.next_multiple_of(2<<20)}else{0}};
                let required=self.cuda.allocation_reserve_bytes()+needed;
                let enough=|cuda:&Cuda|->Result<bool>{
                    Ok(cuda.admission_memory()?.admitted_available > required)
                };
                if !control.all_admit(enough(&self.cuda)?)? {
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
                let patches=if self.encoder_rank!=0 || self.rank==0 {
                    let mut patches=self.cuda.allocate::<bf16>(g.rows()*588)?;
                    if self.rank==0 {
                        let input=self.inputs.get(&image.input)
                            .ok_or_else(||anyhow::anyhow!("missing image input on the ingress rank"))?;
                        ensure!(input.geometry==g,"request image input geometry differs");
                        patches.upload(input.patches.reinterpret_cast())?;
                    }
                    stream.join_default()?;
                    if self.encoder_rank!=0 {
                        // Broadcast raw BF16 patch bits, never model arithmetic.
                        // Each rank follows the same ordered NCCL sequence.
                        unsafe {comm.broadcast_image_patches(&patches,0,stream)?;}
                    }
                    Some(patches)
                }else{None};
                let span=if self.rank==self.encoder_rank {
                    self.encoder.as_ref().unwrap().encode(patches.as_ref().unwrap(),g,stream)?
                }else{ImageSpan {data:self.cuda.allocate(g.tokens()*5120)?,geometry:g}};
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
    }
    let p:Probe=serde_json::from_slice(&std::fs::read(path)?)?;
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
    for location in [0,3] {
        let before=cuda.admission_memory()?;
        let mut media=Media::load_at(&cuda,p.rank,&model,&p.vision,location)?;
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
            records.push(json!({"encoder_rank":location,"step":step,"case":case,"seconds":started.elapsed().as_secs_f64(),
                "receipt":receipt,"load_memory_before":before,"load_memory_after":after,
                "encoder_weight_bytes":media.encoder.as_ref().map_or(0,Vision::weight_bytes),"all_peer_span_hashes_equal":true}));
            drop(bindings);
        }
        stream.synchronize()?;control.agree(&[location as u8])?;drop(media);
    }
    comm.health()?;stream.synchronize()?;
    let report=json!({"rank":p.rank,"records":records,"placement_exact":true,
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
}
