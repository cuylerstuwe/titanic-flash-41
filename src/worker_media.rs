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
        let encoder_key=config.fingerprint()?;
        let encoder=if rank==0 {
            let cp=Checkpoint::from_manifest(&config.checkpoint)?;
            let plan=RankPlan::build(&cp,model,0,4)?;
            let encoder=Vision::new_staged(cuda,&std::fs::read(&config.kernel)?,&config.attention,
                &cp,&plan,&config.stage)?;
            ensure!(encoder.source_sha256()==config.weights_sha256,"loaded vision identity differs");
            Some(encoder)
        }else{None};
        Ok(Self {cuda:cuda.clone(),rank,encoder_key,encoder,cache:SpanCache::new(config.cache_bytes)?,
            inputs:HashMap::new(),encoded:0,skipped:0,transmitted:0})
    }
    pub(crate) fn identity(&self, images:&[Descriptor], tokens:&[u32]) -> Result<Identity> {
        Descriptor::identity(images,self.encoder_key,tokens)
    }
    pub(crate) fn set_inputs(&mut self, inputs:&[Arc<PatchInput>]) -> Result<()> {
        ensure!(self.rank==0,"only the encoding rank receives patch input");
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
                let needed=if self.rank==0 {
                    g.workspace_bytes(true)?
                }else{g.tokens()*5120*2+(16<<20)};
                let enough=|cuda:&Cuda|->Result<bool>{
                    Ok(cuda.admission_memory()?.admitted_available > (8usize<<30)+needed)
                };
                if !control.all_admit(enough(&self.cuda)?)? {
                    stream.synchronize()?;self.cache.release_warm();
                    if !control.all_admit(enough(&self.cuda)?)? {
                        reclamations.push(reclaim(control,(8usize<<30)+needed)?);
                    }
                    if !control.all_admit(enough(&self.cuda)?)? {
                        eprintln!("image admission refused: {}",json!({"rank":self.rank,
                            "needed_bytes":needed,"reclamations":reclamations,
                            "memory":self.cuda.admission_memory()?}));
                        return Err(crate::memory::InputCapacityError(
                            "image encoding cannot retain the UMA reserve",(8usize<<30)+needed).into());
                    }
                }
                let span=if self.rank==0 {
                    let input=self.inputs.get(&image.input)
                        .ok_or_else(||anyhow::anyhow!("missing image input on the encoding rank"))?;
                    ensure!(input.geometry==g,"request image input geometry differs");
                    let mut patches=self.cuda.allocate::<bf16>(input.patches.len())?;
                    patches.upload(input.patches.reinterpret_cast())?;
                    stream.join_default()?;
                    let span=self.encoder.as_ref().unwrap().encode(&patches,g,stream)?;
                    self.encoded+=1;span
                }else{ImageSpan {data:self.cuda.allocate(g.tokens()*5120)?,geometry:g}};
                // No participant exposes this allocation before completion.
                // The root is read-only; all other ranks own fresh destinations.
                unsafe {comm.broadcast_image_span(&span.data,0,stream)?;}
                stream.synchronize()?;comm.health()?;
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
        let receipt=json!({"seconds":began.elapsed().as_secs_f64(),"encoded_images":self.encoded-before.0,
            "prefix_images_skipped":self.skipped-before.1,"broadcast_images":self.transmitted-before.2,
            "cache":self.cache.stats(),"live_request_bytes":bindings.owned_bytes(),
            "cache_reclamations":reclamations,"diagnostic_span_hashes":hashes});
        Ok((bindings,receipt))
    }
    pub(crate) fn release_warm(&mut self) { self.cache.release_warm(); }
    pub(crate) fn clear_inputs(&mut self) { self.inputs.clear(); }
}
