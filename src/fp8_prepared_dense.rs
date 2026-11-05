//! Dense-only FlashInfer consumers for qualified native prepared operands.
//! The native quantizer, fused producer and any zero padding remain owned by
//! Linear; this never invokes the FI quantizer.
use super::{bind, find};
use crate::{cuda::{Cuda,Stream},primitives::Primitives};
use anyhow::{Context,Result,ensure};
use serde_json::Value;
use sha2::{Digest,Sha256};
use std::{cell::Cell,collections::BTreeMap,path::Path};

pub(super) struct PreparedDense {
    primitives:Primitives,
    dense:usize,
    enabled:Cell<bool>,
}
impl PreparedDense {
    pub(super) fn open(cuda:&Cuda,parent:&Path,rows:usize,enabled:bool)->Result<Self> {
        ensure!(matches!(rows,6|16),"prepared shared down row geometry");
        let pins:Value=serde_json::from_str(include_str!("fp8_prepared_shared_down_bundles.json"))?;
        let pin=&pins[rows.to_string()];
        Self::open_pinned(cuda,parent,rows,enabled,pin,640,5120,[16,64],2560,102400)
    }
    pub(super) fn open_index_query(cuda:&Cuda,parent:&Path,enabled:bool)->Result<Self> {
        let pin:Value=serde_json::from_str(include_str!("fp8_prepared_index_query_bundle.json"))?;
        Self::open_pinned(cuda,parent,16,enabled,&pin,1280,4096,[16,128],5120,163840)
    }
    #[allow(clippy::too_many_arguments)]
    fn open_pinned(cuda:&Cuda,parent:&Path,rows:usize,enabled:bool,pin:&Value,
        k:usize,n:usize,tile:[usize;2],scale_a:usize,scale_b:usize)->Result<Self> {
        let root=parent.join(pin["bundle"].as_str().context("prepared dense bundle")?);
        let manifest=std::fs::read(root.join("manifest.json"))?;
        let bindings=std::fs::read(root.join("bindings.json"))?;
        ensure!(format!("{:x}",Sha256::digest(&manifest))==pin["manifest_sha256"].as_str().unwrap_or("")
            &&format!("{:x}",Sha256::digest(&bindings))==pin["bindings_sha256"].as_str().unwrap_or(""),
            "prepared dense artifact identity");
        let m:Value=serde_json::from_slice(&manifest)?;let g=&m["geometry"];
        ensure!(m["architecture"]==121&&m["physical_sms"]==48
            &&m["source"]=="https://github.com/flashinfer-ai/flashinfer"
            &&g["rows"]==rows&&g["in_features"]==k&&g["out_features"]==n
            &&g["mma_tiler_mn"]==serde_json::json!(tile)
            &&g["scale_a_bytes"]==scale_a&&g["scale_b_bytes"]==scale_b
            &&g["split_k_slices"]==1&&g["split_k_atomic_bf16"]==false
            &&g["swap_ab"]==false&&g["tile_k"]==128&&g["max_active_clusters"]==48
            &&g["use_prefetch"]==false,
            "unqualified prepared dense geometry");
        let primitives=Primitives::open(cuda,&root.join("libcanada_primitives.so"),&root.join("bindings.json"))?;
        ensure!(primitives.bindings.abi==2&&primitives.bindings.entries.len()==4
            &&primitives.bindings.source=="https://github.com/flashinfer-ai/flashinfer"
            &&primitives.bindings.library_sha256==pin["library_sha256"].as_str().unwrap_or(""),
            "prepared dense library identity");
        let dense=find(&primitives.bindings.entries,"flashinfer.mxfp8.dense.pdl1")?;
        Ok(Self{primitives,dense,enabled:Cell::new(enabled)})
    }
    pub(super) fn enabled(&self)->bool {self.enabled.get()}
    pub(super) fn set_for_probe(&self,enabled:bool){self.enabled.set(enabled);}
    pub(super) unsafe fn launch(&self,values:u64,weights:u64,scales:u64,
        weight_scales:u64,output:u64,alpha:u64,stream:&Stream)->Result<()> {
        let arguments=bind(&self.primitives.bindings.entries[self.dense],&BTreeMap::from([
            ("x",(values,"pointer")),("w",(weights,"pointer")),
            ("xs",(scales,"pointer")),("ws",(weight_scales,"pointer")),
            ("y",(output,"pointer")),("alpha",(alpha,"pointer")),
        ]))?;
        unsafe{self.primitives.launch_words(self.dense as i32,&arguments,stream)}
    }
}
