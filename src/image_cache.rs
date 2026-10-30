//! Bounded warm image-span retention. Live request owners remain authoritative
//! for their spans; weak entries allow reuse of those spans after LRU eviction.
use crate::{prompt_media::{Identity, ImageKey, Placement}, vision::{Geometry, ImageSpan}};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, VecDeque}, rc::{Rc, Weak}};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub start: usize,
    pub geometry: Geometry,
    pub input: [u8;32],
}
impl Descriptor {
    pub fn key(&self, encoder: [u8;32]) -> ImageKey { ImageKey { input:self.input, encoder } }
    pub fn encode_all(images: &[Self], out: &mut Vec<u8>) -> Result<()> {
        ensure!(images.len() <= crate::config::MODEL_CONTEXT/4, "too many image descriptors");
        out.extend_from_slice(&(images.len() as u32).to_le_bytes());
        for image in images {
            image.geometry.validate()?;
            ensure!(image.start.checked_add(image.geometry.tokens()).is_some_and(|end|end<=crate::config::MODEL_CONTEXT),
                "image position exceeds context");
            for n in [image.start,image.geometry.height,image.geometry.width] {
                out.extend_from_slice(&(n as u32).to_le_bytes());
            }
            out.extend_from_slice(&image.input);
        }
        Ok(())
    }
    pub fn decode_all(bytes: &[u8]) -> Result<Vec<Self>> {
        ensure!(bytes.len() >= 4, "truncated image descriptor count");
        let n=u32::from_le_bytes(bytes[..4].try_into().unwrap())as usize;
        ensure!(n <= crate::config::MODEL_CONTEXT/4 && bytes.len()==4+n*44, "image descriptor extent");
        bytes[4..].chunks_exact(44).map(|item| {
            let start=u32::from_le_bytes(item[..4].try_into().unwrap())as usize;
            let geometry=Geometry { height:u32::from_le_bytes(item[4..8].try_into().unwrap())as usize,
                width:u32::from_le_bytes(item[8..12].try_into().unwrap())as usize }.validate()?;
            ensure!(start.checked_add(geometry.tokens()).is_some_and(|end|end<=crate::config::MODEL_CONTEXT),
                "image descriptor exceeds context");
            Ok(Self {start,geometry,input:item[12..].try_into().unwrap()})
        }).collect()
    }
    pub fn identity(images: &[Self], encoder: [u8;32], tokens: &[u32]) -> Result<Identity> {
        let placements=images.iter().map(|image| Ok(Placement {start:image.start,
            tokens:image.geometry.validate()?.tokens(), key:image.key(encoder)})).collect::<Result<Vec<_>>>()?;
        Identity::new(placements,tokens)
    }
}

pub struct SpanCache {
    budget: usize,
    retained: usize,
    warm: HashMap<ImageKey,Rc<ImageSpan>>,
    live: HashMap<ImageKey,Weak<ImageSpan>>,
    oldest: VecDeque<ImageKey>,
    hits: u64,
    misses: u64,
    evictions: u64,
}
impl SpanCache {
    pub fn new(budget: usize) -> Result<Self> {
        ensure!(budget <= 2usize<<30, "image span cache budget exceeds admission contract");
        Ok(Self {budget,retained:0,warm:HashMap::new(),live:HashMap::new(),oldest:VecDeque::new(),
            hits:0,misses:0,evictions:0})
    }
    fn retain(&mut self,key:ImageKey,span:Rc<ImageSpan>) {
        self.oldest.retain(|k|*k!=key);
        if self.warm.contains_key(&key) { self.oldest.push_back(key);return; }
        let bytes=span.data.bytes();
        if bytes>self.budget { return; }
        while self.retained+bytes>self.budget {
            let victim=self.oldest.pop_front().expect("warm image cache accounting");
            let old=self.warm.remove(&victim).expect("warm image cache identity");
            self.retained-=old.data.bytes();self.evictions+=1;
        }
        self.retained+=bytes;self.warm.insert(key,span);self.oldest.push_back(key);
    }
    pub fn get(&mut self,key:ImageKey) -> Option<Rc<ImageSpan>> {
        if let Some(span)=self.live.get(&key).and_then(Weak::upgrade) {
            self.hits+=1;self.retain(key,span.clone());Some(span)
        }else{self.misses+=1;self.live.remove(&key);None}
    }
    pub fn insert(&mut self,key:ImageKey,span:Rc<ImageSpan>) -> Result<()> {
        let geometry=span.geometry.validate()?;
        ensure!(span.data.len()==geometry.tokens()*5120, "image cache span geometry");
        ensure!(self.live.get(&key).and_then(Weak::upgrade).is_none_or(|old|Rc::ptr_eq(&old,&span)),
            "duplicate image allocation for a live identity");
        self.live.retain(|_,v|v.strong_count()>0);
        self.live.insert(key,Rc::downgrade(&span));self.retain(key,span);Ok(())
    }
    /// Frees optional retention only. Active/suspended requests keep their
    /// immutable spans, and actual UMA admission includes those allocations.
    pub fn release_warm(&mut self) {
        self.warm.clear();self.oldest.clear();self.retained=0;
        self.live.retain(|_,v|v.strong_count()>0);
    }
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({"hits":self.hits,"misses":self.misses,"evictions":self.evictions,
            "warm_bytes":self.retained,"budget_bytes":self.budget,
            "live_bytes":self.live.values().filter_map(Weak::upgrade).map(|s|s.data.bytes()).sum::<usize>(),
            "warm_images":self.warm.len()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_descriptor_wire_is_bounded_and_keeps_full_identity() {
        let images=vec![Descriptor {start:1,geometry:Geometry {height:3,width:6},input:[9;32]}];
        let mut bytes=Vec::new();Descriptor::encode_all(&images,&mut bytes).unwrap();
        assert_eq!(Descriptor::decode_all(&bytes).unwrap(),images);
        for cut in 0..bytes.len() { assert!(Descriptor::decode_all(&bytes[..cut]).is_err()); }
        let mut extra=bytes.clone();extra.push(0);assert!(Descriptor::decode_all(&extra).is_err());
        bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());assert!(Descriptor::decode_all(&bytes).is_err());
        let prompt=[2,129264,129264,129264,129264,129264,3];
        let a=Descriptor::identity(&images,[1;32],&prompt).unwrap();
        let b=Descriptor::identity(&images,[2;32],&prompt).unwrap();
        assert!(!a.matches_prefix(&b,2));
        assert!(Descriptor::identity(&images,[1;32],&prompt[..3]).is_err());
    }
}
