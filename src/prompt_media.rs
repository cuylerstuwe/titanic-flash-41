//! Image identity and GPU span ownership at the language-model input boundary.
//! Token IDs remain canonical. Full image identities accompany prefix keys;
//! they must never be truncated into synthetic vocabulary IDs.
use crate::{cuda::Cuda, vision::ImageSpan};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

pub const IMAGE_TOKEN: u32 = 129264;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageKey {
    /// Hash of encoded content, detail and the pinned input processor.
    pub input: [u8; 32],
    /// Hash of canonical vision weights and the admitted encoder implementation.
    pub encoder: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub start: usize,
    pub tokens: usize,
    pub key: ImageKey,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Identity {
    placements: Vec<Placement>,
}
impl Identity {
    pub fn new(placements: Vec<Placement>, prompt: &[u32]) -> Result<Self> {
        ensure!(prompt.len() <= crate::config::MODEL_CONTEXT, "media prompt capacity");
        let mut previous = 0;
        for p in &placements {
            let end = p.start.checked_add(p.tokens)
                .ok_or_else(|| anyhow::anyhow!("image span overflow"))?;
            ensure!((1..=1024).contains(&p.tokens) && p.start >= previous && end <= prompt.len(),
                "image spans must be ordered, disjoint and inside the prompt");
            ensure!(prompt[previous..p.start].iter().all(|&t| t != IMAGE_TOKEN)
                && prompt[p.start..end].iter().all(|&t| t == IMAGE_TOKEN),
                "image span and canonical token positions differ");
            previous = end;
        }
        ensure!(prompt[previous..].iter().all(|&t| t != IMAGE_TOKEN),
            "image placeholder has no content identity");
        Ok(Self { placements })
    }
    pub fn placements(&self) -> &[Placement] { &self.placements }
    pub fn is_empty(&self) -> bool { self.placements.is_empty() }
    pub fn metadata_bytes(&self) -> usize { self.placements.capacity() * std::mem::size_of::<Placement>() }
    pub fn validate_prompt(&self, prompt: &[u32]) -> Result<()> {
        // An Identity obtained from a shorter checkpoint is not automatically
        // an identity for an entire new prompt. Recheck coverage at admission.
        Self::new(self.placements.clone(), prompt).map(|_| ())
    }
    /// A checkpoint may end inside an image. Keep that image's entire identity
    /// and original span extent, even though only its prefix has entered KV.
    pub fn through(&self, frontier: usize) -> Self {
        Self { placements: self.placements.iter().take_while(|p| p.start < frontier).cloned().collect() }
    }
    /// Used together with exact token-prefix equality, never in place of it.
    pub fn matches_prefix(&self, other: &Self, frontier: usize) -> bool {
        self.placements.iter().take_while(|p| p.start < frontier)
            .eq(other.placements.iter().take_while(|p| p.start < frontier))
    }
}

/// Immutable image spans for one prompt, on one CUDA context. Sharing this
/// owner across execution profiles keeps CUDA graph row pointers alive.
pub struct Bindings {
    identity: Identity,
    spans: Vec<Option<Rc<ImageSpan>>>,
}
impl Bindings {
    pub fn new(cuda: &Cuda, identity: Identity, spans: Vec<Rc<ImageSpan>>) -> Result<Self> {
        Self::suffix(cuda, identity, spans.into_iter().map(Some).collect(), 0)
    }
    /// Images entirely behind an already restored frontier need neither a
    /// fresh vision pass nor a live span. Executing such a skipped row fails
    /// closed; a prefix-cache hit is not permission to approximate its suffix.
    pub fn suffix(cuda: &Cuda, identity: Identity, spans: Vec<Option<Rc<ImageSpan>>>,
        frontier: usize) -> Result<Self> {
        ensure!(identity.placements.len() == spans.len(), "media span count");
        for (placement, span) in identity.placements.iter().zip(&spans) {
            let Some(span) = span else {
                ensure!(placement.start + placement.tokens <= frontier,
                    "unencoded image crosses the reusable frontier");
                continue;
            };
            let geometry = span.geometry.validate()?;
            ensure!(cuda.owns_buffer(&span.data) && span.data.len() == geometry.tokens() * 5120
                && placement.tokens == geometry.tokens(), "image span context or geometry");
        }
        Ok(Self { identity, spans })
    }
    pub fn identity(&self) -> &Identity { &self.identity }
    pub fn owned_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        self.identity.metadata_bytes() + self.spans.capacity() * std::mem::size_of::<Option<Rc<ImageSpan>>>()
            + self.spans.iter().flatten().filter(|s| seen.insert(Rc::as_ptr(s)))
                .map(|s| s.data.bytes()).sum::<usize>()
    }
    pub(crate) fn in_context(&self, cuda: &Cuda) -> bool {
        self.spans.iter().flatten().all(|span| cuda.owns_buffer(&span.data))
    }
    /// Returns None for an ordinary text chunk, without allocating row metadata.
    /// Row addresses are control data; no model arithmetic runs on the CPU.
    pub(crate) fn row_pointers(&self, start: usize, ids: &[i64], live: usize) -> Result<Option<Vec<u64>>> {
        ensure!(live <= ids.len(), "media live row count");
        let end = start.checked_add(live).ok_or_else(|| anyhow::anyhow!("media frontier overflow"))?;
        ensure!(end <= crate::config::MODEL_CONTEXT, "media frontier exceeds context");
        let first = self.identity.placements.partition_point(|p| p.start + p.tokens <= start);
        if self.identity.placements.get(first).is_none_or(|p| p.start >= end) { return Ok(None); }
        let mut pointers = vec![0u64; ids.len()];
        for (placement, span) in self.identity.placements[first..].iter().zip(&self.spans[first..]) {
            if placement.start >= end { break; }
            let span = span.as_ref().ok_or_else(|| anyhow::anyhow!("attempt to re-embed a skipped cached image"))?;
            for position in start.max(placement.start)..end.min(placement.start + placement.tokens) {
                let row = position - start;
                ensure!(ids[row] == i64::from(IMAGE_TOKEN), "image binding overlaps a text token");
                pointers[row] = span.data.ptr() + ((position - placement.start) * 5120 * 2) as u64;
            }
        }
        Ok(Some(pointers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(n: u8) -> ImageKey { ImageKey { input: [n; 32], encoder: [7; 32] } }
    #[test]
    fn prefix_identity_covers_content_detail_encoder_and_partial_images() {
        let tokens = [3, IMAGE_TOKEN, IMAGE_TOKEN, 4, IMAGE_TOKEN, IMAGE_TOKEN, 5];
        let placements = vec![Placement { start: 1, tokens: 2, key: key(1) },
            Placement { start: 4, tokens: 2, key: key(2) }];
        let original = Identity::new(placements.clone(), &tokens).unwrap();
        for field in ["content", "encoder"] {
            let mut changed = placements.clone();
            if field == "content" { changed[0].key.input[31] ^= 1; }
            else { changed[0].key.encoder[31] ^= 1; }
            let changed = Identity::new(changed, &tokens).unwrap();
            assert!(original.matches_prefix(&changed, 1));
            assert!(!original.matches_prefix(&changed, 2));
        }
        let mut later = placements; later[1].key = key(3);
        let later = Identity::new(later, &tokens).unwrap();
        assert!(original.matches_prefix(&later, 4));
        assert!(!original.matches_prefix(&later, 5));
        let checkpoint = original.through(2);
        assert_eq!(checkpoint.placements()[0].tokens, 2);
        assert!(checkpoint.matches_prefix(&original, 2));
        assert!(!checkpoint.matches_prefix(&original, 5));
    }
    #[test]
    fn malformed_or_unidentified_spans_fail_closed() {
        let tokens = [3, IMAGE_TOKEN, IMAGE_TOKEN, 4];
        assert!(Identity::new(vec![], &tokens).is_err());
        for (start, count) in [(0,2), (1,1), (1,4), (usize::MAX,2), (1,0)] {
            assert!(Identity::new(vec![Placement { start, tokens: count, key: key(1) }], &tokens).is_err());
        }
        let one = Placement { start: 1, tokens: 2, key: key(1) };
        assert!(Identity::new(vec![one.clone(), one], &tokens).is_err());
        assert!(Identity::new(vec![], &[2,3,4]).unwrap().is_empty());
    }
}
