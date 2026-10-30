//! Bounded, exact memoization around Recipe 0.2.0's plain-region encoder.
//! Special ranges remain explicit template metadata. Literal marker spellings
//! inside a plain region are never promoted to special tokens. Cache keys are
//! the full region bytes, scoped to one immutable canonical tokenizer owner.
use deepseek_recipe_encoding::{PromptTokenizer, TokenizerEncoder};
use rayon::prelude::*;
use std::{collections::HashMap, ops::Range, sync::{Arc, Mutex}};
use tokenizers::Tokenizer;

struct Entry {
    ids: Arc<[u32]>,
    touched: u64,
    charge: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn canonical_regions_survive_eviction_literals_and_concurrent_hits() {
        use sha2::{Digest, Sha256};
        let raw = std::fs::read(std::env::var("CANADA_TEST_TOKENIZER").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), crate::protocol::TOKENIZER_SHA256);
        let tokenizer = Tokenizer::from_bytes(&raw).unwrap();
        let canonical = PromptTokenizer::new(&tokenizer);
        let cache = RegionCache::new(2048);
        let cases: Vec<_> = (0..8).map(|i| {
            let left = format!("case{i} café 日本語 🦀 {}", "alpha beta gamma delta ".repeat(12));
            let middle = "Literal <think> stays plain. ".repeat(8);
            let end = left.len() + 7 + middle.len();
            (format!("{left}<think>{middle}</think>suffix"), vec![left.len()..left.len()+7,end..end+8])
        }).collect();
        for (text, ranges) in cases.iter().chain(cases.iter().rev()) {
            assert_eq!(cache.encode(&canonical, &tokenizer, text, ranges).unwrap(), canonical.encode_ids(text, ranges).unwrap());
            assert!(cache.metadata().unwrap()["charged_bytes"].as_u64().unwrap() <= 2048);
        }
        let cache = RegionCache::new(32 * 1024);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for (text, ranges) in &cases {
                        assert_eq!(cache.encode(&canonical, &tokenizer, text, ranges).unwrap(), canonical.encode_ids(text, ranges).unwrap());
                    }
                });
            }
        });
        assert!(cache.metadata().unwrap()["entries"].as_u64().unwrap() > 0);
        assert!(cache.encode(&canonical, &tokenizer, "é", &[1..2]).is_err());
    }
}
#[derive(Default)]
struct Regions {
    entries: HashMap<String, Entry>,
    charge: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}
pub(crate) struct RegionCache {
    regions: Mutex<Regions>,
    budget: usize,
}
impl RegionCache {
    pub(crate) fn new(budget: usize) -> Self {
        Self { regions: Mutex::new(Regions::default()), budget }
    }
    pub(crate) fn metadata(&self) -> Result<serde_json::Value, String> {
        let cache = self.regions.lock().map_err(|_| "prompt-region cache poisoned")?;
        Ok(serde_json::json!({"entries":cache.entries.len(),"charged_bytes":cache.charge,
            "budget_bytes":self.budget,"hits":cache.hits,"misses":cache.misses}))
    }
    pub(crate) fn encode(&self, canonical: &PromptTokenizer, tokenizer: &Tokenizer,
        text: &str, specials: &[Range<usize>]) -> Result<Vec<u32>, String> {
        let mut regions = Vec::with_capacity(specials.len() + 1);
        let mut special_ids = Vec::with_capacity(specials.len());
        let mut start = 0;
        for range in specials {
            if range.start < start || range.end < range.start || range.end > text.len()
                || !text.is_char_boundary(range.start) || !text.is_char_boundary(range.end) {
                return Err("invalid canonical special-token range".into());
            }
            regions.push(&text[start..range.start]);
            special_ids.push(*tokenizer.get_added_vocabulary().get_vocab().get(&text[range.clone()])
                .ok_or_else(|| "canonical special token is absent".to_owned())?);
            start = range.end;
        }
        regions.push(&text[start..]);
        let mut encoded: Vec<Option<Arc<[u32]>>> = vec![None; regions.len()];
        {
            let mut cache = self.regions.lock().map_err(|_| "prompt-region cache poisoned")?;
            cache.clock = cache.clock.checked_add(1).ok_or("prompt-region clock exhausted")?;
            let tick = cache.clock;
            for (i, region) in regions.iter().enumerate() {
                if region.is_empty() { encoded[i] = Some(Arc::from([])); }
                else if let Some(entry) = cache.entries.get_mut(*region) {
                    entry.touched = tick;
                    encoded[i] = Some(entry.ids.clone());
                    cache.hits += 1;
                } else { cache.misses += 1; }
            }
        }
        // Each region is independent in the official encoder. Preserve its
        // exact implementation and parallelize misses; never cut a plain
        // region at an arbitrary character/token boundary to obtain reuse.
        let missing: Vec<_> = encoded.iter().enumerate().filter_map(|(i,v)| v.is_none().then_some(i)).collect();
        let computed: Result<Vec<_>, String> = missing.par_iter().map(|&i| {
            canonical.encode_ids(regions[i], &[]).map(|ids| (i, Arc::<[u32]>::from(ids)))
        }).collect();
        let computed = computed?;
        {
            let mut cache = self.regions.lock().map_err(|_| "prompt-region cache poisoned")?;
            let tick = cache.clock;
            for (i, ids) in computed {
                let text = regions[i];
                // Charge payload plus conservative per-entry bookkeeping.
                let charge = text.len().checked_add(ids.len().saturating_mul(4))
                    .and_then(|n| n.checked_add(256)).ok_or("prompt-region size overflow")?;
                if text.len() >= 128 && charge <= self.budget && !cache.entries.contains_key(text) {
                    while cache.charge + charge > self.budget || cache.entries.len() >= 1024 {
                        let oldest = cache.entries.iter().min_by_key(|(_,e)|e.touched)
                            .map(|(text,_)| text.clone()).ok_or("prompt-region accounting mismatch")?;
                        let old = cache.entries.remove(&oldest).unwrap();
                        cache.charge -= old.charge;
                    }
                    cache.entries.insert(text.to_owned(), Entry { ids:ids.clone(), touched:tick, charge });
                    cache.charge += charge;
                }
                encoded[i] = Some(ids);
            }
        }
        let mut result = Vec::new();
        for (i, ids) in encoded.into_iter().enumerate() {
            result.extend_from_slice(&ids.ok_or("missing encoded region")?);
            if let Some(&id) = special_ids.get(i) { result.push(id); }
        }
        Ok(result)
    }
}
