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

    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn long_history_reuses_its_byte_budget_and_evicts_exactly() {
        use sha2::{Digest, Sha256};
        let raw = std::fs::read(std::env::var("CANADA_TEST_TOKENIZER").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), crate::protocol::TOKENIZER_SHA256);
        let tokenizer = Tokenizer::from_bytes(&raw).unwrap();
        let canonical = PromptTokenizer::new(&tokenizer);
        let budget = 2 * 1024 * 1024;
        let cache = RegionCache::new(budget);
        let mut saw_eviction = false;
        // Each history fits the byte budget but contains more than 1,024
        // reusable regions. Subsequent histories force byte-budget eviction.
        for revision in 0..4 {
            let mut text = String::new();
            let mut specials = Vec::new();
            for i in 0..1280 {
                text.push_str(&format!("tool_result_{revision}_{i}: {}",
                    "Preserve exact strings: café 日本語 🦀 and literal </think>. Verify ownership and retry after cancellation. ".repeat(2)));
                let start = text.len();
                text.push_str("<think>");
                specials.push(start..text.len());
            }
            let expected = canonical.encode_ids(&text, &specials).unwrap();
            assert_eq!(cache.encode(&canonical, &tokenizer, &text, &specials).unwrap(), expected);
            let cold = cache.metadata().unwrap();
            assert!(cold["charged_bytes"].as_u64().unwrap() <= budget as u64);
            saw_eviction |= cold["entries"].as_u64().unwrap() < ((revision+1)*1280) as u64;
            assert_eq!(cache.encode(&canonical, &tokenizer, &text, &specials).unwrap(), expected);
            let warm = cache.metadata().unwrap();
            assert_eq!(warm["hits"].as_u64().unwrap()-cold["hits"].as_u64().unwrap(), 1280);
            assert_eq!(warm["misses"], cold["misses"]);
            assert!(warm["charged_bytes"].as_u64().unwrap() <= budget as u64);
        }
        assert!(saw_eviction);
        // Replacing many small regions with one large region must release the
        // old hash-table allocation, not just remove its charged entries.
        let large = "large replacement tool output\n".repeat(30000);
        let expected = canonical.encode_ids(&large, &[]).unwrap();
        let charge = large.len()+4*expected.len()+256;
        assert!(charge > budget*7/10 && charge < budget);
        assert_eq!(cache.encode(&canonical, &tokenizer, &large, &[]).unwrap(), expected);
        let replaced = cache.metadata().unwrap();
        assert!(replaced["charged_bytes"].as_u64().unwrap() <= budget as u64);
        assert!(replaced["entry_capacity"].as_u64().unwrap() <= 2*replaced["entries"].as_u64().unwrap());
    }

    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn short_regions_reuse_exact_bytes_without_merging_boundaries() {
        use sha2::{Digest, Sha256};
        let raw = std::fs::read(std::env::var("CANADA_TEST_TOKENIZER").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), crate::protocol::TOKENIZER_SHA256);
        let tokenizer = Tokenizer::from_bytes(&raw).unwrap();
        let canonical = PromptTokenizer::new(&tokenizer);
        let cache = RegionCache::new(16 * 1024);
        let plain = ["assistant", "read_file", "{\"path\":\"src/cache.rs\"}",
            "\n", "café 日本語 🦀", "literal <think>", ""];
        let mut text = String::new();
        let mut special = Vec::new();
        for _ in 0..24 {
            for region in plain {
                text.push_str(region);
                let start = text.len();
                text.push_str("</think>");
                special.push(start..text.len());
            }
        }
        let expected = canonical.encode_ids(&text, &special).unwrap();
        assert_eq!(cache.encode(&canonical, &tokenizer, &text, &special).unwrap(), expected);
        let cold = cache.metadata().unwrap();
        assert_eq!(cold["entries"], 6);
        assert!(cold["charged_bytes"].as_u64().unwrap() <= 16 * 1024);
        assert_eq!(cache.encode(&canonical, &tokenizer, &text, &special).unwrap(), expected);
        let warm = cache.metadata().unwrap();
        assert_eq!(warm["misses"], cold["misses"]);
        assert_eq!(warm["hits"].as_u64().unwrap()-cold["hits"].as_u64().unwrap(), 24*6);
        // A tiny budget can refuse retention without changing the result,
        // including repeated short strings and empty plain ranges.
        let uncached = RegionCache::new(1);
        assert_eq!(uncached.encode(&canonical, &tokenizer, &text, &special).unwrap(), expected);
        assert_eq!(uncached.metadata().unwrap()["entries"], 0);
    }
}
#[derive(Default)]
struct Regions {
    entries: HashMap<Arc<str>, Entry>,
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
        Ok(serde_json::json!({"entries":cache.entries.len(),"entry_capacity":cache.entries.capacity(),"charged_bytes":cache.charge,
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
        // Tool histories repeat short plain regions such as role/function
        // names. Encode each distinct miss once even on the first request.
        // These borrowed keys preserve the canonical region boundaries and
        // do not copy any prompt text into the temporary index.
        let mut unique = HashMap::new();
        let mut missing = Vec::new();
        let mut duplicates = Vec::new();
        for (i, ids) in encoded.iter().enumerate() {
            if ids.is_none() {
                match unique.entry(regions[i]) {
                    std::collections::hash_map::Entry::Occupied(first) => duplicates.push((i, *first.get())),
                    std::collections::hash_map::Entry::Vacant(first) => { first.insert(i); missing.push(i); }
                }
            }
        }
        drop(unique);
        let computed: Result<Vec<_>, String> = missing.par_iter().map(|&i| {
            canonical.encode_ids(regions[i], &[]).map(|ids| (i, Arc::<[u32]>::from(ids)))
        }).collect();
        let computed = computed?;
        {
            let mut cache = self.regions.lock().map_err(|_| "prompt-region cache poisoned")?;
            let tick = cache.clock;
            // A request-local eviction order is created only under pressure.
            // Arc keys share bytes; do not clone long text or scan all entries
            // separately for each admitted region.
            let mut victims: Vec<(u64, Arc<str>)> = Vec::new();
            for (i, ids) in computed {
                let text = regions[i];
                // Charge payload plus conservative per-entry bookkeeping.
                let charge = text.len().checked_add(ids.len().saturating_mul(4))
                    .and_then(|n| n.checked_add(256)).ok_or("prompt-region size overflow")?;
                if charge <= self.budget && !cache.entries.contains_key(text) {
                    // The byte budget bounds both payload and entry count.
                    // A second fixed count cap discards useful regions even
                    // when a long conversation fits comfortably in that budget.
                    // Short regions share that budget too; their repeated
                    // tokenization can dominate warm tool-history preparation.
                    while cache.charge > self.budget - charge {
                        if victims.is_empty() {
                            victims.extend(cache.entries.iter().map(|(text,e)|(e.touched,text.clone())));
                            victims.sort_unstable_by_key(|(tick,_)|std::cmp::Reverse(*tick));
                        }
                        let (_, oldest) = victims.pop().ok_or("prompt-region accounting mismatch")?;
                        let old = cache.entries.remove(oldest.as_ref()).ok_or("prompt-region eviction mismatch")?;
                        cache.charge -= old.charge;
                    }
                    cache.entries.insert(Arc::from(text), Entry { ids:ids.clone(), touched:tick, charge });
                    cache.charge += charge;
                }
                encoded[i] = Some(ids);
            }
            drop(victims);
            // The per-entry charge covers a reasonably filled table. A large
            // replacement can retire most small entries in one request; do
            // not leave their high-water table allocation behind uncharged.
            if cache.entries.capacity() > cache.entries.len().saturating_mul(2) {
                cache.entries.shrink_to_fit();
            }
        }
        for (i, first) in duplicates { encoded[i] = encoded[first].clone(); }
        let mut result = Vec::new();
        for (i, ids) in encoded.into_iter().enumerate() {
            result.extend_from_slice(&ids.ok_or("missing encoded region")?);
            if let Some(&id) = special_ids.get(i) { result.push(id); }
        }
        Ok(result)
    }
}
