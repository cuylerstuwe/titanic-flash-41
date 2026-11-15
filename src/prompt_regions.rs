//! Bounded, exact memoization around Recipe 0.2.0's plain-region encoder.
//! Special ranges remain explicit template metadata. Literal marker spellings
//! inside a plain region are never promoted to special tokens. Cache keys are
//! the full region bytes, scoped to one immutable canonical tokenizer owner.
//!
//! Prefix reuse (CANADA_SCRATCH_PROMPT_PREFIX_REUSE=1; default off). A long region that differs from a cached one
//! only after a long common prefix (a new question after the same document in ONE message) used to be tokenized
//! whole: 0.9-1.0 ms per 1,000 tokens, 830 ms of a first token at 837K tokens (standing 3, Nov 14). With the switch
//! the cached ids are kept up to a cut before the first differing byte and only the rest is tokenized. The result
//! is the full tokenization, by these properties of the pinned tokenizer (its SHA-256 is checked at load):
//!   - no normalizer; the pre-tokenizer is three `Split`s (digit groups, CJK runs, one alternation over letters,
//!     punctuation and whitespace) and a byte-level map; none has an anchor or a look-behind, the only look-ahead
//!     is one character; BPE merges inside one pre-token. So text after a pre-token boundary never changes the
//!     ids before it once one more character is fixed, and text before the boundary never changes the ids after;
//!   - the cut is taken only where a pre-token starts in ANY text (`cut_at`): after a line break when a visible
//!     ASCII character follows (a break is always the end of its match), where an ASCII digit run starts, and
//!     where it ends;
//!   - every base token is a byte string, so the cached ids give the byte position of each token (`token_bytes`).
//! Two guards on every use: the cut must fall exactly on a token boundary of the cached ids, and the tail is
//! tokenized from an EARLIER cut so that its first ids must repeat the cached ids up to the cut (otherwise the
//! region is tokenized whole and `prefix_reuse_rejects` counts it; expected 0). Tests at the end of this file
//! compare against the whole tokenization on random documents and edits.
use deepseek_recipe_encoding::{PromptTokenizer, TokenizerEncoder};
use rayon::prelude::*;
use std::{collections::HashMap, ops::Range, sync::{Arc, Mutex, OnceLock}};
use tokenizers::Tokenizer;

/// Regions shorter than this are tokenized whole (the saving would be under ~4 ms).
const REUSE_MIN_BYTES: usize = 16 * 1024;
/// The cut lies at least this many bytes before the first differing byte.
const REUSE_MARGIN: usize = 64;
/// The tail is tokenized from a cut at least this many bytes before the splice, and must repeat the cached ids there.
const REUSE_OVERLAP: usize = 256;
/// How far back from the first differing byte a cut is searched.
const REUSE_SCAN: usize = 1 << 20;
/// Cached regions compared with a missed one (most recently used first).
const REUSE_CANDIDATES: usize = 4;

/// True where a pre-token of the pinned tokenizer starts in any text containing these two bytes at `p - 1`, `p`:
/// after a line break followed by a visible ASCII character (the alternation's whitespace and punctuation
/// branches end with their last line break), at the start of an ASCII digit run and at its end (the digit split
/// runs first and isolates every run). ASCII on both sides keeps Unicode digits and spaces out of the rule.
fn cut_at(bytes: &[u8], p: usize) -> bool {
    let (l, r) = (bytes[p - 1], bytes[p]);
    ((l == b'\n' || l == b'\r') && (0x21..=0x7e).contains(&r))
        || (l < 0x80 && !l.is_ascii_digit() && r.is_ascii_digit())
        || (l.is_ascii_digit() && r < 0x80 && !r.is_ascii_digit())
}
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 4096 <= n && a[i..i + 4096] == b[i..i + 4096] { i += 4096; }
    while i < n && a[i] == b[i] { i += 1; }
    i
}
/// Byte length of every base token (a byte-level vocabulary spells one character a byte), or None when the
/// tokenizer could shorten or pad an encoding (then a tail would not be the whole encoding's tail).
fn token_bytes(tokenizer: &Tokenizer) -> Option<Vec<u16>> {
    if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() { return None; }
    let vocab = tokenizer.get_vocab(false);
    let mut table = vec![0u16; *vocab.values().max()? as usize + 1];
    for (token, &id) in &vocab { table[id as usize] = u16::try_from(token.chars().count()).ok()?; }
    Some(table)
}

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

    /// xorshift: the same documents on every run, without a dependency
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    }
    /// Words, blanks and line breaks of every kind, punctuation, digit runs (ASCII and not), CJK, combining marks,
    /// code, JSON and literal marker spellings: whatever could make a pre-token depend on its neighbours.
    const PIECES: &[&str] = &[
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "Tokenization", "boundary", "naïve", "café", "Überraschung",
        " ", " ", " ", "  ", "   ", "\t", "\n", "\n", "\n\n", "\r\n", "\r", " \n", "\n  ", "\n\n\n", "    \n", "\u{a0}", "\u{3000}", "\u{2028}", "\u{85}",
        ".", ",", ";", ":", "!", "?", "...", "--", "->", "=>", "::", "()", "{}", "[]", "\"", "'", "`", "```", "#", "##", "//", "/*", "*/", "<=", "&&", "||", "$", "%", "@", "。", "、", "—", "…", "€", "±",
        "0", "1", "7", "42", "123", "1234", "98765", "20261114", "3.14159", "1,000,000", "0x7f", "v2", "x86_64", "²", "１２３", "٣", "Ⅷ", "½",
        "日本語", "東京", "漢字かな交じり", "カタカナ", "한국어", "中文测试", "🦀", "👍🏽", "é", "e\u{301}", "\u{200b}", "\u{feff}", "ß", "Ω", "я", "عربى",
        "fn main() {", "let x = 1;", "return Ok(());", "{\"key\": \"value\", \"n\": 12345}", "https://example.com/a/b?c=1&d=2", "user@example.com",
        "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>", "<｜DSML｜", "｜DSML｜", "<｜end▁of▁sentence｜>",
        "def f(x):\n    return x + 1\n", "- item\n- item 2\n", "| a | b |\n|---|---|\n", "\\n", "\\", "/", "_", "__init__", "CamelCaseWord", "snake_case_word", "UPPER", "'s", ".txt", "don't",
    ];
    fn document(rng: &mut Rng, bytes: usize) -> String {
        let mut text = String::with_capacity(bytes + 4096);
        while text.len() < bytes {
            let piece = PIECES[rng.below(PIECES.len())];
            // now and then a long run of one piece: digit runs, blank runs, letter runs, breaks
            let repeat = if rng.below(40) == 0 { 2 + rng.below(300) } else { 1 };
            for _ in 0..repeat { text.push_str(piece); }
            if rng.below(3) == 0 { text.push(' '); }
        }
        text
    }
    fn boundary(text: &str, mut at: usize) -> usize {
        at = at.min(text.len());
        while !text.is_char_boundary(at) { at += 1; }
        at
    }
    fn pinned() -> (Tokenizer, PromptTokenizer) {
        use sha2::{Digest, Sha256};
        let raw = std::fs::read(std::env::var("CANADA_TEST_TOKENIZER").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), crate::protocol::TOKENIZER_SHA256);
        let tokenizer = Tokenizer::from_bytes(&raw).unwrap();
        let canonical = PromptTokenizer::new(&tokenizer);
        (tokenizer, canonical)
    }

    /// The property the splice rests on, tested directly: at every position `cut_at` accepts, the whole
    /// tokenization is the tokenization of the text before it followed by that of the text after it, and the base
    /// tokens' byte lengths add up to the text.
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn cuts_start_a_pre_token_in_any_text() {
        let (tokenizer, canonical) = pinned();
        let table = token_bytes(&tokenizer).unwrap();
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let (mut cuts, mut tested) = (0usize, 0usize);
        for _ in 0..60 {
            let text = document(&mut rng, 6000);
            let whole = canonical.encode_ids(&text, &[]).unwrap();
            assert_eq!(whole.iter().map(|&id| table[id as usize] as usize).sum::<usize>(), text.len());
            let bytes = text.as_bytes();
            let positions: Vec<usize> = (1..bytes.len()).filter(|&p| cut_at(bytes, p)).collect();
            cuts += positions.len();
            for &p in &positions {
                let mut joined = canonical.encode_ids(&text[..p], &[]).unwrap();
                joined.extend(canonical.encode_ids(&text[p..], &[]).unwrap());
                assert_eq!(joined, whole, "cut at byte {p}: {:?} | {:?}", &text[boundary(&text, p.saturating_sub(24))..p], &text[p..boundary(&text, p + 24)]);
                tested += 1;
            }
        }
        assert!(cuts > 10_000 && tested == cuts, "{cuts} cuts, {tested} tested");
    }

    /// A cached long region, then the same region with its tail replaced, text inserted, deleted, appended or cut
    /// off: with prefix reuse the ids must be the whole tokenization's, most edits must be served by a splice and
    /// no guard may refuse one.
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn prefix_reuse_is_the_whole_tokenization_on_random_edits() {
        let (tokenizer, canonical) = pinned();
        let mut rng = Rng(0x2545f4914f6cdd1d);
        let (mut edits, mut reuses) = (0u64, 0u64);
        for round in 0..12 {
            let cache = RegionCache::with_prefix_reuse(64 * 1024 * 1024, true);
            let size = 40_000 + rng.below(80_000);
            let base = document(&mut rng, size);
            assert_eq!(cache.encode(&canonical, &tokenizer, &base, &[]).unwrap(), canonical.encode_ids(&base, &[]).unwrap());
            let mut current = base.clone();
            for _ in 0..40 {
                let at = boundary(&current, if rng.below(4) == 0 { rng.below(current.len()) } else { current.len() - rng.below(current.len() / 8) });
                let size = rng.below(3000);
                let fresh = document(&mut rng, size);
                let next = match rng.below(6) {
                    0 => format!("{}{}", &current[..at], fresh),                                   // tail replaced
                    1 => format!("{}{}{}", &current[..at], fresh, &current[at..]),                   // inserted
                    2 => { let end = boundary(&current, at + rng.below(2000)); format!("{}{}", &current[..at], &current[end..]) }   // deleted
                    3 => format!("{}{}", current, fresh),                                            // appended
                    4 => current[..at].to_owned(),                                                   // cut off
                    _ => { let end = boundary(&current, at + rng.below(200)); format!("{}{}{}", &current[..at], &fresh[..boundary(&fresh, fresh.len().min(40))], &current[end..]) }   // a few bytes changed
                };
                if next.len() < REUSE_MIN_BYTES { continue; }
                let expected = canonical.encode_ids(&next, &[]).unwrap();
                assert_eq!(cache.encode(&canonical, &tokenizer, &next, &[]).unwrap(), expected, "round {round}, edit at byte {at}");
                // the same text again is a plain hit
                assert_eq!(cache.encode(&canonical, &tokenizer, &next, &[]).unwrap(), expected);
                edits += 1;
                // half the time continue from the edited text, otherwise edit the base again
                if rng.below(2) == 0 { current = next; } else { current = base.clone(); }
            }
            let meta = cache.metadata().unwrap();
            assert_eq!(meta["prefix_reuse_rejects"], 0, "a guard refused a splice in round {round}");
            reuses += meta["prefix_reuses"].as_u64().unwrap();
        }
        assert!(edits > 300 && reuses * 10 >= edits * 6, "{reuses} splices in {edits} edits");
        // with the switch off nothing is spliced
        let off = RegionCache::new(64 * 1024 * 1024);
        let text = document(&mut rng, 50_000);
        off.encode(&canonical, &tokenizer, &text, &[]).unwrap();
        let changed = format!("{}tail", &text[..boundary(&text, 49_000)]);
        assert_eq!(off.encode(&canonical, &tokenizer, &changed, &[]).unwrap(), canonical.encode_ids(&changed, &[]).unwrap());
        assert_eq!(off.metadata().unwrap()["prefix_reuses"], 0);
    }

    /// What the splice saves on a document of standing 3's largest size (prints; a few seconds).
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn prefix_reuse_on_a_long_document() {
        let (tokenizer, canonical) = pinned();
        let mut rng = Rng(0x94d049bb133111eb);
        let cache = RegionCache::with_prefix_reuse(64 * 1024 * 1024, true);
        let base = document(&mut rng, 3_400_000);
        let start = std::time::Instant::now();
        let first = cache.encode(&canonical, &tokenizer, &format!("{base}\n\nQuestion 0: what changed?"), &[]).unwrap();
        let whole_seconds = start.elapsed().as_secs_f64();
        let mut spliced_seconds = Vec::new();
        for note in 1..6 {
            let text = format!("{base}\n\nQuestion {note}: what changed since revision {}?", note * 17);
            let start = std::time::Instant::now();
            let ids = cache.encode(&canonical, &tokenizer, &text, &[]).unwrap();
            spliced_seconds.push(start.elapsed().as_secs_f64());
            assert_eq!(ids, canonical.encode_ids(&text, &[]).unwrap());
        }
        let meta = cache.metadata().unwrap();
        eprintln!("{} tokens: whole {:.1} ms, spliced {:?} ms, {}", first.len(), 1e3 * whole_seconds,
            spliced_seconds.iter().map(|s| (1e4 * s).round() / 10.0).collect::<Vec<_>>(), meta);
        assert_eq!(meta["prefix_reuses"], 5);
        assert_eq!(meta["prefix_reuse_rejects"], 0);
    }
}
#[derive(Default)]
struct Regions {
    entries: HashMap<Arc<str>, Entry>,
    charge: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    prefix_reuses: u64,
    prefix_reused_tokens: u64,
    prefix_reuse_rejects: u64,
}
/// A cached region offered to a missed one: its text and ids.
type Candidate = (Arc<str>, Arc<[u32]>);
pub(crate) struct RegionCache {
    regions: Mutex<Regions>,
    budget: usize,
    prefix_reuse: bool,
    token_bytes: OnceLock<Option<Vec<u16>>>,
}
impl RegionCache {
    pub(crate) fn new(budget: usize) -> Self {
        Self::with_prefix_reuse(budget, false)
    }
    pub(crate) fn with_prefix_reuse(budget: usize, prefix_reuse: bool) -> Self {
        Self { regions: Mutex::new(Regions::default()), budget, prefix_reuse, token_bytes: OnceLock::new() }
    }
    pub(crate) fn metadata(&self) -> Result<serde_json::Value, String> {
        let cache = self.regions.lock().map_err(|_| "prompt-region cache poisoned")?;
        Ok(serde_json::json!({"entries":cache.entries.len(),"entry_capacity":cache.entries.capacity(),"charged_bytes":cache.charge,
            "budget_bytes":self.budget,"hits":cache.hits,"misses":cache.misses,"prefix_reuse":self.prefix_reuse,
            "prefix_reuses":cache.prefix_reuses,"prefix_reused_tokens":cache.prefix_reused_tokens,
            "prefix_reuse_rejects":cache.prefix_reuse_rejects}))
    }
    /// The ids of `text` from the cached region sharing its longest prefix: Ok(Some((ids, tokens kept))) when the
    /// splice passed both guards, Ok(None) when no cut qualifies, Err(()) when a guard refused it.
    fn reuse_prefix(canonical: &PromptTokenizer, table: &[u16], text: &str, candidates: &[Candidate])
        -> Result<Result<Option<(Vec<u32>, usize)>, ()>, String> {
        let bytes = text.as_bytes();
        let Some((shared, ids)) = candidates.iter().map(|(key, ids)| (common_prefix(key.as_bytes(), bytes), ids))
            .max_by_key(|(shared, _)| *shared) else { return Ok(Ok(None)) };
        if shared < REUSE_MIN_BYTES { return Ok(Ok(None)); }
        let limit = shared - REUSE_MARGIN;
        let floor = limit.saturating_sub(REUSE_SCAN).max(1);
        let Some(cut) = (floor..=limit).rev().find(|&p| cut_at(bytes, p)) else { return Ok(Ok(None)) };
        if cut < floor + REUSE_OVERLAP { return Ok(Ok(None)); }
        let Some(start) = (floor..=cut - REUSE_OVERLAP).rev().find(|&p| cut_at(bytes, p)) else { return Ok(Ok(None)) };
        // Token indices of both cuts in the cached ids: each must be met exactly by the running byte position.
        let (mut at, mut first, mut last) = (0usize, None, None);
        for (index, &id) in ids.iter().enumerate() {
            if at == start { first = Some(index); }
            if at >= cut { if at == cut { last = Some(index); } break; }
            match table.get(id as usize) {
                Some(&n) if n > 0 => at += n as usize,
                _ => return Ok(Err(())),
            }
        }
        let (Some(first), Some(last)) = (first, last) else { return Ok(Err(())) };
        let tail = canonical.encode_ids(&text[start..], &[])?;
        if tail.len() < last - first || tail[..last - first] != ids[first..last] { return Ok(Err(())); }
        let mut out = Vec::with_capacity(first + tail.len());
        out.extend_from_slice(&ids[..first]);
        out.extend_from_slice(&tail);
        Ok(Ok(Some((out, first))))
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
        let table = if self.prefix_reuse { self.token_bytes.get_or_init(|| token_bytes(tokenizer)).as_deref() } else { None };
        let mut candidates: HashMap<usize, Vec<Candidate>> = HashMap::new();
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
                } else {
                    cache.misses += 1;
                    if table.is_some() && region.len() >= REUSE_MIN_BYTES {
                        // Long cached regions that open like this one, most recently used first.
                        let head = &region.as_bytes()[..64];
                        let mut similar: Vec<_> = cache.entries.iter()
                            .filter(|(text, _)| text.len() >= REUSE_MIN_BYTES && &text.as_bytes()[..64] == head)
                            .map(|(text, entry)| (entry.touched, text.clone(), entry.ids.clone())).collect();
                        similar.sort_unstable_by_key(|(touched, _, _)| std::cmp::Reverse(*touched));
                        similar.truncate(REUSE_CANDIDATES);
                        if !similar.is_empty() { candidates.insert(i, similar.into_iter().map(|(_, text, ids)| (text, ids)).collect()); }
                    }
                }
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
        // (region, ids, tokens kept from a cached region, whether a reuse guard refused the splice)
        let computed: Result<Vec<_>, String> = missing.par_iter().map(|&i| {
            let mut refused = false;
            if let (Some(table), Some(similar)) = (table, candidates.get(&i)) {
                match Self::reuse_prefix(canonical, table, regions[i], similar)? {
                    Ok(Some((ids, kept))) => return Ok((i, Arc::<[u32]>::from(ids), kept, false)),
                    Ok(None) => {}
                    Err(()) => refused = true,
                }
            }
            canonical.encode_ids(regions[i], &[]).map(|ids| (i, Arc::<[u32]>::from(ids), 0, refused))
        }).collect();
        let computed = computed?;
        drop(candidates);
        {
            let mut cache = self.regions.lock().map_err(|_| "prompt-region cache poisoned")?;
            let tick = cache.clock;
            for (i, ids, kept, refused) in &computed {
                if *kept > 0 {
                    cache.prefix_reuses += 1; cache.prefix_reused_tokens += *kept as u64;
                    eprintln!("claude prompt prefix reuse: kept {} of {} tokens of a {}-byte region", kept, ids.len(), regions[*i].len());
                }
                if *refused {
                    cache.prefix_reuse_rejects += 1;
                    eprintln!("claude prompt prefix reuse REFUSED by a guard: a {}-byte region was tokenized whole", regions[*i].len());
                }
            }
            let computed = computed.into_iter().map(|(i, ids, _, _)| (i, ids));
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
