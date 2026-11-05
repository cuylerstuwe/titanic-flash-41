//! Opt-in immutable context transport. Disabled unless explicitly configured.
//! Retains immutable messages or trusted compact representations; never responses.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

type Hash = [u8; 32];
const MAX_MESSAGES: usize = 65536;
const BLOB_OVERHEAD: usize = 256;
const SNAPSHOT_OVERHEAD: usize = 512;
const ENTRY_OVERHEAD: usize = 64;

fn hash(parts: &[&[u8]]) -> Hash {
    let mut h = Sha256::new();
    for part in parts {
        h.update(part);
    }
    h.finalize().into()
}
pub fn hex(h: &Hash) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex<const N: usize>(text: &str) -> Option<[u8; N]> {
    if text.len() != N * 2
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let mut out = [0; N];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}
pub fn root() -> Hash {
    hash(&[b"titanic-context-v1/root"])
}
pub fn message_hash(raw: &str) -> Hash {
    hash(&[b"titanic-context-v1/message\0", raw.as_bytes()])
}
pub fn extend(parent: &Hash, message: &Hash) -> Hash {
    hash(&[b"titanic-context-v1/chain\0", parent, message])
}

#[derive(Debug, Serialize)]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
}
fn invalid() -> Error {
    Error {
        status: 400,
        code: "context_invalid",
    }
}
fn missing() -> Error {
    Error {
        status: 409,
        code: "context_miss",
    }
}
fn too_large() -> Error {
    Error {
        status: 413,
        code: "context_too_large",
    }
}
type Result<T> = std::result::Result<T, Error>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Base {
    key: String,
    keep: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u32,
    namespace: String,
    id: String,
    base: Option<Base>,
    // Strings make the hashing contract independent of JSON object key order
    // and different languages' number serialization. Values are never patched.
    append: Vec<String>,
    key: String,
    request: Map<String, Value>,
}

struct Charge {
    bytes: usize,
    live: Arc<AtomicUsize>,
}
impl Charge {
    fn new(bytes: usize, live: &Arc<AtomicUsize>) -> Self {
        live.fetch_add(bytes, Ordering::Relaxed);
        Self {
            bytes,
            live: live.clone(),
        }
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}
struct Blob {
    raw: Box<str>,
    digest: Hash,
    original_bytes: usize,
    images: Vec<Hash>,
    _charge: Charge,
}
#[derive(Clone)]
struct Entry {
    blob: Arc<Blob>,
    chain: Hash,
}
struct Snapshot {
    entries: Vec<Entry>,
    _charge: Charge,
}
struct Retained {
    snapshot: Arc<Snapshot>,
    touched: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Receipt {
    pub key: String,
    pub count: usize,
    pub reused: usize,
    pub id: String,
    pub retained: bool,
}

/// Validated immutable input, without an expanded JSON allocation. It pins the
/// referenced bytes across queueing, cache eviction or a simultaneous branch.
/// Dropping a cancelled request releases those references immediately.
pub struct Prepared {
    scope: Hash,
    chain: Hash,
    hashes: Vec<(Hash, Hash)>,
    head: String,
    prefix: Option<Arc<Snapshot>>,
    keep: usize,
    tail: Vec<String>,
    expanded_bytes: usize,
    streaming: bool,
    receipt: Receipt,
}
impl Prepared {
    pub fn expanded_bytes(&self) -> usize {
        self.expanded_bytes
    }
    pub fn image_keys(&self) -> Vec<Hash> {
        self.prefix.as_ref().map_or(&[][..], |s| &s.entries[..self.keep])
            .iter().flat_map(|e| e.blob.images.iter().copied()).collect()
    }
    pub fn streaming(&self) -> bool {
        self.streaming
    }
    pub fn receipt(&self) -> &Receipt {
        &self.receipt
    }
    fn install_snapshot(&mut self, snapshot: Arc<Snapshot>) {
        self.keep = snapshot.entries.len();
        self.expanded_bytes = self.head.len() + snapshot.entries.iter().map(|e|e.blob.raw.len()).sum::<usize>()
            + self.keep.saturating_sub(1) + 2;
        self.prefix = Some(snapshot);
        self.tail.clear();
        self.hashes.clear();
        self.receipt.retained = true;
    }
    /// Call after execution admission. The caller's reservation remains alive
    /// through both materialization and canonical conversion; refusal allocates
    /// no expanded body and does not invoke the consumer.
    pub fn with_materialized<G, T, E>(
        &self,
        reserve: impl FnOnce(usize) -> std::result::Result<G, E>,
        consume: impl FnOnce(&str) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let _reservation = reserve(self.expanded_bytes)?;
        let mut body = String::with_capacity(self.expanded_bytes);
        body.push_str(&self.head);
        let old = self
            .prefix
            .as_ref()
            .map_or(&[][..], |s| &s.entries[..self.keep]);
        for (i, raw) in old
            .iter()
            .map(|e| e.blob.raw.as_ref())
            .chain(self.tail.iter().map(String::as_str))
            .enumerate()
        {
            if i != 0 {
                body.push(',');
            }
            body.push_str(raw);
        }
        body.push_str("]}");
        debug_assert_eq!(body.len(), self.expanded_bytes);
        consume(&body)
    }
}

/// Trusted representation generated only after successful canonical input
/// conversion. The input digest still identifies the original message bytes.
pub struct StoredMessage {
    pub json: String,
    pub images: Vec<Hash>,
}
impl StoredMessage {
    pub fn plain(raw: &str) -> Self { Self { json: raw.to_owned(), images: Vec::new() } }
    fn charge(&self) -> usize { self.json.len() + BLOB_OVERHEAD + self.images.capacity() * std::mem::size_of::<Hash>() }
}

#[derive(Debug, Serialize)]
pub struct Resolved {
    pub body: String,
    pub key: String,
    pub count: usize,
    pub reused: usize,
    pub id: String,
    pub retained: bool,
}
#[derive(Serialize)]
pub struct Stats {
    pub retained_bytes: usize,
    pub snapshots: usize,
    pub blobs: usize,
    pub evictions: u64,
}

/// The budget counts unique retained blobs plus conservative snapshot metadata.
/// Resolved request bodies remain the caller's ordinary bounded input workspace.
/// A production adapter must reserve that workspace BEFORE materialization.
pub struct Cache {
    max_bytes: usize,
    max_snapshots: usize,
    ttl_ms: u64,
    live: Arc<AtomicUsize>,
    snapshots: HashMap<(Hash, Hash), Retained>,
    blobs: HashMap<(Hash, Hash), Weak<Blob>>,
    evictions: u64,
}
impl Cache {
    pub fn new(max_bytes: usize, max_snapshots: usize, ttl_ms: u64) -> Self {
        Self {
            max_bytes,
            max_snapshots,
            ttl_ms,
            live: Arc::new(AtomicUsize::new(0)),
            snapshots: HashMap::new(),
            blobs: HashMap::new(),
            evictions: 0,
        }
    }
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.blobs.clear();
    }
    pub fn stats(&self) -> Stats {
        Stats {
            retained_bytes: self.live.load(Ordering::Relaxed),
            snapshots: self.snapshots.len(),
            blobs: self.blobs.values().filter(|b| b.strong_count() > 0).count(),
            evictions: self.evictions,
        }
    }
    fn prune(&mut self, now: u64) {
        self.snapshots
            .retain(|_, value| now.saturating_sub(value.touched) < self.ttl_ms);
        self.blobs.retain(|_, value| value.strong_count() > 0);
    }
    fn evict_oldest(&mut self) -> bool {
        let Some(key) = self
            .snapshots
            .iter()
            .min_by_key(|(_, v)| v.touched)
            .map(|(k, _)| *k)
        else {
            return false;
        };
        self.snapshots.remove(&key);
        self.evictions += 1;
        true
    }
    /// `principal` comes from successful authentication, never the request body.
    /// Misses and malformed input must be returned before any inference dispatch.
    pub fn prepare(
        &mut self,
        principal: &str,
        wire: &[u8],
        now: u64,
        limit: usize,
    ) -> Result<Prepared> {
        self.prepare_with_policy(principal, wire, now, limit, true)
    }
    /// Native serving commits only after canonical conversion, so encoded image
    /// bodies need not become retained cache storage even temporarily.
    pub fn prepare_unretained(&mut self, principal: &str, wire: &[u8], now: u64, limit: usize) -> Result<Prepared> {
        self.prepare_with_policy(principal, wire, now, limit, false)
    }
    fn prepare_with_policy(&mut self, principal: &str, wire: &[u8], now: u64, limit: usize, retain: bool) -> Result<Prepared> {
        // This parser is also bounded independently of expanded request length.
        if wire.len() > limit.saturating_mul(2).saturating_add(65536) {
            return Err(too_large());
        }
        let env: Envelope = serde_json::from_slice(wire).map_err(|_| invalid())?;
        if env.version != 1
            || unhex::<16>(&env.namespace).is_none()
            || unhex::<16>(&env.id).is_none()
            || env.append.len() > MAX_MESSAGES
            || env.request.contains_key("messages")
        {
            return Err(invalid());
        }
        let expected = unhex::<32>(&env.key).ok_or_else(invalid)?;
        let scope = hash(&[
            b"titanic-context-v1/scope\0",
            &(principal.len() as u64).to_be_bytes(),
            principal.as_bytes(),
            env.namespace.as_bytes(),
        ]);
        self.prune(now);
        let (base, keep) = if let Some(base) = env.base {
            let key = unhex::<32>(&base.key).ok_or_else(invalid)?;
            let retained = self.snapshots.get_mut(&(scope, key)).ok_or_else(missing)?;
            if base.keep > retained.snapshot.entries.len() {
                return Err(invalid());
            }
            // Pin the immutable prefix while materializing/evicting other entries.
            (Some(retained.snapshot.clone()), base.keep)
        } else {
            (None, 0)
        };
        if keep.saturating_add(env.append.len()) > MAX_MESSAGES {
            return Err(too_large());
        }
        let old = base.as_ref().map_or(&[][..], |s| &s.entries[..keep]);
        let mut chain = old.last().map_or_else(root, |e| e.chain);
        let mut new_hashes = Vec::with_capacity(env.append.len());
        for raw in &env.append {
            let message = serde_json::from_str::<Value>(raw).map_err(|_| invalid())?;
            if !message.is_object() { return Err(invalid()); }
            // Internal processed-image handles may enter only through an
            // authenticated retained snapshot, never from new wire definitions.
            if message.get("content").and_then(Value::as_array).is_some_and(|parts|parts.iter().any(|p|
                p["type"] == "image_url" && p["image_url"]["url"].as_str()
                    .is_some_and(|u|u.starts_with("http://titanic-context.invalid/")))) { return Err(invalid()); }
            let digest = message_hash(raw);
            chain = extend(&chain, &digest);
            new_hashes.push((digest, chain));
        }
        if chain != expected {
            return Err(invalid());
        }
        let count = keep + env.append.len();
        let mut controls = serde_json::to_string(&env.request).map_err(|_| invalid())?;
        controls.pop();
        let separator = if env.request.is_empty() { "" } else { "," };
        let head = format!("{controls}{separator}\"messages\":[");
        let raw_len = old
            .iter()
            .map(|e| e.blob.raw.len())
            .chain(env.append.iter().map(String::len))
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(too_large)?;
        let total = head
            .len()
            .checked_add(raw_len)
            .and_then(|n| n.checked_add(count.saturating_sub(1) + 2))
            .ok_or_else(too_large)?;
        let logical = old.iter().map(|e|e.blob.original_bytes).chain(env.append.iter().map(String::len))
            .try_fold(head.len() + count.saturating_sub(1) + 2, usize::checked_add).ok_or_else(too_large)?;
        if logical > limit || total > limit { return Err(too_large()); }
        let streaming = match env.request.get("stream") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(stream)) => *stream,
            _ => return Err(invalid()),
        };
        let mut result = Prepared {
            scope, chain, hashes: new_hashes,
            head,
            prefix: base.clone(),
            keep,
            tail: env.append,
            expanded_bytes: total,
            streaming,
            receipt: Receipt {
                key: env.key,
                id: env.id,
                count,
                reused: keep,
                retained: false,
            },
        };
        if retain && self.commit(&result, now, |raw| Some(StoredMessage::plain(raw))) {
            result.install_snapshot(self.snapshots[&(scope,chain)].snapshot.clone());
        }
        Ok(result)
    }
    /// Optional post-conversion retention. Failed transformation or budget
    /// refusal leaves the request valid but unretained. Pinned ancestors remain
    /// charged even if their cache handles are evicted while making room.
    pub fn commit(&mut self, prepared: &Prepared, now: u64,
        mut transform: impl FnMut(&str) -> Option<StoredMessage>) -> bool {
        let (scope,chain)=(prepared.scope,prepared.chain);
        if let Some(current)=self.snapshots.get_mut(&(scope,chain)) { current.touched=now;return true; }
        if self.max_snapshots == 0 || self.ttl_ms == 0 { return false; }
        let old=prepared.prefix.as_ref().map_or(&[][..],|s|&s.entries[..prepared.keep]);
        let existing:Vec<_>=prepared.hashes.iter().map(|(h,_)|self.blobs.get(&(scope,*h)).and_then(Weak::upgrade)).collect();
        let mut staged=HashMap::new();
        for ((digest,_),(raw,prior)) in prepared.hashes.iter().zip(prepared.tail.iter().zip(&existing)) {
            if prior.is_none() && !staged.contains_key(digest) {
                let Some(representation)=transform(raw) else { return false; };
                staged.insert(*digest,representation);
            }
        }
        let metadata=SNAPSHOT_OVERHEAD+ENTRY_OVERHEAD*prepared.receipt.count;
        let need=metadata+staged.values().map(StoredMessage::charge).sum::<usize>();
        while self.snapshots.len()>=self.max_snapshots || self.live.load(Ordering::Relaxed).saturating_add(need)>self.max_bytes {
            if !self.evict_oldest(){break;}
        }
        if self.live.load(Ordering::Relaxed).saturating_add(need)>self.max_bytes {self.prune(now);return false;}
        let mut entries=old.to_vec();
        for ((raw,(digest,chain)),prior) in prepared.tail.iter().zip(&prepared.hashes).zip(existing) {
            let blob=prior.or_else(||self.blobs.get(&(scope,*digest)).and_then(Weak::upgrade)).unwrap_or_else(|| {
                let representation=staged.remove(digest).expect("staged unique message");
                let charge=representation.charge();
                let blob=Arc::new(Blob { raw:representation.json.into_boxed_str(),digest:*digest,
                    original_bytes:raw.len(), images:representation.images, _charge:Charge::new(charge,&self.live) });
                self.blobs.insert((scope,*digest),Arc::downgrade(&blob));blob
            });
            debug_assert_eq!(blob.digest,*digest);entries.push(Entry{blob,chain:*chain});
        }
        let snapshot=Arc::new(Snapshot {entries,_charge:Charge::new(metadata,&self.live)});
        self.snapshots.insert((scope,chain),Retained{touched:now,snapshot});
        self.prune(now);assert!(self.live.load(Ordering::Relaxed)<=self.max_bytes);true
    }
    /// Convenience for the CPU diagnostic driver. Serving must use `prepare`
    /// and its guarded materialization after ordinary lifetime admission.
    pub fn resolve(
        &mut self,
        principal: &str,
        wire: &[u8],
        now: u64,
        limit: usize,
    ) -> Result<Resolved> {
        let prepared = self.prepare(principal, wire, now, limit)?;
        let body =
            prepared.with_materialized(|_| Ok::<_, Error>(()), |body| Ok(body.to_owned()))?;
        let r = prepared.receipt;
        Ok(Resolved {
            body,
            key: r.key,
            id: r.id,
            count: r.count,
            reused: r.reused,
            retained: r.retained,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn wire(
        messages: &[Value],
        base: Option<(&str, usize)>,
        append: &[Value],
        namespace: &str,
    ) -> Vec<u8> {
        let mut chain = root();
        for m in messages {
            chain = extend(&chain, &message_hash(&m.to_string()));
        }
        serde_json::to_vec(&json!({"version":1,"namespace":namespace,"id":"00000000000000000000000000000001",
            "base":base.map(|(key,keep)|json!({"key":key,"keep":keep})),"append":append.iter().map(Value::to_string).collect::<Vec<_>>(),
            "key":hex(&chain),"request":{"model":"test","temperature":1,"top_p":1,"seed":18446744073709551615u64}})).unwrap()
    }
    const NS: &str = "0123456789abcdef0123456789abcdef";
    #[test]
    #[ignore = "finite CPU maintenance cost screen, synthetic messages only"]
    fn maintenance_latency_screen() {
        use std::time::Instant;
        for count in [1000usize, 10000, 32000] {
            let mut c = Cache::new(64 << 20, 16, 1_800_000);
            let mut messages: Vec<Value> = (0..count).map(|i|json!({"role":"user",
                "content":format!("Public synthetic message {i}: inspect the next component and report its result.")})).collect();
            let full = wire(&messages, None, &messages, NS);
            let first = c.prepare("authenticated-user", &full, 0, 128 << 20).unwrap();
            let mut key = first.receipt().key.clone();
            drop(first);
            let mut turns = Vec::new();
            for turn in 1..=18u64 {
                let tail = json!({"role":"tool","content":format!("Synthetic result {turn}")});
                let keep = messages.len(); messages.push(tail.clone());
                let delta = wire(&messages, Some((&key, keep)), &[tail], NS);
                let at = Instant::now();
                let prepared = c.prepare_unretained("authenticated-user", &delta, turn, 128 << 20).unwrap();
                let prepare_ms = at.elapsed().as_secs_f64()*1000.;
                let at = Instant::now();
                assert!(c.commit(&prepared, turn, |raw|Some(StoredMessage::plain(raw))));
                let commit_ms = at.elapsed().as_secs_f64()*1000.;
                key = prepared.receipt().key.clone();
                drop(prepared);
                turns.push(json!({"turn":turn,"prepare_ms":prepare_ms,"commit_ms":commit_ms}));
            }
            let at = Instant::now();
            for _ in 0..100 { c.prune(100); }
            let no_expiry_prune_ms = at.elapsed().as_secs_f64()*10.;
            let retained = c.stats();
            let at = Instant::now(); c.prune(2_000_000);
            let expire_all_ms = at.elapsed().as_secs_f64()*1000.;
            assert_eq!(c.stats().retained_bytes,0);
            assert_eq!(c.stats().snapshots,0);
            assert_eq!(c.stats().blobs,0);
            println!("{}", json!({"screen":"context_maintenance","messages":count,"turns":turns,
                "no_expiry_prune_ms":no_expiry_prune_ms,"expire_all_ms":expire_all_ms,"retained":retained}));
        }
    }
    fn resolve(c: &mut Cache, w: &[u8], now: u64) -> Resolved {
        c.resolve("authenticated-user", w, now, 1 << 20).unwrap()
    }
    #[test]
    fn compact_commit_keeps_original_identity_and_size_without_retaining_payload() {
        let mut c=Cache::new(8192,4,1000);
        let message=json!({"role":"user","content":"original encoded fixture ".repeat(5000)});
        let full=wire(&[message.clone()],None,&[message.clone()],NS);
        let prepared=c.prepare_unretained("authenticated-user",&full,0,1<<20).unwrap();
        assert_eq!(c.stats().retained_bytes,0,"preflight must not retain source data");
        let key=prepared.receipt().key.clone();
        assert!(c.commit(&prepared,1,|_|Some(StoredMessage{json:r#"{"role":"user","content":"trusted compact fixture"}"#.into(),images:vec![[9;32]]})));
        assert!(c.stats().retained_bytes<2048);
        let delta=wire(&[message],Some((&key,1)),&[],NS);
        let next=c.prepare_unretained("authenticated-user",&delta,2,1<<20).unwrap();
        assert!(next.expanded_bytes()<1024);assert_eq!(next.receipt().key,key);assert_eq!(next.image_keys(),vec![[9;32]]);
        assert_eq!(c.prepare_unretained("authenticated-user",&delta,3,4096).err().unwrap().code,"context_too_large",
            "compaction must not silently change the logical input-size contract");
        c.clear();assert!(c.stats().retained_bytes>0);
        drop(next);assert_eq!(c.stats().retained_bytes,0);
        // A full resync supplies original definitions even when a cached key
        // exists; it must not implicitly depend on the compact image payload.
        assert!(c.commit(&prepared,4,|_|Some(StoredMessage::plain("{}"))));
        let retry=c.prepare_unretained("authenticated-user",&full,5,1<<20).unwrap();
        assert!(retry.expanded_bytes()>100000);assert!(retry.image_keys().is_empty());
    }
    #[test]
    fn queued_input_survives_eviction_but_releases_its_charge_after_cancellation() {
        let mut c = Cache::new(20000, 2, 1000);
        let a = json!({"role":"user","content":"image-encoded-fixture".repeat(200)});
        let first = resolve(&mut c, &wire(&[a.clone()], None, &[a.clone()], NS), 0);
        let w = wire(&[a], Some((&first.key, 1)), &[], NS);
        let queued = c.prepare("authenticated-user", &w, 1, 1 << 20).unwrap();
        assert_eq!(queued.expanded_bytes(), first.body.len());
        assert!(!queued.streaming());
        c.clear();
        assert_eq!(c.stats().snapshots, 0);
        assert!(
            c.stats().retained_bytes > 0,
            "queued references must remain charged"
        );
        let active = Arc::new(AtomicUsize::new(0));
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        queued
            .with_materialized(
                |size| {
                    assert_eq!(size, first.body.len());
                    active.fetch_add(1, Ordering::Relaxed);
                    Ok::<_, ()>(Guard(active.clone()))
                },
                |body| {
                    assert_eq!(active.load(Ordering::Relaxed), 1);
                    assert_eq!(body, first.body);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(active.load(Ordering::Relaxed), 0);
        let refused: std::result::Result<(), &str> = queued.with_materialized(
            |_| Err::<(), _>("reserve"),
            |_| panic!("must not consume after refused memory admission"),
        );
        assert_eq!(refused, Err("reserve"));
        // An unpolled/cancelled future simply drops this same owned handle.
        drop(queued);
        assert_eq!(c.stats().retained_bytes, 0);
        assert_eq!(
            c.resolve("authenticated-user", &w, 2, 1 << 20)
                .unwrap_err()
                .code,
            "context_miss"
        );
    }
    #[test]
    fn exact_extension_rewind_fork_and_changed_controls() {
        let mut c = Cache::new(1 << 20, 8, 1000);
        let a = json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}},{"type":"text","text":"é 🦀 \"\\\n"}]});
        let b = json!({"role":"assistant","content":"answer","tool_calls":[{"id":"c1","function":{"name":"test","arguments":"{\"x\":1}"}}]});
        let d = json!({"role":"tool","tool_call_id":"c1","content":"result"});
        let first = resolve(
            &mut c,
            &wire(&[a.clone(), b.clone()], None, &[a.clone(), b.clone()], NS),
            0,
        );
        let extended = resolve(
            &mut c,
            &wire(
                &[a.clone(), b.clone(), d.clone()],
                Some((&first.key, 2)),
                &[d.clone()],
                NS,
            ),
            1,
        );
        assert_eq!(
            serde_json::from_str::<Value>(&extended.body).unwrap()["messages"],
            json!([a, b, d])
        );
        assert_eq!(extended.reused, 2);
        let branch = json!({"role":"user","content":"edited user turn"});
        let branched = resolve(
            &mut c,
            &wire(
                &[a.clone(), branch.clone()],
                Some((&first.key, 1)),
                &[branch],
                NS,
            ),
            2,
        );
        assert_ne!(branched.key, extended.key);
        let original = resolve(
            &mut c,
            &wire(&[a.clone(), b.clone()], Some((&first.key, 2)), &[], NS),
            3,
        );
        assert_eq!(original.body, first.body);
        let mut altered: Value =
            serde_json::from_slice(&wire(&[a, b], Some((&first.key, 2)), &[], NS)).unwrap();
        altered["request"]["temperature"] = json!(0);
        altered["request"]["reasoning_effort"] = json!("max");
        altered["request"]["tools"] = json!([]);
        let got = resolve(&mut c, &serde_json::to_vec(&altered).unwrap(), 4);
        let body: Value = serde_json::from_str(&got.body).unwrap();
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["reasoning_effort"], "max");
        assert_eq!(body["seed"], json!(18446744073709551615u64));
    }
    #[test]
    fn auth_namespace_ttl_clear_and_wrong_key_never_silently_shorten() {
        let mut c = Cache::new(1 << 20, 8, 10);
        let a = json!({"role":"user","content":"private"});
        let f = resolve(&mut c, &wire(&[a.clone()], None, &[a.clone()], NS), 0);
        let w = wire(&[a.clone()], Some((&f.key, 1)), &[], NS);
        assert_eq!(
            c.resolve("other-user", &w, 1, 1 << 20).unwrap_err().code,
            "context_miss"
        );
        assert_eq!(
            c.resolve(
                "authenticated-user",
                &wire(
                    &[a.clone()],
                    Some((&f.key, 1)),
                    &[],
                    "11111111111111111111111111111111"
                ),
                1,
                1 << 20
            )
            .unwrap_err()
            .code,
            "context_miss"
        );
        assert_eq!(
            c.resolve("authenticated-user", &w, 10, 1 << 20)
                .unwrap_err()
                .code,
            "context_miss"
        );
        let mut bad: Value =
            serde_json::from_slice(&wire(&[a.clone()], None, &[a.clone()], NS)).unwrap();
        bad["key"] = json!(hex(&root()));
        assert_eq!(
            c.resolve(
                "authenticated-user",
                &serde_json::to_vec(&bad).unwrap(),
                11,
                1 << 20
            )
            .unwrap_err()
            .code,
            "context_invalid"
        );
        assert_eq!(c.stats().snapshots, 0);
        resolve(&mut c, &wire(&[a.clone()], None, &[a], NS), 12);
        c.clear();
        assert_eq!(c.stats().retained_bytes, 0);
        assert_eq!(
            c.resolve("authenticated-user", &w, 13, 1 << 20)
                .unwrap_err()
                .code,
            "context_miss"
        );
    }
    #[test]
    fn retained_memory_is_shared_bounded_and_optional() {
        let mut c = Cache::new(15000, 4, 10000);
        let image = json!({"role":"user","content":"x".repeat(6000)});
        let first = resolve(
            &mut c,
            &wire(&[image.clone()], None, &[image.clone()], NS),
            0,
        );
        for n in 1..80 {
            let turn = json!({"role":"user","content":format!("branch {n}")});
            // Refresh the shared parent, so a bounded sibling eviction does not
            // select the parent snapshot needed to create the next branch.
            resolve(
                &mut c,
                &wire(&[image.clone()], Some((&first.key, 1)), &[], NS),
                n * 2,
            );
            let branch = resolve(
                &mut c,
                &wire(
                    &[image.clone(), turn.clone()],
                    Some((&first.key, 1)),
                    &[turn],
                    NS,
                ),
                n * 2 + 1,
            );
            assert!(branch.retained);
            assert!(c.stats().retained_bytes <= 15000);
            assert!(c.stats().snapshots <= 4);
        }
        assert!(c.stats().evictions > 0);
        assert!(c.stats().blobs <= 4);
        let mut tiny = Cache::new(100, 1, 1000);
        let got = resolve(&mut tiny, &wire(&[image.clone()], None, &[image], NS), 0);
        assert!(!got.retained);
        assert_eq!(tiny.stats().retained_bytes, 0);
    }
    #[test]
    fn expanded_size_and_malformed_protocol_are_checked_before_admission() {
        let mut c = Cache::new(1 << 20, 8, 1000);
        let a = json!({"role":"user","content":"x".repeat(1000)});
        let w = wire(&[a.clone()], None, &[a], NS);
        assert_eq!(
            c.resolve("authenticated-user", &w, 0, 100)
                .unwrap_err()
                .code,
            "context_too_large"
        );
        assert_eq!(c.stats().snapshots, 0);
        let f = resolve(&mut c, &w, 1);
        let mut d: Value = serde_json::from_slice(&w).unwrap();
        d["base"] = json!({"key":f.key,"keep":9});
        d["append"] = json!([]);
        assert_eq!(
            c.resolve(
                "authenticated-user",
                &serde_json::to_vec(&d).unwrap(),
                2,
                1 << 20
            )
            .unwrap_err()
            .code,
            "context_invalid"
        );
        d["base"] = Value::Null;
        d["append"] = json!(["[]"]);
        assert_eq!(
            c.resolve(
                "authenticated-user",
                &serde_json::to_vec(&d).unwrap(),
                2,
                1 << 20
            )
            .unwrap_err()
            .code,
            "context_invalid"
        );
    }
}
