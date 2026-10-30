//! Bounded, volatile operational evidence for a stationary HTTP worker.
//! A flush marks a complete JSON record. Normal serving never opens an event
//! file; disk tracing is an explicit diagnostic option, with bounded rotation.
//! Request/incident records also have their own ring so decode traffic cannot
//! evict cache/admission decisions. Both rings are lost when the process exits.
use std::{
    collections::VecDeque,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

const MEMORY_BYTES: usize = 128 * 1024;
const RECORD_BYTES: usize = 16 * 1024;
const RECORD_COUNT: usize = 512;

#[derive(Clone, Default)]
struct MemoryRing {
    records: VecDeque<Arc<[u8]>>,
    bytes: usize,
    evicted: u64,
    oversized: u64,
}

impl MemoryRing {
    fn push(&mut self, record: Arc<[u8]>) {
        while self.bytes + record.len() > MEMORY_BYTES || self.records.len() >= RECORD_COUNT {
            let old = self.records.pop_front().unwrap();
            self.bytes -= old.len();
            self.evicted += 1;
        }
        self.bytes += record.len();
        self.records.push_back(record);
    }

    fn snapshot(self) -> serde_json::Value {
        serde_json::json!({
            "payload_limit_bytes":MEMORY_BYTES,"record_limit_bytes":RECORD_BYTES,
            "record_count_limit":RECORD_COUNT,"retained_payload_bytes":self.bytes,
            "evicted_records":self.evicted,"oversized_records":self.oversized,
            "records":self.records.iter().map(|record|serde_json::from_slice::<serde_json::Value>(record)
                .unwrap_or_else(|_|serde_json::json!({"invalid_record":true}))).collect::<Vec<_>>()})
    }
}

#[derive(Clone, Default)]
struct MemoryHistory {
    recent: MemoryRing,
    incidents: MemoryRing,
}

fn retain_incident(event: &serde_json::Value) -> bool {
    match event["type"].as_str() {
        // The first token record carries the complete prefill/cache receipt.
        Some("tokens") => event.get("prompt_tokens").is_some(),
        Some("prefill_progress") => event["request_started"] == true || event["cancelled"] == true,
        Some("idle" | "hold" | "output_wait") => false,
        // Preserve new operational event types unless explicitly high-volume.
        _ => true,
    }
}

fn write_record(writer: &mut impl Write, event: &serde_json::Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, event)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

struct MemoryLog {
    history: Arc<Mutex<MemoryHistory>>,
    pending: Vec<u8>,
    oversized: bool,
    incident: bool,
}

impl MemoryLog {
    fn new() -> Self {
        Self {
            history: Arc::new(Mutex::new(MemoryHistory::default())),
            pending: Vec::new(),
            oversized: false,
            incident: false,
        }
    }

    fn record(&mut self, event: &serde_json::Value) -> io::Result<()> {
        self.incident = retain_incident(event);
        write_record(self, event)
    }

    fn snapshot(history: &Mutex<MemoryHistory>) -> serde_json::Value {
        // Clone only bounded Arc references under the writer lock, not bytes or
        // parsed JSON. The two rings share record storage when both retain it.
        let history = {
            history
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        };
        let total_bytes = history.recent.bytes + history.incidents.bytes;
        // Keep the original top-level recent-record schema for existing clients.
        let mut snapshot = history.recent.snapshot();
        snapshot["mode"] = "memory".into();
        snapshot["persisted"] = false.into();
        snapshot["request_incidents"] = history.incidents.snapshot();
        snapshot["total_payload_limit_bytes"] = (2 * MEMORY_BYTES).into();
        // Conservative sum: records present in both rings count twice here.
        snapshot["total_retained_payload_bytes"] = total_bytes.into();
        snapshot
    }
}

impl Write for MemoryLog {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.oversized {
            if bytes.len() > RECORD_BYTES - self.pending.len() {
                self.pending.clear();
                self.oversized = true;
            } else {
                let needed = self.pending.len() + bytes.len();
                if needed > self.pending.capacity() {
                    let capacity = (self.pending.capacity().max(512) * 2)
                        .max(needed)
                        .min(RECORD_BYTES);
                    self.pending.reserve_exact(capacity - self.pending.len());
                }
                self.pending.extend_from_slice(bytes);
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() && !self.oversized {
            return Ok(());
        }
        let incident = std::mem::take(&mut self.incident);
        let mut history = self
            .history
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.oversized {
            history.recent.oversized += 1;
            if incident {
                history.incidents.oversized += 1;
            }
            self.oversized = false;
            return Ok(());
        }
        let record: Arc<[u8]> = std::mem::take(&mut self.pending).into();
        if incident {
            history.incidents.push(record.clone());
        }
        history.recent.push(record);
        Ok(())
    }
}

enum Sink {
    Memory(MemoryLog),
    Disk(DiskLog),
    Off,
}

struct Diagnostics {
    mode: &'static str,
    history: Option<Arc<Mutex<MemoryHistory>>>,
}
static DIAGNOSTICS: OnceLock<Diagnostics> = OnceLock::new();

pub(crate) fn snapshot() -> serde_json::Value {
    match DIAGNOSTICS.get() {
        Some(Diagnostics {
            history: Some(history),
            ..
        }) => MemoryLog::snapshot(history),
        Some(diagnostics) => serde_json::json!({"mode":diagnostics.mode,
            "persisted":diagnostics.mode=="disk","records":[]}),
        None => serde_json::json!({"mode":"not_initialized","persisted":false,"records":[]}),
    }
}

pub(crate) struct EventLog {
    sink: Sink,
}

impl EventLog {
    pub(crate) fn from_env(path: &Path) -> anyhow::Result<Self> {
        let mode = std::env::var("CANADA_EVENT_LOG_MODE").or_else(|error| match error {
            std::env::VarError::NotPresent => Ok("memory".to_owned()),
            other => Err(other),
        })?;
        let mib = match std::env::var("CANADA_EVENT_LOG_MIB") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(error.into()),
        };
        let log = Self::open(path, &mode, mib.as_deref())?;
        let diagnostics = match &log.sink {
            Sink::Memory(log) => Diagnostics {
                mode: "memory",
                history: Some(log.history.clone()),
            },
            Sink::Disk(_) => Diagnostics {
                mode: "disk",
                history: None,
            },
            Sink::Off => Diagnostics {
                mode: "off",
                history: None,
            },
        };
        anyhow::ensure!(
            DIAGNOSTICS.set(diagnostics).is_ok(),
            "worker event log already initialized"
        );
        Ok(log)
    }

    fn open(path: &Path, mode: &str, mib: Option<&str>) -> anyhow::Result<Self> {
        let sink = match mode {
            "memory" => Sink::Memory(MemoryLog::new()),
            "off" => Sink::Off,
            "disk" => {
                let mib: u64 = mib.unwrap_or("64").parse()?;
                anyhow::ensure!(
                    (1..=4096).contains(&mib),
                    "event log limit must be 1..4096 MiB"
                );
                Sink::Disk(DiskLog::open(path, mib * 1024 * 1024)?)
            }
            _ => anyhow::bail!("event log mode must be memory, disk, or off"),
        };
        Ok(Self { sink })
    }
    // Classify the already-redacted value at the producer boundary. No second
    // parse or extra serialization is added to the per-token hot path.
    pub(crate) fn record(&mut self, event: &serde_json::Value) -> io::Result<()> {
        match &mut self.sink {
            Sink::Memory(log) => log.record(event),
            Sink::Disk(log) => write_record(log, event),
            Sink::Off => Ok(()),
        }
    }

    pub(crate) fn worker_failed(&mut self, request_id: Option<u64>) {
        // Keep the original error/retirement path even if diagnostic I/O fails.
        // Never copy arbitrary error text (which may contain request data).
        let _ = self.record(&serde_json::json!({"type":"worker_failure",
            "request_id":request_id,"reason":"native_worker_group_failed"}));
    }
}

struct DiskLog {
    file: Option<BufWriter<File>>,
    path: PathBuf,
    limit: u64,
    bytes: u64,
    rotate_next: bool,
}

impl DiskLog {
    fn open(path: &Path, limit: u64) -> io::Result<Self> {
        if limit == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "zero event log limit",
            ));
        }
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self {
            file: Some(BufWriter::new(file)),
            path: path.to_owned(),
            limit,
            bytes: 0,
            rotate_next: false,
        })
    }

    fn archive(&self, index: u8) -> PathBuf {
        let mut name = self.path.as_os_str().to_owned();
        name.push(format!(".{index}"));
        PathBuf::from(name)
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.as_mut().unwrap().flush()?;
        // Four files in total: the active file and three previous segments.
        match std::fs::remove_file(self.archive(3)) {
            Ok(()) => (),
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error),
        }
        for index in (1..3).rev() {
            match std::fs::rename(self.archive(index), self.archive(index + 1)) {
                Ok(()) => (),
                Err(error) if error.kind() == io::ErrorKind::NotFound => (),
                Err(error) => return Err(error),
            }
        }
        std::fs::rename(&self.path, self.archive(1))?;
        let next = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)?;
        self.file = Some(BufWriter::new(next));
        self.bytes = 0;
        self.rotate_next = false;
        Ok(())
    }
}

impl Write for DiskLog {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        if self.rotate_next {
            self.rotate()?;
        }
        let count = self.file.as_mut().unwrap().write(bytes)?;
        self.bytes += count as u64;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.as_mut().unwrap().flush()?;
        self.rotate_next = self.bytes >= self.limit;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "canada-event-log-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn rotation_preserves_complete_records_and_bounds_history() {
        let directory = Directory::new();
        let path = directory.0.join("events.jsonl");
        let mut log = DiskLog::open(&path, 24).unwrap();
        for index in 0..10 {
            serde_json::to_writer(
                &mut log,
                &serde_json::json!({"index":index,"payload":"longer than one segment"}),
            )
            .unwrap();
            log.write_all(b"\n").unwrap();
            log.flush().unwrap();
        }
        let mut indices = Vec::new();
        for segment in [log.archive(3), log.archive(2), log.archive(1), path] {
            for line in std::fs::read_to_string(segment).unwrap().lines() {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                indices.push(value["index"].as_u64().unwrap());
            }
        }
        assert_eq!(indices, [6, 7, 8, 9]);
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 4);
    }

    #[test]
    fn memory_and_off_never_open_a_file() {
        let directory = Directory::new();
        // The parent does not exist: either mode must work without touching it.
        let path = directory.0.join("absent/events.jsonl");
        for mode in ["memory", "off"] {
            let mut log = EventLog::open(&path, mode, Some("64")).unwrap();
            for _ in 0..100 {
                log.record(&serde_json::json!({"type":"tokens","prompt_tokens":9000}))
                    .unwrap();
                log.record(&serde_json::json!({"type":"tokens","frontier":9001}))
                    .unwrap();
            }
            log.worker_failed(Some(1));
        }
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
    }

    #[test]
    fn ram_ring_bounds_payload_and_count_and_exposes_only_flushed_records() {
        for payload in ["x".repeat(1024), String::new()] {
            let mut log = MemoryLog::new();
            for index in 0..2000 {
                serde_json::to_writer(
                    &mut log,
                    &serde_json::json!({"index":index,"payload":payload}),
                )
                .unwrap();
                log.write_all(b"\n").unwrap();
                log.flush().unwrap();
            }
            log.write_all(b"{\"partial\":").unwrap();
            let snapshot = MemoryLog::snapshot(&log.history);
            let records = snapshot["records"].as_array().unwrap();
            assert!(records.len() <= RECORD_COUNT);
            assert!(snapshot["retained_payload_bytes"].as_u64().unwrap() <= MEMORY_BYTES as u64);
            assert_eq!(records.last().unwrap()["index"], 1999);
            assert_eq!(
                snapshot["evicted_records"].as_u64().unwrap() as usize + records.len(),
                2000
            );
            assert!(
                records
                    .windows(2)
                    .all(|pair| pair[1]["index"].as_u64().unwrap()
                        == pair[0]["index"].as_u64().unwrap() + 1)
            );
        }
    }

    #[test]
    fn oversized_record_is_discarded_without_growth_or_poisoning_the_next_record() {
        let mut log = MemoryLog::new();
        for _ in 0..100 {
            log.write_all(&vec![b'x'; RECORD_BYTES]).unwrap();
        }
        assert!(log.pending.capacity() <= RECORD_BYTES);
        log.flush().unwrap();
        log.write_all(b"{\"ok\":true}\n").unwrap();
        log.flush().unwrap();
        let snapshot = MemoryLog::snapshot(&log.history);
        assert_eq!(snapshot["oversized_records"], 1);
        assert_eq!(snapshot["records"], serde_json::json!([{"ok":true}]));
    }

    #[test]
    fn request_receipts_survive_long_decode_and_keep_existing_recent_view() {
        let mut log = MemoryLog::new();
        let mut expected = Vec::new();
        for id in 1..=3 {
            let begin = serde_json::json!({"type":"tokens","request_id":id,
                "prompt_tokens":124000,"cached_tokens":123904,"request_started":true,
                "retained_session_cache":{"hit":true,"exact_prompt_hit":true}});
            log.record(&begin).unwrap();
            expected.push(begin);
            for step in 0..512 {
                log.record(&serde_json::json!({"type":"tokens","request_id":id,
                    "step":step,"diagnostic_padding":"x".repeat(768)}))
                    .unwrap();
            }
            let finish = serde_json::json!({"type":"finish","request_id":id,"reason":"length"});
            log.record(&finish).unwrap();
            expected.push(finish);
        }
        let snapshot = MemoryLog::snapshot(&log.history);
        assert_eq!(
            snapshot["request_incidents"]["records"],
            serde_json::json!(expected)
        );
        assert_eq!(snapshot["request_incidents"]["evicted_records"], 0);
        assert!(snapshot["evicted_records"].as_u64().unwrap() > 1000);
        assert!(
            snapshot["records"]
                .as_array()
                .unwrap()
                .iter()
                .all(|record| record["request_id"] == 3)
        );
        assert_eq!(
            snapshot["records"].as_array().unwrap().last(),
            expected.last()
        );
        assert_eq!(snapshot["mode"], "memory");
        assert_eq!(snapshot["persisted"], false);
    }

    #[test]
    fn incident_admission_keeps_starts_failures_and_cache_decisions_not_progress_chatter() {
        let mut log = MemoryLog::new();
        let expected = vec![
            serde_json::json!({"type":"ready","rank":0}),
            serde_json::json!({"type":"prefill_progress","request_started":true,"prompt_tokens":124000}),
            serde_json::json!({"type":"prefill_progress","cancelled":true}),
            serde_json::json!({"type":"tokens","prompt_tokens":124000}),
            serde_json::json!({"type":"input_refused","reason":"input_capacity"}),
            serde_json::json!({"type":"input_reclaim","freed_bytes":4096}),
            serde_json::json!({"type":"park","parked":false}),
            serde_json::json!({"type":"park","parked":true}),
            serde_json::json!({"type":"resume","request_id":1}),
            serde_json::json!({"type":"discard","request_id":2}),
            serde_json::json!({"type":"priority","hold":true}),
            serde_json::json!({"type":"finish","reason":"cancelled"}),
            serde_json::json!({"type":"worker_failure","reason":"native_worker_group_failed"}),
            serde_json::json!({"type":"shutdown"}),
            serde_json::json!({"type":"future_operational_event"}),
        ];
        for event in &expected {
            log.record(event).unwrap();
            for kind in ["tokens", "prefill_progress", "idle", "hold", "output_wait"] {
                log.record(&serde_json::json!({"type":kind,"request_id":1}))
                    .unwrap();
            }
        }
        let snapshot = MemoryLog::snapshot(&log.history);
        assert_eq!(
            snapshot["request_incidents"]["records"],
            serde_json::json!(expected)
        );
        assert_eq!(
            snapshot["records"].as_array().unwrap().len(),
            expected.len() * 6
        );
    }

    #[test]
    fn incident_ring_has_independent_byte_count_and_record_limits() {
        for padding in ["x".repeat(1024), String::new()] {
            let mut log = MemoryLog::new();
            for request in 0..2000 {
                log.record(
                    &serde_json::json!({"type":"finish","request_id":request,"padding":padding}),
                )
                .unwrap();
            }
            let before = MemoryLog::snapshot(&log.history);
            let incidents = &before["request_incidents"];
            let retained = incidents["records"].as_array().unwrap().len();
            assert!(retained <= RECORD_COUNT);
            assert!(incidents["retained_payload_bytes"].as_u64().unwrap() <= MEMORY_BYTES as u64);
            assert_eq!(
                incidents["evicted_records"].as_u64().unwrap() as usize + retained,
                2000
            );
            assert_eq!(
                incidents["records"].as_array().unwrap().last().unwrap()["request_id"],
                1999
            );
            if padding.is_empty() {
                assert_eq!(retained, RECORD_COUNT);
            } else {
                assert!(retained < RECORD_COUNT);
            }
            // Completely cycling the recent ring cannot change the other ring.
            for step in 0..2000 {
                log.record(&serde_json::json!({"type":"tokens","step":step,"padding":padding}))
                    .unwrap();
            }
            let after = MemoryLog::snapshot(&log.history);
            assert_eq!(after["request_incidents"], *incidents);
            assert!(
                after["total_retained_payload_bytes"].as_u64().unwrap() <= 2 * MEMORY_BYTES as u64
            );
            assert_eq!(after["total_payload_limit_bytes"], 2 * MEMORY_BYTES);
        }
    }

    #[test]
    fn oversized_incidents_are_counted_without_losing_the_following_valid_record() {
        let mut log = MemoryLog::new();
        log.record(&serde_json::json!({"type":"tokens","prompt_tokens":1,
            "oversized":"x".repeat(RECORD_BYTES)}))
            .unwrap();
        log.record(&serde_json::json!({"type":"tokens","step":1,
            "oversized":"x".repeat(RECORD_BYTES)}))
            .unwrap();
        assert!(log.pending.capacity() <= RECORD_BYTES);
        let finish = serde_json::json!({"type":"finish","reason":"stop"});
        log.record(&finish).unwrap();
        let snapshot = MemoryLog::snapshot(&log.history);
        assert_eq!(snapshot["oversized_records"], 2);
        assert_eq!(snapshot["request_incidents"]["oversized_records"], 1);
        assert_eq!(snapshot["records"], serde_json::json!([finish]));
        assert_eq!(
            snapshot["request_incidents"]["records"],
            snapshot["records"]
        );
    }

    #[test]
    fn explicit_disk_mode_preserves_one_copy_of_each_record() {
        let directory = Directory::new();
        let path = directory.0.join("events.jsonl");
        let mut log = EventLog::open(&path, "disk", Some("1")).unwrap();
        let begin = serde_json::json!({"type":"tokens","prompt_tokens":9000});
        let step = serde_json::json!({"type":"tokens","frontier":9001});
        log.record(&begin).unwrap();
        log.record(&step).unwrap();
        log.worker_failed(Some(1));
        let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            records,
            vec![
                begin,
                step,
                serde_json::json!({"type":"worker_failure",
            "request_id":1,"reason":"native_worker_group_failed"})
            ]
        );
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
    }

    #[test]
    fn initial_open_never_overwrites_prior_evidence() {
        let directory = Directory::new();
        let path = directory.0.join("events.jsonl");
        std::fs::write(&path, "prior").unwrap();
        assert!(DiskLog::open(&path, 64).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "prior");
    }
}
