//! Optional host-counter deltas around existing encoder chunks. These are
//! diagnostic metadata, never model arithmetic or a reason to retry a chunk.
//! Host PSI/vmstat are machine-wide; cgroup and thread counters have narrower
//! scope. Missing counters stay missing, rather than being reported as zero.
use std::{collections::BTreeMap, fs, time::Instant};

pub(crate) struct Snapshot {
    values: BTreeMap<String,u64>,
    unavailable: Vec<String>,
    read_ms: f64,
}

impl Snapshot {
    pub(crate) fn read()->Self {
        let started=Instant::now();
        let mut s=Self {values:BTreeMap::new(),unavailable:Vec::new(),read_ms:0.};
        for (path,prefix,keys) in [
            ("/sys/fs/cgroup/cpu.stat","cgroup_cpu",&["usage_usec","user_usec","system_usec","nr_periods","nr_throttled","throttled_usec"][..]),
            ("/sys/fs/cgroup/memory.events","cgroup_memory",&["low","high","max","oom","oom_kill"][..]),
            ("/proc/thread-self/status","thread",&["voluntary_ctxt_switches","nonvoluntary_ctxt_switches"][..]),
            ("/proc/self/io","process_io",&["read_bytes","write_bytes","rchar","wchar"][..]),
            ("/proc/vmstat","host_vm",&["pgmajfault","compact_stall","compact_success","compact_fail","allocstall_dma","allocstall_dma32","allocstall_normal","allocstall_movable","pgscan_direct","pgscan_kswapd"][..]),
        ] {
            match fs::read_to_string(path) {
                Ok(text)=>for line in text.lines() {
                    let mut words=line.split_whitespace();
                    if let (Some(key),Some(value))=(words.next(),words.next()) {
                        let key=key.trim_end_matches(':');
                        if keys.contains(&key) {
                            if let Ok(value)=value.parse(){s.values.insert(format!("{prefix}.{key}"),value);}
                        }
                    }
                },
                Err(_)=>s.unavailable.push(prefix.into()),
            }
        }
        for resource in ["cpu","memory","io"] {
            match fs::read_to_string(format!("/proc/pressure/{resource}")) {
                Ok(text)=>for line in text.lines() {
                    let mut words=line.split_whitespace();
                    if let Some(kind)=words.next() {
                        if !["some","full"].contains(&kind) {continue;}
                        for word in words {
                            if let Some(value)=word.strip_prefix("total=").and_then(|v|v.parse().ok()) {
                                s.values.insert(format!("host_psi.{resource}.{kind}_usec"),value);
                            }
                        }
                    }
                },
                Err(_)=>s.unavailable.push(format!("host_psi.{resource}")),
            }
        }
        s.read_ms=started.elapsed().as_secs_f64()*1000.;s
    }
    pub(crate) fn finish(self,start:usize,live:usize,capacity:usize,chunk_ms:f64)->serde_json::Value {
        let after=Self::read();
        let delta:BTreeMap<_,_>=after.values.iter().filter_map(|(key,value)|
            self.values.get(key).and_then(|before|value.checked_sub(*before)).map(|d|(key.clone(),d))).collect();
        serde_json::json!({"start":start,"live_rows":live,"physical_rows":capacity,
            "chunk_ms_excluding_counter_reads":chunk_ms,"counter_read_ms":self.read_ms+after.read_ms,
            "delta":delta,"unavailable_before":self.unavailable,"unavailable_after":after.unavailable,
            "scope":"Host PSI/vmstat are system-wide; cgroup counters cover this container; thread counters cover the model host thread. Correlation alone does not establish the cause of a stall."})
    }
}
