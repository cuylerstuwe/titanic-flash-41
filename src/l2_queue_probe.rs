//! Same-allocation M3/M6 screen of L2 hint queue plans (CANADA_SCRATCH_L2_QUEUE with
//! CANADA_SCRATCH_L2_QUEUE_SCREEN=1): every plan against today's separate hints, reversed arms, unchanged
//! arithmetic (full logits, T1 samples and rollback are compared across every arm).
use super::*;
use super::history32_generation_probe::execute;

fn state(target: &Target) -> Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}
/// Back-to-back decode collectives in one captured graph, with no model work between them: what one costs when
/// every rank arrives together (the transport and the kernel's own waits), against the same kernels' times
/// inside a decode step, where a rank also waits for the slowest peer's preceding work. Zero payloads; wall
/// time on this rank over 256 collectives a sample. Every rank runs the same sequence.
/// Every RoCE port's hardware counters (sysfs, read-only): congestion notifications, sequence errors and
/// retransmissions, keyed "device/counter". Empty where sysfs does not show them.
fn roce_counters() -> std::collections::BTreeMap<String, u64> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(devices) = std::fs::read_dir("/sys/class/infiniband") else { return out };
    for device in devices.flatten() {
        let name = device.file_name().to_string_lossy().into_owned();
        let Ok(counters) = std::fs::read_dir(device.path().join("ports/1/hw_counters")) else { continue };
        for counter in counters.flatten() {
            if let Ok(text) = std::fs::read_to_string(counter.path())
                && let Ok(value) = text.trim().parse::<u64>() {
                out.insert(format!("{name}/{}", counter.file_name().to_string_lossy()), value);
            }
        }
    }
    out
}
/// The counters that moved since `before`.
fn roce_delta(before: &std::collections::BTreeMap<String, u64>) -> Value {
    let after = roce_counters();
    Value::Object(after.iter().filter_map(|(key, value)| {
        let delta = value.wrapping_sub(*before.get(key).unwrap_or(value));
        (delta != 0 && !key.ends_with("/lifespan")).then(|| (key.clone(), json!(delta)))
    }).collect())
}
/// CANADA_SCRATCH_MOE_FRONT_GRID_SCREEN: block counts (1..=48, comma separated) for the front-end grid variants.
fn front_grids() -> Result<Vec<u32>> {
    match std::env::var("CANADA_SCRATCH_MOE_FRONT_GRID_SCREEN") {
        Ok(v) => {
            let grids = v.split(',').map(|g| Ok(g.trim().parse::<u32>()?)).collect::<Result<Vec<_>>>()?;
            ensure!(!grids.is_empty() && grids.len() <= 8 && grids.iter().all(|g| (1..=48).contains(g)), "front-end grid list");
            Ok(grids)
        }
        Err(std::env::VarError::NotPresent) => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}
/// CANADA_SCRATCH_MOE_PHASE_GRID_SCREEN: launch grids for the routed experts' two phase kernels, comma separated
/// `<FC1 blocks>x<FC2 blocks>[c]` (exported: 96x96; `c` adds the largest shared-memory carve-out, so more than two
/// blocks can stay resident on an SM). Each block strides the task list by the grid, so any grid is the same
/// arithmetic; with more tasks than blocks the last pass over the list leaves most blocks idle.
fn phase_grids() -> Result<Vec<(u32, u32, bool)>> {
    match std::env::var("CANADA_SCRATCH_MOE_PHASE_GRID_SCREEN") {
        Ok(v) => {
            let grids = v.split(',').map(crate::config::parse_moe_phase_grid).collect::<Result<Vec<_>>>()?;
            ensure!(!grids.is_empty() && grids.len() <= 12, "expert grid list");
            Ok(grids)
        }
        Err(std::env::VarError::NotPresent) => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}
/// Timed repetitions per arm of the plan screen (CANADA_SCRATCH_L2_QUEUE_TIMINGS, default 10).
/// CANADA_SCRATCH_MHC_STATS_LATE_SCREEN: release points of the mHC statistics forks to compare with today's
/// (comma separated <attention><ffn> pairs as CANADA_SCRATCH_MHC_STATS_LATE, e.g. 13,14,15,10,03).
fn stats_lates() -> Result<Vec<(u8, u8)>> {
    match std::env::var("CANADA_SCRATCH_MHC_STATS_LATE_SCREEN") {
        Ok(value) => value.split(',').map(crate::mhc::parse_stats_late).collect(),
        Err(std::env::VarError::NotPresent) => Ok(Vec::new()), Err(error) => Err(error.into()),
    }
}
fn timings() -> Result<usize> {
    match std::env::var("CANADA_SCRATCH_L2_QUEUE_TIMINGS") {
        Ok(v) => { let n = v.parse::<usize>()?; ensure!((4..=400).contains(&n), "L2 queue timings count"); Ok(n) }
        Err(std::env::VarError::NotPresent) => Ok(10),
        Err(e) => Err(e.into()),
    }
}
fn collective_timing(w: &mut Worker) -> Result<Value> {
    use half::bf16;
    let (cuda, stream, comm) = (&w.cuda, &w.stream, &w.comm);
    stream.synchronize()?;
    comm.select_decode_reduction(true, stream)?;
    let mut sum = Vec::new();
    for rows in [3usize, 6, 16] {
        let mut input: Buffer<bf16> = cuda.allocate(rows * 5120)?;
        let mut output: Buffer<bf16> = cuda.allocate(rows * 5120)?;
        input.zero()?; output.zero()?;
        sum.push((rows, input, output));
    }
    let mut gather = Vec::new();
    for rows in [3usize, 6] {
        let mut input: Buffer<bf16> = cuda.allocate(rows * 448)?;
        let mut output: Buffer<bf16> = cuda.allocate(rows * 448 * 4)?;
        input.zero()?; output.zero()?;
        gather.push((rows, input, output));
    }
    stream.join_default()?;
    let mut records = Vec::new();
    // (name, the sums' buffer index, the gathers' buffer index): 64 rounds a graph of each case's collectives.
    for (name, s, g) in [("sum 3 rows", Some(0), None), ("sum 6 rows", Some(1), None), ("sum 16 rows", Some(2), None),
        ("gather 3 x 448", None, Some(0)), ("gather 6 x 448", None, Some(1)),
        ("layer of 3 rows: gather, sum, sum", Some(0), Some(0)), ("layer of 6 rows: gather, sum, sum", Some(1), Some(1))] {
        let mut each = 0usize;
        let mut enqueue = |count: &mut usize| -> Result<()> {
            // With the layer clock armed: marks around every collective of the graph (part 3).
            crate::layer_clock::part(stream, 3)?;
            for _ in 0..64 {
                if let Some(g) = g {
                    let (_, input, output) = &mut gather[g];
                    unsafe { comm.gather_bf16(input, output, 448, stream) }?;
                    *count += 1;
                }
                if let Some(s) = s {
                    let (_, input, output) = &mut sum[s];
                    for _ in 0..if g.is_some() { 2 } else { 1 } {
                        unsafe { comm.sum_bf16(input, output, stream) }?;
                        *count += 1;
                    }
                }
            }
            crate::layer_clock::end(stream)
        };
        enqueue(&mut each)?;
        stream.synchronize()?;
        comm.health()?;
        let mut captured = 0usize;
        let graph = stream.capture(|| enqueue(&mut captured))?;
        ensure!(captured == each, "collective timing capture extent");
        for _ in 0..4 { unsafe { graph.launch(stream) }?; }
        stream.synchronize()?;
        let mut samples = Vec::new();
        let counters = roce_counters();
        // With the clock: one launch a timing (48 of them), each read back, so every graph's own marks survive.
        let clocked = crate::layer_clock::armed();
        let (rounds, launches) = if clocked { (48, 1) } else { (12, 4) };
        let mut clocks = Vec::new();
        for _ in 0..rounds {
            let start = Instant::now();
            for _ in 0..launches { unsafe { graph.launch(stream) }?; }
            stream.synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1e6 / (launches * each) as f64);
            if let Some(marks) = crate::layer_clock::read()? { clocks.push(crate::layer_clock::part_json(&marks, 3)); }
        }
        let moved = roce_delta(&counters);
        comm.health()?;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        eprintln!("rank{} collective timing, {name}: median {:.2} us each (min {:.2})", comm.rank(), sorted[sorted.len() / 2], sorted[0]);
        records.push(json!({"case":name,"collectives_per_graph":each,"launches_per_sample":launches,
            "median_us_each":sorted[sorted.len() / 2],"min_us_each":sorted[0],"samples_us_each":samples,
            "roce_counter_deltas":moved,"clock":clocks}));
    }
    // What a hint beside a collective costs it: the same back-to-back collectives, each with one L2 hint over
    // cold scratch bytes (a different 8 MiB slice of a 512 MiB buffer a round, so every hint reaches memory),
    // begun before the collective and joined after it. "hint only" rounds give the hint's own duration.
    let module = cuda.module(&w.core)?;
    let mut scratch: Buffer<u8> = cuda.allocate(512 << 20)?;
    scratch.zero()?;
    stream.join_default()?;
    let mut hinted = Vec::new();
    for (pace, kib) in [(None, 3072usize), (Some(100u32), 2048), (Some(150), 3072), (Some(200), 3072), (Some(300), 3072),
        (Some(200), 6144), (Some(200), 1024), (Some(100), 1024)] {
        let windows = (0..64).map(|i| crate::l2_prefetch::Window::new_scratch_for_probe(cuda, &module, stream,
            (scratch.ptr() + (i as u64) * (8 << 20), kib * 1024), pace)).collect::<Result<Vec<_>>>()?;
        // (case, sum buffer, gather buffer): the hint alone, beside a 6-row sum, beside the 6 x 448 gather.
        for (name, s, g) in [("hint only", None, None), ("sum 6 rows", Some(1usize), None), ("gather 6 x 448", None, Some(1usize))] {
            let mut enqueue = || -> Result<()> {
                for window in &windows {
                    window.begin(stream)?;
                    if let Some(s) = s { let (_, input, output) = &mut sum[s]; unsafe { comm.sum_bf16(input, output, stream) }?; }
                    if let Some(g) = g { let (_, input, output) = &mut gather[g]; unsafe { comm.gather_bf16(input, output, 448, stream) }?; }
                    window.join(stream)?;
                }
                Ok(())
            };
            enqueue()?;
            stream.synchronize()?;
            comm.health()?;
            let graph = stream.capture(&mut enqueue)?;
            for _ in 0..4 { unsafe { graph.launch(stream) }?; }
            stream.synchronize()?;
            let mut samples = Vec::new();
            for _ in 0..12 {
                let start = Instant::now();
                for _ in 0..4 { unsafe { graph.launch(stream) }?; }
                stream.synchronize()?;
                samples.push(start.elapsed().as_secs_f64() * 1e6 / 256.);
            }
            comm.health()?;
            let mut sorted = samples.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            eprintln!("rank{} collective timing, {name} beside a hint of {kib} KiB at {pace:?} GB/s: median {:.2} us a round (min {:.2})",
                comm.rank(), sorted[sorted.len() / 2], sorted[0]);
            hinted.push(json!({"case":name,"hint_kib":kib,"hint_pace_gbps":pace,"median_us_a_round":sorted[sorted.len() / 2],
                "min_us_a_round":sorted[0],"samples_us_a_round":samples}));
        }
    }
    // What an idle gap or bandwidth-bound work before a collective costs it. Every rank runs the same graph:
    // rounds of [gap] [collective]. A timer delay is equal on every rank, so it adds no skew: a latency that
    // grows with it is the transport waking up (proxy, NIC, link). A cold 64 MiB read takes each rank its own
    // time, so the round then also holds the wait for the slowest rank, as after the experts in a decode layer.
    let mut gapped = Vec::new();
    let mut sink: Buffer<u64> = cuda.allocate(128)?;
    sink.zero()?;
    stream.join_default()?;
    let delay = module.function("ca_probe_delay")?;
    let read = module.function("ca_probe_read")?;
    for (name, gap_us, work, s, g) in [
        ("sum 6 rows", 0u64, false, Some(1usize), None), ("sum 6 rows", 30, false, Some(1), None),
        ("sum 6 rows", 100, false, Some(1), None), ("sum 6 rows", 300, false, Some(1), None),
        ("sum 6 rows", 600, false, Some(1), None), ("gather 6 x 448", 30, false, None, Some(1usize)),
        ("gather 6 x 448", 100, false, None, Some(1)), ("gather 6 x 448", 300, false, None, Some(1)),
        ("none", 0, true, None, None), ("sum 6 rows", 0, true, Some(1), None), ("gather 6 x 448", 0, true, None, Some(1))] {
        let mut enqueue = || -> Result<()> {
            for round in 0..64u64 {
                if gap_us > 0 {
                    let mut ns = gap_us * 1000;
                    unsafe { delay.launch([1, 1, 1], [32, 1, 1], 0, stream, &mut [(&mut ns as *mut u64).cast()]) }?;
                }
                if work {
                    // A different cold 64 MiB slice a round (eight slices of the scratch).
                    let (mut source, mut words, mut out) = (scratch.ptr() + (round % 8) * (64 << 20), (64u64 << 20) / 8, sink.ptr());
                    unsafe { read.launch([96, 1, 1], [128, 1, 1], 0, stream, &mut [(&mut source as *mut u64).cast(),
                        (&mut words as *mut u64).cast(), (&mut out as *mut u64).cast()]) }?;
                }
                if let Some(s) = s { let (_, input, output) = &mut sum[s]; unsafe { comm.sum_bf16(input, output, stream) }?; }
                if let Some(g) = g { let (_, input, output) = &mut gather[g]; unsafe { comm.gather_bf16(input, output, 448, stream) }?; }
            }
            Ok(())
        };
        enqueue()?;
        stream.synchronize()?;
        comm.health()?;
        let graph = stream.capture(&mut enqueue)?;
        for _ in 0..2 { unsafe { graph.launch(stream) }?; }
        stream.synchronize()?;
        let mut samples = Vec::new();
        for _ in 0..8 {
            let start = Instant::now();
            for _ in 0..2 { unsafe { graph.launch(stream) }?; }
            stream.synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1e6 / 128.);
        }
        comm.health()?;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        eprintln!("rank{} collective timing, {name} after {}: median {:.2} us a round (min {:.2})", comm.rank(),
            if work { "a cold 64 MiB read".to_owned() } else { format!("a {gap_us} us delay") }, sorted[sorted.len() / 2], sorted[0]);
        gapped.push(json!({"collective":name,"delay_us":gap_us,"cold_read_mib":if work { 64 } else { 0 },
            "median_us_a_round":sorted[sorted.len() / 2],"min_us_a_round":sorted[0],"samples_us_a_round":samples}));
    }
    Ok(json!({"records":records,"hinted":hinted,"gapped":gapped,
        "scope":"Back-to-back native decode collectives in one captured graph (64 rounds), zero payloads, no model work between them; wall time on this rank over four launches a sample. The hinted cases add one L2 hint over cold scratch bytes beside each collective. No arithmetic or serving change."}))
}
/// One owner's plan screen: every plan against the existing hints at each live row count, arms
/// control/queue/queue/control, with exact logits, samples and rollback in every arm.
#[allow(clippy::too_many_arguments)]
fn screen_owner(target: &mut Target, comm: &Roce, stream: &Stream, cuda: &Cuda, tokens: &[u32], prefix: usize,
    lives: &[usize], rank: usize, records: &mut Vec<Value>, summary: &mut Vec<Value>) -> Result<()> {
    let owner = target.physical_rows();
    let (plans, ffn_plans) = crate::config::l2_queue_plans_for(owner);
    let span = lives.iter().copied().max().unwrap_or(0);
    // The native WO-A A/B (CANADA_SCRATCH_WOA_NATIVE_SCREEN=1, six-row owner) runs as one more variant after
    // the plans, on plan 0 in both arms when there is one; a difference there is recorded, not fatal.
    let woa = crate::config::woa_native_screen() && (owner == 6 || owner == 16);
    // CANADA_SCRATCH_WOA_NATIVE_SCREEN_ALL=1: that A/B on every plan (the hint windows pace the projections, so the
    // native kernel's value depends on the plan and on the memory's speed), not on plan 0 alone.
    let woa_plans = if !woa { 0 } else if crate::config::woa_native_screen_all() { plans.len().max(1) } else { 1 };
    // Front-end grid variants (CANADA_SCRATCH_MOE_FRONT_GRID_SCREEN=1,4,8,...): after the plans, the routed experts'
    // front-end kernel launched with that many blocks instead of 48, on plan 0 in both arms when there is one.
    let grids = front_grids()?;
    // Late statistics forks (CANADA_SCRATCH_MHC_STATS_LATE_SCREEN): after those, on plan 0 in both arms.
    let lates = stats_lates()?;
    let knockouts = if owner == 6 { crate::knockout::screen()? } else { Vec::new() };
    // CANADA_SCRATCH_KNOCKOUT_PLANS=N (default 1): every knockout runs on each of the first N plans (both arms on
    // that plan), since what a group is worth depends on the hint plan it runs beside.
    let knock_plans = match std::env::var("CANADA_SCRATCH_KNOCKOUT_PLANS") {
        Ok(v) => { let n: usize = v.parse()?; ensure!(n >= 1 && n <= plans.len().max(1), "knockout plans"); n }
        Err(std::env::VarError::NotPresent) => 1,
        Err(e) => return Err(e.into()),
    };
    // The third window before the index work (CANADA_SCRATCH_L2_QUEUE_INDEX_EARLY_SCREEN=1): one more variant, last.
    let early = crate::config::l2_queue_index_early_screen();
    // Expert kernel grids (CANADA_SCRATCH_MOE_PHASE_GRID_SCREEN): after everything else, on plan 0 in both arms.
    let phases = phase_grids()?;
    // Bounded candidate mask for the decode consumers (CANADA_SCRATCH_BOUNDED_MASK_SCREEN=1): one last variant.
    let bounded = std::env::var("CANADA_SCRATCH_BOUNDED_MASK_SCREEN").as_deref() == Ok("1");
    // The six-row owner's experts at 2, 4 and 5 live rows (CANADA_SCRATCH_MOE_LIVE_VARIANTS_SCREEN=1): the cache-ca
    // plan from two live rows, the fused FFN sum from two, and both; after everything else, on plan 0 in both arms.
    // The source layers' key emission beside the attention projection (CANADA_SCRATCH_SOURCE_KEYS_FORK_SCREEN=1): last.
    let keys_fork = crate::config::source_keys_fork_screen();
    // The indexed layers' second window with the index work's cold weights first (…_INDEX_LEAD_SCREEN=1): very last.
    let index_lead = crate::config::l2_queue_index_lead_screen();
    // The Engram projection hints begun by layers 0 and 13 (…_ENGRAM_LEAD_SCREEN=1): after that.
    let engram_lead = crate::config::l2_queue_engram_lead_screen();
    // The native expert front-end in the captured graphs (CANADA_SCRATCH_MOE_NATIVE_FRONT_SCREEN=1): last of all.
    let native_front = crate::config::moe_native_front_screen();
    // Layers 28 / 32 / 36 reusing layer 24's candidate union (CANADA_SCRATCH_SPARSE_UNION_SHARE_SCREEN=1): after that,
    // six-row owner only (the sparse consumers are active from a 64K frontier).
    let union_share = owner == 6 && crate::config::sparse_union_share_screen();
    // The consuming layers' index work started where Q-B forks (CANADA_SCRATCH_INDEX_FORK_SCREEN=1): after that.
    let index_fork = crate::config::index_fork_screen();
    // QKV-A on its narrow lowering, fed by the boundary's quantized input (…_QKV_NARROW_PREQUANT_SCREEN=1): six-row
    // owner only, last.
    let qkv_narrow = owner == 6 && crate::config::qkv_narrow_prequant_screen();
    // The router's top-k inside the native expert front-end (CANADA_SCRATCH_MOE_ROUTE_FRONT_SCREEN=1; the native
    // front-end on in both arms): very last.
    let route_front = crate::config::moe_route_front_screen();
    // The earlier owner's four-bit code for the FC1 scales (CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN=1; runtime
    // moe-runtime-m6-scale-codes-v5): six-row owner, against the input-address plan it was patched from. Run with
    // CANADA_SCRATCH_MOE_FC_PDL_M3=0, …_COOPERATIVE_EPILOGUE=0 and …_MOE_CACHE_CA=0 so both arms take those two
    // plans at every live count. After everything else.
    let scale_codes = owner == 6 && std::env::var("CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN").as_deref() == Ok("1");
    // An empty kernel between the router's top-k and the experts at 2-4 live rows (CANADA_SCRATCH_GATE_SEP_SCREEN=1;
    // pico's DSGB10_GATE_SEP, checklist step 8): after everything else.
    let gate_sep = crate::config::gate_sep_screen();
    // The six-row head on its compact exception table against the canonical backing (CANADA_SCRATCH_HEAD_COMPACT_SCREEN=1
    // with CANADA_SCRATCH_HEAD_COMPACT_EXCEPTIONS=1; checklist step 3): six-row owner, very last.
    let head_compact = owner == 6 && crate::config::head_compact_screen();
    // The six-row head's bit-plane layout on that table against the compact head (CANADA_SCRATCH_HEAD_PLANES_SCREEN=1
    // with CANADA_SCRATCH_HEAD_COMPACT_EXCEPTIONS=1; checklist step 5): six-row owner, after that one.
    let head_planes = owner == 6 && crate::config::head_planes_screen();
    let live_variants: &[(&str, usize, usize, bool)] = if owner == 6
        && std::env::var("CANADA_SCRATCH_MOE_LIVE_VARIANTS_SCREEN").as_deref() == Ok("1") {
        &[("cache-ca plan from two live rows", 2, 6, false), ("fused FFN sum from two live rows", 6, 2, false),
          ("FC PDL plan at 2, 4 and 5 live rows", 6, 6, true), ("cache-ca plan and fused sum from two live rows", 2, 2, false),
          ("FC PDL plan at 2, 4 and 5 live rows and fused sum from two", 6, 2, true)]
    } else { &[] };
    ensure!((!plans.is_empty() || woa) && span > 0 && span <= owner && tokens.len() > prefix + 2 * span,
        "L2 queue owner screen extent");
    let original = state(target)?;
    for &live in lives {
        let proposed = &tokens[prefix..prefix+live];
        let changed = &tokens[prefix+span..prefix+span+live];
        ensure!(proposed != changed, "L2 queue changed-input probe must differ");
        let mut expected = None;
        // The direct-read wq_b (CANADA_SCRATCH_WQB_DIRECT_SCREEN=1, six-row owner): after every other variant, on
        // plan 0 or (…_SCREEN_ALL=1) on every plan; other arithmetic, so a difference is recorded, not fatal.
        let direct_plans = if !(owner == 6 && (crate::config::wqb_direct_screen() || crate::config::qkva_direct_screen())) { 0 }
            else if crate::config::wqb_direct_screen_all() { plans.len().max(1) } else { 1 };
        let direct_base = plans.len() + woa_plans + grids.len() + lates.len() + usize::from(early) + phases.len() + usize::from(bounded) + live_variants.len() + usize::from(keys_fork) + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork) + usize::from(qkv_narrow) + usize::from(route_front) + usize::from(scale_codes) + usize::from(gate_sep) + usize::from(head_compact) + usize::from(head_planes);
        // The attention boundaries' statistics by rank quarter (CANADA_SCRATCH_MHC_QUARTER_STATS_SCREEN=1; six-row
        // owner): one more variant after those, on plan 0 in both arms. Bit-identical by construction: the
        // screen's logit and sample comparison is its check.
        let quarter = owner == 6 && crate::config::mhc_quarter_stats_screen();
        // The head by the 4-bit copy with exact rescoring (CANADA_SCRATCH_HEAD4_RESCORE_SCREEN=1; six-row owner): one
        // more variant, on plan 0 in both arms. Other arithmetic by design: the difference is measured (KL of the
        // scheme's logits against the exact head's on the same input), not required to be zero.
        let head4 = owner == 6 && crate::config::head4_rescore_screen();
        let variant_count = direct_base + direct_plans + usize::from(quarter) + usize::from(head4);
        // Injection arms (CANADA_SCRATCH_INJECT_SCREEN; crate::inject): after the knockouts, on plan 0 in both arms.
        let injects = if owner == 6 { crate::inject::screen()? } else { Vec::new() };
        let first_inject = variant_count + knockouts.len() * knock_plans;
        for index in 0..first_inject + injects.len() {
            let inject = (index >= first_inject).then(|| injects[index - first_inject].clone());
            // A knockout arm (CANADA_SCRATCH_KNOCKOUT_SCREEN; crate::knockout): one kernel group left out of the captures.
            let knock_plan = if index >= variant_count && index < first_inject { (index - variant_count) / knockouts.len() } else { 0 };
            let knock = (index >= variant_count && index < first_inject).then(|| knockouts[(index - variant_count) % knockouts.len()].clone());
            // The plan a native WO-A variant runs on (both arms).
            let native_plan = (index >= plans.len() && index < plans.len() + woa_plans).then(|| index - plans.len());
            let native = native_plan.is_some();
            let first_late = plans.len() + woa_plans + grids.len();
            let grid = (index >= plans.len() + woa_plans && index < first_late)
                .then(|| grids[index - plans.len() - woa_plans]);
            let late = (index >= first_late && index < first_late + lates.len()).then(|| lates[index - first_late]);
            let index_early = early && index == first_late + lates.len();
            let first_phase = first_late + lates.len() + usize::from(early);
            let phase = (index >= first_phase && index < first_phase + phases.len()).then(|| phases[index - first_phase]);
            let bounded_mask = bounded && index == first_phase + phases.len();
            let first_live_variant = first_phase + phases.len() + usize::from(bounded);
            let live_variant = (index >= first_live_variant && index < first_live_variant + live_variants.len())
                .then(|| live_variants[index - first_live_variant]);
            let fork_variant = keys_fork && index == first_live_variant + live_variants.len();
            let lead_variant = index_lead && index == first_live_variant + live_variants.len() + usize::from(keys_fork);
            let engram_variant = engram_lead
                && index == first_live_variant + live_variants.len() + usize::from(keys_fork) + usize::from(index_lead);
            let front_variant = native_front && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead);
            let union_variant = union_share && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front);
            let ifork_variant = index_fork && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share);
            let qkv_variant = qkv_narrow && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork);
            let route_variant = route_front && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork)
                + usize::from(qkv_narrow);
            let scale_variant = scale_codes && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork)
                + usize::from(qkv_narrow) + usize::from(route_front);
            let sep_variant = gate_sep && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork)
                + usize::from(qkv_narrow) + usize::from(route_front) + usize::from(scale_codes);
            let compact_variant = head_compact && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork)
                + usize::from(qkv_narrow) + usize::from(route_front) + usize::from(scale_codes) + usize::from(gate_sep);
            let planes_variant = head_planes && index == first_live_variant + live_variants.len() + usize::from(keys_fork)
                + usize::from(index_lead) + usize::from(engram_lead) + usize::from(native_front) + usize::from(union_share) + usize::from(index_fork)
                + usize::from(qkv_narrow) + usize::from(route_front) + usize::from(scale_codes) + usize::from(gate_sep) + usize::from(head_compact);
            let direct_plan = (index >= direct_base && index < direct_base + direct_plans).then(|| index - direct_base);
            let quarter_variant = quarter && index == direct_base + direct_plans;
            let head4_variant = head4 && index == direct_base + direct_plans + usize::from(quarter);
            let mut head4_counts: Vec<Vec<i32>> = Vec::new();
            let mut head4_width = 0usize;
            let mut direct_logits: Vec<Vec<u8>> = Vec::new();
            let mut direct_outputs: Vec<[Vec<u8>; 2]> = Vec::new();
            let mut swapped = 0;
            let mut regridded = 0;
            let mut means = [0f64; 2];
            let mut exact = true;
            for (trial, on) in [false, true, true, false].into_iter().enumerate() {
                let arm = on.then_some(index);
                if let Some(plan) = direct_plan {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(plan), stream)?;
                    target.set_wqb_direct_for_probe(Some(on), stream)?;
                } else if quarter_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    swapped = target.set_quarter_stats_for_probe(on, stream)?;
                } else if head4_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_head4_for_probe(on, stream)?;
                } else if let Some(plan) = native_plan {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(plan), stream)?;
                    target.set_woa_native_for_probe(on, stream)?;
                } else if let Some(grid) = grid {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_moe_front_grid_for_probe(on.then_some(grid), stream)?;
                } else if scale_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_moe_scale_codes_for_probe(on, stream)?;
                } else if route_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_moe_route_front_for_probe(on, stream)?;
                    crate::moe::route_placed();
                } else if compact_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_head_compact_for_probe(on, stream)?;
                } else if planes_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_head_variant_for_probe(if on { crate::lossless_head::PLANES } else { crate::lossless_head::COMPACT }, stream)?;
                    crate::lossless_head::planes_launched();
                } else if sep_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_gate_sep_for_probe(on, stream)?;
                    crate::ffn::gate_sep_launched();
                } else if union_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    ensure!(target.set_sparse_union_share_for_probe(on, stream)?, "candidate union screen on an unlinked owner");
                } else if ifork_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    regridded = target.set_index_fork_for_probe(on, stream)?;
                } else if qkv_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    regridded = target.set_narrow_qkv_for_probe(on, stream)?;
                    ensure!(regridded > 0, "narrow QKV screen on an owner without the alternate lowering");
                } else if front_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_moe_native_front_for_probe(on, stream)?;
                } else if engram_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_l2_engram_lead_for_probe(on, stream)?;
                } else if lead_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_l2_index_lead_for_probe(on, stream)?;
                } else if fork_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    regridded = target.set_source_keys_fork_for_probe(on, stream)?;
                } else if let Some((_, cache_ca, fuse, fc_pdl_other)) = live_variant {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_moe_live_variants_for_probe(on.then_some((cache_ca, fuse, fc_pdl_other)), stream)?;
                } else if bounded_mask {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    regridded = target.set_bounded_mask_for_probe(on, stream)?;
                } else if let Some((up, down, carve)) = phase {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    // A control arm recaptures with the exported grids and (after a carve-out arm) the driver's own carve-out.
                    target.set_moe_phase_grid_for_probe(Some(if on { (up, down, carve.then_some(100)) }
                        else { (96, 96, carve.then_some(-1)) }), stream)?;
                } else if index_early {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_l2_index_early_for_probe(on, stream)?;
                } else if let Some(points) = late {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_mhc_stats_late_for_probe(if on { Some(points) } else { Some((0, 0)) }, stream)?;
                } else if let Some((_, arm)) = &inject {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(0), stream)?;
                    target.set_injection_for_probe(on.then_some(*arm), stream)?;
                } else if let Some((_, mask)) = &knock {
                    target.set_l2_queue_for_probe((!plans.is_empty()).then_some(knock_plan), stream)?;
                    target.set_knockout_for_probe(0, stream)?;
                    if on {
                        // The previous arm ended on the changed input: run this step once in full so every buffer
                        // holds its values, then leave the group out.
                        target.begin_verification(proposed, comm, stream)?;
                        target.finish_verification(0, comm, stream)?;
                        target.set_knockout_for_probe(*mask, stream)?;
                    }
                } else {
                    target.set_l2_queue_for_probe(arm, stream)?;
                }
                for _ in 0..4 {
                    target.begin_verification(proposed, comm, stream)?;
                    target.finish_verification(0, comm, stream)?;
                }
                let captures = target.capture_statistics().1;
                if phase.is_some() && on { regridded = target.moe_regridded(); }
                let mut samples = Vec::new();
                let mut clocks = Vec::new();
                let counters = roce_counters();
                for _ in 0..timings()? {
                    let start = Instant::now();
                    target.begin_verification(proposed, comm, stream)?;
                    target.finish_verification(0, comm, stream)?;
                    samples.push(start.elapsed().as_secs_f64()*1000.);
                    // The forward's parts (initial, before Engram 14, after it) on the device's own clock.
                    if let Some(marks) = crate::layer_clock::read()? {
                        clocks.push(json!((0..3).map(|part| crate::layer_clock::part_json(&marks, part)).collect::<Vec<_>>()));
                    }
                }
                let moved = roce_delta(&counters);
                ensure!(target.capture_statistics().1 == captures, "timed L2 queue graph recapture");
                // A knockout arm is timed only: the checks below run with every kernel back in.
                if knock.is_some() && on { target.set_knockout_for_probe(0, stream)?; }
                if inject.is_some() && on { target.set_injection_for_probe(None, stream)?; }
                let mut checks = Vec::new();
                for input in [proposed, changed] {
                    target.begin_verification(input, comm, stream)?;
                    // Full gathered logits, or (an owner that samples from per-rank shards, as the 16-row one)
                    // this rank's own shard: every rank compares its own.
                    let (logits, count, width, local) = match target.pending_logits() {
                        Ok((logits, count)) => (logits, count, 129280usize, false),
                        Err(_) => {
                            let (logits, count) = target.pending_local_logits_for_probe()?;
                            (logits, count, logits.len() / owner, true)
                        }
                    };
                    ensure!(count == live && width > 0 && logits.len() >= live*width, "L2 queue live logits extent");
                    let bytes = unsafe { cuda.read_device(logits.ptr(), live*width*4) }?;
                    if direct_plan.is_some() && std::ptr::eq(input, proposed) {
                        direct_logits.push(bytes.clone());
                        direct_outputs.push(target.direct_outputs_for_probe(stream)?);
                    }
                    if head4_variant && std::ptr::eq(input, proposed) {
                        direct_logits.push(bytes.clone());
                        head4_width = width;
                        if on { head4_counts.push(target.head4_counts_for_probe(live, stream)?.0); }
                    }
                    let hash = format!("{:x}", Sha256::digest(bytes));
                    // Such an owner samples from its shards only when every row is live; the shard hashes
                    // above cover the other counts.
                    let ids = if !local { target.verification_sample(2026111700, stream)? }
                        else if live == owner { target.verification_history_sample(2026111700, comm, stream)? }
                        else { Vec::new() };
                    target.finish_verification(0, comm, stream)?;
                    ensure!(state(target)? == original, "L2 queue retained state/rollback mismatch");
                    checks.push(json!({"logits_sha256":hash,"sampled_ids":ids}));
                }
                ensure!(checks[0]["logits_sha256"] != checks[1]["logits_sha256"],
                    "changed L2 queue inputs did not change logits");
                if let Some(reference) = &expected {
                    if direct_plan.is_some() || native || grid.is_some() || late.is_some() || index_early || phase.is_some() || bounded_mask || live_variant.is_some() || fork_variant || lead_variant || engram_variant || front_variant || union_variant || ifork_variant || qkv_variant || route_variant || scale_variant || sep_variant || compact_variant || planes_variant || head4_variant { exact &= reference == &checks; }
                    else { ensure!(reference == &checks, "an L2 queue plan changed full logits or target T1 samples"); }
                } else { expected = Some(checks.clone()); }
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                if front_variant && on { swapped = target.moe_swapped(); }
                if route_variant && on { regridded = crate::moe::route_placed(); ensure!(regridded > 0, "fused routing screen placed no router"); }
                // The receipt that this arm's captures ran the plane operator and the compact arm's did not.
                if planes_variant { let launched = crate::lossless_head::planes_launched(); if on { regridded = launched; }
                    ensure!(on == (launched > 0), "the bit-plane head launched {launched} times with the planes {}", if on { "on" } else { "off" }); }
                if sep_variant { let launched = crate::ffn::gate_sep_launched(); if on { regridded = launched; }
                    ensure!(on == (launched > 0) || !(2..=4).contains(&live), "gate separation launched {launched} kernels"); }
                means[usize::from(arm.is_some())] += mean / 2.;
                records.push(json!({"owner_rows":owner,"live":live,"plan":index,"queue":arm.is_some(),"trial":trial,
                    "mean_ms":mean,"samples_ms":samples,"roce_counter_deltas":moved,"clock":clocks}));
            }
            if scale_variant {
                target.set_moe_scale_codes_for_probe(false, stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} four-bit FC1 scale code M{owner} live{live}: control {:.3} ms, coded {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"four-bit code for the FC1 expert scales (half of 3.9 % of an expert's bytes) against the input-address plan",
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if compact_variant {
                target.set_head_compact_for_probe(true, stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                ensure!(exact, "the compact head changed logits or samples in the plan screen");
                eprintln!("rank{rank} compact head M{owner} live{live}: canonical {:.3} ms, compact {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"six-row head: compact exception table against the canonical backing",
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if planes_variant {
                target.set_head_variant_for_probe(crate::lossless_head::COMPACT, stream)?;
                // Floors and medians over both trials of an arm; the means carry the outliers.
                let arm = |queue: bool| { let mut all = records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap())).collect::<Vec<_>>();
                    all.sort_by(f64::total_cmp); (all[0], all[all.len() / 2]) };
                let ((floor0, median0), (floor1, median1)) = (arm(false), arm(true));
                ensure!(exact, "the bit-plane head changed logits or samples in the plan screen");
                eprintln!("rank{rank} plane head M{owner} live{live}: compact {:.3} ms, planes {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, medians {median0:.3} / {median1:.3}, plane launches captured {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"six-row head: bit-plane weight layout (compact + planes) against the compact head",
                    "captured_launches":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0,
                    "control_median_ms":median0,"queue_median_ms":median1,"delta_median_ms":median1-median0}));
                continue;
            }
            if sep_variant {
                target.set_gate_sep_for_probe(false, stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} gate separation M{owner} live{live}: control {:.3} ms, separated {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, captured launches {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"empty kernel (no PDL) between the router's top-k and the routed experts at 2-4 live rows (pico's DSGB10_GATE_SEP)",
                    "captured_launches":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if route_variant {
                target.set_moe_route_front_for_probe(crate::config::moe_route_front(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} fused routing M{owner} live{live}: control {:.3} ms, fused {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, routers placed {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"router top-k inside the native expert front-end (native front-end in both arms)",
                    "layers":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if qkv_variant {
                target.set_narrow_qkv_for_probe(crate::config::qkv_narrow_prequant(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} narrow QKV-A M{owner} live{live}: control {:.3} ms, narrow {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, layers {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"QKV-A on the narrow (16x64) lowering fed by the boundary's quantized input",
                    "layers":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if ifork_variant {
                target.set_index_fork_for_probe(crate::config::index_fork(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} index fork M{owner} live{live}: control {:.3} ms, fork {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, layers {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"consuming layers' index work started at the query fork (side stream in both arms)",
                    "layers":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if union_variant {
                target.set_sparse_union_share_for_probe(crate::config::sparse_union_share(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} shared candidate union M{owner} live{live}: control {:.3} ms, shared {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"layers 28, 32, 36 reuse layer 24's candidate union and gathered keys",
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if front_variant {
                target.set_moe_native_front_for_probe(crate::config::moe_native_front(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} native expert front-end M{owner} live{live}: control {:.3} ms, native {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, nodes replaced {swapped}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                ensure!(live < 2 || swapped >= 40, "native front-end screen replaced {swapped} nodes");
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"native expert front-end in the captured graphs",
                    "exact":exact,"nodes_replaced":swapped,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if engram_variant {
                target.set_l2_engram_lead_for_probe(crate::config::l2_queue_engram_lead(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} Engram lead window M{owner} live{live}: control {:.3} ms, lead {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"Engram projection hinted beside the previous FFN sum (layers 1 and 14)",
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if lead_variant {
                target.set_l2_index_lead_for_probe(crate::config::l2_queue_index_lead(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} index lead window M{owner} live{live}: control {:.3} ms, lead {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"index weights first in the second window (indexed layers)",
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if fork_variant {
                target.set_source_keys_fork_for_probe(crate::config::source_keys_fork(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} source key fork M{owner} live{live}: control {:.3} ms, fork {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, layers {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"source key emission beside the attention projection",
                    "layers":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if let Some((name, _, _, _)) = live_variant {
                target.set_moe_live_variants_for_probe(None, stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} {name} M{owner} live{live}: control {:.3} ms, variant {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":name,
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if bounded_mask {
                target.set_bounded_mask_for_probe(crate::candidates::bounded_mask_decode(), stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} bounded candidate mask M{owner} live{live}: control {:.3} ms, bounded {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, consumers {regridded}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"bounded candidate mask (layers 24..36)",
                    "consumers":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if let Some((up, down, carve)) = phase {
                target.set_moe_phase_grid_for_probe(None, stream)?;
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} expert grids {up}x{down}{} M{owner} live{live}: control {:.3} ms, grids {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, nodes {regridded}, logits and samples exact {exact}",
                    if carve { " carve-out" } else { "" }, means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,
                    "variant":format!("expert grids {up}x{down}{}", if carve { " carve-out 100" } else { "" }),
                    "matched_kernel_nodes":regridded,"exact":exact,"control_ms":means[0],"queue_ms":means[1],
                    "delta_ms":means[1]-means[0],"delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if index_early {
                target.set_l2_index_early_for_probe(crate::config::l2_queue_index_early(), stream)?;
                eprintln!("rank{rank} early index hint M{owner} live{live}: control {:.3} ms, early {:.3} ms ({:+.2}%), logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"third window before the index work",
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if let Some((name, arm)) = &inject {
                target.set_injection_for_probe(None, stream)?;
                eprintln!("rank{rank} injection {name} M{owner} live{live}: control {:.3} ms, injected {:.3} ms ({:+.2}%)",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,
                    "variant":format!("inject at {} {} (timing only)", arm.code,
                        if arm.spin_ns > 0 { format!("spin {} us", arm.spin_ns / 1000) } else { format!("read {} KiB", arm.bytes / 1024) })
                        + &arm.rank.map_or(String::new(), |r| format!(" on rank {r}")),
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if let Some((name, _)) = &knock {
                target.set_knockout_for_probe(0, stream)?;
                eprintln!("rank{rank} knockout {name} on plan {knock_plan} M{owner} live{live}: control {:.3} ms, without {:.3} ms ({:+.2}%)",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,
                    "variant":format!("knockout {name} on plan {knock_plan} (timing only: the group's outputs are reused)"),
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if let Some((attention, ffn)) = late {
                target.set_mhc_stats_late_for_probe(None, stream)?;
                eprintln!("rank{rank} late statistics {attention}{ffn} M{owner} live{live}: control {:.3} ms, late {:.3} ms ({:+.2}%), logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":format!("statistics fork {attention}{ffn}"),
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if let Some(grid) = grid {
                target.set_moe_front_grid_for_probe(None, stream)?;
                eprintln!("rank{rank} front-end grid {grid} M{owner} live{live}: control {:.3} ms, grid {:.3} ms ({:+.2}%), logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":format!("front-end grid {grid}"),
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if head4_variant {
                target.set_head4_for_probe(false, stream)?;
                // Trials ran off, on, on, off: the scheme's logits against the exact head's, row by row.
                ensure!(direct_logits.len() == 4 && head4_width > 0, "4-bit head screen logits");
                let row_kl = |exact: &[u8], scheme: &[u8], row: usize| -> f64 {
                    let get = |b: &[u8], i: usize| f32::from_le_bytes(b[(row * head4_width + i) * 4..(row * head4_width + i) * 4 + 4].try_into().unwrap()) as f64;
                    let (mut me, mut ms) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                    for i in 0..head4_width { me = me.max(get(exact, i)); ms = ms.max(get(scheme, i)); }
                    let (mut ze, mut zs) = (0f64, 0f64);
                    for i in 0..head4_width { ze += (get(exact, i) - me).exp(); zs += (get(scheme, i) - ms).exp(); }
                    let (le, ls) = (me + ze.ln(), ms + zs.ln());
                    (0..head4_width).map(|i| { let a = get(exact, i) - le; a.exp() * (a - (get(scheme, i) - ls)) }).sum()
                };
                let kls: Vec<f64> = (0..live).map(|row| row_kl(&direct_logits[0], &direct_logits[1], row)).collect();
                let noise: Vec<f64> = (0..live).map(|row| row_kl(&direct_logits[0], &direct_logits[3], row)).collect();
                let (kl_mean, kl_max) = (kls.iter().sum::<f64>() / live as f64, kls.iter().cloned().fold(0f64, f64::max));
                let settings = target.head4_counts_for_probe(live, stream)?.1;
                eprintln!("rank{rank} 4-bit head with exact rescoring M{owner} live{live}: control {:.3} ms, scheme {:.3} ms ({:+.2}%), KL(exact || scheme) a row mean {kl_mean:.3e} max {kl_max:.3e} over {head4_width} columns (exact against exact {:.1e}), rows rescored on this rank {:?}, settings {settings:?}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.), noise.iter().cloned().fold(0f64, f64::max), head4_counts.first());
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"4-bit head with exact rescoring",
                    "exact":exact,"kl_rows":kls,"kl_mean":kl_mean,"kl_max":kl_max,"logit_columns":head4_width,"rescored_rows":head4_counts,
                    "control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if quarter_variant {
                target.set_quarter_stats_for_probe(crate::config::mhc_quarter_stats(), stream)?;
                eprintln!("rank{rank} statistics by rank quarter M{owner} live{live}: control {:.3} ms, quarter {:.3} ms ({:+.2}%), layers {swapped}, logits and samples exact {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,"variant":"attention statistics by rank quarter",
                    "layers":swapped,"exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            if let Some(plan) = direct_plan {
                target.set_wqb_direct_for_probe(None, stream)?;
                // The first layer's projection outputs, exported arm against direct arm on the same step: that layer's
                // input is the same in both, so this is the kernels' own difference (BF16 elements that differ, and
                // the largest difference as a share of the largest magnitude), with no downstream amplification.
                let mut kernel_check = Vec::new();
                if direct_outputs.len() >= 2 {
                    for (which, name) in ["qkv_a", "wq_b"].iter().enumerate() {
                        let f = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect() };
                        let (a, b) = (f(&direct_outputs[0][which]), f(&direct_outputs[1][which]));
                        let differing = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs()));
                        let worst = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                        kernel_check.push(json!({"projection": name, "elements": a.len(), "differing": differing,
                            "largest_difference": worst, "largest_magnitude": scale}));
                    }
                    eprintln!("rank{rank} direct projections, first layer M{owner} live{live}: {}", serde_json::Value::Array(kernel_check.clone()));
                }
                // How far the direct arm's logits are from the exported kernel's on the same step (control arm of
                // trial 0 against the direct arm of trial 1): largest difference and top-1 agreement by row.
                let (mut largest, mut same_top, mut rows_seen) = (0f32, 0usize, 0usize);
                if direct_logits.len() >= 2 {
                    let f = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
                    let (a, b) = (f(&direct_logits[0]), f(&direct_logits[1]));
                    let width = a.len() / live;
                    for row in 0..live {
                        let (x, y) = (&a[row*width..(row+1)*width], &b[row*width..(row+1)*width]);
                        largest = x.iter().zip(y).fold(largest, |m, (p, q)| m.max((p - q).abs()));
                        let top = |v: &[f32]| v.iter().enumerate().max_by(|l, r| l.1.total_cmp(r.1)).map(|(i, _)| i).unwrap_or(0);
                        same_top += usize::from(top(x) == top(y)); rows_seen += 1;
                    }
                }
                let floor = |queue: bool| records.iter().rev().take(4).filter(|r| r["queue"] == queue)
                    .flat_map(|r| r["samples_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .fold(f64::INFINITY, f64::min);
                let (floor0, floor1) = (floor(false), floor(true));
                eprintln!("rank{rank} direct wq_b on plan {plan} M{owner} live{live}: exported {:.3} ms, direct {:.3} ms ({:+.2}%), floors {floor0:.3} / {floor1:.3}, largest logit difference {largest:.4}, same top-1 {same_top}/{rows_seen}, bit-identical {exact}",
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,
                    "variant":format!("direct-read projections on plan {plan}"),"first_layer_outputs":kernel_check,
                    "exact":exact,"largest_logit_difference":largest,"same_top1_rows":same_top,"rows":rows_seen,
                    "control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.),
                    "control_floor_ms":floor0,"queue_floor_ms":floor1,"delta_floor_ms":floor1-floor0}));
                continue;
            }
            if native {
                target.set_woa_native_for_probe(false, stream)?;
                eprintln!("rank{rank} native WO-A on plan {} M{owner} live{live}: control {:.3} ms, native {:.3} ms ({:+.2}%), logits and samples exact {exact}",
                    native_plan.unwrap_or(0),
                    means[0], means[1], 100.*(means[1]/means[0]-1.));
                summary.push(json!({"owner_rows":owner,"live":live,"plan":index,
                    "variant":format!("native WO-A partial kernel on plan {}", native_plan.unwrap_or(0)),
                    "exact":exact,"control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                    "delta_percent":100.*(means[1]/means[0]-1.)}));
                continue;
            }
            let plan = &plans[index];
            eprintln!("rank{rank} L2 queue M{owner} live{live} plan{index} {:?}: control {:.3} ms, queue {:.3} ms ({:+.2}%)",
                plan, means[0], means[1], 100.*(means[1]/means[0]-1.));
            let ffn = ffn_plans[index];
            summary.push(json!({"owner_rows":owner,"live":live,"plan":index,
                "windows":plan.iter().map(|w|json!({"kib":w.bytes/1024,"pace_gbps":w.pace,"kv":w.kv,"lead":w.lead,"tail":w.tail,"routed_end":w.routed_end,"warps":w.warps,"wave":w.wave})).collect::<Vec<_>>(),
                "w4":{"kib":ffn.w4.bytes/1024,"pace_gbps":ffn.w4.pace},
                "routed":ffn.routed.map(|r|json!({"kib":r.bytes/1024,"pace_gbps":r.pace,"pieces":r.pieces})),
                "control_ms":means[0],"queue_ms":means[1],"delta_ms":means[1]-means[0],
                "delta_percent":100.*(means[1]/means[0]-1.)}));
        }
    }
    if woa { target.set_woa_native_for_probe(false, stream)?; }
    target.set_l2_queue_for_probe(None, stream)
}
pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    let (plans6, plans16) = (crate::config::l2_queue_plans_for(6).0, crate::config::l2_queue_plans_for(16).0);
    let woa = crate::config::woa_native_screen();
    ensure!(w.current.is_none() && (!plans6.is_empty() || !plans16.is_empty() || woa),
        "isolated L2 queue screen requires plans and no active request");
    ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available >= 9*1024usize.pow(3))?,
        "L2 queue screen memory reserve");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let prefix = match std::env::var("CANADA_SCRATCH_L2_QUEUE_PREFIX") {
        Ok(v) => v.parse::<usize>()?, Err(std::env::VarError::NotPresent) => 131329, Err(e) => return Err(e.into()),
    };
    ensure!((256..=131329).contains(&prefix) && tokens.len() > prefix + 32, "L2 queue public fixture extent");
    execute(w,Command::Begin{images:Vec::new(),id:8_021_400,seed:2026111112,
        greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
    let mut records = Vec::new();
    let mut summary = Vec::new();
    // CANADA_SCRATCH_LAYER_CLOCK=1: device-clock marks around every decode collective of the timed steps (the
    // plan switch before each arm drops the verification graphs, so every timed graph is captured with them).
    if crate::layer_clock::requested() { crate::layer_clock::arm(&w.cuda, &w.core, &w.stream)?; }
    // CANADA_SCRATCH_INJECT_SCREEN: the sensitivity instrument (crate::inject), armed before any graph is captured.
    if !crate::inject::screen()?.is_empty() { crate::inject::arm(&w.cuda, &w.core, &w.stream, rank)?; }
    if !plans6.is_empty() || woa {
        // CANADA_SCRATCH_L2_QUEUE_LIVES6 (default 3,6): the six-row owner's live row counts in the screen.
        let lives6: Vec<usize> = match std::env::var("CANADA_SCRATCH_L2_QUEUE_LIVES6") {
            Ok(v) => v.split(',').map(|x| Ok(x.trim().parse::<usize>()?)).collect::<Result<Vec<_>>>()?,
            Err(std::env::VarError::NotPresent) => vec![3, 6], Err(e) => return Err(e.into()),
        };
        ensure!(!lives6.is_empty() && lives6.iter().all(|l| (1..=6).contains(l)), "six-row owner screen lives");
        screen_owner(&mut w.verify, &w.comm, &w.stream, &w.cuda, &tokens, prefix, &lives6, rank, &mut records, &mut summary)?;
    }
    if !plans16.is_empty() {
        // The 16-row owner takes the session over for its screen and hands it back.
        let longer = w.long_verify.as_mut().context("the 16-row L2 queue screen needs the 16-row owner")?;
        longer.handoff_from(&w.verify)?;
        screen_owner(longer, &w.comm, &w.stream, &w.cuda, &tokens, prefix, &[8, 16], rank, &mut records, &mut summary)?;
        w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;
    }
    let collectives = collective_timing(w)?;
    w.control.agree(&[0x76, 0x66])?;
    std::fs::write(out.join("l2-queue-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "all_logits_tokens_state_exact":true,"summary":summary,"records":records,
        "collective_timing":collectives,
        "scope":"One public frontier, M3/M6 fixed verification, four warm-ups and ten timings per arm, arms control/queue/queue/control per plan. The queue replaces the separate QKV-A and wq_b hints with one queue (QKV-A, wq_b, wo_a, wo_b) over three windows; immutable weight bytes only. Same allocations, canonical arithmetic, exact full logits, T1 samples and rollback in every arm. No serving-policy or outcome-speed claim."
    }))?)?;
    execute(w,Command::Finish(Finish::Length))?;
    Ok(())
}
