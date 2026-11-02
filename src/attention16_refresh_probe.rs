//! Revisit M16 side work after the qualified dense pipeline ordering fixes.
//! The explicit compact-overlap candidate also checks its actual constructor.
use super::history32_generation_probe::execute;
use super::*;

fn moe_screen(
    w: &mut Worker,
    rank: usize,
    out: &Path,
    ids: &[u32],
    prefix: usize,
    case: usize,
    records: &mut Vec<Value>,
) -> Result<()> {
    ensure!(
        std::env::var("CANADA_SCRATCH_MOE_FC_PDL_SCREEN").as_deref() == Ok("1"),
        "expert PDL comparison admission"
    );
    let original = (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?);
    let mut expected = std::collections::BTreeMap::<usize, Vec<u8>>::new();
    for repeat in 0..4u8 {
        for enabled in if repeat % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            w.verify.set_moe_fc_pdl_for_probe(enabled, &w.stream)?;
            for rows in [3usize, 6] {
                let mut times = Vec::new();
                let mut hashes = Vec::new();
                for trial in 0..4u8 {
                    w.stream.synchronize()?;
                    w.control.agree(&[
                        0x9b,
                        case as u8,
                        repeat,
                        enabled as u8,
                        rows as u8,
                        trial,
                    ])?;
                    let at = Instant::now();
                    w.verify
                        .begin_verification(&ids[prefix..prefix + rows], &w.comm, &w.stream)?;
                    w.stream.synchronize()?;
                    let seconds = at.elapsed().as_secs_f64();
                    let (y, n) = w.verify.pending_local_logits_for_probe()?;
                    ensure!(n == rows, "PDL refresh extent");
                    let logits = unsafe { w.cuda.read_device(y.ptr(), rows * 32320 * 4) }?;
                    let hash = format!("{:x}", Sha256::digest(&logits));
                    hashes.push(hash);
                    let reference = expected.entry(rows).or_insert_with(|| logits.clone());
                    let exact = reference.as_slice() == logits.as_slice();
                    if !exact {
                        std::fs::write(
                            out.join(format!("moe-pdl-c{case}-n{rows}-reference.bin")),
                            reference,
                        )?;
                        std::fs::write(
                            out.join(format!(
                                "moe-pdl-c{case}-n{rows}-p{}-r{repeat}-t{trial}.bin",
                                enabled as u8
                            )),
                            &logits,
                        )?;
                    }
                    w.verify.finish_verification(0, &w.comm, &w.stream)?;
                    let state =
                        (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?) == original;
                    if !w.control.all_admit(exact && state)? {
                        std::fs::write(
                            out.join("moe-pdl-refresh-failure.json"),
                            serde_json::to_vec_pretty(&json!({
                            "rank":rank,"prefix":prefix,"repeat":repeat,"pdl":enabled,"rows":rows,"trial":trial,
                            "logits_exact":exact,"state_exact":state,"earlier_records":records}))?,
                        )?;
                        anyhow::bail!("FC PDL changed fixed-prefix logits or retained state");
                    }
                    if trial > 0 {
                        times.push(seconds);
                    }
                }
                records.push(
                    json!({"case":case,"prefix":prefix,"repeat":repeat,"pdl":enabled,
                    "rows":rows,"seconds":times,"logit_hashes":hashes,"all_exact":true}),
                );
                std::fs::write(
                    out.join("moe-pdl-refresh-partial.json"),
                    serde_json::to_vec_pretty(&records)?,
                )?;
            }
        }
    }
    w.verify.set_moe_fc_pdl_for_probe(false, &w.stream)?;
    Ok(())
}

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    if std::env::var("CANADA_SCRATCH_ATTENTION16_COMMIT_SCREEN").as_deref() == Ok("1") {
        commit_screen(w, rank, out)?;
        if std::env::var("CANADA_SCRATCH_POOLED_INHERITED_ROUTE_CAPTURE").as_deref() == Ok("1") {
            super::pooled_route_probe::run(w, rank, out, Value::Null)?;
        }
        return Ok(());
    }
    ensure!(
        std::env::var("CANADA_SCRATCH_ATTENTION16_REFRESH_SCREEN").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_VERIFY16_ATTN_ONLY").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_ATTENTION16_OVERLAP").as_deref() == Ok("0")
            && w.current.is_none()
            && w.long_verify.is_some(),
        "finite M16 schedule refresh admission"
    );
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/long-context-continuation-v2.json"
    ))?;
    let ids: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let mut records = Vec::new();
    let mut moe_records = Vec::new();
    for (case, prefix) in [16413usize, 131109].into_iter().enumerate() {
        let begin = execute(
            w,
            Command::Begin {
                images: Vec::new(),
                id: 7_960_000 + case as u64,
                seed: 20261031,
                greedy: false,
                limit: 1,
                tokens: ids[..prefix].to_vec(),
            },
        )?;
        moe_screen(w, rank, out, &ids, prefix, case, &mut moe_records)?;
        let original = (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?);
        let target = w.long_verify.as_mut().unwrap();
        target.handoff_from(&w.verify)?;
        ensure!(
            target.grouped16_index_layers() == 4,
            "refresh must retain compact M16 indexing"
        );
        let mut expected: Option<Vec<u8>> = None;
        for repeat in 0..2u8 {
            let order = if repeat == 0 {
                [0u8, 1, 2, 3]
            } else {
                [3, 2, 1, 0]
            };
            for mask in order {
                target.set_verify16_schedule_for_probe(mask, &w.stream)?;
                ensure!(
                    target.query_overlap_layers() == (if mask & 1 != 0 { 40 } else { 0 })
                        && target.attention_coefficient_overlap_layers()
                            == (if mask & 2 != 0 { 40 } else { 0 }),
                    "M16 refresh schedule was not applied"
                );
                let mut times = Vec::new();
                let mut hashes = Vec::new();
                for trial in 0..3u8 {
                    w.stream.synchronize()?;
                    w.control.agree(&[0x9a, case as u8, repeat, mask, trial])?;
                    let at = Instant::now();
                    target.begin_verification(&ids[prefix..prefix + 16], &w.comm, &w.stream)?;
                    w.stream.synchronize()?;
                    let seconds = at.elapsed().as_secs_f64();
                    let (y, n) = target.pending_local_logits_for_probe()?;
                    ensure!(n == 16, "M16 refresh extent");
                    let logits = unsafe { w.cuda.read_device(y.ptr(), 16 * 32320 * 4) }?;
                    let hash = format!("{:x}", Sha256::digest(&logits));
                    hashes.push(hash);
                    let exact = expected.as_ref().is_none_or(|e| e == &logits);
                    if expected.is_none() {
                        expected = Some(logits.clone());
                    }
                    if !exact {
                        std::fs::write(
                            out.join(format!("attention16-c{case}-reference.bin")),
                            expected.as_ref().unwrap(),
                        )?;
                        std::fs::write(
                            out.join(format!(
                                "attention16-c{case}-m{mask}-r{repeat}-t{trial}.bin"
                            )),
                            &logits,
                        )?;
                    }
                    target.finish_verification(0, &w.comm, &w.stream)?;
                    let state =
                        (target.carry_digests()?, target.encoder_tail_digest()?) == original;
                    if !w.control.all_admit(exact && state)? {
                        std::fs::write(
                            out.join("attention16-refresh-failure.json"),
                            serde_json::to_vec_pretty(&json!({
                            "rank":rank,"prefix":prefix,"repeat":repeat,"mask":mask,"trial":trial,
                            "logits_exact":exact,"state_exact":state,"earlier_records":records}))?,
                        )?;
                        anyhow::bail!("M16 refresh changed fixed-prefix logits or retained state");
                    }
                    if trial > 0 {
                        times.push(seconds);
                    }
                }
                records.push(
                    json!({"case":case,"prefix":prefix,"begin":begin,"repeat":repeat,"mask":mask,
                    "query_overlap":mask&1!=0,"coefficient_overlap":mask&2!=0,
                    "seconds":times,"logit_hashes":hashes,"all_exact":true}),
                );
                std::fs::write(
                    out.join("attention16-refresh-partial.json"),
                    serde_json::to_vec_pretty(&records)?,
                )?;
            }
        }
        w.verify.handoff_from(target)?;
        execute(w, Command::Finish(Finish::Length))?;
    }
    execute(w, Command::Shutdown)?;
    std::fs::write(
        out.join("moe-pdl-refresh.json"),
        serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"records":moe_records,
        "scope":"Two public fixed prefixes; paired current input-address versus PDL M3/M6, same allocations and model state. Four alternating orders, one warmup plus three timed replays. Complete local vocabulary and rollback state checked on all ranks. No task-quality or serving-policy claim."}))?,
    )?;
    std::fs::write(
        out.join("attention16-refresh.json"),
        serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"records":records,"memory":w.cuda.admission_memory()?,
        "scope":"Two public fixed prefixes, current compact M16 index/head/precision and pipeline-fenced dense primitives. Serial, query-side, coefficient-side and combined schedules in reversed order; one warmup plus two timed replays each. All local vocabulary shards and retained state checked on four ranks. No serving-policy or task-quality claim."}))?,
    )?;
    Ok(())
}

/// Reuse the old failing frontier, including commits that change its history.
/// The first arm uses the actual constructor policy, before any probe rebinding.
fn commit_screen(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(
        crate::mhc::compact16_overlap_enabled()?
            && std::env::var("CANADA_SCRATCH_MOE_FC_PDL_M3").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref() == Ok("1")
            && w.current.is_none()
            && w.long_verify.is_some(),
        "bounded committed overlap screen"
    );
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/long-history-context-v1.json"
    ))?;
    let c = &fixture["cases"][0];
    let prompt: Vec<u32> = serde_json::from_value(c["prompt_ids"].clone())?;
    let n = prompt.len();
    ensure!(n == 274108, "old failure frontier changed");
    let mut teacher = vec![c["anchor"].as_u64().context("fixture anchor")? as u32];
    teacher.extend(serde_json::from_value::<Vec<u32>>(
        c["proposal_ids"].clone(),
    )?);
    ensure!(teacher.len() == 16, "M16 teacher extent");
    let mut changed = teacher.clone();
    changed.rotate_left(8);
    changed[0] = (changed[0] + 37) % 129280;
    let following = prompt[1111..1127].to_vec();
    let begin = execute(
        w,
        Command::Begin {
            images: Vec::new(),
            id: 7_961_001,
            seed: 20261101,
            greedy: false,
            limit: 1,
            tokens: prompt,
        },
    )?;
    let original = (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?);
    let bytes = w.verify.compact_state_image_bytes()?;
    ensure!(
        bytes <= 1024usize.pow(3)
            && w.control.all_admit(
                w.cuda.admission_memory()?.admitted_available >= 8 * 1024usize.pow(3) + bytes
            )?,
        "committed overlap snapshot reserve"
    );
    let image = w.verify.save_state_image(bytes, &w.comm, &w.stream)?;
    // n%64==60: the first partial commit crosses into a new page. Later
    // arms must not inherit future rows written by the earlier arm's +35
    // continuation when whole final pages (including slack) are hashed.
    ensure!(w.control.all_admit(
        w.cuda.admission_memory()?.admitted_available>=9*1024usize.pow(3))?,
        "diagnostic future-page snapshot reserve");
    let physical = w.verify.save_repeat_physical_image(&w.comm, &w.stream)?;
    let mut records = Vec::new();
    let mut references = std::collections::BTreeMap::<String, Vec<u8>>::new();
    let mut states = std::collections::BTreeMap::<usize, Value>::new();
    // Check the selected M3-only constructor, the old control and restoration.
    for (arm, enabled) in [true, false, true].into_iter().enumerate() {
        if arm > 0 {
            w.verify.set_moe_fc_pdl_for_probe(enabled, &w.stream)?;
        }
        for rows in [3usize, 6] {
            for trial in 0..2 {
                w.verify
                    .begin_verification(&teacher[..rows], &w.comm, &w.stream)?;
                w.stream.synchronize()?;
                let (y, live) = w.verify.pending_local_logits_for_probe()?;
                ensure!(live == rows, "M3 policy extent");
                let raw = unsafe { w.cuda.read_device(y.ptr(), rows * 32320 * 4) }?;
                let key = format!("pdl-{rows}");
                let exact = references.entry(key).or_insert_with(|| raw.clone()) == &raw;
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
                let state =
                    (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?) == original;
                ensure!(
                    w.control.all_admit(exact && state)?,
                    "M3-only policy changed logits/state"
                );
                records.push(json!({"kind":"m3_policy","arm":arm,"enabled":enabled,
                    "rows":rows,"trial":trial,"all_exact":true}));
            }
        }
    }
    for (arm, mask) in [3u8, 0, 3].into_iter().enumerate() {
        w.target_lineage = w
            .target_lineage
            .checked_add(1)
            .context("probe lineage exhausted")?;
        w.verify
            .restore_state_image(&image, w.target_lineage, &w.comm, &w.stream)?;
        w.verify.restore_repeat_physical_image(&physical, &w.comm, &w.stream)?;
        ensure!(
            (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?) == original,
            "probe original state restoration"
        );
        let wide = w.long_verify.as_mut().unwrap();
        wide.handoff_from(&w.verify)?;
        if arm > 0 {
            wide.set_verify16_schedule_for_probe(mask, &w.stream)?;
        }
        ensure!(
            wide.grouped16_index_layers() == 4
                && wide.query_overlap_layers() == (if mask == 3 { 40 } else { 0 })
                && wide.attention_coefficient_overlap_layers() == (if mask == 3 { 40 } else { 0 }),
            "committed M16 construction/schedule not selected"
        );
        for (stage, (ids, accepted)) in [(&teacher, 8usize), (&changed, 5), (&following, 16)]
            .into_iter()
            .enumerate()
        {
            let before = (wide.carry_digests()?, wide.encoder_tail_digest()?);
            let mut seconds = Vec::new();
            for trial in 0..3u8 {
                w.stream.synchronize()?;
                w.control
                    .agree(&[0x9c, arm as u8, mask, stage as u8, trial])?;
                let at = Instant::now();
                wide.begin_verification(ids, &w.comm, &w.stream)?;
                w.stream.synchronize()?;
                let elapsed = at.elapsed().as_secs_f64();
                let (y, live) = wide.pending_local_logits_for_probe()?;
                ensure!(live == 16, "commit M16 extent");
                let raw = unsafe { w.cuda.read_device(y.ptr(), 16 * 32320 * 4) }?;
                let key = format!("m16-stage{stage}");
                let reference = references.entry(key.clone()).or_insert_with(|| raw.clone());
                let exact = reference == &raw;
                if !exact {
                    std::fs::write(out.join(format!("{key}-reference.bin")), reference)?;
                    std::fs::write(out.join(format!("{key}-arm{arm}-trial{trial}.bin")), &raw)?;
                }
                wide.finish_verification(
                    if trial == 2 { accepted } else { 0 },
                    &w.comm,
                    &w.stream,
                )?;
                let rollback =
                    trial == 2 || (wide.carry_digests()?, wide.encoder_tail_digest()?) == before;
                if !w.control.all_admit(exact && rollback)? {
                    std::fs::write(
                        out.join("attention16-commit-failure.json"),
                        serde_json::to_vec_pretty(&json!({
                        "rank":rank,"arm":arm,"mask":mask,"stage":stage,"trial":trial,
                        "logits_exact":exact,"rollback_exact":rollback,"earlier_records":records}))?,
                    )?;
                    anyhow::bail!("committed M16 overlap changed logits or rollback");
                }
                if trial > 0 {
                    seconds.push(elapsed);
                }
            }
            let full_state = wide.repeat_state_digests(&w.stream)?;
            let reference = states.entry(stage).or_insert_with(|| full_state.clone());
            let exact = reference == &full_state;
            if !exact {
                std::fs::write(
                    out.join(format!("state-stage{stage}-reference.json")),
                    serde_json::to_vec_pretty(reference)?,
                )?;
                std::fs::write(
                    out.join(format!("state-stage{stage}-arm{arm}.json")),
                    serde_json::to_vec_pretty(&full_state)?,
                )?;
            }
            ensure!(
                w.control.all_admit(exact)?,
                "committed M16 overlap changed full accepted state"
            );
            records.push(json!({"kind":"m16_commit","arm":arm,"mask":mask,"stage":stage,
                "position":wide.position(),"accepted":accepted,"seconds":seconds,
                "full_state_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&full_state)?)),"all_exact":true}));
            std::fs::write(
                out.join("attention16-commit-partial.json"),
                serde_json::to_vec_pretty(&records)?,
            )?;
            if stage == 1 {
                w.verify.handoff_from(wide)?;
                w.verify
                    .begin_verification(&changed[5..11], &w.comm, &w.stream)?;
                w.verify.finish_verification(6, &w.comm, &w.stream)?;
                wide.handoff_from(&w.verify)?;
            }
        }
        w.verify.handoff_from(wide)?;
        w.verify
            .begin_verification(&following[..6], &w.comm, &w.stream)?;
        w.stream.synchronize()?;
        let (y, live) = w.verify.pending_local_logits_for_probe()?;
        ensure!(live == 6, "following M6 extent");
        let raw = unsafe { w.cuda.read_device(y.ptr(), 6 * 32320 * 4) }?;
        let exact = references
            .entry("following-m6".into())
            .or_insert_with(|| raw.clone())
            == &raw;
        w.verify.finish_verification(0, &w.comm, &w.stream)?;
        ensure!(
            w.control.all_admit(exact)?,
            "following M6 changed after M16 overlap commits"
        );
        ensure!(
            w.verify.position() == (n + 35) as u64,
            "changed-history committed frontier"
        );
        records.push(json!({"kind":"following_m6","arm":arm,"all_exact":true}));
    }
    w.target_lineage += 1;
    w.verify
        .restore_state_image(&image, w.target_lineage, &w.comm, &w.stream)?;
    w.verify.restore_repeat_physical_image(&physical, &w.comm, &w.stream)?;
    ensure!(
        w.verify.position() == n as u64
            && w.history.frontier() == n
            && (w.verify.carry_digests()?, w.verify.encoder_tail_digest()?) == original,
        "final original frontier"
    );
    execute(w, Command::Finish(Finish::Length))?;
    if std::env::var("CANADA_SCRATCH_POOLED_INHERITED_ROUTE_CAPTURE").as_deref() != Ok("1") {
        execute(w, Command::Shutdown)?;
    }
    std::fs::write(
        out.join("attention16-commit.json"),
        serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"prefix":n,"begin":begin,"snapshot_bytes":bytes,"records":records,
        "scope":"Old failing 274108-token frontier; constructed combined schedule, serial control, rebound combined. Two partial commits, changed history through M6, full M16 commit and following M6 logits. All local vocab shards, rollback and complete accepted target state. M3-only PDL constructor/control/restoration checked at three and six rows. No task-quality or general throughput claim."}))?,
    )?;
    Ok(())
}
