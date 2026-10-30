//! Short paired fusion screen, with the accumulated copy/schedule candidate.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::Path,time::Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,candidate:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    let query=std::env::var("CANADA_SCRATCH_QUERY_NORM_QUANT_SCREEN").as_deref()==Ok("1");
    let hint=std::env::var("CANADA_SCRATCH_INPUT_HINT_SCREEN").as_deref()==Ok("1");
    let hint_off=std::env::var("CANADA_SCRATCH_INPUT_HINT_OFF_SCREEN").as_deref()==Ok("1");
    let prefix=if hint_off{"input-hint-off"}else if hint{"input-hint"}else if query{"query-norm-quant"}else{"norm-quant"};
    let diagnostic=std::env::var("CANADA_SCRATCH_NORM_QUANT_DIAGNOSTIC").as_deref()==Ok("1");
    let bundled=std::env::var("CANADA_SCRATCH_TINY_ENCODER_SCREEN").as_deref()==Ok("1");
    // Repeat only the tiny timing loop, not another context/quality matrix.
    let (warmup_pairs,measured_pairs)=if diagnostic{(0,1)}else if bundled{(2,6)}else{(2,2)};
    let control_schedule=verify.schedule_counts_for_probe();
    let candidate_schedule=candidate.schedule_counts_for_probe();
    let schedules_match=if hint_off {
        control_schedule[0]==candidate_schedule[0]+37 && control_schedule[1..]==candidate_schedule[1..]
    }else{control_schedule==candidate_schedule};
    ensure!(!hint_off || hint,"input hint disable screen parent admission");
    ensure!((query || std::env::var("CANADA_SCRATCH_NORM_QUANT_SCREEN").as_deref()==Ok("1"))
        && !(query && (diagnostic || bundled))
        && large.context_capacity()==1048576
        && verify.norm_quant_boundaries()==if query{80}else{0} && candidate.norm_quant_boundaries()==80
        && verify.query_norm_quant_layers()==if hint{40}else{0} && candidate.query_norm_quant_layers()==if query{40}else{0}
        && (!hint || (verify.input_hint_layers()==37 && verify.large_input_hint_layers()==0
            && candidate.input_hint_layers()==if hint_off{0}else{37}
            && candidate.large_input_hint_layers()==if hint_off{0}else{37}))
        && verify.fused_shared_quantization_layers()==40 && candidate.fused_shared_quantization_layers()==40
        && verify.cached_index_gather_layers()==8 && candidate.cached_index_gather_layers()==8
        && verify.staged_metadata_bytes()>0 && verify.staged_metadata_bytes()==candidate.staged_metadata_bytes()
        && verify.packed_index_gather_layers()==8 && candidate.packed_index_gather_layers()==8
        && verify.packed_sparse_gather_layers()==4 && candidate.packed_sparse_gather_layers()==4
        && schedules_match
        && verify.fused_compact_publication_layers()==4 && candidate.fused_compact_publication_layers()==4,
        "norm/quant fusion requires otherwise equal accumulated profiles");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    ensure!(format!("{:x}",Sha256::digest(raw))=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb",
        "norm/quant fusion fixture identity");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let frontier=262144;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let prefill=Instant::now();large.prefill_wide_with_prefetch(wides,&tokens[..frontier],comm,stream,true)?;
    let prefill_seconds=prefill.elapsed().as_secs_f64();verify.handoff_from(large)?;
    let carry=verify.carry_digests()?;let encoder=verify.encoder_tail_digest()?;
    let mut records=Vec::new();
    let mut all_exact=true;let mut diagnostic_report=Value::Null;
    let mut reference_snapshot=None;
    for &live in if diagnostic{&[3][..]}else{&[3,6][..]} {
        let mut reference=None;
        let mut by_path: [Option<Value>;2] = [None,None];
        for iteration in 0..warmup_pairs+measured_pairs {
            for path in [iteration%2,1-iteration%2] {
                if path==1 {candidate.handoff_from(verify)?;}
                let target=if path==1 {&mut *candidate}else{&mut *verify};
                // Handoff invalidates cached gathered input. Prime outside the
                // timed interval so this measures the steady verification path.
                target.begin_verification(&tokens[frontier..frontier+live],comm,stream)?;
                target.verification_greedy(stream)?;
                target.finish_verification(0,comm,stream)?;
                stream.join_default()?;
                unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let start=Instant::now();target.begin_verification(&tokens[frontier..frontier+live],comm,stream)?;
                let predictions=target.verification_greedy(stream)?;
                let milliseconds=start.elapsed().as_secs_f64()*1000.;
                let phases=serde_json::to_value(target.verification_metrics()?)?;
                let (logits,rows)=target.pending_logits()?;ensure!(rows==live,"norm/quant fusion live logits");
                let logit_bytes=unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?;
                ensure!(logit_bytes.chunks_exact(4).all(|x|f32::from_le_bytes(x.try_into().unwrap()).is_finite()),
                    "norm/quant fusion nonfinite logits");
                if iteration == 0 {
                    std::fs::write(output.join(format!("{prefix}-first-m{live}-v{path}.f32")),&logit_bytes)?;
                }
                let signature=json!({"logits":format!("{:x}",Sha256::digest(&logit_bytes)),
                    "predictions":predictions});
                let exact=reference.as_ref().is_none_or(|r|r==&signature);
                ensure!(by_path[path].as_ref().is_none_or(|r|r==&signature),
                    "norm/quant repeated same-path output changed");
                by_path[path]=Some(signature.clone());
                if reference.is_none(){reference=Some(signature.clone());}
                all_exact &= exact;
                if diagnostic {
                    stream.synchronize()?;
                    let snapshot=snapshot(cuda,target)?;
                    std::fs::write(output.join(format!("norm-quant-logits-v{path}.f32")),&logit_bytes)?;
                    if path==0 { reference_snapshot=Some(snapshot); }
                    else {diagnostic_report=compare_snapshots(reference_snapshot.as_ref().unwrap(),&snapshot,output)?;}
                }
                target.finish_verification(0,comm,stream)?;
                let rollback=target.position()as usize==frontier && target.carry_digests()?==carry
                    && target.encoder_tail_digest()?==encoder;
                let record=json!({"live_rows":live,"iteration":iteration,"warmup":iteration<warmup_pairs,
                    "fused":path==1,"milliseconds":milliseconds,"phases":phases,"signature":signature,
                    "same_signature":exact,"rollback_exact":rollback,"fused_boundaries":if hint{if hint_off{37-target.input_hint_layers()}else{target.large_input_hint_layers()}}else if query{target.query_norm_quant_layers()}else{target.norm_quant_boundaries()},
                    "capture_statistics":target.capture_statistics()});
                std::fs::write(output.join(format!("{prefix}-m{live}-i{iteration}-v{path}.json")),
                    serde_json::to_vec_pretty(&record)?)?;
                // Same-precision reduction changes are evidence to diagnose,
                // not a reason to discard the timing or abort the other cases.
                ensure!(rollback,"norm/quant fusion changed rollback state; saved evidence");
                if path==1 {verify.handoff_from(candidate)?;}
                records.push(record);
            }
        }
    }
    Ok(json!({"rank":rank,"norm_quant_screen":!query,"query_norm_quant_screen":query&&!hint,"input_hint_screen":hint,"input_hint_off_screen":hint_off,"all_exact":all_exact,"diagnostic":diagnostic,"diagnostic_report":diagnostic_report,"frontier":frontier,
        "warmup_pairs":warmup_pairs,"measured_pairs":measured_pairs,
        "records":records,"prefill_seconds":prefill_seconds,"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":if hint_off{format!("One262K frontier; remove37next-input read-only cache hints, retain80QB/router hints and every arithmetic/precision boundary. {warmup_pairs}warm and{measured_pairs}balanced measured pairs atM3/M6; four-rank logits/state/rollback. No outcome or production claim.")}else if hint{format!("One262K frontier, accumulated mHC/query fusion on both paths;37next-input cache hints use1MiB bulk requests instead of16KiB. QB/router hints, byte ranges, streams and arithmetic unchanged. {warmup_pairs} warmup and {measured_pairs} balanced measured pairs atM3/M6, four-rank logits/state/rollback. Component-guided scheduling screen, no outcome or production claim.")}else if query{format!("One262K frontier, accumulated1M candidate including layout8 mHC fusion on both paths. Fuse existing CuTe RMS/BF16 rounding with the exact FlashInfer MXFP8 quantizer at40Q-B boundaries. {warmup_pairs} excluded warmup pairs and {measured_pairs} balanced measured pairs at M3/M6; full FP32 logits, peer agreement and rollback. Fixed input timing; target sampler remains T1/P1/high reasoning.")}else{format!("One262K frontier and accumulated1M candidate. Fuse canonical BF16 mHC combine/RMS with the unchanged legacy MXFP8 input quantizer at79projection boundaries; retain BF16 routing input. One untimed cache prime per measurement, {warmup_pairs} excluded warmup pairs and {measured_pairs} balanced measured pairs at M3/M6. Full FP32 logits, peer agreement, graph reuse and rollback state. Fixed-input greedy readout isolates kernel timing; intended sampler remains T1/P1/high reasoning.")}}))
}

type Snapshot=Vec<BTreeMap<String,Vec<u8>>>;
fn snapshot(cuda:&Cuda,target:&Target)->Result<Snapshot> {
    let mut result=Vec::new();
    for layer in target.layer_owners() {
        for (boundary,ordinary) in [(layer.attn_boundary(),layer.attention().projection_quantized_input()),
            (layer.ffn_boundary(),layer.ffn().projection_quantized_input())] {
            let(q,scales)=boundary.quantized_input().unwrap_or(ordinary);
            let mut record:BTreeMap<String,Vec<u8>>=boundary.norm_quant_diagnostic_inputs()?.into_iter()
                .map(|(k,v)|(k.to_owned(),v)).collect();
            for (key,ptr,bytes) in [("normalized",boundary.input().ptr(),boundary.input().bytes()),
                ("quantized",q.ptr(),q.bytes()),("mma_scales",scales.ptr(),scales.bytes())] {
                record.insert(key.to_owned(),unsafe{cuda.read_device(ptr,bytes)}?);
            }
            result.push(record);
        }
    }
    ensure!(result.len()==80,"norm snapshot boundary count");Ok(result)
}
fn compare_snapshots(reference:&Snapshot,candidate:&Snapshot,output:&Path)->Result<Value> {
    let mut records=Vec::new();let mut first=None;
    for (index,(a,b)) in reference.iter().zip(candidate).enumerate() {
        ensure!(a.keys().eq(b.keys()),"norm snapshot field contract");
        let mut fields=Vec::new();let mut changed=false;
        for (key,av) in a {
            let bv=&b[key];ensure!(av.len()==bv.len(),"norm snapshot size");
            let different=av.iter().zip(bv).filter(|(x,y)|x!=y).count();
            if ["normalized","quantized","mma_scales"].contains(&key.as_str()) {changed|=different!=0;}
            fields.push(json!({"key":key,"bytes":av.len(),"different_bytes":different,
                "control_sha256":format!("{:x}",Sha256::digest(av)),"candidate_sha256":format!("{:x}",Sha256::digest(bv))}));
        }
        if changed && first.is_none() {
            let dir=output.join("norm-quant-first-difference");std::fs::create_dir(&dir)?;
            for (variant,values) in [("control",a),("candidate",b)] {
                for (key,bytes) in values {std::fs::write(dir.join(format!("{variant}-{key}.bin")),bytes)?;}
            }
            first=Some(index);
        }
        records.push(json!({"index":index,"layer":index/2,"branch":if index%2==0{"attention"}else{"ffn"},"fields":fields}));
    }
    let result=json!({"first_changed_boundary":first,"records":records,
        "scope":"Untimed read-only buffers from one identical accepted prefix. Saves first differing boundary inputs and both FP32 logit tensors; does not establish task quality."});
    std::fs::write(output.join("norm-quant-diagnostic.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
