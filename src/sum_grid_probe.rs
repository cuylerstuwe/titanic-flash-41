//! Short geometry/notification screens for hidden-state sums; no model loaded.
use crate::{cuda::Cuda,roce::Roce};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::path::Path;

fn bits(i:usize,rank:usize,phase:usize)->u16 {
    let h=(i as u32).wrapping_mul(0x9e3779b9)
        .wrapping_add((rank as u32+1)*0x01234567).wrapping_add(phase as u32*0x07654321);
    // Finite BF16 operands spanning signs/exponents, including cancellation.
    const SPECIAL:[u16;8]=[0,0x8000,0x3f80,0xbf80,0x4b00,0xcb00,1,0x8001];
    if i%17==0 {SPECIAL[(i/17+rank+phase)%8]}
    else {((h>>16)as u16&0x807f)|((((h>>9)%20+110)as u16)<<7)}
}

pub fn run(rank:usize,endpoint:&str,nonce:&[u8],root:&Path)->Result<Value>{
    let paired=std::env::var("CANADA_SCRATCH_PAIRED_SUM_GRID_SCREEN").as_deref()==Ok("1");
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let poll=std::env::var("CANADA_SCRATCH_NOTIFY_SCREEN").as_deref()==Ok("2");
    let notify=poll||std::env::var("CANADA_SCRATCH_NOTIFY_SCREEN").as_deref()==Ok("1");
    if notify{comm.attach_notification_probe(root)?;}
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;
    comm.connect(&all)?;
    let affinity=std::env::var("CANADA_SCRATCH_PROXY_AFFINITY_SCREEN").as_deref()==Ok("1");
    ensure!(!(notify&&affinity)&&!(paired&&(notify||affinity)),"separate geometry/notification/affinity screens");
    let original_cpus=if affinity{comm.proxy_cpus_for_probe()?}else{Vec::new()};
    let mut capacities=Vec::new();
    for &cpu in &original_cpus{
        let base=std::path::PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}"));
        let mhz:usize=std::fs::read_to_string(base.join("cpufreq/cpuinfo_max_freq"))?.trim().parse()?;
        let capacity:usize=std::fs::read_to_string(base.join("cpu_capacity"))?.trim().parse()?;
        capacities.push((cpu,mhz,capacity));
    }
    let fast_cpu=capacities.iter().max_by_key(|&&(cpu,freq,cap)|(freq,cap,cpu)).map(|x|x.0);
    ensure!(!affinity||original_cpus.len()>1&&fast_cpu.is_some(),"proxy CPU alternatives");
    let configure=|variant:usize|->Result<()>{
        if affinity{comm.set_proxy_cpus_for_probe(if variant>=2{std::slice::from_ref(fast_cpu.as_ref().unwrap())}else{&original_cpus},&stream)?;}
        Ok(())
    };
    let mut gathered_input=cuda.allocate::<f32>(8)?;
    gathered_input.upload(&vec![f32::from_bits(0x7fc00010+rank as u32);8])?;
    let mut gathered_output=cuda.allocate::<f32>(32)?;
    let mut results=Vec::new();
    for rows in if paired{[6usize,9,12]}else{[1usize,6,16]}{
        let columns=5120usize;let mut input=cuda.allocate::<u16>(rows*columns)?;
        let mut output=cuda.allocate::<u16>(rows*columns)?;
        input.zero()?;stream.join_default()?;
        let grids=if notify{vec![if rows==1{2usize}else if rows==6{4}else{8};4]}
            else if affinity{if rows==1{vec![2usize,1,2,1]}else{vec![8,4,8,4]}}
            else if rows==1{vec![2,1,4]}else{vec![8,4,16]};
        let mut graphs=Vec::new();let mut mixed=Vec::new();
        for (variant,&blocks) in grids.iter().enumerate(){
            if notify{comm.set_notification_probe(variant,&stream)?;}
            comm.set_sum_grid_for_probe(blocks,&stream)?;
            unsafe{comm.sum_bf16(&input,&mut output,&stream)?;}
            stream.synchronize()?;
            graphs.push(stream.capture(||{for _ in 0..8{
                unsafe{comm.sum_bf16(&input,&mut output,&stream)?;}
            }Ok(())})?);
            mixed.push(stream.capture(||unsafe{
                comm.gather_f32(&gathered_input,&mut gathered_output,8,&stream)?;
                comm.sum_bf16(&input,&mut output,&stream)?;
                comm.gather_f32(&gathered_input,&mut gathered_output,8,&stream)
            })?);
        }
        let mut cases=Vec::new();
        for phase in 0..2{
            input.upload(&(0..input.len()).map(|i|bits(i,rank,phase)).collect::<Vec<_>>())?;
            stream.join_default()?;
            let mut baseline=None;let mut hash=String::new();
            let validation=if affinity||notify{vec![0usize,1,2,3,2,1,0]}else{vec![0,1,2,1,0]};
            for variant in validation{
                configure(variant)?;
                unsafe{graphs[variant].launch(&stream)?;mixed[variant].launch(&stream)?;}
                stream.synchronize()?;comm.health()?;
                let raw=gathered_output.download()?;
                for peer in 0..4{ensure!(raw[peer*8..(peer+1)*8].iter()
                    .all(|x|x.to_bits()==0x7fc00010+peer as u32),"sum grid mixed gather bytes");}
                let actual=output.download()?;
                ensure!(actual.iter().all(|&x|half::bf16::from_bits(x).is_finite()),"sum grid finite output");
                if let Some(ref expected)=baseline{ensure!(&actual==expected,"sum grid arithmetic differs");}
                else{
                    hash=format!("{:x}",Sha256::digest(actual.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()));
                    baseline=Some(actual);
                }
            }
            let mut pairs=Vec::new();
            for repeat in 0..if affinity||notify{4}else{2}{
                let mut milliseconds=vec![0f32;grids.len()];
                let mut order=(0..grids.len()).collect::<Vec<_>>();
                order.rotate_left(repeat/2);if repeat%2==1{order.reverse();}
                for &variant in &order{
                    configure(variant)?;
                    unsafe{comm.gather_f32(&gathered_input,&mut gathered_output,8,&stream)?;}
                    for _ in 0..4{unsafe{graphs[variant].launch(&stream)?;}}
                    stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                    for _ in 0..8{unsafe{graphs[variant].launch(&stream)?;}}
                    end.record(&stream)?;milliseconds[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
                }
                pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
            }
            cases.push(json!({"phase":phase,"bytes_exact":true,"mixed_collective_ordering_exact":true,
                "sha256":hash,"pairs":pairs}));
        }
        results.push(json!({"rows":rows,"local_columns":columns,"input_bytes":input.bytes(),"grids":grids,"cases":cases}));
        eprintln!("rank{rank}: sum grid M{rows} exact");
    }
    if affinity{comm.set_proxy_cpus_for_probe(&original_cpus,&stream)?;}
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"all_exact":true,"results":results,
        "variants":if poll{vec!["stock","fresh-control","relaxed-poll-checked-acquire","relaxed-poll-acquire-fence"]}
            else if notify{vec!["stock","fresh-control","final-acq-rel","stage-and-final-acq-rel"]}
            else if affinity{vec!["original-default-grid","original-half-grid","pinned-default-grid","pinned-half-grid"]}else{vec!["default","half","double"]},
        "notification_probe":notify,
        "poll_probe":poll,
        "paired_geometry_probe":paired,
        "proxy_affinity_probe":affinity,"original_cpus":original_cpus,"cpu_capacities":capacities,"fast_cpu":fast_cpu,
        "scope":if paired{"Same native BF16 sum primitive, fixed-rank arithmetic, and protocol at combined M6/M9/M12. Grids8/4/16, two changed operands and reversed warm timings. Exact outputs across geometries and mixed raw gathers. No weights or full model; excludes pair packing/scattering and is not a whole-model performance claim."}
            else if poll{"Relaxed system flag polling followed by either a checked acquire load or system acquire/release fence. Retains original staging/poison/tail/epoch and fixed-rank math; original bounded unsuccessful-poll limit. Current M1/6/16 grids2/4/8; two finite inputs, four rotated/reversed orders, exact four-peer bytes and mixed gathers. Component only."}
            else if notify{"SC versus acquire/release system staging fences; retains acquire of other CTAs, payload publication, peer acquire waits, poison/epoch/tail and fixed-rank math. Stock/fresh-control/final-only/both-staging-fences. Current M1/6/16 grids2/4/8. Two finite inputs, four rotated/reversed timing orders, exact four-peer bytes and mixed gathers. Component only."}
            else if affinity{"Native BF16 sums; only the isolated RDMA progress thread is pinned to an allowed faster CPU, crossed with default/half GPU grid. Hidden5120 M1/6/16; two finite inputs, four rotated/reversed timing orders each, exact outputs and mixed gathers. Affinity restored afterward. No host-global or serving policy change, component only."}else{"Same native BF16 sum primitive, precision, reduction order and protocol. Hidden5120 M1/6/16 launch grids default/half/double. Two finite changed patterns, opposite timing orders, exact outputs against baseline and mixed raw gathers. Component only."}}))
}

/// Finite changing-payload transport screen; no weights, target or CPU oracle.
pub fn run_sequence(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value>{
    ensure!(rank<4&&std::env::var("CANADA_SCRATCH_SUM_GRID_SCREEN").as_deref()==Ok("1"),
        "isolated changing-payload reduction admission");
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let module=cuda.module(&std::fs::read(core)?)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;comm.connect(&all)?;
    let input=cuda.allocate::<u16>(6*5120)?;let mut output=cuda.allocate::<u16>(6*5120)?;
    let gathered=cuda.allocate::<u16>(4*6*5120)?;
    let mut epoch=cuda.allocate::<u32>(1)?;let mut stats=cuda.allocate::<u32>(7)?;
    let epoch_ptr=epoch.ptr();let stats_ptr=stats.ptr();
    let fill=|n:usize,pattern:i32,jitter:i32|->Result<()>{
        let mut x=input.ptr();let mut e=epoch_ptr;let mut n=n as i32;let mut r=rank as i32;
        let mut p=pattern;let mut j=jitter;
        unsafe{module.function("ca_sequence_fill")?.launch([(n as u32).div_ceil(256),1,1],[256,1,1],0,&stream,
            &mut[(&mut x as *mut u64).cast(),(&mut e as *mut u64).cast(),(&mut n as *mut i32).cast(),
                (&mut r as *mut i32).cast(),(&mut p as *mut i32).cast(),(&mut j as *mut i32).cast()])?;}Ok(())
    };
    let check=|ptr:u64,n:usize,pattern:i32,gather:bool|->Result<()>{
        let mut y=ptr;let mut e=epoch_ptr;let mut s=stats_ptr;let mut n=n as i32;let mut p=pattern;let mut g=i32::from(gather);
        unsafe{module.function("ca_sequence_check")?.launch([((n as u32)*if gather{4}else{1}).div_ceil(256),1,1],[256,1,1],0,&stream,
            &mut[(&mut y as *mut u64).cast(),(&mut e as *mut u64).cast(),(&mut s as *mut u64).cast(),
                (&mut n as *mut i32).cast(),(&mut p as *mut i32).cast(),(&mut g as *mut i32).cast()])?;}Ok(())
    };
    let advance=||->Result<()>{let mut e=epoch_ptr;unsafe{module.function("ca_sequence_advance")?.launch([1,1,1],[32,1,1],0,&stream,
        &mut[(&mut e as *mut u64).cast()])?;}Ok(())};
    // Independent negative control: the previous operation's real reduced
    // bytes must fail the next epoch's check. It does not feed later tests.
    epoch.upload(&[0])?;stats.upload(&[0,u32::MAX,u32::MAX,0,0,0,0])?;stream.join_default()?;
    comm.set_sum_grid_for_probe(4,&stream)?;fill(6*5120,0,0)?;
    unsafe{comm.sum_bf16(&input,&mut output,&stream)?;}
    check(output.ptr(),6*5120,0,false)?;stream.synchronize()?;let correct=stats.download()?;
    advance()?;check(output.ptr(),6*5120,0,false)?;stream.synchronize()?;
    let stale=stats.download()?;ensure!(correct[0]==0&&stale[0]>0,"changing-payload oracle negative control");
    let mut results=Vec::new();
    for (variant,blocks,pattern,jitter,mixed)in [(0,4,0,0,false),(1,4,1,0,true),(2,4,1,1,true),(3,8,1,1,true)]{
        comm.set_sum_grid_for_probe(blocks,&stream)?;
        stats.upload(&[0,u32::MAX,u32::MAX,0,0,0,0])?;stream.join_default()?;
        let graph=stream.capture(||{
            for step in 0..64{
                let n=if mixed&&step%4==1{3*5120}else{6*5120};fill(n,pattern,jitter)?;
                let x=unsafe{input.view(0,n)}?;let mut y=unsafe{output.view(0,n)}?;
                unsafe{comm.sum_bf16(&x,&mut y,&stream)?;}
                check(output.ptr(),n,pattern,false)?;
                if mixed&&step%4==3{
                    let mut dst=unsafe{gathered.view(0,n*4)}?;
                    unsafe{comm.gather_bf16(&x,&mut dst,5120,&stream)?;}
                    check(gathered.ptr(),n,pattern,true)?;
                }
                advance()?;
            }Ok(())
        })?;
        let start=std::time::Instant::now();
        for _ in 0..32{unsafe{graph.launch(&stream)?;}}
        stream.synchronize()?;comm.health()?;
        let actual=stats.download()?;let end_epoch=epoch.download()?[0];
        results.push(json!({"variant":variant,"sum_grid":blocks,"pattern":pattern,"rank_stagger":jitter!=0,
            "mixed_shapes_and_gathers":mixed,"reductions":2048,"gathers":if mixed{512}else{0},
            "seconds":start.elapsed().as_secs_f64(),"end_epoch":end_epoch,"mismatched_elements":actual[0],
            "first_bad_epoch":if actual[1]==u32::MAX{None}else{Some(actual[1])},
            "first_error_raw":[actual[2],actual[3],actual[4],actual[5],actual[6]],"passed":actual[0]==0}));
        eprintln!("rank{rank}: sequence variant{variant} mismatches={}",actual[0]);
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"all_exact":results.len()==4&&results.iter().all(|r|r["passed"]==true),
        "negative_control_detected_stale_payload":true,"negative_control_bad_elements":stale[0],"results":results,
        "memory":cuda.admission_memory()?,"scope":"Changing GPU-generated canonical BF16 operands per operation. GPU fixed-rank FP32 sum then BF16 RN, exact raw gathers; M6/M3, both parities, asynchronous rank staggering, sums grids4/8. No model loaded and no CPU tensor arithmetic; isolates transport correctness, not full-model quality or serving speed."}))
}
