//! Diagnostic, never in a serving profile: CANADA_SCRATCH_VERIFIER_NUMERICS_PROBE=FIXTURE.json evaluates the same
//! six teacher-forced rows at each recorded frontier on the ordinary six-row verifier and on the 16-row verifier,
//! and rank 0 writes both owners' full-vocabulary FP32 logits for comparison with DeepSeek's reference forward and
//! its split-K yardstick (adapted from the tail-price probe).
//!
//! Fixture: {"tokens": [...], "frontiers": [p, ...], "out": "/scratch/..."}. At frontier p the accepted history is
//! tokens[..p] and the verified rows are tokens[p..p+6] (row i evaluates position p+i and predicts token p+i+1).
//! Output (rank 0): narrow.f32 and wide.f32, each frontiers x 6 x 129280 little-endian FP32 in fixture order, and
//! receipt.json with the per-frontier determinism and prefix checks. Every trial rolls back to the accepted state.
//! When the 32-row copy verifier is loaded (and 32 tokens follow every frontier) its first six rows go to wide32.f32
//! the same way: copy steps of 16 and 32 rows carry most tokens of a hinted file edit.
use super::{Command,Finish,Worker};
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{Context,Result,ensure};
use serde::Deserialize;
use serde_json::{json,Value};
use std::io::Write;

const VOCAB:usize=129280;
const SHARD:usize=32320;

#[derive(Deserialize)]
struct Fixture {tokens:Vec<u32>,frontiers:Vec<usize>,out:String}

fn state(target:&Target)->Result<Value> {
    Ok(json!([target.position(),target.carry_digests()?,target.encoder_tail_digest()?]))
}

/// Full-vocabulary logits of the first six rows, gathered from all four ranks (rank-major column blocks).
fn first_six(cuda:&Cuda,target:&Target,comm:&Roce,stream:&Stream)->Result<Vec<f32>> {
    let (out,live)=target.pending_local_logits_for_probe()?;
    ensure!(live>=6,"six common rows required");
    let common=unsafe{out.view(0,6*SHARD)}?;
    let mut gathered=cuda.allocate::<f32>(6*VOCAB)?;
    unsafe{comm.gather_f32(&common,&mut gathered,SHARD,stream)?;}
    stream.synchronize()?;
    let full=gathered.download()?;
    for row in 0..6 {
        let bytes=unsafe{cuda.read_device(out.ptr()+(row*SHARD*4) as u64,SHARD*4)}?;
        let local:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        let column=row*VOCAB+comm.rank()*SHARD;
        ensure!(full[column..column+SHARD]==local[..],"diagnostic vocabulary gather order");
    }
    ensure!(full.iter().all(|x|x.is_finite()),"verifier numerics: finite logits");
    Ok(full)
}

fn trial(cuda:&Cuda,small:&mut Target,big:&mut Target,tokens:&[u32],force_wide:bool,
         comm:&Roce,stream:&Stream)->Result<Vec<f32>> {
    let local=tokens.len()==6 && !force_wide;
    if !local{big.handoff_from(small)?;}
    let owner=if local{&mut *small}else{&mut *big};
    if local{owner.begin_local_verification(tokens,comm,stream)?;}else{owner.begin_verification(tokens,comm,stream)?;}
    owner.complete_deferred_verification(comm,stream)?;
    stream.synchronize()?;
    let logits=first_six(cuda,owner,comm,stream)?;
    owner.finish_verification(0,comm,stream)?;
    if !local{small.handoff_from(big)?;}
    Ok(logits)
}

fn bits_equal(a:&[f32],b:&[f32])->bool {a.len()==b.len()&&a.iter().zip(b).all(|(x,y)|x.to_bits()==y.to_bits())}

pub(super) fn run(worker:&mut Worker,rank:usize,path:&str)->Result<()> {
    let fixture:Fixture=serde_json::from_slice(&std::fs::read(path)?)?;
    let n=fixture.tokens.len();
    ensure!(!fixture.frontiers.is_empty()&&fixture.frontiers.iter().all(|&p|p>=1&&p+16<=n)
        &&fixture.tokens.iter().all(|&t|(t as usize)<VOCAB),"verifier numerics fixture extent");
    ensure!(worker.verify.physical_rows()==6&&worker.verify.local_target_enabled(),"verifier numerics: six-row owner");
    let out=std::path::Path::new(&fixture.out);
    // The worker may run as root; the host user that copies and removes the output gets write access to it.
    let open_to_host=|path:&std::path::Path,mode:u32|{
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path,std::fs::Permissions::from_mode(mode))
    };
    let mut files=if rank==0 {
        std::fs::create_dir(out)?;
        open_to_host(out,0o777)?;
        let open=|name:&str|std::fs::OpenOptions::new().create_new(true).append(true).open(out.join(name));
        Some((open("narrow.f32")?,open("wide.f32")?))
    }else{None};
    let with32=worker.copy32_verify.is_some()&&fixture.frontiers.iter().all(|&p|p+32<=n);
    let mut file32=if rank==0&&with32 {
        Some(std::fs::OpenOptions::new().create_new(true).append(true).open(out.join("wide32.f32"))?)
    }else{None};
    let mut receipts=Vec::new();
    for (k,&p) in fixture.frontiers.iter().enumerate() {
        let history=fixture.tokens[..p].to_vec();
        let ids=&fixture.tokens[p..p+16];
        let ids32=if with32{Some(fixture.tokens[p..p+32].to_vec())}else{None};
        worker.execute(Command::Begin{images:Vec::new(),id:46770000+k as u64,seed:87651,greedy:false,limit:24,tokens:history})?;
        ensure!(worker.verify.position()==p as u64&&worker.verify.profile_is_current(),"verifier numerics: accepted history");
        let before=state(&worker.verify)?;
        let wide=worker.long_verify.as_mut().context("verifier numerics: 16-row owner absent")?;
        ensure!(wide.physical_rows()==16,"verifier numerics: 16-row geometry");
        let narrow=trial(&worker.cuda,&mut worker.verify,wide,&ids[..6],false,&worker.comm,&worker.stream)?;
        let narrow_again=trial(&worker.cuda,&mut worker.verify,wide,&ids[..6],false,&worker.comm,&worker.stream)?;
        let wide_six=trial(&worker.cuda,&mut worker.verify,wide,&ids[..6],true,&worker.comm,&worker.stream)?;
        let wide_again=trial(&worker.cuda,&mut worker.verify,wide,&ids[..6],true,&worker.comm,&worker.stream)?;
        let wide_sixteen=trial(&worker.cuda,&mut worker.verify,wide,ids,false,&worker.comm,&worker.stream)?;
        // The 32-row copy verifier: six live rows, then all 32 (the first six must not change).
        let wide32=if let Some(ids32)=&ids32 {
            let copy32=worker.copy32_verify.as_mut().context("verifier numerics: 32-row owner absent")?;
            ensure!(copy32.physical_rows()==32,"verifier numerics: 32-row geometry");
            let six=trial(&worker.cuda,&mut worker.verify,copy32,&ids32[..6],true,&worker.comm,&worker.stream)?;
            let full=trial(&worker.cuda,&mut worker.verify,copy32,ids32,false,&worker.comm,&worker.stream)?;
            Some((six,full))
        }else{None};
        ensure!(before==state(&worker.verify)?,"verifier numerics: a trial changed the accepted state");
        worker.execute(Command::Finish(Finish::Cancelled))?;
        let mut receipt=json!({"frontier":p,"narrow_repeat_bitwise":bits_equal(&narrow,&narrow_again),
            "wide_repeat_bitwise":bits_equal(&wide_six,&wide_again),"wide16_prefix_bitwise":bits_equal(&wide_six,&wide_sixteen),
            "narrow_wide_bitwise":bits_equal(&narrow,&wide_six)});
        if let Some((six,full))=&wide32 {
            receipt["wide32_prefix_bitwise"]=json!(bits_equal(six,full));
            receipt["narrow_wide32_bitwise"]=json!(bits_equal(&narrow,six));
            receipt["wide16_wide32_bitwise"]=json!(bits_equal(&wide_six,six));
            if let Some(file)=file32.as_mut() {
                let bytes:Vec<u8>=six.iter().flat_map(|x|x.to_le_bytes()).collect();
                file.write_all(&bytes)?;
            }
        }
        if let Some((a,b))=files.as_mut() {
            for (file,values) in [(a,&narrow),(b,&wide_six)] {
                let bytes:Vec<u8>=values.iter().flat_map(|x|x.to_le_bytes()).collect();
                file.write_all(&bytes)?;
            }
        }
        eprintln!("verifier_numerics {}",json!({"rank":rank,"k":k,"receipt":receipt}));
        receipts.push(receipt);
    }
    if let Some((a,b))=files {
        a.sync_all()?;b.sync_all()?;
        std::fs::write(out.join("receipt.json"),serde_json::to_vec_pretty(&json!({"tokens":n,
            "frontiers":fixture.frontiers,"rows_per_frontier":6,"vocab":VOCAB,"receipts":receipts}))?)?;
        for name in ["narrow.f32","wide.f32","receipt.json"] {open_to_host(&out.join(name),0o666)?;}
        if let Some(file)=file32 {file.sync_all()?;open_to_host(&out.join("wide32.f32"),0o666)?;}
    }
    eprintln!("verifier_numerics_complete {}",json!({"rank":rank,"frontiers":fixture.frontiers.len()}));
    Ok(())
}
