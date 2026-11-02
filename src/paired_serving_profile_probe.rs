//! Complete profile coexistence for the finite two-owner experiment. Both
//! owners keep their full context, width bank and M16/M32 copy capabilities.
use super::*;

pub(super) fn enabled()->bool {
    std::env::var("CANADA_SCRATCH_PAIRED_SERVING_PROFILE_SCREEN").as_deref()==Ok("1")
}
pub(super) struct Profiles {
    long:Target,
    copy:Target,
    _long_buffers:LongHistoryBuffers,
    _copy_buffers:LongHistoryBuffers,
    pub(super) memory:Value,
    repeats: RepeatCapture,
}
struct RepeatCapture {
    first:std::collections::BTreeMap<(usize,usize,u64),Vec<u8>>,
    directory:PathBuf,
}
impl RepeatCapture {
    fn observe(&mut self,owner:usize,rows:usize,position:u64,rank:usize,bytes:Vec<u8>)->Result<()> {
        ensure!(bytes.len()%4==0&&bytes.chunks_exact(4)
            .all(|b|f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "copy profile produced nonfinite target logits");
        let key=(owner,rows,position);
        if let Some(first)=self.first.get(&key) {
            if first!=&bytes {
                ensure!(first.len()==bytes.len()&&bytes.len()%4==0,"copy repeat logit extent");
                let mut changed=0usize;let mut nonfinite=0usize;let mut max_abs=0f64;
                for(a,b)in first.chunks_exact(4).zip(bytes.chunks_exact(4)) {
                    changed+=usize::from(a!=b);
                    let x=f32::from_le_bytes(a.try_into().unwrap());
                    let y=f32::from_le_bytes(b.try_into().unwrap());
                    if x.is_finite()&&y.is_finite(){max_abs=max_abs.max((x as f64-y as f64).abs());}
                    else{nonfinite+=1;}
                }
                let name=format!("copy-repeat-owner{owner}-M{rows}-pos{position}");
                for(suffix,data)in [("before",first.as_slice()),("after",bytes.as_slice())] {
                    let path=self.directory.join(format!("{name}-{suffix}.f32"));
                    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(path)?;
                    file.write_all(data)?;
                }
                eprintln!("copy_profile_logit_repeat {}",json!({"rank":rank,"owner":owner,
                    "rows":rows,"position":position,"changed_values":changed,
                    "nonfinite_values":nonfinite,"max_abs":max_abs,"capture":name}));
            }
        }else{self.first.insert(key,bytes);}
        Ok(())
    }
}
impl Profiles {
    pub(super) fn new(w:&mut Worker,other:&Target,prepared:&mut PreparedSession,kernels:&Path,out:&Path)->Result<Self> {
        ensure!(enabled()&&w.long_verify.is_some()&&w.copy32_verify.is_some()
            &&w.long_history_buffers.is_some()&&w.copy32_buffers.is_some()
            &&w.large.context_capacity()==1048576&&other.context_capacity()==1048576,
            "both complete copy profiles required");
        let module=w.cuda.module(&w.core)?;
        let fork=|rows|->Result<Target> {
            if std::env::var("CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN").as_deref()==Ok("1") {
                other.fork_profile_lending_prefill_for_probe(&module,&w.core,&w.stream,kernels,rows,Phase::Verify,
                    w.wides.iter().max_by_key(|t|t.physical_rows()).context("complete peer donor bank")?)
            }else{other.fork_profile(&module,&w.core,&w.stream,kernels,rows,Phase::Verify)}
        };
        let mut long=fork(16)?;
        long.set_graph_replay(true)?;
        let compact=Target::compact_history_sampling_requested()?;
        ensure!(compact,"keep serving compact history sampling");
        long.enable_compact_history_sampling(&w.core,&w.stream)?;
        let long_buffers=LongHistoryBuffers::new(&w.cuda,&w.core)?;
        let before=w.cuda.admission_memory()?;
        // With transient-arena loans and immutable encodings shared, the
        // primary M32 constructor consumed 0.44--0.62 GiB on all four devices
        // in the full-capacity s1128 screen. Keep the 9 GiB post-construction
        // requirement, plus 768 MiB for this now-smaller constructor. The
        // independent per-allocation floor remains active throughout.
        let compact_constructor = ["CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN",
            "CANADA_SCRATCH_SHARED_MATRIX_ENCODING_SCREEN",
            "CANADA_SCRATCH_ALLOCATION_FLOOR_SCREEN"].iter()
            .all(|key|std::env::var(key).as_deref()==Ok("1"));
        let construction_headroom = 9*1024usize.pow(3)
            + if compact_constructor {768*1024usize.pow(2)} else {3*1024usize.pow(3)};
        eprintln!("rank{} second M32 admission: available={} required={construction_headroom}",
            w.comm.rank(),before.admitted_available);
        ensure!(w.control.all_admit(before.admitted_available>=construction_headroom)?,
            "second M32 construction would consume serving reserve");
        let mut copy=fork(32)?;
        copy.set_graph_replay(true)?;copy.enable_compact_history_sampling(&w.core,&w.stream)?;
        let copy_buffers=LongHistoryBuffers::with_rows(&w.cuda,&w.core,32)?;
        ensure!(prepared.prepare_copy_conditioning(&copy_buffers,&w.comm,&w.stream)?,
            "second wide copy conditioning missing");
        w.stream.synchronize()?;
        let after=w.cuda.admission_memory()?;
        ensure!(w.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,
            "complete second profiles consumed serving reserve");
        Ok(Self{long,copy,_long_buffers:long_buffers,_copy_buffers:copy_buffers,
            repeats:RepeatCapture{first:std::collections::BTreeMap::new(),directory:out.to_owned()},
            memory:json!({"before_second_M32":before,"after_all_profiles":after,
                "second_M32_construction_headroom_bytes":construction_headroom})})
    }
    pub(super) fn check(&mut self,w:&mut Worker,verify:&mut Target,teacher:&[u32])->Result<Value> {
        let first=check_owner(0,&mut w.verify,w.long_verify.as_mut().unwrap(),w.copy32_verify.as_mut().unwrap(),
            teacher,&w.cuda,&w.comm,&w.stream,&mut self.repeats)?;
        let second=check_owner(1,verify,&mut self.long,&mut self.copy,teacher,&w.cuda,&w.comm,&w.stream,&mut self.repeats)?;
        Ok(json!([first,second]))
    }
}

fn check_owner(owner:usize,base:&mut Target,long:&mut Target,copy:&mut Target,teacher:&[u32],
    cuda:&Cuda,comm:&Roce,stream:&Stream,repeats:&mut RepeatCapture)->Result<Value> {
    // A wider rejected step legally writes additional future cells. Compare
    // published rows, carry, encoder and following logits; whole-page storage
    // hashes incorrectly require that unpublished cells be erased as well.
    let before=published_state(base,teacher,cuda,comm,stream)?;
    let mut records=Vec::new();
    for target in [long,copy] {
        target.handoff_from(base)?;
        let rows=target.physical_rows();
        let ids=teacher.iter().copied().cycle().take(rows).collect::<Vec<_>>();
        target.begin_verification(&ids,comm,stream)?;
        let(y,live)=target.pending_local_logits_for_probe()?;
        ensure!(live==rows,"full copy profile live rows");
        let bytes=unsafe{cuda.read_device(y.ptr(),live*32320*4)}?;
        let logits=format!("{:x}",Sha256::digest(&bytes));
        if owner==0&&rows==32&&target.position()>100000
            &&std::env::var("CANADA_SCRATCH_COPY_PROFILE_TRACE").as_deref()==Ok("1") {
            let phase=if repeats.first.contains_key(&(owner,rows,target.position())){"repeat"}else{"first"};
            target.copy_repeat_trace(&repeats.directory.join(format!(
                "copy-trace-owner{owner}-M{rows}-pos{}-{phase}",target.position())),stream)?;
        }
        repeats.observe(owner,rows,target.position(),comm.rank(),bytes)?;
        let draws=target.verification_history_sample(9173,comm,stream)?;
        target.finish_verification(0,comm,stream)?;
        base.handoff_from(target)?;
        let after=published_state(base,teacher,cuda,comm,stream)?;
        if after!=before {
            eprintln!("copy_profile_rollback {}",json!({"rank":comm.rank(),"owner":owner,
                "rows":rows,"before":before,"after":after}));
        }
        ensure!(after==before,"copy profile rollback changed retained state owner{owner} M{rows}");
        records.push(json!({"rows":rows,"logits":logits,"draws":draws,"rollback_exact":true}));
    }
    Ok(json!(records))
}

fn published_state(target:&mut Target,teacher:&[u32],cuda:&Cuda,comm:&Roce,stream:&Stream)->Result<Value> {
    let following=super::logits(target,teacher,0,cuda,comm,stream)?;
    Ok(json!({"position":target.position(),"following_logits":following,
        "carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?,
        "regions":target.published_state_digests_for_probe(stream)?}))
}
