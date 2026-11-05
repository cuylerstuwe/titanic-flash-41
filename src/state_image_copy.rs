//! Bounded raw-byte snapshot copy plan. No persistent GPU scratch, arithmetic,
//! new stream, changed ownership, or changed allocation policy.
use crate::cuda::{Cuda, Module, Stream};
use anyhow::{Result, ensure};
use std::{cell::Cell, ffi::c_void};

const CHUNK: usize = 128 << 10;
const LIMIT: usize = 128;
thread_local! { static ENABLED: Cell<bool> = const {Cell::new(false)}; }

pub(crate) fn enabled() -> bool { ENABLED.get() }
fn policy(value:Option<&str>)->Result<bool> {
    match value {None|Some("1")=>Ok(true),Some("0")=>Ok(false),
        _=>anyhow::bail!("state copy kernel must be0or1")}
}
pub(crate) fn configured()->Result<bool> {
    let value=match std::env::var("CANADA_SCRATCH_STATE_COPY_KERNEL") {
        Ok(v)=>Some(v),Err(std::env::VarError::NotPresent)=>None,Err(e)=>return Err(e.into()),
    };
    policy(value.as_deref())
}
pub(crate) fn configure_worker()->Result<bool> {
    let value=configured()?;ENABLED.set(value);Ok(value)
}
pub(crate) fn set_for_probe(enabled: bool) -> Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_STATE_COPY_SCREEN").as_deref()==Ok("1"),
        "state copy choice requires the finite file screen");
    ENABLED.set(enabled); Ok(())
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CopyRegion { source: u64, destination: u64, bytes: u64 }
#[repr(C)]
struct Plan { regions: [CopyRegion; LIMIT], end_blocks: [u32; LIMIT] }
const _: () = { assert!(size_of::<CopyRegion>()==24); assert!(size_of::<Plan>()==3584); };

fn aligned(&(source,destination,bytes):&(u64,u64,usize))->bool {(source|destination|bytes as u64)&15==0}
fn plan(regions: &[(u64,u64,usize)]) -> Option<(Plan,u32,u32)> {
    if !(16..=LIMIT).contains(&regions.len()) {return None;}
    let mut plan=Plan{regions:[CopyRegion::default();LIMIT],end_blocks:[0;LIMIT]};
    let mut total=0usize;let mut blocks=0u32;let mut count=0usize;
    for &(source,destination,bytes) in regions {
        if source==0||destination==0||bytes==0 {return None;}
        source.checked_add(bytes as u64)?;destination.checked_add(bytes as u64)?;
        // Keep odd-sized metadata and any subsequent unaligned packed range
        // on ordinary copies, without disqualifying all large aligned regions.
        if !aligned(&(source,destination,bytes)) {continue;}
        total=total.checked_add(bytes)?;
        blocks=blocks.checked_add(u32::try_from(bytes.div_ceil(CHUNK)).ok()?)?;
        plan.regions[count]=CopyRegion{source,destination,bytes:bytes as u64};
        plan.end_blocks[count]=blocks;count+=1;
    }
    (count>=16&&total>=4<<20).then_some((plan,blocks,count as u32))
}
pub(super) fn eligible(regions:&[(u64,u64,usize)])->bool {plan(regions).is_some()}

/// Safety: all copy ranges are valid, pairwise independent and exclusively
/// writable at the destination. Their lifetimes extend through stream join.
/// False means the caller must issue the ordinary copies instead.
pub(super) unsafe fn launch(cuda:&Cuda,module:&Module,regions:&[(u64,u64,usize)],stream:&Stream)->Result<bool> {
    let Some((mut plan,blocks,mut count))=plan(regions) else{return Ok(false);};
    unsafe{module.function("ca_state_copy_plan")?.launch([blocks,1,1],[256,1,1],0,stream,
        &mut[&mut plan as *mut _ as *mut c_void,&mut count as *mut _ as *mut c_void])}?;
    for &(source,destination,bytes) in regions.iter().filter(|r|!aligned(r)) {
        unsafe{cuda.copy_device(destination,source,bytes,stream)}?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidate_default_keeps_an_explicit_copy_control() {
        assert!(policy(None).unwrap());assert!(policy(Some("1")).unwrap());
        assert!(!policy(Some("0")).unwrap());assert!(policy(Some("2")).is_err());
    }
    #[test]
    fn bounded_plan_covers_regions_once_and_declines_unsuitable_geometry() {
        let regions=(0..128).map(|i|(0x10000000+i*0x100000,0x30000000+i*0x100000,
            if i%3==0 {CHUNK+16}else{CHUNK*2})).collect::<Vec<_>>();
        let (p,blocks,count)=plan(&regions).unwrap();assert_eq!(count,128);
        let mut next=0;
        for (i,&(_,_,bytes)) in regions.iter().enumerate() {
            assert_eq!(p.end_blocks[i]-next,bytes.div_ceil(CHUNK) as u32);
            let last=usize::try_from(p.end_blocks[i]-next-1).unwrap()*CHUNK;
            assert!(last<bytes&&bytes-last<=CHUNK);next=p.end_blocks[i];
        }
        assert_eq!(next,blocks);
        assert!(plan(&regions[..15]).is_none());
        let mut bad=regions.clone();bad.push(regions[0]);assert!(plan(&bad).is_none());
        for change in [1,2,3] {
            let mut bad=regions.clone();
            match change {1=>bad[0].1=0,2=>bad[0].2=0,_=>bad[0].0=u64::MAX-15};
            assert!(plan(&bad).is_none());
        }
        let small=vec![(256,512,16);16];assert!(plan(&small).is_none());
        let mut mixed=regions[..64].to_vec();mixed.insert(10,(0x50000000,0x60000000,4));
        mixed.push((0x70000000,0x80000004,4<<20));
        let (p,_,count)=plan(&mixed).unwrap();assert_eq!(count,64);
        assert_eq!(p.regions[10].source,regions[10].0,"odd metadata cannot shift descriptor identity");
        assert!(!aligned(&mixed[10]));assert!(!aligned(mixed.last().unwrap()));
    }
}
