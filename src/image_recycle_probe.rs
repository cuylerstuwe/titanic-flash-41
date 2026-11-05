//! One-device component: recycle an exclusively retired snapshot allocation.
//! No model, persistent pool, cache eviction policy or serving activation.
use crate::{cuda::{Buffer,Cuda},state_image::{Image,Region}};
use anyhow::{Result,ensure,Context};
use serde_json::{Value,json};
use std::{collections::BTreeMap,path::Path,time::Instant};

fn vmstat()->Result<BTreeMap<String,u64>> {
    Ok(std::fs::read_to_string("/proc/vmstat")?.lines().filter_map(|line| {
        let mut w=line.split_whitespace();let k=w.next()?;let v=w.next()?.parse().ok()?;
        (k.starts_with("pgscan_direct")||k.starts_with("pgsteal_direct")
            ||k.starts_with("allocstall")||k=="compact_stall").then(||(k.to_owned(),v))
    }).collect())
}
fn regions<'a>(c:&Cuda,b:&'a Buffer<u8>,reverse:bool)->Result<Vec<Region<'a>>> {
    let n=b.bytes()/32;
    (0..32).map(|i| {
        let physical=if reverse{31-i}else{i};
        Region::byte_range(c,format!("part{i}"),b,physical*n..(physical+1)*n)
    }).collect()
}

pub fn run(core:&Path)->Result<Value> {
    let c=Cuda::open(0)?;c.qualify_gb10()?;let stream=c.stream()?;
    let module=c.module(&std::fs::read(core)?)?;
    let baseline_charge=crate::memory::retained_image_bytes();
    let mut rows=Vec::new();
    for bytes in [64usize<<20,256usize<<20] {
        let mut a=c.allocate::<u8>(bytes)?;let mut b=c.allocate::<u8>(bytes)?;
        let mut values=vec![17u8;bytes];a.upload(&values)?;
        for(arm,recycle)in [false,true,true,false,false,true,true,false].into_iter().enumerate() {
            let old_regions=regions(&c,&a,false)?;
            let mut old=Image::allocate(&c,&old_regions,bytes,&stream)?;
            old.configure_copy_kernel(&module)?;old.save(&old_regions,&stream)?;
            ensure!(crate::memory::retained_image_bytes()==baseline_charge+bytes,
                "old image charge disagrees with physical storage");
            // Distinct source address, changed ordering and changed bytes must
            // all replace the retired image's old descriptor/capture metadata.
            for (i,v)in values.iter_mut().enumerate(){*v=((i%251+arm*29+31)%256)as u8;}
            b.upload(&values)?;let current=regions(&c,&b,true)?;
            let before=vmstat()?;let at=Instant::now();
            let mut image=if recycle {old.try_recycle(&current,bytes,&stream)?.context("compatible recycle refused")?}
                else{drop(old);Image::allocate(&c,&current,bytes,&stream)?};
            let allocation_seconds=at.elapsed().as_secs_f64();let after=vmstat()?;
            ensure!(crate::memory::retained_image_bytes()==baseline_charge+bytes,
                "recycled image lost or duplicated its physical charge");
            let delta=after.into_iter().map(|(k,v)|{let n=v.saturating_sub(*before.get(&k).unwrap_or(&0));(k,n)}).collect::<BTreeMap<_,_>>();
            ensure!(image.bytes()==bytes&&image.digest().is_err(),"recycle retained stale capture state");
            image.configure_copy_kernel(&module)?;let at=Instant::now();image.save(&current,&stream)?;
            let copy_seconds=at.elapsed().as_secs_f64();drop(current);
            ensure!(image.validate(&old_regions,&stream).is_err(),"recycle retained old address identity");
            b.zero()?;let current=regions(&c,&b,true)?;
            unsafe{image.restore(&current,&stream)}?;drop(current);
            ensure!(b.download()?==values,"recycled image restore changed bytes");
            let manifest=image.manifest();let at=Instant::now();drop(image);
            let release_seconds=at.elapsed().as_secs_f64();
            ensure!(crate::memory::retained_image_bytes()==baseline_charge,"released image retained its charge");
            rows.push(json!({"bytes":bytes,"arm":arm,"recycled":recycle,
                "retire_and_allocate_seconds":allocation_seconds,"copy_seconds":copy_seconds,
                "release_seconds":release_seconds,"vmstat_delta":delta,"bytes_exact":true,"charged_storage_exact":true,
                "manifest":manifest,"memory":c.admission_memory()?}));
        }
        // Refuse an incompatible size; the retired image is dropped, so its
        // storage and charge cannot survive as untracked spare capacity.
        let r=regions(&c,&a,false)?;let image=Image::allocate(&c,&r,bytes,&stream)?;
        let short=Region::byte_range(&c,"short",&b,0..bytes/2)?;
        ensure!(image.try_recycle(&[short],bytes,&stream)?.is_none(),"incompatible storage reused");
        ensure!(crate::memory::retained_image_bytes()==baseline_charge,"incompatible image retained spare storage");
    }
    Ok(json!({"records":rows,"scope":"One-device exact-size retired allocation reuse at64/256MiB. ABBA twice; address/order/content identity replaced, stale capture rejected, restored bytes exact, incompatible size refused. vmstat deltas surround retire/allocation. Unloaded model: no inference-state, pressure, real cache-policy, or throughput claim."}))
}
