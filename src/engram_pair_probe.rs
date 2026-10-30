//! Byte-I/O screen only: no model arithmetic, CUDA context or GPU allocation.
use crate::engram::{PackedShard,RowStore,ROW_BYTES};
use anyhow::{Result,ensure};
use serde_json::{json,Value};
use std::{fs::File,io::{Read,Write},path::Path,time::Instant};

fn parallel(a:&RowStore,ia:&[u64],oa:&mut[u8],b:&RowStore,ib:&[u64],ob:&mut[u8])->Result<()> {
    std::thread::scope(|scope|->Result<()>{
        let other=scope.spawn(||b.lookup_into(ib,ob,8));
        let first=a.lookup_into(ia,oa,8);
        let second=other.join().map_err(|_|anyhow::anyhow!("Engram probe worker panicked"))?;
        first?;second?;Ok(())
    })
}
fn median(mut values:Vec<f64>)->f64 {
    values.sort_by(f64::total_cmp);let n=values.len();(values[(n-1)/2]+values[n/2])*0.5
}
pub fn run(backing:&Path,rank:usize)->Result<Value> {
    ensure!(rank<4 && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1"),"Engram pair probe admission");
    let mut stores=Vec::new();
    for layer in [1,14] {
        let p=backing.join(format!("engram-l{layer}-r{rank}of4.bin"));
        let mut bytes=[0u8;48];File::open(&p)?.read_exact(&mut bytes)?;
        let total=u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        stores.push(RowStore::new(PackedShard::open(&p,layer,rank,4,total)?,4*1024*1024)?);
    }
    let(a,b)=(&stores[0],&stores[1]);let mut records=Vec::new();
    let wide=std::env::var("CANADA_SCRATCH_ENGRAM_PAIR_ROWS").as_deref()==Ok("16");
    let cases=if wide {vec![(0,16,"mixed",false),(1,16,"mixed",true),(2,16,"owned",false),
        (3,16,"nonowned",false),(4,16,"duplicate",false)]}else{vec![(0,3,"mixed",false),(1,3,"mixed",true),
        (2,6,"mixed",false),(3,6,"mixed",true),(4,6,"owned",false),
        (5,6,"nonowned",false),(6,6,"duplicate",false)]};
    for (case,rows,kind,warm) in cases {
        let ids=|store:&RowStore,shift:u64|{
            let(lo,hi,total)=store.interval();
            (0..rows*24).map(|i|{
                let i=i as u64;
                if kind=="nonowned" || (kind=="mixed" && i%4!=0) {if rank==0 {total-1}else{0}}
                else {lo+((if kind=="duplicate" {i%3}else{i})*7919+shift*104729)%(hi-lo)}
            }).collect::<Vec<_>>()
        };
        let ia=ids(a,100+case);let ib=ids(b,200+case);
        let mut oa=vec![0xa5;ia.len()*ROW_BYTES];let mut ob=vec![0xa5;ib.len()*ROW_BYTES];
        // Warm the OS pages once. Cold cases clear only our application caches,
        // so this compares queue/thread overhead, not an NVMe cold-read policy.
        parallel(a,&ia,&mut oa,b,&ib,&mut ob)?;
        let expected=(oa.clone(),ob.clone());let mut times=[Vec::new(),Vec::new()];
        for pair in 0..4 {
            for slot in 0..2 {
                let candidate=(pair+slot)%2==1;
                for _ in 0..24 {
                    if !warm {a.clear_for_probe()?;b.clear_for_probe()?;}
                    oa.fill(0xa5);ob.fill(0xa5);
                    let start=Instant::now();
                    if candidate {RowStore::lookup_pair_into(a,&ia,&mut oa,b,&ib,&mut ob)?;}
                    else {parallel(a,&ia,&mut oa,b,&ib,&mut ob)?;}
                    let us=start.elapsed().as_secs_f64()*1e6;
                    ensure!(oa==expected.0 && ob==expected.1,"paired row bytes differ");
                    if pair>0 {times[usize::from(candidate)].push(us);}
                }
            }
        }
        let control=median(times[0].clone());let paired=median(times[1].clone());
        records.push(json!({"case":case,"rows":rows,"kind":kind,"application_cache_warm":warm,
            "all_bytes_equal":true,"samples_per_mode":times[0].len(),"control_median_us":control,
            "paired_median_us":paired,"speed_ratio":control/paired,"samples_us":times}));
    }
    // Rejected domain and shape admissions must not poison the next queue use.
    let invalid=[a.interval().2];let mut one=[0u8;ROW_BYTES];let mut none=[];
    ensure!(RowStore::lookup_pair_into(a,&invalid,&mut one,b,&[],&mut none).is_err(),"invalid row accepted");
    RowStore::lookup_pair_into(a,&[],&mut none,b,&[],&mut [])?;
    let temporary=std::env::temp_dir().join(format!("canada-engram-pair-io-{}",std::process::id()));
    std::fs::create_dir(&temporary)?;
    let drained=(||->Result<()>{
        let mut first=File::create(temporary.join("first"))?;first.write_all(&vec![0x3c;ROW_BYTES*2])?;
        let mut second=File::create(temporary.join("second"))?;second.write_all(&vec![0xa5;ROW_BYTES])?;
        drop(first);drop(second);
        let first=File::open(temporary.join("first"))?;let second=File::open(temporary.join("second"))?;
        let mut reader=crate::engram_io::Reader::new()?;
        ensure!(reader.read_pair(&first,&[0],&second,&[ROW_BYTES as u64]).is_err(),"short paired read accepted");
        let bytes=reader.read_pair(&first,&[ROW_BYTES as u64],&second,&[0])?;
        ensure!(bytes[..ROW_BYTES].iter().all(|&b|b==0x3c) && bytes[ROW_BYTES..].iter().all(|&b|b==0xa5),"paired queue did not recover after short read");
        Ok(())
    })();
    std::fs::remove_dir_all(&temporary)?;drained?;
    Ok(json!({"rank":rank,"all_bytes_equal":true,"records":records,"invalid_domain_rejected":true,
        "empty_pair_accepted":true,"short_read_drained_and_reused":true,"model_compute":false,"serving_policy_changed":false,
        "wide16":wide,"scope":"Canonical immutable Engram files; bounded byte-I/O cases with balanced order, warmup and three measured pairs. Application-cache cold or hot with OS pages warmed. Includes multiple256-entry batches of unique owned misses, duplicate IDs and all-nonowned rows. No GPU/model speed claim."}))
}
