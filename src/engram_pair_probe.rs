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
    let wide=match std::env::var("CANADA_SCRATCH_ENGRAM_PAIR_ROWS").as_deref(){
        Ok("16")=>Some(16),Ok("32")=>Some(32),Err(_)=>None,
        _=>anyhow::bail!("paired I/O screen supports16or32 rows")};
    let cases=if let Some(n)=wide {vec![(0,n,"mixed",false),(1,n,"mixed",true),(2,n,"owned",false),
        (3,n,"nonowned",false),(4,n,"duplicate",false)]}else{vec![(0,3,"mixed",false),(1,3,"mixed",true),
        (2,6,"mixed",false),(3,6,"mixed",true),(4,6,"owned",false),
        (5,6,"nonowned",false),(6,6,"duplicate",false)]};
    let adaptive=std::env::var("CANADA_SCRATCH_ENGRAM_PAIR_ADAPTIVE_SCREEN").as_deref()==Ok("1");
    let main_worker=std::env::var("CANADA_SCRATCH_ENGRAM_PAIR_MAIN_WORKER_SCREEN").as_deref()==Ok("1");
    let pool=if std::env::var("CANADA_SCRATCH_ENGRAM_PAIR_POOL_SCREEN").as_deref()==Ok("1") {
        ensure!(adaptive,"pool screen includes the adaptive control");
        Some(rayon::ThreadPoolBuilder::new().num_threads(if main_worker{1}else{2}).stack_size(512*1024)
            .thread_name(|n|format!("engram-pair-probe-{n}")).build()?)
    }else{None};
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
        let expected=(oa.clone(),ob.clone());let modes=if pool.is_some(){4}else if adaptive{3}else{2};
        let mut times=vec![Vec::new();modes];
        for pair in 0..if pool.is_some(){9}else if adaptive{7}else{4} {
            let mut order:Vec<usize>=(0..modes).collect();order.rotate_left((pair/2)%modes);
            if pair%2==1{order.reverse();}
            for candidate in order {
                for _ in 0..24 {
                    if !warm {a.clear_for_probe()?;b.clear_for_probe()?;}
                    oa.fill(0xa5);ob.fill(0xa5);
                    let start=Instant::now();
                    match candidate {
                        0=>parallel(a,&ia,&mut oa,b,&ib,&mut ob)?,
                        1=>RowStore::lookup_pair_into(a,&ia,&mut oa,b,&ib,&mut ob)?,
                        2=>RowStore::lookup_pair_warm_or_parallel(a,&ia,&mut oa,b,&ib,&mut ob)?,
                        3=>if main_worker{RowStore::lookup_pair_with_worker(pool.as_ref().unwrap(),a,&ia,&mut oa,b,&ib,&mut ob)?;}
                            else{RowStore::lookup_pair_with_pool(pool.as_ref().unwrap(),a,&ia,&mut oa,b,&ib,&mut ob)?;},
                        _=>unreachable!(),
                    }
                    let us=start.elapsed().as_secs_f64()*1e6;
                    ensure!(oa==expected.0 && ob==expected.1,"paired row bytes differ");
                    if pair>0 {times[candidate].push(us);}
                }
            }
        }
        let control=median(times[0].clone());let paired=median(times[1].clone());
        records.push(json!({"case":case,"rows":rows,"kind":kind,"application_cache_warm":warm,
            "all_bytes_equal":true,"samples_per_mode":times[0].len(),"control_median_us":control,
            "paired_median_us":paired,"speed_ratio":control/paired,
            "pool_median_us":pool.as_ref().map(|_|median(times[3].clone())),
            "adaptive_median_us":if adaptive{Some(median(times[2].clone()))}else{None},"samples_us":times}));
    }
    // Rejected domain and shape admissions must not poison the next queue use.
    let invalid=[a.interval().2];let mut one=[0u8;ROW_BYTES];let mut none=[];
    ensure!(RowStore::lookup_pair_into(a,&invalid,&mut one,b,&[],&mut none).is_err(),"invalid row accepted");
    RowStore::lookup_pair_into(a,&[],&mut none,b,&[],&mut [])?;
    if adaptive {
        ensure!(RowStore::lookup_pair_warm_or_parallel(a,&invalid,&mut one,b,&[],&mut none).is_err(),"adaptive invalid row accepted");
        ensure!(RowStore::lookup_pair_warm_or_parallel(a,&[],&mut one,b,&[],&mut none).is_err(),"adaptive invalid output extent accepted");
        RowStore::lookup_pair_warm_or_parallel(a,&[],&mut none,b,&[],&mut [])?;
    }
    if let Some(pool)=&pool {
        if main_worker {
            ensure!(RowStore::lookup_pair_with_worker(pool,a,&invalid,&mut one,b,&[],&mut none).is_err(),"worker invalid row accepted");
            RowStore::lookup_pair_with_worker(pool,a,&[],&mut none,b,&[],&mut [])?;
        }else{
            ensure!(RowStore::lookup_pair_with_pool(pool,a,&invalid,&mut one,b,&[],&mut none).is_err(),"pool invalid row accepted");
            RowStore::lookup_pair_with_pool(pool,a,&[],&mut none,b,&[],&mut [])?;
        }
    }
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
        "wide16":wide==Some(16),"wide_rows":wide,"adaptive_screen":adaptive,"pool_screen":pool.is_some(),"main_worker":main_worker,"scope":"Canonical immutable Engram files; bounded byte-I/O cases with balanced order and warmup. Application-cache cold or hot with OS pages warmed. Includes multiple256-entry batches of unique owned misses, duplicate IDs and all-nonowned rows. No GPU/model speed claim."}))
}
