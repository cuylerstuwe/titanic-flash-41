//! Standalone valid-score parity/performance for the native FP4 MQA plan.
use crate::{
    cuda::{Buffer, Cuda, Stream},
    index::QueryRows,
    layer_probe::{Tap, compare, fixture},
    mqa::{Chunked, Dense, KeyMatrix, Native, Ranges},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
};
#[derive(Deserialize)]
struct Config {
    fixture: PathBuf,
    primitive: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    key_rows: usize,
    columns: usize,
    starts: Vec<i32>,
    lengths: Vec<i32>,
    sources: Vec<Tap>,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    sources: Vec<Tap>,
    expected: Tap,
}
fn i32s(b: &[u8]) -> Result<Vec<i32>> {
    ensure!(b.len() % 4 == 0, "int32 input alignment");
    Ok(b.chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn floats(target: &mut Buffer<f32>, b: &[u8]) -> Result<()> {
    ensure!(b.len() == target.bytes(), "FP32 input bytes");
    target.upload(
        &b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
enum Scorer {
    Whole(Dense),
    Chunks(Chunked),
}
impl Scorer {
    fn inputs_mut(&mut self) -> Result<(&mut QueryRows, &mut KeyMatrix)> {
        match self {
            Self::Whole(x) => x.inputs_mut(),
            Self::Chunks(x) => x.inputs_mut(),
        }
    }
    fn run(&mut self, ranges: &Ranges, stream: &Stream) -> Result<()> {
        match self {
            Self::Whole(x) => x.run(ranges, stream),
            Self::Chunks(x) => {
                for i in 0..x.len() {
                    x.run_chunk(i, ranges, stream)?;
                }
                Ok(())
            }
        }
    }
    fn share_score_scratch(&mut self, storage: Rc<Buffer<f32>>) -> Result<usize> {
        match self {
            Self::Whole(x)=>x.share_score_scratch(storage),
            Self::Chunks(x)=>x.share_score_scratch(storage),
        }
    }
    fn score_bytes(&self) -> usize {
        match self {
            Self::Whole(x) => x.output().bytes(),
            Self::Chunks(x) => x.output().bytes(),
        }
    }
    fn chunks(&self) -> usize {
        match self {
            Self::Whole(_) => 1,
            Self::Chunks(x) => x.len(),
        }
    }
    fn valid(&mut self, ranges: &Ranges, stream: &Stream, lengths: &[i32]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        match self {
            Self::Whole(x) => {
                let all = x.output().download()?;
                for (row, &n) in lengths.iter().enumerate() {
                    for v in &all[row * x.stride()..row * x.stride() + n as usize] {
                        result.extend(v.to_le_bytes());
                    }
                }
            }
            Self::Chunks(x) => {
                for i in 0..x.len() {
                    let rows = x.rows(i)?;
                    x.run_chunk(i, ranges, stream)?;
                    stream.synchronize()?;
                    let all = x.output().download()?;
                    for (local, row) in rows.enumerate() {
                        for v in
                            &all[local * x.stride()..local * x.stride() + lengths[row] as usize]
                        {
                            result.extend(v.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}
pub fn run(path: &Path) -> Result<Value> {
    run_mode(path, false)
}
pub fn run_chunked(path: &Path) -> Result<Value> {
    run_mode(path, true)
}
fn run_mode(path: &Path, chunked: bool) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(cases.cases.len() == 8, "MQA qualification coverage");
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let native = Native::open(&cuda, &p.primitive)?;
    let stream = cuda.stream()?;
    let pool_screen=std::env::var("CANADA_SCRATCH_INDEX_SCORE_POOL_SCREEN").as_deref()==Ok("1");
    let mut backing=pool_screen.then(||cuda.allocate::<f32>(64*1024*1024/4+128)).transpose()?;
    let shared=backing.as_ref().map(|b|unsafe{b.view(64,b.len()-128)}.map(Rc::new)).transpose()?;
    let poison=f32::from_bits(0x7fc055aa);
    let mut reports = Vec::new();
    for c in &cases.cases {
        let root = p.fixture.join(&c.name);
        let queries = Rc::new(QueryRows::new(&cuda, c.rows)?);
        let keys = Rc::new(KeyMatrix::new(&cuda, c.key_rows)?);
        let chunk_rows = if c.rows >= 161 { 128 } else { 4 };
        let budget = if chunked {
            chunk_rows * c.columns.div_ceil(128) * 128 * 4
        } else {
            64 * 1024 * 1024
        };
        let mut dense = if chunked {
            Scorer::Chunks(Chunked::new(
                &cuda,
                native.clone(),
                queries,
                keys,
                c.columns,
                budget,
            )?)
        } else {
            Scorer::Whole(Dense::new(
                &cuda,
                native.clone(),
                queries,
                keys,
                c.columns,
                budget,
            )?)
        };
        ensure!(
            dense.score_bytes() <= budget,
            "score arena exceeded its byte budget"
        );
        if let Some(pool)=&shared {dense.share_score_scratch(pool.clone())?;}
        let ranges = Ranges::new(&cuda, c.key_rows, c.columns, &c.starts, &c.lengths)?;
        {
            let (_, keys) = dense.inputs_mut()?;
            for s in &c.sources {
                let b = fixture(&root, s)?;
                match s.name.as_str() {
                    "keys" => keys.payload.upload(&b)?,
                    "k_scales" => {
                        let mut scales = i32s(&b)?;
                        ensure!(scales.len() == c.key_rows, "key scale source rows");
                        scales.resize(keys.scales.len(), 0);
                        keys.scales.upload(&scales)?;
                    }
                    _ => anyhow::bail!("unknown MQA keys"),
                }
            }
            keys.ready.set(true);
        }
        let mut graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            {
                let (q, _) = dense.inputs_mut()?;
                for s in &phase.sources {
                    let b = fixture(&directory, s)?;
                    match s.name.as_str() {
                        "q" => q.payload.upload(&b)?,
                        "q_scales" => q.scales.upload(&i32s(&b)?)?,
                        "weights" => floats(&mut q.weights, &b)?,
                        _ => anyhow::bail!("unknown MQA query input"),
                    }
                }
                q.ready.set(true);
            }
            // Deliberately invalidate every score byte between changed input
            // phases; the same backing is then reused by the next row profile.
            if let Some(backing)=&mut backing {backing.upload(&vec![poison;backing.len()])?;stream.join_default()?;}
            if graph.is_none() {
                dense.run(&ranges, &stream)?;
                stream.synchronize()?;
                graph = Some(stream.capture(|| dense.run(&ranges, &stream))?);
            }
            for _ in 0..if pool_screen {2}else{20} {
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            }
            let a = cuda.event()?;
            let b = cuda.event()?;
            let mut samples = Vec::new();
            for _ in 0..if pool_screen {8}else{100} {
                a.record(&stream)?;
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
                b.record(&stream)?;
                samples.push(b.elapsed_since(&a)?);
            }
            samples.sort_by(f32::total_cmp);
            stream.synchronize()?;
            let valid = dense.valid(&ranges, &stream, &c.lengths)?;
            let taps = compare(
                &directory,
                std::slice::from_ref(&phase.expected),
                &BTreeMap::from([("logits".into(), valid)]),
            )?;
            if let Some(backing)=&backing {
                let raw=backing.download()?;
                ensure!(raw[..64].iter().chain(&raw[raw.len()-64..]).all(|x|x.to_bits()==poison.to_bits()),
                    "shared MQA score arena guard overwritten");
            }
            phases.push(json!({"name":phase.name,"taps":taps,"gpu_ms":{"p50":samples[(samples.len()-1)/2],
                "p95":samples[(samples.len()*95/100).min(samples.len()-1)],"max":samples[samples.len()-1]}}));
        }
        let exact = phases.iter().all(|p| {
            p["taps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["bitwise_equal"] == true)
        });
        eprintln!("{}", json!({"case":c.name,"all_bitwise_equal":exact}));
        reports.push(json!({"case":c.name,"all_bitwise_equal":exact,"rows":c.rows,"key_rows":c.key_rows,"columns":c.columns,"logits_buffer_bytes":dense.score_bytes(),"admitted_score_bytes":budget,"row_chunks":dense.chunks(),"phases":phases}));
    }
    Ok(
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"chunked":chunked,"shared_score_pool":pool_screen,"poisoned_between_phases_and_guards":pool_screen,"scope":"Every valid FP4 weighted-ReLU MQA score; one bounded reusable score arena if chunked, not full prefill/topK/model parity. Diagnostic CPU readbacks excluded from timing."}),
    )
}

/// Existing native kernels with independent versus shared FP32 score storage.
/// Changed GPU fixtures, bounded ranges, multiple widths/chunks and graph replay
/// test the storage contract without loading any language-model weights.
pub fn run_score_pool(primitive:&Path,core:&Path)->Result<Value> {
    use sha2::{Digest,Sha256};
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let module=cuda.module(&std::fs::read(core)?)?;
    let native=Native::open(&cuda,primitive)?;
    let elements=64*1024*1024/4;let poison=f32::from_bits(0x7fc055aa);
    let mut backing=cuda.allocate::<f32>(elements+128)?;
    let shared=Rc::new(unsafe{backing.view(64,elements)}?);
    let mut records=Vec::new();
    for (rows,columns) in [(6usize,131072usize),(16,262144),(32,1048576),(128,524288)] {
        let q=Rc::new(QueryRows::new(&cuda,rows)?);let k=Rc::new(KeyMatrix::new(&cuda,columns)?);
        // Initialize every input before eager warmup/capture. Measured phases
        // below replace these with changed GPU-generated encoded fixtures.
        for (ptr,bytes) in [(q.payload.ptr(),q.payload.bytes()),(q.scales.ptr(),q.scales.bytes()),
            (q.weights.ptr(),q.weights.bytes()),(k.payload.ptr(),k.payload.bytes()),(k.scales.ptr(),k.scales.bytes())] {
            unsafe{cuda.zero_device(ptr,bytes,&stream)}?;
        }
        q.ready.set(true);k.ready.set(true);
        let mut scorers=Vec::new();
        for variant in 0..2 {
            let mut s=if rows<=16 {Scorer::Whole(Dense::new(&cuda,native.clone(),q.clone(),k.clone(),columns,elements*4)?)}
                else {Scorer::Chunks(Chunked::new(&cuda,native.clone(),q.clone(),k.clone(),columns,elements*4)?)};
            if variant==1 {s.share_score_scratch(shared.clone())?;}
            scorers.push(s);
        }
        let lengths=(0..rows).map(|r|match r%11{0=>0,1=>1,2=>511,_=>columns-r%137}as i32).collect::<Vec<_>>();
        let ranges=Ranges::new(&cuda,columns,columns,&vec![0;rows],&lengths)?;
        let mut graphs=Vec::new();
        for scorer in &mut scorers {
            let mut chunks=Vec::new();
            for chunk in 0..scorer.chunks() {
                let mut enqueue=||match scorer {Scorer::Whole(s)=>s.run(&ranges,&stream),Scorer::Chunks(s)=>s.run_chunk(chunk,&ranges,&stream)};
                enqueue()?;stream.synchronize()?;
                chunks.push(stream.capture(enqueue)?);
            }
            graphs.push(chunks);
        }
        for phase in 0..3 {
            let (mut qp,mut qs,mut qw,mut kp,mut ks,mut m,mut n,mut ph)=
                (q.payload.ptr(),q.scales.ptr(),q.weights.ptr(),k.payload.ptr(),k.scales.ptr(),rows as i32,columns as i32,phase);
            let mut args=[(&mut qp as *mut u64).cast(),(&mut qs as *mut u64).cast(),(&mut qw as *mut u64).cast(),
                (&mut kp as *mut u64).cast(),(&mut ks as *mut u64).cast(),(&mut m as *mut i32).cast(),
                (&mut n as *mut i32).cast(),(&mut ph as *mut i32).cast()];
            unsafe{module.function("ca_mqa_score_pool_fixture")?.launch([256,1,1],[256,1,1],0,&stream,&mut args)}?;
            stream.synchronize()?;
            for chunk in 0..graphs[0].len() {
                backing.upload(&vec![poison;backing.len()])?;stream.join_default()?;
                let mut hashes=Vec::new();let mut nonzero=0usize;
                for variant in 0..2 {
                    unsafe{graphs[variant][chunk].launch(&stream)}?;stream.synchronize()?;
                    let (output,stride,row_range)=match &scorers[variant] {
                        Scorer::Whole(s)=>(s.output(),s.stride(),0..rows),
                        Scorer::Chunks(s)=>(s.output(),s.stride(),s.rows(chunk)?),
                    };
                    let values=output.download()?;let mut hash=Sha256::new();
                    for (local,row) in row_range.enumerate() {
                        for &value in &values[local*stride..local*stride+lengths[row]as usize] {
                            ensure!(value.is_finite(),"score pool left a nonfinite live score");
                            if phase==2 {ensure!(value==0.,"zero-query fixture was not zero");}
                            nonzero+=usize::from(value!=0.);hash.update(value.to_le_bytes());
                        }
                    }
                    hashes.push(format!("{:x}",hash.finalize()));
                }
                ensure!(hashes[0]==hashes[1] && (phase==2||nonzero>0),"shared scores differ from native owned storage");
                let guards=backing.download()?;
                ensure!(guards[..64].iter().chain(&guards[elements+64..]).all(|x|x.to_bits()==poison.to_bits()),"score pool guard changed");
                let mut pairs=Vec::new();
                for repeat in 0..4 {
                    let mut times=[0f32;2];
                    for variant in [repeat%2,1-repeat%2] {
                        let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                        for _ in 0..8 {unsafe{graphs[variant][chunk].launch(&stream)}?;}
                        end.record(&stream)?;times[variant]=end.elapsed_since(&start)?/8.;
                    }
                    pairs.push(json!({"milliseconds":times,"first_variant":repeat%2}));
                }
                records.push(json!({"rows":rows,"columns":columns,"phase":phase,"chunk":chunk,
                    "exact":true,"guards":true,"nonzero_scores":nonzero,"sha256":hashes[0],"pairs":pairs}));
            }
        }
    }
    Ok(json!({"all_exact":true,"records":records,"shared_bytes":elements*4,
        "scope":"Existing native MQA kernels at6/16/32/128rows, changed and zero GPU-generated FP4 fixtures, bounded/empty ranges, graph replay and outer guards. Shared FP32 score backing only; no full-model/state/caching or serving-speed claim."}))
}
