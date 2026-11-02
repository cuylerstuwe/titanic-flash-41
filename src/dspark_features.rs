//! Independent GPU target-feature capture for the DSpark draft network.
//! Reference equations: qualified prefill_sp, target attention inputs37..39.
use crate::{
    cuda::{Buffer, Cuda, Module, Stream},
    target::Target,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
};

pub(crate) struct Features {
    cuda: Cuda,
    module: Module,
    capacity: usize,
    values: Buffer<bf16>,
    live: usize,
    stream: Option<usize>,
}
impl Features {
    pub(crate) fn new(cuda: &Cuda, core: &[u8], capacity: usize) -> Result<Self> {
        ensure!([6, 16, 128].contains(&capacity)
            || (capacity==32 && crate::config::history32_enabled()), "DSpark feature capacity");
        let module = cuda.module(core)?;
        module.function("ca_dspark_target_features")?;
        Ok(Self {
            cuda: cuda.clone(),
            module,
            capacity,
            values: cuda.allocate(capacity * 15360)?,
            live: 0,
            stream: None,
        })
    }
    pub(crate) fn run(&mut self, target: &Target, stream: &Stream) -> Result<()> {
        self.live = 0;
        let (inputs, live) = target.dspark_feature_inputs(stream)?;
        ensure!(
            inputs
                .iter()
                .all(|b| self.cuda.owns_buffer(b) && b.len() == self.capacity * 20480),
            "DSpark feature input capacity/context"
        );
        unsafe { self.enqueue(inputs.map(|b| b.ptr()), live, stream) }
    }
    /// Safety: each pointer holds capacity*20480 BF16 words and remains alive
    /// until this ordered stream completes; output is separate and owned here.
    unsafe fn enqueue(&mut self, mut inputs: [u64; 3], live: usize, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream)
                && live > 0
                && live <= self.capacity
                && inputs.iter().all(|p| *p != 0)
                && self.stream.is_none_or(|s| s == stream.raw() as usize),
            "DSpark feature live rows/stream"
        );
        self.live = 0;
        self.stream = Some(stream.raw() as usize);
        let mut output = self.values.ptr();
        let mut rows = live as i32;
        let mut args: Vec<*mut c_void> = vec![
            (&mut inputs[0] as *mut u64).cast(),
            (&mut inputs[1] as *mut u64).cast(),
            (&mut inputs[2] as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_dspark_target_features")?.launch(
                [(live * 15360).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        self.live = live;
        Ok(())
    }
    pub(crate) fn values(&self) -> Result<(&Buffer<bf16>, usize)> {
        ensure!(self.live > 0, "DSpark feature output not ready");
        Ok((&self.values, self.live))
    }
}

/// Byte-I/O fixture only: expected numeric values come from a GPU Torch oracle.
pub fn fixture(path: &Path) -> Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct Case {
        live: usize,
        expected: PathBuf,
    }
    #[derive(serde::Deserialize)]
    struct Input {
        core: PathBuf,
        capacity: usize,
        sources: [PathBuf; 3],
        cases: Vec<Case>,
    }
    let p: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        [6, 128].contains(&p.capacity)
            && !p.cases.is_empty()
            && p.cases.len() <= 7
            && p.cases.iter().all(|c| c.live > 0 && c.live <= p.capacity),
        "DSpark feature fixture geometry/cases"
    );
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let mut buffers = Vec::new();
    for source in &p.sources {
        let raw = std::fs::read(source)?;
        ensure!(
            raw.len() == p.capacity * 20480 * 2,
            "DSpark raw source size"
        );
        let mut buffer = cuda.allocate::<u8>(raw.len())?;
        buffer.upload(&raw)?;
        buffers.push(buffer);
    }
    let mut features = Features::new(&cuda, &std::fs::read(p.core)?, p.capacity)?;
    let pointers = [buffers[0].ptr(), buffers[1].ptr(), buffers[2].ptr()];
    let mut records = Vec::new();
    for case in p.cases {
        let expected = std::fs::read(case.expected)?;
        ensure!(
            expected.len() == case.live * 15360 * 2,
            "DSpark expected feature size"
        );
        unsafe { features.enqueue(pointers, case.live, &stream) }?;
        stream.synchronize()?;
        let raw = unsafe { cuda.read_device(features.values.ptr(), expected.len()) }?;
        let graph = stream.capture(|| unsafe { features.enqueue(pointers, case.live, &stream) })?;
        for _ in 0..20 {
            unsafe { graph.launch(&stream) }?;
        }
        stream.synchronize()?;
        let replay = unsafe { cuda.read_device(features.values.ptr(), expected.len()) }?;
        records.push(serde_json::json!({"capacity":p.capacity,"live":case.live,
            "bitwise_equal":raw==expected,"replay_equal":replay==expected,"replays":20}));
    }
    Ok(
        serde_json::json!({"all_exact":records.iter().all(|r|r["bitwise_equal"]==true&&r["replay_equal"]==true),
        "records":records,"full_engine_acceptance":false,
        "scope":"GPU HC mean+concatenation only, on real qualified target residuals. No CPU numeric oracle, draft network, Markov proposal, full engine or performance acceptance."}),
    )
}
