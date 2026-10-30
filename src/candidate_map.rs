//! Admitted GPU logical-to-physical candidate mapping. Invalid producer IDs
//! cannot become out-of-bounds attention reads; an error flag survives until
//! the controller checks it before publishing/committing a request.
use crate::{
    attention::Candidates,
    cuda::{Buffer, Cuda, MappedHost, Module, Stream},
};
use anyhow::{Result, ensure};
use std::{cell::Cell, ffi::c_void};
pub struct Map {
    cuda: Cuda,
    module: Module,
    slots: Buffer<i32>,
    keys: usize,
    capacity: usize,
    output: Candidates,
    error: ErrorFlag,
    count: usize,
    width: usize,
    stream: Cell<usize>,
}
// The GPU is the only writer after allocation. Both representations remain
// alive for the entire mapper/graph lifetime. The host reads only after the
// owning stream completes; mapped memory does not replace that dependency.
enum ErrorFlag { Device(Buffer<u32>), Mapped(MappedHost) }
impl ErrorFlag {
    fn pointer(&self) -> u64 {match self {
        Self::Device(v)=>v.ptr(),Self::Mapped(v)=>v.device_ptr(),
    }}
    fn read(&self) -> Result<u32> {match self {
        Self::Device(v)=>Ok(v.download()?[0]),Self::Mapped(v)=>v.status_word(0),
    }}
}
impl Map {
    pub fn new(
        cuda: &Cuda,
        core: &[u8],
        rows: usize,
        width: usize,
        capacity: usize,
        slots: &[i32],
        lengths: &[i32],
    ) -> Result<Self> {
        let mapped=match std::env::var("CANADA_SCRATCH_MAPPED_CANDIDATE_ERRORS") {
            Ok(v) if v=="0"=>false,Ok(v) if v=="1"=>true,
            // s823: valid/corrupted integer addresses, sticky faults, graph
            // replay and ordered-stream checks match the copied control.
            // Eight checks avoid ~41us of tiny DtoH transfers on coherent GB10.
            Err(std::env::VarError::NotPresent)=>true,
            _=>anyhow::bail!("invalid candidate error readout policy"),
        };
        Self::new_with_error_mode(cuda,core,rows,width,capacity,slots,lengths,mapped)
    }
    fn new_with_error_mode(
        cuda:&Cuda,core:&[u8],rows:usize,width:usize,capacity:usize,
        slots:&[i32],lengths:&[i32],mapped:bool,
    ) -> Result<Self> {
        ensure!(
            (1..=1048576).contains(&slots.len())
                && slots.iter().all(|&s| s >= 0 && (s as usize) < capacity),
            "physical candidate slot map"
        );
        ensure!(
            (1..=1024).contains(&width) && lengths.len() == rows && lengths.iter().all(|&n| n >= 0),
            "candidate sequence lengths"
        );
        let clipped = lengths
            .iter()
            .map(|&n| n.min(width as i32))
            .collect::<Vec<_>>();
        let output = Candidates::device_output(cuda, rows, width, capacity, &clipped)?;
        let mut table = cuda.allocate(slots.len())?;
        table.upload(slots)?;
        let error=if mapped {ErrorFlag::Mapped(cuda.mapped_host(4)?)} else {
            let mut flag=cuda.allocate(1)?;flag.zero()?;ErrorFlag::Device(flag)
        };
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            slots: table,
            keys: slots.len(),
            capacity,
            output,
            error,
            count: rows * width,
            width,
            stream: Cell::new(0),
        })
    }
    pub fn run(
        &self,
        selection: &crate::topk::Selection,
        windows: &crate::topk::Windows,
        stream: &Stream,
    ) -> Result<()> {
        self.run_admitted(
            selection.admitted_indices(windows, stream)?,
            windows.lengths(),
            stream,
        )
    }
    pub(crate) fn run_admitted(
        &self,
        selected: &Buffer<i32>,
        lengths: &Buffer<i32>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            selected.len() == self.count
                && lengths.len() == self.count / self.width
                && self.cuda.owns_buffer(lengths)
                && self.cuda.owns_buffer(selected)
                && self.cuda.owns_stream(stream),
            "candidate producer/context"
        );
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "candidate mapper requires one ordered stream"
        );
        self.stream.set(p);
        self.output.bind_stream(stream)?;
        let mut input = selected.ptr();
        let mut slots = self.slots.ptr();
        let mut output = self.output.ids().ptr();
        let mut lengths_ptr = lengths.ptr();
        let mut output_lengths = self.output.lengths().ptr();
        let mut error = self.error.pointer();
        let mut count = self.count as i32;
        let mut keys = self.keys as i32;
        let mut capacity = self.capacity as i32;
        let mut width = self.width as i32;
        let mut args: Vec<*mut c_void> = vec![
            (&mut input as *mut u64).cast(),
            (&mut slots as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut lengths_ptr as *mut u64).cast(),
            (&mut output_lengths as *mut u64).cast(),
            (&mut error as *mut u64).cast(),
            (&mut count as *mut i32).cast(),
            (&mut keys as *mut i32).cast(),
            (&mut capacity as *mut i32).cast(),
            (&mut width as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_map_index_candidates")?.launch(
                [self.count.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        self.output.produced();
        Ok(())
    }
    pub fn output(&self) -> &Candidates {
        &self.output
    }
    /// Synchronize the owning stream first. This flag is controller validation,
    /// not a per-layer CPU/GPU synchronization in the model graph.
    pub fn error(&self) -> Result<u32> {
        self.error.read()
    }
}

/// GPU fault-injection diagnostic. Expectations are integer address metadata;
/// this is not a CPU model or numeric inference path.
pub fn probe(core: &std::path::Path) -> Result<serde_json::Value> {
    let checks=[probe_mode(core,false)?,probe_mode(core,true)?];
    let timing=probe_timing(core)?;
    Ok(serde_json::json!({"all_checks_passed":true,"modes":["copied","mapped"],
        "checks":checks,"timing":timing,
        "scope":"Native integer candidate address/error transport only. Both modes preserve fault bits, safe addresses, graph replay and stream admission. Eight-map readout and graph/host timing; no model arithmetic or whole-model speed claim."}))
}
fn probe_mode(core:&std::path::Path,mapped:bool)->Result<serde_json::Value> {
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let other = cuda.stream()?;
    let bytes = std::fs::read(core)?;
    let mut map = Map::new_with_error_mode(&cuda, &bytes, 3, 8, 16, &[0, 5, 9], &[3, 2, 0],mapped)?;
    let mut ids = cuda.allocate::<i32>(24)?;
    let mut lengths = cuda.allocate::<i32>(3)?;
    let valid = [
        0, 1, 2, -1, -1, -1, -1, -1, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ];
    let expected = [
        0, 5, 9, -1, -1, -1, -1, -1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ];
    ids.upload(&valid)?;
    lengths.upload(&[3, 2, 0])?;
    stream.join_default()?;
    map.run_admitted(&ids, &lengths, &stream)?;
    stream.synchronize()?;
    ensure!(
        map.output().ids().download()? == expected
            && map.output().lengths().download()? == [3, 2, 0]
            && map.error()? == 0,
        "valid mapping/sentinel mismatch"
    );
    let invalid = [
        -2,
        3,
        i32::MAX,
        i32::MIN,
        -1,
        0,
        1,
        2,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        1,
        2,
        -2,
        3,
        i32::MAX,
        i32::MIN,
        -1,
    ];
    ids.upload(&invalid)?;
    lengths.upload(&[-1, 4, i32::MAX])?;
    stream.join_default()?;
    map.run_admitted(&ids, &lengths, &stream)?;
    stream.synchronize()?;
    let sanitized = [
        -1, -1, -1, -1, -1, 0, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, 0, 5, 9, -1, -1, -1, -1, -1,
    ];
    ensure!(
        map.output().ids().download()? == sanitized
            && map.output().lengths().download()? == [0, 3, 3]
            && map.error()? == 5,
        "invalid IDs/lengths were not safely bounded and flagged"
    );
    ids.upload(&valid)?;
    lengths.upload(&[3, 2, 0])?;
    stream.join_default()?;
    let graph = stream.capture(|| map.run_admitted(&ids, &lengths, &stream))?;
    for _ in 0..100 {
        unsafe { graph.launch(&stream) }?;
    }
    stream.synchronize()?;
    ensure!(
        map.error()? == 5 && map.output().ids().download()? == expected,
        "fault flag did not survive later success"
    );
    // Inject a corrupted device slot after normal host admission to exercise
    // the defense immediately before attention can consume an address.
    map.slots.upload(&[0, 16, 9])?;
    stream.join_default()?;
    map.run_admitted(&ids, &lengths, &stream)?;
    stream.synchronize()?;
    ensure!(
        map.error()? == 7 && map.output().ids().download()?[1] == -1,
        "corrupt physical slot not sanitized"
    );
    ensure!(
        map.run_admitted(&ids, &lengths, &other).is_err(),
        "cross-stream producer accepted"
    );
    ensure!(
        Map::new(&cuda, &bytes, 3, 8, 16, &[-1, 5, 9], &[3, 2, 0]).is_err()
            && Map::new(&cuda, &bytes, 3, 8, 16, &[0, 16, 9], &[3, 2, 0]).is_err(),
        "bad slot admission accepted"
    );
    Ok(
        serde_json::json!({"all_checks_passed":true,"mapped":mapped,"graph_replays":100,
        "checked":["valid addresses and -1 sentinel","negative and oversized IDs","negative and oversized lengths",
        "persistent fault flag","corrupted physical slot","cross-stream rejection","host slot bounds"],
        "scope":"GPU candidate mapping safety only; not full-engine reliability"}),
    )
}

fn probe_timing(core:&std::path::Path)->Result<serde_json::Value> {
    use std::time::Instant;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let bytes=std::fs::read(core)?;let mut reports=Vec::new();
    for rows in [6usize,16] {
        let width=512;let slots:Vec<i32>=(0..4096).map(|i|i*2).collect();
        let mut ids=cuda.allocate::<i32>(rows*width)?;
        let selected:Vec<i32>=(0..rows*width).map(|i|if i%17==0{-1}else{((i*7)%4096)as i32}).collect();
        let expected:Vec<i32>=selected.iter().map(|&i|if i<0{-1}else{i*2}).collect();
        ids.upload(&selected)?;let mut lengths=cuda.allocate::<i32>(rows)?;
        lengths.upload(&vec![4096;rows])?;
        let modes:Vec<Vec<Map>>=[false,true].into_iter().map(|mapped|(0..8).map(|_|
            Map::new_with_error_mode(&cuda,&bytes,rows,width,8192,&slots,&vec![4096;rows],mapped)
        ).collect::<Result<Vec<_>>>()).collect::<Result<_>>()?;
        stream.join_default()?;
        let mut graphs=Vec::new();
        for maps in &modes {
            let graph=stream.capture(||{for map in maps{map.run_admitted(&ids,&lengths,&stream)?;}Ok(())})?;
            unsafe{graph.launch(&stream)?;}stream.synchronize()?;
            for map in maps {ensure!(map.error()?==0&&map.output().ids().download()?==expected
                &&map.output().lengths().download()?==vec![512;rows],"timing candidate output mismatch");}
            graphs.push(graph);
        }
        let mut readout=[Vec::new(),Vec::new()];let mut transaction=[Vec::new(),Vec::new()];
        for repeat in 0..6 {
            for mode in if repeat%2==0{[0,1]}else{[1,0]} {
                let maps=&modes[mode];
                let check=||->Result<()>{for map in maps{ensure!(map.error()?==0,"timing candidate fault");}Ok(())};
                for _ in 0..8{check()?;}
                let start=Instant::now();for _ in 0..256{check()?;}
                readout[mode].push(start.elapsed().as_secs_f64()*1e6/256.);
                let start=Instant::now();for _ in 0..128 {
                    unsafe{graphs[mode].launch(&stream)?;}stream.synchronize()?;check()?;
                }
                transaction[mode].push(start.elapsed().as_secs_f64()*1e6/128.);
            }
        }
        stream.synchronize()?;
        reports.push(serde_json::json!({"rows":rows,"maps":8,"width":width,
            "readout_us":readout,"graph_and_readout_us":transaction}));
    }
    Ok(serde_json::json!(reports))
}
