//! Opt-in six-row candidate-only index scoring. DeepInfer's sparse-MQA path
//! (Apache-2.0, 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9, index_attn.py)
//! motivates skipping excluded keys. This independent implementation keeps the
//! qualified SM121 MQA arithmetic and scatters FP32 scores to their original
//! columns before the existing exact top-k. No score/weight precision change.
use crate::{
    candidates::{BLOCK, WINDOW},
    cuda::{Buffer, Cuda, Module, Stream},
    index::{KeyCache, KeySlots, QueryRows},
    mqa::{Dense, KeyMatrix, Native, Ranges},
    topk::{self,Selection,Windows},
};
use anyhow::{Result, ensure};
use std::{ffi::c_void, rc::Rc};

macro_rules! args {
    ($($v:ident),* $(,)?) => { [$(std::ptr::addr_of_mut!($v).cast::<c_void>()),*] };
}

pub(crate) fn key_capacity(columns: usize) -> usize {
    columns.min(6 * WINDOW)
}

pub(crate) struct Sparse6 {
    cuda: Cuda,
    module: Module,
    packed_gather: Option<Module>,
    keys: Rc<KeyMatrix>,
    dense: Dense,
    ranges: Ranges,
    counts: Buffer<i32>,
    ids: Buffer<i32>,
    error: Buffer<u32>,
    output: Buffer<f32>,
    columns: usize,
    blocks: usize,
    stride: usize,
    compact_selection: Option<CompactSelection>,
    fused_remap_sort: bool,
}
struct CompactSelection { windows:Windows, selection:Selection }

impl Sparse6 {
    pub(crate) fn selector_taps(&self)->Result<(usize,Vec<(&'static str,u64,usize)>)> {
        let c=self.compact_selection.as_ref().ok_or_else(||anyhow::anyhow!("compact selector capture"))?;
        Ok((self.dense.stride(),vec![
            ("scores",self.dense.output().ptr(),self.dense.output().bytes()),
            ("compact_block_ids",self.ids.ptr(),self.ids.bytes()),
            ("compact_key_ends",self.ranges.key_ends().ptr(),self.ranges.key_ends().bytes()),
            ("compact_lengths",c.windows.lengths().ptr(),c.windows.lengths().bytes()),
            ("compact_selected",c.selection.indices().ptr(),c.selection.indices().bytes()),
        ]))
    }
    pub(crate) fn gathered_keys(&self)->&KeyMatrix {&self.keys}
    pub(crate) fn new(
        cuda: &Cuda,
        core: &[u8],
        native: Rc<Native>,
        queries: Rc<QueryRows>,
        keys: Rc<KeyMatrix>,
        columns: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            queries.rows() == 6
                && queries.in_context(cuda)
                && columns > WINDOW
                && columns <= 1048576
                && columns % 128 == 0
                && keys.rows() == key_capacity(columns),
            "candidate-only index requires six zero-based rows and aligned bounded context"
        );
        let capacity = keys.rows();
        let stride = columns;
        let compact_bytes = 8 * capacity * 4;
        ensure!(
            6 * stride * 4 + compact_bytes <= budget,
            "sparse index score budget"
        );
        let blocks = columns / BLOCK;
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            packed_gather: None,
            dense: Dense::new(cuda, native, queries, keys.clone(), capacity, compact_bytes)?,
            ranges: Ranges::new(cuda, capacity, capacity, &[0; 6], &[0; 6])?,
            counts: cuda.allocate(blocks.div_ceil(256) + 1)?,
            ids: cuda.allocate(capacity / BLOCK)?,
            error: cuda.allocate(1)?,
            output: cuda.allocate(6 * stride)?,
            keys,
            columns,
            blocks,
            stride,
            compact_selection:None,
            fused_remap_sort:match std::env::var("CANADA_SCRATCH_COMPACT_REMAP_SORT") {
                Ok(value)=>{ensure!((value=="0"||value=="1") && (value=="0" ||
                    std::env::var("CANADA_SCRATCH_SORT_SELECTED").as_deref()==Ok("1")),
                    "compact remap/sort requires the sorted-selection contract");value=="1"},
                Err(std::env::VarError::NotPresent)=>false,
                Err(error)=>return Err(error.into()),
            },
        })
    }

    pub(crate) fn configure_packed_gather(&mut self, kernels: &std::path::Path) -> Result<()> {
        use sha2::{Digest,Sha256};
        let requested=match std::env::var("CANADA_SCRATCH_PACKED_SPARSE_GATHER") {
            Ok(value)=>{ensure!(value=="0"||value=="4","unsupported packed sparse gather");value=="4"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        if !requested {return Ok(());}
        let root=kernels.join("sparse-gather-packed-v1");
        let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let raw=std::fs::read(root.join("gather.cubin"))?;
        ensure!(manifest["architecture"]==121 && manifest["threads"]==128
            && manifest["grid"]==192 && manifest["query_rows"]==6 && manifest["lanes_per_key"]==4
            && manifest["source_sha256"]=="3e15bcdbb8428d2d4f052b278e91e6bd207aa70d7d72a5a54eec27070d6e8cd6"
            && manifest["artifact_sha256"]=="dc78579e7b22ccc9dd0f7538e50908002df1ee5cace166b8ef32913fd04b7c41"
            && format!("{:x}",Sha256::digest(&raw))==manifest["artifact_sha256"].as_str().unwrap_or(""),
            "packed sparse gather artifact/geometry contract");
        self.packed_gather=Some(self.cuda.module(&raw)?);
        Ok(())
    }
    pub(crate) fn packed_gather_enabled(&self)->bool {self.packed_gather.is_some()}

    // Isolated qualification path. Serving constructors do not enable this.
    pub(crate) fn enable_compact_selection(&mut self,native:Rc<topk::Native>)->Result<()> {
        ensure!(self.compact_selection.is_none(),"compact selection already enabled");
        let width=self.keys.rows();
        self.compact_selection=Some(CompactSelection{
            windows:Windows::new(&self.cuda,width,&[0;6],&[0;6],&[0;6])?,
            selection:Selection::new(&self.cuda,native,6,width,self.dense.stride(),512)?,
        });
        Ok(())
    }
    pub(crate) fn compact_selection_enabled(&self)->bool {self.compact_selection.is_some()}
    pub(crate) fn compact_selection_lengths(&self)->Option<&Buffer<i32>> {
        self.compact_selection.as_ref().map(|c|c.windows.lengths())
    }
    pub(crate) fn fused_remap_sort_enabled(&self)->bool {
        self.fused_remap_sort && self.compact_selection.is_some()
    }

    fn launch(
        &self,
        name: &str,
        grid: usize,
        threads: usize,
        arguments: &mut [*mut c_void],
        stream: &Stream,
    ) -> Result<()> {
        let packed=self.packed_gather.as_ref().filter(|_|name=="ca_sparse6_gather_keys");
        let module=packed.unwrap_or(&self.module);
        let name=if packed.is_some(){"ca_sparse6_gather_keys_packed4"}else{name};
        unsafe {
            module.function(name)?.launch(
                [grid as u32, 1, 1],
                [threads as u32, 1, 1],
                0,
                stream,
                arguments,
            )
        }
    }

    /// All live positions share logical origin zero, including retained decode.
    /// `mask` is the current source20 publication. Geometry is checked here;
    /// bounds/overflow faults on GPU poison downstream selection before commit.
    pub(crate) fn run(
        &mut self,
        cache: &KeyCache,
        slots: &KeySlots,
        windows: &Windows,
        mask: &Buffer<u8>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            windows.rows() == 6
                && mask.len() == 6 * self.blocks
                && slots.values().len() == self.columns
                && slots.capacity() == cache.capacity()
                && self.cuda.owns_buffer(slots.values())
                && self.cuda.owns_buffer(mask)
                && self.cuda.owns_buffer(windows.lengths())
                && self.cuda.owns_buffer(cache.data())
                && self.cuda.owns_stream(stream),
            "candidate-only index cache/mask/window/context"
        );
        cache.bind(stream)?;
        self.run_paged(
            cache.data(),
            slots.values(),
            cache.capacity(),
            windows,
            mask,
            stream,
        )
    }

    fn run_paged(
        &mut self,
        cache: &Buffer<u8>,
        slots: &Buffer<i32>,
        cache_capacity: usize,
        windows: &Windows,
        mask: &Buffer<u8>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            windows.rows() == 6
                && slots.len() == self.columns
                && cache.len() == cache_capacity * 68
                && cache_capacity % 64 == 0
                && mask.len() == 6 * self.blocks
                && self.cuda.owns_buffer(cache)
                && self.cuda.owns_buffer(slots)
                && self.cuda.owns_buffer(windows.lengths())
                && self.cuda.owns_buffer(mask)
                && self.cuda.owns_stream(stream),
            "candidate-only raw cache/context"
        );
        self.keys.bind(stream)?;
        unsafe {
            self.cuda
                .zero_device(self.error.ptr(), self.error.bytes(), stream)
        }?;
        let (mut mask_ptr, mut lengths, mut counts, mut ids, mut error) = (
            mask.ptr(),
            windows.lengths().ptr(),
            self.counts.ptr(),
            self.ids.ptr(),
            self.error.ptr(),
        );
        let (mut blocks, mut columns, mut tiles, mut max_blocks) = (
            self.blocks as i32,
            self.columns as i32,
            self.blocks.div_ceil(256) as i32,
            self.ids.len() as i32,
        );
        self.launch(
            "ca_sparse6_union_counts",
            tiles as usize,
            256,
            &mut args![mask_ptr, lengths, counts, blocks, columns, error],
            stream,
        )?;
        let mut ends = self.ranges.key_ends_mut().ptr();
        self.launch(
            "ca_sparse6_union_offsets",
            1,
            32,
            &mut args![counts, tiles, max_blocks, ends, error],
            stream,
        )?;
        self.launch(
            "ca_sparse6_union_ids",
            tiles as usize,
            256,
            &mut args![
                mask_ptr, lengths, counts, ids, blocks, columns, max_blocks, error
            ],
            stream,
        )?;
        let (mut cache_ptr, mut slot_ptr, mut capacity, mut payload, mut scales, mut key_capacity) = (
            cache.ptr(),
            slots.ptr(),
            cache_capacity as i32,
            self.keys.payload.ptr(),
            self.keys.scales.ptr(),
            self.keys.rows() as i32,
        );
        self.launch(
            "ca_sparse6_gather_keys",
            192,
            128,
            &mut args![
                cache_ptr,
                slot_ptr,
                capacity,
                ids,
                ends,
                payload,
                scales,
                columns,
                key_capacity,
                error
            ],
            stream,
        )?;
        self.keys.ready.set(true);
        let (mut output, mut count) = (self.output.ptr(), self.output.len() as i32);
        if self.compact_selection.is_none() {
            self.launch("ca_sparse6_fill_scores",192,256,&mut args![output,count],stream)?;
        }
        self.dense.run(&self.ranges, stream)?;
        let (mut compact, mut compact_stride, mut stride) = (
            self.dense.output().ptr(),
            self.dense.stride() as i32,
            self.stride as i32,
        );
        if let Some(selection)=&self.compact_selection {
            let mut compact_lengths=selection.windows.lengths().ptr();
            return self.launch("ca_sparse6_mask_compact",192,256,
                &mut args![compact,compact_stride,ids,ends,lengths,mask_ptr,blocks,columns,compact_lengths],stream);
        }
        self.launch(
            "ca_sparse6_scatter_scores",
            192,
            256,
            &mut args![
                compact,
                compact_stride,
                output,
                stride,
                ids,
                ends,
                lengths,
                mask_ptr,
                blocks,
                columns
            ],
            stream,
        )
    }

    pub(crate) fn select(&mut self,output:&Selection,windows:&Windows,stream:&Stream)->Result<()> {
        if let Some(compact)=&self.compact_selection {
            compact.selection.run(self.dense.output_mut(),&compact.windows,stream)?;
            let (mut indices,mut scores,mut stride,mut ids,mut ends,mut lengths,mut error)=
                (compact.selection.indices().ptr(),self.dense.output().ptr(),self.dense.stride()as i32,
                self.ids.ptr(),self.ranges.key_ends().ptr(),windows.lengths().ptr(),self.error.ptr());
            if self.fused_remap_sort {
                return output.write_remapped_from(&compact.selection,windows,stream,|mut destination| {
                    self.launch("ca_sparse6_remap_sort_selection",6,512,
                        &mut args![indices,destination,scores,stride,ids,ends,lengths,error],stream)
                });
            }
            self.launch("ca_sparse6_remap_selection",6,512,
                &mut args![indices,scores,stride,ids,ends,lengths,error],stream)?;
            output.copy_remapped_from(&compact.selection,windows,stream)
        }else{output.run(&mut self.output,windows,stream)}
    }

    /// Untimed diagnostic expansion only. The compact graph omits both of
    /// these kernels; byte comparison still uses the original full layout.
    fn materialize_for_probe(&self,windows:&Windows,mask:&Buffer<u8>,stream:&Stream)->Result<()> {
        ensure!(self.compact_selection.is_some(),"diagnostic expansion requires compact mode");
        let(mut output,mut count)=(self.output.ptr(),self.output.len()as i32);
        self.launch("ca_sparse6_fill_scores",192,256,&mut args![output,count],stream)?;
        let(mut compact,mut compact_stride,mut stride,mut ids,mut ends,mut lengths,
            mut mask_ptr,mut blocks,mut columns)=(self.dense.output().ptr(),self.dense.stride()as i32,
            self.stride as i32,self.ids.ptr(),self.ranges.key_ends().ptr(),windows.lengths().ptr(),
            mask.ptr(),self.blocks as i32,self.columns as i32);
        self.launch("ca_sparse6_scatter_scores",192,256,
            &mut args![compact,compact_stride,output,stride,ids,ends,lengths,mask_ptr,blocks,columns],stream)
    }

    pub(crate) fn poison_selection_on_error(
        &self,
        indices: &Buffer<i32>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            indices.len() == 6 * 512 && self.cuda.owns_buffer(indices),
            "sparse selection geometry"
        );
        let (mut error, mut output) = (self.error.ptr(), indices.ptr());
        self.launch(
            "ca_sparse6_poison_selection",
            1,
            32,
            &mut args![error, output],
            stream,
        )
    }
    pub(crate) fn output(&self) -> &Buffer<f32> {
        &self.output
    }
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }
}

/// Hardware qualification, including captured changed inputs and a paired
/// complete gather/score/mask/top-k comparison. Fixture numerical work is GPU
/// generated; the host checks raw bytes, metadata, and elapsed event times.
pub fn probe(
    core: &std::path::Path,
    kernels: &std::path::Path,
    fixture_root: &std::path::Path,
) -> Result<serde_json::Value> {
    use crate::{
        candidates::Consumer,
        layer_probe::{Tap, fixture},
        topk::{self, Selection},
    };
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::io::Read;
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        columns: usize,
        cache_capacity: usize,
        sources: Vec<Tap>,
        phases: Vec<Phase>,
    }
    #[derive(Deserialize)]
    struct Phase {
        name: String,
        lengths: Vec<i32>,
        sources: Vec<Tap>,
        expected: Tap,
    }
    fn integers(bytes: &[u8]) -> Result<Vec<i32>> {
        ensure!(bytes.len() % 4 == 0, "sparse integer fixture alignment");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn cache_fixture(root: &std::path::Path, tap: &Tap, cache: &mut Buffer<u8>) -> Result<String> {
        // A 1M-column FP4 cache is just over the general 64MiB tap limit.
        // Stream its already-packed bytes with a bounded staging allocation;
        // no CPU packing or relaxation of the ordinary fixture contract.
        ensure!(
            tap.name == "cache" && tap.element_bytes == 1 && tap.bytes == cache.bytes(),
            "sparse cache fixture geometry"
        );
        let mut file = std::fs::File::open(root.join("cache.bin"))?;
        ensure!(
            file.metadata()?.len() == tap.bytes as u64,
            "sparse cache fixture length"
        );
        let mut digest = Sha256::new();
        let mut chunk = vec![0u8; (16 * 1024 * 1024).min(tap.bytes)];
        for offset in (0..tap.bytes).step_by(chunk.len()) {
            let n = chunk.len().min(tap.bytes - offset);
            file.read_exact(&mut chunk[..n])?;
            digest.update(&chunk[..n]);
            cache.upload_part(offset, &chunk[..n])?;
        }
        let digest = format!("{:x}", digest.finalize());
        ensure!(digest == tap.sha256, "sparse cache fixture hash");
        Ok(digest)
    }
    fn cache_digest(cuda: &Cuda, cache: &Buffer<u8>) -> Result<String> {
        let mut digest = Sha256::new();
        for offset in (0..cache.bytes()).step_by(16 * 1024 * 1024) {
            digest.update(unsafe {
                cuda.read_device(
                    cache.ptr() + offset as u64,
                    (cache.bytes() - offset).min(16 * 1024 * 1024),
                )
            }?);
        }
        Ok(format!("{:x}", digest.finalize()))
    }
    fn differences(
        actual: &[u8],
        expected: &[u8],
        lengths: &[i32],
        columns: usize,
    ) -> serde_json::Value {
        let mut count = 0usize;
        let mut first = Vec::new();
        for (i, (a, b)) in actual
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .enumerate()
        {
            if i % columns < lengths[i / columns] as usize && a != b {
                count += 1;
                if first.len() < 8 {
                    first.push(json!({"row":i / columns,"column":i % columns,
                        "actual_bits":u32::from_le_bytes(a.try_into().unwrap()),
                        "expected_bits":u32::from_le_bytes(b.try_into().unwrap())}));
                }
            }
        }
        json!({"different_values":count,"first":first})
    }
    fn copy_input(cuda: &Cuda, stream: &Stream, dest: u64, size: usize, raw: &[u8]) -> Result<()> {
        ensure!(raw.len() == size, "sparse input fixture size");
        let mut staged = cuda.allocate::<u8>(size)?;
        staged.upload(raw)?;
        stream.join_default()?;
        unsafe { cuda.copy_device(dest, staged.ptr(), size, stream) }?;
        stream.synchronize()
    }
    fn invoke(
        module: &Module,
        name: &str,
        grid: usize,
        threads: usize,
        args: &mut [*mut c_void],
        stream: &Stream,
    ) -> Result<()> {
        unsafe {
            module.function(name)?.launch(
                [grid as u32, 1, 1],
                [threads as u32, 1, 1],
                0,
                stream,
                args,
            )
        }
    }
    fn sort(module: &Module, selection: &Selection, stream: &Stream) -> Result<()> {
        let mut ids = selection.indices().ptr();
        invoke(
            module,
            "ca_sort_selected512",
            6,
            512,
            &mut args![ids],
            stream,
        )
    }
    let f: Fixture = serde_json::from_slice(&std::fs::read(fixture_root.join("index.json"))?)?;
    ensure!(
        f.cases.iter().map(|c| c.columns).collect::<Vec<_>>() == [32768, 262144, 1048576],
        "sparse fixture contexts"
    );
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = std::fs::read(core)?;
    let module = cuda.module(&core)?;
    let native = Native::open(&cuda, &kernels.join("mqa-v1"))?;
    let selector = topk::Native::open(&cuda, &kernels.join("topk-v2"))?;
    let compact_probe=std::env::var("CANADA_SCRATCH_COMPACT_INDEX_PROBE").as_deref()==Ok("1");
    let gpu_metrics_only=compact_probe
        ||std::env::var("CANADA_SCRATCH_SPARSE_GPU_METRICS_ONLY").as_deref()==Ok("1");
    let mut reports = Vec::new();
    for c in f.cases {
        ensure!(
            c.phases.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
                == ["initial", "changed", "short-ties", "empty", "union-max"],
            "sparse phase coverage"
        );
        let root = fixture_root.join(&c.name);
        let mut cache = cuda.allocate::<u8>(c.cache_capacity * 68)?;
        let mut slots = cuda.allocate::<i32>(c.columns)?;
        let mut original_cache = String::new();
        for tap in &c.sources {
            match tap.name.as_str() {
                "cache" => {
                    original_cache = cache_fixture(&root, tap, &mut cache)?;
                }
                "slots" => slots.upload(&integers(&fixture(&root, tap)?)?)?,
                _ => anyhow::bail!("unexpected sparse key fixture"),
            }
        }
        let queries = Rc::new(QueryRows::new(&cuda, 6)?);
        let full_keys = Rc::new(KeyMatrix::new(&cuda, c.columns)?);
        let mut full = Dense::new(
            &cuda,
            native.clone(),
            queries.clone(),
            full_keys.clone(),
            c.columns,
            64 * 1024 * 1024,
        )?;
        let compact_keys = Rc::new(KeyMatrix::new(&cuda, key_capacity(c.columns))?);
        let mut sparse = Sparse6::new(
            &cuda,
            &core,
            native.clone(),
            queries.clone(),
            compact_keys,
            c.columns,
            64 * 1024 * 1024,
        )?;
        if compact_probe{sparse.enable_compact_selection(selector.clone())?;}
        let mut windows = Windows::new(&cuda, c.columns, &[0; 6], &[0; 6], &[0; 6])?;
        let mut ranges = Ranges::new(&cuda, c.columns, c.columns, &[0; 6], &[0; 6])?;
        let mut mask = cuda.allocate::<u8>(6 * c.columns / BLOCK)?;
        let consumer = Consumer::new(&cuda, &core, 6, c.columns)?;
        let full_selection =
            Selection::new(&cuda, selector.clone(), 6, c.columns, full.stride(), 512)?;
        let sparse_selection =
            Selection::new(&cuda, selector.clone(), 6, c.columns, sparse.stride(), 512)?;
        let mut full_graph = None;
        let mut sparse_graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            for tap in &phase.sources {
                let raw = fixture(&directory, tap)?;
                match tap.name.as_str() {
                    "q" => copy_input(
                        &cuda,
                        &stream,
                        queries.payload.ptr(),
                        queries.payload.bytes(),
                        &raw,
                    )?,
                    "q_scales" => copy_input(
                        &cuda,
                        &stream,
                        queries.scales.ptr(),
                        queries.scales.bytes(),
                        &raw,
                    )?,
                    "weights" => copy_input(
                        &cuda,
                        &stream,
                        queries.weights.ptr(),
                        queries.weights.bytes(),
                        &raw,
                    )?,
                    "mask" => mask.upload(&raw)?,
                    _ => anyhow::bail!("unexpected sparse query fixture"),
                }
            }
            windows.update(&[0; 6], &phase.lengths, &[0; 6])?;
            ranges.update(&[0; 6], &phase.lengths)?;
            queries.ready.set(true);
            full_keys.ready.set(true);
            stream.join_default()?;
            let mut run_full = || -> Result<()> {
                let (
                    mut source,
                    mut slot_ptr,
                    mut payload,
                    mut scales,
                    mut columns,
                    mut ends,
                    mut rows,
                ) = (
                    cache.ptr(),
                    slots.ptr(),
                    full_keys.payload.ptr(),
                    full_keys.scales.ptr(),
                    c.columns as i32,
                    ranges.key_ends().ptr(),
                    6i32,
                );
                invoke(
                    &module,
                    "ca_gather_index_k_bounded",
                    192,
                    128,
                    &mut args![source, slot_ptr, payload, scales, columns, ends, rows],
                    &stream,
                )?;
                full.run(&ranges, &stream)?;
                let stride = full.stride();
                consumer.apply(full.output_mut(), stride, &windows, &mask, 0..6, &stream)?;
                full_selection.run(full.output_mut(), &windows, &stream)?;
                sort(&module, &full_selection, &stream)
            };
            if full_graph.is_none() {
                run_full()?;
                stream.synchronize()?;
                full_graph = Some(stream.capture(&mut run_full)?);
            }
            let mut run_sparse = || -> Result<()> {
                sparse.run_paged(&cache, &slots, c.cache_capacity, &windows, &mask, &stream)?;
                sparse.select(&sparse_selection, &windows, &stream)?;
                sort(&module, &sparse_selection, &stream)?;
                sparse.poison_selection_on_error(sparse_selection.indices(), &stream)
            };
            if sparse_graph.is_none() {
                run_sparse()?;
                stream.synchronize()?;
                sparse_graph = Some(stream.capture(&mut run_sparse)?);
            }
            for _ in 0..20 {
                unsafe {
                    full_graph.as_ref().unwrap().launch(&stream)?;
                    sparse_graph.as_ref().unwrap().launch(&stream)?;
                }
            }
            stream.synchronize()?;
            if compact_probe {
                sparse.materialize_for_probe(&windows,&mask,&stream)?;
                stream.synchronize()?;
            }
            let expected = fixture(&directory, &phase.expected)?;
            let a = unsafe { cuda.read_device(full.output().ptr(), 6 * c.columns * 4) }?;
            let b = unsafe { cuda.read_device(sparse.output().ptr(), 6 * c.columns * 4) }?;
            let error = sparse.error.download()?[0];
            // The existing dense primitive leaves unreachable columns
            // unspecified; they are outside the top-k windows. Do not turn
            // their contents into a false numerical gate or timed extra work.
            let full_reference_exact = phase.lengths.iter().enumerate().all(|(r, &n)| {
                let begin = r * c.columns * 4;
                a[begin..begin + n as usize * 4] == expected[begin..begin + n as usize * 4]
            });
            let sparse_reference_exact = b == expected;
            let full_indices = full_selection.indices().download()?;
            let sparse_indices = sparse_selection.indices().download()?;
            let selected_exact = full_indices == sparse_indices;
            let mut times = [Vec::new(), Vec::new()];
            let start = cuda.event()?;
            let end = cuda.event()?;
            for pair in 0..50 {
                for which in [pair % 2, 1 - pair % 2] {
                    start.record(&stream)?;
                    for _ in 0..10 {
                        unsafe {
                            if which == 0 {
                                full_graph.as_ref().unwrap().launch(&stream)?;
                            } else {
                                sparse_graph.as_ref().unwrap().launch(&stream)?;
                            }
                        }
                    }
                    end.record(&stream)?;
                    times[which].push(end.elapsed_since(&start)? / 10.);
                }
            }
            for t in &mut times {
                t.sort_by(f32::total_cmp);
            }
            let report = json!({"name":phase.name,"full_reference_exact":full_reference_exact,
                "sparse_reference_exact":sparse_reference_exact,"selected_exact":selected_exact,"error":error,
                // Index metadata is retained for an independent GPU optimality
                // check. Exact identities are not the selector's contract at
                // equal-score boundaries (see topk-v2's numerical contract).
                "selected_indices":{"full":full_indices,"sparse":sparse_indices},
                "full_differences":if gpu_metrics_only{serde_json::Value::Null}else{differences(&a,&expected,&phase.lengths,c.columns)},
                "sparse_differences":if gpu_metrics_only{serde_json::Value::Null}else{differences(&b,&expected,&vec![c.columns as i32;6],c.columns)},
                "selected_differences":full_indices.iter().zip(&sparse_indices).enumerate()
                    .filter(|(_, (a,b))| a!=b).take(8)
                    .map(|(i,(a,b))|json!({"row":i/512,"offset":i%512,"full":a,"sparse":b})).collect::<Vec<_>>(),
                "union_keys":sparse.ranges.key_ends().download()?[0],
                "full_gpu_ms":{"median":times[0][25],"p95":times[0][47]},
                "sparse_gpu_ms":{"median":times[1][25],"p95":times[1][47]},
                "scope":"Paired50x10 captured complete gather/score/mask/top-k; no model inference."});
            let mut brief=report.clone();brief.as_object_mut().unwrap().remove("selected_indices");
            eprintln!("{}", json!({"columns":c.columns,"phase_report":brief}));
            phases.push(report);
        }
        // Replay the same graph with invalid metadata. Never free/read past the
        // admitted union or let an internal fault look like ordinary padding.
        let mut invalid = mask.download()?;
        invalid[0] = 2;
        mask.upload(&invalid)?;
        stream.join_default()?;
        unsafe {
            sparse_graph.as_ref().unwrap().launch(&stream)?;
        }
        stream.synchronize()?;
        let invalid_mask_rejected =
            sparse.error.download()?[0] & 1 != 0 && sparse_selection.indices().download()?[0] == -2;
        let mut overflow_rejected = true;
        if c.columns > 6 * WINDOW {
            mask.upload(&vec![1; mask.len()])?;
            stream.join_default()?;
            unsafe {
                sparse_graph.as_ref().unwrap().launch(&stream)?;
            }
            stream.synchronize()?;
            overflow_rejected = sparse.error.download()?[0] & 2 != 0
                && sparse_selection.indices().download()?[0] == -2;
        }
        let cache_unchanged = cache_digest(&cuda, &cache)? == original_cache;
        let all_exact = phases.iter().all(|p| {
            p["full_reference_exact"] == true
                && p["sparse_reference_exact"] == true
                && p["selected_exact"] == true
                && p["error"] == 0
        }) && invalid_mask_rejected
            && overflow_rejected
            && cache_unchanged;
        let report = json!({"columns":c.columns,"all_exact":all_exact,"invalid_mask_rejected":invalid_mask_rejected,
            "overflow_rejected":overflow_rejected,"cache_unchanged":cache_unchanged,
            "compact_key_capacity":key_capacity(c.columns),"phases":phases});
        eprintln!(
            "{}",
            json!({"columns":c.columns,"all_exact":all_exact,
            "invalid_mask_rejected":invalid_mask_rejected,"overflow_rejected":overflow_rejected,
            "cache_unchanged":cache_unchanged})
        );
        reports.push(report);
    }
    Ok(
        json!({"all_exact":reports.iter().all(|r|r["all_exact"]==true),"cases":reports,
        "compact_selection_probe":compact_probe,"production_compact_selection_enabled":false,
        "scope":"Operator bytes, captured metadata, selection and fail-closed gates. Six-row zero-based consumer only. Full-model state/quality and workload speed remain unqualified."}),
    )
}
