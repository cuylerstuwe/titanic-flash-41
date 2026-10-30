//! Independent, pointer-stable GPU primitive graph executor and replay harness.
//! The model lowering will construct these nodes; fixtures can qualify them now.
use crate::{
    cuda::{Buffer, Cuda, Stream},
    primitives::Primitives,
    triton_ops::Operator,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Component, Path},
};

#[derive(Deserialize)]
struct LibrarySpec {
    id: String,
    root: String,
    #[serde(default)]
    library: String,
    #[serde(default)]
    bindings: String,
    #[serde(default)]
    operator: Option<String>,
}
#[derive(Deserialize)]
struct BufferSpec {
    name: String,
    bytes: usize,
    #[serde(default)]
    file: Option<String>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Argument {
    Pointer {
        buffer: String,
        #[serde(default)]
        offset: usize,
    },
    Word {
        value: u64,
    },
}
#[derive(Deserialize)]
struct Node {
    library: String,
    #[serde(default)]
    entry: i32,
    #[serde(default)]
    rows: u32,
    arguments: Vec<Argument>,
}
#[derive(Deserialize)]
struct Output {
    name: String,
    buffer: String,
    #[serde(default)]
    offset: usize,
    bytes: usize,
    expected: String,
}
#[derive(Deserialize)]
struct Program {
    libraries: Vec<LibrarySpec>,
    buffers: Vec<BufferSpec>,
    nodes: Vec<Node>,
    outputs: Vec<Output>,
}
struct PreparedNode {
    library: usize,
    entry: i32,
    rows: u32,
    values: Vec<u64>,
}
enum Loaded {
    Cute(Primitives),
    Triton(Operator),
}
fn basename(name: &str) -> Result<()> {
    ensure!(
        Path::new(name).components().count() == 1
            && matches!(
                Path::new(name).components().next(),
                Some(Component::Normal(_))
            ),
        "fixture file must be a basename"
    );
    Ok(())
}
fn execute(nodes: &[PreparedNode], libraries: &[Loaded], stream: &Stream) -> Result<()> {
    for n in nodes {
        let lib = match &libraries[n.library] {
            Loaded::Triton(op) => {
                unsafe { op.launch(n.rows, &n.values, stream) }?;
                continue;
            }
            Loaded::Cute(lib) => lib,
        };
        if lib.bindings.abi == 2 {
            unsafe { lib.launch_words(n.entry, &n.values, stream) }?;
        } else {
            let e = &lib.bindings.entries[n.entry as usize];
            let np = e.pointers.len();
            let scalars: Vec<_> = n.values[np..].iter().map(|v| *v as u32 as i32).collect();
            unsafe { lib.launch(n.entry, &n.values[..np], &scalars, stream) }?;
        }
    }
    Ok(())
}

pub fn qualify(path: &Path) -> Result<serde_json::Value> {
    let spec: Program = serde_json::from_reader(File::open(path)?)?;
    let root = path.parent().context("fixture has no parent directory")?;
    ensure!(
        !spec.nodes.is_empty() && spec.nodes.len() <= 4096,
        "bounded nonempty primitive graph required"
    );
    let bytes = spec
        .buffers
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(b.bytes))
        .context("program allocation overflow")?;
    ensure!(
        bytes > 0 && bytes <= 16 * 1024 * 1024 * 1024usize,
        "fixture device budget exceeds 16 GiB"
    );
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    ensure!(
        cuda.admission_memory()?.admitted_available > bytes + 2 * 1024 * 1024 * 1024,
        "insufficient memory for bounded primitive graph"
    );
    let mut buffers = BTreeMap::<String, Buffer<u8>>::new();
    let mut staging = vec![0u8; 16 * 1024 * 1024];
    for b in &spec.buffers {
        ensure!(
            b.bytes > 0 && !buffers.contains_key(&b.name),
            "invalid/duplicate fixture buffer"
        );
        let mut buffer = cuda.allocate::<u8>(b.bytes)?;
        buffer.zero()?;
        if let Some(file) = &b.file {
            basename(file)?;
            let mut f = File::open(root.join(file))?;
            ensure!(
                f.metadata()?.len() == b.bytes as u64,
                "fixture data size differs from manifest"
            );
            let mut offset = 0;
            while offset < b.bytes {
                let count = staging.len().min(b.bytes - offset);
                f.read_exact(&mut staging[..count])?;
                buffer.upload_part(offset, &staging[..count])?;
                offset += count;
            }
        }
        buffers.insert(b.name.clone(), buffer);
    }
    drop(staging);
    let mut libraries = Vec::new();
    let mut names = BTreeMap::new();
    for l in &spec.libraries {
        ensure!(!names.contains_key(&l.id), "duplicate library id");
        let base = Path::new(&l.root);
        names.insert(l.id.clone(), libraries.len());
        if let Some(name) = &l.operator {
            libraries.push(Loaded::Triton(Operator::load(&cuda, base, name)?));
        } else {
            basename(&l.library)?;
            basename(&l.bindings)?;
            libraries.push(Loaded::Cute(Primitives::open(
                &cuda,
                &base.join(&l.library),
                &base.join(&l.bindings),
            )?));
        }
    }
    let mut nodes = Vec::new();
    for n in &spec.nodes {
        let library = *names.get(&n.library).context("unknown node library")?;
        let expected = match &libraries[library] {
            Loaded::Triton(op) => op.entry.arguments.len(),
            Loaded::Cute(lib) => {
                let entry = lib
                    .bindings
                    .entries
                    .get(n.entry as usize)
                    .context("unknown node entry")?;
                if lib.bindings.abi == 2 {
                    entry.arguments.len()
                } else {
                    entry.pointers.len() + entry.i32_scalars.len()
                }
            }
        };
        ensure!(
            n.arguments.len() == expected,
            "fixture argument count differs from ABI"
        );
        let values = n
            .arguments
            .iter()
            .map(|a| -> Result<u64> {
                Ok(match a {
                    Argument::Pointer { buffer, offset } => {
                        let b = buffers.get(buffer).context("unknown node buffer")?;
                        ensure!(*offset < b.len(), "pointer outside buffer");
                        b.ptr() + *offset as u64
                    }
                    Argument::Word { value } => *value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        nodes.push(PreparedNode {
            library,
            entry: n.entry,
            rows: n.rows,
            values,
        });
    }
    execute(&nodes, &libraries, &stream)?;
    stream.synchronize()?;
    let mut reports = Vec::new();
    for o in &spec.outputs {
        basename(&o.expected)?;
        let expected = std::fs::read(root.join(&o.expected))?;
        ensure!(
            expected.len() == o.bytes,
            "expected output size differs from manifest"
        );
        let b = buffers.get(&o.buffer).context("unknown output buffer")?;
        let actual = b.download_part(o.offset, o.bytes)?;
        let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        reports.push(serde_json::json!({"name":o.name,"bytes":o.bytes,"different_bytes":different,"bitwise_equal":different==0}));
        ensure!(
            different == 0,
            "primitive graph output {} differs in {different} bytes",
            o.name
        );
    }
    let graph = stream.capture(|| execute(&nodes, &libraries, &stream))?;
    let start = cuda.event()?;
    let end = cuda.event()?;
    start.record(&stream)?;
    for _ in 0..50 {
        unsafe { graph.launch(&stream) }?;
    }
    end.record(&stream)?;
    let graph_ms = end.elapsed_since(&start)? / 50.;
    // Recheck outputs after repeated replay: catches scratch epochs/aliasing bugs.
    for o in &spec.outputs {
        let actual = buffers[&o.buffer].download_part(o.offset, o.bytes)?;
        ensure!(
            actual == std::fs::read(root.join(&o.expected))?,
            "primitive graph replay changed output"
        );
    }
    Ok(
        serde_json::json!({"device_bytes":bytes,"nodes":nodes.len(),"graph_ms":graph_ms,"outputs":reports,"replays":50}),
    )
}
