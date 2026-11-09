//! Staged full (unsplit) canonical FP8 matrices for sequence-parallel row-local
//! projections. An artifact directory under the primitive exports holds, per layer,
//! the canonical FP8 values and 32x32 UE8M0 scales assembled from the four qualified
//! TP4 shards. Each rank checks the digests and that its own qualified shard is
//! byte-identical to its part of the staged matrix before use.
use crate::{blockscaled::Matrix, cuda::{Cuda, Module, Stream}, weights::Weights};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::BTreeMap, path::Path, rc::{Rc, Weak}};

/// How the qualified TP4 shard maps into the full matrix.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Split { Rows, Columns }

thread_local! {
    static CACHE: RefCell<BTreeMap<(usize, String, usize), Weak<Matrix>>> = const { RefCell::new(BTreeMap::new()) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load(cuda: &Cuda, module: &Module, stream: &Stream, weights: &Rc<Weights>, kernels: &Path,
    artifact: &str, prefix: &str, layer: usize, rank: usize, n: usize, k: usize, split: Split) -> Result<Rc<Matrix>> {
    load_parts(cuda, module, stream, weights, kernels, artifact, None, &[prefix], layer, rank, n, k, split)
}
/// `stem` selects files l{layer}-{stem}-weight.bin and manifest keys {stem}-weight_sha256.
/// A row-split shard may be the concatenation of several tensors (e.g. [w1; w3]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_parts(cuda: &Cuda, module: &Module, stream: &Stream, weights: &Rc<Weights>, kernels: &Path,
    artifact: &str, stem: Option<&str>, prefixes: &[&str], layer: usize, rank: usize, n: usize, k: usize,
    split: Split) -> Result<Rc<Matrix>> {
    ensure!(!prefixes.is_empty() && (split == Split::Rows || prefixes.len() == 1), "staged shard composition");
    let key = (cuda.context_key(), format!("{artifact}/{}", stem.unwrap_or("")), layer);
    if let Some(m) = CACHE.with(|c| c.borrow().get(&key).and_then(Weak::upgrade)) {
        ensure!(m.in_context(cuda) && (m.n, m.k) == (n, k), "staged full matrix context/geometry");
        return Ok(m);
    }
    ensure!(rank < 4 && n % 128 == 0 && k % 128 == 0, "staged full matrix geometry");
    let root = kernels.join(artifact);
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let entry = &manifest["layers"][layer.to_string()];
    let tag = stem.map(|s| format!("{s}-")).unwrap_or_default();
    let weight = std::fs::read(root.join(format!("l{layer}-{tag}weight.bin")))?;
    let scale = std::fs::read(root.join(format!("l{layer}-{tag}scale.bin")))?;
    let (sn, sk) = (n / 32, k / 32);
    ensure!(weight.len() == n * k && scale.len() == sn * sk, "staged full matrix size");
    ensure!(entry[format!("{tag}weight_sha256")].as_str() == Some(format!("{:x}", Sha256::digest(&weight)).as_str())
        && entry[format!("{tag}scale_sha256")].as_str() == Some(format!("{:x}", Sha256::digest(&scale)).as_str()),
        "staged full matrix digest");
    let (shard, scale_shard): (Vec<u8>, Vec<u8>) = match split {
        Split::Rows => {
            let rows = n / 4;
            (weight[rank * rows * k..(rank + 1) * rows * k].to_vec(),
             scale[rank * rows / 32 * sk..(rank + 1) * rows / 32 * sk].to_vec())
        }
        Split::Columns => {
            let cols = k / 4;
            ((0..n).flat_map(|i| weight[i * k + rank * cols..i * k + (rank + 1) * cols].iter().copied()).collect(),
             (0..sn).flat_map(|i| scale[i * sk + rank * cols / 32..i * sk + (rank + 1) * cols / 32].iter().copied()).collect())
        }
    };
    // This rank's qualified tensors, in order, must equal its part of the staged matrix.
    let (mut wdone, mut sdone) = (0usize, 0usize);
    for prefix in prefixes {
        let w = weights.view(&format!("{prefix}.weight"))?;
        let s = weights.view(&format!("{prefix}.scale"))?;
        let cols = if split == Split::Rows { k } else { k / 4 };
        ensure!(w.shape.len() == 2 && w.shape[1] == cols as u64 && s.shape.len() == 2
            && s.shape[1] == (cols / 32) as u64 && wdone + w.bytes <= shard.len() && sdone + s.bytes <= scale_shard.len(),
            "staged shard tensor geometry");
        let mut done = 0;
        while done < w.bytes {
            // Compare in diagnostic-staging-sized pieces (device_range is bounded at 16 MiB).
            let len = (16usize << 20).min(w.bytes - done);
            ensure!(weights.device_range(w.offset + done, len)? == shard[wdone + done..wdone + done + len],
                "staged full matrix disagrees with this rank's qualified shard");
            done += len;
        }
        ensure!(weights.device_range(s.offset, s.bytes)? == scale_shard[sdone..sdone + s.bytes],
            "staged full scales disagree with this rank's qualified shard");
        wdone += w.bytes; sdone += s.bytes;
    }
    ensure!(wdone == shard.len() && sdone == scale_shard.len(), "staged shard coverage");
    let mut values = cuda.allocate::<u8>(n * k)?;
    values.upload(&weight)?;
    let mut canonical = cuda.allocate::<u8>(scale.len())?;
    canonical.upload(&scale)?;
    let matrix = Rc::new(Matrix::from_owned(cuda, module, stream, weights, Rc::new(values), &canonical, n, k)?);
    stream.synchronize()?;
    drop(canonical);
    CACHE.with(|c| c.borrow_mut().insert(key, Rc::downgrade(&matrix)));
    Ok(matrix)
}

thread_local! {
    static SHARDS: RefCell<BTreeMap<(usize, String, usize, usize), Weak<Matrix>>> = const { RefCell::new(BTreeMap::new()) };
}
/// One peer's contiguous K-split shard [n, k/4] cut from a staged full column-split
/// matrix; this rank's own part is first verified against its qualified shard.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_column_shard(cuda: &Cuda, module: &Module, stream: &Stream, weights: &Rc<Weights>, kernels: &Path,
    artifact: &str, prefix: &str, layer: usize, rank: usize, peer: usize, n: usize, k: usize) -> Result<Rc<Matrix>> {
    let key = (cuda.context_key(), artifact.to_owned(), layer, peer);
    if let Some(m) = SHARDS.with(|c| c.borrow().get(&key).and_then(Weak::upgrade)) {
        ensure!(m.in_context(cuda) && (m.n, m.k) == (n, k / 4), "staged shard context/geometry");
        return Ok(m);
    }
    ensure!(peer < 4 && rank < 4 && k % 512 == 0, "column shard geometry");
    // Digest + own-shard verification (cached full matrix is reused, not duplicated).
    let _full = load(cuda, module, stream, weights, kernels, artifact, prefix, layer, rank, n, k, Split::Columns)?;
    let root = kernels.join(artifact);
    let weight = std::fs::read(root.join(format!("l{layer}-weight.bin")))?;
    let scale = std::fs::read(root.join(format!("l{layer}-scale.bin")))?;
    let (sk, cols) = (k / 32, k / 4);
    ensure!(weight.len() == n * k && scale.len() == n / 32 * sk, "staged shard source size");
    let w: Vec<u8> = (0..n).flat_map(|i| weight[i * k + peer * cols..i * k + (peer + 1) * cols].iter().copied()).collect();
    let sc: Vec<u8> = (0..n / 32).flat_map(|i| scale[i * sk + peer * cols / 32..i * sk + (peer + 1) * cols / 32].iter().copied()).collect();
    let mut values = cuda.allocate::<u8>(w.len())?;
    values.upload(&w)?;
    let mut canonical = cuda.allocate::<u8>(sc.len())?;
    canonical.upload(&sc)?;
    let matrix = Rc::new(Matrix::from_owned(cuda, module, stream, weights, Rc::new(values), &canonical, n, cols)?);
    stream.synchronize()?;
    drop(canonical);
    SHARDS.with(|c| c.borrow_mut().insert(key, Rc::downgrade(&matrix)));
    Ok(matrix)
}
