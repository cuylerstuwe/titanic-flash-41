//! Header-first, read-only checkpoint loading. Never deserialize arbitrary code.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

const HEADER_LIMIT: u64 = 32 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum Dtype {
    #[serde(rename = "F32")]
    F32,
    #[serde(rename = "BF16")]
    Bf16,
    #[serde(rename = "F8_E4M3")]
    Fp8,
    #[serde(rename = "F8_E8M0")]
    E8m0,
    #[serde(rename = "I8")]
    PackedFp4,
}
impl Dtype {
    pub fn bytes(&self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tensor {
    pub file: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}
impl Tensor {
    pub fn bytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
    pub fn validate(&self, file: &Shard) -> Result<()> {
        ensure!(
            !self.shape.is_empty() && self.shape.len() <= 4,
            "unsupported tensor rank"
        );
        let elems = self
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .context("shape overflow")?;
        let bytes = elems
            .checked_mul(self.dtype.bytes())
            .context("tensor size overflow")?;
        ensure!(
            self.data_offsets[0] <= self.data_offsets[1],
            "reversed offsets"
        );
        ensure!(self.bytes() == bytes, "shape/byte count mismatch");
        ensure!(
            self.data_offsets[1]
                <= file
                    .size
                    .checked_sub(file.data_start)
                    .context("invalid data start")?,
            "tensor past end"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Shard {
    pub size: u64,
    pub data_start: u64,
    #[serde(default)]
    pub header_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Checkpoint {
    pub files: BTreeMap<String, Shard>,
    pub tensors: BTreeMap<String, Tensor>,
}

fn safe_filename(name: &str) -> Result<()> {
    let p = Path::new(name);
    ensure!(
        p.extension().is_some_and(|s| s == "safetensors"),
        "unexpected shard extension"
    );
    ensure!(
        p.components().count() == 1 && matches!(p.components().next(), Some(Component::Normal(_))),
        "shard must be a basename"
    );
    Ok(())
}

impl Checkpoint {
    pub fn from_manifest(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        c.validate()?;
        Ok(c)
    }
    pub fn scan(root: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let index: Index =
            serde_json::from_reader(BufReader::new(File::open(root.join("model.safetensors.index.json"))?))?;
        ensure!(!index.weight_map.is_empty(), "empty checkpoint");
        let names: BTreeSet<_> = index.weight_map.values().cloned().collect();
        let mut c = Self {
            files: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        for name in names {
            safe_filename(&name)?;
            let mut f = File::open(root.join(&name)).with_context(|| format!("open {name}"))?;
            let size = f.metadata()?.len();
            let mut n = [0u8; 8];
            f.read_exact(&mut n)?;
            let n = u64::from_le_bytes(n);
            ensure!(
                n > 1 && n <= HEADER_LIMIT && n + 8 <= size,
                "invalid header size: {name}"
            );
            let mut buf = vec![0u8; n as usize];
            f.read_exact(&mut buf)?;
            let h: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&buf)?;
            let shard = Shard {
                size,
                data_start: n + 8,
                header_sha256: format!("{:x}", Sha256::digest(&buf)),
            };
            for (key, value) in h {
                if key == "__metadata__" {
                    continue;
                }
                ensure!(
                    index.weight_map.get(&key) == Some(&name),
                    "index/header disagreement: {key}"
                );
                #[derive(Deserialize)]
                struct Entry {
                    dtype: Dtype,
                    shape: Vec<u64>,
                    data_offsets: [u64; 2],
                }
                let e: Entry = serde_json::from_value(value).with_context(|| key.clone())?;
                let t = Tensor {
                    file: name.clone(),
                    dtype: e.dtype,
                    shape: e.shape,
                    data_offsets: e.data_offsets,
                };
                t.validate(&shard).with_context(|| key.clone())?;
                ensure!(
                    c.tensors.insert(key.clone(), t).is_none(),
                    "duplicate tensor {key}"
                );
            }
            c.files.insert(name, shard);
        }
        ensure!(
            c.tensors.len() == index.weight_map.len(),
            "index tensor missing in headers"
        );
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.files.is_empty() && !self.tensors.is_empty(),
            "empty manifest"
        );
        let mut ranges = BTreeMap::<&str, Vec<(u64, u64)>>::new();
        for (name, f) in &self.files {
            safe_filename(name)?;
            ensure!(
                f.data_start >= 8 && f.data_start <= f.size,
                "invalid shard metadata"
            );
        }
        for (name, t) in &self.tensors {
            t.validate(self.files.get(&t.file).context("unknown shard")?)
                .with_context(|| name.clone())?;
            ranges
                .entry(&t.file)
                .or_default()
                .push((t.data_offsets[0], t.data_offsets[1]));
        }
        for (name, mut rr) in ranges {
            rr.sort_unstable();
            let mut end = 0;
            for (start, next) in rr {
                ensure!(
                    start == end,
                    "overlap or hole in {name} at {start} expected {end}"
                );
                end = next;
            }
            let s = &self.files[name];
            ensure!(
                end == s.size - s.data_start,
                "unindexed trailing data in {name}"
            );
        }
        Ok(())
    }
    pub fn read_slice(&self, root: &Path, name: &str, offset: u64, out: &mut [u8]) -> Result<()> {
        let t = self.tensors.get(name).context("unknown tensor")?;
        ensure!(
            offset
                .checked_add(out.len() as u64)
                .is_some_and(|e| e <= t.bytes()),
            "slice outside tensor"
        );
        let s = &self.files[&t.file];
        let path = root.join(&t.file);
        let mut f = File::open(&path)?;
        ensure!(
            f.metadata()?.len() == s.size,
            "shard size changed since scan"
        );
        f.seek(SeekFrom::Start(s.data_start + t.data_offsets[0] + offset))?;
        f.read_exact(out)?;
        Ok(())
    }
    pub fn root_files(&self, root: &Path) -> Vec<PathBuf> {
        self.files.keys().map(|p| root.join(p)).collect()
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("required tensor absent: {name}"))
    }
    pub fn require(&self, name: &str, dtype: Dtype, shape: &[u64]) -> Result<()> {
        let t = self.tensor(name)?;
        if t.dtype != dtype || t.shape != shape {
            bail!(
                "unexpected type/shape for {name}: {:?} {:?}",
                t.dtype,
                t.shape
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_path_traversal() {
        for p in [
            "../x.safetensors",
            "/x.safetensors",
            "a/b.safetensors",
            "x.bin",
        ] {
            assert!(safe_filename(p).is_err());
        }
        assert!(safe_filename("model-00001-of-00048.safetensors").is_ok());
    }
    #[test]
    fn rejects_overflow_and_truncation() {
        let s = Shard {
            size: 108,
            data_start: 8,
            header_sha256: String::new(),
        };
        let mut t = Tensor {
            file: "x.safetensors".into(),
            dtype: Dtype::Bf16,
            shape: vec![50],
            data_offsets: [0, 100],
        };
        t.validate(&s).unwrap();
        t.shape = vec![u64::MAX, 4];
        assert!(t.validate(&s).is_err());
        t.shape = vec![51];
        assert!(t.validate(&s).is_err());
    }
}
