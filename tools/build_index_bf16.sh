#!/usr/bin/env bash
# build_index_bf16.sh SOURCE_TREE OUT_DIR VERSION: the engine-owned bf16 index artifact from
# SOURCE_TREE/kernels/index_bf16.cu: index16.cubin, the source, and manifest.json naming both hashes. The engine
# (src/index_bf16.rs) refuses an artifact whose source_sha256 differs from the source it was built with.
set -euo pipefail
SRC=$1/kernels/index_bf16.cu; OUT=$2; VERSION=$3
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
test -s "$SRC" && test ! -e "$OUT"
mkdir -p "$OUT"
"$NVCC" --cubin -arch=sm_121a -std=c++17 -O3 -o "$OUT/index16.cubin" "$SRC"
cp "$SRC" "$OUT/index_bf16.cu"
python3 - "$OUT" "$VERSION" "$("$NVCC" --version | grep -o 'release [0-9.]*')" <<'PY'
import hashlib, json, sys
out, version, release = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump({"schema": 1, "architecture": "sm_121a", "functions": ["ca_idx16_score", "ca_idx16_select", "ca_idx16_dense", "ca_idx16_dense_pair", "ca_idx16_mask_row", "ca_idx16_dense_solo"],
           "source_version": version, "source_sha256": sha(f"{out}/index_bf16.cu"),
           "artifact_sha256": sha(f"{out}/index16.cubin"),
           "build": f"nvcc {release} --cubin -arch=sm_121a -std=c++17 -O3",
           "contract": "Index scores with the reference's bf16 arithmetic (DeepSeek model.py: einsum, relu, mul, sum in bf16) "
                       "written relative to each row's key start, as 16-bit order-preserving keys or f32 values for topk "
                       "windows; each row's exact top-k (ties at the cut by ascending column) written as offset + "
                       "(column - start), ascending, -1 padded; the fused score + select of up to two independent eight-row groups. Adapted from "
                       "pico's indexer (MIT, Copyright (c) 2026 Andy Shu)."},
          open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(dict(built=out, source=version)))
PY
