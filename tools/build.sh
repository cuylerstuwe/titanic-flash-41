#!/usr/bin/env bash
set -euo pipefail
# Run inside deploy/Dockerfile, with the source at /project and persistent /data.
test "$(uname -m)" = aarch64
nvcc --version | grep -F 'V13.0.88'
rustc --version | grep -F 'rustc 1.99.0 '
test -f /opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export CARGO_HOME=/data/build/cargo
export CARGO_TARGET_DIR=/data/build/target
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
mkdir -p /data/runtime /data/build
export CANADA_BUILD_SOURCE_SHA256=$(python3 /project/tools/source_identity.py)
cargo build --manifest-path /project/Cargo.toml --locked --release --features serving
nvcc --cubin -arch=sm_121 --std=c++17 -O3 --fmad=false \
    /project/kernels/core.cu -o /data/runtime/core.cubin.tmp
nvcc --cubin -arch=sm_121 --std=c++17 -O3 --fmad=false \
    /project/kernels/vision.cu -o /data/runtime/vision.cubin.tmp
install -m755 /data/build/target/release/titanic-flash /data/runtime/titanic-flash.tmp
mv /data/runtime/core.cubin.tmp /data/runtime/core.cubin
mv /data/runtime/vision.cubin.tmp /data/runtime/vision.cubin
mv /data/runtime/titanic-flash.tmp /data/runtime/titanic-flash
/data/runtime/titanic-flash build-provenance > /data/runtime/provenance.json
python3 - <<'PY'
import json, os
j=json.load(open('/data/runtime/provenance.json'))
assert j['source_manifest_sha256']==os.environ['CANADA_BUILD_SOURCE_SHA256']
assert j['serving'] and j['vision_protocol'] and not j['storage_fault_injection']
PY
