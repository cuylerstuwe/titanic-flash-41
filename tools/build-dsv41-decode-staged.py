#!/usr/bin/env python3
"""Add staged sparse-cache reads and first-chunk L2 prefetch to the six-row split-decode export.

Supply the canonical split-decode bundle (for headers) and its built prefetch-v1
bundle, followed by the new destination. The CUDA ABI and workspace size stay
compatible with both earlier exports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("prefetch_v1", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cutlass-include", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    base = args.base.resolve()
    previous = args.prefetch_v1.resolve()
    dest = args.destination.resolve()
    original = previous / "decode.cu"
    patch = Path(__file__).resolve().parents[1] / "kernels/dsv41_decode_staged.patch"
    manifest = json.loads((previous / "manifest.json").read_bytes())
    if sha(original) != "712c97be808734d536830e4858d373a7d689fd963ea73878f11807f7af28bd2e":
        raise ValueError("unsupported prefetch-v1 source")
    if manifest["adapter_sha256"] != sha(original) or manifest["base_adapter_sha256"] != sha(base / "decode.cu"):
        raise ValueError("split-decode source does not match the prefetch manifest")
    for name, digest in manifest["header_sha256"].items():
        if sha(base / "include" / name) != digest:
            raise ValueError(f"header digest mismatch for {name}")
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        source = stage / "decode.cu"
        subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(source), str(original), str(patch)], check=True)
        if sha(source) != "37ef8f089866f17956f91d48eadb4a078572171edeff02d9b82bbe297b0f3ff9":
            raise ValueError("unexpected staged split-decode source")
        lib = stage / "libcanada_decode.so"
        flags = ["-shared", "-Xcompiler=-fPIC", "-O3", "-gencode=arch=compute_121a,code=sm_121a",
                 "-std=c++17", "--expt-relaxed-constexpr", "--use_fast_math", "-DNDEBUG"]
        subprocess.run([args.nvcc, *flags, "-I"+str(base/"include"), "-I"+str(args.cutlass_include.resolve()),
                        str(source), "-o", str(lib), "-lcudart"], check=True)
        manifest.update(adapter_sha256=sha(source), library_sha256=sha(lib), generator_sha256=sha(Path(__file__)),
                        patch_sha256=sha(patch), previous_adapter_sha256=sha(original),
                        gather_load_batch=16,
                        prefetch="first selected cache chunk during query load and following chunks during attention")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        stage.rename(dest)


if __name__ == "__main__":
    main()
