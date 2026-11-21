#!/usr/bin/env python3
"""Build the six-row verify bundle from the installed V4.1 split-decode headers.

Run in the same CUDA build image as attention-dsv41-decode-v1. Supply the
original bundle, the new bundle directory and the CUTLASS include directory.
The standard split-decode ABI and workspace layout are shared by both bundles.
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
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cutlass-include", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    base, dest = args.base.resolve(), args.destination.resolve()
    patch = Path(__file__).resolve().parents[1] / "kernels/dsv41_decode_prefetch.patch"
    manifest = json.loads((base / "manifest.json").read_bytes())
    original = base / "decode.cu"
    if sha(original) != "c6d4fb1c116491162be6f81be820025beade6aa6b5302d39f6763152e8436709":
        raise ValueError("unsupported base decode source")
    for name, digest in manifest['header_sha256'].items():
        if sha(base / "include" / name) != digest:
            raise ValueError(f"header digest mismatch for {name}")
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        source = stage / "decode.cu"
        subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(source),
                        str(original), str(patch)], check=True)
        library = stage / "libcanada_decode.so"
        compile_flags = ["-shared", "-Xcompiler=-fPIC", "-O3", "-gencode=arch=compute_121a,code=sm_121a",
                         "-std=c++17", "--expt-relaxed-constexpr", "--use_fast_math", "-DNDEBUG"]
        subprocess.run([args.nvcc, *compile_flags, "-I" + str(base / "include"),
                        "-I" + str(args.cutlass_include.resolve()), str(source), "-o", str(library),
                        "-lcudart"], check=True)
        manifest.update(adapter_sha256=sha(source), library_sha256=sha(library),
                        generator_sha256=sha(Path(__file__)), patch_sha256=sha(patch),
                        base_adapter_sha256=sha(original), headers_from="attention-dsv41-decode-v1",
                        prefetch="next chunk of selected cache rows and scales into L2",
                        selected_shape={"rows": 6, "extra_topk": 512, "splits": 8},
                        compiler_arguments=["nvcc", *compile_flags, "-I<base>/include",
                                            "-I<cutlass>/include", "decode.cu", "-o", "libcanada_decode.so", "-lcudart"])
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(dest)


if __name__ == "__main__":
    main()
