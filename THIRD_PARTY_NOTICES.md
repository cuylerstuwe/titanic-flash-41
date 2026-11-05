# Third-party components

No license is granted here for Titanic Flash 4.1's original code. This file records the separate terms of components used by the build and runtime; it is not a license for the combined application. Existing copyright headers and upstream notices are retained.

| Component | Use | Upstream identity | Terms / included notice |
|---|---|---|---|
| b12x | Compiled GB10 dense, MoE and communication primitives and associated source/ABI files | [repository](https://github.com/local-inference-lab/b12x), `a7d7d29b2ef8869086e0ceaa787321f17544e3c9`; older SG17 operators identify archive SHA-256 `4922d0db9395fb8b537d7e216d509a5747867a45f20c18290e385786609dd05f` in their manifests | [Apache-2.0](licenses/b12x-Apache-2.0.txt) |
| SGLang | Selected low-level operators, top-k and vision-attention source; build-image environment | [repository](https://github.com/sgl-project/sglang), `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9` | [Apache-2.0](licenses/SGLang-Apache-2.0.txt) |
| FlashInfer | RMS and attention-related primitives and transitive top-k source | [repository](https://github.com/flashinfer-ai/flashinfer), version 0.6.18; individual source files retain their commit references | [BSD-3-Clause and embedded notices](licenses/FlashInfer-BSD-3-Clause.txt) |
| DeepGEMM | Sparse attention operators and FP4 conversion source | [repository](https://github.com/deepseek-ai/DeepGEMM), `14073b4e1e706506e193231209738c848d092a1f`; individual files identify earlier source excerpts | [MIT](licenses/DeepGEMM-MIT.txt) |
| vLLM | Source incorporated in upstream top-k operators | [persistent top-k source](https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh) | [Apache-2.0](licenses/vLLM-Apache-2.0.txt) |
| LightLLM | Source ancestry retained by the upstream vision-attention file | [repository](https://github.com/ModelTC/lightllm), `f2a54f0912293f683bf1d1695fd12c4098a5bf82` | [Apache-2.0](licenses/LightLLM-Apache-2.0.txt) |
| DeepSeek recipe | Canonical model protocol, tokenizer integration and image processing crate | [repository](https://github.com/deepseek-ai/deepseek-recipe), crate 0.2.0, `935817c66f5d6e64838f419692545f06787e485f` | [MIT](licenses/DeepSeek-recipe-MIT.txt) |
| PyTorch | Headers / code used by compiled dependency bridges; build environment | [repository](https://github.com/pytorch/pytorch) | [license and third-party notices](licenses/PyTorch-LICENSE.txt) |
| NVIDIA NCCL | Device collectives, dynamically loaded from the container | [repository](https://github.com/NVIDIA/nccl), runtime version 2.30.7 | [BSD-3-Clause](licenses/NCCL-BSD-3-Clause.txt) |
| NVIDIA CUTLASS DSL | Compiler used for selected GPU operators | Version 4.6.2 | [NVIDIA Software License Agreement](licenses/NVIDIA-CUTLASS-DSL.txt); compiler package is not vendored |
| CPython | Public benchmark context and source fixtures | [repository](https://github.com/python/cpython) | [PSF license](licenses/Python-PSF.txt) |
| Tokio | Public source fixtures | [repository](https://github.com/tokio-rs/tokio) | [MIT](licenses/Tokio-MIT.txt) |

This software contains source code provided by NVIDIA Corporation. NVIDIA CUDA compiler/runtime and driver components are obtained separately through the pinned container and host installation and retain NVIDIA's terms. This repository does not redistribute a CUDA toolkit or CUTLASS DSL compiler installation.

## Separately fetched required WO-A operator

The runtime requires the low-level lossless W8 WO-A operator from [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks), revision `64b02dea8681fb1d8e3a659ea432a8e9e9ba3995`. It is **AGPL-3.0**, with the text retained in [licenses/Mia-AGPL-3.0-external.txt](licenses/Mia-AGPL-3.0-external.txt). It is compiled together with the separately licensed SGLang reduction/quantization operator. `tools/prepare_runtime.py --with-external-woa` fetches pinned source and licenses into the local data directory and compiles the operators there.

The setup now also builds the M16/M32 compact WO-A specialization from that same pinned source, with FP32 partial accumulation and one BF16 output rounding. Its four kernels are verified alongside the fourteen small-row operators. The build recipe is in `tools/build_external_woa_mid.py`.

The AGPL source and its compiled WO-A binaries are **not included in this Git repository**. Separate acquisition is not a statement that AGPL obligations disappear: use, modification, network service and redistribution of a combined deployment must respect the applicable licenses. No claim is made that the whole application can be distributed under a permissive license. Other optional AGPL kernel experiments are not part of this release.

## Rust dependencies and modifications

`Cargo.lock` pins the complete Rust dependency graph. Cargo obtains these dependencies under their own declared licenses; their source packages are not vendored. Preserve the relevant license files if distributing a compiled executable or container. The checked GPU bundle contains generated ABI files, local bridge/adaptation code and selected modified operators; per-directory manifests identify sources and artifact hashes. Local changes include bounded row handling, launch geometry, layout/ABI adaptations and synchronization fixes.

The benchmark client is separately downloaded from the linked gist with an integrity check, not redistributed. Generated answers and private prompts are not included. Model weights are separately downloaded from DeepSeek under the checkpoint's own terms.
