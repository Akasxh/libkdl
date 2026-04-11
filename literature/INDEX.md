# Literature Index — Vendor-Agnostic Runtime Dispatch for ML Kernels

**Project:** LLVM Developers' Meeting, Dublin 2026
**Last updated:** 2026-04-06
**Total files:** 37 (20 original + 17 new research notes)
**Total size:** ~952KB of structured literature

---

## How to Use This Index

This index maps every literature file to the **paper section** it supports.
Each entry includes: file path, word count estimate, key systems covered, and relevance tier.

**Relevance tiers:**
- **T1 (Must-cite):** Directly addresses our problem or is the primary reference for a system we compare against
- **T2 (Should-cite):** Provides important context, metrics, or related approaches
- **T3 (Background):** General background or tangentially related

---

## Section 1: Problem Statement & Motivation

*Why runtime dispatch matters. Why existing solutions are insufficient.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `survey.md` | Master survey: taxonomy of all approaches, executive summary, gap analysis | T1 | P3 scores, IREE LOC counts, performance comparisons |
| `findings.md` (repo root) | Synthesized findings across all research phases | T1 | "ML is static" rebuttal, key performance data table |
| `notes/novelty-gaps.md` | 5 novelty gaps identified with assessments | T1 | Gap analysis for each contribution angle |
| `notes/brainstorm-problem-first.md` | Problem decomposition, analogical transfer, idea generation | T2 | 5 subproblems, CFS/DB optimizer analogies |
| `papers-performance-portability.md` | Pennycook P3 metric, Davis et al. ICS 2025, framework comparisons | T1 | P3 scores: Kokkos 0.75-0.99, SYCL 0.46-0.65 |

---

## Section 2: Background — MLIR and GPU Compilation

*MLIR infrastructure, gpu dialect, compilation pipeline.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `mlir-jit-analysis.md` | MLIR ExecutionEngine, gpu-module-to-binary, JIT limitations | T1 | gpu.select_object is compile-time only |
| `papers-ml-compilation.md` | 15+ papers: MLIR (Lattner 2021), TVM, XLA, survey papers | T1 | Foundational references |
| `triton-compiler-approaches.md` | Triton architecture, MLIR dialects, TVM, XLA, StableHLO comparison | T1 | Triton v2.0 MLIR rewrite, backend architecture |
| `mlir-gpu-infrastructure-2026.md` | gpu-module-to-binary pass internals, gpu.select_object, SPIR-V RFC, XeVM target, runtime selection design | T1 | Concrete upstream integration point identified |

---

## Section 3: Related Work — Compile-Time Portability

*Frameworks that achieve portability through compile-time abstraction.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `alpaka-sofie-analysis.md` | ALPAKA RHP model, backend matrix, TMVA-SOFIE integration | T1 | >94% native CUDA perf, CMS Run 3 deployment |
| `papers-runtime-dispatch.md` | 26 papers: Kokkos 3, AdaptiveCpp SSCP, Davis et al. | T1 | Comprehensive paper catalog |
| `hip-rocm.md` | HIP thin header model, hipify, ROCm ecosystem | T2 | ~400 API functions, 90-95% auto-translation |
| `cern-cms-alpaka-production.md` | CMS Run 3, Patatrack production evidence | T1 | Complete |
| `alpaka-perf-portability.md` | ALPAKA benchmark studies | T2 | Complete |

---

## Section 4: Related Work — Runtime Portability Layers

*Systems enabling device selection at execution time.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `sycl-ecosystem.md` | SYCL standard, DPC++, AdaptiveCpp, device selectors | T1 | P3 0.46-0.65, ~150us launch latency |
| `opencl-lessons.md` | OpenCL history, lessons learned, ICD dispatch, SPIR-V legacy | T2 | 15-year trajectory, source ≠ performance portability |
| `vulkan-webgpu.md` | Vulkan compute, cooperative matrices, capability detection | T2 | ~70-80% CUDA perf on NVIDIA, matches ROCm on RDNA3 |
| `spirv-analysis.md` | SPIR-V as universal IR, chipStar, clvk, LLVM RFC | T1 | chipStar 0.75x vs native, 50-80% native perf |
| `adaptivecpp-sscp.md` | AdaptiveCpp single-pass JIT | T1 | Complete |
| `sycl-mlir-cgo2024.md` | SYCL-MLIR 4.3x speedup paper | T1 | Complete |
| `chipstar-2026-spirv-portability.md` | chipStar IJHPCA 2026 | T2 | Complete |

---

## Section 5: Related Work — Full-Stack ML Compilers

*End-to-end systems (IREE, TVM, XLA) with their own dispatch.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `iree-deep-dive.md` | IREE architecture, HAL, Flow/Stream dialects, multi-target | T1 | Issues #50/#12230/#15334, 100K+ LOC runtime |
| `competitive-landscape.md` | All major systems compared: Triton, TVM, IREE, SYCL, Kokkos, etc. | T1 | Complete comparison matrix |
| `iree-2026-state.md` | IREE 2026 current status | T1 | Complete |
| `openxla-pjrt-2026.md` | OpenXLA, PJRT, StableHLO | T2 | Complete |
| `tvm-unity-multi-target.md` | TVM Unity, Relax | T2 | Complete |
| `torch-mlir-bridge.md` | Torch-MLIR multi-target | T2 | Complete |

---

## Section 6: Related Work — Production ML Dispatch

*How real ML frameworks handle kernel selection at runtime.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `production-ml-dispatch.md` | PyTorch dispatcher, torch.compile, cuBLAS/cuDNN heuristics, vLLM | T1 | Rebuts "ML is static" argument |
| `onnxrt-multi-ep.md` | ONNX Runtime execution providers, graph partitioning | T2 | Priority-based EP selection |
| `onnxrt-multi-ep-deep.md` | Deeper ONNX RT EP analysis | T2 | Complete |
| `helix-2025-mixed-gpu.md` | Helix ASPLOS 2025, 3.3x throughput | T1 | Complete |
| `executorch-edge-dispatch.md` | ExecuTorch backend delegation | T2 | Complete |

---

## Section 7: Technical Foundations — JIT, Introspection, Cost Models

*Building blocks for the proposed system.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `papers-jit-gpu.md` | 22 papers: Proteus (CGO 2025), ProSpec, Leo, multi-versioning | T1 | Proteus 2.8x on AMD, 1.78x on NVIDIA |
| `papers-hardware-introspection.md` | CUDA/Vulkan/OpenCL/HIP device query APIs | T1 | Comprehensive API reference |
| `cost-models-kernel-dispatch.md` | Roofline, cuBLAS heuristics, auto-tuning | T1 | Complete |
| `multi-versioned-kernels.md` | CUDA fat binaries, AMD code objects | T1 | Complete |

---

## Section 8: Heterogeneous Runtime Systems

*Closest prior art for runtime dispatch across vendors.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `iris-2024-task-dispatch.md` | IRIS ORNL, IEEE TPDS 2024 | T1 | Complete |
| `hetgpu-binary-compat.md` | HetGPU binary compatibility | T2 | Complete |
| `tmva-sofie-gpu-2025.md` | TMVA-SOFIE GPU with ALPAKA | T1 | Complete |

---

## Section 9: Our Contribution — Design Context

*Files directly informing our proposed system design.*

| File | Coverage | Tier | Key Data |
|------|----------|------|----------|
| `notes/creative-thinking.md` | Creative research frameworks applied to our problem | T3 | Cognitive science approaches |
| `research/abstract.md` | Submitted abstract | T1 | Original submission text |
| `research/reviews.md` | 4 reviewer comments with synthesis | T1 | Action items from reviews |
| `research-log.md` (repo root) | Phase tracking and progress | T3 | 25 agents, 12K+ lines |

---

## Paper Coverage Matrix

Maps each major system to the files where it is analyzed:

| System | Primary File(s) | Depth |
|--------|-----------------|-------|
| **MLIR gpu dialect** | `mlir-jit-analysis.md`, `new/mlir-gpu-infrastructure-2026.md` | Deep |
| **IREE** | `iree-deep-dive.md`, `new/iree-2026-state.md` | Deep |
| **Triton** | `triton-compiler-approaches.md`, `competitive-landscape.md` | Deep |
| **TVM** | `triton-compiler-approaches.md`, `papers-ml-compilation.md`, `new/tvm-unity-multi-target.md` | Medium → Deep |
| **XLA/OpenXLA** | `triton-compiler-approaches.md`, `new/openxla-pjrt-2026.md` | Medium → Deep |
| **SYCL** | `sycl-ecosystem.md`, `new/sycl-mlir-cgo2024.md`, `new/adaptivecpp-sscp.md` | Deep |
| **ALPAKA** | `alpaka-sofie-analysis.md`, `new/cern-cms-alpaka-production.md` | Deep |
| **Kokkos** | `papers-runtime-dispatch.md`, `competitive-landscape.md` | Medium |
| **RAJA** | `papers-runtime-dispatch.md`, `competitive-landscape.md` | Medium |
| **HIP/ROCm** | `hip-rocm.md` | Deep |
| **SPIR-V** | `spirv-analysis.md`, `new/chipstar-2026-spirv-portability.md` | Deep |
| **OpenCL** | `opencl-lessons.md` | Medium |
| **Vulkan** | `vulkan-webgpu.md` | Medium |
| **PyTorch** | `production-ml-dispatch.md` | Deep |
| **ONNX Runtime** | `onnxrt-multi-ep.md`, `new/onnxrt-multi-ep-deep.md` | Medium → Deep |
| **cuBLAS/cuDNN** | `production-ml-dispatch.md`, `new/cost-models-kernel-dispatch.md` | Medium |
| **Proteus** | `papers-jit-gpu.md` | Medium |
| **chipStar** | `spirv-analysis.md`, `new/chipstar-2026-spirv-portability.md` | Medium → Deep |
| **IRIS** | `iris-2024-task-dispatch.md` | Deep (new) |
| **Helix** | `helix-2025-mixed-gpu.md` | Deep (new) |
| **ExecuTorch** | `executorch-edge-dispatch.md` | Medium (new) |
| **TMVA-SOFIE** | `alpaka-sofie-analysis.md`, `new/tmva-sofie-gpu-2025.md` | Deep |

---

## Quick Reference: Key Numbers

| Metric | Value | Source File |
|--------|-------|-------------|
| SYCL P3 portability scores | 0.46-0.65 | `survey.md` |
| Kokkos P3 scores | 0.75-0.99 | `survey.md` |
| ALPAKA vs native CUDA | >94% | `alpaka-sofie-analysis.md` |
| SYCL-MLIR speedup over DPC++ | up to 4.3x | `survey.md` |
| chipStar vs native HIP | 0.75x geometric mean | `spirv-analysis.md` |
| Vulkan vs CUDA on A100 | ~20-30% slower | `survey.md` |
| Vulkan vs ROCm on RDNA3 | 0-50% faster | `survey.md` |
| IREE runtime size | 100K+ LOC | `iree-deep-dive.md` |
| Proteus JIT speedup (AMD) | up to 2.8x | `papers-jit-gpu.md` |
| Helix mixed GPU throughput | 3.3x | `findings.md` |
| cuBLAS recommender accuracy | 93% optimal | `production-ml-dispatch.md` |
| Kernel launch latency (CUDA) | 5-20us | `papers-hardware-introspection.md` |
