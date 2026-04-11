# Mega Research Survey: Heterogeneous GPU Kernel Dispatch in LLVM/MLIR

**Status:** Complete
**Started:** 2026-04-06
**Topic:** Heterogeneous GPU kernel dispatch in LLVM/MLIR

## Summary
- 6 research waves across ~50 files
- ~450+ sources surveyed
- 5 research directions identified and ranked
- 3 abstract variants drafted

## Top Research Direction: libkdl as liboffload's Policy Layer (9.0/10)

**The gap:** LLVM's liboffload (PR #186088) explicitly selects "first compatible image" and defers multi-version selection policy. MLIR's `#gpu.select_object` is compile-time index-based. No cross-vendor runtime dispatch policy exists anywhere in the LLVM/MLIR stack.

**libkdl fills this:** Pre-compiled fat binary with runtime O(1) dispatch table lookup, <0.8% overhead (validated by arXiv:2601.00227), cross-vendor persistent kernel cache.

## Coverage Map
| Dimension | Status | Key Sources | Wave Files |
|-----------|--------|-------------|------------|
| Runtime dispatch | **Complete** | liboffload, IREE HAL, PJRT, ORT EPs | wave-01/02/04 |
| Compilation pipelines | **Complete** | MLIR gpu dialect, Triton, torch-mlir | wave-01/05 |
| Performance characteristics | **Complete** | CUDA 4.7μs, HIP 70μs, O(1) dispatch 1-2μs | wave-02/03 |
| API design | **Complete** | ol*/ur*, Level Zero, CUDA driver API | wave-04/06 |
| Ecosystem integration | **Complete** | PyTorch Inductor, TVM, ExecuTorch, ONNX RT | wave-01/03/04 |
| Hardware abstraction | **Complete** | SPIR-V, fat binaries, HSACO, cubin ABI | wave-01/02/06 |
| Competitive landscape | **Complete** | HetGPU, chipStar, AdaptiveCpp, Alpaka, GPU Ocelot | wave-03/05 |
| Abandoned approaches | **Complete** | HSA, GPU Ocelot, GCC HSA offload, OpenCL | wave-05 |
| Kernel caching | **Complete** | Triton 843s cold, cross-framework comparison | wave-04 |
| Cost models | **Complete** | SparseX CGO 2026, tritonBLAS roofline | wave-03/04 |

## Key Quantitative Data for Poster
| Metric | Value | Source |
|--------|-------|--------|
| CUDA null-kernel dispatch floor | 4.71 μs (H100) | wave-02 TaxBreak |
| HIP launch overhead | 70 μs (ROCm 3.8) | wave-02 Kokkos #3670 |
| Dynamic dispatch overhead | <0.8% e2e | wave-03 arXiv:2601.00227 |
| O(1) dispatch table lookup | 1-2 μs | wave-03 FlashInfer-Bench |
| chipStar SPIR-V portability cost | 0.75x native | wave-05 IJHPCA 2026 |
| HetGPU JIT cold start | 10-200 ms | wave-03 arXiv:2506.15993 |
| Triton JIT cold start | 843 s (Meta PT2) | wave-04 kernel caching |
| AdaptiveCpp JIT speedup | +30% over CUDA | wave-02 IWOCL 2025 |
| CERN CMS Alpaka GPUs | ~450 L4 GPUs | wave-03 CHEP 2025 |

## Competitive Positioning
| System | Dispatch Level | Runtime? | Cross-Vendor? | libkdl Advantage |
|--------|---------------|----------|---------------|------------------|
| liboffload | Mechanism only | Yes | Yes (ol*) | libkdl adds selection policy |
| IREE HAL | Load-time static | Yes | Yes | libkdl: per-kernel adaptive |
| HetGPU | JIT translation | Yes | Yes | libkdl: no toolchain change, pre-compiled |
| AdaptiveCpp | JIT at load | Yes | Yes | libkdl: no JIT overhead, persistent cache |
| Alpaka | Event-level routing | No | Compile-time | libkdl: kernel-level, runtime |
| ONNX RT | Static greedy EP | Partial | Yes | libkdl: adaptive, not first-fit |
| ExecuTorch | AOT-only | No | Compile-time | libkdl: runtime selection |
| PJRT | Per-process plugin | Yes | Single-device | libkdl: mixed device-kind |
| GPU Ocelot | Below vendor RT | Yes | Yes (dead) | libkdl: above vendor RT (correct) |

## Synthesis Documents
- `synthesis/round-01.md` — 310 sources, initial direction ranking
- `synthesis/round-02.md` — 414+ sources, liboffload integration elevated to rank 2

## Abstract Variants
- `abstracts/abstract-v1.md` — "Policy Layer Above liboffload"
- `abstracts/abstract-v2.md` — "ld.so for GPU Kernels"
- `abstracts/abstract-v3.md` — "Bridging Compile-Time and Runtime"

## Files
- `config.yaml` — Run configuration and angle tracking
- `waves/` — 50+ raw wave files from 30 researcher agents
- `synthesis/` — 2 synthesis rounds
- `directions/` — Ranked research directions (final)
- `abstracts/` — Draft abstract candidates
