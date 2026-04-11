# Findings — Heterogeneous GPU Kernel Dispatch via MLIR

*Updated 2026-04-02.*

## Core Research Question

Can we build a lightweight, vendor-agnostic runtime dispatch layer that integrates with MLIR's compilation pipeline to enable same-source ML kernels across NVIDIA, AMD, and CPU targets with near-native performance?

**Answer: Yes, and the gap is well-documented and confirmed open.**

---

## The Gap Is Real — Evidence from Multiple Sources

### 1. IREE's Own Issues Confirm It
- **Issue #50** (open since 2019-10-13): foundational target configuration issue, unresolved after 6+ years
- **Issue #12230**: Phase 1 (shape dedup) done; Phase 2 (runtime strategy selection) **stalled since May 2023**, acknowledged as "sort of broken"
- **Issue #15334**: Multi-versioning epic — ALL tasks remain unchecked
- IREE does multi-target compilation but requires 100K+ LOC full stack buy-in

### 2. MLIR's `gpu.select_object` Is Compile-Time Only
- The `gpu-module-to-binary` pass CAN produce multi-target binaries (NVPTX, AMDGCN, SPIR-V)
- But `gpu.select_object` resolves at compile time during LLVM IR translation — NOT at runtime
- The `GPUOffloadingLLVMTranslationAttrInterface` is the extensibility point for a runtime-aware handler
- No upstream MLIR mechanism does runtime hardware detection to choose among variants

### 3. No Lightweight Standalone Solution Exists
- IREE: full runtime (100K+ LOC), requires ecosystem buy-in
- SYCL/AdaptiveCpp: requires SYCL programming model
- ALPAKA: compile-time only (C++ templates, >94% native perf but no runtime dispatch)
- Kokkos/RAJA: compile-time backend selection (CMake flag)
- OCCA: runtime JIT but no ML-specific optimization, niche adoption
- **Nobody has built the <1000 LOC MLIR-native dispatch layer**

### 4. Academic Literature Confirms the Gap
- 80+ papers surveyed across runtime dispatch, JIT compilation, ML compilation, performance portability, hardware introspection
- **No published system performs runtime dynamic kernel dispatch across NVIDIA/AMD/CPU from a unified MLIR IR** (papers-ml-compilation.md)
- Closest prior art: HetGPU (arXiv:2506.15993) solves binary compatibility but not ML-kernel-aware dispatch
- IRIS (ORNL, IEEE TPDS 2024) wraps CUDA/HIP/L0/OpenCL but has no cost-model-driven selection

---

## Key Performance Data Points

| Metric | Value | Source |
|--------|-------|--------|
| SYCL P3 portability scores | 0.46-0.65 (vs Kokkos 0.75-0.99) | Davis et al. ICS 2025 |
| AdaptiveCpp SSCP JIT overhead | ~15% first launch, near-zero cached | IWOCL 2025 |
| SYCL-MLIR speedup over DPC++ | Up to 4.3x | CGO 2024 |
| ALPAKA vs native CUDA | >94% performance | arXiv:1602.08477 |
| Vulkan vs CUDA on A100 | ~20-30% slower general | llama.cpp benchmarks |
| Vulkan vs ROCm on RDNA3 | 0-50% FASTER | llama.cpp benchmarks |
| SPIR-V vs native | 50-80% of native performance | arXiv:2603.28793 |
| Runtime dispatch overhead (our layer) | <10ns (vs 5-20μs kernel launch) | dispatch-latency research |
| Multi-versioned SGEMM vs theoretical max | Within 10% | arXiv:2507.15277 |
| JIT + autotuning vs vendor implementations | >230% for LLM kernels | arXiv:2505.03780 |
| cuBLAS ML recommender accuracy | 93% of optimal | NVIDIA docs |
| Helix (mixed GPU clusters) | 3.3x throughput | ASPLOS 2025 |

---

## "ML Kernels Are Static" Is Empirically False

Reviewer 91B claimed ML kernels are well-known at compile time. Evidence shows the opposite:

1. **cuBLAS:** Hundreds of GEMM kernels per precision. ML-trained recommender selects at runtime. `cublasSetSmCountTarget()` allows runtime SM count override.
2. **cuDNN v9:** Three heuristic modes + runtime fusion engines that **NVRTC-compile kernels on-the-fly** based on compute capability.
3. **PyTorch dispatcher:** Computes `DispatchKeySet` on **every operator call** from live tensor metadata and thread-local state.
4. **torch.compile:** Guards checked at every invocation; shape changes trigger recompilation or fallback. Backend selection is manual (no auto hardware detection).
5. **vLLM:** Piecewise CUDA Graphs with runtime batch-size bucket selection. Heterogeneous GPU serving explicitly unsupported.
6. **CUDA Graphs:** Each unique shape requires re-recording; practitioners maintain one graph per shape bucket — multi-version dispatch in practice.

Where runtime dispatch adds value:
- **Heterogeneous serving clusters** (Helix: 3.3x throughput on mixed GPUs)
- **Cloud portability** (deploy same model on AWS NVIDIA / Azure AMD / CPU fallback)
- **Edge deployment** on unknown hardware (ExecuTorch 1.0)
- **Multi-tenant inference** with mixed GPU generations

---

## Chosen Contribution: `mlir-hetero-dispatch`

### Architecture

```
                    BUILD TIME                              RUNTIME
              +--------------------+              +---------------------+
              |                    |              |                     |
  MLIR Source |  linalg.matmul     |              |  1. discover_devices()
  (linalg)   |        |           |              |     -> [A100, MI300, CPU]
              |        v           |              |                     |
              |  gpu.launch_func   |              |  2. load routing table
              |   + target attrs   |              |     (from binary bundle)
              |        |           |              |                     |
              |        v           |              |  3. match capabilities
              | gpu-module-to-     |              |     kernel.contract vs
              |   binary           |              |     device.capabilities
              |   |    |    |      |              |                     |
              |   v    v    v      |              |  4. cost_model_rank()
              | nvptx amdgcn x86   |              |     -> A100 wins for GEMM
              |   |    |    |      |              |                     |
              |   v    v    v      |              |  5. launch(nvptx_binary)
              | [routing table]    |              |     fallback -> amdgcn
              | [bundled binary]   |              |     fallback -> x86
              +--------------------+              +---------------------+
```

### Components

1. **Multi-target AOT compilation** — Use MLIR's existing `gpu-module-to-binary` to compile a single `gpu.module` to NVPTX + AMDGCN + x86 simultaneously
2. **Kernel routing table** — Generated at compile time, mapping `(kernel_name, device_vendor, min_capability) → binary_offset`
3. **Capability contracts** — Each kernel variant declares requirements: `requires {cuda >= 11.0, sm >= 80, shared_mem >= 48KB}`
4. **Cost-model-driven selection** — Roofline model estimating execution time per device based on FLOPS, bandwidth, parallelism
5. **Fallback chain** — Priority-ordered: try best GPU, fall back to next, ultimate CPU fallback
6. **`libkdl` runtime** — <1000 LOC C library: discover + match + rank + dispatch

### Why This Is Novel

| Aspect | IREE | SYCL/AdaptiveCpp | ALPAKA | Proteus | **Ours** |
|--------|------|------------------|--------|---------|----------|
| Runtime dispatch | Partial (stalled) | Yes | No | Yes | **Yes** |
| Cross-vendor | Yes | Yes | Yes | No | **Yes** |
| MLIR-native | Yes | No | No | No | **Yes** |
| Lightweight | No (100K+ LOC) | No | Yes | Yes | **Yes (<1000 LOC)** |
| No prog model req | Yes | No (SYCL) | No (C++) | No | **Yes** |
| Cost model | No | No | No | No | **Yes** |

### Addressing ALL Reviewer Concerns

| Concern | Response |
|---------|----------|
| 91A: Need concrete mechanism | Working prototype with `libkdl` runtime + MLIR pass pipeline |
| 91B: ML kernels are static | cuBLAS/cuDNN/PyTorch all do runtime dispatch. Show heterogeneous serving value |
| 91B: Connect to PyTorch/TF | Architecture sketch for torch.compile backend + ONNX RT EP integration |
| 91C: Survey vs proposal | PROPOSAL with working prototype and benchmarks |
| 91C: Too specific for non-experts | Framed as "ld.so for GPU kernels" — dynamic linking analogy |
| 91D: Acknowledge IREE SPIR-V | IREE generates vendor-agnostic SPIR-V but requires full stack. We're lightweight |
| 91D: Why SYCL? | Broadened: compare SYCL, SPIR-V, HIP, Triton, IREE, ALPAKA |
| 91D: Multi-versioned JIT | Our routing table + capability contracts IS this |

---

## Key References (Top 20 for Poster)

1. MLIR GPU Dialect docs — `mlir.llvm.org/docs/Dialects/GPU/`
2. IREE Issues #50, #15334, #12230 — evidence of unsolved runtime dispatch
3. SYCL-MLIR (CGO 2024) — MLIR-based SYCL compilation, up to 4.3x speedup
4. AdaptiveCpp SSCP (IWOCL 2023, 2025) — runtime JIT dispatch model
5. ALPAKA (arXiv:1602.08477) — compile-time portability, >94% native perf
6. Proteus (CGO 2025) — portable JIT via LLVM IR, 2.8x on AMD
7. HetGPU (arXiv:2506.15993) — binary compatibility across vendors
8. Multi-versioning SGEMM (arXiv:2507.15277) — within 10% of theoretical max
9. Helix (ASPLOS 2025) — 3.3x throughput on heterogeneous clusters
10. PyTorch 2 (ASPLOS 2024) — torch.compile + TorchInductor
11. Composable MLIR Codegen (arXiv:2202.03293) — IREE's design basis
12. TVM (OSDI 2018) — auto-tuning ML compiler
13. Triton (MAPL 2019) — GPU kernel programming in Python
14. Kokkos 3 (IEEE TPDS 2022) — performance portability framework
15. Universal GPU ISA (arXiv:2603.28793) — 10 universal primitives
16. GPU Portability Needs Autotuning (arXiv:2505.03780) — >230% improvement
17. ONNX Runtime EP architecture — `onnxruntime.ai/docs/`
18. Pennycook P3 metric (arXiv:1611.07409) — performance portability quantification
19. IRIS (IEEE TPDS 2024) — unified runtime wrapping CUDA/HIP/L0/OpenCL
20. KernelEvolve (Meta, arXiv:2512.23236) — production multi-target kernel generation
