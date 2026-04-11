# Paper Outline — Vendor-Agnostic Runtime Dispatch for ML Kernels via MLIR

**Working Title:** "Closing the Runtime Gap: Lightweight Vendor-Agnostic Kernel Dispatch for MLIR-Compiled ML Workloads"
**Venue:** LLVM Developers' Meeting, Dublin 2026 (Poster → potential workshop paper)
**Authors:** Akash (IIT Patna)
**Status:** DRAFT — literature review in progress

---

## Thesis Statement

MLIR's `gpu-module-to-binary` pass can compile ML kernels to multiple GPU targets simultaneously, but the runtime selection mechanism (`gpu.select_object`) is compile-time only. We propose a lightweight runtime dispatch layer (~500 LOC) that bridges this gap through hardware introspection and cost-model-driven kernel routing, filling a need that IREE acknowledges (issues #50, #12230, #15334) but addresses only with a 100K+ LOC full-stack solution.

---

## Paper Structure

### 1. Introduction (1 page)

**Key arguments:**
1. Heterogeneous GPU environments are now the norm, not the exception (TOP500 data, cloud GPU fleets)
2. MLIR provides the compilation infrastructure for multi-target code generation
3. The "last mile" — runtime dispatch — remains unsolved outside full-stack systems
4. We propose a minimal, composable dispatch layer inspired by `ld.so` for GPU kernels

**Reviewer feedback to address:**
- 91A: Must have concrete contribution, not just survey
- 91C: Better motivation for non-specialists
- 91D: Why not just SPIR-V? Frame as more general than SYCL

**Literature support:**
- `findings.md` — gap evidence
- `survey.md` — problem framing
- `papers-performance-portability.md` — P3 metric for evaluation framing

### 2. Background (1.5 pages)

#### 2.1 MLIR GPU Compilation Pipeline
- linalg → gpu.launch_func → gpu-module-to-binary → {NVPTX, AMDGCN, x86}
- `gpu.select_object` semantics and limitations
- `GPUOffloadingLLVMTranslationAttrInterface` as extensibility point

**Literature:** `mlir-jit-analysis.md`, `new/mlir-gpu-infrastructure-2026.md`

#### 2.2 The Heterogeneous GPU Landscape
- Cloud: AWS (NVIDIA), Azure (AMD MI300X), GCP (TPU + NVIDIA)
- HPC: Frontier (AMD), Aurora (Intel), Perlmutter (NVIDIA)
- Edge: unknown hardware at deployment time

**Literature:** `survey.md` §2

#### 2.3 Why "ML Kernels Are Static" Is Wrong
- PyTorch dispatcher: per-call DispatchKeySet computation
- cuBLAS: ML-trained recommender, 93% optimal
- cuDNN v9: runtime fusion via NVRTC
- CUDA Graphs: per-shape re-recording
- Dynamic shapes, mixed-precision, batch size variation

**Literature:** `production-ml-dispatch.md`, `findings.md` §"ML Kernels Are Static"

### 3. Related Work (2 pages)

#### 3.1 Compile-Time Portability Frameworks
- **ALPAKA:** RHP model, >94% native perf, compile-time backend selection, CMS Run 3 production
- **Kokkos:** P3 0.75-0.99, polymorphic memory layouts, compile-time only
- **RAJA:** Lambda-based execution policies, P3 0.47-1.00, compile-time only

**Key contrast:** Near-native performance but require recompilation per target. Our approach: decouple compilation from target selection.

**Literature:** `alpaka-sofie-analysis.md`, `papers-runtime-dispatch.md` A1-A2, `new/cern-cms-alpaka-production.md`, `new/alpaka-perf-portability.md`

#### 3.2 Runtime Portability Layers
- **SYCL:** Runtime device selection, but P3 0.46-0.65, ~150us launch overhead
- **AdaptiveCpp SSCP:** Single-pass JIT, unified code representation, closest to runtime dispatch
- **OpenCL:** Source portability ≠ performance portability, 15-year cautionary tale
- **Vulkan compute:** Broadest hardware coverage, capability detection via `vkGetPhysicalDeviceFeatures2`

**Key contrast:** Trade performance for portability. Our approach: native-performance kernels selected at runtime.

**Literature:** `sycl-ecosystem.md`, `opencl-lessons.md`, `vulkan-webgpu.md`, `new/adaptivecpp-sscp.md`, `new/sycl-mlir-cgo2024.md`

#### 3.3 Full-Stack ML Compilers
- **IREE:** Most complete MLIR runtime, HAL abstraction, but 100K+ LOC buy-in. Issues #50/#12230/#15334 document unsolved dispatch.
- **Triton:** MLIR-native, near-cuBLAS perf, but single-target at compile time
- **TVM:** Auto-tuning, MetaSchedule, but own IR stack, limited MLIR integration
- **XLA/OpenXLA:** PJRT plugin model, StableHLO, but Google-centric

**Key contrast:** Full ecosystems vs. composable layer. Our approach: thin shim that any MLIR user can adopt.

**Literature:** `iree-deep-dive.md`, `triton-compiler-approaches.md`, `competitive-landscape.md`, `new/iree-2026-state.md`, `new/openxla-pjrt-2026.md`

#### 3.4 Heterogeneous Runtime Systems
- **IRIS (ORNL):** Wraps CUDA/HIP/L0/OpenCL, task-based, but no cost-model-driven selection
- **chipStar:** HIP/CUDA → SPIR-V → any OpenCL/L0 device, 0.75x native, but reverse direction
- **Proteus:** Portable JIT for GPU kernels via LLVM IR, 2.8x AMD, 1.78x NVIDIA
- **HetGPU:** Binary compatibility across GPU vendors

**Key contrast:** Either too heavyweight, wrong direction (source→SPIR-V vs. MLIR→native), or not ML-aware.

**Literature:** `new/iris-2024-task-dispatch.md`, `new/chipstar-2026-spirv-portability.md`, `papers-jit-gpu.md` §1, `new/hetgpu-binary-compat.md`

#### 3.5 SPIR-V as Universal Intermediate
- LLVM RFC: SPIR-V as vendor-agnostic GPU representation
- Performance cost: 50-80% of native
- Reviewer 91D's point: IREE SPIR-V backend CAN generate vendor-agnostic code

**Key insight:** SPIR-V works as a portable fallback tier but cannot match native performance. Our approach: prefer native binaries, SPIR-V as fallback.

**Literature:** `spirv-analysis.md`, `new/mlir-gpu-infrastructure-2026.md`

### 4. Design: Kernel Dynamic Linker (libkdl) (2 pages)

#### 4.1 Architecture Overview
```
BUILD TIME:                          RUNTIME:
MLIR linalg.matmul                   1. discover_devices() → [A100, MI300, CPU]
    → gpu.launch_func + targets      2. load routing table from binary bundle
    → gpu-module-to-binary           3. match kernel.contract vs device.capabilities
    → {nvptx, amdgcn, x86}          4. cost_model_rank() → A100 wins for GEMM
    → bundled binary                 5. dispatch(kernel, device, args)
```

#### 4.2 Kernel Contracts
- Each compiled kernel variant carries a contract: required capabilities, estimated FLOPS, memory footprint
- Contracts are embedded as metadata in the binary bundle

#### 4.3 Device Capability Model
- Unified capability representation across CUDA/HIP/Vulkan/CPU
- Feature matrix: compute units, memory bandwidth, tensor core support, warp size
- Discovery via vendor APIs → normalized capability struct

#### 4.4 Cost Model
- Roofline-based estimation: compute-bound → prefer high-FLOPS device; memory-bound → prefer high-bandwidth
- Static cost hints from compiler + runtime calibration
- Fallback chain: best match → compatible match → CPU

#### 4.5 Dispatch Mechanism
- Overhead target: <10ns per dispatch (vs 5-20us kernel launch)
- Routing table: precomputed at first invocation, cached
- dlopen/dlsym for native binaries, Vulkan pipeline for SPIR-V fallback

**Literature support:** `papers-hardware-introspection.md`, `new/cost-models-kernel-dispatch.md`, `new/multi-versioned-kernels.md`

### 5. Evaluation Strategy (1 page)

#### 5.1 Micro-benchmarks
- Dispatch overhead measurement
- Kernel selection accuracy vs oracle
- Binary loading latency

#### 5.2 Kernel Benchmarks
- GEMM (compute-bound): compare dispatch-selected vs native
- Element-wise (memory-bound): verify bandwidth-aware routing
- Attention kernel (mixed): test multi-device splitting potential

#### 5.3 End-to-End
- ResNet-50 inference: dispatch across NVIDIA + CPU
- BERT tokenization + inference: heterogeneous pipeline
- Throughput on mixed GPU cluster (if access available)

#### 5.4 Comparison Targets
- Native CUDA (upper bound)
- IREE multi-target (full-stack baseline)
- SYCL/AdaptiveCpp (runtime portability baseline)
- ALPAKA (compile-time portability baseline)

### 6. Discussion (0.5 pages)

- **Limitation: compile-time kernel quality** — dispatch can only select among pre-compiled variants; it cannot optimize a poorly compiled kernel
- **Limitation: cost model accuracy** — simple roofline-based model may not capture all workload characteristics
- **Connection to IREE** — our layer is complementary, not competitive; could integrate as a lightweight alternative to IREE's HAL
- **Connection to PyTorch** — torch.compile backend using MLIR multi-target + our dispatch is a natural extension

### 7. Conclusion (0.25 pages)

- MLIR has the compilation infrastructure; the runtime dispatch gap is documented and confirmed
- A lightweight, composable dispatch layer fills this gap without requiring full-stack buy-in
- Future: upstream MLIR integration, cost model learning, multi-kernel pipeline scheduling

---

## Key Reviewer Concerns → Paper Responses

| Reviewer | Concern | Response Section |
|----------|---------|------------------|
| 91A | No concrete contribution | §4 (libkdl design) + §5 (evaluation) |
| 91B | ML kernels are static | §2.3 (empirical rebuttal) |
| 91B | Connect to PyTorch/TF | §2.3 (PyTorch dispatcher), §6 (torch.compile integration) |
| 91C | Survey or proposal? | §1 (clear thesis), §4 (concrete design) |
| 91C | Better motivation for non-experts | §1, §2.2 (accessible framing) |
| 91D | Why SYCL specifically? | §3.2 (SYCL is one of many), §4 (vendor-agnostic by design) |
| 91D | Acknowledge IREE SPIR-V | §3.3 (honest IREE analysis), §3.5 (SPIR-V capabilities) |
| 91D | Multi-versioned JIT approach | §4 (this IS multi-versioned dispatch) |

---

## Citation Priority List

### Must-cite (T1)
1. Lattner et al. 2021 — MLIR: Scaling Compiler Infrastructure (CGO)
2. Davis et al. 2025 — Performance Portability Comparison (ICS)
3. Alpay & Heuveline 2023 — AdaptiveCpp SSCP (IWOCL)
4. Zenker et al. 2016 — ALPAKA (arXiv)
5. Trott et al. 2022 — Kokkos 3 (IEEE TPDS)
6. Ivanov et al. 2024 — Retargeting GPU Workloads (CGO)
7. Georgakoudis et al. 2025 — Proteus JIT (CGO)
8. Pennycook et al. 2016 — Performance Portability Metric (PMBS)
9. IREE issues #50, #12230, #15334 (GitHub)
10. Chen et al. 2018 — TVM (OSDI)

### Should-cite (T2)
11. Alpay & Heuveline 2025 — AdaptiveCpp Adaptivity (IWOCL)
12. chipStar 2026 — SPIR-V portability (IJHPCA)
13. IRIS 2024 — Task-based dispatch (IEEE TPDS)
14. Helix 2025 — Mixed GPU serving (ASPLOS)
15. Yang 2020 — PyTorch Dispatcher (blog, but seminal)
16. SYCL-MLIR 2024 — MLIR compilation for SYCL (CGO)
17. LLVM RFC — SPIR-V as vendor-agnostic GPU IR

---

## Open Questions for Paper

1. **Scope of "kernel"**: Do we dispatch individual GEMM calls or whole fused subgraphs?
   - IREE dispatches fused regions. ONNX RT dispatches subgraphs. cuBLAS dispatches individual kernels.
   - Proposal: kernel-level dispatch with support for graph-level hints.

2. **Cost model complexity**: How sophisticated must the cost model be for useful dispatch?
   - Simple: device type matching (NVIDIA kernel → NVIDIA device)
   - Medium: roofline-based (compute-bound → high-FLOPS device)
   - Complex: learned model (cuBLAS approach, 93% optimal)
   - Proposal: start with roofline, demonstrate value, discuss learned extensions.

3. **Binary format**: What format for the multi-target bundle?
   - CUDA fatbin (NVIDIA-only precedent)
   - IREE FlatBuffer (IREE-specific)
   - ELF sections (Linux-standard, our approach)
   - Proposal: ELF-based bundle with capability metadata sections.

4. **Integration path**: How does this reach real users?
   - Upstream MLIR pass + runtime library
   - torch.compile backend
   - ONNX Runtime execution provider
   - Proposal: standalone library first, upstream integration as future work.
