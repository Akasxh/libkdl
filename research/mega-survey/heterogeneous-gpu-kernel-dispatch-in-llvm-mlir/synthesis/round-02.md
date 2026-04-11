# Synthesis Report — Round 02: Waves 03–04 (New Files)
## Heterogeneous GPU Kernel Dispatch in LLVM/MLIR

**Date:** 2026-04-06
**Scope:** 10 new wave files from waves 03–04, incremental to round-01
**Cumulative:** 45 wave files, ~430 individual sources
**Purpose:** Inform libkdl (Kernel Dynamic Linker) poster for LLVM Developers' Meeting Dublin 2026

---

## 1. New Source Count (Incremental from Round 01)

Round-01 covered 35 wave files (~310 sources). This round adds 10 new wave files:

| Wave File | Sources | Key Topics |
|-----------|---------|------------|
| wave-03-hetgpu-hetir.md | 8 | HetGPU virtual ISA, ZLUDA, GPU Ocelot, Vortex warp emulation |
| wave-03-dynamic-kernel-substitution.md | 9 | FlashInfer-Bench, NVBit, gpu_ext, Luthier, ROCR InterceptQueue |
| wave-03-cost-model-selection.md | 8 | Seer, SparseX CGO 2026, tritonBLAS, NeuSight, ML-MLIR cost model |
| wave-03-alpaka-portability.md | 10 | CMS Alpaka production (CHEP 2023-2025), framework comparison |
| wave-03-onnxrt-execution-providers.md | 10 | ORT EP dispatch, TensorRT RTX, MIGraphX, ONNX-MLIR |
| wave-04-liboffload-multiversion.md | 9 | liboffload ol* API evolution, OffloadBinary multi-image, PR #186088 |
| wave-04-kernel-caching.md | 18 | Cross-framework kernel cache comparison, PT2 843s cold start |
| wave-04-tvm-device-placement.md | 12 | PlanDevices algorithm, Relax VDevice, MATCH, BYOC |
| wave-04-executorch-edge-dispatch.md | 10 | ExecuTorch backend delegation, nested dispatch, edge heterogeneity |
| wave-04-unified-runtime-vs-liboffload.md | 10 | UR vs liboffload, bridging adapter, SYCL upstreaming |

**New sources this round:** ~104
**Cumulative sources:** ~414 unique sources across 45 wave files

---

## 2. Updated Emerging Themes

### Theme 8 (NEW): Binary Interposition Has a Realistic Overhead Floor

Dynamic kernel substitution is feasible at production-acceptable overhead. The design space spans three tiers [wave-03-dynamic-kernel-substitution.md#1, #3, #4]:

| Tier | Mechanism | Overhead | Platform |
|------|-----------|----------|----------|
| Application-level O(1) | Hash table lookup (FlashInfer-Bench) | 1–2 µs, <0.8% e2e | Any |
| PTX trampoline (gpu_ext) | eBPF-style binary rewrite | 3–14% | NVIDIA |
| SASS rewrite (NVBit) | Full instruction instrumentation | 85–93% | NVIDIA |
| HSA InterceptQueue | AQL packet-level substitution | Not published (est. <1 µs) | AMD |
| ISA-level (Luthier) | GCN code object instrumentation | Behind paywall | AMD |

**Critical design recommendation for libkdl:** The NVIDIA path should target `cuLaunchKernel` (driver API) via host-side O(1) dispatch, not binary instrumentation. The AMD path has a first-class supported mechanism: ROCR `InterceptQueue` provides packet-level kernel substitution without binary rewriting [wave-03-dynamic-kernel-substitution.md#6]. This is architecturally cleaner than any CUDA-side approach.

### Theme 9 (NEW): liboffload Is Actively Building Multi-Image Infrastructure — But Explicitly Deferring Selection Policy

As of March 2026, liboffload gained OffloadBinary multi-image container support [wave-04-liboffload-multiversion.md#5, #6]. The `parseOffloadBinary` function already iterates all inner images and checks `isMetadataCompatible()` per-plugin — but loads the **first compatible image** and explicitly defers selection policy: "should we want it, it's better in a follow-up PR" [wave-04-liboffload-multiversion.md#6]. The `ol_kernel_handle_t` → `ol_symbol_handle_t` rename (July 2025, PR #147943) generalizes the symbol abstraction toward `dlsym`-like semantics [wave-04-liboffload-multiversion.md#2].

**Integration point confirmed:** The exact loop where a libkdl-style ranking callback would slot in exists in live code. The `OffloadBinMetadataTy` struct carries `Triple`, `Arch`, `ImageKind`, and arbitrary `StringData` — sufficient metadata for capability matching [wave-04-liboffload-multiversion.md#6].

### Theme 10 (NEW): Kernel Caching Is a Universal Problem; Cross-Vendor Caching Does Not Exist

Every major GPU framework independently invented a persistent kernel binary cache — CUDA ComputeCache, Triton `~/.triton/cache/`, AdaptiveCpp `~/.acpp/apps/`, Intel NEO, chipStar, AOTriton AKS2, MIOpen SQLite, CUTLASS — yet none provides cross-vendor caching [wave-04-kernel-caching.md]. The most striking quantitative finding: Meta's PT2 (torch.compile) shows Triton JIT compilation takes **843 seconds** out of 1,825s total cold start for a large model [wave-04-kernel-caching.md#S17]. A pre-compiled multi-vendor binary cache (as in libkdl's MTB format) eliminates this JIT entirely. Universal design gaps across all frameworks: unbounded cache growth (only CUDA and Intel NEO implement LRU), no cross-process locking (only CUTLASS uses atomic writes), and security is an afterthought (Red Hat's Feb 2026 signed-binary proposal is the first) [wave-04-kernel-caching.md].

### Theme 11 (NEW): The Nested Dispatch Hierarchy Is Pervasive in Edge/Mobile

ExecuTorch reveals a consistent three-level dispatch hierarchy pattern: framework partitioner (AOT) → vendor SDK runtime → hardware unit [wave-04-executorch-edge-dispatch.md#3, #9]. Three production backends independently implement this: CoreML (ANE/GPU/CPU), OpenVINO (iGPU/NPU/CPU), and QNN (Hexagon/Adreno/Kryo). The framework dispatch is AOT and static; the vendor-internal dispatch is runtime and opaque. libkdl occupies the middle ground — runtime dispatch that is transparent rather than opaque [wave-04-executorch-edge-dispatch.md#8].

### Theme 12 (NEW): TVM's Device Placement Has a Fundamental Algorithmic Limitation

TVM's `PlanDevices` pass uses union-find constraint grouping with **no conflict resolution** — it crashes when a single tensor is consumed by operators on different devices (common in residual networks) [wave-04-tvm-device-placement.md#S5]. The Relax redesign (VDevice as first-class type field) fixes the structural fragility but preserves the static dispatch constraint [wave-04-tvm-device-placement.md#S7]. Critically, a community member explicitly proposed `PrimExpr`-typed `vdevice_id` for runtime device selection, and the RFC authors deferred it [wave-04-tvm-device-placement.md#S7]. This is a quotable statement confirming the gap libkdl fills.

### Theme 13 (NEW): Alpaka Production Data Quantifies the Build-Matrix and Tuning Gaps

CMS Run 3 production data (450 GPUs, 30,000+ CPU cores, ~40% HLT runtime on GPU) provides the most rigorous quantification of portability layer limitations [wave-03-alpaka-portability.md#S4]:
- **30–40% default launch parameter penalty** — Alpaka and Kokkos both suffer this unless manually tuned per-device [wave-03-alpaka-portability.md#S3, #S5]
- **11.6x throughput collapse** without a caching memory allocator [wave-03-alpaka-portability.md#S2]
- **Event-level (not kernel-level) dispatch granularity** — CMS SwitchProducer routes entire nodes to one backend per session [wave-03-alpaka-portability.md#S8, #S9]
- **Build matrix scales linearly with vendor count** — separate builds for CUDA, HIP, and CPU [wave-03-alpaka-portability.md#S2]

libkdl's pre-compiled per-device variants directly solve both the tuning gap (optimal params baked into each variant) and the build matrix (single artifact, runtime selection) [wave-03-alpaka-portability.md].

---

## 3. Top 10 NEW Findings (Not in Round 01)

### Finding 1: liboffload PR #186088 — Multi-Image Loading With Explicit Policy Deferral
**Source:** [wave-04-liboffload-multiversion.md#6]
The open PR (March 2026) by Alex Duran generalizes OffloadBinary multi-image support to all plugins (CUDA, AMDGPU, host). The implementation parses all embedded inner images but loads the **first compatible one**. The PR author explicitly states: "For now only the first compatible image in the binary is loaded... it's better in a follow-up PR." The `isMetadataCompatible()` virtual method on `GenericPluginTy` [wave-04-liboffload-multiversion.md#7] is the exact extensibility hook where a `rankImage()` callback would enable libkdl-style selection policy.
**Significance:** This is the clearest available evidence that the LLVM community is building the infrastructure but explicitly leaving the policy layer for someone else. libkdl *is* that policy layer.

### Finding 2: Meta PT2 Cold Start — 843 Seconds of Triton JIT Compilation
**Source:** [wave-04-kernel-caching.md#S17]
Meta's internal workload profiling shows `async_compile.wait` (pure Triton compilation) takes 843.95 seconds — 46.2% of total 1,825s cold start for a large foundation model. Warm cache reduces this to <10ms per kernel. Pre-compiled multi-vendor binaries (libkdl MTB) eliminate this JIT entirely for deployment scenarios.
**Significance:** Strongest quantitative argument for AOT pre-compilation over JIT. The 843s number makes the "why not just JIT?" objection concrete and answerable.

### Finding 3: ROCR InterceptQueue — First-Class Kernel Substitution on AMD
**Source:** [wave-03-dynamic-kernel-substitution.md#6]
AMD's ROCR runtime provides `InterceptQueue`, a supported API primitive for packet-level kernel dispatch interception. `AddInterceptor()` registers a callback; `HandleAsyncDoorbell()` fires before packets reach hardware. This is not a hack — it is a documented production API used by rocprofv2. No application modification required.
**Significance:** Provides a clean, vendor-supported AMD-side dispatch interposition mechanism. Architecturally superior to NVIDIA's LD_PRELOAD-based approach. libkdl's AMD backend should use InterceptQueue rather than binary instrumentation.

### Finding 4: gpu_ext eBPF Trampolines — 3–14% Overhead vs NVBit's 85–93%
**Source:** [wave-03-dynamic-kernel-substitution.md#3]
gpu_ext (arXiv:2512.12615, Dec 2025) uses SIMT-aware warp-leader eBPF trampolines for GPU kernel interposition, achieving 3–14% overhead versus NVBit's 85–93% for equivalent instrumentation. The paper also demonstrates 4.8x throughput improvement and 2x tail latency reduction across LLM inference, GNN training, and vector search.
**Significance:** Establishes the realistic overhead floor for NVIDIA-side transparent kernel substitution. The warp-leader execution pattern (compute per-lane, aggregate via warp leader) is directly applicable to libkdl's dispatch tracing design.

### Finding 5: tritonBLAS Validates max(T_compute, T_memory) Over Weighted Sum
**Source:** [wave-03-cost-model-selection.md#S3]
tritonBLAS (arXiv:2512.04226) achieves 94.7% of exhaustive-tuning performance using an analytical roofline model: `argmin_config max(T_compute, T_memory)` subject to occupancy constraints. Selection runs in 50–80 µs per kernel. Supports both NVIDIA (A100, H100) and AMD ROCm (MI250X, at 91% accuracy).
**Significance:** Directly validates the correction needed in libkdl's cost model: replace the current weighted sum (`w.compute * T_compute + w.memory * T_memory`) with `fmax(T_compute, T_memory)`. The 94.7% quality bar is the target libkdl must match for GEMM variant selection.

### Finding 6: SparseX (CGO 2026) — Runtime Library Selection Accepted at Top Venue
**Source:** [wave-03-cost-model-selection.md#S2]
SparseX, accepted at CGO 2026, solves automatic on-the-fly selection among competing GPU libraries (cuSparse, Sputnik, CLASP, Jigsaw) and processor types (CUDA cores, Tensor Cores, Sparse Tensor Cores) for SpMM. Achieves up to 95.34x over cuSparse via lightweight classifier.
**Significance:** CGO 2026 acceptance validates that "runtime kernel/library selection via predictive model" is a recognized research contribution at a top venue. libkdl's cross-vendor dimension is a strict superset of SparseX's intra-vendor dimension.

### Finding 7: CMS Alpaka — 30–40% Penalty From Default Launch Parameters
**Source:** [wave-03-alpaka-portability.md#S3, #S5]
Both CHEP 2024 (Kortelainen et al.) and arXiv:2601.17526 (Jan 2026) independently confirm that Alpaka and Kokkos suffer 30–40% performance degradation when portability-layer-selected launch parameters are used instead of manually tuned per-device values. On AMD MI100, Alpaka with tuning actually **exceeds native HIP by 23%** due to memory pool effects.
**Significance:** Directly motivates libkdl's per-device pre-compiled variants. Each variant carries its optimal block size and register count; no runtime heuristic needed. The 30–40% penalty is the quantified cost of not having per-device tuning — the gap libkdl eliminates.

### Finding 8: UR vs liboffload — Not Converging, Bridging
**Source:** [wave-04-unified-runtime-vs-liboffload.md]
Unified Runtime (Intel/oneAPI) and liboffload (LLVM mainline, AMD-led) are developing a **bridging adapter** relationship, not merging. UR sits atop liboffload as an adapter consumer: `libsycl → UR → liboffload plugin → native backend`. PR #118503 introduced the reference UR adapter for offload. libsycl was upstreamed into LLVM main in August 2025 targeting LLVM 22.
**Significance:** libkdl should position above both. The dual-layer architecture validates libkdl's approach: define its own minimal dispatch interface, backed by liboffload for OpenMP/CUDA/HIP and UR for SYCL/Level Zero.

### Finding 9: NeuSight — Cross-Architecture GPU Prediction at 2.3% Error
**Source:** [wave-03-cost-model-selection.md#S4]
NeuSight (ASPLOS 2025) decomposes kernel prediction into tile-granularity sub-predictions bounded by per-GPU roofline parameters, enabling transfer to unseen architectures with <9% mean error. GPT-3 latency prediction on unseen H100: 2.3% error versus 30.8% for prior work. Five specialized MLPs with <500 parameters each — small enough to embed in libkdl (~50 KB).
**Significance:** Provides the theoretical basis for libkdl's potential hybrid cost model: analytical roofline for first dispatch, NeuSight-style ML for calibrated prediction on unseen hardware. Key gap: NeuSight is NVIDIA-only in training data; AMD transfer accuracy is not validated.

### Finding 10: TVM Relax — Runtime VDevice Selection Explicitly Deferred
**Source:** [wave-04-tvm-device-placement.md#S7]
In the Relax heterogeneous execution RFC (2023), community member Lunderberg explicitly asked: "Can `vdevice_id` be a `PrimExpr` to support runtime device selection?" The RFC authors responded that dynamic selection was out of scope. The tracking issue closed (Dec 2023) with this gap intact. Post-OctoAI acquisition, Phase 2 goals (cost-model-driven automated placement) were never implemented.
**Significance:** A quotable statement from TVM's own community identifying the exact problem libkdl solves. The deferred `PrimExpr` vdevice_id is the runtime dispatch primitive TVM chose not to build.

---

## 4. Updated Research Direction Ranking

### Direction 1: libkdl as "ld.so for GPU Kernels" — The Core Contribution

| Criterion | Round-01 | Round-02 | Delta | New Evidence |
|-----------|----------|----------|-------|-------------|
| Novelty | 9 | **9** | = | PR #186088 confirms "first compatible wins" is the current liboffload design; no selection policy exists [wave-04-liboffload-multiversion.md#6]. SparseX (CGO 2026) validates the contribution class but only intra-vendor [wave-03-cost-model-selection.md#S2]. |
| Feasibility | 9 | **9** | = | ROCR InterceptQueue confirms AMD-side mechanism is production-ready [wave-03-dynamic-kernel-substitution.md#6]. liboffload `olGetSymbol` API confirmed stable enough for integration [wave-04-liboffload-multiversion.md#2]. |
| Evidence | 9 | **10** | +1 | PT2 843s JIT quantifies the problem [wave-04-kernel-caching.md#S17]. CMS 30–40% default-param penalty quantifies per-device tuning need [wave-03-alpaka-portability.md#S3]. TVM Relax explicitly deferred runtime dispatch [wave-04-tvm-device-placement.md#S7]. |
| Impact | 9 | **9** | = | ExecuTorch's 14 backends on separate `.pte` files confirms the multi-binary distribution problem [wave-04-executorch-edge-dispatch.md#1]. |
| **Composite** | **9.0** | **9.25** | +0.25 | |

### Direction 2: Roofline-Based Cross-Vendor Cost Model

| Criterion | Round-01 | Round-02 | Delta | New Evidence |
|-----------|----------|----------|-------|-------------|
| Novelty | 8 | **8** | = | No cross-vendor cost model found in new waves either. |
| Feasibility | 8 | **9** | +1 | tritonBLAS validates `max(T_compute, T_memory)` at 94.7% quality [wave-03-cost-model-selection.md#S3]. Seer validates compiled decision trees at sub-µs overhead [wave-03-cost-model-selection.md#S1]. |
| Evidence | 7 | **8** | +1 | NeuSight proves tile-decomposed ML transfers to unseen GPUs at <9% error [wave-03-cost-model-selection.md#S4]. ML-MLIR cost model enables build-time contract population [wave-03-cost-model-selection.md#S5]. |
| Impact | 7 | **7** | = | Remains secondary to Direction 1. |
| **Composite** | **7.5** | **8.0** | +0.5 | |

### Direction 3: Empirical Dispatch Overhead Comparison

| Criterion | Round-01 | Round-02 | Delta | New Evidence |
|-----------|----------|----------|-------|-------------|
| Novelty | 6 | **7** | +1 | gpu_ext establishes 3–14% NVIDIA interposition floor [wave-03-dynamic-kernel-substitution.md#3]. ROCR InterceptQueue overhead is unpublished — measurement opportunity [wave-03-dynamic-kernel-substitution.md#6]. |
| Feasibility | 9 | **9** | = | |
| Evidence | 8 | **9** | +1 | Comprehensive overhead table now available from FlashInfer-Bench to NVBit [wave-03-dynamic-kernel-substitution.md]. Alpaka confirms no additional AMD launch latency vs Kokkos's 70 µs [wave-03-alpaka-portability.md#S5]. |
| Impact | 7 | **7** | = | |
| **Composite** | **7.5** | **8.0** | +0.5 | |

### Direction 4: Multi-Variant Kernel Bundle Format (MTB)

| Criterion | Round-01 | Round-02 | Delta | New Evidence |
|-----------|----------|----------|-------|-------------|
| Novelty | 8 | **7** | -1 | liboffload OffloadBinary format now carries multi-image containers with metadata [wave-04-liboffload-multiversion.md#5]. MTB is less novel because the infrastructure is converging from below. |
| Feasibility | 7 | **8** | +1 | `OffloadBinMetadataTy` already carries Triple/Arch/StringData — MTB metadata could align with this [wave-04-liboffload-multiversion.md#5]. |
| Evidence | 7 | **8** | +1 | Cross-framework cache comparison shows no existing cross-vendor archive [wave-04-kernel-caching.md]. AOTriton AKS2 + LZMA confirmed as AMD-only reference [wave-04-kernel-caching.md#S11]. |
| Impact | 6 | **7** | +1 | ExecuTorch's separate `.pte` per target confirms multi-file distribution is a real problem [wave-04-executorch-edge-dispatch.md#1]. |
| **Composite** | **7.0** | **7.5** | +0.5 | |

### Direction 5: Integration with LLVM liboffload as a Policy Layer

| Criterion | Round-01 | Round-02 | Delta | New Evidence |
|-----------|----------|----------|-------|-------------|
| Novelty | 7 | **8** | +1 | PR #186088 author explicitly defers selection policy to follow-up work — confirms the policy gap is recognized [wave-04-liboffload-multiversion.md#6]. |
| Feasibility | 6 | **7** | +1 | Complete API surface now documented [wave-04-liboffload-multiversion.md#3]. `olGetSymbol` rename and extensible `olLaunchKernel` properties [wave-04-liboffload-multiversion.md#2, #8] confirm the API is maturing. |
| Evidence | 8 | **9** | +1 | UR-on-liboffload bridging pattern validates the "sit above liboffload" architecture [wave-04-unified-runtime-vs-liboffload.md]. |
| Impact | 8 | **9** | +1 | SYCL upstreaming (libsycl → UR → liboffload) means libkdl above liboffload inherits the entire SYCL ecosystem [wave-04-unified-runtime-vs-liboffload.md]. |
| **Composite** | **7.3** | **8.25** | +0.95 | |

### Updated Ranking (Round 02)

| Rank | Direction | R1 Score | R2 Score | Movement |
|------|-----------|----------|----------|----------|
| 1 | libkdl as ld.so for GPU kernels | 9.0 | **9.25** | = |
| 2 | liboffload policy layer integration | 7.3 | **8.25** | +2 ↑ |
| 3 | Roofline cross-vendor cost model | 7.5 | **8.0** | = |
| 4 | Empirical dispatch overhead comparison | 7.5 | **8.0** | = |
| 5 | Multi-variant kernel bundle format (MTB) | 7.0 | **7.5** | -1 ↓ |

**Key shift:** Direction 5 (liboffload integration) jumped from rank 5 to rank 2. The new evidence from PR #186088, the UR bridging pattern, and SYCL upstreaming dramatically increase the strategic value of positioning libkdl as liboffload's policy layer. This should be the **secondary narrative** on the poster after the core ld.so contribution.

---

## 5. Competitive Landscape Update

### libkdl vs. HetGPU

| Dimension | HetGPU | libkdl |
|-----------|--------|--------|
| Approach | IR-level virtualization (hetIR → JIT per backend) | Pre-compiled native variants → runtime selection |
| JIT overhead | 10–200 ms cold, 3–8% warm [wave-03-hetgpu-hetir.md#1] | Zero (pre-compiled) |
| Toolchain | Requires recompilation with hetGPU toolchain [wave-03-hetgpu-hetir.md#1] | Consumes existing PTX/HSACO/SPIR-V binaries |
| Migration | Live cross-vendor migration (2.2s for 2GB) [wave-03-hetgpu-hetir.md#1] | Not claimed |
| Non-SIMT HW | Tenstorrent support (unvalidated perf) | Not targeted |
| Warp emulation cost | 30% baseline on non-SIMT hardware [wave-03-hetgpu-hetir.md#5] | N/A (native binaries) |
| Binary compatibility | "Source compatibility" — must use hetGPU compiler [wave-03-hetgpu-hetir.md] | True binary compatibility — consumes existing .cubin/.hsaco |
| **Positioning:** | HetGPU solves a different problem (portable IR + migration). libkdl differentiates on zero JIT overhead, no new toolchain, and true binary-level dispatch. |

### libkdl vs. liboffload

| Dimension | liboffload | libkdl |
|-----------|-----------|--------|
| Role | Mechanism (load blob, look up symbol, launch) | Policy (select among variants, score, dispatch) |
| Multi-version | "First compatible wins" [wave-04-liboffload-multiversion.md#6] | Capability-scored ranking |
| API stability | Pre-1.0, subject to rename (olGetKernel → olGetSymbol) [wave-04-liboffload-multiversion.md#2] | Prototype, not yet upstreamed |
| Format support | OffloadBinary (multi-image, metadata) [wave-04-liboffload-multiversion.md#5] | MTB (multi-vendor, contracts, Bloom filters) |
| **Positioning:** | Complementary, not competing. libkdl is the policy layer liboffload's `parseOffloadBinary` loop is waiting for. |

### libkdl vs. Unified Runtime (UR)

| Dimension | UR | libkdl |
|-----------|-----|--------|
| Owner | Intel/oneAPI (moving to LLVM) | Independent |
| Primary consumer | SYCL/DPC++ | Language-agnostic |
| Multi-version | Per-adapter, no cross-adapter selection | Cross-vendor selection |
| **Positioning:** | UR serves SYCL; libkdl serves the multi-vendor dispatch gap above both UR and liboffload [wave-04-unified-runtime-vs-liboffload.md]. |

### libkdl vs. IREE HAL

| Dimension | IREE HAL | libkdl |
|-----------|----------|--------|
| Variant selection | Static boolean conditions at module load [wave-01-iree-hal.md#7] | Capability-scored runtime ranking |
| Caching | Process-lifetime only (no persistence) [wave-04-kernel-caching.md#S13] | Persistent cross-vendor cache |
| Cost model | None in dispatch path [wave-03-cost-model-selection.md#S7] | Roofline-based cross-vendor scoring |
| **Positioning:** | IREE needs a persistent cache and a runtime cost model. libkdl could integrate as a dispatch policy layer inside IREE HAL. |

### libkdl vs. AdaptiveCpp SSCP

| Dimension | AdaptiveCpp | libkdl |
|-----------|-------------|--------|
| Single binary | Yes (embedded LLVM IR, JIT to native) | Yes (pre-compiled native variants) |
| Cold start | JIT latency on first launch [wave-04-kernel-caching.md#S6] | Zero (AOT) |
| Runtime specialization | +30% over CUDA, +44% over HIP via JIT optimization [wave-03-adaptivecpp (R1)] | No runtime specialization (static variants) |
| Adaptivity database | SQLite appdb, improves over multiple runs [wave-04-kernel-caching.md#S7] | Calibration pass at first load |
| **Positioning:** | Complementary approaches. AdaptiveCpp optimizes via JIT specialization; libkdl optimizes via pre-compiled variant selection. A hybrid approach (libkdl with LLVM IR fallback variants for JIT) combines both. |

### libkdl vs. SparseX (CGO 2026)

| Dimension | SparseX | libkdl |
|-----------|---------|--------|
| Selection scope | Multiple libraries on one GPU type | Multiple vendors across GPU types |
| Selection model | Lightweight classifier | Roofline cost model + Bloom filter |
| Operator focus | SpMM only | General-purpose (GEMM, attention, normalization, SpMV) |
| **Positioning:** | SparseX validates the contribution class at CGO 2026. libkdl's cross-vendor dimension is a strict superset [wave-03-cost-model-selection.md#S2]. |

---

## 6. Poster Narrative Recommendation

### Recommended Story Arc

**Title:** libkdl: ld.so for GPU Kernels — Cross-Vendor Dispatch from Pre-Compiled Native Variants

**Act 1 — The Problem (left panel)**

The LLVM ecosystem has converged on multi-vendor kernel compilation infrastructure (`gpu-module-to-binary` produces NVVM + ROCDL + XeVM objects simultaneously), and every major runtime provides a mechanism for loading GPU kernels (IREE HAL, liboffload `olCreateProgram`, PjRt, ORT Execution Providers). But no existing system selects among pre-compiled vendor-native kernel variants at runtime based on detected hardware capability.

Evidence to cite:
- LLVM Issue #75356 explicitly requests `dlsym()`-for-GPUs [wave-05-ld-so-analogy.md#1]
- liboffload PR #186088 loads "first compatible" image — no ranking [wave-04-liboffload-multiversion.md#6]
- TVM Relax deferred `PrimExpr` vdevice_id for runtime selection [wave-04-tvm-device-placement.md#S7]
- CMS Alpaka requires N builds for N vendors + 30–40% penalty without per-device tuning [wave-03-alpaka-portability.md#S3]

**Act 2 — The Solution (center panel)**

libkdl is `ld.so` for GPU kernels: a standalone C library (~5100 LOC) that loads multi-vendor kernel bundles, queries device capabilities at load time, and dispatches to the optimal pre-compiled variant per kernel per device.

Design diagram: MTB bundle → libkdl loader → capability fingerprint → roofline cost model → vendor driver (CUDA/HIP/CPU)

Key design decisions informed by this survey:
- O(1) dispatch adds <0.8% overhead [wave-03-dynamic-kernel-substitution.md#1]
- `max(T_compute, T_memory)` roofline validated at 94.7% by tritonBLAS [wave-03-cost-model-selection.md#S3]
- AMD path via ROCR InterceptQueue [wave-03-dynamic-kernel-substitution.md#6]
- Positioned as policy layer above liboffload mechanism [wave-04-liboffload-multiversion.md#6]

**Act 3 — Evidence (right panel)**

Benchmark results from prototype on GTX 1650 + CPU:
- Dispatch overhead comparison (libkdl vs direct CUDA vs liboffload)
- Variant selection quality (roofline vs exhaustive autotuning)
- Cold-start elimination: 0 ms (libkdl) vs 843s (Triton JIT) [wave-04-kernel-caching.md#S17]

Competitive positioning table (6 systems):
| System | Runtime dispatch | Cross-vendor | Pre-compiled | Per-kernel |
|--------|-----------------|-------------|-------------|-----------|
| liboffload | Load-time | Multi-image | Yes | Yes (no selection) |
| IREE HAL | Static boolean | Yes | Yes | No cost model |
| Alpaka/CMS | Event-level | Build matrix | Yes | No |
| AdaptiveCpp | JIT | Yes | No (JIT) | Yes |
| HetGPU | JIT | Yes | No (JIT) | Yes |
| **libkdl** | **Load-time** | **Yes** | **Yes** | **Yes (roofline)** |

### Key Positioning Shift from Round 01

Round 02 evidence strengthens the **liboffload integration** narrative. The poster should explicitly frame libkdl as the policy layer for liboffload's multi-image infrastructure:

> "liboffload's `parseOffloadBinary` already iterates all vendor-specific images in a multi-target container. It loads the first compatible one. libkdl provides the missing `rankImage()` callback: score each compatible variant using a hardware-calibrated roofline model, select the highest-ranked one, and cache the decision for subsequent dispatches."

This framing:
1. Positions libkdl as a natural LLVM upstream contribution (not a competing project)
2. Connects to live code (PR #186088, merged #185404)
3. Uses the community's own vocabulary ("follow-up PR" for selection policy)
4. Is implementable: Option A (sit above liboffload) requires no upstream changes

---

## 7. Remaining Gaps — Targets for Waves 5–6

### Gap 1: ROCR InterceptQueue Overhead Measurement
No published benchmark of InterceptQueue pass-through latency. Estimated <1 µs but unconfirmed [wave-03-dynamic-kernel-substitution.md#6]. A micro-benchmark on MI100/MI250X would fill this gap and provide the AMD-side overhead number for the poster.

### Gap 2: OffloadBinary Metadata Schema Compatibility
Can libkdl's MTB capability contracts map onto `OffloadBinMetadataTy.StringData`? The metadata struct carries arbitrary key-value pairs [wave-04-liboffload-multiversion.md#5, #6]. Wave 5–6 should analyze the C++ header (`llvm/include/llvm/Object/OffloadBinary.h`) to determine schema compatibility.

### Gap 3: liboffload `olCreateProgram` Latency vs Direct Driver API
What is the overhead of the `ol*` indirection layer? Compare `olCreateProgram` + `olGetSymbol` + `olLaunchKernel` against direct `cuModuleLoadData` + `cuModuleGetFunction` + `cuLaunchKernel`. This answers the "why not just use liboffload directly?" question.

### Gap 4: Cross-Vendor NeuSight Validation
NeuSight's tile-decomposed ML prediction achieves 2.3% error on unseen NVIDIA GPUs but has never been validated on AMD [wave-03-cost-model-selection.md#S4]. If libkdl claims a NeuSight-style hybrid cost model, AMD transfer accuracy must be characterized or acknowledged as a limitation.

### Gap 5: ExecuTorch Recipe API (RFC #13732) Final Shape
The `combine()` pattern for priority-ordered multi-backend fallback [wave-04-executorch-edge-dispatch.md#6] is directly relevant to libkdl's API design. Track whether it lands in v1.2/v1.3 and adopt the API pattern.

### Gap 6: Signed Binary Verification for Kernel Caches
Red Hat's Feb 2026 Triton signed-binary proposal [wave-04-kernel-caching.md#S18] is the first attempt at supply-chain security for GPU kernel caches. libkdl's MTB format should include a signature field. Wave 5–6 should investigate the OCI-layer approach.

### Gap 7: ONNX-MLIR as ORT EP Replacement
ONNX-MLIR (presented at LLVM DevMtg 2025) compiles monolithically and lacks runtime EP dispatch [wave-03-onnxrt-execution-providers.md#10]. libkdl as "what ONNX-MLIR needs to become a proper ORT EP replacement" is a compelling angle for the Dublin audience. Needs deeper investigation.

### Gap 8: Kokkos AMD Launch Latency Root Cause
Kokkos adds 70+ µs overhead on AMD GPUs (kokkos/kokkos#8738, Dec 2025) while Alpaka does not [wave-03-alpaka-portability.md#S5]. The root cause (fence? memory management? HIP API path?) is not documented. If libkdl can avoid this pathology, it is a differentiator.

---

## 8. Contradictions and Tensions (Updated)

### New Tension: AOT Elimination vs JIT Specialization (Sharpened)
Round 01 identified the JIT vs AOT tension. Round 02 sharpens it with numbers:
- **AOT case:** 843s Triton JIT cold start [wave-04-kernel-caching.md#S17]. AOTriton's zero-cold-start AKS2 format [wave-04-kernel-caching.md#S11]. libkdl's pre-compiled MTB.
- **JIT case:** AdaptiveCpp beats CUDA by 30% via runtime specialization [R1 data]. NeuSight-style ML prediction enables calibration on unseen hardware [wave-03-cost-model-selection.md#S4].
- **Resolution:** libkdl should support both: native variants for production dispatch (zero cold start), LLVM IR/SPIR-V variants as fallback for JIT specialization (handles unseen hardware). The MTB format already supports heterogeneous variant types.

### New Tension: liboffload OffloadBinary vs libkdl MTB Format
liboffload's `OffloadBinary` container now carries multi-image metadata [wave-04-liboffload-multiversion.md#5]. Does libkdl need its own MTB format, or should it extend OffloadBinary?
- **For separate MTB:** MTB carries capability contracts, Bloom filters, autotuning DB — richer metadata than OffloadBinary's StringData map.
- **For extending OffloadBinary:** Ecosystem alignment. One fewer format for the community to learn.
- **Resolution:** libkdl should consume OffloadBinary containers (via `llvm::object::OffloadBinary` parser) AND support its own extended MTB format for richer metadata. The loader should auto-detect both.

---

## 9. Limitations of This Synthesis

1. **Luthier overhead numbers are behind the ISPASS 2025 paywall.** AMD binary instrumentation overhead is not precisely quantified [wave-03-dynamic-kernel-substitution.md#4].

2. **NeuSight AMD transfer accuracy is unvalidated.** All claims about cross-vendor ML prediction must be qualified [wave-03-cost-model-selection.md#S4].

3. **PR #186088 (OffloadBinary generalization) is still open.** If it does not merge, the multi-image loading path differs per plugin. The integration point analysis is based on code in review, not landed code.

4. **TVM PlanDevices bug #15019 may have been silently fixed.** Should verify against current TVM main before citing as an open issue in the poster [wave-04-tvm-device-placement.md#S5].

5. **ExecuTorch RFC #13732 (multi-backend recipes) is not yet merged.** The `combine()` API pattern is speculative [wave-04-executorch-edge-dispatch.md#6].

6. **Meta PT2 843s number is for "a large foundation model" — exact model not named.** The number may not generalize to smaller models [wave-04-kernel-caching.md#S17].

---

*Report generated: 2026-04-06*
*New wave files analyzed: 10*
*Cumulative wave files: 45*
*Estimated unique sources: ~414*
*Next step: Waves 05–06 targeting gaps identified in Section 7*
