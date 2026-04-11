# Topic 15 — Cross-Vendor Roofline Annotation Pass for GPU Kernels

**Topic ID:** 15
**Config key:** `cross-vendor-roofline-annotation`
**Persona:** Performance modeling researcher / LLVM pass author
**Date:** 2026-04-07
**Research depth:** Exhaustive — cross-referenced 7 wave files, 2 literature files, kdl.c
primary source (lines 112–122, 1007–1088), KernelInfo upstream source (PR #102944),
OffloadBinary format (D122069, PR #169425), and 6 peer-reviewed papers

---

## Gap

The roofline model — `attainable = min(peak_flops, peak_bw * AI)` — is vendor-neutral by
construction. Its two hardware parameters (`peak_flops`, `peak_bw`) differ across NVIDIA,
AMD, and CPU, but its formula is identical. A kernel's arithmetic intensity (AI = FLOP /
byte) is architecture-independent: it characterises the kernel, not the target. This means
that roofline classification — "this kernel is compute-bound on H100, memory-bound on GTX
1650" — can in principle be determined at compile time for any device for which specs are
known.

No LLVM pass does this today. The specific failure chain is:

1. **Arithmetic intensity is computable at IR level** for kernels with analyzable loop
   structure (GEMM, Conv2D, pointwise ops). LLVM already walks IR to count instructions
   (`InstructionCount` analysis), count memory accesses (`MemorySSA`, `ScalarEvolution`
   trip-count queries), and classify FP operations (`getArithmeticInstrCost` in
   `TargetTransformInfo`). The components exist; no pass assembles them into a
   roofline classification.

2. **KernelInfo (PR #102944, merged January 2025) is the closest upstream pass**, but
   it is intentionally a "bad code pattern detector." It emits `AllocasStaticSizeSum`,
   `FlatAddrspaceAccesses`, and launch-bound hints as optimization remarks. It explicitly
   does NOT compute `flops`, `bytes_total`, or `arithmetic_intensity`. The design intent
   (PR #102944 description, Joel Denny ORNL): "The ultimate goal of these statistics is
   to help identify bad code patterns." Not a roofline input generator.

3. **Hardware specs have no standard LLVM query path for GPU targets.** `TargetTransformInfo`
   exposes `getArithmeticInstrCost` for CPU cost modeling, but GPU-specific throughput
   parameters (`peak_flops`, `peak_bw`) are not part of TTI's interface. They are either
   queried at runtime via vendor APIs (`cudaGetDeviceProperties`, `hipDeviceGetAttribute`)
   or stored in hand-rolled tables (libkdl: `kdl.c:611–868`, where per-vendor device
   enumeration computes `peak_tflops_f32` and `peak_bw_gbps` from SM count × clock rate
   and memory clock × bus width).

4. **The annotation gap blocks cross-vendor dispatch.** The libkdl runtime dispatcher
   (`kdl_estimate_cost_weighted`, `kdl.c:1013–1088`) uses `flops`, `bytes_total`, and
   device `peak_tflops_f32` / `peak_bw_gbps` to rank kernel variants across NVIDIA, AMD,
   and CPU. The `has_compute_profile` gate at `kdl.c:1016` — `if (!c->has_compute_profile)
   return 1e9` — silently degrades to priority-based dispatch when these fields are absent.
   They are absent for every kernel that has not been hand-annotated. The roofline model
   is theoretically vendor-agnostic; the missing step is a compile-time pass that computes
   the kernel-side inputs.

5. **Every existing cost model is intra-vendor.** cuBLAS (NVIDIA-only), MIOpen (AMD-only),
   Ansor/MetaSchedule (per-device XGBoost, retrain per GPU), NeuSight (trained on NVIDIA
   only, AMD transfer not validated), Fasor (Transformer encoder, NVIDIA training data).
   The cross-vendor dispatch regime — selecting between a CUDA variant and a HIP variant
   for the same logical kernel — is unaddressed by all surveyed systems
   (wave-04-cost-models.md synthesis, wave-03-cost-model-selection.md synthesis).

**The concrete gap:** No LLVM pass annotates a GPU kernel function with its roofline
classification (AI, regime, estimated T_compute, T_memory per vendor class) in a form
that survives to binary metadata for runtime dispatch.

---

## Proposal

**Title:** `RooflineAnnotationPass` — A Compile-Time LLVM Pass Annotating GPU Kernels
with Arithmetic Intensity and Roofline Classification

**One-sentence pitch:** An LLVM analysis pass that walks GPU kernel IR to compute
arithmetic intensity (FP ops / memory bytes), compares it against a vendor-spec table
embedded in the compiler, classifies each kernel as compute-bound, memory-bound, or
ridge-proximate per vendor, and emits the result as `OffloadBinary` property strings
consumed by a cross-vendor runtime dispatcher.

### What data the roofline model needs

| Input | Source | Notes |
|-------|--------|-------|
| `flops` (per invocation) | Static instruction count × trip count (SCEV) | Exact for GEMM/Conv2D; approximate for irregular |
| `bytes_transferred` | `memref.load`/`memref.store` × element width × SCEV trip count | Cache-miss-rate-corrected version requires PGO |
| `arithmetic_intensity` | `flops / bytes_transferred` | Kernel property, device-independent |
| `peak_flops` (per device) | Vendor spec table keyed on target triple / arch string | Compile-time constant per target |
| `peak_bw` (per device) | Vendor spec table keyed on target triple / arch string | Compile-time constant per target |
| `ridge_point` | `peak_flops / peak_bw` | Derived; separates compute-bound from memory-bound |

### Pass design: three layers

#### Layer 1 — Static AI computation (`KernelAIAnalysis`)

A new `FunctionPass` (or `FunctionAnalysis` in the new pass manager) running on GPU
device IR, after LTO, after inlining, at the same pipeline position as KernelInfo:

```
for each gpu.func F:
    flop_count = 0
    byte_count  = 0
    for each BasicBlock BB in F:
        for each Instruction I in BB:
            if I is fadd / fmul / fma / fdiv:
                flop_count += trip_count(BB) * flops_per_instruction(I)
            if I is load / store:
                byte_count += trip_count(BB) * element_bytes(I)
    if IndirectCalls(F) == 0 and InlineAssemblyCalls(F) == 0:
        AI = flop_count / byte_count
        emit_remark(F, "arithmetic_intensity", AI)
        emit_remark(F, "flops", flop_count)
        emit_remark(F, "bytes_total", byte_count)
    else:
        emit_remark(F, "ai_status", "undetermined")
```

Trip count is queried from `ScalarEvolutionAnalysis` for affine loops with constant bounds
(the standard LLVM mechanism used by `LoopVectorize`, `LoopUnroll`, and the
`PolynomialMulVectorizer`). For loops with symbolic trip counts, the pass emits a
conservative lower bound (unrolled body count = 1). This is the same limitation as the
TACO 2021 baseline (arXiv:2001.07104): static analysis reaches 9–52% MAPE, with error
concentrated in memory-bound kernels where cache hit rates are input-variable.

**Validity gate:** KernelInfo already computes `IndirectCalls` and `InlineAssemblyCalls`.
The gate `IndirectCalls == 0 && InlineAssemblyCalls == 0` is borrowed from KernelInfo's
remark output — it is the correct filter for when static instruction counting is valid.
The two passes compose naturally: KernelInfo runs, its remarks are consumed by
`KernelAIAnalysis` as a pre-filter, then AI analysis proceeds on valid kernels only.

#### Layer 2 — Per-vendor classification (`RooflineClassifier`)

A compile-time lookup table of hardware parameters keyed on the LLVM target triple's
architecture string (`sm_80`, `sm_90`, `gfx90a`, `gfx940`, `x86_64`, `aarch64`):

```
struct VendorSpec {
    StringRef arch_prefix;    /* e.g. "sm_", "gfx", "x86" */
    double    peak_tflops_f32; /* FP32 TFLOP/s */
    double    peak_bw_gbps;    /* memory bandwidth GB/s */
    double    ridge_point;     /* peak_tflops * 1e3 / peak_bw (FLOP/byte) */
};

static const VendorSpec kVendorTable[] = {
    /* NVIDIA Hopper */
    {"sm_90", 66.9, 3350.0, 19.97},
    /* NVIDIA Ampere A100 */
    {"sm_80", 19.5, 2000.0, 9.75},
    /* NVIDIA Ada (RTX 40xx) */
    {"sm_89", 82.6,  960.0, 86.0},
    /* NVIDIA Turing (GTX 1650 = sm_75) */
    {"sm_75", 2.98,  192.0, 15.5},
    /* AMD CDNA3 (MI300X) */
    {"gfx940", 163.4, 5300.0, 30.8},
    /* AMD CDNA2 (MI250X) */
    {"gfx90a",  47.9, 3276.0, 14.6},
    /* AMD RDNA3 (RX 7900 XTX) */
    {"gfx1100", 61.4,  960.0, 64.0},
    /* x86 (generic) */
    {"x86_64",   3.5,  460.0,  7.6},
    /* ARM (generic) */
    {"aarch64",  1.2,   51.2, 23.4},
};
```

For each kernel function and each entry in the table, the classifier computes:

```
T_compute[arch] = flops / (spec.peak_tflops_f32 * 1e12)
T_memory[arch]  = bytes / (spec.peak_bw_gbps    * 1e9)
regime[arch]    = (AI > spec.ridge_point) ? COMPUTE_BOUND : MEMORY_BOUND
```

This produces a per-kernel, per-architecture classification without requiring hardware
access at compile time. The classification is emitted as remarks and, critically, written
into the `OffloadBinary` string table (Layer 3).

**Hardware specs: where do they come from?**

Three sources compose at different build contexts:

1. **Built-in table** (default): The `kVendorTable` above covers major current
   architectures. It is a compile-time constant — zero runtime overhead, zero dependency
   on vendor SDKs. This is sufficient for the roofline classification use case, where
   we compare AI to a regime boundary, not predict exact latency.

2. **`TargetTransformInfo` extension** (upstream path): `TTI` already has
   `getArithmeticInstrCost()` for operation latency and throughput modeling.
   Adding `getPeakThroughputFLOPS(ScalarTy)` and `getPeakMemoryBandwidthGBPS()` as
   virtual methods to `TargetTransformInfo` would allow each GPU backend
   (`NVPTXTTIImpl`, `AMDGPUTTIImpl`) to return architecture-aware values computed from
   the target's `MCSubtargetInfo` (which already contains processor clock rate, SM/CU
   count, and cache sizes as feature flags). This is the correct upstream integration
   point — it hooks into the existing TTI abstraction used by all LLVM vectorizers and
   cost-model-aware passes.

3. **Runtime calibration feedback** (dynamic path): libkdl's `kdl_calibrate()` pass
   measures actual `peak_tflops_f32` and `peak_bw_gbps` at first load using STREAM-like
   micro-benchmarks. These calibrated values can be written into a per-device JSON file
   that `RooflineClassifier` reads at compile time if available, overriding the built-in
   table with empirically validated numbers for the exact deployed hardware. This closes
   the compile-time / runtime feedback loop.

#### Layer 3 — OffloadBinary annotation embedding

The computed AI, regime, and cost estimates are written into the `OffloadBinary` string
map as named properties (extending the proposal in topic-07-offloadbinary-metadata.md):

```
"kdl.cost.flops"          = "2097152.0"
"kdl.cost.bytes_total"    = "786432.0"
"kdl.cost.ai"             = "2.667"
"kdl.cost.regime.sm_90"   = "compute_bound"
"kdl.cost.regime.gfx940"  = "memory_bound"
"kdl.cost.ridge.sm_90"    = "19.97"
"kdl.cost.ridge.gfx940"   = "30.8"
```

The `OffloadBinary` format (D122069, PR #169425) uses a `StringMap<StringRef>` that
round-trips unknown keys unchanged. Embedding cost annotations adds no binary-format
breakage — old readers ignore unknown keys. The injection point is `gpu-module-to-binary`
(`mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp`), which produces `gpu.object`
attributes with an extensible `properties` DictionaryAttr. Patching it to carry the
`#kdl.cost` attributes from the `gpu.func` into the final binary properties costs
approximately 30 lines of C++ per backend (NVVM, ROCDL).

#### Runtime consumption

The dispatcher reads the embedded properties at `kdl_load_bundle()` time and populates
`kdl_contract` fields (`kdl.c:112–122`):

```c
contract->flops               = atof(props["kdl.cost.flops"]);
contract->bytes_total         = atof(props["kdl.cost.bytes_total"]);
contract->arithmetic_intensity = atof(props["kdl.cost.ai"]);
contract->has_compute_profile  = 1;
```

With `has_compute_profile = 1`, `kdl_estimate_cost_weighted()` (`kdl.c:1016`) switches
from priority-based fallback to the roofline cost model:

```c
double T_compute = contract->flops    / (device->peak_tflops_f32 * 1e12);
double T_memory  = contract->bytes_total / (device->peak_bw_gbps * 1e9);
double cost      = fmax(T_compute, T_memory) / efficiency + overhead;
```

This is the `max(T_compute, T_memory)` formula validated by tritonBLAS (arXiv:2512.04226)
at 94.7% of exhaustive autotuning quality. It replaces the current weighted-sum
(`w.compute * T_compute + w.memory * T_memory`, `kdl.c:1061–1064`) with the
roofline-correct formulation. The regime annotations (`kdl.cost.regime.*`) allow the
dispatcher to pre-classify the dispatch decision before scoring, enabling Bloom-filter-
style early elimination of structurally mismatched variants (Stream-K++ pattern,
arXiv:2408.11417: 95.8% variant elimination at <100 ns).

---

## Evidence

### Roofline model universality

- Williams, Waterman, Patterson. "Roofline: An Insightful Visual Performance Model for
  Multicore Architectures." CACM 52(4), 2009. https://dl.acm.org/doi/10.1145/1498765.1498785
  The formula `min(peak_flops, peak_bw * AI)` is hardware-agnostic: it applies identically
  to NVIDIA, AMD, CPU, and any future architecture.

- Representative ridge points across current hardware (from `literature/cost-models-kernel-
  dispatch.md`, Section 1.2):

  | Device | Peak FP32 TFLOPS | Peak BW (GB/s) | Ridge Point (FLOP/byte) |
  |--------|-----------------|----------------|------------------------|
  | H100 SXM5 | 66.9 | 3,350 | ~20 |
  | A100 SXM4 | 19.5 | 2,000 | ~9.7 |
  | GTX 1650 (test HW) | 2.98 | 192 | ~15.5 |
  | MI300X | 163.4 | 5,300 | ~30.8 |
  | RX 7900 XTX (RDNA3) | 61.4 | 960 | ~64 |
  | x86 EPYC 9654 | ~3.5 | ~460 | ~7.6 |

  The dramatic spread in ridge points (7.6 to 64 FLOP/byte) means a kernel at AI = 10 is
  compute-bound on H100, memory-bound on GTX 1650, and near the ridge on A100. Static
  compile-time classification per-architecture is therefore strictly more informative than
  a single label.

### tritonBLAS: 94.7% quality from three numbers

- arXiv:2512.04226 (December 2025, `wave-03-cost-model-selection.md` S3):
  tritonBLAS applies `argmin_config max(T_compute, T_memory)` to select Triton GEMM tile
  parameters at JIT time. Achieves **94.7%** of exhaustive autotuning quality on 150,000
  GEMM configurations across NVIDIA A100 and H100. AMD MI250X: **91%**.
  The three inputs are: calibrated hardware latency/bandwidth coefficients + kernel
  `flops` + kernel `bytes_total`. All three are computable at the time the pass runs.

### NeuSight: cross-architecture transfer with hardware specs as input

- ASPLOS 2025. arXiv:2407.13853. ACM DL: https://dl.acm.org/doi/10.1145/3669940.3707265
  (`wave-03-cost-model-selection.md` S4):
  Tile-decomposed ML model using hardware features `{peak_flops, peak_bw, L1/L2 cache}`
  achieves **2.3% mean error** on H100 (unseen during training), **8.9% mean error**
  across all evaluated GPUs. The hardware feature vector is exactly the per-arch vendor
  table proposed above. This validates that vendor specs embedded at compile time are
  sufficient inputs for near-optimal cross-architecture prediction.
  **Qualification:** NeuSight is trained on NVIDIA GPUs only; AMD transfer accuracy is
  not validated. The analytical roofline (no ML) is the only vendor-agnostic predictor.

### TACO 2021: static instruction features achieve 9–52% MAPE

- ACM TACO Vol. 18, No. 1, 2021. arXiv:2001.07104 (`wave-03-cost-model-selection.md` S6):
  Random forest on 189 PTX features (FP instruction counts, memory access counts, branch
  counts) predicts GPU kernel execution time. Median MAPE: **8.86–52.0%** across five
  GPU generations. Power prediction: **1.84–2.94% MAPE** (static instruction mix is
  sufficient for power; latency requires cache behavior which is dynamic).
  For the dispatch use case, only regime classification is needed (compute-bound vs.
  memory-bound), not exact latency — so the 52% worst-case time error does not translate
  to 52% dispatch error.

### KernelInfo: upstream precedent, partial coverage

- `llvm/lib/Analysis/KernelInfo.cpp`, merged commit `18f8106f`, 2025-01-29, PR #102944.
  Source: `llvm/include/llvm/Analysis/KernelInfo.h` (passes `TTI.collectKernelLaunchBounds`
  to the `TargetTransformInfo` interface — confirming TTI is the correct upstream hook for
  per-architecture metrics).
  The pass emits `IndirectCalls` and `InlineAssemblyCalls` counts, which serve as
  validity gates for static FLOP counting. The `FlatAddrspaceAccesses` field is a
  portability signal complementary to roofline.
  **Critical gap confirmed:** No `flops`, `bytes_total`, or `arithmetic_intensity` field
  exists in KernelInfo's output (`wave-08-kernel-info-pass.md`, lines 134–138).

### libkdl: prototype validates the runtime consumer

- `experiments/prototype/src/kdl.c`:
  - `kdl_contract` struct (`kdl.c:112–122`): `flops`, `bytes_total`, `arithmetic_intensity`,
    `has_compute_profile` are defined and consumed.
  - `kdl_estimate_cost_weighted()` (`kdl.c:1013–1088`): uses `peak_tflops_f32` and
    `peak_bw_gbps` from `kdl_device_info`. Device parameters are measured at runtime via
    `kdl_device_info` enumeration (`kdl.c:611–868` for NVIDIA/AMD/CPU). The `has_compute_profile`
    gate at `kdl.c:1016` is the exact injection point.
  - Current cost model uses weighted sum (`kdl.c:1061–1064`); the roofline-correct `fmax`
    replacement is validated by tritonBLAS.

### No cross-vendor cost model exists

- wave-04-cost-models.md (synthesis): "All existing cost models are per-vendor and
  per-architecture — none address cross-vendor runtime selection between a CUDA kernel
  variant and a HIP kernel variant."
- wave-03-cost-model-selection.md (angle assessment): "Every cost model (Seer, SparseX,
  tritonBLAS, NeuSight, ML-MLIR, TACO 2021, cuBLAS, Ansor, Fasor) operates within a
  single vendor's hardware ecosystem. libkdl's roofline cost model... is the only
  published runtime dispatch system that selects across vendors."

---

## Feasibility

**High feasibility — three independent paths, each de-risked.**

### Path A: External annotation tool (zero LLVM changes, deliverable at poster time)

`kdl-contract-gen`: a ~200-LOC Python script that:
1. Runs `opt --passes=kernel-info --pass-remarks-output=remarks.yml` on the GPU device IR
2. Parses the remarks YAML for `IndirectCalls`, `InlineAssemblyCalls` (validity gate),
   `AllocasStaticSizeSum` (shared-mem lower bound), `omp_target_thread_limit`
3. For kernels passing the validity gate, runs a second pass counting FP and memory
   instructions using `opt --passes=instruction-count` and LLVM's remark infrastructure
4. Computes AI = FP_ops / memory_bytes using SCEV-derived trip counts where available
5. Looks up per-arch ridge points from the vendor table and classifies the regime
6. Writes `kdl.cost.*` fields into the MTB contract JSON

This path demonstrates the end-to-end concept using only existing LLVM infrastructure.
It is the poster prototype deliverable, runnable on the GTX 1650 + CPU test system before
the April 7, 2026 deadline.

### Path B: New LLVM analysis pass (~600 LOC, clean upstream contribution)

`KernelAIAnalysis` as a new `FunctionPass` in `llvm/lib/Analysis/`:
- Walk function IR counting FP and memory instructions
- Query `ScalarEvolutionAnalysis` for trip counts (already used by `LoopVectorize`)
- Gate on KernelInfo's call-type summary as a validity pre-filter
- Register immediately after KernelInfo in the LTO pipeline (AMDGPU:
  `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp:1115` — the existing KernelInfo
  registration site)
- Emit `OptimizationRemark` entries for `flops`, `bytes_total`, `arithmetic_intensity`

**Scope:** ~500 LOC new pass + ~30 LOC registration patches per backend. Single GSoC
project scope. Does not modify KernelInfo (separate pass, separate concerns).

### Path C: TTI extension (clean upstream interface, 12–18 months)

Add `getPeakThroughputFLOPS(ScalarTy)` and `getPeakMemoryBandwidthGBPS()` to
`TargetTransformInfo` (`llvm/include/llvm/Analysis/TargetTransformInfo.h`).
- `NVPTXTTIImpl` returns values computed from `MCSubtargetInfo` (SM count × clock × 2
  FLOPs/cycle for FMA, memory clock × bus width / 8)
- `AMDGPUTTIImpl` returns values from `GCNSubtarget` (CU count × wavefront size × 2 × clock,
  HBM bandwidth from memory subsystem feature flags)
- CPU TTI implementations already have this structure for `getCacheSize()` and
  `getNumberOfRegisters()`

This is the correct long-term upstream home for hardware peak specs in LLVM.

### OffloadBinary property embedding (fits topic-07 proposal)

The cost annotation embedding in `OffloadBinary` is already designed and de-risked in
topic-07-offloadbinary-metadata.md (Section: Feasibility). The `StringMap<StringRef>`
mechanism is additive, ABI-safe, and round-trip-safe. The `gpu-module-to-binary`
injection point (`mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp`) is confirmed.
Approximately 30 LOC per backend serializer.

---

## Upstream Path

### Stage 0 — Poster prototype (by April 7, 2026)

- `kdl-contract-gen` Python tool (Path A above): demonstrates the full pipeline
- Demo: GEMM kernel annotated with AI = 2.67 → compute-bound on H100, memory-bound on
  GTX 1650 → libkdl selects different variant per device
- Poster panel: "LLVM knows the FLOPs and bytes — it just never tells the runtime"

### Stage 1 — RFC: KernelAIAnalysis companion pass (6–9 months)

File RFC on LLVM Discourse (GPU/Offloading category):
```
[RFC] KernelAIAnalysis: Static Arithmetic Intensity Pass for GPU Kernels

Motivation: roofline-based cross-vendor dispatch requires AI at the binary
level; KernelInfo covers bad-code detection but not cost annotation.

Proposal: A companion pass KernelAIAnalysis that emits flops/bytes/AI as
remarks and (optionally) writes them into OffloadBinary properties.

Validity gate: borrows KernelInfo's IndirectCalls/InlineAssemblyCalls check.
Scope: ~600 LOC, no modification to KernelInfo.

Prototype: running in libkdl (LLVM Dublin 2026 poster).
```

Natural anchor points:
- MLIR cost model RFC (https://discourse.llvm.org/t/rfc-target-description-and-cost-model-in-mlir/76990):
  concluded "MLIR lacks a standard cross-target cost model interface" — KernelAIAnalysis
  fills this at the IR level for GPU kernels
- KernelInfo PR #102944 authors (Joel Denny, ORNL): natural collaborators; their stated
  follow-on interest is PGO-based (IWOMP 2025 paper), but a static companion is orthogonal

### Stage 2 — TTI extension for peak specs (12–18 months)

Add `getPeakThroughputFLOPS` and `getPeakMemoryBandwidthGBPS` to TTI. Anchor to:
- The existing `getCacheSize(unsigned Level)` pattern in TTI
- NVPTXTTIImpl and AMDGPUTTIImpl already implement target-specific TTI overrides

### Stage 3 — OffloadBinary standardisation (concurrent with Stage 2)

Fold the cost annotation keys into the topic-07 OffloadBinary metadata vocabulary RFC.
The RFC already covers `shared_mem_bytes`, `sgpr_count`, `vgpr_count` — adding
`kdl.cost.flops`, `kdl.cost.bytes_total`, `kdl.cost.ai` as Tier 3 quality keys extends
the same RFC without a separate document.

**Key upstream stakeholders:**
- Joel E. Denny, ORNL — KernelInfo author; PGO complement is natural
- Johannes Doerfert — GPU offloading, OpenMP, liboffload framing
- Joseph Huber, AMD — OffloadBinary format owner, liboffload
- NVPTXTTIImpl maintainers — for TTI extension (NVIDIA LLVM team)
- AMDGPUTTIImpl maintainers — GCN/RDNA backend (AMD)

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **9/10** | No pass annotates GPU kernels with AI for cross-vendor dispatch. KernelInfo is upstream but explicitly not a roofline tool. TTI has no peak-throughput or peak-bandwidth interface. The cross-vendor dimension is unaddressed by all 12+ surveyed cost model systems. |
| **Feasibility** | **9/10** | Path A (Python tool) is demonstrable at poster time with zero LLVM changes. Path B (~600 LOC) is a clean, scoped new pass. Upstream hooks (KernelInfo, TTI, OffloadBinary) are confirmed and extensible. |
| **Evidence Strength** | **9/10** | tritonBLAS (94.7% with 3 numbers, arXiv:2512.04226), NeuSight (2.3% on H100, ASPLOS 2025), TACO 2021 (9–52% MAPE baseline, ACM TACO), KernelInfo PR #102944 (upstream, source-verified), libkdl kdl.c (live prototype, line-cited). All evidence from primary sources. |
| **Impact** | **9/10** | Bridges the only missing link in the static-analysis-to-runtime-dispatch pipeline. Applicable beyond libkdl to any multi-variant GPU kernel system (IREE HAL, liboffload policy layer, cuBLAS successors). TTI extension would benefit all GPU cost-model-aware passes in the ecosystem. |
| **Poster fit** | **8/10** | Strong visual story: roofline diagram with GEMM kernel plotted against two device ridges; table of regime classifications per arch; accuracy vs. tritonBLAS baseline; pass pipeline diagram. Slightly complex but clearly bounded. |
| **Upstream risk** | **3/10** (low risk) | Path A needs no upstream changes. KernelAIAnalysis RFC is low-friction (separate pass, clear precedent). TTI extension is the only potentially contentious step (interface stability). OffloadBinary extension is already scoped in topic-07. |
| **Composite** | **8.83/10** | |

---

## Pitch

Every GPU kernel has an arithmetic intensity — FLOPs divided by memory bytes — that is
computable at compile time for structured ML workloads, and is independent of which GPU
will run the kernel. Every GPU has a ridge point — peak FLOP/s divided by peak bandwidth
— that determines whether a kernel is compute-bound or memory-bound on that specific
device. LLVM already has all the pieces: `ScalarEvolution` for trip counts, instruction
iterators for FP/memory counts, `TargetTransformInfo` for architecture data,
`OffloadBinary` for binary metadata, and `KernelInfo` for the validity gate. What does
not exist is the pass that assembles them. We propose `KernelAIAnalysis`, a companion to
`KernelInfo` that annotates each GPU kernel with its arithmetic intensity, classifies it
as compute-bound or memory-bound per vendor architecture, and embeds the result in the
`OffloadBinary` string table — giving runtime dispatchers like libkdl the three numbers
(FLOPs, bytes, AI) that tritonBLAS proves are sufficient for 94.7% of optimal dispatch
quality, without any runtime profiling, and for the first time across NVIDIA, AMD, and
CPU in a single unified pass.

---

## Risks

1. **Static AI is wrong for irregular kernels.** TACO 2021 shows 52% MAPE for memory-bound
   kernels with input-variable access patterns (sparse, attention with variable seq_len).
   Mitigation: emit `ai_status = "static_estimate"` vs `"pgo_measured"` flag to allow
   the dispatcher to de-weight uncertain annotations. The dispatch problem only needs
   regime classification, not exact latency — the error bar on dispatch decisions is smaller
   than the MAPE on time prediction.

2. **SCEV trip count is not always available.** Kernels with data-dependent loops (while
   loops, early exits, pointer-chased traversals) have no SCEV-derivable trip count.
   Mitigation: emit a conservative lower bound (trip count = 1 for loops without SCEV)
   and set `ai_confidence = "low"`. The validity gate (`IndirectCalls == 0`) eliminates
   the worst cases.

3. **KernelInfo's `isRequired() = true` precedent causes mandatory overhead.** Adding
   another required pass multiplies mandatory compile-time cost. Mitigation: register
   `KernelAIAnalysis` as optional (not `isRequired()`), enabled only with
   `-Rpass=kernel-ai-info` or an explicit `-foffload-cost-annotate` flag.

4. **Vendor table requires maintenance.** As new GPU architectures ship, the built-in
   table becomes stale. Mitigation: (a) TTI extension (Stage 2) moves the data to the
   architecture's own description; (b) runtime calibration feedback (libkdl's
   `kdl_calibrate()`) overrides stale table values with measured actuals.

5. **NeuSight AMD gap.** NeuSight is trained on NVIDIA GPUs; AMD transfer accuracy is
   unvalidated. The analytical roofline (this proposal) is vendor-agnostic by construction
   and does not depend on NeuSight — this risk is for future ML-augmented variants of the
   pass, not the current proposal.

6. **OffloadBinary property round-trip precision.** FP values stored as ASCII strings with
   `%.17g` format preserve full double precision (`strtod` is exact for round-trip). The
   implementation must enforce this format; `%f` or `%g` (6 digits) would lose precision
   for large `flops` values (e.g., GEMM at M=N=K=4096 has ~1.37 × 10^11 FLOPs).

---

## Related Research

- **topic-03-dispatch-cost-attr.md** — sister proposal for the MLIR attribute layer
  (`#kdl.cost` DictionaryAttr on `gpu.func`) and the three-layer propagation protocol;
  this topic focuses on the new LLVM IR pass and the vendor classification layer
- **topic-07-offloadbinary-metadata.md** — OffloadBinary vocabulary standardisation;
  the cost annotation keys proposed here fold into the Tier 3 "quality keys" of that RFC
- **wave-08-kernel-info-pass.md** — complete KernelInfo field analysis; defines the
  validity gate and the partial coverage that motivates this proposal
- **wave-03-cost-model-selection.md** — tritonBLAS, NeuSight, SparseX, TACO 2021 primary
  source analysis; establishes the 94.7% quality ceiling and the MAPE baseline
- **wave-04-cost-models.md** — cuBLAS/CUTLASS/Stream-K++ production precedents; the
  three-tier dispatch architecture (elimination → analytical ranking → calibration)
- **literature/cost-models-kernel-dispatch.md** — roofline model theory, hardware spec
  table, libkdl cost model analysis (Sections 1.1–1.4), recommended `fmax` correction
- **directions/03-roofline-cross-vendor-cost-model.md** — the sister direction document
  covering runtime selection algorithm; this topic provides the missing compile-time input
- **kdl.c:112–122** — `kdl_contract` struct (fields this pass populates)
- **kdl.c:1013–1088** — `kdl_estimate_cost_weighted()` (the runtime consumer)
- **kdl.c:1016** — `has_compute_profile` gate (the injection point where embedded AI
  changes dispatch from priority-based to roofline-based)
- **kdl.c:611–868** — device enumeration computing `peak_tflops_f32` / `peak_bw_gbps`
  at runtime (the runtime analog of the proposed compile-time vendor table)

---

*Generated: 2026-04-07 | Research basis: KernelInfo.cpp (LLVM main), OffloadBinary.h/cpp
(LLVM main), kdl.c (prototype, lines cited), arXiv:2512.04226 (tritonBLAS),
arXiv:2407.13853 (NeuSight ASPLOS 2025), arXiv:2001.07104 (TACO 2021),
arXiv:2408.11417 (Stream-K++), literature/cost-models-kernel-dispatch.md,
wave-03-cost-model-selection.md, wave-04-cost-models.md, wave-08-kernel-info-pass.md,
topic-03-dispatch-cost-attr.md, topic-07-offloadbinary-metadata.md*
