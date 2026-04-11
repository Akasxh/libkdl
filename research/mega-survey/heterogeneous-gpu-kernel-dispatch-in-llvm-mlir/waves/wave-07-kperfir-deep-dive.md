# Wave 07 — KPerfIR Deep Dive: Cross-Vendor MLIR Profiling as Ground Truth for libkdl

**Angle:** KPerfIR OSDI 2025 — compiler-integrated GPU profiling and its integration potential with libkdl dispatch
**Date:** 2026-04-06
**Sources surveyed:** 8 primary (paper, artifact repo, Triton upstream, literature review, related search)

---

## Executive Summary

KPerfIR (OSDI 2025, Guan et al., UCSD/Meta/OpenAI) is a multi-level MLIR dialect infrastructure
that embeds profiling instrumentation directly into the Triton compiler pipeline. It achieves 2%
relative measurement error and 8.2% runtime overhead while covering both NVIDIA H100 and AMD
MI300X at the same abstraction level. Its open-source upstream lives in
`triton-lang/triton/third_party/proton/dialect` as the `proton` dialect (the published name for
KPerfIR's highest-level abstraction).

For libkdl, KPerfIR is not a drop-in dependency — it is fundamentally a compile-time offline
profiler, not a runtime cost oracle. However, it is the best-precision available tool for
generating the ground-truth kernel performance tables that libkdl's `kdl_estimate_cost_weighted`
currently replaces with hardcoded efficiency constants (0.70 NVIDIA, 0.50 AMD, 0.30 CPU). The
integration path is: KPerfIR profiles pre-compiled kernel variants at bundle build time and
writes per-variant performance metadata (clock cycles, compute vs. memory bottleneck ratio,
pipeline utilization) into the MTB (Multi-Target Bundle) JSON contract. libkdl's dispatch loop
consumes these values at runtime, replacing the current static roofline with calibrated
per-variant costs.

**Relevance to libkdl cost model ground truth:** 9/10
**Implementation difficulty of integration:** Medium-High (requires Triton build chain per target)
**Novelty for poster argument:** High — no existing system closes the compile-time profiling →
runtime dispatch feedback loop across vendors

---

## 1. Paper Identity and Availability

| Field | Value |
|-------|-------|
| Title | KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling on Modern AI Workloads |
| Venue | USENIX OSDI 2025 (19th Symposium on Operating Systems Design and Implementation) |
| Authors | Yue Guan, Yuanwei Fang, Keren Zhou, Corbin Robeck, Manman Ren, Zhongkai Yu, Yufei Ding, Adnan Aziz |
| Affiliations | UC San Diego, Meta, George Mason University / OpenAI |
| PDF | https://www.usenix.org/system/files/osdi25-guan.pdf |
| arXiv | https://arxiv.org/abs/2505.21661 (HTML: https://arxiv.org/html/2505.21661v1) |
| Artifact | https://github.com/ChandlerGuan/kperfir_artifact |
| Upstream code | https://github.com/triton-lang/triton/tree/main/third_party/proton/dialect |
| Docs | https://triton-lang.org/main/dialects/ProtonOps.html |
| ACM DL | https://dl.acm.org/doi/10.5555/3767901.3767913 |

---

## 2. Architecture: Multi-Level MLIR Dialect Design

KPerfIR implements a three-tier IR hierarchy. Each tier corresponds to a distinct abstraction
boundary in the Triton compilation pipeline:

### Tier 1: KPerfIR (highest, hardware-agnostic)

The sole operation at this level is `RecordOp`, exposed upstream as `proton.record`:

```
proton.record start "name0"
...
proton.record end "name0"
```

Parameters:
- `isStart` (UnitAttr): distinguishes start/end marker
- `name` (StringAttr): event identifier, unique per function scope

RecordOp carries no hardware semantics — it is a semantic-agnostic program marker whose
interpretation is entirely deferred to conversion pass configuration. This is the portability
keystone: one RecordOp lowers differently on H100 vs MI300X.

### Tier 2: KPerfGPUIR (intermediate, GPU-specific but vendor-neutral)

When RecordOp is lowered, it becomes four concrete operations controlled by pass knobs
(`MetricType`, `Granularity`, `BufferStrategy`):

| Operation | Purpose |
|-----------|---------|
| `InitOp` | Allocate shared-memory profiling buffer, initialize state |
| `ReadCounterOp` | Read hardware performance counter (e.g., `%clock`) into scalar register |
| `StoreCounterOp` | Write counter value + tag to profiling buffer with index management |
| `FinalizeOp` | Writeback profiling buffer to global memory with metadata header |

`Granularity` selects the profiling scope: warp-group, warp, or thread. `MetricType` selects
which hardware counter to read.

### Tier 3: LLVM IR (lowest, vendor-specific)

`startInstrumentationOp` and `stopInstrumentationOp` bridge to hardware ISA:

**NVIDIA path:** PTX `%clock` read. Hardware handles instruction scheduling; compiler inserts
three instructions per record node (clock read, integer move, predicated store), averaging
33 cycles on H100.

**AMD path:** `amdgcn` assembly. AMD exposes instruction scheduling to software (unlike NVIDIA),
requiring three levels of user-configurable scheduling hints:
1. Manual KPerfIR hints (default)
2. Direct `amdgcn` instrumentation
3. Explicit instruction scheduling window with barrier masks

AMD collaborative store strategy: only one timestamp per warp/warp-group, using thread mask
branching with predication. Per-record overhead: 60 cycles on MI300X (vs 33 on H100).

---

## 3. Cross-Vendor Portability: How It Actually Works

The key design insight is that portability is achieved at Tier 1, not at Tier 3. The application
programmer (or compiler pass) only ever sees `RecordOp` — the vendor-specific lowering is
encapsulated entirely within the conversion passes. This mirrors libkdl's own design philosophy
where the `kdl_contract` JSON is target-agnostic and vendor selection happens at dispatch time.

**Abstraction boundary comparison:**

| Layer | KPerfIR concept | libkdl analogue |
|-------|----------------|-----------------|
| User-facing | `proton.record` markers | `kdl_contract` JSON |
| Intermediate | KPerfGPUIR ops (vendor-neutral GPU) | `kdl_device_info` + cost model |
| Hardware | PTX / amdgcn lowering | `cuLaunchKernel` / `hipModuleLaunchKernel` |

One significant limitation: KPerfIR is evaluated only on NVIDIA H100 and AMD MI300X. It has
no SPIR-V lowering path and no direct Intel/Level Zero support. Instrumenting a SYCL or
OpenCL kernel requires Triton-level entry, which is not universal.

---

## 4. Measurement Accuracy: The Trace Replay Algorithm

The 2% relative error claim is not inherent to the counter reads — it is achieved through a
post-processing algorithm called **trace replay** that cancels out instrumentation overhead:

### Synchronous regions

For code regions containing only synchronous instructions:

```
T_actual = CLK_end - CLK_start - T_overhead_per_record
```

The overhead `T_overhead_per_record` is characterized once per target (33 cycles H100,
60 cycles MI300X) and subtracted algebraically.

### Asynchronous regions (tensor cores, async SMEM loads)

For operations like MMA (matrix multiply-accumulate) with async dispatch:

```
placement:
  RecordOp START₁   # before async launch
  RecordOp START₂   # after async launch
  <async barrier>
  RecordOp END       # after wait

T_wait = CLK(START₂) - CLK(START₁)   # overhead cancels out
T_compute = CLK(END) - CLK(START₂) - T_overhead
```

The algebraic cancellation requires `T_MMA - T_exe > T_a + T_b` (functional unit execution
exceeds profiling overhead of ~25 cycles). This holds for virtually all production ML kernel
inner loops.

**Overflow handling:** 32-bit clock values are sufficient because execution is primarily in
loops; per-iteration time is under 1 ms for all measured kernels, keeping counters within range.

---

## 5. Overhead Mechanics: How 8.2% Is Achieved

Three engineering decisions keep overhead low:

**Circular buffer in shared memory.** Instead of flushing profiling data to global memory on
every iteration, KPerfIR uses a circular buffer in shared memory that stores only the most
recent iterations. For SWP GEMM with 3 pipeline stages on H100: 10.9 KB of the 228 KB shared
memory is available for the circular buffer — 1.75% of total. Global memory traffic occurs only
at kernel exit via `FinalizeOp`.

**Register promotion via stack allocation.** Buffer index values are allocated on the stack
rather than in global memory; LLVM's register promotion optimization (`mem2reg`) promotes these
to registers automatically. The compiler pass leverages this by design, generating `alloca`
patterns that LLVM recognizes.

**Minimal instruction footprint.** Each record = 3 instructions (clock read + integer move +
predicated store). 33 cycles total on H100. For a 10,000-iteration kernel inner loop with 2
RecordOps per iteration: 66,000 extra cycles at ~1.5 GHz ≈ 44 μs overhead on a kernel taking
~500 μs — roughly 8.8%, consistent with the paper's figure.

The most complex case (SWP GEMM) shows overhead within 15%. All simpler cases are under 10%.

---

## 6. Performance Model: What KPerfIR Measures and Produces

KPerfIR's output is not just raw cycle counts — it feeds two explicit performance models in the
paper:

### Software Pipelining (SWP) bottleneck model

```
Δ = N_WG × N_pipe × Σᵢ T_comp - Max_i(T_load^i + T_comp^i)
```

- Δ > 0: compute-bound
- Δ < 0: memory-load-bound
- The measured `T_comp` and `T_load` values per pipeline stage come directly from KPerfIR traces

### Warp Specialization (WS) critical path model

```
T_kernel = Σ(i ∈ CriticalPath) T_load/comp^i
```

Critical path is identified by topological sort over producer-consumer warp dependencies,
with stage latencies measured by KPerfIR.

**Validation result:** Flash-Attention 3 predicted at 582.44 TFLOPs, with KPerfIR-guided
optimization achieving a 24.1% speedup over vanilla Triton FA3 and 7.6% over manual FA3.

These models quantify exactly the parameters libkdl's cost model currently approximates with
static efficiency constants: actual compute time, actual memory time, actual pipeline utilization
per target device per kernel variant.

---

## 7. Open-Source Status and Artifact

### Upstream (production-grade)

The KPerfIR dialect is merged into Triton mainline as the `proton` dialect:
- `triton-lang/triton/third_party/proton/dialect/`
- Triton 3.0.0+ (the evaluation version)
- LLVM 19.1 backend requirement

The single upstream operation `proton.record` is documented at
https://triton-lang.org/main/dialects/ProtonOps.html.

### Artifact (evaluation reproduction)

https://github.com/ChandlerGuan/kperfir_artifact contains:
- `Dockerfile` — reproducible H100 environment
- `bench_fa.sh` — FlashAttention benchmark driver
- `benchmark_attn.py` — FA2/FA3 baseline comparisons
- `changes.patch` — diff against Triton 3.0.0
- `ttgir/` — vanilla and improved kernel TTGIR representations

**Requirement:** NVIDIA H100 GPU for full reproduction (AMD path not in artifact).

---

## 8. Integration Design: KPerfIR → libkdl

### Current libkdl cost model (the gap)

`kdl_estimate_cost_weighted` in `experiments/prototype/src/kdl.c:1013` uses static constants:

```c
case KDL_VENDOR_NVIDIA: efficiency = 0.70; break;
case KDL_VENDOR_AMD:    efficiency = 0.50; break;
case KDL_VENDOR_CPU:    efficiency = 0.30; break;
```

And static locality penalties:
```c
case KDL_VENDOR_NVIDIA: locality_score = 50e-6;  break;
case KDL_VENDOR_AMD:    locality_score = 60e-6;  break;
```

These are calibration-free estimates. The function falls back to priority ordering if
`has_compute_profile == 0`, making the `kdl_contract` fields `flops`, `bytes_total`, and
`arithmetic_intensity` the only per-variant data the cost model can consume.

### Proposed integration: KPerfIR as MTB bundle build-time profiler

KPerfIR would run during the MTB bundle build process, not at dispatch runtime. The flow:

```
[Triton kernel source]
    → KPerfIR instrumentation pass
    → compile to PTX (H100) or amdgcn (MI300X)
    → run on target hardware
    → trace replay post-processing
    → per-variant performance profile
        {
          "t_compute_ns": 142.3,
          "t_memory_ns": 89.1,
          "pipeline_bottleneck": "compute",
          "efficiency_measured": 0.847,
          "arithmetic_intensity": 312.4
        }
    → embed in MTB JSON contract
    → libkdl reads at dispatch time
```

This replaces the static `efficiency` constants with measured values per variant per target arch.
The `kdl_contract` struct already has the right fields (`flops`, `bytes_total`,
`arithmetic_intensity`, `has_compute_profile`) — KPerfIR would populate them with truth data.

### Three concrete fields KPerfIR can populate

| `kdl_contract` field | Current source | KPerfIR replacement |
|---------------------|---------------|---------------------|
| `arithmetic_intensity` | Computed from shape metadata | Measured AI from KPerfIR trace (bytes actually transferred vs. FLOPs actually executed) |
| `efficiency` (implicit via cost model) | Static 0.70/0.50/0.30 | Measured `T_actual / T_peak_roofline` per variant per device class |
| Pipeline bottleneck classification | Not present | KPerfIR SWP model outputs `compute_bound` / `memory_bound` flag directly |

### New MTB JSON contract extension

```json
{
  "variant": "gemm_fp16_sm90",
  "target": "nvptx",
  "contract": {
    "flops": 1.34e12,
    "bytes_total": 4.29e9,
    "arithmetic_intensity": 312.4,
    "kperfir_profile": {
      "t_compute_us": 142.3,
      "t_memory_us": 89.1,
      "pipeline_bottleneck": "compute",
      "measured_efficiency": 0.847,
      "profiled_on": "H100-HBM3",
      "profiled_triton_version": "3.0.0",
      "profile_error_pct": 2.0
    }
  }
}
```

libkdl's `kdl_estimate_cost_weighted` would then use `kperfir_profile.measured_efficiency`
directly instead of the static vendor constant when the field is present.

### Dispatch-time algorithm update

```c
/* Proposed: use KPerfIR-measured efficiency if available in contract */
double efficiency;
if (c->kperfir_measured_efficiency > 0.0) {
    efficiency = c->kperfir_measured_efficiency;  /* ground truth */
} else {
    /* fallback to static vendor constants */
    switch (d->vendor) {
        case KDL_VENDOR_NVIDIA: efficiency = 0.70; break;
        case KDL_VENDOR_AMD:    efficiency = 0.50; break;
        default:                efficiency = 0.40; break;
    }
}
```

This is a non-breaking change: bundles without KPerfIR metadata fall back to current behavior;
bundles built with KPerfIR get ground-truth dispatch.

---

## 9. Limitations and Risks for libkdl Integration

### L1: KPerfIR is Triton-only (for now)

KPerfIR instrumentation requires the kernel to be compiled via Triton. Kernels compiled with
CUTLASS, raw HIP C++, or OpenCL cannot be instrumented without porting to Triton first. For
libkdl's current GTX 1650 prototype (CUDA C / PTX kernels, not Triton), KPerfIR is not directly
usable. An alternative profiling step using `cuEventElapsedTime` (already implemented in libkdl
at `kdl.c:299-308`) would be needed for non-Triton kernels.

### L2: AMD artifact not reproduced

The artifact only ships with H100 evaluation. MI300X profiling is documented in the paper
(60 cycles/record, collaborative store strategy) but requires access to MI300X hardware and a
separate Triton build targeting `amdgcn`. This makes cross-vendor ground truth expensive to
obtain at bundle build time.

### L3: Profiling captured at a specific hardware generation

KPerfIR profiles on H100-HBM3 / MI300X. The efficiency value embedded in MTB JSON is specific
to that device class. When libkdl dispatches on an A100, H200, or MI300A, the measured
efficiency from H100 is an approximation, not ground truth. The MTB format would need a
`profiled_on` field (shown in the proposed JSON above) and libkdl would need arch-class matching
logic to select the closest profiled variant.

### L4: Static compilation assumption

KPerfIR is fundamentally an ahead-of-time measurement tool. It cannot profile at dispatch time
without executing the kernel — which is exactly the overhead libkdl is trying to avoid. There
is no runtime feedback loop in the published design. The "runtime dispatch" integration is
necessarily offline (bundle-build-time profiling), not online adaptation.

### L5: 8.2% overhead is non-trivial for production

When profiling to populate MTB JSON, the 8.2% overhead means profiling measurements should
be collected in a dedicated profiling build, not in production execution. This adds a two-build
requirement to the bundle authoring workflow.

---

## 10. Related Work Found During Research

### TritonForge (arXiv:2512.09196, December 2025)
Profiling-guided LLM framework for Triton kernel optimization. Integrates KPerfIR-style metrics
(memory throughput, warp occupancy, instruction stalls from Nsight Compute) into an LLM
generation loop to steer code transformations. Not directly related to dispatch, but demonstrates
the "profiling → code modification" feedback loop that KPerfIR enables.

### The Proton Dialect Slides (LLVM Dev Meeting March 2025)
https://llvm.org/devmtg/2025-03/slides/the_proton_dialect.pdf — presentation at the LLVM
Developers' Meeting (March 2025) documenting the upstream Triton proton dialect. Confirms KPerfIR
graduated from paper to mainline tooling before the paper was published at OSDI.

### Neutrino: Fine-grained GPU Kernel Profiling via Programmable Probing (OSDI 2025 co-paper)
Listed in the OSDI 2025 program alongside KPerfIR. Not yet fetched in detail. Likely provides
complementary profiling at a finer granularity (warp/instruction level). Worth a follow-up wave.

---

## 11. Key Claims for Poster

1. **Ground-truth cost values exist.** KPerfIR proves that 2%-accurate, vendor-neutral kernel
   performance characterization is possible at MLIR level with low overhead. This validates
   libkdl's architectural bet that per-variant cost metadata can be embedded in bundles.

2. **libkdl's static efficiency constants are the weak link.** The current `kdl_estimate_cost_weighted`
   uses 0.70/0.50/0.30 efficiency constants that are not calibrated per kernel or per device
   architecture. KPerfIR-measured efficiency per variant would make libkdl's dispatch genuinely
   data-driven.

3. **The integration gap is novel.** No existing system closes the loop from compile-time
   profiling (KPerfIR) through bundle metadata (MTB JSON) to runtime dispatch selection (libkdl).
   KPerfIR profiles; libkdl decides. This two-system composition is a research contribution.

4. **Cross-vendor is the differentiator.** KPerfIR profiles on H100 and MI300X using the same
   RecordOp abstraction. libkdl selects between H100 and MI300X variants at runtime. The shared
   abstraction level (MLIR RecordOp → MTB efficiency field) is the technical bridge.

---

## Sources

- [KPerfIR paper PDF (USENIX)](https://www.usenix.org/system/files/osdi25-guan.pdf)
- [KPerfIR arXiv HTML](https://arxiv.org/html/2505.21661v1)
- [KPerfIR arXiv abstract](https://arxiv.org/abs/2505.21661)
- [USENIX OSDI 2025 presentation page](https://www.usenix.org/conference/osdi25/presentation/guan)
- [KPerfIR artifact repository](https://github.com/ChandlerGuan/kperfir_artifact)
- [Triton proton dialect docs](https://triton-lang.org/main/dialects/ProtonOps.html)
- [Moonlight literature review of KPerfIR](https://www.themoonlight.io/en/review/kperfir-towards-an-open-and-compiler-centric-ecosystem-for-gpu-kernel-performance-tooling-on-modern-ai-workloads)
- [ACM DL proceedings entry](https://dl.acm.org/doi/10.5555/3767901.3767913)
- [Proton dialect LLVM DevMtg slides](https://llvm.org/devmtg/2025-03/slides/the_proton_dialect.pdf)
- [TritonForge profiling-guided optimization](https://arxiv.org/html/2512.09196v1)
