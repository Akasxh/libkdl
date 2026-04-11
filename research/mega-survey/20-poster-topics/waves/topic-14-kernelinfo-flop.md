# Topic 14: Extending KernelInfo with Static FLOP/Byte Counting for GPU Kernels

**Topic ID:** 14
**Config key:** `kernelinfo-flop-extension`
**Persona:** LLVM analysis pass author
**Date:** 2026-04-07
**Research depth:** Exhaustive — 9 specific sub-questions investigated, cross-referenced
against KernelInfo upstream source, kdl.c prototype (5100 LOC), tritonBLAS, NeuSight,
ORNL IWOMP 2025 paper, LLVM `InstructionCost` / TTI infrastructure, and 7 existing
wave/topic files.

---

## Gap

The LLVM KernelInfo pass (PR #102944, Joel E. Denny, ORNL, merged 2025-01-29) is
upstream, production-quality, and runs at LTO time for every GPU kernel compiled with
AMDGPU or NVPTX backends.  It emits 19 optimization remarks covering stack usage,
call-graph properties, flat-addrspace accesses, and vendor-specific launch bounds.
It does **not** emit FLOP counts, memory byte totals, or arithmetic intensity.

The design gap is explicit and intentional.  From PR #102944 (Denny): *"The ultimate
goal of these statistics is to help identify bad code patterns and ways to mitigate
them."*  The companion IWOMP 2025 paper (McDonough, Denny, Doerfert, LNCS vol. 16123,
pp. 99-113) targets runtime PGO for GPU targets, not static contract generation.  The
roofline model keyword appears only in the IWOMP abstract's keyword list; the system
does not compute arithmetic intensity.

The three numbers that tritonBLAS (arXiv:2512.04226) proves are sufficient to achieve
94.7% of exhaustive autotuning quality -- `flops`, `bytes_total`, `arithmetic_intensity`
-- have no upstream LLVM path that computes them statically from IR and persists them
for a runtime dispatcher to consume.

**Consequence for libkdl:** `kdl_estimate_cost_weighted()` at `kdl.c:1013-1088` gates
on `c->has_compute_profile` (line 1016: `if (!c->has_compute_profile) return 1e9`).
With no KernelInfo FLOP data, every dispatch falls back to priority-based ordering.
The roofline model exists in the code but is never activated by the compiler pipeline.

---

## Proposal

**Title:** `KernelCostInfo` -- A Companion LLVM IR Pass for Static FLOP and Memory Byte
Counting in GPU Kernels

**One-sentence pitch:** Extend the KernelInfo ecosystem with a companion
`KernelCostInfo` pass that counts `fadd`/`fmul`/`fma` and load/store bytes in GPU
kernel IR, gates on KernelInfo's validity flags (`IndirectCalls == 0`,
`InlineAssemblyCalls == 0`), and emits `FlopCount`, `MemBytes`, and
`ArithmeticIntensity` remarks -- closing the static-analysis-to-roofline-dispatch gap
without requiring runtime profiling.

### What to count and how

**FLOP counting (integer instructions, no floating-point type required)**

Walk every `BasicBlock` of a GPU `Function` with kernel calling convention.  For each
`Instruction`, pattern-match on `Instruction::getOpcode()`:

| Opcode | FLOPs credited | Notes |
|--------|---------------|-------|
| `fadd`, `fsub` | 1 | Per element; multiply by vector width if VectorType |
| `fmul` | 1 | |
| `fdiv` | 4 | Approximation; accurate for non-reciprocal hardware paths |
| `fma` / `llvm.fma.*` intrinsic | 2 | Fused: 1 multiply + 1 add counted together |
| `llvm.sqrt.*` | 4 | Approximation matching roofline convention |

FLOPs from SIMD types are scaled by `VectorType::getNumElements()`, which is available
on any `VectorType*` cast of `Instruction::getType()`.

**Loop trip count scaling**

For loop nests where `ScalarEvolution` can provide a concrete trip count
(`SE.getSmallConstantTripCount(Loop*)`), multiply the per-iteration FLOP count by the
trip count.  This is the same mechanism used by `LoopVectorize` and `UnrollAnalysis` --
no new infrastructure required.  When the trip count is `0` (symbolic / unknown), emit
a separate `FlopCountBound` = `unknown` remark and set `has_compute_profile = false` in
the downstream contract.  KernelInfo's `IndirectCalls > 0` already signals a related
conservativeness failure mode; the two gates compose cleanly.

**Memory byte counting**

For each `LoadInst` and `StoreInst`, extract the accessed type width:

```cpp
uint64_t bytes = DL.getTypeStoreSize(I.getType()).getFixedValue();
```

`DataLayout::getTypeStoreSize()` is the correct API -- it returns the allocation size in
bytes including padding, which matches the hardware memory traffic model.  For
`AtomicRMWInst` and `AtomicCmpXchgInst`, count one load + one store of the operand
type width.  For vector loads/stores, count `VectorType::getNumElements() * element_bytes`.

Apply the same SCEV trip-count scaling as FLOP counting.

**`FlatAddrspaceAccesses` as an existing proxy**

KernelInfo already counts `FlatAddrspaceAccesses` -- load/store/atomic ops on
`addrspace(0)` (flat).  This is not a byte count, but it is a validity signal: if
`FlatAddrspaceAccesses > 0` and the target does not have flat address space support, the
kernel should not be dispatched there.  The new `MemBytes` remark complements this --
`FlatAddrspaceAccesses` says *where* memory goes; `MemBytes` says *how much*.

**Arithmetic intensity**

Compute as `FlopCount / MemBytes` (double precision).  Emit only if both denominators
are nonzero and `has_compute_profile` is valid.  The result is `ArithmeticIntensity` in
FLOPs per byte -- the standard roofline x-axis unit.

### Pass architecture: companion, not extension

KernelInfo is registered as `isRequired() = true` via `AMDGPUTargetMachine.cpp:1115`:
```cpp
FPM.addPass(KernelInfoPrinter(this));
```

Adding FLOP counting to KernelInfo risks scope creep the upstream community will reject
-- the PR #102944 comment thread shows Denny explicitly framing it as a bad-pattern
detector.  The correct architecture is a **separate companion pass** `KernelCostInfo`,
registered immediately after `KernelInfoPrinter` in the same `FunctionPassManager`:

```cpp
FPM.addPass(KernelInfoPrinter(this));
FPM.addPass(KernelCostInfoPrinter(this));   // NEW
```

`KernelCostInfoPrinter` consumes the cached `KernelInfoAnalysis` result via the
standard `AM.getResult<KernelInfoAnalysis>(F)` pattern -- KernelInfo's `IndirectCalls`
and `InlineAssemblyCalls` counts are read directly from the cached result, with zero
re-analysis cost.

**Extensibility via `TargetTransformInfo`**

KernelInfo uses `TheTTI.collectKernelLaunchBounds(F, KI.LaunchBounds)` to dispatch
target-specific launch bound collection to the backend TTI implementation
(`llvm/include/llvm/Analysis/TargetTransformInfo.h`).  `KernelCostInfo` should use the
same pattern -- add a `TTI.collectKernelCostHints(F, KCI.CostHints)` virtual method
that targets can override to supply backend-specific FP throughput corrections (e.g.,
AMDGPU MFMA instructions count as 64 FLOPs per instruction but are not `fmul` opcodes).

### Remark output (proposed)

All remarks emitted under pass name `"kernel-cost-info"`:

| Remark key | Type | Description |
|---|---|---|
| `FlopCount` | int64 | Total floating-point operations (FMA counted as 2) |
| `FlopCountBound` | string | `"exact"` or `"lower_bound"` (lower = symbolic trip count present) |
| `MemBytes` | int64 | Total bytes touched by load/store/atomic instructions |
| `MemBytesBound` | string | `"exact"` or `"lower_bound"` |
| `ArithmeticIntensity` | float64 | `FlopCount / MemBytes` in FLOPs/byte |
| `HasComputeProfile` | bool | False if validity gates fail |
| `ValidityGate` | string | Reason for `HasComputeProfile = false` if applicable |

Example output (from projected test file `llvm/test/Analysis/KernelCostInfo/gemm.ll`):

```
remark: gemm.c:12:0: in function 'sgemm_kernel', FlopCount = 2097152
remark: gemm.c:12:0: in function 'sgemm_kernel', FlopCountBound = exact
remark: gemm.c:12:0: in function 'sgemm_kernel', MemBytes = 786432
remark: gemm.c:12:0: in function 'sgemm_kernel', ArithmeticIntensity = 2.667
remark: gemm.c:12:0: in function 'sgemm_kernel', HasComputeProfile = 1
```

Validity gate example (indirect call present):

```
remark: sparse.c:7:0: in function 'spmv_kernel', HasComputeProfile = 0
remark: sparse.c:7:0: in function 'spmv_kernel', ValidityGate = IndirectCalls > 0
```

### What `InstructionCost` buys and why it is the wrong tool here

LLVM's `InstructionCost` (`llvm/include/llvm/CodeGen/BasicTTIImpl.h`,
`TargetTransformInfo::getInstructionCost()`) estimates the **latency or throughput cost**
of an instruction on a specific target in terms of target-dependent cost units.  It
answers "how expensive is this `fmul` on an A100?" -- not "how many FLOPs does this
`fmul` represent?"

For the roofline model, the question is the algorithm-level FLOP count, independent of
target performance.  A 256-wide AVX-512 `fmul` and a scalar `fmul` both represent a
fixed number of arithmetic operations from the roofline perspective; their throughput
costs differ but their FLOP contributions are determined only by the vector width.

`InstructionCost` is the right tool for backend pass decisions (vectorization
profitability, unroll thresholds).  It is the wrong tool for roofline input generation.
`KernelCostInfo` uses raw opcode inspection + `DataLayout` + SCEV -- none of which
require `InstructionCost` -- and produces backend-independent FLOP/byte counts that
remain valid across NVPTX and AMDGPU targets.

---

## Evidence

### KernelInfo current remark list (complete, from upstream source)

Source: `llvm/lib/Analysis/KernelInfo.cpp` (326 lines), commit `18f8106f`, 2025-01-29.
Test corpus: `llvm/test/Analysis/KernelInfo/` (allocas.ll, calls.ll, flat-addrspace/,
launch-bounds/, linkage.ll, openmp/).

| Remark key | Emitted | Notes |
|---|---|---|
| `ExternalNotKernel` | yes | Linkage gate |
| `omp_target_num_teams` | yes | OpenMP launch bound |
| `omp_target_thread_limit` | yes | OpenMP launch bound |
| `maxclusterrank` | yes | NVPTX cluster |
| `maxntidx`, `maxntidy`, `maxntidz` | yes | NVPTX thread dims |
| `amdgpu-max-num-workgroups[0..2]` | yes | AMDGPU workgroups |
| `amdgpu-flat-work-group-size[0..1]` | yes | AMDGPU work-group size |
| `amdgpu-waves-per-eu[0..1]` | yes | AMDGPU occupancy hint |
| `Allocas` | yes | Alloca instruction count |
| `AllocasStaticSizeSum` | yes | Bytes of determinable stack allocs |
| `AllocasDyn` | yes | Dynamic-size alloca count |
| `DirectCalls` | yes | Direct call instructions |
| `IndirectCalls` | yes | Indirect call instructions |
| `DirectCallsToDefinedFunctions` | yes | Module-local direct calls |
| `InlineAssemblyCalls` | yes | Inline assembly calls |
| `Invokes` | yes | InvokeInst count |
| `FlatAddrspaceAccesses` | yes | Load/store on flat addrspace |
| **`FlopCount`** | **NO** | **Proposed `KernelCostInfo` addition** |
| **`MemBytes`** | **NO** | **Proposed `KernelCostInfo` addition** |
| **`ArithmeticIntensity`** | **NO** | **Proposed `KernelCostInfo` addition** |

**Existing validity gates that `KernelCostInfo` can reuse:**
- `IndirectCalls > 0` -- static FLOP counting is undecidable through function pointers
- `InlineAssemblyCalls > 0` -- inline assembly FLOPs are opaque to IR analysis
- `AllocasDyn > 0` -- trip-count-based MemBytes lower-bound uncertain when stack shape varies

### tritonBLAS roofline model inputs

Source: arXiv:2512.04226 (tritonBLAS), December 2025.

tritonBLAS selects among Triton-JIT-compiled GEMM configurations using:

```python
T_compute = flop_count / peak_flops_per_second
T_memory  = byte_count / peak_bandwidth_bytes_per_second
score     = 1.0 / max(T_compute, T_memory)   # higher = better
```

Result: 94.7% of exhaustive autotuning quality on 150,000 GEMM shapes (NVIDIA A100/H100).
AMD ROCm (MI250X): 91%.  Selection time: 50-80 microseconds in Triton JIT context.

**Exactly two per-kernel numbers are required:** `flop_count` and `byte_count`.
The device parameters (`peak_flops`, `peak_bw`) are measured once and cached.
This is precisely what `KernelCostInfo` would emit: `FlopCount` and `MemBytes`.

The libkdl cost model at `kdl.c:1061-1064` currently uses a weighted sum:
```c
double total = w.compute  * compute_time
             + w.memory   * memory_time
             + w.overhead * launch_overhead
             + w.locality * locality_score;
```
The tritonBLAS result implies this should be corrected to
`fmax(compute_time, memory_time)` to reach the 94.7% quality ceiling.  Both the
weighted-sum correction and the `KernelCostInfo`-fed `flops`/`bytes_total` population
are independent improvements that compose.

### `has_compute_profile` gate in libkdl

`kdl.c:1016`:
```c
if (!c->has_compute_profile) return 1e9;  /* fall back to priority */
```

`kdl.c:962-968` (JSON parsing from MTB contract):
```c
double ai = json_get_num(json, "arithmetic_intensity", -1);
if (ai > 0) {
    out->has_compute_profile = 1;
    out->flops = json_get_num(json, "flops", 0);
    out->bytes_total = json_get_num(json, "bytes_read", 0)
                     + json_get_num(json, "bytes_written", 0);
    out->arithmetic_intensity = ai;
}
```

`KernelCostInfo` remarks, consumed by `kdl-contract-gen`, would populate these three
fields automatically.  With `has_compute_profile = 1`, the roofline model activates.
The dispatch decision changes from priority-based to cost-model-based for every kernel
that has static FLOP/byte data -- without any runtime profiling or user annotation.

### `collectKernelLaunchBounds()` as TTI extension pattern

`llvm/include/llvm/Analysis/TargetTransformInfo.h` declares:
```cpp
virtual void collectKernelLaunchBounds(
    const Function &F,
    SmallVectorImpl<std::pair<StringRef, int64_t>> &LB) const;
```

KernelInfo calls this once per function:
```cpp
TheTTI.collectKernelLaunchBounds(F, KI.LaunchBounds);
```

Each backend (AMDGPU TTI, NVPTX TTI) overrides it to read target-specific function
attributes (`amdgpu-flat-work-group-size`, `nvvm.maxclusterrank`).

`KernelCostInfo` would add a parallel virtual method:
```cpp
virtual void collectKernelCostHints(
    const Function &F,
    KernelCostHintsInfo &Hints) const;
```

Where `KernelCostHintsInfo` carries:
- `FlopMultiplier` -- scale factor for target-specific wide instructions (e.g., MFMA
  on AMDGPU emits 64 FLOPs per instruction but is not a standard `fmul` opcode)
- `MemBytesMultiplier` -- cache reuse factor for targets with explicit L1/LDS models
- `ValidityOverride` -- target can force `HasComputeProfile = false` for architectures
  where IR-level counting is known to be unreliable

This follows the exact same extension pattern as `collectKernelLaunchBounds()`.

### NeuSight: three numbers suffice for cross-architecture transfer

Source: NeuSight (ASPLOS 2025), ACM DL: https://dl.acm.org/doi/10.1145/3669940.3707265.

NeuSight decomposes GPU kernel performance prediction into tile-granularity models
bounded by per-GPU roofline ceilings.  Hardware features used: `peak_flops`, `peak_bw`,
L1/L2 cache sizes.  Per-kernel features: tile shape, operation type, memory access
pattern.

**Result:** 2.3% mean error on H100 (not in training set); <9% mean error across all
evaluated GPUs.  Baseline (linear regression): 60.8% error.

NeuSight validates that FLOP count + memory bytes + device roofline parameters are
sufficient for near-optimal cross-architecture dispatch decisions.  The per-kernel FLOP
and byte counts that `KernelCostInfo` would emit are exactly the inputs NeuSight's
analytical bounding step requires.

**NeuSight AMD gap:** NeuSight is trained on NVIDIA-only hardware.  For AMD targets,
the analytical roofline (no ML) is the only published cross-vendor predictor.
`KernelCostInfo` outputs are target-independent; the roofline formula is vendor-agnostic.

### Static FLOP counting accuracy in the literature

Source: TACO 2021 baseline (arXiv:2001.07104, portable execution time prediction).

Static FLOP counting accuracy by regime:
- **Compute-bound kernels with known trip counts** (GEMM, Conv2D): exact agreement with
  dynamic hardware counters -- the loop bounds are compile-time constants and the
  instruction mix is regular.
- **Memory-bound kernels with input-variable access patterns** (SpMV, attention with
  variable sequence length): 52% MAPE for static analysis alone.

Implication for `KernelCostInfo`: the `FlopCountBound = "lower_bound"` / `"exact"`
distinction is essential.  When trip counts are unknown, the emitted `FlopCount` is a
lower bound; `ArithmeticIntensity` is a lower bound; `HasComputeProfile` should remain
`true` but downstream consumers must treat the values as estimates.  The dispatch
problem only requires regime identification (compute-bound vs. memory-bound) -- a
factor-of-2 error in FLOP count does not change the regime classification for a kernel
at 10x the ridge point.

---

## Feasibility

**High.  Three implementation paths, each de-risked independently.**

### Path A: `kdl-contract-gen` bridge (zero new LLVM passes, prototype today)

1. Compile GPU kernel with:
   ```
   clang -foffload-lto -Rpass=kernel-info \
         -pass-remarks-output=remarks.yml kernel.c
   ```
2. Run `kdl-contract-gen remarks.yml` (~200-LOC Python YAML parser) to extract
   `AllocasStaticSizeSum`, `omp_target_thread_limit`, `FlatAddrspaceAccesses` per
   kernel name.
3. User annotates `flops` and `bytes_total` manually (or from offline profiling) in the
   MTB contract JSON.
4. Result: `min_shared_mem_kb`, `flat_addrspace_risk`, `omp_target_thread_limit`
   fields auto-populated; `has_compute_profile = 1` when user provides FLOP/byte data.

This path proves the end-to-end pipeline architecture for the poster with zero LLVM
changes.  Deliverable: `kdl-contract-gen` tool + dispatch accuracy comparison table.

### Path B: `KernelCostInfo` upstream pass (~700 LOC new C++)

| Component | Effort | Infrastructure used |
|---|---|---|
| `KernelCostInfoAnalysis` struct | 50 LOC | Standard LLVM analysis boilerplate |
| FLOP counting walk (`BasicBlock` iterator) | 150 LOC | `Instruction::getOpcode()`, `VectorType` |
| SCEV trip-count scaling | 100 LOC | `ScalarEvolution::getSmallConstantTripCount()` |
| Memory byte counting | 80 LOC | `DataLayout::getTypeStoreSize()` |
| `TTI.collectKernelCostHints()` virtual | 50 LOC | Mirrors `collectKernelLaunchBounds()` |
| `KernelCostInfoPrinter` pass (remark emission) | 100 LOC | `OptimizationRemark` framework |
| AMDGPU TTI override (MFMA correction) | 80 LOC | AMDGPU intrinsic inspection |
| Tests (`llvm/test/Analysis/KernelCostInfo/`) | 90 LOC | FileCheck pattern matching |

**Total: ~700 LOC.** Within scope of a single GSoC or LFX Mentorship project.
All infrastructure (SCEV, DataLayout, TTI virtual dispatch, OptimizationRemark) is
already upstream and well-documented.

### Path C: PGO-guided annotation (ground-truth values)

Use ORNL's GPU PGO infrastructure (PR #93365, PR #94268, McDonough/Denny/Doerfert,
IWOMP 2025) to instrument a kernel for one reference input, measure actual FLOPs and
bytes via hardware performance counters, then write the measured values into the MTB
contract JSON as `flops` and `bytes_total`.  The `has_compute_profile` flag acts as the
"profdata present" marker -- an exact analog of LLVM's `!prof` mechanism for branch
weights.

This path provides ground-truth values for kernels with irregular access patterns where
static counting produces lower bounds.  It composites with Paths A and B: static
counting fills in Path B values where available; PGO overrides them where the static
bound is too loose.

---

## Upstream Path

**Conservative estimate: 12-18 months to upstream acceptance.  Staged milestones.**

### Stage 1 -- Poster + tooling (0-6 months, no upstream changes)

- Present `kdl-contract-gen` at LLVM Dublin 2026 as proof-of-concept.
- Demonstrate: KernelInfo remarks YAML consumed, `AllocasStaticSizeSum` mapped to
  `min_shared_mem_kb`, dispatch behavior change shown on GTX 1650 + CPU.
- Publish the FLOP/byte counting gap as a named limitation of KernelInfo -- create
  community awareness that the companion pass slot is open.

### Stage 2 -- RFC on discourse.llvm.org (3-6 months)

Post an RFC: *"KernelCostInfo: a companion analysis pass for static FLOP and memory byte
counting in GPU kernels."*  Key anchors:

- **Prior art in KernelInfo itself:** the `collectKernelLaunchBounds()` TTI extension
  pattern is the exact template; no new architectural concepts.
- **Companion to KernelInfo, not extension:** avoids scope creep rejection from Denny.
  Two passes registered sequentially; KernelInfo remains narrow by design.
- **MLIR cost model RFC alignment:** Intel PCL's 2024 RFC (discourse.llvm.org/t/rfc-
  target-description-and-cost-model-in-mlir/76990) identified the lack of a standard
  cross-target cost model in MLIR.  `KernelCostInfo` at the LLVM IR level is a
  complementary contribution -- it feeds the ML compiler level that the MLIR RFC targets.
- **Stakeholder alignment:**
  - Joel E. Denny (ORNL) -- KernelInfo author, natural reviewer; the companion-not-
    extension framing respects his design intent.
  - Joseph Huber (AMD) -- liboffload "ld.so for GPU" framing; cost-aware dispatch is a
    natural extension.
  - ORNL GPU PGO team (McDonough, Doerfert) -- PGO path (C) complements static counting;
    shared motivation for contract-quality metrics.

### Stage 3 -- Upstream pass implementation (6-12 months)

1. `llvm/include/llvm/Analysis/KernelCostInfo.h` -- analysis result struct + printer pass
2. `llvm/lib/Analysis/KernelCostInfo.cpp` -- implementation
3. `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp` -- register after `KernelInfoPrinter`
4. `llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp` -- same registration
5. TTI virtual method addition (coordinated with TTI owners)
6. `llvm/test/Analysis/KernelCostInfo/` -- test suite mirroring KernelInfo test structure
7. `llvm/docs/KernelCostInfo.rst` -- documentation

### Stage 4 -- `kdl-contract-gen` integration (12-18 months)

Once `KernelCostInfo` remarks are available in `-pass-remarks-output` YAML, extend
`kdl-contract-gen` to parse `FlopCount`, `MemBytes`, `ArithmeticIntensity`, and emit
them directly into MTB contract JSON.  The `has_compute_profile = 1` gate in libkdl
activates the roofline model automatically.  This closes the end-to-end loop:
compiler analysis output feeds runtime dispatch decision with zero user annotation.

**Upstream acceptance risks:**
- TTI virtual method additions require coordinated review across all backend owners
  (moderate friction, precedented by `collectKernelLaunchBounds()`).
- SCEV-based trip count analysis may produce `lower_bound` for majority of real GPU
  kernels -- managing community expectations about static accuracy is important.
- KernelInfo is "done" from ORNL's perspective; the companion framing must be made
  clearly additive, not a critique of KernelInfo's scope.

---

## Scores

| Criterion | Score | Justification |
|---|---|---|
| **Novelty** | **8/10** | No upstream LLVM pass computes FLOP/byte counts from GPU kernel IR and emits them as machine-readable remarks. The companion-not-extension framing is architecturally clean and is a genuine gap in the KernelInfo ecosystem. |
| **Feasibility** | **9/10** | All infrastructure (SCEV, DataLayout, TTI virtual dispatch, OptimizationRemark) is upstream and mature. Path A (kdl-contract-gen bridge) is demonstrable before the poster deadline with zero LLVM changes. Path B (~700 LOC) is a single mentorship project. |
| **Evidence Strength** | **9/10** | tritonBLAS (94.7% quality with two numbers), NeuSight (2.3% error cross-architecture), KernelInfo (PR #102944, upstream, source-verified), kdl.c cost model (live prototype), TACO accuracy characterization, SCEV/DataLayout/TTI all documented in LLVM source. All primary sources. |
| **Impact** | **8/10** | Activates the roofline cost model in libkdl (currently dead-pathed by `has_compute_profile = false`). Applicable to any runtime dispatcher consuming GPU binaries -- liboffload policy layers, IREE HAL, custom ML serving runtimes. Upstream path creates permanent ecosystem infrastructure. |
| **Upstream viability** | **7/10** | TTI extension pattern is precedented. Companion-not-extension framing avoids scope-creep rejection. SCEV accuracy limitations must be clearly communicated. KernelInfo author engagement is critical. |
| **Composite** | **8.2/10** | |

---

## Pitch

Three sentences for the poster panel:

The LLVM KernelInfo pass runs at LTO time for every GPU kernel but does not count
floating-point operations or memory bytes -- the two numbers that tritonBLAS proves are
sufficient for 94.7% of exhaustive autotuning quality.  We propose `KernelCostInfo`, a
companion IR pass that counts `fadd`/`fmul`/`fma` instructions and load/store bytes,
scales by ScalarEvolution trip counts, and gates on KernelInfo's existing
`IndirectCalls == 0` validity flag -- following the exact `collectKernelLaunchBounds()`
TTI extension pattern already used by KernelInfo.  The emitted `FlopCount`, `MemBytes`,
and `ArithmeticIntensity` remarks feed libkdl's `kdl-contract-gen` tool to auto-
populate `kdl_contract.has_compute_profile`, activating the roofline cost model for
cross-vendor GPU kernel dispatch without any runtime profiling.

**Poster panel structure:**

1. **Gap diagram**: KernelInfo remark list with `FlopCount` / `MemBytes` visually absent.
   Arrow from `has_compute_profile = false` to "priority-only dispatch" in libkdl.
2. **`KernelCostInfo` design**: opcode walk, SCEV trip-count scaling, TTI extension hook,
   validity gate composition with `IndirectCalls`.
3. **Remark output example**: `sgemm_kernel`: FlopCount = 2097152, MemBytes = 786432,
   AI = 2.667.  Before/after: priority dispatch vs. roofline dispatch selecting correctly.
4. **Quality bar**: tritonBLAS 94.7% result; `fmax(T_compute, T_memory)` formula.
5. **Accuracy qualification**: `FlopCountBound = exact` for GEMM with constant bounds;
   `= lower_bound` for sparse kernels.  Regime classification accuracy (not exact latency).
6. **Upstream path**: companion pass → RFC → TTI extension → `kdl-contract-gen` integration.

---

## Risks

1. **SCEV symbolic trip counts are common in production GPU kernels.** Matrix dimension
   `M`, `N`, `K` are typically runtime parameters, making the loop trip count symbolic.
   `FlopCount` will be `lower_bound` for most real GEMM kernels unless the outer loop
   bounds are constant or unrolled.  Mitigation: for the poster, demonstrate on a fixed-
   dimension kernel (M=N=K=256 SGEMM) where counts are exact; acknowledge the limitation
   explicitly.  The `FlopCountBound` remark field exists precisely for this qualification.

2. **MFMA on AMDGPU emits 64 FLOPs per instruction but appears as an intrinsic, not a
   standard `fmul`.** An AMDGPU kernel heavily using `llvm.amdgcn.mfma.*` intrinsics
   will have its FLOPs severely undercounted by naive `fadd`/`fmul` scanning.
   Mitigation: the `TTI.collectKernelCostHints()` extension allows the AMDGPU TTI to
   inject a `FlopMultiplier` correction.  This is a Stage 3 upstream task; the poster
   demo can use NVPTX where standard opcodes dominate.

3. **`IndirectCalls == 0` gate is necessary but insufficient.** Even with only direct
   calls, inlined callees from external modules (where inline status is unknown at LTO
   entry) may have incorrect counts.  Mitigation: `DirectCallsToDefinedFunctions` from
   KernelInfo ensures all callee bodies are visible; gate on
   `IndirectCalls == 0 && DirectCalls == DirectCallsToDefinedFunctions`.

4. **`isRequired() = true` compile-time cost.** KernelInfo always runs; adding
   `KernelCostInfo` immediately after doubles the per-function analysis overhead at LTO
   time.  For large kernel libraries (CUTLASS), this is measurable.  Mitigation: make
   `KernelCostInfo` opt-in via `-passes=kernel-cost-info` initially, not registered by
   default.  Only promote to default registration after performance impact is measured.

5. **Upstream scope competition.** The GPU dialect cleanup RFC (#88170, September 2025)
   is ongoing; LLVM/Offload infrastructure is in active flux.  An LLVM IR-level pass
   proposal (not MLIR) is isolated from this churn -- KernelInfo itself is unaffected by
   MLIR GPU dialect changes.  This is a de-risking advantage of the LLVM IR approach.

6. **NeuSight AMD transfer gap.** All NeuSight results are NVIDIA-only.  The roofline
   formula is vendor-agnostic but per-kernel FLOP/byte counts from `KernelCostInfo` must
   be validated on AMD targets.  This is a poster limitation to state explicitly; the
   AMDGPU MFMA correction (Risk 2) is the primary source of inaccuracy.

---

## Cross-References

- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-08-kernel-info-pass.md`
  -- complete KernelInfo field analysis, alignment table with `kdl_contract`, ORNL PGO context
- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-03-dispatch-cost-attr.md`
  -- three-layer `#kdl.cost` protocol; OffloadBinary embedding path; MLIR-level complement
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/directions/03-roofline-cross-vendor-cost-model.md`
  -- tritonBLAS 94.7%, NeuSight, `fmax` correction, cross-vendor roofline design
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-cost-models.md`
  -- cuBLAS three-tier dispatch, CUTLASS occupancy model, Fasor transferable cost model
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-03-cost-model-selection.md`
  -- tritonBLAS detail, NeuSight ASPLOS 2025, SparseX, Seer decision-tree selector
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` lines 1013-1088
  -- `kdl_estimate_cost_weighted()` (the runtime consumer of `flops` / `bytes_total`)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` lines 112-122
  -- `kdl_contract` struct (`flops`, `bytes_total`, `arithmetic_intensity`, `has_compute_profile`)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` line 1016
  -- `has_compute_profile` gate (exact injection point)
- `llvm/lib/Analysis/KernelInfo.cpp` -- upstream source (326 lines, commit `18f8106f`)
- `llvm/include/llvm/Analysis/TargetTransformInfo.h`
  -- `collectKernelLaunchBounds()` TTI extension pattern to replicate
- `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp:1115`
  -- `FPM.addPass(KernelInfoPrinter(this))` registration point for companion pass

---

*Sources:*
- [KernelInfo LLVM documentation](https://llvm.org/docs/KernelInfo.html)
- [llvm/llvm-project PR #102944](https://github.com/llvm/llvm-project/pull/102944)
- [Profile Generation for GPU Targets — ORNL, IWOMP 2025 (McDonough, Denny, Doerfert)](https://impact.ornl.gov/en/publications/profile-generation-for-gpu-targets/)
- [tritonBLAS (arXiv:2512.04226)](https://arxiv.org/abs/2512.04226)
- [NeuSight — ASPLOS 2025 (ACM DL 10.1145/3669940.3707265)](https://dl.acm.org/doi/10.1145/3669940.3707265)
- [MLIR cost model RFC (LLVM Discourse 2024)](https://discourse.llvm.org/t/rfc-target-description-and-cost-model-in-mlir/76990)
- [TACO 2021 portable execution time prediction (arXiv:2001.07104)](https://arxiv.org/abs/2001.07104)
- `kdl.c` primary source: `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c`
