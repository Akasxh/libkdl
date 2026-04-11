# Wave 08: LLVM KernelInfo Pass — GPU Code Analysis by Joel Denny (ORNL)

**Research question:** Can the upstream LLVM KernelInfo IR pass feed libkdl's capability
contract fields (`flops`, `bytes_total`, `arithmetic_intensity`, `min_shared_mem_kb`,
`omp_target_thread_limit`) at compile time, creating a static-analysis pipeline from
source to dispatch contract?

---

## Finding

The KernelInfo pass is fully upstream in LLVM as of January 29, 2025 (merged PR #102944,
authored by Joel E. Denny at ORNL). The PR was opened August 12, 2024, placing its active
development squarely in the 2024 GPU/offloading discourse. It is an `FunctionPass` that
runs at the end of LTO and emits optimization remarks via `-Rpass=kernel-info`.

The pass does **not** produce arithmetic intensity, FLOP counts, or shared memory byte
totals. It produces a structurally adjacent but distinct set of remarks oriented toward
identifying "bad code patterns" — specifically: stack-frame size (allocas), call graph
properties (indirect calls, inline assembly), flat address space accesses, and launch
bounds. The overlap with libkdl's capability contract is real but partial.

---

## Evidence

### Upstream status

- **Merged commit:** `18f8106f` — "Joel E. Denny, 2025-01-29T17:40:19Z"
  `[KernelInfo] Implement new LLVM IR pass for GPU code analysis (#102944)`
- **PR opened:** 2024-08-12 — active during 2024 LLVM GPU/offloading workshop cycle
- **Subsequent fixes by non-authors** (same day, Jan 29 2025):
  - `953354c7` — Benjamin Kramer: "Fix layering violation, Analysis cannot depend on Passes"
  - `57f17319` — Benjamin Kramer: "Remove unused include"
  - `5dae05f6` — Simon Pilgrim: "Fix MSVC signed/unsigned mismatch"
- **March 2026 update:** `080bc257` — Alexis Engelke: "Remove *WithoutDebug (NFCI)"
- **Documentation:** `llvm/docs/KernelInfo.rst`, linked from `llvm/docs/Passes.rst`
- **Tests:** `llvm/test/Analysis/KernelInfo/` (allocas.ll, calls.ll, flat-addrspace/,
  launch-bounds/, linkage.ll, openmp/)

### Source files (upstream LLVM)

- `llvm/include/llvm/Analysis/KernelInfo.h` — defines `KernelInfoPrinter : PassInfoMixin`
- `llvm/lib/Analysis/KernelInfo.cpp` — 326 lines total
- `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp` line 73 includes `KernelInfo.h`;
  line 1115 registers it: `FPM.addPass(KernelInfoPrinter(this))` at end of LTO

### Metadata fields emitted (complete list)

All fields are emitted as `OptimizationRemark` entries keyed by `"kernel-info"`:

| Remark key | Type | Description |
|---|---|---|
| `ExternalNotKernel` | bool (0/1) | Function has external linkage but no kernel calling conv |
| `omp_target_num_teams` | int64 | OpenMP launch bound: number of teams |
| `omp_target_thread_limit` | int64 | OpenMP launch bound: max threads per team |
| `maxclusterrank` | int64 | NVPTX: max cluster rank (from `nvvm.maxclusterrank`) |
| `maxntidx`, `maxntidy`, `maxntidz` | int64 | NVPTX: max thread block dimensions |
| `amdgpu-max-num-workgroups[0..2]` | int64 | AMDGPU: max workgroups per dimension |
| `amdgpu-flat-work-group-size[0..1]` | int64 | AMDGPU: flat work-group size min/max |
| `amdgpu-waves-per-eu[0..1]` | int64 | AMDGPU: occupancy hint (waves per EU min/max) |
| `Allocas` | int64 | Total alloca instructions in function |
| `AllocasStaticSizeSum` | int64 | Bytes of compile-time-determinable stack allocations |
| `AllocasDyn` | int64 | Number of dynamic-size allocas |
| `DirectCalls` | int64 | Direct call instructions |
| `IndirectCalls` | int64 | Indirect call instructions (function pointers) |
| `DirectCallsToDefinedFunctions` | int64 | Direct calls to module-local functions |
| `InlineAssemblyCalls` | int64 | Calls to inline assembly |
| `Invokes` | int64 | InvokeInst count |
| `FlatAddrspaceAccesses` | int64 | Load/store/atomic ops on flat (non-specialized) address space |

Per-occurrence remarks are emitted inline for each alloca, call, and flat-addrspace
access with source location. Summary properties are emitted once per function at the
function's debug location.

**Architecture-specific launch bounds** are collected via
`TheTTI.collectKernelLaunchBounds(F, KI.LaunchBounds)` where TTI is the target's
`TargetTransformInfo` implementation, making the pass extensible to new backends.

### Remark output format (from test files)

```
remark: test.c:10:0: in artificial function 'test', omp_target_num_teams = 100
remark: test.c:10:0: in artificial function 'test', omp_target_thread_limit = 101
remark: test.c:10:0: in artificial function 'test', maxntidx = 210
remark: test.c:13:0: in artificial function 'h', Allocas = 7
remark: test.c:13:0: in artificial function 'h', AllocasStaticSizeSum = 28
remark: test.c:13:0: in artificial function 'h', AllocasDyn = 2
```

AMDGPU example:
```
remark: test.c:10:0: in artificial function 'all', amdgpu-waves-per-eu[0] = 2
remark: test.c:10:0: in artificial function 'all', amdgpu-waves-per-eu[1] = 9
remark: test.c:10:0: in artificial function 'all', amdgpu-flat-work-group-size[0] = 210
```

### libkdl capability contract fields (kdl.c)

From `kdl.c` lines 112–122 and `kdl_parse_contract()` at lines 940–970:

```c
typedef struct {
    uint32_t target;
    uint32_t min_arch_numeric;
    uint32_t min_shared_mem_kb;   /* checked in kdl_contract_check() */
    uint32_t min_driver_version;
    uint32_t min_vram_mb;
    int      has_compute_profile;
    double   flops;               /* used in roofline cost model */
    double   bytes_total;         /* bytes_read + bytes_written */
    double   arithmetic_intensity;
} kdl_contract;
```

The roofline cost model at lines 1013–1064 uses `flops`, `bytes_total`, and device
`peak_tflops_f32` / `peak_bw_gbps` to estimate kernel execution time and rank variants.

---

## Alignment Analysis: KernelInfo vs. libkdl Contract

### Direct matches (KernelInfo can populate these fields today)

| libkdl contract field | KernelInfo remark | Notes |
|---|---|---|
| `min_shared_mem_kb` | `AllocasStaticSizeSum` | **Partial.** Static alloca sum is a proxy for register-file / shared-mem pressure in languages that lower shared memory to allocas. Accurate for AMDGPU LDS, inexact for NVIDIA shmem. Not a direct measurement. |
| `omp_target_thread_limit` | `omp_target_thread_limit` | **Direct.** Corresponds to `maxThreadsPerBlock` or AMDGPU `flat-work-group-size`. Libkdl does not currently expose this field but it could constrain `block_size` matching. |
| Target arch flags | `maxntidx` / `amdgpu-waves-per-eu` | **Indirect.** Launch bound hints inform occupancy, which informs whether a kernel will saturate the device. Could feed `min_arch_numeric` or occupancy pre-filter. |

### Missing from KernelInfo (must come from elsewhere)

| libkdl contract field | What provides it | Status |
|---|---|---|
| `flops` | Dynamic profiling, LLVM FLOP counting pass (not yet upstream), or static instruction counting | **Not in KernelInfo.** No FP operation analysis in current pass. |
| `bytes_total` | Memory access analysis (e.g., MemorySSA, SCEV-based trip count, or profiling) | **Not in KernelInfo.** `FlatAddrspaceAccesses` counts access instructions but not bytes. |
| `arithmetic_intensity` | Derived from flops / bytes_total | **Not in KernelInfo.** |
| `min_shared_mem_kb` (actual) | AMDGPU: `.group_segment_fixed_size` in metadata; NVPTX: `nvvm.annotations` | **Not in KernelInfo.** The pass does not inspect GPU-dialect shared memory declarations. |

### Critical gap

KernelInfo is intentionally a "bad code pattern detector," not a roofline input
generator. The PR description (jdenny-ornl, PR #102944) states explicitly: "The ultimate
goal of these statistics [is] to help identify bad code patterns and ways to mitigate
them." The `amdgpu-waves-per-eu` and `omp_target_thread_limit` values are **hints set by
the programmer**, not analytically derived. KernelInfo reads and reports them; it does not
compute occupancy.

The companion ORNL paper "Profile Generation for GPU Targets" (McDonough, Denny, Doerfert,
IWOMP 2025, LNCS vol. 16123, pp. 99–113) targets runtime PGO, not static contract
generation. The roofline keyword appears only in the abstract keywords list; the
implementation is profiling-based, not static-analysis-based.

---

## Proposed Integration with libkdl

Despite the gap, KernelInfo represents the closest upstream LLVM touchpoint for a
compile-time contract generator. A two-layer integration is viable:

### Layer 1: KernelInfo-to-MTB contract emitter (feasible today)

A tool (`kdl-contract-gen`) wrapping `opt -passes=kernel-info -pass-remarks-output=...`
can extract remarks at LTO time and populate the following contract JSON fields:

```json
{
  "target": "nvptx",
  "omp_target_thread_limit": 256,
  "min_shared_mem_kb": 16,
  "flat_addrspace_accesses": 3
}
```

The `flat_addrspace_accesses` count is a dispatch-relevant signal: high flat-addrspace
access counts indicate kernels that degrade on hardware without flat address space
support, making CPU fallback more attractive. libkdl does not currently use this but
could add a `flat_addrspace_risk` field to `kdl_contract`.

`AllocasStaticSizeSum` maps to a lower-bound on stack/shared memory footprint and can
gate the `min_shared_mem_kb` contract field conservatively.

### Layer 2: Roofline field population (requires additional passes)

To populate `flops` and `bytes_total`, two options exist:

1. **LLVM instruction-counting pass** — count `fadd`/`fmul`/`fma` instructions and
   load/store bytes. Accurate for unrolled loops; fails for indirect recursion (blocked
   by `InlineAssemblyCalls > 0` or `IndirectCalls > 0` — both already reported by
   KernelInfo). KernelInfo's call-type summary can be used as a pass/fail gate before
   attempting static FLOP counting.

2. **PGO-fed contract** — use the ORNL PGO infrastructure (PR #93365, PR #94268) to
   collect runtime FLOP/byte counters on a reference input and embed them as static
   annotations in the MTB. This creates a "training run" path vs. "zero-profiling" path.

### Integration architecture

```
Source (OpenMP/SYCL/HIP)
  |
  v
clang -foffload-lto -Rpass=kernel-info -pass-remarks-output=remarks.yml
  |
  v
kdl-contract-gen remarks.yml   <-- NEW TOOL
  |
  +-- reads: omp_target_thread_limit, AllocasStaticSizeSum,
  |          FlatAddrspaceAccesses, amdgpu-waves-per-eu,
  |          ExternalNotKernel (skip non-kernel functions)
  |
  v
contract.json  -->  MTB variant header
  |
  v
kdl_load() + kdl_dispatch()  -->  runtime roofline ranking
```

This pipeline requires zero runtime instrumentation and zero hardware availability at
build time. The only new component is `kdl-contract-gen`, a ~200-LOC Python or C
remark-YAML parser.

---

## Risks and Concerns

1. **`AllocasStaticSizeSum` is not shared memory.** On NVIDIA, shared memory is declared
   via `@llvm.nvvm.annotations` metadata or `addrspace(3)` globals, not as allocas.
   KernelInfo does not walk global variables. On AMDGPU, LDS is typically lowered to
   `addrspace(3)` allocas, so the approximation is better there. A separate metadata
   walker is needed for NVIDIA shared memory.

2. **No occupancy calculation.** `amdgpu-waves-per-eu` is a programmer-supplied hint
   stored as a function attribute; it is not computed from register pressure.
   The actual achieved occupancy requires the AMDGPU CodeGen pipeline
   (`AMDGPUAttributorPass`, `AMDGPUPerfHintAnalysis`) which runs after KernelInfo.
   These are post-LTO target passes, not accessible at the LTO boundary where KernelInfo
   runs.

3. **NVPTX register count is absent.** NVPTX register allocation happens in the PTX
   backend; it is not visible in LLVM IR and therefore cannot be reported by any IR
   pass. `ptxas --register-usage-level` exists but is a separate tool outside LLVM.

4. **Pass is `isRequired() = true`.** This means it cannot be skipped silently;
   it always runs when registered. For production builds this is a measurable compile-time
   cost. The cost is bounded (single pass over each GPU function), but for large kernel
   libraries (e.g., CUTLASS) it adds per-function remark emission overhead.

5. **Upstream status is stable but thin.** Only one substantive commit from the author;
   the three follow-up commits were quick layering fixes by other developers. No
   follow-on patches have extended the metric set. This suggests it is "done" from
   ORNL's immediate use case (bad-pattern detection) but not under active expansion.
   Future extension (e.g., FLOP counting) would require a new PR.

6. **Workshop attribution is unclear.** The task prompt references a "2024 LLVM
   GPU/Offloading Workshop" presentation. The PR was opened August 2024 and the
   LLVM/Offload pre-workshop was held October 22, 2024 (confirmed agenda listing
   "LLVM/Offload — Languages, Backends, and Features"). No slides by Denny appear in
   the public `llvm.org/devmtg/2024-10/slides/` directory. The pass may have been
   discussed informally or the slides were not published. The PR review thread on
   GitHub is the authoritative public record.

---

## Related

- `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp` — registers KernelInfo at LTO end
  for AMDGPU; NVPTX uses the same path via the generic offload LTO pipeline
- `llvm/include/llvm/Analysis/TargetTransformInfo.h` — `collectKernelLaunchBounds()`
  virtual method called by KernelInfo to get target-specific launch bounds
- ORNL LLVM fork: `code.ornl.gov/llvm-doe/llvm-project` — Denny's development base;
  `llvm/docs/Passes.rst` in that fork links KernelInfo
- PR #93365, PR #94268 — companion GPU PGO / profraw generation work by the same team
  (McDonough, Denny, Doerfert), IWOMP 2025; runtime complement to KernelInfo's static view
- `kdl.c` lines 1008–1064 — libkdl roofline cost model; `has_compute_profile` gate at
  line 1016 (`if (!c->has_compute_profile) return 1e9`) is the exact injection point
  where KernelInfo-derived contract data would change dispatch behavior

---

## Summary Assessment

KernelInfo is a confirmed, upstream, production-quality pass authored by ORNL that
targets the same compile-time GPU analysis space as libkdl's contract generator. It is
**a natural starting point but not a complete solution.** The concrete value for libkdl is:

- `omp_target_thread_limit` and `amdgpu-waves-per-eu` feed occupancy-aware pre-filtering
- `AllocasStaticSizeSum` provides a lower-bound on memory footprint for `min_shared_mem_kb`
- `FlatAddrspaceAccesses > 0` is a dispatch-relevant portability signal (missing from
  current libkdl contract schema)
- `IndirectCalls > 0` or `InlineAssemblyCalls > 0` flags kernels where static FLOP
  counting would be invalid — a correctness gate for the contract generation pipeline

The `flops` / `bytes_total` / `arithmetic_intensity` fields that drive libkdl's roofline
ranking remain outside KernelInfo's scope and require either PGO instrumentation or a
separate static FLOP-counting IR pass. KernelInfo is the correct **upstream hook point**
for a contract emitter, but it is not the complete contract emitter itself.

---

*Sources:*
- [KernelInfo LLVM 22.0.0git documentation](https://llvm.org/docs/KernelInfo.html)
- [llvm/llvm-project PR #102944](https://github.com/llvm/llvm-project/pull/102944)
- [llvm/lib/Analysis/KernelInfo.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Analysis/KernelInfo.cpp)
- [Profile Generation for GPU Targets — ORNL (McDonough, Denny, Doerfert), IWOMP 2025](https://impact.ornl.gov/en/publications/profile-generation-for-gpu-targets/)
- [LLVM/Offload Workshop 2024 agenda (llvm.swoogo.com)](https://llvm.swoogo.com/2024devmtg/agenda)
- [Joel E. Denny ORNL staff profile](https://www.ornl.gov/staff-profile/joel-e-denny)
