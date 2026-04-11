# Topic 13: Dynamic Hardware Introspection Dialect for MLIR

**Topic ID:** 13
**Config key:** `hw-introspection-dialect`
**Persona:** MLIR dialect designer / heterogeneous runtime architect
**Date:** 2026-04-07
**Research depth:** Exhaustive — IREE HAL docs, CUDA/HIP/Level Zero API refs, MLIR RFC tracker,
GPU dialect source, 8+ wave files + 4 literature files + kdl.c primary source cross-referenced

---

## Gap

Every major GPU runtime exposes a rich hardware capability query API:

| Runtime | Primary Query API | Key Properties Available |
|---------|------------------|--------------------------|
| CUDA | `cuDeviceGetAttribute` / `cudaGetDeviceProperties` | compute capability (major.minor), `multiProcessorCount`, `warpSize`, `sharedMemPerBlock`, `maxThreadsPerBlock`, cooperative launch, managed memory, ~100 total fields |
| HIP (ROCm) | `hipDeviceGetAttribute` / `hipGetDeviceProperties` | `gcnArchName` (e.g., `"gfx90a"`), `multiProcessorCount`, `warpSize` (= 64 on GCN/RDNA), `sharedMemPerBlock`, `cooperativeLaunch`, `managedMemory` |
| Level Zero | `zeDeviceGetProperties` + `zeDeviceGetComputeProperties` | `type` (GPU/CPU/FPGA), `vendorId`, `deviceId`, `numSlices`, `numSubslicesPerSlice`, `numEUsPerSubslice`, `maxGroupSizeX/Y/Z`, max shared local memory |
| OpenCL | `clGetDeviceInfo` | `CL_DEVICE_MAX_COMPUTE_UNITS`, `CL_DEVICE_MAX_WORK_GROUP_SIZE`, `CL_DEVICE_GLOBAL_MEM_SIZE`, `CL_DEVICE_EXTENSIONS` string |
| Vulkan | `vkGetPhysicalDeviceProperties` + `vkGetPhysicalDeviceFeatures2` | `maxComputeWorkGroupSize`, `limits.maxComputeSharedMemorySize`, extension feature structs via `pNext` chain |

IREE's HAL dialect adds a thin layer above these: `hal.device.query` takes a `key` string (e.g.,
`"cuda.compute_capability"`, `"hal.executable.format"`) and returns an `(i1, type)` pair — a
boolean validity flag plus the queried value. This is used in variant condition ops to gate
executable loading at module-load time:

```mlir
%ok, %value = hal.device.query<%device : !hal.device> key("cuda.compute_capability") :
              i1, i32 = -1 : i32
%is_sm90 = arith.cmpi sge, %value, %c90 : i32
```

But `hal.device.query` is IREE-private — it lives in `iree/compiler/Dialect/HAL/IR/HALOps.td`,
not in mainline `mlir/Dialect/GPU/`. There is no equivalent op in the upstream MLIR `gpu`
dialect. As a result, any MLIR-based compilation pipeline that wants to express
"compile kernel variant A if compute_capability >= 90, variant B otherwise" must either:

1. Lower to IREE HAL (pulling in the entire IREE stack), or
2. Implement ad-hoc runtime detection in C/C++ outside the MLIR IR boundary, making the
   query logic invisible to the compiler and unoptimizable (no query hoisting, no CSE, no DCE)

The structural gap is confirmed by two direct observations:

**Observation 1:** MLIR's `gpu` dialect has no op that queries the executing device's
capabilities at runtime. The dialect defines `gpu.func`, `gpu.launch`, `gpu.module`,
`gpu.binary`, and `gpu.launch_func` — but zero introspection ops. The closest upstream
mechanism is `#gpu.select_object`, which resolves selection at **compile time** by index
and emits a single hardcoded binary blob with no runtime detection code whatsoever
(`mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp`).

**Observation 2:** The RFC "An MLIR Dialect for Distributed Heterogeneous Computing"
(discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960,
June 2025, IIT Madras PACE Lab) proposes compile-time target annotation (`schedule` + `task`
with static `target` attributes) but explicitly does **not** address runtime hardware queries.
The RFC assumes target is known at compile time — which is precisely the assumption that
breaks on heterogeneous deployment where the same artifact must run on NVIDIA, AMD, and Intel
GPUs depending on what is available.

**Consequence for runtime dispatch:** Any system that defers vendor/architecture selection to
runtime (libkdl, liboffload policy layers, IREE's planned Phase 2 of Issue #12230, Triton's
per-kernel JIT gating) must issue hardware queries in unstructured host C code. These queries
cannot be analyzed, hoisted, or eliminated by the MLIR compiler. On the CUDA side,
`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ...)` costs ~0.5 µs
per call; on a hot dispatch path that selects among N variants per kernel launch, this is a
measurable overhead. On Level Zero, `zeDeviceGetProperties` is a synchronous blocking call
that must not be issued per-dispatch — yet without an MLIR-level introspection op, no
compiler pass can identify and hoist these calls.

**IREE's open issues confirm the gap is recognized but unsolved:**
- Issue #50 (2019, OPEN): "Add target configuration to compiler for specifying assumed/required
  features" — six years open, the runtime-side capability query is specifically called out as
  unsolved
- Issue #12230 (2023, OPEN, P2): Phase 3 of multi-versioning plan — "query device parallelism
  at runtime and compute tile/workgroup sizes dynamically" — explicitly blocked by "cross-vendor
  parallelism querying extensions is non-existing for Vulkan right now"
- Issue #15334 (2023, OPEN): "trinary capability state (present / not present / unknown)" for
  SPIR-V capabilities not implemented; the query mechanism is missing from the HAL side too

---

## Proposal

**Title:** `hw.query_capability` — A Vendor-Neutral MLIR Op for Runtime Hardware Introspection
That Survives Lowering

**One-sentence pitch:** Add a small `hw` dialect with `hw.query_capability` ops that express
runtime device property queries in MLIR IR, enable compiler-level hoisting and CSE of those
queries, and lower to the correct vendor-specific API call (CUDA/HIP/Level Zero/Vulkan) via the
existing `OffloadingLLVMTranslationAttrInterface` — giving `gpu.select_variant` and any other
multi-variant dispatch mechanism a typed, analyzable introspection substrate.

### The IR Shape

A new lightweight dialect `hw` (Hardware Query) with two ops:

```mlir
// Query a scalar device property.
// key is an enum-typed attribute (not a raw string) for type safety.
// Returns (valid: i1, value: !type) — same success/value pair as IREE's hal.device.query.
%ok, %sm = hw.query_capability #hw.key<cuda_compute_capability_major> : i1, i32

// Query a boolean feature flag.
%ok, %has_tf32 = hw.query_capability #hw.key<cuda_has_tf32> : i1, i1

// Conditional dispatch: branch on capability at runtime.
cf.cond_br %has_tf32, ^use_tf32_kernel, ^use_f32_kernel
```

The key attribute is an enum rather than an opaque string, enabling:
- TableGen-driven type checking (wrong key → compile error, not runtime failure)
- Exhaustiveness checking in lowering patterns (new key → compiler diagnostic if not handled)
- IDE tooling (autocomplete, hover docs)

**Standard key set (initial):**

| Key | Type | CUDA source | HIP source | Level Zero source |
|-----|------|-------------|------------|-------------------|
| `cuda_compute_capability_major` | i32 | `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` | `hipDeviceProp_t.major` | N/A (Intel-only) |
| `cuda_compute_capability_minor` | i32 | `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` | `hipDeviceProp_t.minor` | N/A |
| `multiprocessor_count` | i32 | `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` | `hipDeviceProp_t.multiProcessorCount` | `zeDeviceGetComputeProperties.numSubslicesPerSlice * numSlices` |
| `warp_size` | i32 | `CU_DEVICE_ATTRIBUTE_WARP_SIZE` (= 32) | `hipDeviceProp_t.warpSize` (= 64 GCN/RDNA) | `zeKernelSuggestGroupSize` output |
| `shared_mem_per_block_bytes` | i64 | `CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK` | `hipDeviceProp_t.sharedMemPerBlock` | `zeDeviceGetComputeProperties.maxSharedLocalMemory` |
| `max_threads_per_block` | i32 | `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK` | `hipDeviceProp_t.maxThreadsPerBlock` | `zeDeviceGetComputeProperties.maxGroupSizeX` |
| `has_cooperative_launch` | i1 | `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH` | `hipDeviceProp_t.cooperativeLaunch` | `zeDeviceGetProperties` feature flags |
| `total_global_mem_bytes` | i64 | `cudaDeviceProp.totalGlobalMem` | `hipDeviceProp_t.totalGlobalMem` | `zeDeviceGetMemoryProperties.totalSize` |
| `vendor_id` | i32 | NVIDIA = 0x10DE | AMD = 0x1002 | Intel = 0x8086 |
| `device_arch_string` | memref<?xi8> | PTX arch string e.g., `"sm_90"` | `gcnArchName` e.g., `"gfx90a"` | device name string |

**Connection to `gpu.select_variant`:** The primary consumer of `hw.query_capability` results
is the variant selection guard — the condition that decides which pre-compiled binary to load.
This makes the two proposals (topic-01 and topic-13) a natural pair:

```mlir
// Compile-time: gpu.binary carries NVVM + ROCDL + XeVM objects
%h = gpu.select_variant @kernels {
  // Runtime guard expressed in MLIR IR using hw.query_capability:
  %ok, %sm = hw.query_capability #hw.key<cuda_compute_capability_major> : i1, i32
  %sm90 = arith.cmpi sge, %sm, %c90 : i32
  %valid = arith.andi %ok, %sm90 : i1
  hw.yield %valid : i1
}
gpu.launch_func %h::@matmul_sm90 ...
```

Without `hw.query_capability`, the guard logic must live in C code outside MLIR's visibility.
With it, the guard is:
- **Hoistable**: a compiler pass can identify that `hw.query_capability` for a fixed key is
  invariant over the program lifetime and hoist it to a `llvm.global_ctors` initializer
  (exactly what IREE's `-iree-hal-memoize-device-queries` pass does for `hal.device.query`)
- **CSE-eligible**: two queries for the same key on the same device can be merged
- **Analyzable**: a static analysis pass can determine that a variant requires `warp_size == 32`
  and prune it at compile time when targeting a platform known to be AMD-only

**Lowering strategy:**

The op lowers through the GPU runtime lowering infrastructure, dispatching to the
vendor API based on which GPU runtime is active in the module:

```
hw.query_capability #hw.key<warp_size>
  → (NVVM path)   call @cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device) → i32
  → (ROCDL path)  call @hipDeviceGetAttribute(hipDeviceAttributeWarpSize, device) → i32
  → (XeVM path)   call @zeKernelSuggestGroupSize(...) → inferred from work-group suggestion
  → (CPU path)    return 1 : i32  (warp size = 1 on CPU, i.e., no SIMT lane grouping)
```

The lowering patterns are registered via `ConversionPattern` and can be contributed alongside
the lowerings for `gpu.launch_func` in the existing CUDA/HIP/Level Zero translation paths.

**Caching semantics:** The op carries a `#hw.cache<once>` attribute indicating the result is
invariant over device lifetime (appropriate for architecture queries) vs `#hw.cache<per_call>`
(appropriate for dynamic properties like current occupancy, if those are ever queried). The
`once` variant triggers automatic global hoisting in a standard `HWQueryHoistPass`.

---

## Evidence

### E1 — IREE `hal.device.query` is a production existence proof

- **Source:** `iree/compiler/Dialect/HAL/IR/HALOps.td`, IREE repo; also documented in
  `iree-deep-dive.md §3.3` and `iree-2026-state.md §2.3` in this repo.
- **What it does:** Queries the runtime `!hal.device` for named capability properties. Used in
  `hal.executable.variant` condition ops to gate variant loading. The op is lowered by each HAL
  driver to the appropriate underlying API call (CUDA driver, HIP, Vulkan extension queries).
- **What it cannot do:** It is IREE-private. It depends on the `!hal.device` SSA type, which
  does not exist in mainline MLIR. It is unconsumed by any MLIR-level optimization pass (IREE's
  `-iree-hal-memoize-device-queries` hoists it to globals, but this is IREE-internal). The op
  does not lower through MLIR's standard GPU lowering pipeline.
- **Gap this proves:** The concept is validated at production scale (IREE runs on NVIDIA, AMD,
  ARM Mali, Apple, Qualcomm). The upstream gap is the absence of an equivalent op in mainline
  MLIR's `gpu` dialect or any new dialect.

### E2 — CUDA `cuDeviceGetAttribute` queryable property count

- **Source:** `literature/papers-hardware-introspection.md §1.1`; CUDA Driver API reference
  group `CUDA__DEVICE` (docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)
- **Key facts:** `cudaDeviceProp` struct exposes ~100 fields. `cuDeviceGetAttribute` takes a
  `CUdevice_attribute` enum with 110+ values in CUDA 13.2. All are queryable per-device at
  runtime, cost ~0.5 µs per call. The CUDA runtime already caches these internally after the
  first `cudaGetDeviceProperties` call — meaning an MLIR-level hoisting pass achieves no
  additional runtime savings vs. the CUDA cache, but does achieve compiler-level analyzability.
- **Critical detail:** `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` (= 75) and `_MINOR`
  (= 76) are the primary selection keys used by every CUDA fat-binary selection algorithm
  (`multi-versioned-kernels.md §1.2`). These are the tier-1 keys for the `hw` dialect.

### E3 — HIP `hipDeviceGetAttribute` mirrors CUDA with AMD-specific additions

- **Source:** `literature/papers-hardware-introspection.md §1.4`; ROCm docs
  `rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html`
- **Key facts:** `hipDeviceProp_t` mirrors `cudaDeviceProp` structurally. Critical
  AMD-specific field: `gcnArchName` (e.g., `"gfx90a"`, `"gfx1201"`) — this is the arch
  selection key used by `clang-offload-bundler` at fat-binary load time
  (`wave-02-rocm-hip.md §source3`). `warpSize` is 64 on GCN/RDNA vs 32 on NVIDIA —
  a cardinal difference that affects tile size selection and any code using warp shuffle
  instructions. A `hw.query_capability` for `warp_size` must return 64 on AMD, not 32.
- **Dispatch significance:** Kernels that assume `warpSize == 32` will produce incorrect
  results on AMD if run without a guard check. This is currently caught at runtime by HIP's
  dimension validator (`wave-02-rocm-hip.md §source1`), but an MLIR-level query op could
  catch it at compile time or guard it correctly at launch time.

### E4 — Level Zero `zeDeviceGetProperties` and compute properties

- **Source:** `wave-04-level-zero.md §S1` (Level Zero Core Programming Guide v1.11/v1.15)
- **Key facts:** `zeDeviceGetProperties(device, &props)` fills `ze_device_properties_t`:
  `type` (GPU/CPU/FPGA), `vendorId`, `deviceId`, `numSlices`, `numSubslicesPerSlice`,
  `numEUsPerSubslice` (EU = Execution Unit, Intel's compute unit analog). The product
  `numSlices * numSubslicesPerSlice * numEUsPerSubslice` gives the total EU count (analogous
  to CUDA's `multiProcessorCount`). `zeDeviceGetComputeProperties` adds `maxGroupSizeX/Y/Z`
  and `maxSharedLocalMemory`. These are the fields needed for tile-size selection.
- **No Level Zero analog to compute_capability:** Intel GPUs have no single-integer "compute
  capability" like NVIDIA. The tile-selection key for Intel is the EU count and the subslice
  topology, which require a more complex mapping to the `hw` key set. The `device_arch_string`
  key (returning the device name string) is the closest proxy.

### E5 — RFC #86960 does NOT address runtime queries

- **Source:** `wave-05-llvm-discourse-rfcs.md §10`; RFC at
  discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
  (June 2025, Robert K Samuel, IIT Madras PACE Lab)
- **What the RFC proposes:** A `schedule` op grouping `task` ops each annotated with a
  static `target` attribute (`cpu`, `gpu`, `fpga`). Compile-time task placement, lowering to
  MPI Dialect + existing GPU/CPU dialects.
- **What it explicitly does not do:** Runtime hardware queries. The RFC assumes target
  identity is known at compile time. There is no `hw.query_capability` equivalent.
  The proposal is compile-time assignment, not runtime detection.
- **Implication:** RFC #86960 and the `hw` dialect are complementary, not competing.
  #86960 handles static placement; `hw.query_capability` handles the dynamic case where the
  same artifact runs on unknown hardware. This is a clean design boundary.

### E6 — MLIR `gpu` dialect has no introspection ops (source-verified absence)

- **Source:** `wave-01-mlir-gpu-dialect.md §1`; MLIR GPU Dialect docs
  (mlir.llvm.org/docs/Dialects/GPU/); RFC "Cleaning the GPU Dialect" #88170 (Sep 2025)
- **Confirmed absence:** The GPU dialect op list (as of LLVM 23.x) contains zero ops for
  querying runtime device properties. The cleanup RFC explicitly categorizes ops as:
  (1) target-independent programming model ops, (2) binary management ops, (3) runtime
  interaction ops — and notes that (3) needs a "separate dialect or lowering pass." This
  is the exact slot `hw.query_capability` would fill.
- **`#gpu.select_object` is compile-time only:** Confirmed in
  `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp` — `embedBinary` creates a
  single `@serializedObj` global at translation time with no runtime detection code emitted.

### E7 — IREE `-iree-hal-memoize-device-queries` demonstrates the hoisting value

- **Source:** `iree-2026-state.md §2.3` (HAL compiler passes table)
- **What the pass does:** Hoists `hal.device.query` ops that are invariant over program
  lifetime into `llvm.global_ctors`-style initializers, evaluated once at module load.
  This is the exact optimization that the `HWQueryHoistPass` would implement for the
  upstream `hw` dialect. IREE does this today — inside IREE. The proposal brings it to mainline.
- **Performance impact:** Device query calls cost ~0.5 µs each (CUDA). On a dispatch loop
  running at 1 kHz (1 ms between kernel launches), un-hoisted queries consume 0.05% of
  dispatch budget — negligible per-call, but not negligible at the scale of hundreds of
  distinct kernel launches during model initialization.

### E8 — `zeKernelSuggestGroupSize` reveals Level Zero's introspection philosophy

- **Source:** `wave-04-level-zero.md §S1` — `zeKernelSuggestGroupSize(kernel, gx, gy, gz,
  &sgx, &sgy, &sgz)` returns driver-optimal group size given a global work size
- **Design implication:** Level Zero exposes hardware knowledge through *driver-recommendation*
  rather than raw property queries. The `hw` dialect must accommodate this: for Level Zero,
  `hw.query_capability #hw.key<warp_size>` should delegate to `zeKernelSuggestGroupSize`
  (with a unit workload) rather than a direct property read. This is an implementation detail
  hidden behind the uniform `hw.query_capability` interface.

### E9 — HIP event semantics inversion confirms the need for vendor abstraction

- **Source:** `wave-02-rocm-hip.md §source2` — "The `startEvent` on AMD platforms records
  when the kernel **completes**, not when it starts; this is a semantic inversion vs. CUDA"
- **Implication for introspection:** Hardware behavior is not always semantically identical
  across vendors even when APIs are nominally identical. The `hw` dialect must document
  per-key semantics carefully, noting where vendor behavior diverges. A `hw.query_capability`
  for a timing-related property must specify AMD vs NVIDIA semantics explicitly.

### E10 — libkdl's `kdl_detect_devices()` is the runtime prototype

- **Source:** `experiments/prototype/src/kdl.c` — `kdl_detect_devices()` function
  calls `cuDeviceGetAttribute` (CUDA), `hipGetDeviceProperties` (HIP), and falls back to
  CPU properties. The gathered values populate `kdl_device_info` structs used by the
  cost model in `kdl_estimate_cost_weighted()` (kdl.c:1013–1088).
- **What this proves:** The runtime half of `hw.query_capability` lowering is already
  implemented and tested on GTX 1650 + CPU hardware. The MLIR-level op is the missing
  compiler-side complement that makes the queries analyzable.

---

## Feasibility

**Medium-High. The concept is validated by IREE production use; the implementation scope
is bounded; the main unknowns are dialect naming politics and key-set standardization.**

### What exists today (no new code needed):

- `hal.device.query` in IREE — production-quality blueprint for semantics and lowering
- `cuDeviceGetAttribute` / `hipDeviceGetAttribute` / `zeDeviceGetProperties` — all callable
  from C++ in a lowering pattern with no new library dependencies
- `kdl_detect_devices()` in this repo — the runtime half, verified on hardware
- `HWQueryHoistPass` conceptually equivalent to IREE's `-iree-hal-memoize-device-queries`

### What needs to be written:

| Component | Estimated LOC | Complexity |
|-----------|--------------|------------|
| `hw` dialect TableGen definitions (ops + key enum attr) | ~200 | Low — model after GPU dialect |
| `HWQueryOp` verification logic | ~50 | Low |
| `HWQueryHoistPass` (hoist invariant queries to global_ctors) | ~150 | Medium — needs dominance analysis |
| CUDA lowering pattern (`hw.query_capability` → `cuDeviceGetAttribute`) | ~80 | Low |
| HIP lowering pattern | ~80 | Low |
| Level Zero lowering pattern (with `zeKernelSuggestGroupSize` for warp_size) | ~120 | Medium |
| CPU fallback lowering (return constants) | ~40 | Low |
| Integration with `gpu.select_variant` condition region | ~100 | Medium |
| Lit tests | ~300 | Low-Medium |
| **Total** | **~1120** | **Medium** |

The hardest part is not the code — it is the **key-set standardization RFC**. Choosing which
hardware properties belong in the standard key set (vs. vendor-specific escape hatches) will
require community discussion. IREE's `hal.device.query` uses opaque strings to avoid this
debate; the enum approach forces it. The proposal should start with the 10 keys listed above
and explicitly allow vendor-specific extension via a `#hw.vendor_key<"custom_string">` escape.

**Demo scope for the poster:** Show that on a machine with an NVIDIA GPU, `hw.query_capability
#hw.key<cuda_compute_capability_major>` returns 75 (Turing) and the SM-90-requiring variant
is skipped; fall back to the generic f32 kernel. The Level Zero and HIP paths can be shown
via unit test mocks without requiring Intel/AMD hardware.

**Synergy with topic-01 (`gpu.select_variant`):** The two ops are designed to compose. The
poster can demo them together: `gpu.select_variant` provides the variant container;
`hw.query_capability` provides the typed guard condition. This dual-contribution framing
makes the poster stronger than either op alone.

---

## Upstream Path

### Placement options in llvm-project:

**Option A — New `hw` dialect in `mlir/Dialect/HW/`:**
Precedent: the `hw` directory already exists in CIRCT (circt.llvm.org) for hardware
description — but MLIR upstream does not have a `hw` dialect. A new top-level dialect
requires RFC and community sign-off. This is the cleanest namespace but highest political cost.

**Option B — Extend the `gpu` dialect with introspection ops:**
Add `gpu.query_device_property` to the existing `gpu` dialect. Lower friction: no new dialect,
no namespace discussion. The cleanup RFC (#88170) explicitly creates a "runtime interaction ops"
category that this would fill. Fabian Mora (RFC author) is the natural reviewer.

**Option C — Add to `mlir/Dialect/GPU/Transforms/` as a lowering pass:**
Implement the query as a LLVM IR intrinsic call emission (like `gpu.thread_id` → `llvm.nvvm.read.ptx.sreg.tid.x`),
not as a new dialect op. This is the lowest-friction path but gives up IR analyzability
(the key benefit of the proposal).

**Recommendation: Option B for initial contribution, with Option A as the long-term target.**
Start by adding `gpu.query_device_property` ops within the GPU dialect cleanup RFC context,
demonstrating compiler hoisting benefits. If the key-set grows to cover non-GPU hardware
(FPGAs, NPUs), propose the `hw` dialect split at that point.

| Artifact | Location |
|----------|----------|
| Op definitions | `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td` |
| Key enum attribute | `mlir/include/mlir/Dialect/GPU/IR/GPUAttrDefs.td` |
| Hoist pass | `mlir/lib/Dialect/GPU/Transforms/HoistDeviceQueries.cpp` |
| CUDA lowering | `mlir/lib/Target/LLVMIR/Dialect/GPU/CUDADeviceQueryLowering.cpp` |
| HIP lowering | `mlir/lib/Target/LLVMIR/Dialect/GPU/HIPDeviceQueryLowering.cpp` |
| Level Zero lowering | `mlir/lib/Target/LLVMIR/Dialect/GPU/LevelZeroDeviceQueryLowering.cpp` |
| Tests | `mlir/test/Dialect/GPU/hoist-device-queries.mlir` |
| Integration tests | `mlir/test/Target/LLVMIR/gpu-device-query-cuda.mlir` |

**RFC anchor:** The GPU dialect cleanup RFC (#88170, September 2025, Fabian Mora) is the
correct review context — the cleanup explicitly leaves the "runtime interaction ops" category
vacant, and `gpu.query_device_property` is the most natural candidate to fill it.

**Community buy-in path:**
1. Post LLVM Discourse thread: "Runtime device property queries in the GPU dialect" — reference
   the cleanup RFC, IREE `hal.device.query` as prior art, and the three IREE issues as evidence
   of the gap's persistence
2. Coordinate with Ben Vanik (IREE), Fabian Mora (GPU dialect cleanup), Joseph Huber (liboffload
   `olGetDeviceInfo` is the analogous op in the liboffload C API)
3. The `olGetDeviceInfo(device, OL_DEVICE_INFO_*, ...)` pattern in liboffload's C API
   (PR #118614) is the exact runtime-level counterpart — the Discourse RFC can frame the
   MLIR-level op as the IR-level abstraction over `olGetDeviceInfo`

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **8/10** | The concept exists in IREE-private `hal.device.query`; the upstream gap is confirmed (zero equivalent in mainline MLIR gpu dialect). The enum-typed key approach and compiler-level hoisting pass are new. Score is 8 rather than 9 because IREE prior art means this is porting + standardizing, not wholly new design. |
| **Feasibility** | **8/10** | ~1120 LOC total, mostly boilerplate lowering patterns modeled after existing NVVM/ROCDL paths. The runtime half is already in kdl.c. The key-set RFC is the main risk, not the code. Poster demo achievable with GTX 1650 (existing hardware). |
| **Evidence Strength** | **9/10** | IREE production use on 5+ hardware targets, CUDA/HIP/Level Zero API docs (primary sources), three open IREE issues citing runtime query as an explicit blocker, RFC #86960 confirmed to not address this, GPU dialect source confirmed to lack introspection ops. |
| **Impact** | **8/10** | Enables compiler-level hoisting of hardware queries (performance), typed key checking (correctness), and integration with `gpu.select_variant` for typed guard conditions. Applicable to libkdl, liboffload policy layers, IREE HAL, any MLIR-based multi-variant dispatch. |
| **LLVM Community Fit** | **7/10** | GPU dialect cleanup RFC creates a natural landing zone, but the key-set standardization debate will generate extended discussion. Framing as "port IREE's validated design upstream" is the strongest political strategy. |
| **Composite** | **8.0/10** | |

---

## Pitch

**Three-sentence poster pitch:**

Every major GPU runtime — CUDA, HIP, Level Zero, Vulkan — provides rich device capability
query APIs (`cuDeviceGetAttribute`, `hipDeviceGetAttribute`, `zeDeviceGetProperties`), and
IREE's production HAL dialect uses them via `hal.device.query` to gate variant loading at
runtime. But mainline MLIR's `gpu` dialect has zero introspection ops: variant guards must be
written in unstructured C code outside the IR boundary, invisible to compiler optimization and
impossible to hoist or CSE. We propose `gpu.query_device_property` — a typed, enum-keyed
op that lowers to the correct vendor API, enables compiler-level query hoisting via
`HoistDeviceQueriesPass`, and composes with `gpu.select_variant` to give MLIR-native,
analyzable runtime dispatch guards for the first time.

**Poster panel structure:**

1. Gap diagram: vendor APIs (CUDA/HIP/LZ) → ad-hoc C detection → runtime dispatch (invisible
   to compiler); vs. proposed: vendor APIs → `gpu.query_device_property` → `gpu.select_variant`
2. Op definition: enum key, success/value pair, lowering to CUDA/HIP/Level Zero/CPU
3. Hoisting diagram: un-hoisted (query per launch) vs. hoisted (query once at module load)
4. Integration with topic-01 (`gpu.select_variant`): typed guard condition in variant region
5. IREE `hal.device.query` as production prior art, three open issues as gap evidence
6. Key-set table: 10 standard keys + vendor escape hatch
7. Upstream path: GPU dialect cleanup RFC (#88170) as the landing zone

---

## Risks

1. **Key-set politics.** An enum-typed key set forces a community decision on which hardware
   properties are "standard." Vendors (NVIDIA, AMD, Intel) may want different keys first.
   Mitigation: start with the 10 keys that have direct analogs in CUDA, HIP, and Level Zero
   (warp_size, SM count, shared mem limit, etc.) and add a `#hw.vendor_key<"string">` escape
   hatch to avoid blocking vendor-specific extensions.

2. **`hal.device.query` handles IREE-specific key strings.** IREE's keys include
   `"hal.executable.format"` and `"cuda.compute_capability"` — opaque strings defined by IREE
   internals. The upstream enum approach cannot simply import these. A migration path from
   IREE's string keys to the upstream enum keys would be needed for IREE to adopt the new op.
   This is a non-trivial ecosystem coordination burden.

3. **Level Zero warp size is not a simple property.** Unlike CUDA (`warpSize = 32`) and HIP
   (`warpSize = 64`), Intel's EU SIMD width varies by kernel and can be influenced by
   `zeKernelSuggestGroupSize`. The `warp_size` key cannot return a constant for Level Zero
   without a kernel context. The op definition must either (a) query per-kernel or (b)
   return a conservative lower bound. This is a semantic gap that must be documented and
   tested.

4. **CPU "no-op" lowering is deceptively complex.** On CPU, `warp_size = 1`, `SM count =
   vCPU count`, `shared_mem = L1 cache size`. These values are meaningful for dispatch
   (libkdl uses CPU SM count to determine thread-level parallelism) but are not exposed
   by any GPU-derived API. The CPU lowering patterns require platform-specific system calls
   (`sysconf(_SC_NPROCESSORS_ONLN)`, CPUID for cache size) rather than a uniform GPU API.
   This complexity is hidden at the `hw.query_capability` abstraction level but surfaces in
   the lowering implementation.

5. **Query hoisting correctness requires SSA dominance analysis.** The `HoistDeviceQueriesPass`
   must ensure that hoisted queries execute before any op that consumes their result, and that
   hoisted calls in `global_ctors` do not execute before GPU runtime initialization. This is
   non-trivial for modules with multiple initialization order constraints (e.g., CUDA context
   must exist before `cuDeviceGetAttribute` can return meaningful results).

6. **RFC #88170 (GPU dialect cleanup) may restructure op categories.** If the cleanup RFC
   merges and renames or restructures the "runtime interaction ops" category, the `hw.query_*`
   ops may need renaming before submission. Timing the contribution to land after the cleanup
   RFC is the lower-risk path, at the cost of a 6–12 month delay.

---

## Cross-References

- `literature/papers-hardware-introspection.md` — CUDA/HIP/Vulkan/OpenCL API details, warp
  size semantics, `cudaDeviceProp` field inventory
- `literature/iree-deep-dive.md §3.3` — `hal.device.query` semantics and IREE HAL driver
  lowering pattern
- `literature/iree-2026-state.md §2.3` — `-iree-hal-memoize-device-queries` pass (the hoisting
  pass this proposal brings upstream)
- `wave-04-level-zero.md §S1` — `zeDeviceGetProperties`, `zeDeviceGetComputeProperties`,
  `zeKernelSuggestGroupSize` details
- `wave-02-cuda-driver-api.md §source1` — `cuDeviceGetAttribute` API reference, dispatch cost
  budget (~0.5 µs per call)
- `wave-02-rocm-hip.md §source1,2` — `hipDeviceGetAttribute`, `warpSize = 64` semantics,
  `startEvent` inversion hazard
- `wave-01-mlir-gpu-dialect.md §gap` — confirmed absence of runtime introspection ops in gpu
  dialect; `#gpu.select_object` compile-time-only limitation
- `wave-05-llvm-discourse-rfcs.md §10,11` — RFC #86960 (does not address runtime queries);
  RFC #88170 (GPU dialect cleanup, "runtime interaction ops" category is vacant)
- `research/mega-survey/.../waves/wave-01-iree-hal-runtime-dispatch.md` — IREE variant
  condition op using `hal.device.query` for first-valid-match variant selection
- `experiments/prototype/src/kdl.c` — `kdl_detect_devices()` is the runtime prototype for
  `hw.query_capability` lowering (CUDA and CPU paths implemented)
- `research/mega-survey/20-poster-topics/waves/topic-01-gpu-select-variant.md` — `gpu.select_variant`;
  this proposal provides the typed guard condition that fills topic-01's condition region
- `research/mega-survey/20-poster-topics/waves/topic-03-dispatch-cost-attr.md` — complementary
  proposal; cost annotation propagation assumes device capability queries are already available
  to classify devices; `hw.query_capability` provides the `peak_tflops_f32` and `peak_bw_gbps`
  query substrate that `kdl_estimate_cost_weighted()` needs
