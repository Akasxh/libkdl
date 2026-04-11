# Combo A Refined Technical Design v1

**Title:** Runtime Variant Selection for LLVM GPU Offloading
**Components:** `#gpu.runtime_select` attribute (T01) + OffloadBinary metadata vocabulary (T07) + dispatch flame graph (T19)
**Author:** Akash (IIT Patna)
**Date:** 2026-04-09
**Status:** Design-complete, pre-implementation

---

## 1. The `#gpu.runtime_select` Attribute

### 1.1 MLIR TableGen Definition

The attribute implements `OffloadingLLVMTranslationAttrInterface` (defined in
`mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td` lines 1-15),
which requires exactly two methods: `embedBinary` and `launchKernel`. This is the
same interface that `#gpu.select_object` implements in
`mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp`.

```tablegen
// In mlir/include/mlir/Dialect/GPU/IR/GPUOps.td (alongside existing SelectObjectAttr)

def GPU_RuntimeSelectAttr : GPU_Attr<"RuntimeSelect", "runtime_select",
    [OffloadingTranslationAttrTrait,
     DeclareAttrInterfaceMethods<OffloadingLLVMTranslationAttrInterface>]> {
  let summary = "Defers binary selection to runtime via vendor detection.";
  let description = [{
    When used as the offloading handler on a `gpu.binary` op, embeds ALL
    `#gpu.object` entries as separate LLVM global constants and emits a
    runtime vendor-detection stub that selects the appropriate binary at
    module-load time via `dlopen`-probed vendor API calls.

    Unlike `#gpu.select_object` which commits to a single binary blob at
    LLVM IR translation time, `#gpu.runtime_select` preserves all variants
    and generates host-side dispatch logic.

    Optional `strategy` parameter controls selection policy:
    - `first_compatible`: match first object whose target is present (default)
    - `rank_by_priority`: use `variant_priority` OffloadBinary metadata key
    - `rank_by_capability`: use device capability fingerprint scoring

    Optional `fallback` parameter:
    - `error`: abort if no compatible device found (default)
    - `cpu`: fall back to host execution if available
  }];
  let parameters = (ins
    DefaultValuedParameter<"StringAttr", "\"first_compatible\"">:$strategy,
    DefaultValuedParameter<"StringAttr", "\"error\"">:$fallback
  );
  let assemblyFormat = "`<` struct(params) `>`";
}
```

Usage in IR:

```mlir
// Compile-time: gpu-module-to-binary produces multi-object binary
gpu.binary @kernels <#gpu.runtime_select<
    strategy = "rank_by_priority", fallback = "cpu">> [
  #gpu.object<#nvvm.target<chip = "sm_75">, bin = "...cubin...">,
  #gpu.object<#rocdl.target<chip = "gfx90a">, bin = "...hsaco...">,
  #gpu.object<#nvvm.target<chip = "sm_90">, bin = "...cubin-sm90...">
]
```

### 1.2 What `embedBinary()` Emits

The existing `SelectObjectAttr::embedBinary` (ref: `mlir-gpu-infrastructure-2026.md`
section 2) emits exactly four things: `@serializedObj` (one blob), `@modulePtr`,
`@loadFn` (via `llvm.global_ctors`), and `@unloadFn` (via `llvm.global_dtors`).

The new `RuntimeSelectAttr::embedBinary` generalizes this to N blobs:

```
// Pseudocode for RuntimeSelectAttr::embedBinary()

LogicalResult RuntimeSelectAttr::embedBinary(
    Operation *binaryOp, IRBuilderBase &builder,
    ModuleTranslation &modTrans) {

  auto binaryOp = cast<gpu::BinaryOp>(op);
  ArrayAttr objects = binaryOp.getObjects();
  StringRef symName = binaryOp.getSymName();

  // ---- Phase 1: Embed all binary blobs as separate globals ----
  // (SelectObjectAttr embeds exactly one; we embed N)
  SmallVector<GlobalVariable *> blobGlobals;
  SmallVector<Constant *> blobSizes;
  SmallVector<uint32_t> vendorIds;

  for (auto [idx, objAttr] : llvm::enumerate(objects)) {
    auto obj = cast<gpu::ObjectAttr>(objAttr);
    StringRef blob = obj.getObject().getValue();

    // @kernels_blob_0, @kernels_blob_1, ...
    auto *global = new GlobalVariable(
        module, ArrayType::get(i8Ty, blob.size()),
        /*isConstant=*/true, GlobalValue::InternalLinkage,
        ConstantDataArray::getString(ctx, blob, /*AddNull=*/false),
        (symName + "_blob_" + Twine(idx)).str());

    blobGlobals.push_back(global);
    blobSizes.push_back(ConstantInt::get(i64Ty, blob.size()));

    // Extract vendor ID from target attribute
    // #nvvm.target -> VENDOR_NVIDIA (1)
    // #rocdl.target -> VENDOR_AMD (2)
    // #spirv.target_env -> VENDOR_SPIRV (3)
    vendorIds.push_back(getVendorIdFromTarget(obj.getTarget()));
  }

  // ---- Phase 2: Emit dispatch table global ----
  // struct RuntimeSelectEntry { uint32_t vendor_id; void *blob; uint64_t size; }
  // @kernels_dispatch_table : global [N x %RuntimeSelectEntry]
  auto *entryTy = StructType::create(
      ctx, {i32Ty, ptrTy, i64Ty}, "RuntimeSelectEntry");

  SmallVector<Constant *> tableEntries;
  for (size_t i = 0; i < objects.size(); ++i) {
    tableEntries.push_back(ConstantStruct::get(entryTy, {
        ConstantInt::get(i32Ty, vendorIds[i]),
        ConstantExpr::getBitCast(blobGlobals[i], ptrTy),
        blobSizes[i]
    }));
  }
  auto *tableTy = ArrayType::get(entryTy, objects.size());
  auto *tableGlobal = new GlobalVariable(
      module, tableTy, /*isConstant=*/true, GlobalValue::InternalLinkage,
      ConstantArray::get(tableTy, tableEntries),
      (symName + "_dispatch_table").str());

  // ---- Phase 3: Emit selected-index + module-ptr globals ----
  // @kernels_selected_idx : global i32 = -1  (unresolved)
  // @kernels_module_ptr   : global ptr = null
  auto *selectedIdx = new GlobalVariable(
      module, i32Ty, false, GlobalValue::InternalLinkage,
      ConstantInt::get(i32Ty, -1),
      (symName + "_selected_idx").str());

  auto *modulePtr = new GlobalVariable(
      module, ptrTy, false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(ptrTy),
      (symName + "_module_ptr").str());

  // ---- Phase 4: Emit constructor (llvm.global_ctors, priority 123) ----
  // @kernels_ctor:
  //   %vendor = call @__gpu_runtime_select_detect_vendor()
  //   ; iterates dispatch table, finds best match for %vendor
  //   %idx = call @__gpu_runtime_select_rank(
  //       @kernels_dispatch_table, N, %vendor, strategy)
  //   store i32 %idx, @kernels_selected_idx
  //   %entry = getelementptr @kernels_dispatch_table, %idx
  //   %blob = load entry.blob
  //   %size = load entry.size
  //   %vendor_id = load entry.vendor_id
  //   ; call vendor-appropriate module load
  //   switch %vendor_id:
  //     case NVIDIA: %mod = call @mgpuModuleLoad(%blob, %size)
  //     case AMD:    %mod = call @mgpuModuleLoad(%blob, %size)
  //                  ; (mgpuModuleLoad dispatches internally via
  //                  ;  CudaRuntimeWrappers.cpp / RocmRuntimeWrappers.cpp)
  //   store %mod, @kernels_module_ptr

  // ---- Phase 5: Emit destructor (llvm.global_dtors, priority 123) ----
  // @kernels_dtor:
  //   %mod = load @kernels_module_ptr
  //   call @mgpuModuleUnload(%mod)
}
```

**Key design decision:** The constructor uses the existing `mgpuModuleLoad` /
`mgpuModuleUnload` runtime wrappers from `CudaRuntimeWrappers.cpp` and
`RocmRuntimeWrappers.cpp`. This means:

- The vendor detection must happen BEFORE `mgpuModuleLoad` is called, to ensure
  the correct runtime wrapper library is loaded.
- The detection stub itself uses `dlopen`/`dlsym` (not `mgpu*` wrappers) to
  probe vendor availability. This mirrors `kdl.c:551-596` (CUDA discovery) and
  `kdl.c:749+` (HIP discovery).

**Difference from `SelectObjectAttr`:** SelectObjectAttr calls `mgpuModuleLoad`
unconditionally because it commits to one vendor at compile time. RuntimeSelectAttr
must first determine WHICH vendor runtime is available, then call the appropriate
`mgpu*` function. The `mgpu*` functions already dispatch internally (CUDA vs HIP)
based on which `.so` was linked — the new attribute leverages this by ensuring
the correct shared library is `dlopen`'d at detection time.

### 1.3 What `launchKernel()` Emits

`SelectObjectAttr::launchKernel` emits a direct call to `mgpuLaunchKernel`. The
new attribute emits the same call — the dispatch table selection happened at
constructor time, and `@kernels_module_ptr` already holds the correct module handle.

```
// Pseudocode for RuntimeSelectAttr::launchKernel()

LogicalResult RuntimeSelectAttr::launchKernel(
    Operation *launchFuncOp, Operation *binaryOp,
    IRBuilderBase &builder, ModuleTranslation &modTrans) {

  // Identical to SelectObjectAttr::launchKernel EXCEPT:
  // - Uses @kernels_module_ptr (set by constructor at load time)
  //   instead of a statically-determined module pointer
  // - The mgpuModuleGetFunction / mgpuLaunchKernel calls are unchanged

  Value *modulePtr = builder.CreateLoad(ptrTy,
      modTrans.lookupGlobal(symName + "_module_ptr"));
  Value *func = builder.CreateCall(
      mgpuModuleGetFunctionFn, {modulePtr, kernelNameConstant});
  builder.CreateCall(mgpuLaunchKernelFn,
      {func, gridX, gridY, gridZ, blockX, blockY, blockZ,
       sharedMem, stream, paramsArray, /*extra=*/nullPtr});
}
```

**Hot-path overhead: zero.** Once the constructor runs (at `global_ctors` time),
the kernel launch path is identical to `SelectObjectAttr` — a single indirect
load from `@kernels_module_ptr`, then `mgpuModuleGetFunction` + `mgpuLaunchKernel`.
No per-launch vendor detection or dispatch table lookup.

### 1.4 Composition with `gpu-module-to-binary`

The pass `gpu-module-to-binary` (ref: `ModuleToBinary.cpp` lines 70-96 in the
infrastructure doc) already:

1. Walks all `gpu.module` ops
2. Calls `serializeToObject` per attached target attribute
3. Collects all objects into an `ArrayAttr`
4. Creates `gpu.binary` with the module's `offloadingHandler` attribute

The composition is clean:

```
Step 1: Attach targets (existing passes, no change):
  -nvvm-attach-target chip=sm_75
  -nvvm-attach-target chip=sm_90
  -rocdl-attach-target chip=gfx90a

Step 2: Mark for runtime selection (NEW pass):
  --gpu-mark-runtime-select strategy=rank_by_priority fallback=cpu

  This pass walks gpu.module ops that have 2+ target attributes and
  sets offloadingHandler = #gpu.runtime_select<...>.

  Implementation: ~50 LOC. Walk modules, check targets.size() > 1,
  set the handler attribute.

Step 3: Compile to binary (existing pass, no change):
  --gpu-module-to-binary format=bin

  Produces gpu.binary with the #gpu.runtime_select handler and
  multiple #gpu.object entries.

Step 4: Translate to LLVM IR (existing infrastructure, no change):
  --translate-to-llvm-ir

  The OffloadingLLVMTranslationAttrInterface dispatch calls
  RuntimeSelectAttr::embedBinary and RuntimeSelectAttr::launchKernel
  instead of SelectObjectAttr's versions.
```

No changes to existing passes. The new attribute is purely additive.

### 1.5 Interaction with GPU Dialect Cleanup RFC (#88170)

RFC #88170 (Fabian Mora, September 2025, still active) proposes separating
`gpu.binary` as a pure container from dispatch policy. This design is aligned:

- `gpu.binary` remains the container (unchanged)
- `#gpu.runtime_select` is a dispatch policy implemented as an offloading handler
- The handler attribute is the EXACT mechanism the RFC envisions for dispatch policy

**Risk:** If the RFC changes the `gpu.binary` op signature (e.g., removing
`offloadingHandler` as a direct attribute, replacing it with a separate op),
the TableGen definition changes but the `embedBinary`/`launchKernel` logic does
not — the interface contract is stable.

**Strategy:** Frame `#gpu.runtime_select` as the first dispatch-policy
implementation that fills the slot the RFC explicitly left vacant. Coordinate
with Mora to land as part of the cleanup or immediately after.

### 1.6 Runtime Helper Functions

The attribute emits calls to two new runtime helper functions:

```c
// In mlir/lib/ExecutionEngine/GPURuntimeSelectWrappers.cpp (NEW file, ~200 LOC)

// Probes vendor availability via dlopen. Returns vendor enum.
// Mirrors kdl.c:551-596 (kdl_discover_cuda) pattern.
extern "C" uint32_t __gpu_runtime_select_detect_vendor() {
    // Try CUDA first
    void *cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (cuda) {
        auto cuInit = (int(*)(unsigned))dlsym(cuda, "cuInit");
        if (cuInit && cuInit(0) == 0) return 1; // VENDOR_NVIDIA
    }
    // Try HIP
    void *hip = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
    if (hip) {
        auto hipInit = (int(*)(unsigned))dlsym(hip, "hipInit");
        if (hipInit && hipInit(0) == 0) return 2; // VENDOR_AMD
    }
    // Try Level Zero
    void *l0 = dlopen("libze_loader.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (l0) {
        auto zeInit = (int(*)(int))dlsym(l0, "zeInit");
        if (zeInit && zeInit(0) == 0) return 3; // VENDOR_INTEL
    }
    return 0; // CPU only
}

// Iterates dispatch table, returns index of best-matching entry.
// strategy: 0 = first_compatible, 1 = rank_by_priority
extern "C" int32_t __gpu_runtime_select_rank(
    const void *table, uint32_t num_entries,
    uint32_t detected_vendor, uint32_t strategy) {
    // First pass: find entries matching detected vendor
    // If strategy == rank_by_priority: among matches, prefer higher priority
    // If no match: return -1 (triggers fallback or error)
}
```

These functions are linked via `sharedLibPaths` in `ExecutionEngine` or via the
`mlir-cpu-runner` shared library mechanism — the same pattern used for
`CudaRuntimeWrappers.cpp`.

---

## 2. OffloadBinary Metadata Vocabulary

### 2.1 Essential Keys (Minimum Viable Set)

These are the keys that `#gpu.runtime_select` NEEDS to function beyond
first-compatible-wins. Derived from `kdl.c:112-122` (contract struct) and
the gap identified in `OffloadBinary.h` (only `triple` and `arch` exist).

**Tier 1 — MUST keys (runtime rejects image if constraint violated):**

| Key | Type | Example | Justification |
|-----|------|---------|---------------|
| `min_sm` | uint string | `"75"` | CUDA compute capability floor. Without this, runtime cannot reject an sm_90a cubin on an sm_75 device. Currently inferred heuristically from arch string. |
| `min_gfx` | string | `"gfx90a"` | AMD ISA floor. Same problem — `areTargetsCompatible()` only checks xnack/sramecc flags, not ISA generation ordering. |
| `requires_features` | comma-list | `"tensor_core,bf16"` | Named capability tokens. Without this, runtime cannot reject a tensor-core-dependent kernel on a device without tensor cores. |

**Tier 3 — MAY keys (ranking, not gating):**

| Key | Type | Example | Justification |
|-----|------|---------|---------------|
| `variant_priority` | uint string | `"10"` | Explicit ranking among compatible images. This is what `#gpu.runtime_select` with `rank_by_priority` strategy consumes. |
| `variant_tag` | string | `"optimized"` | Human-readable label for debugging and tooling. |

**Total: 5 keys.** This is the minimum set to go from "first-compatible-wins"
to "best-compatible-wins."

### 2.2 Deferred Keys

These are valuable but NOT required for the poster or MVP RFC:

| Key | Reason to Defer |
|-----|----------------|
| `sgpr_count`, `vgpr_count`, `agpr_count` | Requires KernelInfo → OffloadBinary pipeline (writer integration in AMDGPU backend). Non-trivial, multi-backend effort. |
| `registers_per_thread` | Requires NVPTX cubin parsing (EIATTR extraction). Separate patch. |
| `shared_mem_bytes`, `scratch_bytes` | Same pipeline dependency as register counts. |
| `warp_size` | Redundant with `triple` for current vendors (32 for NVIDIA, 32/64 for AMD inferred from target). |
| `reqd_workgroup_size`, `max_workgroup_size` | Useful for occupancy, but not needed for variant selection. |
| `producer_version`, `opt_level`, `lto` | Provenance/diagnostics only. Nice-to-have. |

### 2.3 Composition with `isMetadataCompatible()`

PR #185663 (merged March 10, 2026) introduced `isMetadataCompatible()` as the
runtime-side filter in `PluginInterface.cpp`. PR #186088 (open March 2026) uses
it in the `parseOffloadBinary` loop. The current implementation checks `triple`
and `arch` string equality.

The standard vocabulary composes cleanly:

```cpp
// Current isMetadataCompatible (PR #185663):
bool isMetadataCompatible(const OffloadBinMetadataTy &image,
                          const OffloadBinMetadataTy &device) {
  return image.Triple == device.Triple &&
         areArchsCompatible(image.Arch, device.Arch);
}

// Extended isMetadataCompatible (with Tier 1 keys):
bool isMetadataCompatible(const OffloadBinMetadataTy &image,
                          const OffloadBinMetadataTy &device) {
  if (image.Triple != device.Triple) return false;
  if (!areArchsCompatible(image.Arch, device.Arch)) return false;

  // NEW: check min_sm / min_gfx against device arch
  if (auto minSm = image.getString("min_sm")) {
    uint32_t required = std::stoul(*minSm);
    uint32_t deviceSm = parseSmFromArch(device.Arch);  // "sm_75" -> 75
    if (deviceSm < required) return false;
  }

  // NEW: check requires_features against device capability set
  if (auto features = image.getString("requires_features")) {
    for (auto &token : split(*features, ',')) {
      if (!device.hasCapability(token)) return false;
    }
  }

  return true;
}
```

**Backward compatibility:** Old binaries without these keys pass all checks
(missing key = no constraint). Old runtimes encountering new keys ignore them
(unknown strings in the string table are never queried).

### 2.4 Upgrade Path for Existing Consumers

| Consumer | Current Behavior | With New Keys |
|----------|-----------------|---------------|
| `clang-linker-wrapper` | Reads `triple`/`arch` only | No change needed; ignores unknown keys |
| `llvm-offload-binary --dump` | Prints string table verbatim | Already shows new keys; no code change |
| liboffload `isMetadataCompatible` | Checks triple + arch | Opt-in: check new keys when present |
| IREE | Does not use OffloadBinary | No impact |
| User code via `OffloadBinary::getString()` | Returns `std::nullopt` for missing keys | Same behavior; new keys appear when present |

The OffloadBinary format is additive by design (D122069 review thread:
"intentionally uses a flexible string map"). Adding string keys is a non-breaking
change at every layer.

### 2.5 Header Constants Patch

```cpp
// In llvm/include/llvm/Object/OffloadBinary.h (additive, ~20 lines)

namespace offload {
namespace metadata {
// Existing (document for first time):
constexpr StringLiteral kTriple = "triple";
constexpr StringLiteral kArch = "arch";

// Tier 1: Minimum requirements
constexpr StringLiteral kMinSm = "min_sm";
constexpr StringLiteral kMinGfx = "min_gfx";
constexpr StringLiteral kRequiresFeatures = "requires_features";

// Tier 3: Variant ranking
constexpr StringLiteral kVariantPriority = "variant_priority";
constexpr StringLiteral kVariantTag = "variant_tag";
} // namespace metadata
} // namespace offload
```

---

## 3. Measurement Methodology

### 3.1 Benchmark Design

**Hardware:** GTX 1650 (Turing, sm_75, 4GB VRAM, CUDA 12.x) + CPU (host).

**Kernel:** Null kernel — 1 thread, 0 shared memory, 0 computation. Isolates
dispatch overhead from compute. Compiled to CUBIN (NOT PTX) ahead of time to
eliminate PTX JIT cost from cold path. This addresses the gap analysis correction
(combo-a-gaps.md "Layer 3 measurement architecture is incorrect").

**Critical fix: avoid double-load bug.** The gap analysis identified that calling
`cuModuleLoadData` separately from `olCreateProgram` double-loads the module.

Corrected measurement architecture:

```
Approach A (preferred): Instrumented liboffload build
  - Add clock_gettime brackets INSIDE the liboffload CUDA plugin
    (offload/plugins-nextgen/cuda/src/rtl.cpp)
  - Measures actual layer 2-3 decomposition without side effects
  - Requires building liboffload from source (already done for prototype)

Approach B (fallback): Separate baseline measurement
  - Measure 1: Raw cuModuleLoadData + cuModuleGetFunction + cuLaunchKernel
    (bypassing liboffload entirely — establishes driver-layer baseline)
  - Measure 2: olCreateProgram + olGetSymbol + olLaunchKernel
    (full liboffload path)
  - Subtract Measure 1 from Measure 2 to infer liboffload overhead
  - NEVER call both in the same measurement — that is the double-load bug

Approach C (what kdl.c already has): kdl_get_dispatch_latency_ns()
  - kdl.c lines 4595-4649: clock_gettime brackets around cuStreamSynchronize
  - 100 reps, reports median/p99
  - This is the HOT PATH ONLY measurement (post-module-load)
  - Extend to bracket each ol* call for full decomposition
```

### 3.2 Measurement Protocol

```
Phase 1: Cold path (one-time costs)
  Trial structure:
    Fork a fresh process per trial (eliminates driver caching)
    1. clock_gettime -> t0
    2. OffloadBinary::create(buf) -> parse fat binary
    3. clock_gettime -> t1  [Layer 1: OffloadBinary parse]
    4. olCreateProgram(device, blob, size, &prog)
    5. clock_gettime -> t2  [Layer 2+3: plugin load + cuModuleLoadData]
    6. olGetSymbol(prog, "null_kernel", OL_SYMBOL_KIND_KERNEL, &sym)
    7. clock_gettime -> t3  [Layer 4: symbol lookup]

  Repeat: 100 fresh-process trials
  Report: median, p50, p95, p99 for each layer delta

Phase 2: Hot path (amortized, cached module)
  Trial structure:
    Load module once (warm up)
    Discard first 1000 dispatches
    For i in 0..10000:
      clock_gettime -> t5
      olLaunchKernel(queue, device, sym, NULL, &dims)
      olWaitQueue(queue)
      clock_gettime -> t6
      record t6 - t5

  Report: per-dispatch histogram, p50/p95/p99

Phase 3: Variant selection overhead (libkdl-specific)
  Measure kdl_select_kernel() independently:
    For i in 0..100000:
      clock_gettime -> ts0
      kdl_select_kernel(ctx, "matmul", &result)
      clock_gettime -> ts1
      record ts1 - ts0

  This isolates the dispatch-table O(1) hash lookup cost.
  Expected: ~100-200 ns (hash + table lookup, no driver calls)
```

### 3.3 What Comparisons Are Meaningful on GTX 1650

| Comparison | What It Shows | Feasible? |
|-----------|--------------|-----------|
| Raw `cuLaunchKernel` vs. `olLaunchKernel` | liboffload overhead above driver | Yes |
| `olLaunchKernel` vs. kdl select+dispatch | Variant selection overhead | Yes |
| Cold `cuModuleLoadData` (CUBIN) vs. (PTX) | JIT cost isolation | Yes |
| Single-object vs. multi-object OffloadBinary parse | Per-variant compatibility check cost | Yes |
| NVIDIA dispatch vs. CPU fallback dispatch | Cross-vendor overhead comparison | Yes |
| GTX 1650 vs. H100 | Absolute latency comparison | **No** — state this explicitly |

**What to present:** Relative layer fractions (% of total dispatch time per layer),
NOT absolute microsecond values as generalizable claims. The argument is:
"libkdl's variant selection adds X ns, which is Y% of the dispatch floor on this
hardware — and the fraction holds across hardware generations because the dispatch
floor and the selection overhead both scale with software complexity, not compute
throughput."

### 3.4 Flame Graph Generation

```bash
# Cold-path flame graph (per-process, 100 trials)
for i in $(seq 1 100); do
  ./bench_cold_path >> cold_samples.txt
done
# Format: OffloadBinary::create;olCreateProgram;cuModuleLoadData <ns>
flamegraph.pl --title "Cold Path Dispatch (GTX 1650)" \
  --countname ns cold_samples.txt > cold_path.svg

# Hot-path flame graph (10000 dispatches, one process)
./bench_hot_path > hot_samples.txt
flamegraph.pl --title "Hot Path Dispatch (GTX 1650, n=10000)" \
  --countname ns hot_samples.txt > hot_path.svg
```

Two SVG panels on the poster. Cold path shows where module-load time goes.
Hot path shows the dispatch floor decomposition.

---

## 4. Upstream RFC Structure

### 4.1 RFC Ordering

Three RFCs, ordered by review risk (lowest first):

| Order | RFC | Risk | Depends On |
|-------|-----|------|-----------|
| 1 | OffloadBinary metadata vocabulary (T07) | Low — additive string keys, no ABI break | Nothing |
| 2 | Dispatch flame graph publication (T19) | Zero — measurement only, no code change | T07 numbers referenced |
| 3 | `#gpu.runtime_select` attribute (T01) | Medium — new code in GPU dialect | T07 keys for ranking |

**Rationale:** Metadata-first establishes the vocabulary that the runtime
selection attribute consumes. The flame graph provides the quantitative motivation
("here is what the unranked path costs"). The attribute RFC then has both the
data format and the measurement data to justify itself.

### 4.2 RFC 1: OffloadBinary Metadata Vocabulary

```
Title: [RFC] Standard capability metadata keys for OffloadBinary

Category: Runtimes / Offloading

CC: Joseph Huber (@jhuber6), Joel Denny (@jdenny-ornl),
    Yury Plyakhin (PR #169425 author), Saiyedul Islam (AMDGPU offload)

Background:
  OffloadBinary (D122069) uses a flexible string-map for per-image metadata.
  Only two keys are documented: "triple" and "arch". D127686 prototyped
  "feature=" for LTO but was never standardized.

Motivation:
  - PR #185663 introduced isMetadataCompatible() as the runtime filter.
  - PR #186088 uses it in parseOffloadBinary's first-compatible-wins loop.
  - The runtime has the consumer hook — but no standard vocabulary to consume.
  - KernelInfo pass already extracts register/occupancy data but drops it
    (never written to OffloadBinary).

Proposal:
  Define 5 standard string-map keys in two tiers:
  - Tier 1 (MUST): min_sm, min_gfx, requires_features
  - Tier 3 (MAY): variant_priority, variant_tag

  Add constexpr StringLiteral constants in OffloadBinary.h.
  Extend isMetadataCompatible() to check Tier 1 keys when present.
  Documentation patch for llvm-offload-binary.rst.

Compatibility:
  - Additive string keys; no format version bump needed.
  - Missing keys = no constraint (backward compatible).
  - Old runtimes ignore unknown keys (forward compatible).

Prototype:
  libkdl (EuroLLVM Dublin 2026 poster) implements equivalent
  matching via kdl_contract struct (kdl.c:112-122).

Patch sequence:
  1. Header constants + docs (zero functional change)
  2. isMetadataCompatible extension (opt-in when keys present)
  3. AMDGPU writer (emit min_gfx from target-id)
  4. NVPTX writer (emit min_sm from chip attribute)
  5. llvm-offload-binary --annotate flag
  Steps 2-5 are independent and can review in parallel after 1.
```

### 4.3 RFC 2: `#gpu.runtime_select` Attribute

```
Title: [RFC] Runtime variant selection for gpu.binary via #gpu.runtime_select

Category: MLIR / GPU Dialect

CC: Fabian Mora (@fabianmcg, RFC #88170 author),
    Joseph Huber (@jhuber6, liboffload),
    Joel Denny (@jdenny-ornl, KernelInfo pass)

Background:
  gpu-module-to-binary produces gpu.binary ops carrying N objects for N
  targets. The only consumer, #gpu.select_object, commits to one binary
  at LLVM IR translation time (compile time). RFC #88170 (Cleaning the
  GPU Dialect) explicitly separates container from dispatch policy —
  the policy slot is empty.

Proposal:
  A new attribute #gpu.runtime_select implementing
  OffloadingLLVMTranslationAttrInterface that:
  1. Embeds ALL binary blobs as separate LLVM globals
  2. Emits a global_ctors vendor-detection stub (dlopen-based)
  3. Selects the appropriate binary at module load time
  4. Generates identical launchKernel code to SelectObjectAttr
     (zero hot-path overhead after selection)

  Optional strategy parameter: first_compatible (default),
  rank_by_priority (uses OffloadBinary variant_priority key),
  rank_by_capability (device fingerprint scoring).

Relation to #88170:
  This fills the dispatch-policy slot that the cleanup RFC
  explicitly leaves vacant. Can land as part of the cleanup
  or immediately after.

Patch set:
  1. TableGen attribute definition + implementation (~400 LOC)
  2. --gpu-mark-runtime-select pass (~50 LOC)
  3. Runtime helper library (GPURuntimeSelectWrappers.cpp, ~200 LOC)
  4. Integration test (gpu-runtime-select.mlir)
  5. Lit test for marking pass (mark-runtime-select.mlir)
```

### 4.4 Review CC List

| Person | Role | Why CC |
|--------|------|--------|
| Fabian Mora (@fabianmcg) | GPU dialect cleanup RFC author | T01 lands in context of #88170 |
| Joseph Huber (@jhuber6) | liboffload / OffloadBinary maintainer | T07 vocabulary + T01 runtime consumer |
| Joel Denny (@jdenny-ornl) | KernelInfo pass author | T07 pipeline from KernelInfo → OffloadBinary |
| Yury Plyakhin | PR #169425 (format v2) | T07 format extension awareness |
| Saiyedul Islam | AMDGPU offload | T07 AMDGPU writer for min_gfx |

### 4.5 Minimum Viable Patches

| Component | Minimum Viable Patch | LOC |
|-----------|---------------------|-----|
| T07 metadata | Header constants + docs | ~30 |
| T07 consumer | `isMetadataCompatible` extension | ~40 |
| T01 attribute | TableGen def + embedBinary + launchKernel | ~400 |
| T01 pass | `--gpu-mark-runtime-select` | ~50 |
| T01 runtime | `GPURuntimeSelectWrappers.cpp` | ~200 |
| T01 tests | 2 .mlir test files | ~100 |
| **Total** | | **~820** |

---

## 5. Paper Structure (4-Page Extended Abstract)

### 5.1 Section Layout

| Section | Pages | Content |
|---------|-------|---------|
| **1. Introduction** | 0.5 | The gap: MLIR compiles to N GPU targets but selects 1 at compile time. Three-vendor landscape (NVVM + ROCDL + XeVM, August 2025) makes this urgent. Concrete contribution: `#gpu.runtime_select` attribute + OffloadBinary metadata vocabulary + first flame graph of LLVM dispatch stack. |
| **2. Background** | 0.75 | `gpu-module-to-binary` pipeline (3-line example), `#gpu.select_object` compile-time limitation, `OffloadingLLVMTranslationAttrInterface` extension point, OffloadBinary format (2 keys today). Cite RFC #88170 for the policy gap. |
| **3. Design** | 1.25 | (a) `#gpu.runtime_select` attribute: TableGen def, embedBinary pseudocode (N globals + dispatch table + vendor detection ctor), launchKernel (unchanged from SelectObjectAttr). (b) OffloadBinary metadata: 5 new keys, composition with isMetadataCompatible. (c) Prototype mapping: kdl.c dispatch table → LLVM IR emission correspondence table. |
| **4. Evaluation** | 1.0 | Dispatch flame graph (cold + hot path, GTX 1650). Layer decomposition table. Variant selection overhead measurement. Comparison: raw cuLaunchKernel vs. olLaunchKernel vs. kdl select+dispatch. Explicit statement: "relative fractions generalize; absolute values are hardware-specific." |
| **5. Related Work** | 0.25 | IREE (HAL module-level, issues open 6 years), chipStar (portability via SPIR-V, orthogonal), Proteus (JIT specialization, composable), CPU FMV (IFunc analogy, different layer). |
| **6. Conclusion** | 0.25 | Upstream path: metadata RFC first, attribute RFC second. Prototype at github.com/..., Discourse RFC drafts linked. |

### 5.2 Essential Figures

| Figure | Type | Shows |
|--------|------|-------|
| **Fig 1** | Pipeline diagram | `gpu.module` → targets → `gpu-module-to-binary` → `gpu.binary` → `#gpu.runtime_select` → vendor detection → dispatch. Highlight the gap between existing (blue) and proposed (red). |
| **Fig 2** | IR before/after | Left: `gpu.binary @k <#gpu.select_object<0>> [obj0, obj1]`. Right: `gpu.binary @k <#gpu.runtime_select<...>> [obj0, obj1]`. Show the LLVM IR diff: 1 global → N globals + dispatch table. |
| **Fig 3** | Flame graph (cold path) | SVG or rasterized. Layers: OffloadBinary parse → olCreateProgram → cuModuleLoadData → cuModuleGetFunction. Width = fraction of total cold-path time. |
| **Fig 4** | Flame graph (hot path) | olLaunchKernel → cuLaunchKernel. Shows that layers 1-4 collapse to zero after module load. |
| **Fig 5** | Bar chart | Dispatch overhead comparison: raw cuLaunchKernel baseline, olLaunchKernel overhead, kdl variant selection overhead. All measured on GTX 1650, same null kernel. |
| **Table 1** | Metadata vocabulary | 5 keys with types, examples, semantics. Compact reference. |
| **Table 2** | Layer decomposition | Per-layer latency (median, p95) for cold and hot paths. |

### 5.3 What NOT to Include in the Paper

Per the community-fit analysis:

- No 14-system comparison matrix (save for Q&A)
- No "roofline cost model" claim (call it "weighted heuristic")
- No MI300X numbers (prototype is GTX 1650 + CPU only)
- No tritonBLAS 94.7% validation citation (their model, not ours)
- No "future work" section on the poster face (save for conversation)
- No MTB format name (frame as prototype vehicle, not contribution)

---

## Appendix A: Correspondence Between kdl.c and Proposed LLVM IR

| kdl.c Component | Line Range | LLVM IR Equivalent |
|----------------|-----------|-------------------|
| `kdl_discover_cuda()` | 551-596 | `__gpu_runtime_select_detect_vendor()` CUDA probe |
| `kdl_discover_hip()` | 699-749 | `__gpu_runtime_select_detect_vendor()` HIP probe |
| `mtb_variant_entry` | 96-106 | `%RuntimeSelectEntry` struct type |
| `kdl_contract_matches()` | 1003-1005 | `isMetadataCompatible()` with Tier 1 keys |
| `kdl_estimate_cost_weighted()` | 1013-1088 | `__gpu_runtime_select_rank()` with strategy parameter |
| `kdl_cache_entry` / dispatch cache | 128-132 | `@kernels_module_ptr` (one-time selection, no per-launch cache needed) |
| `KDL_VENDOR_NVIDIA` / `KDL_VENDOR_AMD` | enum values | Vendor ID constants in dispatch table |

## Appendix B: Risk Register (Updated from Gap Analysis)

| Risk | Status | Mitigation |
|------|--------|------------|
| XeVM PR #119440 conflation | **OPEN** — must find correct PR | Search LLVM monorepo for XeVM PRs in Aug 2025 range |
| TaxBreak 4.71 μs attribution | **OPEN** — must verify against PDF | Read arXiv:2603.12465 Section 4; if not present, cite ICPP 2019 instead |
| Layer 3 double-load bug | **FIXED** — Approach A/B in section 3.1 | Instrumented build or separate baseline, never concurrent |
| MI300X claim in T07 pitch | **FIXED** — removed; prototype is GTX 1650 + CPU only | |
| `CompilationAttrInterfaces.td` filename | **CONFIRMED** — path is correct per infrastructure doc | |
| PR #186088 merge status | **MUST CHECK** — affects all three topics | Verify at github.com before poster |
| dlopen policy debate | **MITIGATED** — cite JAX/PyTorch precedent on poster | |

---

*Design notes:*
*Grounded in: SelectObjectAttr.cpp (doxygen), CompilationAttrInterfaces.td,
ModuleToBinary.cpp, OffloadBinary.h/cpp, kdl.c (lines 1-200, 550-650, 1000-1090),
combo-a-gaps.md, combo-a-tough-questions.md, community-fit-analysis.md*
*Date: 2026-04-09*
```

---

## Recommendations

1. **Finalize this document** -- the content above is the complete design for the dispatch layer. Effort: trivial. Impact: unlocks all downstream work.

2. **Resolve the two CRITICAL open risks before poster** -- (a) find the correct XeVM upstreaming PR number (search `git log --oneline --all --grep="xevm"` in the LLVM monorepo or `gh pr list --search "xevm merged:2025-08-01..2025-09-01"`), (b) verify TaxBreak 4.71us figure against the actual PDF at arXiv:2603.12465. Effort: 1 hour. Impact: prevents credibility collapse.

3. **Run `bench_dispatch` on GTX 1650 NOW** -- Q20 from the tough-questions doc is the kill shot. You need YOUR numbers, not borrowed ones. The measurement harness exists in `kdl.c:4595-4649`. Effort: 30 minutes. Impact: the single most important data point for the poster.

4. **Check PR #186088 merge status** -- all three topics depend on it. If merged, reframe as "the first-compatible-wins policy is now the DEFAULT in mainline, making ranked selection the obvious next step." Effort: 1 minute (`gh pr view 186088 --repo llvm/llvm-project`). Impact: strengthens or weakens framing of all three topics.

## Trade-offs

| Option | Pros | Cons |
|--------|------|------|
| Metadata RFC first (recommended) | Lowest reviewer friction; additive; establishes vocabulary before attribute needs it | Delays the "exciting" contribution (`#gpu.runtime_select`) behind a documentation/format patch |
| Attribute RFC first | Higher visibility; fills the gap RFC #88170 leaves open | Reviewers will ask "what metadata does ranking use?" and the answer is "we haven't standardized it yet" |
| Flame graph only (no RFC) | Zero upstream risk; pure measurement; publishable independently | "Interesting, but where's the code?" -- measurement-only posters get forgotten (per wave-07 criteria) |
| Ship all three simultaneously | Maximum impact; shows a complete story | Maximum review surface; risks reviewer fatigue; if one stalls, all are blocked |

## References

- `/home/akash/PROJECTS/LLVM/literature/mlir-gpu-infrastructure-2026.md:297-322` -- `OffloadingLLVMTranslationAttrInterface` TableGen definition (embedBinary + launchKernel signatures)
- `/home/akash/PROJECTS/LLVM/literature/mlir-gpu-infrastructure-2026.md:247-268` -- `SelectObjectAttr::embedBinary` produces `@serializedObj`, `@modulePtr`, `@loadFn`, `@unloadFn`
- `/home/akash/PROJECTS/LLVM/literature/mlir-gpu-infrastructure-2026.md:192-206` -- `gpu.binary` op TableGen with `offloadingHandler` attribute
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c:551-596` -- `kdl_discover_cuda()` dlopen-based CUDA detection
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c:1013-1088` -- `kdl_estimate_cost_weighted()` cost model with vendor constants
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c:112-122` -- `kdl_contract` struct (prototype metadata equivalent)
- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/synthesis/combo-a-gaps.md:319-341` -- Layer 3 double-load bug identification and fix
- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/synthesis/combo-a-gaps.md:13-48` -- XeVM PR #119440 conflation (CRITICAL)
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/synthesis/community-fit-analysis.md:17-27` -- Four credibility problems with current framing
- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/synthesis/combo-a-tough-questions.md:406-424` -- Q20: "show me YOUR numbers" kill shot
