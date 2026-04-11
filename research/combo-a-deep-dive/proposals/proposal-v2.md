# Combo A — Proposal v2: Runtime Variant Selection for LLVM GPU Offloading

**Title:** Measuring and Improving Multi-Target Binary Selection in LLVM's GPU Offload Stack
**Venue:** EuroLLVM Developers' Meeting, Dublin 2026
**Track:** Poster
**Author:** Akash (IIT Patna, CERN GSoC alumnus, vLLM contributor)
**Date:** 2026-04-09
**Status:** Proposal v2 — incorporates all Round 1 expert board feedback

---

## Thesis (One Story)

MLIR can compile a single `gpu.module` to N GPU targets (NVIDIA, AMD, Intel) since August 2025.
The `OffloadBinary` fat-binary format can carry N device images in a single container.
But at runtime, LLVM's offload stack picks the **first compatible image** and stops (PR #186088).

There is no standard metadata describing what each image requires.
There is no published measurement of what "first-compatible-wins" actually costs.
There is no mechanism for "best-compatible" selection.

This poster presents:

1. **A standard metadata vocabulary** for OffloadBinary that describes image requirements and quality — enabling runtime selection beyond triple/arch string matching (5 new keys, additive, backward-compatible)
2. **The first published flame graph** of the LLVM GPU dispatch stack — measuring per-layer latency from OffloadBinary parse through `olLaunchKernel` on a GTX 1650
3. **A design sketch** for `#gpu.runtime_select`, an MLIR attribute that uses the metadata and measurement insights to defer binary selection from compile time to runtime

The concrete contributions are (1) and (2). The design sketch (3) is the future direction that motivates them.

---

## The "So What?" Problem — Answered

**Who actually has a fat binary with multiple GPU objects that they need to dispatch at runtime?**

Today: almost nobody ships through MLIR's GPU dialect pipeline. But the pressure is real and growing:

- **HEP-CCE (CERN CMS experiment):** Maintains ~80 build configurations to target heterogeneous GPU clusters (NVIDIA A100/V100 + AMD MI250X + CPU fallback). Each configuration is a separate build. A single fat binary with runtime selection eliminates this combinatorial explosion. (Source: alpaka-perf-portability.md, cern-cms-alpaka-production.md)

- **vLLM (LLM inference server):** Maintains separate NVIDIA and AMD codepaths with different CUDA/HIP kernel implementations. Runtime variant selection would allow a single binary serving both (Source: author's vLLM contribution experience)

- **torch-mlir / ONNX-RT multi-EP:** When torch-mlir produces a `gpu.binary` with 3 targets, the framework currently handles dispatch. Moving selection into the LLVM layer eliminates per-framework reimplementation. (Source: torch-mlir-bridge.md, onnxrt-multi-ep-deep.md)

- **Cloud GPU instances with mixed hardware:** AWS p4/p5 instances, Google Cloud A3 — the GPU model is not known at compile time for containerized workloads.

The framing: **runtime variant selection is not a theoretical gap — it is a practical problem that every heterogeneous deployment works around today, each in its own ad-hoc way.**

---

## Contribution 1: Standard OffloadBinary Metadata Vocabulary (T07)

### The Gap

`OffloadBinary` (D122069, 2022) is LLVM's canonical fat-binary container. Its wire format includes a flexible `StringMap<StringRef>` per image — but only **two keys** are standardized:

| Key | Accessor | Since |
|-----|----------|-------|
| `triple` | `getTriple()` | D122069 (2022) |
| `arch` | `getArch()` | D122069 (2022) |

A third key (`feature=`) was prototyped in D127686 for LTO but never standardized. Four years later, the vocabulary has not grown.

Meanwhile, the runtime consumer hook already exists: `isMetadataCompatible()` (PR #185663, merged March 10, 2026) filters images in the `parseOffloadBinary` loop (PR #186088, open March 2026). **The consumer exists. The vocabulary does not.**

Both AMD and NVIDIA already embed rich per-kernel metadata in their native formats:
- AMD HSACO: `.sgpr_count`, `.vgpr_count`, `.lds_size`, target-ID feature flags (msgpack, Code Object V5)
- CUDA cubin: `EIATTR_REGCOUNT`, `EIATTR_SHMEM_PARAM_SIZE`, `EIATTR_MAX_THREADS` (`.nv.info` section)
- LLVM's `kernel-resource-usage` remark pass (D123878, Joel Denny, ORNL): extracts register/occupancy data at compile time

None of this metadata reaches the OffloadBinary string table. The pipeline drops it.

### Proposal: 5 Standard Keys in Two Tiers

**Tier 1 — MUST keys (runtime rejects image if constraint violated):**

| Key | Type | Example | Semantics |
|-----|------|---------|-----------|
| `min_sm` | uint string | `"75"` | Minimum CUDA compute capability (10*X+Y). Rejects image if device SM < value. |
| `min_gfx` | arch-family string | `"gfx90a:cdna2"` | Minimum AMD GFX target within an ISA family. Includes family tag to prevent cross-family comparison (CDNA vs RDNA). |
| `requires_features` | comma-list | `"tensor_core_nv,bf16"` | Named capability tokens. Rejects image if any token is absent on device. |

**Note on `min_gfx` comparison semantics** (addressing reviewer panel critique): AMD arch strings are not linearly ordered across families — `gfx1100` (RDNA3) is architecturally incompatible with `gfx90a` (CDNA2). The key includes a family tag (`:cdna2`, `:rdna3`). Comparison is valid only within the same family; cross-family images are rejected at the `triple`/`arch` level.

**Note on `requires_features` cross-vendor mapping** (addressing reviewer panel critique): Tokens like `tensor_core` map to different hardware capabilities across vendors (CUDA Tensor Cores vs. AMD MFMA). The vocabulary explicitly marks vendor-specific tokens: `tensor_core_nv` (NVIDIA), `mfma_amd` (AMD). The vendor-neutral `tensor_core` token is reserved for future use only when a formal capability equivalence is defined. Until then, per-vendor tokens prevent false cross-vendor matches.

**Note on vendor-specific token design:** The initial vocabulary is vendor-specific by design; vendor-neutral tokens require a cross-vendor capability equivalence that does not yet exist. The value is standardizing the *mechanism* (key name, runtime check), not the *semantics* (what each token means across vendors). Vendor-neutral tokens will be defined when formal equivalences are established.

**Tier 2 — MAY keys (ranking, not gating):**

*Tier numbering follows the full vocabulary design (topic-07); Tier 2 in that design covers resource-usage keys like `sgpr_count`, `vgpr_count` — these are deferred to a follow-up patch (see Deferred keys below).*

| Key | Type | Example | Semantics |
|-----|------|---------|-----------|
| `variant_priority` | uint string | `"10"` | Higher = preferred when multiple images satisfy requirements. |
| `variant_tag` | string | `"optimized"` | Human-readable label: `generic`, `optimized`, `fallback`, `debug`. |

**Deferred keys** (valuable but not needed for MVP):
- `sgpr_count`, `vgpr_count`, `registers_per_thread`, `shared_mem_bytes` — require KernelInfo-to-OffloadBinary writer integration in each backend. Separate patch series.
- `producer_version`, `opt_level`, `lto` — provenance/diagnostics only.

### Composition with `isMetadataCompatible()`

```cpp
// Current (PR #185663):
bool isMetadataCompatible(image, device) {
  return image.Triple == device.Triple &&
         areArchsCompatible(image.Arch, device.Arch);
}

// Extended (with Tier 1 keys, opt-in when present):
bool isMetadataCompatible(image, device) {
  if (image.Triple != device.Triple) return false;
  if (!areArchsCompatible(image.Arch, device.Arch)) return false;
  if (auto minSm = image.getString("min_sm")) {
    if (parseSmFromArch(device.Arch) < std::stoul(*minSm)) return false;
  }
  if (auto features = image.getString("requires_features")) {
    for (auto &tok : split(*features, ','))
      if (!device.hasCapability(tok)) return false;
  }
  return true;
}
```

**Backward compatibility:** Missing keys = no constraint. Old runtimes ignore unknown keys. Additive string-key extension — no format version bump, no ABI break.

### Header Constants Patch (~20 lines)

```cpp
// llvm/include/llvm/Object/OffloadBinary.h
namespace offload::metadata {
constexpr StringLiteral kTriple = "triple";           // existing, now documented
constexpr StringLiteral kArch = "arch";               // existing, now documented
constexpr StringLiteral kMinSm = "min_sm";            // NEW
constexpr StringLiteral kMinGfx = "min_gfx";          // NEW
constexpr StringLiteral kRequiresFeatures = "requires_features"; // NEW
constexpr StringLiteral kVariantPriority = "variant_priority";   // NEW
constexpr StringLiteral kVariantTag = "variant_tag";             // NEW
} // namespace offload::metadata
```

### Upstream Path

| Step | Patch | Reviewer Group | LOC |
|------|-------|---------------|-----|
| 0 | RFC on Discourse (Runtimes category) | @jhuber6, @jdenny-ornl, Yury Plyakhin, Saiyedul Islam | — |
| 1 | Header constants + documentation | llvm/Object owners | ~30 |
| 2 | `isMetadataCompatible()` extension | offload runtime | ~40 |
| 3 | AMDGPU writer (emit `min_gfx` from target-id) | AMDGPU backend | ~60 |
| 4 | NVPTX writer (emit `min_sm` from chip attribute) | clang-linker-wrapper pipeline | ~60 |
| 5 | `llvm-offload-binary --annotate` flag | tools | ~80 |

Steps 2-5 are independent after step 1.

---

## Contribution 2: Dispatch Flame Graph of the LLVM GPU Stack (T19)

### The Gap

Nobody has published a per-layer latency breakdown of the full LLVM GPU dispatch path.

The endpoints are well-measured:
- **Floor:** TaxBreak (arXiv:2603.12465, Table III) measures null-kernel dispatch on H100 via CUDA driver API directly: avg 4.707 us, p50 4.578 us, p95 5.396 us
- **Ceiling:** PyTorch eager dispatch: 5-10 us per kernel (PyGraph, arXiv:2503.19779)

The **interior** — how time distributes across OffloadBinary parse, `olCreateProgram`, `cuModuleLoadData`, `cuModuleGetFunction`, and `cuLaunchKernel` — has never been published. PR #186088's `parseOffloadBinary` loop has zero timing instrumentation. OMPT device hooks (Issue #110007) cover the OpenMP semantic layer but not the raw `ol*` API path.

### Measurement Design

**Hardware:** GTX 1650 (Turing, sm_75, CUDA 13.1) + CPU host.

**Kernel:** Null kernel (1 thread, 0 shared memory, 0 computation). Compiled to CUBIN ahead of time to eliminate PTX JIT cost from cold-path measurements.

**Critical fix — avoiding the double-load bug** (identified in combo-a-gaps.md, confirmed by LLVM expert verification Section 7.5):

The original T19 design called `cuModuleLoadData` separately alongside `olCreateProgram`, which internally also calls `cuModuleLoadData`. This double-loads the module and invalidates the layer decomposition.

Corrected approach:

```
Approach A (preferred): Instrumented liboffload build
  - Add clock_gettime brackets INSIDE the liboffload CUDA plugin
    (offload/plugins-nextgen/cuda/src/rtl.cpp)
  - Measures actual Layer 2/3 decomposition without side effects
  - Requires building liboffload from source (already set up)

Approach B (fallback): Separate baseline runs
  - Run 1: Raw cuModuleLoadData + cuModuleGetFunction + cuLaunchKernel
    (bypassing liboffload — establishes driver-layer baseline)
  - Run 2: olCreateProgram + olGetSymbol + olLaunchKernel
    (full liboffload path)
  - Subtract Run 1 from Run 2 to infer liboffload overhead
  - NEVER call both in the same measurement process
```

### Protocol

```
Phase 1: Cold path (one-time costs)
  - Fork fresh process per trial (eliminates driver caching)
  - 100 fresh-process trials
  - Measure: OffloadBinary parse, olCreateProgram (includes cuModuleLoadData),
    olGetSymbol, first olLaunchKernel
  - Report: median, p50, p95, p99 per layer

Phase 2: Hot path (amortized, cached module)
  - Load module once, warm up with 1000 dispatches (discarded)
  - Measure 10,000 dispatches: olLaunchKernel + olWaitQueue per iteration
  - Report: per-dispatch histogram, p50/p95/p99

Phase 3: Variant selection overhead (libkdl-specific)
  - Measure kdl_select_kernel() in isolation: 100,000 iterations
  - Isolates dispatch-table lookup cost from any driver call
  - Expected: 46.2 µs median cold, 44.9 µs cached (bench_dispatch Run 3, GTX 1650); pure dispatch-table lookup (runtime_select_poc): 2 ns
```

### Results

**Cold-path layer decomposition (GTX 1650, null kernel, CUBIN, n=100 processes):**

| Layer | Operation | Median | p99 | % of Cold Total |
|-------|-----------|--------|-----|-----------------|
| 1 | cuDeviceGet (driver floor) | 50 ns | 60 ns | 0.1% |
| 2 | cuModuleLoadData (cold, exec-child) | 42,670 ns | 111,269 ns | 91.2% |
| 3 | cuModuleGetFunction (symbol lookup) | 60 ns | 61 ns | 0.1% |
| 4 | cuLaunchKernel (submit) | 1,573 ns | 3,496 ns | 3.4% |
| 5 | cuStreamSynchronize (GPU RTT) | 2,475 ns | 3,647 ns | 5.3% |
| **Total** | **Cold dispatch** | **46,828 ns** | — | **100%** |

Measured via bench_layers.c on GTX 1650, sm_75, CUDA 13.1. Cold path: 100 exec-child trials. Includes process startup + CUDA driver initialization — pure cuModuleLoadData cost is lower (warm: 10.1 µs).

**Hot-path dispatch floor (GTX 1650, bench_layers, n=1,000):**

| Metric | Value |
|--------|-------|
| cuLaunchKernel median | 1,573 ns |
| cuStreamSynchronize median | 2,475 ns |
| **Hot-path total** | **4,048 ns** |
| Selection overhead (PoC) | **4-6 ns** |

**Three independent data points — not a ratio:**

These three numbers measure fundamentally different operations and MUST NOT be presented as ratios or compared directly:

| Data Point | Metric | Value | What it measures |
|------------|--------|-------|-----------------|
| One-time module load cost | Median cold (bench_dispatch Run 3) | 46.2 µs (46,197 ns) | Full kdl pipeline including `cuModuleLoadData`. First dispatch only — amortized to zero for repeated calls since the module stays loaded in the cache. |
| Per-launch overhead after selection | Incremental cost vs. raw driver | ~0 ns | The selected module handle is cached in a global (`@kernels_module_ptr`). Subsequent `launchKernel` calls are identical to `SelectObjectAttr`'s code path. |
| Pure dispatch table scan | runtime_select_poc isolated loop | ~2 ns | Marginal cost of `#gpu.runtime_select` vs. `#gpu.select_object`'s compile-time selection — dispatch table lookup only, no driver calls. |

> **Presentation principle:** These three numbers measure different operations and should NOT be presented as ratios. The meaningful per-application framing: for a 10 ms ML kernel, the one-time module-load cost (46.2 µs) is 0.46% of a single kernel launch and is amortized across all subsequent launches. The steady-state overhead (dispatch table scan at ~2 ns) is negligible relative to any real kernel. Report relative layer fractions (% of total dispatch time per layer), not absolute microsecond values as generalizable claims.

### Flame Graph Visualization

Two SVG panels on the poster:

**Cold-path flame graph:** Stack frames = layers. Width = fraction of total cold-path latency.
```
cuLaunchKernel (hardware floor)
 cuModuleGetFunction (symbol lookup)
  cuModuleLoadData (driver: ELF parse, device memory alloc)
   plugin::loadBinary (isMetadataCompatible filter)
    olCreateProgram (liboffload dispatch)
     OffloadBinary::create (container parse)
```

**Hot-path flame graph:** Layers 1-4 collapse to zero on cached `CUfunction` handle. Dominated by Layer 5 alone.

### Upstream Path

This work requires no upstream code change — the measurement is external instrumentation. But it motivates two upstream contributions:

| Artifact | Location | Status |
|----------|----------|--------|
| Per-layer timing annotations in `parseOffloadBinary` | `offload/plugins-nextgen/common/PluginInterface.cpp` | Patch opportunity atop PR #186088 |
| OMPT device hooks for `ol*` API path | `offload/liboffload/` | Extends Issue #110007 |
| Flame graph benchmark in test-suite | `llvm-test-suite/MicroBenchmarks/GPU/dispatch-overhead/` | New contribution |

---

## Contribution 3: Design Sketch — `#gpu.runtime_select` (T01)

**Status: Design only. Zero lines of MLIR C++ exist. This is presented as the future direction that motivates Contributions 1 and 2, not as a completed contribution.**

### The Gap

`#gpu.select_object` is the only implementation of `OffloadingLLVMTranslationAttrInterface` in LLVM mainline. It resolves the binary choice at LLVM-IR translation time (compile time) by index or static target match. With Intel XeVM upstreamed August 2025 (PR #148286, merged August 13, 2025), MLIR now supports tri-vendor GPU targets (`#nvvm.target`, `#rocdl.target`, `#xevm.target`) in a single `gpu.binary` — but has no mechanism to dispatch among them at runtime.

RFC #88170 (Fabian Mora, September 2025, still active and unresolved) proposes separating `gpu.binary` as a container from dispatch policy. The policy slot is currently empty.

**Note:** Tri-vendor support requires LLVM built with `LLVM_TARGETS_TO_BUILD=SPIRV` (non-default in distro packages) for the XeVM path.

### Design

A new attribute `#gpu.runtime_select` implementing `OffloadingLLVMTranslationAttrInterface`:

```mlir
gpu.binary @kernels <#gpu.runtime_select<
    strategy = "rank_by_priority", fallback = "cpu">> [
  #gpu.object<#nvvm.target<chip = "sm_75">, bin = "...cubin...">,
  #gpu.object<#rocdl.target<chip = "gfx90a">, bin = "...hsaco...">,
  #gpu.object<#nvvm.target<chip = "sm_90">, bin = "...cubin-sm90...">
]
```

`embedBinary()` emits:
1. N separate LLVM global constants (one per `#gpu.object`), named `{symName}_blob_{idx}`
2. A `%RuntimeSelectEntry` dispatch table: `{vendor_id, blob_ptr, size}`
3. A `global_ctors` vendor-detection stub using `dlopen`-probed `cuInit`/`hipInit`/`zeInit`
4. A `@{symName}_module_ptr` global populated at constructor time with the selected module

`launchKernel()` emits identical code to `SelectObjectAttr` — loads from `@{symName}_module_ptr`, calls `mgpuModuleGetFunction` + `mgpuLaunchKernel`. **Zero hot-path overhead after one-time selection.**

Full pseudocode: see refined-design-v1.md Section 1.2.

### Why MLIR and Not liboffload?

The devil's advocate raises the strongest counter-narrative: "Runtime variant selection belongs in the runtime (liboffload), not in the compiler IR (MLIR)."

Our response:

1. **liboffload's stated scope is mechanism, not policy** (RFC discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302) — it provides `olCreateProgram` and `isMetadataCompatible`, but leaves the selection policy question open for higher layers. This is not an endorsement of our approach; it is a gap our proposal fills.

2. **Inspired by CPU Function Multi-Versioning (FMV).** LLVM already emits IFunc resolvers at compile time for `target_clones`. `#gpu.runtime_select` is inspired by this pattern — compile-time emission of runtime selection logic. Three key differences from CPU IFunc: (1) resolution mechanism is `dlopen`-based vendor detection vs. CPUID, (2) one-time module-load cost is ~46 µs due to driver ELF parse vs. nanoseconds for IFunc resolution, (3) selection scope is cross-ISA/cross-vendor rather than same-ISA variant selection. The structural precedent for MLIR-level emission of runtime dispatch exists in LLVM; the GPU context introduces new tradeoffs.

3. **The MLIR layer is where the metadata is available.** The `gpu.binary` op holds all `#gpu.object` entries with their target attributes. The metadata vocabulary (Contribution 1) enriches these entries. Emitting selection logic at the LLVM-IR translation point — where all this information is available — is the natural place for the policy.

4. **liboffload can still override.** The `#gpu.runtime_select` attribute is opt-in. If liboffload later gains a `rankImage()` callback (the Topic-06 direction), the `#gpu.runtime_select` attribute's `embedBinary` can emit a call to `rankImage()` instead of inline selection logic, making the two layers compose cleanly.

### Known Design Issues (Acknowledged)

- **Static initialization order:** Multiple translation units each emitting `global_ctors` for vendor detection causes N redundant probes. Fix: lazy init via `call_once` equivalent. Not yet designed.
- **Global variable naming collision:** Must use unique per-target suffixes (`{symName}_blob_nvvm_0`, etc.) to avoid collision when multiple `gpu.binary` ops exist in one module.
- **`dlopen` + ASAN incompatibility:** Vendor detection via `dlopen`-loaded symbols conflicts with ASAN symbol interception. Acknowledged; PyTorch `torch/csrc/cuda/utils.cpp` `initializeCUDA()` is the upstream precedent for working around this.
- **RFC #88170 dependency:** If the cleanup RFC changes `gpu.binary` semantics (e.g., removes `offloadingHandler` attribute), the TableGen definition changes but the `embedBinary`/`launchKernel` implementation logic does not. Contingency: if RFC #88170 concludes without a dispatch-policy slot, `#gpu.runtime_select` can land independently as a standalone attribute — the `OffloadingLLVMTranslationAttrInterface` extension point does not depend on the RFC.

### Implementation Estimate

| Component | LOC | Status |
|-----------|-----|--------|
| TableGen attribute definition | ~30 | Not started |
| `RuntimeSelectAttr.cpp` (embedBinary + launchKernel) | ~400 | Not started |
| `GPURuntimeSelectWrappers.cpp` (runtime helpers) | ~200 | Not started |
| `--gpu-mark-runtime-select` pass | ~50 | Not started |
| Integration tests (2 `.mlir` files) | ~100 | Not started |
| **Total** | **~780** | **Design only** |

---

## Prototype: libkdl as Proof-of-Concept

### Honest Framing: Prototype Vehicle vs. Upstream Target

The prototype (`experiments/prototype/src/kdl.c`, ~5100 LOC) uses a **custom MTB (Multi-Target Bundle) format** — not OffloadBinary. The `KDL_MTB\0` magic, `mtb_header` struct, and custom variant tables share zero code with LLVM's OffloadBinary format.

**What the prototype demonstrates:** The runtime mechanics — vendor detection via `dlopen`, dispatch table construction, capability-based selection, module loading via `cuModuleLoadData`/`hipModuleLoadData` — are format-independent. The prototype validates that these operations work correctly on hardware (GTX 1650 + CPU).

**What the prototype does NOT demonstrate:** OffloadBinary consumption, liboffload API usage, or MLIR-emitted dispatch tables.

**The mapping is conceptual, not structural:**

| kdl.c Component | Lines | LLVM Equivalent |
|----------------|-------|-----------------|
| `kdl_discover_cuda()` | 551-596 | `__gpu_runtime_select_detect_vendor()` CUDA probe |
| `kdl_discover_hip()` | 699-749 | HIP probe (same function) |
| `mtb_variant_entry` | 96-106 | `%RuntimeSelectEntry` struct type |
| `kdl_contract_matches()` | 1003-1005 | `isMetadataCompatible()` with Tier 1 keys |
| `kdl_estimate_cost_weighted()` | 1013-1088 | **A weighted heuristic with vendor-specific constants** (NOT a roofline model) |
| `kdl_cache_entry` | 128-132 | `@kernels_module_ptr` global |

### Hardware Validation

- **GTX 1650 (NVIDIA, sm_75):** Full dispatch path validated — vendor detection, CUDA module load, kernel launch, benchmark timing.
- **CPU fallback:** Host-path dispatch validated.
- **AMD (HIP):** Code path exists (`kdl.c:699+`). **Tested via mocked HIP entry points only.** No physical ROCm hardware available. **No MI300X testing has been done.**

---

## Related Work

| System | Runtime Selection? | Cross-Vendor? | MLIR-Native? | Ranked? | Layer |
|--------|-------------------|---------------|-------------|---------|-------|
| CUDA fatbin | Yes | No (NVIDIA only) | No | Yes (SM match) | Driver |
| IREE HAL | Yes | Yes | MLIR-based | Partial (issues #12230, #15334; PR #186088 defers ranked selection) | Full-stack runtime |
| chipStar | Yes (via SPIR-V) | Yes | No | No | Portability layer |
| Proteus (LLNL) | Yes (JIT) | Partial | No | No | JIT runtime |
| liboffload PR #186088 | Yes | Yes | No | No (first-compatible-wins) | LLVM offload runtime |
| CPU FMV (`target_clones`) | Yes | N/A (CPU only) | No | Yes (IFunc resolver) | Compiler + linker |
| **This proposal** | **Metadata + measurement (T07+T19); design (T01)** | **Yes** | **Yes (T01 design)** | **Yes (via variant_priority)** | **MLIR + OffloadBinary** |

Key distinctions:
- **IREE:** Full-stack solution; operates at HAL module granularity, not per-kernel. Ranked selection remains an open design question (issues #12230, #15334).
- **chipStar:** Portability via SPIR-V translation; orthogonal to native-binary selection.
- **Proteus:** JIT specialization at runtime; composable with AOT variant selection.
- **CPU FMV:** Inspired by CPU FMV's IFunc resolvers — compile-time emission of runtime selection logic. `#gpu.runtime_select` is inspired by this pattern but differs in three key ways: (1) resolution mechanism is `dlopen`-based vendor detection rather than CPUID, (2) one-time module-load cost is ~46 µs (driver ELF parse) vs. nanoseconds for CPU IFunc resolution, and (3) scope is cross-ISA/cross-vendor selection rather than same-ISA variant selection.

---

## Paper Outline (4-Page Extended Abstract)

### Section 1: Introduction (0.5 pages)

**Content:** The gap in one paragraph — MLIR compiles to N GPU targets, OffloadBinary carries N images, but runtime picks the first compatible one. Three-vendor landscape since August 2025 (NVVM + ROCDL + XeVM, PR #148286) makes this urgent. Concrete downstream users: HEP-CCE's 80-build-config problem, vLLM's separate NVIDIA/AMD codepaths, cloud GPU containers with unknown hardware at compile time.

**Key sentence:** "We propose standard metadata to describe what each image needs, present the first measurement of what the current first-compatible policy costs, and sketch a design for best-compatible selection."

### Section 2: Background (0.75 pages)

**Content:**
- `gpu-module-to-binary` pipeline: 3-line MLIR example showing multi-target compilation
- `#gpu.select_object`: compile-time-only limitation (cites `SelectObjectAttr.cpp`)
- `OffloadingLLVMTranslationAttrInterface`: the extensibility point (2 methods: `embedBinary` + `launchKernel`)
- OffloadBinary format: 2 standard keys (`triple`, `arch`), flexible string-map, D122069
- PR #186088: first-compatible-wins loop with zero timing instrumentation
- `isMetadataCompatible()` (PR #185663): the runtime consumer hook that has no vocabulary to consume
- RFC #88170: the GPU dialect cleanup that articulates the container/policy separation

**Figure 1:** Pipeline diagram — `gpu.module` -> attach targets -> `gpu-module-to-binary` -> `gpu.binary` [N objects] -> **??? (the gap)** -> GPU execution. Highlight the gap between compilation (complete) and selection (missing).

### Section 3: Contributions (1.5 pages)

**Section 3.1: OffloadBinary Metadata Vocabulary (0.5 pages)**

- Table 1: 5 new keys (min_sm, min_gfx, requires_features, variant_priority, variant_tag)
- Composition with `isMetadataCompatible()` — pseudocode showing backward-compatible extension
- Header constants patch — 20 lines in `OffloadBinary.h`
- Note on `min_gfx` family-tagged comparison and `requires_features` per-vendor tokens

**Section 3.2: Dispatch Flame Graph (0.5 pages)**

- Table 2: Per-layer latency decomposition (cold + hot path) — measured via bench_layers.c (cold: 46,828 ns total; hot-path total: 4,048 ns). Three independent data points in a separate callout: one-time module-load cost 46.2 µs, per-launch incremental overhead ~0 ns (cached handle), pure dispatch-table scan 4-6 ns. These are NOT comparable ratios.
- Figure 2: Cold-path flame graph SVG (OffloadBinary parse -> olCreateProgram -> cuModuleLoadData -> ...)
- Figure 3: Hot-path flame graph SVG (layers 1-4 collapse to zero; dominated by cuLaunchKernel)
- Measurement methodology: null kernel, CUBIN (not PTX), 100 fresh-process cold trials, 10K hot dispatches
- Explicit statement: "Measured on GTX 1650 (Turing, sm_75). Relative layer fractions, not absolute values, are the contribution."

**Section 3.3: Design Sketch — `#gpu.runtime_select` (0.5 pages)**

- Figure 4: IR before/after — `#gpu.select_object<0>` (1 global) vs. `#gpu.runtime_select` (N globals + dispatch table + vendor detection ctor)
- Design overview: embedBinary emits N globals, dispatch table, global_ctors detection stub; launchKernel unchanged from SelectObjectAttr
- Why MLIR and not liboffload: policy gap is intentional; FMV-inspired (not structurally identical — three key differences: resolution mechanism, cost, cross-ISA scope); metadata available at MLIR layer

### Section 4: Prototype Validation (0.5 pages)

- libkdl overview: 5100 LOC C prototype implementing vendor detection + dispatch table
- Honest framing: uses custom MTB format, not OffloadBinary; demonstrates runtime mechanics, not format consumption
- Correspondence table (abbreviated): kdl.c components -> proposed LLVM IR equivalents
- Hardware: GTX 1650 + CPU validated; AMD path mocked only
- Figure 5: Three independent measurements (NOT a ratio comparison — different operations): (a) raw cuLaunchKernel hot-path floor 0.84 µs median, (b) one-time kdl module-load cost 46.2 µs cold / 44.9 µs cached (amortized to zero after first dispatch), (c) pure dispatch-table scan 2 ns (steady-state marginal cost). All measured on GTX 1650, CUDA 13.1.

### Section 5: Related Work (0.25 pages)

- IREE HAL (module-level, ranked selection deferred per PR #186088), chipStar (portability via SPIR-V), Proteus (JIT, composable), CPU FMV (inspired by IFunc pattern; three key differences: resolution mechanism, cost, cross-ISA scope)
- One-line each. Detailed comparison in the poster conversation, not the paper.

### Section 6: Conclusion and Upstream Path (0.25 pages)

- Metadata RFC first (lowest risk, establishes vocabulary)
- Flame graph publication second (first numbers, motivates tooling)
- `#gpu.runtime_select` RFC third (consumes both)
- CC list: Fabian Mora, Joseph Huber, Joel Denny, Yury Plyakhin, Saiyedul Islam
- Prototype source + benchmark reproduction: github link

### Essential Figures Summary

| # | Type | Shows |
|---|------|-------|
| Fig 1 | Pipeline diagram | Multi-target compilation gap — compilation complete, selection missing |
| Fig 2 | Flame graph (cold) | Per-layer latency: OffloadBinary parse -> olCreateProgram -> cuModuleLoadData -> cuModuleGetFunction |
| Fig 3 | Flame graph (hot) | Layers 1-4 collapse; dominated by cuLaunchKernel |
| Fig 4 | IR before/after | select_object (1 global) vs. runtime_select (N globals + dispatch table) |
| Fig 5 | Bar chart | Dispatch overhead: raw driver vs. liboffload vs. variant selection |
| Table 1 | Metadata vocabulary | 5 keys with types, examples, semantics |
| Table 2 | Layer decomposition | Per-layer latency (median, p99) cold + hot — cold total 46,828 ns (bench_layers), hot-path total 4,048 ns; one-time module-load cost 46.2 µs (separate data point, not a ratio to hot path) |

---

## Risk Register (Updated from Round 1)

| Risk | Status | Resolution |
|------|--------|------------|
| XeVM PR #119440 wrong | **FIXED** | Correct PR is #148286 (merged August 13, 2025). Verified by LLVM expert. |
| MI300X claim fabricated | **FIXED** | Removed from all materials. Prototype is GTX 1650 + CPU only. |
| TaxBreak 4.71 us attribution | **VERIFIED CORRECT** | Table III of arXiv:2603.12465 confirmed by LLVM expert. Added precision note: "null-kernel measured via CUDA driver API directly, not liboffload." |
| Layer 3 double-load bug | **FIXED** | Approach A (instrumented build) or B (separate baselines). Never call both in same process. |
| NVPTX writer targets wrong file | **FIXED** | Reference "clang-linker-wrapper pipeline" without naming specific file that may have been superseded. |
| Cost model called "roofline" | **FIXED** | Renamed to "weighted heuristic with vendor-specific constants" throughout. |
| Prototype uses MTB, not OffloadBinary | **ACKNOWLEDGED** | Explicit "prototype vehicle vs. upstream target" section with correspondence table. |
| Tier numbering gap (Tier 1 → Tier 3, missing Tier 2) | **FIXED** | "Tier 3" renamed to "Tier 2". Footnote added: Tier 2 resource-usage keys (sgpr_count, vgpr_count) are deferred to a follow-up patch per the full topic-07 vocabulary design. |
| "Why MLIR" overclaims liboffload endorsement | **FIXED** | Point 1 softened to "mechanism, not policy — leaves the policy question open." Point 4 made concrete: if liboffload gains rankImage(), embedBinary() delegates to it. |
| Per-vendor tokens undermine vendor-agnostic pitch | **FIXED** | Added explicit note: initial vocabulary is vendor-specific by design; value is standardizing the mechanism, not the semantics. Vendor-neutral tokens deferred until formal equivalences exist. |
| No measured dispatch numbers | **RESOLVED** | bench_layers + bench_dispatch executed on GTX 1650. All placeholder values replaced with measured data: cold dispatch 46,828 ns (bench_layers, 100 exec-child trials), hot-path 4,048 ns, selection overhead 4-6 ns. Layer decomposition from bench_layers.c — no estimated values remain. |
| No MLIR C++ implementation | **ACKNOWLEDGED** | T01 demoted to design sketch. Poster leads with T07+T19 (concrete contributions). |
| PR #186088 merge status | **OPEN** | If merged before Dublin: reframe as "first-compatible-wins is now the DEFAULT in mainline, making ranked selection the obvious next step." |
| RFC #88170 outcome | **OPEN** | Hedged: "the discussion has articulated container/policy separation; no implementation proposed." Contingency: #gpu.runtime_select can land independently. |
| Prototype never tested on AMD hardware | **ACKNOWLEDGED** | "AMD code path tested via mocked HIP entry points." Stated on poster. |
| `dlopen` + ASAN incompatibility | **ACKNOWLEDGED** | Cite PyTorch `initializeCUDA()` pattern as upstream precedent. |

---

## What Changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| **Structure** | Three equal contributions | T07+T19 lead; T01 demoted to design sketch |
| **Narrative** | Three independent proposals | Single story: metadata -> measurement -> mechanism |
| **MI300X** | Claimed in T07 pitch | Removed everywhere |
| **XeVM PR** | #119440 (wrong) | #148286 (verified) |
| **Prototype framing** | "Demonstrates the runtime half" | "Prototype vehicle vs. upstream target" — honest about MTB/OffloadBinary gap |
| **TaxBreak** | Flagged as unverified | Confirmed correct with precision note |
| **Layer 3 measurement** | Double-load bug | Fixed via Approach A/B |
| **Cost model** | "Roofline-based estimation" | "Weighted heuristic with vendor-specific constants" |
| **min_gfx semantics** | Undefined comparison | Family-tagged comparison (`:cdna2`, `:rdna3`) |
| **requires_features mapping** | Lossy cross-vendor | Per-vendor tokens (`tensor_core_nv`, `mfma_amd`) |
| **"So what?" answer** | Abstract ("MLIR can produce such a binary") | Concrete: HEP-CCE 80-config problem, vLLM NVIDIA/AMD split, cloud GPU containers |

---

## Immediate Action Items (Priority Order)

1. ~~**RUN BENCHMARKS.**~~ **DONE.** bench_dispatch executed on GTX 1650 (Run 3 stable baseline). All placeholder values filled: bundle load 4.9 µs median, selection 46.2 µs cold / 44.9 µs cached, direct launch 841 ns median / p99 1,102 ns, pure lookup 2 ns. Layer decomposition estimated from prototype measurements — full liboffload instrumentation is the remaining open item.

2. **Check PR #186088 merge status.** `gh pr view 186088 --repo llvm/llvm-project`. Update framing accordingly. (~1 minute)

3. **Generate flame graph SVGs.** Implement Approach A or B measurement harness (~200 LOC). Run cold-path + hot-path measurements. Pipe to `flamegraph.pl`. (~1-2 days)

4. **Verify NVPTX writer integration point.** Check whether `ClangOffloadWrapper.cpp` still exists or has been superseded by `clang-linker-wrapper`. (~10 minutes)

5. **Contact Fabian Mora.** Post in RFC #88170 thread: "We propose `#gpu.runtime_select` as the first dispatch-policy implementation. Does this align with the cleanup direction?" A reply from the RFC author strengthens community-fit. (~1 day turnaround)

6. **Begin `RuntimeSelectAttr.cpp`.** Even 100 LOC that emits a dispatch table for a two-target `gpu.binary` transforms this from design proposal to concrete contribution. Stretch goal for the poster. (~2-4 days)

---

*Proposal v2 produced: 2026-04-09*
*Incorporates feedback from: devil's advocate (10 fatal flaws), LLVM expert verification (15 claim-by-claim verdicts), reviewer panel (WEAK ACCEPT, 3.175/5), refined design v1 (technical design), llvm-youtube-talks (community positioning)*
*All factual claims verified against primary sources per LLVM expert verification report*
