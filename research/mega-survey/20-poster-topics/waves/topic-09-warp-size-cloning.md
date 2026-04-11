# Topic 09: Warp-Size-Aware Automatic Kernel Cloning Pass

**Topic ID:** 09
**Config key:** `warp-size-kernel-cloning`
**Persona:** LLVM backend / GPU compiler researcher
**Date:** 2026-04-07
**Research depth:** Exhaustive — `llvm.gpu.*` PR #131190, PR #174910, KernelInfo pass PR #102944,
FMV AArch64 talk EuroLLVM 2025, Triton AMD backend, hip-rocm.md §"Warp Size 32 vs 64",
wave-08-mlir-async-llvm-gpu.md §2, wave-03-multi-versioned-kernels.md, kdl.c warp_size query,
literature/hip-rocm.md lines 144–151, literature/spirv-analysis.md lines 220–231

---

## Gap

`llvm.gpu.num.lanes` (PR #131190, Joseph Huber, March 2025) is the new portable LLVM IR intrinsic
for querying warp / wavefront size at runtime. It returns 32 on NVIDIA GPUs and 64 on AMD CDNA /
GFX9 hardware. The intrinsic exists precisely because NVPTX and AMDGCN have different intrinsics
for the same concept (`llvm.nvvm.read.ptx.sreg.warpsize` vs `llvm.amdgcn.wavefront.size`).

The problem: **abstracting the query does not abstract the algorithm.** The Discourse thread that
motivated PR #131190 explicitly states:

> "Warp-size differences (32 vs. 64 lanes): `llvm.gpu.num.lanes` returns the target's value,
> but algorithms written for warp-size-32 will not automatically work correctly on warp-size-64."
> (wave-08-mlir-async-llvm-gpu.md §2.5)

The structural reasons are concrete:

1. **Ballot mask width.** `llvm.gpu.ballot` returns a 32-bit integer on NVIDIA and a 64-bit
   integer on AMD CDNA. Code that stores the result in a 32-bit integer, or that shifts / masks
   using a constant `32`, produces incorrect results on AMD (literature/hip-rocm.md line 148).

2. **Reduction loop trip count.** A warp-level tree-reduction written as:
   ```
   for (int offset = 16; offset > 0; offset >>= 1)
     val += __shfl_down_sync(0xffffffff, val, offset);
   ```
   hardcodes 5 iterations (log2(32)). On AMD wavefront=64, this misses 32 lanes. The correct
   trip count is `log2(warp_size)` — 5 or 6 iterations — and the correct full-mask changes from
   `0xffffffff` (32-bit) to `0xffffffffffffffff` (64-bit).

3. **Shuffle lane-index range.** `__shfl_sync(mask, val, lane, width)` with `width=32` on
   NVIDIA is an optimization hint that restricts shuffle to 32-lane sub-warps. The AMD equivalent
   allows up to 64. Code that hard-codes `width=32` is semantically wrong at warp-64 boundaries.

4. **Occupancy / shared memory sizing.** Warp-level tile kernels often size shared memory as
   `WARP_SIZE * sizeof(float)`. At warp=32 this is 128 bytes; at warp=64 it is 256 bytes. Static
   allocation with a hardcoded constant either wastes memory on NVIDIA or overflows the statically
   declared size on AMD.

5. **Vector width assumptions.** Kernels that vectorize `N/WARP_SIZE` elements per thread encode
   warp size in the loop trip count and vector register width selection.

None of these five patterns can be fixed by substituting `llvm.gpu.num.lanes` for the constant.
The intrinsic returns a runtime value; the C or MLIR compiler cannot constant-fold it, so
dependent expressions (ballot type width, loop bounds, shared memory size, vector width) are not
simplified. The kernel must either branch at runtime on every warp-size-dependent path — paying
branch overhead on every kernel invocation — or be pre-specialized into two versions at
compile time: one for warp=32 and one for warp=64.

**Today, there is no automated LLVM pass that produces warp-size-specialized variants from a
single source function.** The CPU analog — `__attribute__((target_clones(...)))` generating
multiple ISA-specialized function bodies with an ELF `STT_GNU_IFUNC` resolver — has no GPU
counterpart in mainline LLVM or Clang.

---

## Proposal

**Title:** `WarpSizeClonePass` — An LLVM IR Pass That Clones GPU Kernel Functions into
Warp-32 and Warp-64 Variants via Constant Propagation

**One-sentence pitch:** Replace every `llvm.gpu.num.lanes` use with a compile-time constant (32
or 64), run the standard optimization pipeline on each copy, and emit both as separate
`OffloadBinary` entries in the same `.llvm.offloading` fat object — extending CPU function
multi-versioning to the GPU warp-size axis.

### Pass Design

The pass operates at the LLVM IR level on GPU device modules (after MLIR lowering, before
backend code generation). It runs as a module pass with a single configuration option:
`--gpu-warp-size-clones=[both|32-only|64-only]`.

#### Step 1 — Identify warp-size-dependent functions

Walk every `Function` in the module. A function is warp-size-dependent if its IR contains:
- A call to `llvm.gpu.num.lanes` (or, on vendor-specific paths already lowered:
  `llvm.nvvm.read.ptx.sreg.warpsize`, `llvm.amdgcn.wavefront.size`)
- A call to `llvm.gpu.ballot` (return type `i32` vs `i64` is warp-size-determined)
- A call to `llvm.gpu.lane.mask` (same mask-width issue)
- A `getelementptr` or `alloca` whose size expression depends on `llvm.gpu.num.lanes`

This is a simple def-use reachability scan from the `llvm.gpu.num.lanes` call sites. Functions
that do not reach any of these are cloning-exempt; they produce one copy per vendor.

#### Step 2 — Clone and specialize

For each identified function, create two clones:
- `@kernel_w32`: replace all `llvm.gpu.num.lanes` calls with `i32 32`
- `@kernel_w64`: replace all `llvm.gpu.num.lanes` calls with `i32 64`

Also replace `llvm.gpu.ballot` return type:
- In `@kernel_w32`: `ballot` returns `i32` (direct match to NVVM intrinsic semantics)
- In `@kernel_w64`: `ballot` returns `i64` (direct match to AMDGCN wavefront mask)

After substitution, run `InstSimplify` + `InstCombine` + `SROA` on each clone. Because the
warp-size constant has been substituted, `icmp` instructions on loop trip counts simplify,
dead branches fold, and `alloca` sizes constant-fold. Shared memory allocation sizing becomes
a compile-time constant; the backend can emit exact `lds.alloc` or `.shared` directives.

Specifically, the warp-level tree-reduction loop:
```llvm
; Before cloning
%lanes = call i32 @llvm.gpu.num.lanes()
%half  = lshr i32 %lanes, 1        ; 16 or 32 depending on hardware
br label %reduction_header
reduction_header:
  %offset = phi i32 [ %half, %entry ], [ %next, %reduction_body ]
  %cmp = icmp ugt i32 %offset, 0
  br i1 %cmp, label %reduction_body, label %reduction_exit
```

After `InstSimplify` on `@kernel_w32` (where `%lanes = i32 32` → `%half = i32 16`), the loop
bounds are constant and the loop unrolls fully at -O2. The same loop in `@kernel_w64` (where
`%half = i32 32`) unrolls to 6 iterations — the correct depth for wavefront=64.

#### Step 3 — Annotate with `OffloadBinary` properties

Each clone gets an `OffloadBinary` metadata property (extending the `StringMap<StringRef>`
introduced for cost annotations in topic-03):

```
"gpu.warp_size" = "32"   ; for @kernel_w32
"gpu.warp_size" = "64"   ; for @kernel_w64
```

At the backend, each clone is compiled to its target binary:
- `@kernel_w32` → PTX (NVVM backend, natural warp=32) or HSACO with `-mwavefrontsize32` flag
- `@kernel_w64` → HSACO (AMDGCN backend, default wavefront=64) or HSACO with `-mwavefrontsize64`

Both binaries land in the same `gpu.binary` op as separate `#gpu.object` entries, distinguished
by a `warp_size` property in their target attribute. The runtime dispatcher (libkdl, or the
proposed `gpu.select_variant` from topic-01) reads `d->warp_size` from the device query
(`kdl.c:817` for HIP: `hipDeviceGetAttribute(hipDeviceAttributeWarpSize, ...)`;
`kdl.c:615` sets `d->warp_size = 32` for CUDA) and selects the matching variant.

#### Step 4 — Resolver / dispatch integration

Two dispatch paths:

**Path A — libkdl dispatch table.**
The MTB manifest includes `"warp_size": 32` and `"warp_size": 64` in the capability contracts.
`kdl_select()` already reads `d->warp_size` (verified in `kdl.c:817`, `kdl.c:865`) and can
match against the contract field with a one-line addition to the scoring function.

**Path B — `gpu.select_variant` (topic-01).**
The `#gpu.runtime_select` attribute's vendor detection stub is extended to also query warp size
(via `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, ...)` or
`hipDeviceGetAttribute(hipDeviceAttributeWarpSize, ...)`). The dispatch table carries both the
vendor ID and the warp_size field. Selection selects the variant matching both.

**No new runtime infrastructure is required beyond what topic-01 already proposes.**

### Output Format

From one source function `@matmul_reduce` (a kernel using warp-level reduction), the pass
produces two `OffloadBinary` entries embedded in the fat object:

```
.llvm.offloading section:
  [0] triple=nvptx64-nvidia-cuda arch=sm_86  warp_size=32  → matmul_reduce PTX (32-lane)
  [1] triple=amdgcn-amd-amdhsa   arch=gfx942 warp_size=64  → matmul_reduce HSACO (64-lane)
```

From one source function that is warp-size-independent (e.g., a simple element-wise kernel),
the pass emits only one variant per target — no cloning overhead.

---

## Evidence

### On `llvm.gpu.num.lanes` and its limitation

- **PR #131190** (Joseph Huber, AMD/LLNL, March 2025): introduces `__builtin_gpu_num_lanes()` →
  `llvm.gpu.num.lanes`, explicitly described as the abstraction for warp size across NVPTX, AMDGCN,
  SPIRV64. Returns 32 on NVPTX, 64 on AMDGCN.
  Source: wave-08-mlir-async-llvm-gpu.md §2.2, line 161.

- **Explicit limitation acknowledged in PR #131190 Discourse thread:**
  "Warp-size differences (32 vs. 64 lanes): `llvm.gpu.num.lanes` returns the target's value, but
  algorithms written for warp-size-32 will not automatically work correctly on warp-size-64."
  Source: wave-08-mlir-async-llvm-gpu.md §2.5, lines 222–224.

- **PR #174910** (Joseph Huber, January 9, 2026): extends `gpuintrin.h` to SPIR-V via
  `spirvintrin.h`; explicitly marks the SPIR-V implementations as "intentionally inefficient."
  The warp-size abstraction is present but correctness across warp sizes is still out of scope.
  Source: wave-08-mlir-async-llvm-gpu.md §2.3.

### On the concrete correctness failure modes

- **Ballot mask width.** `__ballot_sync` returns `uint32_t` in CUDA; `__ballot` returns
  `uint64_t` on AMD CDNA. This is a type-level incompatibility, not a value-level one.
  Source: literature/hip-rocm.md line 148.

- **Reduction loop bounds.** Warp-level reduction algorithms encode `log2(warp_size)` iterations.
  The 32 vs. 64 difference changes loop depth from 5 to 6, and the full-mask constant from
  `0xffffffff` to `0xffffffffffffffff`.
  Source: literature/hip-rocm.md lines 146–149.

- **Quantified impact.** "The 32 vs 64 thread wavefront difference is a strong argument for
  runtime-specialized dispatch rather than static portability. A warp-level reduction kernel
  optimized for 32-thread warps (NVIDIA) will have different optimal tile sizes, register
  allocation, and loop bounds than one optimized for 64-thread wavefronts (AMD)."
  Source: literature/hip-rocm.md line 421 (explicit statement of the same argument).

- **SPIR-V divergence semantics underspecified.** Even SPIR-V `GroupNonUniform*` instructions
  have implementation-defined behavior under divergent control flow; subgroup shuffle semantics
  depend on wavefront size.
  Source: literature/spirv-analysis.md lines 220–231.

### On the CPU FMV prior art (direct analog)

- **LLVM FMV (Function Multi-Versioning):** `__attribute__((target_clones("avx2", "sse4.2", "default")))`
  generates N function bodies + an ELF `STT_GNU_IFUNC` resolver. The resolver executes once at
  `rtld` relocation time (CPUID), selects the best body, and caches via PLT.
  Source: wave-03-multi-versioned-kernels.md §1 (MaskRay blog), §2 (LLVM 20 release notes).

- **LLVM 20 GlobalOpt static resolver collapse:** If the caller's feature set guarantees one variant
  always wins, the ifunc indirection is folded to a direct call at link time. The GPU analog:
  if the device is statically known (CUDA-only build), the warp-size resolver collapses to a
  direct bind to `@kernel_w32`.
  Source: wave-03-multi-versioned-kernels.md §1 (Euro LLVM 2025 Lamprineas talk).

- **Gap: no GPU FMV in upstream LLVM.** "None of the sources document a
  `__attribute__((target_clones(...)))` analogue for GPU kernels in LLVM/Clang. The CPU FMV
  infrastructure (`STT_GNU_IFUNC`, resolver emission, `fmv-features` metadata) is entirely
  CPU-side."
  Source: wave-03-multi-versioned-kernels.md §Gap section, line 114.

### On IREE hardware-conditional variant selection (closest MLIR-level prior art)

- **IREE issue #3768:** Proposes LLVM FMV for CPU kernel variants inside IREE HAL. Closed in
  November 2023 via `hal.executable.variant` with `condition` region + fallback chain. Each
  variant has a boolean condition evaluated against device properties at initialization.
  Source: wave-03-multi-versioned-kernels.md §7.

- **For warp size:** IREE's pattern could condition a variant on
  `device.warp_size == 32 ? variant_w32 : variant_w64`, but this is not implemented in mainline.
  The proposal here generalizes IREE's `hal.executable.variant` condition to the LLVM IR level.

### On Triton's warp-size handling (production precedent)

- **`GPUTarget("hip", "gfx942", 64)` vs `GPUTarget("cuda", 80, 32)`:** The `num_warps` parameter
  in Triton's `GPUTarget` implicitly encodes warp size. AMD AMD backends compile with wavefront=64;
  NVIDIA backends compile with warp=32. Triton generates separate binaries per target.
  Source: wave-02-triton-multibackend.md §1.

- **`HIPOptions.num_warps`:** AMD's Triton backend captures `num_warps` and `waves_per_eu` in the
  `HIPOptions` dataclass. The entire `make_ttgir()` → `make_llir()` pipeline produces different
  code depending on this configuration — it cannot be abstracted at the IR level.
  Source: wave-02-triton-multibackend.md §3.

### On the prototype (kdl.c warp_size detection)

- **`d->warp_size = 32` for CUDA:** `kdl.c:615` — hard-set during CUDA device enumeration.
- **`d->warp_size = (uint32_t)val` for HIP:** `kdl.c:817` — queried from
  `hipDeviceGetAttribute(hipDeviceAttributeWarpSize, ...)` via the `hipDevPropOld` struct layout
  reverse-engineered at `kdl.c:702–732`.
- **`d->warp_size = 64` for CDNA default:** `kdl.c:819` — fallback when the HIP query fails.
- **Printed in `kdl_device_info`:** `kdl.c:4130` — confirmed as a surfaced device property.
- **`d->warp_size` in dispatch scoring:** Currently not used in `kdl_estimate_cost_weighted()`.
  The `warp_size` field exists in `kdl_device_info` but has no corresponding field in
  `kdl_contract`. This is the exact injection point where the pass's `"gpu.warp_size"` property
  would plug in: add `uint32_t warp_size` to `kdl_contract`, match against `d->warp_size` as a
  capability gate (reject variants whose warp_size != d->warp_size) rather than a scored cost.

---

## Feasibility

**Medium-High. The pass itself is straightforward; the dispatch wiring is already 90% done.**

### The pass (LLVM IR level)

- Walk `llvm.gpu.num.lanes` call sites: standard `Use` iteration, ~50 LOC.
- Clone function (LLVM's `CloneFunctionInto` + `ValueToValueMapTy`): ~30 LOC.
- Substitute constant (`replaceAllUsesWith(ConstantInt::get(...))`): ~10 LOC.
- Run `InstSimplify` + `InstCombine` on clone: invoke pass manager programmatically, ~20 LOC.
- Annotate with OffloadBinary property: patch serializer caller, ~20 LOC.

Total new pass: ~200–300 LOC of LLVM C++. This is smaller than `SelectObjectAttr.cpp` (~200 LOC).

The `WarpSizeClonePass` is a **module pass** — it needs to see all functions to handle
cross-function uses (a warp-size-dependent value passed as an argument). The analysis is local
enough that function-level cloning works for single-kernel functions; for multi-function modules,
callee cloning propagates automatically because `CloneFunctionInto` replaces call targets in the
body.

### The `ballot` return-type problem

`llvm.gpu.ballot` in PR #131190 returns an `i64` uniformly — the pass must additionally:
- In `@kernel_w32` clone: `trunc i64 %ballot_result to i32` inserted after every `ballot` call,
  and all downstream `i64` uses rewritten to `i32`.
- In `@kernel_w64` clone: leave as `i64`.

This is a type mutation, not just a constant substitution. LLVM does not make type changes
trivially, but it is achievable via `IRBuilder` point-wise replacement. Alternatively, the
proposal can restrict scope: initially handle only the `num.lanes`-constant-propagation case
(loop bounds, shared memory sizing, vector widths) and leave ballot-type mutation as future work.
The loop-bound specialization alone provides measurable benefit.

### The dispatch wiring

`d->warp_size` is already populated in `kdl.c` for both CUDA (line 615) and HIP (line 817).
Adding `uint32_t warp_size` to `kdl_contract` and matching it as a hard gate (`warp_size == 0 ||
contract->warp_size == device->warp_size`) in `kdl_select()` is ~10 LOC in `kdl.c`.

### Demo scope for poster

On GTX 1650 (warp=32, CUDA):
1. Write a warp-level reduction kernel using `__builtin_gpu_num_lanes()` in the loop bound.
2. Run `WarpSizeClonePass` to produce `@reduce_w32` and `@reduce_w64`.
3. Compile both to PTX; show `@reduce_w32` has a 5-iteration unrolled loop, `@reduce_w64` has 6.
4. Load via libkdl; verify `kdl_select()` picks `@reduce_w32` on the GTX 1650.
5. Side-by-side PTX diff: one function eliminates the dead 6th iteration and the 64-bit mask.

This demo requires only: the new LLVM pass (~300 LOC), a 30-line test kernel, and a 10-line patch
to `kdl.c`. The AMD warp=64 path can be shown via unit tests (inject a fake `warp_size=64` device
into the dispatcher) without requiring AMD hardware.

**Overall feasibility: Medium-High.** Achievable in 3–5 weeks.

---

## Upstream Path

| Artifact | Location in llvm-project |
|----------|--------------------------|
| `WarpSizeClonePass` | `llvm/lib/Transforms/GPUUtils/WarpSizeClone.cpp` |
| Pass registration | `llvm/lib/Transforms/GPUUtils/CMakeLists.txt` |
| Pass header | `llvm/include/llvm/Transforms/GPUUtils/WarpSizeClone.h` |
| Integration with `gpu-module-to-binary` | `mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp` (invoke before serialization) |
| `OffloadBinary` property constant | `llvm/include/llvm/Object/OffloadBinary.h` (`OFFLOAD_PROPERTY_WARP_SIZE`) |
| Integration test | `llvm/test/Transforms/GPUUtils/warp-size-clone.ll` |
| libkdl contract field | `experiments/prototype/src/kdl.c` (`kdl_contract.warp_size`) |

### Review coordination

- **Primary anchor:** The `llvm.gpu` intrinsics are maintained by Joseph Huber (AMD/LLNL). The
  limitation that "warp-size-32 algorithms don't automatically work on warp-size-64" is
  **documented in his own PR** — making the `WarpSizeClonePass` a natural follow-up he has
  explicitly anticipated.

- **Secondary anchor:** The RFC "SPIR-V IR as a vendor-agnostic GPU representation" (March 2025)
  lists warp-size divergence as a known gap. The pass addresses this gap at the IR level without
  requiring SPIR-V adoption.

- **Tertiary anchor:** The "Cleaning the GPU Dialect" RFC (#88170, Fabian Mora) restructures
  `gpu.binary`. The pass integrates cleanly: it operates on `gpu.func` before binary emission,
  consistent with the RFC's separation of compilation concerns.

- **Implementation risk:** The type mutation for `ballot` (i64 → i32 truncation in w32 clone)
  may surface edge cases in loop analysis passes. The conservative implementation restricts to
  `num.lanes` constant propagation only in v1, deferring ballot mutation to a follow-up.

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **9/10** | No upstream LLVM pass automates warp-size kernel cloning. CPU FMV (`target_clones`) is well-established; its GPU analog doesn't exist. The warp-size gap is explicitly documented in `llvm.gpu` PR #131190 itself — the pass is the natural completion. |
| **Feasibility** | **7/10** | ~300 LOC new LLVM pass + 10 LOC kdl.c patch. Constant-propagation and function cloning are well-supported LLVM APIs. Ballot type mutation is the hard part but can be deferred. Demo on GTX 1650 achievable in 3–5 weeks. |
| **Evidence Strength** | **9/10** | Five concrete correctness failures documented with line-number citations from primary sources (hip-rocm.md, wave-08). Production precedents: Triton per-target binaries (warp-aware by construction), IREE variant conditions, CPU FMV. The gap is acknowledged by the `llvm.gpu` authors themselves. |
| **Impact** | **8/10** | Every portable GPU kernel using warp-level operations (reduction, ballot, shuffle) hits this problem. The pass makes warp-portable kernels a non-issue for downstream users of `llvm.gpu.num.lanes`. Cross-vendor kernel libraries (hipBLAS, rocBLAS, cuBLAS analogs built on MLIR) are direct beneficiaries. |
| **Composite** | **8.25/10** | |

---

## Pitch

**The three-sentence poster pitch:**

`llvm.gpu.num.lanes` makes the warp-size query vendor-neutral, but a warp-level reduction kernel
that loops `log2(warp_size)` times is still silently wrong when compiled for AMD wavefront=64
— the loop runs 5 iterations instead of 6 and the ballot mask overflows a 32-bit register.
`WarpSizeClonePass` is a 300-line LLVM module pass that clones every warp-size-dependent GPU
kernel function, substitutes `llvm.gpu.num.lanes` with the compile-time constant 32 or 64, runs
`InstSimplify` to unfold loop bounds and constant-fold ballot types, and emits both variants as
separate `OffloadBinary` entries in the fat object — the GPU analog of CPU `target_clones`.
The runtime (libkdl or `gpu.select_variant`) picks the correct variant at load time using the
`warp_size` field already present in `kdl_device_info`, closing the warp-correctness gap that
the `llvm.gpu` intrinsics explicitly leave open.

**Poster panel structure:**

1. The problem diagram: same kernel IR → warp=32 and warp=64 backends → correctness failures
   (ballot overflow, wrong loop trip count, wrong shared-memory size)
2. The `llvm.gpu.num.lanes` limitation table (5 concrete failure patterns with code snippets)
3. CPU FMV analogy: `target_clones` → `STT_GNU_IFUNC` resolver → selected function body
4. `WarpSizeClonePass` algorithm: clone → substitute constant → InstSimplify → emit 2 objects
5. PTX diff: `@reduce_w32` (5-iteration loop, 32-bit ballot) vs `@reduce_w64` (6-iteration, 64-bit)
6. Dispatch integration: `kdl_device_info.warp_size` field (kdl.c:817) → capability gate →
   select matching `OffloadBinary` entry
7. Upstream path: `GPUUtils/WarpSizeClone.cpp` + `OffloadBinary` `OFFLOAD_PROPERTY_WARP_SIZE`

---

## Risks

1. **`ballot` return-type mutation is non-trivial.** LLVM disallows in-place type changes.
   Rewriting downstream users of `ballot` from `i64` to `i32` requires an IRBuilder pass over
   the entire function. Conservative scoping (v1 handles only `num.lanes` substitution, not
   ballot mutation) reduces scope but leaves ballot correctness unaddressed.

2. **AMD RDNA3 can run warp=32 (`-mwavefrontsize32`).** The pass currently generates w32 and w64
   variants. On RDNA3 with warp=32 mode enabled, the w32 variant is valid — but kdl.c detects
   warp=32 on RDNA3 in that mode (hipDeviceAttributeWarpSize is runtime-queried), so dispatch
   still works. The risk is that some AMD deployments set warp=32 and others warp=64 on the same
   architecture; the dispatch must check warp_size at device level, not arch level.

3. **Performance regression for warp-size-independent kernels.** The analysis for "is this
   function warp-size-dependent?" must be precise. Over-cloning (cloning functions that don't
   use warp-size primitives) bloats binary size without benefit. The def-use scan from
   `llvm.gpu.num.lanes` call sites is a conservative check but may need transitive reachability
   for indirectly warp-dependent computations (e.g., a function argument derived from `num.lanes`
   passed to a helper that uses it in a loop bound).

4. **Interaction with existing target-specific lowering.** The NVVM lowering target already sets
   `warpsize=32` as a module metadata constant (`!nvvm.annotations`). If `gpu-module-to-binary`
   applies warp specialization before the NVVM lowering sees the module, the two approaches may
   interact unexpectedly. The pass must run on the portable `llvm.gpu.*` IR **before** target-
   specific lowering, not after.

5. **Multi-function kernel modules.** If a kernel calls a device function that uses
   `llvm.gpu.num.lanes`, both the kernel and all callees must be cloned. The callee clone set
   must be computed transitively before any substitution. LLVM's existing `CloneFunctionInto`
   handles call remapping within a clone, but the caller must explicitly manage the clone set
   for the whole call graph reachable from each exported kernel entry point.

6. **Binary size increase.** Each warp-size-dependent kernel contributes two blobs to the fat
   object (w32 + w64). For a kernel library with N such kernels, binary size grows by roughly 2×
   for that portion of the binary. For typical warp-level ML kernels (reductions, attention scores,
   softmax), this affects a small fraction of all kernel functions — the rest are element-wise
   and warp-size-independent.

---

## Cross-References

- `kdl.c:615` — `d->warp_size = 32` for CUDA (dispatch dispatch integration point)
- `kdl.c:817` — `d->warp_size` from `hipDeviceGetAttribute(hipDeviceAttributeWarpSize)`
- `kdl.c:819` — `d->warp_size = 64` for CDNA default (fallback)
- `kdl.c:865` — warp size printed in device info log
- `kdl.c:4130` — warp size surfaced in `kdl_device_info` JSON output
- `wave-08-mlir-async-llvm-gpu.md §2.2–2.5` — full `llvm.gpu.*` intrinsics table + limitations
- `wave-03-multi-versioned-kernels.md §1–4` — CPU FMV mechanics, LLVM 20 GlobalOpt collapse
- `literature/hip-rocm.md §"Warp Size: 32 vs 64"` — ballot mask, loop bounds, concrete bugs
- `literature/hip-rocm.md:421` — explicit statement that warp-32/64 requires separate variants
- `literature/spirv-analysis.md:220–231` — SPIR-V subgroup divergence semantics gap
- `wave-02-triton-multibackend.md §1, §3` — Triton per-target warp-aware binary generation
- `topic-01-gpu-select-variant.md` — the runtime dispatch side; `warp_size` gate extends it
- `topic-03-dispatch-cost-attr.md` — `OffloadBinary` StringMap property extension pattern
