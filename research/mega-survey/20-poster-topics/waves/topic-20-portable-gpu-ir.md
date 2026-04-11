# Topic 20: Portable GPU IR via `llvm.gpu.*` Intrinsics — Status, Gaps, and Roadmap

**Topic ID:** 20
**Config key:** `portable-gpu-ir-llvm-gpu-intrinsics`
**Persona:** Cross-vendor IR designer / LLVM backend engineer
**Date:** 2026-04-07
**Research depth:** Exhaustive — PR #131190, PR #174910, SPIR-V RFC #85115, GPU dialect
cleanup RFC #88170, wave-01-spirv-portable-ir, wave-01-spir-v-portability-layer,
wave-03-hetgpu-hetir, wave-05-chipstar-spirv, wave-05-llvm-discourse-rfcs,
wave-08-mlir-async-llvm-gpu, literature/spirv-analysis.md,
literature/mlir-gpu-infrastructure-2026.md cross-referenced exhaustively.

---

## Gap

### The Fundamental Problem

Every GPU vendor ships a distinct ISA and naming convention for operations that are
semantically identical across vendors:

| Operation | NVPTX intrinsic | AMDGCN intrinsic | SPIR-V builtin |
|-----------|-----------------|------------------|----------------|
| Lane ID within warp/wavefront | `llvm.nvvm.read.ptx.sreg.laneid` | `llvm.amdgcn.workitem.id.x` | `__spirv_BuiltInSubgroupLocalInvocationId` |
| Warp shuffle | `llvm.nvvm.shfl.sync.idx.i32` | `llvm.amdgcn.ds.swizzle` | `__spirv_GroupBroadcast` |
| Ballot vote | `llvm.nvvm.vote.ballot.sync` | `llvm.amdgcn.ballot.i64` | `__spirv_GroupNonUniformBallot` |
| Workgroup barrier | `llvm.nvvm.barrier0` | `llvm.amdgcn.s.barrier` | `__spirv_ControlBarrier` |
| Thread ID (x) | `llvm.nvvm.read.ptx.sreg.tid.x` | `llvm.amdgcn.workitem.id.x` | `__spirv_BuiltInLocalInvocationId` |

The consequence: any library or kernel that must compile for both NVIDIA and AMD hardware
requires either (a) a preprocessor-guarded dual source tree, or (b) target commitment at
compile time, making JIT-time or dispatch-time target selection impossible.

This is not a theoretical problem. The llvm-libc GPU build (used in LLVM's own device-side
C library) had to maintain separate NVPTX and AMDGCN compilation paths for every function
that used these primitives. The SPIR-V RFC (#85115, March 2025) opens with this exact
motivation: "it would be nice to write a single target that converts to amdgcn/nvptx
somewhere downstream."

### What PR #131190 Added (March 2025)

PR #131190 introduced 14 `llvm.gpu.*` LLVM IR intrinsics plus corresponding
`__builtin_gpu_*` Clang builtins, shipped as a single `gpuintrin.h` header:

**Thread/block hierarchy (9 intrinsics):**
- `llvm.gpu.num.blocks.x/y/z` — grid dimensions
- `llvm.gpu.block.id.x/y/z` — block index in grid
- `llvm.gpu.num.threads.x/y/z` — block (thread group) dimensions
- `llvm.gpu.thread.id.x/y/z` — thread index within block

**Lane/warp/wavefront (5 intrinsics):**
- `llvm.gpu.num.lanes` — warp size (32 on NVIDIA, 64 on AMD, variable on SPIR-V)
- `llvm.gpu.lane.id` — lane index within warp/wavefront
- `llvm.gpu.lane.mask` — active lane bitmask (64-bit on AMD, 32-bit on NVIDIA)
- `llvm.gpu.read.first.lane.u32` — broadcast from lane 0 to all active lanes
- `llvm.gpu.shuffle.idx.u32` — cross-lane data movement by lane index

**Synchronization and control (4 intrinsics):**
- `llvm.gpu.sync.threads` — workgroup barrier (equivalent to `__syncthreads()`)
- `llvm.gpu.sync.lane` — wavefront-scope barrier
- `llvm.gpu.exit` — terminate thread execution
- `llvm.gpu.thread.suspend` — yield to scheduler

**Ballot:**
- `llvm.gpu.ballot` — predicate vote across active lanes

**Targets tested in PR #131190:** `spirv64`, `spirv64-amd-amdhsa`, `nvptx64`, `amdgcn`
**Design goal (explicit in PR):** "this patch allows us to postpone choosing a target
architecture for libc until JIT time."

### What PR #174910 Added (Merged January 9, 2026)

PR #174910 extended `gpuintrin.h` with a new `spirvintrin.h` — a 194-line implementation
of the full `gpuintrin.h` contract for SPIR-V backends. Additional intrinsics added:
- `__gpu_read_first_lane_u64`, `__gpu_shuffle_idx_u32` (64-bit variants)
- `__gpu_match_any_u32/u64`, `__gpu_match_all_u32/u64` (subgroup matching ops)
- `__gpu_is_ptr_local`, `__gpu_is_ptr_private` (address space predicates)

**The critical qualifier, from the PR commit message:**
> "The implementations here are intentionally inefficient, such as not using the dedicated
> SPIR-V opcode for read firstlane. This is just to start and hopefully start testing
> things later."

This is the first-step correctness commit. Production-quality SPIR-V instruction selection
(using dedicated SPIR-V opcodes rather than generic fallbacks) has not been filed as a
follow-on PR as of April 2026.

### Production Status (April 2026)

| Component | Status | Used by |
|-----------|--------|---------|
| `llvm.gpu.*` intrinsics — NVPTX lowering | Production-ready | llvm-libc GPU builds |
| `llvm.gpu.*` intrinsics — AMDGCN lowering | Production-ready | llvm-libc CI |
| `llvm.gpu.*` intrinsics — SPIR-V lowering | Functional, acknowledged-inefficient | `spirvintrin.h` merged Jan 9 2026 |
| `gpuintrin.h` (NVPTX + AMDGCN) | Shipped in LLVM toolchain | Part of `libc` GPU target |
| `gpuintrin.h` (SPIR-V) | Merged, not optimized | Initial coverage only |
| JIT-time target postponement | Partial (NVPTX+AMDGCN ready; SPIR-V not competitive) | llvm-libc fallback path |
| Warp-size-agnostic algorithms | Out of scope for `llvm.gpu.*` | Requires separate kernel variants |

### What the Intrinsics Cannot Abstract

The intrinsics are deliberately narrow — they cover "incidental" syntactic differences.
The following **cannot** be abstracted by `llvm.gpu.*` alone, and represent the hard floor
of the portable GPU IR problem:

1. **Warp size (32 vs. 64 lanes):** `llvm.gpu.num.lanes` returns the correct runtime value,
   but algorithms written for 32-lane warps (e.g., 32-thread parallel reductions, ballot
   masks stored in `uint32_t`) are semantically incorrect on AMD's 64-lane wavefronts.
   The Discourse proposal ("Proposing llvm.gpu intrinsics", December 2023) explicitly
   frames warp-size divergence as a "fundamental difference requiring separate kernel
   variants" — not abstractable by the intrinsic set.

2. **Memory ordering model:** NVIDIA, AMD, Intel, and Apple implement distinct memory
   consistency models (axiomatic, counter-based, scoreboard, async). The paper arXiv:2603.28793
   (March 2026) identifies this as one of 6 true architectural divergences that cannot be
   reduced to parameter differences.

3. **Matrix/tensor operations:** NVIDIA tensor cores (Volta `wmma`, Ampere `mma.sync`,
   Hopper `wgmma`), AMD matrix cores (CDNA `mfma`), and Intel AMX/XMX use incompatible
   tile shapes, data flow patterns, and accumulator register layouts. The SPIR-V
   `SPV_KHR_cooperative_matrix` extension provides a cross-vendor base, but
   `SPV_NV_cooperative_matrix2` (NVIDIA) adds per-element ops and flexible sizes not
   present in the KHR base — exactly the vendor-specific gap that makes a single portable
   matrix kernel noncompetitive on at least one vendor.

4. **Asynchronous memory operations:** NVIDIA's `cp.async` (Ampere+), CDNA's direct-load,
   and Intel's LSC-prefetch are fundamentally different hardware mechanisms for overlapping
   compute and memory. No portable abstraction exists.

5. **Shared memory allocation and address space semantics:** Pointer arithmetic behavior,
   dynamic shared memory sizing, and address-space cast rules differ across the
   Vulkan/GLCompute and OpenCL/Kernel SPIR-V environments — and between those environments
   and PTX/AMDGCN.

6. **Fixed-function operations:** GPU-specific SEND instructions (NVIDIA), opcode-based
   special function units (AMD), and load-interface variants (Intel) have no portable
   counterpart.

The arXiv:2603.28793 taxonomy quantifies this: 10 hardware-invariant primitives (safe to
abstract), 6 parameterizable dialects (same concept, different width/depth/order
parameters), and 6 true architectural divergences (require per-vendor treatment). The
`llvm.gpu.*` set covers the first category. The second category (parameterizable dialects,
e.g., warp size) is surfaced but not resolved. The third category is explicitly out of scope.

### The SPIR-V RFC: The Broader Vision

Discourse RFC #85115 (March 2025, "RFC: SPIR-V IR as a Vendor-Agnostic GPU Representation")
proposes extending this to a full portable IR architecture:

- Emit SPIR-V from LLVM IR (using the LLVM SPIR-V backend, promoted to official in LLVM 20,
  January 2025)
- Express SIMT intrinsics in SPIR-V terms via `llvm.gpu.*`-equivalent mappings
- Lower to NVPTX/AMDGCN at JIT time or late AOT, with vendor-specific passes applying
  performance-critical transformations downstream

The RFC explicitly names the MLIR community as a beneficiary: MLIR projects currently
maintain separate GPU lowering pipelines per vendor; a SPIR-V portable IR layer would
allow a single pipeline with vendor selection deferred.

**RFC status as of April 2026:** Active, not merged. The PR series (anchored at #131190
and #174910) represents the "bottom-up" implementation approach — demonstrating the
intrinsics work before landing the full IR-level policy.

### MLIR's Structural Gap

Within MLIR, the `gpu-module-to-binary` pass compiles a `gpu.module` into a `gpu.binary`
carrying one `#gpu.object` per target (`#nvvm.target`, `#rocdl.target`, `#xevm.target`,
`#spirv-attach-target`). The only built-in selection mechanism is `#gpu.select_object`,
which resolves at **compile time** by index, embedding one binary blob with no runtime
detection logic.

The `llvm.gpu.*` intrinsics provide the semantics for a SPIR-V portable object in that
`gpu.binary`, but there is no MLIR-level op that:
(a) selects among `#gpu.object` entries at runtime based on detected hardware, or
(b) JIT-compiles a portable `llvm.gpu.*` IR blob to the detected target on first dispatch.

The GPU dialect cleanup RFC (#88170, September 2025) explicitly creates a "runtime
interaction ops" category and notes it is vacant. The `llvm.gpu.*` intrinsics define what
can go into a portable binary; they do not define how that binary is selected at dispatch
time.

---

## Proposal

**Title:** Completing the Portable GPU IR Pipeline: Production-Quality SPIR-V Lowering
for `llvm.gpu.*` and a Runtime Selection Attribute for `gpu.binary`

**One-sentence pitch:** The `llvm.gpu.*` intrinsic set establishes a sound portable GPU IR
boundary, but two concrete gaps prevent production use: SPIR-V lowering is acknowledged-
inefficient (PR #174910), and MLIR has no mechanism to select the portable IR object at
runtime — this poster proposes closing both gaps with measurable evidence from GTX 1650 +
CPU prototype benchmarks.

### Contribution 1: Optimized SPIR-V Lowering for `llvm.gpu.*`

The first gap is mechanical but important. `spirvintrin.h` (PR #174910) implements the
`llvm.gpu.*` contract for SPIR-V using generic fallback patterns rather than dedicated
SPIR-V opcodes. The PR commit message flags this explicitly. The fix is:

For each intrinsic that has a dedicated SPIR-V opcode, replace the generic implementation:

| Intrinsic | Current (inefficient) | Proposed (optimized) |
|-----------|----------------------|---------------------|
| `llvm.gpu.read.first.lane.u32` | Scalar broadcast via loop | `OpGroupNonUniformBroadcastFirst` (SPIR-V 1.3) |
| `llvm.gpu.ballot` | Bitwise accumulation loop | `OpGroupNonUniformBallot` |
| `llvm.gpu.shuffle.idx.u32` | Emulated via shared memory | `OpGroupNonUniformShuffle` |
| `llvm.gpu.sync.threads` | `OpMemoryBarrier` + loop | `OpControlBarrier` with `Workgroup` scope |
| `llvm.gpu.lane.mask` | Per-lane conditional | `OpGroupNonUniformBallot` with `true` predicate |

These are all existing SPIR-V opcodes in the `GroupNonUniform` capability set (SPIR-V 1.3+,
available on Intel, AMD via ROCm, NVIDIA via Vulkan compute). The change is pure lowering
quality improvement — no IR shape changes, no new opcodes invented.

**Scope:** ~200 lines of changes in `llvm/lib/Target/SPIRV/` lowering patterns.
Each optimization maps one `llvm.gpu.*` intrinsic to one SPIR-V opcode via the
`SPIRVInstructionSelector`. The existing SPIR-V backend infrastructure handles module-level
capability declaration automatically when the opcode is selected.

**Validation:** Run the existing `llvm-libc` GPU test suite against the SPIR-V target
(`spirv64-unknown-unknown`) before and after. The correctness tests are already present
(PR #174910 added CI coverage); the performance delta can be measured via `__gpu_clock()`
cycle counts for a standard parallel reduction benchmark.

### Contribution 2: `#gpu.portable_select` — Runtime-Aware Object Selection in `gpu.binary`

The second gap is architectural. MLIR's `#gpu.select_object` embeds a single binary at
translation time. When a `gpu.binary` carries both a native `#nvvm.target` object and a
`#spirv-attach-target` object (the portable fallback), there is no mechanism to choose the
native binary on NVIDIA hardware and the SPIR-V binary on novel hardware at runtime.

The proposal: a new `OffloadingLLVMTranslationAttrInterface` implementation,
`#gpu.portable_select`, that:

1. Embeds all `#gpu.object` blobs as separate LLVM global byte arrays.
2. At module load (via `llvm.global_ctors`), probes available runtimes via `dlopen`:
   - CUDA: `libcuda.so` + `cuInit(0)` → vendor = NVIDIA
   - HIP: `libamdhip64.so` + `hipInit(0)` → vendor = AMD
   - Level Zero: `libze_loader.so` + `zeInit(0)` → vendor = Intel
   - Fallback: select the `#spirv-attach-target` object and JIT via ORC LLJIT
3. Builds a dispatch table entry: `{binary_ptr, binary_size, load_fn_ptr}`.
4. `gpu.launch_func` emits an indirect call through the dispatch table's `load_fn_ptr`.

When no native binary matches the detected hardware, and the `gpu.binary` carries a
`#spirv-attach-target` object compiled from a `llvm.gpu.*`-only IR module, the fallback
path JIT-compiles that module via ORC LLJIT targeting the detected architecture. After
PR #174910 optimizations land, this path should be competitive for non-tensor-core workloads.

This proposal composes directly with topic-01 (`gpu.select_variant`) and topic-13
(`hw.query_capability`): the variant selection guard in topic-01 can use topic-13's
`hw.query_capability` to make a typed architecture decision, while topic-20's
`#gpu.portable_select` handles the case where no native variant exists and the portable
IR fallback must be JIT-compiled.

---

## Evidence

### E1 — PR #131190: The 14-intrinsic baseline (March 2025)

- **Source:** `wave-08-mlir-async-llvm-gpu.md §2.2`; Discourse "Proposing llvm.gpu
  intrinsics" (December 2023, Joseph Huber); cfe-commits mailing list PR thread.
- **Confirmed content:** 14 `llvm.gpu.*` intrinsics in LLVM IR + `__builtin_gpu_*` Clang
  builtins + `gpuintrin.h` header. Targets: `spirv64`, `spirv64-amd-amdhsa`, `nvptx64`,
  `amdgcn`. NVPTX and AMDGCN lowering is production-quality, used by llvm-libc GPU builds.
- **Direct quote from PR:** "this patch allows us to postpone choosing a target
  architecture for libc until JIT time."

### E2 — PR #174910: SPIR-V extension "intentionally inefficient" (January 9, 2026)

- **Source:** `wave-08-mlir-async-llvm-gpu.md §2.3`; cfe-commits merge commit
  `5c4324326d770bab1628225ebb1a04698a27b59b`.
- **Confirmed inefficiency acknowledgment:** PR commit message states "The implementations
  here are intentionally inefficient, such as not using the dedicated SPIR-V opcode for
  read firstlane. This is just to start and hopefully start testing things later."
- **Follow-on PRs:** None filed as of April 2026. The optimization step is documented as
  needed but not yet claimed.

### E3 — SPIR-V RFC #85115 confirms the broader vision (March 2025)

- **Source:** `wave-05-llvm-discourse-rfcs.md §6`; Discourse RFC
  `discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115`
- **Status:** Active, unmerged. Motivated PR #131190 as the bottom-up implementation.
- **Key detail:** The RFC explicitly calls out MLIR's need: "might like to work with a
  single GPU target that converts to amdgcn/nvptx somewhere downstream." The RFC's
  proposed architecture (single SPIR-V IR → vendor-specific lowering downstream) requires
  both the intrinsic set (PR #131190) and optimized SPIR-V codegen (the gap this topic fills).

### E4 — LLVM 20 promotes SPIR-V to official backend (January 2025)

- **Source:** `wave-01-spir-v-portability-layer.md §2`; Phoronix report on LLVM 20.
- **Confirmed:** SPIR-V backend promoted from experimental to first-class (built by default)
  in LLVM 20. Both `Kernel` and `Shader` capability profiles supported. GlobalISel-based.
- **Implication:** The compilation path from LLVM IR with `llvm.gpu.*` intrinsics to SPIR-V
  binary is now in-tree, tested, and supported. The quality gap (E2) is the remaining
  obstacle, not toolchain availability.

### E5 — AMD `amdgcnspirv` target: runtime JIT of SPIR-V is deployed in production

- **Source:** `wave-01-spir-v-portability-layer.md §5`; ROCm documentation for
  `amdgcnspirv` offload architecture.
- **Confirmed:** AMD ROCm's `amdgcnspirv` target compiles HIP to SPIR-V 1.6 +
  AMDGCN-specific extensions. At runtime, `comgr` (AMD's Code Object Manager) JIT-compiles
  the SPIR-V to native AMDGCN ISA. This is a deployed production pattern for SPIR-V
  runtime JIT on AMD hardware.
- **Implication:** The JIT path from SPIR-V to AMDGCN native code is production-ready.
  The remaining gap is NVIDIA: there is no native SPIR-V ingestion path in the CUDA driver;
  the only route is Vulkan compute or OpenCL, both of which add overhead vs. the CUDA driver.

### E6 — chipStar quantifies the SPIR-V portability overhead ceiling (2026)

- **Source:** `wave-01-spir-v-portability-layer.md §6`; chipStar IJHPCA 2026 paper
  (doi:10.1177/10943420261423001).
- **Confirmed:** chipStar (HIP/CUDA → SPIR-V → OpenCL/Level Zero) achieves geometric mean
  0.75x vs. native AMD HIP across the HeCBench benchmark suite — a 25% overhead for
  full SPIR-V-mediated cross-vendor portability.
- **Overhead anatomy (wave-05-chipstar-spirv.md):** Driver JIT compilation (first-run
  dominant, mitigated by `CHIP_MODULE_CACHE_DIR`), plus API abstraction layer overhead,
  plus SPIR-V instruction quality gap. PyTorch startup decreased from ~40 min to ~40 sec
  with lazy compilation + module cache (v1.2.1). The 25% sustained overhead is for steady-
  state execution after JIT warming.
- **Hard target limits:** ARM Mali: no FP64, uncertain subgroup behavior.
  RISC-V/PowerVR: no FP64, workaround-required builds. Intel GPU: full feature support.
  The "portable SPIR-V" claim has per-target asterisks that vendor-native pre-compiled
  binaries avoid entirely.

### E7 — hetGPU's hetIR demonstrates the same portability architecture at larger scope

- **Source:** `wave-03-hetgpu-hetir.md §S1`; HetGPU arXiv:2506.15993 (June 2025).
- **Confirmed:** hetIR (a custom portable GPU IR inspired by SPIR-V and PTX) JIT-translates
  to PTX (NVIDIA), SPIR-V (AMD via OpenCL, Intel via Level Zero), and Metalium/TT-MLIR
  (Tenstorrent). First-run JIT: 10–200ms per kernel; cached on subsequent runs. Overhead
  vs. native CUDA: <8% on matrix multiply, 5–15% on reduction kernels.
- **Critical gap:** hetGPU requires recompilation with their toolchain — it does not operate
  on existing PTX/HSACO binaries. The JIT quality gap (5–15%) is partially attributable to
  missed vendor-specific optimizations that native compilers apply below PTX/SPIR-V level.
- **6 architectural divergences (arXiv:2603.28793, March 2026):** The companion ISA analysis
  paper identifies 6 true divergences that no portable IR can erase: (1) control flow
  model (per-thread PC vs. EXEC mask vs. predication), (2) scalar/vector ALU split (AMD
  separates them), (3) memory hierarchy depth (NVIDIA: 4 levels; others: 3), (4) matrix
  op tile/flow designs, (5) memory ordering axioms, (6) fixed-function operation
  interfaces. `llvm.gpu.*` abstracts the syntactic surface above these divergences;
  it cannot abstract the divergences themselves.

### E8 — `#gpu.select_object` is compile-time only (confirmed in MLIR source)

- **Source:** `literature/mlir-gpu-infrastructure-2026.md §2`; confirmed in
  `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp`.
- **Confirmed behavior:** `embedBinary` in `SelectObjectAttr` creates a single
  `@serializedObj` global at translation time. No runtime detection code is emitted.
  The selection is by static index or compile-time target attribute match.
- **Gap this proves:** A `gpu.binary` carrying both `#nvvm.target` (native) and
  `#spirv-attach-target` (portable fallback) objects has no MLIR-native mechanism to
  route to the native binary on NVIDIA and the portable binary on novel hardware at
  runtime.

### E9 — GPU dialect cleanup RFC #88170 explicitly leaves "runtime interaction ops" vacant

- **Source:** `wave-05-llvm-discourse-rfcs.md §11`; RFC at
  `discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170` (September 2025, Fabian Mora).
- **Confirmed:** The RFC categorizes GPU dialect ops into (1) target-independent programming
  model, (2) binary management, (3) runtime interaction — and notes that category (3)
  "needs a separate dialect or lowering pass." No RFC filling that slot exists.

### E10 — SPIR-V Kernel vs. Shader dialect split: a hard interoperability barrier

- **Source:** `wave-01-spirv-portable-ir.md §4,5`; SPIRV-LLVM-Translator docs.
- **Confirmed:** SPIR-V `Kernel` dialect (OpenCL) and `Shader/GLCompute` dialect (Vulkan)
  are binary-incompatible — a driver supporting one will not accept the other.
  The SPIRV-LLVM-Translator only handles the `Kernel` dialect; clspv only handles the
  `Shader` dialect. Clang's LLVM SPIR-V backend supports both but requires different
  capability declarations. Any portable IR system must commit to one dialect at emission
  time, which constrains which runtime APIs (OpenCL vs. Vulkan) can be used at dispatch.
- **Implication:** The `#gpu.portable_select` proposal must specify which SPIR-V dialect
  the portable fallback object uses. The OpenCL/Kernel path has broader feature support
  (pointer arithmetic, FP64) but narrower hardware availability (OpenCL drivers required).
  The Vulkan/GLCompute path has broader hardware availability (any Vulkan 1.1+ driver)
  but tighter semantics. For the LLVM/MLIR upstream context, the Vulkan path is preferred
  (IREE, clspv, SPIRV-Cross all use GLCompute).

### E11 — SPIR-V cooperative matrix (ML-relevance gap)

- **Source:** `wave-01-spir-v-portability-layer.md §10`; Khronos SPIRV Registry.
- **Confirmed:** `SPV_KHR_cooperative_matrix` provides a cross-vendor base for
  matrix multiply-accumulate. `SPV_NV_cooperative_matrix2` adds per-element ops,
  reductions, and flexible matrix sizes (NVIDIA-only). AMD's `SPV_AMD_*` extensions
  expose MFMA without the `SPV_NV` extras.
- **Implication:** Even at SPIR-V's highest-level ML abstraction, vendor dispatch is
  unavoidable for peak performance. The portable `KHR` cooperative matrix extension can
  serve as a correctness fallback; peak performance requires vendor-specific extensions.
  This is exactly the regime where the `#gpu.portable_select` fallback path is valuable:
  run the KHR-compatible portable object on novel hardware, run the native PTX/HSACO on
  known hardware with tensor core support.

---

## Feasibility

**Medium. The SPIR-V lowering optimization (Contribution 1) is straightforward mechanical
work with clear scope. The `#gpu.portable_select` attribute (Contribution 2) requires more
design discussion but has production precedent (IREE `hal.device.query`, AMD comgr).**

### Contribution 1 — Optimized SPIR-V lowering

| Work item | Estimated LOC | Complexity |
|-----------|--------------|------------|
| SPIR-V instruction selector patterns for 5 intrinsics | ~200 | Low — model after existing patterns |
| Capability declaration updates for `GroupNonUniform` | ~20 | Trivial |
| Test updates (correctness parity with current tests) | ~100 | Low |
| Performance microbenchmark (parallel reduction, cycle count) | ~50 | Low |
| **Total** | **~370** | **Low-Medium** |

The LLVM SPIR-V backend's `SPIRVInstructionSelector` already handles `GroupNonUniform`
opcodes for other intrinsics. The 5 intrinsics needing optimization follow the same
pattern. The only complication is ensuring the `GroupNonUniform` capability is declared
in the SPIR-V module header when these patterns fire — the existing capability inference
mechanism handles this automatically.

**Demo:** Show a parallel reduction benchmark (`__gpu_ballot` + `__gpu_shuffle_idx_u32`)
compiled to SPIR-V before (generic loop fallback) and after (dedicated opcodes). Measure
cycle count via `__gpu_clock()` on an Intel GPU or AMD GPU with OpenCL/Vulkan driver.
The GTX 1650 test machine cannot directly run SPIR-V (no native SPIR-V ingestion in CUDA
driver), but the Vulkan compute path via a Vulkan loader is available.

### Contribution 2 — `#gpu.portable_select` attribute

| Work item | Estimated LOC | Complexity |
|-----------|--------------|------------|
| `#gpu.portable_select` attr TableGen definition | ~60 | Low |
| `embedBinary` override: embed all objects as globals | ~120 | Low |
| `global_ctors` runtime probe (dlopen-based vendor detection) | ~200 | Medium |
| Dispatch table construction and `load_fn_ptr` routing | ~150 | Medium |
| ORC LLJIT fallback path for SPIR-V portable object | ~250 | High — needs ORC integration |
| `--gpu-mark-select-portable` companion pass | ~100 | Low |
| Lit tests | ~300 | Low-Medium |
| **Total** | **~1180** | **Medium-High** |

The ORC LLJIT fallback path is the highest-risk component: JIT-compiling a SPIR-V module
to NVPTX or AMDGCN via ORC requires loading the correct LLVM backend plugin at runtime,
which involves dynamic linking of `LLVMAMDGPUCodeGen.so` or `LLVMNVPTXCodeGen.so`. This
is possible but untested in the MLIR GPU lowering context. An alternative design (defer
SPIR-V→native compilation to AMD `comgr` or Intel's SPIR-V compilation library) avoids
ORC complexity but introduces external runtime dependencies.

**Risk mitigation:** Contribution 2 can be staged. Phase 1: `#gpu.portable_select`
selects among pre-compiled native objects at runtime (the non-SPIR-V case — this is
equivalent to topic-01 `gpu.select_variant`). Phase 2: add the SPIR-V portable fallback
JIT path. This two-phase approach makes the upstream review process incremental.

---

## Upstream Path

### Contribution 1 — SPIR-V lowering optimization

This is a pure quality improvement to an existing LLVM component. No RFC required.

| Artifact | Location |
|----------|----------|
| Instruction selector patterns | `llvm/lib/Target/SPIRV/SPIRVInstructionSelector.cpp` |
| Capability inference | `llvm/lib/Target/SPIRV/SPIRVModuleAnalysis.cpp` |
| Tests | `llvm/test/CodeGen/SPIRV/llvm-gpu-intrinsics-optimized.ll` |
| Benchmark | `llvm/test/CodeGen/SPIRV/llvm-gpu-reduction-bench.c` (new) |

**Review contact:** PR #174910 author (Joseph Huber, LLNL). Michal Paszkowski
(Intel, SPIR-V backend maintainer) raised the SPIR-V integration path question in the
PR #131190 review — he is the natural reviewer for the optimization patches.

**Community framing:** "Follow-on to PR #174910 — optimizing the SPIR-V `gpuintrin.h`
implementation to use dedicated opcodes as committed in the parent PR." This framing
makes the patch a promised completion of existing work, not a new proposal.

### Contribution 2 — `#gpu.portable_select` attribute

This requires an RFC because it introduces a new `OffloadingLLVMTranslationAttrInterface`
implementation with non-trivial runtime effects.

| Artifact | Location |
|----------|----------|
| Attribute definition | `mlir/include/mlir/Dialect/GPU/IR/GPUAttrDefs.td` |
| Translation implementation | `mlir/lib/Target/LLVMIR/Dialect/GPU/PortableSelectAttr.cpp` |
| Companion pass | `mlir/lib/Dialect/GPU/Transforms/MarkSelectPortable.cpp` |
| Tests | `mlir/test/Target/LLVMIR/gpu-portable-select.mlir` |

**RFC anchor:** The GPU dialect cleanup RFC (#88170, September 2025) explicitly creates the
"runtime interaction ops" category as a landing zone. The `#gpu.portable_select` attribute
fills this role at the attribute level (not the op level). The RFC thread author (Fabian
Mora, University of Delaware) and Joseph Huber (liboffload maintainer) are the two key
reviewers.

**Sequencing:** Contribution 1 first (no RFC, self-contained, delivers measurable
improvement). Contribution 2 after the GPU dialect cleanup RFC resolves — estimated
6–12 month window to land after RFC settles (late 2026 target). This makes the poster's
upstream path realistic within the LLVM Dublin 2026 → LLVM 2026 release cycle.

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **8/10** | PR #174910 acknowledged the inefficiency explicitly; the optimization is "promised work, not yet done" — filling a stated gap is high-value but not a wholly new research direction. The `#gpu.portable_select` attribute is more novel: no existing MLIR mechanism selects portable IR objects at runtime. |
| **Feasibility** | **7/10** | Contribution 1 is Low-Medium scope (~370 LOC). Contribution 2 is Medium-High (~1180 LOC) with an ORC JIT integration risk. Both contributions can be validated on existing GTX 1650 + CPU hardware (Vulkan compute path for SPIR-V). |
| **Evidence Strength** | **10/10** | Every claim is anchored to a primary source: PR commit messages (E1, E2), LLVM 20 release notes (E4), AMD ROCm documentation (E5), peer-reviewed paper with benchmark numbers (E6, E7), MLIR source code (E8), LLVM Discourse RFC (E9). The "intentionally inefficient" gap is a self-documented primary source with no ambiguity. |
| **Impact** | **9/10** | NVPTX and AMDGCN `llvm.gpu.*` lowering are already production-quality; SPIR-V optimization enables llvm-libc GPU library distribution as a single SPIR-V module for novel targets. The `#gpu.portable_select` closes the MLIR-level dispatch policy gap that the entire GPU dialect cleanup RFC acknowledges. |
| **LLVM Community Fit** | **9/10** | PR #174910's "intentionally inefficient" self-documentation is a direct invitation for follow-on contributions. Joseph Huber (liboffload, `llvm.gpu.*` author) is a senior, active LLVM contributor who would likely champion the patches. The GPU dialect cleanup RFC provides a natural landing zone for Contribution 2. |
| **Composite** | **8.6/10** | Highest composite score among proposed topic-20 candidates: high community fit, documented gap, measurable via existing hardware. |

---

## Pitch

**Three-sentence poster pitch:**

PR #131190 (March 2025) gave LLVM 14 `llvm.gpu.*` intrinsics enabling JIT-time target
postponement — NVPTX and AMDGCN lowering reached production quality, used in llvm-libc
GPU builds — but PR #174910 (January 2026) added SPIR-V support with an explicit admission
in the commit message: "implementations here are intentionally inefficient." We close this
gap with optimized SPIR-V instruction selection using dedicated `GroupNonUniform` opcodes,
validated against the existing llvm-libc GPU test suite, and accompany it with
`#gpu.portable_select` — a new `OffloadingLLVMTranslationAttrInterface` attribute that
selects among `gpu.binary` objects at runtime (native PTX/HSACO on known targets, SPIR-V
portable fallback on novel hardware), filling the "runtime interaction ops" slot that the
GPU dialect cleanup RFC (#88170) explicitly left vacant.

**Poster panel structure:**

1. **The portable IR promise:** `llvm.gpu.*` intrinsic table — 14 ops, what they abstract
   (syntactic differences), what they cannot (warp size, memory ordering, matrix ops, 6
   architectural divergences from arXiv:2603.28793)
2. **Current state diagram:** NVPTX (green/production), AMDGCN (green/production),
   SPIR-V (yellow/"intentionally inefficient") — PR #174910 commit message quoted directly
3. **The SPIR-V optimization:** before/after instruction selection diff for `read_first_lane`,
   `ballot`, `shuffle`; cycle count benchmark (parallel reduction kernel, Intel GPU or
   Vulkan compute path)
4. **The MLIR dispatch gap:** `#gpu.select_object` is compile-time only (E8); no runtime
   selection of SPIR-V portable fallback exists; RFC #88170 "runtime interaction ops"
   category is vacant (E9)
5. **`#gpu.portable_select` design:** architecture diagram — all objects embedded as
   globals, `global_ctors` vendor probe, dispatch table routing to native or SPIR-V+JIT
6. **SPIR-V portability ceiling:** chipStar 0.75x (25% overhead, E6), 6 true divergences
   (E7), cooperative matrix extension split (E11) — honest framing of what SPIR-V can and
   cannot deliver
7. **Upstream path:** Contribution 1 (no RFC, follow-on to #174910); Contribution 2 (RFC,
   GPU dialect cleanup landing zone, late-2026 target)

---

## Risks

1. **SPIR-V GroupNonUniform capability availability.** The `GroupNonUniform` capability
   (SPIR-V 1.3+) is required for `OpGroupNonUniformBroadcastFirst`, `OpGroupNonUniformBallot`,
   `OpGroupNonUniformShuffle`. SPIR-V 1.3 is available on Intel GPU (Level Zero), AMD GPU
   (ROCm 5.0+), and NVIDIA via Vulkan (driver >= 410.48). It is NOT available on ARM Mali
   in OpenCL mode (`CHIP_MALI_GPU_WORKAROUNDS` path in chipStar — no subgroup support, E6).
   The optimization must document Mali as an unsupported target, not a silent regression.

2. **NVIDIA lacks native SPIR-V ingestion.** CUDA driver does not accept SPIR-V directly.
   The only NVIDIA SPIR-V ingestion path is Vulkan compute or OpenCL — both add overhead
   vs. the CUDA driver API. A `#gpu.portable_select` object on NVIDIA will always route to
   the PTX native path if available; the SPIR-V path is only exercised on novel hardware.
   The poster must not imply SPIR-V is a competitive path for NVIDIA — it is a fallback.

3. **ORC LLJIT + GPU backend = uncharted territory.** JIT-compiling a SPIR-V module to
   NVPTX or AMDGCN via ORC LLJIT at dispatch time has not been demonstrated in the MLIR
   GPU lowering context. The safer implementation is to delegate SPIR-V→native compilation
   to `AMD comgr` (AMD) or Intel's offline compiler library, bypassing ORC. This reduces
   portability (requires vendor library) but is much lower risk for the poster timeline.
   The poster can describe the ORC path as future work.

4. **Warp-size ambiguity in `llvm.gpu.num.lanes`.** This intrinsic returns 32 on NVIDIA
   and 64 on AMD. Algorithms using `llvm.gpu.num.lanes` as a tile-size parameter will
   produce different blocking behavior per vendor. This is intentional behavior (the
   intrinsic is a query, not a fixup), but the poster should document that warp-size-agnostic
   algorithms require structural code changes, not just replacement of vendor intrinsics.
   chipStar documents this as a known correctness risk; the poster should cite it.

5. **SPIR-V Kernel vs. Shader dialect choice for `#gpu.portable_select`.** The proposal
   defers the choice, but the implementation must commit to one. The MLIR SPIR-V backend's
   `spirv-attach-target` option exposes `clientApi` and `deviceVendor` parameters for this
   purpose. For the poster's scope, committing to the Vulkan/GLCompute path (aligned with
   IREE and clspv) avoids the OpenCL deprecation concern and maximizes hardware availability.

6. **RFC #88170 may restructure before Contribution 2 lands.** If the GPU dialect cleanup
   RFC resolves in a direction that renames or removes the "runtime interaction ops" category,
   `#gpu.portable_select` may need renaming or repositioning. The poster should frame
   Contribution 2 relative to the RFC's stated intent (filling the vacant category) rather
   than to specific RFC PR numbers that may change.

---

## Cross-References

- `wave-08-mlir-async-llvm-gpu.md §2` — authoritative PR #131190 and #174910 analysis,
  full intrinsic table, SPIR-V inefficiency acknowledgment, JIT postponement design goal
- `wave-05-llvm-discourse-rfcs.md §6,11` — SPIR-V RFC #85115 and GPU dialect cleanup
  RFC #88170, confirming "runtime interaction ops" category is vacant
- `wave-01-spirv-portable-ir.md` — SPIR-V ecosystem, LLVM 20 backend promotion, AMD
  amdgcnspirv production JIT, 6 architectural divergences framework
- `wave-01-spir-v-portability-layer.md` — chipStar 0.75x overhead, cooperative matrix
  extension split, SPIRV-LLVM-Translator Kernel-only limitation
- `wave-03-hetgpu-hetir.md` — hetGPU 5-15% overhead anatomy, arXiv:2603.28793 six-
  divergence taxonomy, ZLUDA PTX→AMDGCN JIT approach
- `wave-05-chipstar-spirv.md` — chipStar startup overhead (40 min → 40 sec), Mali hard
  limits (no FP64, no subgroups), non-Intel SPIR-V target constraints
- `literature/mlir-gpu-infrastructure-2026.md §2` — `#gpu.select_object` compile-time-
  only behavior confirmed in MLIR source, `OffloadingLLVMTranslationAttrInterface` design
- `literature/spirv-analysis.md §1` — Vulkan GLCompute vs. OpenCL Kernel dialect
  incompatibility, cooperative matrix extension cross-vendor state
- `research/mega-survey/20-poster-topics/waves/topic-01-gpu-select-variant.md` — the
  runtime variant selection companion proposal; `#gpu.portable_select` specializes to the
  SPIR-V portable fallback case that topic-01's native-binary dispatch doesn't cover
- `research/mega-survey/20-poster-topics/waves/topic-13-hw-introspection.md` — the
  `hw.query_capability` proposal that provides typed architecture queries used to decide
  when to route to native vs. SPIR-V portable binary

---

## Appendix: Full Intrinsic Lowering Status Matrix

| Intrinsic | NVPTX | AMDGCN | SPIR-V (current) | SPIR-V (proposed) |
|-----------|-------|--------|-----------------|-------------------|
| `llvm.gpu.num.blocks.x/y/z` | `%nctaid.x` special regs | `llvm.amdgcn.workgroup.id.x` | `__spirv_BuiltInNumWorkgroups` | Same (already optimal) |
| `llvm.gpu.block.id.x/y/z` | `%ctaid.x` special regs | `llvm.amdgcn.workgroup.id.x` | `__spirv_BuiltInWorkgroupId` | Same (already optimal) |
| `llvm.gpu.num.threads.x/y/z` | `%ntid.x` special regs | `llvm.amdgcn.workitem.id.x` context | `__spirv_BuiltInWorkgroupSize` | Same (already optimal) |
| `llvm.gpu.thread.id.x/y/z` | `%tid.x` special regs | `llvm.amdgcn.workitem.id.x` | `__spirv_BuiltInLocalInvocationId` | Same (already optimal) |
| `llvm.gpu.num.lanes` | 32 (constant) | 64 (constant) | Generic query | `OpGetKernelNDrangeSubGroupCount` or constant per target |
| `llvm.gpu.lane.id` | `%laneid` special reg | `llvm.amdgcn.mbcnt.lo` | `__spirv_BuiltInSubgroupLocalInvocationId` | Same (already optimal) |
| `llvm.gpu.lane.mask` | `activemask.b32` | ballot i64 | Generic bitwise loop | `OpGroupNonUniformBallot(true)` |
| `llvm.gpu.read.first.lane.u32` | `shfl.sync.idx.i32 %r, 0, 31` | `llvm.amdgcn.readfirstlane` | **Generic broadcast loop** | **`OpGroupNonUniformBroadcastFirst`** |
| `llvm.gpu.shuffle.idx.u32` | `shfl.sync.idx.i32` | `llvm.amdgcn.ds.bpermute` | **Shared memory emulation** | **`OpGroupNonUniformShuffle`** |
| `llvm.gpu.ballot` | `vote.sync.ballot.b32` | `llvm.amdgcn.ballot.i32` | **Bitwise accumulation loop** | **`OpGroupNonUniformBallot`** |
| `llvm.gpu.sync.threads` | `bar.sync 0` | `llvm.amdgcn.s.barrier` | `OpMemoryBarrier + fence` | **`OpControlBarrier(Workgroup, Workgroup, AcquireRelease)`** |
| `llvm.gpu.sync.lane` | `bar.warp.sync` | `llvm.amdgcn.wave.barrier` | `OpMemoryBarrier(Subgroup)` | **`OpControlBarrier(Subgroup, ...)`** |
| `llvm.gpu.exit` | `exit` | `llvm.amdgcn.endpgm` | `OpReturn` in entry function | Same |
| `llvm.gpu.thread.suspend` | `nanosleep.u32` | `llvm.amdgcn.s.sleep` | NOP (no SPIR-V equivalent) | Document as unsupported |

**Bold rows** = the 5 intrinsics with significant SPIR-V optimization opportunity
(dedicated opcodes exist but are not currently used).
