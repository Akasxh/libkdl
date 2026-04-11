# Wave 08 — MLIR Async Runtime Dispatch + llvm.gpu Portable Intrinsics

**Angles covered:** mlir-async-runtime-dispatch, llvm-gpu-intrinsics-portable-ir
**Date:** 2026-04-06
**Sources surveyed:** 18 primary (MLIR docs, source files, Discourse threads, mailing list, PR analyses)

---

## Executive Summary

Two complementary LLVM-ecosystem developments are analyzed here. They bear directly on
libkdl's architecture and poster positioning.

**Angle 1 — MLIR async runtime dispatch:** The MLIR `async` dialect and the GPU dialect's
`gpu.async.token` mechanism are two structurally separate async systems. The `async` dialect
lowers to LLVM coroutines and a CPU thread-pool runtime (`mlirAsyncRuntime`). The GPU dialect's
`!gpu.async.token` maps directly to CUDA stream/HIP queue dependencies and lowers to
`@mgpuLaunchKernel` calls — entirely bypassing the CPU async runtime. A bridge pass
(`gpu-async-region`, `AsyncRegionRewriter.cpp`) can promote GPU tokens into `async.execute`
regions for host-side DAG coordination, but the actual GPU kernel dispatch call is never
mediated by the MLIR async runtime's thread pool. This architecture confirms that there is no
MLIR-level runtime dispatch point where a policy layer (like libkdl) can intercept GPU kernel
launch. The dispatch point is always a lowered-to-LLVM runtime call (`@mgpuLaunchKernel`,
`@olEnqueueKernelLaunch`, or the CUDA/HIP driver directly).

**Angle 2 — llvm.gpu portable intrinsics:** PR #131190 (March 2025) introduced `__builtin_gpu_*`
clang builtins and corresponding `llvm.gpu.*` LLVM intrinsics abstracting over NVPTX, AMDGCN,
and SPIRV64. The design goal is explicit: "postpone choosing a target architecture for libc until
JIT time." A subsequent PR #174910 (merged January 9, 2026) extended `gpuintrin.h` with an
initial `spirvintrin.h` for SPIR-V backends, currently marked intentionally inefficient. As of
April 2026: the intrinsic set covers thread/block IDs, lane operations (shuffle, ballot,
read_first_lane), synchronization barriers, and utility ops; NVPTX and AMDGCN lowering is
production-quality; SPIR-V lowering is functional but acknowledged as suboptimal. This is the
IR-level foundation that makes single-binary SPIR-V portable dispatch feasible — but not yet
production-competitive.

For libkdl: the async architecture finding reinforces that libkdl operates correctly at the
right level (below MLIR, above raw driver calls). The `llvm.gpu` intrinsics finding means the
SPIR-V-as-one-binary strategy is maturing but not yet complete — multi-versioned native binaries
remain the pragmatic near-term architecture, validating libkdl's multi-version bundle design.

---

## Angle 1: MLIR Async Runtime and GPU Dispatch

### 1.1 The Two Async Systems in MLIR

MLIR contains two distinct async token/execution systems that are often confused:

**System A — `async` dialect (CPU coroutine system):**
- Ops: `async.execute`, `async.token`, `async.value`, `async.await`, `async.yield`
- Lowering path: `async-to-async-runtime` pass → `async-to-llvm` pass → LLVM coroutines
  (`llvm.coro.id`, `llvm.coro.begin`, `llvm.coro.suspend`, etc.)
- Runtime: `mlirAsyncRuntime` (CPU-only, `lib/ExecutionEngine/AsyncRuntime.cpp`)
- Runtime functions: `mlirAsyncRuntimeExecute`, `mlirAsyncRuntimeAwaitTokenAndExecute`,
  `mlirAsyncRuntimeCreateToken`, `mlirAsyncRuntimeEmplaceToken`, etc.
- Thread pool: `llvm::DefaultThreadPool` — schedules coroutine resumption on CPU threads
- **No GPU content whatsoever.** The runtime contains zero references to CUDA, HIP, or any GPU API.

**System B — `gpu` dialect async tokens (GPU stream system):**
- Type: `!gpu.async.token` — returned by `gpu.launch_func async`, `gpu.alloc async`,
  `gpu.memcpy async`, `gpu.wait async`, etc.
- Semantic: maps directly to a CUDA stream event / HIP queue completion event
- Lowering: `gpu-to-cubin` / `convert-gpu-to-rocdl` + `gpu-to-llvm` conversion passes produce
  calls to:
  - `@mgpuModuleLoad(binary_blob)` — loads a compiled GPU binary
  - `@mgpuModuleGetFunction(module, "kernel_name")` — symbol lookup
  - `@mgpuLaunchKernel(func, gridX/Y/Z, blockX/Y/Z, sharedMem, stream, params)` — actual launch
  - `@mgpuStreamCreate` / `@mgpuStreamSynchronize` / `@mgpuStreamDestroy` — stream lifecycle
- These `@mgpu*` symbols resolve to the MLIR CUDA/HIP runtime wrappers (`mlir_cuda_runtime.so`,
  `mlir_rocm_runtime.so`) or directly to the CUDA driver API in production deployments.

### 1.2 The Bridge: AsyncRegionRewriter

The pass `gpu-async-region` (`mlir/lib/Dialect/GPU/Transforms/AsyncRegionRewriter.cpp`) bridges
the two systems for host-side DAG construction:

- **ThreadTokenCallback**: walks GPU ops in a region, inserts `gpu.wait async` to produce tokens,
  and threads tokens as dependencies between GPU ops
- **DeferWaitCallback**: converts synchronous `gpu.wait` ops inside `async.execute` bodies into
  `async.execute` dependencies — promoting GPU-level synchronization into host-level async DAG edges
- **SingleTokenUseCallback**: ensures each `!gpu.async.token` has exactly one consumer (duplicates
  tokens as needed)

This pass allows GPU pipelines to compose with the CPU async dialect for host-side scheduling
(e.g., launch kernel A, then asynchronously transfer result, then launch kernel B). However:

**Critical finding:** The bridge adds DAG dependency structure at the host MLIR level. It does
NOT move the actual GPU kernel dispatch into the MLIR async runtime. The `@mgpuLaunchKernel` call
is still generated at the bottom of the lowering pipeline and executes on whatever CUDA stream was
threaded through. The MLIR async runtime thread pool only handles the host-side "wait and resume
coroutine" logic — it never touches the GPU.

### 1.3 ExecutionEngine's Role

`mlir::ExecutionEngine` wraps ORC LLJIT (LLVM's `LLJIT` class). It JIT-compiles **host** MLIR
into machine code. For GPU execution, the GPU kernels are compiled to binary blobs
(cubin/hsaco/SPIR-V) during `gpu-module-to-binary` pass, embedded as global string literals
in the host IR, then loaded at runtime by `@mgpuModuleLoad`. The ExecutionEngine JITs the host
wrapper that calls the `@mgpu*` runtime functions — it never JITs device code. Device code is
always pre-compiled.

This is a hard architectural constraint: ExecutionEngine = CPU host JIT only. GPU device code
= AOT blob embedded in the module. There is no device-side JIT path via ExecutionEngine.

### 1.4 The Single Async Dependency Constraint

A community thread (October 2025, LLVM Discourse) noted that `gpu.launch_func` accepts only a
**single** async dependency token, enforced by an explicit check in the implementation. This is a
known structural limitation — if a kernel needs to wait for multiple async operations (e.g., two
concurrent memory copies), the programmer must explicitly serialize them through a `gpu.wait` op
that joins multiple tokens before launch. The thread signals this is being discussed but no RFC
to remove the constraint was filed as of April 2026.

### 1.5 Implications for libkdl

| Finding | Implication |
|---------|-------------|
| MLIR async runtime is CPU-only | libkdl correctly operates below MLIR, at the `@mgpuLaunchKernel`/`olEnqueueKernelLaunch` level |
| `gpu.binary` + `#gpu.select_object` is compile-time only | No MLIR-native runtime selection policy exists — libkdl's gap is confirmed at the MLIR level |
| ExecutionEngine = host JIT only | libkdl's use of liboffload or direct driver API for device dispatch is architecturally correct |
| gpu-async-region bridge exists | libkdl could in future expose a `kdl.async.token` type that composes with MLIR's DAG construction — this is a future research direction |
| Single async dependency limit | libkdl dispatch does not need to solve this — it operates on a per-kernel basis, not in a DAG scheduling role |

---

## Angle 2: llvm.gpu Intrinsics — Portable IR for Cross-Vendor Kernels

### 2.1 Motivation and Context

Source: Discourse thread "Proposing llvm.gpu intrinsics" (December 2023, Joseph Huber et al.),
connected to RFC "SPIR-V IR as a vendor-agnostic GPU representation" (March 2025).

The problem: NVPTX and AMDGCN have functionally identical operations (e.g., thread ID in warp,
warp shuffle, ballot vote) with different intrinsic names:
- NVPTX: `llvm.nvvm.read.ptx.sreg.laneid`, `llvm.nvvm.shfl.sync.idx.i32`
- AMDGCN: `llvm.amdgcn.workitem.id.x`, `llvm.amdgcn.ds.swizzle`
- SPIRV64: OpenCL-style builtin calls via `__spirv_BuiltInSubgroupLocalInvocationId`

The proposed abstraction: a single set of `llvm.gpu.*` LLVM IR intrinsics that lower to the
correct target-specific sequence during code generation. Critically, they enable compiling
LLVM IR to GPU code without committing to a target at IR-generation time — enabling JIT-time
target selection for IR-level portable binaries.

### 2.2 The Intrinsic Set (PR #131190, March 2025)

PR #131190 introduced `__builtin_gpu_*` clang builtins and corresponding `llvm.gpu.*` intrinsics:

**Thread/block hierarchy:**
| Builtin | Intrinsic | Semantics |
|---------|-----------|-----------|
| `__builtin_gpu_num_blocks_x/y/z()` | `llvm.gpu.num.blocks.x/y/z` | Grid dimension |
| `__builtin_gpu_block_id_x/y/z()` | `llvm.gpu.block.id.x/y/z` | Current block index |
| `__builtin_gpu_num_threads_x/y/z()` | `llvm.gpu.num.threads.x/y/z` | Block dimension |
| `__builtin_gpu_thread_id_x/y/z()` | `llvm.gpu.thread.id.x/y/z` | Thread index in block |

**Lane/warp/wavefront:**
| Builtin | Intrinsic | Semantics |
|---------|-----------|-----------|
| `__builtin_gpu_num_lanes()` | `llvm.gpu.num.lanes` | Warpsize (32 on NVIDIA, 64 on AMD) |
| `__builtin_gpu_lane_id()` | `llvm.gpu.lane.id` | Lane index within warp/wavefront |
| `__builtin_gpu_lane_mask()` | `llvm.gpu.lane.mask` | Active lane bitmask |
| `__builtin_gpu_read_first_lane_u32(v)` | `llvm.gpu.read.first.lane.u32` | Broadcast from lane 0 |
| `__builtin_gpu_shuffle_idx_u32(v, i)` | `llvm.gpu.shuffle.idx.u32` | Cross-lane data movement |
| `__builtin_gpu_ballot(pred)` | `llvm.gpu.ballot` | Vote across active lanes |

**Synchronization and control:**
| Builtin | Intrinsic | Semantics |
|---------|-----------|-----------|
| `__builtin_gpu_sync_threads()` | `llvm.gpu.sync.threads` | Workgroup barrier |
| `__builtin_gpu_sync_lane()` | `llvm.gpu.sync.lane` | Wavefront barrier |
| `__builtin_gpu_exit()` | `llvm.gpu.exit` | Terminate thread |
| `__builtin_gpu_thread_suspend()` | `llvm.gpu.thread.suspend` | Scheduler yield |

**Targets tested in PR #131190:** spirv64, spirv64-amd-amdhsa, nvptx64, amdgcn

### 2.3 PR #174910 — SPIR-V gpuintrin.h Extension (Merged January 9, 2026)

Author: Joseph Huber (same as PR #131190 driving author)
Commit: 5c4324326d770bab1628225ebb1a04698a27b59b

What it adds:
- New header `spirvintrin.h` — 194-line implementation of the `gpuintrin.h` contract for SPIR-V
- Extends `gpuintrin.h` to `#include <spirvintrin.h>` when targeting SPIR-V
- Adds additional intrinsics beyond PR #131190's baseline:
  - `__gpu_read_first_lane_u64`, `__gpu_shuffle_idx_u32` (64-bit variants)
  - `__gpu_match_any_u32/u64`, `__gpu_match_all_u32/u64` (subgroup matching)
  - `__gpu_is_ptr_local`, `__gpu_is_ptr_private` (address space predicates)

**Explicitly acknowledged as production-incomplete:**
> "The implementations here are intentionally inefficient, such as not using the dedicated SPIR-V
> opcode for read firstlane. This is just to start and hopefully start testing things later."

This is the first-step commit establishing correctness and CI coverage before optimization.

### 2.4 JIT-Time Target Postponement: The Core Claim

The explicit design goal from PR #131190:
> "this patch allows us to postpone choosing a target architecture for libc until JIT time"

The mechanism: compile libc (or any library) to LLVM IR using only `llvm.gpu.*` intrinsics, with
no target-specific intrinsics present. Store this IR as a "portable GPU IR module." At JIT time,
select NVPTX, AMDGCN, or SPIRV64 as the code generation target and compile the module to the
appropriate native binary. The `llvm.gpu.*` intrinsics lower during this compilation.

This directly maps to an alternative libkdl bundle format:
- **Current libkdl:** bundle contains N pre-compiled native binaries (PTX, HSACO)
- **Future libkdl (JIT path):** bundle also contains one LLVM IR blob with `llvm.gpu.*`; dispatch
  selects either a pre-compiled native binary (performance path) or JIT-compiles the IR to the
  detected target (fallback / novel target path)

### 2.5 Scope and Limitations of llvm.gpu Intrinsics

The intrinsics are intentionally narrow:

**What they cover:** Thread hierarchy queries, lane/warp operations, barriers, ballot, shuffle.
These are the operations that differ only syntactically between NVPTX, AMDGCN, and SPIRV.

**What they do NOT cover:**
- Memory model semantics (volatile, acquire/release — these differ fundamentally between vendors)
- Shared memory allocation and addressing (vendor-specific address space models)
- Warp-size differences (32 vs. 64 lanes): `llvm.gpu.num.lanes` returns the target's value,
  but algorithms written for warp-size-32 will not automatically work correctly on warp-size-64
- Architecture-specific performance characteristics (NVIDIA tensor cores, AMD matrix ops)
- Asynchronous memory operations (cp.async on Ampere, direct load on CDNA)

The Discourse thread (December 2023) explicitly frames these as abstractions over "incidental"
differences, not "fundamental" ones. Warp-size divergence is noted as a fundamental difference
that requires separate kernel variants — not abstractable by `llvm.gpu.*` alone.

### 2.6 Reviewer Concern: SPIR-V Integration Path

One reviewer (Michal Paszkowski) raised a question about how the pass integrating `llvm.gpu`
intrinsics fits into the SPIR-V lowering pipeline, noting that SPIR-V translation typically
occurs via either SPIRV-LLVM-Translator or the in-tree SPIR-V backend. This is an open design
question as of early 2025: should `llvm.gpu.*` → SPIRV instruction mapping happen in the LLVM
SPIR-V backend (preferred for in-tree coherence) or via SPIRV-LLVM-Translator (existing path)?
PR #131190 takes the LLVM backend approach; PR #174910 follows the same path. This design
decision remains important for any consumer (including a libkdl JIT path) building on these
intrinsics.

### 2.7 Current Production Status (April 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| `llvm.gpu.*` intrinsics (NVPTX) | Production-ready | Used by llvm-libc GPU builds |
| `llvm.gpu.*` intrinsics (AMDGCN) | Production-ready | Tested in CI |
| `llvm.gpu.*` intrinsics (SPIRV64, via PR #174910) | Functional, acknowledged-inefficient | Merged Jan 9 2026; marked as initial step |
| `gpuintrin.h` (NVPTX + AMDGCN) | Shipped in LLVM toolchain | Part of llvm-libc GPU support |
| `gpuintrin.h` (SPIRV64) | Merged, not optimized | `spirvintrin.h` added Jan 2026 |
| JIT-time target postponement for libc | Partially implemented | NVPTX+AMDGCN ready; SPIRV incomplete |
| Warp-size-agnostic algorithms | Out of scope | Requires separate kernel variants |

---

## Cross-Angle Synthesis: Implications for libkdl

### The Dispatch Architecture Diagram

```
[MLIR IR with gpu.module]
         |
  gpu-module-to-binary pass
         |
  [gpu.binary: cubin blob + hsaco blob]   <-- AoT, N binaries per gpu.module
         |
  #gpu.select_object (COMPILE-TIME)       <-- picks one binary at MLIR lowering time
         |
  [LLVM IR: global byte array + @mgpuModuleLoad() call]
         |
  ExecutionEngine (host JIT, ORC LLJIT)
         |
  Host machine code with calls to:
  @mgpuModuleLoad -> loads selected binary
  @mgpuModuleGetFunction -> symbol lookup
  @mgpuLaunchKernel -> GPU kernel dispatch   <-- ACTUAL DISPATCH POINT
         |
  [GPU stream / CUDA stream / HIP queue]
```

libkdl operates at the `@mgpuLaunchKernel` (or `olEnqueueKernelLaunch`) level — below MLIR,
after all MLIR passes have run. This is correct: no MLIR pass performs runtime selection;
`#gpu.select_object` is compile-time. The async runtime is CPU-only and does not mediate GPU
dispatch. There is no MLIR-level hook that would allow runtime kernel selection — it must be
done at the runtime library level, which is exactly where libkdl sits.

### The llvm.gpu Intrinsics as libkdl Bundle Format Extension

The `llvm.gpu.*` intrinsic set defines the boundary of what is portably expressible in a
single LLVM IR module across NVPTX, AMDGCN, and SPIRV64. A libkdl bundle entry of type
`BINARY_LLVM_IR_GPU` (future extension) could store this portable IR and trigger JIT compilation
at first dispatch on a novel target. The production gaps (SPIR-V inefficiency, warp-size
abstraction missing) mean this path is a complement to, not replacement of, pre-compiled native
binaries in the bundle. A reasonable libkdl bundle structure for a single kernel:

```
kdl_bundle_t {
  entry[0]: type=BINARY_PTX,    target="sm_75",  binary=<PTX blob>        // NVIDIA, fast path
  entry[1]: type=BINARY_HSACO,  target="gfx1030", binary=<HSACO blob>     // AMD, fast path
  entry[2]: type=BINARY_LLVM_GPU_IR, target="*", binary=<llvm.gpu IR blob> // fallback JIT
}
```

At dispatch: check if native binary exists for detected target. If yes, use it. If no, JIT-compile
the `llvm.gpu` IR blob for the target using ORC JIT + the target's code generator.

---

## Gaps and Follow-On Angles

1. **`gpu.launch_func` single async dependency constraint** — the October 2025 thread is
   unresolved. If lifted, it would enable better MLIR-level pipeline parallelism for GPU kernels.
   Worth monitoring for libkdl's future integration story.

2. **SPIR-V `llvm.gpu` optimization pass** — the spirvintrin.h initial commit (Jan 2026) is
   acknowledged as inefficient. The follow-on PRs that optimize SPIR-V instruction selection
   (using dedicated SPIR-V opcodes for read_first_lane, etc.) have not yet been filed as of
   April 2026. This is the next critical milestone for the portable-IR-JIT path.

3. **mlir-air async concurrency model** — MLIR-AIR (October 2025, emerging from AMD/Xilinx work)
   uses async execution engines as first-class MLIR constructs targeting spatial AI hierarchies.
   It is a potential alternative coordination layer for multi-accelerator dispatch that would
   subsume some of libkdl's orchestration role in spatial compute contexts.

4. **GPU dialect cleanup RFC (#88170, September 2025)** — discussed in wave-05-llvm-discourse-rfcs.
   The resolution of what `gpu.binary` should/should not do will determine whether a
   `gpu.runtime_select` op is ever proposed in MLIR, which would directly compete with or
   complement libkdl at the MLIR level.

---

## Source Registry

| # | Source | Type | URL | Relevance |
|---|--------|------|-----|-----------|
| 1 | MLIR Async Dialect docs | Official docs | https://mlir.llvm.org/docs/Dialects/AsyncDialect/ | 7/10 |
| 2 | MLIR GPU Dialect docs | Official docs | https://mlir.llvm.org/docs/Dialects/GPU/ | 9/10 |
| 3 | AsyncRuntime.cpp source | Source | https://mlir.llvm.org/doxygen/AsyncRuntime_8cpp_source.html | 8/10 |
| 4 | AsyncRegionRewriter.cpp | Source | https://mlir.llvm.org/doxygen/AsyncRegionRewriter_8cpp_source.html | 8/10 |
| 5 | Proposing llvm.gpu intrinsics thread | Discourse | https://discourse.llvm.org/t/proposing-llvm-gpu-intrinsics/75374 | 9/10 |
| 6 | PR #131190 cfe-commits thread | Mailing list | https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg538873.html | 9/10 |
| 7 | PR #174910 merge commit | Mailing list | http://www.mail-archive.com/cfe-commits@lists.llvm.org/msg645088.html | 9/10 |
| 8 | RFC SPIRV IR vendor-agnostic | Discourse | https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115 | 9/10 |
| 9 | gpu.launch_func single async dep | Discourse | https://discourse.llvm.org/t/gpu-why-does-gpu-launch-func-only-accept-a-single-async-dependency/88510 | 7/10 |
| 10 | Where do gpu async tokens come from | Discourse | https://discourse.llvm.org/t/where-do-gpu-async-tokens-come-from/71426 | 7/10 |
| 11 | How to lower async gpu ops | Discourse | https://discourse.llvm.org/t/how-to-lower-the-combination-of-async-gpu-ops-in-gpu-dialect/72796 | 7/10 |
| 12 | MLIR async to LLVM coroutines | Mailing list | https://lists.llvm.org/pipermail/mlir-commits/2020-October/002112.html | 6/10 |
| 13 | libc GPU building docs | Official docs | https://libc.llvm.org/gpu/building.html | 7/10 |
| 14 | libc GPU using docs | Official docs | https://libc.llvm.org/gpu/using.html | 7/10 |
| 15 | MLIR passes list | Official docs | https://mlir.llvm.org/docs/Passes/ | 6/10 |
| 16 | AsyncToAsyncRuntime.cpp | Source | https://mlir.llvm.org/doxygen/AsyncToAsyncRuntime_8cpp_source.html | 7/10 |
| 17 | AsyncToLLVM.cpp | Source | https://mlir.llvm.org/doxygen/AsyncToLLVM_8cpp_source.html | 7/10 |
| 18 | This year in LLVM 2025 (nikic) | Blog | https://www.npopov.com/2026/01/31/This-year-in-LLVM-2025.html | 3/10 (no GPU content) |

---

## Cross-References to Prior Waves

- `gpu.binary` + `#gpu.select_object` compile-time selection: wave-01-mlir-gpu-dialect-dispatch, literature/mlir-gpu-infrastructure-2026.md
- SPIR-V portable IR RFC (#85115): wave-01-spirv-portable-ir, wave-05-llvm-discourse-rfcs (#6)
- liboffload `olEnqueueKernelLaunch` dispatch point: wave-02-llvm-offload-runtime, wave-04-liboffload-multiversion
- GPU dialect cleanup RFC (#88170): wave-05-llvm-discourse-rfcs (#11)
- ExecutionEngine as host-JIT-only: literature/mlir-jit-analysis.md §1.3
