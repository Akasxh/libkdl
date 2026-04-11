# Wave 04 — Level Zero oneAPI Backend Dispatch

**Angle:** Level Zero oneAPI Backend Dispatch
**Query:** Level Zero oneAPI unified runtime dispatch Intel GPU SPIR-V kernel submission
**Date:** 2026-04-06

---

## Source Index

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | Level Zero Core Programming Guide (spec v1.11) | https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/PROG.html | Current (spec v1.15.31) | Spec/Docs | 10/10 |
| S2 | Introduction to Level Zero API for Heterogeneous Programming — Juan Fumero | https://jjfumero.github.io/posts/2021/09/introduction-to-level-zero/ | Sep 2021 | Blog/Tutorial | 9/10 |
| S3 | Level Zero Immediate Command Lists — Intel Developer Guide | https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html | 2023-2025 | Docs | 9/10 |
| S4 | Level Zero UR Reference Document — Unified Runtime Spec | https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html | Current | Spec/Docs | 9/10 |
| S5 | oneAPI Unified Runtime: Introduction and Adapter Model | https://oneapi-src.github.io/unified-runtime/core/INTRO.html | Current | Spec/Docs | 9/10 |
| S6 | Intel oneAPI DPC++/C++ Compiler 2025 Release Notes | https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html | 2025 | Release Notes | 8/10 |
| S7 | PoCL Level Zero Driver — Portable Computing Language 7.1 docs | https://portablecl.org/docs/html/level0.html | Mar 2025 | Docs | 8/10 |
| S8 | chipStar: Making HIP/CUDA applications cross-vendor portable — IJPP 2026 | https://journals.sagepub.com/doi/10.1177/10943420261423001 | Feb 2026 | Paper | 8/10 |
| S9 | Zero in on Level Zero: oneAPI Open Backend Approach — Intel | https://www.intel.com/content/www/us/en/developer/articles/technical/zero-in-on-level-zero-oneapi-open-backend-approach.html | 2021 | Blog/Docs | 8/10 |
| S10 | Unified Runtime Design — intel.github.io/llvm | https://intel.github.io/llvm/design/UnifiedRuntime.html | Current | Docs | 8/10 |

---

## Source Summaries

### S1 — Level Zero Core Programming Guide (Spec v1.11/1.15) [10/10]

**URL:** https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/PROG.html
**Type:** Official specification documentation
**Date:** Current (latest release: v1.15.31)

The authoritative reference for the Level Zero API's kernel dispatch lifecycle. Covers the complete pipeline from module loading through kernel submission.

**Key details:**

- `zeModuleCreate(context, device, &desc, &module, &buildLog)` accepts `ZE_MODULE_FORMAT_IL_SPIRV` or `ZE_MODULE_FORMAT_NATIVE`. When SPIR-V is provided, the driver performs JIT compilation to device-native ISA at module creation time (not lazily).
- Build options string (e.g., `-ze-opt-level=2`) can be passed to influence the device compiler.
- Specialization constants can override SPIR-V constant values at module creation time — analogous to CUDA's template instantiation but at runtime.
- `zeKernelCreate(module, &desc, &kernel)` creates a kernel object by name lookup within the module. Returns `ZE_RESULT_ERROR_INVALID_KERNEL_NAME` if not found, `ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED` for unresolved imports.
- `zeKernelSetGroupSize(kernel, x, y, z)` or `zeKernelSuggestGroupSize(kernel, globalX, globalY, globalZ, &groupX, &groupY, &groupZ)` — driver can override with hardware-optimal tile dimensions.
- `zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch, signal, numWait, waitList)` copies all argument state at append time, enabling safe reuse with new arguments.
- `zeCommandListAppendLaunchKernel` with indirect launch variant reads group counts from a device buffer — critical for GPU-driven dispatch (no CPU roundtrip).
- `zeModuleDynamicLink` resolves import symbols across separately compiled modules — enables dynamic linking on GPU analogous to `dlopen`.
- `zeModuleGetNativeBinary` extracts the device-compiled binary for application-level disk caching (AOT caching pattern).
- `zeCommandListCreateImmediate` creates a combined command-list/queue object where each append immediately submits to device — the low-latency path.
- Cooperative kernels via `zeCommandListAppendLaunchCooperativeKernel` allow cross-workgroup barrier synchronization on supported hardware.

**Relevance to libkdl:** The `zeModuleCreate`/`zeKernelCreate`/`zeCommandListAppendLaunchKernel` triple is Level Zero's exact equivalent of CUDA's `cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`. The `zeModuleGetNativeBinary` + `ZE_MODULE_FORMAT_NATIVE` round-trip is precisely the AOT caching mechanism libkdl needs for Intel GPUs.

---

### S2 — Introduction to Level Zero API — Juan Fumero [9/10]

**URL:** https://jjfumero.github.io/posts/2021/09/introduction-to-level-zero/
**Type:** Tutorial blog post (GPU systems researcher, University of Edinburgh)
**Date:** Sep 2021

End-to-end worked example of SPIR-V kernel dispatch via Level Zero on Intel HD Graphics 630.

**Key details:**

- Full compile chain: `clang -cc1 -triple spir kernel.cl -emit-llvm-bc -o kernel.bc` then `llvm-spirv kernel.bc -o kernel.spv`, producing a SPIR-V binary consumed directly by `zeModuleCreate`.
- Demonstrated **14x speedup** on 1024x1024 matrix multiply vs sequential CPU, on Gen9 integrated graphics.
- Driver behavior on group size: suggested 32x32 was auto-overridden to 256x1 by the driver — shows the driver has meaningful tile selection logic that differs from CUDA's purely programmer-driven model.
- Identified gaps at the time: no documented examples for events/barriers/timers, no CUDA/OpenCL migration guides, silent failure on closed command list reuse.
- SPIR-V loading via `moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV` — driver compiles at `zeModuleCreate` call.

**Relevance to libkdl:** Demonstrates that offline-compiled SPIR-V (via standard Clang/llvm-spirv) works directly with Level Zero, validating a SPIR-V as a portable kernel IR that libkdl could target. The auto-optimization of group sizes is a dispatch consideration — libkdl should use `zeKernelSuggestGroupSize` rather than hardcoding.

---

### S3 — Level Zero Immediate Command Lists — Intel Developer Guide [9/10]

**URL:** https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html
**Type:** Intel developer documentation
**Date:** 2023 (updated through 2025.3)

Defines the two submission modes and their performance tradeoffs.

**Key details:**

- **Regular command lists:** `zeCommandListAppendLaunchKernel` records into a list, `zeCommandQueueExecuteCommandList` submits the batch. Enables deferred work accumulation and batched GPU submission — better for throughput.
- **Immediate command lists (`zeCommandListCreateImmediate`):** Programming and submission occur together per-operation. Multiple immediate command lists can run concurrently on a single hardware queue, enabling GPU-side kernel concurrency.
- Performance tradeoff: immediate lists have higher per-operation host submission overhead, problematic for kernels under ~10 microseconds. Regular lists amortize submission cost across a batch.
- Default mode history: immediate command lists became default for Intel Data Center GPU Max Series (PVC) on Linux starting with oneAPI 2023.2. L0 v2 adapter (2025.3) supports **only** immediate in-order mode.
- Runtime control: `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1/0` (SYCL path) and `LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=all/0` (OpenMP path).

**Relevance to libkdl:** libkdl should select submission mode based on kernel duration characteristics. For ML inference kernels (typically milliseconds), immediate mode is appropriate and reduces dispatch latency. The 10-microsecond boundary is a useful heuristic for the dispatch layer's mode selection.

---

### S4 — Level Zero UR Reference Document — Unified Runtime Spec [9/10]

**URL:** https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html
**Type:** Official UR specification reference
**Date:** Current (UR 0.12)

Configuration and behavioral reference for the Level Zero adapter within the oneAPI Unified Runtime.

**Key details:**

- `UR_L0_USE_IMMEDIATE_COMMANDLISTS` env var controls command list mode: disabled, per-queue, or per-thread variants — fine-grained control allowing different submission strategies per use case.
- 40+ environment variables governing copy engine selection, memory allocation strategies, queue synchronization behavior, USM configuration, debug/trace modes.
- The adapter is a thin translation layer mapping UR API calls (`urKernelCreate`, `urEnqueueKernelLaunch`, etc.) to Level Zero calls (`zeKernelCreate`, `zeCommandListAppendLaunchKernel`).
- The L0 v2 adapter (2025.3) is a redesigned version focusing on maximizing individual queue mode performance; supports immediate in-order mode only; activated via `SYCL_UR_USE_LEVEL_ZERO_V2=1`.

**Relevance to libkdl:** The UR adapter's env-var surface demonstrates what operational knobs exist. A libkdl Level Zero backend could expose similar knobs. The v2 adapter's design (single-mode, maximum per-mode performance) is a model for libkdl's Intel backend architecture.

---

### S5 — oneAPI Unified Runtime: Introduction and Adapter Model [9/10]

**URL:** https://oneapi-src.github.io/unified-runtime/core/INTRO.html
**Type:** Official UR specification documentation
**Date:** Current (UR 0.12)

Defines the UR abstraction layer architecture that sits between DPC++ runtime and device backends.

**Key details:**

- UR provides a unified C API that maps to backend adapters: `ur_adapter_level_zero`, `ur_adapter_cuda`, `ur_adapter_hip`, `ur_adapter_opencl`, `ur_adapter_native_cpu`.
- Adapter discovery: configurable search paths via `UR_ADAPTERS_FORCE_LOAD` and `UR_ADAPTERS_SEARCH_PATH` — dynamic loader pattern directly analogous to libkdl's plugin model.
- Free-threaded design: APIs are thread-safe when different object handles are used — enables parallel kernel submissions from multiple threads.
- Minimal validation: adapters skip parameter validation by default; optional validation layers handle this — keeps hot path overhead minimal.
- Object hierarchy: platforms > devices > contexts > queues > programs > kernels — standard heterogeneous compute model matching CUDA/HIP.
- New in 2025.3: `urEnqueueKernelLaunchWithArgsExp` combines argument setting and kernel launch into a single call — reduces per-kernel overhead on hot dispatch paths.

**Relevance to libkdl:** UR's adapter loader pattern (`UR_ADAPTERS_FORCE_LOAD`) is essentially what libkdl implements: a runtime-loaded backend abstraction. The `urEnqueueKernelLaunchWithArgsExp` single-call dispatch API is worth adopting in libkdl's Intel backend interface for reduced overhead.

---

### S6 — Intel oneAPI DPC++/C++ Compiler 2025 Release Notes [8/10]

**URL:** https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html
**Type:** Official Intel release notes
**Date:** 2025 (versions 2025.0 through 2025.3)

Documents the evolution of Level Zero v2 adapter through the 2025 release cycle.

**Key details:**

- **2025.2:** L0 v2 adapter introduced as experimental feature behind `SYCL_UR_USE_LEVEL_ZERO_V2=1`.
- **2025.3:** L0 v2 adapter promoted to default for Intel Arc B-Series Graphics (Xe2 / Battlemage) and Intel Core Ultra 200V Series (Xe2-LP) on both Linux and Windows.
- Performance claim: "significantly reduces host runtime overhead and improves latency of kernel submissions" — specific numbers not published, but confirmed improvement in submission latency direction.
- L0 v2 design: immediate in-order mode only, maximizing per-queue performance rather than supporting multiple submission modes.
- Also in 2025.3: faster kernel compilation and Graph USM allocation support in SYCL Graphs.
- `urEnqueueKernelLaunchWithArgsExp` extension unifies argument binding and launch into one UR call.

**Relevance to libkdl:** The v2 adapter trajectory (2025.2 experimental → 2025.3 default) shows Intel actively optimizing the submission path. libkdl targeting Intel GPUs should track the v2 adapter path and use immediate in-order mode for Xe2+ targets.

---

### S7 — PoCL Level Zero Driver — Portable Computing Language 7.1 [8/10]

**URL:** https://portablecl.org/docs/html/level0.html
**Type:** Open-source project documentation
**Date:** Mar 2025 (PoCL 7.x)

PoCL implements an OpenCL-over-Level-Zero driver using LLVM/Clang and SPIRV-LLVM-Translator. Provides a secondary data point on Level Zero dispatch from an independent runtime.

**Key details:**

- Compilation pipeline: Clang → LLVM IR → SPIRV-LLVM-Translator → SPIR-V → `zeModuleCreate` (driver compiles to GEN ISA).
- Two compilation modes:
  - Standard (AOT): compile at `clBuildProgram` time (equivalent to `zeModuleCreate`).
  - JIT (`POCL_LEVEL0_JIT=1`): lazy compilation at first `clEnqueueNDRangeKernel` — beneficial for programs with thousands of kernels (heavily-templated HPC codes).
- Achieved OpenCL CTS conformance with both OpenCL C and SPIR-V compilation modes (conformance stamp: January 2025).
- Known limitations: FP64/FP16 math test failures, 64-bit atomics risk GPU hang, program-scope variables not supported, subgroup queries missing.
- Uses `SPIRV-LLVM-Translator` with fallback to `llvm-spirv` binary when lib not available.

**Relevance to libkdl:** PoCL's Level Zero driver is the closest open-source analog to what a libkdl Intel backend would need to do: load SPIR-V via Level Zero, manage module lifecycle, dispatch kernels. Its JIT/AOT mode selection logic maps directly to libkdl's caching strategy. The conformance test results validate the viability of the approach.

---

### S8 — chipStar: Making HIP/CUDA Cross-Vendor Portable — IJPP 2026 [8/10]

**URL:** https://journals.sagepub.com/doi/10.1177/10943420261423001
**Type:** Peer-reviewed paper
**Date:** February 2026 — International Journal of High Performance Computing Applications

Peer-reviewed validation of HIP/CUDA portability to Level Zero (and OpenCL) via SPIR-V, with performance data.

**Key details:**

- chipStar compiles unmodified CUDA/HIP source to SPIR-V binary, then dispatches via either Level Zero or OpenCL runtime.
- Two backends: OpenCL backend (older, more mature) and Level Zero backend (newer, lower latency path).
- HIPCC compilation pipeline: `hipcc --targets=spirv64` → SPIR-V → runtime dispatch via chosen backend.
- Performance: geometric mean of **0.75x vs native AMD HIP** across benchmarks — 25% overhead for full cross-vendor portability through an open-standard chain.
- Demonstrated production viability: GAMESS-GPU-HF (quantum chemistry) ported successfully with competitive performance.
- Evaluations on diverse platforms including Intel GPU, ARM Mali, PowerVR, RISC-V — Level Zero path covers Intel discrete and integrated.
- Paper notes HIP/CUDA feature mismatches with OpenCL that required extensions; Level Zero backend has fewer such gaps due to closer semantic alignment.
- Contributors include Intel, AMD, Argonne National Laboratory — broad institutional backing.

**Relevance to libkdl:** chipStar's architecture is the production-grade existence proof for the libkdl concept: a single SPIR-V binary, dispatched via a pluggable backend (Level Zero for Intel, OpenCL for others). The 0.75x geometric mean overhead establishes a baseline for cross-vendor portability cost through the Level Zero path.

---

### S9 — Zero in on Level Zero: oneAPI Open Backend Approach — Intel [8/10]

**URL:** https://www.intel.com/content/www/us/en/developer/articles/technical/zero-in-on-level-zero-oneapi-open-backend-approach.html
**Type:** Intel developer article
**Date:** 2021 (original; concepts remain current)

Intel's overview of Level Zero's design philosophy and how it fits within the oneAPI stack.

**Key details:**

- Level Zero is positioned as the "direct-to-metal" layer — analogous to Vulkan for compute, below OpenCL. Sits beneath SYCL/DPC++, OpenCL, and OpenMP in the software stack.
- Key design intent: expose hardware capabilities that higher-level APIs abstract away: explicit memory management, fine-grain synchronization, command queuing controls, multi-device topology.
- Supports: function pointers on GPU, virtual functions, unified memory (USM), I/O capabilities, fine-grain explicit controls for HPC.
- SPIR-V is the primary IR input (`ZE_MODULE_FORMAT_IL_SPIRV`) — Level Zero is explicitly designed as the SPIR-V runtime for Intel GPUs.
- The `zeModuleCreate` + `zeKernelCreate` + `zeCommandListAppendLaunchKernel` sequence parallels CUDA's `cuModuleLoad` + `cuModuleGetFunction` + `cuLaunchKernel` exactly.
- Context creation is explicit; multiple contexts per device are supported — important for isolation between concurrent dispatch agents.

**Relevance to libkdl:** Confirms Level Zero is the intended interface for libkdl's Intel GPU backend. The explicit parallel to CUDA's module/function/launch API means libkdl's abstraction layer over CUDA can be mirrored almost 1:1 for Level Zero, minimizing backend-specific code.

---

### S10 — Unified Runtime Design — intel.github.io/llvm [8/10]

**URL:** https://intel.github.io/llvm/design/UnifiedRuntime.html
**Type:** DPC++ compiler design documentation
**Date:** Current (intel/llvm sycl branch)

Describes how DPC++ uses UR as the interface to backend runtimes.

**Key details:**

- UR is the interface layer between the DPC++ (SYCL) runtime and device-specific backend runtimes.
- Each Plugin object owns a `ur_adapter_handle_t` representing one backend (Level Zero, OpenCL, CUDA, HIP).
- DPC++ ESIMD kernels are automatically recognized and dispatched without user setup — the runtime queries kernel metadata.
- Adapter libraries are shared objects loaded at runtime via the UR loader — confirmed dynamic plugin pattern.
- Both Linux and Windows supported across all adapters including Level Zero.
- Issue noted: Level Zero leak checker doesn't work on Windows due to race between plugin DLL unload and static variable destruction — a known platform-specific hazard.

**Relevance to libkdl:** UR's Plugin/adapter model is the direct architectural precedent for libkdl's backend plugin system. The per-adapter shared object loading validates that libkdl's `dlopen`-based backend discovery approach is the industry standard pattern for this class of runtime.

---

## Synthesis

### Level Zero Dispatch Pipeline (Complete Flow)

```
[Offline compile]
kernel.cl → clang (SPIR target) → kernel.bc → llvm-spirv → kernel.spv

[Runtime: libkdl Intel backend]
zeInit()
zeDriverGet() → zeDeviceGet() → zeContextCreate()
zeCommandListCreate() or zeCommandListCreateImmediate()
zeCommandQueueCreate()

zeModuleCreate(ctx, dev, {ZE_MODULE_FORMAT_IL_SPIRV, spv_data}, &module, &log)
  // driver JIT-compiles SPIR-V → GEN ISA here

zeKernelCreate(module, {kernel_name}, &kernel)
zeKernelSuggestGroupSize(kernel, gx, gy, gz, &sgx, &sgy, &sgz)
zeKernelSetGroupSize(kernel, sgx, sgy, sgz)
zeKernelSetArgumentValue(kernel, 0, sizeof(ptr), &buf_ptr)
...

ze_group_count_t groups = {grid_x/sgx, grid_y/sgy, grid_z/sgz};
zeCommandListAppendLaunchKernel(cmdList, kernel, &groups, signal, 0, nullptr)

// Regular path:
zeCommandListClose(cmdList)
zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, fence)
zeCommandQueueSynchronize(queue, timeout)

// Immediate path (v2 adapter, Xe2+):
// No explicit submit — each Append is immediate
zeCommandListHostSynchronize(immCmdList, timeout)
```

### Key Architectural Observations

1. **SPIR-V is the native IR.** Level Zero is explicitly designed around SPIR-V as the kernel IR (`ZE_MODULE_FORMAT_IL_SPIRV`). This is not an afterthought — unlike CUDA's PTX which is text-based and NVIDIA-specific, SPIR-V is binary and vendor-neutral.

2. **Two-level dispatch abstraction.** The UR layer (oneapi-src/unified-runtime) sits between DPC++/SYCL and Level Zero, providing the same adapter pattern libkdl needs. libkdl's architecture directly parallels UR's design.

3. **Immediate vs. regular command lists.** The split maps cleanly to "latency-sensitive" vs. "throughput-sensitive" dispatch — libkdl should expose this as a backend configuration option. The L0 v2 adapter betting on immediate-only suggests Intel's direction for the future.

4. **AOT caching via `zeModuleGetNativeBinary`.** Level Zero explicitly supports exporting compiled native binary and re-loading it (`ZE_MODULE_FORMAT_NATIVE`). This is libkdl's disk caching mechanism for Intel targets — avoids SPIR-V JIT overhead on subsequent loads.

5. **Dynamic linking on GPU.** `zeModuleDynamicLink` provides GPU-side dynamic linking analogous to `ld.so`, resolving import symbols across separately compiled modules. This is the Level Zero analog to libkdl's core concept.

6. **Performance baseline from chipStar.** 0.75x geometric mean vs native HIP through a SPIR-V/Level Zero path establishes the overhead floor for a fully-portable dispatch chain. libkdl targeting native Level Zero (not via chipStar) should achieve higher, closer to 1.0x.

### Gaps and Risks

- Level Zero lacks widely-published head-to-head kernel dispatch latency benchmarks vs CUDA/HIP. Phoronix's early (2020) benchmarks covered Gen9/Gen11 iGPU micro-benchmarks but not dispatch latency specifically.
- The L0 v2 adapter (immediate-only) means existing code using regular command lists must be migrated when targeting Xe2+ with the default adapter.
- PoCL's Level Zero driver still has known FP64/FP16 failures and 64-bit atomic hazards as of March 2025 — indicates the Level Zero ecosystem is still maturing for full HPC coverage.
- `zeModuleCreate` performs synchronous JIT compilation — for cold-start latency, libkdl must cache native binaries via `zeModuleGetNativeBinary`.

---

## Relevance to libkdl / Vendor-Agnostic Kernel Dispatch

Level Zero is the most complete open-standard GPU dispatch API for Intel hardware and provides direct SPIR-V support — making it the correct backend for libkdl's Intel GPU path. The API parallels CUDA/HIP closely enough that libkdl's abstraction layer can mirror CUDA's module/function/launch pattern with minimal adaptation. The UR adapter model (oneapi-src/unified-runtime) is an existing implementation of exactly the libkdl architecture pattern: dynamic adapter loading, unified object hierarchy, free-threaded dispatch. The chipStar 2026 paper provides peer-reviewed validation that SPIR-V → Level Zero dispatch achieves ~75% of native performance — a strong baseline for a portability-first dispatch system.

---

## Sources

- [Level Zero Core Programming Guide (spec v1.11)](https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/PROG.html)
- [Introduction to Level Zero API — Juan Fumero](https://jjfumero.github.io/posts/2021/09/introduction-to-level-zero/)
- [Level Zero Immediate Command Lists — Intel](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)
- [Level Zero UR Reference Document](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html)
- [Unified Runtime Introduction](https://oneapi-src.github.io/unified-runtime/core/INTRO.html)
- [Intel oneAPI DPC++/C++ Compiler 2025 Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html)
- [PoCL Level Zero Driver docs](https://portablecl.org/docs/html/level0.html)
- [chipStar 2026 paper — IJPP](https://journals.sagepub.com/doi/10.1177/10943420261423001)
- [Zero in on Level Zero — Intel](https://www.intel.com/content/www/us/en/developer/articles/technical/zero-in-on-level-zero-oneapi-open-backend-approach.html)
- [Unified Runtime Design — intel/llvm](https://intel.github.io/llvm/design/UnifiedRuntime.html)
