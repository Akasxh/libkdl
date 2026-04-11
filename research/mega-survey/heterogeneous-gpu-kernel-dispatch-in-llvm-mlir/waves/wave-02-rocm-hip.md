# Wave 02 — ROCm HIP Runtime Kernel Dispatch

**Survey:** Heterogeneous GPU Kernel Dispatch in LLVM/MLIR
**Angle:** ROCm HIP Runtime Kernel Dispatch
**Search query:** "ROCm HIP runtime kernel dispatch hipModuleLoad hipLaunchKernel code object v5"
**Priority source types:** docs, PR, commit
**Date:** 2026-04-06

---

## Sources

### Source 1: HIP Kernel Execution and Modules — DeepWiki (ROCm/rocm-systems)
- **URL:** https://deepwiki.com/ROCm/rocm-systems/2.3-hip-kernel-execution
- **Date:** Current (ROCm 7.x)
- **Type:** Synthesized documentation / codebase analysis
- **Relevance:** 10/10
- **Novelty:** 8/10
- **Summary:** Comprehensive breakdown of HIP's module loading and kernel dispatch internals. `hipModuleLoad` (file path), `hipModuleLoadData` (memory buffer), and `hipModuleLoadDataEx` (with JIT options) all delegate to `PlatformState::instance().loadModule()`, which handles device-specific code object extraction and kernel registration. Kernel argument passing supports both `kernelParams` arrays and the structured `HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE/END` extra-buffer format.
- **Key technical detail:** `hipModuleGetFunction` returns a `hipFunction_t` pointing to a `hip::DeviceFunc` object encapsulating `amd::Kernel` plus device-specific metadata. The launch validation pipeline in `ihipLaunchKernel_validate` enforces dimension bounds (gridDimX ≤ INT32_MAX, gridDimY/Z ≤ UINT16_MAX+1), `maxWorkGroupSize`, and `localMemSizePerCU` before command enqueue. Stream capture replaces direct execution with `GraphKernelNode` creation, with an optimized path that pre-captures AQL packets for near-zero-overhead graph replay.

---

### Source 2: HIP Extended Kernel Launch APIs — DeepWiki (ROCm/hip)
- **URL:** https://deepwiki.com/ROCm/hip/5.2-extended-kernel-launch-apis
- **Date:** Current (ROCm 7.x)
- **Type:** Synthesized documentation / codebase analysis
- **Relevance:** 9/10
- **Novelty:** 7/10
- **Summary:** Documents three extended launch APIs with progressively higher-level interfaces: `hipExtModuleLaunchKernel` (module handle, integrated timing), `hipExtLaunchKernel` (function pointer, dim3 dimensions), and `hipExtLaunchKernelGGL` (type-safe C++ template with variadic compile-time argument validation). All three feed a common `amd::NDRangeKernelCommand` creation path.
- **Key technical detail:** `hipExtAnyOrderLaunch` flag — permitting out-of-order execution within a stream — is explicitly **not supported on AMD GFX9xx boards**. The `startEvent` on AMD platforms records when the kernel **completes**, not when it starts; this is a semantic inversion vs. CUDA's stream-ordered events, critical for latency measurement in a dispatch benchmarking layer. For libkdl, using `hipExtModuleLaunchKernel` with profiling events is the correct path to measure per-kernel dispatch overhead.

---

### Source 3: AMDGPU Backend User Guide — LLVM Project Documentation
- **URL:** https://llvm.org/docs/AMDGPUUsage.html
- **Date:** Current (LLVM 23.x)
- **Type:** Official LLVM documentation
- **Relevance:** 10/10
- **Novelty:** 7/10
- **Summary:** Canonical reference for AMD code object format versioning (V2 through V6). Covers ELF structure of `.hsaco` files: `.text` section for GCN/RDNA machine code, `.rodata` for read-only data, note records carrying ISA version and feature flags, and symbol tables for kernel entry points. Code Object V5 (default since ROCm 5.x, `--mcode-object-version=5`) refines metadata for kernarg layout, wavefront size, and implicit kernel arguments.
- **Key technical detail:** Code Object V6 introduces "generic processors" with versioning — a forward-compatibility mechanism analogous to PTX virtual ISA, allowing code objects to run on architectures defined after compilation. The `amdhsa_kernel_descriptor` is a fixed-size struct in `.rodata` containing VGPR/SGPR counts, scratch memory size, workgroup size constraints, and the AQL dispatch setup fields; it is the AMD equivalent of CUDA's `cuFunctionLoadingMode` metadata. Multi-arch fat binaries use `clang-offload-bundler` to pack per-gfx code objects into a single container; the HIP runtime iterates bundle entries at load time and selects the one matching the detected GPU's ISA.

---

### Source 4: HIP RTC (Runtime Compilation) — ROCm Official Docs
- **URL:** https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
- **Date:** Current (HIP 7.2)
- **Type:** Official documentation
- **Relevance:** 9/10
- **Novelty:** 7/10
- **Summary:** HIP RTC exposes a `hiprtc*` API surface parallel to NVRTC. `hiprtcCreateProgram` + `hiprtcCompileProgram` + `hiprtcGetCode` produces a code object binary from a source string, which is then loaded via `hipModuleLoadData`. The `--gpu-architecture` flag (e.g., `--gpu-architecture=gfx906:sramecc+:xnack-`) targets specific ISA variants including feature flags.
- **Key technical detail:** `-fgpu-rdc` mode produces LLVM bitcode rather than a final binary, enabling deferred multi-arch bundling via `hiprtcLinkCreate` + `hiprtcLinkAddData` + `hiprtcLinkComplete`. "HIPRTC assumes WGP mode by default for RDNA GPUs" (overridable with `-mcumode`) — a dispatch-affecting default that a vendor-agnostic layer must normalize. The `hiprtcAddNameExpression` / `hiprtcGetLoweredName` pair is the RTC equivalent of `dlsym`: mapping source-level kernel names to mangled symbol names in the compiled binary.

---

### Source 5: HIP Fat Binary Format and `__hipRegisterFatBinary` — Clang HIP Support Docs
- **URL:** https://clang.llvm.org/docs/HIPSupport.html
- **Date:** Current (Clang 23.x)
- **Type:** Official Clang documentation
- **Relevance:** 9/10
- **Novelty:** 8/10
- **Summary:** Documents the full compilation pipeline from HIP C++ source to fat binary to runtime registration. In non-RDC mode, each translation unit produces a self-contained fat binary (one fully-linked device image per enabled gfx target). In RDC mode, per-TU bitcode is device-linked across TUs before packaging. At startup, Clang-generated constructors call `__hipRegisterFatBinary` with a descriptor struct, registering all kernels and device symbols with the runtime.
- **Key technical detail:** When a host-side kernel launch stub fires, the HIP runtime uses the fat binary handle and kernel name registered at startup to resolve which device image to dispatch — this is a two-phase dispatch: (1) static registration at program load, (2) dynamic selection at launch time. This registration/resolution split is structurally identical to `ld.so`'s symbol table construction vs. PLT slot resolution. A `libkdl` layer could intercept at the fat binary registration boundary to implement its own architecture-selection and caching policy.

---

### Source 6: ROCm TheRock Multi-Arch kpack — GitHub Issue #3531
- **URL:** https://github.com/ROCm/TheRock/issues/3531
- **Date:** February 2026
- **Type:** GitHub issue (active development)
- **Relevance:** 8/10
- **Novelty:** 9/10
- **Summary:** Documents the kpack system — ROCm's runtime mechanism for splitting multi-architecture fat binaries into per-gfx `.kpack` archive files and loading them on demand. The ELF splitter extracts HIP code objects from `.hipk` ELF sections and packages them per architecture. At runtime, the system expands `@GFXARCH@` template variables to the bare arch name (e.g., `gfx1201`) and opens the matching `.kpack` archive to load code objects via `hipModuleLoad`.
- **Key technical detail:** Critical architectural limitation: "Tensile-generated kernels are loaded at runtime from an external database, not compiled into the library's ELF as standard HIP code objects" — the kpack splitter does not intercept this path. Libraries like rocBLAS and hipBLASLt use a completely separate `.co` file loading path via `hipModuleLoad`, bypassing the fat-binary ELF mechanism entirely. This two-track system (ELF-embedded fat binaries vs. external `.co` databases) is a direct architectural parallel to the problem libkdl must solve: unified dispatch regardless of how kernel binaries were packaged.

---

### Source 7: ROCm vs CUDA Kernel Launch Latency — Kokkos Issue + Benchmarks
- **URL:** https://github.com/kokkos/kokkos/issues/3670
- **Date:** 2021–2024 (ongoing)
- **Type:** Open-source project issue / performance investigation
- **Relevance:** 8/10
- **Novelty:** 8/10
- **Summary:** Empirical comparison of HIP vs. CUDA kernel launch overhead from the Kokkos portability framework team. HIP's current launch mechanism measures approximately 70 µs per kernel in the unoptimized path; optimized batched HIP achieves ~25 µs vs. CUDA's ~3 µs for batched and ~8 µs for fenced launches. Even the optimized HIP path is 4–5x slower than CUDA's batched path.
- **Key technical detail:** The 7–14x latency gap (unoptimized HIP vs. CUDA) stems from HIP's AQL packet submission path going through additional validation layers and the ROCm thunk/kernel-mode driver (KFD) doorbell mechanism. ROCm 7.1 (November 2025) specifically targets this gap with "lower module-load latency, faster kernel-metadata retrieval, and improved doorbell batching for graph launches." For a libkdl dispatch layer, this baseline gap means that any additional overhead from dispatch indirection is dominated by the existing HIP overhead — making aggressive optimization of the lookup path less critical on ROCm than on CUDA.

---

### Source 8: ROCm Architecture and Components Overview — DeepWiki
- **URL:** https://deepwiki.com/ROCm/ROCm/2-architecture-and-components
- **Date:** Current (ROCm 7.x)
- **Type:** Synthesized documentation / codebase analysis
- **Relevance:** 8/10
- **Novelty:** 6/10
- **Summary:** ROCm's layered dispatch stack from top to bottom: HIP C++ API → CLR (Common Language Runtime, formerly HIP runtime) → ROCR (HSA Runtime, user-space) → thunk (libhsakmt, ioctl bridge) → KFD (Kernel Fusion Driver, kernel-mode). The Code Object Manager (Comgr) operates as a compiler-support library alongside this stack, used by HIP RTC and the CLR for in-process compilation and code object introspection.
- **Key technical detail:** Comgr (now in ROCm/llvm-project under `amd/comgr`) provides `amd_comgr_action_*` APIs for: compiling source to bitcode (`AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC`), assembling, linking, disassembling, and inspecting code objects. Critically, Comgr can enumerate the kernels and their metadata within a code object binary — providing the introspection primitive a kernel dynamic linker needs to build its symbol table from an arbitrary `.hsaco` or fat binary blob without executing it.

---

### Source 9: ROCR Runtime (HSA Runtime) — Architecture Overview
- **URL:** https://deepwiki.com/ROCm/rocm-systems/4-rocr-runtime-(hsa-runtime)
- **Date:** Current (ROCm 7.x)
- **Type:** Synthesized documentation / codebase analysis
- **Relevance:** 8/10
- **Novelty:** 7/10
- **Summary:** ROCR is AMD's HSA specification implementation, sitting below the HIP CLR layer. `GpuAgent` initialization performs ISA detection (lines 133–186 of `amd_gpu_agent.cpp`), determining the gfx family (gfx9/gfx10/gfx11/gfx12) and feature flags (SRAMECC, XNACK). `hsa_executable_load_agent_code_object` is the HSA-level API for loading a compiled code object into an executable, binding it to a specific agent (GPU device).
- **Key technical detail:** Architecture selection at the HSA layer happens per-agent: the caller is responsible for providing the correctly-targeted code object (already matched to the agent's ISA) before calling `hsa_executable_load_agent_code_object`. The HIP CLR layer above performs the fat-binary architecture matching (iterating clang-offload-bundle entries) before handing the selected code object down to ROCR. This separation of concerns — HIP does selection, HSA does loading — is the correct model for libkdl to follow: selection policy lives in the dispatch layer, binary loading in the backend.

---

### Source 10: ROCm 7.1 Release — Kernel Launch and Module Load Improvements
- **URL:** https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.1/README.html
- **Date:** November 2025
- **Type:** AMD Engineering Blog
- **Relevance:** 7/10
- **Novelty:** 8/10
- **Summary:** ROCm 7.1 release notes enumerate targeted performance improvements to the dispatch path: reduced time-to-first-kernel through lower `hipModuleLoad` latency and faster kernel metadata retrieval, plus improved doorbell batching for HIP graph launches. Positions these improvements as addressing the Kokkos-documented latency gap vs. CUDA, particularly for graph-heavy ML training workloads.
- **Key technical detail:** The "faster kernel-metadata retrieval" improvement implies that metadata lookup — the step of resolving a kernel name to its `amdhsa_kernel_descriptor` and resource requirements — was a measurable bottleneck. This validates the design choice in libkdl to pre-build and cache a kernel symbol table at module load time rather than performing on-demand metadata parsing at each dispatch. ROCm 7.1 effectively implements a version of this optimization at the runtime level.

---

## Angle Assessment

**Coverage:** High. ROCm/HIP's kernel dispatch stack is well-documented across four distinct layers (HIP C++ → CLR → ROCR/HSA → KFD), with clear separation of responsibilities at each boundary. The fat-binary architecture selection mechanism (clang-offload-bundler containers, `__hipRegisterFatBinary` at startup, per-launch symbol resolution) is structurally well-understood.

**Relevance to libkdl:** Very high. The HIP dispatch pipeline maps to the `ld.so` analogy at multiple levels:

| ld.so concept | HIP / ROCm equivalent |
|---|---|
| `dlopen(path)` | `hipModuleLoad(path)` / `hipModuleLoadData(blob)` |
| `dlsym(handle, name)` | `hipModuleGetFunction(module, name)` |
| lazy PLT binding | Not natively supported; ROCm 7.1 improves module-load latency instead |
| fat-binary arch selection | `clang-offload-bundler` container unwrap in CLR at `__hipRegisterFatBinary` time |
| symbol interposition | No direct equivalent; Comgr introspection can achieve similar inspection |
| link-time optimization | `hiprtcLinkCreate` + `-fgpu-rdc` bitcode linking |
| shared library soname versioning | Code Object format version field (V3/V4/V5/V6) |
| external `.so` kernel database | Tensile `.co` file path in rocBLAS/hipBLASLt (bypasses fat-binary mechanism) |

**Key gaps and risks found:**

1. **Two-track binary delivery problem:** ROCm has two incompatible runtime kernel loading paths — ELF-embedded fat binaries (via `__hipRegisterFatBinary`) and external `.co` file databases (Tensile/rocBLAS). The kpack system (TheRock) only handles the first path. A libkdl implementation targeting ROCm must handle both tracks or explicitly exclude the Tensile path.

2. **Architecture selection is pre-dispatch, not lazy:** Unlike CUDA's `CUDA_MODULE_LOADING=LAZY`, HIP resolves architecture selection at registration time (`__hipRegisterFatBinary`), not at first launch. This limits opportunities for deferred architecture selection based on actual workload patterns. A libkdl intercept at the registration boundary could implement genuine lazy selection.

3. **Latency baseline gap vs. CUDA:** HIP kernel launch overhead is 4–14x higher than CUDA even before any additional dispatch indirection. For a vendor-agnostic dispatch layer, this means the absolute overhead contribution from libkdl lookup is relatively less important on ROCm than on CUDA — the optimization priority should be minimizing `hipModuleLoad` calls (amortize across kernels) rather than minimizing per-launch lookup.

4. **`hipExtAnyOrderLaunch` gfx9 limitation:** Out-of-order kernel scheduling within a stream is unsupported on gfx9 (Vega/MI series). A dispatch layer that uses out-of-order scheduling as an optimization must gate this capability on runtime architecture detection.

5. **`startEvent` semantic inversion:** HIP's `startEvent` fires on kernel completion, not launch. Any timing infrastructure in libkdl must account for this when computing dispatch latency on AMD vs. NVIDIA.

**Suggested new research angles from this investigation:**

- HSA Signals and AQL queue direct submission — bypassing the HIP CLR layer entirely for minimal-overhead dispatch (analogous to CUDA's `cuLaunchKernel` vs. PTX direct submission)
- Comgr `amd_comgr_action_*` API as a universal code object introspection layer — could serve as the AMD backend for libkdl's kernel symbol enumeration
- ROCm kpack ELF splitter design as a reference architecture for libkdl's multi-arch binary packaging strategy
- XNACK and SRAMECC feature flags as dispatch metadata — kernels compiled for `gfx906:sramecc+:xnack-` cannot run on `gfx906:sramecc-:xnack+`; libkdl must track feature flags, not just base ISA names

---

## Sources (Inline Reference List)

- [HIP Kernel Execution and Modules — DeepWiki](https://deepwiki.com/ROCm/rocm-systems/2.3-hip-kernel-execution)
- [Extended Kernel Launch APIs — DeepWiki](https://deepwiki.com/ROCm/hip/5.2-extended-kernel-launch-apis)
- [AMDGPU Backend User Guide — LLVM Docs](https://llvm.org/docs/AMDGPUUsage.html)
- [HIP RTC Programming Guide — ROCm Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html)
- [HIP Support — Clang Documentation](https://clang.llvm.org/docs/HIPSupport.html)
- [Multi-arch kpack Issue #3531 — ROCm/TheRock](https://github.com/ROCm/TheRock/issues/3531)
- [HIP Launch Latency Investigation — kokkos/kokkos Issue #3670](https://github.com/kokkos/kokkos/issues/3670)
- [Architecture and Components — DeepWiki ROCm](https://deepwiki.com/ROCm/ROCm/2-architecture-and-components)
- [ROCR Runtime — DeepWiki ROCm Systems](https://deepwiki.com/ROCm/rocm-systems/4-rocr-runtime-(hsa-runtime))
- [ROCm 7.1 Release Blog — AMD](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.1/README.html)
- [ROCm Code Object Format — ReadTheDocs](https://rocmdoc.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Codeobj-format.html)
- [HSA Runtime API — ROCm Docs](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/)
- [ROCm-CompilerSupport (Comgr) — GitHub](https://github.com/ROCm/ROCm-CompilerSupport)
- [Execution Control Reference — HIP 7.1 Docs](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/reference/hip_runtime_api/modules/execution_control.html)
