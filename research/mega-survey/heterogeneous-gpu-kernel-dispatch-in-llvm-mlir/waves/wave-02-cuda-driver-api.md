# Wave 02 — CUDA Driver API Lazy Module Loading

**Survey:** Heterogeneous GPU Kernel Dispatch in LLVM/MLIR
**Angle:** CUDA Driver API Lazy Module Loading
**Search query:** "CUDA driver API cuModuleLoadData cuLaunchKernel lazy loading JIT PTX runtime"
**Priority source types:** docs, blog, paper
**Date:** 2026-04-06

---

## Sources

### Source 1: CUDA Driver API — Module Management Reference
- **URL:** https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- **Date:** Current (CUDA 13.2)
- **Type:** Official documentation
- **Relevance:** 10/10
- **Novelty:** 7/10
- **Summary:** Canonical reference for `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleLoadFatBinary`, `cuModuleUnload`, `cuModuleGetFunction`. Covers the full lifecycle of a CUDA module: load from file/memory/fatbinary, extract function handles, unload. The `cuModuleLoadData` path accepts a NULL-terminated PTX string or a cubin/fatbin blob, driving on-the-fly JIT compilation by the driver when PTX is supplied. `cuModuleGetLoadingMode` exposes the lazy-loading state.
- **Key technical detail:** `cuModuleLoadDataEx` accepts `CUjit_option` arrays (e.g., `CU_JIT_MAX_REGISTERS`, `CU_JIT_TARGET`, `CU_JIT_LOG_VERBOSE`) giving the caller full control over compilation policy. This is the exact interface a kernel dynamic linker would sit on top of.

---

### Source 2: CUDA Lazy Loading (Programming Guide § 4.7)
- **URL:** https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html
- **Date:** Current (CUDA 13.2)
- **Type:** Official documentation
- **Relevance:** 10/10
- **Novelty:** 8/10
- **Summary:** Documents CUDA's built-in lazy module loading, introduced in CUDA 11.7 (driver 515+). Setting `CUDA_MODULE_LOADING=LAZY` defers loading of all functions in a module until the moment of first use, reducing device memory footprint and initialization time for applications that import large libraries but invoke only a few kernels. Tradeoff: a one-time latency spike on first call to any new kernel.
- **Key technical detail:** Lazy loading is transparent at the `cuLaunchKernel` boundary — the driver intercepts the first launch of a deferred function, JIT-compiles and loads it, then proceeds. This is structurally analogous to PLT/GOT lazy binding in ELF: the first call pays the resolution cost, subsequent calls are fast. `cuModuleGetLoadingMode` lets code query whether lazy mode is active.

---

### Source 3: CUDA Context-Independent Module Loading (cuLibraryLoad)
- **URL:** https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
- **Date:** 2023 (CUDA 12.0)
- **Type:** NVIDIA Technical Blog
- **Relevance:** 10/10
- **Novelty:** 9/10
- **Summary:** Introduces `cuLibraryLoad`/`cuLibraryGetKernel` (CUDA 12.0 Driver API) and their runtime twins `cudaLibraryLoad`/`cudaLibraryGetKernel`. Modules loaded via these APIs are context-independent: the driver automatically propagates the module into every CUDA context that is created or destroyed, and the resulting `CUkernel` handle can be launched directly via `cuLaunchKernel` without per-context `CUmodule`/`CUfunction` bookkeeping.
- **Key technical detail:** This is the closest NVIDIA has come to a "kernel registry" primitive. A single `cuLibraryLoad` call at startup registers a code object system-wide; any context gets the kernel for free. For libkdl, this maps directly to the global kernel symbol table the dispatch layer would maintain.

---

### Source 4: Dynamic Loading in the CUDA Runtime (cudaLibraryLoad)
- **URL:** https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime
- **Date:** 2023–2024
- **Type:** NVIDIA Technical Blog
- **Relevance:** 9/10
- **Novelty:** 8/10
- **Summary:** Extends context-independent loading to the CUDA runtime API. `cudaLibraryLoad` and `cudaLibraryGetKernel` let pure-runtime-API applications (no explicit driver API calls) do the same on-demand loading. Key claim: kernel handles (`cudaKernel_t`) can be shared across separately linked CUDA runtime instances — enabling inter-library kernel dispatch without any shared global state.
- **Key technical detail:** `cudaLibraryLoad` accepts a cubin/PTX/fatbin blob from memory, matching the `dlopen`+`dlsym` pattern exactly. The returned `cudaLibrary_t` is the equivalent of a `void*` handle from `dlopen`.

---

### Source 5: CUDA 12.0 Runtime LTO via nvJitLink
- **URL:** https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/
- **Date:** December 2022 (CUDA 12.0)
- **Type:** NVIDIA Technical Blog
- **Relevance:** 8/10
- **Novelty:** 9/10
- **Summary:** Introduces `nvJitLink`, a standalone library for JIT link-time optimization. Accepts device objects, PTX, cubin, LTO-IR, and host object files; produces a final cubin via online linking. Decouples LTO from the CUDA driver version — applications compiled with CUDA 12.x LTO-IR can link at runtime against any driver in the same major release, removing the compatibility constraint of driver-bundled PTX-JIT.
- **Key technical detail:** `nvJitLinkAddData` + `nvJitLinkComplete` + `cuModuleLoadData` is the canonical pipeline for composing multiple device translation units at runtime. For a kernel dynamic linker, this is the "runtime relocation" step — analogous to how `ld.so` resolves inter-library symbol references via the PLT.

---

### Source 6: PTX Compiler API (libNVPTXCompiler)
- **URL:** https://docs.nvidia.com/cuda/ptx-compiler-api/index.html
- **Date:** Current (CUDA 13.2)
- **Type:** Official documentation
- **Relevance:** 8/10
- **Novelty:** 8/10
- **Summary:** A standalone shared library that compiles PTX to cubin entirely in-process, independent of the CUDA driver. `nvPTXCompilerCreate` + `nvPTXCompilerCompile` + `nvPTXCompilerGetCompiledProgramSize`/`GetCompiledProgram` produces a cubin blob. The blob can then be fed into `cuModuleLoadData`. Crucially, this separates compilation from the driver context lifecycle, enabling ahead-of-time compilation without a live device context.
- **Key technical detail:** The PTX Compiler API was designed specifically to decouple compilation from loading — a clear two-phase model. For libkdl, Phase 1 (compile PTX → cubin) can run at install time or first-use, and Phase 2 (load cubin → CUmodule) runs per-context at dispatch time, mirroring how `ldconfig` pre-links vs. `ld.so` loads at execution.

---

### Source 7: Understanding the Overheads of Launching CUDA Kernels (ICPP 2019)
- **URL:** https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf
- **Date:** 2019 (ICPP)
- **Type:** Academic paper (poster)
- **Relevance:** 8/10
- **Novelty:** 7/10
- **Summary:** Empirical measurement study of kernel launch overhead. Distinguishes CPU-side launch overhead (the `cuLaunchKernel` call itself, ~2–4 µs) from the GPU-side scheduling and warp setup overhead. Uses kernel fusion as a probe technique to isolate the bare dispatch cost from execution time.
- **Key technical detail:** `cuLaunchKernel` CPU-side latency was measured at approximately 2.5 µs on a mobile A5000 and 3.5 µs on a datacenter A40. This sets the baseline budget for any additional dispatch overhead introduced by a kernel dynamic linker layer; a lookup + pointer indirection should stay well below 1 µs to remain negligible.

---

### Source 8: NVRTC — NVIDIA Runtime Compilation
- **URL:** https://docs.nvidia.com/cuda/nvrtc/index.html
- **Date:** Current (CUDA 13.2)
- **Type:** Official documentation
- **Relevance:** 7/10
- **Novelty:** 6/10
- **Summary:** NVRTC compiles CUDA C++ source strings to PTX at runtime, bypassing disk I/O and subprocess spawning. The PTX output is then fed into `cuModuleLoadData` or `nvJitLink`. First-call compilation latency is ~600 ms; subsequent calls hit the driver-managed compute cache (~256 MiB default). The Jitify library wraps NVRTC + module caching into a single header.
- **Key technical detail:** The compute cache (`~/.nv/ComputeCache`) is driver-invalidated on driver upgrade, forcing recompilation. For a kernel dynamic linker that caches cubins persistently, the cache invalidation policy must track driver version and GPU architecture — exactly as `ldconfig` tracks shared-library soname and ABI version.

---

### Source 9: Understanding PTX — Assembly Language of CUDA GPU Computing
- **URL:** https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/
- **Date:** 2024
- **Type:** NVIDIA Technical Blog
- **Relevance:** 7/10
- **Novelty:** 6/10
- **Summary:** Explains PTX as a virtual ISA: forward-compatible across GPU generations within the same major architecture, with the CUDA runtime automatically JIT-compiling PTX to cubin if no matching cubin is present in the fatbinary. Covers the fatbinary structure: a container holding multiple (PTX, compute capability) pairs, with the runtime selecting the best-matching cubin or PTX fallback.
- **Key technical detail:** The runtime's fat-binary selection algorithm is the GPU equivalent of the ELF multi-arch dispatch: iterate candidates, score by ISA match, fall back to PTX-JIT if no cubin matches. libkdl would replicate this selection logic but extend it to vendor-neutral IR (SPIR-V, MLIR) as additional candidates alongside cubin/PTX.

---

### Source 10: Jitify — Single-Header CUDA Runtime Compilation Library
- **URL:** https://github.com/NVIDIA/jitify
- **Date:** Active (2024–2025)
- **Type:** Open-source library (GitHub)
- **Relevance:** 7/10
- **Novelty:** 7/10
- **Summary:** NVIDIA-maintained C++ library that wraps NVRTC + `cuModuleLoadData` + an in-process kernel cache (keyed by source hash + GPU arch). Handles header inclusion, preprocessor defines, and caching transparently. Used in production by cuDF, RAPIDS, and other NVIDIA data-science libraries.
- **Key technical detail:** Jitify's caching model — hash source string + compile flags + GPU arch → cubin blob, store in `std::unordered_map` — is a direct prototype of the kernel cache layer libkdl would need. The existing code is Apache-licensed and could be adapted as the NVIDIA-backend cache in a vendor-agnostic dispatch layer.

---

## Angle Assessment

**Coverage:** High. CUDA's kernel loading machinery is thoroughly documented and has matured through four distinct phases: (1) classic `cuModuleLoad`/`cuModuleLoadData` (CUDA 1–10), (2) PTX Compiler API decoupling compilation from loading (CUDA 11), (3) lazy module loading via `CUDA_MODULE_LOADING=LAZY` (CUDA 11.7), and (4) context-independent `cuLibraryLoad`/`cuKernel` with cross-library handle sharing (CUDA 12.0).

**Relevance to libkdl:** Extremely high. The CUDA Driver API is the ground truth for what a GPU kernel dynamic linker must do on NVIDIA hardware. The API surface maps cleanly onto the `ld.so` analogy:

| ld.so concept | CUDA Driver API equivalent |
|---|---|
| `dlopen(path)` | `cuLibraryLoad(blob, ...)` |
| `dlsym(handle, name)` | `cuLibraryGetKernel(lib, name)` |
| lazy PLT binding | `CUDA_MODULE_LOADING=LAZY` |
| cache invalidation on soname change | compute cache invalidated on driver upgrade |
| in-process relocation | `nvJitLink` online linking |
| multi-arch `RPATH` selection | fat-binary cubin/PTX selection |

**Key gaps found:**
1. CUDA's lazy loading is opt-in via environment variable, not the default — a kernel dynamic linker should make lazy-by-default the policy, not leave it to the environment.
2. Context-independent loading (`cuLibraryLoad`) does not yet support on-the-fly PTX compilation — it requires a pre-compiled cubin or fatbin. Combining `nvPTXCompiler` + `cuLibraryLoad` into a single pipeline is not documented as a recommended pattern.
3. `cuLaunchKernel` dispatch cost (~2.5–3.5 µs CPU-side) sets the overhead budget for any dispatch indirection layer.

**Suggested new angles from this research:**
- NVRTC + nvJitLink integration patterns in production ML frameworks (cuDF, RAPIDS) — evidence that JIT-compiled kernel dispatch is production-viable
- Green Contexts (CUDA 12.4) as a partitioning primitive for heterogeneous sub-device dispatch
- Compute cache management policy across driver versions — an unsolved problem for long-lived kernel registries

---

## Sources (Inline Reference List)

- [CUDA Driver API Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
- [CUDA Lazy Loading — Programming Guide § 4.7](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)
- [CUDA Context-Independent Module Loading — NVIDIA Blog](https://developer.nvidia.com/blog/cuda-context-independent-module-loading/)
- [Dynamic Loading in the CUDA Runtime — NVIDIA Blog](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime)
- [CUDA 12.0 Runtime LTO via nvJitLink — NVIDIA Blog](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/)
- [PTX Compiler API Documentation](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html)
- [Understanding the Overheads of Launching CUDA Kernels — ICPP 2019](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf)
- [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [Understanding PTX — NVIDIA Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
- [Jitify — NVIDIA GitHub](https://github.com/NVIDIA/jitify)
- [CUDA Driver API vs Runtime API](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html)
- [CUDA Pro Tip: Fat Binaries and JIT Caching — NVIDIA Blog](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
- [nvJitLink Documentation](https://docs.nvidia.com/cuda/nvjitlink/index.html)
- [Green Contexts — CUDA Programming Guide § 4.6](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)
