# Wave 06 — Dynamic Linking GPU Kernels: ELF Code Objects and Runtime Loading

**Survey:** Heterogeneous GPU Kernel Dispatch in LLVM/MLIR
**Angle:** dynamic-linking-gpu-kernels
**Search queries:**
- "dynamic linking GPU kernel shared library loading runtime ELF code object"
- "CUDA cuModuleLoad dynamic GPU kernel loading driver API"
- "AMD HSA HSACO ELF GPU code object dynamic loading hsa_executable_load"
- "GPU shared library dynamic linking kernel dlopen equivalent runtime 2025"
- "CUDA dynamic parallelism separate compilation linking device code cuLink"
- "LLVM liboffload GPU kernel binary ELF code object runtime loading offload API"
- "ROCm comgr code object manager HSACO dynamic loading kernel symbol"
- "NVRTC runtime compilation PTX JIT CUDA kernel compile load execute"
- "SPIR-V OpenCL clCreateProgramWithBinary GPU binary loading cross-vendor"
- "CUDA context-independent module loading cudaLibraryLoad 2023 2024"
- "name-based kernel loading GPU ELF symbol table runtime dispatch llvm offload"
**Priority source types:** docs, blog, GitHub issue, official API reference
**Date:** 2026-04-06

---

## Core Research Question

How do existing GPU runtimes (CUDA, ROCm/HSA, OpenCL/SPIR-V, LLVM offload) actually load binary GPU code at runtime? What are the binary formats, the loading APIs, the symbol resolution mechanisms, and where are the gaps that libkdl fills?

This wave complements wave-05 (which covered the `dlopen`/`dlsym` analogy at a conceptual level) by focusing on the concrete technical mechanisms: what bytes are exchanged, what APIs are called, what ELF sections matter, and what the real runtime loading pipelines look like.

---

## Sources

### Source 1 — CUDA Driver API: cuModuleLoad / cuModuleGetFunction
- **URL:** https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html
- **URL:** https://leimao.github.io/blog/CUDA-Driver-Runtime-Load-Run-Kernel/
- **Date:** Current (CUDA 13.2); blog 2022
- **Type:** Official documentation + technical blog
- **Relevance:** 10/10
- **Novelty:** 6/10
- **Summary:** The CUDA Driver API provides the foundational GPU dynamic loading primitives. `cuModuleLoad(path)` loads a PTX, CUBIN, or FATBIN file from disk into a CUmodule. `cuModuleLoadData(data_ptr)` loads from an in-memory blob — the runtime analog of `dlopen()` with `RTLD_LAZY`. `cuModuleGetFunction(module, "kernel_name")` is `dlsym()` — it resolves a named kernel to a `CUfunction` handle. The driver performs architecture selection internally: for a fatbin, it picks the cubin matching the executing SM, or falls back to PTX JIT.

  Full loading pipeline:
  ```
  cuInit(0)
  → cuCtxCreate(&ctx, 0, device)
  → cuModuleLoad(&module, "/path/to/kernel.cubin")   // or cuModuleLoadData(blob)
  → cuModuleGetFunction(&func, module, "my_kernel")  // symbol resolution by name
  → cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, stream, args, 0)
  ```

- **Key detail for libkdl:** `cuModuleLoadData` + `cuModuleGetFunction` is the exact two-step load-then-resolve pattern libkdl abstracts over. The CUDA driver does lazy context binding but requires the host to manage module lifetimes explicitly. The new `cuLibraryLoad` (CUDA 12.0) solves lifetime management by making modules context-independent — see Source 2.

---

### Source 2 — CUDA 12.0: Context-Independent Module Loading (cuLibraryLoad / cudaLibraryLoad)
- **URL:** https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
- **URL:** https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/
- **Date:** 2023 (CUDA 12.0 launch)
- **Type:** NVIDIA Technical Blog (two posts)
- **Relevance:** 10/10
- **Novelty:** 9/10
- **Summary:** CUDA 12.0 elevated `dlopen`/`dlsym` semantics to a first-class, context-independent API. The key problem solved: prior to 12.0, libraries using `cuModuleLoad` had to track every CUDA context and reload modules per-context, since modules were context-bound. CUDA 12.0 introduces `cuLibraryLoadFromFile` / `cuLibraryLoad` that load device code once and distribute it automatically across all current and future CUDA contexts. `cuLibraryGetKernel(lib, "name")` returns a `CUkernel` — a context-independent function handle. The CUDA driver resolves execution context at launch time, not at load time.

  Full new API:
  ```
  cuLibraryLoadFromFile(&lib, "kernel.fatbin", ...)
  cuLibraryGetKernel(&kernel_handle, lib, "my_kernel")
  // Kernel handle valid across all contexts from any library instance
  cuLaunchKernel(kernel_handle, grid, block, ...)
  ```

  The runtime API counterparts (`cudaLibraryLoad`, `cudaLibraryGetKernel`, `cudaKernel_t`) enable kernel handles to be shared between separately-linked CUDA runtime instances — a key capability for plugin architectures.

- **Key detail for libkdl:** This is the single most important NVIDIA reference for libkdl's positioning. NVIDIA has recognized the per-context management burden and built an explicit `dlopen`-for-GPU API. However, it is CUDA-only. libkdl's `kdl_load_bundle()` + `kdl_select_kernel()` are the vendor-agnostic equivalents. The absence of a HIP or OpenCL counterpart to `cuLibraryLoad` is an explicit gap.

---

### Source 3 — AMD HSA Runtime: hsa_executable and HSACO ELF Loading
- **URL:** https://github.com/HSAFoundation/HSA-Runtime-AMD
- **URL:** https://llvm.org/docs/AMDGPUUsage.html
- **URL:** https://github.com/HSAFoundation/HSA-Runtime-Reference-Source/blob/master/inc/amd_hsa_elf.h
- **Date:** Ongoing (HSA spec 1.2+, LLVM 23.0git)
- **Type:** Open-source runtime + LLVM docs + header file
- **Relevance:** 10/10
- **Novelty:** 7/10
- **Summary:** AMD's GPU runtime uses a multi-stage loading pipeline distinct from CUDA's module system.

  HSA code object loading pipeline:
  ```
  hsa_code_object_reader_create_from_memory(blob, size, &reader)
  → hsa_executable_create_alt(profile, rounding_mode, "", &executable)
  → hsa_executable_load_agent_code_object(executable, agent, reader, "", NULL)
  → hsa_executable_freeze(executable, "")
  → hsa_executable_get_symbol_by_name(executable, "my_kernel.kd", &agent, &symbol)
  → hsa_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object)
  → // kernel_object is a uint64_t handle — the kernel descriptor address in GPU VA space
  ```

  HSACO format: a standard ELF64 file (`ET_DYN` type — dynamic/shared-object semantics). The `.text` section holds GCN ISA code. Kernel descriptor structs (`amd_kernel_code_t` in V2/V3, `kernel_descriptor_t` in V4/V5) are stored as ELF symbols. ABI versioning: `ELFABIVERSION_AMDGPU_HSA_V2` through `V5` in the ELF OS/ABI field.

  The `freeze` step is critical — it corresponds to ELF relocation processing. After freeze, the executable is immutable and kernel objects (addresses in GPU VA) can be extracted.

- **Key detail for libkdl:** AMD's HSA loading API is significantly more verbose than CUDA's but exposes more of the ELF machinery. The `ET_DYN` ELF type for HSACO is not coincidental — the HSA runtime performs ELF dynamic linking (relocation patching) as part of `hsa_executable_load_agent_code_object`. The libkdl challenge: wrapping this 6-step AMD pipeline behind the same `kdl_load_bundle()` interface as CUDA's 2-step `cuLibraryLoad/GetKernel`.

---

### Source 4 — NVRTC: CUDA Runtime Compilation (Source-to-PTX JIT)
- **URL:** https://docs.nvidia.com/cuda/nvrtc/index.html
- **URL:** https://github.com/NVIDIA/jitify
- **URL:** https://saurabh-s-sawant.github.io/blog/2024/GPU-JIT/
- **Date:** Current (NVRTC 13.2); Jitify updated 2024
- **Type:** Official API docs + library + technical blog
- **Relevance:** 9/10
- **Novelty:** 7/10
- **Summary:** NVRTC is CUDA's source-level JIT: it accepts CUDA C++ as a string and outputs PTX. The PTX is then loaded via `cuModuleLoadData`. This enables a complete "no binary pre-compilation" workflow where kernel source ships with the application and is compiled on first use or on-demand.

  NVRTC pipeline:
  ```
  nvrtcCreateProgram(&prog, kernel_src, "kernel.cu", 0, NULL, NULL)
  → nvrtcCompileProgram(prog, num_opts, opts)
  → nvrtcGetPTXSize(prog, &ptx_size); nvrtcGetPTX(prog, ptx_buf)
  → cuModuleLoadData(&module, ptx_buf)  // PTX → cubin JIT happens inside the driver
  → cuModuleGetFunction(&func, module, "my_kernel")
  ```

  Jitify (NVIDIA open-source) wraps this workflow into a simple C++ header with kernel caching, template instantiation support, and automatic `#include` resolution.

- **Key detail for libkdl:** NVRTC + cuModuleLoadData is the standard pattern for ML frameworks that need to specialize kernels at runtime (e.g., Triton's CUDA backend). The PTX is the intermediate; the cubin is generated by the driver JIT. The disk cache (`~/.nv/ComputeCache`) is the GPU analog of `ld.so`'s shared library cache (`ldconfig`). AMD's equivalent is `comgr` + `hipRtcCompileProgram` (HIP RTC).

---

### Source 5 — CUDA Separate Compilation and Device Linking (cuLink API)
- **URL:** https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/
- **URL:** https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html (cuLink section)
- **Date:** 2012 (concept), 2024 (current CUDA 12.x)
- **Type:** NVIDIA Technical Blog + Driver API docs
- **Relevance:** 8/10
- **Novelty:** 6/10
- **Summary:** CUDA supports relocatable device code (RDC) compiled with `--device-c`, which produces `.rdc` object files containing unresolved device symbol references. These are linked at runtime using the `cuLink` API:

  ```
  cuLinkCreate(num_opts, opts, vals, &linkState)
  → cuLinkAddFile(linkState, CU_JIT_INPUT_PTX, "kernel1.ptx", ...)
  → cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN, cubin_blob, ...)
  → cuLinkComplete(linkState, &cubin_out, &cubin_size)  // produces final cubin
  → cuModuleLoadData(&module, cubin_out)
  ```

  Dynamic Parallelism (kernels launching kernels) requires RDC + linking against `cudadevrt` (device runtime library). NVRTC supports generating relocatable PTX for RDC scenarios.

- **Key detail for libkdl:** The cuLink API is the "ld" step for GPU device code — it resolves inter-kernel symbol references. libkdl's bundle concept implicitly handles this: a `.kdl` bundle contains already-linked, complete binaries. But for future work, libkdl could expose a linking layer for partially-compiled kernel fragments.

---

### Source 6 — LLVM/Clang Offloading Design: .llvm.offloading ELF Section and __tgt_bin_desc
- **URL:** https://clang.llvm.org/docs/OffloadingDesign.html
- **URL:** https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- **Date:** 2024–2026 (current Clang 23.0git)
- **Type:** Official Clang docs + LLVM Discourse RFC
- **Relevance:** 9/10
- **Novelty:** 8/10
- **Summary:** LLVM's unified offloading infrastructure embeds device code in host ELF using a custom section:

  - `.llvm.offloading` section (with `SHF_EXCLUDE` flag): holds the device binary, identified by magic bytes `0x10FF10AD`. Stripped from the final executable by default; extracted by the linker wrapper.
  - `omp_offloading_entries` section: holds a table of `__tgt_offload_entry` structs, one per kernel or global variable. Each entry contains the symbol address, name string, size, and flags.
  - `__tgt_bin_desc`: top-level descriptor registered with `libomptarget` via a global constructor at program start.

  Kernel dispatch flow (OpenMP offload):
  ```
  // At startup: __tgt_register_lib(&bin_desc) called by global ctor
  // At a #pragma omp target region:
  __tgt_target_kernel(loc, device_id, grid, block, kernel_entry_ptr, args)
  → libomptarget looks up kernel_entry_ptr in registered entry table
  → dispatches to appropriate plugin (cuda, amdhsa, cpu)
  ```

  The key limitation: kernels must be registered at compile time. The entry table is static. There is no runtime registration path — the `__tgt_get_kernel_handle(name)` API proposed in issue #75356 (wave-05 source) would add this.

- **Key detail for libkdl:** libkdl does not use `libomptarget` at all. It reads kernel bundles (which carry their own ELF sections and symbol tables) and dispatches via vendor driver APIs directly. The `.llvm.offloading` embedding model is relevant because libkdl bundles could potentially use the same magic-byte format for interoperability with LLVM toolchain tooling.

---

### Source 7 — LLVM Issue #75356: Name-Based Kernel Loading Gap in libomptarget
- **URL:** https://github.com/llvm/llvm-project/issues/75356
- **Date:** December 2023 — open as of April 2026
- **Type:** GitHub issue / unimplemented RFC
- **Relevance:** 10/10
- **Novelty:** 9/10
- **Summary:** The Chapel GPU support team identified that `libomptarget` cannot dispatch kernels not registered at compile time. The proposed `__tgt_get_kernel_handle(name)` + `__tgt_launch_kernel_via_handle(handle, ...)` interface was prototyped by Johannes Doerfert (LLNL) but never merged. As of April 2026 the issue is in the LLVM/Offload Development project backlog with no committed implementation.

  The root cause: `libomptarget`'s `__tgt_offload_entry` table is generated by the compiler and statically linked. There is no runtime API to extend this table or look up kernels by name without a pre-registered entry. `cuModuleGetFunction` and `hipModuleGetFunction` both support name-based lookup natively; `libomptarget` does not expose this capability upward.

- **Key detail for libkdl:** This is the most important gap reference. libkdl implements `kdl_select_kernel(bundle, "kernel_name")` — precisely the `dlsym`-style name-based lookup that LLVM's own offload runtime is missing. The LLVM issue confirms this is a recognized, unresolved gap in the ecosystem (not a problem libkdl is solving that nobody noticed).

---

### Source 8 — OpenCL / SPIR-V: clCreateProgramWithIL and Cross-Vendor Binary Loading
- **URL:** https://www.khronos.org/blog/offline-compilation-of-opencl-kernels-into-spir-v-using-open-source-tooling
- **URL:** https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html
- **URL:** https://www.intel.com/content/www/us/en/developer/articles/case-study/spir-v-default-interface-to-intel-graphics-compiler-for-opencl-workloads.html
- **Date:** 2020–2024
- **Type:** Khronos blog + spec + Intel case study
- **Relevance:** 8/10
- **Novelty:** 7/10
- **Summary:** OpenCL 2.1+ provides the most mature cross-vendor GPU binary loading API in the ecosystem:

  ```
  // Load SPIR-V binary (vendor-neutral IL)
  clCreateProgramWithIL(context, spirv_blob, spirv_size, &errcode)
  → clBuildProgram(program, 1, &device, opts, NULL, NULL)
  → clCreateKernel(program, "my_kernel", &errcode)    // name-based lookup
  → clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf)
  → clEnqueueNDRangeKernel(queue, kernel, dims, NULL, global_size, local_size, ...)
  ```

  The `clCreateProgramWithBinary` variant loads vendor-specific binaries (HSACO for AMD, PTX/cubin for NVIDIA via their OpenCL implementation) — not portable. `clCreateProgramWithIL` with SPIR-V is the portable path. SPIR-V is compiled to device-native ISA by the vendor OpenCL runtime's JIT backend.

  Intel uses SPIR-V as the exclusive interface to its OpenCL compiler (`ocloc`), bypassing the older SPIR binary format entirely.

- **Key detail for libkdl:** OpenCL/SPIR-V is the most complete prior art for cross-vendor dynamic kernel loading. libkdl's bundle format could be described as "a SPIR-V container but for heterogeneous targets that don't all have OpenCL implementations" — CPU fallback, NVIDIA cubin, AMD HSACO, all in one archive. The critical difference: OpenCL requires the full OpenCL runtime stack, which may not be present on all targets; libkdl uses vendor driver APIs directly.

---

### Source 9 — ROCm comgr: Code Object Manager for Dynamic Compilation and Loading
- **URL:** https://github.com/ROCm/ROCm-CompilerSupport
- **URL:** https://rocmdoc.readthedocs.io/en/latest/Tutorial/rocncloc.html
- **Date:** ROCm 4.x+
- **Type:** Open-source library + ROCm documentation
- **Relevance:** 8/10
- **Novelty:** 7/10
- **Summary:** AMD's `comgr` (Code Object Manager) is the AMD equivalent of NVRTC — a runtime compilation and code object manipulation library. It provides:
  - `amd_comgr_compile_source_to_relocatable()` — compiles OpenCL C or HIP source to HSACO
  - `amd_comgr_link_object_to_executable()` — links multiple code objects
  - `amd_comgr_get_data_kind()` — inspects code object format (BC, relocatable, executable, bytes)
  - Metadata queries: extract kernel names, argument info, workgroup sizes from a code object

  comgr wraps the LLVM/AMDGPU backend and provides a stable ABI for ROCm runtimes to use without linking against LLVM directly.

  Symbol resolution gap: unlike `cuModuleGetFunction`, HSA executables surface kernel objects as `uint64_t` VA addresses (kernel descriptor pointers), not named function pointers. `hsa_executable_get_symbol_by_name()` does the name-to-address resolution, but it requires the full executable freeze step first.

- **Key detail for libkdl:** comgr provides everything needed for AMD-side JIT in libkdl's future work: compile OpenCL C → HSACO → load via HSA runtime. The metadata query APIs (`amd_comgr_iterate_symbols`) are the AMD equivalent of `nm` or `objdump --syms` for kernel bundles — useful for libkdl's kernel discovery path.

---

### Source 10 — Dynamic Loading in the CUDA Runtime (cudaLibraryLoad, 2023)
- **URL:** https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/
- **Date:** 2023 (CUDA 12.0)
- **Type:** NVIDIA Technical Blog
- **Relevance:** 9/10
- **Novelty:** 8/10
- **Summary:** Companion post to Source 2, focusing on the CUDA *runtime* API (not driver) side of dynamic loading. Key additions in CUDA 12.0:
  - `cudaLibraryLoad(blob, ...)` / `cudaLibraryLoadFromFile(path, ...)` — runtime-level `dlopen`
  - `cudaLibraryGetKernel(lib, "name")` → `cudaKernel_t` — runtime-level `dlsym`
  - `cudaGetKernel(hostFuncPtr)` → `cudaKernel_t` — converts a traditional `__global__` function pointer to a kernel handle
  - Kernel handle (`cudaKernel_t`) is shareable across separately-linked CUDA runtime instances

  This matters for plugin systems: a framework can `dlopen` a .so containing CUDA kernels and receive `cudaKernel_t` handles without needing a shared CUDA runtime context.

- **Key detail for libkdl:** `cudaKernel_t` sharing across plugin `.so` boundaries is the NVIDIA-specific solution to a problem libkdl solves universally. A libkdl bundle loaded from a plugin `.so` can dispatch to any backend without the plugin knowing the backend type.

---

## Key Technical Patterns Identified

### Pattern 1: The Universal GPU Load-Then-Dispatch Pipeline

Every GPU runtime follows the same logical sequence:
```
[Binary blob] → [Load into runtime] → [Symbol resolution by name] → [Handle] → [Launch]
```

| Step | CUDA Driver | CUDA Runtime (12.0+) | HSA/ROCm | OpenCL/SPIR-V |
|------|-------------|----------------------|-----------|---------------|
| Load binary | `cuModuleLoadData(blob)` | `cudaLibraryLoad(blob)` | `hsa_executable_load_agent_code_object(exec, reader)` | `clCreateProgramWithIL(ctx, spirv, size)` |
| Compile/JIT | Driver internal | Driver internal | `hsa_executable_freeze(exec)` (relocations) | `clBuildProgram(prog, ...)` |
| Name resolution | `cuModuleGetFunction(mod, "name")` | `cudaLibraryGetKernel(lib, "name")` | `hsa_executable_get_symbol_by_name(exec, "name.kd", ...)` | `clCreateKernel(prog, "name")` |
| Handle type | `CUfunction` | `cudaKernel_t` | `uint64_t` (kernel descriptor VA) | `cl_kernel` |
| Launch | `cuLaunchKernel(func, ...)` | `cuLaunchKernel(handle, ...)` | AQL dispatch packet to HSA queue | `clEnqueueNDRangeKernel(queue, kernel, ...)` |

### Pattern 2: Static vs. Dynamic Kernel Registration

The major architectural divide:
- **CUDA/HIP driver API**: fully dynamic — any named symbol in a loaded module is callable
- **CUDA/HIP runtime API (pre-12.0)**: static — only `__global__` functions in the compilation unit are accessible
- **CUDA runtime 12.0+**: dynamic — `cudaLibraryLoad` restores dynamic capability at runtime level
- **libomptarget/OpenMP**: fully static — entry table is compiler-generated and immutable
- **OpenCL**: fully dynamic — any kernel in a program object is accessible by name

### Pattern 3: Binary Format Complexity vs. API Simplicity

| Format | Binary Type | Portability | JIT needed? |
|--------|-------------|-------------|-------------|
| CUBIN | ELF64 (NVIDIA-specific) | sm_XY only | No |
| PTX | Text IR | Any NVIDIA GPU | Yes (driver JIT) |
| FATBIN | Container (CUBIN + PTX) | All NVIDIA, forward-compat | Conditional |
| HSACO | ELF64 (AMD-specific, ET_DYN) | Target gfx only | No |
| SPIR-V | Binary IL | Any OpenCL vendor | Yes (vendor JIT) |
| .bc (LLVM) | LLVM bitcode | Any LLVM backend | Yes |

libkdl's `.kdl` bundle format corresponds to FATBIN at the concept level — a multi-target container. The difference: FATBIN is NVIDIA-only; `.kdl` carries CUBIN, HSACO, and native CPU objects in one archive.

---

## Angle Assessment

- **Relevance to libkdl:** 10/10 — This angle is the foundational technical substrate of libkdl. Every API table entry above describes something libkdl either wraps, replaces, or extends.
- **Novelty of findings:** 7/10 — The individual mechanisms (cuModuleLoad, HSA executable load, SPIR-V IL load) are well-documented. What is novel is the cross-vendor comparison table and the identification of the static-vs-dynamic divide as the core architectural tension.
- **Key gap confirmed:** The `libomptarget` name-based loading gap (Source 7 / LLVM issue #75356) is open as of April 2026. libkdl's `kdl_select_kernel(bundle, name)` is a complete implementation of the missing capability, generalized across vendors.
- **Poster contribution angle:** The table in Pattern 1 above is directly poster-worthy. It shows at a glance that every vendor has solved dynamic GPU loading independently, with incompatible APIs and handle types. libkdl provides a uniform abstraction row across this table.

---

## Connections to Prior Waves

- **Wave 05 (ld-so-analogy):** This wave provides the concrete API details supporting the conceptual analogy there. The tables here are the "evidence layer" for wave-05's claims.
- **Wave 02 (fat-binaries):** The binary format table in Pattern 3 extends wave-02's FATBIN/CUBIN focus with HSA and SPIR-V entries.
- **Wave 02 (cuda-driver-api):** That wave covers PTX JIT caching; this wave covers the broader loading pipeline from which PTX JIT is one step.
- **Wave 05 (gpu-kernel-jit):** NVRTC (Source 4) and comgr (Source 9) are the source-level JIT tools feeding into the loading pipeline described here.

---

## Sources

- [CUDA Driver API: Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
- [Load CUDA Kernel at Runtime Using CUDA Driver APIs](https://leimao.github.io/blog/CUDA-Driver-Runtime-Load-Run-Kernel/)
- [CUDA Context-Independent Module Loading](https://developer.nvidia.com/blog/cuda-context-independent-module-loading/)
- [Dynamic Loading in the CUDA Runtime](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime)
- [NVRTC: Runtime Compilation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [NVIDIA Jitify](https://github.com/NVIDIA/jitify)
- [Just-In-Time Compiled CUDA Kernel — Saurabh S. Sawant](https://saurabh-s-sawant.github.io/blog/2024/GPU-JIT/)
- [Separate Compilation and Linking of CUDA C++ Device Code](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/)
- [HSA Runtime AMD](https://github.com/HSAFoundation/HSA-Runtime-AMD)
- [LLVM AMDGPU User Guide](https://llvm.org/docs/AMDGPUUsage.html)
- [AMD HSA ELF Header](https://github.com/HSAFoundation/HSA-Runtime-Reference-Source/blob/master/inc/amd_hsa_elf.h)
- [ROCm CompilerSupport (comgr)](https://github.com/ROCm/ROCm-CompilerSupport)
- [Offloading Design & Internals — Clang](https://clang.llvm.org/docs/OffloadingDesign.html)
- [RFC: Introducing llvm-project/offload — LLVM Discourse](https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302)
- [[Offload] Name-based kernel loading — LLVM Issue #75356](https://github.com/llvm/llvm-project/issues/75356)
- [Offline Compilation of OpenCL Kernels into SPIR-V](https://www.khronos.org/blog/offline-compilation-of-opencl-kernels-into-spir-v-using-open-source-tooling)
- [The OpenCL SPIR-V Environment Specification](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html)
- [SPIR-V: Default Interface to Intel Graphics Compiler for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/case-study/spir-v-default-interface-to-intel-graphics-compiler-for-opencl-workloads.html)
- [HIP Porting Driver API](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.2/how-to/hip_porting_driver_api.html)
