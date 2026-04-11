# Wave 02: Dynamic Kernel Loading & GPU Fat Binaries
Search query: "GPU fat binary dynamic loading ELF cubin hsaco kernel image selection runtime"
Sources found: 10
Date: 2026-04-06

## Sources

### 1. Runtime Fatbin Creation Using the NVIDIA CUDA Toolkit 12.4 Compiler — NVIDIA Technical Blog
- URL: https://developer.nvidia.com/blog/runtime-fatbin-creation-using-the-nvidia-cuda-toolkit-12-4-compiler/
- Type: blog/docs
- Date: 2024-03 (CUDA 12.4 release)
- Relevance: 9/10
- Novelty: 9/10
- Summary: Introduces the `nvFatbin` library for programmatic fat binary construction at runtime — previously only possible via the `fatbinary` CLI tool. The API accepts CUBIN, PTX, and LTO-IR inputs, creates an in-memory fatbin without writing to disk or spawning subprocesses, and has no dependency on the CUDA driver (can run on systems without a GPU). Critically, this is the first official NVIDIA API enabling runtime fat binary *creation*, not just consumption.
- Key detail: Core API sequence is `nvFatbinCreate()` → `nvFatbinAddCubin()` / `nvFatbinAddPTX()` / `nvFatbinAddLTOIR()` → `nvFatbinGet()` → `nvFatbinDestroy()`. The resulting blob can be handed directly to the CUDA driver via `cuModuleLoadData()`. This is the closest NVIDIA analogy to what libkdl needs: programmatic multi-arch binary bundling.

### 2. CUDA Binary Utilities — NVIDIA Official Documentation (v13.2)
- URL: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- Type: docs
- Date: 2026-03 (current)
- Relevance: 8/10
- Novelty: 5/10
- Summary: Canonical reference for the CUDA binary ecosystem: cubin (ELF-format device code for a specific SM architecture), PTX (virtual ISA text), and fatbin (multi-arch container). The CUDA driver performs architecture selection at load time: it prefers the cubin matching the detected SM version, falls back to PTX JIT compilation if no cubin matches. Tools `cuobjdump` and `nvdisasm` can inspect and disassemble any fatbin or cubin.
- Key detail: cubin is a standard ELF with CUDA-specific sections. A fatbin wraps one or more cubins plus optional PTX. The JIT-compiled result is cached in `~/.nv/ComputeCache` (256 MiB default, 4 GiB max). Cache invalidation is automatic on driver upgrade. This two-level lookup (cubin-exact → PTX-JIT) is the canonical GPU "dynamic version selection" algorithm.

### 3. nvFatbin Library Reference — NVIDIA Official Documentation (v13.0+)
- URL: https://docs.nvidia.com/cuda/nvfatbin/index.html
- Type: docs
- Date: 2025 (v13.0 current release)
- Relevance: 9/10
- Novelty: 8/10
- Summary: Full API reference for the `nvFatbin` runtime library. The library's stated purpose is "supporting dynamic loading of the most optimized variant for a given architecture" — creating fatbins containing multiple variants of a single CUDA source. Key property: no reliance on the CUDA driver, usable on CPU-only systems. Accepts inputs from the same CUDA toolkit major version plus older inputs, enabling cross-version packaging.
- Key detail: The Fatbin Creator APIs are explicitly designed for "dynamic loading of the most optimized variant for a given architecture" — this is the exact problem libkdl solves in a vendor-agnostic way. nvFatbin only handles NVIDIA targets; libkdl is the cross-vendor generalization.

### 4. User Guide for AMDGPU Backend — LLVM Official Documentation (v23.0.0git)
- URL: https://llvm.org/docs/AMDGPUUsage.html
- Type: docs
- Date: 2025–2026 (continuously updated)
- Relevance: 8/10
- Novelty: 6/10
- Summary: Comprehensive reference for the AMDGPU LLVM backend and the HSACO (HSA Code Object) format. HSACO is a standard ELF file with vendor-specific sections: `.text` contains AMDGCN ISA instructions, `.note` section stores metadata (kernel descriptors, AMDGPU architecture flags) whose schema varies by code object version (V2 through V6). Kernel symbols are in the ELF symbol table; the HSA runtime resolves them by name.
- Key detail: HSACO versions V3+ store kernel descriptors directly in the ELF `.text` section as 64-byte structs preceding each kernel. The ELF `e_flags` field encodes the target AMDGPU architecture (e.g., `gfx1030` for RDNA2). This means the runtime can detect architecture compatibility by inspecting ELF headers alone — analogous to the ABI version check in `ld.so`. Code Object V6 (ROCm 6.x+) is the current production format.

### 5. Offloading Design & Internals — Clang Official Documentation (v23.0.0git)
- URL: https://clang.llvm.org/docs/OffloadingDesign.html
- Type: docs
- Date: 2025–2026 (continuously updated)
- Relevance: 9/10
- Novelty: 7/10
- Summary: Describes LLVM's fat object pipeline in detail. Device binaries are embedded into host objects in `.llvm.offloading` sections (magic bytes `0x10FF10AD`). The `clang-linker-wrapper` scans input objects for these sections at link time, routes device binaries to appropriate linkers, and produces `__tgt_bin_desc` runtime descriptors. At program startup, a global constructor calls `__tgt_register_lib()` to register all embedded device images with `libomptarget`. Kernel lookup currently relies on pre-registered entry tables compiled into the binary.
- Key detail: The `.llvm.offloading` section uses a "string map" format where each device image carries a triple, architecture, and offload kind tag. This is structurally identical to a fat binary directory. The `__tgt_bin_desc` descriptor passed to the runtime is effectively a device-side dynamic linking table — exactly what libkdl generalizes.

### 6. ⚙ D125165: [Clang] Introduce clang-offload-packager tool — LLVM Phabricator
- URL: https://reviews.llvm.org/D125165
- Type: PR/review
- Date: 2022-05 (landed in LLVM 15)
- Relevance: 8/10
- Novelty: 7/10
- Summary: Introduces `clang-offload-packager`, designed to replace `clang-offload-bundler` for modern GPU offloading workflows. Creates a binary package marked by `0x10FF10AD` magic bytes, followed by a version field and a compact string-map-based directory. Multiple images (one per target) are concatenated; the magic-byte delimiter allows the linker to locate all embedded offload sections even after merging during relocatable linking.
- Key detail: The binary format's magic bytes survive linker section merges — this is a key design property. The format is explicitly described as "behaves similarly to CUDA's fatbinary" but vendor-agnostic. The reason for replacing `clang-offload-bundler` is that bundler output was "not valid input to the rest of LLVM" — the packager produces proper ELF-embeddable blobs. This establishes that LLVM does not yet have a standardized cross-vendor kernel-container format, leaving an opening for libkdl.

### 7. [Offload] Name-based kernel loading — LLVM GitHub Issue #75356
- URL: https://github.com/llvm/llvm-project/issues/75356
- Type: issue/RFC
- Date: 2023-12
- Relevance: 10/10
- Novelty: 10/10
- Summary: Chapel language team (Johannes Doerfert et al.) identifies a critical gap in LLVM's offload runtime: `libomptarget` only supports kernels registered in a static compile-time table, but dynamic languages need to find kernels by name at runtime (like `cuModuleGetFunction` or `hipModuleGetFunction`). The proposed solution introduces two new APIs: `__tgt_get_kernel_handle(name)` and `__tgt_launch_kernel_via_handle(handle, ...)`. This is the closest official LLVM equivalent to `dlsym()` for GPU kernels.
- Key detail: The existing offload runtime fundamentally cannot support dynamic kernel discovery because it assumes the kernel table is fixed at compile time. The proposed `__tgt_get_kernel_handle()` API is a `dlsym()`-for-GPUs equivalent. libkdl is a complete implementation of this missing concept — not just a single API, but a full linker layer. This issue validates that the problem libkdl solves is recognized as a gap in the LLVM ecosystem.

### 8. HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs — arXiv 2506.15993
- URL: https://arxiv.org/html/2506.15993v1
- Type: paper
- Date: 2025-06
- Relevance: 9/10
- Novelty: 9/10
- Summary: Proposes `hetGPU`, a system for vendor-agnostic GPU binary portability via a portable intermediate representation (`hetIR`) — a virtual GPU ISA with SPMD semantics, abstract memory ops, and virtualized special functions. At load time, the runtime JIT-compiles hetIR to the detected hardware: NVIDIA via PTX→SASS, AMD via SPIR-V→GCN, Intel via SPIR-V→Level Zero, Tenstorrent via Metalium assembly. The system caches translated kernels to amortize JIT cost. First-run overhead: 50–200 ms per kernel; sustained execution overhead: 5–15% vs. native.
- Key detail: hetGPU implements the ld.so analogy at the IR level: device detection → cache lookup → backend selection → lazy JIT compilation → cached execution. While not named a "linker," it performs exactly the operations libkdl targets: selecting the right code for the detected hardware from a multi-target binary. The 5–15% overhead from abstract IR translation is the cost libkdl avoids by storing pre-compiled native variants directly (no IR-to-ISA translation at dispatch time).

### 9. CUDA Pro Tip: Understand Fat Binaries and JIT Caching — NVIDIA Technical Blog
- URL: https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
- Type: blog
- Date: 2014 (classic reference, still accurate)
- Relevance: 7/10
- Novelty: 4/10
- Summary: Foundational explanation of CUDA's two-level architecture selection: exact cubin match first, then PTX JIT with persistent caching. Compilation command `nvcc -arch=compute_10 -code=compute_10,sm_10,sm_13` produces a fat binary with PTX plus two cubins. The CUDA driver selects the "most appropriate translation" at launch — first exact SM match, then PTX fallback. Cache stored in `~/.nv/ComputeCache` with 256 MiB default.
- Key detail: The fat binary selection algorithm has not changed since 2014: exact-match cubin preferred, PTX-JIT as fallback. This two-level lookup is what libkdl should implement for its vendor-native backends — with the addition of a cross-vendor dispatch layer on top.

### 10. Dynamic Loading in the CUDA Runtime — NVIDIA Technical Blog
- URL: https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime
- Type: blog/docs
- Date: 2023
- Relevance: 8/10
- Novelty: 8/10
- Summary: Introduces `cudaLibraryLoad()` and `cudaLibraryGetKernel()` APIs in CUDA 12.x, enabling explicit control over when GPU device code is loaded, on-the-fly compilation and linking, and kernel handle sharing between CUDA runtime instances. The API mirrors Linux's `dlopen()`/`dlsym()` semantics at the CUDA runtime level — you load a "library" (a fatbin or cubin), get a kernel handle by name, then launch it. Kernel handles can be shared across CUDA runtime instances in the same process.
- Key detail: `cudaLibraryLoad()` + `cudaLibraryGetKernel()` is CUDA's answer to "dlopen for GPU kernels" — but it only works for CUDA and requires explicit integration. libkdl provides the same capability vendor-agnostically, with the runtime dispatch table transparent to the kernel caller. The existence of this API confirms that NVIDIA itself recognizes the dlopen-for-GPU pattern as the right abstraction.

---

## Angle Assessment

**Angle:** Dynamic Kernel Loading & GPU Fat Binaries
**Coverage:** Excellent — all major dimensions covered (CUDA fatbin, HSACO ELF, LLVM packaging toolchain, runtime loading APIs, name-based kernel lookup gap, academic hetGPU paper)

**Novelty for libkdl:**
- LLVM GitHub issue #75356 is the most directly validating finding: the LLVM project explicitly lacks a `dlsym()`-for-GPUs equivalent and there is no clean solution in the current offload runtime. libkdl fills this gap.
- `nvFatbin` (CUDA 12.4) and `cudaLibraryLoad()` (CUDA 12.x) show that NVIDIA recognizes the runtime kernel loading problem and is building vendor-specific solutions. libkdl is the vendor-agnostic generalization.
- hetGPU (arXiv 2025) is the closest academic prior art — but it operates at the IR translation level (10–200 ms overhead), while libkdl stores pre-compiled native variants and dispatches in microseconds.

**Key gaps libkdl addresses:**
1. No vendor-agnostic fat binary format that spans CUDA cubins, AMD HSACO, and CPU ELF in a single container.
2. No cross-vendor `dlopen()`/`dlsym()` API for GPU kernels — each vendor has its own (`cuModuleGetFunction`, `hipModuleGetFunction`, `hsa_executable_get_symbol`).
3. LLVM's offload runtime (`libomptarget`) statically registers kernels at compile time; dynamic name-based lookup is an open GitHub issue with no merged solution.
4. No architecture-aware kernel selection library analogous to `ld.so`'s hardware-capability directory scheme.

**Recommended follow-up angles:**
- How does `libhsa-runtime64.so`'s `hsa_executable_get_symbol_by_name()` work internally — is it walking an ELF symbol table? If so, that's the exact mechanism libkdl should expose cross-vendor.
- Is there any existing use of `LD_PRELOAD`-style kernel interposition for GPU kernels? (NVBit, ROCm's `rocprofv2` — possible angle for libkdl's "preload" extension.)
- CUDA Graphs vs. explicit kernel handles: does `cudaLibraryGetKernel()` work inside a CUDA Graph? This affects libkdl's dispatch API design.
