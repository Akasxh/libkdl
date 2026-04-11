# Wave 02: LLVM Offload Runtime Architecture
Search query: "LLVM libomptarget offload runtime plugin architecture GPU dispatch"
Sources found: 9
Date: 2026-04-06

Note: This file complements wave-02-llvm-offloading.md (which covers RFCs, roadmap PRs, and the liboffload C API surface). This wave focuses on: plugin runtime internals (GenericPluginTy/NextGen), the clang-linker-wrapper link-time pipeline, name-based kernel loading issues, cross-language consumers (Chapel, Fortran, OpenACC), and the 2025 DevMtg technical state-of-the-art talk.

## Sources

### 1. Clang Offloading Design & Internals — Clang 23.0.0git Documentation
- URL: https://clang.llvm.org/docs/OffloadingDesign.html
- Type: docs
- Date: 2024–2026 (continuously maintained, current as of LLVM 23.x)
- Relevance: 9/10
- Novelty: 5/10
- Summary: The canonical reference for the new unified offloading driver pipeline. Covers the full compilation model: device images are compiled to fat objects that embed device binaries in the `.llvm.offloading` ELF section (magic bytes `0x10FF10AD`), extracted and linked by clang-linker-wrapper at link time, then wrapped with global constructor symbols that call `__tgt_register_lib()` at program startup to hand the `__tgt_bin_desc` descriptor to libomptarget. The document distinguishes the old multi-pass compilation model (one compilation per target, separate device object files) from the new fat-object model (single compilation, embedded device data).
- Key detail: The stable interface between compile time and runtime is the `.llvm.offloading` binary section format, not any public C API. A runtime like libkdl that consumes pre-compiled GPU binaries must understand this format — or reuse clang-linker-wrapper's extraction logic — to interoperate with standard LLVM-compiled outputs.

### 2. Clang Linker Wrapper — Clang 23.0.0git Documentation
- URL: https://clang.llvm.org/docs/ClangLinkerWrapper.html
- Type: docs
- Date: 2024–2026 (continuously maintained)
- Relevance: 8/10
- Novelty: 7/10
- Summary: Documents clang-linker-wrapper as the unified link-time tool that: (1) scans input objects for `.llvm.offloading` sections, (2) extracts embedded device images (ELF/PTX/HSACO/SPIR-V), (3) invokes target-specific device linkers (lld for amdgpu/nvptx, llvm-link for bitcode LTO), (4) wraps the final linked device image with the `__tgt_bin_desc` registration symbols, and (5) passes all device objects to the host linker. The tool replaces the older `clang-offload-wrapper` tool. As of LLVM 19, clang-linker-wrapper is the default for all GPU languages (CUDA, HIP, OpenMP) after PR #84420 merged.
- Key detail: The wrapping step creates `__tgt_offload_entry` structures that enumerate all kernel entry points by name and address within the device image. This per-kernel name table is the data structure that libomptarget uses for kernel dispatch. Understanding how clang-linker-wrapper populates this table is prerequisite to implementing libkdl's kernel registry in a compatible format.

### 3. LLVM/OpenMP Runtimes — NextGen Plugin Documentation
- URL: https://openmp.llvm.org/design/Runtimes.html
- Type: docs
- Date: 2024–2026 (continuously maintained)
- Relevance: 9/10
- Novelty: 6/10
- Summary: The definitive technical reference for the NextGen plugin infrastructure underlying both libomptarget and liboffload. The plugin system is composed of C++ abstract base classes: `GenericPluginTy` (per-vendor plugin singleton, handles device enumeration), `GenericDeviceTy` (per-device state, owns streams/events/memory pools), `GenericKernelTy` (per-kernel metadata and launch configuration), and `GenericImageTy` (loaded device binary). The CUDA plugin `dlopen`s `libcuda.so` at runtime if not found at build time; the AMDGPU plugin `dlopen`s `libhsa-runtime64.so`. Both plugins implement asynchronous kernel launch via a stream/event abstraction with the same interface.
- Key detail: `GenericKernelTy` stores kernel metadata including: number of threads, grid dimensions, dynamic shared memory requirements, and the kernel function handle obtained from the vendor API (e.g., `CUfunction` from `cuModuleGetFunction`, `hipFunction_t` from `hipModuleGetFunction`). The kernel is identified within a device image by its name string — exactly the name-based lookup model libkdl's registry uses. The plugin layer is where the abstraction barrier between LLVM's runtime model and vendor driver APIs actually lives.

### 4. [Offload] Name-based Kernel Loading — GitHub Issue #75356
- URL: https://github.com/llvm/llvm-project/issues/75356
- Type: issue
- Date: December 13, 2023
- Relevance: 10/10
- Novelty: 10/10
- Summary: Issue opened by Brad Chamberlain (Chapel language team) reporting that Chapel GPU support could not use libomptarget/llvm-offload because the runtime assumes device binaries contain a compiler-generated table of kernel entry points (`omp_offloading_entries`). Chapel's compiler does not generate this table; instead the Chapel runtime performs name-based kernel lookup through driver APIs (`cuModuleGetFunction`, `hipModuleGetFunction`). The issue requests that llvm/offload expose an interface for name-based kernel extraction from arbitrary device binaries — without requiring an entry table. Joseph Huber responded positively, noting this use case is a design goal of liboffload's `olCreateKernel(program, "kernel_name")` interface.
- Key detail: This issue is the clearest public articulation of the mismatch between compiler-generated kernel dispatch (entry-table-based, assumes the compiler knows all kernels at link time) and runtime kernel dispatch (name-based, allows the runtime to select kernels dynamically). libkdl's multi-version dispatch model has exactly the same requirement: it must look up kernels by name from pre-compiled binaries, not from compiler-generated entry tables. This issue validates the design and confirms it is a recognized gap in LLVM's current infrastructure.

### 5. [Offload] Develop a New API for the "Plugins" — GitHub Issue #79304
- URL: https://github.com/llvm/llvm-project/issues/79304
- Type: issue
- Date: January 24, 2024
- Relevance: 9/10
- Novelty: 9/10
- Summary: Issue by Joseph Huber (AMD) formally proposing to rewrite and export the LLVM offload plugin interface as a stable C API. The stated problem: "The plugins abstract over vendor-dependent libraries and provide a common interface used by OpenMP and other languages to provide language-specific features, but they are currently not exported and the API is ill-defined." The proposal outlines a minimal `plugin.h` API surface covering: device enumeration, image loading, kernel lookup by name, argument setting, kernel launch, and synchronization. This is the issue that drove the `offload-tblgen`/liboffload C API work (PR #118614, PR #122106).
- Key detail: The issue explicitly states the plugin API is "not exported" in its current form — the `GenericPluginTy` C++ interface is internal-only. The entire liboffload `ol`-prefixed C API exists precisely to solve this problem. A libkdl integration that wants to call into LLVM's offload stack must use the `ol*` C API, not the internal C++ plugin classes directly. This architectural decision has direct implications for libkdl's integration path.

### 6. [Offload] Provide a Kernel Library Useable by the Offload Runtime — PR #104168
- URL: https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg474856.html
- Type: PR (mailing list)
- Date: September 2024
- Relevance: 8/10
- Novelty: 9/10
- Summary: PR proposing a "kernel library" component for the offload runtime — a set of pre-compiled GPU utility kernels (memory operations, atomic helpers, device printf) compiled for all supported targets and embedded directly into the offload runtime rather than compiled per-application. The implementation compiles these kernels as LLVM GPU programs and links them into libomptarget.so. This shows that the offload runtime itself follows the same binary-embedding pattern it exposes to user code: kernel objects stored as device ELF sections, extracted and loaded by the plugin at device init time.
- Key detail: The kernel library pattern (pre-compiled kernels embedded in a runtime library, loaded on-demand per device) is architecturally identical to what libkdl implements as a "kernel bundle" — a multi-target archive of pre-compiled kernels. The fact that LLVM's own runtime uses this pattern internally validates the design and suggests a kernel bundle format compatible with the offload runtime's internal loading mechanism could enable deep integration.

### 7. Implementing OpenMP Offload Support in the AMD Next Generation Fortran Compiler — SC'25 Workshop Paper
- URL: https://dl.acm.org/doi/10.1145/3731599.3767478
- Type: paper (SC'25 workshop)
- Date: November 2025
- Relevance: 7/10
- Novelty: 8/10
- Summary: SC'25 workshop paper (AMD team) describing AMD Flang's integration with LLVM's offload infrastructure for OpenMP GPU dispatch. The compiler emits the same MLIR `omp` dialect as Clang, lowered through the OpenMPIRBuilder to device IR, then through the same clang-linker-wrapper fat-object pipeline. Key shared infrastructure reused: LLVM metadata generation, code generation for OpenMP reductions, the unified offloading driver. Performance on AMD MI250x and MI300x reported as comparable to ROCm's classic Flang for representative HPC workloads.
- Key detail: This paper confirms that the LLVM offload infrastructure now handles three production languages (C, C++, Fortran) through a single shared runtime path. The convergence means libkdl can target a single runtime interface (the `.llvm.offloading` format + liboffload) and reach code compiled from all three languages. For a CERN/HPC use case where legacy Fortran HPC kernels coexist with C++ ML kernels, this is a meaningful portability point.

### 8. RFC: Implementation of OpenACC 3.3 for Offload in Clang — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-implementation-of-openacc-3-3-for-offload-in-clang/74405
- Type: RFC
- Date: October 2023
- Relevance: 7/10
- Novelty: 8/10
- Summary: RFC proposing OpenACC 3.3 support in Clang/Flang targeting the offload runtime. The key architectural decision: OpenACC uses `FortranOffloadRuntime` (a thin F18 runtime library) as a wrapper over `liboffload`, with OpenACC directives lowered through MLIR's `acc` dialect → LLVM IR → same fat-object pipeline. This means OpenACC-accelerated code will use the same `__tgt_register_lib` / plugin dispatch path as OpenMP target regions. As of LLVM 20, the F18 Flang docs confirm `FortranOffloadRuntime` depends on `liboffload` and links via `-fopenacc`/`-fopenmp`.
- Key detail: OpenACC joining the liboffload ecosystem means the plugin interface now serves OpenMP, SYCL, CUDA, HIP, and OpenACC — five distinct GPU programming models through one runtime layer. libkdl, positioned above liboffload, would automatically gain compatibility with all five without per-model changes. This is the strongest concrete argument for the "policy above mechanism" architecture.

### 9. The LLVM Offloading Infrastructure — Joseph Huber, LLVM DevMtg 2025 Technical Talk
- URL: https://llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf
- Type: talk slides (LLVM DevMtg 2025)
- Date: October 2025
- Relevance: 10/10
- Novelty: 10/10
- Summary: Joseph Huber's (AMD) comprehensive technical talk at the 2025 LLVM Developers' Meeting on the state and direction of LLVM's offloading infrastructure. Covers the full stack from clang-linker-wrapper (link-time fat-object creation) through the NextGen plugin layer (GenericPluginTy/GenericDeviceTy) to liboffload's `ol`-prefixed C API. Key points: (1) liboffload is now the recommended interface for non-OpenMP GPU dispatch; (2) the plugin API redesign (Issue #79304, PRs #118614/#122106) is production-ready for CUDA and AMDGPU; (3) PTX JIT support in `olCreateProgram` is in progress (Issue #149284); (4) the "not-compiler runtime" use case — user-space applications that want GPU dispatch without a compiler-managed runtime — is an explicit first-class goal. The talk frames LLVM's offload stack as "ld.so for GPU code" in the same metaphor used by libkdl.
- Key detail: The "ld.so for GPU code" framing in Huber's 2025 talk is the same metaphor used in the libkdl project description. This is not a conflict — it is strong community validation. However, Huber's stack stops at loading and launching: it does not implement capability-based selection among multiple compiled variants of the same kernel. The talk's roadmap does not include multi-version dispatch policy. libkdl's contribution is unambiguously additive to what this talk describes.

---

## Angle Assessment

- Coverage: Deep coverage of the plugin architecture runtime internals, link-time pipeline, name-based loading gap, and cross-language consumers. Complements wave-02-llvm-offloading.md (which covers the RFC/API design layer) by going further into the implementation and operational details.

- Surprise findings: Two unexpected items: (1) OpenACC joining liboffload via `FortranOffloadRuntime` means the runtime now unifies five GPU programming models — much broader reach than the original OpenMP + CUDA framing; (2) Joseph Huber explicitly used the "ld.so for GPU code" metaphor in his 2025 DevMtg talk — the same framing as libkdl. This is direct community confirmation of the problem framing, but also means libkdl must differentiate clearly: LLVM's version does loading/launch, libkdl adds selection policy.

- Gaps: (1) No analysis yet of how the `omp_offloading_entries` table format compares to libkdl's kernel registry format in detail — a format compatibility analysis would clarify the integration path; (2) No coverage of the OMPT (OpenMP Tools Interface) device profiling layer — relevant for instrumentation of dispatch overhead; (3) The PR #104168 kernel library pattern (embedded runtime kernels) needs a deeper read to determine if the binary format is exactly compatible with libkdl bundles.

- Suggested follow-up angles:
  1. `omp_offloading_entries` vs libkdl registry format — binary compatibility analysis
  2. OMPT device profiling — observability of kernel dispatch events for overhead measurement
  3. LLVM GPU libc (`libgpu`) — Huber's 2023 DevMtg LibC-for-GPUs talk shows the same multi-target precompiled-binary distribution model that libkdl uses; examining it may reveal a more mature precedent for the bundle format
  4. `offload/liboffload/API/README.md` on `llvmorg-21.1.3` — check if API stability status changed between LLVM 19 (unstable) and LLVM 21
  5. CUPTI callbacks issue #85770 — LLVM 18+ OpenMP target offloading breaks NVIDIA profiling tools; relevant to libkdl's interaction with profilers when built on top of liboffload
