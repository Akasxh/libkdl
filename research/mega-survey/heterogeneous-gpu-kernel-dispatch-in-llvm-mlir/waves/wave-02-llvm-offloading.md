# Wave 02: LLVM Unified Offloading Infrastructure
Search query: LLVM offloading runtime libomptarget unified offloading multi-target GPU plugin
Sources found: 10
Date: 2026-04-06

## Sources

### 1. Offloading Design & Internals — Clang Official Documentation
- URL: https://clang.llvm.org/docs/OffloadingDesign.html
- Type: docs
- Date: 2024–2025 (continuously updated, current as of LLVM 23.x)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Canonical description of LLVM's new offloading driver pipeline. Device images are compiled to fat objects by embedding them in the `.llvm.offloading` section with magic bytes `0x10FF10AD`. The clang-linker-wrapper scans input objects for this section, extracts device files, routes them to appropriate device link jobs (including LTO for bitcode), and wraps the final linked device image with loading symbols. At startup, a `__tgt_bin_desc` descriptor is passed to libomptarget via `__tgt_register_lib()` through a global constructor, making all embedded images available to the runtime.
- Key detail: The new driver moves almost all device link work out of the compiler driver and into clang-linker-wrapper at link time. This means the embedded image format (`.llvm.offloading` section) is the stable interface between compilation and runtime — a critical integration point for a KDL that wants to pre-compile and package multi-target kernel images.

### 2. [RFC] Introducing `llvm-project/offload` — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- Type: RFC
- Date: October 2023
- Relevance: 10/10
- Novelty: 9/10
- Summary: Foundational RFC by Johannes Doerfert (LLNL) proposing that libomptarget be moved, renamed, and evolved into a standalone `llvm/offload` sub-project serving as a shared offloading runtime for the entire LLVM ecosystem — covering OpenMP, CUDA, HIP, SYCL, AI accelerators, FPGAs, and remote machines. The stated goals are: unified user experience, reduced code duplication across vendor runtimes, interoperability between offloading models, and broader portability. The RFC received unanimous positive feedback from major stakeholders including AMD, NVIDIA, Intel, and LLNL.
- Key detail: The explicit framing is that every hardware vendor currently builds their own LLVM offloading runtime downstream — this RFC proposes to upstream that work into a shared foundation. This is the most direct statement in the LLVM ecosystem that a general-purpose GPU dispatch runtime is needed and desired, making it the strongest architectural parallel to libkdl's thesis.

### 3. [RFC] `llvm-project/offload` Roadmap — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-llvm-project-offload-roadmap/75611
- Type: RFC
- Date: November 2023
- Relevance: 9/10
- Novelty: 8/10
- Summary: Follow-up RFC detailing the concrete roadmap for the llvm/offload sub-project. Proposes a new stable C API (`liboffload`) as the public interface, with the existing OpenMP runtime API preserved as a functional wrapper over it. The roadmap includes: (1) moving libomptarget source to `offload/`, (2) introducing a TableGen-based API definition system (offload-tblgen), (3) implementing a minimal but complete device/program/kernel/memory API, and (4) enabling SYCL and Unified Runtime interop via the same plugin layer. The design is explicitly modeled on oneAPI Unified Runtime's API structure.
- Key detail: The roadmap draws a hard line between the OpenMP-semantic API (libomptarget, which requires registered images and target regions) and the new lower-level API (liboffload, which takes arbitrary binary blobs and kernel names). This two-layer design is exactly the split that a KDL would need: liboffload gives raw dispatch capability while libomptarget preserves OpenMP correctness.

### 4. Draft PR #122106: Implement the Remaining Initial Offload API — GitHub
- URL: https://github.com/llvm/llvm-project/pull/122106
- Type: PR
- Date: January 2025
- Relevance: 10/10
- Novelty: 10/10
- Summary: This PR implements the complete initial version of the liboffload C API, intended to be "usable for simple offloading programs." The API surface covers: `olMemAlloc`/`olMemFree` (host/device/shared allocation), `olCreateQueue`/`olFinishQueue` (async stream abstraction), `olEnqueueDataWrite`/`olEnqueueDataRead`/`olEnqueueDataCopy` (data movement), `olCreateProgram`/`olReleaseProgram` (binary blob → program), `olCreateKernel`/`olSetKernelArgValue`/`olSetKernelArgsData` (kernel extraction by name + argument binding), and `olEnqueueKernelLaunch` (grid dispatch). All functions are implemented as thin wrappers over the existing NextGen plugin infrastructure.
- Key detail: `olCreateProgram` takes an arbitrary binary blob (ELF, PTX, HSACO, SPIR-V) and creates a program object — no OpenMP pragma or target region required. `olCreateKernel` looks up a kernel symbol by name within that program. This is functionally equivalent to what libkdl does with its kernel registry, but as an official LLVM-maintained API. The API is explicitly marked unstable and evolving, but the capability is there today.

### 5. PR #118614: Introduce offload-tblgen and Initial New API Implementation — GitHub
- URL: https://github.com/llvm/llvm-project/pull/118614
- Type: PR
- Date: November 2024 (merged)
- Relevance: 8/10
- Novelty: 8/10
- Summary: Introduces offload-tblgen, a TableGen-based tool for generating the liboffload C API headers, validation code, and print infrastructure from `.td` descriptor files. The API is defined in `offload/liboffload/API/` as TableGen records, enabling consistent code generation of the function signatures, parameter validation, and debug printing. The design is described as "loosely based on equivalent tooling in Unified Runtime." The initial commit implements only device enumeration/querying to validate the infrastructure — kernel operations came in subsequent PRs.
- Key detail: The TableGen approach means the API is machine-generated and can be extended by adding new `.td` records. The API lives in `offload/liboffload/API/` with `OffloadAPI.td` as the root file. This is a production-grade design pattern, not a prototype — it signals that LLVM intends liboffload to be a maintained, versioned API surface.

### 6. PR #120145: Add Initial SPIR-V Support to clang-linker-wrapper — GitHub
- URL: https://github.com/llvm/llvm-project/pull/120145
- Type: PR
- Date: December 2024
- Relevance: 8/10
- Novelty: 9/10
- Summary: Extends clang-linker-wrapper to support OpenMP offloading to generic SPIR-V (initial target: Intel GPUs via `spirv64-intel-unknown`). Since no production SPIR-V linker exists, the implementation manually constructs an ELF binary containing the SPIR-V offloading image in the format expected by liboffload's plugin infrastructure. This demonstrates that the `.llvm.offloading` / liboffload path is intentionally target-agnostic — adding a new target is a matter of implementing a plugin and constructing the appropriate binary wrapper.
- Key detail: The SPIR-V path confirms that the offloading binary format (ELF + `.llvm.offloading` section) is the vendor-neutral kernel packaging format LLVM is converging on. A KDL could emit/consume this format to interoperate with any target that liboffload supports — without needing to know vendor APIs directly.

### 7. [Offload] New Subproject + Pending Move of libomptarget — LLVM Discourse
- URL: https://discourse.llvm.org/t/offload-new-subproject-pending-move-of-libomptarget/78185
- Type: RFC/announcement
- Date: April 2024
- Relevance: 8/10
- Novelty: 7/10
- Summary: Formal announcement that `offload/` has been created as a new top-level LLVM sub-project directory, and that the migration of libomptarget source from `openmp/libomptarget/` to `offload/` was imminent. This is the execution step following the October 2023 RFC. As of this posting, the plugins-nextgen code and the DeviceRTL were being moved into the new location. Confirms the directory structure: `offload/libomptarget/`, `offload/liboffload/`, `offload/plugins-nextgen/`, `offload/DeviceRTL/`.
- Key detail: The actual code migration from `openmp/libomptarget` to `offload/` completed in mid-2024. Any KDL built on this infrastructure must target `offload/` paths, not the legacy `openmp/libomptarget/` location. The migration was non-trivial — standalone build support required multiple backport PRs (e.g., #118643) to keep release branches functional.

### 8. NextGen Plugin Infrastructure — LLVM/OpenMP Runtime Docs
- URL: https://openmp.llvm.org/design/Runtimes.html
- Type: docs
- Date: 2024 (continuously updated)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Describes the NextGen plugin architecture that underlies both libomptarget and liboffload. All plugins inherit from a `GenericPluginTy` C++ base class that provides: device enumeration, image registration, kernel loading, memory allocation, data transfer, and stream/event management. Vendor-specific plugins (CUDA, AMDGPU, GenericELF64) override only the hardware-specific portions. The `GenericPluginTy` abstractions directly correspond to the liboffload C API operations: `ol_program` = registered image, `ol_kernel` = loaded kernel function, `ol_queue` = plugin stream.
- Key detail: The NextGen plugin for CUDA uses `dlopen(libcuda.so)` at runtime (not link-time) — identical to what libkdl does for vendor isolation. The AMDGPU plugin similarly `dlopen`s `libhsa-runtime64.so`. This means both libomptarget and liboffload are already architected as late-binding dispatch systems, exactly matching libkdl's design premise.

### 9. Issue #149284: Allow liboffload CUDA Plugin to Accept PTX Binaries — GitHub
- URL: https://github.com/llvm/llvm-project/issues/149284
- Type: issue
- Date: 2025
- Relevance: 8/10
- Novelty: 9/10
- Summary: Feature request (with input from Joseph Huber/AMD) to allow `olCreateProgram()` to accept PTX text directly, eliminating the need for users to manually call `cuLinkCreate`/`cuLinkAddData`/`cuLinkComplete` before invoking liboffload. Currently users must JIT-compile PTX to CUBIN themselves and pass the CUBIN to `olCreateProgram()`. Huber notes "We already do this kind of stuff in the JIT engine" — confirming liboffload has internal JIT plumbing that is not yet exposed at the API level.
- Key detail: This issue reveals a gap in the current liboffload API: it handles pre-compiled binaries but does not yet expose a JIT compilation path at the `olCreateProgram` layer. A KDL that wants to support PTX kernels at runtime would either need to implement its own cuLinkXxx wrapper (as shown in the issue) or wait for this feature. This is an active development gap as of early 2026.

### 10. LLVM Looking to Better Collaborate Around Common AI/GPU/FPGA Offloading — Phoronix
- URL: https://www.phoronix.com/news/LLVM-Offload-More-Common
- Type: news/analysis
- Date: October 2023
- Relevance: 7/10
- Novelty: 5/10
- Summary: Phoronix coverage of the llvm/offload RFC, confirming broad industry interest. Notes that "hardware vendors are relying on LLVM when it comes to offloading compute work to GPUs, AI accelerators, FPGAs" but each builds their own downstream runtime, creating massive code duplication. The article contextualizes the community response as "basically unanimous support." Useful as a third-party validation of the ecosystem consensus that the existing fragmentation is a recognized problem.
- Key detail: The framing in this article mirrors libkdl's problem statement almost verbatim: fragmented vendor runtimes, no shared dispatch layer, poor portability. The difference is that llvm/offload approaches this from the compiler/language-runtime side, while libkdl approaches it from the binary/dynamic-linking side — these are complementary, not competing, solutions.

---

## Angle Assessment

**How mature is LLVM's unified offloading infrastructure today?**

The `offload/` sub-project is real and merged into LLVM mainline as of mid-2024. The liboffload C API (`ol`-prefixed functions) is functional but explicitly unstable. As of January 2025 (PR #122106), the API covers the complete basic dispatch loop: allocate memory, create program from binary blob, extract kernel by name, bind arguments, enqueue launch on a queue, synchronize. This is sufficient for a KDL prototype integration.

**Plugin architecture alignment with libkdl:**

The NextGen plugin system (`GenericPluginTy` + vendor-specific subclasses, loaded via `dlopen`) is structurally identical to libkdl's plugin model. Both use late-binding vendor library discovery, abstract over device enumeration, and route kernel launches through a unified interface. The key difference: libomptarget/liboffload requires kernels to be packaged in the `.llvm.offloading` ELF format, whereas libkdl uses its own KDL binary format. Interop is feasible — libkdl could emit `.llvm.offloading`-formatted objects that liboffload can load.

**Gaps relevant to KDL:**

1. No runtime kernel symbol enumeration — `olCreateKernel` requires knowing the kernel name in advance (no `dl_iterate_phdr`-style discovery over kernel ELFs).
2. No PTX JIT path at `olCreateProgram` layer yet (Issue #149284) — pre-compiled binaries only.
3. No multi-version selection logic — liboffload does not have a policy engine for choosing among multiple kernels of the same name compiled for different architectures/microarchitectures.
4. API is explicitly unstable — building stable KDL on top requires either vendoring or version pinning.

**Strategic relevance to libkdl:**

liboffload represents the strongest possible validation of libkdl's core thesis: the LLVM community itself has identified the need for a general-purpose, OpenMP-agnostic GPU kernel dispatch runtime and is building one. libkdl's contributions (multi-version dispatch, kernel symbol resolution, dlopen-style kernel loading semantics) fill the gaps that liboffload explicitly does not address. The poster should position libkdl as a user-space dispatch policy layer that sits above liboffload's mechanism layer — complementary infrastructure, not a competing runtime.
