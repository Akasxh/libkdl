# Wave 06 — ROCm Code Object Format: Binary Internals and Runtime Loading

**Survey:** Heterogeneous GPU Kernel Dispatch in LLVM/MLIR
**Angle:** rocm-hip-code-object-loading
**Search query:** "ROCm HIP code object loading runtime kernel dispatch amdgpu ELF"
**Priority source types:** official docs, LLVM source, ROCR runtime source, Clang docs
**Date:** 2026-04-06

---

> **Scope note:** `wave-02-rocm-hip.md` covered the HIP dispatch *API stack* (hipModuleLoad → CLR → ROCR → KFD) with high depth. This wave goes one level lower: the *binary format internals* — ELF section layout, kernel descriptor wire format, AQL packet construction from descriptor fields, the ROCR loader's ELF parsing logic, and the in-flight 2025 changes to the `.amdhsa.kd` section. These layers are the "object file format" that any kernel dynamic linker must understand to parse AMD binaries.

---

## Sources

### Source 1: LLVM AMDGPU Backend User Guide — Code Object Format Reference
- **URL:** https://llvm.org/docs/AMDGPUUsage.html
- **Date:** Current (LLVM 23.0.0git, confirmed 2026)
- **Type:** Official LLVM documentation
- **Relevance:** 10/10
- **Novelty:** 7/10 (authoritative, not novel)
- **Summary:** Canonical specification of AMD code object versions V2 through V6. The code object is a standard ELF relocatable (or shared) object. Key ELF sections: `.text` (GCN/RDNA machine code), `.rodata` (read-only data including kernel descriptors in V3–V5), `.amdhsa.kd` (new dedicated section for kernel descriptors, see Source 3), `.note` (ISA version metadata, feature flags, and code object version as ELF note records). Code Object V5 is the default since ROCm 5.x (`-mcode-object-version=5`); V6 adds generic-processor versioning for forward-compatible deployment.
- **Key technical detail:** The `amdhsa_kernel_descriptor` struct (64 bytes, 64-byte aligned) is the fundamental dispatch primitive. It contains: `group_segment_fixed_size` (LDS bytes), `private_segment_fixed_size` (scratch bytes per work-item), `kernarg_size`, `kernel_code_entry_byte_offset` (byte offset from descriptor start to first instruction — enables position-independent kernel placement), and packed bit fields for SGPR/VGPR setup. The CP hardware reads this descriptor directly when processing an AQL kernel dispatch packet; the descriptor address (not the instruction address) is what goes into the AQL packet's `kernel_object` field.

---

### Source 2: ROCm Code Object Format Specification — ReadTheDocs
- **URL:** https://rocmdoc.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Codeobj-format.html
- **Date:** Legacy (v2-era), but foundational
- **Type:** Official AMD SDK documentation
- **Relevance:** 9/10
- **Novelty:** 6/10
- **Summary:** Specifies the `amd_kernel_code_t` 256-byte structure used in Code Object V2. Provides exact bit-field layouts for `compute_pgm_rsrc1_t` and `compute_pgm_rsrc2_t` — the hardware register images that CP copies into SH_PGM_RSRC1/2 before wavefront launch. Describes the AQL dispatch sequence: (1) obtain AQL queue pointer, (2) obtain `amd_kernel_code_t` pointer from the loaded code object, (3) allocate kernarg segment via `hsa_memory_allocate`, (4) write grid/workgroup dims and kernarg pointer into an AQL packet, (5) atomically set the packet format field from `INVALID` to `KERNEL_DISPATCH`, (6) ring the hardware doorbell.
- **Key technical detail:** The 64-byte alignment requirement on kernel descriptors is a hardware constraint, not a software convention. The CP's packet processor reads the descriptor at the address stored in the AQL packet's `kernel_object` field using a direct memory fetch; misalignment causes a CP fault (hardware exception), not a software error. This is directly analogous to how the OS loader must page-align `.text` segments — the constraint is architectural. For libkdl: any in-memory kernel descriptor table must preserve this alignment guarantee when copying descriptors between code objects.

---

### Source 3: LLVM PR #122930 — Emit AMDHSA Kernel Descriptors to `.amdhsa.kd` Section
- **URL:** https://github.com/llvm/llvm-project/pull/122930
- **Date:** January–February 2025 (open, stalled)
- **Type:** LLVM upstream pull request
- **Relevance:** 9/10
- **Novelty:** 10/10
- **Summary:** Proposes moving kernel descriptor emission from `.rodata` (where they were interspersed with other symbols) into a dedicated `.amdhsa.kd` section with `SHF_ALLOC | SHF_GNU_RETAIN` flags. Implementation in `AMDGPUTargetELFStreamer.cpp` adds section push/pop and explicit 64-byte alignment. The `SHF_GNU_RETAIN` flag is critical: it prevents `--gc-sections` from discarding descriptor symbols that appear unreferenced from a static-linking perspective.
- **Key technical detail:** The stated runtime optimization is that the ROCm runtime could `memcpy` the entire `.amdhsa.kd` section to a pinned GPU-accessible buffer in a single operation rather than iterating symbol table entries filtered by the `.kd` name suffix. However, as of February 2025 the PR is **stalled due to kernel launch failures** — the ROCR runtime's loader (`executable.cpp`) still expects descriptor symbols in `.rodata`, and the runtime modifications to consume `.amdhsa.kd` directly have not been merged. This is an active binary format evolution with a broken compatibility bridge: the LLVM backend is ahead of the runtime. For libkdl: any parser of AMD code objects must handle *both* layouts (descriptors in `.rodata` and descriptors in `.amdhsa.kd`) since both will exist in the wild during the transition period.

---

### Source 4: ROCR Runtime Loader — `executable.cpp` Source Analysis
- **URL:** https://github.com/ROCm/ROCR-Runtime/blob/master/src/loader/executable.cpp
- **Date:** Current (ROCm 7.x codebase)
- **Type:** Open-source runtime implementation
- **Relevance:** 10/10
- **Novelty:** 9/10
- **Summary:** The ROCR loader's `LoadCodeObject` function is the HSA-level implementation of `hsa_executable_load_agent_code_object`. It parses the ELF via the `AmdHsaCode` class, validates code object version (V1–V6), checks ISA compatibility against the target agent, and then processes the symbol table. Kernel symbols (ELF symbol type `STT_AMDGPU_HSA_KERNEL`) are extracted and registered into the executable's symbol table for later lookup via `hsa_executable_get_symbol`.
- **Key technical detail (ELF symbol filtering logic):**
  ```cpp
  for (size_t i = 0; i < code->SymbolCount(); ++i) {
      if (majorVersion >= 2 &&
          code->GetSymbol(i)->elfSym()->type() != STT_AMDGPU_HSA_KERNEL &&
          code->GetSymbol(i)->elfSym()->binding() == STB_LOCAL)
          continue;
      status = LoadSymbol(agent, code->GetSymbol(i), majorVersion);
  }
  ```
  This means that in V2+ code objects, only symbols with type `STT_AMDGPU_HSA_KERNEL` (value `10`, AMD-specific ELF symbol type extension) or non-LOCAL binding are loaded into the executable. The kernel descriptor address is resolved during `LoadDefinitionSymbol` and cached in a `KernelSymbol` object. Later queries via `KernelSymbol::GetInfo` return pre-parsed fields (kernarg size, group segment size, private segment size, wavefront size) without re-reading the ELF. This pre-caching is the HSA-level analog of the ROCm 7.1 "faster kernel-metadata retrieval" optimization.

---

### Source 5: HSA AQL Queue and Dispatch Packet Architecture — ROCR Runtime DeepWiki
- **URL:** https://deepwiki.com/ROCm/ROCR-Runtime
- **Date:** Current (ROCm 7.x)
- **Type:** Synthesized documentation / codebase analysis
- **Relevance:** 9/10
- **Novelty:** 7/10
- **Summary:** Describes the full AQL queue lifecycle: `hsa_queue_create()` → `AMD::AqlQueue` instantiation → ring buffer allocation → KFD driver registration → hardware doorbell configuration → scratch memory allocation. The `HSA_PACKET_TYPE_KERNEL_DISPATCH` packet (64 bytes) contains `kernel_object` (pointer to kernel descriptor), `kernarg_address` (pointer to argument buffer), `grid_size_{x,y,z}`, `workgroup_size_{x,y,z}`, and `group_segment_size` / `private_segment_size` overrides. The GPU's Command Processor (CP) polls the ring buffer and executes packets via microcode that reads the kernel descriptor, configures SPI/SH registers, and launches wavefronts.
- **Key technical detail:** The dispatch sequence from user-space perspective:
  1. `hsa_executable_get_symbol_by_name()` → `hsa_executable_symbol_get_info(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT)` → 64-bit address of the kernel descriptor
  2. Allocate kernarg region: `hsa_memory_allocate()` in `HSA_AMD_SEGMENT_GLOBAL` with `HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT` flag
  3. Copy kernel arguments into the kernarg buffer (layout defined by code object metadata, not by a runtime struct)
  4. Write AQL dispatch packet to the queue ring buffer at `queue->base_address + (queue->write_index % queue->size) * 64`
  5. Atomically store `HSA_PACKET_TYPE_KERNEL_DISPATCH` into `packet->header` (releases packet to CP)
  6. Write `queue->write_index + 1` to the hardware doorbell register

  This is a **fully userspace-driven dispatch** — no syscall after queue creation. The doorbell write is a memory-mapped I/O operation into a PCIe BAR region, not an ioctl. This is architecturally distinct from OpenCL's `clEnqueueNDRangeKernel` which goes through a driver daemon for every dispatch.

---

### Source 6: HIP Fat Binary Format — Clang HIP Support Documentation
- **URL:** https://clang.llvm.org/docs/HIPSupport.html
- **Date:** Current (Clang 23.0.0git)
- **Type:** Official Clang documentation
- **Relevance:** 8/10
- **Novelty:** 6/10
- **Summary:** The fat binary container (`.hip_fatbin` section in the host ELF, global symbol `__hip_fatbin`) is created by `clang-offload-bundler` (legacy) or `clang-offload-packager` (new path). Each bundle entry is identified by a target triple of the form `hip-amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-`. The runtime iterates entries at `__hipRegisterFatBinary` time, matches against the current device's ISA via `hipGetDeviceProperties`, and extracts the matching HSACO blob via `hiprtcGetCode`-compatible deserialization.
- **Key technical detail:** Non-RDC compilation produces one fully-linked HSACO per target per translation unit; RDC mode produces relocatable device code (`.bc` bitcode) per TU which is device-linked across TUs in a separate step, then packaged. The SPIR-V path (`--offload-arch=amdgcnspirv`) produces a single `spirv64-amd-amdhsa` blob instead of per-gfx ISA — this blob is **not** a valid HSACO and requires JIT recompilation at runtime before use (see Source 7).

---

### Source 7: ROCm SPIR-V for AMDGCN — Portable Generic Code Objects
- **URL:** https://rocm.docs.amd.com/projects/llvm-project/en/develop/conceptual/spirv.html
- **Date:** Current (ROCm 7.x / LLVM 20+)
- **Type:** Official ROCm/LLVM documentation
- **Relevance:** 8/10
- **Novelty:** 9/10
- **Summary:** `--offload-arch=amdgcnspirv` targets the triple `spirv64-amd-amdhsa` — a portable AMD-flavored SPIR-V that is architecture-agnostic at compile time. At runtime, when this SPIR-V code object is loaded (via `hipModuleLoadData` or `__hipRegisterFatBinary`), the ROCm runtime invokes the Comgr JIT compilation path to lower it to the concrete GFX target of the active device. This is the AMD equivalent of CUDA's PTX virtual ISA JIT path.
- **Key technical detail:** The SPIR-V path has specific compile-time limitations: GPU-specific preprocessor macros (`__gfx906__` etc.) are undefined, wavefront size is not a compile-time constant (must use `__builtin_amdgcn_wavefront_size()` at runtime), and `__builtin_amdgcn_processor_is(name)` provides runtime ISA queries. The JIT compilation cost (Comgr `AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE`) is amortized if the runtime caches the lowered HSACO — which the current ROCm runtime does in a per-process code cache, but not persistently across processes (no persistent kernel cache file like CUDA's `.cubin` cache). For libkdl: a SPIR-V code object is a viable **single-binary portability vehicle** for AMD kernels, but requires the libkdl implementation to detect the `spirv64-amd-amdhsa` code object type and invoke the Comgr JIT path, similar to how a generic linker handles architecture-independent bitcode.

---

### Source 8: HIP Module API — Driver API Port Reference
- **URL:** https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.2/how-to/hip_porting_driver_api.html
- **Date:** ROCm 6.2 (2024)
- **Type:** Official ROCm documentation
- **Relevance:** 8/10
- **Novelty:** 5/10
- **Summary:** Maps CUDA driver API functions to HIP equivalents. `hipModuleLoadData(module, image)` accepts a pointer to a code object binary in memory. `hipModuleLoadDataEx(module, image, numOptions, options, optionValues)` accepts JIT compilation options but on the HIP-Clang path these options are **silently ignored** (compilation already happened ahead-of-time). `hipModuleGetFunction(function, module, name)` retrieves a kernel handle by name. `hipModuleLaunchKernel` dispatches with explicit grid/block dimensions.
- **Key technical detail:** The `image` parameter to `hipModuleLoadData` must point to a valid HSACO ELF binary (for native path) or a fat binary container (for multi-arch path). The function does **not** accept raw LLVM bitcode or SPIR-V — those require the HIP RTC path (`hiprtcCompileProgram` → `hiprtcGetCode` → `hipModuleLoadData`). This is a format validation boundary: the module loader performs ELF magic byte validation and rejects non-ELF inputs. For libkdl: the AMD backend's `load_module` function must be prepared to handle both raw HSACO (ELF, magic `\x7fELF`) and fat binary containers (different magic) as distinct input types with different processing paths.

---

### Source 9: ROCm 7.2 Release Notes — Dynamic Code Object Loading Fixes
- **URL:** https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html
- **Date:** 2025
- **Type:** Official release notes
- **Relevance:** 7/10
- **Novelty:** 7/10
- **Summary:** ROCm 7.2 explicitly fixes "issues arising during dynamic code object loading" — confirming that the dynamic loading path (`hipModuleLoadData` called at arbitrary runtime points, not just at program startup) had known bugs in ROCm 7.1 and earlier. Also introduces `MultiKernelDispatch` thread trace support across all ASICs for profiling multiple kernel dispatches within a single trace session.
- **Key technical detail:** The dynamic loading fix is directly relevant to libkdl's design: if libkdl implements deferred or on-demand kernel loading (analogous to `dlopen` called mid-execution rather than at startup), it would have triggered the now-fixed bugs on ROCm 7.1. ROCm 7.2 is the minimum safe version for dynamic code object loading in production. The `MultiKernelDispatch` trace support enables profiling libkdl's dispatch overhead accurately — previously, back-to-back dispatches within a single trace session would produce merged timing data.

---

### Source 10: Clang Offload Bundler Format — Bundle Version 3
- **URL:** https://clang.llvm.org/docs/ClangOffloadBundler.html
- **Date:** Current (Clang 23.0.0git)
- **Type:** Official Clang documentation
- **Relevance:** 7/10
- **Novelty:** 6/10
- **Summary:** The clang-offload-bundler packs multiple device code objects into a single bundle container. Version 3 is now the default format; version 2 can be forced via `COMPRESSED_BUNDLE_FORMAT_VERSION=2` for compatibility with older HIP runtimes. Target identifiers in bundles use the format `<offload-kind>-<triple>` (e.g., `hip-amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-`). The `HIPAMD` vs `HIP` offload kind distinction is now purely historical — the ABI version is embedded directly in the code object ELF metadata, not in the bundle entry identifier.
- **Key technical detail:** Bundle format version 3 adds compression (zlib/zstd) of individual code object payloads within the bundle. The runtime must decompress before parsing the ELF. For libkdl's fat binary parser: after extracting a bundle entry by matching the target identifier, the extracted bytes may be compressed and must be decompressed before they constitute a valid ELF that can be passed to `hipModuleLoadData` or parsed for kernel symbols. This is an under-documented format detail that would cause silent failures (corrupt ELF magic bytes) without explicit compression handling.

---

## Angle Assessment

**Coverage:** High for format internals. This wave covers the binary format layer that wave-02 omitted: ELF section layout, kernel descriptor wire format, the `.amdhsa.kd` section evolution, ROCR loader ELF parsing logic, AQL packet construction, userspace doorbell mechanism, SPIR-V portable code objects, and fat binary compression. The chain from compiler output to CP hardware execution is now fully traced.

**Relevance to libkdl:** Critical. A kernel dynamic linker for AMD GPUs must operate at exactly this layer — parsing ELF code objects, extracting kernel descriptors, managing the AQL dispatch interface, and handling the multiple binary input formats (HSACO, fat binary v2/v3, SPIR-V). The format is specified but in active evolution (PR #122930 `.amdhsa.kd` transition).

**ld.so analogy mapping (format-level):**

| ld.so binary concept | AMD code object equivalent |
|---|---|
| ELF `.text` section | `.text` section in HSACO (GCN/RDNA machine code) |
| ELF `.rodata` section | `.rodata` + `.amdhsa.kd` (kernel descriptors — in transition) |
| ELF `.dynamic` section | `.note` (ISA version, code object version, feature flags) |
| Symbol table (`STT_FUNC`) | Symbol table with `STT_AMDGPU_HSA_KERNEL` entries |
| PLT entry (call stub) | AQL dispatch packet (64-byte, written to queue ring buffer) |
| GOT entry (data address) | `kernel_object` field in AQL packet (descriptor address) |
| `DT_SONAME` (library identity) | GFX ISA target ID (e.g., `gfx906:sramecc-:xnack-`) |
| `.so` fat binary (multi-arch) | clang-offload-bundler container (hip fat binary) |
| Position-independent code | `kernel_code_entry_byte_offset` in descriptor (offset, not absolute address) |
| Shared library lazy binding | Not native; SPIR-V path approximates it (JIT on first load) |
| Compressed debug info | Bundle v3 compressed payloads (zlib/zstd per entry) |

**Key risks and gaps found:**

1. **Dual descriptor layout problem (`.rodata` vs `.amdhsa.kd`):** PR #122930 is stalled because the ROCR runtime doesn't yet consume `.amdhsa.kd`. The LLVM backend and runtime are out of sync. Any libkdl code object parser deployed in 2025-2026 will encounter both layouts and must handle both. The detection heuristic: check for the `.amdhsa.kd` section; if absent, fall back to scanning `.rodata` for symbols with the `.kd` suffix.

2. **Fat binary compression in bundle v3:** Bundle format v3 compresses individual code object payloads. Documentation does not prominently advertise this. A parser treating bundle entries as raw ELF bytes after extraction will silently receive corrupt data for v3 bundles. Must check the bundle version header and decompress before ELF parsing.

3. **SPIR-V code objects require Comgr JIT — no fallback:** If a fat binary contains only `spirv64-amd-amdhsa` entries (no per-gfx HSACO), loading requires Comgr's JIT path. If Comgr is unavailable or the SPIR-V payload is malformed, there is no graceful degradation to a pre-compiled fallback. libkdl should treat SPIR-V-only bundles as requiring an explicit "compile first" step with a distinct error code, not a transparent load-time operation.

4. **`STT_AMDGPU_HSA_KERNEL` is an AMD ELF extension:** Standard ELF tooling (readelf, objdump without AMDGPU support) will not classify these as function symbols. A libkdl ELF parser using a generic ELF library must explicitly handle `STT_AMDGPU_HSA_KERNEL = 10` as the AMD kernel symbol type. Using a generic `STT_FUNC` filter will miss all kernels.

5. **64-byte descriptor alignment is a hard CP constraint:** Any in-memory kernel descriptor table built by libkdl (e.g., for caching descriptors from multiple code objects into a single dispatch-ready buffer) must preserve 64-byte alignment per descriptor. `malloc`/`new` do not guarantee this; `posix_memalign(64)` or `std::aligned_alloc(64, size)` are required.

6. **Doorbell write is MMIO, not a syscall:** Dispatch is fully userspace after queue creation. This means libkdl's AMD dispatch path has the same zero-syscall property as the native HIP path — the overhead is purely the AQL packet write and MMIO doorbell, both measurable in tens of nanoseconds. The latency gap to CUDA is in the HIP CLR validation layers above this, not in the raw dispatch mechanism.

**Suggested new research angles from this investigation:**

- **Comgr API as libkdl introspection backend:** `amd_comgr_action_*` can enumerate kernels, extract metadata, and JIT-compile SPIR-V to HSACO — a complete introspection + compilation primitive for an AMD backend without depending on ROCR's executable loader.
- **AQL packet batching:** Multiple kernel dispatches can be written to the queue ring buffer before a single doorbell write. This batched submission path eliminates per-kernel MMIO overhead — a potential libkdl optimization for workloads submitting many small kernels.
- **Feature flag versioning in target IDs:** `gfx906:sramecc+:xnack-` vs. `gfx906:sramecc-:xnack+` are ABI-incompatible. libkdl's AMD selector must match target IDs including feature flags, not just the base gfx number — a matching problem with combinatorial state space for AMD targets.
- **Code Object V6 generic processors:** V6's versioned generic processor mechanism (analogous to PTX virtual ISA) may eventually allow a single code object to run on multiple GFX generations, collapsing the multi-gfx fat binary into a single portable binary. Tracking the V6 adoption timeline in the ROCm ecosystem.

---

## Sources (Inline Reference List)

- [AMDGPU Backend User Guide — LLVM 23.0.0git](https://llvm.org/docs/AMDGPUUsage.html)
- [ROCm Code Object Format — ReadTheDocs](https://rocmdoc.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Codeobj-format.html)
- [LLVM PR #122930: Emit AMDHSA kernel descriptors to `.amdhsa.kd`](https://github.com/llvm/llvm-project/pull/122930)
- [ROCR Runtime Loader — executable.cpp](https://github.com/ROCm/ROCR-Runtime/blob/master/src/loader/executable.cpp)
- [ROCR Runtime Architecture — DeepWiki](https://deepwiki.com/ROCm/ROCR-Runtime)
- [HIP Kernel Execution and Modules — DeepWiki](https://deepwiki.com/ROCm/rocm-systems/2.3-hip-kernel-execution)
- [ROCm SPIR-V Support for AMDGCN — ROCm Docs](https://rocm.docs.amd.com/projects/llvm-project/en/develop/conceptual/spirv.html)
- [HIP Support — Clang 23.0.0git Documentation](https://clang.llvm.org/docs/HIPSupport.html)
- [Clang Offload Bundler — Clang 23.0.0git Documentation](https://clang.llvm.org/docs/ClangOffloadBundler.html)
- [Porting CUDA Driver API — HIP 6.2 Documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.2/how-to/hip_porting_driver_api.html)
- [HIP Compilers — HIP 7.2 Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html)
- [ROCm 7.2.0 Release Notes — ROCm Documentation](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)
- [ROCR Runtime API — ROCm Docs](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/api.html)
- [HSA Runtime AMD — HSAFoundation GitHub](https://github.com/HSAFoundation/HSA-Runtime-AMD)
