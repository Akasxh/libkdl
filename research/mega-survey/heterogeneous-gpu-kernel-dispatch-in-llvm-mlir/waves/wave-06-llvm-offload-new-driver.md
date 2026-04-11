# Wave 06: LLVM New Offload Driver RFC and Toolchain Redesign

**Search angle:** llvm-offload-new-driver-rfc
**Search queries:**
- "LLVM new offload driver clang-linker-wrapper unified offloading RFC 2024 2025"
- "LLVM OffloadBinary format clang-offload-packager heterogeneous GPU dispatch"
- "LLVM offload toolchain redesign liboffload runtime 2024 2025 2026"
- "Joseph Huber LLVM offloading infrastructure 2025 devmtg slides liboffload plugin architecture"
- "LLVM offload name-based kernel loading issue 75356 dynamic kernel dispatch"
- "GPU Offloading Workshop LLVM 2025 slides liboffload SPIR-V Intel AMD NVIDIA"
- "LLVM RFC introducing llvm-project offload subproject 74302 governance 2023 2024"

**Sources found:** 10
**Date:** 2026-04-06

---

## Sources

### 1. Offloading Design & Internals — Clang 23.0.0 Official Documentation
- URL: https://clang.llvm.org/docs/OffloadingDesign.html
- Type: docs (canonical, continuously updated)
- Date: 2024–2026 (current as of LLVM 23.x)
- Relevance: 10/10
- Novelty: 5/10 (canonical reference, partially covered in wave-02; this wave extracts toolchain-side details not previously documented)
- Summary: The definitive architectural description of LLVM's new offloading driver. The compilation pipeline has five stages: (1) host compilation producing bitcode with offloading metadata, (2) device compilation using exported symbols from host, (3) fat object creation embedding device images in `.llvm.offloading` ELF sections, (4) device linking via clang-linker-wrapper at link time, (5) runtime registration through global constructors invoking `__tgt_register_lib()`. The new driver consolidates device link work that was previously scattered across multiple compiler driver invocations into a single clang-linker-wrapper pass at link time. Device images are described by three nested structs: `__tgt_offload_entry` (per-kernel metadata with name, addr, size, flags), `__tgt_device_image` (wraps one linked binary with ImageStart/ImageEnd pointers and associated entry table), and `__tgt_bin_desc` (root descriptor carrying all device images + entry tables for all target architectures). The `addr` field in `__tgt_offload_entry` is null at link time and filled by the runtime during image registration — this is the dynamic binding step.
- Key detail: The `.llvm.offloading` ELF section is the **stable, vendor-neutral interface** between compilation and runtime in the new driver model. Any tool that can produce or consume this section participates in the LLVM offload toolchain. For libkdl, emitting kernels in this format enables transparent interoperability with every runtime that speaks the new driver protocol.

### 2. [RFC] Use the New Offloading Driver for CUDA and HIP by Default — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-use-the-new-offloding-driver-for-cuda-and-hip-compilation-by-default/77468
- Type: RFC
- Date: 2024 (transitional)
- Relevance: 9/10
- Novelty: 7/10
- Summary: This RFC proposed making `--offload-new-driver` the default for CUDA and HIP, completing the transition that had already happened for OpenMP. The new driver had been the default for OpenMP offloading for some time prior. The RFC motivated the change by listing new driver advantages over the legacy path: support for device-side LTO, static libraries containing device code, Windows support, compatible with standard host linkers, and eliminating the fragile `clang-cuda-link` intermediate step. The discussion concluded with PR #84420 moving HIP and PR #122312 moving CUDA to the new driver default. As of LLVM 20 (early 2025), both CUDA and HIP use the new driver by default — the old driver path is now the legacy opt-out.
- Key detail: The switch is significant for libkdl: any LLVM 20+ build of a CUDA or HIP program now produces fat objects with `.llvm.offloading` sections by default. This means the OffloadBinary format is no longer opt-in experimental infrastructure — it is the default output format for CUDA/HIP/OpenMP compilation in current LLVM.

### 3. [clang][llvm] Move HIP and CUDA to New Driver by Default (PR #84420) — LLVM Mailing List
- URL: https://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20240826/614432.html
- Type: PR (merged 2024)
- Date: August 2024
- Relevance: 8/10
- Novelty: 7/10
- Summary: The concrete merge of the new-driver-by-default transition for HIP. The PR updates tests to account for the changed default. The patch summary confirms that with this change, "the new offloading driver provides a unified interface between the different offloading languages" for all three major GPU programming models (CUDA, HIP, OpenMP) simultaneously. The PR also confirms that the legacy driver path remains available via `--no-offload-new-driver` for downstream consumers that have not yet migrated.
- Key detail: The transition is complete as of late 2024. LLVM's toolchain now has a single unified driver path producing `.llvm.offloading` artifacts for all three major GPU programming models. This unification is a prerequisite for libkdl to treat multi-vendor kernel packages uniformly — the compilation artifacts from CUDA, HIP, and OpenMP kernels are now structurally identical containers.

### 4. [RFC] Introducing `llvm-project/offload` — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- Type: RFC
- Date: October 22, 2023
- Relevance: 10/10
- Novelty: 6/10 (covered in wave-02; cited here for governance and toolchain-side details)
- Summary: The founding RFC for the `offload/` sub-project by Johannes Doerfert (LLNL/AMD). Key governance outcome: regular biweekly coordination meetings established January 24, 2024 (alternating with OpenMP in LLVM meeting). The RFC explicitly names the target community: OpenMP, CUDA, HIP, SYCL, AI accelerators, FPGAs, and "remote threads or machines." The stated design principle is that liboffload should be a **mechanism library** — providing raw dispatch capability — while language runtimes (libomptarget, CUDA runtime, HIP runtime) provide the policy and semantic layer above it. This mechanism/policy split is formalized in the RFC text.
- Key detail: The governance structure established here (biweekly meetings, Discourse coordination) means the LLVM offload community has an active forum for upstream contributions. A libkdl-style selection policy extension to liboffload (e.g., a `rankImage` virtual hook) would have a clear submission path through this community channel.

### 5. [RFC] `llvm-project/offload` Roadmap — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-llvm-project-offload-roadmap/75611
- Type: RFC
- Date: November 2023
- Relevance: 9/10
- Novelty: 6/10 (partially covered in wave-02; this wave extracts toolchain-pipeline details)
- Summary: Follow-up RFC detailing the concrete implementation roadmap. Three-layer architecture: (1) `liboffload` — new stable C API for raw dispatch (`ol`-prefixed functions, TableGen-generated), (2) language bindings — libomptarget wraps liboffload for OpenMP semantics, SYCL/Unified Runtime adapters planned, (3) plugins — vendor-specific backends (CUDA, AMDGPU, Level Zero) implementing `GenericPluginTy`. The roadmap explicitly defers multi-version selection to "future work" — the immediate goal was minimum viable dispatch (load binary, launch kernel by name, memory ops). The roadmap also specifies that the compilation-side toolchain (clang-linker-wrapper, clang-offload-packager, llvm-offload-binary) feeds into the runtime via the `.llvm.offloading` ELF section as the stable interface.
- Key detail: The roadmap formally separates the **toolchain side** (producing `.llvm.offloading` artifacts) from the **runtime side** (consuming them via liboffload). libkdl operates at the intersection: it needs to both produce multi-target OffloadBinary containers (toolchain-side) and implement selection policy when consuming them (runtime-side). The roadmap's toolchain/runtime split is the natural seam where libkdl inserts.

### 6. Clang Linker Wrapper — Clang 23.0.0 Documentation
- URL: https://clang.llvm.org/docs/ClangLinkerWrapper.html
- Type: docs
- Date: 2024–2026 (current)
- Relevance: 9/10
- Novelty: 8/10
- Summary: Full specification of the clang-linker-wrapper tool. The pipeline: (1) scan all linker inputs for `.llvm.offloading` sections containing `llvm-offload-binary`-formatted data, (2) extract and group device files by target triple + architecture, (3) invoke the appropriate device linker (lld for AMDGPU/NVPTX, ld for host-targeting device code), (4) for bitcode architectures (AMDGPU, SPIR-V), run LTO passes before device linking, (5) wrap the final linked device image in a new object file containing: (a) the image data as a static symbol, and (b) global constructor/destructor functions that call `__tgt_register_lib()`/`__tgt_unregister_lib()`, (6) pass the wrapped object to the system linker for final host linking. The tool handles multi-target compilation transparently: when multiple device images exist for different architectures (e.g., sm_80 + sm_89 + gfx1030), all are wrapped into the same descriptor. The `--relocatable` mode enables eager device code extraction for distributable static libraries.
- Key detail: The `generic` architecture wildcard in clang-linker-wrapper permits a single device image to be linked with any device code sharing the same target triple. This is the toolchain-level analogue of libkdl's "capability contract" — a coarse compatibility filter applied at device link time rather than at runtime. The difference: clang-linker-wrapper's generic match is binary (match/no-match), while libkdl's capability contracts express continuous compatibility scores.

### 7. clang-offload-packager and OffloadBinary Format — Clang 20.1.0 Documentation + OffloadBinary.h
- URL: https://releases.llvm.org/20.1.0/tools/clang/docs/ClangOffloadPackager.html
- URL: https://llvm.org/docs/CommandGuide/llvm-offload-binary.html
- URL: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/OffloadBinary.h
- Type: docs + source header
- Date: 2024–2026 (LLVM 20/23)
- Relevance: 10/10
- Novelty: 9/10
- Summary: Full OffloadBinary format specification from primary source. The format is defined in `llvm/include/llvm/Object/OffloadBinary.h` as `llvm::object::OffloadBinary`. Binary layout (version 2): 4-byte magic `{0x10, 0xFF, 0x10, 0xAD}`, 32-bit version (currently 2), 64-bit total size, 64-bit entry block offset, 64-bit entry count. Each `Entry` struct holds: `ImageKind` (enum: None, Object, Bitcode, CUBIN, Fatbinary, PTX, SPIRV), `OffloadKind` (bitmask: None=0, OpenMP=bit0, CUDA=bit1, HIP=bit2, SYCL=bit3), 64-bit flags, offset+size to string metadata block, offset+size to binary image payload. The string metadata block is a key-value string table; standard keys include `triple` and `arch`. The `OffloadFile` wrapper class provides target identification via (triple, arch) pairing and supports deep copying. Version 2 adds explicit value-size tracking in `StringEntry`, enabling binary (non-null-terminated) metadata values. Multiple OffloadBinary records are concatenated in the `.llvm.offloading` section — each self-describes its size, so iterating requires no external index. The `create()` static method parses a buffer and optionally selects a specific image by index (version 2+).
- Key detail: The `OffloadKind` bitmask allows a single image to be tagged as valid for multiple offloading models simultaneously (e.g., OpenMP+HIP for an HIP kernel with OpenMP interop wrappers). The string metadata StringMap is extensible — any arbitrary key-value pair can be embedded. libkdl's capability contracts (e.g., `"min_sm": "89"`, `"requires_tensor_core": "1"`) could be embedded directly in this string table without format changes. The `OffloadBinary` C++ class is the correct API surface for a libkdl tool that needs to pack multi-target kernel bundles.

### 8. [Driver][clang-linker-wrapper] Add Initial Support for OpenMP Offloading to Generic SPIR-V (PR #120145) — GitHub
- URL: https://github.com/llvm/llvm-project/pull/120145
- Type: PR (merged December 2024)
- Date: December 2024
- Relevance: 8/10
- Novelty: 8/10
- Summary: This PR demonstrates the extensibility of the new offload driver to non-NVIDIA/AMD targets. The implementation adds `spirv64-intel-unknown` as a new target triple, introduces `SPIRVOpenMPToolChain` extending the existing SPIR-V toolchain, and — because no production SPIR-V linker exists — manually constructs an ELF binary wrapping the SPIR-V offloading image in the format expected by liboffload plugins. The binary is then embedded in the `.llvm.offloading` section via the standard path. The SPIR-V approach confirms that the OffloadBinary/`.llvm.offloading` path is target-agnostic by design: adding support for a new architecture requires only (a) a plugin implementing `GenericPluginTy` on the runtime side, and (b) a toolchain producing a properly-wrapped binary on the compilation side.
- Key detail: The ELF-wrapping approach used in this PR is exactly the mechanism libkdl's offline compiler could use to produce SPIR-V kernel packages: wrap a SPIR-V binary in an ELF container, embed it with OffloadBinary metadata (`ImageKind=SPIRV`, `triple=spirv64-intel-unknown`), and the existing liboffload Level Zero plugin can load it. No new format required — libkdl's cross-vendor packaging goal is achievable within the existing OffloadBinary container.

### 9. [Offload] Name-Based Kernel Loading (Issue #75356) — GitHub
- URL: https://github.com/llvm/llvm-project/issues/75356
- Type: issue
- Date: November 2023
- Relevance: 9/10
- Novelty: 8/10
- Summary: Opened by the Chapel language team, this issue documents the fundamental mismatch between the LLVM offload driver's assumption (binaries contain a compile-time-generated kernel registry table) and dynamic kernel dispatch requirements (kernels discovered at runtime by name). Chapel's GPU runtime calls `cuModuleGetFunction`/`hipModuleGetFunction` directly, bypassing the `__tgt_offload_entry` table mechanism entirely. Johannes Doerfert (LLNL) provided a proof-of-concept proposing two new API functions: `__tgt_get_kernel_handle(image, "kernel_name") -> KernelHandle` and `__tgt_launch_kernel_via_handle(handle, args...)`. The design intention is to "launch all kernels in the device image, regardless if they were registered via our current mechanism or not" — decoupling runtime kernel dispatch from compile-time kernel registration. As of April 2026, this issue is still open and the PoC has not been upstreamed.
- Key detail: This issue is the clearest articulation in the LLVM community of the exact gap that libkdl fills: the compile-time registration model breaks dynamic dispatch. The proposed `__tgt_get_kernel_handle` interface is functionally equivalent to libkdl's `kdl_kernel_lookup(bundle, "kernel_name")` — but libkdl has an implementation while LLVM's PoC does not yet have an upstream PR. This is a strong differentiator for the Dublin poster.

### 10. The LLVM Offloading Infrastructure — Joseph Huber (AMD), LLVM Dev Meeting 2025
- URL: https://llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf
- URL: https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832
- Type: conference talk + workshop slides
- Date: October 2025 (LLVM Developers' Meeting)
- Relevance: 9/10
- Novelty: 9/10
- Summary: Joseph Huber's (AMD) presentation at the 2025 LLVM Developers' Meeting covers the current state of the LLVM offloading infrastructure and the `llvm/offload` sub-project. A half-day GPU/Offloading Workshop at the same meeting (October 27, 8:30am–12:30pm) covered the complete state of liboffload across CUDA/AMDGPU/SPIR-V targets, with a dedicated session titled "LLVM Offloading — Where are We Going?" The workshop confirmed: (1) liboffload `ol*` API is the production dispatch path as of LLVM 20, (2) SPIR-V/Intel GPU support via liboffload is actively landing (PR #120145 series), (3) the biweekly offload coordination meetings continue, and (4) the OffloadBinary multi-image container format (PR #185404, #186088) was identified as a key forward-looking capability. The presentation also covered the C/C++ toolchain for GPUs (LLVM DevMtg 2024, huber.pdf) — enabling GPU-native C++ without offloading language extensions.
- Key detail: The 2025 workshop explicitly asked "Where are we going?" on offloading — and the answer includes multi-target OffloadBinary containers but leaves selection policy as open work. Citing this talk in the Dublin poster positions libkdl as a concrete response to this stated community need, with direct lineage to the LLVM offload community's own roadmap discussions.

---

## Angle Assessment

### What is the current state of LLVM's offload toolchain redesign?

**Status as of April 2026: Complete and default.**

The new offload driver is no longer experimental. As of LLVM 20 (early 2025):
- OpenMP has used the new driver by default since LLVM 14
- HIP migrated to new driver default via PR #84420 (August 2024)
- CUDA migrated to new driver default via PR #122312 (December 2024/LLVM 20)

The unified pipeline is: `clang` (compilation) → fat object with `.llvm.offloading` section → `clang-linker-wrapper` (device linking) → `__tgt_register_lib()` at startup → liboffload/libomptarget (dispatch). The `clang-offload-packager` and `llvm-offload-binary` tools create and inspect `.llvm.offloading` sections independently of the compiler.

The old driver path (`--no-offload-new-driver`) still exists for legacy compatibility but is not the default for any language.

### How does the OffloadBinary format work?

**Precise format (from `OffloadBinary.h`, version 2):**

```
Header:
  magic[4]        = { 0x10, 0xFF, 0x10, 0xAD }
  version         = 2 (uint32_t)
  size            = total binary size (uint64_t)
  entry_offset    = offset to Entry array (uint64_t)
  entry_count     = number of entries (uint64_t)

Per entry (Entry struct):
  image_kind      = ImageKind enum (uint16_t):
                    Object=1, Bitcode=2, CUBIN=3, Fatbinary=4, PTX=5, SPIRV=6
  offload_kind    = OffloadKind bitmask (uint16_t):
                    OpenMP=1, CUDA=2, HIP=4, SYCL=8
  flags           = uint64_t
  str_offset      = offset to StringEntry array (uint64_t)
  str_size        = string table byte count (uint64_t)
  img_offset      = offset to binary payload (uint64_t)
  img_size        = binary payload byte count (uint64_t)

String table:
  key-value pairs of null-terminated strings
  Standard keys: "triple" (e.g., "nvptx64-nvidia-cuda"),
                 "arch"   (e.g., "sm_89")
  Arbitrary additional keys permitted
```

Multiple OffloadBinary records are concatenated — each self-describes its size enabling sequential iteration without an external index. The `create()` API supports index-based selection within a concatenated blob (version 2 feature).

**Critical structural capability:** The StringMap is unbounded. Any key-value metadata can be embedded without format changes. libkdl capability contracts (SM version, ISA features, memory requirements, benchmark-derived scores) map directly to OffloadBinary StringMap entries.

### Where does libkdl integrate?

The LLVM offload toolchain creates a natural three-level hierarchy, and libkdl occupies the **policy layer** between level 2 and level 3:

```
Level 1: Compilation toolchain (clang → clang-offload-packager → llvm-offload-binary)
         Produces: OffloadBinary containers in .llvm.offloading ELF sections

Level 2: liboffload mechanism (olCreateProgram, olGetSymbol, olLaunchKernel)
         Consumes: single selected binary blob
         Policy: "first compatible image wins" (PR #186088, open March 2026)

[libkdl POLICY LAYER: multi-version selection, capability matching, cost scoring]

Level 3: Kernel dispatch (olLaunchKernel → plugin → GPU driver API)
```

**Two confirmed integration options:**

**Option A — libkdl above liboffload (recommended for Dublin prototype):**
libkdl parses the `.llvm.offloading` section using `llvm::object::OffloadBinary::create()`, applies its capability matching and cost scoring across all embedded images, selects the optimal image for the current device, and passes the selected binary blob to `olCreateProgram()`. liboffload handles plugin dispatch, kernel lookup (via `olGetSymbol()`), and launch. This path is implementable today with no upstream changes.

**Option B — libkdl extends liboffload internally:**
Contribute a `virtual Expected<int> rankImage(const OffloadBinMetadataTy&, DeviceId)` virtual hook to `GenericPluginTy`, replacing the `break` in PR #186088's `parseOffloadBinary` loop with a ranking call. The community appetite exists (PR #186088 explicitly defers this), and the governance path is clear (biweekly meetings, Discourse RFC). This is the longer-term upstream contribution path.

**The name-based kernel loading gap (Issue #75356) is the strongest differentiator:**

The current offload driver model requires compile-time kernel registration via `__tgt_offload_entry` tables. Dynamic dispatch (discovering kernels at runtime, loading pre-compiled kernel objects, dlopen-style kernel management) is explicitly not supported — Issue #75356 proposes `__tgt_get_kernel_handle` but has no upstream PR. libkdl's `kdl_kernel_lookup()` is an implementation of this exact missing capability. The Dublin poster can cite Issue #75356 directly as the community's own articulation of the problem libkdl solves.

### Key gaps in the new offload driver relevant to libkdl

1. **No dynamic kernel loading**: The compile-time `__tgt_offload_entry` table model prevents runtime kernel discovery. Issue #75356 is 2+ years old with no upstream PR.

2. **No multi-version selection policy**: PR #186088's "first compatible image wins" is the current default. No `rankImage` API exists.

3. **No PTX JIT path in olCreateProgram**: Issue #149284 is still open (April 2026). Callers must JIT-link PTX to CUBIN externally before calling `olCreateProgram`.

4. **API instability**: `olGetKernel` was renamed to `olGetSymbol` in PR #147943 (July 2025), 3 months after the initial API merged. A libkdl integration must include a version-detection shim.

5. **No kernel enumeration**: `olGetSymbol(program, name)` requires prior knowledge of kernel names. No `olEnumerateSymbols(program, callback)` exists. libkdl's MTB format includes an explicit kernel manifest that avoids this limitation.

### Relevance to libkdl (1–10): 10

The new offload driver toolchain is the **production infrastructure** that libkdl must interoperate with. The OffloadBinary format (`.llvm.offloading` section, magic `0x10FF10AD`, StringMap metadata) is the convergence point for multi-vendor GPU kernel packaging in LLVM. libkdl's KDL bundle format should be either a superset of OffloadBinary (additional metadata fields) or use OffloadBinary as its on-disk serialization format.

### Novelty (1–10): 7

The new offload driver story is well-known at the LLVM community level. The novel contributions of this wave are: (1) the precise OffloadBinary v2 binary layout from `OffloadBinary.h` (not previously documented in the wave series), (2) the specific timeline of the CUDA/HIP driver-default transitions (PR #84420, #122312), (3) the connection between Issue #75356's `__tgt_get_kernel_handle` PoC and libkdl's `kdl_kernel_lookup`, and (4) the three-level hierarchy showing exactly where libkdl sits in the toolchain stack.

---

## Cross-references to prior waves

- liboffload ol* API complete state: wave-02-llvm-offloading, wave-04-liboffload-multiversion
- PR #186088 (multi-image selection gap): wave-04-liboffload-multiversion (Source 6)
- PR #185404 (Level Zero OffloadBinary): wave-04-liboffload-multiversion (Source 5)
- Issue #149284 (PTX JIT gap): wave-02-llvm-offloading (Source 9), wave-04-liboffload-multiversion (Source 4)
- `ol_symbol_handle_t` rename PR #147943: wave-04-liboffload-multiversion (Source 2)
- SPIR-V via liboffload (PR #120145): wave-02-llvm-offloading (Source 6), wave-05-chipstar-spirv

## Suggested follow-up angles

1. `llvm::object::OffloadBinary` C++ parsing API — read `OffloadBinary.cpp` in LLVM source to understand the full iterator pattern for scanning concatenated OffloadBinary records in a section; this is the exact code libkdl's bundle reader should model
2. `offload/plugins-nextgen/common/PluginInterface.cpp` — read `parseOffloadBinary` function from PR #186088 to extract the exact loop structure and identify the `break` replacement point for libkdl's policy injection
3. `__tgt_register_lib` startup registration — trace the call graph from the global constructor generated by clang-linker-wrapper through libomptarget's image registration to understand the latency budget libkdl must match
4. `clang-linker-wrapper --override-image` flag — test whether this flag can be used to replace a kernel image at link time with a libkdl-packaged multi-target bundle, enabling drop-in integration without source changes
5. LLVM Offload Workshop 2025 full slide deck — fetch individual talk slides from discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832 to extract any discussion of selection policy or "where are we going" that post-dates the wave-02 survey
