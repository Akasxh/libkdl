# Wave 03: Multi-Versioned Kernel Selection
Search query: "multi-versioned kernel selection function multi-versioning GPU architecture dispatch resolver"
Sources found: 10
Date: 2026-04-06

## Sources

### 1. Function Multi-Versioning for AArch64 — Euro LLVM Developers' Meeting 2025 (Lamprineas, Arm)
- URL: https://llvm.org/devmtg/2025-04/slides/technical_talk/lamprineas_function_multi-versioning.pdf
- Type: conference talk / slides
- Date: 2025-04 (Euro LLVM Dev Meeting)
- Relevance: 9/10
- Novelty: 8/10
- Summary: Covers the state-of-the-art for LLVM's FMV (Function Multi-Versioning) on AArch64, including LLVM 20's GlobalOpt pass that can statically resolve ifunc resolver calls when the caller's architecture features are known at compile time, and TableGen-generated dependency tables for feature detection. The talk addresses the core dispatch problem: software deployed on diverse devices with different optional instructions (e.g., dotproduct) needs a one-time runtime check to route to the best-compiled variant.
- Key detail: LLVM 20 (March 2025) added `GlobalOpt` support that eliminates the ifunc dispatcher overhead when the caller's feature set guarantees a specific variant will always win — the indirect call is collapsed into a direct call. This is the "static resolver optimization" that libkdl could borrow: if device capabilities are fully known at load time, skip the resolver and bind directly.

### 2. Function Multi-Versioning — MaskRay (Fangrui Song)
- URL: https://maskray.me/blog/2023-02-05-function-multi-versioning
- Type: blog / deep technical reference
- Date: 2023-02 (updated periodically)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Definitive reference on CPU FMV mechanics: `__attribute__((target_clones(...)))` generates N optimized function bodies plus an ELF `STT_GNU_IFUNC` resolver symbol. The resolver is invoked exactly once by `rtld` at relocation time, executes `cpuid` / HWCAP detection, and returns a function pointer. All future calls go through a PLT entry to the selected body — no re-dispatch overhead. Key limitation: ifunc calls defeat all interprocedural optimizations (inlining, devirtualization) because the compiler cannot reason about which body will be selected.
- Key detail: The resolver-PLT indirection in CPU FMV is a 1-level lookup: `cpuid → resolver(once) → PLT → selected_fn`. GPU kernel dispatch is conceptually the same but the "feature query" is a runtime API call (`cuDeviceGetAttribute`, `hipGetDeviceProperties`) and the "resolver" runs in host code. libkdl's design mirrors this precisely at the kernel granularity.

### 3. LLVM 20 Release Notes — FMV and GlobalOpt Changes
- URL: https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html
- Type: official docs / changelog
- Date: 2025-03-11
- Relevance: 8/10
- Novelty: 7/10
- Summary: Documents two significant FMV advances in LLVM 20: (1) `GlobalOpt` can now statically resolve calls to multi-versioned functions for AArch64 when the call site can guarantee which variant will win, removing the ifunc overhead entirely; (2) FMV feature dependency tables are now auto-generated via TableGen from the ARMTargetDefEmitter, eliminating hand-maintained lists. Multi-versioned functions gain `fmv-features` metadata annotating them with their target feature sets.
- Key detail: The `fmv-features` metadata (comma-separated ACLE feature names) is the IR-level encoding of "which variant is this." For GPU kernels, the equivalent would be attaching target architecture metadata (sm_86, gfx1100) to each compiled kernel variant in a shared library — exactly what libkdl's `.kdl` section does.

### 4. Architecture Specific Code Generation and Function Multiversioning — LLVM Dev Meeting 2014 (Christopher)
- URL: https://llvm.org/devmtg/2014-10/Slides/Christopher-Function%20Multiversioning%20Talk.pdf
- Type: conference slides
- Date: 2014-10
- Relevance: 7/10
- Novelty: 4/10
- Summary: The original LLVM FMV design talk. Establishes the ifunc-based dispatch chain, the `target` and `target_clones` attribute semantics, and the motivation (AVX2-optimized code path vs. SSE2 fallback). Shows the full dispatch chain: application code → PLT stub → ifunc resolver (runs once at relocation) → optimal implementation. Priority ordering: most-specific target wins, `default` is the ultimate fallback.
- Key detail: The priority ordering mechanism (more specific target = higher priority) is directly applicable to GPU kernel selection. For CUDA fat binaries, sm_90 > sm_80 > PTX is the same hierarchy. libkdl needs an analogous priority table keyed on architecture capability levels.

### 5. CUDA Fat Binary Architecture Selection — NVIDIA NVCC and Binary Utilities Documentation
- URL: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- Type: official docs
- Date: 2026-03 (current)
- Relevance: 9/10
- Novelty: 5/10
- Summary: The CUDA driver implements the canonical two-level GPU kernel "multi-versioning" algorithm: (1) find an exact-match or backward-compatible cubin for the detected SM version (cubin compiled for sm_X.Y runs on sm_X.Z where Z >= Y); (2) fall back to JIT-compiling the embedded PTX. The selection runs in the CUDA driver at `cuModuleLoad` / `cuModuleLoadData` time, not at kernel-launch time. The result is cached in `~/.nv/ComputeCache`.
- Key detail: The SM-compatibility rule (same major, higher-or-equal minor) is an implicit versioning lattice. A runtime resolver for GPU kernels needs to implement this partial order: exact match > compatible cubin > PTX JIT. For cross-vendor dispatch (NVIDIA + AMD + CPU), the same principle extends to a heterogeneous capability lattice — libkdl's resolver traverses exactly this lattice.

### 6. [FMV][AArch64] Do Not Emit ifunc Resolver on Use — LLVM PR #97761
- URL: https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg452381.html
- Type: PR / patch review
- Date: 2024-07
- Relevance: 8/10
- Novelty: 8/10
- Summary: Patches Clang to defer ifunc resolver emission until the point of function definition rather than at each call site, reducing binary size and avoiding redundant resolver stubs. The PR discussion reveals the design tension: resolvers should be emitted exactly once, but the current Clang implementation could emit them per-TU in some LTO scenarios. This is the same problem that will arise if libkdl is implemented as a header-only resolver: link-time deduplication of the dispatch table must be explicit.
- Key detail: The fix aligns with the model that a resolver is a singleton keyed on the versioned symbol name. For GPU kernels, this maps to: one resolver function per kernel name, registered in the `.kdl` descriptor section, invoked once at module load time.

### 7. IREE HAL Executable Variant Conditional Enablement — Issue #3768
- URL: https://github.com/iree-org/iree/issues/3768
- Type: GitHub issue / design discussion
- Date: 2021 (closed 2023)
- Relevance: 8/10
- Novelty: 7/10
- Summary: Proposes extending IREE's HAL to use LLVM function multiversioning for CPU kernel variants, enabling a single `hal.executable.target` to generate multiple architecture-specific variants rather than separate compilation pipelines. Ultimately solved via `hal.executable.variant` conditional enablement: each variant has an optional boolean `condition` region evaluated at runtime against device properties; failing variants are skipped and the next fallback is tried. Multiple exports can chain as fallbacks forming a priority-ordered dispatch tree.
- Key detail: IREE's solution is essentially an FMV resolver at the MLIR/HAL level: `hal.executable.variant` with `condition` + `fallback` chains implement exactly the "check feature → select variant" pattern that CPU FMV uses with ifunc. This is the closest existing MLIR-level analogue to what libkdl needs. The design confirms that the FMV resolver pattern scales to heterogeneous GPU targets.

### 8. Performance Portability Through ML-Guided Kernel Selection in SYCL Libraries (Lawson, 2020)
- URL: https://arxiv.org/abs/2008.13145
- Type: paper (Parallel Computing journal, 2021)
- Date: 2020-08 (published 2021)
- Relevance: 7/10
- Novelty: 7/10
- Summary: Proposes using unsupervised clustering to select a minimal subset of kernel variants to compile (reducing binary bloat from the full Cartesian product of configurations), then trains a lightweight classifier to select the best variant at runtime given input size, hardware, and other features. Applied to SYCL library kernels across Intel GPU, NVIDIA GPU, and CPU. Achieves near-vendor-library performance without manual tuning.
- Key detail: This is the only paper that treats GPU kernel multi-versioning as an ML problem. The clustering step (offline, at library build time) is analogous to "which variants to pre-compile and package." The classifier (runtime, O(1) lookup) is the resolver. For libkdl, the offline step is already manual (developer chooses which SM/GFX variants to compile), but the runtime classifier concept could inform a smarter scoring function than simple "best compatible architecture."

### 9. HIP Target ID and Multi-Architecture Compilation — Clang/ROCm Documentation
- URL: https://clang.llvm.org/docs/HIPSupport.html
- Type: official docs
- Date: 2024-current
- Relevance: 8/10
- Novelty: 5/10
- Summary: Clang/HIP supports `--offload-arch=gfx1030,gfx1100,gfx906` to generate a single fat binary with per-architecture cubins for each listed GFX target. Target IDs extend arch names with feature flags (e.g. `gfx906:xnack+`). At runtime, `hipModuleLoad` selects the image whose target ID best matches the detected device, using a capability-matching algorithm similar to CUDA's SM compatibility rules.
- Key detail: HIP's `--offload-arch=native` detects the current system GPU and uses that as the single target — useful for dev builds but not for distribution. The `target ID` mechanism is HIP's "FMV feature string": the human-readable capability tag attached to each kernel image variant. libkdl's per-variant architecture metadata should be structurally identical to HIP target IDs.

### 10. LLVM/Clang Offloading Design — Multi-Arch Binary Embedding and Runtime Image Selection
- URL: https://clang.llvm.org/docs/OffloadingDesign.html
- Type: official docs
- Date: 2024-current (Clang 20+)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Describes the full LLVM offloading pipeline: device code compiled to per-arch object files, bundled into fat binaries via magic-byte-prefixed containers, embedded in ELF `.llvm.offloading` sections, and registered with the offload runtime via constructor-injected `__tgt_register_lib()`. The `__tgt_bin_desc` struct holds an array of `__tgt_device_image` entries (one per arch), plus host-side entry tables. The runtime iterates the image array and selects the image whose triple matches the loaded device.
- Key detail: The `__tgt_device_image` struct is LLVM's multi-versioned kernel descriptor: `{ImageStart, ImageEnd, EntriesBegin, EntriesEnd}` tagged by a target triple string. The runtime resolver is a simple linear scan of this array matching the triple. libkdl uses an identical pattern but with a richer metadata header (`.kdl` section) and a smarter scorer that ranks by capability proximity rather than exact-string match.

---

## Synthesis

### Core Pattern: The Resolver-Variant Model

All implementations of multi-versioned kernel selection share the same three-part structure:

1. **Variant table**: A compile-time-constructed list of `(capability_tag, code_pointer)` pairs — whether this is an ELF ifunc resolver table, a CUDA fat binary's cubin array, an IREE `hal.executable.variant` array, or libkdl's `.kdl` section.

2. **Resolver / selector**: A function invoked once at load time (for ELF ifunc: rtld relocation; for CUDA: `cuModuleLoad`; for IREE: HAL device initialization; for libkdl: `kdl_open()`) that queries device/CPU capabilities and returns the best matching entry.

3. **Priority / scoring function**: The rule for "best match" — for CPU FMV this is a bitmask intersection scoring (most features matched wins); for CUDA it is the SM compatibility partial order (same major, higher-or-equal minor, exact > compatible > JIT); for IREE it is a conditional expression evaluated against device properties.

### Gap: No First-Class GPU FMV in LLVM

None of the sources document a `__attribute__((target_clones(...)))` analogue for GPU kernels in LLVM/Clang. The CPU FMV infrastructure (`STT_GNU_IFUNC`, resolver emission, `fmv-features` metadata) is entirely CPU-side (x86, AArch64). GPU kernel multi-versioning is handled at the runtime level (CUDA driver, HIP runtime, IREE HAL) rather than at the compiler IR/linker level. This is the gap libkdl addresses.

### Relevance to libkdl

- **Resolver design**: libkdl's `kdl_open()` function is the GPU analogue of an ELF ifunc resolver: it runs once, queries device capabilities, scores each registered variant, and caches the selection.
- **Metadata format**: The `.kdl` ELF section mirrors the `__tgt_bin_desc` / `__tgt_device_image` pattern but with richer capability metadata (vendor, architecture string, SM/GFX version, feature flags).
- **Priority ordering**: libkdl's scorer should implement the SM-compatibility partial order (for CUDA) and GFX target-ID matching (for HIP), with a cross-vendor fallback chain (CUDA → HIP → SPIR-V → CPU).
- **Static resolution optimization**: LLVM 20's GlobalOpt static resolver collapse (source 1, 3) is directly applicable: if `kdl_open()` is called in a context where the device is statically known (e.g., a CUDA-only build), the resolver can be constant-folded at link time.

---

## Key Findings for the Poster

1. CPU FMV (ifunc + target_clones) is the canonical prior art for multi-versioned dispatch. libkdl extends this pattern to heterogeneous GPU targets.
2. LLVM 20's `GlobalOpt` static resolver optimization demonstrates that the compiler community recognizes resolver overhead as worth eliminating — same motivation applies to GPU kernel dispatch.
3. IREE HAL's `hal.executable.variant` with conditional enablement and fallback chains is the closest existing MLIR-level implementation of GPU multi-versioning. IREE solved this problem at the MLIR layer; libkdl solves it at the binary/ELF layer for pre-compiled kernels.
4. No LLVM/Clang proposal exists to extend FMV to GPU offload targets. This is an explicit gap and a potential future direction for libkdl to propose upstream.
5. ML-guided kernel selection (source 8) offers a richer resolver model than simple capability matching — relevant for future work where libkdl's scorer could incorporate input-size heuristics.
