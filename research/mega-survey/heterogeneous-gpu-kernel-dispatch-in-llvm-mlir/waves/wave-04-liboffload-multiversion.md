# Wave 04: liboffload ol* API Status and Multi-Version Kernel Selection Gap

**Search angle:** liboffload-multi-version-selection — investigating the current state of LLVM's liboffload ol* API and whether anyone is building multi-version kernel selection policy on top of it.

**Search queries:**
- "olCreateProgram olCreateKernel multi-version kernel selection llvm offload"
- "llvm liboffload API status 2025 2026"
- "llvm offload plugin interface ol* API"
- PR #122106, Issue #149284, PR #186088, PR #185404, PR #147943

**Sources found:** 9
**Date:** 2026-04-06

---

## Sources

### 1. PR #122106: [Offload] Implement the Remaining Initial Offload API — GitHub (MERGED)
- URL: https://github.com/llvm/llvm-project/pull/122106
- Type: PR (merged 2025-04-22)
- Date: January 2025 (merged April 2025)
- Relevance: 10/10
- Novelty: 9/10 (partially covered in wave-02-llvm-offloading, but now confirmed merged with precise API surface)
- Summary: This PR completed the initial liboffload C API, merging April 22, 2025. The implemented API surface covers: `olMemAlloc`/`olMemFree` (host/device/shared), `olCreateQueue`/`olWaitQueue`/`olFinishQueue` (async stream abstraction), `olEnqueueDataWrite`/`olEnqueueDataRead`/`olEnqueueDataCopy` (data movement), `olCreateProgram`/`olReleaseProgram` (binary blob to program handle), `olGetKernel`/`olSetKernelArgValue`/`olSetKernelArgsData` (kernel extraction by name + argument binding), and `olEnqueueKernelLaunch` (grid dispatch). Files modified confirm TableGen definitions in `offload/liboffload/API/`: `Common.td`, `Device.td`, `Event.td`, `Kernel.td`, `Memory.td`, `Program.td`, `Queue.td`. Unit tests added under `offload/unittests/OffloadAPI/` including `program/olCreateProgram.cpp` and `kernel/olGetKernel.cpp`.
- Key detail: `olCreateProgram(Device, ProgData, ProgDataSize, &Program)` takes an arbitrary binary blob with no format constraint at the API level — format detection happens at runtime in the plugin. `olGetKernel(Program, "kernel_name", &Kernel)` performs name-based lookup within the loaded program. The PR explicitly notes "The API should still be considered unstable." This is the confirmed mechanism layer as of LLVM 19-20.

### 2. PR #147943: [Offload] Change `ol_kernel_handle_t` → `ol_symbol_handle_t` — GitHub (MERGED)
- URL: https://github.com/llvm/llvm-project/pull/147943
- Type: PR (merged 2025-07-10)
- Date: July 2025
- Relevance: 9/10
- Novelty: 10/10 (not covered in any prior wave)
- Summary: API rename merged July 10, 2025. `ol_kernel_handle_t` has been renamed to `ol_symbol_handle_t` as "the first step in making symbols represent both kernels and global variables." The PR body states: "In the future, we want `ol_symbol_handle_t` to represent both kernels and global variables." This is a generalization of the kernel concept toward a unified symbol abstraction — analogous to `dlsym` returning any symbol, not just function pointers. The corresponding API function has evolved: `olGetKernel(Program, name)` → `olGetSymbol(Program, name, kind, &Symbol)` where `kind` is `OL_SYMBOL_KIND_KERNEL` or `OL_SYMBOL_KIND_GLOBAL_VARIABLE`.
- Key detail: The current (2025-07-10+) API uses `olGetSymbol` not `olCreateKernel` or `olGetKernel`. Any libkdl integration must use `olGetSymbol(program, "kernel_name", OL_SYMBOL_KIND_KERNEL, &symbol)` then `olLaunchKernel(queue, device, symbol, args_data, &launch_size_args)`. This API change is invisible from prior waves but critical for implementation. The `ol_symbol_handle_t` handle is owned by the program and does not need separate destruction — reducing lifecycle management complexity for libkdl.

### 3. Current ol* API Surface (from TableGen .td files, main branch, April 2026)
- URL: https://github.com/llvm/llvm-project/blob/main/offload/liboffload/API/OffloadAPI.td
- Type: source (TableGen API definitions)
- Date: April 2026 (continuously maintained)
- Relevance: 10/10
- Novelty: 10/10 (first precise enumeration of current complete API surface)
- Summary: The current liboffload API is defined across 10 `.td` files: APIDefs.td, Common.td, Platform.td, Device.td, Memory.td, Queue.td, Event.td, Program.td, Kernel.td, Symbol.td. Complete confirmed function inventory:
  - **Program:** `olCreateProgram(Device, ProgData, ProgDataSize, &Program)`, `olReleaseProgram(Program)`
  - **Symbol:** `olGetSymbol(Program, Name, Kind, &Symbol)`, `olGetSymbolInfo(Symbol, PropName, PropSize, PropValue)`, `olGetSymbolInfoSize(Symbol, PropName, &PropSizeRet)`
  - **Kernel:** `olLaunchKernel(Queue, Device, Kernel/Symbol, ArgumentsData, LaunchSizeArgs)`, `olCalculateOptimalOccupancy(Device, Kernel, SharedMemory, &GroupSize)`
  - **Memory:** `olMemAlloc(Device, Type, Size, &Ptr)`, `olMemFree(Device, Ptr)`, `olMemcpy(Queue, Device, Dst, DstDevice, Src, SrcDevice, Size, &Event)`
  - **Queue:** `olCreateQueue(Device, &Queue)`, `olDestroyQueue(Queue)`, `olWaitQueue(Queue)`
  - **Event:** `olWaitEvent(Event)`, `olDestroyEvent(Event)`
  - **Platform:** `olGetPlatform(NumEntries, Platforms, &NumPlatforms)`, `olGetPlatformInfo(...)`, `olGetPlatformInfoSize(...)`
  - **Device:** `olIterateDevices(Callback, UserData)`, `olGetDeviceInfo(...)`, `olGetDeviceInfoSize(...)`
- Key detail: No `olSelectKernel`, no `olEnumerateKernels`, no `olQueryKernelVariants`, no `olGetBestKernel` — there is zero API surface for multi-version selection policy in the current liboffload. The API provides mechanism (load binary blob, look up symbol by name, launch) but has no vocabulary for "I have multiple compiled variants of this kernel, select the best one for this device." This is the confirmed policy gap libkdl fills.

### 4. Issue #149284: [offload] Allow liboffload CUDA Plugin to Accept PTX Binaries — GitHub (OPEN)
- URL: https://github.com/llvm/llvm-project/issues/149284
- Type: issue (open, July 2025)
- Date: July 17, 2025
- Relevance: 8/10
- Novelty: 8/10 (referenced in wave-02, now with full primary source text)
- Summary: Issue opened by Ross Brunton (Intel/Codeplay) reporting that users of `olCreateProgram` must manually invoke `cuLinkCreate`/`cuLinkAddData`/`cuLinkComplete` before passing a CUBIN to `olCreateProgram` when the input is PTX text. Brunton provides example code from Unified Runtime showing a `ProgramCreateCudaWorkaround` function. Joseph Huber (AMD) comments: "We already do this kind of stuff in the JIT engine, the main annoyance is detecting it since PTX is a textual format. Beyond that, should be pretty easy to slot it next to where we discern between ELF and IR." The issue remains open as of April 2026.
- Key detail: The workaround code directly calls `olCreateProgram(hContext->Device->OffloadDevice, RealBinary, RealLength, &hProgram->OffloadProgram)` — confirming this is the live production call path. libkdl's CUDA backend uses `cuModuleLoadData` (equivalent to `cuLinkComplete` → `cuModuleLoad`) and bypasses this issue entirely. If libkdl wraps liboffload instead, it would inherit this PTX limitation for CUDA unless the feature lands.

### 5. PR #185404: [Offload][L0] Add support for OffloadBinary format in L0 plugin — GitHub (MERGED)
- URL: https://github.com/llvm/llvm-project/pull/185404
- Type: PR (merged 2026-03-11)
- Date: March 2026
- Relevance: 9/10
- Novelty: 10/10 (not covered in any prior wave)
- Summary: Merged March 11, 2026. Adds support in the Level Zero plugin for the `OffloadBinary` container format — a format that wraps multiple inner images with metadata (triple, arch, ImageKind, OffloadKind) in a single blob. The PR teaches `isImageCompatible` and `loadBinary` to unwrap `OffloadBinary` containers, extract metadata, check compatibility with the plugin's target, and load the inner image. Introduces `OffloadBinMetadataTy` struct carrying: `ImageKind`, `OffloadKind`, `Triple` (as string), `Arch` (as string), and a `StringMap<string>` for additional key-value metadata. Adds a new virtual method `isMetadataCompatible(const OffloadBinMetadataTy&)` to `GenericPluginTy` that plugins override to express compatibility constraints beyond raw binary magic bytes.
- Key detail: This PR introduces the critical infrastructure for multi-image container support in liboffload. An `OffloadBinary` can contain multiple inner images (e.g., PTX for SM 7.5 + HSACO for gfx1030 + SPIR-V for Intel). The metadata-based compatibility check (`isMetadataCompatible`) allows per-plugin filtering before the expensive binary parsing step. The compatibility model is "first compatible image wins" — not a selection policy.

### 6. PR #186088: [OFFLOAD] Generalize support for OffloadBinary images — GitHub (OPEN)
- URL: https://github.com/llvm/llvm-project/pull/186088
- Type: PR (open, March 2026)
- Date: March 12, 2026
- Relevance: 10/10
- Novelty: 10/10 (not covered in any prior wave — most directly relevant source found)
- Summary: Open PR by Alex Duran (adurang) generalizing the OffloadBinary multi-image support introduced in #185404 from Level Zero to all plugins (CUDA, AMDGPU, host). The implementation: `parseOffloadBinary(MemoryBufferRef)` iterates over all embedded inner images, extracts `OffloadBinMetadataTy` for each, calls `isMetadataCompatible(metadata)` per-plugin, then `isDeviceCompatible(DeviceId, InnerImage)` — and loads the **first** compatible image. The PR author explicitly calls out the design limitation: "For now only the first compatible image in the binary is loaded. While it might be desirable to add support for loading multiple images, our current interface is limiting (i.e., it returns a single Image) and it's unclear if in all cases this behavior is desirable so we would need to add more options to control it. So, should we want it, it's better in a follow-up PR."
- Key detail: This is the clearest available community statement that multi-version image selection is a known future work item within liboffload itself. The interface returns `Expected<DeviceImageTy*>` — a single image pointer — and refactoring to return a ranked set of candidates would require a non-trivial API change. The `parseOffloadBinary` function (added in PluginInterface.cpp) already parses all variants and iterates them; the only missing piece is a ranking/selection callback. This is the exact hook point where a libkdl-style policy layer would integrate.

### 7. PR #185663: [OFFLOAD] Add interface to extend image validation — GitHub (MERGED)
- URL: https://github.com/llvm/llvm-project/pull/185663
- Type: PR (merged 2026-03-10)
- Date: March 2026
- Relevance: 8/10
- Novelty: 9/10 (not covered in prior waves)
- Summary: Merged one day before #185404. Adds `isMetadataCompatible(const OffloadBinMetadataTy&)` as a virtual method on `GenericPluginTy` with a default implementation returning `true` (always compatible). Individual plugin subclasses override this to implement target-specific validation. The PR was motivated by "as discussed in #185404 we might want to provide a way for plugins to validate images not recognized by the common layer." This is the extensibility hook that allows metadata-driven image filtering without modifying the core loading path.
- Key detail: The extensibility pattern here (virtual method on GenericPluginTy, default-true, overridden by plugins) is exactly the pattern a libkdl policy injection point would use. A "selector callback" API could follow the same pattern: `virtual Expected<int> rankImage(const OffloadBinMetadataTy&, int DeviceId)` returning a score, with the loader selecting the highest-ranked compatible image. This would be a minimal, backwards-compatible extension to support multi-version selection policy within liboffload itself.

### 8. PR #184343: [offload] Add properties parameter to olLaunchKernel — GitHub (OPEN)
- URL: https://github.com/llvm/llvm-project/pull/184343
- Type: PR (open, February 2026)
- Date: February 2026
- Relevance: 7/10
- Novelty: 8/10 (not covered in prior waves)
- Summary: Open PR adding a `properties` struct parameter to `olLaunchKernel` to enable future API extensions without breaking changes. Initial use: a `cooperative_launch` boolean flag and `argument_size` field (required by Level Zero). The properties mechanism follows the Unified Runtime pattern for extensible API calls. Motivation for argument_size: Level Zero requires argument size information at kernel launch time rather than at argument-setting time, necessitating a late-binding size specification.
- Key detail: The extensible properties mechanism on `olLaunchKernel` is the established pattern for adding new launch-time policy without breaking the core API. A multi-version selection hint (e.g., `ol_launch_properties_t::preferred_variant = "sm_89_optimized"`) could be added following this pattern — allowing callers to override automatic selection. This is direct evidence that the API design already accommodates future policy extensions.

### 9. Issue #79304: [Offload] Develop a New API for the "Plugins" — GitHub
- URL: https://github.com/llvm/llvm-project/issues/79304
- Type: issue (referenced in wave-02-llvm-offload-runtime, verified still open as of April 2026)
- Date: January 24, 2024
- Relevance: 9/10
- Novelty: 7/10 (anchor issue, already documented in prior waves)
- Summary: The originating issue for the liboffload C API redesign. Joseph Huber's (AMD) framing: "The plugins abstract over vendor-dependent libraries and provide a common interface used by OpenMP and other languages to provide language-specific features, but they are currently not exported and the API is ill-defined." The issue was the motivation for offload-tblgen (PR #118614) and the initial API (PR #122106). The issue is still open as of April 2026 because it serves as a tracking issue for the broader API design effort — new capability requests (like PTX binary input, #149284; properties extension, #184343) are linked back to it.
- Key detail: Reviewing the issue thread as of April 2026: no comment proposes or mentions multi-version kernel selection, dispatch policy, or capability-based kernel ranking. The issue is entirely focused on mechanism (loading, launching, memory management). This confirms the design intent: liboffload is a mechanism library and the community has not proposed extending it with policy.

---

## Angle Assessment

### What is the current state of the liboffload ol* API?

**Status as of April 2026:**

The liboffload API is real, merged, and functional. The complete dispatch loop is operational:

```c
// 1. Load binary blob (ELF/CUBIN/HSACO/SPIR-V — format auto-detected)
olCreateProgram(device, binary_data, binary_size, &program);

// 2. Look up kernel symbol by name (as of July 2025, olGetKernel → olGetSymbol)
olGetSymbol(program, "kernel_name", OL_SYMBOL_KIND_KERNEL, &symbol);

// 3. Launch
olLaunchKernel(queue, device, symbol, args_data, &launch_size_args);
```

Key API evolution events (in chronological order):
- **Nov 2024** (PR #118614): offload-tblgen infrastructure merged
- **Apr 2025** (PR #122106): Complete initial API merged — `olCreateProgram`, `olGetKernel`, `olEnqueueKernelLaunch`
- **Jul 2025** (PR #147943): `ol_kernel_handle_t` renamed to `ol_symbol_handle_t`; `olGetKernel` → `olGetSymbol(program, name, kind, &symbol)`
- **Mar 2026** (PR #185663): `isMetadataCompatible` extensibility hook added to `GenericPluginTy`
- **Mar 2026** (PR #185404): Level Zero plugin gains OffloadBinary multi-image container support
- **Mar 2026** (PR #186088): CUDA + AMDGPU plugins gain OffloadBinary support — first-compatible-wins selection

**API is explicitly unstable.** PR #122106 body: "The API should still be considered unstable and it's very likely we will need to change the existing entry points." This means libkdl integration must either vendor-pin a commit or abstract behind a compatibility shim.

### Does multi-version kernel selection policy exist in liboffload?

**No. Confirmed gap. Evidence:**

1. The complete function inventory (Source 3) shows zero API surface for selection policy.
2. PR #186088 (Source 6) implements multi-image container parsing with "first compatible image wins" and explicitly defers multi-image selection to "a follow-up PR."
3. Issue #79304 (Source 9) — the tracking issue for the entire API design — contains no mention of selection policy across its entire comment thread.
4. The liboffload roadmap RFC (wave-05-llvm-discourse-rfcs, Source 2) explicitly states the API is "intentionally lower-level than any language model" — mechanism, not policy.

### What is the integration surface for libkdl?

**Confirmed integration points (from live code in PR #186088):**

The `parseOffloadBinary` function in `PluginInterface.cpp` already iterates all inner images from an OffloadBinary container. The loop structure is:

```cpp
for (auto &[ExtractedMetadata, InnerImage] : InnerImages) {
    if (!Plugin.isMetadataCompatible(Metadata)) continue;  // metadata filter
    if (!Plugin.isDeviceCompatible(DeviceId, InnerImage)) continue;  // binary filter
    // LOAD first compatible → break
    break;
}
```

The integration point for libkdl-style multi-version selection is the replacement of `break` with a ranking callback. The `OffloadBinMetadataTy` struct already carries: `Triple`, `Arch`, `ImageKind`, `OffloadKind`, and arbitrary `StringData` key-value pairs — sufficient metadata for a capability-contract match or roofline cost score.

**libkdl integration path (two options):**

Option A — Sit above liboffload: libkdl performs its own multi-image selection (using its existing capability matching logic), then calls `olCreateProgram` with the chosen binary blob. liboffload handles the rest (plugin dispatch, kernel lookup, launch). This requires libkdl to parse OffloadBinary containers itself (the `llvm::object::OffloadBinary` class is in `llvm/include/llvm/Object/OffloadBinary.h`).

Option B — Extend liboffload: Propose a `rankImage` virtual hook on `GenericPluginTy` following the same pattern as `isMetadataCompatible`. This would require an upstream contribution but would integrate selection policy directly into liboffload's loading path. Given that PR #186088's author already identified this as follow-up work, there is community appetite.

Option A is the correct choice for the LLVM Dublin poster: it is implementable now, does not require upstream API changes, and clearly demonstrates libkdl as a policy layer above the mechanism.

### What is novel about the liboffload multi-image situation as of April 2026?

The most surprising finding is PR #186088 (open, March 2026): the community is actively building multi-image container support in liboffload's plugin layer, but is explicitly deferring selection policy. The code already has a loop over all candidates — the "break on first match" is a deliberate design choice pending a fuller solution. libkdl's MTB (Multi-Target Bundle) format and capability-contract matching is the policy layer this loop is waiting for.

The second surprise: the `ol_kernel_handle_t` → `ol_symbol_handle_t` rename (PR #147943, July 2025) was not in any prior wave. This changes the call site for any libkdl integration from `olGetKernel` to `olGetSymbol(program, name, OL_SYMBOL_KIND_KERNEL, &sym)`. The rename also signals a more general symbol abstraction (covering global variables, not just kernels), which could support libkdl's global constant management.

### Risks and open questions

1. **API instability**: liboffload is explicitly unstable. The `olGetKernel` → `olGetSymbol` rename (3 months after the initial API merged) demonstrates this is not theoretical. A libkdl integration must abstract the API boundary with a version-detection shim.

2. **PTX binary gap**: Issue #149284 is still open. `olCreateProgram` does not accept PTX text directly — users must JIT-link PTX to CUBIN first. libkdl's own CUDA backend bypasses this by calling CUDA driver APIs directly, but a liboffload-integrated libkdl would need to handle this.

3. **PR #186088 not yet merged**: The generalization of OffloadBinary support to CUDA/AMDGPU is still open. If merged, it becomes the production multi-image loading path for all vendors — and the "first match" policy becomes the default liboffload behavior that libkdl must override.

4. **No API for kernel enumeration**: `olGetSymbol` requires knowing the kernel name in advance. There is no `olEnumerateSymbols(program, callback)` equivalent — no way to discover what kernels a binary contains via liboffload. libkdl's MTB format stores a kernel manifest explicitly, avoiding this limitation.

5. **Level Zero only for now**: PR #185404 (OffloadBinary support) is merged for Level Zero. PR #186088 (generalization to all plugins) is still open. CUDA and AMDGPU OffloadBinary support is pending review as of April 2026.

---

## Cross-references to prior waves

- liboffload mechanism layer (complete): wave-02-llvm-offloading (Sources 1-4, 8-9), wave-02-llvm-offload-runtime (Sources 3-5, 9)
- Name-based kernel loading gap (Issue #75356): wave-02-llvm-offload-runtime (Source 4)
- Mechanism vs. policy split: wave-05-llvm-discourse-rfcs (Sources 1, 2, 4)
- Policy direction (libkdl above liboffload): directions/01-policy-layer-above-liboffload.md

## Suggested follow-up angles

1. `llvm::object::OffloadBinary` class — parse the C++ header in `llvm/include/llvm/Object/OffloadBinary.h` to understand the full metadata schema and whether it can carry capability constraints compatible with libkdl's contract format
2. `offload/unittests/OffloadAPI/program/olCreateProgram.cpp` — read the unit test to understand what binary formats are exercised in practice (CUDA CUBIN? AMDGPU HSACO? SPIR-V?)
3. PR #184343 (olLaunchKernel properties) — if merged before Dublin, the properties extension could carry a libkdl-compatible variant hint at launch time rather than at load time
4. OMPT device observability (Issue #110007, wave-05-discourse S13) — profiling hooks for kernel dispatch events; relevant for measuring libkdl overhead on top of liboffload
5. OffloadBinary format specification — document the binary layout (magic, version, metadata section, image section) for the libkdl MTB compatibility analysis
