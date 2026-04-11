# Direction 02: libkdl as the Policy Layer for LLVM liboffload

**Composite Score: 8.75/10**
**Rank: 2 of 8**

---

## Title

**The Missing rankImage() Callback: libkdl as Multi-Version Selection Policy for LLVM's liboffload Infrastructure**

## One-Sentence Description

Position libkdl as the dispatch policy layer that LLVM's liboffload mechanism layer explicitly defers, replacing the current "first compatible image wins" selection in `parseOffloadBinary` with a capability-scored ranking callback.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| liboffload PR #186088 (multi-image loading) | wave-04-liboffload-multiversion | Author explicitly states: "For now only the first compatible image... it's better in a follow-up PR" |
| OffloadBinary v2 format specification | wave-06-llvm-offload-new-driver | StringMap metadata is extensible — libkdl capability contracts map directly to existing fields |
| liboffload roadmap RFC | wave-06-llvm-offload-new-driver (Source 5) | Three-layer architecture: toolchain → mechanism → dispatch. Multi-version selection is "future work" |
| UR-on-liboffload bridging | wave-04-unified-runtime-vs-liboffload | Validates "sit above liboffload" architecture pattern |
| SYCL upstreaming (libsycl Aug 2025) | wave-04-unified-runtime-vs-liboffload | libkdl above liboffload inherits SYCL ecosystem |
| LLVM offload biweekly meetings | wave-06-llvm-offload-new-driver (Source 4) | Clear governance path for upstream contributions |
| SPIR-V OpenMP offloading PR #120145 | wave-06-llvm-offload-new-driver (Source 8) | Demonstrates OffloadBinary extensibility — adding new target requires only a plugin + toolchain wrapper |
| New offload driver default for CUDA/HIP | wave-06-llvm-offload-new-driver (Sources 2,3) | All LLVM 20+ CUDA/HIP/OpenMP produce .llvm.offloading by default — libkdl's input format is ubiquitous |
| Issue #75356 (name-based kernel loading) | wave-06-llvm-offload-new-driver (Source 9) | Proposed __tgt_get_kernel_handle has no upstream PR; libkdl's kdl_kernel_lookup is the implementation |
| `isMetadataCompatible()` virtual method | wave-04-liboffload-multiversion | Exact extensibility hook where rankImage() callback slots in |

---

## Novelty Argument

The LLVM offload infrastructure has converged on a complete toolchain-to-runtime pipeline:

```
clang → .llvm.offloading section → clang-linker-wrapper → __tgt_register_lib() → liboffload → vendor plugin
```

As of LLVM 20, this pipeline is the default for CUDA, HIP, and OpenMP. The OffloadBinary format supports multi-image containers with per-image metadata. The `parseOffloadBinary` function iterates all images and checks compatibility. But it uses `break` on the first compatible match.

Nobody has proposed the ranking callback. The PR #186088 author defers it. The roadmap RFC lists it as "future work." The biweekly offload meetings have discussed it but produced no RFC.

libkdl is not a competing project — it is the answer to the question the LLVM offload community is already asking.

---

## Feasibility Plan

**Option A — Above liboffload (recommended for poster deadline):**
1. Parse `.llvm.offloading` sections using `llvm::object::OffloadBinary::create()`
2. Iterate all embedded images, extract metadata (Triple, Arch, StringMap)
3. Apply libkdl's capability matching and roofline scoring
4. Pass selected binary blob to `olCreateProgram()`
5. Use `olGetSymbol()` + `olLaunchKernel()` for dispatch

This requires no upstream changes. Implementable in <500 LOC wrapper.

**Option B — Extend liboffload (upstream contribution path):**
1. Add `virtual Expected<int> rankImage(const OffloadBinMetadataTy&, DeviceId)` to `GenericPluginTy`
2. Replace `break` in `parseOffloadBinary` loop with ranking call
3. Submit via LLVM Discourse RFC + biweekly meeting discussion

**Poster should demonstrate Option A and propose Option B as the upstream path.**

---

## Poster Potential

**Yes — strong community-alignment narrative.**

- Diagram showing the three-level hierarchy with libkdl at the policy seam
- Code diff: before (first-compatible `break`) vs. after (rankImage callback)
- OffloadBinary metadata table showing libkdl capability contracts as StringMap entries
- Timeline: Issue #75356 (Nov 2023) → PR #186088 (Mar 2026) → libkdl (Dublin 2026)

This fills a poster panel as the "LLVM integration story" complementing the core contribution.

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **8/10** | Policy layer concept is novel; integration point is precisely identified in live code (PR #186088). |
| **Feasibility** | **9/10** | Option A requires no upstream changes; wraps existing API. Option B has clear governance path. |
| **Evidence** | **9/10** | PR #186088 text, roadmap RFC, OffloadBinary v2 spec all confirm the gap and the integration point. |
| **Impact** | **9/10** | Positions libkdl as natural upstream contribution. Inherits SYCL/OpenMP/CUDA/HIP ecosystems. |
| **Composite** | **8.75/10** | |
