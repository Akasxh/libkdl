# PR Status Check — LLVM Offload Image Dispatch
**Date:** 2026-04-09
**Scope:** PRs #186088, #185663, related new PRs since March 2026, competing RFCs, EuroLLVM program

---

## 1. PR #186088 — [OFFLOAD] Generalize support for OffloadBinary images

**Status: OPEN — stalled, awaiting direction**

- **URL:** https://github.com/llvm/llvm-project/pull/186088
- **Author:** Alex Duran (`adurang`)
- **Created:** 2026-03-12
- **Labels:** `offload`, `backend:AMDGPU`
- **Review requests:** `jdoerfert`, `hansangbae`

**What it does:**
Generalizes the OffloadBinary-wrapping support that was merged for L0 (Intel) in #185404 to ALL plugins (CUDA, AMDGPU, host). Key changes:

- Adds `OffloadBinMetadataTy` struct (triple, arch, ImageKind, OffloadKind, StringMap) to `PluginInterface.h`
- Extends `loadBinaryImpl()` signature across all plugins to accept `const OffloadBinMetadataTy*`
- Adds `isMetadataCompatible()` virtual hook on `GenericPluginTy` — plugins can override to reject images by triple/kind before attempting to load inner bytes
- `GenericDeviceTy::loadBinary()` now handles `file_magic::offload_binary` natively: parses all inner images, calls `isMetadataCompatible()` + `isDeviceCompatible()` per entry, loads first compatible match
- `isPluginCompatible()` / `isDeviceCompatible()` both extended to recurse into OffloadBinary containers
- L0 plugin's `isImageCompatible()` simplified: pure SPIR-V only; `isMetadataCompatible()` added to check `spirv64-intel` triple + IMG_SPIRV/IMG_Object kind

**Current limitation explicitly noted in PR body:**
> "For now only the first compatible image in the binary is loaded. While it might be desirable to add support for loading multiple images, our current interface is limiting (returns a single Image) and it's unclear if in all cases this behavior is desirable... better in a follow-up PR."

**Review timeline:**
- 2026-03-12: `sarnex` (Intel, L0 maintainer) commented
- 2026-03-12/13: `jhuber6` (AMD, offload lead) multiple comments
- 2026-04-07: `adurang` pinged `jhuber6`: "what do you want to do with this? I don't mind working on this but it's not at the top of my list."
- No LGTM/approval yet — 28 days open with no merge decision

**Relevance to libkdl:** This is the upstream "fat binary" dispatch path. The `isMetadataCompatible()` hook is structurally identical to what libkdl implements via `kdl_match()`. libkdl goes further: multi-image scoring (not just first-match), runtime device feature queries, and dynamic loading without fat-binary packaging. PR #186088's "first compatible image" limitation is precisely the gap libkdl fills.

---

## 2. PR #185663 — [OFFLOAD] Add interface to extend image validation

**Status: MERGED 2026-03-10**

- **URL:** https://github.com/llvm/llvm-project/pull/185663
- **Author:** Alex Duran (`adurang`)
- **Merged:** 2026-03-10T17:41:24Z
- **Label:** `offload`

**What it does:**
Adds a plugin-level extension hook `validateImage()` so plugins can validate image formats not recognized by the common layer. Used by L0 to validate pure SPIR-V images. This is the prerequisite for #185404.

**Context:** Part of the 3-PR chain: #185663 → #185404 → #186088.

---

## 3. The OffloadBinary PR Chain — Full Map (Since Feb 2026)

| PR | Title | Status | Date |
|----|-------|--------|------|
| [#185663](https://github.com/llvm/llvm-project/pull/185663) | [OFFLOAD] Add interface to extend image validation | MERGED | 2026-03-10 |
| [#185404](https://github.com/llvm/llvm-project/pull/185404) | [Offload][L0] Add support for OffloadBinary format in L0 plugin | MERGED | 2026-03-11 |
| [#185413](https://github.com/llvm/llvm-project/pull/185413) | [llvm][offload] Change Intel's SPIRV wrapper from ELF to OffloadBinary | MERGED | 2026-03-11 |
| [#186088](https://github.com/llvm/llvm-project/pull/186088) | [OFFLOAD] Generalize support for OffloadBinary images (all plugins) | **OPEN** | 2026-03-12 |
| [#185425](https://github.com/llvm/llvm-project/pull/185425) | [llvm][tools] Extend llvm-objdump to support nested OffloadBinaries | **OPEN** | 2026-03-09 |
| [#184774](https://github.com/llvm/llvm-project/pull/184774) | [llvm][tools] Add support to llvm-offload-binary to unbundle images | **OPEN** | 2026-03-05 |

**Pattern:** The chain is partially merged — the L0-specific pieces landed, but the generalization to CUDA/AMDGPU (the part that would affect libkdl's target plugins) is stalled at #186088.

---

## 4. New Related PRs Since March 2026 — Offload Label

Notable PRs merged or open since 2026-03-01 relevant to dispatch/runtime:

| PR | Title | Status | Date |
|----|-------|--------|------|
| [#190814](https://github.com/llvm/llvm-project/pull/190814) | [Offload] Disable RPC doorbell queries on some ISAs | OPEN | 2026-04-07 |
| [#190708](https://github.com/llvm/llvm-project/pull/190708) | [PGO][AMDGPU] Add uniformity-aware offload profile format and instrumentation | OPEN | 2026-04-06 |
| [#190588](https://github.com/llvm/llvm-project/pull/190588) | [offload] Fix kernel record/replay and add extensible mechanism | OPEN | 2026-04-06 |
| [#190319](https://github.com/llvm/llvm-project/pull/190319) | offload: Parse triple using to identify amdgcn-amd-amdhsa | MERGED | 2026-04-03 |
| [#189731](https://github.com/llvm/llvm-project/pull/189731) | [Offload] Run liboffload unit tests as a part of check-offload | MERGED | 2026-04-01 |
| [#188485](https://github.com/llvm/llvm-project/pull/188485) | [Offload] Enable multilib building for OpenMP/Offload | MERGED | 2026-03-26 |
| [#188067](https://github.com/llvm/llvm-project/pull/188067) | [libc] Support AMDGPU device interrupts for the RPC interface | MERGED | 2026-03-24 |
| [#187597](https://github.com/llvm/llvm-project/pull/187597) | [OFFLOAD] Improve resource management of the plugin | MERGED | 2026-03-25 |
| [#187602](https://github.com/llvm/llvm-project/pull/187602) | [OpenMP][cuda][HIP] Support for external dev of GPU-INITIATED functions via offload RPC | OPEN | 2026-03-19 |
| [#186972](https://github.com/llvm/llvm-project/pull/186972) | [HIPSPV] Add in-tree SPIR-V backend support for chipStar | OPEN | 2026-03-17 |
| [#186856](https://github.com/llvm/llvm-project/pull/186856) | [Offload] Add support for measuring elapsed time between events | MERGED | 2026-04-01 |
| [#186261](https://github.com/llvm/llvm-project/pull/186261) | [OpenMP] Emit aggregate kernel prototypes and remove libffi dependency | MERGED | 2026-03-20 |

**Dispatch-relevant:** None of the new PRs implement "variant selection" or "rankImage" semantics. The closest is #190319 (better triple parsing for AMDGPU images) and #190588 (extensible kernel record/replay). No PR since March 2026 has introduced scoring/ranking of multiple compatible images.

---

## 5. Competing RFC Proposals — Discourse Search

**No direct competing RFC found** for GPU runtime kernel dispatch/image selection in the 2026 timeframe.

What exists on discourse.llvm.org:
- **[RFC] Introducing llvm-project/offload** (thread #74302) — foundational RFC, ongoing updates
- **[RFC] llvm-project/offload roadmap** (thread #75611) — roadmap items including image loading; post #15 by `jdoerfert` mentions multi-image handling as future work but no formal RFC
- **[RFC] Cleaning the GPU dialect** (thread #88170, Sept 2025) — MLIR gpu dialect restructuring, not runtime dispatch
- **[RFC] SYCL Kernel Lowering** (thread #74082) — frontend lowering, not runtime

**GPU/Offloading Workshop 2025 (Oct 2025, US Dev Meeting):** Slides posted to discourse thread #88832. The Huber (AMD) talk PDF at `llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf` covers LLVM/Offload infrastructure but is focused on overall architecture, not specifically image variant selection.

**Assessment:** No competing upstream RFC specifically proposes a "rankImage" or scored multi-variant dispatch mechanism. The upstream approach (PR #186088) uses first-match with metadata pre-filtering — a clear gap vs. libkdl's scoring model.

---

## 6. EuroLLVM 2026 Dublin Program — Poster Session

**Event:** April 13–15, 2026 — Clayton Hotel Burlington Road, Dublin
**Poster session:** Wednesday April 15, 3:15–4:15 PM

**GPU/offload/heterogeneous sessions confirmed in the program:**

### MLIR Workshop (Monday April 13)
| Time | Talk |
|------|------|
| 9:05-9:30 AM | CUDA Tile IR |
| 10:00-10:30 AM | Auto-tuning MLIR schedules for Intel GPUs |
| 1:00-1:30 PM | From Graphs to Warps: Semantic Interoperability |
| 3:00-3:30 PM | MLIR-RAJA: Bridging AI Models and HPC |
| 3:30-3:55 PM | Training-Aware Compilation for Custom AI Accelerators |

### Main Conference (Tue-Wed)
| Time | Talk | Speaker |
|------|------|---------|
| Tue 1:45-2:15 PM | rocMLIR: High-Performance ML Compilation for AMD GPUs with MLIR | Pablo Martinez |
| Tue 11:00 AM-12:00 PM | Creating a runtime using the LLVM_ENABLE_RUNTIMES system | Michael Kruse |
| Tue 5:15-5:45 PM | Writing a Formal Execution and Memory Model for AMD GPU Sync Primitives | Pierre van Houtryve |
| Wed 4:45-5:45 PM | HIVM: MLIR Dialect Stack for Ascend NPU Compilation | (unlisted) |

**Poster session specifics:** The agenda page does not enumerate individual poster titles — the poster board list is not publicly published on the web-accessible schedule as of 2026-04-09. The confirmed format is an open poster hall during the Wednesday afternoon break.

**No other poster on heterogeneous kernel dispatch or GPU runtime dispatch confirmed.** The closest thematically competing talk is "Creating a runtime using the LLVM_ENABLE_RUNTIMES system" (Kruse, Tue) — a tutorial, not a poster, and focused on build infrastructure not dispatch semantics.

---

## Summary Assessment

| Question | Finding |
|----------|---------|
| PR #186088 status | OPEN, stalled — no approval after 28 days; `adurang` explicitly deprioritized it |
| PR #185663 status | MERGED 2026-03-10 |
| Related new PRs (image dispatch) | None implement scoring/ranking; upstream remains first-match only |
| Competing RFC (Discourse) | None found for variant selection / rankImage / dynamic dispatch scoring |
| EuroLLVM poster competition | No confirmed competing poster on heterogeneous kernel dispatch; GPU talks present but different angles |

**Strategic implication for libkdl:**
The upstream OffloadBinary chain (especially the stalled #186088) validates the problem space but confirms the gap. The "first compatible image wins" limitation noted in #186088's body is precisely libkdl's contribution. The poster faces no direct upstream proposal competition as of April 9, 2026.

---

## Sources

- https://github.com/llvm/llvm-project/pull/186088
- https://github.com/llvm/llvm-project/pull/185663
- https://github.com/llvm/llvm-project/pull/185404
- https://github.com/llvm/llvm-project/pull/185413
- https://github.com/llvm/llvm-project/pull/185425
- https://github.com/llvm/llvm-project/pull/184774
- https://llvm.swoogo.com/2026eurollvm/agenda
- https://discourse.llvm.org/t/rfc-llvm-project-offload-roadmap/75611
- https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832
- https://llvm.org/devmtg/2026-04/
