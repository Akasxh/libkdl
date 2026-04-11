# Combo A Reviewer Panel — EuroLLVM Dublin 2026 Poster Track

**Submission:** "Runtime Variant Selection for LLVM GPU Offloading"
**Components:** T01 (`gpu.select_variant`), T07 (OffloadBinary metadata keys), T19 (dispatch flame graph)
**Authors:** Akash (IIT Patna)
**Review date:** 2026-04-09
**Review mode:** THOROUGH (no escalation to ADVERSARIAL warranted — issues found are fixable, not systemic)

---

## Review 1: Novelty (3.5 / 5)

### What is new

The gap is real and well-documented. No MLIR-native runtime variant selection mechanism exists today. `#gpu.select_object` is compile-time-only, and the RFC "Cleaning the GPU Dialect" (#88170) explicitly leaves the dispatch-policy slot vacant. The OffloadBinary string table has only two standard keys (`triple`, `arch`) since D122069 in 2022 — four years with no vocabulary extension. The per-layer latency decomposition of the LLVM offload stack is genuinely unpublished.

### What is not new

The *concept* of runtime multi-binary dispatch is well-established:

1. **IREE's HAL** has done multi-target dispatch since 2019. The open issues (#50, #12230, #15334) show incomplete *ranked* selection — but the basic mechanism of selecting among compiled variants at runtime exists and ships. The distinction (HAL module granularity vs. kernel granularity) is real but narrow.

2. **CPU Function Multi-Versioning** (`target_clones`, IFunc resolvers) is the direct structural analogue in LLVM itself. The proposal acknowledges this in the tough-questions document but the topic files do not discuss it. A reviewer who works on FMV will immediately ask why this isn't an extension of the existing resolver infrastructure.

3. **CUDA fatbin's built-in runtime selection** has done exactly this for NVIDIA-only binaries since CUDA 2.0 (~2008). The CUDA driver selects the best cubin from a fatbin based on device SM version at `cuModuleLoadData` time. The proposal extends this concept to cross-vendor selection, but must acknowledge the CUDA precedent explicitly or risk appearing unaware of it.

4. **chipStar, Proteus, IRIS** all provide forms of cross-vendor dispatch. The proposal correctly distinguishes from each (portability vs. peak performance, JIT vs. AOT, etc.) but the differentiation is in supporting documents, not in the topic files themselves.

### Comparison verdict

| System | Runtime selection? | Cross-vendor? | MLIR-native? | Ranked? |
|--------|-------------------|---------------|-------------|---------|
| CUDA fatbin | Yes | No (NVIDIA only) | No | Yes (SM match) |
| IREE HAL | Yes | Yes | MLIR-based | Partial (issues open) |
| chipStar | Yes (via SPIR-V) | Yes | No | No |
| Proteus | Yes (JIT) | Partial | No | No |
| liboffload PR #186088 | Yes | Yes | No | No (first-compatible-wins) |
| **This proposal** | Yes | Yes | Yes | Yes (proposed) |

The novelty is incremental but real: the specific combination of MLIR-native + cross-vendor + ranked selection + OffloadBinary metadata vocabulary is new. The individual pieces are not.

**Score: 3.5/5** — Novel contribution to the MLIR ecosystem specifically, but the broader concept has extensive prior art. Acceptable for a poster; would be weak for a full paper.

---

## Review 2: Technical Soundness (3 / 5)

### T01: `gpu.select_variant` — Mostly sound, one design concern

The lowering strategy through `OffloadingLLVMTranslationAttrInterface` is correct. Verified: `SelectObjectAttr.cpp` implements `embedBinary` + `launchKernel` as a two-method interface, and the proposal's `#gpu.runtime_select` attribute is a clean extension of this pattern. The LLVM IR emission plan (N global arrays + dispatch table + `global_ctors` + indirect call) uses well-established patterns from the NVVM lowering path.

**Design concern:** The `dlopen`-based vendor detection at `global_ctors` time means the dispatch table is populated *before main()*. This creates a static initialization order problem: if multiple translation units each embed a `gpu.select_variant`, the vendor detection runs N times. The proposal does not discuss deduplication or lazy initialization. This is a known C++ problem (static init order fiasco) and the solution (lazy init via `std::call_once` or equivalent) is straightforward — but its absence from the design is a gap.

**Cost model concern:** The prototype's cost model uses hardcoded locality constants (`50e-6` for NVIDIA, `60e-6` for AMD at `kdl.c:1051-1054`). This is acknowledged in the tough-questions document (Q7) but not in the topic file. The paper-outline calls it a "roofline-based estimation" (Section 4.4), which it is not — it is a weighted heuristic with vendor-specific magic numbers. The mismatch between the paper-outline's claim and the prototype's reality would be caught by any reviewer who reads the code.

### T07: OffloadBinary metadata — Sound but incomplete

The four-tier vocabulary is well-structured. The gap analysis is correct: `OffloadBinary.h` has only `getTriple()` and `getArch()`. The `areTargetsCompatible()` function in `OffloadBinary.cpp` hard-codes AMD xnack/sramecc parsing and nothing else.

**Completeness issue:** The `requires_features` token vocabulary maps vendor-specific features to vendor-neutral names (`tensor_core` → CUDA Tensor Core / AMD MFMA / SPIR-V cooperative_matrix). This mapping is lossy: CUDA tensor cores and AMD MFMA have different capabilities (different matrix sizes, different accumulation precision). A fat binary with `requires_features=tensor_core` that was compiled for `sm_90a` Warp Specialization would match on an MI300X — but the kernel wouldn't work because MFMA doesn't support the same intrinsics. The vocabulary needs a way to express vendor-specific capabilities that cannot be cross-mapped, or the "vendor-neutral" abstraction creates a correctness bug.

**The `min_gfx` ordering problem:** The proposal uses `min_gfx` with an arch string like `gfx90a`. But AMDGPU arch strings are not linearly ordered — `gfx1100` (RDNA3) is architecturally incompatible with `gfx90a` (CDNA2). A simple `>=` comparison is incorrect. The proposal does not specify the comparison semantics.

### T19: Dispatch flame graph — Measurement design has a known flaw

The gap analysis already identified this (and credit to the author for the self-audit): the Layer 3 measurement calls `cuModuleLoadData` *separately* from `olCreateProgram`, which double-loads the module. This is not a minor issue — it invalidates the layer decomposition because the driver's module cache may return a cached result for the second load, giving a misleadingly low Layer 3 time.

The measurement harness in `kdl.c` (lines 4595-4649) times `cuStreamSynchronize` as a dispatch proxy, not `cuLaunchKernel`. These measure different things: `cuStreamSynchronize` includes GPU idle detection overhead. The harness is a reasonable foundation but requires extension to match the poster's claims.

**Score: 3/5** — The MLIR extension point usage is technically correct. The OffloadBinary gap is real. But the cost model is misrepresented, the metadata vocabulary has a correctness gap in cross-vendor feature mapping, the `min_gfx` comparison semantics are undefined, and the flame graph measurement design has a known flaw. All fixable, but currently unsound in aggregate.

---

## Review 3: Significance (4 / 5)

### Would LLVM contributors use this?

**Yes, with high confidence for T07.** The OffloadBinary metadata vocabulary fills a gap that every LLVM backend team encounters. The `isMetadataCompatible` consumer hook (PR #185663, merged March 2026) already exists — the vocabulary is the missing producer side. This has immediate practical value for anyone writing a multi-target fat binary pipeline.

**Probably yes for T01.** The three-vendor MLIR GPU landscape (NVVM + ROCDL + XeVM since August 2025) creates real demand for runtime selection. The GPU/Offloading Workshop themes for 2024-2025 repeatedly surface runtime dispatch as an open question. Intel's XeVM team has immediate motivation — without runtime dispatch, their upstream target attribute cannot be used in heterogeneous deployments.

**Moderate for T19.** A measurement study with no code change has lower long-term impact. However, the LLVM community is performance-conscious and "first published numbers" for a previously unmeasured path carries significant visibility. The accfg poster (LLVM DevMtg 2024) set the precedent that dispatch overhead quantification is poster-worthy.

### Is the problem important enough?

Yes. The gap between MLIR's multi-target compilation capability and the absence of runtime dispatch is a real, visible, documented gap. The RFC #88170 discussion and the liboffload PR #186088's "first-compatible-wins" approach both confirm the community recognizes this problem. This is not a manufactured gap.

**Score: 4/5** — Real problem, real community demand, reasonable likelihood of adoption for T07 and T01.

---

## Review 4: Presentation Quality (2.5 / 5)

### "Three contributions in one poster" is too much

This is the single biggest presentation risk. A poster has roughly 3-4 minutes of a passerby's attention. Three distinct contributions — a new MLIR op, a metadata vocabulary standard, and a measurement study — require three different mental models:

1. T01 is a **compiler infrastructure** contribution (MLIR ops, LLVM IR lowering)
2. T07 is a **format specification** contribution (string keys, ABI contracts)
3. T19 is an **empirical measurement** contribution (flame graphs, latency numbers)

Each of these is poster-worthy on its own. Combined, the narrative thread is: "we identified a gap in runtime dispatch, designed the mechanism (T01), defined the metadata it needs (T07), and measured the overhead it must beat (T19)." This narrative works *if* the poster leads with the gap and positions T01/T07/T19 as three panels of a single story. But the current topic files don't have a unified narrative arc — they were written as independent proposals.

### What's the narrative arc?

The paper-outline (paper-outline.md) has a clear thesis — "libkdl as ld.so for GPU kernels" — but the three topics don't converge on that thesis cleanly. T01 proposes an MLIR op. T07 proposes metadata keys. T19 proposes a measurement. The connecting thread ("all three are needed for runtime variant selection") is implicit, not explicit.

**Recommendation:** The poster must open with the gap (one sentence: "MLIR compiles to three GPU vendors but can't select among them at runtime"), then show the three contributions as layers of a single solution (metadata enables selection, the op implements it, the flame graph validates it). The current materials don't have this structure.

### Can this be explained in a poster?

T07 (metadata keys) is highly visual — a table of keys on the poster is immediately legible. T19 (flame graph) is inherently visual — two SVG panels. T01 (MLIR op) is the hardest to present — MLIR syntax and LLVM IR lowering details don't render well on a poster. The proposed MLIR snippet (`gpu.select_variant @kernels attributes {strategy = #gpu.rank_by_device}`) is clean, but the lowering details (global arrays, dispatch tables, `global_ctors`) require a diagram that doesn't exist yet.

**Score: 2.5/5** — Three contributions is ambitious for a poster. Each is individually presentable, but the combined narrative needs a unified structure that doesn't exist in the current materials. This is the highest-priority fix.

---

## Review 5: Reproducibility (2.5 / 5)

### Prototype availability

The prototype (`kdl.c`, ~5157 LOC) exists and is in the repository. It implements vendor detection, dispatch table construction, and binary loading via `cuModuleLoadData`/`hipModuleLoadData`. The build system is `make` in `experiments/prototype/src/`. This is reproducible for the runtime half.

### MLIR half does not exist

`gpu.select_variant` is a proposal — no MLIR C++ implementation exists. `#gpu.runtime_select` has not been written. The 300-500 LOC estimate is plausible (based on `SelectObjectAttr.cpp` as template) but unverified. A poster presenting a proposed op with no implementation will face the "show me the code" challenge from the LLVM community.

### Measurement reproducibility

**Hardware:** GTX 1650 is consumer hardware, widely available. Good for reproducibility.

**The MI300X claim is false.** The topic-07 pitch claims "a prototype already running on GTX 1650 + MI300X under libkdl." The project instructions and all other materials say "GTX 1650 + CPU." No MI300X testing has been done. This is a factual error that undermines reproducibility claims.

**The flame graph does not exist yet.** T19 proposes to produce flame graphs but none have been generated. The measurement code in `kdl.c:4595-4649` times `cuStreamSynchronize`, not the per-layer decomposition the poster proposes. The ~200 LOC extension has not been written.

**No numbers have been measured.** The tough-questions document (Q20) is bracingly honest: `[INSERT MEASURED VALUE]` and `[INSERT MEASURED BASELINE]` appear as placeholders. The poster deadline is 2026-04-07, which has already passed (today is 2026-04-09). If the measurements still have placeholders, the poster is not ready.

### liboffload API instability

The `ol*` API is explicitly unstable. `olGetKernel` was renamed to `olGetSymbol` three months after initial API. Any reproduction attempt must pin to a specific LLVM commit, which is not specified in the materials.

**Score: 2.5/5** — Prototype exists for the runtime half. MLIR half is vaporware. Flame graph is vaporware. Actual numbers have not been measured. MI300X claim is false. The reproducibility story is incomplete.

---

## Overall Score: WEAK ACCEPT

The contribution addresses a genuine gap in the LLVM/MLIR ecosystem. The technical direction is sound — extending `OffloadingLLVMTranslationAttrInterface` is the correct approach, the OffloadBinary metadata vocabulary fills a real need, and the dispatch latency decomposition would be the first of its kind. The research preparation is unusually thorough (the gap analysis, tough-questions document, and self-audit are excellent).

However, the poster is not ready for submission in its current state.

---

## Detailed Comments

### What must be fixed before the poster

1. **Unify the narrative.** The three contributions need a single-page visual story: gap → metadata (T07) → mechanism (T01) → validation (T19). Currently they read as three independent proposals. Without a unified narrative, the poster will confuse rather than convince.

2. **Produce actual measurements.** The poster deadline has passed. Q20 of the tough-questions document has `[INSERT MEASURED VALUE]` placeholders. Run `bench_dispatch` on the GTX 1650 and report: (a) raw `cuLaunchKernel` baseline latency, (b) `kdl_select_kernel` overhead, (c) variant selection as percentage of total dispatch. These are the numbers the poster lives or dies on.

3. **Remove the MI300X claim.** Topic-07's pitch says "GTX 1650 + MI300X." Replace with "GTX 1650 + CPU-fallback." False hardware claims are credibility-destroying.

4. **Fix the XeVM PR number.** PR #119440 is the ELF section pass, not XeVM upstreaming. Find the correct PR (likely #148286 or adjacent). This is a factual error that any LLVM regular will catch.

5. **Fix the TaxBreak attribution.** The "4.71 us H100 floor" figure needs verification against the actual paper. If TaxBreak doesn't measure null-kernel dispatch floor, find the correct source or remove the claim. The entire T19 anchor depends on this number being correctly attributed.

6. **Fix the cost model description.** Do not call the prototype's weighted heuristic a "roofline model" (paper-outline.md Section 4.4). The code uses hardcoded vendor constants (`kdl.c:1051-1054`). Call it what it is: a "weighted heuristic with vendor-specific constants." Misrepresenting this will be caught instantly by anyone who reads the code.

7. **Fix the Layer 3 measurement design.** The proposed code calls `cuModuleLoadData` outside `olCreateProgram`, which double-loads. Either instrument the plugin source or use a separate baseline measurement, as the gap analysis already recommends.

8. **Define `min_gfx` comparison semantics.** AMDGPU arch strings are not linearly ordered (CDNA vs RDNA are incompatible families). The vocabulary proposal must specify how the runtime compares `min_gfx` values, or the Tier 1 keys have a correctness bug.

### What would make this a strong accept

1. **A working `#gpu.runtime_select` attribute**, even as a proof-of-concept that passes one MLIR test. The LLVM community respects code over proposals. Even 200 LOC that emits the dispatch table LLVM IR for a two-target `gpu.binary` would transform this from "interesting idea" to "concrete contribution."

2. **Measured numbers on the poster.** Actual GTX 1650 dispatch latency for: (a) raw `cuLaunchKernel`, (b) `cuLaunchKernel` through liboffload `olLaunchKernel`, (c) variant selection overhead. Three numbers. First published. This alone justifies the poster.

3. **The flame graph SVGs.** Two panels (cold-path, hot-path) with real data. Flame graphs are universally legible and would dominate the visual space of the poster.

4. **An RFC draft for the OffloadBinary vocabulary.** Not submitted — but visible on the poster as "RFC planned" with the vocabulary table printed. This signals seriousness about upstream contribution.

### What should be cut

1. **Cut T01 to a design sketch.** The MLIR op is the most ambitious component and the least prototyped. Present it as a 2-panel design diagram (MLIR syntax + LLVM IR lowering), not as a full contribution. The poster's strength is T07 + T19 — concrete metadata + concrete numbers.

2. **Cut the roofline cost model discussion entirely.** The prototype uses hardcoded constants. Presenting this as a "cost model" invites scrutiny that the current implementation cannot survive. Say "pluggable ranking via `variant_priority` metadata key" and defer cost model sophistication to future work.

3. **Cut any comparison to IREE, chipStar, or Proteus from the poster itself.** These comparisons belong in a paper, not a poster. On the poster, one sentence: "Unlike full-stack solutions (IREE, chipStar), this operates at the LLVM offload layer with ~500 LOC." Keep the detailed comparisons in the paper-outline for a future full paper.

---

## Aggregate Scores

| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 3.5/5 | 25% | 0.875 |
| Technical Soundness | 3.0/5 | 25% | 0.750 |
| Significance | 4.0/5 | 20% | 0.800 |
| Presentation Quality | 2.5/5 | 15% | 0.375 |
| Reproducibility | 2.5/5 | 15% | 0.375 |
| **Weighted Total** | | | **3.175/5** |

---

## Meta-Review: Strengths and Weaknesses

### Strengths (acknowledge briefly)

- The research preparation is genuinely exceptional. The gap analysis (`combo-a-gaps.md`), tough-questions document, and self-audit demonstrate a level of rigor that is unusual for a poster submission. The author has done more due diligence than most published papers.
- The OffloadBinary metadata vocabulary (T07) is the cleanest, most immediately impactful contribution. It fills a documented gap, has a clear upstream path, and the runtime consumer hook (PR #185663) already exists.
- The choice to target `OffloadingLLVMTranslationAttrInterface` as the extension point for T01 is technically correct and shows genuine understanding of the MLIR GPU infrastructure.

### Weaknesses

- **No measured data.** A poster about dispatch overhead with placeholder numbers is not ready for submission.
- **Scope overreach.** Three contributions in one poster dilutes each. T07 + T19 would be a stronger poster than T01 + T07 + T19.
- **Prototype-proposal gap.** The runtime half (C prototype) and the compiler half (MLIR proposal) are disconnected. The poster claims they form a single contribution, but `kdl.c` and `#gpu.runtime_select` share no code, no tests, and no integration path.
- **Factual errors.** The XeVM PR number, MI300X claim, and TaxBreak attribution are all wrong or unverified. These are fixable but indicate rushed preparation.

---

## Reviewer Confidence

- **Reviewer 1 (Novelty):** 4/5 — familiar with MLIR GPU dialect, IREE, chipStar, and LLVM offload ecosystem
- **Reviewer 2 (Technical Soundness):** 3/5 — verified claims against codebase where possible; some LLVM-internal claims rely on secondary sources
- **Reviewer 3 (Significance):** 4/5 — familiar with EuroLLVM poster track expectations and community priorities
- **Reviewer 4 (Presentation):** 3/5 — experience with poster presentations but not LLVM-specific poster norms
- **Reviewer 5 (Reproducibility):** 4/5 — verified prototype code, build system, and measurement harness directly

---

## Recommendation to Program Committee

**WEAK ACCEPT** — conditional on addressing items 1-8 in "What must be fixed."

The gap is real, the direction is correct, and the research depth is impressive. But the poster is not submission-ready: no measured data, three disconnected contributions without a unified narrative, and several factual errors. If the author produces actual GTX 1650 numbers, unifies the narrative, and fixes the factual errors, this becomes a solid poster contribution to EuroLLVM Dublin 2026.

The strongest path forward: lead with T07 (metadata vocabulary) + T19 (flame graph with real numbers), position T01 (`gpu.select_variant`) as the motivating design that shows *why* the metadata and measurements matter, and present `kdl.c` as the proof-of-concept that validates the runtime half.

---

*Review completed: 2026-04-09*
*Reviewer panel: conference paper reviewer simulation*
*Materials reviewed: topic-01, topic-07, topic-19, paper-outline.md, combo-a-gaps.md, combo-a-tough-questions.md, kdl.c (lines 560-590, 1040-1070, 4595-4649)*
