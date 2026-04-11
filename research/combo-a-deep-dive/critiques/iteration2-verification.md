# Iteration 2 — Adversarial Verification Report

**Reviewer:** Critic (Iteration 2 verification pass)
**Date:** 2026-04-10
**Mode:** THOROUGH (no escalation to ADVERSARIAL — no CRITICAL findings, <3 MAJOR findings)
**Scope:** Poster (`poster/poster-combo-a.html`), Paper (`proposals/extended-abstract-v3.tex`), Proposal (`proposals/proposal-v2.md`), RFCs, Q&A cards

---

## Pre-commitment Predictions

Before investigation, predicted the 5 most likely residual issues after a 45-issue iteration 1:

1. **Number inconsistency across documents** — Hot-path and cold-path figures likely still diverge between poster (visual rounding) and paper (precise pinned data). *CONFIRMED — see Major #1, Minor #1.*
2. **IREE #50 reference not fully purged** — Given the breadth of files, some reference likely survived. *CONFIRMED — see Major #2.*
3. **`tensor_core` vs `tensor_core_nv` inconsistency** — The vendor-specific token fix may not have reached the poster. *CONFIRMED — see Major #3.*
4. **XeVM `device` vs `chip` regression** — MLIR IR examples might have inconsistent parameter names. *NOT FOUND — all `.tex` and poster MLIR examples consistently use `chip`. Clean.*
5. **PR #186088 author regression** — The Huber-to-Duran fix may not have propagated everywhere. *NOT FOUND — `extended-abstract-v3.tex:342` correctly says `Duran, A.` Clean.*

Hit rate: 3/5 predictions confirmed. The two clean items (XeVM chip, PR author) verify that those iteration-1 fixes landed correctly.

---

## Verification Results

### Check 1: Number Consistency Across Poster + Paper + Proposal

#### Hot-path total

| Document | Value | Source Data |
|----------|-------|-------------|
| **Poster** stat box (line 984) | **4.0 µs** | — |
| **Poster** table (line 1044) | **4.0 µs** | — |
| **Paper** (line 201) | **4,104 ns** (4.1 µs) | Pinned 3-run cross-run median |
| **Proposal** (line 239) | **4,048 ns** (4.0 µs) | Unpinned single-run median |
| **Q&A cards** (line 18) | **4.26 µs** (4,257 ns) | Layer-benchmark-results.md (unpinned mean) |
| **Pinned benchmark data** | **4,104.0 ns** median | Cross-run median of 3 pinned runs |

**Verdict:** The paper correctly uses pinned data (4,104 ns). The poster rounds to 4.0 µs — this is 4,104 rounded down, which is aggressive (should be 4.1 µs to match the paper). The Q&A cards use 4.26 µs which comes from a *different statistic* (unpinned mean, not pinned median). The proposal uses 4,048 ns (unpinned median). **Three different numbers across four documents.** The hostile reviewer in R5 flagged this exact issue (review-hostile-r5.md line 125: "The poster should use 4.3 us, not 4.0") — the fix was NOT applied.

#### Cold module load (`cuModuleLoadData`)

| Document | Value | Source Data |
|----------|-------|-------------|
| **Poster** flame graph (line 952) | **42.7 µs** | Unpinned single-run median |
| **Poster** stat box (line 988) | **42.7 µs** | Same |
| **Poster** table (line 1039) | **42.7 µs** | Same |
| **Poster** Key Findings (line 1115) | **42.7 µs** | Same |
| **Paper** cold table (line 180) | **35,953 ns** (~36.0 µs) | Pinned 3-run cross-run median |
| **Paper** figure caption (line 249) | **~40.1 µs** | Cold end-to-end (all layers) |
| **Proposal** (line 225) | **42,670 ns** (42.7 µs) | Unpinned single-run |
| **Q&A cards** (line 16) | **42.7 µs** | Unpinned |
| **Pinned benchmark data** (line 79) | **35,952.7 ns** median | Cross-run median of 3 pinned runs |

**Verdict:** The paper correctly switched to pinned data (35,953 ns). The poster still uses the unpinned figure (42.7 µs). The Q&A cards also use unpinned. **The paper and poster now report different cold-path numbers** — 36.0 µs vs 42.7 µs. This is the exact inconsistency the iteration-1 academic reviewer flagged. The paper fix landed but the poster was not updated to match.

#### Selection overhead

| Document | Value | Source |
|----------|-------|--------|
| **Poster** (line 975, 992, 1045) | **6 ns** / **4-6 ns** | PoC OffloadBinary microbench |
| **Paper** (line 218) | **3 ns** | PoC (100K iterations, 3 pinned runs) |
| **Proposal** (line 240) | **4-6 ns** | PoC |
| **Q&A cards** (line 14) | **6 ns** | PoC OffloadBinary mode |
| **Pinned benchmark** (line 184) | **3 ns** (all 3 runs) | Cross-run: 3 ns, CV 0.0% |

**Verdict:** Paper says 3 ns (pinned). Poster says 4-6 ns (unpinned/range). Q&A says 6 ns. These are not contradictory (3 ns is the pinned amortized value; 4-6 ns is the range across modes), but the presentation is confusing. No one will understand why the paper says 3 and the poster says 6. Both are defensible individually but the inconsistency is an attack surface.

#### cuLaunchKernel / cuStreamSync

| Metric | Poster | Paper | Pinned Data |
|--------|--------|-------|-------------|
| cuLaunchKernel | 1.6 µs | 1,650 ns | 1,650 ns (pinned median) |
| cuStreamSync | 2.5 µs | 2,454 ns | 2,454 ns (pinned median) |

**Verdict:** These are consistent. The poster rounds (1.6 and 2.5 match 1,650 and 2,454 ns). **Clean.**

#### Percentage claims

| Document | cuModuleLoadData % of cold total |
|----------|----------------------------------|
| **Poster** (line 952) | **72%** (42.7 / ~59.4 total?) |
| **Paper** (line 180) | **89.6%** (35,953 / ~40,120 total) |

**Verdict:** The poster says cuModuleLoadData is "72% of total cold dispatch latency" (line 1115). The paper says 89.6%. These cannot both be right. The poster's 72% appears to be 42.7/59.4 (where 59.4 is the full cold end-to-end including process startup), while the paper's 89.6% is 35,953/40,120 (just the layers). **Different denominators, presented as the same metric.** This will confuse anyone cross-referencing the two.

---

### Check 2: IREE #50 Removal

#### Outward-facing files (poster, paper, Q&A):

| File | `#50` present? | `6 years`/`six years` present? |
|------|---------------|-------------------------------|
| `poster/poster-combo-a.html` | **NO** | **NO** |
| `proposals/extended-abstract-v3.tex` | **NO** | **NO** |
| `proposals/qa-cards-final.md` | **YES** (line 358, Q26 title) | **YES** (Q26, but it's a hostile question to *defend against*) |

**Verdict on outward-facing:** The poster and paper are clean. The Q&A card Q26 still contains `#50` and `6 years` — but this is correct: Q26 is a *hostile question card* that prepares the author for an attack about the old `#50` claim. The card's answer says "Do NOT say: IREE issue #50 proves they haven't done this." This is defensive preparation, not an outward claim. **Acceptable.**

#### Internal/non-outward files:

| File | Status |
|------|--------|
| `proposals/proposal-v2.md` line 385 | **STILL HAS** `#50, #12230, #15334 open 6 years` |
| `proposals/proposal-v2.md` line 393 | **STILL HAS** `Ranked selection incomplete after 6 years` |
| `proposals/proposal-v2.md` line 454 | **STILL HAS** `issues open 6 years` |
| `proposals/visitor-personas.md` lines 84, 88, 102 | Has `#50` and `6 years` |
| `proposals/refined-design-v1.md` line 718 | Has `issues open 6 years` |
| `proposals/extended-abstract.tex` line 387 | Has `#50, #12230, #15334` and `6 years` |
| `poster/poster.html` (OLD, stale) | Has `#50` |
| `poster/index.html` (OLD, stale) | Has `#50` |
| `poster/slides.tex` (OLD, stale) | Has `#50` |
| `poster/pitch-script.md` | Has `#50` |

**Verdict:** The two outward-facing files that matter (poster-combo-a.html and extended-abstract-v3.tex) are clean. However, `proposal-v2.md` — which is a current document, not stale — still contains the problematic `#50` and `6 years` claims in three places. If this proposal is shared with anyone (e.g., a reviewer, a Discourse post), the IREE conflation attack surface is still present.

---

### Check 3: XeVM `chip` vs `device`

Grep results:
- `device = ` in `.tex` files: **0 matches**
- `chip = ` in `.tex` files: 6 matches (all correct: `chip = "sm_75"`, `chip = "gfx90a"`, `chip = "pvc"`, `chip = "sm_90"`)
- Poster HTML: uses `chip=` consistently
- Both RFCs: use `chip =` consistently

**Verdict: CLEAN.** All MLIR IR examples consistently use `chip` parameter. Fix verified.

---

### Check 4: PR #186088 Author

`extended-abstract-v3.tex` line 342: `Duran, A.`

**Verdict: CLEAN.** The bibitem correctly attributes PR #186088 to Alex Duran. Fix verified.

---

### Check 5: "Roofline Model" Ban

| File | `roofline` present? |
|------|-------------------|
| `poster/poster-combo-a.html` | **NO** |
| `proposals/extended-abstract-v3.tex` | **NO** |
| `proposals/qa-cards-final.md` | **YES** — but only in defensive context: Q8 says "NOT roofline" and "Do NOT say: Roofline model" |
| `proposals/rfc-FINAL.md` | **NO** |
| `proposals/rfc-runtime-select.md` | **NO** |
| `proposals/proposal-v2.md` line 369, 488, 513 | **YES** — but only in risk register ("FIXED: Renamed to weighted heuristic") and correspondence table (explicitly says "NOT a roofline model") |

**Verdict: CLEAN.** No outward-facing file uses "roofline model" as a claim. All remaining instances are defensive (risk register, Q&A defense cards). Fix verified.

---

## Findings

### Major Findings

#### MAJOR #1: Poster uses unpinned data while paper uses pinned data — numbers now DISAGREE

- **Confidence:** HIGH
- **Evidence:**
  - Poster: `42.7 µs` cold module load (4 occurrences: lines 952, 988, 1039, 1115), `4.0 µs` hot-path (lines 984, 1044)
  - Paper: `35,953 ns` (~36 µs) cold (line 180), `4,104 ns` (4.1 µs) hot-path (line 201)
  - Source: Poster uses unpinned single-run median; paper uses pinned 3-run cross-run median
- **Why this matters:** The iteration-1 academic solver correctly fixed the paper to use pinned data. But the poster was not updated. Anyone reading both will see 42.7 µs in the poster and ~36 µs in the paper for the same metric. The hostile reviewer explicitly identified this (review-hostile-r5.md line 119). This is exactly the kind of inconsistency that destroys credibility in a poster session when someone is reading the paper in one hand and looking at the poster with the other.
- **Fix:** Pick one dataset and use it everywhere. The pinned 3-run cross-run medians are the higher-quality data. Update the poster to:
  - Cold `cuModuleLoadData`: 36.0 µs (from 42.7 µs)
  - Hot-path total: 4.1 µs (from 4.0 µs)
  - Flame graph percentage: recalculate with pinned total (~40.1 µs base → 36.0/40.1 = 89.6%)
  - Update Q&A cards definitive numbers table to match

#### MAJOR #2: `proposal-v2.md` still contains IREE #50 and "6 years" claims (3 instances)

- **Confidence:** HIGH
- **Evidence:**
  - Line 385: `Partial (issues #50, #12230, #15334 open 6 years)`
  - Line 393: `Ranked selection incomplete after 6 years.`
  - Line 454: `IREE HAL (module-level, issues open 6 years)`
- **Why this matters:** The proposal is a current document (v2, dated 2026-04-09). If used as source material for a Discourse RFC post, blog post, or shared with reviewers, the IREE #50 conflation attack surface survives. The paper and poster were fixed but the proposal was missed.
- **Fix:** In `proposal-v2.md`:
  - Line 385: Remove `#50` from the issue list, change to `(issues #12230, #15334; PR #186088 defers ranked selection)` — matching the paper's Related Work section
  - Line 393: Remove "after 6 years" or replace with "remains an open design question (issues #12230, #15334)"
  - Line 454: Same treatment — remove "6 years"

#### MAJOR #3: Poster uses `tensor_core` (not `tensor_core_nv`) in the metadata vocabulary table

- **Confidence:** HIGH
- **Evidence:**
  - `poster/poster-combo-a.html` line 921: `<code>"tensor_core,bf16"</code>` — missing the `_nv` vendor suffix
  - `extended-abstract-v3.tex` line 121: `"tensor\_core\_nv,bf16"` — correct
  - `proposals/rfc-FINAL.md` line 29: `tensor_core_nv,bf16` — correct
  - `proposals/proposal-v2.md` line 80: `"tensor_core_nv,bf16"` — correct
- **Why this matters:** The iteration-1 fix explicitly changed `tensor_core` to `tensor_core_nv` with a detailed rationale about preventing false cross-vendor capability matches. The paper and RFCs have the correct vendor-specific token. The poster — the most visible document at the conference — still uses the old vendor-neutral `tensor_core`. A questioner who reads the poster's `requires_features` example will see `tensor_core` and immediately ask "does that match across vendors?" — exactly the attack the fix was designed to prevent.
- **Fix:** In `poster/poster-combo-a.html` line 921, change `"tensor_core,bf16"` to `"tensor_core_nv,bf16"`.

---

### Minor Findings

#### MINOR #1: Poster hot-path rounds 4,104 ns down to 4.0 µs instead of 4.1 µs

The hostile reviewer in R5 said "The poster should use 4.3 us, not 4.0" (review-hostile-r5.md line 204). Even with the pinned median (4,104 ns = 4.1 µs), the poster rounds to 4.0. Rounding 4.1 to 4.0 is a 2.5% understatement — minor but avoidable.

#### MINOR #2: Poster cold-path percentage (72%) contradicts paper (89.6%)

The poster says `cuModuleLoadData` is "72% of total cold dispatch latency." The paper says 89.6%. These use different denominators (poster: total including unaccounted overhead; paper: sum of measured layers). Neither is wrong per se, but the discrepancy is confusing. After updating the poster to pinned data, this should be recalculated and one consistent percentage used.

#### MINOR #3: Selection overhead: poster says 6 ns / 4-6 ns, paper says 3 ns

Both are defensible (different measurement modes), but at a poster session where someone is comparing documents, "6 ns" vs "3 ns" looks sloppy. Consider settling on "3-6 ns" consistently, or explain the range.

#### MINOR #4: Context.md claims MLIR sketch is 883 lines; actual count is 889

Line 18 of `context.md`: "883 lines". Actual `wc -l`: 889. Trivial but sloppy if anyone checks.

#### MINOR #5: Q&A definitive numbers use unpinned data

Q&A cards (line 16-18) list `42.7 µs` cold and `4.26 µs` hot-path — both unpinned. If the canonical numbers shift to pinned, the Q&A table needs updating too.

---

### What's Missing (Gap Analysis)

1. **No explicit "which dataset" declaration on the poster.** The poster uses unpinned data without stating it. The paper declares "pinned 3-run medians" in the methodology. The poster should similarly state its data source.

2. **Poster has no methodology footnote on whether data is pinned or unpinned.** The caption at line 980 says "100 cold trials (exec-child isolation), 10K hot iterations" but does not mention CPU pinning. If the data is updated to pinned, add "CPU-pinned (taskset -c 0)" to the caption.

3. **The `extended-abstract.tex` (v2, not v3) still contains the old IREE #50 reference** (line 387). This file appears to be superseded by `extended-abstract-v3.tex`, but if anyone opens the wrong file, they get the stale claims.

---

### Ambiguity Risks

- **"4.0 µs" on poster** → Interpretation A: 4,000 ns exactly. Interpretation B: rounding of 4,104 ns. Risk: someone back-calculates 1.6+2.5=4.1 and wonders why it says 4.0. Mismatch is small but noticeable.

- **"72% of cold dispatch" on poster vs "89.6%" in paper** → Interpretation A: different benchmarks. Interpretation B: someone did bad math. Risk: credibility hit if noticed.

---

### Multi-Perspective Notes

- **Executor perspective:** If I need to fix these issues, I need to: (1) update 4 numbers in the poster HTML (cold, hot, percentage, selection range), (2) change one string in the poster (`tensor_core` → `tensor_core_nv`), (3) edit 3 lines in proposal-v2.md. Total: ~30 minutes of work. Straightforward.

- **Stakeholder perspective:** The paper is in good shape — pinned data, CV%, separated measurements, qualified claims. The poster is the weak link: it's the most visible artifact and has the most stale numbers. The Q&A cards track the poster (unpinned), not the paper (pinned), so the verbal defense will cite different numbers than the paper. Fix the poster first, Q&A second.

- **Skeptic perspective:** The core work is sound. Three concrete contributions, measured on real hardware, honest about limitations (design only for C3, custom format for prototype, AMD mocked). The issues are all presentation consistency, not substance. No structural problems.

---

## Score: Complete Package

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| **Novelty** | 3.5 | Metadata vocabulary is incremental but needed. Layer decomposition is genuinely first-published. Design sketch is future work. Honest framing helps. |
| **Soundness** | 4.0 | Paper methodology is rigorous (pinned, 3-run, CV%, footnoted caveats). Poster numbers lag the paper — fixable. Prototype honestly framed. No fabricated claims survived. |
| **Significance** | 3.5 | Real problem (HEP-CCE 80-build, vLLM split codepaths). Solution is narrow (~30 LOC header patch + ~40 LOC runtime extension). Impact is bounded but practical. |
| **Presentation** | 3.5 | Paper is clean and well-structured. Poster design is strong (A0, visual hierarchy, flame graph). But number inconsistencies between poster and paper are a presentation failure. The `tensor_core` vs `tensor_core_nv` gap undermines the "vendor-specific by design" pitch. |
| **Reproducibility** | 4.0 | Real hardware (GTX 1650), real benchmarks (bench_layers, bench_dispatch, runtime_select_poc), LOC counts verified (664/889/319/5157 all within range of claims). Pinned data with cross-run statistics. Protocol documented. |
| **Overall** | **3.7 / 5** | Down from estimated 4.3 because the iteration-1 fixes created a *new* problem: the paper and poster now disagree on core numbers. Fixing the 3 MAJOR issues (poster data update, proposal IREE cleanup, tensor_core_nv) would bring this to **4.2-4.3**. |

---

## VERDICT: REVISE

### Overall Assessment

The iteration-1 fixes correctly addressed the paper (pinned data, separated measurements, qualified claims, PR author, IREE #50 removal). But the fixes were not propagated to the poster — creating a new inconsistency that is worse than the original problem. Before iteration 1, everything used unpinned data (internally consistent but lower quality). Now the paper uses pinned and the poster uses unpinned — externally inconsistent. The `tensor_core_nv` fix similarly landed in the paper and RFCs but not the poster. These are all straightforward to fix (~30 minutes), but they must be fixed before the poster session.

### Verdict Justification

Three MAJOR findings, all HIGH confidence, all with specific evidence. None involve substance or methodology — all are fix-propagation failures. Review stayed in THOROUGH mode (no CRITICAL findings, exactly 3 MAJOR findings, no systemic pattern — these are consistent with a "solver didn't update all files" failure mode, not a deeper problem).

Realist check: all three MAJOR findings are real and would be noticed at a poster session. The number inconsistency (Major #1) is the highest-risk: a visitor comparing the paper handout to the poster will immediately notice 42.7 vs 36.0 µs. The `tensor_core` issue (Major #3) will be noticed by anyone who reads the paper first, where `tensor_core_nv` is explicitly motivated with a rationale. No mitigating factors downgrade these — a poster session is exactly the environment where cross-document consistency matters most.

### What would change verdict to ACCEPT

1. Update poster-combo-a.html cold-path to pinned data (36.0 µs) and hot-path to 4.1 µs
2. Fix `tensor_core` → `tensor_core_nv` in poster line 921
3. Remove IREE #50 and "6 years" from proposal-v2.md (3 lines)
4. Recalculate poster percentage to match pinned data (89.6% or recalculated denominator)
5. Update Q&A definitive numbers to match chosen dataset

After these fixes: ACCEPT-WITH-RESERVATIONS (the minor inconsistencies in selection overhead range and rounding are livable).

---

## Open Questions (Unscored)

1. **Which cold-path number is the "canonical" one for verbal defense?** The pinned 3-run cross-run median (35,953 ns ≈ 36.0 µs) or the cold end-to-end including all layers (~40.1 µs)? The poster's stat box should show one number; the verbal answer should explain the other.

2. **Should the poster caption declare "pinned" methodology?** The paper does. The poster's caption at line 980 does not mention CPU pinning. Adding "CPU-pinned (taskset -c 0)" would make the methodology transparent.

3. **The `extended-abstract.tex` (non-v3) file still has stale data.** Is this file dead or might someone accidentally use it? Consider deleting or adding a "SUPERSEDED" header.
