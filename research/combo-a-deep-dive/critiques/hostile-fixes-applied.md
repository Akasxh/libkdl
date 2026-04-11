# Hostile Review (R5) — Fixes Applied

**Source:** review-hostile-r5.md (2026-04-10)
**Applied:** 2026-04-10

---

## Content Fixes Applied

### 1. IREE #50 Conflation (Attack 1, Danger 9/10)

**Problem:** Poster and paper cited IREE Issue #50 as evidence of "ranked selection unimplemented for 6 years." Issue #50 is about device selection policy, not kernel binary selection.

**Files changed:**
- `poster/poster-combo-a.html` line 1151: Changed `(#50 open 6yr)` to `(PR #186088 defers ranked selection)` in the IREE row of the Related Work table.
- `research/combo-a-deep-dive/proposals/extended-abstract-v3.tex` line 303: Removed #50 cite and "6 years" claim. Replaced with reference to issues #12230, #15334 (more relevant) and PR #186088 (directly analogous).
- `research/combo-a-deep-dive/proposals/qa-cards-final.md` Q1: Removed #50 reference and "up to six years" claim. Replaced with PR #186088 cite.
- `research/combo-a-deep-dive/proposals/qa-cards-final.md` summary table row 1: Updated key phrase.

### 2. OffloadBinary Format Qualification (Attack 3, Danger 7/10)

**Problem:** Poster claimed "LLVM OffloadBinary format" without noting the PoC implements a simplified subset missing `ImageKind`, `OffloadKind`, and `Flags` fields.

**Files changed:**
- `poster/poster-combo-a.html` line 1086: Changed "LLVM OffloadBinary format (magic `0x10FF10AD`, 14,064 bytes)" to "LLVM OffloadBinary format (compatible magic `0x10FF10AD`, simplified entry layout; 14,064 bytes)".

### 3. Single-GPU Scope Note (Attack 5, Danger 7/10)

**Problem:** No explicit limitations/scope note on poster despite all measurements being from a single consumer GPU.

**Files changed:**
- `poster/poster-combo-a.html`: Added scope note after Key Findings item 4: "Measured on GTX 1650 (sm_75). Layer fractions are hardware-specific; the decomposition methodology generalizes. AMD path validated via mocked HIP; physical ROCm pending."

---

## Q&A Preparation (Cards Appended)

Three new cards added to `qa-cards-final.md` as Category 6 (Hostile Reviewer Defenses):

| Card | Attack | Danger | Core Defense |
|------|--------|--------|-------------|
| Q26 | IREE #50 conflation | 9/10 | Concede error, pivot to PR #186088 as the directly analogous cite |
| Q27 | "No LLVM code" / vaporware | 8/10 | Two concrete contributions + one labeled design sketch; poster session starts the upstream conversation |
| Q28 | OffloadBinary format wrong | 7/10 | Acknowledge immediately; selection mechanism is format-field-independent |

---

## Attacks NOT Fixed (Q&A Preparation Only)

These attacks are valid but do not require content changes — they require rehearsed verbal responses:

- **Attack 2 (no MLIR C++):** Poster already labels C3 as "Design Sketch" with "Zero lines of MLIR C++ exist." No content change needed. Q27 card prepared.
- **Attack 4 (single GPU hardware):** Addressed by scope note (Fix 3). Hardware limitation is real and acknowledged.
- **Attack 6 (5 keys is trivial):** Design discipline argument is correct defense. No content change needed.
- **Attack 7 (inconsistent numbers):** Extended abstract already uses pinned 3-run medians. Poster numbers are internally consistent.
- **Attack 8 (trivial linear scan):** 3 entries is the realistic case. No content change needed.
- **Attack 9 (microbenchmark measures wrong thing):** Both 6 ns and 380 ns are reported. No content change needed.
- **Attack 10 (D127686 precedent):** Process risk acknowledged. Q20 card already covers this.
