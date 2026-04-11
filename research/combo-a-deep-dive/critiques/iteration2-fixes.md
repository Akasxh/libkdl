# Iteration 2 Fixes Applied

**Date:** 2026-04-10
**Source:** iteration2-verification.md (3 MAJOR findings)

---

## MAJOR #1: Poster numbers updated to pinned data (match paper)

**File:** `poster/poster-combo-a.html`

| Location | Old (unpinned) | New (pinned) |
|----------|---------------|--------------|
| Line 952: flame graph | 42.7 us, 72% | 36.0 us, 90% |
| Line 984: stat box hot-path | 4.0 us | 4.1 us |
| Line 988: stat box cold load | 42.7 us | 36.0 us |
| Line 989: stat box description | ~1/23,000th | ~1/28,000th |
| Line 1039: table cold median | 42.7 us | 36.0 us |
| Line 1039: table cold notes | 100 exec-child trials | 100 exec-child trials, CPU-pinned |
| Line 1044: table hot-path total | 4.0 us | 4.1 us |
| Line 1115: Key Findings paragraph | 72% ... 42.7 us | ~90% ... 36.0 us of ~40.1 us end-to-end |
| Line 980: caption | (no pinning info) | Added "CPU-pinned (taskset -c 0)" and "3-run cross-run medians" |

**Rationale:** Paper (extended-abstract-v3.tex) uses pinned 3-run cross-run medians (35,953 ns cold, 4,104 ns hot). Poster was still showing unpinned single-run medians. Now both documents use the same dataset.

## MAJOR #2: Removed IREE #50 and "6 years" from proposal-v2.md

**File:** `proposals/proposal-v2.md`

| Line | Old | New |
|------|-----|-----|
| 385 | `Partial (issues #50, #12230, #15334 open 6 years)` | `Partial (issues #12230, #15334; PR #186088 defers ranked selection)` |
| 393 | `Ranked selection incomplete after 6 years.` | `Ranked selection remains an open design question (issues #12230, #15334).` |
| 454 | `issues open 6 years` | `ranked selection deferred per PR #186088` |

**Rationale:** IREE issue #50 was a meta-tracker, not a ranked-selection issue. The "6 years" claim conflated unrelated issues. Replaced with specific issue numbers and PR #186088 reference, matching the paper's Related Work section.

## MAJOR #3: Fixed vendor-neutral capability token in poster

**File:** `poster/poster-combo-a.html`

| Line | Old | New |
|------|-----|-----|
| 921 | `"tensor_core,bf16"` | `"tensor_core_nv,bf16"` |

**Rationale:** The `_nv` vendor suffix prevents false cross-vendor capability matches (e.g., NVIDIA tensor cores vs AMD matrix cores). Paper, RFCs, and proposal already had the correct `tensor_core_nv` token. Poster was the only document with the stale vendor-neutral form.

---

## Verification

```
42.7 refs in poster:          0 (PASS)
#50/6yr refs in proposal:     0 (PASS)
tensor_core,bf16 in poster:   0 (PASS)
36.0 refs in poster:          4 (correct: flame, stat box, table, key findings)
4.1 us refs in poster:        2 (correct: stat box, table)
tensor_core_nv in poster:     1 (correct: metadata table)
~90% in poster:               1 (correct: key findings)
PR #186088 in proposal:      10 (correct: existing + 3 new refs)
```

All three MAJOR findings resolved. Expected post-fix score: 4.2-4.3 / 5 (per verification report).
