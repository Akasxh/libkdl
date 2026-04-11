# Adversarial Review Loop — Context File

## Project
EuroLLVM Dublin 2026 poster: "Measuring and Improving Multi-Target Binary Selection in LLVM's GPU Offload Stack"

## Current Score: 4.0/5 (target: 4.5+)

## Iteration Log
- Iteration 1: Starting adversarial review loop

## Key Files (for reviewer context)
- Poster: poster/poster-combo-a.html (40KB, A0 standalone)
- Paper: research/combo-a-deep-dive/proposals/extended-abstract-v3.tex (4 pages)
- Proposal: research/combo-a-deep-dive/proposals/proposal-v2.md
- RFC #1: research/combo-a-deep-dive/proposals/rfc-FINAL.md
- RFC #2: research/combo-a-deep-dive/proposals/rfc-runtime-select.md
- PoC: experiments/prototype/src/runtime_select_poc.c (664 LOC)
- MLIR sketch: experiments/prototype/src/RuntimeSelectAttr.cpp.sketch (883 lines)
- Runtime wrappers: experiments/prototype/src/GPURuntimeSelectWrappers.c (319 LOC)
- Benchmarks: research/combo-a-deep-dive/research/benchmark-results.md, layer-benchmark-results.md, pinned-benchmark-results.md
- Q&A: research/combo-a-deep-dive/proposals/qa-cards-final.md (25 cards)

## Key Numbers
- Selection overhead: 3-6 ns (PoC with real OffloadBinary)
- cuModuleLoadData cold: 42.7 µs | warm: 10.0 µs
- cuLaunchKernel: 1.6 µs | cuStreamSync: 2.5 µs
- Hot-path total: 4.0-4.3 µs
- Overhead vs 10ms kernel: < 0.1%

## Known Fixed Issues (don't re-flag these)
- MI300X removed from all materials
- XeVM PR corrected to #148286
- TaxBreak 4.707µs confirmed as "average" not "median"
- CUdevice type fixed to int
- Tier numbering fixed (1,2 not 1,3)
- IFunc analogy softened to "inspired by"
- tensor_core → tensor_core_nv in all files
- CUDA 12.x → 13.1
- Layer table: real bench_layers data (not estimates)
- OffloadBinary format disclaimer added

## Issues Found This Loop
(Updated each iteration)

## Iteration 1 — Review Round 5 (R5)

### Reviewers deployed:
1. Pedantic LLVM maintainer — RUNNING
2. Academic OSDI/SOSP reviewer — DONE (7 issues: 4 MAJOR, 3 MODERATE)
3. Hostile competitor — RUNNING
4. Visual design critic — DONE (10 issues: P0-P3)
5. Fresh-eyes newcomer — DONE (10 accessibility issues)

### Key findings:
- Paper uses unpinned numbers when pinned data exists (ACADEMIC #2)
- 46.2µs kdl overhead vs 3ns PoC overhead conflated (ACADEMIC #3)
- Title too small for A0 poster (DESIGN #1)
- 13+ unexpanded acronyms (NEWCOMER #2)
- No reading order on poster (NEWCOMER #5)
- "first published" claim too broad (ACADEMIC #4)

### Solvers launched:
- Design solver: fixing CSS (P0-P3 first)
- Academic solver: fixing paper (use pinned data, separate measurements)
- Newcomer solver: fixing accessibility (acronyms, reading order, "so what" box)

### Solver Status:
- ✅ Design solver: 10/10 CSS fixes applied (poster 7.5→9/10)
- ✅ Academic solver: paper updated with pinned data, 4 pages confirmed
- 🔄 Newcomer solver: adding acronym expansions, reading order, "so what" box
- 🔄 Pedantic solver: fixing XeVM chip param, struct layout note
- 🔄 Hostile solver: fixing IREE #50 conflation, adding OffloadBinary disclaimer to poster

### Iteration 1 Score (estimated):
- Before: 4.0/5 (R4)
- After design + academic fixes: ~4.2/5
- After all 5 solver passes: target 4.3-4.5/5

### Issues remaining after iteration 1:
- Need iteration 2 re-review to verify all fixes landed correctly
- Need to git commit everything

### Iteration 1 Complete (pending 2 solvers):
- ALL 5 reviewers finished
- 3/5 solvers complete (design ✅, academic ✅, newcomer ✅)
- 2 solvers running (pedantic, hostile)
- PR #186088 bibitem author fixed: Huber → Duran
- Paper: pinned data + CV% + separated measurements + qualified claims = 4 pages, 0 errors
- Poster: 43KB, 10 design fixes applied, 0 banned terms
- Estimated post-iteration-1 score: 4.3/5

### Issues to verify in Iteration 2:
- Hot-path number consistency (4.0 vs 4.1 vs 4.3 µs across files)
- Cold-path number consistency (pinned 36.0 µs in paper, 42.7 µs may still be in poster)
- IREE #50 reference removed from poster
- OffloadBinary "simplified subset" disclaimer in poster

## Iteration 2 — Verification Pass
- Verified all iteration 1 fixes
- Checked: number consistency, IREE #50 removal, XeVM chip param, PR author, roofline ban

## Iteration 2 — COMPLETE
- Found 3 MAJOR regressions from iteration 1 fixes
- All 3 fixed and verified:
  - Poster numbers normalized to pinned data (36.0µs, 4.1µs)
  - IREE #50 purged from proposal-v2.md (was missed in iteration 1)
  - tensor_core → tensor_core_nv in poster
- Verification score: 3.7 → estimated 4.2-4.3 after fixes

## Iteration 3 — Final Score
- Need one more verification pass to confirm no new regressions
- Then git commit everything
