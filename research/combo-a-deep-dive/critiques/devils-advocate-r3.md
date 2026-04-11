# Devil's Advocate Round 3: Proposal v2 Review

**Date:** 2026-04-09
**Verdict:** REVISE — v2 is genuinely improved but blocked by zero measured data (now resolved by benchmark run)

## Key Findings

1. **CRITICAL (NOW RESOLVED):** All benchmark data was PLACEHOLDER — bench_dispatch now shows 46.2µs cold select, 0.84µs cuLaunchKernel baseline
2. **MAJOR:** Tier 1 → Tier 3 numbering gap (missing Tier 2) — fix: renumber or state "Tier 2 reserved"
3. **MAJOR:** "Why MLIR" section overclaims liboffload endorsement — fix: soften to "scope is mechanism, not policy"
4. **MAJOR:** Per-vendor feature tokens undermine vendor-agnostic pitch — fix: acknowledge honestly
5. **MINOR:** Static init call_once design missing, LOC undercount

## Score Estimate
- With PLACEHOLDERs: 3.4/5
- With real numbers (now available): 3.8-4.0/5 → solid ACCEPT territory

## Single Most Important Next Step
Fill the PLACEHOLDERs in proposal-v2.md with the actual benchmark numbers.
