# Pedantic Review R5 -- Fixes Applied

**Source:** `review-pedantic-r5.md`
**Date:** 2026-04-10

## Fixed

| # | Issue | File(s) | Fix |
|---|-------|---------|-----|
| 1 | `RuntimeSelectEntry` struct layout inconsistency | `RuntimeSelectAttr.cpp.sketch` | Added contract comment at struct definition (line ~244) explaining the LLVM IR struct vs C runtime struct divergence and why it is intentional. |
| 2 | `#xevm.target<device = "pvc">` wrong param name | `extended-abstract-v3.tex`, `extended-abstract.tex` | Changed `device` to `chip` in both files. |
| 3 | `DefaultValuedParameter` missing builder expression | `RuntimeSelectAttr.cpp.sketch` | Changed to `$_builder.getStringAttr(...)` pattern for both `$strategy` and `$fallback`. |
| 4 | PR #186088 bibliography has fabricated title | `extended-abstract-v3.tex`, `extended-abstract.tex` | Changed to `[OFFLOAD] Generalize support for OffloadBinary images`. |
| 5 | `select_variant` in MLIR keyword list does not exist | `extended-abstract-v3.tex`, `extended-abstract.tex` | Replaced with `select_object` (the actual upstream attribute). |
| 7 | `getOrInsertFunction` calling convention inconsistent | `RuntimeSelectAttr.cpp.sketch` | Switched all three declarations to `FunctionCallee` pattern (no `.getCallee()`) and updated call sites to use `CreateCall(FunctionCallee, Args)`. |
| 8 | "implicit ptr decay" misleading comment | `RuntimeSelectAttr.cpp.sketch` | Changed to `// GV is Constant* with opaque ptr type`. |
| 10 | `_POSIX_C_SOURCE 200112L` inconsistent | `GPURuntimeSelectWrappers.c` | Changed to `200809L` for consistency with sibling files. |

## Not Fixed (requires external verification)

| # | Issue | Reason |
|---|-------|--------|
| 6 | RFC #88170 URL format inconsistency (Discourse vs GitHub) | Needs manual verification of which URL is canonical. Discourse URL in tex files is likely correct. |
| 9 | PR #185663 author attribution | Needs GitHub verification of actual PR author. |

## Build Verification

```
pdflatex -interaction=nonstopmode extended-abstract-v3.tex  (x2)
Output written on extended-abstract-v3.pdf (4 pages, 338802 bytes).
```
