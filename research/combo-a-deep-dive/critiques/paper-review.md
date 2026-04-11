# Extended Abstract Review: Runtime Variant Selection in LLVM's GPU Offload Stack

**Reviewer:** Internal Review
**Date:** 2026-04-09
**File:** `research/combo-a-deep-dive/proposals/extended-abstract.tex`
**Verdict:** REQUEST CHANGES (before fixes) -> COMMENT (after fixes applied)

---

## Review Checklist Results

| # | Check | Result |
|---|-------|--------|
| 1 | Follows structure from proposal-v2.md? | PARTIAL -- structure matches but omits Known Design Issues subsection |
| 2 | All claims properly cited? | MOSTLY -- two uncited claims (vLLM, HEP-CCE from literature notes only) |
| 3 | Overclaims (especially "roofline")? | PASS -- correctly says "weighted heuristic" at line 339 |
| 4 | MI300X claim removed? | PASS -- no MI300X mention anywhere |
| 5 | XeVM PR correctly cited as #148286? | PASS -- line 67 and bibitem correct |
| 6 | Leads with T07+T19, demotes T01? | PASS -- C1 and C2 concrete, C3 labeled "design sketch" |
| 7 | TikZ figures correct and useful? | FIXED -- was inverted, now proportionally correct |
| 8 | Tables well-formatted? | PARTIAL -- overfull hboxes remain (presentation polish) |
| 9 | Related work fair and accurate? | PASS -- honest framing of IREE, chipStar, CPU FMV |
| 10 | Grammar, spelling, formatting? | FIXED -- compilation-breaking errors resolved |

---

## Issues Found and Fixed

### CRITICAL (fixed)

**[CRITICAL-1] TikZ flame graph had inverted width proportions** -- FIXED
File: `extended-abstract.tex:283-300` (original lines)
Issue: OffloadBinary parse was widest (12cm) and olLaunchKernel was narrowest (4cm). The paper's own data shows olCreateProgram is 73% of total latency.
Fix applied: Replaced with proportionally correct stacked horizontal bar showing 73% olCreateProgram, 15% launch, 9% parse, 4% symbol lookup.

**[CRITICAL-2] `\LLVM` undefined control sequence + bare `#` characters** -- FIXED
File: `extended-abstract.tex:59-60` (original lines)
Issue: `\LLVM` is not a LaTeX macro; `#` is a macro parameter character in running text. Both cause fatal compilation errors.
Fix applied: `\LLVM` -> `LLVM`; all `PR~#` -> `PR~\#` and `RFC~#` -> `RFC~\#`.

### HIGH (fixed)

**[HIGH-1] `language=MLIR` not defined in listings** -- FIXED
File: `extended-abstract.tex:89, 131, 302` (original lines)
Issue: Three `lstlisting` blocks specify `language=MLIR` but the listings package has no MLIR definition.
Fix applied: Added `\lstdefinelanguage{MLIR}{...}` in the preamble with appropriate keywords.

**[HIGH-2] Tier 1 -> Tier 3 numbering gap** -- FIXED
File: `extended-abstract.tex:153, 173` (original lines)
Issue: Jumped from Tier 1 (MUST keys) to Tier 3 (MAY keys) with no explanation.
Fix applied: Renamed to "Tier 2: MAY Keys". Updated footnote about deferred resource-usage keys to not conflict with the Tier 2 label.

**[HIGH-3] Cold-path table header implied rigorous measurement** -- FIXED
File: `extended-abstract.tex:222` (original line)
Issue: Header said "n=100 processes" implying 100 process-forked trials, but 4 of 5 layer values are estimates.
Fix applied: Changed to "Estimated cold-path layer decomposition (GTX 1650, null kernel, CUBIN, prototype measurements):"

**[HIGH-4] Bare `#` characters in running text** -- FIXED (included in CRITICAL-2 fix)

### MEDIUM

**[MEDIUM-1] Overfull hbox warnings (tables exceed margins)** -- NOT FIXED
File: Lines 163-172 (Tier 1 table, 766pt overfull), 183-191 (Tier 2 table), 328-336 (embedBinary design)
Issue: Several tables and code blocks overflow page margins significantly.
Fix: Use `\resizebox{\textwidth}{!}{...}` or switch to `tabularx` with `X` columns. The Tier 1 metadata table (4 columns with descriptions) is the worst offender.

**[MEDIUM-2] Abstract says "five new keys" without tier context** -- NOT FIXED
File: Line 49
Issue: Minor ambiguity given the deferred Tier 2 keys mentioned in the footnote.
Fix: Say "five initial keys in two tiers" or remove the count from the abstract.

**[MEDIUM-3] `requires_features` example uses vendor-specific tokens** -- NO FIX NEEDED
File: Line 169 (`tensor_core_nv,bf16`) vs. proposal-v2.md line 80 (`tensor_core,bf16`)
Status: The paper is more correct than the proposal. The vendor-specific tokens are the intended design.

**[MEDIUM-4] TaxBreak 4.707 us labeled "median" when source says "average"** -- FIXED
File: `extended-abstract.tex:143` (original line)
Issue: TaxBreak Table III reports 4.707 as average (avg), not median. p50 is 4.578 us.
Fix applied: Changed to "4.707 us average (p50: 4.578 us)".

**[MEDIUM-5] "Concrete Next Steps" section is project management, not publication content** -- NOT FIXED
File: Lines 396-404
Issue: Items like "Verify PR #186088 merge status" and "Generate flame graph SVGs" are TODO items, not research conclusions. The "Upstream Path" section already covers forward-looking content.
Fix: Remove Section 6.2 or rewrite as "Future Work" with research framing. This is an author content decision.

### LOW

**[LOW-1] n= values differ between protocol description and actual tables** -- NOT FIXED
File: Lines 215 vs. 257
Issue: Protocol says 100 trials / 10,000 dispatches; tables say n=1,000. The measurements come from the prototype benchmark which uses 1,000 iterations.
Fix: Align protocol description with actual execution, or add a note.

**[LOW-2] `\footnotesize{...}\normalsize` fragile scoping** -- NOT FIXED
File: Lines 241, 270
Issue: Non-standard size switching. Should use `{\footnotesize ...}` grouping.
Fix: Replace with group-scoped font size changes.

**[LOW-3] LOC estimate differs from proposal-v2.md** -- NOT FIXED
File: Line 391 (~600 LOC) vs. proposal-v2.md (~780 LOC total)
Issue: Minor inconsistency in implementation estimates.

**[LOW-4] Missing `\usepackage{url}`** -- FIXED
File: Preamble
Fix applied: Added `\usepackage{url}` before `\usepackage{hyperref}`.

---

## Positive Observations

1. **Honest prototype framing** (lines 347-353): Explicitly states what the prototype does and does NOT demonstrate, including "custom MTB format, not OffloadBinary" and "share zero code with LLVM's format." This level of honesty will earn reviewer trust.

2. **"Weighted heuristic" not "roofline"** (line 363): Correctly describes `kdl_estimate_cost_weighted()` without the overclaim that plagued earlier drafts.

3. **MI300X completely absent**: Clean removal of the unsupported hardware claim.

4. **XeVM PR #148286 correctly cited**: Both in the introduction and bibliography.

5. **T07+T19 lead correctly**: Contributions 1 and 2 presented as concrete; Contribution 3 labeled "Design only. Zero lines of MLIR C++ exist." This is the right emphasis for a poster.

6. **CPU FMV analogy** (lines 374, 398): Using IFunc resolvers as the structural precedent is the paper's strongest argumentative move.

7. **Backward compatibility argument** (line 213): "Missing keys = no constraint" is clean and convincing.

8. **Related work table** (lines 376-389): Fair comparison. IREE HAL acknowledged as closest with specific issue numbers. chipStar, Proteus included.

9. **Vendor-specific token explanation** (lines 174-178): Addresses reviewer panel critique about cross-vendor capability mapping with specificity.

---

## Structural Compliance with proposal-v2.md

**Present:**
- Introduction with the three-layer gap (compilation complete, selection missing)
- Downstream user motivation (HEP-CCE, vLLM, cloud containers)
- Background with OffloadBinary format, RFC #88170, PR #186088
- Contributions 1/2/3 in order of concreteness
- Prototype with honest framing
- Related work comparison table
- Upstream path

**Missing from paper but in proposal-v2.md:**
- Known Design Issues subsection (static init, naming collision, dlopen+ASAN) -- should be a paragraph in Contribution 3
- XeVM SPIRV build caveat (proposal line 292) -- one sentence needed
- `clang-linker-wrapper` integration detail for metadata writers
- `--gpu-mark-runtime-select` pass mention

These omissions are acceptable for a 4-page abstract but should appear in the poster itself.

---

## Compilation Status (Post-Fix)

- **Errors:** 0 (was 5 fatal errors)
- **Undefined control sequences:** 0 (was 1)
- **Undefined languages:** 0 (was 3)
- **Overfull hbox:** 10 (presentation polish, not fatal)
- **Pages:** 7
- **PDF output:** 277,727 bytes, renders correctly

---

## Recommendation

**COMMENT** (after fixes applied)

All CRITICAL and HIGH issues have been resolved. The paper compiles cleanly and the content is factually accurate on the key claims (XeVM PR, TaxBreak numbers, no MI300X, no roofline overclaim, correct T07+T19 emphasis). The remaining MEDIUM issues are presentation polish (table widths, section naming) and one content decision (removing the TODO-style "Concrete Next Steps").

The paper is ready for author review of the remaining MEDIUM items before submission.
