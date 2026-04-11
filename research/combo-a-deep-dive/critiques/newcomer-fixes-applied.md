# Newcomer Accessibility Fixes Applied

**Source review:** `critiques/review-newcomer-r5.md`
**Target file:** `poster/poster-combo-a.html`
**Date:** 2026-04-10

---

## Fix 1: Plain-English "So What" callout box (Issue 9)

**Location:** New `<div class="callout teal">` inserted between thesis strip and 3-column body (after line ~834).

**Change:** Added a centered teal callout with text:
> **In plain English:** When you compile a GPU program for multiple vendors (NVIDIA, AMD, Intel), LLVM currently picks the first binary that works. We measure the cost and propose a way to pick the **best** one.

---

## Fix 2: Acronym expansions on first use (Issue 2)

**Locations:**

1. **Left column, first paragraph** — Expanded "NVVM + ROCDL + XeVM" to:
   - NVVM (NVIDIA Virtual Machine)
   - ROCDL (ROCm Device Library)
   - XeVM (Intel Xe GPU)

2. **Center column, C1 metadata table** — `min_sm` row now reads:
   - "Min CUDA compute capability (e.g. sm_75 = NVIDIA Turing, compute capability 7.5)"
   - `min_gfx` row expands ISA: "(Instruction Set Architecture)"

3. **Center column, C3 code block** — Inline expansions in bin strings:
   - `"...cubin (CUDA Binary)..."`
   - `"...hsaco (HSA Code Object)..."`

---

## Fix 3: Number anchors for key measurements (Issue 3)

**Location:** Center column, C2 stat boxes.

**Changes:**
- 42.7 us stat box label changed from "Cold module load (one-time cost)" to:
  "Cold module load (~1/23,000th of a second -- faster than a single DRAM page fault)"
- 6 ns stat box label changed from "Selection overhead (100K iterations)" to:
  "Selection overhead (faster than a single L2 cache access)"

---

## Fix 4: Reading order numbers on column headers (Issue 5)

**Locations:** First `<h2>` in each column.

**Changes:**
- Left column: Added circled `1` badge (blue background) before "The Gap"
- Center column: Added circled `2` badge (teal background) before "C1: OffloadBinary..."
- Right column: Added circled `3` badge (orange background) before "Dispatch Overhead"

Badges use inline `<span>` with matching column color, white text, pill shape.

---

## Fix 5: One-line MLIR comment in gpu.binary code block (Issue 6)

**Location:** Center column, C3, `<pre>` block.

**Change:** Added second comment line after `// MLIR: defer binary selection to runtime`:
```
// Compile once -> 3 vendor binaries -> pick best at runtime
```

---

## Fix 6: Parenthetical descriptions for Related Work entries (Issue 10)

**Location:** Bottom bar, Related Work Comparison table, first column of each row.

**Changes:**
- IREE HAL -> IREE HAL *(Google's ML compiler runtime)*
- chipStar -> chipStar *(HIP-over-SPIR-V portability layer)*
- Proteus (LLNL) -> Proteus (LLNL) *(GPU JIT specializer)*

Descriptions use 8pt dimmed text to avoid visual clutter.

---

## Fix 7: PR reference context (Issue 8)

**Location:** Left column, "Upstream Evidence of the Gap", PR #186088 entry.

**Change:** Rewrote from:
> `liboffload` loads "first compatible image" -- selection policy explicitly deferred to higher layers

To:
> `liboffload` loads first compatible GPU binary -- explicitly defers ranked selection to higher layers

---

## Fix 8: Simplified ld.so analogy (Issue 4)

**Location:** Right column, "Key Findings" card, finding #4.

**Change:** Rewrote from pure systems-programming jargon to lead with accessible analogy:
> Think of it like a phone app store that auto-installs the right version for your device. `#gpu.runtime_select` picks the right GPU binary once at program start; every subsequent kernel launch uses that choice with zero overhead. (Technically: resolves in `global_ctors`, hot path is a single pointer load -- same pattern as the dynamic linker's PLT.)

The technical details are preserved in a parenthetical for experts, but the primary explanation is now accessible.

---

## Issues NOT addressed (design/structural, not content fixes)

- **Issue 1** (thesis strip jargon): Partially addressed by Fix 1 (plain-English callout). Full thesis strip rewrite deferred -- would change too much shared structure.
- **Issue 7** (architecture diagram too compressed): Structural/visual issue better suited to the design solver. The diagram itself was not modified.
- **Issue 3 partial** (kernel duration context in percentage table): The existing table already provides kernel type labels. Adding inline anchors to the stat boxes was the higher-impact fix.

---

## Summary

8 of 10 review issues addressed with content-only changes. 2 remaining issues (thesis strip rewording, architecture diagram annotation) require design/layout collaboration and are deferred. Total diff: ~30 lines changed, ~10 lines added. No CSS modifications -- all changes are HTML content within existing card structure.
