# LaTeX Build Log — extended-abstract.tex

**Date:** 2026-04-09
**File:** `research/combo-a-deep-dive/proposals/extended-abstract.tex`
**Output:** `research/combo-a-deep-dive/proposals/extended-abstract.pdf`
**Tool:** pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian)

---

## Build Result

**Status: SUCCESS** — PDF produced after two-pass compilation.

```
pdflatex -interaction=nonstopmode extended-abstract.tex  # pass 1
pdflatex -interaction=nonstopmode extended-abstract.tex  # pass 2 (cross-refs)
```

Output written on extended-abstract.pdf (7 pages, 199474 bytes).

---

## Page Count

**7 pages**

---

## Errors

### Package Listings — `language=MLIR` undefined (3 occurrences)

```
! Package Listings Error: Couldn't load requested language.
! Package Listings Error: language mlir undefined.
```

Triggered at lines 89, 131, and 295 — all `\begin{lstlisting}[language=MLIR]` blocks.
The `listings` package does not have a built-in MLIR language definition. pdflatex
recovers and continues; the blocks compile but lose syntax highlighting.

**Fix (if desired):** Add a custom `lstdefinelanguage{MLIR}` block in the preamble,
or fall back to `language=C++` / `language={[LLVM]Assembler}` for those blocks.

---

## Warnings

### Overfull `\hbox` boxes

| Lines        | Overfull width  | Location (approx.)             |
|--------------|-----------------|--------------------------------|
| 141--146     | 48.05 pt        | Code listing or wide verbatim  |
| 162          | 32.14 pt        | Single long line               |
| 156--165     | 766.95 pt       | Very wide code block           |
| 176--184     | 49.70 pt        | Code/text                      |
| 225--237     | 32.96 pt        | Text paragraph                 |
| 304--309     | 66.25 pt        | Code listing                   |
| 310--312     | 87.56 pt        | Code listing                   |
| 321--322     | 27.25 pt        | Text or code                   |

The 766.95 pt overfull at lines 156--165 is severe (content extends far past the right
margin). Likely a long URL, a verbatim line, or a listing block without line wrapping.

**Fix:** Add `breaklines=true` to the global `\lstset{}` and/or wrap long URLs with
`\url{}` inside a `\begin{sloppypar}` block.

### Underfull `\hbox` boxes (minor)

Two underfull boxes at lines 339--340 (badness 1661 and 3849) — cosmetic only,
caused by hyphenation in narrow columns. No action needed.

### Font Warning

```
LaTeX Font Warning: Font shape `OMS/cmtt/m/n' undefined
LaTeX Font Warning: Some font shapes were not available, defaults substituted.
```

Cosmetic. Monospace bold/italic fallback used; no visible content lost.

---

## Figures

**1 figure** present (lines 270--287): TikZ-drawn flame graph visualization of
cold-path dispatch decomposition. No external image files (`\includegraphics`) used.
Figure rendered via `tikzpicture` — no missing file errors.

---

## Summary

| Item                    | Status                                      |
|-------------------------|---------------------------------------------|
| Build                   | PASS (2-pass clean)                         |
| Pages                   | 7                                           |
| PDF size                | 199,474 bytes (~195 KB)                     |
| Hard errors             | 0 (MLIR language errors are non-fatal)      |
| Non-fatal pkg errors    | 3 × `language=MLIR undefined`               |
| Overfull boxes          | 8 (1 severe at 766 pt, rest cosmetic)       |
| Underfull boxes         | 2 (cosmetic)                                |
| Missing figures         | 0 (figure uses TikZ, no external files)     |
| Cross-references stable | Yes (page count unchanged across both passes) |

---

## Recommended Follow-up

1. Add `breaklines=true, breakatwhitespace=true` to the `\lstset{}` preamble to fix
   the 766 pt overfull and other wide code blocks.
2. Define a minimal `lstdefinelanguage{MLIR}` or alias it to `[LLVM]Assembler` to
   eliminate the package errors and restore highlighting.
3. (Optional) Wrap any bare long URLs in `\url{}` inside `\begin{sloppypar}`.
