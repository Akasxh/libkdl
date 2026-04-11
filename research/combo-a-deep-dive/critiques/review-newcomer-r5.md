# Newcomer Accessibility Review (R5)

**Reviewer persona:** First-time EuroLLVM attendee. CS masters student. Knows C++ and basic compilers. Does NOT know MLIR, GPU programming, or LLVM offload infrastructure.

**Artifact reviewed:** `poster/poster-combo-a.html` (poster-combo-a, full 3-column layout)

**Date:** 2026-04-10

---

## Issue 1: The title and thesis strip are impenetrable on first contact

**Where:** Title bar and dark thesis strip at the very top of the poster.

**What confused me:** The title says "Multi-Target Binary Selection in LLVM's GPU Offload Stack." I know what LLVM is, but "GPU Offload Stack" means nothing to me. Is that a library? A set of tools? A runtime? The thesis strip then hits me with `gpu.module`, `OffloadBinary`, and "N device images" without explaining any of these terms. I'm reading the single most important sentence of the poster -- the thesis statement -- and I already need three definitions I don't have.

- `OffloadBinary` -- Is this a file format? A data structure? An executable?
- `gpu.module` -- Is this Python? MLIR syntax? A build system concept?
- "device images" -- Images like pictures? Compiled blobs? Something else?

**Suggested fix:** Add a single parenthetical or inline clause to the thesis strip: "MLIR compiles one `gpu.module` (an MLIR operation representing a GPU kernel collection) to 3 GPU vendors. `OffloadBinary` (LLVM's container format for compiled GPU code) carries N device images (vendor-specific compiled binaries)." That is three short parentheticals. They cost maybe 15 words and save every non-MLIR reader from being lost at line one.

---

## Issue 2: Acronyms used but never expanded

**Where:** Throughout the poster, but most densely in Contribution 1 (center column, metadata table) and the code blocks.

**What confused me:** The poster uses at least these acronyms/terms without ever expanding them:

| Term | Where it appears | What I assume it might mean |
|------|------------------|-----------------------------|
| NVVM | Metadata table, code block | Something NVIDIA-related? |
| ROCDL | Thesis strip, code block | Something AMD-related? |
| XeVM | Left column ("Upstream Evidence") | Intel? |
| HSACO | Code block (`...hsaco...`) | No idea |
| CUBIN | Multiple places | Compiled CUDA binary? |
| sm_75, sm_86, sm_90 | Everywhere | Some kind of GPU version number? |
| gfx90a | Metadata table | AMD GPU identifier? |
| ISA | Metadata table ("within ISA family") | Instruction set architecture, but in what context? |
| FMV | Bottom of C3 card | "Function Multi-Versioning"? Expanded once but buried in small text |
| IFunc | Bottom of C3 card | Never expanded at all |
| PLT | Key Findings, finding #4 | "Procedure Linkage Table"? Not obvious to non-systems people |
| GEMM | Right column table | Matrix multiply? Not everyone knows this |
| FFN | Right column table | Feed-forward network? |
| HEP-CCE | Left column | Something CERN? |

I counted 13+ unexpanded acronyms. A poster is not a paper -- there is no glossary section. If I'm standing 1 meter away trying to read this, I cannot Google "HSACO" on the spot.

**Suggested fix:** Either (a) add a tiny glossary strip somewhere (even in the bottom bar), or (b) expand each acronym on first use. At minimum, NVVM/ROCDL/CUBIN/HSACO need one-time expansions since they are central to the contribution. "sm_75" should say "(NVIDIA compute capability 7.5)" at least once.

---

## Issue 3: Numbers without anchoring -- is 42.7 microseconds good or bad?

**Where:** The flame graph (center column, C2) and the data table (right column, "Dispatch Overhead").

**What confused me:** The poster's headline numbers are 42.7 us for module loading and 6 ns for selection. The poster clearly wants me to think 6 ns is impressively small. But I have no frame of reference for any of these numbers:

- 42.7 us -- is that slow? Fast? Normal for GPU operations?
- 6 ns -- the poster says "noise floor" but compared to what?
- 4.0 us hot-path -- is that competitive with other frameworks?

The "Selection as % of ML Kernel Duration" table helps somewhat by showing 6 ns against kernel durations (100 ms, 10 ms, etc.), but even there -- what is a "large GEMM, transformer" kernel duration in practice? If I don't know that 100 ms is typical, the percentage is just a number.

**Suggested fix:** Add one anchoring comparison. Something like: "For reference, a single DRAM access takes ~100 ns. Our selection at 6 ns is 16x faster than reading one cache line from main memory." Or: "A typical PyTorch kernel launch overhead is ~10 us. Our selection adds 0.06% to that." Give me ONE concrete thing I already understand to compare against.

---

## Issue 4: The "ld.so analogy" in Key Finding #4 assumes systems programming knowledge

**Where:** Right column, "Key Findings" card, finding #4.

**What confused me:** "Like the dynamic linker resolving shared libraries at process startup then jumping directly through the PLT, `#gpu.runtime_select` resolves once in `global_ctors`, then the hot path is a single pointer load."

This sentence contains: dynamic linker, shared libraries, PLT, global_ctors, pointer load. That is five systems-programming concepts chained together in one analogy. I took a compilers class, not an OS class. I don't know what PLT is. I don't know what `global_ctors` does. The analogy is supposed to make the concept MORE accessible, but it actually makes it less accessible to anyone who isn't a Linux systems programmer.

**Suggested fix:** Replace or supplement with a simpler analogy: "Like how your phone connects to WiFi once at startup and then all apps use that connection without re-scanning, `#gpu.runtime_select` picks the right GPU binary once at program start, and every subsequent kernel launch uses that choice with zero overhead." Keep the ld.so analogy too if you want, but don't make it the ONLY explanation.

---

## Issue 5: No reading order -- where do I start?

**Where:** The poster as a whole (3-column layout).

**What confused me:** I walk up to the poster. There are three columns, a top strip, and a bottom bar. Where do I start reading? The title and thesis strip are clear entry points, but after that:

- Left column is "The Gap" -- the problem statement
- Center column is "Our Contributions" (C1, C2, C3)
- Right column is "Evidence"

This is a reasonable structure, but it is not signposted. There are no numbered steps, no arrows between columns, no "Start Here" indicator. I spent a few seconds figuring out whether to read top-to-bottom within each column or left-to-right across columns. For a poster that someone scans in 30-60 seconds while walking by, that hesitation is costly.

**Suggested fix:** Add a subtle visual flow indicator. Even just labeling the columns with circled numbers -- (1) The Gap, (2) Our Contributions, (3) Evidence -- or adding a single line in the thesis strip like "Read left to right: Problem, Solution, Proof" would eliminate the hesitation.

---

## Issue 6: The code blocks assume MLIR syntax literacy

**Where:** Center column, C1 (`isMetadataCompatible` code) and C3 (`gpu.binary` MLIR code block).

**What confused me:** The C3 code block shows:

```
gpu.binary @kernels <#gpu.runtime_select<
    strategy = "rank_by_priority",
    fallback = "cpu">> [
  #gpu.object<#nvvm.target<chip="sm_75">, bin="...cubin...">,
  ...
]
```

I do not know MLIR syntax. I see angle brackets, hash symbols, and nested attributes, and I cannot parse the structure. Is `#gpu.runtime_select` a type? A function call? An annotation? The poster assumes I can read MLIR like I read C++. I cannot.

The `isMetadataCompatible` C++ code in C1 is much more readable -- it looks like normal C++ with some API calls. But the MLIR block in C3 is the central design contribution and it is the least readable part of the poster to a newcomer.

**Suggested fix:** Add a 1-line comment above the MLIR block: `// Read as: "this binary bundle contains 3 GPU variants; at runtime, pick the best one by priority, fall back to CPU."` That single sentence lets me understand the intent even if I can't parse the syntax. The code becomes an illustration of the idea rather than the only way to understand it.

---

## Issue 7: The architecture diagram (C3) is too compressed

**Where:** Center column, C3 card, the horizontal box-and-arrow diagram.

**What confused me:** The architecture diagram shows: `N gpu.object blobs -> @kernels_blob_0/1/2 -> Dispatch Table + Vendor Detect (global_ctors) -> @kernels_module_ptr`. I can see a pipeline, but:

- What are "@kernels_blob_0" etc.? Global variables? Files?
- "Vendor Detect" -- how does this happen? Magic?
- "global_ctors" -- what is this? (Same issue as the ld.so analogy.)
- The arrow from Dispatch Table to @kernels_module_ptr -- what does this step actually DO?

The diagram shows boxes and arrows but not what happens INSIDE the boxes. For someone unfamiliar with LLVM's lowering pipeline, the boxes are opaque labels connected by arrows.

**Suggested fix:** Add one sentence below the diagram explaining the flow in plain English: "At compile time, each GPU binary variant is embedded as a global constant. At program startup, a constructor function detects the GPU vendor and picks the best variant. All subsequent kernel launches use that single cached choice." This maps each box to an understandable action.

---

## Issue 8: PR and issue references mean nothing without context

**Where:** Left column, "Upstream Evidence of the Gap" section.

**What confused me:** The poster lists PR #148286, PR #186088, PR #185663, Issue #75356, and RFC #88170. These are presented as evidence that the gap exists. But to me, these are just numbers. I don't follow LLVM development. I don't know if PR #185663 is a big deal or a minor cleanup. The colored badges (purple "merged", green "open") help a little, but the actual significance of each PR is unclear.

For example: "PR #185663 `isMetadataCompatible()` consumer hook merged -- but only checks `triple` + `arch` strings." I gather this means something was added but it's incomplete. But I don't know what a "consumer hook" is in this context, what `triple` means (target triple? I vaguely remember that from compilers class), or why checking only two strings is insufficient.

**Suggested fix:** The PR numbers are useful for experts who will look them up. But for the rest of us, add a brief plain-English parenthetical after each: "PR #185663 (merged June 2025) added a hook to check if a GPU binary matches a device -- but it only checks two fields, not enough for smart selection." The PR number stays for credibility; the explanation stays for comprehension.

---

## Issue 9: The "So What" is buried

**Where:** Left column, "Why This Matters Now" card.

**What confused me:** After reading the entire poster, I can piece together the story: LLVM can compile code for multiple GPU vendors, but at runtime it doesn't intelligently pick which compiled version to actually use. This poster proposes metadata and a selection mechanism to fix that.

But this plain-English summary does not appear anywhere on the poster in that form. The thesis strip says it in MLIR jargon. The "Why This Matters Now" card lists use cases (CERN, vLLM, cloud containers) but these are motivation, not explanation. I had to reconstruct the "elevator pitch" myself from scattered pieces.

If a friend asked me "what's that poster about?" after I walked away, I think I could say: "It's about making programs that contain GPU code for NVIDIA and AMD automatically pick the right version at runtime, like how your computer picks the right driver." But the poster never gave me that sentence directly.

**Suggested fix:** Add a single "In Plain English" callout box, ideally near the top (thesis strip area or top of center column): "When software ships with GPU code compiled for multiple vendors (NVIDIA, AMD, Intel), today's LLVM picks blindly. We propose metadata and a selection mechanism so it picks the best match -- like how a web browser picks the right video codec for your device." One sentence, zero jargon, memorable analogy.

---

## Issue 10: The Related Work table assumes I know these systems

**Where:** Bottom bar, "Related Work Comparison" table.

**What confused me:** The table lists IREE HAL, chipStar, Proteus (LLNL), liboffload, and CPU FMV. I have heard of none of these except vaguely IREE. The table tells me whether each has "Runtime Select" and "Cross-Vendor" support, but I don't know what these systems ARE, so the comparison doesn't land.

- IREE HAL -- a runtime? A compiler? What does HAL stand for?
- chipStar -- is this a CUDA compatibility layer? Something else?
- Proteus (LLNL) -- a JIT compiler from a national lab?
- liboffload -- part of LLVM? A separate project?

The table is clearly designed for experts who know the landscape. For me, it's a grid of green/orange/grey dots next to names I don't recognize.

**Suggested fix:** Add a 1-word descriptor after each name in the table. Not a full explanation, just enough to place each system: "IREE HAL (Google ML compiler runtime)", "chipStar (HIP-over-SPIR-V portability)", "Proteus (LLNL JIT specializer)", "liboffload (LLVM's own offload runtime)". One parenthetical per row, five extra seconds of reading, and now the comparison actually communicates.

---

## Summary: The "Can I Explain This to a Friend?" Test

After reading the full poster carefully (which took me several minutes, not the 60 seconds a poster walk-by gets), here is what I THINK it's about:

> LLVM can now compile GPU code targeting NVIDIA, AMD, and Intel from a single source. But when you run the program, the system just grabs the first compiled version that seems compatible, with no smart matching. This poster proposes (1) a set of metadata tags so each compiled version can describe what GPU it needs, (2) measurements showing the selection process is essentially free (6 nanoseconds), and (3) a design for an MLIR mechanism to do this automatically.

I am about 70% confident this summary is correct. The 30% uncertainty comes from not understanding whether this is about a single program shipping fat binaries for multiple GPUs (like Apple's Universal Binaries) or about a cloud system dynamically choosing at deployment time. The poster doesn't clearly distinguish these scenarios.

**Bottom line:** The technical content appears solid and well-evidenced. The poster is written for LLVM insiders. A newcomer CAN extract the story, but it requires real effort and tolerance for unexplained jargon. Ten small additions -- mostly parenthetical expansions and one plain-English summary sentence -- would make this poster accessible to the broader EuroLLVM audience, not just the GPU/MLIR specialists.
