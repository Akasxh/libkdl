# Community Fit Analysis: libkdl at EuroLLVM Dublin 2026

**Perspective:** Senior LLVM compiler engineer, 5-year DevMtg veteran, MLIR RFC reviewer
**Date:** 2026-04-06
**Deadline:** 2026-04-07
**Verdict:** The prototype is real and good. The framing will kill it. Reframe now.

---

## Q1: Would a standalone C runtime library excite LLVM compiler engineers?

**No. Not as currently framed.**

### The Problems

**Problem 1: Custom binary format instead of OffloadBinary.**
`kdl.c:63-64` defines `MTB_MAGIC "KDL_MTB\0"` with `MTB_VERSION 1`. Grep for `OffloadBinary` in `kdl.c` returns zero hits. The LLVM project already ships `OffloadBinary` (magic `0x10FF10AD`, documented in `llvm/include/llvm/Object/OffloadBinary.h`). Every LLVM compiler engineer will immediately ask: "Why didn't you extend OffloadBinary?" A custom format reads as NIH syndrome. This is the single largest credibility risk.

**Problem 2: The "roofline cost model" is hardcoded constants.**
`kdl.c:1008` says "Iteration 3: Improved cost model (roofline + efficiency factors)." The actual implementation at `kdl.c:1051-1054`:
```c
case KDL_VENDOR_CPU:    locality_score = 0.0;    break;
case KDL_VENDOR_NVIDIA: locality_score = 50e-6;  break;
case KDL_VENDOR_AMD:    locality_score = 60e-6;  break;
default:                locality_score = 100e-6;  break;
```
This is a weighted heuristic with magic numbers. It is not a roofline model. The poster strategy claims "roofline cost model validated at 94.7% by tritonBLAS" — but that is tritonBLAS's result using `max(T_compute, T_memory)`, not what this code does. Any compiler engineer who reads the source (and they will) sees the gap between claim and implementation. Credibility dies in 5 seconds.

**Problem 3: Zero MLIR integration.**
The prototype does not consume `gpu.binary`, does not interact with `gpu-module-to-binary`, does not parse `.llvm.offloading` sections. It is a standalone C library with its own format, its own loader, its own everything. At a venue where every talk involves MLIR passes and LLVM infrastructure, this feels like a parallel universe.

**Problem 4: "ld.so for GPU kernels" is a systems metaphor, not an LLVM metaphor.**
The analogy is compelling to Linux kernel developers and systems programmers. LLVM compiler engineers think in passes, dialects, and lowering pipelines. The ld.so framing signals "I built something outside your ecosystem." It is the right metaphor for the wrong audience.

### What They WILL Appreciate

The prototype is genuinely good engineering:
- `kdl.c:228,248` — real `cuModuleLoadData`/`hipModuleLoadData` function pointers via dlopen (no link-time GPU deps)
- `kdl.c:568,749` — actual runtime loading of CUDA/HIP symbols
- 5157 LOC of real C code that builds and runs
- `bench_dispatch` and `demo_dispatch` are real compiled ELF executables
- The dlopen-based vendor discovery is architecturally clean

**Verdict:** The prototype is solid engineering. But it is an MLSys contribution currently wearing LLVM clothing. As-is, the response will be "cool, submit this to MLSys or OSDI" — and then they walk away.

---

## Q2: What WOULD excite LLVM compiler engineers?

Ranked by community resonance, highest to lowest:

### Tier 1: Maximum resonance (pick one of these)

**(a) A `rankImage()` callback contribution to liboffload**
PR #186088 literally defers variant selection to "a follow-up PR." If Akash shows up saying "I implemented the follow-up PR that Huber deferred, here are the benchmarks," that is a direct upstream contribution. The poster becomes: "we measured the cost of first-compatible-wins vs. ranked selection, here is the data, here is the patch."

Joseph Huber is at EuroLLVM Dublin. He will stop at this poster.

- Effort: Medium (need to write a draft patch against liboffload, even if incomplete)
- Impact: Maximum — only framing where the poster leads to an actual upstream commit
- Risk: If the patch is bad, it is worse than not having one

**(b) First cross-vendor dispatch overhead measurement on real hardware**
The community genuinely lacks this data. The numbers in the poster strategy doc are borrowed from other papers (FlashInfer, TaxBreak). If Akash measures on his GTX 1650: (i) raw `cuLaunchKernel` latency, (ii) liboffload `ol*` API latency, (iii) libkdl selection + dispatch latency, (iv) CPU fallback latency — all on the same hardware, same kernel — that is a novel data point nobody else has published.

- Effort: Low (the `bench_dispatch` binary already exists and runs)
- Impact: High — numbers from real hardware on a poster are conversation magnets
- Risk: GTX 1650 is modest hardware; some people will discount it. Counter: "modest hardware is the point — this is deployment, not vanity benchmarks"

### Tier 2: Strong resonance

**(c) The prototype as-is, reframed as "policy layer exploration above liboffload"**
Keep the code. Change the narrative. Instead of "here is libkdl, a new library," frame it as: "We explored what a policy layer above liboffload would look like. Here is what we learned, here are the measurements, here is why `rankImage()` needs these three inputs." The prototype becomes evidence for a design argument, not a product pitch.

- Effort: Zero code changes, just poster text changes
- Impact: Medium-high — reframes from "competing project" to "exploration that feeds upstream"

### Tier 3: Not feasible in 1 day

**(d) An MLIR dialect (`gpu.dispatch` op)** — Requires RFC, working pass, tests. Vaporware on a poster is the worst outcome. Do not attempt.

**(e) KernelInfo + cost model pipeline** — Requires LLVM build setup and understanding Denny's pass internals. Not feasible.

**(f) A design RFC with no code** — "Design-only posters get 'interesting' and are forgotten" (wave-07 research). You have code. Use it.

---

## Q3: The combination that maximizes poster engagement

Given: working prototype, strong survey data, GTX 1650, CERN/GSoC credibility, 1 day.

### The Optimal 1-Day Strategy

**Step 1 (2 hours): Run actual benchmarks.**
Execute `bench_dispatch` on the GTX 1650. Record:
- Raw `cuLaunchKernel` dispatch latency (floor)
- libkdl `kdl_select_kernel` + dispatch latency
- CPU-fallback dispatch latency
- Repeat 1000x, report median/p99
These are YOUR numbers. Nobody can challenge them because nobody else ran this experiment.

**Step 2 (1 hour): Write a 2-paragraph Discourse RFC draft.**
Title: "[RFC] Runtime kernel variant selection for liboffload"
Body: "PR #186088 defers selection policy. Here is what we learned building a prototype. We propose a `rankImage()` callback with these inputs: device capability fingerprint, kernel resource metadata (from KernelInfo pass), and a pluggable cost function. Here are measurements from a GTX 1650."
Do not post it yet — just have the URL ready. Having a draft RFC signals "I want to contribute upstream."

**Step 3 (4 hours): Build the poster with this structure:**

| Panel | Title | Content |
|-------|-------|---------|
| **LEFT** | **The Deferred Follow-Up** | PR #186088 loads first-compatible image. Issue #75356 (open 2.5 years) requests dlsym-for-GPUs. Huber 2025: "ld.so for GPU code." Three quotes, one sentence: "The compile-time half is done. The runtime policy is not." |
| **CENTER** | **A 5100-LOC Exploration** | 5-box architecture: OffloadBinary -> device fingerprint -> weighted selection -> vendor driver -> execution. 3-line API snippet. ld.so analogy table (4 rows). Key sentence: "This prototype explores what `rankImage()` could look like as a liboffload extension." |
| **RIGHT** | **Measured on a GTX 1650** | YOUR dispatch overhead bar chart (not borrowed numbers). Cold start: <5ms measured. LOC context: libkdl 5100 / liboffload core ~3000 / IREE HAL ~15000. QR to GitHub repo + Discourse RFC draft. |

**Step 4: Prepare the live demo.**
`KDL_FORCE_CPU=1 ./bench_dispatch` vs. `./bench_dispatch` with GPU. Same binary, different hardware detection, different variant selected. 30 seconds. Communicates the entire thesis without slides.

---

## Q4: What Akash must NOT do

### Credibility Traps

1. **Do NOT present the 14-system comparison matrix on the poster.** It screams "survey paper" and signals "I did not build anything, I read papers." The competitive landscape is for Q&A only. If someone asks about IREE or chipStar, you know the answer. But the poster surface is for YOUR work.

2. **Do NOT claim the cost model is a "roofline model."** It is hardcoded constants (`kdl.c:1051-1054`). When someone asks "what is your compute/memory bandwidth model?" and you point to `locality_score = 50e-6`, you lose all credibility. Call it "a weighted heuristic" and say "roofline refinement using tritonBLAS's `max(T_compute, T_memory)` is the planned next step." Honesty is respected; overclaiming is not.

3. **Do NOT use the MTB format name prominently.** Nobody at LLVM cares about a new binary format that is not OffloadBinary. If you mention MTB, the immediate question is "why not OffloadBinary?" and the honest answer is "I haven't integrated OffloadBinary yet." Frame MTB as a prototype vehicle, not a contribution.

4. **Do NOT present this as a finished product.** 5100 LOC with hardcoded constants and no OffloadBinary integration is a prototype. Own it. "Working exploration" is respected in this community. "Production-ready library" will be stress-tested by people who write production compilers, and it will be found wanting.

5. **Do NOT quote the 843s Triton cold start as if libkdl solves it.** That is Meta's internal profiling number from a specific PT2 workload. libkdl avoids JIT compilation for pre-compiled kernels — a different claim. Conflating the two is intellectually dishonest, and compiler engineers trained in formal semantics will catch it immediately.

6. **Do NOT borrow tritonBLAS's 94.7% validation number.** That is their roofline model, not yours. Your cost model uses hardcoded vendor constants. Citing someone else's validation for your different implementation is academic dishonesty in this community.

7. **Do NOT skip the MLIR Workshop on April 13.** The CUDA Tile IR talk (Springer/Chelini, NVIDIA) and the rocMLIR talk (Martinez, AMD) will give you the exact vocabulary and framing the audience uses the next day at the poster session. You need to speak their language.

### Audience Misjudgments

8. **Do NOT assume people care about PyTorch/TF integration.** This is EuroLLVM, not MLSys. The audience builds compilers, not ML frameworks. Frame the contribution in compiler infrastructure terms.

9. **Do NOT frame this as "my project."** Frame it as "an exploration for the community." LLVM people contribute to a shared project. "I built this for LLVM" lands. "I built this for myself" does not.

10. **Do NOT have a "future work" section on the poster.** Everything on the poster must be done. Not planned, not proposed — done and measured. Future work goes in conversation, not on paper.

---

## Q5: The Reframe

### One-sentence pitch (optimized for LLVM compiler engineers):

> "liboffload loads the first compatible GPU image from a fat binary — we built and measured the runtime policy layer that picks the *best* one, adding 7ns of overhead on a GTX 1650."

### Why this works for THIS audience:

- **Starts with liboffload** — positions inside the ecosystem, not outside it
- **"first compatible"** — every offload engineer knows this is PR #186088's limitation
- **"measured"** — not projected, not borrowed, not simulated
- **"7ns"** — a specific number from specific hardware (update with your actual measurement)
- **"GTX 1650"** — humble hardware signals honest benchmarking, not H100 vanity metrics

### Three-panel poster structure:

```
┌─────────────────────┬────────────────────────┬──────────────────────┐
│  THE DEFERRED        │  A 5100-LOC             │  MEASURED ON A       │
│  FOLLOW-UP           │  EXPLORATION            │  GTX 1650            │
│                      │                         │                      │
│  PR #186088:         │  [5-box architecture    │  [Bar chart:         │
│  "first compatible   │   diagram]              │   cuLaunchKernel     │
│  wins"               │                         │   vs libkdl select   │
│                      │  OffloadBinary          │   + dispatch         │
│  Issue #75356:       │  → device fingerprint   │   vs CPU fallback]   │
│  open since Nov 2023 │  → weighted selection   │                      │
│                      │  → vendor driver        │  Cold start: <5ms    │
│  Huber DevMtg 2025:  │  → execution            │  (measured)          │
│  "ld.so for GPU      │                         │                      │
│   code"              │  3-line API:            │  5100 LOC C          │
│                      │  load → select →        │  (liboffload: ~3K)   │
│  "The compile-time   │  dispatch               │  (IREE HAL: ~15K)   │
│  half is done.       │                         │                      │
│  The runtime policy  │  "Explores what         │  [QR: GitHub repo]   │
│  is not."            │  rankImage() could      │  [QR: Discourse RFC] │
│                      │  look like"             │                      │
└─────────────────────┴────────────────────────┴──────────────────────┘

   libkdl: Runtime Variant Selection for LLVM GPU Offloading
   S. Akash — IIT Patna / CERN — EuroLLVM Dublin 2026
```

---

## The 30-Second Elevator Pitch (for the poster session hallway)

"You know how liboffload's `parseOffloadBinary` loop just loads the first matching image? PR 186088 explicitly defers ranked selection. I built a 5000-line C prototype that implements that missing policy — capability fingerprint, weighted variant scoring, fallback chain — and measured it on a GTX 1650. Dispatch overhead is under 10 nanoseconds. I want to turn this into an upstream RFC for a `rankImage()` callback. Can I show you the numbers?"

This pitch works because:
1. It starts with something they know (liboffload's `parseOffloadBinary`)
2. It cites a specific PR they can verify
3. It claims measured results, not projections
4. It ends with a request for feedback on upstream integration
5. It takes 30 seconds

---

## Honest Assessment of Strengths

Despite the harsh critique above, Akash has genuine advantages that most poster presenters lack:

1. **Working code.** 5157 lines of C that compiles to a real `.so` and real executables. Most posters at this level are 1000-3000 LOC student projects. The code is architecturally clean — dlopen-based vendor discovery, hash-table dispatch, three independent backends.

2. **CERN/GSoC credibility.** The LLVM community knows CERN (ROOT, CMS). The GSoC connection signals "this person knows how open source works." Use it.

3. **Survey depth.** 520 sources across 7 waves is overkill for a poster — but it means every question has an answer. The Q&A prep in the strategy doc covers every likely challenge. This is a massive advantage: the poster is the hook, the Q&A is where you win.

4. **The gap is real.** Issue #75356 (2.5 years open), PR #186088's explicit deferral, Huber's own metaphor — these are not manufactured claims. The LLVM community itself articulated this gap. Akash is the first person to build something that fills it.

5. **Modest hardware.** A GTX 1650 is not impressive, but that is the point. Numbers from humble hardware are more credible than H100 vanity benchmarks. "If it works on a 1650, it works everywhere" is a stronger claim than "we had access to a DGX."

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| "Why not OffloadBinary?" question | **95%** | High | Honest answer: "Prototype used MTB for rapid iteration; OffloadBinary consumption is the planned upstream path. The formats are structurally similar." |
| "Is the cost model real?" challenge | **80%** | Critical | Do not call it roofline. Call it "weighted heuristic" and show the code. Honesty defuses this completely. |
| "How does this relate to IREE?" | **70%** | Medium | "IREE HAL dispatches at module granularity with static boolean selection. libkdl operates at kernel granularity with analytical ranking. They are complementary." |
| Joseph Huber visits the poster | **60%** | High (positive) | Be ready to discuss `rankImage()` callback design for liboffload. This is the single most important conversation at the poster session. |
| "Is this upstreamable?" | **50%** | High | "The prototype explores the design space. The upstream path is a `rankImage()` callback in liboffload, consuming existing OffloadBinary metadata. Here is a Discourse RFC draft." |
| Someone has already done this | **5%** | Critical | The 520-source survey found no prior art. But check the MLIR Workshop talks on April 13 for any late-breaking work. |

---

## References

- `experiments/prototype/src/kdl.c:63-64` — MTB_MAGIC custom format, no OffloadBinary integration
- `experiments/prototype/src/kdl.c:1008` — comment says "roofline" but implementation is weighted heuristic
- `experiments/prototype/src/kdl.c:1051-1054` — hardcoded locality_score constants per vendor
- `experiments/prototype/src/kdl.c:228,248` — real cuModuleLoadData/hipModuleLoadData function pointers
- `experiments/prototype/src/kdl.c:568,749` — actual runtime loading of CUDA/HIP symbols via dlsym
- `experiments/prototype/src/bench_dispatch` — real compiled ELF executable (21280 bytes, Apr 2)
- `experiments/prototype/src/libkdl.so` — real shared library (109576 bytes, Apr 2)
- `wave-07-llvm-poster-criteria.md:230-231` — "Design-only posters get 'interesting' and are forgotten"
- `wave-07-llvm-devmtg-gpu-landscape.md:295-301` — runtime multi-version selection absent from 3 years of workshops
- `wave-07-llvm-poster-criteria.md:129-134` — EuroLLVM 2026 has 12 posters, none on cross-vendor runtime dispatch
- `directions/01-libkdl-ld-so-for-gpu-kernels.md:80-81` — prototype verified on GTX 1650 + CPU
- `poster-strategy-final.md:111-115` — the "So What?" argument (keep for Q&A, not poster face)

---

*Analysis by: architect agent (community fit review pass)*
*Generated: 2026-04-06*
*Perspective: LLVM community insider, not sympathetic observer*
