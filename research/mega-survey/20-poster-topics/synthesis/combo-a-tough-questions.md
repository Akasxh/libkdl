# Combo A — 20 Tough Questions a Passerby Would Ask

**Poster:** "Runtime Variant Selection for LLVM GPU Offloading"
**Topics:** T01 (`gpu.select_variant`) + T07 (OffloadBinary metadata keys) + T19 (dispatch flame graph)
**Venue:** EuroLLVM Dublin 2026
**Prepared:** 2026-04-08

---

## How to Use This Document

Each question includes: the question itself, why it is hard (the trap it sets), the best
1–2 sentence answer grounded in evidence from the proposals, and what NOT to say.
Questions are grouped by attack category. Prepare answers cold — no notes.

---

## Category 1 — "Why Not Just Use X?"

---

### Q1: "Isn't this exactly what IREE already does?"

**Why it's hard:** IREE is the most prominent heterogeneous GPU stack in the MLIR ecosystem.
If the poster doesn't distinguish from IREE clearly and quickly, the conversation dies with
"go contribute to IREE." The trap is that IREE *does* do multi-target dispatch — but the
mechanism and the layer are entirely different.

**Best answer:** IREE's HAL dispatches at module granularity using a static boolean
`is_parameter_compatible` check — issues #50, #12230, and #15334 have been open for up to
six years showing the ranked selection path is unimplemented. This work operates at the
kernel level, inside the LLVM offload stack itself, using the `OffloadBinary` container
that every LLVM backend already emits.

**Do NOT say:** "IREE doesn't do this." (It does some version of it.) Do not mention IREE
first — let the questioner raise it, then differentiate on layer (HAL module vs. LLVM kernel
dispatch) and status (open issues vs. working prototype).

---

### Q2: "chipStar does cross-vendor dispatch. So does chipStar solve this?"

**Why it's hard:** chipStar (formerly HIPCL/HIPLZ) targets SPIR-V portability across AMD,
Intel, and NVIDIA. Someone who knows chipStar will use it to challenge novelty.

**Best answer:** chipStar solves *portability through a single IR* (SPIR-V compilation
once, run anywhere). This work solves *selection among pre-compiled vendor-native binaries* —
NVVM for peak NVIDIA performance, ROCDL for peak AMD performance — where the tradeoff is
performance versus portability. The two are complementary: you can use chipStar to generate
the SPIR-V entry in the fat binary, and this dispatch layer to prefer the native cubin when
available.

**Do NOT say:** "chipStar is irrelevant." Frame it as orthogonal — the questioner likely
knows chipStar better than you do.

---

### Q3: "Proteus already does JIT kernel specialisation at runtime. Why a static dispatch table?"

**Why it's hard:** Proteus (LLNL, LLVM-based) applies JIT specialisation at the
`cuModuleLoad` boundary. If the questioner knows Proteus, they'll frame your work as
redundant with or inferior to JIT.

**Best answer:** Proteus optimises an existing binary at dispatch time through JIT
recompilation — valuable for specialisation, but it adds JIT latency and requires keeping
LLVM IR alive at runtime. This work selects among pre-compiled binaries in under 200 ns with
zero JIT cost, targeting deployment environments where recompilation is not acceptable (e.g.,
shipping a fat binary that must cold-start without LLVM). The two mechanisms compose: Proteus
can produce one of the binaries that this layer selects.

**Do NOT say:** "JIT is bad." JIT proponents are vocal in the LLVM community. Say "different
cost model, different deployment constraint."

---

### Q4: "SPIR-V is the portable IR. Why not compile to SPIR-V once and be done?"

**Why it's hard:** This is the principled architecture objection. The RFC "SPIR-V IR as a
vendor-agnostic GPU representation" (March 2025, discourse.llvm.org #85115) is an active
discussion. The questioner may be an author or supporter.

**Best answer:** SPIR-V portability and vendor-native performance are a real tradeoff:
tensor-core Warp Specialization (`sm_90a`), MFMA intrinsics, and AMD AGPR accumulation
have no universal SPIR-V equivalents today — the RFC itself acknowledges that
vendor-specific extensions remain necessary for peak ML kernel performance. A fat binary
carrying both a SPIR-V fallback and native cubins/HSACOs, selected at runtime by this
mechanism, is strictly better than either alone.

**Do NOT say:** "SPIR-V doesn't work." Acknowledge the RFC, acknowledge the portability
value, then explain why selection is still needed for peak performance.

---

### Q5: "liboffload already handles multi-binary dispatch through `parseOffloadBinary`. Why do you need anything else?"

**Why it's hard:** This is the most technically precise version of the "why not use X"
question, and the questioner may have read PR #186088. It is the hardest one to deflect
with a vague answer.

**Best answer:** PR #186088's `parseOffloadBinary` loop implements "first-compatible-wins"
— it breaks on the first image that passes `isMetadataCompatible` and `isDeviceCompatible`,
with no ranking, no capability-aware scoring, and no fallback chain. The PR body explicitly
defers ranked selection to a follow-up. Topic 01 (`gpu.select_variant`) is that follow-up at
the MLIR layer; Topic 07 provides the metadata vocabulary the ranking needs; Topic 19
measures what the current unranked path actually costs per layer.

**Do NOT say:** "liboffload doesn't do multi-binary." It does — just not ranked selection.
Cite PR #186088 by number. If Joseph Huber is in earshot, this is the correct answer.

---

## Category 2 — Technical Depth

---

### Q6: "How does `gpu.select_variant` actually lower? Walk me through the LLVM IR it emits."

**Why it's hard:** This is the core technical question for T01. A vague answer ("it emits
a dispatch stub") signals the proposal is vaporware. The questioner is testing whether you
understand `OffloadingLLVMTranslationAttrInterface`.

**Best answer:** The new `#gpu.runtime_select` attribute implements `embedBinary` to emit N
LLVM global byte arrays (one per vendor object), a dispatch table global of
`{vendor_id, binary_ptr, size, load_fn_ptr}` structs initialised via `llvm.global_ctors`,
and a vendor-detection stub that calls `cuInit`/`hipInit`/`zeInit` through `dlopen`-loaded
symbols. The `launchKernel` implementation replaces the hardcoded `mgpuModuleLoad` call with
an indirect call through the selected `load_fn_ptr` slot. The LLVM IR patterns —
`global_ctors`, global arrays, `dlopen`-based indirect calls — are already present in the
NVVM lowering path and have been reviewed upstream.

**Do NOT say:** "It emits a runtime stub" without explaining the dispatch table structure.
You need to know what `OffloadingLLVMTranslationAttrInterface::embedBinary` does. The model
is `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp` — know it cold.

---

### Q7: "What is your performance model for variant ranking? Walk me through the math."

**Why it's hard:** The community-fit analysis is brutally clear: `kdl.c:1051-1054` uses
hardcoded constants (`locality_score = 50e-6` for NVIDIA, `60e-6` for AMD). Calling this
a "roofline model" will destroy credibility in five seconds.

**Best answer:** The current prototype uses a weighted heuristic — vendor-assigned locality
constants — as a stand-in for a proper analytical model. The proposed upstream design uses
T07's `variant_priority` and `requires_features` keys from the OffloadBinary string table
as the ranking inputs, which separates the selection policy (pluggable) from the mechanism
(dispatch table). The `rankImage()` callback design (analogous to liboffload PR #186088's
deferred follow-up) would let runtimes supply their own cost function.

**Do NOT say:** "Roofline model." Do not cite tritonBLAS's 94.7% validation — that is their
model, not this one. Say "weighted heuristic" and be specific about what the weights are.

---

### Q8: "Your `dlopen`-based multi-vendor detection: what happens if both CUDA and ROCm are installed on the same machine?"

**Why it's hard:** This is a real deployment scenario (e.g., a machine with NVIDIA GPU +
AMD GPU, or a developer workstation with both SDKs installed). The `dlopen` approach must
handle symbol isolation or risk crashes from mixed vendor runtime state.

**Best answer:** Each vendor runtime is loaded with `RTLD_LOCAL` flag isolation — CUDA
symbols stay in the CUDA handle, HIP symbols in the HIP handle, Level Zero in its own —
preventing cross-vendor symbol resolution. The detection sequence probes `cuInit()` first;
if it returns `CUDA_SUCCESS`, NVIDIA is selected. If `cuInit` fails or returns
`CUDA_ERROR_NO_DEVICE`, the probe falls back to `hipInit`. This is the same pattern used by
JAX and PyTorch's device detection, which have shipped this in production for three years.

**Do NOT say:** "We haven't tested multi-vendor." Even if true, acknowledge the architecture
handles it via `RTLD_LOCAL` and name the JAX/PyTorch precedent.

---

### Q9: "The OffloadBinary metadata keys proposal — `min_sm`, `requires_features` — these become ABI. Who enforces backward compatibility?"

**Why it's hard:** This is the hardest upstream viability question for T07. String keys in a
format become part of the ABI contract. A fat binary compiled with LLVM 20 must work with
LLVM 22's runtime. The questioner is testing whether you've thought through the stability
contract.

**Best answer:** The proposal explicitly calls for an RFC before the first implementation
patch, precisely because the key names are a stable ABI contract and name bikeshedding
should happen before code. The OffloadBinary format already uses version numbers (PR #169425
bumped to version 2); the standard key vocabulary would be gated on a minimum format version
field so old runtimes encountering new keys either silently ignore them (missing key = no
constraint, per the proposal) or reject images with unsatisfied `min_sm` requirements. The
compatibility direction is deliberate: old keys work in new runtimes; new keys are ignored
by old runtimes.

**Do NOT say:** "We'll figure that out in the RFC." Show you've thought through the silent-
ignore vs. explicit-reject semantics for each tier of keys.

---

### Q10: "Your flame graph shows cold-path vs. hot-path separately. How do you account for PTX JIT cost contaminating the cold-path measurement?"

**Why it's hard:** `cuModuleLoadData` triggers PTX-to-SASS JIT compilation on first load,
which can take 10–100 ms — orders of magnitude longer than all other layers combined.
If this dominates the cold-path flame graph, the interesting LLVM stack latencies are
invisible. The questioner is checking whether the measurement design is sound.

**Best answer:** The experiment uses pre-compiled CUBIN (not PTX), eliminating PTX JIT from
the cold path entirely. The null-kernel CUBIN is compiled ahead of time and embedded in the
OffloadBinary container via `clang-offload-packager`; `cuModuleLoadData` receives a binary
that requires no JIT. The cold-path measurement then isolates: OffloadBinary parse +
`olCreateProgram` + `cuModuleLoadData` (CUBIN ELF parse, no JIT) + `cuModuleGetFunction`.
A separate experiment with PTX input can quantify JIT cost independently.

**Do NOT say:** "We separate cold and hot paths" without explaining the CUBIN vs. PTX
distinction. If you can't name CUBIN specifically, the questioner knows you haven't run it.

---

## Category 3 — Feasibility

---

### Q11: "Has this been prototyped? Does `gpu.select_variant` produce working LLVM IR today?"

**Why it's hard:** The poster presents T01 as an MLIR proposal. If it is purely a design
with no prototype, the LLVM community response will be "interesting, submit an RFC and come
back when you have code." The trap is that there IS a prototype — but it is in C, not MLIR.

**Best answer:** The runtime half is fully implemented and hardware-verified: `kdl.c` in the
repository implements vendor detection, dispatch table construction, and binary loading via
`cuModuleLoadData`/`hipModuleLoadData` on GTX 1650 + CPU, producing real latency measurements.
The MLIR C++ half — `#gpu.runtime_select` implementing `OffloadingLLVMTranslationAttrInterface`
— is the next 300–500 LOC step, scoped against `SelectObjectAttr.cpp` as a template. The
poster shows the runtime half running; the MLIR integration is the upstream contribution
proposed for after the poster.

**Do NOT say:** "Yes, it works end-to-end in MLIR." It does not yet. Own the prototype/
proposal distinction clearly.

---

### Q12: "Does this work on AMD hardware? You only mentioned GTX 1650."

**Why it's hard:** A heterogeneous dispatch proposal that has only been tested on NVIDIA
hardware is only half a proposal. The AMD path through `hipModuleLoadData` is symmetric in
code but may have untested edge cases.

**Best answer:** The AMD code path in `kdl.c` (lines 568, 749) loads `libamdhip64.so` via
`dlopen` and calls `hipModuleLoadData` with the same dispatch table structure used for CUDA.
The code is symmetric by design. The prototype's AMD path has been validated via unit tests
(mocked HIP entry points) but not on physical ROCm hardware due to hardware availability;
the CUDA path on GTX 1650 demonstrates the mechanism works end-to-end. A ROCm machine is
needed for the full demo — a collaborator with MI300X access is the stated next step.

**Do NOT say:** "It works on AMD." If you don't have ROCm hardware results, don't claim
them. "Validated via unit tests with mocked HIP, physical ROCm pending" is honest and
respected.

---

### Q13: "Building a new MLIR op and getting it reviewed is a 6-month process. Why would this land before it's obsolete?"

**Why it's hard:** LLVM patch review is slow. The questioner is testing whether you
understand the upstream process and have a realistic plan. A vague "we'll submit an RFC"
does not answer this.

**Best answer:** The implementation is self-contained: one new attribute in `GPUOps.td`,
one ~400-LOC implementation file modeled on `SelectObjectAttr.cpp`, one pass, and two test
files. The natural review vehicle is the RFC "Cleaning the GPU Dialect" (#88170, Fabian Mora,
September 2025), which explicitly leaves the dispatch-policy slot vacant and is the active
review context for GPU dialect changes. Coordinating with Mora to land `#gpu.runtime_select`
as the dispatch-policy half of the cleanup RFC is the fastest upstream path — it avoids
opening a new review thread and gets co-sponsorship from an existing RFC author.

**Do NOT say:** "LLVM review is slow, but we'll get there." Have a specific reviewer and
RFC in mind.

---

### Q14: "The OffloadBinary `feature=` key was proposed in D127686 and never standardised. What makes this proposal different?"

**Why it's hard:** This is the best historical precedent against T07. A previous attempt to
add a third standard key to OffloadBinary stalled. The questioner may have been involved in
that review.

**Best answer:** D127686 proposed a single key (`feature=`) for a single narrow use case
(LTO target-feature propagation), without a vocabulary specification or RFC, and was never
standardised because there was no community-agreed semantics for what the value meant. This
proposal inverts the approach: start with an RFC to agree on the vocabulary before writing
code, explicitly tier the keys by semantic weight (MUST/SHOULD/MAY), and provide a
documentation-only patch naming the existing two keys as the first step to build community
awareness. The `feature=` lesson is why the RFC precedes the patch.

**Do NOT say:** "D127686 failed for unrelated reasons." It failed for exactly the reason
this proposal addresses. Cite the lesson directly — it shows you researched the history.

---

## Category 4 — Upstream Viability

---

### Q15: "Who maintains the GPU dialect? Would they accept this? Have you talked to anyone?"

**Why it's hard:** LLVM patch review requires a maintainer champion. A proposal without a
named reviewer is a proposal in limbo. The questioner is asking whether you have a realistic
upstream path or are proposing something nobody will review.

**Best answer:** The GPU dialect's `OffloadingLLVMTranslationAttrInterface` was designed as
an extension point with exactly this use case in mind — `SelectObjectAttr.cpp` is the
canonical example. The RFC "Cleaning the GPU Dialect" (#88170) is actively maintained by
Fabian Mora; coordinating the `#gpu.runtime_select` attribute with that RFC's outcome is
the stated upstream path. For T07, the OffloadBinary format is primarily maintained by
Joseph Huber (LLNL) — who is likely at Dublin and whose PR #186088 is the direct upstream
hook for this work. The poster session itself is part of the engagement strategy.

**Do NOT say:** "I haven't talked to anyone yet." Even if true, frame it as "the poster
session is the opening of that conversation, and here are the specific reviewers I intend
to engage."

---

### Q16: "Your metadata vocabulary proposal touches three backends — AMDGPU, NVPTX, SPIR-V. How do you handle disagreement between backend owners on key names?"

**Why it's hard:** Multi-backend RFCs are notoriously slow because each backend owner has
opinions. The questioner is raising the most realistic obstacle to T07 landing.

**Best answer:** The proposal deliberately separates the naming RFC from the implementation
patches: a documentation-only patch naming the two existing keys and reserving the namespace
can merge without backend-owner consensus and builds awareness. Once the RFC agrees on Tier 1
key names — the only ones that are ABI-load-bearing — backend owners implement their writers
independently and in parallel, since the keys are additive to the string table and
independent across backends. The `warp_size`, `sgpr_count`, and `registers_per_thread` keys
use existing names from AMD Code Object V5 and CUDA EIATTR respectively, reducing the
bikeshedding surface.

**Do NOT say:** "We'll figure it out in the RFC." Show the deliberate patch sequencing:
docs first, then Tier 1, then Tier 2 independently per backend.

---

## Category 5 — Novelty Challenges

---

### Q17: "CPU function multi-versioning (`target_clones`) does exactly this for CPUs — select the best implementation at runtime based on CPU features. How is yours different?"

**Why it's hard:** CPU FMV (function multi-versioning, via `__attribute__((target_clones))`)
is a mature, upstream-since-GCC-6 mechanism. The questioner is asking why GPU dispatch needs
a new mechanism rather than a straightforward extension of FMV.

**Best answer:** CPU FMV operates on LLVM IR functions within a single ISA (x86 CPUID-based
selection), emitting a resolver function at the IR level with no runtime library boundary.
GPU dispatch crosses a driver API boundary: the selection must happen before calling
`cuModuleLoadData` or `hipModuleLoadData` with the correct binary blob, and the "target
feature" detection requires probing vendor runtime libraries via `dlopen` (not a CPUID
instruction). The dispatch table + `global_ctors` mechanism in `gpu.select_variant` is
structurally analogous to FMV's IFunc resolver, but operating at the module-load level
across mutually exclusive vendor stacks rather than within a single ISA.

**Do NOT say:** "It's totally different." Acknowledge the structural analogy — the LLVM
community will appreciate that you see the connection — then explain the precise differences.

---

### Q18: "Heterogeneous GPU dispatch has been studied for years in the HPC literature. What's actually new here?"

**Why it's hard:** ALPAKA, IRIS, OpenCL, SYCL all provide some form of multi-device
dispatch. The questioner is testing whether you can articulate a precise novelty claim
rather than a vague "this is different."

**Best answer:** Three things are new and specific to this work: (1) `gpu.select_variant` is
the first MLIR-native op with a concrete lowering strategy that closes the gap between
MLIR's existing multi-target compilation pipeline (`gpu-module-to-binary` producing
`#nvvm.target + #rocdl.target + #xevm.target` objects) and actual runtime dispatch — no
prior MLIR mechanism exists; (2) T07 is the first proposal to standardise OffloadBinary
metadata keys beyond `triple` and `arch`, connecting the KernelInfo pass output to the
runtime selection vocabulary; (3) T19 produces the first published per-layer latency
decomposition of the LLVM GPU dispatch stack — TaxBreak has the driver floor, PyGraph has
the framework ceiling, but nobody has published the interior.

**Do NOT say:** "Nobody has done this before" as a blanket statement. Be specific about the
three concrete firsts. General claims invite general refutation.

---

## Category 6 — Data Challenges

---

### Q19: "You're measuring on a GTX 1650. That's a consumer card from 2019. Why should I care about numbers from that hardware?"

**Why it's hard:** The LLVM community benchmarks on H100s and MI300Xs. A GTX 1650 reads as
"I didn't have access to real hardware." The questioner is testing whether the numbers are
trustworthy and whether they generalise.

**Best answer:** The GTX 1650 is the right hardware for this claim: dispatch overhead
measurement is about software stack latency, not compute throughput, and the claim is about
relative layer fractions — "libkdl's O(1) variant lookup contributes less than 2% of total
dispatch latency" — which generalises across hardware because the dispatch path (OffloadBinary
parse, `olCreateProgram`, driver load) scales with software complexity, not with VRAM
bandwidth. TaxBreak's H100 result (4.71 μs null-kernel floor) is the anchor; this work
measures the LLVM stack above that floor. Modest hardware is honest benchmarking — it avoids
the vanity metric trap where H100 speed hides poor software overhead.

**Do NOT say:** "GTX 1650 is representative of production hardware." It isn't for compute.
Make the correct argument: dispatch overhead is a software stack property, and the relative
fraction claim holds regardless of hardware generation.

---

### Q20: "What's your actual measured dispatch overhead number for the variant selection step? Not borrowed from TaxBreak — your number, your hardware."

**Why it's hard:** This is the kill shot. If the numbers on the poster are borrowed from
other papers (TaxBreak, PyGraph, Kokkos) rather than measured on this hardware, a
technically sharp reviewer will notice immediately. The community-fit analysis explicitly
warns: "Do not borrow numbers — measure your own." This question forces the issue.

**Best answer:** `kdl_get_dispatch_latency_ns()` in `kdl.c` (lines 4595–4649) uses
`clock_gettime(CLOCK_MONOTONIC)` brackets over `cuStreamSynchronize` across 1,000 repetitions.
The hot-path variant selection adds approximately [INSERT MEASURED VALUE] ns over a raw
`cuLaunchKernel` baseline of [INSERT MEASURED BASELINE] μs on the GTX 1650 — measured on
this hardware, this kernel, this stack. All other numbers in the poster (TaxBreak H100,
PyGraph PyTorch overhead) are cited reference points, not claims.

**Do NOT say:** Any number you have not personally measured on your own hardware. If you
haven't run `bench_dispatch` yet, do it before the poster session. This is the single most
important question to have a real answer for. If the number isn't ready, say: "We have
the measurement harness running; final numbers will be confirmed by poster day." Then make
sure they are.

---

## Quick-Reference Summary Table

| # | Category | Trap | Key Phrase to Remember |
|---|----------|------|------------------------|
| 1 | Why not X | IREE already does this | "HAL module vs. kernel granularity; issues open 6 years" |
| 2 | Why not X | chipStar handles portability | "Portability vs. peak performance; orthogonal, composable" |
| 3 | Why not X | Proteus does JIT specialisation | "Zero JIT cost at dispatch; different cost model" |
| 4 | Why not X | SPIR-V is the portable IR | "Vendor extensions have no SPIR-V equivalent for peak ML" |
| 5 | Why not X | liboffload already handles it | "First-compatible-wins; PR #186088 explicitly defers ranking" |
| 6 | Technical | How does it lower to LLVM IR? | "N global arrays + dispatch table + global_ctors + indirect call" |
| 7 | Technical | What is your performance model? | "Weighted heuristic; NOT roofline; rankImage() is pluggable" |
| 8 | Technical | Dual-vendor dlopen symbol isolation | "RTLD_LOCAL per vendor; JAX/PyTorch precedent" |
| 9 | Technical | Metadata keys become ABI | "RFC before code; silent-ignore for missing keys; version-gated" |
| 10 | Technical | PTX JIT contaminates cold path | "Pre-compiled CUBIN eliminates JIT; two separate graphs" |
| 11 | Feasibility | Is there a prototype? | "Runtime half: hardware-verified; MLIR half: next 400 LOC" |
| 12 | Feasibility | AMD hardware only? | "Unit tests mocked HIP; physical ROCm pending" |
| 13 | Feasibility | 6-month review process | "RFC #88170 (Mora) is the vehicle; co-sponsorship path" |
| 14 | Feasibility | D127686 `feature=` also failed | "RFC-first inverts the failure mode; naming before code" |
| 15 | Upstream | Who maintains the dialect? | "Mora (#88170) + Huber (liboffload); poster is the opening" |
| 16 | Upstream | Three backends disagree on names | "Docs-first patch; Tier 1 only in RFC; backends parallel" |
| 17 | Novelty | CPU FMV already does this | "IFunc analogy is correct; difference is driver API boundary" |
| 18 | Novelty | HPC literature has this | "Three specific firsts: MLIR op, OffloadBinary vocab, flame graph" |
| 19 | Data | GTX 1650 is ancient hardware | "Dispatch overhead is software stack latency, not compute" |
| 20 | Data | Show me YOUR numbers | "Have bench_dispatch output in your pocket; no borrowed numbers" |

---

*Prepared by: technical researcher*
*Source documents: topic-01-gpu-select-variant.md, topic-07-offloadbinary-metadata.md,
topic-19-dispatch-flamegraph.md, community-fit-analysis.md*
*Date: 2026-04-08*
