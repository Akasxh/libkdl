# Devil's Advocate: Combo A Destruction Report

**Reviewer:** Critic (adversarial mode)
**Date:** 2026-04-09
**Verdict:** This proposal has serious structural dishonesty problems that will be exposed in under 60 seconds by anyone who has read the LLVM source.

---

## 10 Fatal Flaws

### F1. The prototype does not use OffloadBinary — it uses a proprietary format (CRITICAL)

The entire Combo A narrative is: "we propose standard OffloadBinary metadata keys (T07), a runtime selection op that consumes them (T01), and a flame graph measuring the OffloadBinary dispatch path (T19)."

The prototype (`kdl.c`) uses **`KDL_MTB\0`** — a custom Multi-Target Bundle format defined at line 63 of `kdl.c` with its own `mtb_header` struct. There is zero OffloadBinary code in the prototype. No `0x10FF10AD` magic. No `OffloadBinary::create()`. No `olCreateProgram`. No `olGetSymbol`. No `olLaunchKernel`.

The proposals repeatedly claim the prototype "demonstrates the runtime half" of the upstream contribution. It demonstrates a runtime half *of a completely different format*. The mapping between MTB dispatch and OffloadBinary dispatch is hand-waved ("the poster maps libkdl's MTB dispatch loop directly to the LLVM IR") but never implemented or tested.

**Impact:** Any attendee who runs `grep OffloadBinary kdl.c` gets zero results and the credibility of all three topics collapses simultaneously.

**Fix:** Either rewrite the prototype to actually consume OffloadBinary containers (using `clang-offload-packager` to produce them), or be brutally honest on the poster: "the prototype uses a custom MTB format; the upstream proposal translates these concepts to OffloadBinary." The first option is the only one that survives scrutiny.

---

### F2. The "O(1) variant lookup adds ~100-200 ns" claim has no supporting measurement (CRITICAL)

Topic-19 states: "libkdl's O(1) hash-table variant lookup contributes less than 2% of that budget." The tough-questions doc (Q20) has a literal `[INSERT MEASURED VALUE]` placeholder.

`grep` for "100.*ns", "200.*ns", "O(1)", and "hash.*lookup" in `kdl.c` returns zero results. The `experiments/prototype/results/` directory contains only three pre-generated PNGs and a demo MTB file — no raw latency data, no CSV, no measurement output.

The `kdl_get_dispatch_latency_ns()` function (lines 4605-4649) measures `cuStreamSynchronize` over 100 reps — this measures stream flush latency, not variant selection latency. There is no function that times `kdl_select_kernel()` in isolation.

**Impact:** The poster's central quantitative claim — the number that validates the entire libkdl thesis — does not exist as a measurement. It is a guess.

**Fix:** Instrument `kdl_select_kernel()` with `clock_gettime` brackets. Run `bench_dispatch` (which does exist at `benchmarks/bench_dispatch.c`). Produce actual numbers. Do this before writing a single word on the poster.

---

### F3. Topic-19's flame graph measures a path the prototype cannot execute (CRITICAL)

The flame graph proposal instruments: `OffloadBinary::create()` → `olCreateProgram` → `cuModuleLoadData` → `olGetSymbol` → `olLaunchKernel`.

The prototype calls: `dlopen("libcuda.so.1")` → `cuInit` → `cuCtxCreate` → `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel`.

These are two completely different dispatch paths. The prototype bypasses liboffload entirely. To produce the flame graph as proposed, the author must build liboffload from LLVM source and write a *new* measurement harness that calls the `ol*` API — the existing `kdl.c` infrastructure cannot be "extended" for this because it doesn't use liboffload at all.

The proposal says "the prototype timing infrastructure already exists in `kdl.c`" and "estimated work: 1-2 weeks." The actual work is: build liboffload, write a new 200+ LOC harness from scratch that uses `ol*` APIs, debug linking against an explicitly unstable API, and produce measurements. That is not an extension of existing code.

**Fix:** Rewrite the feasibility section honestly. The flame graph requires a new harness, not an extension of kdl.c.

---

### F4. The XeVM PR number is wrong and was already caught — but the fix is missing (HIGH)

The gap analysis (combo-a-gaps.md) identified this as "the most serious factual error" on 2026-04-08. PR #119440 is the ELF section pass, not XeVM upstreaming. As of 2026-04-09 (one day later), the error is still present in topic-01.

**Impact:** At Dublin, anyone who has touched the MLIR GPU dialect in the last year will know PR #119440 is Renaud-K's ELF section pass, not Intel XeVM. This is a credibility-destroying error on the poster's core "tri-vendor urgency" claim.

**Fix:** Find the actual XeVM PR. The gap analysis suggests #148286 or nearby. Do not present until this is verified.

---

### F5. gpu.select_variant does not exist as code — not even a sketch (HIGH)

There is no C++, no TableGen, no MLIR test file for `gpu.select_variant` or `#gpu.runtime_select` anywhere in the repository. The proposal describes it with precision ("~300-500 LOC C++") but the implementation is 0 LOC.

For an LLVM poster, "I propose an op and here is my prototype in a different language using a different format" is weak. For comparison, the accfg poster (ETH Zurich, 2024) presented working MLIR passes with upstream patches. This proposal presents a C prototype that shares no code, no format, and no API with the proposed MLIR contribution.

**Impact:** The poster will be perceived as a design proposal, not a contribution. Design proposals without code are what RFCs are for, not poster sessions.

**Fix:** Implement a minimal `RuntimeSelectAttr.cpp` (even 100 LOC that emits the multi-blob globals without the vendor detection stub). Show it producing LLVM IR on one test case. This transforms the poster from "we propose" to "we implemented."

---

### F6. The RFC "Cleaning the GPU Dialect" (#88170) is unresolved and could invalidate T01 (HIGH)

Topic-01's upstream path depends entirely on coordinating with RFC #88170 (Fabian Mora, September 2025). The RFC is active/unresolved as of April 2026. If the RFC concludes by restructuring `gpu.binary` semantics — for example, removing the container model entirely in favor of a different representation — the `#gpu.runtime_select` attribute has no landing point.

The proposal treats the RFC outcome as favorable without hedging: "coordinate with Fabian Mora to land `#gpu.runtime_select` as the dispatch-policy half of the cleanup." This presumes the cleanup creates a policy slot. It might not.

**Fix:** Add an explicit contingency: "If RFC #88170 concludes without a dispatch-policy slot, `#gpu.runtime_select` can land independently as a standalone attribute — the `OffloadingLLVMTranslationAttrInterface` extension point does not depend on the cleanup RFC's structural changes."

---

### F7. Topic-07's NVPTX writer path references a file that likely doesn't exist (HIGH)

Topic-07 proposes implementing the NVPTX metadata writer in `clang/tools/clang-offload-wrapper/ClangOffloadWrapper.cpp`. The gap analysis notes this file may have been superseded by the new unified driver (default since LLVM 19, PR #84420). The correct location is likely under `clang/tools/clang-linker-wrapper/` or `offload/`.

This is not a cosmetic error — it means the implementation plan for one of the three topics targets a wrong (possibly deleted) file. An attendee who works on the offload toolchain will immediately know this.

**Fix:** Verify the actual file in the LLVM monorepo before the poster. The proposal should name the correct file or say "the NVPTX writer integration point is in the clang-linker-wrapper pipeline" without naming a specific file that might not exist.

---

### F8. TaxBreak "4.71 μs" attribution is unverified and load-bearing (HIGH)

The gap analysis flags this as critical. The TaxBreak paper (arXiv:2603.12465) is about LLM inference overhead decomposition, not null-kernel microbenchmarking. The specific figures "4.71 μs avg, p50: 4.578 μs, p95: 5.396 μs" could not be verified against the paper.

Topic-19's entire pitch is: "the community has the floor (TaxBreak) and the ceiling (PyGraph) but zero published data on the interior." If the floor number is wrong or misattributed, the positioning collapses.

**Fix:** Read the actual TaxBreak PDF. Confirm the numbers. If they come from a different source (ICPP 2019, NVIDIA blog), fix the attribution.

---

### F9. Topic-19's Layer 3 measurement design double-loads the CUDA module (HIGH)

The measurement code in topic-19 calls `olCreateProgram` (which internally calls `cuModuleLoadData`) AND separately calls `cuModuleLoadData` with the same blob. This creates two module instances, not a decomposition of one dispatch. The gap analysis caught this.

**Fix:** Use approach (b) from the gap analysis: measure `cuModuleLoadData` in a separate run bypassing liboffload entirely, then subtract from `olCreateProgram` total to infer plugin overhead.

---

### F10. The prototype has never been tested on AMD hardware (HIGH)

The project docs say "verified on GTX 1650 + CPU." The tough-questions doc (Q12) says "physical ROCm pending." The kdl.c AMD code path loads `libamdhip64.so` via dlopen and calls HIP functions — but this path has only been validated via "unit tests (mocked HIP entry points)."

A "heterogeneous GPU kernel dispatch" poster that has only been tested on one vendor's hardware is not heterogeneous. It is single-vendor dispatch with aspirational multi-vendor code paths.

Topic-07's pitch falsely claims "a prototype already running on GTX 1650 + MI300X under libkdl." The gap analysis already flagged this as fabricated. No MI300X appears anywhere in the prototype or its results.

**Fix:** Remove MI300X from every claim. State "validated on NVIDIA (GTX 1650) and CPU; AMD code path tested via mocked HIP" and nothing more.

---

## 10 Weak Claims (stated as fact, actually assumptions)

### W1. "Novelty Score: 9/10 — No MLIR-native runtime variant selection mechanism exists today"

This assumes IREE's HAL dispatch, chipStar's SPIR-V runtime selection, and Proteus's JIT specialization don't count. They do count as prior art in runtime variant selection for ML workloads. The 9/10 is defensible only within the narrow frame of "MLIR GPU dialect op" — which is a definitional trick, not a novelty assessment.

### W2. "~300-500 LOC C++ for the attribute implementation"

Based on "SelectObjectAttr.cpp is ~200 LOC; the new version is roughly 2-2.5x." This ignores that the new version must emit vendor detection stubs with `dlopen` logic, `global_ctors` initialization, and multi-blob management — functionality that has no precedent in SelectObjectAttr.cpp. 500-1000 LOC is more realistic, and the `dlopen` policy debate alone could double the review scope.

### W3. "libkdl's O(1) hash-table variant lookup adds ~100-200 ns"

As established in F2, this number has never been measured. The "O(1)" characterization is also misleading — `kdl_select_kernel_internal` (line 1286) iterates through variants with a linear scan of capability checks, not a hash lookup.

### W4. "Estimated work: 1-2 weeks of focused effort" (Topic-19)

Requires building liboffload from source, writing a new measurement harness for the `ol*` API (which the prototype doesn't use), debugging against an unstable API, and producing publication-quality flame graphs. 1-2 weeks assumes zero debugging time with an API that renamed a core function 3 months after introduction.

### W5. "The community is actively seeking this contribution"

Based on: the GPU/Offloading Workshop themes are "where are we going?" and RFC #88170 has "three pages of active discussion." Having discussion is not the same as seeking this specific contribution. The RFC discussion may conclude that runtime dispatch belongs in liboffload (below MLIR), not in the GPU dialect.

### W6. "The `OffloadingLLVMTranslationAttrInterface` is already the correct extension point and has precedent"

Correct as a technical statement. But the precedent (SelectObjectAttr) is compile-time-only. Using the same interface for runtime dispatch — which requires emitting vendor-detection stubs, `dlopen` calls, and dispatch tables — is a fundamentally different use that may face reviewer objections about scope creep of the interface.

### W7. "relative layer fractions, not absolute values, generalize across hardware"

This assumes the ratio of time spent in OffloadBinary parse vs. cuModuleLoadData vs. cuLaunchKernel is hardware-independent. It is not. On H100 with NVLink, `cuModuleLoadData` may be faster relative to `cuLaunchKernel` than on GTX 1650 with PCIe 3.0. The proportional breakdown is hardware-dependent.

### W8. "Making opt-in trivial" (the --gpu-mark-select-variant pass)

Claims the pass "walks `gpu.binary` ops that carry two or more `#gpu.object` entries and replaces the implicit `#gpu.select_object<0>` handler." This assumes the current `#gpu.select_object<0>` is always the default handler and can be safely replaced. If downstream users depend on deterministic compile-time selection (for reproducibility, testing, or debugging), an opt-in pass that changes default behavior is not trivial — it is a behavioral change that requires careful opt-out support.

### W9. "Topic-07 has the cleanest evidence base" (gap analysis)

Topic-07 proposes 20+ new string-map keys across 4 tiers, touching 3 backends. D127686 tried to add ONE key (`feature=`) and failed to standardize. The evidence that one key failed is not evidence that 20+ keys will succeed — it is evidence of the opposite.

### W10. "The poster maps libkdl's MTB dispatch loop directly to the LLVM IR that `#gpu.runtime_select::embedBinary` would emit"

This mapping is asserted but never demonstrated. The MTB format uses custom headers (`KDL_MTB\0`, `mtb_header`), custom kernel tables, and custom variant matching. OffloadBinary uses a completely different wire format with different metadata semantics. "Directly maps" is a stretch — "is loosely analogous to" is accurate.

---

## 5 "Someone Already Did This" Risks

### S1. Topic-06 (rankImage) from your OWN survey is the same contribution as Topic-01

Topic-06 proposes `rankImage()` — a variant selection callback for liboffload — targeting the exact same PR #186088 gap with the exact same "first-compatible-wins" problem. The only difference is the layer: Topic-01 operates at MLIR level, Topic-06 at liboffload level. If Joseph Huber (who is "likely at Dublin") has already implemented `rankImage()` or something equivalent as a follow-up to PR #186088 by April 2026, Topic-01's MLIR-level approach becomes redundant. Check whether this follow-up landed.

### S2. IREE's Phase 2-3 multi-versioning could land before Dublin

Topics 01 and 07 lean heavily on IREE issues #50, #12230, #15334 being open for "up to six years." If IREE ships ranked HAL variant selection in 2026 Q1-Q2 (which is plausible given active development), the "nobody has done this" claim weakens considerably.

### S3. liboffload's own isMetadataCompatible evolution

PR #185663 (merged March 10, 2026) introduces `isMetadataCompatible` as a virtual method. PR #186088 adds `OffloadBinMetadataTy` with a `StringMap<string>` for arbitrary key-value metadata. The liboffload team is building the runtime consumer infrastructure for metadata-based selection. They may define their own de facto metadata vocabulary simply by using it in the compatibility checks — without waiting for an RFC. The de facto standard could preempt Topic-07's de jure proposal.

### S4. Proteus (LLNL) runtime specialization

Proteus already operates at the `cuModuleLoad` boundary with LLVM-based JIT. If Proteus adds a "select among pre-compiled variants" mode (which is simpler than their JIT mode), it directly competes with Topic-01's dispatch mechanism. Proteus has an existing LLNL team, existing upstream presence, and institutional backing.

### S5. CUDA 13 / ROCm 7 driver-level multi-version selection

NVIDIA and AMD both have proprietary fat-binary selection logic in their drivers. If CUDA 13 (expected late 2026) or ROCm 7 adds a standard driver-level API for ranked multi-version selection from a single binary, the entire LLVM-level dispatch layer becomes unnecessary. The "LLVM should do this" argument only holds if the vendors don't.

---

## The Single Biggest "So What?" Problem

**A busy LLVM contributor will ask: "Who actually has a fat binary with NVVM + ROCDL + XeVM objects that they need to dispatch at runtime?"**

The answer today is: nobody in production. The proposal correctly identifies that MLIR *can* produce such a binary — but no real-world ML framework (PyTorch, JAX, TensorFlow, Triton) currently ships fat binaries through the MLIR GPU dialect pipeline with multiple vendor targets. They all use their own dispatch mechanisms.

The proposal is solving a problem that exists in theory (MLIR's multi-target compilation capability) but not in practice (no downstream consumer). The gap is real at the infrastructure level, but the absence of any downstream user who would actually call `gpu.select_variant` today means the contribution has zero immediate users.

The counter-argument ("build it and they will come" — MLIR infra enables future frameworks) is valid but weak for a poster. A poster needs to demonstrate impact, not potential. The strongest version of this poster would show ONE real framework (even a toy) producing a fat binary through MLIR and dispatching it with `gpu.select_variant` on two different GPUs. Without that, it's an infrastructure proposal with no demonstrated user.

---

## The Strongest Counter-Narrative

**"This entire direction is wrong. Runtime variant selection belongs in the runtime (liboffload), not in the compiler IR (MLIR). Putting dispatch policy in MLIR violates separation of concerns and will be rejected by both the MLIR and liboffload maintainers."**

The argument:

1. MLIR's job is to compile code. liboffload's job is to run it. Dispatch policy — which binary to load on which device — is a runtime decision that depends on runtime state (which devices are present, their load, their temperature). Encoding this in MLIR IR bakes compile-time assumptions into what should be a runtime decision.

2. The `OffloadingLLVMTranslationAttrInterface` was designed for compile-time code generation, not runtime policy emission. Using it to emit `dlopen`-based vendor detection stubs, `global_ctors` dispatch tables, and indirect calls is scope creep that transforms the MLIR translation layer into a runtime code generator.

3. liboffload (PR #186088) is already building the runtime selection infrastructure at the correct layer. The `isMetadataCompatible` + `isDeviceCompatible` pipeline in `PluginInterface.cpp` is where selection policy belongs. Adding a `rankImage()` callback (Topic-06) is the right fix at the right layer. `gpu.select_variant` in MLIR is the right fix at the wrong layer.

4. Joseph Huber (liboffload maintainer, likely at Dublin) has explicitly stated that liboffload excludes selection policy from its roadmap — but this means he wants it *not in liboffload*, not that he wants it *in MLIR*. He may want it in user code (the "Not-Compiler Runtime Library GPUs" talk). Putting it in MLIR may be opposed by both the MLIR dialect maintainers (who want the GPU dialect to be clean per RFC #88170) and the liboffload maintainers (who want dispatch to be a runtime concern).

5. The strongest precedent — IREE's HAL, JAX's device placement, PyTorch's device dispatch — all implement selection in their runtimes, not in their IR layers. Nobody compiles selection policy into their IR. There's a reason for this.

This counter-narrative is the one the poster must address head-on. If the presenter cannot articulate why MLIR is the right layer for dispatch policy (not just "it can be done here" but "it must be done here"), the strongest audience members will walk away unconvinced.

---

## Summary

The Combo A proposal identifies real gaps — nobody disputes that MLIR lacks runtime dispatch, OffloadBinary metadata is impoverished, and the dispatch path is uninstrumented. The problem is not the gaps; the problem is the evidence:

- The prototype uses a different format than the proposal targets
- The central quantitative claim has never been measured
- The flame graph requires infrastructure the prototype doesn't have
- Key citations are wrong or unverified
- The "heterogeneous" claim rests on single-vendor hardware testing
- Zero lines of MLIR C++ exist

This is a proposal that is 80% narrative and 20% implementation, presented as though the ratio is inverted. At EuroLLVM Dublin, the audience will be 90% people who can read LLVM source. They will find these gaps in real-time.

**The path forward:** spend the next 2 weeks implementing, not writing. Get `RuntimeSelectAttr.cpp` to emit multi-blob LLVM IR for one test case. Rewrite `bench_dispatch` to time `kdl_select_kernel()` in isolation. Produce actual numbers. Then the poster has substance behind the narrative.

---

*Devil's advocate review completed: 2026-04-09*
*Mode: ADVERSARIAL (escalated after F1+F2+F3 — three critical findings in the first three items investigated)*
