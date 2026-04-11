# Hostile Competitor Review — Round 5

**Reviewer persona:** Senior engineer who works on IREE / chipStar / Proteus and feels threatened by this work. Looking for any reason to dismiss it at the poster session.

**Materials reviewed:** poster-combo-a.html, qa-cards-final.md, elevator-pitch.md, runtime_select_poc.c, offloadbinary_parse.c, kdl.c, pinned-benchmark-results.md, layer-benchmark-results.md, real-offloadbinary-results.md, rfc-FINAL.md, context.md

**Date:** 2026-04-10

---

## Attack 1: "IREE already does this, and better — you're reinventing our HAL with worse abstractions"

**The attack (IREE engineer):**

"We've had multi-target dispatch in IREE's HAL since 2019. Our `hal.device.query` plus executable format handles device/executable binding, multi-device scheduling, buffer management, and execution ordering — a complete dispatch stack. Your 'contribution' is a 5-key string table and a design sketch for an MLIR attribute that doesn't exist yet. You claim IREE issue #50 shows ranked selection is 'unimplemented' — but that issue is about *device selection policy*, not binary selection. IREE's `hal.executable.export` already resolves the correct backend binary per device. You're conflating device selection with kernel variant selection to make your gap look bigger than it is."

**Best defense:**

The distinction is layer, not functionality. IREE's HAL is a full-stack runtime that owns the execution model: buffers, scheduling, device management. It works — for IREE users. But torch-mlir users going through `gpu.binary` don't get IREE's HAL. ONNX-RT multi-EP users don't. Anyone compiling through MLIR's GPU dialect to `#gpu.select_object` gets compile-time resolution or nothing. The contribution is at the LLVM layer: below any framework, above the driver. IREE and this work operate at different abstraction levels and compose rather than compete.

On issue #50: you are correct that it is about device selection policy, not kernel binary selection. The poster's claim of "6 years open" should be narrowed. The stronger cite is PR #186088 in liboffload, which explicitly defers ranked image selection — that is directly analogous. Do not over-claim on IREE issues.

**Danger level:** 9/10. If the IREE person pins you on the #50 conflation, your credibility drops for the rest of the conversation.

---

## Attack 2: "Your PoC doesn't use LLVM at all — it's a standalone C library pretending to be LLVM infrastructure"

**The attack (any LLVM committer):**

"I looked at your code. `runtime_select_poc.c` is 664 lines of C that calls `dlopen` and `cuModuleLoadData`. `kdl.c` is 5,100 lines of C that calls `dlopen` and `cuModuleLoadData`. Neither file includes a single LLVM header. Neither links against a single LLVM library. You wrote a userspace CUDA wrapper and slapped 'LLVM OffloadBinary' on the struct names. Where is the MLIR pass? Where is the `OffloadingLLVMTranslationAttrInterface` implementation? You have a 'sketch' file that Python reports as ASCII text, not actual C++. This is a C library with an LLVM poster around it."

**Best defense:**

This is a fair hit and the strongest version of the "vaporware" critique. The honest answer: the poster presents two concrete contributions (metadata vocabulary, flame graph) and one design sketch. The prototype validates the *runtime mechanics* — dispatch table construction, vendor detection, variant selection, kernel launch — that `#gpu.runtime_select`'s `embedBinary()` would emit as LLVM IR. The C prototype is the C equivalent of what the LLVM IR would lower to. No MLIR C++ exists yet. The design sketch (`RuntimeSelectAttr.cpp.sketch`) is a detailed pseudocode roadmap, not a compilable implementation.

The poster explicitly presents C3 as a "Design Sketch" — not a landed patch. The metadata RFC (C1) and the flame graph (C2) are the concrete contributions. Lean into those two and frame C3 as future work. If pressed, say: "The poster session is where we start the upstream conversation, not where we present a merged patch."

**Danger level:** 8/10. Accurate critique. The defense is honest framing, not rebuttal.

---

## Attack 3: "Your OffloadBinary format is wrong — you're missing `ImageKind`, `OffloadKind`, and `Flags` fields"

**The attack (liboffload maintainer or anyone who has read OffloadBinary.h):**

"You claim your PoC uses 'LLVM OffloadBinary format (magic `0x10FF10AD`)'. I pulled up `llvm/include/llvm/Object/OffloadBinary.h`. The real format has per-entry fields including `ImageKind` (enum: `IMG_None`, `IMG_Object`, `IMG_Bitcode`, `IMG_Cubin`), `OffloadKind` (enum: `OFK_None`, `OFK_OpenMP`, `OFK_CUDA`, `OFK_HIP`), and `Flags`. Your `ObEntryHeader` struct has five uint64_t fields: `the_size`, `image_offset`, `image_size`, `string_offset`, `string_size`. You're missing at least three fields. Your PoC files are NOT interoperable with `clang-offload-packager` or `clang-linker-wrapper`. You hid this — your `real-offloadbinary-results.md` has a disclaimer, but your poster says 'LLVM OffloadBinary format' with no qualification. That's misleading."

**Best defense:**

This is accurate. The PoC implements a simplified subset: correct magic, correct overall container structure (file header + variable-length entries with string tables), but the per-entry metadata encoding differs from the actual `OffloadBinary.h`. The poster should include a footnote: "PoC implements the OffloadBinary container structure (magic, header, string table) as a round-trip demonstration; per-entry fields are a simplified subset not interoperable with LLVM tooling."

The key defense: the selection mechanism (dispatch table scan, vendor detection, ranking) is independent of the exact binary format fields. Adding `ImageKind`, `OffloadKind`, and `Flags` to the struct is a straightforward engineering task that doesn't change the 6 ns selection overhead or the flame graph results. The format difference affects interoperability, not the dispatch mechanism being demonstrated.

But do not try to claim the format is correct. It isn't. Acknowledge it immediately and pivot to what the PoC *does* demonstrate.

**Danger level:** 7/10. Recoverable if you acknowledge instantly. Fatal if you try to defend the format as correct.

---

## Attack 4: "You tested on ONE consumer GPU from 2019 — a GTX 1650 on PCIe 3.0. This tells me nothing about production deployment"

**The attack (ML infra engineer or cloud GPU vendor):**

"Your entire measurement suite runs on a GTX 1650 — a $150 consumer card from 2019 with 896 CUDA cores, 4 GB VRAM, and PCIe 3.0. Production ML inference runs on H100 NVLink, A100 SXM, MI300X. Your `cuStreamSynchronize` measures 2.5 us — that's PCIe 3.0 round-trip latency. On NVLink that would be sub-microsecond. Your `cuModuleLoadData` warm at 10 us — on H100 with HBM3 that could be 3x faster. You can't claim dispatch overhead ratios generalize across hardware when the absolute numbers are dominated by your antique interconnect. And you have zero AMD data — not even on a Radeon. For a 'heterogeneous dispatch' poster, you tested on exactly one vendor on exactly one architecture."

**Best defense:**

The GTX 1650 is the right hardware for the specific claim being made: dispatch overhead is a software stack latency measurement, not a compute throughput benchmark. The claim is about *relative layer fractions* — "variant selection contributes 6 ns to a 4.26 us hot-path" — not about absolute microsecond values. TaxBreak measured H100 null-kernel dispatch at 4.71 us via the same CUDA driver API — the same order of magnitude as our 4.26 us, confirming the driver-level baseline is stable across generations.

On AMD: the AMD code path exists in `kdl.c` (lines 740-768, `dlopen("libamdhip64.so")`) and is architecturally symmetric. It has been validated via mocked HIP entry points but not on physical ROCm hardware. Say: "Physical ROCm validation is the stated next step. The CUDA path demonstrates the mechanism end-to-end; the AMD path is structurally identical."

On the interconnect point: `cuStreamSynchronize` on a null kernel is indeed dominated by PCIe latency. On NVLink/NVSwitch it would be lower. This strengthens the "selection is negligible" argument — if the total hot-path shrinks, 6 ns becomes an even smaller fraction. Do not claim the absolute numbers transfer.

**Danger level:** 7/10. The AMD gap is real and the single-GPU limitation is real. The defense is strong on the ratio argument but cannot paper over the absence of multi-vendor data for a multi-vendor dispatch poster.

---

## Attack 5: "Nobody actually ships multi-vendor fat binaries through MLIR — you're solving a problem that doesn't exist"

**The attack (skeptical industry engineer):**

"Name one production system that compiles through MLIR's `gpu-module-to-binary`, produces a fat binary with NVIDIA and AMD objects, and dispatches at runtime. Not 'could' — *does*. IREE compiles its own flatbuffers. PyTorch uses TorchInductor with vendor-specific backends. TensorFlow uses XLA with per-target compilation. JAX uses PJRT with per-device plugins. Nobody is producing `gpu.binary` containers with multiple vendor objects and needing runtime selection. Your three 'users' — CERN HEP-CCE, vLLM, cloud containers — none of them use MLIR's GPU dialect today. CERN uses Alpaka. vLLM uses CUDA/HIP directly. Cloud containers use vendor SDKs. You're proposing infrastructure for a workflow that doesn't exist."

**Best defense:**

This is the strongest "so what?" attack. The honest answer: today, nobody ships multi-vendor fat binaries through MLIR because the runtime selection mechanism doesn't exist — that's the gap this poster identifies. The workflow doesn't exist because the last mile is missing.

The stronger version: the compile-time half *does* work today. `gpu-module-to-binary` can produce a `gpu.binary` with NVVM + ROCDL + XeVM objects. `#gpu.select_object` can pick one at compile time. The gap is specifically the runtime selection — which is why PR #186088, PR #185663, and RFC #88170 all exist in flight: the upstream developers are building toward this workflow. The poster is not inventing a problem; it's filling the last gap in a pipeline that upstream is actively constructing.

For CERN/vLLM: these are motivating use cases showing where the pain is, not claims of current MLIR adoption. Frame as: "These are the deployments that would benefit from this capability once it exists in LLVM."

**Danger level:** 8/10. The "build it and they will come" defense is valid for infrastructure proposals but weak for a poster. The strongest counter-evidence is the three upstream PRs/RFCs that show the LLVM community is actively building toward this capability.

---

## Attack 6: "Your 'metadata vocabulary' is just 5 string keys — that's a documentation patch, not a research contribution"

**The attack (academic reviewer or senior LLVM developer):**

"Let me make sure I understand your 'Contribution 1.' You're proposing 5 string keys: `min_sm`, `min_gfx`, `requires_features`, `variant_priority`, `variant_tag`. The header patch is 30 lines. The `isMetadataCompatible()` extension is 40 lines. This is a documentation RFC and a trivial code patch. It's useful infrastructure work — I'm not saying it shouldn't be done — but calling it a 'contribution' at a poster session is generous. Where is the science? Where is the design space exploration? Why these 5 keys and not 10? Why not `max_sm` for forward compatibility? Why not `optimal_occupancy`? You picked 5 obvious strings and called it a vocabulary."

**Best defense:**

The contribution is the vocabulary *design*, not the line count. The reason only 2 keys have been standardized in 4 years — despite the StringMap existing since D122069 — is not technical difficulty but naming-and-semantics agreement. D127686 proposed a `feature=` key and stalled because there was no vocabulary specification. The contribution is: (1) tiered semantics (MUST-check vs. MAY-use), (2) backward compatibility rules (missing key = no constraint, old runtimes ignore unknown keys), (3) vendor-specific token naming (`tensor_core_nv` not `tensor_core`) to avoid premature cross-vendor equivalence claims, and (4) composition with the existing `isMetadataCompatible()` hook.

On "why not more keys": Tier 2 (resource-usage keys like `sgpr_count`, `vgpr_count`) is explicitly deferred. Starting with 5 keys is deliberate scoping — the RFC process requires consensus, and a 5-key proposal has better odds of merging than a 15-key proposal. The `variant_priority` key alone enables ranked selection, which is the gap PR #186088 explicitly defers.

On "where is the science": this is an infrastructure contribution for a developers' meeting poster, not a PLDI paper. The science is in the flame graph (C2), which is the first published per-layer latency decomposition of the LLVM GPU dispatch stack.

**Danger level:** 6/10. A defensible critique. The answer is about design discipline, not line count.

---

## Attack 7: "Your flame graph numbers are inconsistent across your own documents"

**The attack (careful reader who cross-references):**

"Your poster stat box says '4.0 us hot-path total.' Your Q&A cards say '4.26 us.' Your elevator pitch says 'cold module load: 54.6 microseconds' but your poster says '42.7 us.' Your pinned benchmark data shows cold-path medians of 35.8, 36.2, 35.9 us across three runs — not 42.7 us. The 42.7 us comes from the *unpinned* single run. Which number is correct? If you can't keep your own measurements consistent across documents, why should I trust any of them?"

**Best defense:**

This is a valid consistency critique. The numbers come from different benchmark runs:

- **4.0 us vs. 4.26 us:** The poster stat box rounds 4.26 us down to 4.0 us for visual presentation. The Q&A cards use the precise value (4,257 ns from bench_layers). Both come from the same run. The poster should use 4.3 us, not 4.0.
- **42.7 us vs. 54.6 us vs. 36.0 us cold:** 42.7 us is the median from the unpinned bench_layers run. 54.6 us is the mean from the same run (inflated by p99 outliers). 36.0 us is the cross-run median from pinned runs. The poster uses the unpinned median. The pinned number is more reliable. The elevator pitch incorrectly uses the mean.
- **Correct approach:** Use the pinned cross-run medians as the canonical numbers. The poster should say "36 us cold-path median" and "4.25 us hot-path median" from pinned, CPU-affinity-controlled runs.

Acknowledge the inconsistency immediately. Say: "The poster uses the unpinned first-run median. Pinned benchmarks show 36 us cold and 4.25 us hot-path. The qualitative conclusion — selection overhead is negligible relative to module load and launch — holds across all runs."

**Danger level:** 6/10. Embarrassing but not fatal. The relative fractions are consistent even if the absolute numbers vary. Fix the poster before the session.

---

## Attack 8: "Your `select_best_entry` function is a linear scan over 3 entries — of course it's 6 ns. That's not a dispatch mechanism, it's a toy loop"

**The attack (performance engineer):**

"Let me read your selection function: it's a `for` loop over `g_num_entries` (3 entries), comparing two integers per entry. Three iterations of an integer comparison loop is going to be single-digit nanoseconds on any modern CPU — that's a trivially obvious result. You measured the time to execute 6 integer comparisons and called it 'selection overhead.' The real question is: what happens at 100 entries? 1,000? With string-matching on `requires_features`? With the `device.hasCapability()` call that your RFC admits 'does not yet exist in liboffload'? Your 6 ns number is meaningless because it measures a toy configuration."

**Best defense:**

The 6 ns number for 3 entries is the correct measurement for the real-world configuration: a fat binary carries 2-4 variants (e.g., sm_75 + sm_86 + sm_90 + SPIR-V fallback), not 1,000. Nobody ships a fat binary with 100 GPU variants. The dispatch table is architecturally bounded by the number of compiled targets, which in MLIR's `gpu.binary` is typically 3-5.

On scaling: the selection is O(N) in the number of entries. At 30 entries (hypothetical extreme), it would be ~40 ns — still noise relative to the 4.26 us hot-path. The poster already makes this point ("scales to ~40 ns for 30 variants").

On `requires_features` string matching: the PoC does not implement feature token matching because `device.hasCapability()` doesn't exist yet. This is an honest limitation. The `isMetadataCompatible()` extension (C1) would add string comparison per feature token — still sub-microsecond for realistic token counts (1-3 features per variant). The ranking function (variant_priority integer comparison) stays at 6 ns.

**Danger level:** 5/10. The "toy loop" framing is emotionally effective but technically wrong — 3 entries is the realistic case, not a toy one.

---

## Attack 9: "You claim 'zero hot-path overhead after one-time selection' but your prototype re-runs selection on every microbenchmark iteration"

**The attack (careful code reader):**

"Your poster says: 'After one-time selection, the module handle is cached in a global pointer. Subsequent launchKernel calls are identical to compile-time #gpu.select_object.' But your microbenchmark in `runtime_select_poc.c` lines 643-652 calls `select_best_entry()` 100,000 times in a loop. That's measuring the selection function hot in L1 cache. In the real `#gpu.runtime_select` design, selection runs once in `global_ctors` and then the hot path is a global pointer load. Your 6 ns number measures repeated selection, not the one-time cost. The one-time cost (single-shot) is 380-521 ns from your own elevator pitch data. Which is the real number?"

**Best defense:**

Both numbers are real, and both are reported. The 380-521 ns single-shot number includes instruction cache misses and TLB misses on first access. The 4-6 ns amortized number measures the selection function in steady state (which is what matters if a different code path calls it again, e.g., for a different kernel). In the `#gpu.runtime_select` design, selection runs once per `gpu.binary` module in `global_ctors`. The cost is 380-521 ns per module, one time. After that, it's a global pointer load — truly zero overhead.

The Q&A cards report both numbers. The poster headline uses 6 ns because it's the per-dispatch cost if selection were repeated (conservative upper bound). The one-time 380-521 ns cost is more relevant for the `global_ctors` design but harder to present on a poster. Acknowledge the distinction. Say: "6 ns is the selection function cost if called repeatedly. 380 ns is the cold single-shot cost including cache misses. Both are negligible relative to the 42.7 us module load that follows."

**Danger level:** 5/10. Technically accurate critique about what the microbenchmark actually measures. The defense is solid if you clearly distinguish the two numbers.

---

## Attack 10: "The D127686 precedent should scare you — the LLVM community has already rejected metadata vocabulary standardization once. What makes you think this time is different?"

**The attack (long-time LLVM contributor):**

"D127686 proposed a `feature=` key for LTO target-feature propagation. It was never standardized. It didn't fail for lack of technical merit — it failed because nobody could agree on semantics. Your `requires_features` key has the exact same problem: who defines the token vocabulary? You say 'tensor_core_nv' — but NVIDIA doesn't use that string anywhere. Where does 'bf16' come from? Is it the CUDA device attribute `CUDA_DEVICE_ATTR_COMPUTE_MODE` or something else? You're going to hit the same naming bikeshed that killed D127686, and your poster will be a historical footnote just like that revision."

**Best defense:**

This is the strongest upstream-viability attack. The answer: D127686 failed because it proposed code before semantics — a single `feature=` key with no specified value vocabulary, no tiering, and no RFC. This proposal explicitly inverts that: RFC first, code second. The 5-key vocabulary is deliberately minimal and the token names are taken from existing vendor terminology (`tensor_core` from NVIDIA's own documentation, `mfma` from AMD's ISA manual, `bf16` from IEEE 754-2008 extension widely used across both vendors).

The naming bikeshed risk is real. The mitigation strategy is: (1) post the RFC on Discourse *before* any code, (2) start with Tier 1 keys only (3 keys, not 5), (3) use the documentation-only patch as step zero — naming the existing 2 keys and reserving the namespace costs zero risk and builds awareness. The poster session is the start of the conversation with the people (Huber, Mora, Denny) who would review the RFC.

Concede that the bikeshedding risk is the primary obstacle. Do not claim it will be easy. Say: "D127686 is exactly why the RFC precedes any code. We learned from that failure."

**Danger level:** 8/10. This is the attack that can't be fully rebutted because it's predicting a social/process failure, not a technical one. The best you can do is show you've learned from the precedent and have a concrete engagement plan.

---

## Summary: Ranked by Danger

| # | Attack | Danger | Key Vulnerability |
|---|--------|--------|-------------------|
| 1 | IREE already does this better | 9/10 | Conflation of IREE issue #50 with kernel binary selection |
| 5 | Nobody ships multi-vendor fat binaries | 8/10 | No production user of the proposed workflow exists today |
| 10 | D127686 precedent: metadata standardization already failed | 8/10 | Social/process risk cannot be technically rebutted |
| 2 | PoC doesn't use LLVM at all | 8/10 | No MLIR C++ exists; "sketch" file is pseudocode |
| 3 | OffloadBinary format is wrong | 7/10 | Missing fields, not interoperable, poster has no disclaimer |
| 4 | One GPU, one vendor, 2019 hardware | 7/10 | Zero AMD data for a heterogeneous dispatch poster |
| 7 | Numbers inconsistent across documents | 6/10 | 4.0 vs 4.26 us; 42.7 vs 36.0 vs 54.6 us cold |
| 6 | 5 string keys is not a contribution | 6/10 | Low line count, design discipline argument is subjective |
| 8 | Selection is a trivial linear scan | 5/10 | 3 entries is realistic, but "toy loop" framing is effective |
| 9 | Microbenchmark measures wrong thing | 5/10 | 6 ns vs 380 ns ambiguity, but both are negligible |

## Pre-Session Action Items

1. **Fix the poster stat box:** Change "4.0 us" to "4.3 us" hot-path total. The rounding is indefensible when the Q&A cards say 4.26.
2. **Add OffloadBinary disclaimer to poster:** One line in the Prototype Validation card: "PoC implements OffloadBinary container structure; per-entry fields are a simplified subset."
3. **Reconcile cold-path number:** Pick 42.7 us (unpinned median) or 36.0 us (pinned cross-run median) and use ONE number everywhere. The pinned number is more rigorous.
4. **Soften IREE issue #50 claim:** The poster says "Partial (#50 open 6yr)" in the related work table. Change to "#50, #12230" or remove the duration claim. Be precise about what #50 is about.
5. **Rehearse the "no MLIR C++" answer:** This will come up. Have the "two concrete contributions, one design sketch" framing ready without hesitation.
6. **Do NOT mention IREE first.** Let them raise it. Your prepared answer is better as a response than as a preemptive defense.
