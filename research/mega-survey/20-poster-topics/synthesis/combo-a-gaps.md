# Combo A Gap Analysis: Factual Verification and Risk Assessment

**Topics audited:** Topic-01 (gpu.select_variant), Topic-07 (OffloadBinary Metadata), Topic-19 (Dispatch Flamegraph)
**Auditor role:** technical researcher / factual checker
**Date:** 2026-04-08
**Method:** cross-referenced all claims against wave files in this repo, web searches
against live LLVM GitHub/docs/Discourse, and source-verified kdl.c.

---

## TOPIC 01: Runtime Variant Selection Op for MLIR GPU Dialect

### Critical Error: PR Number Conflation

**The most serious factual error in the entire Combo A set.**

Topic-01 cites:

> PR #119440: ELF section option for gpu-module-to-binary (December 2024)

and also cites:

> Intel XeVM upstreamed August 2025 — phoronix.com/news/Intel-XeVM-MLIR-In-LLVM — PR #119440

**Both citations point to PR #119440. They refer to different things.**

PR #119440 is `[mlir][gpu] Adding ELF section option to the gpu-module-to-binary pass` by
Renaud-K. Confirmed merged December 16, 2024. Source:
https://github.com/llvm/llvm-project/pull/119440 and commit
`9919295cfd05222159246d7448ec42392e98fbf2`.

The Intel XeVM upstreaming is a completely separate PR. The MLIR commit note reference is
`b9b2661`, and the integration test PR is #148286 (`[Mlir-commits] [mlir] Add XeVM target
and XeVM dialect integration tests`), merged around August 2025. The first XeVM dialect
RFC is discourse.llvm.org/t/mlir-rfc-dialect-xevm-proposal-for-new-xevm-dialect/86955
(June 2025). Phoronix reported the upstreaming on August 19, 2025.

**The correct attribution is:** XeVM was upstreamed in ~August 2025 via a PR in the
#140000-#155000 range (not #119440). The actual PR number is unknown from public web
search and should be verified directly in the LLVM monorepo before the poster is
finalized. Do not cite #119440 as the XeVM upstreaming PR.

**Risk:** A reviewer familiar with the LLVM repo will immediately know PR #119440 is the
ELF section pass, not XeVM. This destroys credibility on the core claim that tri-vendor
GPU targets "urgently need" runtime dispatch.

**Fix:** Replace `PR #119440` in the XeVM citation with the correct PR number. Keep
PR #119440 only for the ELF section claim (where it is correct).

---

### Verified: SelectObjectAttr.cpp Path and Description

`mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp` — confirmed correct path via
MLIR doxygen at https://mlir.llvm.org/doxygen/SelectObjectAttr_8cpp_source.html. The
description that `embedBinary` creates a single `@serializedObj` global at translation
time is confirmed accurate. The `OffloadingLLVMTranslationAttrInterface` extension point
(Source 4, `CompilationAttrInterfaces.td`) is the correct mechanism for a new
`#gpu.runtime_select` attribute.

### Verified: ModuleToBinary.cpp Path

`mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp` — confirmed via MLIR doxygen at
https://mlir.llvm.org/doxygen/ModuleToBinary_8cpp_source.html.

### Unverified: RFC #88170 Resolution Status

The "Cleaning the GPU Dialect" RFC (discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170)
was posted September 4, 2025. Web search as of April 2026 returns three pages of
discussion but no resolution announcement. The RFC is **still active / unresolved**.
Topic-01 treats this RFC as a stable anchor: "coordinate with Fabian Mora to land
`#gpu.runtime_select` as the dispatch-policy half of the cleanup." This is premature —
the cleanup's final shape is unknown. If the RFC concludes by removing `gpu.binary`
semantics or replacing it with something structurally different, the proposed
`#gpu.runtime_select` landing point may shift.

**Risk:** Medium. The RFC being unresolved is actually an *opportunity* — topic-01's
proposal can be framed as filling the explicit policy gap the RFC identifies. But the
poster should not claim the RFC "leaves a policy slot vacant" as if that framing is
final RFC consensus; it should say "the RFC discussion has separated container from
dispatch policy, and no implementation of the policy has been proposed."

### Verified: `OffloadingLLVMTranslationAttrInterface` as Extension Point

Confirmed from MLIR GPU dialect docs and `SelectObjectAttr.cpp` doxygen. The two-method
interface (`embedBinary` + `launchKernel`) is the correct and sufficient extension point.
The existing `SelectObjectAttr` implementation is the correct template (~200 LOC confirmed
by doxygen source view).

### Unverified: `CompilationAttrInterfaces.td` Path

The proposal cites `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td`. This
path could not be confirmed via web search (the file is not in the doxygen index under
that exact name). Alternative candidate:
`mlir/include/mlir/Dialect/GPU/IR/GPUOps.td` or a file named
`CompilationAttributes.td`. Verify the exact filename against the LLVM monorepo before
citing it. This is a low-risk error (wrong filename does not affect the technical claim)
but will be caught in code review.

### Gaps Not Checked in Topic-01

- No analysis of what happens to `gpu.select_variant` if SPIR-V-as-portable-IR (RFC
  discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115) lands
  before the poster. That RFC would change the "three vendor binaries" premise to "one
  SPIR-V binary, JIT-compiled." Topic-01 briefly mentions this in Evidence §8 but does
  not address the scenario where SPIR-V makes runtime selection unnecessary. The rebuttal
  — SPIR-V cannot yet encode all vendor-specific performance extensions — is in wave-05
  §6 but is absent from topic-01 itself.

- The `dlopen`-based multi-vendor linking approach has a known LLVM community sensitivity
  (mentioned in feasibility). No Discourse thread or PR has been located where this
  pattern was explicitly debated for MLIR-internal code. The poster should pre-empt this
  by citing the JAX/PyTorch precedent *on the poster itself*, not just in internal notes.

### Implementation Risk Summary (Topic-01)

| Risk | Severity | Mitigation |
|------|----------|------------|
| XeVM PR number is wrong (#119440) | HIGH — reviewer credibility | Find and cite the correct PR |
| RFC #88170 unresolved, final shape unknown | MEDIUM | Reframe as "filling the gap RFC identifies" |
| `CompilationAttrInterfaces.td` filename unverified | LOW | Verify in monorepo before poster |
| SPIR-V RFC not addressed in the proposal | MEDIUM | Add one sentence acknowledging it |
| `dlopen` linking policy debate unaddressed | MEDIUM | Cite JAX/PyTorch precedent explicitly |

---

## TOPIC 07: Standard Capability Metadata Keys for OffloadBinary

### Verified: D122069 as OffloadBinary Origin

D122069 (`[Object] Add binary format for bundling offloading metadata`) confirmed as the
originating patch via https://reviews.llvm.org/D122069. The description — flexible string
map, `getTriple()` → `"triple"`, `getArch()` → `"arch"` — is confirmed accurate by the
live `OffloadBinary.h` at https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/OffloadBinary.h.

The LLVM llvm-offload-binary tool documentation at
https://llvm.org/docs/CommandGuide/llvm-offload-binary.html confirms: "arch" and "triple"
are the only documented standard keys. The claim that only two standard string keys are
defined is **confirmed correct**.

### Verified: PR #169425 Format Version 2

PR #169425 (`[Offloading] Extend OffloadBinary format to support multiple metadata
entries`) confirmed via https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg654252.html,
posted November 25, 2025. The PR adds a `ValueSize`-aware per-entry field and a
`StringEntryV1` struct for backward compatibility. The description "version 2 format bump,
adds EntriesCount and per-entry ValueSize-aware string entries" is **confirmed accurate**.

**One discrepancy:** Topic-07 describes PR #169425 as adding an `EntriesCount` field.
The mail-archive patch shows the format adds per-entry size tracking (`ValueSize` in
`StringEntryV1`) and backward compat via `StringEntryV1`. The exact field name
`EntriesCount` is not confirmed in the patch excerpt — verify this specific detail
against the merged code before the poster.

### Verified: D127686 `feature=` Key Not Standardized

D127686 (`[Offloading] Embed the target features in the OffloadBinary`) confirmed at
https://reviews.llvm.org/D127686. It prototyped a `feature=` key for LTO target-feature
propagation. The claim that it was "never standardised into a documented vocabulary" is
consistent with the fact that the current `llvm-offload-binary` docs and `OffloadBinary.h`
do not mention `feature=`. **Confirmed — no standard vocabulary beyond `triple`/`arch`.**

### Verified: KernelInfo Pass Emits but Does Not Write to OffloadBinary

`KernelInfo` (documented at https://llvm.org/docs/KernelInfo.html) and the AMDGPU
remarks pass D123878 (`[AMDGPU] Add remarks to output some resource usage`, April 2022)
both confirm that SGPR/VGPR/occupancy/LDS data is emitted as *remarks* only. The
`llvm-offload-binary` tool documentation contains no reference to these values being
embedded in the string table. **The pipeline gap claim is confirmed.**

### Verified: `areTargetsCompatible` AMD-Only Parsing

The `areTargetsCompatible()` function is confirmed in `OffloadBinary.cpp` (visible via
https://llvm.org/docs/doxygen/OffloadBinary_8cpp_source.html). It parses AMD `xnack+/-`
and `sramecc+/-` flags from the arch string. CUDA-side capability checking is not present
in that function. **Claim confirmed accurate.**

### Unverified: NVPTX Writer Path via `clang-offload-wrapper`

Topic-07 proposes writer integration for NVPTX via
`clang/tools/clang-offload-wrapper/ClangOffloadWrapper.cpp`. This file path has not been
confirmed in the current codebase — the offload wrapper may now live under
`clang/tools/clang-linker-wrapper/` or the `offload/` directory following the 2024
migration (wave-05 §3, new unified driver PR #84420). The CUDA backend writer path is
plausible but the specific file needs verification before it is cited in the poster as
the implementation location.

**Risk:** Medium. The new unified offloading driver (default since LLVM 19) changed the
compilation pipeline; the old `clang-offload-wrapper` may have been superseded. The
correct file is likely in `offload/tools/` or `clang/tools/clang-linker-wrapper/`.

### Missing Research: No Competing RFC Found — But Recent Activity Exists

Web searches confirm no competing RFC has been filed on OffloadBinary metadata vocabulary
as of April 2026. However, PR #185404 (`[Offload][L0] Add support for OffloadBinary
format in L0 plugin`, merged March 11, 2026) introduced `OffloadBinMetadataTy` — a struct
carrying `Triple`, `Arch`, `ImageKind`, `OffloadKind`, and `StringMap<string>` for
additional key-value metadata. This struct is the *runtime-side* consumer side of the
very metadata vocabulary Topic-07 proposes to define.

**This is a significant relationship that topic-07 does not explicitly cite.** The
`isMetadataCompatible` method introduced in PR #185663 (merged March 10, 2026) is the
runtime filter that would *consume* the standardized keys. The poster should position
the key vocabulary proposal as the *producer side* of this already-merged consumer
infrastructure. This strengthens the proposal considerably — the runtime hook already
exists; the standard vocabulary is the only missing piece.

### Missing Research: `llvm-offload-binary --annotate` Precedent

The proposal adds an `--annotate` flag to `llvm-offload-binary`. The current tool already
has `--dump` and `--info` functionality visible at
https://llvm.org/docs/CommandGuide/llvm-offload-binary.html. Verify that `--annotate` is
not a name collision with existing flags, and that the tool is in
`llvm/tools/llvm-offload-binary/` (this path is consistent with the docs).

### Implementation Risk Summary (Topic-07)

| Risk | Severity | Mitigation |
|------|----------|------------|
| NVPTX writer path (`ClangOffloadWrapper.cpp`) may be wrong file | MEDIUM | Find correct file post-LLVM-19 driver migration |
| PR #169425 `EntriesCount` field name unconfirmed | LOW | Verify in merged code |
| `isMetadataCompatible` / PR #185663 not cited — strongest supporting evidence | MEDIUM | Add explicit citation: this PR already provides the runtime consumer |
| PR #186088 (CUDA/AMDGPU generalization) not yet merged as of April 2026 | MEDIUM | Note as open PR; if merged, multi-image support is now default |

---

## TOPIC 19: End-to-End Dispatch Overhead Flame Graph

### Critical: TaxBreak Does NOT Measure Null-Kernel Dispatch Floor

**This is the most substantive technical gap in topic-19.**

Topic-19 asserts:

> "CUDA null-kernel floor (H100): 4.71 μs avg (p50: 4.578 μs, p95: 5.396 μs) — TaxBreak arXiv:2603.12465"

The TaxBreak paper (confirmed published March 2026, authors: Prabhu Vellaisamy et al.)
is a **trace-driven methodology for decomposing LLM inference overhead** into framework
time, CUDA library time, and kernel launch-path time. It was validated on H100/H200.

However, the paper's focus is decomposing *production LLM kernel* launch overhead across
the PyTorch/CUDA stack — it is not a null-kernel microbenchmark paper. The specific
figures "4.71 μs avg, p50: 4.578 μs, p95: 5.396 μs" could not be verified against the
paper's abstract or HTML via web search. These numbers may be:

1. From a different paper (e.g., the ICPP 2019 poster cited in wave-03, or NVIDIA's
   own blog post about CUDA Graph node overhead).
2. Correctly from TaxBreak but misattributed — TaxBreak may report these as the
   `cuLaunchKernel` component of its decomposition, not as null-kernel floor.
3. From the wave-03 research notes which may themselves have merged multiple sources.

**The "4.71 μs H100 floor" is cited as the entire foundation for the poster's novelty
claim** ("the community has the floor"). If this number is wrong or misattributed, the
poster's baseline collapses. This must be verified against the actual TaxBreak PDF before
the poster is finalized.

**Action required:** Read TaxBreak arXiv:2603.12465 Section 4 (or wherever measurements
appear) and confirm: (a) does the paper report null-kernel dispatch latency, or production
kernel latency? (b) are the p50/p95 figures present? (c) is H100 the hardware? If the
figures come from a different source, correct the attribution.

### Verified: `olGetKernel` → `olGetSymbol` Rename (PR #147943)

Topic-19's measurement code uses `olGetSymbol(prog, "null_kernel", OL_SYMBOL_KIND_KERNEL, &sym)`
which is consistent with the rename confirmed in wave-04 (PR #147943, merged July 10,
2025). **Correct API usage confirmed.** The measurement code would fail to compile
against pre-July-2025 liboffload.

### Verified: PR #186088 parseOffloadBinary Loop — No Timing Instrumentation

Wave-04 confirms PR #186088 (open March 2026) adds `parseOffloadBinary` to
`PluginInterface.cpp` with a first-compatible-wins loop and zero timing instrumentation.
This is the direct evidence for the claim "the community does not know how long
image-compatibility checking takes per variant." **Confirmed.**

### Critical: PR #186088 Open Status as of April 2026

Topic-19 refers to PR #186088 as "open March 2026." If this PR has merged by poster
date (April 7, 2026), the implementation details (especially the
`isMetadataCompatible`/`isDeviceCompatible` filtering path) are in mainline and the
measurement harness should target the merged code path. The poster should verify the
current status of this PR.

### Critical: kdl.c Lines 4595–4649 — Verified, With Important Caveat

Verified: `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` lines 4590–4649
contain `kdl_get_dispatch_latency_ns()`. The function exists at approximately those lines
(actual function header at line ~4605 based on the output at lines 4590+).

**However, the function does NOT time `cuLaunchKernel` directly.** It times
`cuStreamSynchronize` as a *proxy* for hot-path dispatch overhead, with 100 repetitions.
Topic-19's description states "kdl_get_dispatch_latency_ns() which times
cuStreamSynchronize as a proxy for hot-path dispatch overhead" — this is accurate. But
the measurement design on the poster (which proposes extending this to bracket each
`ol*` call directly) is a meaningful extension, not just "using the existing harness."
The poster language should be precise: the existing function provides a *foundation*;
the flame graph requires new per-layer bracketing code. This is already acknowledged in
the feasibility section but the pitch conflates the two.

### Verified: OMPT Device Hooks Issue #110007

Web search could not find Issue #110007 directly in search results (the adjacent
Issue #110008 is about Flang/OpenMP). However, wave-04 references Issue #110007 as
tracking OMPT device hook expansion, sourced from wave-07-llvm-devmtg-gpu-landscape.md.
The cited Dhruva Chakrabarti talk is confirmed from wave-05 §13 (GPU/Offloading Workshop
2024 agenda). **The issue number should be verified directly at
https://github.com/llvm/llvm-project/issues/110007 before citing on the poster.**

### Missing Research: liboffload API Stability — Version Pinning Not Planned

Topic-19 acknowledges API instability as a risk and says "pin to a specific LLVM commit."
What it does not say: liboffload has no published ABI stability guarantee or versioning
policy as of April 2026. The `olGetKernel` → `olGetSymbol` rename (3 months after the
initial API) demonstrates the rate of change. For a poster presented in April 2026 at
Dublin, the measurement code should be pinned to a tag (e.g., `llvmorg-21.x`) or a
specific dated commit, and this should be stated explicitly to pre-empt the question
"which version?"

### Missing Research: Layer 3 Measurement Architecture Is Incorrect

Topic-19's measurement design shows:

```c
/* Layer 3: implicitly inside olCreateProgram:
   cuModuleLoadData measured separately via direct CUDA driver call */
cuModuleLoadData(&cumod, blob_data);
```

This is architecturally confused: `cuModuleLoadData` is *inside* `olCreateProgram` (it
is what `olCreateProgram` calls internally for CUDA). Calling it separately with the
same blob would load a *second* instance of the module, not measure layer 3 of the
`olCreateProgram` path. To actually measure layer 3 independently, either:

(a) Instrument the liboffload CUDA plugin source (add `clock_gettime` inside
    `CUDA_PLUGIN::loadBinary`), or
(b) Use a separate measurement pass: call `cuModuleLoadData` directly (bypassing
    liboffload) to establish the driver-call baseline, then subtract from
    `olCreateProgram` total to infer plugin overhead.

The measurement code as written in the proposal conflates these two approaches and would
double-load the module. This must be fixed before implementation.

### Missing Research: GTX 1650 vs H100 Comparison Scope

Topic-19 acknowledges "relative layer fractions, not absolute values, generalize across
hardware." However, the GTX 1650 lacks Ampere+ features (no hardware MIG, no CUDA
Graph hardware acceleration) that affect the dispatch floor. The H100 "4.71 μs" figure
used as the anchor comes from a paper that specifically studied H100/H200 in LLM
inference scenarios. The GTX 1650 dispatch floor is unknown — topic-19 estimates
"8–12 μs" based on reasoning about older PCIE bandwidth, but no citation supports this.

**Risk:** A conference attendee with an A100 or H100 will ask "why is your floor 2x the
published H100 number?" The answer ("consumer hardware, older PCIE") is correct but needs
to be supported by at least one citation (e.g., the 2019 ICPP poster cited for "2–4 μs
on 2019 hardware" is the right anchor — cite that for the GTX 1650 baseline estimate).

### Implementation Risk Summary (Topic-19)

| Risk | Severity | Mitigation |
|------|----------|------------|
| "4.71 μs H100 floor" attribution to TaxBreak unverified | HIGH — core claim | Read TaxBreak PDF; confirm numbers or find correct source |
| Layer 3 measurement design is double-loading the module | HIGH — implementation | Fix: use separate cuModuleLoadData baseline pass, not concurrent |
| OMPT Issue #110007 not directly verified | LOW | Check github.com/llvm/llvm-project/issues/110007 |
| liboffload version not pinned | MEDIUM | Pin to a specific LLVM tag in the poster |
| GTX 1650 baseline estimate uncited | MEDIUM | Cite ICPP 2019 for pre-H100 hardware baseline |

---

## Cross-Cutting Issues Across All Three Topics

### Issue A: PR #186088 Status Is Load-Bearing for Topics 01, 07, and 19

All three topics depend on PR #186088 (`[OFFLOAD] Generalize support for OffloadBinary
images`) being open. If this PR merged in early April 2026 (submitted March 12), the
situation changes:

- Topic-01: the "first-compatible-wins" policy the poster proposes to replace is now
  the *default* in liboffload, making the gap more visible.
- Topic-07: the `isMetadataCompatible` consumer hook is now live for CUDA and AMDGPU,
  making the vocabulary proposal the obvious next step.
- Topic-19: the `parseOffloadBinary` loop being timed is in production code, so
  instrumentation has broader applicability.

**Action:** Verify current status of PR #186088 directly.

### Issue B: MI300X in Topic-07 Pitch Not Supported by Prototype

Topic-07's pitch says "a prototype already running on GTX 1650 + MI300X under libkdl."
However, the project instructions and `experiments/prototype/src/kdl.c` describe the prototype as
verified on "GTX 1650 + CPU." No MI300X hardware is mentioned anywhere else in the
research files. This claim appears to be aspirational. Remove MI300X from the pitch
unless AMD hardware testing has actually been done.

### Issue C: wave-08 Sources Not Cited in Any of the Three Topics

The wave-08 files (`wave-08-mlir-async-llvm-gpu.md`, `wave-08-kernel-info-pass.md`,
`wave-08-recent-gpu-dispatch-2025-2026.md`) are in the research base but not cited by
topic-01, 07, or 19. `wave-08-kernel-info-pass.md` is particularly relevant to
Topic-07's claim that `KernelInfo` data is not written to OffloadBinary. Verify that
wave-08 does not contain contradicting evidence before finalizing Topic-07.

### Issue D: GPU/Offloading Workshop 2025 Slides Not Directly Read

All three topics cite the GPU/Offloading Workshop 2025 slides
(discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832) and Joseph Huber's
technical talk slides (llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf). Neither
has been directly read in the research waves — they are cited secondhand from wave-05 §12.
These slides are directly relevant to all three topics and should be read before the
poster is submitted. In particular, the "Not-Compiler Runtime Library GPUs" slide deck
(633KB) may contain statements directly supporting or contradicting the proposals.

---

## Priority Action List

### Before poster draft is written:

1. **[CRITICAL] Correct the XeVM PR number in Topic-01.** PR #119440 is the ELF section
   pass, not XeVM. Find the correct XeVM initial upstreaming PR (likely in the
   #140000–#155000 range, possibly #148286 or related). Check
   https://github.com/llvm/llvm-project/pulls?q=xevm+merged:2025-08-01..2025-09-01.

2. **[CRITICAL] Verify TaxBreak "4.71 μs" figure source.** Read
   https://arxiv.org/html/2603.12465 Section 4 or equivalent. Confirm whether TaxBreak
   measures null-kernel dispatch or production kernel dispatch, and whether the p50/p95
   figures are present. If not from TaxBreak, find the correct citation.

3. **[HIGH] Fix Layer 3 measurement architecture in Topic-19.** The proposed code calls
   `cuModuleLoadData` separately from `olCreateProgram`, which double-loads. Redesign to
   either instrument liboffload source or use separate baseline measurement.

4. **[HIGH] Remove MI300X from Topic-07 pitch.** Prototype is verified on GTX 1650 +
   CPU only. Replace "GTX 1650 + MI300X" with "GTX 1650 + CPU-fallback."

### Before poster is submitted:

5. **[MEDIUM] Check PR #186088 merge status.** If merged, update all three topics to
   reflect that multi-image OffloadBinary support is now in mainline.

6. **[MEDIUM] Verify `CompilationAttrInterfaces.td` exact filename** in the LLVM
   monorepo. Likely `CompilationAttributes.td` or embedded in `GPUOps.td`.

7. **[MEDIUM] Verify NVPTX writer path** in Topic-07. The old
   `clang-offload-wrapper/ClangOffloadWrapper.cpp` may be superseded by the new unified
   driver. Check the `offload/` tree.

8. **[MEDIUM] Add `isMetadataCompatible` (PR #185663) as an explicit citation in
   Topic-07.** This PR's runtime consumer hook is the strongest evidence for the
   vocabulary proposal's timeliness.

9. **[MEDIUM] Verify OMPT Issue #110007** at the GitHub URL. If the issue has been
   closed or merged, update Topic-19 accordingly.

10. **[LOW] Read wave-08 files** for potential contradictions to Topic-07's KernelInfo
    claims.

11. **[LOW] Directly read the GPU/Offloading Workshop 2025 slides** (Huber PDF) before
    finalizing all three topics.

---

## Confidence Ratings (post-audit)

| Topic | Core Technical Gap | Evidence Quality | Implementation Feasibility |
|-------|--------------------|-----------------|---------------------------|
| Topic-01 | Confirmed real (9/10) | 7/10 — XeVM PR error degrades this | Medium — dlopen policy risk real |
| Topic-07 | Confirmed real (9/10) | 8/10 — D122069/PR#169425 solid | Medium-Low risk — additive keys |
| Topic-19 | Confirmed real (8/10) | 6/10 — TaxBreak 4.71 μs unverified | High — but Layer 3 design broken |

All three proposals address genuine, confirmed gaps. The errors are fixable before
submission. Topic-07 has the cleanest evidence base once the MI300X claim is removed and
the NVPTX writer path is corrected. Topic-01 is strongest conceptually but carries the
most reputational risk from the PR number error.

---

*Audit completed: 2026-04-08. Researcher: technical researcher agent.*
*Sources consulted: wave files in this repo, MLIR doxygen, llvm.org docs, GitHub PRs
#119440/#147943/#148286/#169425/#185404/#185663/#186088, reviews.llvm.org D122069/D123878/D127686,
arxiv.org/abs/2603.12465, discourse.llvm.org RFC #88170, phoronix.com/news/Intel-XeVM-MLIR-In-LLVM.*
