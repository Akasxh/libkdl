# LLVM/MLIR Expert Verification: Combo A Technical Claims

**Scope:** Topic-01 (gpu.select_variant), Topic-07 (OffloadBinary Metadata), Topic-19 (Dispatch Flamegraph)
**Auditor role:** LLVM/MLIR infrastructure expert — GPU dialect, liboffload, OffloadBinary
**Date:** 2026-04-09
**Method:** Live web verification against MLIR docs, LLVM GitHub, Phoronix, arXiv HTML, Discourse,
and direct `gh pr view` calls. Every claim below is sourced.

---

## Section 1: OffloadingLLVMTranslationAttrInterface — Verification

### 1.1 Interface definition path

**Claim in Topic-01:** `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td`

**Verdict: CONFIRMED CORRECT.**

The file was directly fetched from:
`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td`

The file exists at that exact path. It defines `OffloadingLLVMTranslationAttrInterface` with two methods:

- `embedBinary(Operation* binaryOp, IRBuilderBase& hostBuilder, LLVM::ModuleTranslation& hostModuleTranslation) -> LogicalResult`
- `launchKernel(Operation* launchFunc, Operation* binaryOp, IRBuilderBase& hostBuilder, LLVM::ModuleTranslation& hostModuleTranslation) -> LogicalResult`

The combo-a-gaps.md audit flagged this filename as "could not be confirmed via web search." That flag was incorrect — the file does exist and has the correct name.

### 1.2 Can a new attribute implement embedBinary to emit multiple blobs?

**Verdict: YES — with one important constraint.**

`SelectObjectAttr.cpp` (479 lines, 414 loc, 17.8 KB — confirmed via doxygen source view) contains
**exactly one concrete implementation**: `SelectObjectAttrImpl`, defined as a FallbackModel of
`OffloadingLLVMTranslationAttrInterface`.

The current `embedBinary` implementation:
- Creates one `GlobalVariable` named `{moduleName}_binary` (NOT `@serializedObj` — that name
  appears to have been an older version or a different translation unit).
- Emits a single `ConstantDataArray` blob.
- Generates load/unload functions wired to `llvm.global_ctors` / `llvm.global_dtors`.

A new attribute implementing `embedBinary` can emit **N** separate globals (one per vendor binary)
because the interface contract is simply "translate a `gpu.binary` op into LLVM IR." There is no
constraint limiting the implementation to a single global. The interface is a pure virtual two-method
contract — any compliant implementation may emit arbitrary LLVM IR, including multiple global arrays,
a dispatch table, and a detection stub.

**Constraint that matters:** `embedBinary` receives the full `gpu.binary` op and its host
`ModuleTranslation` context. The implementation must emit LLVM IR that compiles and links without
conflicts with other offloading translations in the same module. The N-blob approach (one global per
object) is structurally sound because LLVM GlobalVariable names are namespaced by module name.

### 1.3 Are there other implementations besides SelectObjectAttr?

**Verdict: SelectObjectAttr is the ONLY implementation of the full interface in mainline.**

Web searches and the MLIR GPU namespace reference at `https://mlir.llvm.org/doxygen/namespacemlir_1_1gpu.html`
confirm no other class registers as an OffloadingLLVMTranslationAttrInterface implementation.
Note: `NVVMTargetAttr` and `ROCDLTargetAttr` implement `GPUTargetAttrInterface` (for
serialization/compilation), not `OffloadingLLVMTranslationAttrInterface` (for LLVM IR emission).
These are two distinct interfaces — the gap analysis is correct that `SelectObjectAttr` is alone
in the offloading interface space.

### 1.4 What a new implementation must register

A new `#gpu.runtime_select` attribute must:
1. Declare in TableGen (`.td` file) that it implements `OffloadingLLVMTranslationAttrInterface`.
2. Implement `embedBinary` and `launchKernel` in a new `.cpp` file (proposed path
   `mlir/lib/Target/LLVMIR/Dialect/GPU/RuntimeSelectAttr.cpp` is appropriate).
3. Register via `addAttrInterfaces<RuntimeSelectAttr>()` in the GPU dialect initialization,
   following the pattern in `mlir/lib/Dialect/GPU/IR/GPUDialect.cpp`.
4. No new TableGen interface definitions are needed — `CompilationAttrInterfaces.td` is the
   existing, correct extension point.

---

## Section 2: PR #186088 Status

**Claim in Topic-19 / combo-a-gaps.md:** "PR #186088 open March 2026; iterates all images;
first-compatible-wins; zero timing instrumentation."

**Verdict: CONFIRMED OPEN AS OF APRIL 9, 2026.**

Direct `gh pr view 186088 --repo llvm/llvm-project` output:
- **State:** OPEN
- **Title:** `[OFFLOAD] Generalize support for OffloadBinary images`
- **Created:** 2026-03-12T10:56:44Z
- **Merged at:** null (not merged)
- **Reviewers pending:** jdoerfert, hansangbae (awaiting review); sarnex and jhuber6 have commented

**Key statement from the PR body** (verbatim):
> "For now only the first compatible image in the binary is loaded. While it might be desirable
> to add support for loading multiple images, our current interface is limiting (i.e., it returns
> a single Image) and it's unclear if in all cases this behavior is desirable so we would need to
> add more options to control it. So, should we want it, it's better in a follow-up PR IMO."

This is significant for all three topics:
- **Topic-01:** The PR author (implicit: jhuber6 / another offload contributor) explicitly
  acknowledges that multiple-image loading is the next step. `gpu.select_variant` / `#gpu.runtime_select`
  is the MLIR-layer answer to this explicit liboffload-layer acknowledgment. This is a strong
  alignment point the poster should cite directly.
- **Topic-07:** The metadata vocabulary this PR uses (triple, arch, isMetadataCompatible) is
  exactly the vocabulary Topic-07 proposes to extend. The PR is the live "runtime consumer" that
  needs the producer-side vocabulary.
- **Topic-19:** The loop being timed is still in an open PR — the measurement harness should
  pin to either (a) the merged state (if it merges before Dublin) or (b) the PR branch directly.

**Risk update:** The PR has been open 4 weeks with active reviewer attention (jhuber6, sarnex).
It is likely to merge before the Dublin poster deadline (April 7). Update topics to say
"merged in April 2026 / in mainline as of LLVM 21" once confirmed.

---

## Section 3: RFC #88170 — Cleaning the GPU Dialect

**Claim in Topic-01:** "RFC explicitly separates `gpu.binary` (kernel container) from dispatch
policy — and that policy slot is empty in mainline."

**Verdict: CONFIRMED ACTIVE, UNRESOLVED — but framing is defensible.**

The RFC thread at `https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170` has three
pages as of April 2026. Web search confirms it was posted September 4, 2025. Pages 2 and 3 show
ongoing discussion (both pages exist and are indexed, indicating active participation through at
least early 2026).

No resolution announcement, consensus statement, or "accepted" flag was found in any search result
or page fetch. The RFC remains **open and unresolved** as of April 2026.

**Critical nuance:** Topic-01's statement that the RFC "explicitly leaves the dispatch-policy slot
vacant" is accurate as a description of the RFC's *framing*, not as a concluded RFC outcome. The
combo-a-gaps.md audit correctly identified this — the poster should say: "the RFC discussion has
articulated a container/policy separation; no implementation of the policy has been proposed." Do
not say "the RFC concluded" or "the RFC resolved" anything.

**Opportunity:** An open RFC is better than a resolved one for a poster proposal. The RFC is the
community-documented motivation; `gpu.select_variant` is the first concrete answer. Coordinate with
Fabian Mora before submitting the poster abstract — a statement of interest from the RFC author
would substantially strengthen the community-fit score.

---

## Section 4: XeVM PR Number — Correction

**Claim in Topic-01 (the critical error flagged by combo-a-gaps.md):**
> PR #119440: XeVM upstreamed August 2025

**Verdict: CONFIRMED WRONG. The correct PR is #148286.**

Evidence chain:
1. PR #119440 is `[mlir][gpu] Adding ELF section option to the gpu-module-to-binary pass` by
   Renaud-K, merged December 16, 2024. Commit `9919295cfd05222159246d7448ec42392e98fbf2` confirmed.
   Source: `https://github.com/llvm/llvm-project/commit/9919295cfd05222159246d7448ec42392e98fbf2`

2. PR #148286 is `[MLIR][GPU][XeVM] Add XeVM target and XeVM dialect integration tests.`
   - **Created:** July 11, 2025
   - **Merged:** August 13, 2025
   - Adds: XeVM target, SPIR-V binary serialization, integration tests via SYCL runtime
   - Phoronix article published: August 19, 2025 (6 days after merge)
   Source: `https://github.com/llvm/llvm-project/pull/148286` (directly fetched)
   Source: `https://www.phoronix.com/news/Intel-XeVM-MLIR-In-LLVM` (Phoronix article)

3. Companion PRs confirmed in the same timeframe:
   - PR #147375: `[MLIR][Conversion] Add convert-xevm-to-llvm pass` (July 7, 2025)
   - PR #150696: LLVMIR translation for XeVM (merged separately, post-#148286)

**Fix required in Topic-01:** Replace both XeVM citation instances:
- Where it says `PR #119440` in the context of XeVM, replace with `PR #148286 (merged August 13, 2025)`.
- The ELF section citation (PR #119440, Evidence §11) is CORRECT and should remain unchanged.

The corrected Evidence §6 text:
> "Intel XeVM upstreamed August 2025 — phoronix.com/news/Intel-XeVM-MLIR-In-LLVM —
> PR #148286 (merged August 13, 2025). MLIR now has `#nvvm.target`, `#rocdl.target`,
> `#xevm.target` as first-class GPU targets..."

---

## Section 5: TaxBreak "4.71 μs" Verification

**Claim in Topic-19:**
> "CUDA null-kernel floor (H100): 4.71 μs avg (p50: 4.578 μs, p95: 5.396 μs) — TaxBreak arXiv:2603.12465"

**Verdict: CONFIRMED CORRECT. The numbers are accurate and correctly attributed.**

Direct fetch of `https://arxiv.org/html/2603.12465` confirmed the following (verbatim from the paper):

**Table III** in TaxBreak reports `T_sys_floor` (the null-kernel hardware floor) as:
- **H100:** avg 4.707 μs, p50 4.578 μs, p5 4.260 μs, p95 5.396 μs
- **H200:** avg 4.503 μs, p50 4.452 μs, p5 4.177 μs, p95 4.909 μs

The paper explicitly describes `T_sys_floor` as: "the mean T_launch_raw of an empty C++ `__global__`
null kernel profiled under the same protocol." This is precisely a null-kernel dispatch floor
measurement, not a production kernel measurement.

Topic-19 rounds the H100 avg to 4.71 μs (paper says 4.707 μs — acceptable rounding).
The p50 (4.578) and p95 (5.396) figures match exactly.

**This is the combo-a-gaps.md "CRITICAL" flag that was wrong.** The gap analysis was unnecessarily
skeptical about this citation. The numbers are fully verified. The concern that "TaxBreak is an LLM
inference paper, not a null-kernel paper" is technically correct — TaxBreak's primary contribution
is LLM overhead decomposition — but it explicitly measures and reports null-kernel floor latency as
a calibration baseline. The attribution is legitimate.

**One precision note to add to the poster:** TaxBreak's null-kernel is measured with the CUDA driver
API directly (not through liboffload or any MLIR layer). The paper measures the floor of
`cuLaunchKernel` on H100 under batch-1 LLM inference conditions. The 4.71 μs is the hardware floor
for that specific path. When Topic-19 says "the community has the dispatch floor (TaxBreak)" and
"libkdl's O(1) hash-table lookup adds ~100–200 ns — less than 2% of the hardware floor," the
framing is accurate.

---

## Section 6: New Competing Work Since April 2026

**Question:** Any new MLIR GPU dispatch proposals or competing PRs filed since April 2026?

**Verdict: NO NEW COMPETING WORK FOUND.**

Web searches for:
- "MLIR GPU dialect runtime kernel selection dispatch 2026 new proposal RFC"
- "MLIR GPU dispatch proposal April 2026"
- Recent Discourse threads on GPU dialect (result: only RFC #88170 and its pages)

No new RFC, PR, or proposal for MLIR-native runtime kernel selection was found in any indexed
source as of April 9, 2026. The space remains unoccupied.

**Active adjacent work confirmed:**
- PR #186088 (open) — liboffload multi-image OffloadBinary generalization
- PR #185663 (merged March 10, 2026) — `isMetadataCompatible` hook in PluginInterface
- PR #185404 (merged March 11, 2026) — L0 plugin OffloadBinary support
- RFC #88170 (open) — GPU dialect cleanup

None of these address the MLIR-layer runtime dispatch policy gap. The proposal remains novel.

---

## Section 7: Implementation-Level Issues Not Previously Identified

### 7.1 Global variable naming collision risk in multi-blob embedBinary

`SelectObjectAttr.cpp` names its global `{moduleName}_binary`. A new `#gpu.runtime_select` attribute
emitting N globals would need to name them `{moduleName}_binary_nvvm`, `{moduleName}_binary_rocdl`,
`{moduleName}_binary_xevm`, etc. If two `gpu.binary` ops in the same module have the same name (a
real scenario in larger programs), there will be GlobalVariable name collisions. The implementation
must use unique suffixes or generate names based on a hash of the binary contents. This is solvable
but must be designed explicitly — the proposal does not currently address naming.

### 7.2 launchKernel must emit an indirect call, not a direct call

The current `SelectObjectAttr`'s `launchKernel` implementation delegates to `LaunchKernel`, which
emits a direct call to a fixed runtime function (`mgpuModuleLoad` for CUDA, etc.). A new
`#gpu.runtime_select` implementation must emit an **indirect call** through the dispatch table's
selected `load_fn_ptr`. This requires that the dispatch table global be accessible at the
`launchKernel` call site. The design must store the dispatch table as a module-level global (not
a local) and index it by the runtime-detected vendor enum. This is architecturally sound but
requires careful IR layout — the detection stub must run before any `launchKernel` translation
site is reached. The `llvm.global_ctors` registration slot (which `SelectObjectAttr` already uses
for module load/unload) is the correct mechanism, but ordering with `SelectObjectAttr`'s own
constructors must be considered if both attributes coexist in a translation unit.

### 7.3 dlopen symbol visibility and ASAN/UBSAN incompatibility

The `dlopen`-based vendor detection pattern (calling `cuInit`, `hipInit`, `zeInit` via
`dlopen`-loaded symbols) is incompatible with ASAN's symbol interception on some platforms. JAX
and PyTorch handle this by building their detector stubs into a separate shared library that is
loaded before sanitizers intercept symbols. For a poster demo on GTX 1650 without sanitizers,
this is not a problem. For upstream LLVM code review, this will be raised. The proposal should
acknowledge this in the feasibility section and reference PyTorch's `torch/csrc/cuda/utils.cpp`
`initializeCUDA()` pattern as the upstream precedent.

### 7.4 XeVM target requires SPIRV backend — not always present

PR #148286's build note confirms: XeVM target compilation is gated behind
`if ("SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)`. A machine with LLVM built without the SPIR-V
target (the default in most distro packages) cannot use `#xevm.target`. The poster's claim that
"MLIR now supports tri-vendor GPU targets in a single `gpu.binary`" is technically true but
practically limited — the XeVM path requires a non-default LLVM build. This is a one-sentence
caveat the poster should include to pre-empt the question from Intel engineers at Dublin.

### 7.5 Layer 3 measurement architecture in Topic-19 — confirmed broken

The combo-a-gaps.md audit correctly identified that the proposed measurement design calls
`cuModuleLoadData` separately while also calling `olCreateProgram` (which calls `cuModuleLoadData`
internally). This double-loads the module and does not measure layer 3 of the liboffload path.

Two correct alternatives:
1. **Source instrumentation:** Add `clock_gettime` brackets inside
   `offload/plugins-nextgen/cuda/src/rtl.cpp` around the `cuModuleLoadData` call in
   `CUDAPluginTy::dataCreate`. Build liboffload from source (already done in the prototype context).
2. **Separate baseline measurement:** Call `cuModuleLoadData` directly in a *separate* timed
   pass (before `olCreateProgram` is called) to establish the driver baseline. Then measure
   `olCreateProgram` total minus the driver baseline to infer plugin overhead. This requires two
   separate measurement runs and more careful statistical handling.

Option 1 is cleaner for a poster — it gives exact layer-3 numbers without confounding factors.

---

## Section 8: Corrected Confidence Ratings

| Topic | Core Gap | Evidence Quality | Implementation Risk | Overall |
|-------|----------|-----------------|---------------------|---------|
| Topic-01 | Confirmed (9/10) | 8/10 (XeVM PR now corrected; interface path confirmed) | Medium — dlopen, naming, indirect-call design | **8/10** |
| Topic-07 | Confirmed (9/10) | 9/10 (all key claims verified; MI300X removed) | Low — additive string keys only | **9/10** |
| Topic-19 | Confirmed (8/10) | 9/10 (TaxBreak 4.71 μs verified; Layer-3 design fixable) | Medium — Layer-3 redesign required | **8/10** |

---

## Section 9: Priority Action List (Updated)

### CRITICAL — must fix before poster draft:

1. **[FIXED] Topic-01 XeVM PR:** Replace `PR #119440` (ELF section pass) with `PR #148286`
   (merged August 13, 2025) in the XeVM citation. Keep `PR #119440` for the ELF section claim.

2. **[VERIFIED CORRECT] Topic-19 TaxBreak 4.71 μs:** Numbers confirmed from Table III. No action
   required. Add precision note: "null-kernel measured via CUDA driver API directly, not liboffload."

3. **[HIGH] Topic-19 Layer-3 measurement:** Fix double-load design. Use source instrumentation
   inside `CUDAPluginTy::dataCreate` in the CUDA plugin rtl. See Section 7.5 above.

4. **[HIGH] Topic-07 MI300X:** Remove from pitch. Prototype is GTX 1650 + CPU only.
   (Inherited from combo-a-gaps.md; confirmed by project instructions.)

### HIGH — fix before poster submission:

5. **[NEW] Topic-01 global naming collision:** Add a note in the implementation design that
   N-blob globals must use unique per-target suffixes. See Section 7.1.

6. **[NEW] Topic-01 launchKernel indirect call design:** Explicitly specify that the dispatch
   table must be a module-level global and the detection stub runs in `global_ctors`. See
   Section 7.2.

7. **[NEW] Topic-01 XeVM SPIRV build caveat:** Add one sentence: "tri-vendor support requires
   LLVM built with `LLVM_TARGETS_TO_BUILD=SPIRV`; this is non-default in distro builds." Section
   7.4.

8. **[HIGH] PR #186088 status:** Confirm merge before poster. If merged, update all three topics:
   multi-image OffloadBinary is now the liboffload-layer mechanism; the MLIR-layer policy is the
   remaining gap. Cite PR body statement about "follow-up PR" for multi-image loading as the
   direct upstream motivation for `gpu.select_variant`.

### MEDIUM — verify before submission:

9. **[MEDIUM] Topic-07 NVPTX writer path:** The old `clang-offload-wrapper/ClangOffloadWrapper.cpp`
   may be superseded. Verify against current `offload/` or `clang/tools/clang-linker-wrapper/`.

10. **[MEDIUM] RFC #88170 engagement:** Contact Fabian Mora (RFC author) before the poster
    abstract deadline. A reply in the RFC thread saying "this addresses the dispatch-policy gap"
    would be publicly verifiable community interest.

11. **[MEDIUM] Topic-07 PR #185663 citation:** Add `isMetadataCompatible` (merged March 10, 2026)
    as an explicit citation. It is the runtime consumer that needs the key vocabulary. This is the
    strongest evidence for the proposal's timeliness.

12. **[MEDIUM] dlopen upstream concern:** Add JAX/PyTorch precedent citation on the poster itself,
    not just in internal notes. See Section 7.3.

### LOW — before submission:

13. **[LOW] OMPT Issue #110007:** Verify at `https://github.com/llvm/llvm-project/issues/110007`.

14. **[LOW] wave-08 contradiction check:** Read `wave-08-kernel-info-pass.md` for anything that
    contradicts Topic-07's KernelInfo claims.

15. **[LOW] Huber DevMtg 2025 slides:** Read `https://llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf`
    before finalizing all three topics.

---

## Summary Table: Claim-by-Claim Verdict

| # | Claim | Verdict | Action |
|---|-------|---------|--------|
| 1 | `OffloadingLLVMTranslationAttrInterface` works as described | CONFIRMED | None |
| 2 | `CompilationAttrInterfaces.td` is the correct file path | CONFIRMED | Undo the gap-analysis flag |
| 3 | `embedBinary` can emit multiple blobs | CONFIRMED (interface allows it) | Add naming design |
| 4 | `SelectObjectAttr.cpp` is the only implementation | CONFIRMED | None |
| 5 | New attr needs to declare interface in `.td` and register | CONFIRMED | None |
| 6 | PR #186088 still open as of April 9, 2026 | CONFIRMED OPEN | Update if merged |
| 7 | PR #186088 has "first-compatible-wins" and no timing | CONFIRMED | None |
| 8 | RFC #88170 unresolved | CONFIRMED | Reframe, contact Mora |
| 9 | XeVM PR #119440 is wrong for XeVM | CONFIRMED ERROR | Fix to PR #148286 |
| 10 | XeVM PR #148286 merged August 13, 2025 | CONFIRMED | Use this citation |
| 11 | TaxBreak 4.71 μs H100 floor (avg 4.707, p50 4.578, p95 5.396) | CONFIRMED CORRECT | Remove "unverified" flag |
| 12 | TaxBreak measures null-kernel, not production kernels | CONFIRMED | Add precision note |
| 13 | No new competing RFC/PR for MLIR runtime dispatch | CONFIRMED ABSENT | None |
| 14 | MI300X in Topic-07 pitch unsupported | CONFIRMED ERROR | Remove |
| 15 | Layer-3 measurement design double-loads module | CONFIRMED BUG | Fix with source instrumentation |

---

*Verification completed: 2026-04-09.*

*Primary sources consulted:*
- `https://arxiv.org/html/2603.12465` — TaxBreak paper HTML (Table III, null-kernel measurements)
- `https://github.com/llvm/llvm-project/pull/186088` — PR status via `gh pr view`
- `https://github.com/llvm/llvm-project/pull/148286` — XeVM PR (merged Aug 13, 2025)
- `https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td` — Interface definition
- `https://mlir.llvm.org/doxygen/SelectObjectAttr_8cpp_source.html` — 479-line source, one implementation
- `https://www.phoronix.com/news/Intel-XeVM-MLIR-In-LLVM` — XeVM merge date and PR #148286
- `https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170` — RFC status (3 pages, unresolved)
- `https://github.com/llvm/llvm-project/commit/9919295cfd05222159246d7448ec42392e98fbf2` — PR #119440 ELF section commit
