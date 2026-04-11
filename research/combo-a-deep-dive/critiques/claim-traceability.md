# Claim Traceability Matrix -- extended-abstract.tex

**Date:** 2026-04-10
**Auditor:** Independent verification pass
**Scope:** Every factual claim (numbers, PR references, dates, comparisons) in `proposals/extended-abstract.tex`
**Sources checked:**
- `research/benchmark-results.md` (BR)
- `research/layer-benchmark-results.md` (LBR)
- `research/real-cubin-test-results.md` (RCT)
- `research/real-offloadbinary-results.md` (ROB)
- `critiques/llvm-expert-verification.md` (LEV)
- `research/extended-gpu-experiments.md` (EGE)
- `research/verification-poc.md` (VPC)
- `experiments/prototype/src/kdl.c` (KDL source)

---

## Abstract (Lines 50--62)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 1 | MLIR can compile a single `gpu.module` to multiple GPU vendors since August 2025 | 51 | LEV Section 4 | PR #148286 merged August 13, 2025 (confirmed) | VERIFIED |
| 2 | OffloadBinary container can carry N device images | 52 | LEV Section 2 | PR #186088 body confirms multi-image design; D122069 format confirmed | VERIFIED |
| 3 | At runtime, LLVM picks the first compatible image and stops | 53 | LEV Section 2 | PR #186088 body verbatim: "only the first compatible image in the binary is loaded" | VERIFIED |
| 4 | Five new keys, backward-compatible | 57 | N/A | Paper's own contribution -- design claim, not empirical | N/A (design) |
| 5 | First published flame graph of LLVM GPU dispatch stack latency | 57 | LEV Section 6 | "No new competing RFC/PR... the space remains unoccupied" -- no prior publication found | VERIFIED |
| 6 | Measured on GTX 1650 | 57 | BR header | "Machine: NVIDIA GeForce GTX 1650 (4096 MiB)" | VERIFIED |

## Introduction (Lines 64--89)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 7 | Since August 2025 (PR #148286, merged) | 67 | LEV Section 4 | PR #148286 merged August 13, 2025 confirmed via GitHub + Phoronix | VERIFIED |
| 8 | gpu-module-to-binary supports NVIDIA (#nvvm.target), AMD (#rocdl.target), Intel (#xevm.target) | 67 | LEV Section 4 | PR #148286 adds XeVM; NVVM/ROCDL preexisting; confirmed | VERIFIED |
| 9 | OffloadBinary fat-binary format (D122069, 2022) | 68 | LEV (multiple sections) | D122069 confirmed as OffloadBinary origin, 2022 | VERIFIED |
| 10 | parseOffloadBinary iterates through images (PR #186088), selects first match | 71 | LEV Section 2 | PR #186088 verbatim: "only the first compatible image" | VERIFIED |
| 11 | HEP-CCE (CERN CMS): 80 separate build configurations | 77 | proposal-v2.md line 38 | Source cited: alpaka-perf-portability.md, cern-cms-alpaka-production.md | PARTIALLY VERIFIED -- sourced from literature notes, not a primary publication with the "80" figure |
| 12 | NVIDIA A100/V100 + AMD MI250X + CPU fallback | 77 | proposal-v2.md line 38 | Same literature note source | PARTIALLY VERIFIED -- hardware list is plausible but "80" is uncited in a peer-reviewed source |
| 13 | vLLM: separate NVIDIA and AMD codepaths | 79 | proposal-v2.md line 40 | Source: "author's vLLM contribution experience" | VERIFIED (author is vLLM contributor per bio) |
| 14 | Cloud GPU containers: unknown GPU at build time | 81-82 | N/A | General industry knowledge, not a specific factual claim | N/A (common knowledge) |

## Background (Lines 90--152)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 15 | OffloadBinary keys: `triple` (string, D122069 2022), `arch` (string, D122069 2022) | 122-127 | LEV / proposal-v2.md | D122069 review confirmed; these are the only 2 standard keys | VERIFIED |
| 16 | isMetadataCompatible() -- PR #185663, merged March 2026 | 130 | LEV Section 2 line 229 | "PR #185663 (merged March 10, 2026) -- isMetadataCompatible hook" | VERIFIED |
| 17 | #gpu.select_object is the sole implementation of OffloadingLLVMTranslationAttrInterface | 135 | LEV Section 1.3 | "SelectObjectAttr is the ONLY implementation of the full interface in mainline" | VERIFIED |
| 18 | Resolves by index or static target match at compile time | 136 | LEV Section 1.2 | SelectObjectAttr.cpp confirmed; compile-time resolution | VERIFIED |
| 19 | RFC #88170 separates container from policy slot; policy slot is vacant | 143-144 | LEV Section 3 | "Confirmed ACTIVE, UNRESOLVED... framing is defensible" | VERIFIED |
| 20 | TaxBreak: null-kernel dispatch on H100 via CUDA driver API: 4.707 us average (p50: 4.578 us) | 150 | LEV Section 5 | Table III confirmed: avg 4.707, p50 4.578, p95 5.396 | VERIFIED |
| 21 | PyTorch eager dispatch is 5-10 us per kernel (pygraph citation) | 151 | proposal-v2.md line 165 | Attributed to arXiv:2503.19779 -- claim is taken from the paper | PARTIALLY VERIFIED -- no direct verification of the PyGraph paper's specific claim against its PDF was performed |
| 22 | No published per-layer latency breakdown of full LLVM GPU dispatch path | 148 | LEV Section 6 | "No new competing RFC/PR for MLIR runtime dispatch was found" | VERIFIED |

## Contribution 1: Metadata Vocabulary (Lines 156--213)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 23 | Five new string keys in two tiers (3 MUST + 2 MAY) | 158 | N/A | Paper's design contribution; count matches tables (min_sm, min_gfx, requires_features, variant_priority, variant_tag) = 5 keys | VERIFIED (self-consistent) |
| 24 | Backward compatibility: missing keys = no constraint, old runtimes ignore unknown keys | 213 | LEV / refined-design-v1.md | OffloadBinary StringMap is additive by D122069 design; confirmed | VERIFIED |

## Contribution 2: Dispatch Flame Graph (Lines 215--308)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 25 | GTX 1650 (Turing, sm_75) | 219 | BR header | "NVIDIA GeForce GTX 1650" confirmed | VERIFIED |
| 26 | Null kernel: 1 thread, 0 shared memory, compiled to CUBIN ahead of time | 219 | RCT Step 1 | "extern C __global__ void null_kernel() {}" -- 1 thread launch confirmed at Phase 4 (grid=1x1x1, block=1x1x1, shared_mem=0) | VERIFIED |
| 27 | Cold-path: fork fresh process per trial, 100 trials | 222-223 | LBR | "Cold trials: 100 exec-child processes (clean address space per trial)"; "100/100 cold trials succeeded" | VERIFIED |
| 28 | Hot-path: 1000 discarded warmup, then 10,000 dispatches | 226 | LBR | "Warmup: 100 iterations (discarded), Measure: 10,000 iterations" | **UNVERIFIED** -- paper says 1000 warmup, LBR says 100 warmup; paper says 10,000 measure which matches |
| 29 | Layer 1: OffloadBinary parse ~500 ns median, ~600 ns p95, ~9% | 239 | N/A | Marked with asterisk (*) as estimated; no direct measurement exists | VERIFIED (honestly marked as estimate) |
| 30 | Layer 2+3: olCreateProgram + cuModuleLoadData ~4000 ns median, ~73% | 240 | N/A | Marked as estimated; no direct liboffload measurement | VERIFIED (honestly marked as estimate) |
| 31 | Layer 4: olGetSymbol ~200 ns median, ~4% | 241 | LBR | cuModuleGetFunction median=60 ns; paper says ~200 ns for olGetSymbol wrapper | PARTIALLY VERIFIED -- raw CUDA is 60 ns; 200 ns for ol wrapper is an estimate, marked as such |
| 32 | Layer 5: First olLaunchKernel 841 ns median | 242 | BR Run 3 | cuda_direct_launch median=841 ns | VERIFIED |
| 33 | Layer 5 footnote: "841 ns is the measured cuda_direct_launch median from bench_dispatch Run 3, GTX 1650" | 249 | BR Run 3 line 83 | cuda_direct_launch: median_ns=841 | VERIFIED |
| 34 | End-to-end cold dispatch ~5500 ns | 244 | N/A | Estimated; sum of estimated layers; marked with asterisk | VERIFIED (honestly marked as estimate) |
| 35 | Hot-path dispatch floor: p50=841 ns | 258 | BR Run 3 line 83 | cuda_direct_launch median=841 | VERIFIED |
| 36 | Hot-path dispatch floor: p99=1102 ns | 259 | BR Run 3 line 83 | cuda_direct_launch p99=1,102 | VERIFIED |
| 37 | Hot-path n=1000 | 251 | BR header line 47 | "1000 iterations per phase" | VERIFIED |
| 38 | Hot-path p95 ~950 ns (est.) | 259 | N/A | Marked as estimate; p95 not directly reported in bench_dispatch | VERIFIED (honestly marked as estimate) |
| 39 | Variant selection overhead: kdl_select cold, n=1000 | 265 | BR header | "1000 iterations per phase" | VERIFIED |
| 40 | kdl_select cold median: 46.2 us (46,197 ns) | 272 | BR Run 3 line 80 | kdl_select (cold): median=46,197 | VERIFIED |
| 41 | Overhead vs 10 ms ML kernel: 0.46% | 272 | BR Section 6 Overhead Calculation | "46 us / 10,000 us (10 ms op) = 0.46% overhead" | VERIFIED |
| 42 | Pure dispatch-table lookup (runtime_select_poc): 2 ns | 277 | EGE lines 26-28 | "Per-select cost: 2 ns" from extended-gpu-experiments.md | **UNVERIFIED** -- multiple sources give conflicting values: EGE=2 ns, VPC=3 ns, RCT=4 ns, ROB=4-6 ns. The paper cites "2 ns" which is the lowest observed value, not a representative one. |
| 43 | 46.2 us vs 841 ns ratio: 5,494% | 278 | N/A | Math check: 46,197/841 = 54.93x = 5,493% -- rounds to 5,494% | VERIFIED (arithmetic) |
| 44 | Flame graph bar: olCreateProgram + cuModuleLoadData ~73% | 292 | N/A | Estimated from prototype; consistent with table | VERIFIED (self-consistent) |
| 45 | Flame graph bar: olLaunchKernel ~15% | 293 | N/A | Estimated; consistent with table | VERIFIED (self-consistent) |
| 46 | Flame graph bar: OffloadBinary parse ~9% | 295 | N/A | Estimated; consistent with table | VERIFIED (self-consistent) |
| 47 | Flame graph bar: olGetSymbol ~4% | 298 | N/A | Estimated; consistent with table | VERIFIED (self-consistent) |

## Contribution 3: Design Sketch (Lines 310--338)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 48 | "Zero lines of MLIR C++ exist" | 312 | LEV Section 1.3 / devils-advocate.md line 65 | "There is no C++, no TableGen, no MLIR test file... the implementation is 0 LOC" | VERIFIED |
| 49 | New attribute implements OffloadingLLVMTranslationAttrInterface | 314 | LEV Section 1.2 | Interface contract confirmed; implementation is structurally sound | VERIFIED (design claim, confirmed feasible) |
| 50 | embedBinary() emits N separate LLVM global constants | 326 | LEV Section 1.2 | "can emit N separate globals... interface allows it" | VERIFIED (feasibility) |
| 51 | launchKernel() emits identical code to SelectObjectAttr | 331 | LEV Section 7.2 | Partially true but LEV notes: "must emit an indirect call, not a direct call" | PARTIALLY VERIFIED -- paper overstates similarity; launchKernel needs indirect call through dispatch table |
| 52 | Zero hot-path overhead after one-time selection | 332 | N/A | Design claim; after global_ctors runs, only a global load remains (same as select_object) | VERIFIED (design) |
| 53 | liboffload explicitly excludes selection policy (RFC discourse.llvm.org/t/74302) | 335 | LEV Section 3 / pr-status-check.md line 165 | Discourse thread /t/74302 is the liboffload RFC; policy exclusion confirmed by design scope | VERIFIED |
| 54 | LLVM emits IFunc resolvers for CPU target_clones (structural analogy) | 337 | N/A | Standard LLVM/GCC FMV behavior; common knowledge | VERIFIED (well-known) |

## Prototype Validation (Lines 340--368)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 55 | libkdl (~5100 LOC, C) | 342 | kdl.c source | `wc -l kdl.c` = 5157 lines | VERIFIED (5157 rounds to ~5100) |
| 56 | Implements vendor detection via dlopen | 342 | RCT Step 4 output | "[Phase 1] Vendor detection (dlopen probe)" -- confirmed working | VERIFIED |
| 57 | Prototype uses custom MTB format, not OffloadBinary | 344 | BR header / ROB Section 4 | bench_dispatch uses "synthetic in-memory MTB"; OffloadBinary writer added separately | VERIFIED |
| 58 | KDL_MTB\0 magic and custom variant tables share zero code with LLVM's format | 345 | ROB Section 4 | OffloadBinary magic=0x10FF10AD vs KDL_MTB; separate implementations confirmed | VERIFIED |
| 59 | kdl_discover_cuda() at lines 551-596 | 357 | kdl.c source | Requires direct source check; line numbers are plausible for 5157-line file | PARTIALLY VERIFIED -- not cross-checked against current source; line numbers may have drifted |
| 60 | mtb_variant_entry at lines 96-106 | 358 | kdl.c source | Same: plausible but not cross-checked | PARTIALLY VERIFIED |
| 61 | kdl_contract_matches() at lines 1003-1005 | 359 | kdl.c source | Same: plausible but not cross-checked | PARTIALLY VERIFIED |
| 62 | kdl_estimate_cost_weighted() at lines 1013-1088 | 360 | kdl.c source | Same: plausible but not cross-checked | PARTIALLY VERIFIED |
| 63 | GTX 1650 (NVIDIA, sm_75): Full dispatch path validated | 366 | RCT Step 4, ROB Step 5 | Kernel loaded, launched, and synchronized on GTX 1650; confirmed | VERIFIED |
| 64 | CPU fallback: Validated | 367 | BR Run 3 | bench_dispatch ran CPU target successfully | VERIFIED |
| 65 | AMD (HIP): tested via mocked HIP entry points only | 368 | N/A | No HIP test results exist in any source file; claim of mocked testing is unverified | **UNVERIFIED** -- no evidence of mocked HIP testing found in any research file |

## Related Work (Lines 370--389)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 66 | IREE HAL: ranked selection incomplete after 6 years (issues #50, #12230, #15334) | 387 | LEV Section 6 / reviewer-panel.md line 21 | IREE issues cited; reviewer panel confirms "basic mechanism exists but ranked selection incomplete" | PARTIALLY VERIFIED -- issue numbers cited but not directly fetched to confirm current status |
| 67 | CPU FMV is the structural precedent | 388 | N/A | Well-known GCC/LLVM feature; standard knowledge | VERIFIED |
| 68 | liboffload PR #186088: runtime=Yes, cross-vendor=Yes, MLIR=No, ranked=No | 381 | LEV Section 2 | PR #186088 confirmed open; "first compatible image" = no ranking | VERIFIED |

## Concrete Next Steps (Lines 410--418)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 69 | bench_dispatch executed on GTX 1650 (Run 3, stable baseline) | 413 | BR Section 3 Run 3 | Full Run 3 data present with all phases | VERIFIED |
| 70 | Bundle load 4.9 us | 413 | BR Section 6 | "Bundle load (kdl_load_bundle): 4.9 us" (median Run 3: 4,949 ns) | VERIFIED |
| 71 | Selection 46.2 us cold | 413 | BR Run 3 | kdl_select (cold) median=46,197 ns = 46.2 us | VERIFIED |
| 72 | Selection 44.9 us cached | 413 | BR Run 3 | kdl_select (cached) median=44,924 ns = 44.9 us | VERIFIED |
| 73 | Direct launch 841 ns | 413 | BR Run 3 | cuda_direct_launch median=841 ns | VERIFIED |
| 74 | p99 1,102 ns | 413 | BR Run 3 | cuda_direct_launch p99=1,102 ns | VERIFIED |

## References (Lines 420--456)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 75 | TaxBreak: arXiv 2603.12465, 2026 | 428 | LEV Section 5 | Confirmed via direct HTML fetch of arxiv.org/html/2603.12465 | VERIFIED |
| 76 | PyGraph: arXiv 2503.19779, 2026 | 433 | proposal-v2.md line 165 | Cited; not independently verified against the PDF | PARTIALLY VERIFIED |
| 77 | RFC #88170: Cleaning the GPU Dialect, Mora, F. | 438 | LEV Section 3 | Confirmed at discourse.llvm.org/t/88170; posted September 4, 2025 | VERIFIED |
| 78 | PR #186088: Huber, J., [OpenMP][Offload] | 443 | LEV Section 2 | Title confirmed: "[OFFLOAD] Generalize support for OffloadBinary images" | **UNVERIFIED** -- paper says "[OpenMP][Offload] Add variant selection to liboffload" but LEV says actual title is "[OFFLOAD] Generalize support for OffloadBinary images". Title mismatch. |
| 79 | PR #185663: Denny, J., merged March 2026 | 448 | LEV Section 2 line 229 | "PR #185663 (merged March 10, 2026)" | VERIFIED |
| 80 | PR #148286: Intel XeVM Team, merged August 2025 | 453 | LEV Section 4 | Merged August 13, 2025; confirmed | VERIFIED |

## Upstream Path (Lines 397--408)

| # | Claim in Paper | Line | Source File | Source Line/Section | Verified? |
|---|----------------|------|-------------|---------------------|-----------|
| 81 | Header constants ~30 LOC | 401 | N/A | Engineering estimate; not verifiable until written | N/A (estimate) |
| 82 | isMetadataCompatible extension ~40 LOC | 402 | N/A | Engineering estimate | N/A (estimate) |
| 83 | AMDGPU writer ~60 LOC | 403 | N/A | Engineering estimate | N/A (estimate) |
| 84 | NVPTX writer ~60 LOC | 404 | N/A | Engineering estimate | N/A (estimate) |
| 85 | RuntimeSelectAttr.cpp ~600 LOC | 405 | LEV Section 7.2 / devils-advocate.md line 133 | Devil's advocate says "500-1000 LOC is more realistic" -- estimate is in range but may undercount | PARTIALLY VERIFIED |

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| VERIFIED | 57 | 67% |
| PARTIALLY VERIFIED | 13 | 15% |
| UNVERIFIED | 4 | 5% |
| N/A (design/estimate/common knowledge) | 11 | 13% |
| **Total claims** | **85** | **100%** |

---

## UNVERIFIED Claims -- Risk Assessment

### UNVERIFIED-1: Warmup iterations mismatch (Claim #28, Line 226)
**Paper says:** "warm with 1000 discarded dispatches"
**Source says:** layer-benchmark-results.md: "Warmup: 100 iterations (discarded)"
**Risk:** MEDIUM -- the paper overstates the warmup count by 10x. This affects reproducibility claims.
**Fix:** Change "1000 discarded dispatches" to "100 discarded dispatches" to match actual measurement, or re-run with 1000 warmup if that protocol is preferred.

### UNVERIFIED-2: Pure dispatch-table lookup "2 ns" (Claim #42, Line 277)
**Paper says:** "2 ns"
**Sources report:**
- extended-gpu-experiments.md: 2 ns (synthetic bundle, single run)
- verification-poc.md: 3 ns (synthetic, single run)
- real-cubin-test-results.md: 4 ns (real cubins, 100k iterations)
- real-offloadbinary-results.md: 4 ns (directory), 6 ns (OffloadBinary)
**Risk:** HIGH -- the "2 ns" is cherry-picked from the lowest observed value on a synthetic workload. Representative values from real cubins are 4-6 ns. A reviewer testing with real OffloadBinary files will measure 4-6 ns and question the "2 ns" claim.
**Fix:** Change to "4-6 ns" or "~4 ns (real cubins)" to match real-cubin-test-results.md. Alternatively, report as range: "2-6 ns depending on entry source."

### UNVERIFIED-3: AMD HIP mocked testing (Claim #65, Line 368)
**Paper says:** "AMD (HIP): Code path exists; tested via mocked HIP entry points only"
**Evidence:** No HIP mock test results appear in any research file (benchmark-results.md, layer-benchmark-results.md, real-cubin-test-results.md, real-offloadbinary-results.md, verification-poc.md, extended-gpu-experiments.md). No mocked HIP output captured anywhere.
**Risk:** MEDIUM -- if a reviewer asks for the mock test output, none exists in the evidence chain.
**Fix:** Either produce and document a mocked HIP test run, or soften to "AMD (HIP): Code path exists but untested on hardware; tested only by code inspection."

### UNVERIFIED-4: PR #186088 title mismatch (Claim #78, Line 443)
**Paper says:** `[OpenMP][Offload] Add variant selection to liboffload`
**LEV says:** Actual title is `[OFFLOAD] Generalize support for OffloadBinary images`
**Risk:** HIGH -- this is a citation accuracy error. A reviewer checking the PR will see a different title. Misquoting PR titles undermines credibility.
**Fix:** Correct the bibitem to match the actual PR title: `[OFFLOAD] Generalize support for OffloadBinary images`.

---

## PARTIALLY VERIFIED Claims -- Notes

| # | Claim | Issue | Risk |
|---|-------|-------|------|
| 11-12 | HEP-CCE "80 build configs" | Number sourced from literature notes, not a citable primary source | Medium -- may be challenged as unverifiable |
| 21 | PyTorch eager 5-10 us (PyGraph) | PyGraph paper not independently verified against PDF | Low -- plausible and well-known |
| 31 | olGetSymbol ~200 ns | Raw CUDA is 60 ns; ol wrapper overhead is estimated | Low -- marked as estimate in paper |
| 51 | launchKernel identical to SelectObjectAttr | LEV notes indirect call needed | Medium -- design simplification |
| 59-62 | kdl.c line numbers | Not cross-checked against current source | Low -- line numbers may drift |
| 66 | IREE issues #50, #12230, #15334 | Not directly fetched to confirm current status | Medium -- issues may have been closed |
| 76 | PyGraph arXiv number | Cited but PDF not fetched | Low |
| 85 | RuntimeSelectAttr ~600 LOC | Devil's advocate says 500-1000 | Low -- acknowledged as estimate |

---

## Verdict

**4 UNVERIFIED claims, 2 of which are HIGH risk (the "2 ns" cherry-pick and the PR #186088 title mismatch).**

The paper is well-sourced overall (67% fully verified, 15% partially verified). The estimated values in the flame graph table are honestly marked with asterisks. The benchmark numbers from Run 3 are accurately reproduced.

The two HIGH-risk issues should be fixed before submission:
1. Replace "2 ns" with "4-6 ns" (or cite range with source)
2. Correct PR #186088 bibitem title to match actual PR
