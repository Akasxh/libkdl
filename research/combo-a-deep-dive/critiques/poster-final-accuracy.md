# Poster Final Accuracy Check — poster-combo-a.html

**Date:** 2026-04-10
**Reviewer:** Critic (automated accuracy audit)
**Cross-reference:** `research/combo-a-deep-dive/research/pinned-benchmark-results.md` (pinned, 3-run cross-run)

---

## Verdict: 7 issues found. Score: 3/5

The poster's primary numbers (cold, hot, launch, sync medians) are correct against the pinned benchmark. However, there is a systematic problem: **the poster mixes pinned medians with unpinned p99 values** in the same table, and uses the **unpinned selection overhead (6 ns)** despite the pinned benchmark document recommending 3 ns. One arithmetic error exists. The negative-list items are clean.

---

## Check 1: Numbers vs Pinned Benchmark Data

### Medians (from poster stat boxes and flame graph)

| Poster Claim | Pinned 3-Run Median | Match? |
|---|---|---|
| Cold module load: **36.0 us** | 35,952.7 ns = **36.0 us** | **PASS** (rounded correctly) |
| Hot-path total: **4.1 us** | 4,104.0 ns = **4.10 us** | **PASS** |
| Selection overhead: **6 ns** | **3 ns** (pinned, 100K iters, 0% CV) | **FAIL** — see Issue #1 |
| cuLaunchKernel: **1.6 us** | 1,650.0 ns = **1.65 us** | **PASS** (rounded to 1 sig fig) |
| cuStreamSync: **2.5 us** | 2,454.0 ns = **2.45 us** | **PASS** (rounded to 1 sig fig) |
| cuModuleGetFunction: **60 ns** | 63.3 ns | **PASS** (reported as unpinned median, both are ~60 ns) |

### p99 values (from poster Layer Decomposition table, lines 1039-1044)

| Poster p99 Claim | Pinned 3-Run p99 | Unpinned p99 | Source Used |
|---|---|---|---|
| Cold: **111.3 us** | 59,601.7 ns = **59.6 us** | 111,269 ns = **111.3 us** | **UNPINNED** |
| Warm: **16.3 us** | 56,434.3 ns = **56.4 us** | 16,311 ns = **16.3 us** | **UNPINNED** |
| GetFunction: **61 ns** | 90 ns | 61 ns | **UNPINNED** |
| LaunchKernel: **3.5 us** | 3,011 ns = **3.0 us** | 3,496 ns = **3.5 us** | **UNPINNED** |
| StreamSync: **3.6 us** | 3,600.7 ns = **3.6 us** | 3,647 ns = **3.6 us** | **Both match** |
| Hot-path p99: **7.1 us** | 3,011 + 3,601 = **6.6 us** | 3,496 + 3,647 = **7.1 us** | **UNPINNED** |

---

## Issues Found

### Issue #1: Selection overhead uses unpinned value (6 ns), not pinned (3 ns) — MAJOR

**Evidence:** Pinned benchmark Section 7 "Recommended Poster Numbers" explicitly states:
> `| OffloadBinary variant selection | **3 ns** | runtime_select_poc, 100K iters |`

The 6 ns value is the **unpinned** measurement from `real-offloadbinary-results.md:251`. The pinned benchmark shows all 3 runs at exactly 3 ns with 0% CV — the most stable measurement in the entire dataset. The poster's caption (line 980) says "CPU-pinned (taskset -c 0)" and "3-run cross-run medians" but then reports 6 ns instead of 3 ns.

The poster uses 6 ns in: thesis strip (line 822), flame graph (line 975), flame bar CSS callout (line 754), stat box (line 992), layer table (line 1045), overhead table header (line 1056), all 4 overhead percentage rows (lines 1058-1061), and the callout text (line 1066). That is 11+ occurrences.

**Severity:** MAJOR. The number is defensible as "4-6 ns range across modes" (per `iteration2-verification.md:67`), but the poster claims pinned methodology while citing unpinned data. Either cite 3 ns (matching pinned data) or change the methodology description to not claim pinned.

**Fix:** Either (a) change all "6 ns" to "3 ns" to match the pinned benchmark, or (b) change to "3-6 ns" as a range and remove the specific "CPU-pinned" claim from the caption, or (c) keep 6 ns but label it as "unpinned worst-case."

### Issue #2: p99 column uses unpinned data while medians use pinned data — MAJOR

**Evidence:** The Layer Decomposition table (lines 1039-1044) presents medians from pinned 3-run data (36.0 us, 1.6 us, 2.5 us) but p99 values from the unpinned single-run data (111.3 us, 16.3 us, 61 ns, 3.5 us). See the comparison table above — every p99 value except cuStreamSync matches the unpinned dataset, not the pinned dataset.

The caption (line 1049) says "100 cold trials via exec-child isolation, 10K warm iterations" but does not specify pinned vs unpinned. The flame graph caption (line 980) explicitly says "CPU-pinned."

**Severity:** MAJOR. Mixing data from two different measurement conditions in the same table without disclosure is a methodological integrity issue. A reviewer who tries to reproduce the p99 values using pinned runs will get different numbers (e.g., 59.6 us vs 111.3 us for cold load — a 2x discrepancy).

**Fix:** Either (a) replace all p99 values with pinned 3-run p99 averages (59.6, 56.4, 90, 3.0, 3.6 us), or (b) add a footnote: "p99 values from single unpinned run; medians from 3-run pinned cross-run."

### Issue #3: "1 MHz → 0.4% overhead" arithmetic is wrong at 6 ns — MINOR

**Evidence:** Line 1066: `"At 6 ns per dispatch, a kernel launched at 1 MHz frequency incurs 0.4% selection overhead."`

Math: 6 ns × 1,000,000 calls/sec = 6,000,000 ns/sec = 6 ms/sec. 6 ms / 1000 ms = **0.6%**, not 0.4%.

The 0.4% figure is correct for **4 ns** (from `real-cubin-test-results.md:175`: "At 4 ns per call, a kernel launched at 1 MHz frequency incurs 0.4% selection overhead"). The poster changed the selection value to 6 ns without updating the arithmetic.

**Severity:** MINOR. The qualitative point holds (it's small), but reviewers who check the math will notice.

**Fix:** Either change to "0.6%" (if keeping 6 ns) or change to "0.3%" (if using 3 ns) or change the selection value to 4 ns to match the 0.4%.

### Issue #4: IREE row cites PR #186088 (a liboffload PR) as IREE evidence — MINOR

**Evidence:** Line 1153: `<td><span class="dot-p">Partial</span> (PR #186088 defers ranked selection)</td>` in the IREE HAL row.

PR #186088 is "[OFFLOAD] Generalize support for OffloadBinary images" — a **liboffload** PR by Duran, not an IREE PR. The correct IREE issue references are #12230 and #15334 (per `extended-abstract-v3.tex:307` and `proposal-v2.md:393`).

The liboffload PR #186088 is already correctly cited in its own row (line 1173-1178).

**Severity:** MINOR. The observation that IREE has partial ranked selection is accurate, but the evidence citation is misattributed.

**Fix:** Change to `(issues #12230, #15334)` in the IREE row.

### Issue #5: Flame graph bar widths are NOT proportional to actual data — MINOR

**Evidence:** Line 980 caption: `"Width = fraction of total cold-path latency."`

Actual fractions (pinned medians, total ~40,150 ns):
- cuModuleLoadData: 35,953/40,150 = **89.5%** → bar width=100% (rescaled, OK)
- cuStreamSync: 2,454/40,150 = **6.1%** → bar width=18% (**3x exaggerated**)
- cuLaunchKernel: 1,650/40,150 = **4.1%** → bar width=12% (**3x exaggerated**)
- cuModuleGetFn: 63/40,150 = **0.16%** → bar width=3% (**19x exaggerated**)
- Selection: 6/40,150 = **0.015%** → bar width=2.5% (**167x exaggerated**)

The smaller bars are heavily inflated for legibility.

**Severity:** MINOR. This is a common poster design choice — making small bars visible. But the caption explicitly claims proportionality. Either the caption or the widths should change.

**Fix:** Either (a) change caption to "Width approximately proportional; smaller layers enlarged for legibility," or (b) make widths proportional (cuStreamSync ~7%, cuLaunchKernel ~4.5%, others <1% — but these would be near-invisible, so option (a) is better).

### Issue #6: cuModuleLoadData warm median differs between table and pinned data — MINOR

**Evidence:** Line 1040: `cuModuleLoadData warm` median = `10.0 us`.

- Unpinned median: 10,069 ns = 10.1 us (rounds to 10.0 us — acceptable)
- Pinned 3-run median: 9,615 ns = 9.6 us

If the poster is using pinned data for medians (as it does for cold=36.0 us), the warm median should be 9.6 us, not 10.0 us. 10.0 us matches the unpinned data.

**Severity:** MINOR. The difference (9.6 vs 10.0 us) is small and within noise, but reinforces the data-mixing pattern from Issue #2.

**Fix:** Change to 9.6 us if standardizing on pinned data.

### Issue #7: "cuModuleLoadData" flame bar claims "90%" but label says "36.0 us — 90%" — Acceptable approximation

**Evidence:** Line 952: `36.0 µs — 90%`

Actual: 35,953/40,150 = 89.5%. Rounded to 90% is acceptable for a poster.

**Severity:** Not an issue. Flagging for completeness only.

---

## Check 2: PR References

| Reference | Poster Description | Verified? |
|---|---|---|
| PR #148286 | XeVM target upstreamed (Aug 2025) | **PASS** — confirmed in `pr-status-check.md` and `extended-abstract-v3.tex:360` |
| PR #186088 | liboffload loads first compatible GPU binary | **PASS** — confirmed as "[OFFLOAD] Generalize support for OffloadBinary images", OPEN status |
| PR #185663 | `isMetadataCompatible()` consumer hook merged | **PASS** — confirmed MERGED 2026-03-10 |
| Issue #75356 | Chapel team needs dlsym-for-GPUs, open 2.5 years | **PASS** — "Name-Based Kernel Loading" opened Nov 2023, ~2.4 years to April 2026 |
| RFC #88170 | GPU dialect cleanup separates container from dispatch policy | **PASS** — confirmed in `pr-status-check.md:101` |
| D122069 | OffloadBinary original (2022) | **PASS** — confirmed in `rfc-review.md:128` |

---

## Check 3: Negative List

| Prohibited Item | Present in poster? |
|---|---|
| MI300X | **NO** — not found |
| "roofline model" | **NO** — not found |
| IREE #50 | **NO** — not found |

**PASS** — all three negative-list items are absent.

---

## Check 4: RFC #88170 Description

Poster (line 873): `"GPU dialect cleanup separates container from dispatch policy — the policy slot is empty"`

Verified against `pr-status-check.md:101`: `"[RFC] Cleaning the GPU dialect (thread #88170, Sept 2025) — MLIR gpu dialect restructuring, not runtime dispatch"`

The poster's characterization ("separates container from dispatch policy") is a fair summary. The "policy slot is empty" is the poster's interpretation (the gap thesis), which is supported by the evidence chain.

**PASS**

---

## Check 5: "So What" Callout Box

Present at lines 839-841:
> "In plain English: When you compile a GPU program for multiple vendors (NVIDIA, AMD, Intel), LLVM currently picks the first binary that works. We measure the cost and propose a way to pick the best one."

**PASS** — present and accurate.

---

## Check 6: Metadata Table (5 Keys)

| # | Key | Tier | Type | Example | Present? |
|---|---|---|---|---|---|
| 1 | `min_sm` | MUST | uint | `"75"` | **PASS** |
| 2 | `min_gfx` | MUST | arch:family | `"gfx90a:cdna2"` | **PASS** |
| 3 | `requires_features` | MUST | comma-list | `"tensor_core_nv,bf16"` | **PASS** |
| 4 | `variant_priority` | MAY | uint | `"10"` | **PASS** |
| 5 | `variant_tag` | MAY | string | `"optimized"` | **PASS** |

5 keys, correct tiers, correct types, correct examples.

Note: The example `"tensor_core_nv"` uses the `_nv` suffix (vendor-qualified), which was a fix from an earlier iteration. **PASS**.

---

## Check 7: MLIR Syntax

The `#gpu.runtime_select` block (lines 1004-1010) uses:
- `gpu.binary @kernels` — valid MLIR syntax for a named gpu.binary op
- `#gpu.runtime_select<strategy = "rank_by_priority", fallback = "cpu">` — proposed new attribute (design sketch, not yet upstream)
- `#gpu.object<#nvvm.target<chip="sm_75">, bin="...">` — valid existing MLIR GPU syntax
- `#rocdl.target<chip="gfx90a">` — valid existing MLIR GPU syntax

The syntax is consistent with MLIR GPU dialect conventions. The proposed attribute follows MLIR parameterized-attribute syntax correctly.

**PASS** — valid MLIR syntax for a design sketch.

---

## Check 8: Related Work Table

| System | Runtime Select? | Cross-Vendor? | MLIR-Native? | Ranked? | Accurate? |
|---|---|---|---|---|---|
| IREE HAL | Yes | Yes | MLIR-based | Partial | **PASS** (IREE does runtime dispatch; ranked selection is genuinely partial) |
| chipStar | Yes (SPIR-V) | Yes | No | No | **PASS** |
| Proteus (LLNL) | Yes (JIT) | Partial | No | No | **PASS** |
| liboffload #186088 | Yes | Yes | No | No (first-wins) | **PASS** |
| CPU FMV | Yes | N/A | No | Yes (IFunc) | **PASS** |
| This Work | Metadata+Measurement+Design | Yes | Yes | Yes | **PASS** |

Note: IREE "Ranked?" column cites PR #186088 instead of IREE issues — see Issue #4 above.

---

## Check 9: Author Info

Line 803: `S. Akash — IIT Patna | CERN GSoC | vLLM contributor | EuroLLVM Dublin 2026`

**PASS** — matches project documentation.

---

## Check 10: Flame Graph Bar Widths

See Issue #5 above. Widths are not proportional to actual data but are inflated for legibility. The caption claims proportionality.

---

## Summary

| Check | Result |
|---|---|
| 1. Numbers match pinned data | **PARTIAL** — medians match, p99s use unpinned, selection uses unpinned |
| 2. PR references correct | **PASS** |
| 3. RFC #88170 described correctly | **PASS** |
| 4. Negative list clean | **PASS** |
| 5. Flame graph widths proportional | **FAIL** — widths inflated, caption claims proportionality |
| 6. Metadata table correct | **PASS** |
| 7. MLIR syntax valid | **PASS** |
| 8. Related work table accurate | **PASS** (with minor IREE citation issue) |
| 9. Author info correct | **PASS** |
| 10. "So What" callout present | **PASS** |

**Score: 3/5** — Two MAJOR issues (data source mixing, selection overhead mismatch), one arithmetic error, and two minor citation/visualization issues prevent a clean bill of health. The qualitative conclusions hold regardless, but the specific numbers in the poster do not consistently match the pinned benchmark document that is designated as the authoritative source.
