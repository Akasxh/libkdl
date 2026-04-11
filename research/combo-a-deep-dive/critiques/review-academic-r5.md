# Academic Rigor Review R5 — Extended Abstract v3

**Reviewer posture:** OSDI/SOSP-level skeptical systems reviewer  
**Date:** 2026-04-10  
**Target:** `proposals/extended-abstract-v3.tex`  
**Supporting data:** `layer-benchmark-results.md`, `pinned-benchmark-results.md`, `ptx-vs-cubin-results.md`

---

## Issue 1: No Confidence Intervals or Error Bars on Any Reported Number (MAJOR)

The paper reports single-point medians and p99 values (e.g., "cuModuleLoadData cold: 42,670 ns", "cuLaunchKernel: 1,573 ns") without confidence intervals, standard deviations, or any measure of statistical uncertainty. The pinned benchmark data (`pinned-benchmark-results.md`) actually contains 3-run cross-run statistics with stddev and CV, but **none of this reaches the paper**.

This matters because:

- The cold module load has a cross-run CV of 1.3% on the mean but 13.3% on p99 (pinned data, Section 2). Reporting a single p99 without a CI is misleading.
- The warm module load (Layer 2) shows a cross-run mean CV of **15.5%** and a p99 CV of **117.8%** (pinned data). The paper reports a single "10,069 ns" median as though it were a stable number, but Run 1 produced a 1.28 ms outlier (max=1,275,655 ns) that inflates the mean by 30%.
- The "46.2 us" selection overhead (line 215 of the .tex) comes from a single Run 3 of bench_dispatch. The pinned cross-run median is 52.6 us — a 14% discrepancy. Which number is correct? The paper does not disclose that it cherry-picked one run.

**Fix:** Report cross-run medians with 95% CIs or at minimum (min, median, max) across runs. For the cold-path table (Section 3.2.1), add a "CV%" or "95% CI" column. Three runs is the bare minimum for any claims about reproducibility.

---

## Issue 2: The Paper Reports Unpinned Numbers While Pinned Data Exists (MAJOR)

The cold-path layer decomposition table (lines 171-189 of the .tex) reports numbers from the **unpinned** `layer-benchmark-results.md` (single run, 2026-04-09), not the pinned 3-run data collected one day later (`pinned-benchmark-results.md`, 2026-04-10). Specifically:

| Layer | Paper reports (unpinned) | Pinned 3-run median |
|-------|--------------------------|---------------------|
| L2 cold | 42,670 ns | 35,953 ns |
| L4 | 1,573 ns | 1,650 ns |
| L5 | 2,475 ns | 2,454 ns |

The pinned data has lower variance (L4 cross-run CV of 1.0% pinned vs. unknown unpinned since there's only one unpinned run) and is methodologically superior (CPU affinity eliminates migration noise). The paper uses the inferior dataset. Worse, the pinned data shows cold module load is 32% faster when pinned (37,029 ns vs. 54,633 ns mean) — this means the unpinned cold-path number conflates OS scheduler jitter with actual CUDA driver cost.

**Fix:** Use pinned cross-run medians as the primary reported numbers. Footnote the unpinned numbers if desired. The recommended poster numbers in `pinned-benchmark-results.md` Section 7 already solve this — the paper just has not been updated to use them.

---

## Issue 3: Selection Overhead "46.2 us" Conflates Entirely Different Operations (MAJOR)

The paper (line 207-218) labels the `kdl_select` cold measurement as "Selection overhead" and reports 46.2 us. But `bench_dispatch.c` (line 17 of its source) describes itself as testing "CPU-only path" with a "synthetic MTB bundle, CPU target path" (`pinned-benchmark-results.md` Section 4). This 46.2 us includes:

1. Parsing a synthetic MTB (Multi-Target Binary) format — **not** the LLVM OffloadBinary format
2. Hash table lookups in the kdl runtime — **not** the proposed `isMetadataCompatible()` scan
3. CPU fallback selection logic — **not** the GPU runtime_select path

Meanwhile, the `runtime_select_poc` (which actually parses real OffloadBinary files and does the proposed metadata-based selection) measures **3 ns** per selection (pinned, 100K iterations) and **90 ns** one-shot cold selection (pinned cross-run mean).

Presenting the 46.2 us kdl prototype number alongside the 3 ns OffloadBinary number in the same paper, under the same "Selection overhead" heading, creates confusion. A reader cannot tell which number applies to the proposed LLVM design. The 46.2 us number measures a completely different code path (libkdl's MTB format) from the proposed contribution (OffloadBinary metadata).

**Fix:** Clearly separate the two measurements. State explicitly: "libkdl prototype (MTB format, not OffloadBinary): 46.2 us cold. OffloadBinary dispatch-table lookup (runtime_select_poc): 3 ns steady-state, ~90 ns one-shot." Make clear which number is relevant to the proposed LLVM upstream design.

---

## Issue 4: "first published layer-by-layer latency decomposition" Claim is Unsubstantiated (MAJOR)

Line 66 claims: "the first published layer-by-layer latency decomposition of the LLVM GPU dispatch stack." This is a priority claim that requires evidence of a literature search. The paper cites TaxBreak (arXiv:2603.12465) and PyGraph (arXiv:2503.19779), both of which decompose GPU dispatch latency in similar ways:

- TaxBreak Table III reports null-kernel floor measurements (T_sys_floor) broken down by GPU model, which is exactly a dispatch latency decomposition.
- NVIDIA's own CUPTI profiling tools have documented per-API-call latency breakdowns for years (e.g., nsight-systems traces of cuModuleLoadData, cuLaunchKernel).
- ICPP 2019 and GTC papers have published CUDA driver API call latencies.

The paper's actual contribution here is decomposing the *LLVM offloading stack specifically* (not CUDA in general), but the claim as written — "first published layer-by-layer latency decomposition" — is broader than what's defensible. No systematic literature search is documented.

**Fix:** Qualify the claim: "the first published layer-by-layer latency decomposition **of the LLVM GPU offload path specifically**" — and add a sentence acknowledging that CUDA driver-level microbenchmarks exist elsewhere (TaxBreak, nsight-systems profiles). Alternatively, drop "first" entirely and let the contribution stand on its own merit.

---

## Issue 5: Cold-Path Measurement Confounds exec Overhead with cuModuleLoadData (MODERATE)

The paper reports cold-path cuModuleLoadData as 42,670 ns median (line 179). But `bench_layers.c` lines 282-349 show the cold-child measurement includes:

1. `execve(/proc/self/exe)` — process creation
2. `dlopen("libcuda.so.1")` — shared library loading
3. `cuInit(0)` — CUDA driver initialization
4. `cuDeviceGet(0)` — device enumeration
5. `cuCtxCreate()` — context creation
6. `cuModuleLoadData()` — the actual module load

Only step (6) is what the table header says is being measured. The paper acknowledges this in the results markdown ("Includes: process startup, dlopen(libcuda.so.1), cuInit, cuDeviceGet, cuCtxCreate, and cuModuleLoadData") but the paper's table header simply says "cuModuleLoadData (cold)" — which implies the measurement isolates that single call.

The warm measurement (10,069 ns) is a much cleaner measurement of cuModuleLoadData in isolation. The difference between warm and cold (42,670 - 10,069 = ~32,600 ns) is almost entirely exec/init/ctx overhead, not module loading.

**Fix:** Rename the cold row to "Cold init + cuModuleLoadData" or "Full cold path (exec → cuModuleLoadData)" in the table. Add a footnote clarifying what the cold measurement includes. The paper gets close with the "(cold)" qualifier, but "cuModuleLoadData (cold)" misleads readers into thinking cuModuleLoadData itself is 4x slower cold vs. warm.

---

## Issue 6: PTX vs CUBIN Benchmark Uses Different Binary Sizes (MODERATE)

The `ptx-vs-cubin-results.md` reports CUBIN at 2984 bytes, but the layer benchmark uses a 4328-byte CUBIN. Meanwhile, the PTX is only 85 bytes ("hand-crafted minimal"). The paper's layer decomposition table says "4328-byte CUBIN" (line 165). These are different cubins of the same null kernel — presumably different compilation flags or toolchain versions produced different ELF sizes.

The 85-byte hand-crafted PTX vs. 2984-byte nvcc-compiled CUBIN is not an apples-to-apples comparison. PTX JIT overhead scales with instruction count — an 85-byte PTX has virtually nothing to compile. The claimed 3.3x JIT cost multiplier (`ptx-vs-cubin-results.md`) is therefore a **lower bound** on real-world JIT overhead, but the results markdown claims exactly the opposite: "3.3-8.2x multiplier on a trivial null kernel is a lower bound — real kernels with more PTX instructions will have higher JIT costs."

The lower-bound reasoning is correct but poorly supported. The paper does not use the PTX vs CUBIN data directly, but the supporting data should at minimum acknowledge the binary size mismatch.

**Fix:** If this data is cited anywhere, note that the CUBIN sizes differ between benchmarks (2984 vs. 4328 bytes) and explain why. Use consistently generated artifacts.

---

## Issue 7: Sample Sizes Vary Wildly Without Justification (MODERATE)

The paper mixes:
- n=10,000 warm iterations for bench_layers
- n=100 cold trials for bench_layers exec-child
- n=1,000 iterations for bench_dispatch (kdl_select)
- n=100,000 iterations for runtime_select_poc selection microbenchmark

There is no power analysis or justification for why these specific sample sizes were chosen. The cold-trial count of 100 is particularly concerning because each trial involves `execve()` and full CUDA init — the distribution is heavy-tailed (p99 at 111,269 ns vs. median 42,670 ns, a 2.6x ratio) and 100 samples is insufficient to characterize the tail reliably. A p99 estimate from 100 samples has wide confidence bounds.

The selection microbenchmark uses 100K iterations but reports only the integer-truncated per-call average (3 ns). At this resolution, the measurement is at the noise floor of `clock_gettime(CLOCK_MONOTONIC)` itself (~20-30 ns on Linux). The 3 ns figure is computed as `(total_time) / 100000`, meaning individual measurements are not captured — the per-iteration variance is unknown.

**Fix:** Either justify the sample sizes (e.g., "100 cold trials chosen because each takes ~55 ms, for a total measurement time of ~5.5 seconds") or standardize them. For the 3 ns claim, acknowledge that this is below the timer resolution and therefore represents an amortized average, not a per-call measurement. Report the total batch time and iteration count explicitly in the paper.

---

## Issue 8: No Comparison Against liboffload or Any Real Offloading Stack (MAJOR)

The paper's entire motivation is about the LLVM GPU offload stack, but **no benchmark measures liboffload**. The layer decomposition measures raw CUDA driver API calls. The selection overhead measures the libkdl prototype. The runtime_select_poc measures a custom C implementation.

The relevant comparison for an LLVM-focused paper would be:
1. End-to-end dispatch through `liboffload` (current LLVM runtime) — what does `parseOffloadBinary` + selection + `cuModuleLoadData` + `cuLaunchKernel` actually cost today?
2. The proposed `runtime_select` path — what would it cost if implemented?

Neither measurement exists. The paper hand-waves this in `layer-benchmark-results.md` Section "Overhead Analysis" by proposing subtraction: "liboffload dispatch overhead: measured_total - 4.26 us." But the measured_total is never obtained.

The paper acknowledges "Contributions (1) and (2) are concrete; (3) motivates them" — but even Contribution 2 (the layer decomposition) is of raw CUDA APIs, not of the LLVM offloading stack. A reviewer will ask: "You claim to measure 'the LLVM GPU dispatch stack' but you measured CUDA driver calls. Where is liboffload in your measurements?"

**Fix:** Either (a) build and benchmark through liboffload and report the actual overhead, or (b) explicitly acknowledge this gap in the paper: "We measure the CUDA driver API baseline; liboffload overhead on top of these layers is not measured in this work and remains future work." Do not claim you measured "the LLVM GPU dispatch stack" when you measured the CUDA driver API.

---

## Issue 9: Flame Graph Visualization is Not Actually a Flame Graph (MINOR)

The TikZ figure (lines 226-247) is labeled a "flame graph" but is actually a stacked bar chart with bars of different widths. A flame graph has a specific meaning in the performance engineering community (Brendan Gregg, 2012): it visualizes stack traces with width proportional to sample count. What the paper shows is proportional to latency fraction, not stack depth or call frequency. The bars are also stacked vertically in the wrong order (widest at bottom in a flame graph represents the deepest/most-called frame, not the most expensive).

This is a terminology issue, not a data issue. But an audience at EuroLLVM will include people who know what flame graphs are.

**Fix:** Call it a "latency decomposition chart" or "proportional latency bar chart." Reserve "flame graph" for actual stack-sampled visualizations.

---

## Issue 10: Frequency Governor Not Controlled — Measurements Vulnerable to DVFS (MODERATE)

`pinned-benchmark-results.md` line 7 states: "Governor: Default (no root for `cpupower frequency-set -g performance`)." This means the CPU was running with dynamic frequency scaling enabled during all measurements. For sub-microsecond measurements (Layer 1 at 30 ns, Layer 3 at 60 ns, selection at 3 ns), DVFS can introduce measurement artifacts:

- The CPU may be in a low-power state at the start of a measurement burst, producing slower initial iterations.
- Turbo boost behavior varies with thermal state, meaning the effective clock changes across the 10,000-iteration run.
- The 100 warmup iterations may not be sufficient to stabilize the CPU frequency.

The pinned data shows Layer 1 (cuDeviceGet) went from 50 ns unpinned to 30 ns pinned — a 40% change. Some of this could be DVFS-related rather than migration-related. Without controlling the governor, you cannot distinguish these effects.

**Fix:** Either run with `performance` governor (acknowledge if you cannot), or add a disclaimer: "Measurements were taken with the default (ondemand/schedutil) frequency governor. Sub-microsecond measurements may be affected by DVFS." For a poster (not a full paper), this is acceptable if disclosed.

---

## Summary

| # | Issue | Severity | Fixable before poster? |
|---|-------|----------|------------------------|
| 1 | No confidence intervals | MAJOR | Yes — data exists in pinned-benchmark-results.md |
| 2 | Uses unpinned data when pinned data exists | MAJOR | Yes — swap numbers |
| 3 | "Selection overhead" conflates kdl and OffloadBinary | MAJOR | Yes — clarify labeling |
| 4 | "first published" priority claim unsubstantiated | MAJOR | Yes — qualify the claim |
| 5 | Cold-path confounds exec overhead with cuModuleLoadData | MODERATE | Yes — rename table row |
| 6 | PTX vs CUBIN uses different binary sizes | MODERATE | Yes — note the discrepancy |
| 7 | Sample sizes vary without justification | MODERATE | Yes — add justification |
| 8 | No actual liboffload measurement | MAJOR | No — requires liboffload build |
| 9 | "Flame graph" is not a flame graph | MINOR | Yes — rename |
| 10 | Frequency governor uncontrolled | MODERATE | Partially — disclose limitation |

**Bottom line:** The benchmarking methodology is more careful than average (execve for cold isolation, CPU pinning, cross-run statistics), but the paper fails to surface its own best data. The pinned 3-run cross-run statistics are solid work — they just are not in the paper. The biggest conceptual gap is Issue 8: no actual LLVM offloading stack measurement. The "first published" claim (Issue 4) is the most likely to draw fire from reviewers.
