# Evaluation Plan: Profiled Adaptive Dispatch

**Authors:** Akash (IIT Patna)
**Context:** Extension to the LLVM Dublin 2026 poster on libkdl
**Date:** 2026-04-14

---

## 1. Research Questions

**RQ1 (Convergence):** How many kernel invocations does the profiled dispatcher
require to identify the optimal variant?

**RQ2 (Regret):** What is the cumulative regret of the profiled dispatcher
compared to an oracle that always selects the best variant?

**RQ3 (Overhead):** What is the per-invocation overhead of profiling (event
creation, timing, statistics update) compared to unprofiled dispatch?

**RQ4 (Accuracy):** Does profiled dispatch outperform roofline-only dispatch,
and by how much?

**RQ5 (Robustness):** How does the profiled dispatcher behave under workload
shifts, thermal throttling, and concurrent kernel interference?

---

## 2. Workloads

### 2.1 GEMM (Compute-Bound)

**Kernel:** Matrix multiplication C = A * B, single-precision (FP32).
**Arithmetic intensity:** AI = 2*N / (3 * sizeof(float)) = N/6 FLOP/byte.
For N=1024: AI ~ 170 >> ridge point of any GPU. Clearly compute-bound.

**Shapes:**
| Label | M | N | K | GFLOP | AI (FLOP/byte) | Regime |
|-------|---|---|---|-------|-----------------|--------|
| small | 256 | 256 | 256 | 0.034 | ~43 | Compute |
| medium | 1024 | 1024 | 1024 | 2.15 | ~170 | Compute |
| large | 4096 | 4096 | 4096 | 137.4 | ~682 | Compute |
| tall-skinny | 4096 | 64 | 4096 | 2.15 | ~10 | Memory (on some devices) |

**Variants to compile:**
1. Naive triple-loop (CPU baseline)
2. Tiled 16x16 (GPU, generic)
3. Tiled 32x32 with shared memory (GPU, requires smem >= 8KB)
4. Vectorized 64x64 with register blocking (GPU, requires sm_70+)
5. cuBLAS SGEMM (oracle upper bound, not a dispatched variant)

**Why this workload:** GEMM is the canonical compute-bound kernel. It tests
whether the profiler can distinguish between tiling strategies that have
identical roofline estimates but different real performance.

### 2.2 Softmax (Memory-Bound)

**Kernel:** Row-wise softmax: y_i = exp(x_i - max(x)) / sum(exp(x - max(x))).
**Arithmetic intensity:** ~5 FLOP per element, 2 reads + 1 write per element,
AI ~ 5 / (3 * 4) ~ 0.42 FLOP/byte. Deeply memory-bound.

**Shapes:**
| Label | Rows | Cols | Elements | Regime |
|-------|------|------|----------|--------|
| bert-base | 512 | 768 | 393K | Memory |
| bert-large | 512 | 1024 | 524K | Memory |
| llm-vocab | 1 | 32000 | 32K | Latency |
| batched | 64 | 32000 | 2M | Memory |

**Variants to compile:**
1. Per-row sequential (CPU)
2. One-block-per-row (GPU, simple)
3. Warp-shuffle reduction (GPU, requires warp-level primitives)
4. Multi-pass online softmax (GPU, numerically stable, higher instruction count)

**Why this workload:** Softmax is memory-bound, so the roofline correctly
identifies bandwidth as the bottleneck. The profiler's value here is in
distinguishing variants that trade numerical stability for throughput,
and in identifying the optimal block size for different sequence lengths.

### 2.3 Flash Attention (Mixed)

**Kernel:** Fused attention: softmax(QK^T / sqrt(d)) * V with tiled
accumulation to avoid materializing the full attention matrix.

**Arithmetic intensity:** Varies with sequence length. For seq_len=512, d=64:
the fused kernel achieves AI ~ 20-50 depending on tiling. The unfused version
(separate GEMM + softmax + GEMM) is memory-bound due to intermediate
materialization.

**Shapes:**
| Label | batch | heads | seq_len | d | Regime |
|-------|-------|-------|---------|---|--------|
| short | 8 | 12 | 128 | 64 | Mixed |
| medium | 8 | 12 | 512 | 64 | Compute |
| long | 1 | 12 | 2048 | 64 | Compute |

**Variants to compile:**
1. Unfused: GEMM + softmax + GEMM (3 separate kernels)
2. Fused tile-64 (single kernel, tile size 64)
3. Fused tile-128 (single kernel, tile size 128, requires more smem)

**Why this workload:** Attention is the workload where variant selection
matters most in practice. The fused vs unfused choice and the tile size
are not captured by the roofline model; profiling is essential.

---

## 3. Metrics

### 3.1 Primary Metrics

**Time-to-convergence (TTC):** Number of kernel invocations until the profiled
dispatcher enters the EXPLOIT phase for a given context. Measured per-context.
Lower is better. Theoretical bound: N * N_warmup (Section 2.2 of THEORY.md).

**Cumulative regret (CR):** Sum of excess execution time over the oracle-optimal
variant across all invocations:

```
CR(t) = sum_{i=1}^{t} [ T_{selected}(i) - T_{oracle}(i) ]
```

Measured in microseconds. The oracle is determined by exhaustive profiling
of all variants (100 invocations each) before the experiment.

**Profiling overhead (PO):** Per-invocation wall-clock overhead introduced by
the profiling infrastructure (event creation/destruction, Welford's update,
convergence check):

```
PO = T_{profiled_dispatch} - T_{unprofiled_dispatch}
```

Measured in nanoseconds using `clock_gettime(CLOCK_MONOTONIC)`.

### 3.2 Secondary Metrics

**Selection accuracy:** Fraction of invocations where the profiled dispatcher
selects the oracle-optimal variant. Measured post-convergence.

**Speedup over roofline-only:** Ratio of mean kernel time under roofline-only
dispatch to mean kernel time under profiled dispatch. Values > 1.0 indicate
profiled dispatch is better.

**Noise-to-gap ratio (NGR):** sigma / Delta_min for each context, where sigma
is the standard deviation of the optimal variant's timing and Delta_min is
the gap to the second-best variant. NGR < 1 predicts easy convergence;
NGR > 1 predicts the profiler will struggle.

---

## 4. Baselines

### 4.1 Roofline-Only (Current libkdl)

The existing `kdl_select_kernel` with `kdl_estimate_cost_weighted()`. This is
the current production path. Expected performance: correct ranking for clearly
compute-bound or memory-bound kernels; possibly wrong for mixed workloads or
when distinguishing same-architecture variants.

### 4.2 First-Wins

Select the first contract-matching variant in the routing table. No cost model,
no profiling. This is the degenerate baseline: if the bundle's variant ordering
happens to be correct, it performs optimally; otherwise, it performs poorly.

### 4.3 cuBLAS Heuristic (93% Line)

For GEMM workloads, the cuBLAS trained recommender system achieves 93% of
optimal geometric mean performance across all problem sizes (documented in
NVIDIA cuBLAS documentation, also noted in our literature review at
`literature/production-ml-dispatch.md`). This is the industry standard for
vendor-specific runtime dispatch. Our profiled dispatcher should match or
exceed 93% on the problems it handles, since it has access to actual timing
data rather than a trained model.

### 4.4 Oracle (Exhaustive Profiling)

Run every variant 100 times on every context, select the one with the lowest
mean. This is the theoretical optimum that the profiled dispatcher should
converge to. It is not a practical baseline (too expensive for production)
but defines the ceiling.

### 4.5 Random Selection

Select a variant uniformly at random for each invocation. Expected performance:
mean of all variant means. This is the floor; any reasonable dispatcher must
beat it.

---

## 5. Experimental Protocol

### 5.1 Hardware

**Primary (available):**
- **GPU:** NVIDIA GeForce GTX 1650 (Turing TU117, 896 CUDA cores, 4GB GDDR5)
  - Peak FP32: 2.98 TFLOPS
  - Memory BW: 192 GB/s (128-bit GDDR5)
  - Ridge point: ~15.5 FLOP/byte
  - Compute capability: sm_75
- **CPU:** Intel Core i7-10750H (6 cores / 12 threads, 2.6 GHz base)
- **Driver:** CUDA 12.x
- **OS:** Linux 6.x

**Extended (if access is obtained):**
- NVIDIA A100 SXM4 80GB (sm_80, 19.5 TFLOPS, 2 TB/s)
- AMD MI300X (gfx942, 163.4 TFLOPS, 5.3 TB/s)
- CPU-only node (AMD EPYC 9654, 96 cores)

### 5.2 Measurement Protocol

1. **System preparation:**
   - Set GPU to persistence mode (`nvidia-smi -pm 1`)
   - Lock GPU clocks to base frequency (`nvidia-smi -lgc <base_freq>`)
     to reduce variance from GPU Boost
   - Disable CPU frequency scaling (`cpupower frequency-set -g performance`)
   - Close unnecessary background processes

2. **Warmup:**
   - 10 invocations of each kernel on each device, discarded
   - Ensures instruction caches, TLBs, and memory pages are warmed

3. **Oracle establishment:**
   - Run each variant 100 times per context
   - Record mean, stddev, min, max, p50, p99
   - Select oracle-optimal variant per context

4. **Profiled dispatch experiment:**
   - Reset all profile caches
   - Run the workload for T = 10000 invocations per context
   - Record per-invocation: selected variant, execution time, dispatch phase
   - Repeat 5 times with different random seeds (for epsilon-greedy randomness)

5. **Baseline experiments:**
   - Roofline-only: same protocol, T = 10000 invocations
   - First-wins: same protocol
   - Random: same protocol, 5 seeds
   - cuBLAS (GEMM only): `cublasSgemm` with default heuristic

### 5.3 Statistical Analysis

- Report mean and 95% confidence intervals across 5 runs
- Use Welch's t-test to determine if profiled dispatch significantly
  outperforms roofline-only dispatch
- Report effect size (Cohen's d) for the improvement
- Use Kruskal-Wallis test for non-parametric comparison when distributions
  are non-normal

---

## 6. Expected Results

### 6.1 Convergence (RQ1)

**Prediction:** TTC = 15-30 invocations for most contexts (N=3-5 variants,
N_warmup=5). For single-variant contexts (only one viable variant after
contract matching), TTC = 1 (immediate convergence).

**Interesting case:** The tall-skinny GEMM (M=4096, N=64, K=4096) where
the roofline may misrank variants because AI is near the ridge point.
The profiler should take longer to converge here due to smaller gaps.

### 6.2 Regret (RQ2)

**Prediction:** Cumulative regret saturates (flattens) after TTC invocations.
Total regret ~ 14 * mu* (from THEORY.md Section 2.3). For a 1ms kernel,
this is ~14ms total, amortized to zero over 10000 invocations.

**Visualization:** Plot CR(t) vs t for each baseline. The profiled dispatcher's
curve should flatten while roofline-only stays flat (but at a higher level
if the roofline is wrong) and random grows linearly.

### 6.3 Overhead (RQ3)

**Prediction:** Profiling overhead during exploration ~ 5-10us per invocation
(dominated by cuEventCreate/Destroy). During exploitation: 0ns additional
overhead (no events, no timing). The 5-10us exploration overhead is < 1%
of a 1ms kernel.

**Breakdown expected:**
| Component | Time | When |
|-----------|------|------|
| Context hash | ~10ns | Always |
| Profile cache lookup | ~5ns | Always |
| cuEventCreate (x2) | ~2us | Exploration only |
| cuEventRecord (x2) | ~0.5us | Exploration only |
| cuEventSynchronize | ~1us | Exploration only |
| cuEventDestroy (x2) | ~2us | Exploration only |
| Welford update | ~5ns | Exploration only |
| Convergence check | ~50ns | Exploration only |
| **Total (exploration)** | **~6us** | |
| **Total (exploitation)** | **~15ns** | |

### 6.4 Accuracy (RQ4)

**Prediction:** Profiled dispatch achieves >= 99% selection accuracy
post-convergence (the rare 1% failure comes from contexts where two variants
are within the noise floor of each other, making the "correct" choice
ambiguous and irrelevant).

**Speedup over roofline-only:** Varies by workload.
- GEMM (all sizes except tall-skinny): 1.0x (roofline is already correct)
- GEMM (tall-skinny): 1.1-1.3x (roofline misranks near ridge point)
- Softmax: 1.0-1.05x (memory-bound, roofline is correct)
- Attention (fused vs unfused): 1.3-2.0x (roofline cannot distinguish fusion)

---

## 7. Ablation Studies

### 7.1 N_WARMUP Sensitivity

Vary N_WARMUP in {1, 2, 3, 5, 10, 20} and measure TTC and selection accuracy.
**Hypothesis:** N_WARMUP = 3 is sufficient for most contexts; N_WARMUP = 5
provides a safety margin; N_WARMUP > 5 adds no benefit.

### 7.2 Roofline Prior Ablation

Compare three exploration orderings:
1. Roofline-ordered (our approach)
2. Random-ordered
3. Reverse-roofline-ordered (worst case)

**Hypothesis:** Roofline ordering reduces exploration regret by 30-50% compared
to random ordering, because the roofline's top pick is correct >60% of the time.

### 7.3 Convergence Criterion Ablation

Compare convergence criteria:
1. Statistical separation (our approach, CONFIDENCE_THRESHOLD = 2.0)
2. Fixed exploration budget (N * N_warmup invocations, then stop)
3. Annealing (never fully stop, epsilon -> 0 asymptotically)

**Hypothesis:** Statistical separation converges faster than fixed budget
when variant gaps are large, and produces better results than annealing
because annealing wastes invocations on unnecessary continued exploration.

---

## 8. Presentation Plan

### 8.1 Poster Figures

**Figure 1: Regret curves.** Cumulative regret vs invocation count for all
baselines on the medium GEMM workload. Shows the profiled dispatcher's
curve flattening while roofline-only is flat-but-higher and random grows
linearly.

**Figure 2: Convergence scatter.** TTC vs noise-to-gap ratio for all
(workload, context) pairs. Shows the correlation between NGR and convergence
difficulty.

**Figure 3: Overhead breakdown.** Stacked bar chart of profiling overhead
components during exploration vs exploitation phases.

**Figure 4: Speedup heatmap.** Workload x metric heatmap showing profiled
dispatch speedup over roofline-only for each workload.

### 8.2 Key Numbers for the Poster

Target one-line claims to validate:
- "Converges in <30 kernel invocations"
- "Constant regret: O(N^2) not O(sqrt(T))"
- "Zero overhead after convergence (same 3-6ns cache lookup)"
- "Up to 1.3x over roofline-only for near-ridge-point workloads"
