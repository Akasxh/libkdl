# Profiled Adaptive Dispatch for Cross-Vendor GPU Kernel Selection

## Problem Statement

**Authors:** Akash (IIT Patna)
**Context:** Extension to the LLVM Dublin 2026 poster on libkdl (Kernel Dynamic Linker)
**Date:** 2026-04-14

---

## 1. The Selection Problem

libkdl currently selects among N pre-compiled kernel variants using a roofline cost
model: given a kernel's arithmetic intensity and a device's peak FLOPS and bandwidth,
it estimates execution time and picks the cheapest variant. This is a *static*
cost model -- it produces the same ranking every time for a given (kernel, device) pair.

The roofline model is wrong in predictable ways:

1. **It overpredicts.** The roofline assumes perfect occupancy, perfect cache reuse,
   and zero bank conflicts. Real kernels achieve 30-70% of the roofline bound
   (Volkov and Demmel, SC08). The gap varies per variant.

2. **It cannot distinguish same-target variants.** Two sm_80 cubins for the same
   GEMM -- one tile-based, one vectorized -- have identical roofline estimates
   but may differ by 2x in practice. The roofline model has no information to
   rank them.

3. **It ignores microarchitectural effects.** L2 cache partitioning, warp scheduler
   stalls, register pressure, shared memory bank conflicts, and TLB miss rates
   all affect real performance but are invisible to the roofline.

4. **It cannot adapt.** When the system is under load (concurrent kernels, memory
   pressure, thermal throttling), the optimal variant may change. The roofline model
   is oblivious to runtime state.

**The question:** can we replace or augment the roofline estimate with actual
measurement, converging to the true best variant through online profiling?

---

## 2. Formal Problem Definition

### 2.1 Multi-Armed Bandit Formulation

We model kernel variant selection as a **contextual multi-armed bandit**:

- **Arms:** N kernel variants {v_1, ..., v_N} compiled for the target device.
  Typically N in {2, ..., 8}; one kernel rarely has more than 8 viable variants
  after contract matching filters incompatible targets.

- **Context:** A tuple c = (kernel_name, shape_hash, device_id) that identifies
  the dispatch scenario. Contexts are discrete and cacheable: the same GEMM with
  the same (M, N, K) on the same GPU always produces the same context.

- **Reward:** Negative execution time. Pulling arm v_i in context c yields
  reward r_i(c) = -T_i(c) + epsilon, where T_i(c) is the true execution time
  and epsilon ~ N(0, sigma^2) is measurement noise from hardware jitter.

- **Objective:** Minimize cumulative regret over T invocations:

  ```
  R(T) = sum_{t=1}^{T} [ T_{a_t}(c_t) - T_{v*}(c_t) ]
  ```

  where v* = argmin_i T_i(c) is the oracle-optimal variant for context c,
  and a_t is the arm pulled at time t.

### 2.2 Key Structural Properties

This bandit instance has properties that make it **degenerate** -- far easier
than the general case:

**Property 1: Small action space.**
N < 10. After contract matching (architecture compatibility, shared memory
requirements, VRAM constraints), the typical viable set is 2-5 variants.
For comparison, cuBLAS selects among ~16 SGEMM kernels for small matrices
and more for large ones. Our problem is an order of magnitude smaller.

**Property 2: Near-deterministic rewards.**
GPU kernel execution times are highly concentrated. For a fixed (kernel, shape, device),
the coefficient of variation is typically CV = sigma/mu < 5%.

Evidence:
- CUDA event timing on a warmed-up GPU yields sub-microsecond jitter for
  millisecond-scale kernels.
- The primary noise sources are L2 cache state (cold vs warm), concurrent
  kernel interference, and clock frequency variation (GPU Boost).
- After a warmup pass (1-2 invocations to fill caches), successive timings
  are stable to within 2-3%.
- This is in stark contrast to, e.g., web request latencies where CV > 100%
  and heavy tails dominate.

**Property 3: Discrete, cacheable contexts.**
The context space is finite and small in practice. ML workloads exhibit
a small number of distinct (kernel, shape) pairs:
- A transformer model has O(10) unique operator shapes (QKV projection,
  attention, FFN, layernorm, each with 1-3 shape variants from dynamic batching).
- Each shape is dispatched hundreds or thousands of times during inference serving.
- Once the best variant is identified for a context, the answer is valid
  indefinitely (until hardware or driver changes).

**Property 4: Informative prior.**
The roofline cost model, while imprecise, is positively correlated with true
performance. It correctly identifies the binding regime (compute vs memory bound)
and provides a reasonable initial ranking. This prior can bootstrap exploration
by ordering which variants to try first.

### 2.3 Why This Is a Degenerate Bandit

A standard K-armed bandit with sub-Gaussian rewards has minimax regret
Theta(sqrt(KT)). This bound is tight when:
- K is large relative to T
- Arm gaps are small
- Rewards are noisy

None of these conditions hold here:

1. **K is tiny (2-8).** Exhaustive exploration costs at most K * N_warmup
   samples, where N_warmup ~ 3-5 suffices for a high-confidence estimate
   given CV < 5%. Total exploration cost: 10-40 kernel invocations.

2. **Gaps are large.** Variant performance differences are typically 10-50%
   for different tiling strategies, and 2-10x across different architectures.
   With sigma/mu < 5%, the gap-to-noise ratio is 2-10, meaning a handful
   of samples suffices to distinguish the best arm.

3. **Rewards are near-deterministic.** With CV < 5%, the confidence interval
   after N_warmup = 5 samples is approximately +/- 2*sigma/sqrt(5) ~ +/- 4.5%
   of the mean. This is tight enough to resolve the typical 10-50% gap
   between variants.

**Consequence:** The explore-then-commit strategy is optimal. Spend O(N)
invocations exploring, then exploit forever. The cumulative regret is
bounded by a constant:

```
R(T) = O(N * N_warmup * Delta_max) = O(N^2)
```

where Delta_max is the maximum suboptimality gap. This is **constant regret**
-- it does not grow with T. The problem self-terminates.

For comparison:
- UCB1 achieves O(K * log(T) / Delta) regret -- logarithmic growth in T,
  which is optimal for the general case but unnecessary here.
- Thompson Sampling achieves the same asymptotic bound with better constants.
- Both are overkill: they hedge against adversarial or highly stochastic
  environments that do not arise in GPU kernel selection.

---

## 3. Why the Roofline Alone Is Insufficient

The roofline model serves as a Bayesian prior -- it tells us which arm to
pull first and gives a rough ranking. But it cannot close the loop:

| Factor | Roofline captures? | Profiling captures? |
|--------|-------------------|---------------------|
| Peak FLOPS/BW ratio | Yes | Yes (implicitly) |
| Achievable occupancy | No (assumes 100%) | Yes |
| Shared memory bank conflicts | No | Yes |
| Register spill to local memory | No | Yes |
| L2 cache hit rate | No | Yes |
| Instruction mix efficiency | No | Yes |
| Warp divergence | No | Yes |
| Concurrent kernel interference | No | Yes |
| Thermal throttle state | No | Yes |
| Driver/runtime overhead variance | No | Yes |

The roofline is a necessary starting point (it prevents exploring obviously
wrong candidates) but measurement is the only way to find the true best.

---

## 4. The Opportunity

The combination of these structural properties means profiled dispatch is:

1. **Cheap to implement.** The exploration phase adds N * N_warmup kernel
   launches to the cold-start path. For N=4, N_warmup=5, this is 20 extra
   invocations -- amortized over thousands of subsequent dispatches.

2. **Cheap at runtime.** After convergence, the profiled dispatch path is
   identical to the current cached dispatch path: a hash table lookup
   returning a function pointer. The overhead is the same 3-6ns we already
   measured for libkdl's selection cache.

3. **Strictly better than static.** The profiled dispatcher converges to the
   oracle-optimal variant (with high probability), while the roofline model
   achieves at best the cuBLAS-equivalent 93% of optimal.

4. **Composable with the roofline.** The roofline prior reduces exploration
   by ordering candidates. If the roofline ranking is already correct (which
   it often is for clear compute-bound or memory-bound kernels), the profiler
   confirms it in N_warmup samples and never explores further.

---

## 5. Scope and Assumptions

This analysis assumes:

- **Stationary performance.** A variant's execution time for a given context
  does not drift over time. This holds in practice unless thermal throttling
  or concurrent workloads change the effective hardware capability. We address
  non-stationarity through periodic re-validation (Section 4 of ALGORITHM.md).

- **Independent contexts.** The optimal variant for context c does not depend
  on what was dispatched for context c'. This holds when kernels do not
  persistently modify shared state (e.g., L2 cache pollution). For cache-sensitive
  workloads, the context could be extended to include a "preceding kernel" field.

- **Honest timing.** CUDA/HIP event elapsed time accurately reflects kernel
  execution time. This requires proper synchronization (event-based, not
  wall-clock) and warmup to eliminate cold-start effects.

- **Variant set is fixed.** The set of compiled variants does not change
  at runtime. JIT compilation of new variants is out of scope (that is a
  separate, complementary optimization).
