# Theoretical Analysis: Regret Bounds for Profiled Kernel Dispatch

**Authors:** Akash (IIT Patna)
**Context:** Extension to the LLVM Dublin 2026 poster on libkdl
**Date:** 2026-04-14

---

## 1. Setup and Notation

Let K = {v_1, ..., v_N} be the set of viable kernel variants after contract
matching, with N < 10. For a fixed context c = (kernel_name, shape_hash, device_id):

- mu_i = E[T_i(c)] is the expected execution time of variant v_i
- sigma_i^2 = Var[T_i(c)] is the variance (from hardware jitter)
- v* = argmin_i mu_i is the oracle-optimal variant
- mu* = mu_{v*} is the optimal expected time
- Delta_i = mu_i - mu* >= 0 is the suboptimality gap of arm i
- Delta_max = max_i Delta_i is the maximum gap

**Empirical parameter regime** (from GPU benchmarking literature and our
measurements on GTX 1650):

| Parameter | Typical Range | Source |
|-----------|--------------|--------|
| N (viable variants) | 2-8 | Contract matching filters; libkdl routing table |
| mu_i | 0.1ms - 50ms | GEMM 256x256 to 4096x4096 on GTX 1650 |
| CV = sigma_i/mu_i | 1-5% | CUDA event timing, warmed GPU |
| Delta_i / mu* | 10-50% | Tiling variants on same arch |
| Delta_i / mu* (cross-arch) | 2x-10x | GPU vs CPU, or different GPU gens |

---

## 2. Regret Bound for Explore-Then-Commit

### 2.1 The Strategy

**Phase 1 (Exploration):** For each of the N variants, execute it N_w times
(where N_w is the warmup/sample count per arm). Record the sample mean:

```
hat{mu}_i = (1/N_w) * sum_{j=1}^{N_w} T_i^{(j)}
```

Total exploration invocations: N * N_w.

**Phase 2 (Exploitation):** Select v_hat = argmin_i hat{mu}_i and dispatch
v_hat for all subsequent invocations of context c.

### 2.2 Probability of Correct Selection

By Hoeffding's inequality, for each arm i:

```
P(|hat{mu}_i - mu_i| > epsilon) <= 2 * exp(-2 * N_w * epsilon^2 / R^2)
```

where R is the range of the random variable (for kernel times, R ~ 2 * sigma_i
in practice, since the distribution is tightly concentrated).

For the selection to be correct, we need:

```
hat{mu}_{v*} < hat{mu}_i   for all i != v*
```

This holds with high probability when:

```
epsilon < Delta_min / 2
```

where Delta_min = min_{i: Delta_i > 0} Delta_i is the smallest nonzero gap.

Setting epsilon = Delta_min / 2 and solving for N_w:

```
N_w >= (R^2 / (2 * epsilon^2)) * ln(2N / delta)
    = (2 * R^2 / Delta_min^2) * ln(2N / delta)
```

where delta is the failure probability.

**In our regime:**
- R ~ 2 * sigma, sigma ~ 0.03 * mu (CV = 3%), so R ~ 0.06 * mu
- Delta_min ~ 0.10 * mu (10% gap, conservative)
- N = 5, delta = 0.01

```
N_w >= (2 * (0.06*mu)^2 / (0.10*mu)^2) * ln(2*5 / 0.01)
    = (2 * 0.0036 / 0.01) * ln(1000)
    = 0.72 * 6.91
    ~ 5.0
```

**Result:** N_w = 5 samples per arm suffice for 99% correct selection.
This is remarkably small, confirming the degeneracy of the problem.

### 2.3 Cumulative Regret Bound

The regret decomposes into exploration regret and exploitation regret:

**Exploration regret:**
During the N * N_w exploration phase, the expected regret is at most:

```
R_explore <= N * N_w * Delta_max
```

since each exploration pull incurs at most Delta_max regret.

**Exploitation regret:**
With probability >= 1 - delta, the exploitation phase incurs zero regret
(the correct arm was selected). With probability delta, the wrong arm is
selected, incurring at most Delta_max regret per invocation for the remaining
T - N*N_w invocations.

```
R_exploit <= delta * (T - N*N_w) * Delta_max
```

**Total expected regret:**

```
R(T) <= N * N_w * Delta_max + delta * T * Delta_max
```

Setting delta = 1/T (adapt failure probability to horizon):

```
R(T) <= N * N_w * Delta_max + Delta_max
     = O(N * N_w * Delta_max)
     = O(N^2 * Delta_max)        [since N_w = O(N) in the worst case]
```

This is **constant in T**. The regret does not grow with the number of
invocations. After the exploration phase, the algorithm performs optimally.

More precisely, with the parameters above (N=5, N_w=5, Delta_max ~ 0.5*mu*):

```
R(T) <= 5 * 5 * 0.5 * mu* + mu*
     = 13.5 * mu*
```

That is, the total lifetime regret is equivalent to **~14 extra kernel
invocations at the optimal rate**. For a GEMM running in 1ms, this is 14ms
of total regret, amortized over an arbitrarily long serving session.

---

## 3. Comparison to Standard Bandit Algorithms

### 3.1 UCB1 (Auer et al., 2002)

UCB1 selects the arm minimizing:

```
hat{mu}_i - sqrt(2 * ln(t) / n_i)
```

where n_i is the number of times arm i has been pulled and t is the current
round.

**Regret bound:** O(K * ln(T) / Delta_min)

**Why it is overkill here:**
- UCB1 is designed for settings where the optimal arm is unknown and the
  algorithm must balance exploration and exploitation indefinitely.
- The ln(T) factor means UCB1 never fully stops exploring. At round t=10^6,
  it still occasionally pulls suboptimal arms to tighten confidence bounds.
- In our setting, after N_w = 5 samples per arm, we have 99%+ confidence
  in the correct arm. Continued exploration wastes kernel invocations.
- UCB1's strength is in adversarial or non-stationary environments with
  many arms and small gaps. None of these apply to GPU kernel selection.

**Concrete comparison (N=5, T=10^6, Delta_min=0.1*mu):**
- UCB1 regret: O(5 * ln(10^6) / (0.1*mu)) = O(5 * 13.8 / 0.1) * mu = ~690 * mu
- Explore-then-commit: O(25 * 0.5) * mu = ~13 * mu
- Explore-then-commit wins by **50x** because it exploits the structure.

### 3.2 Thompson Sampling

Thompson Sampling maintains a posterior distribution over each arm's mean
and samples from the posteriors to select arms.

**Regret bound:** O(K * ln(T) / Delta_min), same order as UCB1 but with
better empirical constants (Agrawal and Goyal, 2012).

**Why it is overkill here:**
- Thompson Sampling's advantage over UCB1 is primarily in the constants:
  it explores more efficiently by exploiting posterior concentration.
- But in our regime, even the crude explore-then-commit already converges
  in 25 total samples. There is no room for Thompson Sampling to improve.
- The Bayesian machinery (maintaining and sampling from posterior distributions)
  adds implementation complexity with no benefit.
- Thompson Sampling is most valuable when arms have similar means and the
  algorithm must carefully distinguish them. Our arms differ by 10-50%.

### 3.3 Epsilon-Greedy

Epsilon-greedy selects a random arm with probability epsilon and the
empirically best arm with probability 1 - epsilon.

**Regret bound:** O(K * T * epsilon) for fixed epsilon; O(K * T^{2/3}) for
optimally decaying epsilon.

**Why we use it anyway (in modified form):**
- Epsilon-greedy with a *terminating* epsilon schedule (epsilon -> 0 after
  convergence) is equivalent to explore-then-commit.
- It is trivial to implement: a random number check and a branch.
- The "epsilon-greedy with convergence detection" variant naturally maps
  to our three-phase algorithm (cold start, exploration, exploitation).
- It does not require confidence bound computation (UCB1) or posterior
  sampling (Thompson), reducing the code path in a latency-critical
  dispatch loop.

### 3.4 Summary Table

| Algorithm | Regret R(T) | Stops exploring? | Implementation cost | Fit for our problem |
|-----------|-------------|------------------|--------------------|--------------------|
| Explore-then-commit | **O(N^2)** | **Yes** | Trivial | **Optimal** |
| Epsilon-greedy (decaying) | O(N * T^{2/3}) | Eventually | Trivial | Acceptable (with termination) |
| UCB1 | O(N * ln(T)) | No | Moderate | Overkill |
| Thompson Sampling | O(N * ln(T)) | No | High (posteriors) | Overkill |
| Roofline only (no profiling) | O(T * Delta_roofline) | N/A | Zero | **Unbounded** if model is wrong |

The critical insight: roofline-only dispatch has *linear* regret if the model
is miscalibrated (it picks the wrong variant every time). All profiling-based
approaches have at most *logarithmic* regret. Explore-then-commit achieves
*constant* regret by exploiting the problem structure.

---

## 4. The Roofline as Bayesian Prior

The roofline cost model is not useless -- it is an informative prior that
reduces the exploration cost.

### 4.1 Prior-Guided Exploration

Instead of exploring arms in arbitrary order, we explore in roofline-rank
order (cheapest estimated first). This has two benefits:

1. **Early termination.** If the roofline's top pick is confirmed by profiling
   (its measured time is within the noise floor of the best possible), we can
   skip exploring the remaining arms entirely.

2. **Reduced regret during exploration.** By trying likely-good arms first,
   the regret accumulated during exploration is lower than random-order
   exploration. The expected exploration regret becomes:

   ```
   R_explore <= sum_{i=1}^{N} N_w * Delta_{pi(i)}
   ```

   where pi is the roofline-induced ordering. If the roofline ranking is
   correct, Delta_{pi(1)} = 0 and the sum is dominated by the first few
   suboptimal arms tried.

### 4.2 When the Prior Is Correct

For clearly compute-bound kernels (large GEMM, AI >> ridge point) or clearly
memory-bound kernels (elementwise ops, AI << ridge point), the roofline model
correctly identifies the binding regime and ranks variants appropriately.

In these cases, profiling serves as *confirmation*, not *discovery*. The
exploration phase completes in N_w samples (just profiling the top-ranked
variant to confirm it) rather than N * N_w.

### 4.3 When the Prior Is Wrong

The roofline model fails when:
- The kernel is near the ridge point (mixed compute/memory)
- Two variants have similar roofline estimates but different real performance
  (e.g., different tiling strategies for the same architecture)
- The efficiency factor is miscalibrated (e.g., the 70% NVIDIA / 50% AMD
  heuristic in libkdl's current cost model)

In these cases, profiling discovers the true optimum where the model cannot.
The profiler never performs worse than the roofline, and often performs
significantly better.

### 4.4 Formal Bayesian Interpretation

Let pi_0 be the prior distribution over arm orderings induced by the roofline
model. In the simplest case, pi_0 assigns probability 1 to the roofline
ranking. A Bayesian explore-then-commit algorithm would:

1. Sample arms in prior-rank order
2. Update posterior over arm means using observed times
3. Stop when the posterior probability of the current best exceeding any
   other arm drops below threshold delta

This is formally a **best-arm identification** problem with an informative
prior. The sample complexity for (epsilon, delta)-PAC identification with
a correct prior is:

```
N_total = O(N_w * H_prior)
```

where H_prior <= N is the effective number of arms the prior cannot
distinguish. When the prior is fully correct, H_prior = 1 and we need
only N_w samples total. When the prior is uninformative, H_prior = N
and we recover the full exploration cost.

---

## 5. Convergence Guarantees

### 5.1 Per-Context Convergence

For a single context c with N viable variants:

- **Convergence time:** N * N_w invocations of context c (worst case).
  With N=5, N_w=5: 25 invocations.
- **Convergence quality:** With probability >= 1 - delta, the selected
  variant is within epsilon of optimal.
- **Post-convergence overhead:** Zero additional profiling overhead.
  Dispatch reduces to the existing 3-6ns cache lookup.

### 5.2 System-Wide Convergence

For a workload with C distinct contexts (e.g., C=15 for a transformer model):

- **Total exploration cost:** C * N * N_w invocations distributed across
  all contexts. With C=15, N=5, N_w=5: 375 invocations total.
- **Wall-clock exploration time:** For 1ms average kernel time:
  375 * 1ms = 375ms. The system converges in under half a second.
- **Amortization:** For a serving workload running for hours, the 375ms
  exploration cost is negligible (< 0.01% of a 1-hour session).

### 5.3 Non-Stationarity Handling

If hardware conditions change (thermal throttle, concurrent workload shift),
the profiled variant may become suboptimal. We handle this with periodic
re-validation:

- Every R invocations (e.g., R = 10000), re-profile the current best
  variant and one random alternative.
- If the alternative outperforms the current best by more than 2*sigma,
  trigger a full re-exploration of the context.
- Expected cost: 2 / R additional profiling overhead per invocation.
  With R = 10000 and 1ms kernels: 0.2us amortized overhead.
