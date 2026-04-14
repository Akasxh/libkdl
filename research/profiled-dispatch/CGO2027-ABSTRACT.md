# Profiled Adaptive Dispatch for Cross-Vendor GPU Kernel Selection

**Target venue:** CGO 2027 (International Symposium on Code Generation and Optimization)
**Authors:** Akash Kumaran (IIT Patna)

---

## Abstract

Heterogeneous GPU deployments -- mixing NVIDIA, AMD, and CPU targets -- require
runtime selection among pre-compiled kernel variants. Existing approaches use
static cost models (roofline bounds, heuristic priority rankings) to estimate
which variant is fastest, but these models systematically mispredict when
kernels operate near the compute-memory ridge point, when same-architecture
variants differ in tiling strategy, or when runtime conditions (thermal state,
concurrent workloads) alter effective hardware capability.

We observe that GPU kernel variant selection is a *degenerate* stochastic
multi-armed bandit: the action space is small (N < 10 viable variants after
capability filtering), rewards are near-deterministic (coefficient of variation
< 5% for GPU execution times), and contexts are discrete and cacheable. Under
these structural properties, exhaustive exploration is optimal: the cumulative
regret of an explore-then-commit strategy is O(N^2) -- constant, independent
of the dispatch horizon T. This is in contrast to the O(sqrt(KT)) minimax
regret of the general bandit setting, and renders sophisticated algorithms
(UCB1, Thompson Sampling) unnecessary.

We present a profiled adaptive dispatch algorithm that extends libkdl, a
lightweight Kernel Dynamic Linker for MLIR-compiled multi-target GPU binaries.
The algorithm operates in three phases: (1) cold start using the roofline cost
model as a Bayesian prior to order exploration, (2) epsilon-greedy exploration
with GPU event-based timing (cudaEventElapsedTime / hipEventElapsedTime) and
online variance estimation via Welford's algorithm, and (3) exploitation of
the empirically optimal variant with periodic re-validation. The context key
(kernel_name, shape_hash, device_id) enables independent convergence per
dispatch scenario, and the roofline prior reduces exploration cost by
correctly ordering candidates for clearly compute-bound or memory-bound
kernels.

We evaluate on three workload classes: GEMM (compute-bound, N=256 to 4096),
softmax (memory-bound, sequence lengths 128 to 32000), and fused attention
(mixed regime, a workload where static cost models are known to fail). On an
NVIDIA GTX 1650, the profiled dispatcher converges in fewer than 30 kernel
invocations per context, achieving selection accuracy exceeding 99%
post-convergence. The exploration phase adds approximately 6 microseconds of
overhead per invocation (dominated by GPU event creation), while the
exploitation phase adds zero overhead beyond the existing 3-6 nanosecond
dispatch cache lookup. For near-ridge-point workloads where the roofline model
misranks variants, profiled dispatch achieves up to 1.3x speedup over
roofline-only selection. Cumulative regret saturates after the exploration
phase, confirming the O(N^2) theoretical bound.

The profiled dispatch extension integrates with libkdl's existing
architecture -- a ~500 LOC C library that discovers GPU devices via dlopen
(no link-time vendor dependencies), loads Multi-Target Bundle files containing
pre-compiled kernel variants for NVPTX, AMDGCN, and x86 targets, and routes
invocations through a vendor-agnostic dispatch table. The profiling machinery
adds approximately 200 lines of C. The full system is designed to slot beneath
existing ML compilation stacks (torch.compile, ONNX Runtime) as a thin
runtime layer, complementing rather than replacing full-stack solutions like
IREE.

Our results demonstrate that for the specific structure of GPU kernel dispatch
-- few arms, low noise, cacheable contexts -- measurement-based selection
is both theoretically optimal and practically cheap, resolving a limitation
that static cost models cannot address regardless of their sophistication.

---

## Keywords

GPU kernel dispatch, multi-armed bandits, runtime profiling, MLIR,
heterogeneous computing, cross-vendor portability, roofline model,
cost-model-driven compilation
