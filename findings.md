# Findings — Bridging Runtime Gaps in LLVM: Vendor-Agnostic Dispatch for ML Kernels

*EuroLLVM Dublin 2026 — S. Akash, IIT Patna*

## Core Research Question

Can we build a lightweight, adaptive runtime dispatch layer for LLVM's GPU offload stack that selects the best pre-compiled kernel variant using online profiling, and is it provably near-optimal?

**Answer: Yes.** We formalize kernel variant selection as a degenerate multi-armed bandit and show that exhaustive exploration followed by permanent exploitation achieves O(N²) constant regret — near-oracle performance with negligible overhead.

---

## Phase 1: Dispatch Stack Measurement

### The Gap
MLIR compiles one `gpu.module` to 3+ GPU vendors (NVIDIA, AMD, Intel). The `OffloadBinary` carries N device images. But at runtime, `liboffload` picks the **first compatible image** (PR #186088). No metadata vocabulary, no measurement, no ranking.

### Key Measurements (GTX 1650 sm_75, CUDA 13.1)

| Layer | Median | Share of Cold Path |
|-------|--------|--------------------|
| cuModuleLoadData (cold) | 36.0 µs | 89.6% |
| cuModuleGetFunction | 63 ns | 0.2% |
| cuLaunchKernel | 1.65 µs | 4.1% |
| cuStreamSynchronize | 2.45 µs | 6.1% |
| **Selection overhead** | **3-6 ns** | **< 0.02%** |

**Insight:** Module loading dominates at 90%. Selection is essentially free. The question becomes: **what information should drive the selection?**

### Metadata Vocabulary
5 new keys for OffloadBinary's StringMap: `min_sm`, `min_gfx`, `requires_features`, `variant_priority`, `variant_tag`. Backward-compatible, ~30 LOC patch.

---

## Phase 2: Profiled Adaptive Dispatch (MAB Formulation)

### The Formulation
- **Arms:** N pre-compiled kernel variants (typically N < 10)
- **Reward:** Negative execution time
- **Context:** (kernel_name, shape_hash, device_id) — cacheable key
- **Objective:** Minimize cumulative regret

### Why This Is a Degenerate Bandit
1. **Small action space:** N < 10
2. **Near-deterministic:** CV < 5%
3. **Cacheable contexts:** Same tuple → same answer forever

**Result:** O(N²) constant regret. UCB1/Thompson Sampling unnecessary.

### Experimental Results (5-Scenario Suite)

| Scenario | Finding |
|----------|---------|
| Near-identical (12% spread, 8% noise) | Converges despite noise |
| Context-dependent | Different shapes → different optima |
| Scaling (N=2→64) | 6/25/48/73/85 dispatches |
| Non-stationary | Static fails; re-validation needed |
| **Comparison** | **Profiled = 86% of oracle, 7.3x < random** |

---

## Prototype: 5,157 + 664 + 238 LOC

## Open Questions
1. Real CUDA kernels (not simulated)
2. Non-stationarity handling (sliding window)
3. Multi-GPU cross-device dispatch
4. vLLM integration
