# Bridging Runtime Gaps in LLVM: Vendor-Agnostic Dispatch for ML Kernels

**EuroLLVM Dublin 2026 — Poster Session — S. Akash, IIT Patna**
**Repo:** https://github.com/Akasxh/libkdl

---

## The Problem

MLIR compiles one `gpu.module` to NVIDIA + AMD + Intel GPUs via `gpu-module-to-binary`. The result is packed into an `OffloadBinary` (magic `0x10FF10AD`) with N device images. But at runtime, `liboffload` (PR #186088) picks the **first compatible image** and stops. No ranking, no metadata, no intelligence.

5 upstream signals confirm the gap: PR #148286 (XeVM), PR #186088 (first-wins), PR #185663 (isMetadataCompatible with no policy), Issue #75356 (Chapel team asking since 2023), RFC #88170 (policy slot empty).

## Phase 1: Dispatch Stack Measurement

First published per-layer latency breakdown of LLVM's GPU dispatch path.

| Layer | Median | Share |
|-------|--------|-------|
| cuModuleLoadData (cold) | **36.0 µs** | **89.6%** |
| cuModuleLoadData (warm) | 9.6 µs | — |
| cuModuleGetFunction | 63 ns | 0.2% |
| cuLaunchKernel | 1.65 µs | 4.1% |
| cuStreamSynchronize | 2.45 µs | 6.1% |
| **Selection overhead** | **3–6 ns** | **< 0.02%** |

Hardware: GTX 1650 sm_75, CUDA 13.1, null kernel CUBIN. 100 cold trials (exec-child isolation), 10K warm, 3-run CPU-pinned medians.

**Key insight:** Module loading dominates at 90%. Selection at 3–6 ns is essentially free — faster than an L2 cache access.

## Metadata Vocabulary (5 new keys)

| Key | Tier | Purpose |
|-----|------|---------|
| `min_sm` | MUST | Min CUDA compute capability |
| `min_gfx` | MUST | Min AMD GFX version |
| `requires_features` | MUST | Named capability tokens (tensor_core, bf16) |
| `variant_priority` | MAY | Higher = preferred among compatible |
| `variant_tag` | MAY | Human label: generic, optimized, fallback |

Backward-compatible: fits in OffloadBinary's existing StringMap, ~30 LOC patch to `isMetadataCompatible()`.

## Phase 2: Profiled Adaptive Dispatch (Multi-Armed Bandit)

Since selection is free (3–6 ns), the question is: **what information drives it?**

### Formulation
- **Arms:** N pre-compiled kernel variants (N < 10)
- **Reward:** Negative execution time
- **Context:** (kernel_name, shape_hash, device_id) — cacheable
- **Objective:** Minimize cumulative regret

### Why it's a degenerate bandit
- N < 10 arms (few variants)
- σ² < 5% variance (near-deterministic)
- Cacheable contexts (same tuple = same answer forever)

**Result:** Exhaustive exploration (N × warmup samples) then permanent exploitation. O(N) constant regret. UCB1/Thompson Sampling unnecessary.

### Algorithm (3 phases)
```
fn dispatch(ctx, variants[N], warmup=3):
  key = (ctx.kernel, ctx.shape, ctx.device)
  if key in cache: return cache[key]           // EXPLOIT
  if stats[key].count < N*warmup:              // EXPLORE
    arm = stats[key].count % N
    t = time(variants[arm])
    stats[key].update(arm, t)
    return variants[arm]
  cache[key] = argmin(stats[key].median)       // LOCK
  return cache[key]
```

### Experimental Results (5-scenario benchmark suite)

| Scenario | Finding |
|----------|---------|
| Near-identical (12% spread, 8% noise) | Converges despite noise |
| Context-dependent | Different shapes → different optima |
| Scaling (N=2→64) | 6/25/48/73/85 dispatches to converge |
| Non-stationary | Static fails; re-validation needed |
| **Comparison** | **Profiled = 83% of oracle, 7.3x less regret than random** |

## `#gpu.runtime_select` Design (Proposed)

```mlir
gpu.module @matmul_variants
  { gpu.runtime_select = #gpu.runtime_select<
      policy = "best_compatible",
      fallback = "first_valid"> }
{  gpu.func @matmul_sm80 { min_sm=80 } ...
   gpu.func @matmul_sm90 { min_sm=90 } ... }
```

Plugs into `GPUOffloadingLLVMTranslationAttrInterface`. Emits dispatch table in `global_ctors`. Zero hot-path overhead after one-time selection (same as CPU IFunc/FMV via `target_clones`).

## Implementation

- `kdl.c`: 5,157 LOC — dispatch library
- `runtime_select_poc.c`: 664 LOC — end-to-end PoC
- `profiled_dispatch.c`: 238 LOC — MAB profiler
- `bench_mab_suite.c`: 514 LOC — 5-scenario benchmark
- No external deps beyond CUDA driver API. Builds with `gcc -O2`.

## Related Work

| System | Multi-vendor | Metadata | Policy | Measured | Upstream |
|--------|:-:|:-:|:-:|:-:|:-:|
| IREE HAL | ✅ | ✅ | 🟡 | ❌ | ❌ |
| chipStar | ✅ | ❌ | ❌ | ❌ | ❌ |
| Proteus (CGO 2025) | ❌ | ❌ | ✅ | ✅ | ❌ |
| liboffload | ✅ | 🟡 | ❌ | ❌ | ✅ |
| CPU FMV | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Ours** | ✅ | ✅ | ✅ | ✅ | ❌ |

## Key Differentiators
- vs Triton/Helion: they recompile per target (JIT); we select from pre-compiled (AOT)
- vs cuBLAS: NVIDIA-only, proprietary heuristic; we're cross-vendor, open, MLIR-native
- vs IREE: 100K+ LOC full stack; we're lightweight standalone

## Contact
- S. Akash — IIT Patna, CERN GSoC, vLLM contributor
- 2201ee54_sakash@iitp.ac.in / drakathakash@gmail.com
- https://github.com/Akasxh/libkdl
