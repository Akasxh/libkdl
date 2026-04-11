# Direction 03: Cross-Vendor AOTriton — Generalizing AKS2 Dispatch

**Final Score: 7.75/10** (Rank #4 of 6)
**Scoring History:** Round 1: 8.25 → Round 2: 8.00 → Final: 7.75 (declining)

---

## One-Sentence Description

Generalize AMD's AOTriton dispatch pattern (AKS2 archives, SQLite autotuning, hierarchical arch naming, per-device funcache) to support NVIDIA (CUBIN) and CPU (ELF) targets alongside HSACO, producing a cross-vendor ahead-of-time kernel dispatch library.

---

## Score Breakdown

| Criterion | Score | Justification |
|-----------|------:|---------------|
| Novelty | 8/10 | AOTriton exists only for AMD; cross-vendor generalization is new. However, AdaptiveCpp and chipStar provide alternative cross-vendor stories that reduce uniqueness |
| Feasibility | 7/10 | Requires implementing CUDA + CPU backends alongside existing HIP path; more engineering than Direction A |
| Evidence Strength | 9/10 | AOTriton's production deployment proves the pattern works; ALPAKA CMS data confirms per-device tuning is critical |
| Impact | 7/10 | Narrower audience than Direction A; Triton community would benefit but LLVM community less engaged |

---

## Evidence Summary

### AOTriton Architecture (wave-02-triton S7-S8)

AOTriton is AMD's ahead-of-time Triton kernel dispatch library for HSACO binaries:

- **AKS2 archives:** LZMA-compressed packages containing pre-compiled HSACO kernels for multiple AMD GPU architectures
- **Hierarchical arch naming:** `gfx942` (all MI300), `gfx942_mod0` (MI300X only) — enables fine-grained device targeting
- **SQLite autotuning DB:** Per-architecture lookup of optimal kernel configurations, keyed by problem shape
- **Per-device funcache:** `std::unordered_map<std::string, CUfunction>` (HIP equivalent) guarded by mutex, populated on first dispatch
- **V3 `OpAttnFwd`:** Supports backend enumeration — Triton HSACO or `aiter` assembly for the same operation

### Supporting Evidence

| Source | What It Shows | Wave |
|--------|--------------|------|
| AOTriton V3 design | Production-validated multi-arch kernel dispatch with autotuning | wave-02-triton S7-S8 |
| ALPAKA CMS production | 40% performance penalty without per-device tuning; >94% native with tuning | wave-05-alpaka S1-S4 |
| Stream-K++ Bloom filter | 95.8% variant elimination for GEMM kernel selection | wave-04-cost-models |
| MIOpen find-and-cache | SQLite DB storing per-architecture optimal kernel configurations | wave-05-kernel-caching S8 |
| Triton cache architecture | Hash-based cache keyed by (source + config + backend); cross-backend isolation | wave-05-kernel-caching S3 |

---

## Novelty Argument

### What AOTriton does (AMD-only)

```
Triton kernel source
    ↓ AOT compile (per gfx target)
AKS2 archive = {
    gfx90a: { matmul_M128_N128_K32.hsaco, ... },
    gfx942: { matmul_M128_N128_K32.hsaco, ... },
    gfx942_mod0: { matmul_M128_N128_K64.hsaco, ... }  // MI300X-specific
}
    ↓ Runtime dispatch
hipGetDeviceProperties → match gfx target → SQLite lookup → funcache → hipLaunchKernel
```

### What cross-vendor AOTriton would do

```
Triton kernel source
    ↓ AOT compile (per target: CUDA + AMD + CPU)
Cross-Vendor Archive = {
    sm_80:   { matmul_M128_N128_K32.cubin, ... },
    sm_90:   { matmul_M128_N128_K32.cubin, ... },
    gfx90a:  { matmul_M128_N128_K32.hsaco, ... },
    gfx942:  { matmul_M128_N128_K32.hsaco, ... },
    x86_64:  { matmul_M128_N128_K32.so, ... }
}
    ↓ Runtime dispatch
Device detection → match target → DB lookup → funcache → vendor launch API
```

### Gap this fills

AOTriton's design patterns are directly adoptable for cross-vendor dispatch:
- Godel-numbered tuning keys → extend to NVIDIA SM versions
- Hierarchical arch naming → add NVIDIA SM hierarchy (`sm_80` < `sm_86` < `sm_90`)
- LZMA-compressed archives → same for CUBIN + HSACO + ELF
- SQLite autotuning DB → add per-SM optimal configurations

No existing system provides this cross-vendor AOT kernel archive with device-aware dispatch.

---

## Why This Direction Is Declining

### Relative to Round 1

1. **AdaptiveCpp SSCP** (wave-03-adaptivecpp) provides a more elegant cross-vendor story: single LLVM IR binary, JIT to any target, +30% over CUDA with adaptivity. This is architecturally superior for the "compile once" use case.

2. **chipStar** (wave-04-chipstar) provides another cross-vendor story: SPIR-V as the portable binary, deployed at exascale (Aurora). Though 0.75x native, it covers all SPIR-V platforms.

3. **Direction A** (libkdl as policy layer) subsumes this direction: libkdl's MTB format IS the cross-vendor generalization of AKS2, but framed more broadly as a policy layer above liboffload rather than a Triton-specific tool.

### Result

The cross-vendor AOTriton framing is useful as a **mental model** for the LLVM audience ("think AOTriton, but for NVIDIA + AMD + CPU") but is not the primary contribution. It is a framing device, not the research direction.

---

## Feasibility Plan

### What would be required

| Component | Effort | Challenge |
|-----------|--------|-----------|
| CUBIN variant support in archive format | 1 week | Straightforward: same archive format, different binary type |
| NVIDIA autotuning DB population | 2 weeks | Need to benchmark Triton kernels across SM versions; requires multiple NVIDIA GPUs |
| SM compatibility lattice | 2 days | Document SM partial order (sm_80 runs on sm_86 but not sm_90a) |
| CPU fallback variant | 1 week | Compile Triton kernels to CPU via Triton CPU backend or direct C codegen |
| Cross-vendor dispatch logic | 1 week | Device detection → target matching → DB lookup → vendor-specific launch |
| **Total** | **~5 weeks** | Requires multiple GPU types for autotuning |

### Hardware limitation

- GTX 1650 (SM 7.5) available — sufficient for CUDA path
- No AMD GPU available — HIP path is design-only
- No multi-GPU setup — cross-vendor runtime dispatch cannot be demonstrated end-to-end

---

## Poster Role

### Current recommendation: framing device, not primary direction

The poster should mention AOTriton as the closest single-vendor prior art and frame libkdl as "AOTriton generalized cross-vendor." This gives the audience an immediate mental model without requiring a separate contribution section.

**Suggested mention (50 words in Related Work):**

> "AMD's AOTriton [ref] pre-compiles Triton kernels to per-architecture HSACO archives with SQLite autotuning DB. libkdl generalizes this pattern cross-vendor: the MTB format is a multi-vendor analog of AKS2, and the roofline cost scorer replaces AOTriton's per-shape lookup with a device-independent analytical model."

---

## Key Design Patterns to Adopt from AOTriton

| AOTriton Pattern | libkdl Adoption | Status |
|-----------------|-----------------|--------|
| AKS2 archive (LZMA-compressed multi-arch) | MTB bundle (ELF-based multi-vendor) | Implemented in prototype |
| Hierarchical arch naming (gfx942 > gfx942_mod0) | SM + GFX partial order lattice | Designed, not benchmarked |
| SQLite autotuning DB | Roofline scorer + calibration cache | Roofline implemented; calibration planned |
| Per-device funcache (mutex-guarded hash map) | Dispatch table (kernel_name + device_idx → function pointer) | Implemented in prototype |
| Godel-numbered tuning keys | Capability contract JSON | Implemented in prototype |

---

## Key References

1. AOTriton (AMD, 2024-2025) — AKS2 archives, hierarchical arch naming, SQLite autotuning
2. MIOpen kernel cache (ROCm 6.4) — per-architecture cache with find database
3. Triton cache architecture (Red Hat, May 2025) — hash-based cross-backend isolation
4. Stream-K++ Bloom filter (arXiv:2408.11417) — fast variant elimination
5. ALPAKA CMS CHEP 2024 — 40% tuning gap validates per-device variant need
