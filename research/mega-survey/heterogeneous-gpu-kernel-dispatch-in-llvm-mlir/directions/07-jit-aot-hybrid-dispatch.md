# Direction 07: Hybrid AOT+JIT Dispatch — Pre-Compiled Variants with LLVM IR Fallback

**Composite Score: 7.25/10**
**Rank: 7 of 8**

---

## Title

**Best of Both Worlds: Pre-Compiled Native Dispatch with LLVM IR JIT Fallback for Unseen Hardware**

## One-Sentence Description

Extend libkdl's MTB format to carry LLVM IR bitcode alongside native variants, enabling Proteus-style JIT specialization as a fallback when no pre-compiled variant matches the runtime hardware.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| Proteus (CGO 2025) | wave-05-gpu-kernel-jit | 2.8x AMD, 1.78x NVIDIA via LLVM-IR-level constant folding; portable across CUDA+HIP |
| AdaptiveCpp SSCP | wave-03-adaptivecpp | +30% over CUDA, +44% over HIP via runtime JIT specialization |
| NVRTC/HIPRTC JIT paths | wave-05-gpu-kernel-jit | NVRTC cold: ~600ms; HIPRTC mirrors API; comgr handles AMD backend |
| MLIR mgpuModuleLoadJIT | wave-05-gpu-kernel-jit | NVIDIA-only PTX JIT in MLIR; no AMD equivalent — asymmetric gap |
| chipStar 40min→40s caching | wave-05-chipstar-spirv | SPIR-V JIT cost at scale is production-critical; caching alone reduces 60x |
| cuTENSOR 6.9x JIT speedup | wave-05-gpu-kernel-jit | JIT-specialized plans outperform pre-compiled by 6.9x for specific contractions on H100 |
| OffloadBinary ImageKind | wave-06-llvm-offload-new-driver | ImageKind enum includes Bitcode(2); LLVM IR is already a valid OffloadBinary payload type |

---

## Novelty Argument

The AOT vs JIT debate is a false dichotomy. libkdl's MTB format already supports heterogeneous variant types. Adding LLVM IR bitcode as a variant type creates a three-tier dispatch:

```
Tier 1: Pre-compiled native variant for exact hardware match (0 ms cold start)
Tier 2: Pre-compiled native variant for compatible hardware (minimal mismatch)
Tier 3: LLVM IR bitcode → JIT compile → cache result (Proteus-style, one-time cost)
```

No existing system implements all three tiers in a unified dispatch framework:
- AOTriton: Tier 1 only (AMD, per-gfx HSACO)
- AdaptiveCpp: Tier 3 only (always JIT from LLVM IR)
- CUDA fatbin: Tiers 1+2 only (cubin + PTX JIT, NVIDIA-only)
- libkdl: All three tiers, cross-vendor

The cuTENSOR 6.9x speedup demonstrates that JIT specialization can dramatically exceed AOT for workload-specific configurations. The 843s PT2 cold start demonstrates that JIT alone is unacceptable for production. The hybrid resolves both.

---

## Feasibility Plan

1. Add `KDL_VARIANT_LLVM_IR` to variant type enum in kdl.c
2. When no native variant matches, invoke LLVM's OrcJIT (or vendor JIT: NVRTC for NVIDIA, comgr for AMD) on the IR bitcode
3. Cache the JIT result using the same cache key scheme (from wave-06-kernel-binary-abi)
4. For poster: demonstrate fallback path on a simple kernel where native variant is deliberately excluded

**Risk:** Linking against LLVM's JIT infrastructure adds significant binary size and build complexity. For the poster, the fallback can be demonstrated via NVRTC (PTX intermediate) without full OrcJIT.

---

## Poster Potential

**Limited as standalone — best as a "future work" bullet on the main poster.**

- Diagram: three-tier dispatch hierarchy
- Cold start comparison: Tier 1 (<5ms) vs. Tier 3 (~600ms NVRTC) vs. no-libkdl (843s Triton)
- cuTENSOR 6.9x number as motivation for JIT path

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **7/10** | Three-tier hybrid is novel in combination; individual tiers exist in other systems. |
| **Feasibility** | **7/10** | Variant type extension is easy; JIT infrastructure integration is heavy (LLVM OrcJIT or vendor JIT). |
| **Evidence** | **8/10** | Proteus 2.8x, cuTENSOR 6.9x, AdaptiveCpp +30% all validate JIT benefit. 843s cold start validates AOT necessity. |
| **Impact** | **7/10** | Handles the "what about unseen hardware?" objection elegantly. Future-proofs libkdl. |
| **Composite** | **7.25/10** | |
