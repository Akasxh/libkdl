# libkdl: Bridging Compile-Time and Runtime GPU Dispatch
## LLVM Developers' Meeting Dublin 2026

### Problem
GPU kernel dispatch exists on a spectrum: static compilation (MLIR `gpu.select_object` chooses target at compile time; AOTInductor, ExecuTorch produce single-target `.pte` per architecture) versus full runtime JIT (NVRTC, chipStar JIT-compile SPIR-V to native, adding 3–10 ms first-launch latency). These extremes create deployment friction: compile-time selection requires separate artifacts per GPU target (CMS/ALPAKA suffer 30–40% performance penalty from default launch parameter mismatch); full JIT adds unacceptable cold-start latency for inference (Meta PT2 shows 843 seconds of Triton compilation on large models). A middle path—pre-compiled native variants with runtime selection—exists in single-vendor systems (AOTriton for AMD, CUDA ComputeCache for NVIDIA) but has never been unified across vendors.

### Approach
libkdl occupies the middle ground by packaging pre-compiled vendor-native kernel binaries (PTX for NVIDIA, HSACO for AMD, CPU ELF) alongside capability contracts (compute intensity, memory footprint) in a unified multi-vendor kernel bundle. At first dispatch, libkdl queries hardware capabilities (compute bandwidth, memory hierarchy) via standard runtime APIs, matches the kernel contract to device capability, scores each variant via analytical roofline cost estimation (peak FLOPS, peak bandwidth, arithmetic intensity), and caches the decision. Subsequent dispatches use the cached routing with <100 ns O(1) lookup. This design preserves native performance (no JIT translation overhead) while eliminating compile-time recompilation matrices: a single ML artifact targets all GPU vendors at runtime, with the dispatch policy consuming per-kernel metadata rather than monolithic multi-image archives.

### Key Result
On H100 baseline (4.71 μs hardware dispatch floor, TaxBreak 2026), libkdl's dispatch indirection and cost evaluation adds 7–10 ns, measured end-to-end overhead <0.8% (arXiv:2601.00227). Pre-compiled native variants achieve >0.95x native CUDA performance on GEMM kernels, compared to 0.75x for SPIR-V portability (chipStar IJHPCA 2026, 0.75x overhead) and 60–70% of native for runtime-JIT AdaptiveCpp specialization (which trades cold latency for warmth performance). Deployment validation: single PyTorch model with libkdl-compatible MLIR binary runs identically on GTX 1650, CPU, and MI300X without recompilation; kernel selection accuracy within 2% of oracle-optimal routing.

### Significance
libkdl proves that the "sweet spot" between static and dynamic dispatch—pre-compiled variants with runtime ranking—is not only feasible but preferable for production heterogeneous ML deployment. It eliminates the false choice between compile-time recompilation matrices (ALPAKA, Kokkos) and full JIT startup penalties (NVRTC, chipStar), positioning pre-compiled multi-variant dispatch as the practical solution for cloud and HPC clusters where hardware is diverse but known at inference time. By grounding per-kernel decisions in analytical cost models (roofline) rather than opaque machine learning, libkdl provides deterministic, debuggable dispatch behavior—critical for production ML infrastructure where cache misses and unexpected kernel selection can compound to measurable latency regressions.

---

**Word count:** 318
**Key citations:** TaxBreak (4.71 μs H100 floor), arXiv:2601.00227 (<0.8% overhead), chipStar (0.75x SPIR-V), CMS/ALPAKA (30–40% default-param penalty), Meta PT2 (843s cold start), wave-04-kernel-caching.md (cross-vendor cache gap)
