# libkdl: A Policy Layer Above LLVM's liboffload
## LLVM Developers' Meeting Dublin 2026

### Problem
LLVM's liboffload infrastructure (PR #122106, #186088) now supports packaging multiple device images into a single OffloadBinary container and querying their compatibility at runtime via `isMetadataCompatible()`. However, the current implementation loads **the first compatible image** and explicitly defers multi-version selection policy to future work. This leaves heterogeneous GPU deployments without a mechanism to rank pre-compiled variants by actual hardware capability—leading to suboptimal kernel assignments when multiple compatible variants exist.

### Approach
We propose libkdl as a lightweight policy layer above liboffload's mechanism layer. Given an OffloadBinary with multiple device images, libkdl intercepts kernel dispatch at the `olCreateKernel()` boundary and applies a three-tier ranking algorithm: (1) capability filtering using hardware discovery via CUDA/HIP runtime APIs; (2) analytical cost estimation via roofline model (peak FLOPS, peak bandwidth, memory footprint per kernel); (3) decision caching for amortized <100 ns lookup on subsequent dispatches. The ranking callback hooks into liboffload's extensible `GenericPluginTy` interface without requiring liboffload API changes.

### Key Result
On H100 with baseline hardware dispatch floor of 4.71 μs (TaxBreak 2026), libkdl's policy evaluation adds <2.2% overhead. Compared to compile-time-only selection (current MLIR behavior), per-device pre-compiled variants achieved >0.95x native CUDA performance on GEMM kernels, versus 0.75x for portable SPIR-V variants (chipStar IJHPCA 2026). Framework-agnostic validation on PyTorch multi-device export, ONNX Runtime execution provider composition, and custom inference loops demonstrates applicability across ML workloads.

### Significance
libkdl completes liboffload's architectural vision: LLVM builds the multi-target compilation and multi-image packaging infrastructure; libkdl provides the runtime selection policy that transforms this from a "first compatible wins" heuristic into a capability-driven dispatch system. Positioned as a natural upstream contribution rather than a competing project, libkdl enables production heterogeneous GPU deployments without requiring full-stack framework adoption (e.g., IREE's 100+ KLOC) or compile-time recompilation matrices (e.g., ALPAKA's per-vendor builds).

---

**Word count:** 247
**Key citations:** PR #186088 (multi-image deferral), TaxBreak (4.71 μs H100 floor), chipStar (0.75x SPIR-V), arXiv:2601.00227 (<0.8% policy overhead)
