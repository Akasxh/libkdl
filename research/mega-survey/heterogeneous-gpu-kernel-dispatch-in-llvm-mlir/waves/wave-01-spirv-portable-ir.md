# Wave 01: SPIR-V as Portable GPU IR
Search query: SPIR-V portable intermediate representation GPU kernel multi-vendor NVIDIA AMD Intel
Sources found: 10
Date: 2026-04-06

## Sources

### 1. SPIR-V — The Industry Open Standard Intermediate Language for Parallel Compute and Graphics
- URL: https://www.khronos.org/spirv/
- Type: docs
- Date: ongoing (latest: SPIR-V 1.6 Rev 6, July 2025)
- Relevance: 9/10
- Novelty: 4/10
- Summary: The authoritative Khronos Group specification page for SPIR-V. SPIR-V is a binary intermediate language for parallel compute and graphics, adopted as the mandatory IR in Vulkan, the optional IR in OpenCL 2.1+, and now planned as the Direct3D Interchange format replacing DXIL from Shader Model 7 (announced by Microsoft, September 2024). The binary format reduces driver complexity by eliminating front-end compilers from device drivers.
- Key detail: SPIR-V 1.6 Rev 6 (July 2025) adds clarifications on variable pointers and cooperative matrix capabilities — direct relevance to ML kernel dispatch scenarios.

### 2. SPIR-V — Wikipedia
- URL: https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
- Type: docs
- Date: 2025 (maintained)
- Relevance: 7/10
- Novelty: 3/10
- Summary: Comprehensive overview of SPIR-V's history, structure, and adoption. SSA-form binary IR with explicit control-flow graphs; adopted in OpenCL 2.1 core spec and Vulkan. Original SPIR (now legacy as of 2025) was a subset of LLVM IR targeting OpenCL 1.x/2.0. Modern toolchains have fully migrated to SPIR-V.
- Key detail: Microsoft's September 2024 announcement to adopt SPIR-V as D3D Interchange format from SM7 onward is a significant signal of cross-ecosystem convergence — reduces the "NVIDIA only" objection to SPIR-V.

### 3. AMD Lands Support For Vendor Flavored SPIR-V Within LLVM
- URL: https://www.phoronix.com/news/LLVM-AMDGCN-Flavored-SPIR-V
- Type: blog
- Date: June 2024
- Relevance: 10/10
- Novelty: 9/10
- Summary: AMD merged AMDGCN-flavored SPIR-V into LLVM in June 2024. This is SPIR-V that relaxes portability preconditions in exchange for AMDGCN-specific capabilities: inline AMDGCN assembly (via SPV_INTEL_inline_assembly extension), target-specific built-ins, and better feature-set alignment with actual AMDGCN hardware. Compiled with `-target spirv64-amd-amdhsa`.
- Key detail: This is a canonical demonstration that "portable SPIR-V" and "vendor-optimized SPIR-V" are two distinct compilation targets — directly relevant to libkdl's problem of selecting the right kernel variant at dispatch time.

### 4. SPIR-V Support in LLVM and Clang (LLVM Dev Meeting 2021 Slides)
- URL: https://llvm.org/devmtg/2021-11/slides/2021-SPIR-V-SupportinLLVMandClang.pdf
- Type: docs
- Date: November 2021
- Relevance: 8/10
- Novelty: 5/10
- Summary: Presents the roadmap for upstreaming LLVM's SPIR-V backend (completed in 2022). The LLVM SPIR-V backend was designed to supersede the SPIRV-LLVM-Translator for new workflows. Covers the toolchain architecture: Clang → LLVM IR → SPIR-V backend → SPIR-V binary, and the parallel path via SPIRV-LLVM-Translator.
- Key detail: The SPIR-V backend is designated as a permanent LLVM target — meaning SPIR-V is now a first-class compilation target alongside PTX and AMDGCN within the LLVM compilation pipeline.

### 5. SPIRV-LLVM-Translator (KhronosGroup GitHub)
- URL: https://github.com/KhronosGroup/SPIRV-LLVM-Translator
- Type: docs
- Date: active (2024-2025)
- Relevance: 9/10
- Novelty: 5/10
- Summary: Bi-directional LLVM IR ↔ SPIR-V translation library targeting the OpenCL/compute "Kernel" capability dialect. Explicitly cannot handle vendor-specific built-ins (e.g., NVVM intrinsics). Used by Intel's oneAPI and upstream OpenCL toolchains. Expected to be superseded by the native LLVM SPIR-V backend for new use cases.
- Key detail: The translator only supports the `Kernel` capability dialect — it cannot round-trip `Shader` dialect SPIR-V produced by clspv, confirming the fragmentation of SPIR-V into non-interoperable dialects.

### 6. Clspv — OpenCL C to Vulkan Compute Shaders (Google GitHub)
- URL: https://github.com/google/clspv
- Type: docs
- Date: active (2024-2025)
- Relevance: 8/10
- Novelty: 6/10
- Summary: Google's compiler that translates OpenCL C into SPIR-V in the Vulkan `Shader` dialect (not the `Kernel` dialect). Works by transforming LLVM IR through a set of module passes. Enables OpenCL kernels to run on Vulkan drivers (via the clvk runtime layer), giving NVIDIA Vulkan access to OpenCL workloads without CUDA.
- Key detail: The existence of two incompatible SPIR-V dialects (`Kernel` vs `Shader`) is a critical practical limitation: a SPIR-V binary produced for OpenCL drivers cannot be directly consumed by Vulkan drivers and vice versa, breaking naive "write once, run anywhere" assumptions.

### 7. AMD AI Compiler Engineer Lands A Generic MLIR To SPIR-V Pass In LLVM 19
- URL: https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V
- Type: blog
- Date: 2024 (LLVM 19 cycle)
- Relevance: 9/10
- Novelty: 8/10
- Summary: An AMD AI compiler engineer upstreamed a generic MLIR-to-SPIR-V lowering pass into LLVM 19. Supports lowering of `arith`, `vector` (1-D, sizes 2/3/4/8/16), `scf`, `ub`, `index`, `func`, and `math` dialects directly to SPIR-V. The `gpu` dialect and `tensor` dialect conversions are planned but not yet upstreamed.
- Key detail: The GPU dialect → SPIR-V lowering gap means that MLIR's primary GPU abstraction layer cannot yet be compiled to portable SPIR-V via the upstream path — projects must either bridge through custom lowerings or use vendor-specific pipelines (e.g., nvvm for NVIDIA).

### 8. SPIR-V Dialect — MLIR Documentation
- URL: https://mlir.llvm.org/docs/Dialects/SPIR-V/
- Type: docs
- Date: active (LLVM main)
- Relevance: 9/10
- Novelty: 4/10
- Summary: Official documentation for the MLIR SPIR-V dialect. The dialect models the SPIR-V binary format directly, with explicit support for SPIR-V versions, extensions, and capability sets. Provides conversion patterns from higher-level MLIR dialects and serialization to binary SPIR-V modules. Designed to bridge the gap between high-level ML IRs and GPU driver-consumable SPIR-V.
- Key detail: The dialect exposes version/extension/capability targeting at IR construction time — which means portability decisions (e.g., Kernel vs Shader capability, vendor extensions) must be resolved during compilation, not at runtime dispatch, creating a fundamental tension with dynamic dispatch models.

### 9. HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs
- URL: https://arxiv.org/abs/2506.15993
- Type: paper
- Date: June 2025 (arXiv 2506.15993)
- Relevance: 10/10
- Novelty: 10/10
- Summary: HetGPU proposes a system (compiler + runtime + abstraction layer) enabling a single GPU binary to execute on NVIDIA, AMD, Intel, and Tenstorrent hardware via a custom hetIR (inspired by SPIR-V and PTX). The runtime dynamically translates hetIR to the target GPU's native code at dispatch time. For SPIR-V targets (Intel, AMD via Vulkan/OpenCL), a dedicated hetIR→SPIR-V module handles translation. Preliminary results show live kernel migration across disparate GPUs with minimal overhead.
- Key detail: HetGPU's hetIR functions as an architecture-agnostic GPU IR that subsumes SPIR-V as a backend target — this is the closest published prior work to libkdl's dynamic dispatch concept, and demonstrates that SPIR-V alone is insufficient for true binary portability (it must be generated from a higher-level IR at runtime).

### 10. Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors
- URL: https://arxiv.org/abs/2603.28793
- Type: paper
- Date: March 2026 (arXiv 2603.28793)
- Relevance: 9/10
- Novelty: 10/10
- Summary: First systematic cross-vendor ISA analysis across NVIDIA (PTX v1.0–v9.2, Fermi–Blackwell), AMD (RDNA 1-4, CDNA 1-4), Intel (Gen11–Xe-HPC), and Apple (M-series, reverse-engineered). Identifies 10 hardware-invariant computational primitives, 6 parameterizable dialects (same concept, different parameters), and 6 true architectural divergences. Proposes an abstract execution model for a vendor-neutral GPU ISA. Validated on NVIDIA T4 and Apple M1; matches or exceeds native performance on 5/6 benchmark-platform pairs.
- Key detail: The 6 true architectural divergences (not just parameter differences) define exactly where SPIR-V abstraction breaks down and where runtime dispatch with target-specific kernel variants is unavoidable — directly frames libkdl's design space.

---

## Angle Assessment

- Coverage: This angle is moderately well-explored in official documentation and toolchain repositories. The SPIR-V spec, MLIR dialect, and LLVM backend are well-documented. Academic work on SPIR-V portability *in practice* (performance benchmarks, limitations under realistic ML workloads) is sparse.
- Surprise findings:
  - Microsoft's adoption of SPIR-V for Direct3D (SM7+) is a major ecosystem signal not widely cited in GPU compute literature — potentially the strongest argument for SPIR-V as the long-term portable kernel IR.
  - The `gpu` dialect → SPIR-V lowering gap in MLIR is a concrete, current blocker for any project trying to use SPIR-V as a first-class output from MLIR-based ML compilers.
  - AMD's AMDGCN-flavored SPIR-V formalizes the "portability vs. performance" tradeoff as an explicitly supported compilation mode, not just an unofficial workaround.
  - The arXiv 2603.28793 paper (March 2026) identifying exactly 6 true architectural divergences provides a principled, empirical bound on SPIR-V's portability ceiling.
- Gaps:
  - No published performance benchmarks comparing portable SPIR-V vs. vendor-native (PTX/AMDGCN) execution overhead for ML workloads (matmul, attention, etc.).
  - No published work on runtime SPIR-V selection/specialization (analogous to libkdl's dispatch model).
  - NVIDIA's CUDA ecosystem lacks native SPIR-V ingestion — the only path is Vulkan compute or OpenCL, both of which are non-trivial for CUDA-centric ML workflows.
  - No analysis of SPIR-V cooperative matrix (`KHR_cooperative_matrix`) adoption across vendors for ML tensor operations.
- Suggested follow-up angles:
  1. MLIR gpu dialect → SPIR-V lowering gap: current state, workarounds, roadmap (LLVM Discourse threads)
  2. NVIDIA + SPIR-V: Vulkan compute as an alternative to PTX for portable ML kernels — adoption, overhead, toolchain maturity
  3. SPIR-V cooperative matrix extension across vendors — what works, what is vendor-locked
  4. clvk runtime layer: using Vulkan as a universal OpenCL-over-SPIR-V substrate for multi-vendor dispatch
  5. Performance delta: portable SPIR-V vs. native ISA for GEMM/attention workloads — any published data?
  6. The `hetIR` from HetGPU paper as a concrete SPIR-V successor design — compare to libkdl's kernel bundle format
