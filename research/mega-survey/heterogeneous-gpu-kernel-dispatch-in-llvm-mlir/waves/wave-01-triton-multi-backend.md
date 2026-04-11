# Wave 01: Triton Multi-Backend Compilation
Search query: "Triton compiler multi-backend AMD NVIDIA CPU code generation"
Sources found: 10
Date: 2026-04-06

## Sources

### 1. Triton Kernel Compilation Stages — PyTorch Blog
- URL: https://pytorch.org/blog/triton-kernel-compilation-stages/
- Type: blog
- Date: 2024
- Relevance: 10/10
- Novelty: 7/10
- Summary: Canonical reference for Triton's shared multi-backend compilation pipeline. The five-stage IR chain (Python AST → TTIR → TTGIR → LLIR → vendor binary) is universal across all vendor backends. Optimization passes are stratified into three tiers: MLIR-generic (CSE, DCE, inlining), GPU-generic (Pipeline, Prefetch, MatmulAcceleration), and vendor-specific (NVIDIA: TMA/AsyncDot/WarpSpecialization; AMD: OptimizeLDSUsage/BlockPingpong). The Triton cache under `$HOME/.triton/cache` stores all pipeline stages keyed by (kernel_hash, arch, num_warps, num_stages).
- Key detail: The cache stores every intermediate stage (TTIR, TTGIR, LLIR, binary) alongside metadata, meaning Triton already has an embryonic per-stage binary artifact store — the exact architecture a multi-vendor dispatch library would need.

### 2. MLIR Dialects and IR System — DeepWiki triton-lang/triton
- URL: https://deepwiki.com/triton-lang/triton/3-mlir-dialects-and-ir-system
- Type: docs
- Date: 2024–2025
- Relevance: 9/10
- Novelty: 8/10
- Summary: Documents Triton's multi-level dialect hierarchy: Triton → TritonGPU → TritonNvidiaGPU / TritonAMDGPU, with each level providing progressively finer hardware targeting. The PassManager orchestrates dialect-to-dialect transformations; backend-specific dialects (e.g., TritonAMDGPUDialect) are loaded dynamically at compile time. Layout encodings — describing how tensors are distributed across cores, warps, and wavefronts — are first-class IR citizens, making tensor tiling and memory placement explicit in the IR rather than implicit in codegen heuristics.
- Key detail: The generic/hardware-specific dialect split is the defined abstraction boundary in Triton's design. For a runtime dispatch system like libkdl, the TTGIR level (hardware-generic GPU IR) is the highest level at which a pre-compiled binary can be stamped with meaningful metadata before diverging into vendor-specific lowering.

### 3. A Deep Dive Into AMD Triton Compilation — Medium (nzhangnju)
- URL: https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45
- Type: blog
- Date: 2024
- Relevance: 9/10
- Novelty: 8/10
- Summary: Step-by-step walkthrough of the AMD HIP backend: `make_ttir()` → `make_ttgir()` → `make_llir()` → `make_amdgcn()` → `make_hsaco()`. AMD-specific TTGIR passes include `OptimizeLDSUsage`, `BlockPingpong` (inter-warp instruction interleaving), and `StreamPipeline` (global-memory prefetching into LDS). LLVM IR is translated to AMDGCN assembly via `translateLLVMIRToASM` with target triple `amdgcn-amd-amdhsa`, then linked into an HSACO ELF by ROCm's linker.
- Key detail: `hipGetDeviceProperties` is called inside `make_llir()` to read `gcnArchName` and set LLVM target attributes — device detection is baked into compilation, not deferred to runtime. This means AOT compilation without target hardware requires explicit `HIPOptions.arch` overrides, validating cross-device pre-compilation as non-trivial.

### 4. triton/third_party/amd/backend/compiler.py — GitHub Source
- URL: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Type: source/commit
- Date: Active (2025)
- Relevance: 10/10
- Novelty: 9/10
- Summary: Primary source for AMD backend implementation in Triton. The `HIPOptions` dataclass captures all compilation knobs: `arch` (gfx942, gfx950, gfx1250), `num_warps`, `waves_per_eu`, `schedule_hint` (e.g., "attention", "memory-bound-attention"), `enable_fp_fusion`, FP8 emulation flags, and sanitizer support. `add_stages()` registers all pipeline stages and supports both Triton and Gluon kernel languages. Architecture-conditional code paths handle async copy (gfx950/gfx1250 only) and in-thread transpose (gfx942 only).
- Key detail: The `schedule_hint` parameter is AMD-exclusive — a named optimization strategy with no NVIDIA equivalent. Kernels compiled with different `schedule_hint` values produce different binaries for identical source. A dispatch system must tag binaries with their exact `HIPOptions` and preserve this metadata; it cannot be recovered from the binary at runtime.

### 5. Intel XPU Backend for Triton — GitHub Repository
- URL: https://github.com/intel/intel-xpu-backend-for-triton
- Type: repository
- Date: Active (2024–2025)
- Relevance: 8/10
- Novelty: 8/10
- Summary: Intel's out-of-tree Triton backend for Intel GPUs (Arc, Data Center GPU Max). Originally targeted SPIR-V via a translator, the backend shifted toward Intel's native LLVM GPU target (`llvm-target`) for performance, prioritizing deep LLVM integration over SPIR-V portability. Intel-specific passes handle 2D block IO and hardware DPAS (dot-product accumulate systolic) instructions. Active development tracked via GitHub milestones; SPIR-V translator dependency remains a fragility surface — upstream Triton changes broke SPV_KHR_bfloat16 support (issue #5574).
- Key detail: Intel deliberately moved away from SPIR-V as the portable target IR in favor of native LLVM GPU codegen. This is a direct data point against "SPIR-V as universal dispatch format" — the only production backend that was SPIR-V-based abandoned it for performance reasons.

### 6. triton-lang/triton-cpu — Experimental CPU Backend
- URL: https://github.com/triton-lang/triton-cpu
- Type: repository
- Date: Active (2024–2025)
- Relevance: 8/10
- Novelty: 9/10
- Summary: Official experimental CPU backend for Triton, maintained as a long-lived development branch to minimize divergence from the Triton main repo. Uses MLIR's vector dialect for CPU-side vectorization (in contrast to triton-shared which uses the linalg dialect and lacks target-dependent optimizations). Supports multithreading via auto-generated OpenMP C code from a Python script. Built by passing `TRITON_CPU_BACKEND=1`; at the Triton community meetup in August 2024, triton-cpu demonstrated performance comparable to or better than PyTorch's CPU execution on several benchmarks.
- Key detail: The CPU backend follows the same `third_party/[vendor]/backend/` plugin structure as AMD and NVIDIA backends — the architecture for adding non-GPU targets to Triton is already established. This validates the claim that Triton's backend model is extensible by design, not by accident.

### 7. ML-Triton: A Multi-Level Compilation and Language Extension — arXiv:2503.14985
- URL: https://arxiv.org/abs/2503.14985
- Type: paper
- Date: March 2025 (revised March 26, 2025)
- Relevance: 7/10
- Novelty: 9/10
- Summary: Proposes extending Triton's single workgroup-to-thread lowering into a three-tier hierarchy: workgroup → warp → intrinsic. Motivation is that modern GPUs (Hopper SM90, CDNA3 gfx942) expose warp-level MMA and blocked load instructions not reachable through Triton's current tile-to-thread-direct lowering without hand-written intrinsics. Extends the Triton language with user-set compiler hints and warp-level programming constructs. Validated on Intel GPUs achieving over 95% of expert-written kernel performance.
- Key detail: Multi-level lowering means future kernel binaries may need to be stamped with a hardware feature tier beyond just architecture name — warp-level MMA capability, async copy support, and blocked load availability all become binary-level properties that a dispatch system must track and match at runtime.

### 8. vLLM Triton Attention Backend Deep Dive — vLLM Blog
- URL: https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
- Type: blog
- Date: March 2026
- Relevance: 9/10
- Novelty: 8/10
- Summary: Technical deep dive into the production Triton attention backend in vLLM — approximately 800 lines of Triton code running the same source on NVIDIA H100, AMD MI300X, and Intel XPU. On H100, the Triton backend achieves 100.7% of FlashAttention3 performance for long decode; on MI300X, it delivers approximately 5.8x speedup over earlier HIP implementations. The same paged-attention Triton kernel is the default backend on AMD (ROCm) and is used on Intel XPU for fp32 (where FlashAttention does not support fp32). Covers CUDA graph interactions, parallelization strategies, and benchmarking methodology.
- Key detail: 800 lines of Triton replaces approximately 70,000 lines of hand-written FlashAttention3 CUDA/HIP, with competitive performance across three vendors using one source file. This is the strongest real-world evidence that Triton-based portability is production-viable — and that pre-compiled Triton binaries per vendor could be packaged and dispatched at load time.

### 9. Bridging the Gap: Compiling Triton Kernels Onto RISC-V Targets — RISC-V Summit 2024
- URL: https://static.sched.com/hosted_files/riscvsummit2024/0b/202410-RISC-V-NA-Summit-Compiling%20and%20Optimizing%20Triton%20Kernels%20Onto%20RISC-V%20Targets%20Based%20on%20MLIR.pdf
- Type: talk/slides
- Date: October 2024
- Relevance: 7/10
- Novelty: 9/10
- Summary: Terapines Technology presents an end-to-end software stack for compiling Triton kernels to RISC-V AI chips using MLIR/LLVM. The pipeline lowers Triton kernels and neural network graphs from PyTorch/ONNX/TensorFlow/JAX through a range of MLIR dialects, applying coarse-grained optimizations (loop tiling, kernel fusion, auto-vectorization) before targeting RISC-V. Discusses Triton language limitations for non-GPU architectures and proposes extensions. The approach enables reuse of existing Triton kernel libraries (from PyTorch and vLLM) on RISC-V-based AI chips.
- Key detail: RISC-V is already a Triton compilation target via MLIR — confirming that Triton's backend model extends to non-GPU ISAs. The talk explicitly identifies Triton language limitations for non-GPU targets, pointing to missing abstractions for memory hierarchy and vector width that GPU-centric Triton assumes.

### 10. vLLM Triton Backend: State-of-the-Art Performance on NVIDIA and AMD — IBM Research / PyTorch Conference 2025
- URL: https://research.ibm.com/publications/vllm-triton-backend-how-to-get-state-of-the-art-performance-on-nvidia-and-amd-with-just-triton
- Type: talk/paper
- Date: 2025
- Relevance: 9/10
- Novelty: 8/10
- Summary: IBM Research, Red Hat, and AMD collaboration demonstrating a Triton-only vLLM attention backend achieving state-of-the-art performance across GPU vendors without any hand-written CUDA or HIP kernels. Uses Triton autotuning with a persistent database for per-platform kernel configuration, enabling portability without sacrificing performance. The backend is now the default for AMD in vLLM and runs efficiently on NVIDIA and Intel using the same kernel source.
- Key detail: Platform portability is achieved through autotuning, not source-level abstraction — the same kernel source is compiled per-platform with different tile sizes and pipeline depths. This is precisely the AOT dispatch model: single source, multiple pre-compiled configurations, runtime selection by hardware ID.

---

## Angle Assessment

- **Coverage:** Well-explored from production usage (vLLM), AMD backend internals, and the three-tier pass hierarchy. The RISC-V angle and ML-Triton hierarchical lowering are underexplored in the broader survey. The CPU backend's plugin architecture design is a unique data point for libkdl's backend extensibility design.

- **Surprise findings:** Two significant surprises: (1) Intel explicitly abandoned SPIR-V as the portable target in favor of native LLVM GPU codegen — direct counter-evidence to SPIR-V as a universal dispatch layer. (2) The RISC-V backend is already production-targeted via Terapines, meaning Triton's backend model has been validated on non-GPU ISAs in 2024.

- **Gaps:** No existing system pre-compiles Triton kernels cross-vendor (NVIDIA + AMD + Intel + CPU) into a single distributable archive with runtime dispatch. Triton's cache is per-process and per-machine, with no standard packaging format. The ML-Triton hierarchical lowering work implies that future kernels will require richer binary metadata (warp-level feature tiers) not yet captured in any dispatch design.

- **Suggested follow-up angles:**
  - `AOTriton AKS2 archive format` — the only existing cross-architecture (intra-AMD) kernel archive format; reverse-engineer for cross-vendor generalization
  - `triton-lang/triton issue #935` — community discussion on AOT pre-compilation and binary shipping
  - `Triton autotuning database persistence` — IBM/Red Hat's per-platform autotuning store; compare with AOTriton's SQLite scheme
  - `Triton Gluon language` — listed as a second kernel language in AMD `compiler.py`, unexplored implications for multi-language dispatch
  - `PyTorch torch.compile + TorchInductor backend selection` — how PyTorch decides which Triton backend to invoke at kernel dispatch time
