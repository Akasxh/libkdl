# Wave 02: Triton Multi-Backend Compilation
Search query: Triton compiler AMD HIP backend SPIR-V multi-target kernel compilation
Sources found: 10
Date: 2026-04-06

## Sources

### 1. Triton Compiler Development Tips — Lei.Chat()
- URL: https://www.lei.chat/posts/triton-compiler-development-tips/
- Type: blog
- Date: 2024 (updated periodically)
- Relevance: 10/10
- Novelty: 9/10
- Summary: The single most technically complete reference on Triton's multi-backend architecture. Documents the full IR chain (TTIR → TTGIR → LLIR → vendor binary), the `third_party/[vendor]/backend/compiler.py` plugin model, and the `GPUTarget` API enabling AOT compilation without target hardware present. Backend selection is set at CMake time via `-DTRITON_CODEGEN_BACKENDS="amd;nvidia"`, and each backend registers its stages dynamically via `add_stages()`. Cross-architecture AOT compilation is shown with `GPUTarget("cuda", 80, 32)` vs `GPUTarget("hip", "gfx942", 64)`.
- Key detail: The entire vendor abstraction boundary is the `compiler.py` file in each backend directory. The same TTIR/TTGIR pipeline runs identically up to the LLIR stage; only LLIR-to-binary diverges per vendor. This is the cleanest articulation of where a runtime dispatch system like libkdl would need to intercept: at the TTGIR→LLIR boundary or at binary ingestion time.

### 2. A Deep Dive Into AMD Triton Compilation — Medium (nzhangnju)
- URL: https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45
- Type: blog
- Date: 2024
- Relevance: 9/10
- Novelty: 8/10
- Summary: Step-by-step walkthrough of the AMD HIP backend pipeline: `make_ttir()` → `make_ttgir()` → `make_llir()` → `make_amdgcn()` → `make_hsaco()`. AMD-specific passes at the TTGIR level include `OptimizeLDSUsage`, `BlockPingpong` (inter-warp instruction interleaving), and `StreamPipeline` (prefetching). The final LLVM IR is translated to AMDGCN assembly via LLVM's `translateLLVMIRToASM` with target `amdgcn-amd-amdhsa`, then linked into an HSACO ELF by ROCm's linker.
- Key detail: The `make_llir()` function invokes `hipGetDeviceProperties` to detect `gcnArchName` and configures the LLVM module accordingly. AMD-specific optimization passes are inserted between standard MLIR CSE/DCE passes and the LLVM backend — creating a clean three-tier pass structure: generic → GPU-generic → vendor-specific.

### 3. triton/third_party/amd/backend/compiler.py — GitHub Source
- URL: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Type: commit/source
- Date: Active (latest 2025)
- Relevance: 10/10
- Novelty: 9/10
- Summary: Primary source for AMD backend implementation. `HIPOptions` dataclass captures all compilation knobs: `arch` (gfx942, gfx950, gfx1250), `num_warps`, `waves_per_eu`, `schedule_hint` (e.g., "attention", "memory-bound-attention"), `enable_fp_fusion`, FP8 emulation flags, and ConSan/FPSan sanitizer support. `add_stages()` registers all pipeline stages dynamically, supporting both Triton and Gluon kernel languages. Architecture-conditional paths exist for async copy (gfx950/gfx1250 only) and in-thread transpose (gfx942 only).
- Key detail: The `schedule_hint` parameter is AMD-exclusive — a named optimization strategy applied during TTGIR lowering. This has no NVIDIA equivalent, demonstrating that the vendor-specific layer carries non-trivial semantic information that a dispatch system must preserve per-binary rather than recomputing at runtime.

### 4. intel/intel-xpu-backend-for-triton — GitHub Repository
- URL: https://github.com/intel/intel-xpu-backend-for-triton
- Type: commit/repository
- Date: Active (2024–2025)
- Relevance: 8/10
- Novelty: 8/10
- Summary: Intel's out-of-tree Triton backend for Intel GPUs (Arc, Data Center GPU Max). The backend originally targeted SPIR-V but has since shifted toward `llvm-target` (Intel's native GPU LLVM backend) rather than emitting SPIR-V via a translator. Environment variables `MLIR_ENABLE_DUMP=1` and `LLVM_IR_ENABLE_DUMP=1` expose the IR pipeline; the compilation chain follows Triton→MLIR→LLVM with Intel-specific passes for 2D block IO and hardware DPAS (dot-product accumulate systolic) operations. Active SPIR-V translator dependency tracked via GitHub milestones through 2024–2025.
- Key detail: The shift from SPIR-V target to LLVM target is architecturally significant — Intel chose deep LLVM integration over portable SPIR-V emission, prioritizing performance over portability. Issue #5574 tracks an SPV_KHR_bfloat16 extension breakage from an upstream Triton update, showing the fragility of cross-backend compatibility in practice.

### 5. Triton Kernel Compilation Stages — PyTorch Blog
- URL: https://pytorch.org/blog/triton-kernel-compilation-stages/
- Type: blog
- Date: 2024
- Relevance: 9/10
- Novelty: 7/10
- Summary: Canonical reference for the shared compilation pipeline used across all Triton vendor backends. The five stages — Python AST → TTIR → TTGIR → LLIR → vendor binary — are universal. Optimization passes are categorized into three tiers: MLIR generic (CSE, DCE, inlining), GPU-generic (Pipeline, Prefetch, MatmulAcceleration), and vendor-specific (NVIDIA: TMA/AsyncDot/WarpSpecialization; AMD: OptimizeLDSUsage/BlockPingpong). The Triton cache stores every stage of the pipeline under `$HOME/.triton/cache` with a hash key computed from kernel source + compilation parameters + target architecture.
- Key detail: Each cached kernel directory contains TTIR, TTGIR, LLIR, and platform binary files alongside metadata. This means Triton already implements a per-stage binary artifact store — the architecture for a multi-vendor dispatch library already exists in embryonic form within Triton's own caching layer.

### 6. NVIDIA CUDA Backend — DeepWiki triton-lang/triton
- URL: https://deepwiki.com/triton-lang/triton/5.6-nvidia-cuda-backend
- Type: docs/wiki
- Date: 2024–2025
- Relevance: 8/10
- Novelty: 7/10
- Summary: Documents the NVIDIA-specific compilation path from TTGIR to PTX to CUBIN. The LLIR stage injects NVVM dialect intrinsics (e.g., `llvm.nvvm.read.ptx.sreg.tid.x()`). Architecture-conditional passes include HopperWarpSpec (SM90), BlackwellTMEM (SM100+). The `JITFunction` class manages compilation and caching, generating cubin binaries delivered to `cuModuleLoadData()` and `cuModuleGetFunction()` for kernel handle resolution. PTX vectorized loads use `ld.global.v4.b32` instructions with L1/L2 cache policy annotations.
- Key detail: The NVIDIA and AMD backends have identical TTIR→TTGIR→LLIR pipeline structure; the divergence is entirely in the LLIR→binary phase. This validates the feasibility of a unified IR format for shipping pre-compiled kernels with vendor-specific binary blobs — the architectural split already exists in Triton's own design.

### 7. AOTriton: Ahead-of-Time Triton Kernel Libraries on ROCm — PyTorch Conference 2024
- URL: https://pytorch2024.sched.com/event/1fHnF/lightning-talk-aotriton-ahead-of-time-triton-kernel-libraries-on-rocm-jeff-daily-amd
- Type: PR/talk
- Date: September 2024
- Relevance: 10/10
- Novelty: 10/10
- Summary: AMD's AOTriton project (Jeff Daily, AMD) is the closest existing analog to libkdl. It pre-compiles Triton kernels to HSACO at build time, packages them into AKS2 archives (LZMA-compressed collections of per-architecture HSACO files), and dispatches at runtime using `hipGetDeviceProperties` to read `gcnArchName`. A SQLite autotuning database maps (dtype, shape, causal-mask) → optimal (block_size, num_warps, staging) per GPU architecture. The C++ shim layer bridges Python-defined kernels to the runtime dispatch logic.
- Key detail: Architecture naming is hierarchical: `gfx942` selects all MI300X/MI300A/MI325X variants; `gfx942_mod0` selects MI300X only. This coarse/fine architecture selection is the exact multi-granularity dispatch mechanism that libkdl would need to implement.

### 8. ROCm/aotriton — GitHub + DeepWiki Architecture
- URL: https://deepwiki.com/ROCm/aotriton
- Type: commit/docs
- Date: 2024–2025
- Relevance: 10/10
- Novelty: 10/10
- Summary: Full technical architecture of AOTriton's runtime dispatch system. Three-layer pipeline: (1) `hipGetDeviceProperties` hardware detection → `gcnArchName` → internal target enum; (2) `OpAttnFwdContext::lookup_optimal(gpu)` queries SQLite for kernel configuration using Godel-numbered keys; (3) `TritonKernel::invoke()` checks per-device `funcache_` (shared-mutex-guarded), decompresses HSACO from `.aks2` via `PackedKernel::filter()`, loads with `hipModuleLoadDataEx()`, and resolves function handle via `hipModuleGetFunction()`. Code generation via `RootGenerator` produces header/source/manifest triplets from Python kernel definitions.
- Key detail: The V3 `OpAttnFwd` operator pattern supports backend enumeration — at dispatch time, either a Triton HSACO kernel or an `aiter` assembly kernel can be selected for the same operation. This multi-implementation dispatch for identical semantics is the core design pattern libkdl needs, but extended cross-vendor (NVIDIA/AMD/CPU) rather than intra-vendor.

### 9. ML-Triton: A Multi-Level Compilation and Language Extension — arXiv 2503.14985
- URL: https://arxiv.org/abs/2503.14985
- Type: paper
- Date: March 2025
- Relevance: 7/10
- Novelty: 8/10
- Summary: Extends Triton's single-level tile abstraction with a three-tier lowering: workgroup → warp → intrinsic. The motivation is that modern GPUs (including Hopper/Ada on NVIDIA and CDNA3 on AMD) expose warp-level MMA and blocked load instructions that Triton's tile-to-thread direct lowering cannot exploit without manually written intrinsics. Validated on Intel GPUs achieving >95% of expert-written performance. The approach decouples hardware abstraction layers, making each lowering step cleanly separable.
- Key detail: Multi-level lowering explicitly separates concerns across logical GPU hierarchy levels. For a dispatch system, this means kernel binaries could be stamped with their minimum hardware feature tier (warp-level MMA required, async copy optional), enabling smarter runtime fallback selection.

### 10. KPerfIR: Compiler-Centric GPU Kernel Performance Tooling — arXiv 2505.21661
- URL: https://arxiv.org/html/2505.21661v1
- Type: paper
- Date: May 2025
- Relevance: 6/10
- Novelty: 7/10
- Summary: Introduces KPerfIR, a multi-level profiling IR built on top of Triton's dialect hierarchy. KPerfIR operates at three levels: hardware-agnostic RecordOp markers → KPerfGPUIR (vendor-independent GPU counter collection) → LLVM IR instrumentation. Supports both NVIDIA and AMD backends through the same pass infrastructure. The multi-level design mirrors KPerfIR's own claim: the separation of hardware-agnostic semantics from vendor-specific lowering is the correct abstraction boundary.
- Key detail: The KPerfGPUIR→LLVM lowering is configurable based on hardware features detected at pass time, showing how a common GPU IR layer can drive vendor-divergent code generation — directly applicable to libkdl's need for a vendor-neutral kernel metadata representation.

---

## Angle Assessment

**Overall relevance to libkdl / runtime heterogeneous dispatch:** HIGH (9/10)

Triton's multi-backend architecture is the most mature real-world implementation of the exact problem libkdl addresses — compiling a single kernel source to multiple vendor binaries and dispatching at runtime. The key findings are:

### Critical Design Insights for libkdl

1. **The correct abstraction boundary is TTGIR→LLIR.** All vendor-specific semantics (warp scheduling, memory layout, async copy availability) are encoded in TTGIR passes. Below TTGIR, the pipeline is truly vendor-specific. libkdl should operate at the binary level (post-HSACO/PTX/CUBIN generation), treating pre-compiled vendor binaries as opaque artifacts.

2. **AOTriton's AKS2+dispatch pattern is the direct prior art.** The `gfx942` → autotuning DB → HSACO decompression → `hipModuleLoadDataEx()` → `funcache_` pattern is exactly what libkdl implements, but AOTriton is AMD-only. libkdl's contribution is extending this to a vendor-neutral dispatch layer covering NVIDIA, AMD, and CPU backends.

3. **Cache key semantics are already solved.** Triton caches per (kernel_hash, arch, num_warps, num_stages, ...). A cross-vendor dispatch library can adopt the same key scheme, extended with vendor discriminator. The Godel numbering scheme in AOTriton V3 is worth adopting for shape-dependent autotuning.

4. **Intel deliberately abandoned SPIR-V as the portable IR.** The XPU backend shifted from SPIR-V target to native LLVM target for performance. This validates libkdl's approach of carrying vendor-native binaries (PTX, HSACO, SPIR-V where required) rather than targeting a lowest-common-denominator portable IR.

5. **The `schedule_hint` and architecture-conditional pass problem.** AMD's `schedule_hint` parameter produces semantically different binaries for the same source. A dispatch system cannot regenerate these at runtime — it must pre-compile and tag each binary with the exact `HIPOptions` used at build time.

### Gaps and Open Questions

- No existing system does cross-vendor AOT dispatch (NVIDIA + AMD + Intel in a single library)
- Triton's cache is per-process-per-machine; there is no standard format for shipping pre-compiled multi-vendor kernel archives
- Intel's SPIR-V translator dependency remains an unresolved compatibility surface (issue #5574 shows upstream Triton changes can silently break the XPU backend)
- ML-Triton's warp-level IR suggests future Triton versions may require more levels of binary specialization than currently (warp count, MMA tier, async copy availability all become binary-level properties)

### Recommended Next Searches

- `triton-lang/triton` AOT compilation issue #935 thread — community discussion on pre-compilation and shipping binaries
- Triton Conference 2024 talks on multi-vendor performance parity (AMD vs. NVIDIA benchmark data)
- `intel-xpu-backend-for-triton` SPIR-V translator migration issues for portability risks
- PyTorch `torch.compile` + Triton kernel selection logic — how PyTorch chooses vendor backend at dispatch time
