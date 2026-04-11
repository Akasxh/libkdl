# Wave 05 — GPU Kernel JIT Compilation
**Angle:** gpu-kernel-jit-compilation
**Query:** GPU kernel JIT compilation runtime LLVM NVPTX AMDGPU backend
**Date:** 2026-04-06

---

## Summary

GPU kernel JIT compilation is a mature but fragmented space. The primary paths are: (1) NVRTC for CUDA source-to-PTX at runtime, (2) AMD HIPRTC for HIP source-to-AMDGCN at runtime, (3) MLIR's `mgpuModuleLoadJIT` for PTX-via-driver JIT in the GPU dialect (NVIDIA-only), (4) Proteus for LLVM-IR-level runtime specialization portable across CUDA and HIP, and (5) OCCA for directive-based JIT across CUDA/HIP/OpenCL/Metal/OpenMP. Cold-start JIT overhead ranges from ~600 ms (single NVRTC kernel) to 8+ seconds (cuTENSOR complex plans). Once kernels are cached, re-launch overhead drops to 2-3 ms. The key insight for libkdl: JIT is not just a compilation strategy but a **dispatch correctness mechanism** — it enables architecture-adaptive code selection at runtime, which is exactly the problem libkdl solves at the binary level without requiring re-compilation.

---

## JIT Compilation Paths

### 1. NVRTC (NVIDIA Runtime Compilation)
- Compiles CUDA C++ source strings directly to PTX at runtime, without spawning nvcc as a subprocess
- API: `nvrtcCreateProgram` → `nvrtcCompileProgram` → `nvrtcGetPTX` → `cuModuleLoadData` (CUDA driver loads PTX, JITs to SASS)
- Overhead (cold): ~600 ms per kernel (cuDF measurements)
- Overhead (cached): ~3 ms per kernel (cuDF, 130 KB cached binary)
- Speedup over nvcc for analytical queries: 8x faster compilation throughput
- Cache path: `LIBCUDF_KERNEL_CACHE_PATH` env var; also CUDA driver's own implicit PTX cache (`~/.nv/ComputeCache/`)
- **Jitify** (NVIDIA/jitify): single-header wrapper over NVRTC providing `JitCache`, serialization, template instantiation, automatic header resolution
  - ROCm fork (ROCm/jitify) ports Jitify to HIPRTC for AMD
  - The two forks are maintained separately — no unified abstraction

### 2. HIPRTC (AMD HIP Runtime Compilation)
- Mirrors NVRTC API: `hiprtcCreateProgram` → `hiprtcCompileProgram` → `hiprtcGetCode` → `hipModuleLoadData`
- AMD LLVM compiles HIP source → LLVM IR → AMDGCN assembly → HSACO binary
- Triton on AMD uses this path: `@triton.jit` decorator → JITFunction → HIPRTC → hsaco
- AMD comgr (Code Object Manager) handles the low-level IR→binary finalization; sits below HIPRTC
- No `mgpuModuleLoadJIT` equivalent in MLIR for AMD: asymmetric gap vs NVIDIA path

### 3. MLIR GPU Dialect: `mgpuModuleLoadJIT`
- Added in LLVM PR #66220 (fabianmcg) — NVIDIA-only
- Enables PTX-format GPU modules to be loaded and JIT-compiled by the NVIDIA driver at runtime
- Three formats via `GPUTargetAttrInterface.createObject`:
  - `ISA` (PTX) — JIT by driver at load time
  - `Binary` (cubin) — AOT for specific arch
  - `Fatbin` — default, embeds both cubin + PTX
- The PTX/JIT path allows tests to pass across different compute capabilities without recompilation
- **Admitted overhead**: PR notes "significant runtime performance hit" vs fatbin
- AMD gap: no `mgpuModuleLoadJIT` equivalent; AMD path must go through `hipModuleLoad` with pre-compiled HSACO

### 4. Proteus (LLNL/Olympus-HPC, CGO 2025)
- Language-agnostic LLVM IR-level JIT, portable across CUDA (NVRTC) and HIP (HIPRTC)
- Core technique: **runtime constant folding** — replaces kernel arguments, launch dimensions, and loop bounds with runtime constants, enabling classical optimizations (loop unrolling, CFG simplification, constant propagation) that AOT cannot apply
- Three integration modes: `__attribute__((annotate("jit", ...)))` annotations, DSL API, string-based C++ frontend
- Caching: in-memory + persistent disk cache; first-run specializes, subsequent runs reuse
- Performance vs AOT:
  - AMD GPUs: up to **2.8x speedup** end-to-end
  - NVIDIA GPUs: up to **1.78x speedup** end-to-end
  - vs Jitify (NVIDIA-specific): Proteus achieves **1.23x higher end-to-end speedup** on average due to IR-level vs source-level specialization
- RAJA integration: Proteus embedded in LLNL's RAJA Portability Suite; two mission-critical LLNL applications already using it
- SC25 result: no slowdowns on CUDA or HIP backends for RAJAPerf benchmarks with Proteus JIT enabled

### 5. OCCA (Argonne/libocca)
- Directive-based portable kernel language (OKL — OCCA Kernel Language, extension of C)
- JIT compiles OKL at runtime to CUDA, HIP, OpenCL, Metal, OpenMP backends
- Selects target backend at runtime → true vendor-agnostic dispatch via JIT
- Performance: faster than other frameworks for small-scale problems; JIT enables runtime-informed optimization
- Limitation: low HPC adoption (only 3 identified codes with OCCA ports); OKL portability layer adds abstraction overhead

### 6. cuTENSOR / CUTLASS JIT
- cuTENSOR JIT: triggered during `cutensorCreatePlan()` (blocking)
  - Cold compilation: 1–8 seconds depending on kernel complexity and host CPU
  - Cached: shared across library handles, writable to disk via `cutensorWriteKernelCacheToFile()`
  - Cache invalidation: cuTENSOR version, CUDA version, or GPU model mismatch
  - JIT speedup vs pre-compiled: **6.9x** on H100 PCIe (774 → 5374 GFLOPs/s) for specific contraction plans
- CUTLASS DSL: Python → custom IR → MLIR → PTX/SASS via JIT; Apache TVM-FFI for reduced host overhead

---

## JIT vs AOT: Quantitative Overhead Summary

| System | Cold JIT Latency | Cached Latency | Peak Speedup vs AOT |
|--------|-----------------|----------------|---------------------|
| cuDF/NVRTC (single kernel) | ~600 ms | ~3 ms | 1–4x (string transforms) |
| CUDA driver PTX JIT | 17 ms (simple kernel) | 2 ms (SASS cache) | architecture-adaptive |
| cuTENSOR plan | 1–8 seconds | ~few ms | 6.9x (H100, specific plan) |
| Proteus/LLVM-IR | JIT in background thread | cached | 2.8x AMD, 1.78x NVIDIA |
| NVRTC 64-kernel batch | ~5 seconds first run | few ms | — |

**Key insight**: Cold JIT overhead is always paid once per kernel per process (or per architecture). The breakeven depends on data volume: cuDF breaks even at ~1–3B rows (batch 100M). For long-running ML training jobs, JIT amortizes easily; for short-lived inference, cache pre-population is critical.

---

## Relevance to libkdl

libkdl operates at the **pre-compiled binary dispatch** level — selecting among fat-binary objects or pre-compiled variants. This is architecturally distinct from but complementary to JIT:

1. **JIT as a fallback path in libkdl**: When no pre-compiled variant matches the current device (e.g., new GPU arch), libkdl could invoke NVRTC/HIPRTC as a fallback, caching the result for subsequent dispatches. The `mgpuModuleLoadJIT` PR shows LLVM infrastructure already supports this three-tier approach (fatbin default, cubin for known arch, PTX/JIT as fallback).

2. **Cache sharing**: libkdl's dispatch table and JIT cache can be unified — the `dl_open` analog for GPU kernels would check in-memory dispatch table first, persistent binary cache second, JIT compile third.

3. **The AMD asymmetry is a libkdl concern**: MLIR has `mgpuModuleLoadJIT` only for NVIDIA. AMD requires pre-compiled HSACO. libkdl's value proposition on AMD is higher because there is no driver-level PTX JIT fallback — libkdl must ship the right binary or have a comgr-backed compilation path.

4. **Proteus validates the IR-level approach**: Proteus shows that operating on LLVM IR (not source, not PTX) is the portable JIT abstraction. libkdl could embed IR bitcode in kernel objects alongside pre-compiled binaries, enabling Proteus-style specialization without source distribution.

---

## Sources

- [NVRTC 13.2 Documentation — NVIDIA](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [Efficient Transforms in cuDF Using JIT Compilation — NVIDIA Blog](https://developer.nvidia.com/blog/efficient-transforms-in-cudf-using-jit-compilation/)
- [Just-In-Time Compiled CUDA Kernel — Saurabh S. Sawant (2024)](https://saurabh-s-sawant.github.io/blog/2024/GPU-JIT/)
- [Give a JIT on GPUs: NVRTC for Code-Generating Database Systems — IEEE (2024)](https://ieeexplore.ieee.org/document/10555104/)
- [NVIDIA/jitify — GitHub](https://github.com/NVIDIA/jitify)
- [ROCm/jitify — GitHub](https://github.com/ROCm/jitify)
- [CUDA Pro Tip: Fat Binaries and JIT Caching — NVIDIA Blog](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
- [Just In Time Compilation — cuTENSOR Docs](https://docs.nvidia.com/cuda/cutensor/latest/just_in_time_compilation.html)
- [JIT Compilation and Caching — CUTLASS DeepWiki](https://deepwiki.com/NVIDIA/cutlass/3.3-jit-compilation-and-caching)
- [Proteus: Portable Runtime Optimization of GPU Kernels (CGO 2025) — ACM DL](https://dl.acm.org/doi/10.1145/3696443.3708939)
- [Proteus — CGO 2025 Conference Page](https://2025.cgo.org/details/cgo-2025-papers/24/Proteus-Portable-Runtime-Optimization-of-GPU-Kernel-Execution-with-Just-In-Time-Comp)
- [LLNL Researchers Develop GPU JIT Compiler — HPCwire](https://www.hpcwire.com/off-the-wire/llnl-researchers-develop-gpu-jit-compiler-for-large-scale-hpc-applications/)
- [Olympus-HPC/proteus — GitHub](https://github.com/Olympus-HPC/proteus)
- [Extending RAJA with JIT Optimization — SC25 Workshop (ACM DL)](https://dl.acm.org/doi/10.1145/3731599.3767492)
- [[mlir][gpu][NVPTX] Enable NVIDIA GPU JIT compilation path — LLVM PR #66220](https://github.com/llvm/llvm-project/pull/66220)
- [Offloading Design & Internals — Clang Documentation](https://clang.llvm.org/docs/OffloadingDesign.html)
- [HIP Compilers — ROCm Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html)
- [libocca/occa — GitHub](https://github.com/libocca/occa)
- [Implementing Multi-GPU Miniapps Across Portability Frameworks — arXiv 2511.02655](https://arxiv.org/html/2511.02655)
- [Just-In-Time Compilation in SYCL — Intel oneAPI Docs](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/just-in-time-compilation-in-sycl.html)
- [OpenCL JIT Compilation Benchmarks — Karl Rupp (2016)](https://www.karlrupp.net/2016/01/opencl-just-in-time-jit-compilation-benchmarks/)

---

## Angle Assessment

**Relevance: 9/10**
JIT compilation is the alternative to libkdl's pre-compiled dispatch model. Understanding where JIT excels (architecture adaptation, constant folding speedups) and where it fails (cold-start latency, no AMD driver JIT fallback) directly defines the design space libkdl occupies and the claims it can make.

**Novelty: 7/10**
NVRTC and Jitify are well-established. The novel material is: (1) Proteus (CGO 2025) showing portable LLVM-IR JIT with 2.8x speedups, (2) MLIR's asymmetric JIT support (NVIDIA only, PR #66220), and (3) the quantified cold/warm latency gap that makes pre-compiled dispatch + libkdl-style dynamic linking a better default for production inference. The AMD comgr gap (no driver-level PTX JIT fallback) is an underappreciated asymmetry that strengthens libkdl's AMD value case.

**Key Gap Identified:**
No existing work unifies pre-compiled binary dispatch (libkdl's model) with JIT fallback into a single runtime abstraction. MLIR has per-vendor JIT paths; Proteus operates at IR level but requires annotation/recompilation of source; Jitify requires CUDA source strings. libkdl operating on pre-compiled fat-binary objects with optional IR bitcode embedding would be a novel layer between full AOT (no flexibility) and full JIT (high latency).
