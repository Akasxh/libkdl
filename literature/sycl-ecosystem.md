# SYCL Ecosystem Survey

*Research for LLVM Dublin 2026 poster on heterogeneous GPU kernel dispatch via MLIR.*
*Last updated: 2026-04-02*

---

## 1. SYCL 2020 Specification Overview

SYCL is a Khronos Group open standard for single-source heterogeneous C++ programming.
Originally developed as a higher-level programming model within the OpenCL working group
(provisional spec March 2014, ratified 2015), it became an independent Khronos working
group in September 2019. The current version is SYCL 2020 revision 11.

### 1.1 Key SYCL 2020 Features

- **Unified Shared Memory (USM):** Pointer-based memory management with three allocation
  types -- device (physically on device), host (physically on host but device-accessible),
  and shared (unified virtual address space with automatic migration). Supports both
  explicit and implicit data movement. This was a major addition over SYCL 1.2.1's
  buffer/accessor-only model.

- **Sub-groups:** Expose hardware SIMD lanes as a programming abstraction. Enable
  cooperative operations below the work-group level, critical for GPU performance.

- **Specialization Constants:** Compile-time constants that can be set at runtime before
  kernel launch. Enable kernel specialization without recompilation.

- **Group Algorithms:** Collective operations (reduce, scan, broadcast) across work-groups
  and sub-groups.

- **Backend Generalization:** SYCL 2020 introduced a generic backend concept, decoupling
  from OpenCL. Any acceleration API can serve as a backend while maintaining full
  interoperability with the target API.

- **Reductions:** Built-in reduction operations that the runtime can optimize per-device.

- **Kernel Bundles:** Explicit management of compiled kernel objects for fine-grained
  control over JIT/AOT compilation.

Sources:
- [SYCL 2020 Specification (Khronos)](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
- [SYCL 2020 Features (Intel)](https://www.intel.com/content/www/us/en/developer/articles/technical/sycl-2020-features-dpc-language-oneapi-c.html)
- [SYCL 2020 Reference Guide (PDF)](https://www.khronos.org/files/sycl/sycl-2020-reference-guide.pdf)

### 1.2 Device Discovery

SYCL provides device discovery through `device_selector` (SYCL 1.2.1) and the newer
aspect-based device selection (SYCL 2020). The runtime enumerates all available platforms
and devices. The `sycl::device` class exposes:
- `is_cpu()`, `is_gpu()`, `is_accelerator()`
- `get_info<>()` for detailed queries: vendor, driver version, max work-group size,
  local memory size, sub-group sizes, etc.

Built-in selectors: `default_selector_v`, `gpu_selector_v`, `cpu_selector_v`.
Custom selectors can score devices based on arbitrary criteria.

**Critical limitation:** One queue maps to exactly one device. The mapping is fixed at
queue construction and cannot change. Multi-device dispatch requires multiple queues
with explicit orchestration by the programmer.

Sources:
- [Device Discovery with SYCL (Intel)](https://www.intel.com/content/www/us/en/developer/articles/technical/device-discovery-with-sycl.html)
- [ENCCS SYCL Workshop: Device Discovery](https://enccs.github.io/sycl-workshop/device-discovery/)

### 1.3 Queue Management and Kernel Dispatch

SYCL uses a task-graph execution model:

1. **Queue:** Connects host to a single device. All work is submitted as actions to a queue.
2. **Command Group:** Groups a kernel with its data requirements (accessor declarations).
   Submitted via `queue::submit()`.
3. **Task Graph:** The runtime builds a DAG from command groups. Nodes are actions (kernel
   invocations, data transfers). Edges are data dependencies (inferred from accessors) or
   explicit dependencies.
4. **Execution:** The runtime schedules kernels as soon as all dependencies are satisfied.
   Execution is asynchronous from the host.

The runtime handles:
- Automatic data transfer between host and device based on accessor modes
- Dependency resolution and scheduling
- Kernel compilation (JIT or AOT depending on implementation)

**SYCL Graph Extension:** For AI/ML workloads with many small kernels, SYCL Graph
(command_graph) was introduced to batch kernel submissions. Benchmarks show dramatic
improvement: from ~45,665 us without graph to ~117-121 us with graph for repeated
kernel patterns.

Sources:
- [ENCCS: Queues, Command Groups, Kernels](https://enccs.github.io/sycl-workshop/queues-cgs-kernels/)
- [SYCL Graph (Intel)](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-offload-many-kernels-sycl-graph.html)

---

## 2. SYCL Implementations: DPC++ vs AdaptiveCpp vs ComputeCpp

### 2.1 Intel oneAPI DPC++ (Data Parallel C++)

**Architecture:** Built on LLVM/Clang. Hosted in the `sycl` branch of `intel/llvm`,
regularly synced with upstream LLVM main.

**Compilation model (multi-pass):**
1. Clang driver invokes device compiler once per target, then host compiler.
2. Device code is lowered to LLVM IR, then translated to SPIR-V via `llvm-spirv` tool.
3. `sycl-post-link` performs final transformations on the device LLVM IR module.
4. `clang-offload-wrapper` creates a "wrapper object" embedding device binaries.
5. Final linking produces a "fat binary" -- host binary with embedded device binaries.

**Target support:**
- Intel CPUs, Intel GPUs (Arc, Xe, Data Center GPU Max) -- primary
- NVIDIA GPUs via CUDA plugin (developed by Codeplay)
- AMD GPUs via HIP/ROCm plugin (developed by Codeplay under DOE contract)

**Strengths:** Most mature SYCL implementation. Backed by Intel with significant
investment. Integrates with oneAPI libraries (oneMKL, oneDNN, oneTBB). Best Intel
GPU support.

**Weaknesses:** Multi-pass compilation is slower. NVIDIA/AMD support via plugins is
less mature than native toolchains. Vendor-driven -- Intel priorities dominate roadmap.

Sources:
- [DPC++ Compiler Architecture](https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html)
- [DPC++ Offloading Design](https://intel.github.io/llvm/design/OffloadDesign.html)
- [intel/llvm on GitHub](https://github.com/intel/llvm)

### 2.2 AdaptiveCpp (formerly hipSYCL, Open SYCL)

**Architecture:** Community-driven, independent SYCL implementation. Academic origin
(Heidelberg University).

**Compilation model -- Generic SSCP (Single-Source, Single Compiler Pass):**

AdaptiveCpp is the only SYCL compiler that parses source code a single time for both
host and device compilation. The process:

1. **Compile time (Stage 1):** During regular host C++ compilation, AdaptiveCpp extracts
   LLVM IR for kernels with backend-independent representations of builtins and kernel
   annotations. This LLVM IR is embedded in the host binary.

2. **Runtime (Stage 2, JIT):** When kernels launch, AdaptiveCpp translates the generic
   LLVM IR to whatever the target hardware requires: PTX (NVIDIA), amdgcn (AMD, 38+
   architectures in ROCm 5.3), or SPIR-V (Intel).

**Performance:** SSCP compilation takes ~15% longer than host-only compilation, but is
>2x faster than multi-pass compilation targeting three AMD GPU architectures.

**Target support:** NVIDIA (CUDA/PTX), AMD (ROCm/amdgcn), Intel (SPIR-V/OpenCL),
CPUs (OpenMP).

**Strengths:**
- Fastest compilation. Single-pass avoids redundant parsing.
- Runtime JIT enables true hardware adaptation -- can target whatever GPU is present
  at execution time without recompilation.
- Vendor-independent. No single company controls the roadmap.
- The JIT model aligns closely with the "multi-versioned kernel dispatch" concept
  from our poster's research question.

**Weaknesses:**
- Smaller team and funding than DPC++.
- JIT compilation adds startup latency.
- Less mature library ecosystem compared to oneAPI.

**2025 development:** "Adaptivity in AdaptiveCpp: Optimizing Performance by Leveraging
Runtime Information During JIT-Compilation" (IWOCL 2025) -- using runtime hardware
information to guide JIT optimization decisions.

Sources:
- [AdaptiveCpp SSCP Architecture](https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/)
- [AdaptiveCpp GitHub](https://github.com/AdaptiveCpp/AdaptiveCpp)
- [AdaptiveCpp Compilation Docs](https://adaptivecpp.github.io/AdaptiveCpp/compilation/)
- [IWOCL 2025 paper](https://dl.acm.org/doi/10.1145/3731125.3731127)

### 2.3 ComputeCpp (Codeplay) -- Discontinued

Codeplay's proprietary SYCL implementation. Played a significant role in SYCL's early
history but development was **discontinued in 2023**. Codeplay now focuses entirely on
contributing to the open-source DPC++ project (NVIDIA and AMD plugins).

### 2.4 Other Implementations

- **triSYCL:** Research/experimental implementation. Not production-ready.
- **neoSYCL:** Academic implementation targeting NEC SX-Aurora vector processors.
- **SimSYCL:** Development/debugging/simulation implementation, not for production
  deployment (published at IWOCL 2024).

### 2.5 Implementation Comparison Summary

| Feature | DPC++ | AdaptiveCpp | ComputeCpp |
|---|---|---|---|
| Status (2025) | Active, Intel-backed | Active, community | Discontinued (2023) |
| Compilation | Multi-pass (fat binary) | Single-pass SSCP + JIT | Multi-pass |
| Intel GPU | Primary target | Via SPIR-V | Limited |
| NVIDIA GPU | Plugin (Codeplay) | Native CUDA/PTX | Supported |
| AMD GPU | Plugin (Codeplay) | Native HIP/amdgcn | Limited |
| CPU | Yes (OpenCL) | Yes (OpenMP) | Yes |
| Runtime JIT | No (AOT or driver JIT) | Yes (LLVM IR to target) | No |
| SYCL 2020 conformance | High | High (some gaps) | Partial |
| Compilation speed | Slower (multi-pass) | Faster (~15% over host) | Slow |

---

## 3. Performance Benchmarks: SYCL vs Native CUDA/HIP

### 3.1 Headline Result

The consensus from 2023-2025 literature is that **well-optimized SYCL code can match
or slightly exceed native CUDA/HIP performance** on the same hardware. The overhead is
not fundamental to the programming model but arises from toolchain differences (inlining,
unrolling, block sizes vs work-group sizes).

### 3.2 Specific Results

**Codeplay/oneAPI benchmarks (2023):**
- On NVIDIA A100, SYCL with OpenSYCL+atomics was 18% faster than CUDA+atomics.
- Across multiple benchmarks, SYCL matched or exceeded native CUDA on NVIDIA GPUs.
- Source: [Codeplay Blog](https://codeplay.com/portal/blogs/2023/04/06/sycl-performance-for-nvidia-and-amd-gpus-matches-native-system-language)

**Protein Database Search (Costanzo et al., 2023):**
- CUDA and SYCL achieved similar performance on NVIDIA devices.
- SYCL demonstrated superior architectural efficiency in 3 of 4 test cases on non-NVIDIA
  hardware.
- Tested on NVIDIA, AMD, and Intel GPUs.
- Source: [arXiv:2309.09609](https://arxiv.org/abs/2309.09609)

**HPC study on H100 (2025):**
- SYCL achieved slightly better performance than CUDA on the NVIDIA H100-based Zaratan
  system.

**Edge computing evaluation (2024):**
- 3% performance gap transitioning from CUDA to SYCL on NVIDIA hardware.
- Source: [Journal of Supercomputing](https://link.springer.com/article/10.1007/s11227-024-05957-6)

**Polybench suite (DPC++ vs AdaptiveCpp vs CUDA):**
- Disparity between DPC++ and AdaptiveCpp was negligible at ~5%.
- Both within competitive range of native CUDA.

### 3.3 Performance Sources of Variation

The primary causes of performance gaps are:
1. Memory address widths (32-bit vs 64-bit pointers)
2. Memory access widths and sub-word access patterns
3. Tuning parameters: CUDA block sizes vs SYCL local work-group sizes
4. Toolchain options: inlining thresholds, unrolling strategies
5. Backend maturity: DPC++ AMD backend showed 7x-14x scheduling overhead vs other
   backends in SYCL-Bench 2020

### 3.4 Performance Portability Scores (P3 Metric)

From the ICS 2025 evaluative comparison across Summit, Perlmutter, Frontier, Corona
supercomputers:

| Application | Kokkos | RAJA | SYCL | OpenMP |
|---|---|---|---|---|
| XSBench | 0.82 | 0.70 | 0.46 | 0.60 |
| BabelStream | 0.99 | 1.00 | 0.65 | 0.42 |
| CloverLeaf | 0.75 | 0.98 | 0.56 | 0.83 |
| su3_bench | 0.75 | 0.47 | 0.47 | 0.60 |

**SYCL consistently scored lower than Kokkos and RAJA on performance portability.**
The authors concluded: "SYCL appears to often perform worse than other programming
models" except on older Summit hardware.

On A100 specifically, Alpaka and Kokkos matched native CUDA, whereas SYCL was ~10x
slower and std::par ~2x slower.

**This is the most damaging data point against choosing SYCL for vendor-agnostic dispatch.**

Sources:
- [ICS 2025 Performance Portability Study](https://arxiv.org/html/2402.08950v1)
- [SYCL-Bench 2020](https://dl.acm.org/doi/fullHtml/10.1145/3648115.3648120)
- [OSTI Performance Portability Report](https://www.osti.gov/servlets/purl/1996690)

---

## 4. Runtime Overhead

### 4.1 Kernel Dispatch Latency

- Kernel launch latency in AdaptiveCpp/hipSYCL on NVIDIA: ~150 microseconds, comparable
  to kernel execution time of ~130 us for small kernels.
  Source: [AdaptiveCpp GitHub Issue #87](https://github.com/OpenSYCL/OpenSYCL/issues/87)

- Many small ML kernels have execution durations comparable to or smaller than launch cost.
  This is the exact problem SYCL Graph was designed to solve.

### 4.2 SYCL-Bench Runtime Scheduling Overhead

SYCL-Bench 2020 measures scheduling overhead by launching series of small kernels with
linear dependencies:

- **DPC++ AMD backend:** 7x slowdown vs AdaptiveCpp, 14x vs Tesla V100S with same
  implementation.
- **Buffer-accessor vs USM:** Different memory models show different scheduling overhead
  characteristics.
- System time (host-side runtime preparation) is the primary measurement metric.

### 4.3 SYCL Graph Mitigation

For repeated kernel patterns (common in ML inference):
- Without SYCL Graph: ~45,665 us total
- With SYCL Graph: ~117-121 us total
- This is a ~380x improvement for the graph submission path.

Sources:
- [SYCL-Bench 2020 Paper](https://dl.acm.org/doi/fullHtml/10.1145/3648115.3648120)
- [Intel SYCL Graph Article](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-offload-many-kernels-sycl-graph.html)

---

## 5. SYCL in HPC

### 5.1 Aurora (Argonne, Intel GPUs)

Aurora is the US DOE exascale supercomputer at Argonne National Laboratory based on
Intel Data Center GPU Max (Ponte Vecchio). SYCL/DPC++ is a primary programming model.
The Aurora Early Science Program has been porting scientific applications to SYCL.

### 5.2 Frontier (ORNL, AMD GPUs)

Frontier at Oak Ridge uses AMD Instinct MI250X GPUs. SYCL support came via a DOE-funded
contract awarded to Codeplay in 2021 to implement AMD GPU support in the DPC++ compiler.
A pilot port of the DPC++ SYCL compiler is available on Frontier.

### 5.3 Codeplay's Role

Codeplay (acquired by Intel in 2022) serves as the primary contractor for extending
DPC++ to non-Intel hardware:
- Developed NVIDIA GPU plugin for DPC++
- Developed AMD GPU plugin for DPC++ under DOE contract
- Previously developed ComputeCpp (now discontinued)

### 5.4 NERSC Perlmutter

NERSC, ALCF, and Codeplay partnered on SYCL support for next-generation supercomputers.
SYCL is positioned as a "write once, run on DOE's diverse GPU fleet" solution.

### 5.5 Portable Libraries

oneMKL interface library provides a single API dispatching to vendor-optimized backends:
cuBLAS (NVIDIA), rocBLAS (AMD), Intel oneMKL. This is available on Aurora, Frontier,
and Perlmutter.

Sources:
- [ALCF/ORNL Codeplay Contract](https://www.alcf.anl.gov/news/argonne-and-oak-ridge-national-laboratories-award-codeplay-software-further-strengthen-sycl)
- [Portable SYCL with oneMKL (ALCF)](https://www.alcf.anl.gov/events/portable-sycl-code-using-onemkl-amd-intel-and-nvidia-gpus)
- [NERSC/ALCF/Codeplay Partnership](https://www.hpcwire.com/off-the-wire/nersc-alcf-codeplay-partner-on-sycl-for-next-generation-supercomputers/)

---

## 6. SYCL + MLIR Integration

### 6.1 SYCL-MLIR Compiler (Tiotto et al., CGO 2024)

This is the most directly relevant work to our poster.

**Problem:** Traditional SYCL compilers lower device code to LLVM IR too early, losing
high-level SYCL semantics that could enable powerful optimizations.

**Solution:** Define a SYCL MLIR dialect representing core SYCL concepts:
- Work-item position in execution grid
- Memory access through buffers and accessors
- Kernel boundaries and launch parameters

**Architecture:**
1. Process device code Clang AST -> generate combination of MLIR dialects (including
   SYCL dialect) instead of going directly to LLVM IR.
2. Multi-step lowering through progressively lower MLIR dialects.
3. Device code module nested inside host code module (MLIR's nesting principle) ->
   enables joint host-device optimization.
4. Final lowering to LLVM IR -> SPIR-V via standard DPC++ path.

**Key innovation:** Host-device co-analysis extracts constant parameters and accessor
aliasing information from host code to optimize device code.

**Performance:** Geo-mean speedup of 1.18x over DPC++ on Intel Data Center GPU Max 1100.
Up to 4.3x speedup on individual benchmarks from the Polybench suite.

**Limitations:** MLIR-based C/C++ frontends still have limitations. Joint host-device
optimization potential not fully exploited yet.

Sources:
- [Tiotto et al., arXiv:2312.13170](https://arxiv.org/abs/2312.13170)
- [Codeplay Blog Post](https://codeplay.com/portal/blogs/2024/02/09/experiences-building-an-mlir-based-sycl-compiler)
- [Lomuller, LLVM Dev Meeting 2023 Lightning Talk (PDF)](https://llvm.org/devmtg/2023-05/slides/Lightning-Talks/02-Lomuller-SYCL-MLIR.pdf)

### 6.2 MLIR SPIR-V Dialect

MLIR has a native SPIR-V dialect with conversion paths from GPU dialect and memref
dialect. LLVM 20 promoted SPIR-V to an official backend enabled by default.

An RFC on the LLVM discourse proposes "SPIR-V IR as a vendor-agnostic GPU representation"
-- aligning with our poster's thesis.

Sources:
- [MLIR SPIR-V to LLVM Conversion](https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/)
- [LLVM Discourse RFC: SPIR-V as vendor-agnostic GPU IR](https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115)

### 6.3 IREE's Approach

IREE uses MLIR-to-SPIR-V lowering but targets Vulkan (Shader capability), not OpenCL
(Kernel capability). This means IREE's SPIR-V output is not interchangeable with SYCL's
SPIR-V path. IREE has had vendor-agnostic dispatch as an issue since project inception
(issue #50).

---

## 7. SYCL + ML Frameworks

### 7.1 PyTorch Integration (2024-present)

PyTorch 2.4 added Intel GPU support via SYCL:
- Aten operators implemented in SYCL for Intel Data Center GPU Max Series
- Integration with oneDNN for optimized kernels
- torch.compile backend for Intel GPUs via Triton
- Intel Extension for PyTorch (IPEX) uses DPC++ compiler

This is significant: it demonstrates that SYCL can serve as a backend for major ML
frameworks, not just standalone HPC codes.

### 7.2 TensorFlow

Historical TensorFlow-SYCL work exists but has not seen major updates in 2024-2025.
The ML ecosystem has largely consolidated around PyTorch.

Sources:
- [PyTorch 2.4 Intel GPU Support](https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-2-4-supports-gpus-accelerate-ai-workloads.html)
- [PyTorch SYCL Custom Operators Tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html)
- [Khronos News: PyTorch 2.4 SYCL](https://www.khronos.org/news/permalink/pytorch-2.4-using-sycl-for-deep-learning-on-intel-gpus)

---

## 8. Strengths and Weaknesses

### 8.1 Strengths

1. **Open standard:** Khronos-governed, not controlled by a single vendor. Multiple
   implementations exist.

2. **Single-source C++:** Host and device code in the same file using standard C++.
   No language extensions (unlike CUDA's `__global__`). Lower learning curve for C++
   developers.

3. **Backend generality:** SYCL 2020's generic backend concept enables targeting any
   acceleration API. Not tied to OpenCL anymore.

4. **HPC adoption:** Deployed on three DOE exascale/pre-exascale systems (Aurora,
   Frontier, Perlmutter). Backed by national lab investment.

5. **Near-native performance:** When well-tuned, SYCL matches CUDA/HIP within a few
   percent on individual hardware targets.

6. **ML framework integration:** PyTorch 2.4+ supports SYCL backend. Connection to
   real ML ecosystem exists.

7. **MLIR synergy:** SYCL-MLIR work (CGO 2024) shows up to 4.3x speedup from MLIR-based
   compilation, directly relevant to MLIR-centric dispatch.

### 8.2 Weaknesses

1. **Performance portability is poor in practice:** P3 scores of 0.46-0.65 vs Kokkos
   (0.75-0.99) and RAJA (0.47-1.00). "SYCL appears to often perform worse than other
   programming models" across diverse hardware.

2. **Queue-device binding is rigid:** One queue = one device, fixed at construction.
   No built-in multi-device dispatch, load balancing, or dynamic migration. The
   programmer must orchestrate multi-device execution manually.

3. **No binary portability:** Must recompile for each platform. Precompiled libraries
   remain vendor-specific (except AdaptiveCpp's JIT approach).

4. **Fragmented ecosystem:** Practical backend coverage depends on which implementation
   you use. DPC++ is best for Intel, AdaptiveCpp is best for multi-vendor, but neither
   covers everything equally well.

5. **Runtime overhead for small kernels:** ~150 us launch latency can dominate for
   ML inference workloads with many small operations. SYCL Graph mitigates but adds
   complexity.

6. **CPU performance gap:** SYCL is generally less efficient on CPUs compared to native
   OpenMP or vendor-specific approaches.

7. **Ecosystem maturity:** Fewer third-party libraries, less community content, and
   more variability in vendor support compared to CUDA. AMD support is community-driven,
   not AMD-official.

8. **Compile times:** Full SYCL toolchain compilation (especially multi-pass DPC++)
   is significantly slower than native CUDA compilation.

---

## 9. "Why SYCL Specifically?" -- Answering Reviewer 91D

Reviewer 91D asked: "Why SYCL specifically? The idea is more general -- multi-versioned
kernels specialized at JIT time. SPIR-V and even HIP/CUDA can do this."

### Honest Assessment

The reviewer is correct that SYCL is not uniquely positioned for vendor-agnostic dispatch.
The alternatives:

| Approach | Dispatch Model | Strengths | Weaknesses |
|---|---|---|---|
| SYCL | Runtime queue per device, manual multi-device | Open standard, C++ single-source, MLIR work exists | Poor P3 scores, rigid queue model |
| Kokkos | Compile-time backend selection | Best P3 scores, DOE adoption | No runtime hardware discovery |
| RAJA | Compile-time backend selection | Excellent P3, LLNL backing | No runtime dispatch |
| SPIR-V (direct) | IR-level portability | LLVM 20 official backend, IREE uses it | Not a programming model, needs host API |
| ALPAKA | Header-only, compile-time | Zero runtime overhead, CERN uses it | Verbose API, low adoption |
| HIP | Source portability CUDA<->AMD | AMD-native performance | Not truly vendor-agnostic |

### Where SYCL Fits in Our Poster

For the poster, SYCL should be presented as **one approach among several**, not the
primary mechanism. The honest framing:

1. **SYCL provides the runtime model** (device discovery, queue management, kernel
   submission) but has performance portability problems.

2. **AdaptiveCpp's SSCP/JIT model** is the closest existing implementation to our
   "multi-versioned kernel dispatch" concept -- it stores generic LLVM IR and JIT-compiles
   to the target at runtime. This is directly relevant.

3. **MLIR provides the compilation infrastructure** -- the SYCL-MLIR work (CGO 2024)
   demonstrates that MLIR-based compilation outperforms traditional SYCL lowering.

4. **The gap we address:** No existing system combines MLIR's high-level optimization
   with AdaptiveCpp-style runtime JIT dispatch and proper multi-device orchestration.
   SYCL's rigid queue model and poor P3 scores show that the runtime layer needs
   rethinking, not just adoption.

### Recommended Poster Framing

Instead of "SYCL-based dispatch," frame as: **"MLIR-native runtime dispatch informed
by SYCL's device model but using multi-versioned kernel JIT (a la AdaptiveCpp SSCP)
with MLIR dialect-level optimization (a la SYCL-MLIR)."**

This addresses the reviewer's critique by:
- Acknowledging SYCL's limitations honestly
- Drawing on the best ideas from SYCL (device discovery, USM) without buying the
  whole stack
- Connecting to MLIR (the venue's core technology)
- Showing awareness of alternatives (Kokkos, ALPAKA, SPIR-V)

---

## 10. Key Papers and References

### Primary Sources

1. Tiotto et al., "Experiences Building an MLIR-based SYCL Compiler," CGO 2024.
   [arXiv:2312.13170](https://arxiv.org/abs/2312.13170)
   -- SYCL-MLIR compiler with up to 4.3x speedup.

2. Alpay et al., "SYCL-Bench: A Versatile Cross-Platform Benchmark Suite," Euro-Par 2020.
   [Springer](https://link.springer.com/chapter/10.1007/978-3-030-57675-2_39)
   -- Canonical SYCL benchmark suite.

3. SYCL-Bench 2020: Benchmarking SYCL 2020 on AMD, Intel, and NVIDIA GPUs, IWOCL 2024.
   [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3648115.3648120)
   -- Updated benchmarks with runtime overhead measurements.

4. Evaluative Comparison of Performance Portability Across GPU Programming Models, ICS 2025.
   [arXiv:2402.08950](https://arxiv.org/html/2402.08950v1)
   -- P3 scores showing SYCL underperforms Kokkos/RAJA.

5. Costanzo et al., "Comparing Performance and Portability between CUDA and SYCL," 2023.
   [arXiv:2309.09609](https://arxiv.org/abs/2309.09609)
   -- CUDA vs SYCL on protein database search.

6. SYCL in the Edge: Performance and Energy Evaluation, J. Supercomputing 2024.
   [Springer](https://link.springer.com/article/10.1007/s11227-024-05957-6)
   -- 3% overhead on edge devices.

7. Homerding & Tramm, "Evaluating the Performance of hipSYCL for HPC," SYCLcon 2020.
   [ALCF](https://www.alcf.anl.gov/sites/default/files/SYCLcon_2020_Homerding_Tramm.pdf)
   -- Early hipSYCL HPC evaluation.

### Specifications and Documentation

8. [SYCL 2020 Specification, Revision 11](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
9. [DPC++ Compiler Architecture](https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html)
10. [AdaptiveCpp SSCP Documentation](https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/)

### LLVM/MLIR Integration

11. [LLVM Discourse: SPIR-V as Vendor-Agnostic GPU IR](https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115)
12. [MLIR SPIR-V Dialect](https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/)
13. [Lomuller, "Leveraging MLIR for Better SYCL Compilation," LLVM Dev Meeting 2023](https://llvm.org/devmtg/2023-05/slides/Lightning-Talks/02-Lomuller-SYCL-MLIR.pdf)

### HPC Deployments

14. [ALCF/ORNL Codeplay Contract for AMD SYCL Support](https://www.alcf.anl.gov/news/argonne-and-oak-ridge-national-laboratories-award-codeplay-software-further-strengthen-sycl)
15. [NERSC/ALCF/Codeplay Partnership](https://www.hpcwire.com/off-the-wire/nersc-alcf-codeplay-partner-on-sycl-for-next-generation-supercomputers/)

### ML Integration

16. [PyTorch 2.4 Intel GPU via SYCL](https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-2-4-supports-gpus-accelerate-ai-workloads.html)
17. [PyTorch SYCL Custom Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html)

---

## 11. Implications for Poster Design

### What to take from SYCL:
- Device discovery and introspection API design
- USM as a memory model for heterogeneous access
- The concept of backend-agnostic kernel submission
- SYCL-MLIR's proof that MLIR-based compilation outperforms traditional lowering

### What to avoid from SYCL:
- Rigid one-queue-one-device binding
- Multi-pass compilation overhead
- The assumption that a single programming model solves dispatch

### What to cite:
- AdaptiveCpp SSCP as closest prior art for MLIR-native JIT dispatch
- SYCL-MLIR (CGO 2024) as proof of MLIR compilation benefits
- P3 performance portability data to motivate why a new approach is needed
- IREE SPIR-V as an alternative architecture (per reviewer 91D)

### Open questions:
1. Can MLIR dialect-level JIT compilation match AdaptiveCpp's LLVM-IR-level JIT
   while preserving higher-level optimization opportunities?
2. What is the minimum runtime overhead for MLIR-based kernel dispatch vs SYCL's
   ~150 us launch latency?
3. Can we achieve Kokkos-level P3 scores (0.75-0.99) while retaining SYCL's runtime
   flexibility?
