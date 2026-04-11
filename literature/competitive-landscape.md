# Competitive Landscape: Heterogeneous GPU Dispatch Solutions
## Research for LLVM Dublin 2026 Poster
**Last updated:** 2026-04-10

---

## Overview

This document maps all major existing solutions for heterogeneous GPU kernel dispatch and performance portability, structured for contrast against a novel dynamic runtime dispatch contribution. Solutions are grouped by their fundamental dispatch strategy: compile-time static selection, runtime dispatch, hybrid, or lightweight abstraction.

---

## 1. Compile-Time Approaches

### 1.1 Triton (OpenAI)

**Architecture Summary**
Triton is a Python DSL and MLIR-based compiler that lowers user-written tile-level programs to GPU machine code. The programmer operates on blocks of data ("Blocked Program, Scalar Threads"), and the compiler handles thread-level parallelism decomposition, shared memory allocation, synchronization, and instruction selection automatically. Version 2.0 (2022) rewrote the backend entirely in MLIR; the current release is 3.6.0 (January 2026).

The compilation pipeline proceeds through a stack of custom MLIR dialects:
- `tt` (core Triton operations)
- `ttg` (GPU-generic tile and layout ops)
- `ttng` / `nvws` (NVIDIA-specific warp specialization)
- `nvg` (NVIDIA GPU lowering)
- `TritonAMDGPUOps` (AMD-specific lowering)
- Final lowering to NVVM IR (PTX) or ROCDL IR (AMDGCN)

**Portability Scope**
- NVIDIA GPUs: Compute Capability 8.0+ (Ampere and newer)
- AMD GPUs: ROCm 6.2+
- CPU: Experimental / under development
- Intel GPU: Not supported
- Platform: Linux only (no Windows/macOS)

**Performance Overhead vs Native**
Triton targets parity with handwritten CUDA/HIP for compute-bound workloads. Automatic optimizations include coalescing, thread swizzling, prefetching, vectorization, tensor core-aware instruction selection, shared memory management, and async copy scheduling. In practice, Triton-generated FlashAttention kernels match or exceed cuBLAS-level performance for standard shapes. Overhead relative to hand-tuned Cutlass is workload-dependent.

**MLIR Integration**
Deep and native. Triton's entire backend is MLIR. Nine custom dialects drive the compilation pipeline; there is no path to PTX or AMDGCN that bypasses MLIR. This is the most MLIR-native system in the ML compiler space.

**Developer Experience**
High productivity for kernel authors familiar with Python. The tiled programming model is narrower than CUDA — developers write block-level logic, not thread-level logic. Debugging is harder than CUDA due to opaque compilation stages, but the `MLIR_ENABLE_DUMP` environment variable exposes per-pass IR. No turnkey test suite exists.

**Active Development Status**
Highly active. 5,943+ commits, version 3.6.0 released January 2026, 855 open issues. Backed by OpenAI; used as the compiler for PyTorch 2.x `torch.compile` on GPU.

**Key Limitations**
- Requires relatively modern NVIDIA hardware (CC 8.0+)
- No Intel GPU support
- Linux-only
- CPU backend experimental
- Kernel granularity locked to tile abstraction — not suitable for irregular compute patterns
- LLVM API instability makes custom LLVM version pinning necessary
- No dynamic kernel selection at runtime: backend chosen at compile time

---

### 1.2 TVM (Apache)

**Architecture Summary**
TVM is a general-purpose ML compiler with a two-level IR: `relax::Function` (high-level computational graph with control flow) and `tir::PrimFunc` (low-level tensor operations with explicit loop structures and threading). Compilation proceeds through model import (from PyTorch, ONNX, TensorFlow), graph-level optimization, and TensorIR schedule optimization via MetaSchedule (auto-tuning) or DLight (tuning-free). Code generation targets LLVM for CPUs and device-specific backends for GPUs.

**Portability Scope**
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/OpenCL)
- Intel GPUs (OpenCL)
- CPUs: x86, ARM via LLVM
- Vulkan (experimental)
- WebGPU (experimental)
- Metal (macOS/iOS)

**Performance Overhead vs Native**
Auto-tuning via MetaSchedule can match cuBLAS-level performance for static-shape workloads after sufficient tuning time. DLight provides tuning-free reasonable performance for dynamic shapes. Untuned schedules can be 2-5x slower than optimal. Tuning time can be hours for large models.

**MLIR Integration**
Limited. TVM predates MLIR and has its own IR stack. There is an experimental MLIR ingestion path, but TVM does not use MLIR as its primary IR or backend. The Apache TVM project has discussed MLIR integration but it is not production-ready.

**Developer Experience**
High initial complexity. Requires understanding TensorIR scheduling semantics. Schedule writing is manual unless MetaSchedule is used. PyTorch and ONNX frontends ease model import. Active Python and C++ APIs.

**Active Development Status**
Active. Apache top-level project, v0.18+ maintained. Used in production by AWS (TVM on SageMaker), Qualcomm, and others.

**Key Limitations**
- MetaSchedule auto-tuning restricted to static shapes
- No native MLIR IR; integration is shallow
- Tuning is expensive (hours of search)
- Complex scheduling API for custom kernels
- No runtime dispatch — backend selected at compile time
- DLight tuning-free mode trades performance for convenience

---

### 1.3 XLA (Google / OpenXLA)

**Architecture Summary**
XLA (Accelerated Linear Algebra) is a whole-program ML compiler operating on StableHLO as its portable IR. Compilation proceeds through target-independent optimization passes (CSE, operation fusion, buffer analysis), then backend-specific optimizations, then LLVM-based code generation. The GPU backend fuses operations suited to the GPU threading model before lowering to NVPTX (NVIDIA) or equivalent paths for other targets.

**Portability Scope**
- NVIDIA GPUs: LLVM NVPTX backend (primary GPU target)
- AMD GPUs: ROCm (via OpenXLA project; production quality as of 2024)
- TPUs: Google custom backend (primary design target)
- CPUs: LLVM
- Intel GPUs: In development

**Performance Overhead vs Native**
Near-native for fused workloads. XLA's fusion engine can match hand-tuned cuDNN for training workloads. AutoTuning (XProf/persisted autotuning) further reduces overhead. Overhead primarily from untuned fusion decisions for non-standard shapes.

**MLIR Integration**
First-class. StableHLO is an MLIR dialect and serves as the canonical IR entering XLA. The XLA team has been migrating internal representations to MLIR. `hlo-to-mlir` and `mlir-to-hlo` bridges exist. The MHLO (MLIR HLO) dialect is widely used. XLA leverages MLIR "to bring the best capabilities into a single compiler toolchain."

**Developer Experience**
Not intended as a kernel-authoring interface. Users program in JAX, TensorFlow, or PyTorch (torch-xla), and XLA compiles transparently. Debugging opaque JIT compilation is challenging; `XLA_FLAGS=--xla_dump_to=...` helps. No manual scheduling.

**Active Development Status**
Highly active. Production backbone for Google TPU training, JAX ecosystem, and TensorFlow. OpenXLA is the open-source steward (founded 2022).

**Key Limitations**
- Not a kernel-authoring interface; developer cannot write custom kernels in XLA IR
- Compilation latency can be significant for first execution
- AMD ROCm path less mature than CUDA
- No dynamic backend selection at runtime
- Shape dynamism limited — XLA prefers static shapes (traced programs)
- Tight coupling to JAX/TF/PyTorch-XLA front-ends

---

### 1.4 IREE (MLIR-native ML Runtime)

**Architecture Summary**
IREE (Intermediate Representation Execution Environment) is a holistic MLIR-based compiler and runtime for ML models. It combines scheduling and execution into a single compilation unit. The compiler operates through a dialect stack: model import to `Flow` dialect (data flow and partitioning), then `Stream` dialect (execution partitioning and scheduling), then `HAL` (Hardware Abstraction Layer) dialect (hardware-specific operations), then target-specific code generation (IREEGPU, IREECPU, etc.). The runtime is thin and designed for deployment in both datacenter and embedded (30KB binary) contexts.

**Portability Scope**
- NVIDIA GPUs: CUDA backend (full production support)
- AMD GPUs: ROCm/HIP backend
- Cross-vendor: Vulkan backend (NVIDIA, AMD, Intel, mobile via SPIR-V)
- Apple Silicon: Metal backend
- CPUs: ARM, x86, RISC-V
- WebGPU (experimental)
- AMD AIE (experimental)
- Platforms: Linux, Windows, macOS, Android, iOS, bare metal

**Performance Overhead vs Native**
Competitive for standard ML workloads. The CUDA backend supports sm_60 through sm_90 with architecture-specific tuning. Optimization levels O0-O3 available. Performance trails hand-tuned libraries for highly specialized operations but competitive with framework-compiled code.

**MLIR Integration**
Native and foundational. IREE is built on MLIR from the ground up. The entire pipeline from model ingestion to hardware binary is expressed in MLIR dialects: Flow, Stream, HAL, IREE-specific codegen dialects (IREEGPU, IREECPU, IREECodegen), and final lowering to LLVM IR or SPIR-V. The HAL dialect is the key abstraction enabling multi-target dispatch.

**Developer Experience**
Moderate. ML framework users (TensorFlow, PyTorch, JAX, ONNX) interact through import tools that produce MLIR. Direct MLIR authoring is possible for advanced users. The HAL abstraction makes target-specific code invisible to model authors. Runtime bindings exist for Python, C, and C++.

**Active Development Status**
Active. Supported by Google and an open-source community. Used in production for inference deployment at Google. Latest releases in 2025-2026.

**Key Limitations**
- Primarily an inference compiler; training workflows less developed
- HAL abstraction has overhead for very fine-grained kernel dispatch
- Dynamic shape support exists but is more complex than static compilation
- Custom kernel injection requires understanding the MLIR dialect stack
- AMD and Vulkan paths less optimized than CUDA path
- No dynamic backend selection at runtime — target chosen at compile time (though Vulkan enables cross-vendor deployment at runtime via SPIR-V)

---

## 2. Runtime Approaches

### 2.1 SYCL (DPC++ / AdaptiveCpp)

**Architecture Summary**
SYCL (Specification for Unified Kernels for C++) is a Khronos standard (SYCL 2020 is current) defining a single-source C++ programming model for heterogeneous computing. Kernels are written in standard C++ and compiled by SYCL implementations to run on diverse devices. Intel's DPC++ (built on LLVM) and AdaptiveCpp (formerly hipSYCL, multi-compiler) are the two main implementations. The device selector mechanism allows runtime selection of execution targets: `gpu_selector_v`, `cpu_selector_v`, or custom selectors querying platform/device properties. SPIR-V is the portable intermediate representation for GPU targets.

**Portability Scope (DPC++)**
- Intel GPUs: Arc, Data Center GPU Flex, Data Center GPU Max, Iris Xe, UHD (native)
- NVIDIA GPUs: via Codeplay oneAPI plugin (CUDA PTX path)
- AMD GPUs: via Codeplay oneAPI plugin (HIP path)
- CPUs: all major ISAs (x86, ARM) via OpenMP or native host
- FPGAs: Intel FPGAs via OpenCL backend

**Portability Scope (AdaptiveCpp)**
- CPUs and GPUs from Intel, NVIDIA, AMD, Apple
- Multipass compilation: separate binaries per target, combined at link time
- SSCP (Single-Source Single-Compiler Pass): single compilation produces multi-target binary via JIT at runtime

**Performance Overhead vs Native**
DPC++ on Intel hardware: near-native. On NVIDIA via Codeplay plugin: 5-15% overhead vs native CUDA for compute-bound kernels (plugin adds indirection through Level Zero / SPIR-V translation). AdaptiveCpp SSCP mode adds JIT cost on first execution but achieves near-native performance thereafter. SYCL unified shared memory (USM) avoids some buffer copy overhead relative to SYCL 1.2 buffer/accessor model.

**MLIR Integration**
None in the SYCL standard. DPC++ compiles through LLVM, not MLIR. Some research projects (e.g., Polygeist, MLIR-based SYCL frontends) exist but are not production-ready. AdaptiveCpp has no documented MLIR path.

**Developer Experience**
Standard C++ with queue/kernel lambda model. Familiar to C++ developers; kernel code restrictions (no virtual functions, no RTTI, no exceptions inside kernels) require adaptation. Device selector API is clean. Debugging across vendors is harder than single-vendor CUDA.

**Active Development Status**
Active. SYCL 2020 ratified. DPC++ maintained by Intel as part of oneAPI. AdaptiveCpp actively developed (2025-2026 releases). Khronos SYCL working group ongoing.

**Key Limitations**
- NVIDIA/AMD support via plugins; not first-class citizens in Intel DPC++
- SPIR-V translation overhead for non-Intel GPU targets
- No MLIR integration
- Kernel restrictions limit expressiveness relative to CUDA
- Cross-vendor runtime selection exists but compile-time backend switching still requires recompilation for some paths
- Performance portability not guaranteed across vendors without tuning

---

### 2.2 OpenCL

**Architecture Summary**
OpenCL (Open Computing Language) is the original Khronos standard for heterogeneous computing. The host-device model separates host C/C++ code from device kernels written in OpenCL C (or C++ for OpenCL). Kernels are compiled at runtime (online) or offline to SPIR-V. The runtime selects from available platforms and devices via `clGetPlatformIDs` / `clGetDeviceIDs`. OpenCL 3.0 (current) adopts a modular optional-feature model to improve deployment flexibility.

**Portability Scope**
- CPUs: multi-core (AMD, Intel, ARM) via Pocl, Intel OpenCL SDK
- NVIDIA GPUs: via NVIDIA OpenCL runtime
- AMD GPUs: via ROCm OpenCL (clr)
- Intel GPUs: via Intel OpenCL runtime
- DSPs, FPGAs, AI accelerators: vendor-specific runtimes
- `clspv`/`clvk`: OpenCL over Vulkan for platforms without native drivers

**Performance Overhead vs Native**
Generally 10-30% overhead vs CUDA for GPU workloads due to runtime compilation, less aggressive optimization, and more conservative memory model. Offline SPIR-V compilation narrows the gap. Modern AMD/Intel implementations are more competitive; NVIDIA's OpenCL support is legacy and under-optimized.

**MLIR Integration**
None native. MLIR has a SPIR-V dialect that targets OpenCL execution via `cl.kernel` decorations; IREE and other tools use this path for CPU/embedded targets. OpenCL itself has no MLIR awareness.

**Developer Experience**
Verbose and low-level. Boilerplate for platform/device enumeration, context creation, program compilation, and kernel launch is substantial. OpenCL C is more restrictive than CUDA C. The buffer model (explicit memory objects) is cumbersome vs. CUDA unified memory. Generally considered developer-hostile relative to modern alternatives.

**Active Development Status**
Maintained but not growing. OpenCL 3.0.19 is the latest maintenance release. NVIDIA has de-prioritized OpenCL in favor of CUDA. Industry momentum has shifted to SYCL and Vulkan Compute. Still important for legacy systems, FPGAs, and embedded targets.

**Key Limitations**
- NVIDIA support is legacy and unmaintained
- Verbose programming model
- Online compilation adds startup latency
- No MLIR integration
- Feature fragmentation across vendors (OpenCL 3.0 optional features)
- Lower performance ceiling than CUDA/HIP due to conservative optimization assumptions
- Largely superseded by SYCL and Vulkan for new GPU work

---

### 2.3 Vulkan Compute

**Architecture Summary**
Vulkan is the modern Khronos low-level graphics and compute API. Compute pipelines (`VkPipeline` with `VK_PIPELINE_STAGE_COMPUTE_BIT`) execute SPIR-V shader/compute modules on any Vulkan-conformant device. SPIR-V is the universal binary IR consumed by Vulkan drivers, which then translate to hardware-specific ISA (PTX→SASS for NVIDIA, GCN/RDNA for AMD, etc.). The Vulkan HAL in IREE and WebGPU's backend both use this path for cross-vendor GPU execution.

**Portability Scope**
- NVIDIA GPUs: all Vulkan-conformant cards (GTX 10xx and newer)
- AMD GPUs: all GCN and RDNA cards
- Intel GPUs: Gen9 and newer
- ARM Mali, Qualcomm Adreno, Apple (via MoltenVK)
- Broadest hardware coverage of any GPU compute API

**Performance Overhead vs Native**
Close to native on AMD and Intel where Vulkan is the primary compute API. On NVIDIA, 5-20% overhead vs CUDA because NVIDIA optimizes for CUDA first; SPIR-V→SASS translation quality is improving but lags CUDA. Subgroup operations and memory barriers in Vulkan Compute perform comparably to CUDA warp intrinsics on most hardware.

**MLIR Integration**
Strong indirect integration. MLIR has a dedicated SPIR-V dialect (`mlir::spirv`) that compiles to the SPIR-V binary format consumed by Vulkan. IREE's Vulkan backend uses this path. The MLIR GPU dialect can lower to SPIR-V for Vulkan dispatch. This makes Vulkan + SPIR-V + MLIR a natural multi-vendor GPU target.

**Developer Experience**
Very verbose and explicit. Vulkan Compute requires manual pipeline creation, descriptor set management, synchronization (barriers, semaphores), and command buffer recording. Intended for library/framework authors, not end users. Tools like `vkFFT` and `VkCompute` provide higher-level abstractions. SPIR-V assembly is not human-writable; GLSL or HLSL-to-SPIR-V compilation (via `glslang`/`dxc`) is the standard path.

**Active Development Status**
Highly active. Vulkan 1.3+ is current. Strong adoption on mobile (Android), console (Nintendo Switch), and cross-platform desktop. WebGPU standardizes over Vulkan/Metal/D3D12. WebGPU compute shaders use WGSL compiled to SPIR-V.

**Key Limitations**
- Verbose, low-level API not suitable for ML kernel authors directly
- NVIDIA performance trails CUDA
- No MLIR integration at the API level; integration only through SPIR-V dialect
- No dynamic kernel dispatch across vendors in a single binary without pre-compilation of all targets
- Lack of native support for tensor/matrix operations (no equivalent to CUDA tensor cores at API level)
- Limited subgroup support fragmentation across vendors

---

## 3. Hybrid Approaches

### 3.1 ALPAKA

**Architecture Summary**
ALPAKA (Abstraction Library for Parallel Kernel Acceleration) is a header-only C++20 library that provides performance portability through abstraction rather than hiding of the underlying parallelism model. Kernels are written as C++ function objects following the CUDA grid-blocks-threads decomposition, but using ALPAKA's API instead of CUDA intrinsics. The correct backend implementation is instantiated via C++ template specialization at compile time; no runtime dispatch overhead exists because backend selection is resolved by the compiler. Backend switching requires recompilation with different CMake flags.

**Portability Scope**
- NVIDIA GPUs: CUDA 12.0+ (nvcc or clang as CUDA compiler)
- AMD GPUs: HIP 6.0+
- Intel GPUs: SYCL/oneAPI 2024.2+
- CPUs: OpenMP 2.0+, std::thread, TBB 2.2+, serial (single-core)
- Header-only: no pre-built runtime library required

**Performance Overhead vs Native**
Minimal at steady state. Since backend selection is compile-time via C++ templates, there is no runtime dispatch overhead. The abstraction layers are zero-cost in the C++ sense — template instantiation produces backend-specific code equivalent to writing native CUDA/HIP directly. In practice, minor overhead from the function-object kernel interface vs. `__global__` functions has been measured at < 2% in HPC benchmarks.

**MLIR Integration**
None. ALPAKA is a pure C++ template library with no MLIR awareness. Integration would require an MLIR-to-C++ code generation path.

**Developer Experience**
Moderate. The ALPAKA programming model is close to CUDA, easing adoption for GPU-familiar developers. CMake integration via `alpaka_add_executable` is required; switching backends is a CMake reconfiguration. Single kernel implementation works across all backends. Limitation: `clang` as CUDA compiler cannot combine with OpenMP backends.

**Active Development Status**
Active. Developed at CERN (EP-SFT group) and Helmholtz centers (HZDR, etc.). Used in high-energy physics simulation (CMSSW, Alpaka-based detector simulation). Latest version supports C++20 and HIP 6.0+.

**Key Limitations**
- Compile-time backend selection only — no runtime dispatch across GPU vendors
- Recompilation required to switch backends
- Clang + CUDA + OpenMP combination unsupported
- No MLIR integration
- Abstraction is inherently CUDA-shaped; non-CUDA-like accelerators (e.g., RISC-V vector, dataflow accelerators) require new backend development
- No auto-tuning; performance depends on developer-specified parallelism parameters

---

### 3.2 Kokkos

**Architecture Summary**
Kokkos is a C++ performance portability ecosystem (Core + Kernels + Tools) developed at Sandia National Laboratories (SNL) and maintained as a Linux Foundation project. The programming model requires developers to express algorithms using Kokkos execution policies and memory spaces, which the framework maps to target architectures. Data layout policies (e.g., `LayoutLeft` vs. `LayoutRight`) enable memory access pattern optimization per backend. Backend selection is compile-time via CMake configuration, producing a single-backend binary.

**Portability Scope**
- NVIDIA GPUs: CUDA
- AMD GPUs: HIP
- Intel GPUs: SYCL/Level Zero
- CPUs: OpenMP, HPX, C++ threads
- Additional backends in development

**Performance Overhead vs Native**
Near-native. Kokkos is production-deployed in major HPC codes (Trilinos, Sierra, Nalu-Wind) with performance competitive with native CUDA/HIP implementations. The data layout abstraction is the key differentiator: layout policies ensure coalesced access per backend without code changes. Measured overhead in HPC benchmarks typically < 5%.

**MLIR Integration**
None. Kokkos is a C++ abstraction library with no MLIR integration. Future work on Kokkos-to-MLIR translation has been proposed in research but is not implemented.

**Developer Experience**
Moderate-to-high learning curve. Developers must understand execution spaces, memory spaces, and layout policies — concepts not present in CUDA. The reward is a single codebase running across all supported backends. Kokkos Kernels provides pre-optimized linear algebra and graph algorithms. Kokkos Tools enables profiling without code changes.

**Active Development Status**
Highly active. Version 5.1.0 released March 2026. 15,300+ commits, Linux Foundation project, OpenSSF certified. Used in production at DoE national labs and major HPC centers. C++20 required as of v5.0.

**Key Limitations**
- Compile-time backend selection only — one backend per binary
- Must recompile to switch GPU vendor
- No MLIR integration
- Higher abstraction overhead in developer understanding vs. CUDA
- Data layout abstraction can conflict with existing codebases using raw pointers
- No native support for ML-specific operations (tensor contractions, attention); relies on Kokkos Kernels library

---

### 3.3 RAJA

**Architecture Summary**
RAJA is a C++ performance portability library from Lawrence Livermore National Laboratory (LLNL). It decouples loop bodies (expressed as C++ lambda functions) from execution policies (expressed as C++ template parameters). A loop using `RAJA::forall<ExecPolicy>(range, body)` can switch backends by changing `ExecPolicy` without modifying the loop body. RAJA does not provide data management (no equivalent to Kokkos memory spaces); it assumes developers manage memory separately.

**Portability Scope**
- NVIDIA GPUs: CUDA (fully supported)
- AMD GPUs: HIP (fully supported)
- CPUs: Sequential, OpenMP multithreading, SIMD (experimental), TBB (experimental)
- Accelerator offload: OpenMP target offload (experimental), SYCL (experimental)

**Performance Overhead vs Native**
Aims for performance parity with direct use of underlying programming models (OpenMP, CUDA, HIP). Lambda capture overhead is typically negligible with modern compilers. For GPU workloads, RAJA adds minimal abstraction beyond CUDA/HIP kernel launch. Measured overhead < 3% vs. native CUDA in DoE application benchmarks.

**MLIR Integration**
None documented.

**Developer Experience**
Lower barrier than Kokkos for existing C++ developers — no need to learn memory space concepts. Lambda-based policy model is familiar to C++14+ developers. Downside: no integrated memory management means developers must separately handle CUDA `cudaMalloc`/HIP `hipMalloc` or use UMPIRE (LLNL memory management library). OpenMP target offload and SIMD backends are experimental.

**Active Development Status**
Active. Maintained by LLNL. Used in production simulation codes (HYDRA, ALE3D). Releases continue as of 2025-2026. Typically deployed alongside UMPIRE for memory management.

**Key Limitations**
- No integrated memory management (must use UMPIRE or manual allocation)
- OpenMP target, SIMD, TBB backends experimental
- No MLIR integration
- Compile-time policy selection — no runtime dispatch across vendors
- Less ecosystem richness than Kokkos (no Kernels library equivalent)
- AMD/Intel support less mature than CUDA path

---

## 4. Lightweight / JIT / Header-Only

### 4.1 OCCA (Open Concurrent Compute Abstraction)

**Architecture Summary**
OCCA provides a unified JIT-compiled portable kernel framework. Kernels are written in OKL (OCCA Kernel Language), a directive-annotated extension of C, and compiled at runtime for the target backend. Compiled kernels are cached to avoid recompilation on subsequent runs. OCCA manages device enumeration, memory allocation, and kernel launch through a unified C++/C/Fortran API. Backend selection occurs at runtime based on available devices and user configuration. The framework is designed for scientific computing (adopted by U.S. DoE and Shell) rather than ML.

**Portability Scope**
- NVIDIA GPUs: CUDA
- AMD GPUs: HIP
- Intel GPUs/CPUs: Data Parallel C++ (DPC++)
- Cross-platform: OpenCL
- CPUs: OpenMP
- Apple: Metal
- Latest release: v2.0 (April 2024), 3,290 commits

**Performance Overhead vs Native**
JIT compilation introduces startup latency (first run); kernel caching eliminates this on subsequent runs. Runtime backend selection adds no steady-state overhead beyond the JIT startup cost. Performance of generated code is bounded by the underlying backend (CUDA, HIP, etc.).

**MLIR Integration**
None.

**Developer Experience**
High productivity for scientific kernel developers familiar with C. OKL annotations are minimal and non-intrusive. Runtime backend selection without recompilation is a key differentiator. C, C++, and Fortran APIs provide broad language support. Transparency of backend mapping is a design principle.

**Active Development Status**
Active but niche. Production use at DoE and Shell. v2.0 released April 2024. Less community momentum than Kokkos/RAJA in HPC, and not adopted in ML.

**Key Limitations**
- OKL requires annotation-style programming (not standard C/C++)
- JIT startup latency on first kernel execution
- Requires C++17 minimum
- Backend availability dependent on installed drivers
- No MLIR integration
- Limited adoption in ML ecosystem
- Less performance optimization sophistication than Triton or TVM

---

## 5. MLIR GPU Infrastructure (Foundation Layer)

### 5.1 MLIR GPU Dialect

**Architecture Summary**
The MLIR GPU dialect provides vendor-neutral abstractions for GPU kernel launches following a CUDA/OpenCL-like programming model. `gpu.launch` and `gpu.launch_func` operations abstract kernel dispatch; `gpu.module` and `gpu.binary` operations hold serialized GPU binaries. Target attributes on `gpu.module` specify compilation targets (NVVM, ROCDL, SPIR-V), enabling multi-target compilation from a single IR. The binary offloading mechanism supports dynamic selection of GPU binary objects per target at translation time.

**Portability Scope**
- NVVM target: NVIDIA (PTX via NVPTX LLVM backend)
- ROCDL target: AMD (AMDGCN via AMDGPU LLVM backend)
- SPIR-V target: Vulkan/OpenCL cross-vendor

**Performance Overhead vs Native**
The GPU dialect is a compiler IR layer, not a runtime. Overhead is zero at runtime — it is compiled away. The quality of generated code depends on the lowering pipeline applied to the dialect.

**MLIR Integration**
The GPU dialect is MLIR. It is the foundation on which IREE, Triton, XLA's MLIR path, and research compilers are built. It is not a user-facing API but a building block.

**Active Development Status**
Active. Maintained in the LLVM monorepo. Key infrastructure for the MLIR-based ML compiler ecosystem.

---

## 6. Comparison Matrix

| Framework | Category | NVIDIA | AMD | Intel GPU | CPU | Dispatch Model | MLIR | Performance vs Native | Developer Experience | Status |
|-----------|----------|--------|-----|-----------|-----|----------------|------|-----------------------|---------------------|--------|
| **Triton** | Compile-time | Yes (CC 8.0+) | Yes (ROCm 6.2+) | No | Experimental | Compile-time (per-target binary) | Native (9 dialects) | ~Native (tile-level) | Python DSL, high prod. | Active (v3.6, Jan 2026) |
| **TVM** | Compile-time | Yes (CUDA) | Yes (ROCm) | Yes (OpenCL) | Yes (LLVM) | Compile-time (per-target) | Shallow | Near-native (tuned) | Complex scheduling | Active (Apache TLP) |
| **XLA** | Compile-time | Yes (NVPTX) | Yes (ROCm) | In dev | Yes | Compile-time (per-target) | First-class (StableHLO) | Near-native (fused) | Framework-transparent | Active (Google/OpenXLA) |
| **IREE** | Compile-time | Yes (CUDA) | Yes (HIP) | Yes (Vulkan) | Yes (ARM/x86) | Compile-time (HAL, multi-target via Vulkan) | Native (full stack) | Competitive | MLIR authoring/import | Active (Google) |
| **SYCL/DPC++** | Runtime | Yes (plugin) | Yes (plugin) | Yes (native) | Yes | Runtime device selector | None | Near-native (Intel), 5-15% overhead (NVIDIA) | Standard C++ | Active (Intel oneAPI) |
| **AdaptiveCpp** | Runtime | Yes | Yes | Yes | Yes | Runtime (SSCP JIT) | None | Near-native (JIT) | Standard C++ (SYCL) | Active |
| **OpenCL** | Runtime | Yes (legacy) | Yes | Yes | Yes | Runtime device selector | None (SPIR-V path) | 10-30% overhead | Verbose, low-level | Maintained, declining |
| **Vulkan Compute** | Runtime | Yes | Yes | Yes | No | Runtime (via SPIR-V driver) | Via SPIR-V dialect | ~Native AMD/Intel, 5-20% NVIDIA | Very verbose, expert | Active |
| **ALPAKA** | Hybrid | Yes (CUDA 12+) | Yes (HIP 6+) | Yes (SYCL) | Yes (OMP/TBB) | Compile-time only | None | < 2% overhead | CUDA-shaped C++ | Active (CERN/HEP) |
| **Kokkos** | Hybrid | Yes (CUDA) | Yes (HIP) | Yes (SYCL) | Yes (OMP/HPX) | Compile-time only | None | < 5% overhead | Moderate learning curve | Active (v5.1, LF project) |
| **RAJA** | Hybrid | Yes (CUDA) | Yes (HIP) | Experimental | Yes (OMP) | Compile-time only | None | < 3% overhead | Lambda-based, familiar | Active (LLNL) |
| **OCCA** | JIT/lightweight | Yes (CUDA) | Yes (HIP) | Yes (DPC++) | Yes (OMP) | Runtime JIT | None | Backend-bounded + JIT startup | OKL annotations, C API | Active (v2.0, Apr 2024) |

---

## 7. Key Gaps Identified

Examining the landscape, several critical gaps emerge that motivate a novel contribution:

**1. No solution combines compile-time ML kernel optimization with runtime GPU vendor selection in a single binary.**
Every ML compiler (Triton, TVM, XLA, IREE) requires recompilation per target. Every runtime dispatch system (SYCL, OpenCL, Vulkan, OCCA) lacks ML-specific kernel optimization. The two capabilities have not been unified.

**2. MLIR's GPU dialect supports multi-target compilation (`gpu.binary` with multiple target attributes) but no ML compiler exposes this for dynamic runtime vendor selection.**
The MLIR infrastructure for embedding multiple GPU binaries and selecting at runtime exists but is unused in production ML compilers.

**3. IREE's HAL dialect is the closest existing approach** — it provides hardware abstraction that could support runtime backend selection. However, IREE selects targets at compile time and does not dispatch dynamically based on runtime-detected hardware.

**4. AdaptiveCpp's SSCP (Single-Source Single-Compiler Pass)** comes closest to a runtime-dispatch model: one binary, multiple targets, JIT selection. But it operates at the SYCL level without ML-specific kernel optimization (no tile-level auto-tuning, no tensor core awareness).

**5. No solution provides zero-overhead heterogeneous dispatch with ML-tuned kernels across NVIDIA + AMD + Intel GPU in a single deployment artifact.**

These gaps define the contribution space for the LLVM Dublin 2026 poster: a dynamic dispatch architecture that builds on MLIR's multi-target `gpu.binary` infrastructure to select vendor-optimal ML kernels at runtime without recompilation.

---

## 7.1 New Entries (April 2026 Update)

### HetGPU (arXiv:2506.15993, June 2025)

**Architecture Summary**
HetGPU tackles vendor lock-in through runtime binary translation: a compiler emits architecture-agnostic GPU IR with execution state metadata, and a runtime dynamically translates this IR to native code for NVIDIA, AMD, Intel, and Tenstorrent GPUs. The system supports unmodified GPU binary migration across vendors.

**Dispatch Strategy:** Runtime IR translation (JIT-style). Not AOT variant selection.

**Relevance to This Work:** Complementary, not competing. HetGPU translates a single binary to run everywhere; our work selects among pre-compiled vendor-native binaries (NVPTX, AMDGCN, x86) for peak per-vendor performance. HetGPU accepts translation overhead for portability; we accept build-time compilation cost for zero runtime translation. The two compose: HetGPU could produce one of the binary variants our dispatch layer selects from.

**Key Limitation:** Runtime translation adds overhead. For latency-sensitive ML inference, pre-compiled native binaries outperform translated ones.

### KernelEvolve (Meta, ISCA 2026, arXiv:2512.23236)

**Architecture Summary**
Agentic kernel generation system deployed at Meta. Uses search-based optimization to generate kernels targeting NVIDIA, AMD, MTIA, and CPU simultaneously. Achieves 60% inference throughput improvement in production. The system generates multi-target kernel variants at design time through iterative refinement.

**Dispatch Strategy:** Design-time generation, not runtime dispatch. Produces the kernel variants but does not address which variant runs on which hardware at deployment time.

**Relevance to This Work:** Strongest industry validation that multi-target kernel generation is production-relevant. Positions our contribution as the runtime dispatch step that follows KernelEvolve's generation step: KernelEvolve creates the variants, our layer selects among them at inference time.

### AdaptiveCpp SSCP (IWOCL 2025)

**Architecture Summary**
Single-Source, Single Compilation Pass mode in AdaptiveCpp (formerly hipSYCL). JIT-compiles at first launch, specializing based on runtime information: work-group sizes, pointer alignments, kernel argument values. Achieves near-native performance after first-launch JIT cost.

**Dispatch Strategy:** Runtime JIT specialization from SYCL source. Requires SYCL programming model.

**Relevance to This Work:** Different approach to the same multi-target problem. AdaptiveCpp specializes one source at JIT time; we route among pre-compiled AOT binaries with zero JIT cost. For deployments where SYCL adoption is infeasible or JIT latency is unacceptable, our 3 ns AOT selection provides multi-target benefit without the programming model lock-in.

### Universal GPU ISA (arXiv:2603.28793, March 2026)

**Architecture Summary**
First systematic cross-vendor analysis of GPU instruction set architectures spanning NVIDIA (Fermi–Blackwell), AMD (RDNA 1–4, CDNA 1–4), Intel (Gen11–Xe-HPC), and Apple (G13). Identifies 10 hardware-invariant computational primitives, 6 parameterizable dialects, and 6 true architectural divergences.

**Dispatch Strategy:** Theoretical ISA unification. Proposes that SPIR-V could distribute programs targeting the universal ISA.

**Relevance to This Work:** Validates our capability contract design (our `requires_features` metadata maps to their "parameterizable dialects"). The 6 true divergences confirm that vendor-native optimized kernels will continue to outperform universal binaries for compute-intensive ML — our dispatch layer bridges this divergence gap by carrying both portable and optimized variants.

---

## 8. References and Sources

- Triton v3.6.0 README and dialect documentation: https://triton-lang.org / https://github.com/openai/triton
- TVM Architecture Guide: https://tvm.apache.org/docs/arch/index.html
- OpenXLA/XLA Architecture: https://openxla.org/xla/architecture
- IREE Documentation (Framework Integration, CUDA backend, Dialect reference): https://iree.dev
- SYCL 2020 Specification: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
- Intel DPC++ Compiler: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html
- AdaptiveCpp Documentation: https://adaptivecpp.github.io/AdaptiveCpp/
- OpenCL 3.0 Overview: https://www.khronos.org/opencl/
- Vulkan Specification: https://docs.vulkan.org
- ALPAKA GitHub: https://github.com/alpaka-group/alpaka
- Kokkos GitHub: https://github.com/kokkos/kokkos (v5.1.0, March 2026)
- RAJA Documentation: https://raja.readthedocs.io/en/main/
- OCCA GitHub: https://github.com/libocca/occa (v2.0, April 2024)
- MLIR GPU Dialect: https://mlir.llvm.org/docs/Dialects/GPU/
- HIP Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/
- HetGPU: https://arxiv.org/abs/2506.15993
- KernelEvolve (Meta, ISCA 2026): https://arxiv.org/abs/2512.23236
- AdaptiveCpp SSCP (IWOCL 2025): https://dl.acm.org/doi/10.1145/3731125.3731127
- Universal GPU ISA: https://arxiv.org/abs/2603.28793
- GPU Portability Needs Autotuning: https://arxiv.org/abs/2505.03780
- Proteus (CGO 2025): https://dl.acm.org/doi/10.1145/3696443.3708939
- Helix (ASPLOS 2025): https://arxiv.org/abs/2406.01566
