# ALPAKA and ROOT TMVA-SOFIE: Deep Analysis for LLVM Dublin Poster

*Research compiled 2026-04-02 for "Heterogeneous GPU Kernel Dispatch via MLIR" poster.*

---

## 1. ALPAKA Abstraction Model

### 1.1 Core Design: Redundant Hierarchical Parallelism

ALPAKA (Abstraction Library for Parallel Kernel Acceleration) is a **header-only C++20 library** that achieves GPU-agnostic behavior through a compile-time template metaprogramming strategy. The foundational paper (Zenker et al., 2016) introduces **Redundant Hierarchical Parallelism (RHP)** -- a five-level abstraction model that maps uniformly across heterogeneous hardware:

| Level    | Abstraction                  | Memory Tier     | Hardware Analog (GPU)   | Hardware Analog (CPU)     |
|----------|------------------------------|-----------------|-------------------------|---------------------------|
| Grid     | Complete task                | Global memory   | GPU grid                | Process                   |
| Block    | Independent subtask          | Shared memory   | Thread block            | Thread group              |
| Warp     | Lock-step execution group    | (implicit)      | Warp (32 threads)       | SIMD lane group           |
| Thread   | Individual execution unit    | Registers       | CUDA thread             | CPU thread / fiber        |
| Element  | Per-thread data elements     | Vector registers| Loop body               | SIMD element              |

The key insight is **redundancy**: if a hardware target does not support a given level (e.g., CPUs have no warp-level synchronization), that level is simply collapsed. The same kernel source maps to different hardware by ignoring unsupported levels and utilizing available ones. This differs fundamentally from abstraction-hiding approaches (like OpenCL) which force a lowest-common-denominator model.

**Source:** [arXiv:1602.08477](https://arxiv.org/abs/1602.08477); [Alpaka Abstraction Docs](https://alpaka.readthedocs.io/en/latest/basic/abstraction.html)

### 1.2 How Template-Based Portability Works

Kernels are written as **C++ function objects (functors)** with a templated `operator()`:

```cpp
struct MyKernel {
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, ...) const -> void {
        // acc provides thread indexing, shared memory, sync primitives
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        // ...kernel body...
    }
};
```

The accelerator type `TAcc` is a **compile-time template parameter** resolved via CMake configuration. The same source compiles to:
- `AccGpuCudaRt` -> NVIDIA GPU via CUDA
- `AccGpuHipRt` -> AMD GPU via HIP
- `AccCpuOmp2Blocks` -> CPU via OpenMP
- `AccCpuSerial` -> Single-threaded CPU

Backend selection requires changing **one typedef or CMake flag**, not the kernel source. The compiler generates fully specialized native code for each target -- there is zero runtime dispatch overhead because the backend is fixed at compile time.

**Source:** [Alpaka GitHub](https://github.com/alpaka-group/alpaka); [Alpaka Intro Docs](https://alpaka.readthedocs.io/en/latest/basic/intro.html)

### 1.3 Memory Model

ALPAKA uses an **explicit, data-structure-agnostic memory model**:
- **Buffers** are the primary abstraction -- simple typed memory regions with device-awareness
- No implicit data migrations (unlike CUDA Unified Memory)
- Deep copies between host and device are explicit API calls
- Shared memory is allocated per-block via `alpaka::declareSharedVar`
- The developer retains full control over allocation, layout, and transfer

This explicit model is both a strength (predictable performance) and a burden (more boilerplate than CUDA managed memory).

---

## 2. ALPAKA Backend Matrix

### 2.1 Supported Backends (as of v2.1.1, December 2025)

| Backend             | Target Hardware        | Minimum Version    | Status      |
|---------------------|------------------------|--------------------|-------------|
| `AccGpuCudaRt`      | NVIDIA GPUs            | CUDA 12.0+         | Production  |
| `AccGpuHipRt`       | AMD GPUs               | HIP 6.0+           | Production  |
| `AccGpuSyclIntel`   | Intel GPUs/CPUs/FPGAs  | oneAPI 2024.2+      | Production  |
| `AccCpuOmp2Blocks`  | Multi-core CPU         | OpenMP 2.0+        | Production  |
| `AccCpuOmp2Threads` | Multi-core CPU         | OpenMP 2.0+        | Production  |
| `AccCpuThreads`     | Multi-core CPU         | C++20 std::thread  | Production  |
| `AccCpuTbbBlocks`   | Multi-core CPU         | TBB                | Production  |
| `AccCpuSerial`      | Single-core CPU        | (none)             | Production  |

**Notable:** The SYCL backend is production-ready as of 2025, targeting Intel GPUs. CPU architectures supported include x86, ARM, RISC-V, and Power 8+.

**Source:** [Alpaka GitHub README](https://github.com/alpaka-group/alpaka); [Alpaka CMake Docs](https://alpaka.readthedocs.io/en/latest/advanced/cmake.html)

### 2.2 Cupla: CUDA Migration Layer

The **cupla** project provides a CUDA-like API wrapper on top of ALPAKA, enabling existing CUDA codebases to be ported with minimal changes (primarily `__global__` functions to functors, `*.cu` to `*.cpp`). However:
- Host-side API is **not thread-safe**
- Initial "find & replace" ports yield poor CPU performance until the element level is properly utilized
- cupla adds some host-side API call overhead vs. pure ALPAKA

**Source:** [cupla GitHub](https://github.com/alpaka-group/cupla)

---

## 3. ROOT TMVA-SOFIE

### 3.1 Architecture: Parser-Generator Pipeline

SOFIE (System for Optimized Fast Inference code Emit) generates **standalone C++ inference code** from trained ML models. The pipeline:

```
ONNX/Keras/PyTorch model
    -> Parser (RModelParser_ONNX, etc.)
    -> SOFIE Internal Representation (IR)
    -> Code Generator
    -> C++ header file (.hxx) + weights file (.dat)
    -> Compile with BLAS/Eigen -> native inference binary
```

Key characteristics:
- **No runtime dependency on ML frameworks** -- generated code is self-contained C++
- Supports 80+ ONNX operators (MatMul, Conv, LSTM, GRU, BatchNorm, attention ops, etc.)
- Weights stored either in separate `.dat` binary or embedded in header
- Two usage modes: Session-based (loads weights at runtime) or standalone function

**Source:** [SOFIE README](https://github.com/root-project/root/blob/master/tmva/sofie/README.md); [ROOT TMVA Manual](https://root.cern/manual/tmva/)

### 3.2 Code Generation Strategy

SOFIE performs **ahead-of-time code generation**: the ONNX graph is traversed, each operator is mapped to a C++ code template, and the full inference function is emitted as a header file. This is fundamentally different from:
- **ONNX Runtime**: Interprets/JITs the graph at runtime
- **TensorRT**: Compiles to GPU-specific engine at build time
- **MLIR-based approaches**: Lower through dialect hierarchy with optimization passes

SOFIE's approach prioritizes **integration simplicity** in the ROOT/HEP ecosystem over maximum optimization flexibility. The generated code is human-readable, debuggable, and embeddable in existing C++ workflows without additional runtime dependencies.

### 3.3 Performance Benchmarks

From CHEP 2024 and ACAT 2024 presentations:
- SOFIE has been benchmarked against **ONNX Runtime and LWTNN** on deep neural networks (5 fully-connected layers, 200 units each) processing 5M events single-threaded
- Recent optimizations include **Structure-of-Arrays memory allocation with reuse**, operator fusion, and kernel-level optimizations that "significantly reduce data movement and improve inference latency"
- SOFIE with SYCL GPU inference reported **up to 258x speedup** over plain C++ code for large convolutional models on Intel GPUs

**Source:** [SOFIE ACAT 2024](https://indico.cern.ch/event/1330797/contributions/5796633/); [SOFIE Benchmarks (ResearchGate)](https://www.researchgate.net/publication/396279763_Benchmark_Studies_of_Machine_Learning_Inference_using_SOFIE)

---

## 4. SOFIE + ALPAKA Integration: Current Status

### 4.1 Timeline and Status

| Date        | Milestone                                                        |
|-------------|------------------------------------------------------------------|
| 2022        | SOFIE CPU-only, ONNX parser complete                             |
| 2023        | SYCL backend developed (Panagou, Moneta, Sengupta)               |
| 2024        | ACAT 2024: SYCL GPU inference presented, 258x speedup claimed    |
| 2025 (GSoC) | GPU support project proposed: CUDA, ROCm, ALPAKA exploration     |
| 2025 (Sep)  | ACAT 2025: "SOFIE now supports portable GPU inference via SYCL and ALPAKA, using cuBLAS and rocBLAS" |

### 4.2 Current Architecture (as of ACAT 2025)

Per the ACAT 2025 presentation by Enrico Lupi (CERN/INFN Padova), with Lorenzo Moneta and Sanjiban Sengupta:
- SOFIE can now generate code targeting **cuBLAS (NVIDIA)** and **rocBLAS (AMD)** through ALPAKA
- Users select GPU stack based on platform preference
- Graph optimizations: SoA memory allocation, operator fusion, kernel-level opts
- Validated on: **ParticleNet, ATLAS GNNs, Smart Pixels models**
- Integration serves the **Next-Gen Triggers Project** for High-Luminosity LHC

### 4.3 Limitations

1. **Code generation, not runtime dispatch**: SOFIE generates different code paths for different backends at build time. There is no runtime detection/selection of the optimal backend.
2. **ALPAKA integration is recent** (2025): maturity and operator coverage on GPU backends likely trails the CPU path significantly.
3. **No MLIR involvement**: SOFIE's code generation is template-based C++ emission, not compiler IR lowering. No optimization passes, no dialect hierarchy, no progressive lowering.
4. **Operator coverage gap**: While 80+ ONNX ops are supported on CPU, GPU coverage via ALPAKA likely covers only BLAS-heavy operators (GEMM, Conv) initially.

**Source:** [ACAT 2025 Indico](https://indico.cern.ch/event/1488410/contributions/6561436/); [GSoC 2025 SOFIE-GPU Proposal](https://hepsoftwarefoundation.org/gsoc/2025/proposal_TMVA-SOFIE-GPU.html); [SOFIE SYCL Report (Zenodo)](https://zenodo.org/records/8385777)

---

## 5. ALPAKA Performance: Benchmarks vs. Native

### 5.1 Micro-Benchmarks (Original Paper)

From the 2016 paper (Zenker et al.):
- ALPAKA CUDA kernels achieve **>94% relative performance** vs. native CUDA on matrix operations
- Single-source DGEMM delivers **~20% of theoretical peak** consistently across AMD, Intel, and NVIDIA hardware
- HASEonGPU application ported in 3 weeks achieved **identical execution times** on NVIDIA GPUs

### 5.2 CMS Pixel Reconstruction (Production Scale)

The CMS experiment adopted ALPAKA as its **official portability layer for Run 3 HLT** (High Level Trigger). Key findings from the Patatrack standalone benchmark:

- ALPAKA yields **comparable or better performance** than direct CUDA/HIP on CMS pixel tracking
- A 2.5x throughput improvement was achieved for CUDA by minimizing API calls for device memory allocation
- ALPAKA was judged **easiest to work with** among Kokkos, SYCL, std::par, and OpenMP offloading for this codebase
- SYCL showed **overhead concerns**; std::par required **many more kernels**; OpenMP had **added data movement costs**

### 5.3 Cross-Framework Comparison (CHEP 2024)

The CMS pixel reconstruction was ported to Alpaka, Kokkos, SYCL, std::par, and OpenMP offloading. Event processing throughput was compared on NVIDIA GPUs, AMD GPUs, and CPUs:
- ALPAKA is "flexible and adds only little constraints"
- All portability layers achieved **near-native throughput** on GPU targets
- CPU performance varied more significantly across frameworks

**Source:** [Fermilab-Conf-23-080](https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf); [EPJ Web Conf. CHEP 2024](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11008/epjconf_chep2024_11008.html); [ResearchGate: Performance Portability CMS](https://www.researchgate.net/publication/368563469_Performance_portability_for_the_CMS_Reconstruction_with_Alpaka)

---

## 6. Key Papers and Projects

### 6.1 Foundational Papers

| Paper | Authors | Venue | Key Contribution |
|-------|---------|-------|------------------|
| "Alpaka -- An Abstraction Library for Parallel Kernel Acceleration" | Zenker, Worpitz, Widera, Huebl, Juckeland, Knupfer, Nagel, Bussmann | arXiv:1602.08477 (2016) | Original RHP model, API design, micro-benchmarks |
| "Performance portability for the CMS Reconstruction with Alpaka" | CMS Collaboration | Fermilab-Conf-23-080 (2023) | Production-scale validation in CMS HLT |
| "Evaluating Performance Portability with CMS Heterogeneous Pixel Reconstruction" | Kortelainen et al. | EPJ Web Conf. CHEP 2024 | Multi-framework comparison (Alpaka, Kokkos, SYCL, std::par, OpenMP) |
| "Line Segment Tracking: Phase 2 CMS HLT" | CMS Collaboration | arXiv:2407.18231 (2024) | Hardware-agnostic tracking algorithm using ALPAKA |

### 6.2 SOFIE Papers

| Paper | Authors | Venue | Key Contribution |
|-------|---------|-------|------------------|
| "C++ Code Generation for Fast Inference of Deep Learning Models in ROOT/TMVA" | Moneta et al. | ResearchGate (2021) | Original SOFIE architecture |
| "Accelerating ML Inference on GPUs with SYCL using SOFIE" | Moneta, Sengupta | ACAT 2024 | SYCL GPU backend, 258x speedup |
| "TMVA SOFIE: Enhancements in ML Inference through graph optimizations and heterogeneous architectures" | Lupi, Sengupta, Moneta | ACAT 2025 | ALPAKA+SYCL integration, cuBLAS/rocBLAS |
| "Benchmark Studies of ML Inference using SOFIE" | Moneta et al. | CHEP 2024 | Comparison vs ONNX Runtime, LWTNN |

### 6.3 alpaka-group GitHub Organization

- **Organization:** [github.com/alpaka-group](https://github.com/alpaka-group)
- **Main repo:** [alpaka-group/alpaka](https://github.com/alpaka-group/alpaka) -- v2.1.1 (Dec 2025)
- **Next-gen:** [alpaka-group/alpaka3](https://github.com/alpaka-group/alpaka3) -- major rewrite with breaking changes, no releases yet
- **CUDA migration:** [alpaka-group/cupla](https://github.com/alpaka-group/cupla)
- **Institutional home:** Helmholtz-Zentrum Dresden-Rossendorf (HZDR), TU Dresden
- **Key maintainers:** Erik Zenker (HZDR), Benjamin Worpitz (LogMeIn/HZDR), Rene Widera (HZDR), Axel Huebl (HZDR/LBNL)
- **Funding:** EU Horizon 2020 Grant No. 654220

### 6.4 Patatrack Project

The **Patatrack** project at CMS pioneered heterogeneous reconstruction for HEP:
- Original CUDA-based pixel tracking, later ported to ALPAKA, Kokkos, SYCL, HIP
- ALPAKA selected as official portability solution for CMS Run 3
- Standalone benchmark: [github.com/cms-patatrack/pixeltrack-standalone](https://github.com/cms-patatrack/pixeltrack-standalone)
- GSoC projects (2021, 2022) added ALPAKA and SYCL backends

**Source:** [Patatrack GitHub](https://github.com/cms-patatrack); [GSoC 2021 Proposal](https://hepsoftwarefoundation.org/gsoc/2021/proposal_CMSalpaka.html); [CERN Openlab Report](https://openlab-archive-6-7.web.cern.ch/sites/default/files/2022-02/CERN_openlab_SUM_report_Abhinav_Ramesh.pdf)

---

## 7. ALPAKA Limitations and Gaps

### 7.1 Compile-Time Only Portability

ALPAKA's most fundamental limitation for the purposes of this poster: **the backend is fixed at compile time.** You cannot:
- Detect available hardware at runtime and dispatch accordingly
- Load-balance across heterogeneous devices dynamically
- Fall back from GPU to CPU if the GPU is busy or unavailable
- Ship a single binary that runs on both NVIDIA and AMD GPUs

Each target requires a **separate compilation**. In a heterogeneous cluster with mixed NVIDIA/AMD nodes, you need multiple binaries or a build matrix. This is the exact gap that runtime dispatch via MLIR would fill.

### 7.2 Template Complexity

ALPAKA "extensively makes use of advanced functional C++ template meta-programming techniques." Practical consequences:
- **Compilation times** are significantly longer than native CUDA/HIP
- **Error messages** from template instantiation failures are notoriously opaque
- **IDE support** (autocomplete, navigation) struggles with deep template nesting
- **Learning curve** is steep -- the CMS collaboration invested significant effort in training developers
- Debugging template-heavy code requires expertise in C++ metaprogramming

### 7.3 No Optimization Passes

ALPAKA is a **library**, not a **compiler**. It provides:
- Abstraction over parallelism hierarchies (good)
- Zero-overhead backend mapping via templates (good)
- But **no domain-specific optimization** (no operator fusion, no memory layout transformation, no auto-tuning)

The compiler (nvcc, hipcc, dpcpp) optimizes the instantiated code, but there is no intermediate representation where cross-kernel optimizations could be applied. Compare this to MLIR, which provides:
- Progressive lowering through dialects
- Optimization passes at each level (linalg -> affine -> scf -> llvm)
- Tiling, fusion, vectorization as composable transforms

### 7.4 Single-Node Scope

ALPAKA targets **single-node heterogeneous execution**. It does not address:
- Distributed multi-node dispatch
- Network-aware kernel placement
- Cross-node memory management

### 7.5 No ML-Specific Abstractions

Unlike MLIR's linalg dialect (which understands operations like matmul, conv, pooling at a semantic level), ALPAKA operates at the raw parallelism level. There is no notion of "this is a GEMM" that could trigger specialized library calls (cuBLAS, rocBLAS) automatically.

---

## 8. ALPAKA + MLIR: What Would It Take?

### 8.1 The Complementarity Thesis

ALPAKA and MLIR address **different layers** of the heterogeneous computing problem:

| Concern                    | ALPAKA Approach              | MLIR Approach                        |
|----------------------------|------------------------------|--------------------------------------|
| Portability mechanism      | C++ templates (compile-time) | IR dialects (compile/JIT-time)       |
| Backend selection          | CMake/typedef (build-time)   | Target triple + HAL (compile/runtime)|
| Optimization               | Delegated to backend compiler| Composable passes at each IR level   |
| Memory management          | Explicit buffers             | Memref type with layout transforms   |
| Kernel representation      | C++ functor                  | MLIR operation in dialect            |
| Runtime dispatch            | Not supported                | Possible via JIT + HAL (e.g., IREE)  |
| ML domain awareness        | None                         | linalg, tosa, stablehlo dialects     |

### 8.2 Hypothetical Integration Paths

**Path A: MLIR generates ALPAKA C++ (code emission)**
- An MLIR backend could emit ALPAKA-compatible C++ functors from linalg/scf operations
- Benefit: leverage ALPAKA's mature backend matrix for actual execution
- Drawback: still compile-time portability only; no runtime dispatch; adds a C++ compilation step

**Path B: MLIR replaces ALPAKA's role entirely (native MLIR lowering)**
- Use MLIR's gpu dialect -> NVVM/ROCDL/SPIRV lowering directly
- Benefit: runtime dispatch via JIT, optimization passes, no template complexity
- Drawback: less mature than ALPAKA for HEP-specific patterns; requires MLIR runtime (e.g., IREE HAL)
- This is essentially what IREE does already

**Path C: ALPAKA-informed MLIR dialect (hybrid)**
- Define an MLIR dialect that captures ALPAKA's RHP abstraction
- Lower RHP operations to gpu/scf/vector dialects with target-specific optimization
- Benefit: formalizes ALPAKA's proven abstraction model in compiler IR; enables both AOT and JIT
- Drawback: significant engineering effort; unclear community demand

### 8.3 The IREE Precedent

IREE already implements much of what a "runtime dispatch via MLIR" system needs:
- **HAL (Hardware Abstraction Layer)**: abstracts CUDA, Vulkan/SPIR-V, CPU targets behind common API
- **Ahead-of-time compilation** for any combination of targets
- **Runtime device selection** and dispatch
- Extensible codegen interfaces for custom backends

The gap IREE does not fill (and ALPAKA does): **tight integration with HEP workflows**, ROOT/CMSSW frameworks, and the existing physics reconstruction codebases that CMS/ATLAS maintain.

### 8.4 What SOFIE+ALPAKA Misses That MLIR Could Provide

1. **Runtime hardware detection and dispatch**: Ship one artifact, run on available hardware
2. **Cross-operator optimization**: SOFIE generates operator-by-operator C++ code; MLIR could fuse adjacent operators, optimize memory layout across the full graph
3. **JIT specialization**: Adapt kernel parameters (tile sizes, vectorization widths) to the specific GPU model at runtime
4. **Progressive lowering**: SOFIE goes ONNX -> C++ in one step; MLIR provides intermediate representations where domain-specific and target-specific optimizations can be applied independently

---

## 9. Critical Assessment for the Poster

### 9.1 What ALPAKA Does Well

- **Proven at production scale**: CMS Run 3 HLT uses ALPAKA in production for pixel tracking
- **Near-zero runtime overhead**: >94% of native CUDA performance
- **Broad backend coverage**: CUDA, HIP, SYCL, OpenMP, TBB, serial -- all from single source
- **Mature ecosystem**: 10+ years of development, CHEP/ACAT publications, GSoC projects, institutional backing from HZDR
- **Clean separation of concerns**: Algorithm vs. parallelization vs. memory management

### 9.2 What Gaps Remain (Poster Contribution Space)

1. **No runtime dispatch** -- the single biggest limitation for heterogeneous environments
2. **No compiler-level optimization** -- missed fusion/tiling opportunities
3. **No JIT capability** -- cannot adapt to hardware discovered at deployment time
4. **Template complexity tax** -- high barrier to entry, long compile times
5. **SOFIE+ALPAKA generates static C++** -- cannot exploit MLIR's progressive lowering

### 9.3 Poster Narrative

The poster should position ALPAKA as **the state-of-the-art for compile-time portability** in HEP, while arguing that MLIR-based dispatch addresses the remaining gaps:

> "ALPAKA proves that single-source heterogeneous execution is achievable with near-native performance. But its compile-time resolution model cannot serve environments where target hardware is unknown until deployment. We propose that MLIR's compilation infrastructure -- with JIT lowering, runtime HAL, and domain-specific optimization passes -- can complement ALPAKA's proven abstraction model to deliver runtime-adaptive dispatch for ML kernels in heterogeneous GPU environments."

This positions the work as **building on ALPAKA's success** (credible, since Akash did GSoC on SOFIE+ALPAKA at CERN) rather than competing with it.

---

## 10. References

### Papers
1. Zenker, E. et al. "Alpaka -- An Abstraction Library for Parallel Kernel Acceleration." [arXiv:1602.08477](https://arxiv.org/abs/1602.08477) (2016).
2. CMS Collaboration. "Performance portability for the CMS Reconstruction with Alpaka." [Fermilab-Conf-23-080](https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf) (2023).
3. Kortelainen, M. et al. "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code." [EPJ Web Conf. CHEP 2024](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11008/epjconf_chep2024_11008.html) (2024).
4. Moneta, L. et al. "C++ Code Generation for Fast Inference of Deep Learning Models in ROOT/TMVA." [ResearchGate](https://www.researchgate.net/publication/354072102_C_Code_Generation_for_Fast_Inference_of_Deep_Learning_Models_in_ROOTTMVA) (2021).
5. Lupi, E., Sengupta, S., Moneta, L. "TMVA SOFIE: Enhancements in ML Inference through graph optimizations and heterogeneous architectures." [ACAT 2025](https://indico.cern.ch/event/1488410/contributions/6561436/) (2025).
6. Panagou, I.-M., Moneta, L., Sengupta, S. "Inference of ML models on Intel GPUs with SYCL and Intel OneAPI using SOFIE." [Zenodo](https://zenodo.org/records/8385777) (2023).
7. CMS Collaboration. "Line Segment Tracking: Improving Phase 2 CMS HLT Tracking." [arXiv:2407.18231](https://arxiv.org/html/2407.18231) (2024).
8. Lattner, C. et al. "MLIR: A Compiler Infrastructure for the End of Moore's Law." [arXiv:2002.11054](https://arxiv.org/pdf/2002.11054) (2020).

### Software and Documentation
9. alpaka-group. "Alpaka: Abstraction Library for Parallel Kernel Acceleration." [GitHub](https://github.com/alpaka-group/alpaka) -- v2.1.1.
10. alpaka-group. "Alpaka3 (next-gen)." [GitHub](https://github.com/alpaka-group/alpaka3).
11. alpaka-group. "Cupla: CUDA-like wrapper for Alpaka." [GitHub](https://github.com/alpaka-group/cupla).
12. ROOT Project. "TMVA SOFIE." [GitHub](https://github.com/root-project/root/tree/master/tmva/sofie).
13. IREE Project. [iree.dev](https://iree.dev/).
14. CMS Patatrack. "Standalone pixel tracking." [GitHub](https://github.com/cms-patatrack/pixeltrack-standalone).
15. Alpaka Documentation. [alpaka.readthedocs.io](https://alpaka.readthedocs.io/en/latest/).

### Conference Presentations
16. "TMVA SOFIE" at ROOT Workshop 2022. [Fermilab Indico](https://indico.fnal.gov/event/23628/contributions/237964/attachments/154980/201725/TMVA_SOFIE_ROOTWS_May2022%20.pdf).
17. "Accelerating ML Inference on GPUs with SYCL using SOFIE" at ACAT 2024. [CERN Indico](https://indico.cern.ch/event/1330797/contributions/5796633/).
18. "Using Alpaka in CMSSW framework" at HSF Frameworks WG 2023. [CERN Indico](https://indico.cern.ch/event/1281987/contributions/5386025/).
19. HSF GSoC 2025: "TMVA SOFIE GPU Support." [HSF](https://hepsoftwarefoundation.org/gsoc/2025/proposal_TMVA-SOFIE-GPU.html).
20. Helmholtz Software Directory: Alpaka. [helmholtz.software](https://helmholtz.software/software/alpaka).
