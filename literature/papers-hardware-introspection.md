# Papers: GPU Hardware Introspection and Capability-Based Dispatch

Research compiled for LLVM Dublin 2026 poster on vendor-agnostic runtime dispatch for ML kernels.
Focus: how GPU hardware is queried at runtime and how that information drives kernel selection.

---

## 1. Runtime Hardware Query APIs

### 1.1 CUDA Compute Capability — NVIDIA

**API:** `cudaGetDeviceProperties()` / `cudaDeviceGetAttribute()` / `cuDeviceGetAttribute()` / `nvmlDeviceGetCudaComputeCapability()`

**Structure:** `cudaDeviceProp` — exposes `major`, `minor` (compute capability), `multiProcessorCount`, `clockRate`, `totalGlobalMem`, `warpSize`, `maxThreadsPerBlock`, concurrentKernels flag, cooperative launch support, managed memory support, and ~100 additional fields.

**Compile-side counterpart:** `nvcc` `-gencode arch=compute_XY,code=sm_XY` embeds multiple PTX/CUBIN blobs in a fatbin; the CUDA runtime picks the best match for the detected device at load time. Architecture-specific features (e.g., `sm_90a` Hopper features) require explicit guard macros since they are not forward-compatible.

**Sources:**
- [CUDA Runtime API — cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)
- [Compute Capabilities — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [How to Query Device Properties — NVIDIA Blog](https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/)
- [Blackwell family-specific features — NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)

---

### 1.2 Vulkan Device Features

**API:** `vkGetPhysicalDeviceFeatures()` / `vkGetPhysicalDeviceFeatures2()` (Vulkan 1.1+)

**Structures:** `VkPhysicalDeviceFeatures` (core), `VkPhysicalDeviceFeatures2` (extensible via `pNext` chain for extension-specific structs). Each field is `VK_TRUE`/`VK_FALSE` indicating support. Extension chains cover variable pointers, raytracing, 16-bit storage, cooperative matrices, etc.

**Dispatch implication:** Function pointers for physical-device-level commands may point to dispatch code that selects different real implementations per `VkPhysicalDevice`. SPIR-V `SpecializationConstants` allow pipeline-time specialization without recompiling shaders: values are injected at `vkCreateComputePipeline` time, enabling hardware-tailored variants from one SPIR-V blob.

**Sources:**
- [vkGetPhysicalDeviceFeatures](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceFeatures.html)
- [vkGetPhysicalDeviceFeatures2](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceFeatures2.html)
- [Vulkan Features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html)
- [SPIR-V Specialization Constants (Intel oneAPI)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/specialization-constants.html)

---

### 1.3 OpenCL Device Queries

**API:** `clGetDeviceInfo(cl_device_id, cl_device_info param_name, ...)`

**Key params:** `CL_DEVICE_TYPE`, `CL_DEVICE_MAX_COMPUTE_UNITS`, `CL_DEVICE_MAX_WORK_GROUP_SIZE`, `CL_DEVICE_GLOBAL_MEM_SIZE`, `CL_DEVICE_LOCAL_MEM_SIZE`, `CL_DEVICE_EXTENSIONS`, `CL_DEVICE_PARTITION_PROPERTIES`. The extensions string is the primary mechanism for feature detection at the OpenCL level.

**Dispatch pattern:** SOCL (an OpenCL implementation) dynamically dispatches kernels over processing devices to maximize utilization. Platform-aware programming using OpenCL's `clGetDeviceIDs` + `clGetDeviceInfo` is the standard pattern for selecting device at runtime, though performance is not automatically portable.

**Sources:**
- [clGetDeviceInfo — Khronos Registry](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html)
- [OpenCL Runtime — pyopencl docs](https://documen.tician.de/pyopencl/runtime_platform.html)

---

### 1.4 HIP Device Properties — AMD ROCm

**API:** `hipGetDeviceProperties(hipDeviceProp_t*, int device)`

**Structure fields:** device name, `totalGlobalMem`, `sharedMemPerBlock`, `regsPerBlock`, `maxThreadsPerBlock`, `multiProcessorCount`, `warpSize`, `cooperativeLaunch`, `managedMemory`, `gcnArchName` (e.g., `"gfx90a"` for MI250X). The `gcnArchName` field is the AMD analog to CUDA's compute capability and is used to select architecture-specific kernels.

**Usage pattern:** Query `multiProcessorCount` and total memory to rank available GPUs; query feature flags for conditional dispatch. HIP's portability goal means many fields mirror CUDA's `cudaDeviceProp` semantically.

**Sources:**
- [HIP Runtime API — Device Management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html)
- [HIP Deep Dive: Device Properties — Medium](https://gahan9.medium.com/hip-deep-dive-unlock-amd-gpu-secrets-with-device-properties-memory-queries-a1ccc4fb8ed0)

---

## 2. GPU Micro-benchmarking and Runtime Characterization

### Paper 2.1

**Title:** Dissecting GPU Memory Hierarchy through Microbenchmarking
**Authors:** Xinxin Mei, Xiaowen Chu
**Venue/Year:** arXiv 1509.02308; published IEEE TPDS, 2016
**Key Contribution:** First disclosure of cache properties (data cache, texture cache, TLB) for NVIDIA Fermi, Kepler, and Maxwell architectures. Pointer-chasing methodology for latency measurement. Revealed Maxwell's superiority in shared memory performance under bank conflict.
**URL:** [https://arxiv.org/abs/1509.02308](https://arxiv.org/abs/1509.02308) | [IEEE TPDS](https://ieeexplore.ieee.org/document/7445236/)

---

### Paper 2.2

**Title:** Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking
**Authors:** Zhe Jia, Marco Maggioni, Benjamin Staiger, Daniele P. Scarpazza
**Venue/Year:** arXiv 1804.06826, Technical Report, 2018
**Key Contribution:** Microarchitectural details of Volta through empirical testing and PTX disassembly. Documents memory hierarchy geometry, instruction encoding, and quantitative comparison against Kepler/Maxwell/Pascal. Methodology is the template for subsequent GPU dissection papers.
**URL:** [https://arxiv.org/abs/1804.06826](https://arxiv.org/abs/1804.06826)

---

### Paper 2.3

**Title:** Dissecting the NVidia Turing T4 GPU via Microbenchmarking
**Authors:** Zhe Jia, Marco Maggioni, Jeffrey Smith, Daniele Paolo Scarpazza
**Venue/Year:** arXiv 1903.07486, 2019
**Key Contribution:** Turing architecture characterization. TensorCore throughput on low-precision operands quantified. Full instruction space mapped (encoding shares Volta format with novel additions). Memory hierarchy found to double cache sizes vs Pascal GP104. Thermal/power throttling impact documented.
**URL:** [https://arxiv.org/abs/1903.07486](https://arxiv.org/abs/1903.07486)

---

### Paper 2.4

**Title:** Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis
**Authors:** Aaron Jarmusch, Sunita Chandrasekaran
**Venue/Year:** arXiv 2512.02189, December 2025 (revised March 2026)
**Key Contribution:** First detailed microbenchmark characterization of NVIDIA Blackwell B200. Open-source PTX/CUDA microbenchmark suite released. Studies Tensor Memory (TMEM), 5th-gen tensor cores supporting FP4/FP6/FP8, and the dedicated decompression engine. Shows 1.85x ResNet-50 and 1.55x GPT-1.3B throughput vs H200 with 32% better energy efficiency.
**URL:** [https://arxiv.org/abs/2512.02189](https://arxiv.org/abs/2512.02189)

---

### Paper 2.5

**Title:** Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks
**Authors:** Aaron Jarmusch, Nathan Graddon, Sunita Chandrasekaran (University of Delaware)
**Venue/Year:** arXiv 2507.10789, July 2025
**Key Contribution:** Microarchitectural analysis of Blackwell SM execution pipelines, memory hierarchy, 5th-gen tensor cores (FP4/FP6 support), latency/throughput/cache behavior. Compares Blackwell (RTX 5080) against Hopper (H100 PCIe). Documents power efficiency under varied workloads. Reveals subtle tuning metrics not in vendor documentation.
**URL:** [https://arxiv.org/abs/2507.10789](https://arxiv.org/abs/2507.10789)

---

### Paper 2.6

**Title:** Neutrino: Fine-grained GPU Kernel Profiling via Programmable Probing
**Authors:** Songlin Huang, Chenshu Wu (University of Hong Kong)
**Venue/Year:** OSDI 2025 (19th USENIX Symposium on Operating Systems Design and Implementation), Boston, MA
**Key Contribution:** Programmable GPU kernel profiler using assembly-layer probing. Achieves instruction-level granularity. Introduces Densified Memory Access Timeline (DMAT) representation. Hardware-independent (works on both NVIDIA and AMD). Python DSL + TOML interface. Supports cooperative probes and eBPF-like maps. Only platform-independent runtime GPU kernel profiler. Open-sourced at https://github.com/open-neutrino/neutrino. Received all OSDI AE badges (Available, Functional, Reproduced).
**URL:** [https://www.usenix.org/conference/osdi25/presentation/huang-songlin](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)

---

## 3. Adaptive Algorithm Selection Based on Hardware Properties

### Paper 3.1

**Title:** TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
**Authors:** Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy
**Venue/Year:** OSDI 2018, pp. 578–594
**Key Contribution:** End-to-end compiler that generates hardware-specific kernel code via a unified IR and a learned cost model for hardware-aware auto-scheduling. Targets CPUs, GPUs, FPGAs from a single program description. The compiler queries hardware characteristics and uses a statistical cost model to select tiling, vectorization, and parallelism parameters. Foundation for subsequent ML-based kernel optimization work.
**URL:** [https://www.usenix.org/conference/osdi18/presentation/chen](https://www.usenix.org/conference/osdi18/presentation/chen) | [arXiv 1802.04799](https://arxiv.org/abs/1802.04799)

---

### Paper 3.2

**Title:** Ansor: Generating High-Performance Tensor Programs for Deep Learning
**Authors:** Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica
**Venue/Year:** OSDI 2020
**Key Contribution:** Hardware-aware auto-scheduling that samples from a hierarchical representation of the search space, refines with evolutionary search and a learned cost model, and performs multi-subgraph task scheduling. Achieves up to 3.8x over Intel CPUs, 2.6x over ARM CPUs, 1.7x over NVIDIA GPUs vs prior state-of-the-art. Hardware properties (memory bandwidth, compute throughput) implicitly captured by the cost model trained per target.
**URL:** [https://arxiv.org/abs/2006.06762](https://arxiv.org/abs/2006.06762)

---

### Paper 3.3

**Title:** Input-Aware Auto-Tuning of Compute-Bound HPC Kernels
**Authors:** Philippe Tillet, David D. Cox
**Venue/Year:** SC17 (International Conference for High Performance Computing, Networking, Storage and Analysis), November 2017; arXiv 1802.05371
**Key Contribution:** ISAAC framework: predictive modeling to drive parameterized PTX templates toward hardware- and input-specific kernels. Addresses the gap where libraries like cuBLAS optimize for specific matrix shapes but degrade on others. Up to 3x gains over cuBLAS and cuDNN on Maxwell and Pascal after a few hours of tuning. Precursor to Triton's design philosophy.
**URL:** [https://arxiv.org/abs/1802.05371](https://arxiv.org/abs/1802.05371) | [ACM DL](https://dl.acm.org/doi/10.1145/3126908.3126939)

---

### Paper 3.4

**Title:** AS2: Adaptive Sorting Algorithm Selection for Heterogeneous Workloads and Systems
**Authors:** (Authors not fully extracted from search results)
**Venue/Year:** Future Generation Computer Systems, Volume 171, 2025, Article 107850
**Key Contribution:** ML model that considers both data-internal factors (size, distribution, type) and external factors (thread count, hardware) to predict execution time for multiple sorting algorithms and select the optimal one. Achieves up to 99.68% accuracy in optimal selection and up to 1.83x speedup vs state-of-the-art. Demonstrates that hardware characterization as an input feature substantially improves algorithm selection accuracy — directly relevant to the dispatch problem.
**URL:** [https://www.sciencedirect.com/science/article/abs/pii/S0167739X25001554](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25001554)

---

### Paper 3.5

**Title:** Performance Portability through Machine Learning Guided Kernel Selection in SYCL Libraries
**Authors:** John Lawson
**Venue/Year:** arXiv 2008.13145 (August 2020); published Parallel Computing, 2021
**Key Contribution:** Uses unsupervised clustering to select a deployment subset of kernel configurations per hardware architecture, then uses classification at runtime to pick the best kernel for a given problem size. Fully automated — relies only on benchmark data, no manual effort for new hardware. Shows that a parameterized GEMM kernel selected this way is competitive with or better than hand-optimized BLAS on desktop, integrated, and mobile GPUs. Direct template for capability-based dispatch without explicit hardware querying.
**URL:** [https://arxiv.org/abs/2008.13145](https://arxiv.org/abs/2008.13145) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167819121000624)

---

### Paper 3.6

**Title:** A Survey of CPU-GPU Heterogeneous Computing Techniques
**Authors:** Sparsh Mittal, Jeffrey S. Vetter
**Venue/Year:** ACM Computing Surveys, Volume 47, Issue 4, July 2015. DOI: 10.1145/2788396
**Key Contribution:** Comprehensive survey of heterogeneous computing techniques at runtime, algorithm, programming, compiler, and application levels. Reviews workload partitioning, scheduling strategies, and adaptive dispatch. Argues that merely offloading to GPU is suboptimal; combined CPU+GPU strategies are needed. Standard reference for the field.
**URL:** [https://dl.acm.org/doi/10.1145/2788396](https://dl.acm.org/doi/10.1145/2788396)

---

## 4. cuDNN and cuBLAS Runtime Kernel Selection

### Reference 4.1: cuBLAS Runtime Heuristics

**Source type:** NVIDIA documentation + technical blog
**Key Facts:**
- cuBLAS contains hundreds of SGEMM implementations. At runtime, dimensions and data types determine which kernel is dispatched via internal heuristics.
- The recommender system (machine-learning-trained on actual timing data) achieves 93% of the best available performance across the measured problem space (geomean).
- `cublasLtMatmulAlgoGetHeuristic` exposes the heuristics API so users can override default selection or perform their own autotuning.
- The library is fine-tuned per SM architecture, data type (FP32, FP16, BF16, INT8), and matrix size.

**Sources:**
- [What is cuBLAS — GPU Glossary (modal.com)](https://modal.com/gpu-glossary/host-software/cublas)
- [Grouped GEMM APIs in cuBLAS — NVIDIA Blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [How to Optimize a CUDA Matmul for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)

---

### Reference 4.2: cuDNN Algorithm Selection

**Source type:** NVIDIA documentation
**Key Facts:**
- `cudnnFindConvolutionForwardAlgorithm` runs all available convolution algorithms and reports timing; `cudnnGetConvolutionForwardAlgorithm_v7` uses heuristics for zero-overhead selection.
- cuDNN Backend API (v8+) provides engine-config selection: query heuristics mode A or B, filter for functional configs, optionally benchmark all functional configs (auto-tune mode), fall back to fallback heuristic if nothing found.
- EngineConfigs from heuristics queries are now guaranteed to be executable (confirmed runnable on the queried device/precision/layout combination).
- Heuristics are hardware-specific: algorithm optimal on A100 may not be optimal on H100.

**Sources:**
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/developer-guide/index.html)
- [cudnn_cnn Library — NVIDIA cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html)

---

## 5. Hardware-Aware Kernel Selection and Architecture-Specific Optimization

### Paper 5.1

**Title:** KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization
**Authors:** Nina Wiedemann, Quentin Leboutet, Michael Paulitsch, Diana Wofk, Benjamin Ummenhofer
**Venue/Year:** arXiv 2603.12440, March 2026
**Key Contribution:** Evolutionary framework combining MAP-Elites quality-diversity search, meta-prompt evolution (co-evolves prompts with kernels), and template-based parameter optimization targeting specific hardware. Distributed remote hardware access for rapid benchmarking. Average 2.3x speedup on KernelBench for SYCL kernels vs baseline. Hardware-awareness is explicit: kernels are evaluated on the actual target device.
**URL:** [https://arxiv.org/abs/2603.12440](https://arxiv.org/abs/2603.12440)

---

### Paper 5.2

**Title:** SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization
**Authors:** Arya Tschand, Muhammad Awad, Ryan Swann, Kesavan Ramakrishnan, Jeffrey Ma, Keith Lowery, Ganesh Dasika, Vijay Janapa Reddi
**Venue/Year:** arXiv 2508.20258, August 2025
**Key Contribution:** LLM-based performance engineering agent that explicitly supplies profiling data, architectural specs, and scheduling context to generate hardware-specific spatial optimizations. Found optimal swizzling pattern for a GEMM kernel on a disaggregated architecture in under 5 minutes vs 2 weeks for expert engineers. Up to 2.06x speedup and 70% improvement in L2 hit rate across 9/10 tested ML and science kernels.
**URL:** [https://arxiv.org/abs/2508.20258](https://arxiv.org/abs/2508.20258)

---

### Paper 5.3

**Title:** Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
**Authors:** Philippe Tillet, H.T. Kung, David Cox
**Venue/Year:** MAPL 2019 (3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages). DOI: 10.1145/3315508.3329973
**Key Contribution:** Tile-centric GPU programming model. The compiler takes tile programs and applies tile-level optimizations (memory coalescing, shared memory management, SM-level parallelism) automatically, adapting to hardware parameters at compile time. Enables writing hardware-efficient kernels without specifying low-level GPU details. Foundation for OpenAI's Triton, now the standard way to write portable performant GPU kernels in PyTorch.
**URL:** [https://dl.acm.org/doi/10.1145/3315508.3329973](https://dl.acm.org/doi/10.1145/3315508.3329973) | [Harvard PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

---

### Paper 5.4

**Title:** CLBlast: A Tuned OpenCL BLAS Library
**Authors:** Cedric Nugteren
**Venue/Year:** arXiv 1705.05249, May 2017 (updated April 2018). Related: IEEE IPDPS 2018
**Key Contribution:** Vendor-agnostic OpenCL BLAS library with fully parameterized kernels tunable per device via CLTune autotuner. Maintains a central tuning database indexed by device. At runtime, selects pretuned parameters matching the queried OpenCL device. Supports GEMM, TRSM, and full BLAS hierarchy across NVIDIA, AMD, Intel, ARM Mali, and embedded GPUs. Direct model for what a cross-vendor dispatch library looks like.
**URL:** [https://arxiv.org/abs/1705.05249](https://arxiv.org/abs/1705.05249)

---

### Paper 5.5

**Title:** Kernel Tuner: A Search-Optimizing GPU Code Auto-Tuner
**Authors:** Ben van Werkhoven
**Venue/Year:** Future Generation Computer Systems, Volume 90, 2019, pp. 347–358. DOI: 10.1016/j.future.2018.08.004
**Key Contribution:** Open-source framework for autotuning CUDA, OpenCL, and C GPU kernels with user-defined parameter spaces. Integrates 20+ search optimization methods (Bayesian optimization, genetic algorithms, simulated annealing, etc.). Separates the tuning concern from kernel code: kernels are parameterized, tuner explores parameter space on target hardware, stores results, and selects best configuration for deployment.
**URL:** [https://www.sciencedirect.com/science/article/pii/S0167739X18313359](https://www.sciencedirect.com/science/article/pii/S0167739X18313359) | [GitHub](https://github.com/KernelTuner/kernel_tuner)

---

## 6. Performance Models Foundational to Dispatch Decisions

### Paper 6.1

**Title:** Roofline: An Insightful Visual Performance Model for Multicore Architectures
**Authors:** Samuel Williams, Andrew Waterman, David A. Patterson
**Venue/Year:** Communications of the ACM, Volume 52, Issue 4, April 2009, pp. 65–76. DOI: 10.1145/1498765.1498785
**Key Contribution:** Introduced the Roofline model: characterizes kernel performance as a function of arithmetic intensity (FLOPs/Byte) relative to two hardware ceilings — peak compute and peak memory bandwidth. Provides the analytical framework for deciding at dispatch time whether a kernel variant is compute-bound or memory-bound on a given device, which directly informs which implementation to select. Still the dominant analytical model used in GPU kernel optimization.
**URL:** [https://dl.acm.org/doi/abs/10.1145/1498765.1498785](https://dl.acm.org/doi/abs/10.1145/1498765.1498785)

---

### Reference 6.2: Autotuning in HPC Applications (Survey)

**Title:** Autotuning in High-Performance Computing Applications
**Authors:** Prasanna Balaprakash, Jack Dongarra, Todd Gamblin, Mary Hall, Jeffrey K. Hollingsworth, Boyana Norris, Richard Vuduc
**Venue/Year:** Proceedings of the IEEE, July 31, 2018. DOI in IEEE Xplore.
**Key Contribution:** Defines autotuning as "the automatic generation of a search space of possible implementations of a computation that are evaluated through models and/or empirical measurement to identify the most desirable implementation." Surveys search strategies (Nelder-Mead, Bayesian, ML methods). Standard reference framing the autotuning subfield.
**URL:** [https://ieeexplore.ieee.org/document/8423171/](https://ieeexplore.ieee.org/document/8423171/)

---

## 7. Heterogeneous Runtime and Vendor-Agnostic Dispatch

### Paper 7.1

**Title:** HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs
**Authors:** Yiwei Yang, Yusheng Zheng, Tong Yu, Andi Quinn
**Venue/Year:** arXiv 2506.15993, June 2025
**Key Contribution:** System comprising compiler, runtime, and abstraction layer enabling a single GPU binary to execute on NVIDIA, AMD, Intel, and Tenstorrent hardware. Compiler emits architecture-agnostic IR with execution state metadata. Runtime JIT-translates IR to native code at load time after detecting the target GPU. Handles fundamental differences: warp-based SIMT (NVIDIA/AMD) vs many-core RISC-V (Tenstorrent). Enables live GPU migration across vendors with minimal overhead. Most direct existence proof that vendor-agnostic binary dispatch is achievable.
**URL:** [https://arxiv.org/abs/2506.15993](https://arxiv.org/abs/2506.15993)

---

### Paper 7.2

**Title:** Alpaka: An Abstraction Library for Parallel Kernel Acceleration
**Authors:** Erik Zenker, Benjamin Worpitz, René Widera, Axel Huebl, Guido Juckeland, Andreas Knüpfer, Wolfgang E. Nagel, Michael Bussmann
**Venue/Year:** arXiv 1602.08477; IPDPSW 2016
**Key Contribution:** C++ template library with abstract hierarchical parallelism model. Single source kernel code runs on x86 (multi-core), NVIDIA (CUDA), AMD (HIP), and Intel (SYCL) with a single-line backend change. Exploits all available hardware parallelism levels; gracefully degrades for unsupported levels. Template-based design makes backend switching zero-overhead at runtime (selected at compile time), but the abstraction layer design informs how dispatch can be structured.
**URL:** [https://arxiv.org/abs/1602.08477](https://arxiv.org/abs/1602.08477)

---

### Paper 7.3

**Title:** Kokkos 3: Programming Model Extensions for the Exascale Era
**Authors:** Christian R. Trott, Damien Lebrun-Grandié, Daniel Arndt, Jan Ciesko, Vinh Dang, Nathan Ellingwood, Rahulkumar Gayatri, Evan Harvey, Daisy S. Hollman, Dan Ibanez, Nevin Liber, Jonathan Madsen, Jeff Miles, David Poliakoff, Amy Powell, Sivasankaran Rajamanickam, Mikael Simberg, Dan Sunderland, Bruno Turcksin, Jeremiah Wilke
**Venue/Year:** IEEE Transactions on Parallel and Distributed Systems, Volume 33, Issue 4, 2022, pp. 805–817. DOI: 10.1109/TPDS.2021.3097283
**Key Contribution:** C++ performance portability programming model targeting all major HPC platforms (CUDA, HIP, SYCL, HPX, OpenMP, C++ threads). Abstractions for parallel execution and memory hierarchy. New in v3: hierarchical parallelism, containers, task graphs, arbitrary-sized atomics for exascale readiness. Backend is selected at compile time but the programming model is the reference for what vendor-agnostic kernel dispatch looks like in production HPC codes.
**URL:** [https://ieeexplore.ieee.org/document/9485033/](https://ieeexplore.ieee.org/document/9485033/)

---

### Paper 7.4

**Title:** RAJA: Portable Performance for Large-Scale Scientific Applications
**Authors:** D.A. Beckingsale, J. Burmark, R. Hornung, H. Jones, W. Killian, A.D. Kunen, O. Pearce, P. Robinson, B.S. Ryujin, T.R.W. Scogland
**Venue/Year:** P3HPC Workshop at SC19, 2019. Published IEEE Xplore.
**Key Contribution:** C++ portability layer (LLNL) enabling single-source code across CPU, CUDA GPU, and HIP backends. Three large production codes achieved 17x, 13x, 12x speedups on GPU vs CPU nodes with unchanged application code. RAJA, combined with Umpire (memory) and CHAI (data movement), forms a complete heterogeneous dispatch stack. Real-world validation of the portability-without-performance-loss claim.
**URL:** [https://ieeexplore.ieee.org/document/8945721/](https://ieeexplore.ieee.org/document/8945721/)

---

### Paper 7.5

**Title:** Enabling Dynamic Selection of Implementation Variants in Component-Based Parallel Programming for Heterogeneous Systems (COMPAR)
**Authors:** Suejb Memeti
**Venue/Year:** HeteroPar 2023 (21st International Workshop on Algorithms, Models and Tools for Parallel Computing on Heterogeneous Platforms), Springer 2023. arXiv 2311.03543
**Key Contribution:** COMPAR framework for runtime selection among multiple implementation variants (CPU OpenMP, CUDA GPU, HIP GPU, etc.) using compiler directives + StarPU runtime. Provides unified view of variants with intelligent context-aware selection. Promotes code reuse and simplifies heterogeneous programming while maintaining optimal performance.
**URL:** [https://arxiv.org/abs/2311.03543](https://arxiv.org/abs/2311.03543)

---

### Paper 7.6

**Title:** Runtime Support for Performance Portability on Heterogeneous Distributed Platforms
**Authors:** Polykarpos Thomadakis, Nikos Chrisochoides
**Venue/Year:** arXiv 2303.02543, March 2023
**Key Contribution:** Runtime framework enabling portable performance across heterogeneous nodes in a distributed setting. Achieves up to 300% improvement on single devices and linear scalability on four-GPU nodes. Up to 40% better than MPI+CUDA on distributed Jacobi proxy application. Demonstrates that runtime-level abstractions (not just compile-time) can maintain portability with performance.
**URL:** [https://arxiv.org/abs/2303.02543](https://arxiv.org/abs/2303.02543)

---

## 8. PyTorch ATen Dispatch and ISA Dynamic Dispatch

### Reference 8.1: PyTorch ATen/Dispatcher

**Source type:** Documentation and technical blog
**Key Facts:**
- Every PyTorch operator declared in `native_functions.yaml`. The dispatcher maintains a table of function pointers indexed by (operator, dispatch key). Dispatch key is computed from input tensor devices/types.
- Backends: CPU, CUDA, HIP, MPS, XLA, etc. Kernel registration fills cells in the operator × dispatch key table. Exact registrations take precedence over catch-all kernels.
- CPU kernels support ISA-level dispatch: `ATEN_CPU_CAPABILITY` environment variable overrides auto-detected ISA level (values: `avx2`, `avx512`, `avx512_vnni`, `avx512_bf16`, `amx`, `avx512_fp16`). The effective level is `min(env_override, hw_max)`.
- Intel Extension for PyTorch (IPEX) extends this with additional ISA levels (AVX512_VNNI, AVX512_BF16, AMX).

**Sources:**
- [Understanding ATen — Red Hat Developer](https://developers.redhat.com/articles/2026/02/19/understanding-aten-pytorchs-tensor-library)
- [ISA Dynamic Dispatching — IPEX docs](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/isa_dynamic_dispatch.html)
- [Let's Talk About the PyTorch Dispatcher — ezyang's blog](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [Extending Dispatcher for New Backend — PyTorch Tutorials](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)

---

### Reference 8.2: LLVM/Clang Function Multiversioning (CPU Analog)

**Source type:** LLVM documentation and code review
**Key Facts:**
- GCC/Clang `__attribute__((target("avx512f")))` generates architecture-specific function versions. `target_clones` attribute generates multiple clones automatically with an auto-generated resolver.
- `cpu_dispatch` / `cpu_specific` attributes (Intel-originated, implemented in Clang): specify individual versions and the dispatch resolver function list.
- The ifunc resolver is called once at relocation time by rtld; calls `__cpu_indicator_init` to inspect CPU feature bits from `__cpu_model` / `__cpu_features2` (populated via CPUID).
- No direct GPU analog exists: GPU feature detection uses runtime API queries (see Section 1), not static linker-level multiversioning.

**Sources:**
- [Function Multi-versioning — MaskRay blog](https://maskray.me/blog/2023-02-05-function-multi-versioning)
- [Architecture Specific Code Generation and Function Multiversioning — LLVM DevMtg 2014](https://llvm.org/devmtg/2014-10/Slides/Christopher-Function%20Multiversioning%20Talk.pdf)
- [Implement cpu_dispatch/cpu_specific Multiversioning — LLVM review](https://reviews.llvm.org/D47474)

---

## 9. MLIR/LLVM GPU Target and Compute Capability

### Reference 9.1: MLIR GPU Dialect and Target Attributes

**Source type:** MLIR documentation
**Key Facts:**
- MLIR's `gpu` dialect provides middle-level abstractions for GPU kernel launch (analogous to CUDA/OpenCL). Target attributes attach to GPU modules specifying serialization scheme: target triple, architecture name (e.g., `"sm_80"`), and target features (e.g., `"+ptx81,+sm_80"`).
- `gpu-lower-to-nvvm` pipeline accepts customization: SM capability, PTX version, optimization level. Lowering produces NVPTX assembly then CUBIN.
- `gpu-module-to-binary` pass serializes each GPU module with all its target attributes into binary objects.
- AMD path: AMDGPU backend uses GCN architecture strings (e.g., `gfx90a`) as target features.
- Compute capability selection at compile time; no MLIR-level runtime dispatch mechanism per se — dispatch must be implemented at the host runtime layer.

**Sources:**
- [GPU Dialect — MLIR LLVM](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR GPU Compilation — Stephen Diehl](https://www.stephendiehl.com/posts/mlir_gpu/)
- [NVIDIA GPU compute capability list in MLIR — LLVM Discourse](https://discourse.llvm.org/t/how-to-find-out-the-list-of-nvidia-gpu-compute-capability-i-e-sm-xx-support-for-a-particular-version-of-mlir/79687)

---

## 10. Broader Context: Performance Portability Surveys

### Paper 10.1

**Title:** An Evaluative Comparison of Performance Portability across GPU Programming Models
**Authors:** (LLNL authors, LLNL-CONF-855581)
**Venue/Year:** 2024, arXiv 2402.08950
**Key Contribution:** Empirical comparison of CUDA, HIP, Kokkos, RAJA, OpenMP, OpenACC, and SYCL on NVIDIA and AMD hardware. Finds Kokkos and RAJA offer the most promise as performance-portable models. Provides concrete data on performance gaps across models — useful for motivating a dispatch layer that can pick the best backend per device.
**URL:** [https://arxiv.org/abs/2402.08950](https://arxiv.org/abs/2402.08950)

---

## Key Themes and Gaps for the Poster

### What exists:
1. **Rich runtime query APIs** for all major GPU families (CUDA, Vulkan, OpenCL, HIP) — hardware introspection is solved at the API level.
2. **Library-level dispatch** (cuBLAS, cuDNN, CLBlast) uses ML-trained heuristics or exhaustive search to select kernels per hardware — effective but vendor-specific.
3. **Auto-scheduling/tuning** (TVM/Ansor, Triton, Kernel Tuner, Input-Aware Autotuning) — generates hardware-specific kernels but requires offline tuning per device.
4. **Performance portability layers** (Kokkos, RAJA, Alpaka, SYCL) — compile-time backend selection, not runtime vendor detection.
5. **Microbenchmarking suites** (Jia et al. Volta/Turing/Blackwell series, Neutrino) — document undisclosed hardware properties needed to build accurate dispatch heuristics.

### The gap (motivation for this poster):
- No system unifies: (a) runtime hardware introspection across NVIDIA/AMD/CPU, (b) dynamic dispatch to the best kernel variant for that specific hardware, (c) vendor-agnostic abstraction at the ML framework level.
- HetGPU (2025) addresses binary compatibility but not ML-kernel-aware dispatch.
- PyTorch's dispatcher handles device types but not intra-device capability variants (e.g., A100 vs H100 both show as `cuda:0`).
- The poster's contribution: a prototype dispatch runtime that queries compute capability / hardware properties at startup and routes to pre-compiled kernel variants, demonstrating this is both feasible and impactful for ML workloads.

---

*Last updated: 2026-04-02*
*Compiled for LLVM Dublin 2026 poster: "Vendor-Agnostic Runtime Dispatch for ML Kernels on Heterogeneous GPUs"*
