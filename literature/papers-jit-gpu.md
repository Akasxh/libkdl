# Literature Survey: JIT Compilation for GPUs
## For: LLVM Dublin 2026 Poster — Heterogeneous GPU Kernel Dispatch
**Compiled:** 2026-04-02
**Target paper count:** 15+ | **Actual:** 22 papers + 3 technical references

---

## Theme 1: GPU JIT Compilation — Core Systems

### 1. Proteus: Portable Runtime Optimization of GPU Kernel Execution with Just-in-Time Compilation
- **Authors:** Giorgis Georgakoudis, Konstantinos Parasyris, David Beckingsale
- **Venue:** 23rd ACM/IEEE International Symposium on Code Generation and Optimization (CGO 2025)
- **Year:** 2025
- **DOI:** https://doi.org/10.1145/3696443.3708939
- **URL:** https://dl.acm.org/doi/abs/10.1145/3696443.3708939
- **Key contribution:** Easy-to-use, portable, lightweight JIT compilation framework for GPU kernels. Embeds seamlessly into C++ codebases; accelerates CUDA, HIP, and host-only C/C++. Core technique is runtime constant folding, which replaces runtime values with constants during JIT compilation to turbocharge classical compiler optimizations (loop unrolling, control-flow simplification, constant propagation). Achieves up to 2.8× speedup on AMD, 1.78× on NVIDIA vs AOT; 1.23× better than CUDA-specific Jitify on average.
- **Relevance:** This is the closest prior art to the poster's central claim. Must cite prominently. Uses LLVM IR as the portable substrate — directly relevant.

### 2. Extending RAJA Parallel Programming Abstractions with Just-In-Time Optimization
- **Authors:** Bowen J., Parasyris K., Beckingsale D., Ben-Nun T., Stitt T., Georgakoudis G.
- **Venue:** SC '25 Workshops (P3HPC), International Conference for High Performance Computing, Networking, Storage and Analysis
- **Year:** 2025
- **DOI:** https://doi.org/10.1145/3731599.3767492
- **URL:** https://dl.acm.org/doi/10.1145/3731599.3767492
- **Key contribution:** Extends Proteus to support indirect kernel launching through RAJA's abstractions. Introduces dimension specialization JIT optimization. Speedups from 1.2× up to 23× on AMD MI250X; 1.1× to 15× on NVIDIA V100 with no slowdowns.
- **Relevance:** Shows Proteus+RAJA integration — application-level JIT benefit, not just kernel-level.

### 3. The OoO VLIW JIT Compiler for GPU Inference
- **Authors:** Paras Jain, Xiangxi Mo, Ajay Jain, Alexey Tumanov, Joseph E. Gonzalez, Ion Stoica
- **Venue:** arXiv preprint (2019)
- **Year:** 2019
- **arXiv:** 1901.10008
- **URL:** https://arxiv.org/abs/1901.10008
- **Key contribution:** VLIW-inspired Out-of-Order JIT compiler that coalesces and reorders execution kernels at runtime for throughput-optimal GPU utilization while satisfying latency SLOs. Demonstrates 7.7× opportunity gap through spatial coalescing. Targets ML inference underutilization.
- **Relevance:** Motivates runtime dispatch: static kernel scheduling leaves GPU throughput on the table.

### 4. Leo: A Profile-Driven Dynamic Optimization Framework for GPU Applications
- **Authors:** Naila Farooqui, Christopher J. Rossbach, Yuan Yu, Karsten Schwan
- **Venue:** USENIX Conference on Timely Results in Operating Systems (TRIOS '14)
- **Year:** 2014
- **URL:** https://www.usenix.org/conference/trios14/technical-sessions/presentation/farooqui
- **PDF:** https://www.usenix.org/system/files/conference/trios14/trios14-paper-farooqui.pdf
- **Key contribution:** Automated GPU optimization using dynamic instrumentation to inform profile-driven optimizations. Achieves 1.12× to 27× kernel runtime speedup, 9–40% end-to-end improvement. Uses JIT recompilation to construct control-flow graphs on-the-fly.
- **Relevance:** Early work on profile-guided GPU JIT — establishes the pattern our system generalizes.

---

## Theme 2: Multi-Versioned Kernels and Runtime Specialization

### 5. Retargeting and Respecializing GPU Workloads for Performance Portability
- **Authors:** Ivan R. Ivanov, Oleksandr Zinenko, Jens Domke, Toshio Endo, William S. Moses
- **Venue:** 22nd IEEE/ACM International Symposium on Code Generation and Optimization (CGO 2024)
- **Year:** 2024
- **DOI:** https://doi.org/10.1109/CGO57630.2024.10444828
- **URL:** https://dl.acm.org/doi/10.1109/CGO57630.2024.10444828
- **PDF:** https://c.wsmoses.com/papers/polygeist24.pdf
- **Key contribution:** Introduces alternative regions in IR to support compile-time multi-versioning. Replicates kernel body region with different thread coarsening factors — automatic multi-versioned kernel generation from a single CUDA source via the Polygeist/MLIR toolchain.
- **Relevance:** Direct technical precedent for multi-version dispatch. Shows MLIR-based approach to the problem.

### 6. ProSpec: Profile-guided Specialization for GPU Kernels
- **Authors:** (Authors not fully extracted — ScienceDirect behind paywall)
- **Venue:** Information and Software Technology (Elsevier), 2025
- **Year:** 2025
- **DOI:** https://doi.org/10.1016/j.infsof.2025.00240X (pii: S095058492500240X)
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S095058492500240X
- **Key contribution:** Profile-guided specialization for GPU kernels — offloads profile collection to CPUs, analyzes inefficiency patterns dependent on multiple hot values, generates optimization feedback for automatic kernel specialization.
- **Relevance:** Profile-guided approach to choosing kernel variants; complements our runtime dispatch model.

### 7. Stream-K++: Adaptive GPU GEMM Kernel Scheduling and Selection using Bloom Filters
- **Authors:** Harisankar Sadasivan, Muhammed Emin Ozturk, Muhammad Osama, Chris Millette, Astha Rai, Maksim Podkorytov, John Afaganis, Carlus Huang, Jing Zhang, Jun Liu
- **Venue:** arXiv (cs.DC), 2024
- **Year:** 2024
- **arXiv:** 2408.11417
- **URL:** https://arxiv.org/abs/2408.11417
- **Key contribution:** Adaptive GEMM kernel selection mechanism: expands Stream-K scheduling from 3 to 7 policies; introduces bloom-filter-based configuration selection that eliminates up to 95.8% of unsuitable configs while maintaining 100% true-negative rate. Up to 43% improvement on AMD MI250X.
- **Relevance:** Runtime kernel selection at the operator level — exactly the problem our system addresses for ML workloads.

### 8. GPU Performance Portability Needs Autotuning
- **Authors:** Burkhard Ringlein, Thomas Parnell, Radu Stoica
- **Venue:** arXiv (cs.AR), 2025
- **Year:** 2025
- **arXiv:** 2505.03780
- **URL:** https://arxiv.org/abs/2505.03780
- **Key contribution:** Demonstrates that combining JIT compilation with kernel parameter autotuning achieves portable LLM inference across GPU vendors without code changes. Explores up to 15× more configurations than manual approaches; achieves up to 230% improvement over vendor-optimized implementations; reduces kernel code size by 70×.
- **Relevance:** Motivates JIT+autotuning as the solution to performance portability in LLM inference specifically.

---

## Theme 3: Auto-Tuning GPU Kernels

### 9. Ansor: Generating High-Performance Tensor Programs for Deep Learning
- **Authors:** Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica
- **Venue:** 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI '20)
- **Year:** 2020
- **arXiv:** 2006.06762
- **URL:** https://www.usenix.org/conference/osdi20/presentation/zheng
- **PDF:** https://www.usenix.org/system/files/osdi20-zheng.pdf
- **Key contribution:** Auto-scheduler for tensor programs that samples from hierarchical search space, uses evolutionary search + learned cost models. Up to 3.8× on Intel CPUs, 2.6× on ARM, 1.7× on NVIDIA GPUs vs prior SOTA.
- **Relevance:** Establishes learned cost model + search as the auto-tuning paradigm our dispatch can use.

### 10. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- **Authors:** Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy
- **Venue:** 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18)
- **Year:** 2018
- **arXiv:** 1802.04799
- **URL:** https://www.usenix.org/conference/osdi18/presentation/chen
- **Key contribution:** End-to-end compiler with graph-level and operator-level optimizations for performance portability across diverse hardware. Novel learning-based cost modeling for optimization search. 1.2×–3.8× speedup over prior frameworks.
- **Relevance:** Foundational work on ML compiler auto-tuning; Ansor is a follow-on. TVM uses JIT compilation internally.

### 11. Benchmarking Optimization Algorithms for Auto-Tuning GPU Kernels
- **Authors:** Richard Schoonhoven, Ben van Werkhoven, Kees Joost Batenburg
- **Venue:** IEEE Transactions on Evolutionary Computation, 2022
- **Year:** 2022
- **arXiv:** 2210.01465
- **URL:** https://arxiv.org/abs/2210.01465
- **Key contribution:** Comprehensive empirical study comparing 16 black-box optimization algorithms for GPU kernel auto-tuning across 26 kernel spaces on 9 GPUs. Introduces PageRank-based difficulty metric that predicts tuning performance.
- **Relevance:** Benchmarks the landscape of search strategies for auto-tuning — informs algorithm choice for our dispatch.

### 12. Kernel Tuner: A Search-Optimizing GPU Code Auto-Tuner
- **Authors:** Ben van Werkhoven
- **Venue:** Future Generation Computer Systems, Vol. 90, pp. 347–358
- **Year:** 2019
- **URL:** https://www.sciencedirect.com/science/article/pii/S0167739X18313359
- **Key contribution:** Open-source Python framework for auto-tuning CUDA, HIP, OpenCL, and C kernels. Supports 20+ search strategies (Bayesian optimization, genetic algorithms, simulated annealing, etc.). Widely used reference framework.
- **Relevance:** Establishes the tooling infrastructure for offline tuning that complements online JIT dispatch.

### 13. MLKAPS: Machine Learning and Adaptive Sampling for HPC Kernel Auto-tuning
- **Authors:** Mathys Jam, Eric Petit, Pablo de Oliveira Castro, David Defour, Greg Henry, William Jalby
- **Venue:** arXiv preprint, 2025
- **Year:** 2025
- **arXiv:** 2501.05811
- **URL:** https://arxiv.org/abs/2501.05811
- **Key contribution:** ML-driven auto-tuner that generates decision trees for runtime kernel selection without manual effort. Scalable to large input/design spaces. Applied to Intel MKL kernels, achieving geomean speedup of 1.30× (dgetrf) and 1.18× (dgeqrf) on 85%+ of inputs.
- **Relevance:** Decision tree for runtime selection is exactly a lightweight dispatch policy — directly applicable.

### 14. CLTune: A Generic Auto-Tuner for OpenCL Kernels
- **Authors:** Cedric Nugteren, Valeriu Codreanu
- **Venue:** arXiv, 2017 (originally presented at ASAP 2015)
- **Year:** 2015/2017
- **arXiv:** 1703.06503
- **URL:** https://arxiv.org/abs/1703.06503
- **Key contribution:** Generic auto-tuner for OpenCL kernels that explores user-defined search spaces of compile-time and runtime parameters (workgroup size, tile sizes, vector types, unroll factors). Supports simulated annealing and particle swarm optimization.
- **Relevance:** Early cross-vendor (OpenCL) auto-tuning tool — precursor to vendor-agnostic JIT dispatch.

---

## Theme 4: Fat Binaries and Multi-Architecture GPU Binaries

### 15. HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs
- **Authors:** Yiwei Yang, Yusheng Zheng, Tong Yu, Andi Quinn
- **Venue:** arXiv (cs.AR), 2025
- **Year:** 2025
- **arXiv:** 2506.15993
- **URL:** https://arxiv.org/abs/2506.15993
- **Key contribution:** System (compiler + runtime + abstraction layer) enabling a single GPU binary to execute on NVIDIA, AMD, Intel, and Tenstorrent hardware. Emits architecture-agnostic IR with metadata; runtime performs dynamic translation to native code. Enables live GPU migration across disparate hardware with minimal overhead.
- **Relevance:** Most ambitious fat-binary / heterogeneous binary work to date. Our dispatch system is a design point between fat-binary (too static) and full dynamic translation (too heavy).

### 16. CUDA Fat Binary and PTX JIT Compilation (NVIDIA Technical Reference)
- **Source:** NVIDIA Developer Blog: "CUDA Pro Tip: Understand Fat Binaries and JIT Caching"
- **URL:** https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
- **Key contribution:** Documents CUDA's fat binary format: embeds multiple cubins and PTX versions; CUDA driver selects optimal translation at runtime; PTX serves as forward-compatible virtual ISA allowing execution on future GPUs via JIT.
- **Relevance:** The incumbent NVIDIA approach to multi-arch binary — our work generalizes this across vendors.

### 17. AMDGPU Generic Targets: Build-Once-Run-on-Multiple-GPUs
- **Source:** LLVM upstream development, 2024 (Phoronix report)
- **URL:** https://www.phoronix.com/news/LLVM-AMDGPU-Generic-GFX
- **Key contribution:** LLVM AMDGPU backend adds generic targets (gfx9-generic, gfx10.1-generic, gfx10.3-generic, gfx11-generic) that target a GPU generation family. One binary runs across multiple GPUs in the family at the cost of some optimization specificity. Merged into LLVM mainline in February 2024.
- **Relevance:** AMD's answer to NVIDIA fat binary — intra-vendor multi-arch. Our work provides cross-vendor generalization.

---

## Theme 5: Portable Compilation Infrastructure (MLIR, SPIR-V, LLVM)

### 18. MLIR: A Compiler Infrastructure for the End of Moore's Law
- **Authors:** Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko
- **Venue:** arXiv, 2020 (presented at CGO 2021)
- **Year:** 2020
- **arXiv:** 2002.11054
- **URL:** https://arxiv.org/abs/2002.11054
- **Key contribution:** Compiler infrastructure addressing software fragmentation for heterogeneous hardware. Multi-level IR with progressive lowering; enables code generators, translators, and optimizers at different abstraction levels. Foundation for IREE, Triton, and modern ML compilers.
- **Relevance:** The infrastructure substrate our heterogeneous dispatch system would be built on.

### 19. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
- **Authors:** Philippe Tillet, H. T. Kung, David Cox
- **Venue:** 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL '19), co-located with PLDI 2019
- **Year:** 2019
- **DOI:** https://doi.org/10.1145/3315508.3329973
- **URL:** https://dl.acm.org/doi/10.1145/3315508.3329973
- **PDF:** https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- **Key contribution:** Tile-based intermediate language and JIT compiler for neural network computations. @triton.jit walks Python AST to generate Triton IR, which is then compiled to LLVM IR and PTX. Enables high-performance GPU kernels from Python-level code.
- **Relevance:** Triton is now PyTorch's default GPU kernel backend — understanding its JIT model is essential context.

### 20. One Pass to Bind Them: The First Single-Pass SYCL Compiler with Unified Code Representation Across Backends
- **Authors:** Aksel Alpay, Vincent Heuveline
- **Venue:** Proceedings of the 2023 International Workshop on OpenCL (IWOCL '23)
- **Year:** 2023
- **DOI:** https://doi.org/10.1145/3585341.3585351
- **URL:** https://dl.acm.org/doi/abs/10.1145/3585341.3585351
- **Key contribution:** First SYCL compiler with single-source, single compiler pass (SSCP) design. Stores unified LLVM IR in the application binary; runtime lowers to PTX (NVIDIA), SPIR-V (Intel), or amdgcn (AMD). Achieves "universal" binaries running across all vendors with only 20% compilation overhead vs regular clang host build.
- **Relevance:** Demonstrates feasibility of LLVM IR as the universal portable binary format — key technical reference for our design.

### 21. SPIR-V: The Industry Open Standard Intermediate Language for Parallel Compute and Graphics
- **Source:** Khronos Group Whitepaper, 2015
- **URL:** https://registry.khronos.org/SPIR-V/papers/WhitePaper.pdf
- **Key contribution:** Defines SPIR-V binary format as portable parallel compute IR. Splits compiler chain: front-ends emit SPIR-V; drivers JIT-compile to native ISA at runtime. Used by Vulkan, OpenCL, SYCL. Released November 2015 alongside OpenCL 2.1.
- **Relevance:** SPIR-V is the incumbent cross-vendor IR; our work must position against and possibly leverage it.

### 22. Intel oneAPI / NEO: JIT Compilation via SPIR-V for Intel GPUs
- **Source:** Intel Developer Documentation: "Just-In-Time Compilation in SYCL"
- **URL:** https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/just-in-time-compilation-in-sycl.html
- **Key contribution:** In Intel's JIT flow, SYCL device code is compiled to SPIR-V and embedded in the fat binary; the NEO compute runtime translates SPIR-V to Intel GPU ISA at kernel launch time. Full pipeline: DPC++ → SPIR-V → NEO JIT → GEN ISA.
- **Relevance:** Documents Intel's vendor-specific JIT compilation path — the third pillar alongside NVIDIA (NVRTC/PTX) and AMD (HIPRTC/AMDGPU IR).

---

## Theme 6: Runtime AMD and NVIDIA Compilation Infrastructure

### 23. AMD HIPRTC: Runtime Compilation for HIP
- **Source:** AMD ROCm Documentation: "Programming for HIP Runtime Compiler (RTC)"
- **URL:** https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
- **Key contribution:** HIPRTC API allows compiling HIP kernels at runtime from source strings. Leverages AMD's Code Object Manager (Comgr) internally. AMDGPU IR is compiled (JIT or AOT) to GFX ISA. Multi-stage compilation allows forward compatibility across GPU generations.
- **Relevance:** AMD's JIT compilation runtime — the AMD-side analog of NVRTC that our dispatch infrastructure must interact with.

### 24. NVIDIA NVRTC and Jitify: CUDA Runtime Compilation
- **Source:** NVIDIA Technical Blog: "Efficient Transforms in cuDF Using JIT Compilation" + GitHub: NVIDIA/jitify
- **URL:** https://developer.nvidia.com/blog/efficient-transforms-in-cudf-using-jit-compilation/
- **URL2:** https://github.com/NVIDIA/jitify
- **Key contribution:** NVRTC compiles CUDA C++ source strings to PTX at runtime. Jitify is a single-header C++ library that simplifies NVRTC usage, handles header dependencies, and manages JIT cache. Used in cuDF for kernel fusion; achieves up to 435× speedup over pandas.apply on GV100. JIT compilation time ~600 ms, managed via cache.
- **Relevance:** The incumbent NVIDIA JIT infrastructure. Proteus is shown to outperform Jitify by 1.23× on average.

---

## Summary Table

| # | Paper | Year | Venue | Theme |
|---|-------|------|-------|-------|
| 1 | Proteus (Georgakoudis et al.) | 2025 | CGO | JIT Core |
| 2 | Extending RAJA with JIT (Bowen et al.) | 2025 | SC Workshops | JIT Core |
| 3 | OoO VLIW JIT (Jain et al.) | 2019 | arXiv | JIT Core |
| 4 | Leo (Farooqui et al.) | 2014 | TRIOS | JIT Core |
| 5 | Retargeting GPU Workloads (Ivanov et al.) | 2024 | CGO | Multi-Version |
| 6 | ProSpec (TBD) | 2025 | IST (Elsevier) | Multi-Version |
| 7 | Stream-K++ (Sadasivan et al.) | 2024 | arXiv | Multi-Version |
| 8 | GPU Portability Needs Autotuning (Ringlein et al.) | 2025 | arXiv | Multi-Version |
| 9 | Ansor (Zheng et al.) | 2020 | OSDI | Auto-Tuning |
| 10 | TVM (Chen et al.) | 2018 | OSDI | Auto-Tuning |
| 11 | Benchmarking Autotuning (Schoonhoven et al.) | 2022 | IEEE Trans. Evol. Comp. | Auto-Tuning |
| 12 | Kernel Tuner (van Werkhoven) | 2019 | FGCS | Auto-Tuning |
| 13 | MLKAPS (Jam et al.) | 2025 | arXiv | Auto-Tuning |
| 14 | CLTune (Nugteren, Codreanu) | 2015 | ASAP / arXiv | Auto-Tuning |
| 15 | HetGPU (Yang et al.) | 2025 | arXiv | Fat Binary |
| 16 | CUDA Fat Binary (NVIDIA) | — | NVIDIA Blog | Fat Binary |
| 17 | AMDGPU Generic Targets (LLVM) | 2024 | LLVM upstream | Fat Binary |
| 18 | MLIR (Lattner et al.) | 2020 | CGO 2021 | Infrastructure |
| 19 | Triton (Tillet et al.) | 2019 | MAPL@PLDI | Infrastructure |
| 20 | AdaptiveCpp SSCP (Alpay, Heuveline) | 2023 | IWOCL | Infrastructure |
| 21 | SPIR-V Whitepaper (Khronos) | 2015 | Khronos | Infrastructure |
| 22 | Intel NEO JIT (Intel) | — | Intel Docs | Infrastructure |
| 23 | AMD HIPRTC (AMD) | — | ROCm Docs | Vendor Runtime |
| 24 | NVIDIA NVRTC + Jitify (NVIDIA) | — | NVIDIA | Vendor Runtime |

---

## Key Gaps and Research Opportunities Identified

1. **No cross-vendor unified JIT dispatch paper exists.** Proteus is portable but targets one vendor at a time. AdaptiveCpp SSCP and HetGPU are the closest, but neither provides adaptive dispatch based on runtime workload characteristics.

2. **LLVM ORC JIT for GPU targets** — the LLVM GPU news newsletters and upstream PRs (e.g., embedding LLVM IR for future JIT in OpenMP offloading) show intent but no published research paper exists on using LLVM ORC JIT directly for NVPTX/AMDGPU targets. This is a gap our poster can claim.

3. **Runtime workload-adaptive kernel selection** — Stream-K++ and MLKAPS address this for specific operators (GEMM). Generalizing to arbitrary ML kernels with a JIT dispatch layer is novel.

4. **Profile-guided cross-vendor specialization** — Leo (2014) and ProSpec (2025) do profile-guided optimization on single vendors. Cross-vendor profile-guided dispatch is unstudied.

5. **Fat binary vs JIT trade-off analysis** — HetGPU goes full dynamic translation; CUDA fat binary goes full static. No paper studies the design space between these extremes with a principled cost model.

---

## Search Queries Used
- "JIT compilation GPU kernels" (ACM DL, arXiv, USENIX)
- "multi-versioned GPU kernels runtime specialization"
- "LLVM ORC JIT GPU targets NVPTX AMDGPU"
- "fat binary GPU multi-architecture binary CUDA PTX"
- "adaptive GPU kernel selection auto-tuning runtime"
- "profile-guided GPU kernel optimization"
- "AMD ROCm HIP JIT runtime compilation"
- "Intel oneAPI NEO compiler GPU JIT SPIR-V"
- "Proteus portable runtime optimization GPU JIT CGO 2025"
- "GPU performance portability autotuning Triton PyTorch"
- "Ansor TVM OSDI 2020 authors"
- "Triton compiler MAPL 2019 Tillet"
- "AdaptiveCpp hipSYCL SSCP JIT runtime LLVM IR"
- "HetGPU binary compatibility GPU heterogeneous"
- "CLTune generic auto-tuner OpenCL kernels"
- "MLIR compiler infrastructure heterogeneous"
