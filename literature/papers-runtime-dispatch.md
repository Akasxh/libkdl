# Literature Review: Heterogeneous Runtime GPU Kernel Dispatch
## LLVM Dublin 2026 Poster — Vendor-Agnostic Runtime Dispatch for ML Kernels

**Search conducted:** 2026-04-02
**Coverage:** SC, HPDC, PPoPP, CGO, CC, ASPLOS, ICS, Euro-Par, IWOCL, arXiv (2021–2026)
**Total papers catalogued:** 26

---

## Theme A: Portable GPU Programming Models

Papers establishing the foundations and frameworks for writing GPU code that targets multiple hardware vendors from a single source.

---

### A1. Kokkos 3: Programming Model Extensions for the Exascale Era
**Authors:** Christian R. Trott, Damien Lebrun-Grandie, Daniel Arndt, Jan Ciesko, Vinh Dang, Nathan Ellingwood, Rahulkumar Gayatri, Evan Harvey, Daisy S. Hollman, Dan Ibanez, Nevin Liber, Jonathan Madsen, Jeff Miles, David Poliakoff, Amy Powell, Sivasankaran Rajamanickam, Mikael Simberg, Dan Sunderland, Bruno Turcksin, Jeremiah Wilke
**Venue:** IEEE Transactions on Parallel and Distributed Systems, Vol. 33(4), pp. 805–817
**Year:** 2022
**DOI:** 10.1109/TPDS.2021.3097283
**URL:** https://ieeexplore.ieee.org/document/9485033/

Kokkos is the canonical C++ performance portability library used across US national labs. This paper describes version 3 extensions including hierarchical parallelism, containers, task graphs, and arbitrary-sized atomic operations targeting exascale heterogeneous architectures. It compiles a single-source application to CUDA, HIP, OpenMP, and SYCL backends. Demonstrated 12–17x speedup on GPU-only vs CPU-only nodes in large production codes.

**Relevance:** Baseline for evaluating any new dispatch mechanism against established portability layers.

---

### A2. Taking GPU Programming Models to Task for Performance Portability
**Authors:** Joshua H. Davis, Pranav Sivaraman, Joy Kitson, Konstantinos Parasyris, Harshitha Menon, Isaac Minn, Giorgis Georgakoudis, Abhinav Bhatele
**Venue:** Proceedings of the 39th ACM International Conference on Supercomputing (ICS '25)
**Year:** 2024 (arXiv), published ICS 2025
**DOI:** 10.1145/3721145.3730423
**arXiv:** https://arxiv.org/abs/2402.08950

The most comprehensive recent empirical comparison of GPU programming model portability. Evaluates CUDA, HIP, Kokkos, RAJA, OpenMP, OpenACC, and SYCL across five scientific proxy applications on NVIDIA (A100, V100) and AMD (MI250X) systems at Summit, Frontier, Perlmutter, and Corona. Finds that Kokkos and RAJA offer the best performance portability — neither makes guarantees about performance consistency across hardware. Uses Spack-based reproducible methodology.

**Relevance:** Directly motivates runtime dispatch: static compilation cannot guarantee performance portability; dynamic selection is needed.

---

### A3. AdaptiveCpp (hipSYCL): One Pass to Bind Them — First Single-Pass SYCL Compiler with Unified Code Representation
**Authors:** Aksel Alpay, Vincent Heuveline
**Venue:** Proceedings of the 2023 International Workshop on OpenCL (IWOCL '23), Article 7, pp. 1–12
**Year:** 2023
**DOI:** 10.1145/3585341.3585348
**URL:** https://dl.acm.org/doi/10.1145/3585341.3585348

AdaptiveCpp (formerly hipSYCL) introduces a single-source, single compiler pass (SSCP) architecture that produces unified code representation across all backends (NVIDIA, AMD, Intel GPU, CPU) enabling true runtime specialization. The generic JIT compiler compiles device kernels at application start using LLVM-based infrastructure, allowing the same binary to run on any hardware discovered at runtime.

**Relevance:** The only SYCL implementation with unified JIT compilation across all backends — directly relevant to runtime dispatch architecture.

---

### A4. Adaptivity in AdaptiveCpp: Optimizing Performance by Leveraging Runtime Information During JIT-Compilation
**Authors:** Aksel Alpay, Vincent Heuveline
**Venue:** Proceedings of the 13th International Workshop on OpenCL and SYCL (IWOCL '25)
**Year:** 2025
**DOI:** 10.1145/3731125.3731127
**URL:** https://dl.acm.org/doi/10.1145/3731125.3731127

Extends AdaptiveCpp with runtime-adaptive JIT that leverages dynamic hardware information during compilation — kernel parameters, occupancy, memory layout — to generate optimized code at launch time. Achieves up to 30% improvement over CUDA in geometric mean. Supports CPUs, Intel, NVIDIA, and AMD GPUs from a single binary without recompilation.

**Relevance:** Strongest current evidence that JIT-based runtime dispatch can beat vendor-specific static compilation.

---

### A5. chipStar: Making HIP/CUDA Applications Cross-Vendor Portable via OpenCL and SPIR-V
**Authors:** Paulius Velesko, Pekka Jääskeläinen, Henry Linjamäki, Michal Babej, Peng Tu, Sarbojit Sarkar, Ben Ashbaugh, Colleen Bertoni, Jenny Chen, Philip C. Roth, Wael Elwasif, Rahulkumar Gayatri, Jisheng Zhao, Karol Herbst, Kevin Harms, Brice Videau
**Venue:** The International Journal of High Performance Computing Applications
**Year:** 2026
**DOI:** 10.1177/10943420261423001
**URL:** https://journals.sagepub.com/doi/10.1177/10943420261423001

chipStar enables unmodified CUDA and HIP programs to run on any OpenCL/SPIR-V capable device — including RISC-V/PowerVR and ARM Mali — by compiling through LLVM to SPIR-V and dispatching via OpenCL or Level Zero at runtime. Achieves geometric mean 0.75 vs native AMD HIP. Demonstrates that SPIR-V is a viable runtime portable IR for CUDA/HIP programs without source changes.

**Relevance:** Key reference for SPIR-V as the portable runtime dispatch layer for existing CUDA/HIP kernels.

---

### A6. GROMACS on AMD GPU-Based HPC Platforms: Using SYCL for Performance and Portability
**Authors:** Andrey Alekseenko, Szilárd Páll, Erik Lindahl
**Venue:** Proceedings of the Cray User Group (CUG 2024), pp. 71–84
**Year:** 2024
**arXiv:** https://arxiv.org/abs/2405.01420
**DOI:** 10.1145/3725789.3725797

Production case study of GROMACS adopting SYCL as primary GPU backend since 2022, targeting AMD MI250X on LUMI/Frontier (exascale). Demonstrates that portability is achievable without major performance sacrifice. Key finding: AdaptiveCpp's instant submission mode improved SYCL runtime performance by up to 22% by eliminating latency-sensitive scheduling overhead. Shows SYCL runtime task-launch behavior is a critical dispatch bottleneck.

**Relevance:** Real-world evidence that SYCL-based dispatch, when properly optimized, matches CUDA performance in production HPC workloads.

---

### A7. Performant Unified GPU Kernels for Portable Singular Value Computation Across Hardware and Precision
**Authors:** Evelyne Ringoot, Rabab Alomairy, Valentin Churavy, Alan Edelman
**Venue:** 54th International Conference on Parallel Processing (ICPP '25)
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2508.06339
**DOI:** 10.1145/3754598.3754667

Demonstrates that a single Julia implementation using KernelAbstractions.jl and GPUArrays.jl can outperform most vendor-specific linear algebra libraries (MAGMA, SLATE, rocSOLVER, oneMKL) for SVD on matrices >1024x1024 across NVIDIA, AMD, Intel, and Apple Metal GPUs. Achieves 80–90% of cuSOLVER performance for large matrices. First GPU-accelerated SVD supporting Apple Metal and half precision.

**Relevance:** Demonstrates that unified kernel dispatch achieves competitive performance without architecture-specific code.

---

### A8. Retargeting and Respecializing GPU Workloads for Performance Portability
**Authors:** Ivan R. Ivanov, Oleksandr Zinenko, Jens Domke, Toshio Endo, William S. Moses
**Venue:** 22nd IEEE/ACM International Symposium on Code Generation and Optimization (CGO '24), Edinburgh
**Year:** 2024
**DOI:** 10.1109/CGO57630.2024.10444828
**URL:** https://dl.acm.org/doi/10.1109/CGO57630.2024.10444828

Extends Polygeist to accept CUDA source and compile it to both NVIDIA and AMD targets through architecture-aware retargeting and respecialization — including kernel granularity selection, shared memory layout adaptation, and occupancy tuning. Evaluated on Rodinia v3 (24 benchmarks) and HeCBench (400 benchmarks). Produces portable binaries that specialize to the detected hardware at compile time without manual porting effort.

**Relevance:** Directly addresses the CUDA-to-heterogeneous compilation gap; core technique for portable ML kernel dispatch.

---

## Theme B: Runtime Dispatch Mechanisms

Papers focusing on the dynamic selection, scheduling, and execution of kernels based on runtime hardware detection.

---

### B1. HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs
**Authors:** Yiwei Yang, Yusheng Zheng, Tong Yu, Andi Quinn
**Venue:** arXiv
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2506.15993

Proposes a three-layer system — compiler (emits architecture-agnostic GPU IR), runtime (dynamically translates IR to native code), and abstraction layer (unified thread/memory/synchronization model) — that allows a single GPU binary to execute on NVIDIA, AMD, Intel, and Tenstorrent hardware. Addresses the SIMT-vs-MIMD execution model gap between conventional GPU warps and Tenstorrent's RISC-V cores. Demonstrates live migration of GPU workloads across vendors.

**Relevance:** Closest existing work to a complete vendor-agnostic runtime dispatch stack.

---

### B2. Toward a Universal GPU Instruction Set Architecture
**Authors:** Ojima Abraham, Onyinye Okoli
**Venue:** arXiv
**Year:** 2026
**arXiv:** https://arxiv.org/abs/2603.28793

First systematic cross-vendor ISA analysis spanning NVIDIA (PTX v1.0–v9.2), AMD (RDNA 1–4, CDNA 1–4), Intel (Gen11 through Xe-HPC), and Apple (G13 reverse-engineered), covering 16 microarchitectures and 5,000+ pages of documentation. Identifies 10 hardware-invariant computational primitives, 6 parameterizable dialects, and 6 fundamental architectural divergences. Proposes an abstract execution model validated on NVIDIA T4 and Apple M1 with near-native performance on 5/6 benchmark-platform pairs.

**Relevance:** Provides ISA-level grounding for a portable dispatch IR — identifies what can be unified and what cannot.

---

### B3. Runtime Support for Performance Portability on Heterogeneous Distributed Platforms
**Authors:** Polykarpos Thomadakis, Nikos Chrisochoides
**Venue:** arXiv
**Year:** 2023
**arXiv:** https://arxiv.org/abs/2303.02543

Presents a unified runtime framework for heterogeneous multi-GPU systems that achieves up to 300% improvement on single devices and outperforms MPI+CUDA by up to 20% for large messages in distributed settings. Integrates task-based scheduling with over-decomposition and GPU-aware communication, maintaining <10% overhead for small messages. Demonstrates linear scalability across four GPUs.

**Relevance:** Runtime dispatch infrastructure for multi-device heterogeneous nodes.

---

### B4. GPU Performance Portability needs Autotuning
**Authors:** Burkhard Ringlein, Thomas Parnell, Radu Stoica
**Venue:** arXiv
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2505.03780

Argues that JIT compilation alone is insufficient for GPU portability — systematic autotuning of kernel parameters is required. Case study on flash attention for LLM inference across NVIDIA A100 and AMD MI250. Explores 15x more parameter configurations than baselines, achieves >230% improvement over vendor-optimized implementations, reduces kernel code size by 70x by eliminating manual specializations. Demonstrates vendor-agnostic LLM inference without code changes.

**Relevance:** Quantifies the performance gap that runtime autotuning closes over static compilation — key argument for JIT dispatch.

---

### B5. Bringing Auto-tuning to HIP: Analysis of Tuning Impact and Difficulty on AMD and Nvidia GPUs
**Authors:** Milo Lurati, Stijn Heldens, Alessio Sclocco, Ben van Werkhoven
**Venue:** Euro-PAR 2024 (Best Paper Award)
**Year:** 2024
**arXiv:** https://arxiv.org/abs/2407.11488
**DOI:** 10.1007/978-3-031-69577-3_7

Extends Kernel Tuner to support AMD HIP, providing the first systematic auto-tuning comparison across AMD and NVIDIA hardware. Key finding: auto-tuning impact is 10x higher on AMD than NVIDIA (2x) — AMD kernels tuned for NVIDIA perform poorly, while AMD-tuned kernels transfer better to NVIDIA. Demonstrates that HIP provides code portability but not performance portability without hardware-specific tuning.

**Relevance:** Empirical case for runtime hardware detection + device-specific dispatch over static cross-compilation.

---

### B6. A Few Fit Most: Improving Performance Portability of SGEMM on GPUs Using Multi-Versioning
**Authors:** Robert Hochgraf, Sreepathi Pai
**Venue:** arXiv
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2507.15277

Proposes "portability tuning" — generating multiple code variants (multi-versioned kernels) at compile time and selecting the best at runtime based on detected device characteristics. Applied to SGEMM using CLBlast data. Achieves performance within 10% of the theoretical maximum on seen and unseen devices without device-specific retuning. Superior generalization compared to single-kernel autotuning.

**Relevance:** Multi-versioning is a concrete runtime dispatch mechanism; this paper provides methodology and empirical support.

---

### B7. Implementing Multi-GPU Scientific Computing Miniapps Across Performance Portable Frameworks
**Authors:** Johansell Villalobos, Josef Ruzicka, Silvio Rizzi
**Venue:** arXiv (Symposium on Computer Architecture and High Performance Computing, SBAC-PAD)
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2511.02655

Benchmarks N-body and structured grid applications across Kokkos, OpenMP, RAJA, and OCCA on four NVIDIA A100 GPUs (Polaris supercomputer). Key finding: OCCA's JIT compilation produces faster execution for small-scale problems. Identifies OCCA's portability advantage from its runtime-JIT dispatch model, while revealing scalability challenges relative to static frameworks.

**Relevance:** Empirical comparison showing JIT-based runtime (OCCA) outperforms static frameworks for small kernels — supports dynamic dispatch argument.

---

### B8. Zorua: Enhancing Programming Ease, Portability, and Performance in GPUs by Decoupling Programming Models from Resource Management
**Authors:** Nandita Vijaykumar, Kevin Hsieh, Gennady Pekhimenko, Samira Khan, Ashish Shrestha, Saugata Ghose, Phillip B. Gibbons, Onur Mutlu
**Venue:** 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO '16); extended arXiv 2018
**Year:** 2016 (conference), 2018 (arXiv)
**arXiv:** https://arxiv.org/abs/1802.02573

Foundational hardware-level paper decoupling the programmer-specified resource allocation from actual on-chip GPU resource management through virtualization. Demonstrates that tight coupling between software resource specification and hardware allocation causes portability and performance problems across GPU generations. Virtualizes registers, shared memory, and thread blocks transparently.

**Relevance:** Architectural motivation for why software-level dispatch abstraction is necessary — hardware won't solve this alone.

---

## Theme C: Multi-Target Compilation

Papers on compiler infrastructure that generates code for multiple GPU targets from a single IR.

---

### C1. Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor Compiler Construction
**Authors:** Nicolas Vasilache, Oleksandr Zinenko, Aart J.C. Bik, Mahesh Ravishankar, Thomas Raoux, Alexander Belyaev, Matthias Springer, Tobias Gysi, Diego Caballero, Stephan Herhut, Stella Laurenzo, Albert Cohen
**Venue:** arXiv (presented in MLIR open meetings, influences IREE and XLA)
**Year:** 2022
**arXiv:** https://arxiv.org/abs/2202.03293

Describes MLIR's structured and retargetable approach to tensor compiler construction via progressive lowering through Linalg → GPU → NVVM/ROCDL/SPIR-V dialects. Key principle: compiler passes compose modularity across abstraction levels, enabling the same Linalg-on-tensors program to target NVIDIA, AMD, and CPU without changing the algorithm. Foundation for IREE's multi-backend support.

**Relevance:** Core infrastructure enabling multi-target compilation in the MLIR ecosystem — directly relevant to LLVM toolchain work.

---

### C2. MLIR-Based Code Generation for GPU Tensor Cores
**Authors:** Navdeep Katel, Vivek Khandelwal, Uday Bondhugula
**Venue:** 31st ACM SIGPLAN International Conference on Compiler Construction (CC '22), Seoul
**Year:** 2022
**DOI:** 10.1145/3497776.3517770
**URL:** https://dl.acm.org/doi/10.1145/3497776.3517770

Builds an MLIR transformation and lowering pipeline to automatically generate near-peak-performance matmul code targeting NVIDIA tensor cores, including fusion with pointwise operators. Demonstrates that the MLIR GPU dialect can generate code competitive with hand-tuned CUTLASS through structured transformations. Extends the "Linalg on tensors" paradigm to hardware-specific instruction selection.

**Relevance:** Concrete demonstration of MLIR multi-target GPU compilation pipeline; near-peak performance without vendor-specific manual tuning.

---

### C3. TinyIREE: An ML Execution Environment for Embedded Systems from Compilation to Deployment
**Authors:** Hsin-I Cindy Liu, Marius Brehler, Mahesh Ravishankar, Nicolas Vasilache, Ben Vanik, Stella Laurenzo
**Venue:** IEEE Micro, Vol. 42(5), pp. 9–16
**Year:** 2022
**arXiv:** https://arxiv.org/abs/2205.14479
**DOI:** 10.1109/MM.2022.3178068

Presents IREE (Intermediate Representation Execution Environment), an MLIR-based end-to-end compiler and runtime that scales from embedded microcontrollers to datacenter GPUs. IREE dispatches ML workloads to CPU, GPU (via Vulkan/SPIR-V), and accelerators using a unified HAL (hardware abstraction layer). Generates SPIR-V for GPU targets enabling cross-vendor portability via the Vulkan runtime.

**Relevance:** Most complete production-grade example of multi-target ML compilation with runtime dispatch via SPIR-V/Vulkan.

---

### C4. Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs
**Authors:** Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yibo Zhu, Gennady Pekhimenko
**Venue:** 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '23), Vol. 2
**Year:** 2023
**DOI:** 10.1145/3575693.3575702
**URL:** https://dl.acm.org/doi/10.1145/3575693.3575702

Introduces task-mappings as first-class scheduling primitives embedded in tensor programs, enabling fine-grained control over computation assignment and ordering. Supports end-to-end compilation from PyTorch/ONNX to CUDA kernels. Outperforms ONNX Runtime and TVM by up to 1.48x while reducing tuning time 20x over AutoTVM. Generates hardware-centric schedules that are input-size agnostic.

**Relevance:** Demonstrates that embedding scheduling decisions in tensor programs achieves better portability across problem sizes than external autotuners.

---

### C5. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation
**Authors:** Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, et al. (50+ authors)
**Venue:** 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '24), Vol. 2
**Year:** 2024
**DOI:** 10.1145/3620665.3640366
**URL:** https://dl.acm.org/doi/10.1145/3620665.3640366

Describes the architecture of torch.compile: TorchDynamo (Python-level JIT capturing computation graphs), TorchInductor (backend compiler generating Triton for GPU and C++/OpenMP for CPU), and dynamic shape support. Achieves 2.27x inference speedup on NVIDIA A100 across 180+ real-world models. Multi-backend: Triton for GPU (CUDA and HIP), C++ for CPU — runtime selects backend based on detected device.

**Relevance:** De facto ML runtime dispatch mechanism in production; shows how dynamic compilation + backend selection works at scale.

---

### C6. KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
**Authors:** Gang Liao, Hongsen Qin, Ying Wang, et al. (38 authors)
**Venue:** arXiv
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2512.23236

Agentic framework that automates kernel generation and optimization across NVIDIA GPUs, AMD GPUs, and Meta's MTIA v3 accelerators. Uses graph-based search across Triton and CuTe DSL abstractions. Achieves 100% correctness on 250 benchmarks and 160 PyTorch operators across three hardware platforms; reduces kernel development time from weeks to hours; achieves up to 17x improvement over PyTorch baselines in production recommendation models.

**Relevance:** Industrial-scale validation of automated multi-target kernel generation with runtime dispatch across heterogeneous hardware.

---

### C7. AMOS: Enabling Automatic Mapping for Tensor Computations on Spatial Accelerators
**Authors:** Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang
**Venue:** 49th Annual International Symposium on Computer Architecture (ISCA '22)
**Year:** 2022
**DOI:** 10.1145/3470496.3527440
**URL:** https://dl.acm.org/doi/10.1145/3470496.3527440

Introduces an automatic mapping framework that maps tensor computations to hardware intrinsics (CUDA WMMA, PTX MMA, etc.) without requiring manual templates. Unlike AutoTVM/UNIT, AMOS is not restricted to fixed memory layouts, enabling exploration of a much larger optimization space. Demonstrated on NVIDIA tensor cores with results competitive to vendor-tuned CUTLASS.

**Relevance:** Compiler-level automatic intrinsic selection demonstrates the dispatch problem at hardware instruction granularity.

---

## Theme D: Performance Portability — Empirical Studies and Applications

Papers measuring or achieving performance portability across real GPU hardware, providing empirical grounding for dispatch research.

---

### D1. Evaluating Performance Portability of GPU Programming Models (SC '23 Poster)
**Authors:** Multiple (LLNL team)
**Venue:** SC '23 Technical Poster
**Year:** 2023
**URL:** https://sc23.supercomputing.org/proceedings/tech_poster/poster_files/rpost217s3-file3.pdf
**OSTI Report:** https://www.osti.gov/servlets/purl/2305595

Technical poster from Lawrence Livermore National Laboratory evaluating CUDA, HIP, Kokkos, OpenMP, OpenACC, and SYCL across production HPC proxy applications on NVIDIA (A100, V100) and AMD (MI250X) at Summit, Frontier, Perlmutter, and Corona. Companion to the ICS 2025 full paper (A2 above). Finds that no single programming model achieves consistently high performance across all apps and platforms.

**Relevance:** Empirical data directly motivating runtime selection over static compilation choice.

---

### D2. Towards Portability at Scale: A Cross-Architecture Performance Evaluation of a GPU-Enabled Shallow Water Solver
**Authors:** Johansell Villalobos, Daniel Caviedes-Voullieme, Silvio Rizzi, Esteban Meneses
**Venue:** SBAC-PAD 2025 (Symposium on Computer Architecture and High Performance Computing)
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2511.01001

Evaluates the SERGHEI-SWE Kokkos-based shallow water equations solver at scale across Frontier (AMD), JUWELS Booster (NVIDIA), JEDI (NVIDIA), and Aurora (Intel) — up to 1024 GPUs. Achieves speedup of 32 with >90% efficiency. Identifies memory bandwidth as the primary performance constraint across all architectures, with portability metrics below 70% for problem sizes requiring architecture-specific Kokkos team tuning.

**Relevance:** Shows that even Kokkos-based portability breaks down at scale without runtime tuning — supporting adaptive dispatch.

---

### D3. Analyzing Performance Portability for a SYCL Implementation of the 2D Shallow Water Equations
**Authors:** (Multiple; University of Innsbruck and collaborators)
**Venue:** The Journal of Supercomputing (Springer)
**Year:** 2025
**DOI:** 10.1007/s11227-025-07063-7
**URL:** https://link.springer.com/article/10.1007/s11227-025-07063-7

Analyzes performance portability of a SYCL implementation of 2D shallow water equations (discontinuous Galerkin, unstructured meshes) across x86/ARM CPUs, NVIDIA/AMD/Intel GPUs, and Intel Stratix 10 FPGAs. Finds good portability across CPU variants and GPU vendors. GPUs achieve up to 10x higher energy efficiency than CPUs per degree of freedom per joule. SYCL reaches comparable performance to CUDA on NVIDIA targets.

**Relevance:** Multi-vendor SYCL performance data across CPU+GPU+FPGA demonstrates the breadth of targets a dispatch layer must handle.

---

### D4. Bringing Auto-tuning to HIP (see also Theme B, B5)
Already catalogued under Theme B as B5.

---

### D5. Juliana: Automated Julia CUDA.jl Code Translation Across Multiple GPU Platforms
**Authors:** de la Calle, García, and collaborators
**Venue:** Parallel Processing and Applied Mathematics (PPAM 2024); extended in Future Generation Computer Systems (2025)
**Year:** 2024/2025
**DOI (journal):** 10.1016/j.future.2025.107813
**URL:** https://www.sciencedirect.com/science/article/pii/S0167739X25001086

Juliana automatically translates Julia CUDA.jl GPU code into abstract multi-backend code using KernelAbstractions.jl, enabling execution on NVIDIA, AMD, Intel, and Apple GPUs from a single translated codebase. Evaluated on Rodinia, miniBUDE, BabelStream, and Oceananigans.jl. Performance overhead under 7% on the Rodinia suite. Demonstrates automated code migration to portable dispatch without manual rewrites.

**Relevance:** Automated porting pipeline from CUDA to multi-backend dispatch — engineering path for migrating existing ML kernels.

---

### D6. Alpaka: An Abstraction Library for Parallel Kernel Acceleration
**Authors:** René Widera, Axel Huebl, et al.
**Venue:** IPDPS Workshops (original 2016); active through 2022+ with PIConGPU OpenACC/OpenMP backend paper (Euro-Par 2022)
**Year:** 2016 (original); 2022 (challenges paper)
**arXiv (original):** https://arxiv.org/abs/1602.08477
**2022 challenges paper:** https://link.springer.com/chapter/10.1007/978-3-030-97759-7_5

Alpaka is a header-only C++17 abstraction library supporting CUDA, HIP, SYCL, OpenMP (target and 2.0+), std::thread, and serial backends from a single kernel source. Used in PIConGPU particle-in-cell code. The 2022 paper documents challenges porting Alpaka-based codes to directive-based offloading (OpenACC/OpenMP target), revealing that immature compiler support creates portability gaps independent of the abstraction layer design.

**Relevance:** Alternative to Kokkos/RAJA; its multi-backend dispatch approach and documented portability gaps are directly comparable.

---

### D7. Implementing Multi-GPU Scientific Computing Miniapps (see also Theme B, B7)
Already catalogued under Theme B as B7.

---

## Theme E: Additional Closely Related Work

Shorter entries on relevant papers that inform the research landscape.

---

### E1. Ansor: Generating High-Performance Tensor Programs for Deep Learning
**Authors:** Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, et al.
**Venue:** OSDI '20
**Year:** 2020
**URL:** https://www.usenix.org/conference/osdi20/presentation/zheng

Auto-scheduler for TVM that generates high-performance tensor programs without manual templates, using hierarchical search space sampling + evolutionary search + learned cost model. Improves GPU performance by up to 1.7x over AutoTVM on NVIDIA. Baseline for comparing dynamic dispatch approaches.

---

### E2. Hidet (see Theme C, C4)

---

### E3. Sylkan: Towards a Vulkan Compute Target Platform for SYCL
**Authors:** Peter Thoman, Daniel Gogl, Thomas Fahringer
**Venue:** IWOCL '21
**Year:** 2021
**DOI:** 10.1145/3456669.3456683
**URL:** https://dl.acm.org/doi/fullHtml/10.1145/3456669.3456683

Prototype SYCL compiler and runtime targeting Vulkan/SPIR-V, expanding SYCL beyond OpenCL to any Vulkan-capable device. Analyzes semantic mismatch between SYCL and Vulkan compute models, proposes workarounds. Opens SYCL dispatch to the broad Vulkan hardware ecosystem including mobile GPUs and embedded platforms.

---

### E4. One Pass to Bind Them: hipSYCL SSCP (see Theme A, A3)

---

### E5. HPVM: Heterogeneous Parallel Virtual Machine
**Authors:** Maria Kotsifakou, Prakalp Srivastava, Matthew D. Sinclair, Rakesh Komuravelli, Vikram Adve, Sarita Adve
**Venue:** PPoPP '18
**Year:** 2018
**DOI:** 10.1145/3178487.3178493
**URL:** https://dl.acm.org/doi/10.1145/3178487.3178493

Foundational paper introducing a hierarchical dataflow graph IR for heterogeneous systems enabling both compiler IR, virtual ISA, and runtime scheduling from the same representation. Supports GPU, CPU, vector ISA, and FPGA targets. Later extended (HPVM-HDC, 2024) to hyperdimensional computing accelerators.

---

## Summary Table

| # | Paper | Venue | Year | Theme |
|---|-------|-------|------|-------|
| A1 | Kokkos 3 | IEEE TPDS | 2022 | Portable Models |
| A2 | Taking GPU Models to Task (ICS) | ICS '25 | 2024/25 | Portable Models |
| A3 | AdaptiveCpp SSCP (IWOCL) | IWOCL '23 | 2023 | Portable Models |
| A4 | Adaptivity in AdaptiveCpp | IWOCL '25 | 2025 | Portable Models |
| A5 | chipStar | IJHPCA | 2026 | Portable Models |
| A6 | GROMACS + SYCL (AMD) | CUG '24 | 2024 | Portable Models |
| A7 | Unified GPU SVD (ICPP) | ICPP '25 | 2025 | Portable Models |
| A8 | Retargeting GPU Workloads (CGO) | CGO '24 | 2024 | Portable Models |
| B1 | HetGPU binary compatibility | arXiv | 2025 | Runtime Dispatch |
| B2 | Universal GPU ISA | arXiv | 2026 | Runtime Dispatch |
| B3 | Runtime Support Heterogeneous | arXiv | 2023 | Runtime Dispatch |
| B4 | GPU Portability Needs Autotuning | arXiv | 2025 | Runtime Dispatch |
| B5 | Auto-tuning to HIP (Euro-PAR) | Euro-PAR '24 | 2024 | Runtime Dispatch |
| B6 | Multi-Versioning SGEMM | arXiv | 2025 | Runtime Dispatch |
| B7 | Multi-GPU Miniapps Frameworks | arXiv | 2025 | Runtime Dispatch |
| B8 | Zorua GPU virtualization | MICRO '16 | 2016 | Runtime Dispatch |
| C1 | Composable MLIR Codegen | arXiv | 2022 | Multi-Target Compilation |
| C2 | MLIR GPU Tensor Cores (CC) | CC '22 | 2022 | Multi-Target Compilation |
| C3 | TinyIREE (IEEE Micro) | IEEE Micro | 2022 | Multi-Target Compilation |
| C4 | Hidet (ASPLOS) | ASPLOS '23 | 2023 | Multi-Target Compilation |
| C5 | PyTorch 2 (ASPLOS) | ASPLOS '24 | 2024 | Multi-Target Compilation |
| C6 | KernelEvolve (Meta) | arXiv | 2025 | Multi-Target Compilation |
| C7 | AMOS (ISCA) | ISCA '22 | 2022 | Multi-Target Compilation |
| D1 | LLNL Portability Eval (SC) | SC '23 | 2023 | Empirical Studies |
| D2 | SERGHEI Shallow Water Scale | SBAC-PAD | 2025 | Empirical Studies |
| D3 | SYCL Shallow Water Journal | J. Supercomputing | 2025 | Empirical Studies |
| D5 | Juliana Julia CUDA Translator | FGCS | 2024/25 | Empirical Studies |
| D6 | Alpaka Library | IPDPS/Euro-Par | 2016–2022 | Empirical Studies |
| E1 | Ansor TVM (OSDI) | OSDI '20 | 2020 | Related |
| E3 | Sylkan Vulkan+SYCL | IWOCL '21 | 2021 | Related |
| E5 | HPVM Virtual Machine | PPoPP '18 | 2018 | Related |

---

## Key Gaps Identified (Poster Contribution Space)

1. **No unified, dynamic runtime dispatcher for ML kernels** that selects between pre-compiled CUDA, HIP, and SPIR-V variants based on runtime hardware detection. HetGPU (B1) attempts binary compatibility but lacks ML-specific dispatch heuristics.

2. **Autotuning is offline** in most work (Ansor, AMOS, TVM). AdaptiveCpp (A4) is closest to runtime-adaptive JIT but does not target PyTorch/CUDA ecosystem kernels directly.

3. **The ML ecosystem dispatch gap**: PyTorch 2 (C5) dispatches to Triton (GPU) or C++ (CPU) but lacks SPIR-V/Vulkan paths for non-CUDA/HIP platforms. IREE (C3) handles this for inference but not training kernels.

4. **Multi-versioning (B6) is applied only to BLAS**: no work applies multi-versioned kernel dispatch to attention, normalization, or other ML-critical kernels across NVIDIA/AMD/Intel.

5. **Performance on emerging hardware** (Intel Xe, Apple Metal, Tenstorrent) is underexplored for ML workloads — Ringoot et al. (A7) covers SVD, but ML training kernels (attention, softmax, GEMM) are not covered.

---

## Research Connections to Poster Themes

**Dynamic dispatch value (reviewer concern #2):**
- A2 (ICS 2025) shows static models don't guarantee portability
- B4 (Ringlein 2025) quantifies 230% improvement from JIT+autotuning
- B5 (Euro-PAR 2024) shows 10x higher AMD tuning impact vs NVIDIA

**Beyond SYCL comparison (reviewer concern #3):**
- SPIR-V: A5 (chipStar), C3 (IREE/TinyIREE), E3 (Sylkan)
- HIP: A8 (Polygeist/CGO), B5 (auto-tuning to HIP)
- Alpaka: D6
- Multi-versioned JIT: B6 (multi-versioning SGEMM)

**PyTorch/TF ecosystem connection (reviewer concern #4):**
- C5 (PyTorch 2 / torch.compile)
- C4 (Hidet — ONNX/PyTorch frontend)
- C6 (KernelEvolve — Meta production)

**IREE SPIR-V acknowledgment (reviewer concern #5):**
- C1 (Composable MLIR codegen — IREE design basis)
- C3 (TinyIREE — production IREE paper)
