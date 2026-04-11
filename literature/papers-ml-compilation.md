# ML Compilation Literature Survey

**Purpose:** Reference collection for LLVM Dublin 2026 poster on vendor-agnostic runtime dispatch for ML kernels across heterogeneous GPU environments.
**Date compiled:** 2026-04-02
**Target:** 15+ papers organized by theme

---

## Theme 1: Foundational ML Compiler Frameworks

### P1 — MLIR: Scaling Compiler Infrastructure for Domain-Specific Computation
- **Authors:** Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko
- **Venue:** CGO 2021 (IEEE/ACM International Symposium on Code Generation and Optimization)
- **Year:** 2021
- **Key Contribution:** Introduces MLIR — a reusable, extensible IR framework that reduces software fragmentation in compiler ecosystems. Dialects are composable IR fragments; the framework connects front-end ML frameworks to heterogeneous hardware targets without rebuilding lowering pipelines per backend. Directly relevant as the foundation for nearly every modern ML compiler.
- **URL:** https://dl.acm.org/doi/10.1109/CGO51591.2021.9370308
- **PDF:** https://rcs.uwaterloo.ca/~ali/cs842-s23/papers/mlir.pdf
- **arXiv preprint:** https://arxiv.org/pdf/2002.11054

### P2 — TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- **Authors:** Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Cody Hao Yu, Eric Liang, Yida Wang, Jingning Liu, Yao Wang, Josh Fromm, Zhiheng Li, Robert Tibshirani, Randal Bryson, Trevor Darrell, Arvind Krishnamurthy, Luis Ceze, Carlos Guestrin, Joseph Gonzalez, Alvin Cheung, Ion Stoica
- **Venue:** OSDI 2018 (USENIX Symposium on Operating Systems Design and Implementation)
- **Year:** 2018
- **Key Contribution:** End-to-end ML compiler covering graph-level IR, operator-level optimization via tensor expressions, and a learning-based cost model for rapid hardware-specific tuning. Introduces decoupled compute/schedule separation (inspired by Halide) applied to deep learning. Supports CPU, GPU, ARM, FPGA backends from a single frontend.
- **URL:** https://arxiv.org/abs/1802.04799
- **PDF:** https://www.usenix.org/system/files/osdi18-chen.pdf

### P3 — The Deep Learning Compiler: A Comprehensive Survey
- **Authors:** Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hua Yang, Zhongyi Liu, Jingyuan Yang, Hanwen Liang, Rui Ran
- **Venue:** IEEE Transactions on Parallel and Distributed Systems
- **Year:** 2021 (arXiv 2020)
- **Key Contribution:** Systematic survey of DL compilers (TVM, XLA, Glow, ONNC, nGraph, Tensor Comprehensions) covering multi-level IR design, frontend/backend optimizations, and performance benchmarks across hardware. Benchmark comparison of TVM, nGraph, Glow, XLA on MobileNetV2 across CPU/GPU. Essential reference for framing the heterogeneous dispatch problem.
- **URL:** https://arxiv.org/abs/2002.03794
- **PDF:** https://arxiv.org/pdf/2002.03794

### P4 — Compiler Technologies in Deep Learning Co-Design: A Survey
- **Authors:** (Intelligent Computing journal)
- **Venue:** Intelligent Computing (Science journal)
- **Year:** 2023
- **Key Contribution:** Surveys compiler technologies for deep learning hardware co-design. Covers MLIR as unified IR for software/hardware, heterogeneous architecture runtimes bridging host programs and kernels, and current compiler design patterns for accelerators (GPU, TPU, FPGA).
- **URL:** https://spj.science.org/doi/10.34133/icomputing.0040

---

## Theme 2: Auto-Scheduling and Auto-Tuning

### P5 — Ansor: Generating High-Performance Tensor Programs for Deep Learning
- **Authors:** Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica
- **Venue:** OSDI 2020 (USENIX Symposium on Operating Systems Design and Implementation)
- **Year:** 2020
- **Key Contribution:** Fully automated auto-scheduler for tensor programs. Generates diverse, high-performance schedules without manual templates using hierarchical search space construction and evolutionary search with a learned cost model. Integrated into Apache TVM as `tvm.auto_scheduler`. Supersedes the original TVM template-based AutoTVM.
- **URL:** https://arxiv.org/abs/2006.06762
- **PDF:** https://arxiv.org/pdf/2006.06762

### P6 — MetaSchedule: Unified Machine Learning-Based Tensor Program Optimization
- **Authors:** Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, Tianqi Chen
- **Venue:** NeurIPS 2022
- **Year:** 2022
- **Key Contribution:** Successor to Ansor. Introduces a domain-specific probabilistic programming abstraction for search space construction; stochastic schedule primitives parameterize transformations, enabling flexible domain-expert customization. Decouples the search strategy from the schedule representation and supports TensorIR (successor to TVM's Tensor Expression). More composable than Ansor and handles complex modern workloads.
- **URL:** https://openreview.net/forum?id=nyCr6-0hinG
- **PDF:** https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf

### P7 — Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines
- **Authors:** Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman Amarasinghe
- **Venue:** PLDI 2013 (ACM SIGPLAN Conference on Programming Language Design and Implementation)
- **Year:** 2013
- **Key Contribution:** Decouples algorithm specification from scheduling (tile sizes, vectorization, parallelism, recomputation). The compute/schedule split is the intellectual precursor to TVM's optimization model and MLIR's transform dialect. Demonstrates that systematic schedule-space search achieves up to 5x speedup over expert-tuned CUDA/C. Foundational for all subsequent tensor program optimization work.
- **URL:** https://dl.acm.org/doi/10.1145/2491956.2462176
- **PDF:** https://people.csail.mit.edu/jrk/halide-pldi13.pdf

### P8 — Learning to Optimize Halide with Tree Search and Random Programs
- **Authors:** Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tian Qi Chen, Matthijs Holleman, Shoaib Kamil, Alvin Cheung, Jonathan Ragan-Kelley
- **Venue:** SIGGRAPH 2019 (ACM Transactions on Graphics)
- **Year:** 2019
- **Key Contribution:** ML-based cost model for Halide schedule search using beam search + random program sampling to train predictors. Eliminates manual tuning. Directly informs MetaSchedule and Ansor cost model design.
- **URL:** https://halide-lang.org/papers/halide_autoscheduler_2019.pdf

---

## Theme 3: Operator Fusion and Graph Optimization

### P9 — Operator Fusion in XLA: Analysis and Evaluation
- **Authors:** Daniel Snider, Ruofan Liang
- **Venue:** arXiv
- **Year:** 2023
- **Key Contribution:** Analyzes XLA's fusion pipeline — the most impactful single optimization in XLA. Covers horizontal/vertical fusion, how XLA avoids intermediate memory writes via GPU register streaming, and interaction with common subexpression elimination (CSE) and dead code elimination (DCE). Quantifies fusion impact on memory bandwidth and kernel launch overhead.
- **URL:** https://arxiv.org/abs/2301.13062
- **PDF:** https://arxiv.org/pdf/2301.13062

### P10 — SpaceFusion: Advanced Deep Learning Operator Fusion via Space-Mapping Graph
- **Authors:** (EuroSys 2025)
- **Venue:** EuroSys 2025 (Twentieth European Conference on Computer Systems)
- **Year:** 2025
- **Key Contribution:** Graph-based approach to operator fusion using space-mapping graphs to identify and exploit fusion opportunities beyond sequential patterns. Addresses cases where standard linear fusion misses cross-subgraph opportunities. Recent state-of-the-art on kernel fusion for inference.
- **URL:** https://dl.acm.org/doi/10.1145/3689031.3696087

### P11 — DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion
- **Authors:** Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, Bin Ren
- **Venue:** PLDI 2021
- **Year:** 2021
- **Key Contribution:** Proposes a mathematical-property-based fusion rule system that enables aggressive, safe operator fusion beyond element-wise patterns. Covers convolution, normalization, and activation as fusible groups. Demonstrated large speedups on mobile hardware.
- **URL:** https://arxiv.org/abs/2108.13342
- **PDF:** https://arxiv.org/pdf/2108.13342

---

## Theme 4: MLIR Dialects for ML (StableHLO, MHLO, TOSA, Linalg)

### P12 — StableHLO: A Portability Layer for ML Compilers (OpenXLA)
- **Authors:** OpenXLA team (Google and collaborators)
- **Venue:** OpenXLA project / technical report
- **Year:** 2023 (announced March 2023 with OpenXLA launch)
- **Key Contribution:** StableHLO is a versioned, serializable MLIR dialect providing a stable contract between ML frameworks (JAX, TensorFlow, PyTorch via torch-mlir) and ML compilers (XLA, IREE). Based on MHLO but adds backward-compatibility guarantees through MLIR bytecode serialization. Addresses the fragmentation problem between frameworks and compiler backends. Supports dynamism, quantization, and sparsity. Migration target from MHLO.
- **URL:** https://openxla.org/stablehlo
- **GitHub:** https://github.com/openxla/stablehlo

### P13 — Compiler Support for Sparse Tensor Computations in MLIR
- **Authors:** Aart J.C. Bik, Penporn Koanantakool, Tatiana Shpeisman, Nicolas Vasilache, Bixia Zheng, Fredrik Kjolstad
- **Venue:** ACM Transactions on Architecture and Code Optimization
- **Year:** 2022
- **Key Contribution:** Extends MLIR's `sparse_tensor` dialect to support general sparse tensor operations via compiler-driven code generation. Demonstrates that the Linalg-on-Tensors abstraction can be extended with sparsity annotations and automatically lowered to efficient sparse code. Relevant for sparse ML model optimization.
- **URL:** https://dl.acm.org/doi/10.1145/3544559

### P14 — The MLIR Transform Dialect: Your Compiler Is More Powerful Than You Think
- **Authors:** Martin Paul Lücke, Oleksandr Zinenko, William S. Moses, Michel Steuwer, Albert Cohen
- **Venue:** CGO 2025 (IEEE/ACM International Symposium on Code Generation and Optimization)
- **Year:** 2025
- **Key Contribution:** Presents the Transform dialect — first-class IR for expressing compiler pass composition and fine-grained optimization control. Enables user-defined optimization strategies without recompiling the compiler. Supports auto-tuning via parameterized transform schedules. Adds ≤2.6% compile-time overhead. Directly relevant to dynamic dispatch and runtime-configurable kernel selection.
- **URL:** https://www.steuwer.info/files/publications/2025/CGO-The-MLIR-Transform-Dialect.pdf
- **arXiv:** https://www.arxiv.org/pdf/2409.03864v2

---

## Theme 5: Heterogeneous Execution and Portability

### P15 — IREE: An MLIR-Based Machine Learning Compiler and Runtime Toolkit
- **Authors:** IREE team (Google)
- **Venue:** GitHub / OpenXLA project documentation
- **Year:** 2022-2023 (active development; integrated into OpenXLA 2023)
- **Key Contribution:** IREE (Intermediate Representation Execution Environment) is an end-to-end MLIR-based compiler and runtime. Its Hardware Abstraction Layer (HAL) provides a single interface across CPU, GPU (via Vulkan/CUDA/Metal), and custom accelerators. Tiled kernels, vectorized loops, and async dispatches are expressed once in IR and mapped to device backends by swapping the codegen target. Scales from mobile edge to datacenter. The execution environment for OpenXLA.
- **URL:** https://github.com/iree-org/iree
- **Design roadmap:** https://iree.dev/developers/design-docs/design-roadmap/

### P16 — PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation
- **Authors:** Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al.
- **Venue:** ASPLOS 2024 (29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems)
- **Year:** 2024
- **Key Contribution:** Describes TorchDynamo (Python bytecode JIT capturing FX graphs without breaking Python dynamism) and TorchInductor (code generator emitting Triton for GPU and C++/OpenMP for CPU). Achieves 2.27x inference and 1.41x training speedup on 180+ real models. Defines the `torch.compile()` interface. Critical for understanding how dynamic PyTorch dispatch connects to static kernel compilation.
- **URL:** https://dl.acm.org/doi/10.1145/3620665.3640366
- **PDF:** https://docs.pytorch.org/assets/pytorch2-2.pdf

### P17 — Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
- **Authors:** Philippe Tillet, H. T. Kung, David Cox
- **Venue:** MAPL 2019 (3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages)
- **Year:** 2019
- **Key Contribution:** Introduces Triton — a tile-centric GPU programming model with an LLVM-based IR. Abstracts away SIMT threading, memory coalescing, shared memory conflicts, and tensor core scheduling. Exposes intra-instance parallelism via block operations. Achieves parity with cuBLAS (>90% peak device performance on some tasks) and outperforms other DSLs by 2-3x. Basis for TorchInductor's GPU code generation.
- **URL:** https://dl.acm.org/doi/10.1145/3315508.3329973
- **PDF:** https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

### P18 — ML-Triton: A Multi-Level Compilation and Language Extension to Triton GPU Programming
- **Authors:** (Intel/academic collaboration)
- **Venue:** arXiv
- **Year:** 2025
- **Key Contribution:** Extends Triton's compilation flow to support workgroup-level, warp-level, and intrinsic-level programming via multi-level MLIR lowering. Adds compiler hints and warp-level programming extensions to the Triton language. Relevant for understanding how Triton is being extended to support more heterogeneous execution targets.
- **URL:** https://arxiv.org/abs/2503.14985
- **PDF:** https://arxiv.org/pdf/2503.14985

### P19 — A Survey on Deep Learning Hardware Accelerators for Heterogeneous HPC Platforms
- **Authors:** (ACM Computing Surveys)
- **Venue:** ACM Computing Surveys
- **Year:** 2025 (arXiv 2023, accepted 2025)
- **Key Contribution:** Broad survey of GPU, TPU, FPGA, ASIC, NPU, RISC-V-based accelerators for deep learning in HPC. Covers PIM (Processor-In-Memory), non-volatile memory, neuromorphic processors. Provides taxonomy of hardware targets that a heterogeneous dispatch system must address.
- **URL:** https://dl.acm.org/doi/full/10.1145/3729215
- **arXiv:** https://arxiv.org/abs/2306.15552

### P20 — Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem
- **Authors:** (Modular / external collaborators)
- **Venue:** SC 2025 Workshops (International Conference for High Performance Computing, Networking, Storage and Analysis)
- **Year:** 2025
- **Key Contribution:** Evaluates Mojo (MLIR-based language from Modular) for HPC science kernels across NVIDIA H100 and AMD MI300A. Tests stencil (memory-bound), BabelStream (memory-bound), miniBUDE (compute-bound), Hartree-Fock (atomic operations). Competitive with CUDA/HIP for memory-bound kernels; gaps on AMD for atomic ops and fast-math compute-bound kernels. Demonstrates single-source vendor-agnostic GPU programming via MLIR.
- **URL:** https://dl.acm.org/doi/10.1145/3731599.3767573
- **arXiv:** https://arxiv.org/abs/2509.21039

---

## Theme 6: Compiler Optimization Strategies (RL, Polyhedral, Learning-Based)

### P21 — Target-Independent XLA Optimization Using Reinforcement Learning
- **Authors:** Milan Ganai et al.
- **Venue:** NeurIPS 2022 ML for Systems Workshop
- **Year:** 2022
- **Key Contribution:** Applies deep RL to find optimal XLA HLO compiler pass ordering. Achieves average 13.3% reduction in operation count on GPT-2 training graphs compared to fixed pass sequences. Demonstrates that pass ordering is a learnable, hardware-independent optimization target.
- **URL:** https://arxiv.org/abs/2308.14364
- **PDF:** https://mlforsystems.org/assets/papers/neurips2022/paper11.pdf

### P22 — LOOPer: A Learned Automatic Code Optimizer for Polyhedral Compilers
- **Authors:** (EPFL / academic)
- **Venue:** arXiv
- **Year:** 2024
- **Key Contribution:** First polyhedral auto-scheduler using a deep learning-based cost model covering a large space of affine transformations. Achieves 1.84x geometric mean speedup over Tiramisu and 1.42x over Pluto on PolyBench benchmarks. Shows that learned optimization generalizes across polyhedral program structures relevant to ML kernels.
- **URL:** https://arxiv.org/abs/2403.11522
- **PDF:** https://arxiv.org/pdf/2403.11522

### P23 — Towards a High-Performance AI Compiler with Upstream MLIR
- **Authors:** (Academic/industry)
- **Venue:** arXiv
- **Year:** 2024
- **Key Contribution:** Proposes a compilation flow using open-source MLIR passes (Linalg-on-Tensors, cache-level optimizations, micro-kernel lowering) to achieve high performance from generic linear algebra abstractions. Bridges TensorFlow/PyTorch frontends to vectorized CPU/GPU backends without proprietary tooling.
- **URL:** https://arxiv.org/abs/2404.15204
- **PDF:** https://arxiv.org/html/2404.15204v1

---

## Theme 7: Cross-Framework Portability (torch-mlir, ONNX-MLIR, PJRT)

### P24 — Torch-MLIR: Bridging PyTorch and MLIR Ecosystems
- **Authors:** LLVM/PyTorch community
- **Venue:** LLVM Developer Meeting 2024 (poster), ongoing project
- **Year:** 2022-2024
- **Key Contribution:** Provides first-class MLIR compiler support for PyTorch via `Torch` and `TorchConversion` MLIR dialects. FxImporter handles `torch.export` models (dynamic shapes, mutations, symbolic shape expressions). ONNX importer converts ONNX to Torch dialect. Multiple lowering paths (Linalg, StableHLO, TOSA) target different hardware stacks. The standard gateway from PyTorch to IREE, XLA, and custom MLIR backends.
- **URL:** https://github.com/llvm/torch-mlir
- **Poster PDF:** https://llvm.org/devmtg/2024-10/slides/poster/Wang-MLIR-and-PyTorch-Poster.pdf

### P25 — PJRT: Simplifying ML Hardware and Framework Integration
- **Authors:** Google Open Source
- **Venue:** Google Open Source Blog
- **Year:** 2023
- **Key Contribution:** PJRT (Pretty much Just a Runtime) defines a hardware-and-framework-agnostic runtime API. Hardware vendors implement PJRT plugins; ML frameworks (JAX, TensorFlow, PyTorch/XLA) call PJRT to dispatch compute. Enables runtime selection of backend (NVIDIA GPU, TPU, CPU) without framework recompilation. Directly relevant to the dynamic dispatch problem.
- **URL:** https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html

---

## Summary Table

| # | Paper | Year | Venue | Relevance to Poster |
|---|-------|------|-------|---------------------|
| P1 | MLIR: Scaling Compiler Infrastructure | 2021 | CGO | Core IR framework |
| P2 | TVM: End-to-End Optimizing Compiler | 2018 | OSDI | TVM baseline |
| P3 | Deep Learning Compiler Survey | 2021 | IEEE TPDS | Survey anchor |
| P4 | Compiler Technologies Survey | 2023 | Intelligent Computing | Heterogeneous context |
| P5 | Ansor Auto-Scheduler | 2020 | OSDI | Auto-tuning baseline |
| P6 | MetaSchedule | 2022 | NeurIPS | Auto-tuning SoTA |
| P7 | Halide (Algorithm/Schedule Split) | 2013 | PLDI | Scheduling foundations |
| P8 | Learning to Optimize Halide | 2019 | SIGGRAPH | Learned scheduling |
| P9 | Operator Fusion in XLA | 2023 | arXiv | Fusion analysis |
| P10 | SpaceFusion | 2025 | EuroSys | Fusion SoTA |
| P11 | DNNFusion | 2021 | PLDI | Fusion systems |
| P12 | StableHLO Portability Layer | 2023 | OpenXLA | Dialect portability |
| P13 | Sparse Tensor MLIR | 2022 | TACO | Linalg/dialect design |
| P14 | MLIR Transform Dialect | 2025 | CGO | Dynamic optimization |
| P15 | IREE Compiler/Runtime | 2022-23 | OpenXLA | HAL dispatch model |
| P16 | PyTorch 2 / TorchInductor | 2024 | ASPLOS | Dynamic→static bridge |
| P17 | Triton Compiler | 2019 | MAPL | GPU kernel generation |
| P18 | ML-Triton Multi-Level | 2025 | arXiv | Multi-target Triton |
| P19 | DL Accelerators Survey | 2025 | ACM CSUR | Hardware taxonomy |
| P20 | Mojo Performance-Portable | 2025 | SC | Vendor-agnostic GPU |
| P21 | XLA RL Optimization | 2022 | NeurIPS ML4Sys | Pass ordering |
| P22 | LOOPer Polyhedral | 2024 | arXiv | Learned polyhedral |
| P23 | High-Perf AI MLIR | 2024 | arXiv | MLIR pipeline |
| P24 | Torch-MLIR | 2022-24 | LLVM DevMtg | PyTorch→MLIR |
| P25 | PJRT Runtime API | 2023 | Google OSS | Runtime dispatch API |

---

## Research Gaps Identified (Relevant to Poster Contribution)

1. **No single system performs runtime (dynamic) kernel dispatch across NVIDIA/AMD/CPU from a unified MLIR IR.** IREE chooses backend at compile time via `--iree-hal-target-backends`; PJRT dispatches at the framework level but doesn't expose IR-level dispatch. The poster's contribution fills this gap.

2. **Comparison of dispatch overhead**: No published benchmark directly compares IREE HAL, TVM PackedFunc runtime, TorchInductor Triton dispatch, and custom multi-versioned kernel selection on identical workloads across vendor GPUs.

3. **Transform Dialect for runtime dispatch**: P14 (MLIR Transform Dialect) opens the door to expressing dispatch strategies as first-class IR, but no work has exploited this for vendor-agnostic runtime selection.

4. **StableHLO as a dispatch-neutral representation**: StableHLO (P12) is designed for portability but currently targets static compile-time lowering. Using it as an intermediate for deferred, runtime-dispatched kernel selection is unexplored.

---

## Key URLs for Further Reading

- MLIR publications list: https://mlir.llvm.org/pubs/
- IREE design roadmap: https://iree.dev/developers/design-docs/design-roadmap/
- OpenXLA architecture: https://openxla.org/xla/architecture
- TVM Apache: https://tvm.apache.org/
- Triton language: https://triton-lang.org/
- torch-mlir GitHub: https://github.com/llvm/torch-mlir
- IREE comparative benchmarks: https://github.com/iree-org/iree-comparative-benchmark
