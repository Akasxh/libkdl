# Performance Portability for GPUs — Literature Survey

Compiled: 2026-04-02
Scope: Metrics, frameworks, comparisons, and applications of GPU performance portability.
Relevance: LLVM Dublin 2026 poster on vendor-agnostic runtime dispatch for ML kernels.

---

## 1. Foundational Metrics Papers

### 1.1 The Pennycook Metric (Original)

**Title:** A Metric for Performance Portability
**Authors:** S. J. Pennycook, J. D. Sewall, V. W. Lee
**Venue:** 7th International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS), SC 2016
**Year:** 2016
**arXiv:** https://arxiv.org/abs/1611.07409
**DOI:** 10.48550/arXiv.1611.07409

**Key Contribution:**
Introduced the first formal, quantitative definition of "performance portability." Prior usage was entirely subjective. The metric is the harmonic mean of efficiency values across a set of platforms H:

    PP(a,p,H) = |H_e| / sum_{i in H_e} 1/e_i(a,p)

where e_i(a,p) is either architectural efficiency (fraction of peak hardware performance) or application efficiency (fraction of best observed performance). The harmonic mean penalizes poor outliers — an application that fails on even one platform scores low.

**Why it matters for this poster:** Gives a rigorous, citable baseline for any claim about "portability" of a dispatch strategy. Any evaluation of SYCL/HIP/CUDA dispatch choices should be framed in these terms.

---

### 1.2 The Pennycook Metric (Revised Implications)

**Title:** Implications of a metric for performance portability
**Authors:** S. J. Pennycook, J. D. Sewall, V. W. Lee
**Venue:** *Future Generation Computer Systems*, Vol. 92, pp. 947–958
**Year:** 2017 (published online 2017)
**DOI:** 10.1016/j.future.2017.08.007
**ScienceDirect:** https://www.sciencedirect.com/science/article/abs/pii/S0167739X17300559

**Key Contribution:**
Identified theoretical flaws in the original 2016 metric definition (specifically around the use of application efficiency), proposed a revised metric, and provided guidelines for correct usage. Demonstrated that naive application of the metric leads to misleading conclusions.

---

### 1.3 Reformulation of the Performance Portability Metric

**Title:** Reformulation of the performance portability metric
**Authors:** A. Marowka
**Venue:** *Software: Practice and Experience*, Wiley
**Year:** 2022
**DOI:** 10.1002/spe.3002

**Key Contribution:**
Further revision and clarification of the Pennycook metric. Addresses edge cases and proposes cleaner formulations for multi-platform comparison.

---

## 2. Broad Survey / Comparison Papers

### 2.1 Evaluative Comparison across GPU Programming Models (Davis et al., 2024)

**Title:** An Evaluative Comparison of Performance Portability across GPU Programming Models
**Authors:** Joshua H. Davis, Pranav Sivaraman, Isaac Minn, Konstantinos Parasyris, Harshitha Menon, Giorgis Georgakoudis, Abhinav Bhatele
**Venue:** arXiv preprint (2402.08950); presented at ICS 2025 (Taking GPU Programming Models to Task for Performance Portability)
**Year:** 2024 (preprint Feb 2024; ICS version 2025)
**arXiv:** https://arxiv.org/abs/2402.08950
**OSTI (LLNL report):** https://www.osti.gov/servlets/purl/2305595

**Models Evaluated:** CUDA, HIP, Kokkos, RAJA, OpenMP, OpenACC, SYCL
**Hardware:** NVIDIA and AMD GPU systems at leadership-class facilities (Frontier, Summit)
**Proxy Applications:** 4 proxy apps spanning different arithmetic intensities

**Key Findings:**
- Kokkos and RAJA "offer the most promise empirically as performance portable programming models"
- Kokkos performance portability metric: 0.82–0.99 across applications
- RAJA: 0.70–1.00
- Both can "be competitive with CUDA and HIP on many system and application pairs"
- RAJA better for low arithmetic intensity (e.g., BabelStream-like kernels)
- Kokkos better for complex, larger kernels (e.g., XSBench)
- SYCL: inconsistent — excelled on older Summit system but unreliable on Frontier
- OpenMP and OpenACC: easier porting, but unable to match Kokkos/RAJA portability
- Data movement overhead frequently exceeded kernel execution time

**Relevance:** The primary comparative reference for any Kokkos/RAJA/SYCL discussion at the poster.

---

### 2.2 Performance Portability across Diverse Computer Architectures (Deakin et al., 2019)

**Title:** Performance Portability across Diverse Computer Architectures
**Authors:** Tom Deakin, Simon McIntosh-Smith, James Price, Andrei Poenaru, Patrick Atkinson, Codrin Popa, Justin Salmon
**Venue:** 2nd International Workshop on Performance, Portability and Productivity in HPC (P3HPC), SC 2019
**Year:** 2019
**DOI:** 10.1109/P3HPC49587.2019.00006
**PDF:** https://research-information.bris.ac.uk/ws/portalfiles/portal/211492749/p3hpc.pdf

**Key Contribution:**
Broadest pre-2024 performance portability study. Five mini-applications × five programming models × six CPUs + five GPUs + one vector architecture. Applied Pennycook metric rigorously. Authors describe it as "the broadest and most rigorous performance portability study to date" at time of publication.

---

### 2.3 On the Performance Portability of OpenACC, OpenMP, Kokkos and RAJA (Marowka, 2022)

**Title:** On the Performance Portability of OpenACC, OpenMP, Kokkos and RAJA
**Authors:** Ami Marowka
**Venue:** HPC Asia 2022 — International Conference on High Performance Computing in Asia-Pacific Region, Virtual, Japan, Jan 12–14, 2022, pp. 103–114
**Year:** 2022
**DOI:** 10.1145/3492805.3492806

**Key Findings:**
324 case studies across diverse application domains, CPU/GPU architectures, and compilers. All four frameworks achieved >80% performance portability with no significant differences across architectures and compilers. Provides statistical backing for framework claims.

---

### 2.4 Implementing Multi-GPU Miniapps Across Portability Frameworks (Villalobos et al., 2025)

**Title:** Implementing Multi-GPU Scientific Computing Miniapps Across Performance Portable Frameworks
**Authors:** Johansell Villalobos, Josef Ruzicka, Silvio Rizzi
**Venue:** arXiv preprint (2511.02655)
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2511.02655

**Frameworks:** Kokkos, RAJA, OCCA, OpenMP
**Hardware:** Polaris supercomputer (Argonne), single node with 4× NVIDIA A100

**Key Findings (N-body, 10k particles):**
- RAJA: 19.02 s (best)
- Kokkos: 26.96 s
- OpenMP: 21.01 s
- OCCA: 139.82 s (with reductions) → 8.96 s (without, pure JIT path) — dramatic JIT compensation effect

**Key Findings (structured grid simulation):**
- OCCA: 6.07 s (best — JIT advantage for smaller kernels)
- RAJA: 8.02 s
- Kokkos: 15.98 s
- OpenMP: 45.77 s (inter-node sync issues)

**Critical insight:** OCCA's JIT compilation can outperform static compilation for certain workloads but its lack of optimized reduction algorithms is a significant gap. Directly relevant to the poster's JIT dispatch angle.

---

## 3. Kokkos — Architecture and Key Papers

### 3.1 Kokkos Original Paper

**Title:** Kokkos: Enabling manycore performance portability through polymorphic memory access patterns
**Authors:** H. Carter Edwards, Christian R. Trott, Daniel Sunderland
**Venue:** *Journal of Parallel and Distributed Computing*, Vol. 74, No. 12, pp. 3202–3216
**Year:** 2014
**DOI:** 10.1016/j.jpdc.2014.07.003

**Key Contribution:**
Foundational Kokkos paper. Introduced the core abstraction: polymorphic memory layouts decoupled from algorithms. A `Kokkos::View` adapts its layout (LayoutLeft vs LayoutRight) per backend at compile time. This is the unique differentiator — other models support execution spaces but only Kokkos supports data layout policies essential for cache efficiency on heterogeneous memory hierarchies.

**Backends at time of paper:** CUDA, OpenMP. Now expanded to: CUDA, HIP, SYCL, HPX, OpenMP, C++ threads.

---

### 3.2 Kokkos 3: Exascale Extensions

**Title:** Kokkos 3: Programming Model Extensions for the Exascale Era
**Authors:** Christian R. Trott, Damien Lebrun-Grandie, Daniel Arndt, et al.
**Venue:** *IEEE Transactions on Parallel and Distributed Systems*, Vol. 33, No. 4, pp. 805–817
**Year:** 2021 (published online; IEEE TPDS October 2021 issue)
**DOI:** 10.1109/TPDS.2021.3097283
**OSTI:** https://www.osti.gov/pages/biblio/1825555

**Key Contribution:**
Documents Kokkos 3 additions: hierarchical parallelism (TeamPolicy), arbitrary-sized atomic operations, task graphs, and container abstractions. Prepares the model for Frontier (AMD MI250X) and Aurora (Intel Ponte Vecchio) exascale systems. Five DOE labs now co-develop the ecosystem (Sandia, Oak Ridge, ANL, LLNL, CEA France).

---

## 4. RAJA — Architecture and Key Papers

### 4.1 RAJA: Portable Performance for Large-Scale Scientific Applications

**Title:** RAJA: Portable Performance for Large-Scale Scientific Applications
**Authors:** D. A. Beckingsale, J. Burmark, R. Hornung, H. Jones, W. Killian, A. J. Kunen, O. Pearce, P. Robinson, B. S. Ryujin, T. R. W. Scogland
**Venue:** 2019 IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC), SC 2019, Denver, CO, pp. 71–81
**Year:** 2019
**DOI:** 10.1109/P3HPC49587.2019.00012
**OSTI:** https://www.osti.gov/servlets/purl/1573949
**PDF:** http://www.cs.millersville.edu/~wkillian/files/RAJA.P3HPC-SC19.pdf

**Key Contribution:**
Primary RAJA design paper. RAJA provides C++ loop-level abstractions: a loop body is expressed as a lambda; the execution policy (OpenMP, CUDA, HIP, SYCL) is a separate template parameter. This separates algorithm from parallelism. Porting is incremental — existing vendor-specific code can coexist with RAJA-ized loops. Most LLNL ASC applications now depend on RAJA.

**Backends supported:** CUDA (NVIDIA), HIP (AMD), SYCL (Intel), OpenMP, sequential
**Key differentiator vs Kokkos:** More gradual adoption path; better suited for codebases with existing single-vendor GPU code. Kokkos requires more upfront restructuring but yields better performance on complex kernels.

---

### 4.2 Enabling RAJA on Intel GPUs with SYCL (2024)

**Title:** Enabling RAJA on Intel GPUs with SYCL
**Authors:** Brian Homerding (ANL) et al.
**Venue:** Proceedings of the International Workshop on OpenCL and SYCL (IWOCL/SYCLcon 2024)
**Year:** 2024
**DOI:** 10.1145/3648115.3648131

**Key Contribution:**
Documents the SYCL backend addition to RAJA targeting Intel Data Center GPUs (Ponte Vecchio / Max series). Demonstrates RAJA now spans all three major GPU vendors: NVIDIA (CUDA), AMD (HIP), Intel (SYCL). Performance evaluation on Aurora (Intel GPU cluster at ANL).

---

## 5. Alpaka — Architecture and Key Papers

### 5.1 Alpaka Original Paper

**Title:** Alpaka — An Abstraction Library for Parallel Kernel Acceleration
**Authors:** Erik Zenker, Benjamin Worpitz, René Widera, Axel Huebl, Guido Juckeland, Andreas Knüpfer, Wolfgang E. Nagel, Michael Bussmann
**Venue:** IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW 2016)
**Year:** 2016
**arXiv:** https://arxiv.org/abs/1602.08477
**DOI:** 10.48550/arXiv.1602.08477

**Key Contribution:**
Header-only C++20 abstraction for GPU/CPU portability. Defines a hierarchical redundant parallelism model that mirrors GPU thread hierarchies abstractly. Single kernel implementation; changing one source line switches the backend. Backends: CUDA, HIP, SYCL, OpenMP 2.0+, std::thread, serial. Supports NVIDIA, AMD, Intel GPUs plus x86/ARM/RISC-V/Power CPUs.

**Key differentiator:** Lower-level abstraction than Kokkos/RAJA — exposes the thread hierarchy directly, giving more control. Adopted by CERN CMS as the performance portability solution for LHC Run 3 detector reconstruction after evaluating OpenMP, alpaka, Kokkos, and SYCL.

---

### 5.2 Performance Portability for CMS Reconstruction with Alpaka (2023)

**Title:** Performance portability for the CMS Reconstruction with Alpaka
**Authors:** Bocci, Czirkos, Pilato, Pantaleo, Hugo, Kortelainen, Redjeb
**Venue:** *Journal of Physics: Conference Series* (CHEP 2023)
**Year:** 2023
**OSTI:** https://www.osti.gov/pages/biblio/1958450
**Fermilab preprint:** https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf

**Key Contribution:**
Real-world case study: CMS at CERN chose alpaka over Kokkos, SYCL, and std::par for GPU-accelerated pixel track reconstruction. Single codebase runs on NVIDIA, AMD, and Intel GPUs, plus CPU. Demonstrates production-grade portability for a mission-critical HEP workload. CMS has since adopted alpaka as the primary portability solution for Run 3 reconstruction code.

---

### 5.3 Application of Portability Solutions to Track Reconstruction Kernels (2024)

**Title:** Application of performance portability solutions for GPUs and many-core CPUs to track reconstruction kernels
**Authors:** Ka Hei Martin Kwok, Matti Kortelainen, Giuseppe Cerati, Alexei Strelchenko, Oliver Gutsche, et al. (29 total)
**Venue:** 26th International Conference on Computing in High Energy & Nuclear Physics (CHEP 2023), published 2024
**arXiv:** https://arxiv.org/abs/2401.14221
**Year:** 2024

**Frameworks Compared:** Kokkos, SYCL, C++17 std::execution::par, Alpaka
**Hardware:** NVIDIA and AMD GPUs, Intel many-core CPUs
**Workload:** Kalman filter track reconstruction mini-apps (p2z, p2r)

**Key Contribution:**
High-energy physics real-workload evaluation. Explores memory layout strategies and their interaction with portability abstractions. Provides performance data across all three GPU vendor platforms using a realistic scientific application.

---

## 6. OCCA — JIT Compilation for Portable GPU Kernels

### 6.1 OCCA: A Unified Approach to Multi-Threading Languages (2014)

**Title:** OCCA: a unified approach to multi-threading languages
**Authors:** D. D. Medina, A. St-Cyr, T. Warburton
**Venue:** SIAM Journal on Scientific Computing (SISC)
**Year:** 2014
**URL:** https://libocca.org/

**Key Contribution:**
Introduced OCCA (Open Concurrent Computing Abstraction). Core insight: GPU kernels are written once in OKL (OCCA Kernel Language), a directive-based C extension. At **runtime**, OCCA JIT-compiles the OKL source to the target backend (CUDA, OpenCL, OpenMP, HIP, Metal, SYCL). Device selection happens at runtime, not compile time — enabling concurrent use of multiple devices and runtime-adaptive dispatch. This is architecturally distinct from Kokkos/RAJA (which are compile-time polymorphic).

**Backends:** CUDA, HIP, SYCL, OpenMP, OpenCL, Metal, C++
**Key differentiator:** Runtime JIT compilation vs compile-time template instantiation. Allows dynamic device selection and kernel recompilation with different parameters without restarting the application.

**Relevance to poster:** OCCA is the primary reference for "JIT-based portable GPU dispatch" — the approach most analogous to what a dynamic dispatch runtime would look like.

---

## 7. SYCL Performance Portability

### 7.1 Evaluating SYCL Portability on Bandwidth-Bound Applications (2023)

**Title:** Evaluating the performance portability of SYCL across CPUs and GPUs on bandwidth-bound applications
**Authors:** Istvan Z. Reguly
**Venue:** Proceedings of the SC '23 Workshops (P3HPC workshop), 2023
**arXiv:** https://arxiv.org/abs/2309.10075
**DOI:** 10.1145/3624062.3624180

**Hardware:** Intel Data Center GPU Max 1100, NVIDIA A100, AMD MI250X; Intel Xeon (Ice Lake), AMD EPYC (Genoa-X), Ampere Altra
**Compilers:** DPC++, hipSYCL/OpenSYCL

**Key Findings:**
- SYCL on GPU "on average slightly outperforms native approaches"
- CPU performance lags behind native implementations
- SYCL "cannot entirely resolve performance portability challenges across all architectures" but "provides a single programming model and ecosystem to target most current HPC architectures productively"

---

### 7.2 Comparing CUDA and SYCL for Protein Database Search (2023)

**Title:** Comparing Performance and Portability between CUDA and SYCL for Protein Database Search on NVIDIA, AMD, and Intel GPUs
**Authors:** Manuel Costanzo, Enzo Rucci, Carlos García Sánchez, Marcelo Naiouf, Manuel Prieto-Matías
**Venue:** 2023 IEEE 35th International Symposium on Computer Architecture and High Performance Computing (SBAC-PAD)
**Year:** 2023
**arXiv:** https://arxiv.org/abs/2309.09609

**Key Findings:**
- CUDA and SYCL achieve similar performance on NVIDIA devices
- SYCL demonstrated "remarkable code portability to other GPU architectures (AMD and Intel)"
- SYCL achieved superior architectural efficiency in 3 of 4 test cases on non-NVIDIA devices

---

### 7.3 SYCL Performance Portability across CPUs, GPUs, and Hybrid Systems (2024/2025)

**Title:** Analyzing the Performance Portability of SYCL across CPUs, GPUs, and Hybrid Systems with SW Sequence Alignment
**Authors:** Manuel Costanzo, Enzo Rucci, Carlos García-Sánchez, Marcelo Naiouf, Manuel Prieto-Matías
**Venue:** *Future Generation Computer Systems* (journal)
**Year:** 2025 (submitted Dec 2024, published 2025)
**arXiv:** https://arxiv.org/abs/2412.08308
**DOI:** 10.1016/j.future.2025.107838

**Key Findings:**
Extended study covering NVIDIA, AMD, Intel GPUs, multiple CPU vendors, and hybrid CPU-GPU systems. SYCL maintained "comparable performance to CUDA on NVIDIA GPUs" with "similar architectural efficiency on AMD and Intel in the majority of cases." Performance limitations in multi-GPU and hybrid configurations attributed to workload distribution, not SYCL itself.

---

### 7.4 HeCBench: Benchmark Suite for SYCL Portability (2023)

**Title:** A Benchmark Suite for Improving Performance Portability of the SYCL Programming Model
**Authors:** Zheming Jin, Jeffrey S. Vetter
**Venue:** 2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), Raleigh, NC
**Year:** 2023
**DOI:** 10.1109/ISPASS57527.2023.10158214
**GitHub:** https://github.com/zjin-lcf/HeCBench

**Key Contribution:**
200+ benchmarks from HPC, ML, finance implemented in OpenMP, SYCL, HIP, CUDA. With recent LLVM versions, SYCL's median slowdown vs CUDA/HIP is <1% on both NVIDIA and AMD. Many OpenMP benchmarks significantly underperform.

---

## 8. Triton — ML Kernel Language for GPU Portability

### 8.1 Triton: Intermediate Language and Compiler for Tiled Neural Network Computations

**Title:** Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
**Authors:** Philippe Tillet, H. T. Kung, David Cox
**Venue:** Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL '19), co-located with PLDI 2019, Phoenix, AZ
**Year:** 2019
**DOI:** 10.1145/3315508.3329973
**PDF:** https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

**Key Contribution:**
Triton abstracts GPU programming at the *tile* level rather than the thread level. A C-based language with LLVM-based IR; tile-level operations are lowered to GPU instructions via novel optimization passes (shared memory allocation, vectorization, data reuse). Goal: researchers with no CUDA experience write kernels "on par with what an expert would produce."

**Portability angle:** Triton is hardware-agnostic at the language level. The backend (NVIDIA PTX, AMD AMDGCN, Intel SPIR-V) is selected at compile time by the Triton compiler. Recent versions target NVIDIA Blackwell. AMD backend via ROCm is actively developed. This is compile-time portability via a compiler IR, not runtime dispatch.

---

## 9. Runtime JIT Dispatch and Multi-Versioning

### 9.1 Proteus: Portable Runtime Optimization with JIT Compilation (2025)

**Title:** Proteus: Portable Runtime Optimization of GPU Kernel Execution with Just-in-Time Compilation
**Venue:** 23rd ACM/IEEE International Symposium on Code Generation and Optimization (CGO 2025)
**Year:** 2025
**DOI:** 10.1145/3696443.3708939

**Key Contribution:**
Lightweight, language-agnostic JIT system. Extracts LLVM IR from compiled GPU programs, recompiles at runtime with device-specific optimizations. More portable than language-specific JIT (e.g., CUDA's nvcc). Key finding: JIT recompilation overhead is high — median compilation time for Xgemm on NVIDIA Quadro equals ~900 kernel executions. Addresses the fundamental tension between JIT overhead and runtime optimization benefit.

**Relevance to poster:** Direct precedent for a runtime dispatch system using LLVM IR as the portable intermediate representation.

---

### 9.2 A Few Fit Most: SGEMM Multi-Versioning for Portability (2025)

**Title:** A Few Fit Most: Improving Performance Portability of SGEMM on GPUs using Multi-Versioning
**Authors:** Robert Hochgraf, Sreepathi Pai
**Venue:** arXiv (cs.PL)
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2507.15277

**Key Contribution:**
"Portability tuning" framework. Automatically generates multiple specialized SGEMM kernel variants. At deployment, selects the best-fitting variant for the detected GPU without retuning. On CLBlast benchmark data, reaches within 10% of theoretical maximum performance on unseen devices. Directly addresses the static-kernel portability problem without requiring full JIT recompilation.

**Relevance:** The multi-versioned dispatch model is a key design option for the poster's proposed runtime.

---

## 10. Broader Context: HPC Application Portability

### 10.1 Taking GPU Programming Models to Task for Performance Portability (ICS 2025)

**Title:** Taking GPU Programming Models to Task for Performance Portability
**Authors:** Joshua H. Davis et al. (UMD / LLNL)
**Venue:** ACM International Conference on Supercomputing (ICS 2025)
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2402.08950
**PDF:** https://pssg.cs.umd.edu/assets/papers/2025-06-portability-ics.pdf

**Key Context:** Conference version of the 2024 arXiv preprint (see §2.1). As of November 2024, nine of the top ten TOP500 systems employ co-processors or accelerators, making performance portability a first-order concern for all HPC software.

---

### 10.2 BabelStream: Measuring Attainable Memory Bandwidth

**Title:** Performance Portability across Diverse Computer Architectures (also referenced in BabelStream context)
**Lead Researcher:** Tom Deakin, Simon McIntosh-Smith (University of Bristol)
**Benchmark:** BabelStream — reimplementation of McCalpin STREAM benchmark in all major parallel programming models (CUDA, HIP, SYCL, Kokkos, RAJA, OpenMP, OpenCL, etc.)
**Venue series:** P3HPC workshops at SC (2019, 2020, 2021, 2022)
**URL:** https://hpc.tomdeakin.com/projects/babelstream

**Key Contribution:**
Standard community benchmark for measuring memory bandwidth portability. Includes dot-product kernel. Widely used in all major portability comparison studies as the low arithmetic intensity reference. Results: RAJA often outperforms Kokkos on BabelStream-class workloads; SYCL shows high variance.

---

## 11. Summary Comparison Table

| Framework | Type | Portability Mechanism | Key Backends | Strength | Weakness |
|-----------|------|----------------------|--------------|----------|----------|
| **Kokkos** | C++ library | Compile-time template polymorphism | CUDA, HIP, SYCL, OpenMP, HPX | Best for complex kernels; unique memory layout policies | Requires upfront restructuring; compile-time backend selection |
| **RAJA** | C++ library | Compile-time execution policies | CUDA, HIP, SYCL, OpenMP | Gradual adoption; best for low-intensity kernels | Less control over memory layout vs Kokkos |
| **Alpaka** | C++ header-only | Compile-time backend selection | CUDA, HIP, SYCL, OpenMP, std::thread | Fine-grained thread control; HEP production use (CMS CERN) | More verbose API; smaller community than Kokkos |
| **SYCL** | Standard (ISO) | Compile-time + runtime (via OpenCL model) | NVIDIA (via Codeplay), AMD (ROCm), Intel | Vendor-neutral standard; CPU portability | Inconsistent GPU performance; compiler fragmentation (DPC++, hipSYCL, ComputeCpp) |
| **OCCA** | C++ library + OKL | **Runtime JIT compilation** | CUDA, HIP, SYCL, OpenCL, OpenMP, Metal | Runtime device selection; no recompile needed; concurrent multi-device | JIT overhead; no optimized reductions; smaller community |
| **Triton** | Python DSL + compiler | Compile-time tiled IR lowering | NVIDIA PTX, AMD AMDGCN, Intel SPIR-V | ML-focused; expert-level performance without CUDA expertise | ML workloads only; not general HPC |
| **OpenMP** | Pragma standard | Compile-time offload | NVIDIA, AMD, Intel (via target offload) | Lowest porting cost; zero new API | Worst performance portability; vendor support uneven |
| **OpenACC** | Pragma standard | Compile-time offload | NVIDIA (best), AMD (partial) | Very low porting cost | Poor AMD support; effectively NVIDIA-centric in practice |

---

## 12. Key URLs and Sources

- Kokkos GitHub: https://github.com/kokkos/kokkos
- RAJA GitHub: https://github.com/LLNL/RAJA
- OCCA GitHub: https://github.com/libocca/occa
- Alpaka GitHub: https://github.com/alpaka-group/alpaka
- Triton GitHub: https://github.com/triton-lang/triton
- BabelStream: https://hpc.tomdeakin.com/projects/babelstream
- Performance Portability website: https://performanceportability.org/
- HeCBench: https://github.com/zjin-lcf/HeCBench
- Kokkos citing page: https://kokkos.org/citing-kokkos/
- RAJA documentation: https://raja.readthedocs.io/
- Exascale Computing Project (Kokkos/RAJA): https://www.exascaleproject.org/research-project/kokkos-raja/

---

## 13. Gaps and Open Questions (Relevant to Poster Contribution)

1. **No paper directly addresses runtime-adaptive dispatch across NVIDIA/AMD/CPU with ML-specific kernels.** OCCA comes closest but is not ML-focused. This is the gap the poster fills.

2. **Multi-versioning (§9.2) + runtime dispatch (§9.1) have not been combined** for ML workloads. Hochgraf/Pai do SGEMM; Proteus does general IR JIT. Neither targets ML kernel libraries.

3. **SYCL's inconsistency on AMD (Frontier)** noted in Davis et al. (§2.1) suggests that a dispatch system should not rely on a single portability layer — it should maintain per-vendor optimized paths and select at runtime.

4. **PyTorch/JAX integration** is absent from all reviewed papers. The CMS/CERN work (§5.2, §5.3) is the closest real-world adoption case but is HEP-specific.

5. **JIT overhead for ML kernels** (Proteus, §9.1) remains a known cost. The poster should address amortization strategies (caching compiled artifacts, AOT + JIT hybrid).
