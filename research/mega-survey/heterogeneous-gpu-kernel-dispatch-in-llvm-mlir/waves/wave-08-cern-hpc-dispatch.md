# Wave 08 — HPC Production Heterogeneous Dispatch: CERN, Frontier, Aurora, EuroHPC

**Angle:** Real-world production deployment of heterogeneous GPU dispatch at large HPC facilities
**Query:** CERN CMS/ATLAS, Oak Ridge Frontier (AMD), ALCF Aurora (Intel), EuroHPC (LUMI/JUPITER), mixed-vendor clusters
**Date:** 2026-04-06
**Priority source types:** paper, facility docs, CHEP proceedings, HPC user guides

---

## Source Index

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | Bocci et al. "Experience with the alpaka performance portability library in the CMS software" (CHEP 2025, Fermilab-Conf-25-0145) | https://lss.fnal.gov/archive/2025/conf/fermilab-conf-25-0145-cms-csaid.pdf | 2025 | Paper | 10/10 |
| S2 | "Running GPU-enabled CMSSW workflows through the production system" (CHEP 2024) | https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11021.pdf | 2024 | Paper | 10/10 |
| S3 | "Using the ATLAS experiment software on heterogeneous resources" (CHEP 2025) | https://www.epj-conferences.org/articles/epjconf/pdf/2025/22/epjconf_chep2025_01104.pdf | 2025 | Paper | 9/10 |
| S4 | Williams et al. "Aurora: Architecting Argonne's First Exascale Supercomputer for Accelerated Scientific Discovery" | https://arxiv.org/abs/2509.08207 | 2025 | Paper | 8/10 |
| S5 | "Early Application Experiences on Aurora at ALCF: Moving From Petascale to Exascale Systems" (Cray User Group 2025) | https://dl.acm.org/doi/10.1145/3725789.3725791 | 2025 | Paper | 8/10 |
| S6 | Frontier User Guide — OLCF Documentation | https://docs.olcf.ornl.gov/systems/frontier_user_guide.html | Current | Docs | 7/10 |
| S7 | LUMI GPU Programming Models Evaluation (Springer 2022) | https://link.springer.com/chapter/10.1007/978-3-031-10419-0_6 | 2022 | Paper | 7/10 |
| S8 | "Packaging HEP Heterogeneous Mini-apps for Portable Benchmarking and Facility Evaluation on Modern HPCs" arXiv:2505.08933 (CHEP 2025) | https://arxiv.org/abs/2505.08933 | 2025 | Paper | 9/10 |
| S9 | EuroHPC JU Supercomputers Overview | https://www.eurohpc-ju.europa.eu/supercomputers/our-supercomputers_en | 2025 | Docs | 7/10 |
| S10 | "HetGPU: The pursuit of making binary compatibility towards GPUs" arXiv:2506.15993 | https://arxiv.org/abs/2506.15993 | 2025 | Paper | 8/10 |
| S11 | NVIDIA acquires SchedMD / Slurm — December 2025 | https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/ | 2025 | News | 6/10 |
| S12 | "The Future of Scheduling in Athena on HPCs" (CERN CDS, CHEP 2025) | https://cds.cern.ch/record/2924094/files/ATL-SOFT-PROC-2025-030.pdf | 2025 | Paper | 8/10 |
| S13 | ALCF Aurora Learning Paths: Performance, Portability, Productivity | https://www.alcf.anl.gov/aurora-learning-paths-performance-portability-and-productivity | 2025 | Docs | 7/10 |
| S14 | "Break GPU Vendor Lock-In: Distributed MLOps across mixed AMD and NVIDIA Clusters" | https://medium.com/weles-ai/break-gpu-vendor-lock-in-distributed-mlops-across-mixed-amd-and-nvidia-clusters-9cf5e1af767f | 2024 | Blog | 5/10 |

---

## Source Summaries

### S1 — Bocci et al. CHEP 2025 (CMS ALPAKA Production Status) [10/10]

The definitive production status paper for CMS heterogeneous dispatch as of 2025. Key numbers for the poster:

- **2024 HLT farm:** >30,000 CPU cores, ~450 GPUs (NVIDIA L4)
- **GPU coverage:** ~40% of HLT runtime offloaded to GPU across 5 algorithms
- **Throughput:** ~100 kHz proton-proton collision events processed in real-time
- **Deployment model:** Separate CMSSW builds per GPU vendor — CUDA-enabled build for NVIDIA nodes, serial/CPU build for CPU-only nodes
- **The unresolved grid problem:** CMS grid nodes at Tier-1 and Tier-2 sites have heterogeneous GPU fleets. The production system maps GPU-enabled workflows to NVIDIA nodes; AMD GPU nodes are not yet served by the same CMSSW builds.

**Critical production pain point (libkdl motivation):**
> "To develop and maintain a single code base; to use different toolchains to build the code for each supported back-end, and link them into a single application."

This is the CMS description of their current solution — **not** of the problem they would like to solve. They still maintain separate build pipelines and grid-site-specific CMSSW releases. A researcher submitting a CMS reconstruction job must match their CMSSW build to the GPU vendor at the target site. This is operationally expensive at the scale of ~170 WLCG Tier-2 sites.

---

### S2 — "Running GPU-enabled CMSSW workflows through the production system" (CHEP 2024) [10/10]

The infrastructure paper describing how CMS maps GPU jobs to grid resources. Key findings:

- **Grid adaptation required:** Both the CMS production system (WMAgent) and CMSSW reconstruction code had to be adapted to efficiently use heterogeneous platforms at Tier-1 and Tier-2 sites.
- **Site-by-site GPU discovery:** The production system queries each compute site for GPU availability and type, then routes GPU-enabled CMSSW builds accordingly.
- **Separate binaries remain:** The ALPAKA migration reduces code duplication at the source level but does **not** eliminate the per-vendor build requirement. Tier-2 sites with NVIDIA GPUs receive the CUDA-enabled CMSSW build; CPU-only sites receive the serial build. There is no single universal binary.
- **"Heterogeneous Tier-2" problem:** When a Tier-2 site has both NVIDIA and AMD GPUs in the same cluster, CMS cannot serve both with a single CMSSW build. They must either maintain two builds or leave one GPU type idle.

**Scale of the problem:** The WLCG has ~170 grid sites across 42 countries. Maintaining GPU-enabled CMSSW builds per vendor per release cycle is a significant operational burden as AMD and Intel GPUs appear in procurement cycles.

---

### S3 — "Using the ATLAS experiment software on heterogeneous resources" (CHEP 2025) [9/10]

ATLAS's status paper on heterogeneous computing as of 2025. Key findings directly relevant to libkdl:

- **Athena on ARM + GPU:** ATLAS successfully runs Athena on ARM64 (via PanDA) and on NVIDIA GPUs (CUDA 12.4.1, gcc 13.1) at various WLCG sites.
- **NVIDIA-only GPU production:** As of CHEP 2025, ATLAS GPU support in the production Athena framework is **NVIDIA-only via CUDA**. AMD and Intel GPU paths are in development but not in production.
- **Scheduler blindness problem:** In Athena's event processing model, algorithms schedule GPU work themselves — the framework-level scheduler is unaware of GPU dispatch. This means the CPU core blocks while GPU work runs. The scheduler has no visibility into GPU state or multi-device topology.
- **The HL-LHC pressure:** With the High-Luminosity LHC expected from 2030, ATLAS data volumes will require GPU offloading for >50% of reconstruction. The current NVIDIA-only CPU-blocking model is explicitly described as insufficient.
- **PanDA workflow manager:** Jobs are submitted with `+RequireGPUs = 1` annotations; the PanDA system at WLCG sites matches jobs to GPU-equipped nodes. No vendor-neutral runtime dispatch — the match is at job-submission time, not kernel-dispatch time.

**FastCaloSim portability status:** The ATLAS calorimeter simulation (FastCaloSim) has been ported to CUDA, HIP, Kokkos, ALPAKA, SYCL, OpenMP, and std::par. It runs on NVIDIA, AMD, and Intel GPUs. This is the **most portable ATLAS algorithm** and represents the leading edge of what ATLAS can do — multi-vendor portability achieved through framework compilation, not runtime dispatch.

---

### S4 — Williams et al. arXiv:2509.08207 "Aurora: Architecting Argonne's First Exascale Supercomputer" [8/10]

The architectural paper for ALCF Aurora (Intel Data Center GPU Max 1550 / Ponte Vecchio).

**Hardware facts:**
- 10,624 compute nodes
- Each node: 2× Intel Xeon Max 9470 CPUs (Sapphire Rapids with HBM) + 6× Intel Data Center GPU Max 1550 (Ponte Vecchio)
- Total: 63,744 Intel GPUs — the world's largest Intel GPU deployment
- Peak: ~2 exaFLOPS (FP64 mixed precision)
- Memory: Each PVC has 128 GB HBM2e across 4 stacks

**The portability challenge — confirmed at exascale scale:**

> "All of the ESP, ECP, INCITE, and ALCC projects targeted performance portability across the pre-exascale and future post-exascale supercomputing systems including the major vendors' CPUs and GPUs (Intel, AMD, NVIDIA, and Arm), with performance portability requiring using and/or developing a portability layer to program the GPUs."

This is a direct statement from the official Aurora architecture paper: **every major application project at Aurora required writing or adopting a portability layer** to target Intel PVC alongside their existing NVIDIA and AMD code paths. The portability layer is not optional — it is the required abstraction for any application targeting multiple facilities.

**ALCF tracked >1100 bugs and issues** across 40+ applications during Aurora bring-up. Many were compiler bugs in Intel's oneAPI/DPC++ toolchain. These bugs required per-vendor workarounds — additional evidence that a single binary cannot currently be compiled once and run everywhere.

**Programming model matrix for Aurora:**
- Supported: MPI, Intel oneAPI, OpenMP (target offload), SYCL/DPC++, Kokkos, RAJA, HIP (via CHIP-SPV/chipStar), OpenCL
- Not supported: CUDA (NVIDIA proprietary, no Intel implementation)
- Migration path: SYCLomatic (formerly DPCT) for CUDA → SYCL translation

**The dispatch model:**
Aurora uses Intel's Level Zero API as the low-level GPU dispatch layer. oneAPI and SYCL compile to SPIR-V; Level Zero JIT-compiles SPIR-V to PVC native ISA at first launch. This is a JIT-heavy model — fundamentally different from CUDA's AOT+fat-binary model and ROCm's AOT+code-object model.

**Pain point for libkdl motivation:** An application optimized for Aurora (SYCL/Level Zero) cannot be run unchanged on Frontier (HIP/ROCm) or an NVIDIA cluster (CUDA). Scientists must maintain three codebases or use a portability layer that adds complexity and potential overhead. The per-facility binary problem is not solved; it is managed.

---

### S5 — "Early Application Experiences on Aurora at ALCF" (Cray User Group 2025) [8/10]

Practical application porting experiences at Aurora, from the first year of production (January–December 2025).

**Key findings relevant to dispatch:**

1. **SYCL host_task performance bug:** A confirmed SYCL issue requiring manual synchronization workarounds for Intel GPU Level Zero plugin. Applications that relied on standard SYCL event semantics failed silently or experienced deadlocks. This is exactly the kind of runtime-level fragility that a dispatch layer should abstract away.

2. **Strong scaling overhead:** GPU offload and communication overhead creates inefficiencies that require "significant performance optimization effort" per application. The portability layer handles the API, not the optimization — matching the ALPAKA finding that default heuristics deliver ~40% suboptimal performance.

3. **OpenMP as the portability bridge:** Several applications adopted portable OpenMP target-offload code to achieve vendor-agnostic GPU porting, avoiding SYCL's Intel-specific toolchain complexity. This confirms that no single portability solution dominates even within a single facility.

4. **15 of 21 ECP applications partnered with ALCF** for Aurora porting — indicating the scale of manual engineering required. Each of these 15 applications represents an independent porting effort, typically resulting in a separate code path per target GPU vendor.

---

### S6 — Frontier User Guide (OLCF, current) [7/10]

Frontier (Oak Ridge Leadership Computing Facility) hardware facts:

- **9,408 compute nodes**
- Each node: 1× AMD EPYC "Trento" 7A53 CPU (64 cores) + 4× AMD MI250X GPU (each with 2 GCDs → 8 effective GPUs per node)
- Total: 37,632 MI250X modules (~75,264 GCDs) — all AMD
- Peak: 1.102 exaFLOPS Rmax (HPL)
- Memory per node: 512 GB DDR4 + 8× 64 GB HBM2E

**Key architectural insight for libkdl:**

Frontier is **AMD-only**. Every GPU in the facility is an MI250X. There is no mixed-vendor GPU dispatch problem within Frontier itself — but there is a severe **ecosystem isolation** problem for scientists who also use NVIDIA facilities (Summit was NVIDIA V100; Perlmutter at NERSC is NVIDIA A100; most cloud computing is NVIDIA).

**The cross-facility dispatch problem:**

The DOE exascale portfolio splits across AMD (Frontier, LUMI), Intel (Aurora), and NVIDIA (Perlmutter at NERSC, future Crossroads). A scientist running the same workload across all three must maintain three GPU code paths:
1. HIP/ROCm for Frontier and LUMI
2. SYCL/oneAPI or CHIP-SPV for Aurora
3. CUDA for Perlmutter and NVIDIA cloud resources

No existing tool solves all three cases with a single binary. This is the cross-facility equivalent of the per-node dispatch problem libkdl addresses.

**Dispatch model on Frontier:**

Frontier uses Slurm for workload management. Applications dispatch kernels via the ROCm runtime (HIP API). The HSA (Heterogeneous System Architecture) runtime underlies the MI250X stack. There is no runtime dispatch layer above HSA — application code calls `hipLaunchKernelGGL` or AMDGPUs-specific APIs directly. No policy layer exists for fallback or multi-variant selection.

---

### S7 — LUMI GPU Programming Models Evaluation (Springer 2022) [7/10]

**Citation:** Evaluating GPU Programming Models for the LUMI Supercomputer. Springer 2022. https://link.springer.com/chapter/10.1007/978-3-031-10419-0_6

LUMI (Finland, EuroHPC) is the largest AMD GPU cluster in Europe:
- AMD EPYC "Trento" CPUs + AMD Instinct MI250X GPUs (same generation as Frontier)
- 10,160 compute nodes in the GPU partition (LUMI-G)
- ~80,000 AMD GCDs total
- ROCm software stack exclusively

**Programming model evaluation for LUMI found:**
- HIP is the primary path for CUDA-porting (hipify achieves 80-95% automatic conversion but remaining code requires manual work)
- OpenMP target offload is viable but shows performance variability across AMD ROCm versions
- SYCL is supported via Intel DPC++ targeting AMD via LLVM HIP backend — but performance is inconsistent
- Kokkos HIP backend: near-native performance, recommended for new portability-layer-based development

**The LUMI porting bottleneck:**
LUMI explicitly requires developers to either: (a) rewrite CUDA code in HIP, (b) use portable programming models (Kokkos, OpenMP), or (c) rely on hipify with manual patches. Option (c) produces a HIP binary — not interoperable with NVIDIA facilities. Options (a) and (b) require either separate HIP and CUDA codebases or a portability framework build.

**Porting documentation evidence of the pain point:**
LUMI's official documentation includes a guide "Preparing codes for LUMI: converting CUDA applications to HIP" — the existence of this guide as a primary onboarding document quantifies the scope of the problem: **every CUDA application at LUMI required a separate porting effort**.

---

### S8 — arXiv:2505.08933 "Packaging HEP Heterogeneous Mini-apps for Portable Benchmarking" (CHEP 2025) [9/10]

**Citation:** Forti, A. et al. (BNL, LBNL, FNAL). "Packaging HEP Heterogeneous Mini-apps for Portable Benchmarking and Facility Evaluation on Modern HPCs." arXiv:2505.08933, CHEP 2025.

This paper directly quantifies the facility-evaluation dispatch problem. The HEP-CCE (High Energy Physics Center for Computational Excellence) packaged four portable HEP mini-apps for cross-facility evaluation:

1. **Patatrack** (CMS pixel tracking): CUDA, HIP, ALPAKA, Kokkos, SYCL backends
2. **FastCaloSim** (ATLAS calorimeter sim): CUDA, HIP, ALPAKA, Kokkos, SYCL, OpenMP, std::par
3. **p2r** (Kalman filter tracking for ATLAS/DUNE): Multiple portability frameworks
4. **WireCell Toolkit** (DUNE neutrino reconstruction): Multiple backends

**The deployment bottleneck the paper explicitly identifies:**

> "While these mini-apps had been developed, they still required a significant amount of manual intervention to deploy on a new facility."

The solution required: containerization (Singularity/Apptainer), automated Spack builds, facility-specific environment configurations. Even with these tools, each **facility deployment is a separate engineering task** — because the binary for Frontier (HIP) is not the binary for Aurora (SYCL) is not the binary for Perlmutter (CUDA).

**Facilities evaluated in the paper:** Frontier (AMD MI250X), Aurora (Intel PVC), Perlmutter (NVIDIA A100), and Polaris (NVIDIA A100). Each required a separate build of each mini-app.

**The deployment matrix this creates:**
- 4 mini-apps × 4 facilities × ~5 backends = ~80 build configurations maintained simultaneously
- This is the combinatorial explosion that libkdl's multi-variant bundle format is designed to replace

**Direct relevance to libkdl:** If each mini-app were packaged as a libkdl Multi-Target Bundle (MTB), a single `patatrack.kdl` file would contain pre-compiled variants for each GPU target. Deployment to any facility reduces to: copy the `.kdl` file, run. No facility-specific build, no Spack environment, no containerization required for the dispatch layer itself.

---

### S9 — EuroHPC JU Supercomputers Overview [7/10]

The EuroHPC fleet as of 2026 spans three major GPU vendors, making it the world's most diverse large-scale GPU deployment:

| System | Location | GPU Vendor | GPUs | Status |
|--------|----------|------------|------|--------|
| LUMI | Finland | AMD MI250X | ~80K GCDs | Production |
| MareNostrum 5 GPU partition | Spain | NVIDIA H100 | ~1,120 | Production |
| Leonardo | Italy | NVIDIA A100 | ~3,456 | Production |
| JUPITER | Germany | NVIDIA GH200 (Booster) + ARM CPU | ~24K | Production (exascale) |
| DAEDALUS | Greece | NVIDIA GH200 | TBD | Planned 2026 |
| Discoverer | Bulgaria | AMD MI250X | ~2,560 | Production |
| Meluxina | Luxembourg | NVIDIA A100 | ~400 | Production |
| Vega | Slovenia | NVIDIA A100 | ~240 | Production |

**The heterogeneity problem at EuroHPC scale:**

The EuroHPC document from the Segler Consulting analysis states explicitly:
> "The heterogeneity of the EuroHPC fleet introduces a significant hidden cost in the form of software complexity and optimization overhead, which undermines the vision of a seamless, unified ecosystem. An AI developer building a model on LUMI must use the AMD ROCm software stack, while to run or scale that same model on Leonardo or MareNostrum 5, they would need to port their code to NVIDIA's CUDA platform — a non-trivial engineering task."

**The EFP response (too late, too high-level):**
The EuroHPC Federation Platform (EFP) project started January 2025, with a Minimum Viable Federation Platform planned for Q1/2026. This is a data/workflow federation platform — it does not address the kernel dispatch binary problem. Users will still need separate GPU binaries for LUMI vs. Leonardo even with the EFP layer in place.

**Key statistic:** Of 11 EuroHPC petascale/exascale systems, at least 4 use AMD GPUs and at least 6 use NVIDIA GPUs. A scientist with compute allocations across EuroHPC must maintain separate AMD and NVIDIA builds, or use a portability framework that adds overhead and complexity.

---

### S10 — HetGPU arXiv:2506.15993 (2025) [8/10]

**Citation:** "HetGPU: The pursuit of making binary compatibility towards GPUs." arXiv:2506.15993 (June 2025). https://arxiv.org/abs/2506.15993

A parallel research effort directly targeting the same problem as libkdl, published June 2025. Key architectural differences from libkdl:

**HetGPU's approach:**
- Defines an **architecture-agnostic GPU IR** — a new intermediate representation for GPU execution state
- A **JIT compiler layer** dynamically translates this IR to each target GPU's native ISA at first launch
- Handles SIMT (NVIDIA/AMD warps) vs. MIMD (Tenstorrent RISC-V cores) execution models
- Supports live migration: serializes GPU execution state mid-run and deserializes on a different GPU vendor

**Results:** "Unmodified GPU binaries compiled with hetGPU can be migrated across disparate GPUs with minimal overhead." Supports NVIDIA, AMD, Intel, and Tenstorrent hardware.

**Critical difference from libkdl:**
- HetGPU is a JIT-based universal translator — it compiles once to the hetGPU IR, then JIT-translates to each target. Performance is bounded by JIT quality; no per-device AOT optimization.
- libkdl is a selection-based multi-variant dispatcher — it carries pre-compiled AOT variants per device, selected at load time. Performance matches native because each variant was compiled natively.
- HetGPU solves "run anywhere" at the cost of per-device peak performance.
- libkdl solves "run anywhere at peak performance" at the cost of binary size (multiple variants stored).

**Complementarity:** HetGPU and libkdl solve different points on the portability/performance tradeoff. HetGPU is a universal fallback; libkdl is a performance-preserving dispatcher. A complete solution might combine both: libkdl selects among native variants when available, falls back to HetGPU translation for unrecognized targets.

**Threat assessment for poster:** HetGPU's June 2025 publication means it will be known to the LLVM community by Dublin 2026. The poster must acknowledge it and clearly differentiate: HetGPU achieves portability through JIT translation; libkdl achieves portability through AOT variant selection. Different performance guarantees, different use cases.

---

### S11 — NVIDIA Acquires SchedMD/Slurm (December 2025) [6/10]

**Event:** NVIDIA acquired SchedMD (maker of Slurm) on December 15, 2025.

**Scale context:** Slurm manages workloads on >50% of the top 100 supercomputers globally. This includes Frontier (AMD), Perlmutter (NVIDIA), and many EuroHPC systems.

**Relevance to vendor-neutral dispatch:**
The acquisition has reignited concerns about vendor lock-in at the workload scheduler level. If the scheduler that places jobs on GPU-equipped nodes is controlled by the GPU vendor with the largest existing install base, there is structural pressure against neutral multi-vendor kernel dispatch.

**HPC community response:** nextplatform.com commentary: "Nvidia nearly completes its control freakery." Multiple HPC community voices expressed concern that open-source commitments may not prevent feature prioritization for NVIDIA hardware.

**Indirect relevance to libkdl:** A vendor-neutral kernel dispatch library (libkdl) that runs equally well on AMD Frontier and NVIDIA Perlmutter nodes becomes more strategically important, not less, when the workload scheduler becomes vendor-aligned. libkdl operates below the scheduler level — once a job is placed on a GPU node, libkdl handles which GPU variant to execute, independent of which vendor's scheduler allocated the node.

---

### S12 — "The Future of Scheduling in Athena on HPCs" (ATLAS CHEP 2025) [8/10]

**Citation:** "The Future of Scheduling in Athena on HPCs." ATL-SOFT-PROC-2025-030. CDS: https://cds.cern.ch/record/2924094/files/ATL-SOFT-PROC-2025-030.pdf

This paper directly addresses the ATLAS scheduler's inability to be aware of GPU dispatch, and proposes architectural changes to the Athena framework for multi-device, heterogeneous scheduling.

**Current problem (explicit statement):**
> "The algorithms themselves are given the task of scheduling work onto GPUs, though the Athena scheduler is unaware of this, leaving the CPU core blocked while GPU work runs."

**Proposed future architecture:**
- Introduce GPU-awareness into the Athena data-flow scheduler
- Allow the scheduler to overlap CPU and GPU work within a single event
- Support for HPC-style asynchronous multi-stream dispatch

**Why this matters for libkdl:**
The ATLAS scheduler paper identifies the same abstraction gap libkdl addresses at a lower level. ATLAS needs a framework-level scheduler that understands GPU dispatch; libkdl provides the kernel-level dispatch mechanism that such a scheduler would call. The paper implicitly calls for exactly the kind of runtime dispatch policy layer that libkdl implements — a layer that knows which GPU variant to run on which device, without the application code needing to hardcode that decision.

**Timeline:** The new scheduling architecture is targeted for prototype testing before HL-LHC operations (2029-2030 timeframe). This is the exact deployment window where libkdl's runtime dispatch model is relevant.

---

### S13 — ALCF Aurora Learning Paths: Performance, Portability, Productivity [7/10]

ALCF's official documentation for Aurora application development reveals the dispatch model in use:

**Programming model hierarchy:**
1. **SYCL/DPC++** (recommended, native to Aurora)
2. **OpenMP target offload** (portable, for existing OpenMP HPC codes)
3. **Kokkos** with HIP or SYCL backend (for Kokkos-based codes)
4. **CUDA migration via SYCLomatic** (for CUDA-first codebases)

**The dispatch chain for a CUDA application reaching Aurora:**
```
CUDA source → SYCLomatic → SYCL source → DPC++ compiler → SPIR-V IR → Level Zero JIT → PVC native ISA
```

This four-step translation chain is lossy. SYCLomatic handles ~95% of CUDA automatically; the remaining 5% requires manual porting. The SPIR-V → PVC JIT happens at runtime, adding first-launch latency. ALCF documents multiple cases where the JIT output is suboptimal compared to hand-tuned PVC code.

**Key finding:** ALCF explicitly publishes "Aurora Learning Paths: Migrating from CUDA to SYCL" as a primary user resource. The existence of this migration guide acknowledges that no automatic, zero-overhead, zero-effort path exists from CUDA to Aurora — every application requires human engineering time.

---

## Synthesis: The Production Heterogeneous Dispatch Problem in Numbers

### Scale of Heterogeneity (2025–2026 Production Systems)

| Facility | GPU Vendor | Count | Dispatch API | Runtime |
|----------|------------|-------|-------------|---------|
| Frontier (ORNL) | AMD MI250X | 37,632 modules | HIP/ROCm | HSA |
| Aurora (ALCF) | Intel PVC | 63,744 | SYCL/Level Zero | Level Zero JIT |
| Perlmutter (NERSC) | NVIDIA A100 | 6,144 | CUDA | CUDA runtime |
| LUMI (Finland, EuroHPC) | AMD MI250X | ~80K GCDs | HIP/ROCm | HSA |
| JUPITER (Germany, EuroHPC) | NVIDIA GH200 | ~24K | CUDA | CUDA runtime |
| Leonardo (Italy, EuroHPC) | NVIDIA A100 | 3,456 | CUDA | CUDA runtime |
| CERN HLT (CMS) | NVIDIA L4 | ~450 | CUDA→ALPAKA | CUDA runtime |
| WLCG Tier-1/2 grid | Mixed (NVIDIA majority, AMD growing) | ~10,000+ | Varies | Varies |

**No two large-scale DOE/EuroHPC facilities use the same GPU vendor.**

### The Build Matrix Problem

A HEP physicist running the same reconstruction algorithm across Frontier, Aurora, and Perlmutter must maintain:
- HIP build for Frontier/LUMI (AMD ROCm)
- SYCL/DPC++ build for Aurora (Intel Level Zero)
- CUDA build for Perlmutter, Leonardo, JUPITER

Or they must adopt a portability framework (ALPAKA, Kokkos, OpenMP) that adds compilation complexity and potentially ~30-40% performance overhead without explicit tuning.

The HEP-CCE paper (S8) quantifies this: 4 mini-apps × 4 facilities × ~5 backends = ~80 build configurations maintained simultaneously for a single benchmarking study.

### The Runtime Fallback Gap

Across all surveyed facilities, **no production system has a runtime fallback dispatch mechanism**:

- Frontier: AMD-only — no fallback needed internally, but no mechanism to serve NVIDIA code
- Aurora: Intel-only — CUDA is not supported; no runtime fallback path
- CERN HLT: NVIDIA GPUs exclusively in 2024 farm — CPU fallback via ALPAKA switch modules requires compile-time configuration
- ATLAS Athena: NVIDIA CUDA only in production — AMD/Intel support is in development but not shipping
- EuroHPC: Workload manager (Slurm) routes jobs to vendor-specific nodes; no single binary runs on all nodes

**The gap libkdl fills:** A kernel bundle (MTB) containing CUDA, HIP, SYCL, and CPU fallback variants, with runtime hardware fingerprinting to select the correct variant, eliminates the entire build matrix and enables a single artifact to be deployed to any of the above facilities.

---

## Key Pain Points — Direct Libkdl Motivation

### Pain Point 1: Per-Vendor Build Matrix at Grid Scale (CMS/ATLAS)

CMS maintains separate CMSSW builds for NVIDIA and CPU-only grid sites. ATLAS is NVIDIA-only in production. The WLCG grid has ~170 sites with evolving GPU procurement — AMD GPUs are entering Tier-2 sites as cost-competitive alternatives to NVIDIA, but the CMS and ATLAS production systems cannot serve them without separate builds.

**Quantification:** The CMS CHEP 2024 paper documents that "parts of the production system infrastructure have been adapted to successfully map, schedule and run available GPU-enabled workflows on different sites." This adaptation is non-trivial and must be repeated for each new GPU vendor type at each new site.

### Pain Point 2: Cross-Facility Portability at DOE Scale

DOE has three exascale systems: Frontier (AMD), Aurora (Intel), and Frontier's planned successor. NERSC's Perlmutter is NVIDIA. A scientist with allocations across multiple facilities — common in DOE programs like INCITE and ALCC — must maintain separate GPU code for each.

The Aurora architecture paper (S4) confirms: "performance portability requiring using and/or developing a portability layer to program the GPUs" — not "here is the portability layer," but "you need to develop one." The current state is every application team builds their own solution.

### Pain Point 3: Intel Aurora SYCL JIT Overhead

Aurora's Level Zero JIT model imposes first-launch latency and produces suboptimal code for performance-critical kernels. ALCF documentation notes applications requiring "significant performance optimization effort" post-porting. A libkdl MTB containing a pre-compiled, hand-tuned PVC variant would eliminate the JIT step entirely and deliver the same performance as the ALCF-optimized build, without the user needing Intel-specific build infrastructure.

### Pain Point 4: GPU Failure and CPU Fallback in Production

The CERN HLT farm documented "switch modules" in CMSSW required for CPU fallback when GPUs are unavailable. The ATLAS scheduler paper (S12) explicitly identifies that the Athena scheduler is "unaware" of GPU dispatch state, blocking CPU cores. These are production problems caused by the absence of a runtime dispatch layer with fallback capability.

### Pain Point 5: NVIDIA Slurm Acquisition and Vendor Lock-In Pressure

NVIDIA's December 2025 acquisition of SchedMD (Slurm) raises structural concerns about the independence of GPU workload scheduling at the system-software level. A vendor-neutral kernel dispatch library that operates independently of the workload scheduler provides an architectural hedge against scheduler-level vendor capture.

---

## Critical Comparison: How HPC Facilities Handle Heterogeneous Dispatch Today

| Facility | Dispatch Model | Fallback Mechanism | Binary Portability | Pain Points |
|----------|---------------|-------------------|-------------------|-------------|
| CERN HLT | Runtime selection among pre-compiled ALPAKA variants | Switch modules (CPU fallback, compile-time configured) | No — separate builds per vendor | Build matrix, AMD GPUs not served |
| ATLAS Athena | Algorithm-driven GPU scheduling (NVIDIA CUDA only) | Manual framework-level switching | No | Scheduler blindness, NVIDIA-only |
| Frontier | HIP/ROCm, AMD-only | No runtime fallback (AMD-only fleet) | No — HIP binary runs only on AMD | Cross-facility isolation from CUDA sites |
| Aurora | SYCL/Level Zero JIT | No runtime fallback (Intel-only fleet) | No — SPIR-V/Level Zero runs only on Intel | JIT overhead, CUDA not supported |
| LUMI | HIP/ROCm, AMD-only | No runtime fallback | No | Same as Frontier |
| EuroHPC grid | Job scheduler routes to vendor-specific nodes | No kernel-level fallback | No | Per-system build required |
| **libkdl (proposed)** | **Runtime hardware fingerprint → AOT variant selection** | **CPU fallback variant in bundle** | **Yes — single MTB file** | **Prototype stage; not at scale** |

---

## Relevance to libkdl: Argument Structure for Poster

### The Argument Chain

1. **Scale of the problem is large and growing:** The three DOE exascale systems use three different GPU vendors (AMD, Intel, NVIDIA). EuroHPC's 11 systems span AMD and NVIDIA. CMS processes 100 kHz events across 450 GPUs and ~170 WLCG sites. This is not a niche portability problem — it is the dominant operational challenge in large-scale scientific computing.

2. **Current solutions require per-vendor builds:** Every surveyed production system — CMS, ATLAS, Frontier, Aurora, LUMI, EuroHPC sites — maintains separate GPU binaries per vendor. No production system has a single binary that runs correctly across AMD, NVIDIA, and Intel GPUs.

3. **Portability frameworks reduce source duplication but not binary fragmentation:** ALPAKA, Kokkos, and OpenMP offload reduce the lines of source code per algorithm, but still require separate compilation per vendor target. The HEP-CCE paper (S8) documents ~80 build configurations for just 4 mini-apps across 4 facilities.

4. **JIT-based solutions (hetGPU, Level Zero on Aurora) trade performance for portability:** Aurora's SYCL/Level Zero JIT model achieves write-once-compile-once, but at the cost of JIT latency and suboptimal code quality vs. AOT. The ALCF tracks 1100+ bugs partly because JIT-compiled code exposes vendor-specific codegen issues at runtime.

5. **libkdl's multi-variant AOT approach fills the gap:** Pre-compile variants per GPU target (NVPTX, AMDGPU, PVC, CPU), package in a single MTB bundle, select at load time via hardware fingerprinting. The result is: (a) single deployable artifact, (b) native AOT performance per device, (c) no JIT overhead, (d) automatic CPU fallback. This is what ALPAKA-at-CMS achieves for one experiment's fixed hardware fleet — libkdl generalizes it to arbitrary deployment targets.

### Poster Quote

> "CMS deploys heterogeneous reconstruction at 100 kHz event throughput across 450 GPUs and 170 WLCG sites — but still maintains separate CMSSW builds per GPU vendor. DOE's three exascale systems use AMD, Intel, and NVIDIA GPUs respectively; a scientist with allocations across all three maintains three GPU codebases. HEP-CCE needed ~80 build configurations to benchmark 4 mini-apps across 4 facilities. libkdl's Multi-Target Bundle format packages all variants in a single artifact; runtime hardware fingerprinting selects the correct one. One binary. Any GPU."

---

## Key References

1. Bocci, A. et al. "Experience with the alpaka performance portability library in the CMS software." EPJ Web Conf. CHEP 2025. Fermilab-Conf-25-0145. https://lss.fnal.gov/archive/2025/conf/fermilab-conf-25-0145-cms-csaid.pdf
2. "Running GPU-enabled CMSSW workflows through the production system." EPJ Web Conf. CHEP 2024. https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11021.pdf
3. "Using the ATLAS experiment software on heterogeneous resources." EPJ Web Conf. CHEP 2025. https://www.epj-conferences.org/articles/epjconf/pdf/2025/22/epjconf_chep2025_01104.pdf
4. Williams, T.J. et al. "Aurora: Architecting Argonne's First Exascale Supercomputer for Accelerated Scientific Discovery." arXiv:2509.08207 (2025). https://arxiv.org/abs/2509.08207
5. "Early Application Experiences on Aurora at ALCF: Moving From Petascale to Exascale Systems." Cray User Group 2025. https://dl.acm.org/doi/10.1145/3725789.3725791
6. Frontier User Guide. OLCF. https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
7. Forti, A. et al. "Packaging HEP Heterogeneous Mini-apps for Portable Benchmarking and Facility Evaluation on Modern HPCs." arXiv:2505.08933, CHEP 2025. https://arxiv.org/abs/2505.08933
8. "HetGPU: The pursuit of making binary compatibility towards GPUs." arXiv:2506.15993 (2025). https://arxiv.org/abs/2506.15993
9. "The Future of Scheduling in Athena on HPCs." ATL-SOFT-PROC-2025-030. CHEP 2025. https://cds.cern.ch/record/2924094/files/ATL-SOFT-PROC-2025-030.pdf
10. EuroHPC JU Supercomputers. https://www.eurohpc-ju.europa.eu/supercomputers/our-supercomputers_en
11. "Preparing codes for LUMI: converting CUDA applications to HIP." LUMI. https://lumi-supercomputer.eu/preparing-codes-for-lumi-converting-cuda-applications-to-hip/
12. NVIDIA acquires SchedMD. NVIDIA Blog, December 2025. https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/
13. ALCF Aurora Learning Paths: Performance, Portability, Productivity. https://www.alcf.anl.gov/aurora-learning-paths-performance-portability-and-productivity
14. "Evaluating GPU Programming Models for the LUMI Supercomputer." Springer 2022. https://link.springer.com/chapter/10.1007/978-3-031-10419-0_6

---

## Novelty of This Wave (1–10): 9

This wave provides the first systematic cross-facility analysis of production heterogeneous GPU dispatch specifically from the angle of **what pain points validate libkdl's use case**. Prior waves (05, 07) covered ALPAKA's CMS deployment and the LLVM community discussion. This wave adds: Aurora's exascale Intel GPU portability challenges (quantified), Frontier's AMD ecosystem isolation, LUMI/EuroHPC's cross-vendor deployment gap, ATLAS's scheduler blindness, the HEP-CCE build matrix quantification, HetGPU as a competing approach, and the NVIDIA Slurm acquisition as a structural risk amplifier.

## Angle Assessment: Relevance to libkdl (1–10): 10

The production HPC dispatch problem is the strongest possible motivation for libkdl. The argument is no longer theoretical — it is operational at the scale of 170 WLCG sites, three exascale systems, and one experiment processing 100 kHz events. The HEP-CCE paper (S8) provides the single most concrete quantification: ~80 build configurations for 4 mini-apps. That is the exact complexity that libkdl's MTB format eliminates.

## Cross-References to Prior Waves

- CMS ALPAKA production: wave-05-alpaka.md, literature/cern-cms-alpaka-production.md
- ALPAKA performance data: literature/alpaka-perf-portability.md
- HetGPU binary compat: literature/hetgpu-binary-compat.md, wave-03-hetgpu-hetir.md
- IREE HAL runtime dispatch: wave-01-iree-hal-runtime-dispatch.md
- IRIS task dispatch: literature/iris-2024-task-dispatch.md
- Liboffload mechanism layer: wave-06-llvm-offload-new-driver.md
- LLVM DevMtg GPU landscape: wave-07-llvm-devmtg-gpu-landscape.md

## Suggested Follow-Up Angles

1. **CHEP 2025 "Running GPU workflows" paper** (full text) — The CHEP 2024 paper (S2) documents the CMS grid adaptation for GPU workflows. The CHEP 2025 equivalent may include AMD GPU sites at Tier-2; fetch and read.
2. **ORNL Frontier application portfolio** — OLCF publishes the list of Frontier production applications (INCITE, ALCC, ASCR). Cross-referencing which also run on NVIDIA (Perlmutter) or Intel (Aurora) facilities quantifies the cross-facility portability problem in terms of specific active scientific workloads.
3. **ATLAS FastCaloSim portability paper** — The ATLAS calorimeter simulation ported to 7+ backends is the most portable HEP algorithm. A detailed analysis of how FastCaloSim manages its backend matrix is a precise case study for libkdl's MTB format.
4. **HetGPU paper full read** — arXiv:2506.15993 is directly competitive with libkdl. Full reading and differentiation analysis needed before Dublin submission.
5. **JUPITER (Jülich) application software stack** — JUPITER is Europe's first exascale system and uses NVIDIA GH200 + ARM CPU. Its software stack choices (CUDA vs. portability frameworks) for production applications will inform the EuroHPC deployment model for 2026–2030.
