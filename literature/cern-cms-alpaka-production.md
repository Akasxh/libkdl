# CMS Run 3 HLT with ALPAKA: Production Heterogeneous Dispatch at CERN

*Research compiled 2026-04-06 for "libkdl: Kernel Dynamic Linker" poster, LLVM Dublin 2026.*

**Relevance to libkdl:** 9/10 — CMS is the highest-profile production deployment of compile-time GPU portability at scale. Their explicit pain point — needing separate binaries per vendor, dual code paths per algorithm, and a complex "switch" scheduling infrastructure — is the exact problem libkdl's runtime dispatch solves. This is the strongest real-world motivation case for the poster.

---

## 1. Overview

The CMS (Compact Muon Solenoid) experiment at CERN's LHC adopted ALPAKA as its official performance portability solution for GPU-accelerated reconstruction in Run 3 (2022–present). This followed a systematic evaluation of all major portability frameworks: OpenMP, Kokkos, SYCL, std::par, and ALPAKA. CMS chose ALPAKA as "the more mature and better performing solution."

The deployment context is demanding: the High-Level Trigger (HLT) must process ~100 kHz of proton-proton collision events in near-real-time, reducing data from ~40 TB/s (raw detector readout) to ~1 GB/s (written to tape). Every millisecond of reconstruction latency matters.

**Primary sources:**
- Bocci et al., "Performance portability for the CMS Reconstruction with Alpaka," CHEP 2023, Fermilab-Conf-23-080 (2023).
- Kortelainen et al., "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code," EPJ Web Conf. CHEP 2024.
- Bocci et al., "Experience with the alpaka performance portability library in the CMS software," EPJ Web Conf. CHEP 2025, Fermilab-Conf-25-0145.
- "Run-3 Commissioning of CMS Online HLT reconstruction using GPUs," EPJ Web Conf. CHEP 2024.

---

## 2. Hardware Heterogeneity in the HLT Farm

### 2024 HLT Farm Configuration

The 2024 HLT farm expansion added 18 new nodes, each with:
- 2× AMD EPYC "Bergamo" 9754 CPUs (192 cores per node)
- 3× NVIDIA L4 GPUs per node

**Total farm capacity (2024):** >30,000 CPU cores and ~450 GPUs.

The GPU fleet is **exclusively NVIDIA** in the current production HLT farm. AMD GPUs are not yet deployed at HLT, but the ALPAKA porting work ensures future readiness for AMD hardware — particularly relevant as AMD becomes more competitive in HPC procurement.

### GPU Workloads in Production (Run 3)

Four detector subsystem algorithms were GPU-offloaded from the start of Run 3:

| Algorithm | Subsystem | Backend | Status |
|-----------|-----------|---------|--------|
| Pixel local reconstruction | Pixel Tracker | CUDA → ALPAKA migration | Production |
| Pixel-only track & vertex reconstruction (Patatrack) | Pixel Tracker | CUDA → ALPAKA migration | Production |
| ECAL local reconstruction | Electromagnetic Calorimeter | CUDA → ALPAKA migration | Production |
| HCAL local reconstruction | Hadronic Calorimeter | CUDA → ALPAKA migration | Production |
| Hadronic PF clustering (HCAL HB+HE) | Particle Flow | ALPAKA-native | Deployed 2024 HLT |

Together these algorithms account for **~40% of the HLT runtime** being offloaded to GPUs.

---

## 3. ALPAKA Migration Timeline in CMSSW

### Phase 1: CUDA-only (Run 3 start, 2022)
- Pixel tracking (Patatrack), ECAL, HCAL ported to native CUDA
- Two separate code paths per algorithm: CPU (legacy) + GPU (CUDA-only)
- Complex HLT "switch" modules to route events to CPU or GPU depending on availability
- Increased developer burden: each algorithm needed two implementations

### Phase 2: ALPAKA Migration (2023–2024)
- Framework support for ALPAKA added to CMSSW
- New Structure-of-Arrays (SoA) data format adopted across algorithms
- Existing CUDA algorithms migrated to ALPAKA equivalents
- Patatrack pixel tracking, ECAL, HCAL ported to single ALPAKA codebase
- First ALPAKA-native algorithm: hadronic PF clustering, deployed for 2024 HLT

### Phase 3: Expansion (Run 4 target, 2027+)
- Goal: GPU utilization for ≥50% of HLT capacity in Run 4
- Goal: ≥80% GPU utilization in Run 5
- More reconstruction algorithms being ported to ALPAKA

---

## 4. Compile-Time vs Runtime Dispatch: The CMS Tradeoff

This is the core tension relevant to libkdl.

### How ALPAKA Handles Dispatch in CMSSW

ALPAKA's single source code is **compiled for each backend separately** and linked into a **single application binary**. The backend selection then happens at **runtime** — CMSSW inspects available hardware and routes events to the appropriate compiled variant. From the CHEP 2025 paper:

> "To develop and maintain a single code base; to use different toolchains to build the code for each supported back-end, and link them into a single application; to seamlessly select the best backend at runtime, and implement portable reconstruction algorithms that run efficiently on CPUs and GPUs from different vendors."

This is important: ALPAKA in CMSSW achieves **runtime device selection**, but via a **compile-multiple-link-one** strategy — not true runtime dispatch of a single kernel to arbitrary hardware. The kernel for each backend is compiled ahead-of-time; the runtime layer selects which compiled variant to invoke.

### The Remaining Problem

Prior to ALPAKA, the CUDA-only deployment required:
- Dual algorithm implementations (CPU + CUDA) for every offloaded algorithm
- Switch modules in the HLT configuration to route events
- Separate validation, testing, and maintenance of both paths

ALPAKA resolves the **code duplication** problem but does NOT solve:
1. The need to compile all desired backends at build time (a new GPU vendor requires recompilation)
2. Dynamic adaptation to unexpected hardware (e.g., a GPU failing mid-run, needing seamless CPU fallback with the same binary)
3. Shipping a single pre-compiled artifact to heterogeneous grid nodes with unknown GPU types

CMS mitigates (3) by maintaining separate CMSSW builds per target platform. Grid nodes with NVIDIA GPUs get the CUDA-enabled build; CPU-only nodes get the serial build.

---

## 5. Patatrack: The Pioneer Project

The **Patatrack** project pioneered heterogeneous reconstruction at CMS and drove ALPAKA adoption. Key history:

- **2018:** CUDA-only GPU pixel tracking prototype (Felice Pantaleo, Andrea Bocci, CERN)
- **2020:** Deployed in HLT production, achieving significant throughput gains
- **2021 GSoC:** ALPAKA backend added to pixeltrack-standalone testbed
- **2022 GSoC:** SYCL backend added via ALPAKA
- **2023:** CMS formally adopts ALPAKA; Patatrack code migrated to ALPAKA

Patatrack standalone benchmark (`github.com/cms-patatrack/pixeltrack-standalone`) remains the community reference for comparing portability frameworks on a realistic HEP reconstruction workload. It has been ported to: CUDA, HIP, SYCL, ALPAKA, Kokkos, std::par, OpenMP offloading, and Julia.

---

## 6. Performance Data

### ALPAKA vs Native CUDA/HIP (CHEP 2023, 2024 results)

From the portability evaluation using the CMS pixel reconstruction (Patatrack) codebase:

- **NVIDIA A100 (CUDA backend):** ALPAKA achieves performance "very close to the native CUDA version"
- **AMD MI100 (HIP backend):** ALPAKA achieved **23% better performance** than native HIP (researchers flagged this as unexpected, requiring further profiling)
- **Overall conclusion:** "Alpaka was found to yield comparable, or in some cases better, performance than the direct CPU, CUDA, and HIP versions"

### The Launch Parameter Caveat

A critical finding from the CHEP 2024 portability study:

> "alpaka and Kokkos suffer from a ~40% slowdown if the kernel launch parameters determined by the portability layer are used."

This overhead disappears when launch parameters (block size, register count) are explicitly specified to match hand-tuned native values. This is a significant practical concern for any portability layer — default heuristics are not sufficient for peak performance; expert knowledge of the target GPU is still required.

### 2.5x Throughput Improvement Trick

Separately, the Fermilab CMS group found a 2.5× improvement in CUDA throughput by minimizing API calls for device memory allocation — demonstrating that portability-layer overhead is often in the host-side API, not the kernel execution itself.

### SYCL and std::par Are Significantly Worse

For the same CMS pixel reconstruction workload on NVIDIA A100:
- SYCL: ~10× slower than native CUDA
- std::par: ~2× slower than native CUDA
- ALPAKA/Kokkos: ~native

This is the strongest available evidence that SYCL is not yet production-viable for performance-critical HEP applications on NVIDIA hardware.

---

## 7. Framework Evaluation: Why CMS Chose ALPAKA Over Alternatives

From the official CMS decision documentation (Bocci et al., CHEP 2023):

| Framework | Assessment by CMS |
|-----------|-------------------|
| ALPAKA | "Most mature and better performing" — selected |
| Kokkos | Viable but "more verbose, more constraints on data structures" |
| SYCL | "Overhead concerns" on NVIDIA hardware |
| std::par | "Required many more kernels" for equivalent functionality |
| OpenMP offloading | "Added data movement costs" |

---

## 8. Relevance to libkdl

### Direct Motivation Cases

1. **Heterogeneous grid dispatch:** CMS must deploy CMSSW builds to thousands of grid nodes with mixed hardware. They maintain per-platform builds. libkdl would enable shipping one artifact that discovers hardware at load time.

2. **GPU fallback:** When GPUs are unavailable (maintenance, failures), CMS needs CPU fallback. Currently this requires the "switch" module infrastructure. libkdl's runtime selection would make this transparent.

3. **Future AMD/Intel GPUs:** CMS's ALPAKA investment is explicitly about readiness for AMD and Intel GPUs in future LHC runs. But they still need separate builds. libkdl would resolve this.

4. **The 40% launch parameter overhead:** This demonstrates that hardware-specific tuning is essential. A runtime dispatch system (libkdl) that carries multiple kernel variants tuned per GPU model addresses this directly.

### Argument Structure for Poster

> "CMS demonstrated at production scale that a single-source portability layer (ALPAKA) can match native performance with the right tuning. But they still compile separate binaries per GPU vendor and maintain switch infrastructure for CPU fallback. libkdl's runtime dispatch with hardware fingerprinting and per-variant selection is the natural evolution of this approach — eliminating the build matrix while preserving per-device optimization."

---

## 9. Key References

1. Bocci, A. et al. "Performance portability for the CMS Reconstruction with Alpaka." Fermilab-Conf-23-080 (2023). https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf
2. Kortelainen, M. et al. "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code." EPJ Web Conf. CHEP 2024. https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11008/epjconf_chep2024_11008.html
3. Bocci, A. et al. "Experience with the alpaka performance portability library in the CMS software." EPJ Web Conf. CHEP 2025. Fermilab-Conf-25-0145. https://lss.fnal.gov/archive/2025/conf/fermilab-conf-25-0145-cms-csaid.pdf
4. "Run-3 Commissioning of CMS Online HLT reconstruction using GPUs." EPJ Web Conf. CHEP 2024. https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11020/epjconf_chep2024_11020.html
5. "Heterogeneous Reconstruction of Hadronic Particle Flow Clusters with the Alpaka Portability Library." EPJ Web Conf. CHEP 2025. https://www.epj-conferences.org/articles/epjconf/abs/2025/22/epjconf_chep2025_01171/epjconf_chep2025_01171.html
6. CMS Patatrack. "Standalone pixel tracking." https://github.com/cms-patatrack/pixeltrack-standalone
7. CERN Document Server. "Heterogeneous Reconstruction of Hadronic PF Clusters with Alpaka." CMS-DP-2024-026. https://cds.cern.ch/record/2898660
8. "Patatrack or: How CMS Learned to Stop Worrying and Love the GPU." Andrea Bocci, CERN. Invited seminar, IN2P3, May 5, 2025. https://indico.in2p3.fr/event/34440/
