# ALPAKA Performance Portability: Benchmarks, P3 Scores, and Framework Comparison

*Research compiled 2026-04-06 for "libkdl: Kernel Dynamic Linker" poster, LLVM Dublin 2026.*

**Relevance to libkdl:** 8/10 — ALPAKA is the primary compile-time portability baseline against which libkdl's runtime dispatch approach should be positioned. The performance data establishes what is achievable with compile-time selection; libkdl needs to demonstrate comparable per-device performance with the addition of runtime flexibility.

---

## 1. Original Paper: Zenker et al. 2016

**Citation:** Zenker, E., Worpitz, B., Widera, R., Huebl, A., Juckeland, G., Knüpfer, A., Nagel, W.E., Bussmann, M. "Alpaka — An Abstraction Library for Parallel Kernel Acceleration." IEEE IPDPSW 2016. arXiv:1602.08477.

**Institutional home:** Helmholtz-Zentrum Dresden-Rossendorf (HZDR), TU Dresden, Germany. Funded by EU Horizon 2020 Grant No. 654220.

### Core Design: Redundant Hierarchical Parallelism

ALPAKA defines a **four-level parallelism hierarchy** (Grid → Block → Thread → Element) that is a superset of GPU thread hierarchies. All levels are always present in the programming model; hardware that does not support a level (e.g., CPUs have no warp-level synchronization) simply collapses it. This is the "redundancy" property — kernels are written for the richest model and degrade gracefully.

| Level | GPU analog | CPU analog | Memory tier |
|-------|-----------|-----------|-------------|
| Grid | CUDA grid | Process | Global |
| Block | Thread block | Thread group | Shared |
| Thread | CUDA thread | OS thread / fiber | Registers |
| Element | Loop body / SIMD element | SIMD lane | Vector registers |

Note: Some CMS documentation refers to 5 levels (adding Warp); the paper itself defines 4. The warp level is implicit within a Block.

### Benchmark Results (Original Paper)

**DGEMM single-source kernel:**
- Achieves **~20% of theoretical peak performance** consistently across AMD, Intel, and NVIDIA hardware
- Matches the performance of respective native implementations (i.e., the portability overhead is zero — native code at the same optimization level also achieves ~20% of theoretical peak for this benchmark)

**DGEMM CUDA overhead:**
- ALPAKA CUDA kernels demonstrate **>94% relative performance** vs native CUDA for almost all matrix sizes
- Overhead ≤6% across the benchmark range

**HASEonGPU (Monte Carlo integration, real application):**
- ALPAKA version using the CUDA backend on NVIDIA K20: **identical execution times** vs native CUDA
- "Does not show any overhead at all"

**Key finding:** Portability overhead in ALPAKA is primarily in host-side API call overhead, not kernel execution. With proper tuning, kernel-level performance matches native.

---

## 2. CMS Production Benchmarks (2023–2025)

### 2.1 Patatrack Pixel Reconstruction (CHEP 2023, CHEP 2024)

**Workload:** CMS heterogeneous pixel track and vertex reconstruction (Patatrack). Includes: pixel cluster finding, local reconstruction, cellular automaton track building, Kalman filter fitting, primary vertex finding.

**Hardware tested:** NVIDIA A100, AMD MI100, Intel Xeon CPUs (for portability evaluation). Production HLT uses NVIDIA T4/L4.

**NVIDIA A100 results (CHEP 2024):**
- ALPAKA (CUDA backend): "very close to native CUDA version"
- Kokkos (CUDA backend): "very close to native CUDA version"
- SYCL (DPC++ to CUDA): ~10× slower than native CUDA
- std::par (nvc++): ~2× slower than native CUDA

**AMD MI100 results (CHEP 2024):**
- ALPAKA (HIP backend): **23% better** than native HIP
  - Cause unknown — researchers flagged for further profiling
  - Likely due to ALPAKA's memory allocation patterns or launch config being accidentally better-tuned than the "native" reference
- Kokkos (HIP backend): "very close to native HIP"

**The kernel launch parameter caveat (critical finding):**
> "alpaka and Kokkos suffer from a ~40% slowdown if the kernel launch parameters determined by the portability layer are used."

Resolution: explicit specification of launch parameters (block size + registers per thread) matching hand-tuned native values eliminates the overhead. This means performance-portable code still requires hardware-specific tuning knowledge to achieve peak performance — the portability layer handles the API, not the optimization.

### 2.2 Event Throughput (Patatrack Standalone Testbed)

Raw event throughput numbers from the portability evaluation:
- **Native CUDA (9 concurrent events):** 1840 ± 20 events/s
- **ALPAKA CUDA (memory pool):** Comparable (specific number not published in accessible text)
- **ALPAKA CUDA (without memory pool):** ~159 ± 1 events/s — 11.6× drop

This reveals that ALPAKA's performance is highly dependent on memory allocation strategy. The memory pool optimization (pre-allocating a pool rather than calling cudaMalloc per-event) is essential for throughput-oriented workloads. This is a key operational finding for any GPU dispatch system.

### 2.3 CHEP 2025: Production Experience Paper

"Experience with the alpaka performance portability library in the CMS software" (Bocci et al., EPJ Web Conf. CHEP 2025, Fermilab-Conf-25-0145):

Key claims:
- ALPAKA "achieves near-native performance" in production HLT deployment
- "Single code base using different toolchains for each backend, linked into a single application"
- "Seamlessly select the best backend at runtime"
- Successfully running on NVIDIA L4 GPUs in 2024 HLT farm

---

## 3. Cross-Framework Performance Portability Comparison

### 3.1 Davis et al. 2024/2025 (LLNL/UMD, ICS 2025)

**Citation:** Davis, J.H. et al. "Taking GPU Programming Models to Task for Performance Portability." ICS 2025. arXiv:2402.08950.

**Frameworks:** CUDA, HIP, Kokkos, RAJA, OpenMP, OpenACC, SYCL
**Hardware:** Frontier (AMD MI250X), Summit (NVIDIA V100)
**Workloads:** 4 proxy apps across arithmetic intensity spectrum
**Note:** ALPAKA not included in this study (DOE-centric framework selection)

**P3 scores (Pennycook metric, harmonic mean of efficiency across platforms):**
- Kokkos: 0.82–0.99 (best overall)
- RAJA: 0.70–1.00 (best for low arithmetic intensity)
- SYCL: inconsistent — good on Summit (V100), unreliable on Frontier (MI250X)
- OpenMP/OpenACC: portable but cannot match Kokkos/RAJA portability scores

**Headline finding:** "Kokkos and RAJA offer the most promise empirically as performance portable programming models" — but this study did not test ALPAKA.

### 3.2 Evaluating Portable Parallelization Strategies for HEP (arXiv:2306.15869, 2023)

**Citation:** Kwok et al. "Evaluating Portable Parallelization Strategies for Heterogeneous Architectures in High Energy Physics." arXiv:2306.15869 (2023).

**Frameworks:** Kokkos, SYCL, OpenMP, std::par, ALPAKA
**Workload:** p2z and p2r mini-apps (Kalman filter, DUNE, ATLAS, CMS tracking)
**Hardware:** NVIDIA A100, AMD MI100, Intel Xeon CPUs

**Per-framework assessment (qualitative from paper):**

| Framework | API Complexity | Documentation | Community | CPU Performance |
|-----------|---------------|---------------|-----------|-----------------|
| ALPAKA | Verbose (lower-level) | Good | Medium | Strong |
| Kokkos | Moderate | Excellent | Large | Strong |
| SYCL | Moderate | Good | Intel-backed | Variable |
| OpenMP | Low | Extensive | Largest | Variable |
| std::par | Minimal | Poor | N/A | Compiler-dependent |

**ALPAKA-specific findings:**
- Matches native performance when tuned
- Lower-level API gives more control but requires more boilerplate
- ~40% slowdown without explicit launch parameter tuning (confirmed)
- Chosen by CMS over all alternatives for production use

### 3.3 Frontiers in Big Data: HEP Tracking Portability (2024)

**Citation:** "Exploring code portability solutions for HEP with a particle tracking test code." Frontiers in Big Data (2024). DOI: 10.3389/fdata.2024.1485344.

**Key finding:** Performance "varies significantly depending on the details of the implementation" — the portability framework is less important than the quality of the implementation within it.

---

## 4. ALPAKA Backend Matrix (v2.1.1, December 2025)

| Backend | Target | Minimum Version | Production Status |
|---------|--------|-----------------|-------------------|
| AccGpuCudaRt | NVIDIA GPUs | CUDA 12.0+ | Production |
| AccGpuHipRt | AMD GPUs | HIP 6.0+ | Production |
| AccGpuSyclIntel | Intel GPUs/CPUs | oneAPI 2024.2+ | Production |
| AccCpuOmp2Blocks | Multi-core CPU | OpenMP 2.0+ | Production |
| AccCpuOmp2Threads | Multi-core CPU | OpenMP 2.0+ | Production |
| AccCpuThreads | Multi-core CPU | C++20 std::thread | Production |
| AccCpuTbbBlocks | Multi-core CPU | TBB | Production |
| AccCpuSerial | Single-core CPU | None | Production |

CPU architecture support: x86, ARM, RISC-V, Power 8+.

**Next-gen:** alpaka3 (major rewrite, breaking API changes) is under active development at `github.com/alpaka-group/alpaka3` but has no releases as of 2025.

---

## 5. ALPAKA vs IREE: Portability Model Comparison

| Property | ALPAKA | IREE |
|----------|--------|------|
| Portability mechanism | C++ templates (compile-time) | MLIR IR + JIT/AOT (compiler) |
| Backend selection | CMake/typedef → separate build | Target triple → single artifact |
| Runtime dispatch | Single binary, multiple compiled variants | Yes, via HAL |
| Optimization | Delegated to backend compiler (nvcc/hipcc) | MLIR passes at each IR level |
| Memory model | Explicit typed buffers | Memref with layout transforms |
| ML-specific ops | None (raw parallelism) | linalg, tosa, stablehlo dialects |
| HEP adoption | CMS Run 3 production | None |
| Maturity | 10+ years | 5+ years |

---

## 6. P3 Portability Metric Applied to ALPAKA

The Pennycook P3 metric (harmonic mean of efficiency across platforms H):

    PP(a, p, H) = |H_e| / Σ_{i ∈ H_e} 1/e_i(a, p)

where e_i is either architectural efficiency (fraction of peak hardware performance) or application efficiency (fraction of best observed performance on H).

**ALPAKA estimated P3 score (from published data):**
- NVIDIA A100: ~1.00 (near-native after tuning)
- AMD MI100: ~1.00 (reportedly 23% better than native HIP reference)
- Intel GPU (SYCL backend): Production as of 2025
- CPU (OpenMP/TBB): Near-native

If ALPAKA maintains near-native performance across A100, MI100, and Intel GPU, its P3 score would be **0.90–0.95+** — competitive with or exceeding Kokkos/RAJA scores from Davis et al.

**Caveat:** Without explicit performance-tuned launch parameters, the 40% slowdown penalty brings effective efficiency to ~0.60 on GPU targets, yielding a P3 score of roughly **0.55–0.65**. Proper tuning is not optional.

---

## 7. Key Limitations for Poster Positioning

1. **No runtime dispatch:** Backend is compile-time fixed per translation unit. The "runtime selection" in CMSSW means choosing between pre-compiled variants at job startup — not dispatching kernels to arbitrary hardware discovered mid-run.

2. **Template complexity tax:** Deep C++ template metaprogramming → long compile times, opaque error messages, steep learning curve. CMS invested significant developer training.

3. **No optimization passes:** ALPAKA is a library, not a compiler. It cannot fuse adjacent operators, transform memory layouts, or apply domain-specific optimizations. The backend compiler (nvcc, hipcc) optimizes individual kernels but has no cross-kernel view.

4. **Launch parameter brittleness:** The ~40% slowdown without explicit tuning means portability does not automatically yield performance. Per-device tuning of block sizes and register counts is still required.

5. **Single-node scope:** No distributed dispatch, no network-aware kernel placement.

---

## 8. Positioning Statement for Poster

> "ALPAKA is the production state-of-the-art for single-source GPU portability, validated at scale by CMS Run 3. With proper tuning it achieves >94% of native CUDA performance on NVIDIA and exceeds native HIP performance on AMD. But its compile-time backend selection model cannot adapt to hardware discovered at deployment time, and its 40% default-parameter penalty reveals that portability without runtime optimization is incomplete. libkdl's multi-variant dispatch model addresses both: pre-compiled variants tuned per device, selected at runtime via hardware fingerprinting — delivering ALPAKA-level per-device performance with IREE-style runtime flexibility."

---

## 9. Key References

1. Zenker, E. et al. "Alpaka — An Abstraction Library for Parallel Kernel Acceleration." IEEE IPDPSW 2016. arXiv:1602.08477. https://arxiv.org/abs/1602.08477
2. Bocci, A. et al. "Performance portability for the CMS Reconstruction with Alpaka." Fermilab-Conf-23-080 (2023). https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf
3. Kortelainen, M. et al. "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code." EPJ Web Conf. CHEP 2024. https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11008.pdf
4. Bocci, A. et al. "Experience with the alpaka performance portability library in the CMS software." EPJ Web Conf. CHEP 2025. Fermilab-Conf-25-0145. https://lss.fnal.gov/archive/2025/conf/fermilab-conf-25-0145-cms-csaid.pdf
5. Davis, J.H. et al. "Taking GPU Programming Models to Task for Performance Portability." ICS 2025. arXiv:2402.08950. https://arxiv.org/abs/2402.08950
6. Kwok, K.H.M. et al. "Evaluating Portable Parallelization Strategies for Heterogeneous Architectures in High Energy Physics." arXiv:2306.15869 (2023). https://arxiv.org/html/2306.15869
7. Pennycook, S.J. et al. "A Metric for Performance Portability." arXiv:1611.07409 (2016). https://arxiv.org/abs/1611.07409
8. Pennycook, S.J. et al. "Implications of a metric for performance portability." Future Generation Computer Systems, Vol. 92 (2019). DOI: 10.1016/j.future.2017.08.007
9. alpaka-group. "Alpaka v2.1.1." GitHub. https://github.com/alpaka-group/alpaka
10. alpaka-group. "alpaka3 (next-gen)." GitHub. https://github.com/alpaka-group/alpaka3
11. "Exploring code portability solutions for HEP with a particle tracking test code." Frontiers in Big Data (2024). DOI: 10.3389/fdata.2024.1485344. https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1485344/full
