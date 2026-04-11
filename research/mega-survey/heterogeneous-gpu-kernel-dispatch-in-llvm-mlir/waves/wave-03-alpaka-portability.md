# Wave 03 — Alpaka Performance Portability Layer

**Angle:** alpaka-portability-layer
**Query:** "Alpaka performance portability abstraction layer heterogeneous computing CUDA HIP SYCL"
**Date:** 2026-04-06
**Priority source types:** Papers, CERN CMS production reports, Alpaka GitHub

---

## Source Index

| # | Title | URL | Date | Type | Relevance | Novelty |
|---|-------|-----|------|------|-----------|---------|
| S1 | Zenker et al. "Alpaka — An Abstraction Library for Parallel Kernel Acceleration" | https://arxiv.org/abs/1602.08477 | 2016 | Paper | 9/10 | 6/10 |
| S2 | Bocci et al. "Performance portability for the CMS Reconstruction with Alpaka" (Fermilab-Conf-23-080) | https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf | 2023 | Production Report | 10/10 | 9/10 |
| S3 | Kortelainen et al. "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code" (CHEP 2024) | https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11008.pdf | 2024 | Paper | 10/10 | 9/10 |
| S4 | Bocci et al. "Experience with the alpaka performance portability library in the CMS software" (CHEP 2025) | https://www.epj-conferences.org/articles/epjconf/pdf/2025/22/epjconf_chep2025_01141.pdf | 2025 | Production Report | 10/10 | 10/10 |
| S5 | "Evaluating Application Characteristics for GPU Portability Layer Selection" | https://arxiv.org/html/2601.17526 | Jan 2026 | Paper | 9/10 | 10/10 |
| S6 | Kwok et al. "Evaluating Portable Parallelization Strategies for HEP" | https://arxiv.org/html/2306.15869 | 2023 | Paper | 8/10 | 7/10 |
| S7 | "Heterogeneous reconstruction of hadronic PF clusters with Alpaka" (CHEP 2025) | https://www.epj-conferences.org/articles/epjconf/abs/2025/22/epjconf_chep2025_01171/epjconf_chep2025_01171.html | 2025 | Production Report | 9/10 | 8/10 |
| S8 | CMS CHEP 2024: Run-3 Commissioning of CMS Online HLT reconstruction using GPUs | https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11020.pdf | 2024 | Production Report | 9/10 | 8/10 |
| S9 | Performance of Heterogeneous Algorithm Scheduling in CMSSW (CHEP 2024) | https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11017/epjconf_chep2024_11017.html | 2024 | Paper | 8/10 | 7/10 |
| S10 | alpaka-group/alpaka GitHub (v2.2.0-rc) + alpaka3 (next-gen) | https://github.com/alpaka-group/alpaka | 2026 | Docs/Source | 8/10 | 7/10 |

---

## Source Summaries

### S1 — Zenker et al. 2016 (Foundational Paper) [Relevance 9/10, Novelty 6/10]

**Citation:** Zenker, E. et al. "Alpaka — An Abstraction Library for Parallel Kernel Acceleration." IEEE IPDPSW 2016. arXiv:1602.08477.

The foundational paper introducing the **Redundant Hierarchical Parallelism (RHP)** model that defines the Alpaka design philosophy. Kernels are expressed as C++ function objects (functors) with a templated `operator()` taking `TAcc const& acc` as the first argument. The accelerator type is a compile-time template parameter — zero runtime overhead because the backend resolves at compile time into native CUDA PTX, AMD HSACO, or OpenMP-annotated x86 code.

**Key benchmark results:**
- DGEMM on NVIDIA (K20): >94% of native CUDA performance across all matrix sizes
- HASEonGPU Monte Carlo port: identical execution time vs. native CUDA
- Portability across AMD, Intel, NVIDIA at ~20% of theoretical peak consistently

**Relevance to libkdl:** Establishes the performance ceiling for compile-time portability. Alpaka's ~94% efficiency is the target libkdl must match per-variant while adding runtime flexibility on top.

---

### S2 — Bocci et al. 2023 (CMS CHEP 2023 — Production Adoption) [Relevance 10/10, Novelty 9/10]

**Citation:** Bocci, A. et al. "Performance portability for the CMS Reconstruction with Alpaka." Fermilab-Conf-23-080 (2023).

The definitive paper documenting CMS's formal adoption of Alpaka as the official portability layer for Run 3 HLT after evaluating OpenMP, Kokkos, SYCL, std::par, and Alpaka. CMS selected Alpaka as "the more mature and better performing solution."

**Production context:** The CMS HLT must process ~100 kHz of proton-proton collision events in near-real-time, reducing ~40 TB/s of raw detector data to ~1 GB/s written to tape. GPU reconstruction covers ~40% of HLT runtime.

**Critical technical finding — compile-multiple-link-one architecture:**
CMS implements "runtime selection" by compiling all desired backends separately and linking them into one application. A runtime hardware-discovery layer then routes events to the appropriate compiled variant using CMSSW's **SwitchProducer** mechanism:

```
Alpaka source (single codebase)
  → nvcc + AccGpuCudaRt → CUDA module (.so)
  → hipcc + AccGpuHipRt → HIP module (.so)
  → g++ + AccCpuSerial  → CPU fallback module (.so)
  → linked into CMSSW binary
  → runtime: SwitchProducer inspects available GPUs → routes to best available variant
```

This is NOT true single-binary cross-vendor dispatch. It is multiple pre-compiled specializations bundled into one executable with a runtime routing layer.

**Memory pool finding (critical for throughput):** Without a caching allocator for host and device memory, Alpaka CUDA throughput collapses ~11.6x (from ~1840 events/sec to ~159 events/sec). Memory allocation API call overhead is the dominant bottleneck, not kernel execution.

**CMS evaluation summary:**
| Framework | CMS Assessment |
|-----------|----------------|
| Alpaka | "Most mature and better performing" — selected |
| Kokkos | "More verbose, more constraints on data structures" |
| SYCL | "Overhead concerns" on NVIDIA hardware |
| std::par | "Required many more kernels" |
| OpenMP offloading | "Added data movement costs" |

---

### S3 — Kortelainen et al. 2024 (CHEP 2024 — Quantitative Benchmarks) [Relevance 10/10, Novelty 9/10]

**Citation:** Kortelainen, M. et al. "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code." EPJ Web Conf. CHEP 2024.

The most quantitatively detailed benchmark comparing Alpaka, Kokkos, SYCL, std::par, and OpenMP for the CMS pixel reconstruction (Patatrack algorithm) on NVIDIA A100 and AMD MI100.

**NVIDIA A100 results:**
- Alpaka (CUDA backend): effectively matches native CUDA when properly tuned
- Kokkos (CUDA backend): matches native CUDA
- SYCL (DPC++ → CUDA): ~10x slower than native CUDA
- std::par (nvc++): ~2x slower than native CUDA

**AMD MI100 results:**
- Alpaka (HIP backend): **23% better than native HIP** (unexpected result, likely due to allocation pattern differences in the portability layer's memory pool)
- Kokkos (HIP backend): matches native HIP

**Critical finding — default launch parameter penalty (~40% overhead):**
> "alpaka and Kokkos suffer from a ~40% slowdown if the kernel launch parameters determined by the portability layer are used."

Resolution: Explicitly specifying block size and register count matching hand-tuned native values eliminates the overhead entirely. The portability layer abstracts the API but does NOT automatically discover optimal launch parameters — that still requires per-device expert knowledge.

**Direct implication for libkdl:** libkdl's multi-variant model (pre-compiled with explicitly tuned params per GPU model, selected at runtime via hardware fingerprinting) directly solves this. Each variant carries its optimal block size; no runtime heuristic needed.

---

### S4 — Bocci et al. 2025 (CHEP 2025 — Most Recent Production Status) [Relevance 10/10, Novelty 10/10]

**Citation:** Bocci, A. et al. "Experience with the alpaka performance portability library in the CMS software." EPJ Web Conf. CHEP 2025. Fermilab-Conf-25-0145.

The most current (2025) production status report. Key claims:
- Alpaka "achieves near-native performance" in production HLT on NVIDIA L4 GPUs
- Single code base, multiple backend toolchains, linked into one application
- "Seamlessly select the best backend at runtime" (via SwitchProducer)

**Concrete production deployment numbers (CMS HLT, 2024):**
- 2024 HLT farm expansion: **18 new nodes added**, each with 2x AMD EPYC Bergamo 9754 CPUs + 3x NVIDIA L4 GPUs
- **Total: >30,000 CPU cores and ~450 GPUs** in the HLT farm
- All GPU-accelerated algorithms run via Alpaka

**Production GPU workloads (Run 3, 2025 status):**
| Algorithm | Subsystem | HLT Runtime Share |
|-----------|-----------|-------------------|
| Pixel local reconstruction | Pixel Tracker | significant |
| Pixel-only track & vertex (Patatrack) | Pixel Tracker | significant |
| ECAL local reconstruction | EM Calorimeter | significant |
| HCAL local reconstruction | Hadronic Calorimeter | significant |
| Hadronic PF clustering (new 2024) | Particle Flow | new 2024 |

Collectively, these five algorithms account for **~40% of total HLT runtime** running on GPU.

**Roadmap:** CMS targets ≥50% GPU utilization by Run 4 (2027+) and ≥80% by Run 5.

---

### S5 — arXiv:2601.17526 (January 2026 — Most Recent Systematic Evaluation) [Relevance 9/10, Novelty 10/10]

**Citation:** "Evaluating Application Characteristics for GPU Portability Layer Selection." arXiv:2601.17526 (January 2026).

The most recent (2026) systematic evaluation of GPU portability layers including Kokkos, Alpaka, SYCL, OpenMP, and std::par, studying which application characteristics favor each framework.

**Key Alpaka-specific findings:**

1. **Launch latency advantage:** "Large additional penalties were not observed for SYCL, HIP, OpenMP, or Alpaka." Contrast: Kokkos adds tens of microseconds (sometimes exceeding 70 µs) on AMD GPUs specifically. Alpaka has no such AMD-specific overhead.

2. **Launch parameter tuning still required:** "Launch parameters need to be manually specified for Alpaka and Kokkos, otherwise the libraries choose suboptimal values and performance is about 30% worse." (Consistent with S3's ~40% finding; variance by workload type.)

3. **No native vendor library API bridge:** Alpaka provides no APIs targeting vendor-optimized routines (cuFFT, rocFFT, cuBLAS, rocBLAS). Applications requiring these must drop to vendor-specific code, breaking the portability abstraction.

4. **Explicit memory model:** Alpaka's explicit allocation model (unlike CUDA Unified Memory) is both a strength (predictable performance) and a burden (more boilerplate). The ~11.6x throughput collapse without a memory pool (S2) illustrates the operational cost.

5. **API verbosity:** "Alpaka offers a lower-level API than other portability layers like Kokkos, resulting in rather verbose application code."

6. **Compilation overhead:** Kokkos adds 10-20% compilation overhead depending on code complexity; Alpaka's deep template instantiation also impacts compile times significantly.

---

### S6 — Kwok et al. 2023 (Multi-Framework HEP Survey) [Relevance 8/10, Novelty 7/10]

**Citation:** Kwok, K.H.M. et al. "Evaluating Portable Parallelization Strategies for Heterogeneous Architectures in High Energy Physics." arXiv:2306.15869 (2023).

Multi-framework comparison across Kokkos, SYCL, OpenMP, std::par, and Alpaka using HEP mini-apps (Kalman filter tracking, DUNE, ATLAS, CMS). Hardware: NVIDIA A100, AMD MI100, Intel Xeon.

**Per-framework assessment relevant to Alpaka:**

| Framework | API Complexity | Community | GPU Performance |
|-----------|---------------|-----------|-----------------|
| Alpaka | Verbose (lower-level) | Medium (HEP-heavy) | Strong (when tuned) |
| Kokkos | Moderate | Large (DOE) | Strong (when tuned) |
| SYCL | Moderate | Intel-backed | Poor on NVIDIA |

**Confirmed Alpaka finding:** ~40% slowdown without explicit launch parameter tuning. CMS selected Alpaka over all alternatives for production use despite verbosity, because performance with tuning is unmatched.

---

### S7 — CHEP 2025: Hadronic PF Clustering with Alpaka [Relevance 9/10, Novelty 8/10]

**Citation:** "Heterogeneous reconstruction of hadronic particle flow clusters with the Alpaka Portability Library." EPJ Web Conf. CHEP 2025. https://www.epj-conferences.org/articles/epjconf/abs/2025/22/epjconf_chep2025_01171/epjconf_chep2025_01171.html

Documents the newest production Alpaka algorithm deployed in 2024 HLT — hadronic Particle Flow clustering. This is significant because it represents the fifth GPU-accelerated reconstruction algorithm in CMS, and it was built on Alpaka from scratch (not a CUDA port). Validates the feasibility of Alpaka-first development for new HEP algorithms.

**Dispatch model:** Same compile-multiple-link-one pattern as other CMS algorithms. The CUDA backend variant is the production path on current L4 GPUs; the CPU serial backend provides automatic fallback.

---

### S8 — CHEP 2024: Run-3 HLT Commissioning [Relevance 9/10, Novelty 8/10]

**Citation:** "Run-3 Commissioning of CMS Online HLT reconstruction using GPUs." EPJ Web Conf. CHEP 2024. https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11020.pdf

Documents the operational experience of running GPU-enabled CMSSW through CMS's production system. Key operational details:

- GPU workflows running through the same production pipeline as CPU-only workflows
- Backend selection at runtime via the SwitchProducer mechanism (not bare dlopen)
- CMS HLT began GPU use in 2022 data taking; Run 3 (2022-2025) is the operational baseline
- Separate CMSSW builds are maintained per platform (CUDA-enabled vs CPU-only) and distributed to appropriate grid nodes

The CMSSW scheduler routes entire events to GPU or CPU based on available hardware on the worker node. This is not kernel-level dispatch granularity — it is event-level routing. An event either runs its GPU algorithms on GPU (if one is available) or falls back to CPU entirely.

---

### S9 — CHEP 2024: Heterogeneous Algorithm Scheduling in CMSSW [Relevance 8/10, Novelty 7/10]

**Citation:** "Performance of Heterogeneous Algorithm Scheduling in CMSSW." EPJ Web Conf. CHEP 2024. https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11017/epjconf_chep2024_11017.html

Studies the impact of different scheduling design choices on event processing throughput for the Run-3 HLT application. Key architectural detail: CMSSW uses a task-based concurrent framework. GPU algorithms submit work asynchronously and the scheduler continues processing other events while GPU kernels execute. This is a critical design for high-throughput — the HLT must pipeline 100 kHz of events across CPUs and GPUs simultaneously.

**SwitchProducer mechanism (confirmed detail):** The SwitchProducer associates multiple module labels to one logical label. At runtime, the scheduler inspects available hardware on the worker node and selects which concrete module to schedule. If the node has an NVIDIA GPU, the CUDA backend module runs. If no GPU is present, the CPU serial backend module runs. This is the entire runtime dispatch mechanism in production — no dynamic code generation, no JIT.

---

### S10 — alpaka GitHub: v2.2.0-rc + alpaka3 [Relevance 8/10, Novelty 7/10]

**URL:** https://github.com/alpaka-group/alpaka (v2.2.0-rc) + https://github.com/alpaka-group/alpaka3

**Current backend matrix (v2.2.0-rc, 2026):**

| Backend | Target | Minimum Requirement | Status |
|---------|--------|---------------------|--------|
| AccGpuCudaRt | NVIDIA GPUs | CUDA 12.0+ | Production |
| AccGpuHipRt | AMD GPUs | HIP 6.0+ | Production |
| AccGpuSyclIntel | Intel GPUs/CPUs/FPGAs | oneAPI 2024.2+ | Production |
| AccCpuOmp2Blocks | Multi-core CPU | OpenMP 2.0+ | Production |
| AccCpuOmp2Threads | Multi-core CPU | OpenMP 2.0+ | Production |
| AccCpuThreads | Multi-core CPU | C++20 std::thread | Production |
| AccCpuSerial | Single-core CPU | None | Production |

CPU architectures: x86, ARM, RISC-V, Power 8+.

**alpaka3 (active major rewrite):** Breaking API changes from v2. Key changes: platforms must now be instantiated explicitly, SYCL backend updated to USM pointers + SYCL 2020, OpenMP 5/OpenACC/Boost.Fiber backends removed, namespace flattened. Adds experimental `std::mdspan` support for multi-dimensional views. Currently pre-release (680 commits on dev, 10 contributors, no v3.x release yet as of April 2026).

---

## Technical Architecture: The Alpaka Dispatch Model

### Layer 1 — Kernel Abstraction (Compile Time)

Alpaka kernels are C++ function objects. The accelerator type is a template parameter resolved entirely at compile time:

```cpp
struct MyKernel {
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* data, int n) const {
        auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        if (idx[0u] < n) data[idx[0u]] *= 2.0f;
    }
};
// Instantiation — AccGpuCudaRt resolved to PTX code path
using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, uint32_t>;
alpaka::exec<Acc>(queue, workDiv, MyKernel{}, data, n);
```

The same source produces: CUDA PTX with `nvcc`, AMDGPU HSACO with `hipcc`, x86 binary with `g++`. Zero runtime overhead at the kernel level.

### Layer 2 — Runtime Dispatch (CMSSW SwitchProducer)

CMS runtime dispatch is implemented at the **framework scheduler level**, not inside Alpaka itself:

1. All backend variants compiled separately and linked into one CMSSW binary
2. SwitchProducer associates a logical module label to multiple concrete implementations
3. At event processing startup on a given worker node, CMSSW inspects available GPU hardware
4. Scheduler routes all events for that node to the GPU backend (if GPU present) or CPU serial backend (if not)
5. Backend selection is node-level and session-level, not per-event or per-kernel

This means: a single CMSSW binary works on both GPU nodes and CPU-only nodes. But the routing granularity is coarse — an entire node commits to one backend for an entire processing session.

### Layer 3 — Memory Management (Production-Critical)

The default Alpaka allocation path calls the underlying vendor allocator (cudaMalloc/hipMalloc) per request. In production:
- Without caching allocator: ~159 events/sec (allocation overhead dominates)
- With caching allocator (memory pool): ~1840 events/sec

CMS implements a custom caching allocator for both host and device memory. This is not part of Alpaka itself — it is a CMSSW-level addition. Production deployment requires this allocator.

---

## Production Deployment Numbers (CERN CMS, 2024-2025)

| Metric | Value | Source |
|--------|-------|--------|
| Total HLT CPU cores | >30,000 | S4 |
| Total HLT GPUs | ~450 NVIDIA GPUs | S4 |
| GPU model (2024 expansion) | NVIDIA L4 | S4 |
| CPU model (2024 expansion) | AMD EPYC Bergamo 9754 | S4 |
| New nodes added (2024) | 18 nodes (2x CPU + 3x L4 each) | S4 |
| Input event rate | ~100 kHz (proton-proton collisions) | S2 |
| Raw data rate | ~40 TB/s reduced to ~1 GB/s | S2 |
| GPU fraction of HLT runtime | ~40% of total HLT | S2, S4 |
| Algorithms on GPU (2025) | 5 (pixel local, Patatrack, ECAL, HCAL, hadronic PF) | S4, S7 |
| Throughput with memory pool | ~1840 events/sec | S2 |
| Throughput without memory pool | ~159 events/sec (11.6x penalty) | S2 |
| Run 4 GPU target | ≥50% of HLT capacity | S4 |
| Run 5 GPU target | ≥80% of HLT capacity | S4 |

---

## Performance Data Summary

| Benchmark | Hardware | Alpaka vs. Native | Condition | Source |
|-----------|----------|-------------------|-----------|--------|
| DGEMM | NVIDIA K20 | >94% of native CUDA | All matrix sizes | S1 |
| HASEonGPU (MC) | NVIDIA K20 | 100% (identical) | Production port | S1 |
| CMS pixel (Patatrack) | NVIDIA A100 | ~100% of native CUDA | Tuned launch params | S3 |
| CMS pixel (Patatrack) | AMD MI100 | 123% of native HIP | Tuned (unexpected, memory pool effect) | S3 |
| CMS pixel (Patatrack) | NVIDIA A100 | ~60% of native CUDA | Default launch params (40% penalty) | S3 |
| CMS pixel (Patatrack) | NVIDIA L4 | Near-native | Memory pool enabled, Run 3 prod | S4 |
| Generic kernel launch latency | AMD GPUs | No additional penalty | vs. Kokkos (70 µs overhead) | S5 |

**Estimated P3 portability score (Pennycook metric):**
- Tuned (explicit launch params, memory pool): **~0.90–0.95** across NVIDIA + AMD + CPU
- Default (portability layer picks params): **~0.55–0.65** (30-40% penalty on GPU targets)

---

## Framework Comparison Matrix

### Alpaka vs. Kokkos

| Dimension | Alpaka | Kokkos |
|-----------|--------|--------|
| API level | Lower-level, more control | Higher-level, more abstractions |
| Verbosity | More boilerplate | Less boilerplate |
| AMD GPU launch latency | No additional penalty | +70 µs overhead on AMD (confirmed Dec 2025, kokkos/kokkos#8738) |
| GPU performance (tuned) | ~native | ~native |
| GPU performance (untuned) | 30-40% penalty | 30-40% penalty |
| Vendor library APIs | None (cuFFT, rocFFT must be called directly) | Provides Kokkos-Random, Kokkos-FFT |
| Community | Medium (HEP-heavy) | Large (DOE exascale) |
| HEP production use | CMS Run 3 (5 algorithms) | ATLAS experiments, alternatives |
| P3 score (estimated) | 0.90-0.95 (tuned) | 0.82-0.99 (Davis et al. 2025) |

**Key differentiator:** Kokkos adds significant AMD GPU launch latency (tens-to-hundreds of µs per kernel launch); Alpaka does not. For AMD-heavy future CMS runs, this is decisive.

### Alpaka vs. SYCL/DPC++

| Dimension | Alpaka | SYCL (DPC++) |
|-----------|--------|--------------|
| NVIDIA A100 performance | ~native (tuned) | ~10x slower than native CUDA |
| AMD GPU performance | ~native or better | Improving but lagging |
| Intel GPU | Production (oneAPI backend) | Native |
| Compile model | C++ templates (library) | Device compiler + runtime JIT |
| HEP production | CMS Run 3 (450 GPUs) | Not in HEP production |

**Key differentiator:** SYCL's 10x overhead on NVIDIA A100 is disqualifying for CMS's 100 kHz HLT requirement.

### Alpaka vs. AdaptiveCpp SSCP

| Dimension | Alpaka | AdaptiveCpp SSCP |
|-----------|--------|------------------|
| Backend selection | Compile-time (CMake flag) | Runtime (JIT on first dispatch) |
| Single binary | No (build matrix per vendor) | Yes (embedded LLVM IR) |
| Cold-start overhead | None (pre-compiled) | JIT latency, mitigated by cache |
| GPU performance | ~native (tuned) | Within ±10% of AOT |
| Adaptivity | None — fixed at compile time | Runtime specialization (+30-44%) |
| HEP production | CMS Run 3 | Not yet |

---

## Critical Gaps for libkdl Positioning

### Gap 1 — No True Runtime Dispatch (Most Critical)

Alpaka's backend is fixed at compile time per translation unit. CMS's "runtime selection" is a routing decision among pre-compiled variants — not dispatch from a single kernel representation to arbitrary hardware. Consequences:
- A new GPU vendor requires a new build and new CMSSW deployment
- Hardware discovered at deployment time cannot be served without anticipating it at build time
- CMS maintains separate CMSSW builds per platform (CUDA-enabled, CPU-only); grid nodes receive the appropriate build based on hardware inventory

**libkdl directly addresses this:** hardware fingerprinting at load time selects the optimal pre-compiled variant from a single artifact, covering vendors not anticipated at build time.

### Gap 2 — 30-40% Default Launch Parameter Penalty

The portability layer handles API abstraction but not optimization. Near-native performance requires explicit per-device tuning (block size, register count) that embeds device-specific knowledge into the codebase. This penalizes teams without per-GPU expertise.

**libkdl directly addresses this:** per-device variants are compiled with tuned launch parameters embedded at compile time; the dispatch mechanism selects the right variant at runtime without requiring per-device tuning knowledge at the application level.

### Gap 3 — Event-Level (Not Kernel-Level) Dispatch Granularity

CMS's SwitchProducer routes at the event level: an entire CMSSW worker node commits to one backend for an entire session. This prevents mixed-vendor dispatch within a single node (e.g., running CUDA for one kernel and HIP for another on different GPUs in the same machine).

### Gap 4 — Memory Pool Is Not Bundled

The ~11.6x throughput collapse without a memory pool is a deployment hazard. Production requires CMSSW-level caching allocator. Any adoption outside the CMS ecosystem must reimplement this. libkdl's memory management layer should bundle this by default.

### Gap 5 — No Compiler-Level Optimization Across Kernels

Alpaka is a library, not a compiler. No cross-kernel operator fusion, no memory layout transforms, no pass-based optimization. The backend compiler (nvcc/hipcc) optimizes individual kernels but cannot see across kernel boundaries. Compare with MLIR: tiling, fusion, vectorization as composable passes.

### Gap 6 — Build Matrix Complexity

Supporting N vendors requires N builds, N toolchain configurations, N CI matrix entries, N binary artifacts. CMS manages this through institutional build infrastructure. For teams without HEP-scale DevOps, this is a significant operational burden.

---

## Relevance to libkdl: Positioning Statement

Alpaka is the production state-of-the-art for single-source GPU portability, validated at scale by CMS Run 3: five reconstruction algorithms covering ~40% of HLT runtime, ~100 kHz event processing, >30,000 CPU cores and ~450 NVIDIA GPUs. With proper tuning and a memory pool, it achieves >94% of native CUDA on NVIDIA and exceeds native HIP performance on AMD MI100.

However, Alpaka's compile-time backend selection creates a build matrix that scales with vendor count. CMS mitigates by maintaining separate builds per platform and using a SwitchProducer for event-level routing. This is viable at HEP scale with institutional infrastructure but impractical for decentralized deployment (cloud, edge, grid nodes with unknown GPU inventory).

**libkdl's complementary value proposition:**
- Pre-compiled per-device variants (Alpaka's performance) + runtime fingerprint-based selection (no build matrix)
- Memory pool bundled by default (not a separately engineered add-on)
- Kernel-level (not event-level) dispatch granularity — enables heterogeneous node mixing
- Single binary artifact works across NVIDIA/AMD/CPU without anticipating hardware at build time

### Positioning Table

| System | Runtime dispatch | Per-device tuning | Single binary | HEP production | Memory pool |
|--------|-----------------|-------------------|---------------|----------------|-------------|
| Alpaka (CMS) | Event-level routing | Manual, required | No (build matrix) | Yes (450 GPUs, Run 3) | External (required) |
| Kokkos | Same model | Manual, required | No | Partial | External |
| AdaptiveCpp SSCP | Yes (JIT) | Automatic | Yes | No | N/A |
| IREE HAL | Partial (variant conditions) | AOT pass-based | Yes | No | N/A |
| **libkdl** | **Kernel-level fingerprint** | **Pre-compiled per device** | **Yes** | **Prototype** | **Bundled** |

---

## Key Citations for Poster

1. **S2 (Bocci 2023)** — CMS adopted Alpaka over all alternatives; production context: 100 kHz HLT, 40% runtime on GPU.
2. **S3 (Kortelainen 2024)** — 40% launch parameter penalty quantified; 23% AMD MI100 improvement; justifies per-device variant model.
3. **S4 (Bocci 2025)** — Most recent production numbers: 450 GPUs, 30,000+ CPU cores, 5 algorithms, Run 4/5 targets.
4. **S5 (arXiv:2601.17526, Jan 2026)** — Most recent systematic evaluation; confirms Alpaka's AMD launch latency advantage over Kokkos; 30% untuned penalty.
5. **S1 (Zenker 2016)** — Original design paper: >94% native CUDA is the compile-time performance ceiling.

---

## Angle Assessment

**Relevance to survey:** 10/10 — Alpaka is the highest-TRL heterogeneous GPU dispatch system in any production environment. CMS Run 3 with 450 GPUs and 100 kHz event rate is the most demanding real-world GPU dispatch validation in scientific computing.

**Novelty of angle:** 8/10 — The Alpaka story is well-documented but the libkdl positioning against it (filling the runtime dispatch gap and the build matrix gap) is an original contribution angle.

**Key differentiator from other agents' angles:** This report focuses specifically on the dispatch model architecture — SwitchProducer mechanism, event-level vs. kernel-level granularity, and the build-matrix burden — rather than the performance benchmarks in isolation.

**Recommended synthesis priority:** HIGH — Alpaka/CMS is the primary prior art for any claim about production-scale GPU kernel dispatch. Every libkdl claim must be positioned against this baseline.

---

## References

1. Zenker, E. et al. "Alpaka — An Abstraction Library for Parallel Kernel Acceleration." arXiv:1602.08477 (2016). https://arxiv.org/abs/1602.08477
2. Bocci, A. et al. "Performance portability for the CMS Reconstruction with Alpaka." Fermilab-Conf-23-080 (2023). https://lss.fnal.gov/archive/2023/conf/fermilab-conf-23-080-cms.pdf
3. Kortelainen, M. et al. "Evaluating Performance Portability with the CMS Heterogeneous Pixel Reconstruction code." CHEP 2024. https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11008.pdf
4. Bocci, A. et al. "Experience with the alpaka performance portability library in the CMS software." CHEP 2025. https://www.epj-conferences.org/articles/epjconf/pdf/2025/22/epjconf_chep2025_01141.pdf
5. "Evaluating Application Characteristics for GPU Portability Layer Selection." arXiv:2601.17526 (Jan 2026). https://arxiv.org/html/2601.17526
6. Kwok, K.H.M. et al. "Evaluating Portable Parallelization Strategies for HEP." arXiv:2306.15869 (2023). https://arxiv.org/html/2306.15869
7. "Heterogeneous reconstruction of hadronic PF clusters with Alpaka." CHEP 2025. https://www.epj-conferences.org/articles/epjconf/abs/2025/22/epjconf_chep2025_01171/epjconf_chep2025_01171.html
8. "Run-3 Commissioning of CMS Online HLT reconstruction using GPUs." CHEP 2024. https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11020.pdf
9. "Performance of Heterogeneous Algorithm Scheduling in CMSSW." CHEP 2024. https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_11017/epjconf_chep2024_11017.html
10. alpaka-group. "alpaka v2.2.0-rc." GitHub. https://github.com/alpaka-group/alpaka
11. alpaka-group. "alpaka3 (next-gen)." GitHub. https://github.com/alpaka-group/alpaka3
12. Kokkos/kokkos issue #8738: "HIP backend slower than native HIP on AMD MI300A." GitHub (Dec 2025). https://github.com/kokkos/kokkos/issues/8738
13. Bocci, A. "Patatrack or: How CMS Learned to Stop Worrying and Love the GPU." IN2P3 Indico (May 2025). https://indico.in2p3.fr/event/34440/
14. "Running GPU-enabled CMSSW workflows through the production system." CHEP 2024. https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_11021.pdf
