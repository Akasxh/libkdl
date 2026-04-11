# Wave 02: Kernel Dispatch Overhead Benchmarks
Search query: "GPU kernel dispatch latency overhead microseconds runtime cost measurement"
Sources found: 10
Date: 2026-04-06

> **Note:** wave-03-dispatch-overhead.md covers overlapping sources with deeper synthesis.
> This wave contributes complementary sources (HIP/ROCm parity data, KLAP, mobile GPU,
> framework-layer overhead breakdown) not duplicated in wave-03.

---

## Sources

### 1. Understanding the Overheads of Launching CUDA Kernels (ICPP 2019 Poster, Tsukuba HPCS)
- URL: https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf
- Type: paper (HPC conference poster)
- Date: August 2019 (ICPP, Kyoto)
- Relevance: 8/10
- Novelty: 6/10
- Summary: Systematic decomposition of CUDA kernel launch overhead into CPU-side (API call latency) and GPU-side (kernel start latency) components by Lingqi Zhang, Mohamed Wahib, and Satoshi Matsuoka. Uses high-resolution CPU timers combined with CUDA events to isolate each component. Establishes a canonical baseline methodology that later papers cite.
- Key detail: CPU-side cuLaunchKernel latency measured at **2–4 μs** depending on GPU model and system state; GPU-side kernel start latency approximately **1 μs** after the API call returns — total per-launch overhead **3–5 μs** on hardware circa 2019.

---

### 2. Constant Time Launch for Straight-Line CUDA Graphs (NVIDIA Technical Blog)
- URL: https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
- Type: blog (NVIDIA official)
- Date: September 2024
- Relevance: 9/10
- Novelty: 8/10
- Summary: Documents NVIDIA's architectural work in CUDA Toolkit 12.6 to reduce CUDA Graph launch overhead on Ampere hardware. Per-launch CPU overhead moved from a linear model to near-constant time, with a 25–40% speedup in graph instantiation and up to 15% improvement in repeat device runtime. Provides exact before/after microsecond measurements across graph sizes.
- Key detail: Repeat launch overhead reduced from `2 μs + 200 ns/node` to **`2.5 μs + ~1 ns/node`** (effectively constant-time for 10+ node graphs). 100-node graph device runtime: 61 μs → 53 μs. 1025-node graph device runtime: 629 μs → 567 μs. First-launch CPU overhead for 100 nodes: 25 μs → 15 μs.

---

### 3. PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch (arXiv:2503.19779)
- URL: https://arxiv.org/abs/2503.19779
- Type: paper (compiler/runtime systems, arXiv preprint)
- Date: March 2025
- Relevance: 9/10
- Novelty: 8/10
- Summary: Compiler framework that maximizes CUDA Graph coverage and benefits for ML workloads in PyTorch. Documents that individual CUDA kernel launches incur 5–10 μs CPU-side overhead each. Finds that 25% of CUDA Graph uses degrade performance due to parameter copy overhead during graph replay — worst case 29% regression for EOS benchmark.
- Key detail: DALLE-2 BS=1 on H100: 740 kernels, GPU execution **3.4 ms**, end-to-end **14 ms** — **75% of latency is kernel launch overhead alone**. Individual kernel launch CPU overhead: **5–10 μs**. Parameter copy in graph replay: up to **24%** of execution time.

---

### 4. Minimizing GPU Kernel Launch Overhead in Deep Learning Inference on Mobile GPUs (ACM HotMobile 2021)
- URL: https://dl.acm.org/doi/10.1145/3446382.3448606
- Type: paper (ACM workshop, mobile systems)
- Date: February 2021
- Relevance: 8/10
- Novelty: 7/10
- Summary: Analyzes inference time of CNNs on mobile GPUs (Adreno 650, Mali G76), revealing that kernel launch overhead accounts for a disproportionate fraction of total inference time for lightweight models. Develops a performance model predicting optimal kernel-flush period for minimizing overhead, then validates it empirically. Distinct contribution: mobile GPU overhead is structurally larger than server GPU overhead due to weaker host-side CPU.
- Key detail: Speedups of **up to 64%** on Adreno 650 (TensorFlow Lite) and **up to 31%** on Mali G76 (ARM Compute Library) purely from launch overhead reduction. Launch overhead constitutes the dominant cost for lightweight NNs on mobile GPUs — not compute.

---

### 5. HIP Launch Latency Investigation — Kokkos Issue #3670 (GitHub)
- URL: https://github.com/kokkos/kokkos/issues/3670
- Type: issue thread (practitioner benchmark, open-source project)
- Date: December 2020
- Relevance: 8/10
- Novelty: 8/10
- Summary: Christian Trott (Kokkos core developer) benchmarks HIP kernel launch overhead on MI25 (ROCm 3.8) versus CUDA equivalents. Finds HIP overhead 8–23× higher than CUDA depending on fence configuration, motivating a redesign of the Kokkos HIP launch path. Still highly cited as the canonical cross-platform launch latency comparison in portability-layer discussions.
- Key detail: HIP (baseline): **70 μs/kernel** (batched or fenced). HIP (optimized): **25 μs** batched / **41 μs** fenced. CUDA local launch: **3 μs** batched / **8 μs** fenced. CUDA global launch: **5 μs** batched / **10 μs** fenced. AMD HIP overhead is **8–23× higher** than CUDA on equivalent operations as of ROCm 3.8.

---

### 6. KLAP: Kernel Launch Aggregation and Promotion for Optimizing Dynamic Parallelism (MICRO 2016)
- URL: https://ielhajj.github.io/publications/paper/paper-klap-micro16.pdf
- Type: paper (MICRO, top-tier computer architecture)
- Date: October 2016
- Relevance: 7/10
- Novelty: 7/10
- Summary: Compiler techniques (kernel launch aggregation and promotion) for reducing the number of kernel launches spawned by CUDA dynamic parallelism. Aggregates launches from warp/block/kernel scope into a single aggregated kernel to reduce total launch count and increase occupancy. Source-to-source compiler implementation published on GitHub (illinois-impact/klap).
- Key detail: Geometric mean speedup of **6.58× over regular dynamic parallelism** via kernel launch aggregation; kernel launch promotion achieves **30.44× throughput improvement** for recursive producer-consumer algorithms. Demonstrates that launch count reduction is the single most effective optimization for dynamic-parallelism workloads.

---

### 7. Reduce CUDA Launch Overhead — Numba Issue #3003 (GitHub)
- URL: https://github.com/numba/numba/issues/3003
- Type: issue thread (practitioner benchmark, open-source project)
- Date: May 2018 (measurements remain architecturally relevant)
- Relevance: 6/10
- Novelty: 6/10
- Summary: Documents the gap between raw CUDA C kernel launch overhead and framework-mediated Python dispatch overhead in Numba. Identifies that the overhead is caused by Python-side setup code executed before the cuLaunchKernel call, not the CUDA call itself. Motivates JIT compilation of CPU-side launcher code to close the gap.
- Key detail: Numba CUDA kernel launch overhead: **~200 μs** (Python launch path). CuPy equivalent: **~25 μs**. Native CUDA C launch floor: **~5–10 μs** on Linux. Python dispatch overhead adds **20–40× overhead** over the native floor — entirely in host-side dispatch code, not driver.

---

### 8. Boosting Performance of Iterative Applications via Kernel Batching with CUDA Graphs (arXiv:2501.09398)
- URL: https://arxiv.org/html/2501.09398v1
- Type: paper (HPC, arXiv preprint)
- Date: January 2025
- Relevance: 8/10
- Novelty: 7/10
- Summary: Evaluates CUDA Graph-based kernel batching for iterative HPC applications on A100 and GH200. Develops a performance model identifying the optimal batch size that balances graph creation cost against amortized per-launch savings. Finds optimal batch size is hardware-independent at 50–100 kernels/graph; small kernels (<100 μs execution) benefit most.
- Key detail: Speedup with optimal batching: **>1.4× for skeleton applications**, **1.2×** for FDTD Maxwell solver on GH200. Graph creation cost is linear at ~4.2×10⁻⁶ μs/node, amortized rapidly after 50+ iterations. Performance gains most pronounced for kernels executing in **<100 μs** — exactly the ML inference regime.

---

### 9. Dispatch Kernel Overhead (OpenCL) — NVIDIA Developer Forums
- URL: https://forums.developer.nvidia.com/t/dispatch-kernel-overhead-opencl/48792
- Type: forum thread (practitioner measurement)
- Date: 2015 (hardware floor still architecturally relevant)
- Relevance: 6/10
- Novelty: 5/10
- Summary: Practitioner measurement comparing OpenCL vs CUDA kernel dispatch overhead on the same NVIDIA hardware. Documents that OpenCL imposes substantially higher per-kernel dispatch latency than CUDA due to the additional abstraction layers in the OpenCL ICD and runtime, even on NVIDIA's own OpenCL driver.
- Key detail: OpenCL kernel dispatch overhead measured at **~0.7 ms** CPU-side vs GPU execution of **~0.2 ms** — dispatch overhead is **3.5× the actual compute time** for small kernels. CUDA dispatch overhead on the same hardware is ~10–20× lower than OpenCL. This gap largely persists in modern SYCL-over-OpenCL paths.

---

### 10. In Pursuit of High-Fidelity GPU Kernel Benchmarking (Standard Kernel Blog)
- URL: https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/
- Type: blog (GPU performance engineering)
- Date: 2024
- Relevance: 7/10
- Novelty: 6/10
- Summary: Establishes methodological best practices for benchmarking GPU kernels when dispatch overhead is close to or exceeds computation time. Documents that naive CUDA event timing conflates dispatch overhead with kernel execution for sub-10 μs kernels, and that CUDA Graph replay averaging is required for accurate sub-microsecond measurement. Validates CUDA events, CUDA graphs, and Triton's do_bench as the three reliable methodologies.
- Key detail: Reliable measurement floor: **>10 μs execution time required** for naive timing. PyTorch profiler reports **zero timing** for sub-microsecond kernels. CUDA Graphs enable accurate measurement of ~1 μs kernels via replay averaging. Switch latency between successive kernels: **1–3 μs average**.

---

## Consolidated Dispatch Overhead Reference Table

| Measurement | Value | Hardware / Context | Source |
|---|---|---|---|
| CUDA null-kernel floor (H100) | 4.71 μs avg | H100 SXM | wave-03 / TaxBreak |
| CUDA null-kernel floor (H200) | 4.50 μs avg | H200 SXM | wave-03 / TaxBreak |
| cuLaunchKernel CPU latency | 2–4 μs | circa-2019 discrete GPU | ICPP 2019 |
| GPU-side kernel start latency | ~1 μs | after API returns | ICPP 2019 |
| PyTorch eager kernel dispatch | 5–10 μs | H100, per-kernel | PyGraph |
| CUDA Graph per-node (Ampere, CUDA 12.6) | ~1 ns/node + 2.5 μs base | Ampere | NVIDIA Blog |
| HIP launch (baseline, ROCm 3.8) | 70 μs | MI25 | Kokkos #3670 |
| HIP launch (optimized, ROCm 3.8) | 25 μs batched / 41 μs fenced | MI25 | Kokkos #3670 |
| CUDA launch (local, Kokkos) | 3 μs batched / 8 μs fenced | NVIDIA GPU | Kokkos #3670 |
| OpenCL dispatch overhead | ~0.7 ms | NVIDIA GPU | NVIDIA Forums |
| Numba Python CUDA dispatch | ~200 μs | Python-mediated | numba #3003 |
| CuPy CUDA dispatch | ~25 μs | Python-mediated | numba #3003 |
| Mobile GPU (Adreno 650) launch fraction | >64% of inference time | Adreno 650, lightweight CNN | HotMobile 2021 |
| KLAP speedup (aggregation) | 6.58× geomean | Kepler-era GPU | MICRO 2016 |

---

## Angle Assessment
- Coverage: Well-explored for CUDA; substantially under-measured for HIP/ROCm (the Kokkos #3670 issue from 2020 remains the primary cross-platform data point). Mobile GPU and framework-layer overhead are documented but with fewer peer-reviewed papers.
- Surprise findings: HIP overhead (70 μs unoptimized, 25 μs optimized) is **8–23× higher** than CUDA on equivalent hardware as of ROCm 3.8 — a dramatic gap that directly impacts libkdl's AMD backend design. Python-mediated dispatch (Numba: 200 μs) adds 40× overhead over the native floor, all in host code — demonstrating that the dispatch layer implementation language matters enormously.
- Gaps:
  - No modern (ROCm 5+/6+) equivalent of Kokkos #3670 with current AMD hardware (MI300X etc.)
  - No published measurement of Level Zero (oneAPI) dispatch floor vs CUDA/HIP
  - No measurement of WebGPU kernel dispatch overhead (browser GPU path)
  - SYCL overhead vs CUDA is documented anecdotally (1.79× slower end-to-end per wave-03) but not at the raw dispatch-floor level
- Suggested follow-up angles:
  1. ROCm 6 / MI300X null-kernel dispatch floor (AMD hardware parity with TaxBreak's H100 data)
  2. Level Zero vs CUDA vs HIP raw dispatch latency — especially for Intel Arc/Xe targets
  3. WebGPU dispatch overhead for edge/browser ML deployment
  4. hipGraph (AMD equivalent of CUDA Graphs) overhead reduction: does AMD's graph replay reach parity with CUDA 12.6's ~1 ns/node?
  5. Multi-stream / concurrent dispatch contention: how does dispatch overhead scale with N concurrent CUDA streams?
