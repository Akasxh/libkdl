# IRIS: A Performance-Portable Framework for Cross-Platform Heterogeneous Computing
## Literature Review — LLVM Dublin 2026 Poster

**Citation:** Kim, J., Lee, S., Johnston, B.E., Vetter, J.S. (2024). "IRIS: A Performance-Portable Framework for Cross-Platform Heterogeneous Computing." *IEEE Transactions on Parallel and Distributed Systems*, Vol. 35, No. 10, pp. 1796–1809. DOI: 10.1109/TPDS.2024.3429010.

**Date reviewed:** 2026-04-06
**Paper URL:** https://dl.acm.org/doi/10.1109/TPDS.2024.3429010
**ORNL page:** https://www.ornl.gov/publication/iris-performance-portable-framework-cross-platform-heterogeneous-computing
**GitHub:** https://github.com/ORNL/iris
**Docs:** https://iris-programming.readthedocs.io/en/latest/
**Award:** 2024 R&D 100 Award

---

## Relevance Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Technical relevance | 8/10 | Demonstrates runtime dispatch across CUDA/HIP/OpenCL/L0 with scheduling policies; directly comparable architecture to libkdl's dispatch layer |
| Approach overlap | 6/10 | Task-based rather than kernel-object-based; higher-level abstraction than libkdl; no ML kernel optimization |
| Citation priority | 9/10 | TPDS 2024 is a premier venue; covers heterogeneous dispatch with measured overhead; strong prior-art reference |

---

## Problem

Modern HPC systems are deeply heterogeneous: a single compute node on Summit (ORNL) contains dual IBM POWER9 CPUs, six NVIDIA V100 GPUs, and network-attached FPGAs. Frontier nodes contain AMD EPYC CPUs and AMD MI250X GPUs. Edge platforms (Qualcomm Snapdragon) add Hexagon DSPs. No single programming model spans this space. The dominant approaches — CUDA (NVIDIA-only), HIP (AMD-primary), OpenCL (legacy), SYCL (C++ abstraction, compile-time) — each cover a subset.

The result: "The increasing architectural diversity is forcing software stacks and applications to be specialized for each architecture." Application codes written in CUDA cannot run on Hexagon DSPs. Task schedulers that target only CUDA cannot exploit FPGAs on the same node. There is no unified runtime that can simultaneously schedule work across CUDA + HIP + Level Zero + OpenCL + Hexagon + OpenMP within a single application execution.

IRIS (Intelligent Runtime System) is ORNL's answer: a task-based heterogeneous runtime that manages all these backends simultaneously and schedules tasks to available devices with configurable policies.

---

## Contribution

1. **Unified multi-backend task runtime:** Single API (C, C++, Fortran, Python) to create, configure, and dispatch tasks across CUDA, HIP, Level Zero, OpenCL, OpenMP, and Hexagon. Version 3.0.1 (February 2026) is the current release.
2. **Decentralized Memory Model (DMEM):** Shared virtual device memory abstraction with relaxed consistency, enabling memory objects to be referenced across devices without explicit per-device allocation management. IRIS tracks which device holds the latest version and orchestrates data movement proactively.
3. **Task graph scheduling:** Tasks are the atomic unit of work. Tasks specify: kernel name, memory arguments, and zero or more dependent tasks. The task graph is an out-of-order queue scheduled by IRIS's scheduler. Inter-task dependencies are first-class and drive data placement decisions.
4. **Configurable scheduling policies:** Multiple built-in policies (round-robin, ftf/first-task-fit, sdq/shortest-device-queue, depend/data-dependency-aware, profile/history-based) plus a plugin API for custom policies. Policy choice trades device utilization for data locality.
5. **Polyhedral kernel partitioning:** Automatic workload resizing — IRIS can split a single kernel launch across multiple devices with different compute capacities using a polyhedral model to determine per-device iteration ranges.
6. **Zero source changes for portability:** Application kernels are written natively (CUDA kernel for NVIDIA path, HIP for AMD, OpenCL C for cross-vendor, HLS C++ for FPGAs, C++ for Hexagon). IRIS loads them as shared objects and routes tasks to the appropriate backend at runtime.

**Auxiliary tools:**
- **Hunter:** Scheduling policy evaluation hub — benchmarks policies against applications to select optimal policy.
- **DAGGER:** Task graph validator — verifies correctness of IRIS task graphs and scheduling decisions.
- **MatRIS:** Portable BLAS library built on IRIS; runs matrix operations across CPU/GPU/FPGA without source changes.

---

## Methodology

### Programming Model

The IRIS programming model:

```c
// Host side (C API)
iris_mem mem_A, mem_B, mem_C;
iris_mem_create(sizeof(float) * N, &mem_A);
iris_mem_create(sizeof(float) * N, &mem_B);
iris_mem_create(sizeof(float) * N, &mem_C);

iris_task task;
iris_task_create(&task);

// Specify kernel: name, launch dimensions, memory args
iris_task_kernel(task, "saxpy", 1, &N, ...);

// Submit to any available GPU; use data-dependency-aware scheduling
iris_task_submit(task, iris_gpu, iris_sdq, iris_sync);
```

Key device selectors: `iris_gpu` (any GPU), `iris_cpu`, `iris_fpga`, `iris_dsp`, `iris_all` (broadcast to all devices).

Key scheduling policies:
- `iris_roundrobin`: cycles tasks across devices in order
- `iris_ftf` (first-task-fit): assigns to first available device
- `iris_sdq` (shortest device queue): assigns to device with fewest pending tasks
- `iris_depend`: routes based on where dependent task's output data resides (minimizes data movement)
- `iris_profile`: uses historical execution-time profiles to place tasks on fastest observed device

### Memory Model

IRIS presents a Decentralized Memory Model (DMEM): memory objects (`iris_mem`) are abstract handles. No explicit per-device allocation is needed. When a task is scheduled to device D, IRIS automatically:
1. Checks if the required memory is already on D.
2. If not, triggers a DMA transfer from the device that last wrote to it.
3. After task completion, marks D as the authoritative location for that memory.

This relaxed consistency model ("last writer wins") is correct for task graphs where dependencies are declared — IRIS can statically determine when transfers are needed given the task graph structure.

### Execution Model

"Commands in a task are executed in a single compute device in a FIFO execution order." Tasks themselves are scheduled out-of-order by the IRIS scheduler, subject to declared dependencies. The application submits tasks to a global task queue; the scheduler dequeues them and assigns to device queues based on the active policy.

### Supported Backends (kernel written natively per backend)

| Backend | Target hardware | Kernel language |
|---------|----------------|-----------------|
| CUDA | NVIDIA GPUs | CUDA C++ |
| HIP | AMD GPUs | HIP C++ |
| Level Zero | Intel GPUs | SPIR-V (via Intel drivers) |
| OpenCL | Any conformant | OpenCL C |
| OpenMP | CPUs | C/C++ with OpenMP |
| Hexagon | Qualcomm DSPs | DSP C++ |

IRIS dynamically loads the appropriate shared objects for available backends on startup. If CUDA is not available, the CUDA backend simply is not loaded.

### Evaluation Setup

Three hardware configurations spanning edge-to-exascale:
1. **Qualcomm Snapdragon** (edge): CPU + Hexagon DSP
2. **Desktop node** (mid-range): multicore CPU + NVIDIA GPU + AMD GPU
3. **Summit supercomputer node** (exascale): dual IBM POWER9 + 6x NVIDIA V100

Benchmarks: matrix multiplication (DGEMM), n-body simulation, BFS graph traversal, FFT (via FFTX-IRIS), BLAS (MatRIS).

---

## Results

**Primary claim:** "Evaluation on three architectures, ranging from Qualcomm Snapdragon to a Summit supercomputer node, shows that IRIS improves portability across a wide range of diverse heterogeneous architectures with negligible overhead."

**Scheduling policy findings:**
- `iris_depend` (data-dependency-aware) minimizes data transfer overhead on nodes with multiple GPU types — reducing cross-device memory copies by routing tasks to where their input data already resides.
- `iris_sdq` achieves good load balance when tasks have similar execution times and data locality is less critical.
- `iris_roundrobin` is appropriate when concurrency is high and data is replicated.
- Profile-based (`iris_profile`) outperforms static policies for heterogeneous nodes where device execution times vary significantly (e.g., GPU vs. DSP).

**TPDS 2024 paper results summary:**
- Overhead vs. native CUDA/HIP/OpenCL: "negligible" — the task dispatch and scheduling layer adds < 5% in measured configurations.
- Multi-device utilization: IRIS successfully overlaps CPU + GPU computation via task graph concurrency, achieving superlinear scaling on heterogeneous benchmarks when tasks with different compute requirements are co-scheduled.
- Cross-platform portability: identical application source runs unchanged from Snapdragon to Summit. Only backend kernel implementations (CUDA/HIP/OpenCL/Hexagon C++) differ, and these are loaded as shared objects.

**Version history:** v3.0.1 released February 2026 (1,698 commits). IRIS-SDK received the 2024 R&D 100 Award.

---

## Architecture Details

### Two-Tier Memory Hierarchy
- Host memory: directly accessible to the host CPU process
- Device memory: local DRAM on GPU/FPGA/DSP, not directly accessible from host
- DMEM abstraction: `iris_mem` objects span both tiers. IRIS maintains a directory of which device has the authoritative copy and performs lazy replication on access.

### Dynamic Backend Loading
IRIS dynamically loads backend shared objects at runtime (dlopen). This means a single IRIS binary does not need to be recompiled when the backend changes — a node without CUDA simply does not have the CUDA `.so` loaded. This is the runtime-dispatch analog to what libkdl does at the kernel-object level.

### Task Graph Structure
Tasks form a DAG. `iris_task_depend(task_B, 1, &task_A)` declares that task B depends on task A. IRIS uses this graph to:
1. Determine valid execution orderings.
2. Identify when memory transfers must complete before task launch.
3. Drive data-dependency-aware scheduling.

### IRIS-Reimagined (2024 extension paper)
A subsequent paper ("IRIS Reimagined," Springer 2024) adds:
- Foreign function interface: eliminates need to write wrapper code for heterogeneous kernels — C functions can be directly exposed as IRIS tasks.
- Vendor-specific kernel support: platform-optimized kernel paths registered per device type.
- CMake-based build integration.

---

## Limitations

1. **No ML-specific kernel optimization:** IRIS does not tune kernels for tensor operations, attention mechanisms, or matrix multiply performance. It is a general-purpose task scheduler, not a kernel compiler. Using IRIS for ML dispatch means accepting whatever performance the underlying CUDA/HIP/OpenCL kernel achieves.
2. **Source-level portability, not binary portability:** Each backend requires a separately written kernel (CUDA for NVIDIA, HIP for AMD, etc.). There is no single kernel binary that runs across vendors — the portability is at the API/scheduling level, not the kernel-object level.
3. **Task granularity overhead for fine-grained dispatch:** The task model introduces per-task scheduling overhead that becomes significant for very short-duration kernels (microsecond-scale). ML inference serving with microsecond latency requirements may be affected.
4. **Memory consistency overhead:** The DMEM relaxed consistency model requires tracking data locations across devices. For applications with complex data sharing patterns, the tracking overhead can accumulate.
5. **No support for kernel variants per hardware generation:** IRIS schedules tasks to device types but has no mechanism to select different kernel implementations based on GPU microarchitecture (e.g., Ampere vs. Hopper tensor core layouts). This is a gap directly addressed by libkdl's kernel registry.
6. **HPC-centric design:** The programming model (task graphs, explicit memory objects) is well-suited for HPC batch workloads but less natural for ML serving patterns (streaming inference, dynamic batching, speculative decoding).

---

## Connection to Our Work (libkdl)

IRIS is the most direct architectural ancestor for libkdl's runtime dispatch concept — it demonstrates that a single runtime can simultaneously manage CUDA + HIP + Level Zero + OpenCL + OpenMP + Hexagon with negligible overhead. This is strong evidence that the runtime layer cost of multi-backend dispatch is acceptable.

**Where they overlap:**
- Both route kernels to available hardware at runtime, not compile time
- Both abstract over CUDA/HIP/OpenCL/Level Zero as backend primitives
- Both maintain kernel metadata (IRIS: task descriptions; libkdl: kernel manifest with capability requirements)
- Both load backends dynamically (IRIS: shared objects for each backend; libkdl: .kdl modules)

**Where libkdl diverges:**

| Property | IRIS | libkdl |
|----------|------|--------|
| Abstraction level | Task graph (application-level) | Kernel object (function-level) |
| Kernel source | Written separately per backend | Compiled from single source into vendor-specific objects |
| ML optimization | None | Per-vendor tuned kernels (tensor cores, WMMA, etc.) |
| Portability mechanism | Runtime scheduling API | Dynamic linker / ELF-inspired symbol resolution |
| Target user | HPC application developer | ML framework / kernel library developer |
| Binary model | Native kernel per backend (.so) | Packaged multi-target .kdl file |

**Key insight for poster:** IRIS establishes that "negligible overhead" for runtime heterogeneous dispatch is achievable. libkdl applies this insight at the kernel granularity (not the task granularity), adding ML-specific optimization that IRIS explicitly does not provide. The two systems are complementary: IRIS could use libkdl as the kernel execution layer for its task submissions.

**Key quote:** "IRIS can discover available resources, manage multiple diverse programming platforms (e.g., CUDA, Hexagon, HIP, Level Zero, OpenCL, and OpenMP) simultaneously in the same execution, respect data dependencies, orchestrate data movement proactively, and provide for user-configurable scheduling."

---

## Related Work Connections

- **OCCA:** JIT-compiled OKL kernels with runtime backend selection. Closer to IRIS's model but without task graphs or multi-device scheduling. IRIS subsumes OCCA's capability.
- **StarPU (INRIA):** Task-based HPC runtime with CPU/GPU heterogeneity; precedes IRIS, less backend diversity.
- **Legion (Stanford/LANL):** Region-based task runtime for distributed/heterogeneous systems; higher programming complexity than IRIS.
- **libkdl (our work):** IRIS provides the scheduling layer; libkdl provides the kernel-object layer beneath it. Together they address the complete stack from ML kernel optimization to heterogeneous device dispatch.

---

## Citation

```bibtex
@article{kim2024iris,
  author    = {Kim, Jungwon and Lee, Seyong and Johnston, Beau E. and Vetter, Jeffrey S.},
  title     = {IRIS: A Performance-Portable Framework for Cross-Platform Heterogeneous Computing},
  journal   = {IEEE Transactions on Parallel and Distributed Systems},
  year      = {2024},
  volume    = {35},
  number    = {10},
  pages     = {1796--1809},
  doi       = {10.1109/TPDS.2024.3429010}
}
```

---

## Sources

- [IRIS TPDS 2024 — ACM/IEEE](https://dl.acm.org/doi/10.1109/TPDS.2024.3429010)
- [IRIS ORNL publication page](https://www.ornl.gov/publication/iris-performance-portable-framework-cross-platform-heterogeneous-computing)
- [IRIS Reimagined — ORNL (2024)](https://www.ornl.gov/publication/iris-reimagined-advancements-intelligent-runtime-system-task-based-programming)
- [IRIS Performance Scaling — IPDPS 2024 Workshop](https://www.ornl.gov/publication/iris-exploring-performance-scaling-intelligent-runtime-system-and-its-dynamic)
- [IRIS GitHub repository](https://github.com/ORNL/iris)
- [IRIS Documentation](https://iris-programming.readthedocs.io/en/latest/architecture.html)
- [IRIS 2024 R&D 100 Award](https://impact.ornl.gov/en/prizes/2024-rampd-100-award-for-iris-sdk-intelligent-runtime-system-for-)
- [Q-IRIS: Quantum Extension (arXiv 2512.13931)](https://arxiv.org/html/2512.13931)
