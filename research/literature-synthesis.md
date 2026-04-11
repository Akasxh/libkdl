# Literature Synthesis — Vendor-Agnostic Runtime Dispatch for ML Kernels via MLIR

**Purpose:** Paper-ready synthesis of ALL literature reviewed. Organized by argument, not by system.
**Last updated:** 2026-04-10
**Source base:** 24 literature files (added HetGPU, KernelEvolve, Universal GPU ISA, AdaptiveCpp SSCP)

---

## 1. The Problem Is Real and Well-Documented

### 1.1 Heterogeneous GPU Environments Are the Norm

**Evidence:**
- 9/10 TOP500 systems (Nov 2024) employ co-processors/accelerators [Davis et al., ICS 2025]
- Cloud providers offer mixed GPU fleets: AWS (NVIDIA A100/H100), Azure (AMD MI300X), GCP (TPU + NVIDIA)
- HPC: Frontier (AMD MI250X), Aurora (Intel Ponte Vecchio), Perlmutter (NVIDIA A100), LUMI (AMD MI250X)
- Edge: Qualcomm Adreno, ARM Mali, x86 CPUs, Apple Silicon
- Model serving must handle variable hardware without per-target binary management

**Source files:** `survey.md` §2, `findings.md`

### 1.2 No Lightweight Vendor-Agnostic Dispatch Layer Exists

**The landscape (compile-time vs runtime):**

| System | Multi-target? | Runtime dispatch? | MLIR native? | LOC | Constraint |
|--------|:---:|:---:|:---:|---:|---|
| MLIR gpu dialect | Yes | No (compile-time `select_object`) | Yes | — | Selection is first-object-wins |
| IREE | Yes | Yes (HAL) | Yes | 100K+ | Full-stack buy-in required |
| Triton | Yes (NVIDIA, AMD) | No | Yes | — | Single target at compile time |
| TVM | Yes | Limited | No (own IR) | — | Auto-tuning, not dispatch |
| XLA/PJRT | Yes | Plugin model | Via StableHLO | — | Google ecosystem |
| SYCL/DPC++ | Yes | Yes (device_selector) | No | — | P3 scores 0.46-0.65 |
| AdaptiveCpp | Yes | Yes (JIT) | No | — | SYCL programming model required |
| ALPAKA | Yes | No (compile-time) | No | — | Recompilation per target |
| Kokkos | Yes | No (compile-time) | No | — | P3 0.75-0.99 but recompile |
| IRIS (ORNL) | Yes | Yes | No | — | No cost-model selection |
| chipStar | Yes (via SPIR-V) | JIT per device | No | — | HIP→SPIR-V, 0.75x native |
| Proteus | Yes (CUDA, HIP) | JIT | No | — | JIT specialization, not dispatch |
| HetGPU | Yes (4 vendors) | Yes (IR translate) | No | — | Runtime translation, not AOT selection |
| AdaptiveCpp SSCP | Yes | Yes (JIT) | No | — | Requires SYCL; JIT first-launch cost |
| KernelEvolve (Meta) | Yes (4 targets) | No (design-time) | No | — | Generates variants, doesn't dispatch |
| **libkdl (proposed)** | **Yes** | **Yes (cost-model)** | **Yes** | **~500** | **Our contribution** |

**The gap:** Nobody has built a lightweight (<1000 LOC), MLIR-native runtime dispatch layer that selects among pre-compiled multi-target kernel variants based on hardware capability matching and cost-model ranking.

**Source files:** `findings.md` §"No Lightweight Standalone Solution", `notes/novelty-gaps.md` Gap 3, `competitive-landscape.md`

### 1.3 IREE Acknowledges the Gap

Three open IREE issues document the unsolved nature of this problem:

- **Issue #50** (open since 2019-10-13): "At runtime we should then match against those to select the best suited for the given runtime configuration." — Foundational target configuration issue, unresolved after 6+ years.
- **Issue #12230**: Phase 1 (shape dedup) done; Phase 2 (runtime strategy selection) **stalled since May 2023**. Acknowledged as "sort of broken."
- **Issue #15334**: Multi-versioning epic — ALL tasks remain unchecked as of 2026.

**Source files:** `iree-deep-dive.md` §7, `notes/novelty-gaps.md` Gap 1

---

## 2. "ML Kernels Are Static" Is Empirically False

**Reviewer 91B's claim:** "In ML, the kernels are well known at compile time as well as the size of the tensors."

**Counter-evidence (6 independent sources):**

### 2.1 cuBLAS Runtime Kernel Selection
- Hundreds of GEMM kernels per precision stored in cuBLAS
- ML-trained recommender system selects optimal kernel at runtime → 93% of oracle performance
- `cublasSetSmCountTarget()` allows runtime SM count override
- **Source:** `production-ml-dispatch.md` §3

### 2.2 cuDNN v9 Runtime Fusion
- Three heuristic modes (A/B/fallback) with different accuracy/latency tradeoffs
- Runtime fusion engines that **NVRTC-compile kernels on-the-fly** based on detected compute capability
- **Source:** `production-ml-dispatch.md` §4

### 2.3 PyTorch Dispatcher
- Computes `DispatchKeySet` on **every operator call** from live tensor metadata and thread-local state
- Union of 4 independent inputs: tensor contributions, local include/exclude sets, global set
- Explicitly designed for runtime extensibility to new backends without recompilation
- **Source:** `production-ml-dispatch.md` §1

### 2.4 torch.compile Guards
- Guards checked at every invocation; shape changes trigger recompilation or fallback
- Backend selection is manual (no auto hardware detection)
- **Source:** `production-ml-dispatch.md` §2

### 2.5 Dynamic Shapes in Production
- vLLM: Piecewise CUDA Graphs with runtime batch-size bucket selection
- LLM serving: sequence length varies per request → different optimal kernels
- Heterogeneous GPU serving explicitly unsupported in vLLM
- **Source:** `production-ml-dispatch.md` §5

### 2.6 Where Runtime Dispatch Adds Value
- **Heterogeneous serving clusters:** Helix achieves 3.3x throughput on mixed GPUs [ASPLOS 2025]
- **Cloud portability:** Deploy same model on AWS NVIDIA / Azure AMD / CPU fallback
- **Edge deployment:** ExecuTorch 1.0 — unknown hardware at deployment time
- **Multi-tenant inference:** Mixed GPU generations in shared clusters

**Source files:** `production-ml-dispatch.md`, `findings.md` §"ML Kernels Are Static", `new/helix-2025-mixed-gpu.md`, `new/executorch-edge-dispatch.md`

---

## 3. Taxonomy of Approaches

### 3.1 Compile-Time Portability (Best Performance, No Runtime Flexibility)

**ALPAKA** (Helmholtz/CERN):
- Header-only C++20, Redundant Hierarchical Parallelism (RHP), 5-level abstraction
- Backends: CUDA 12.0+, HIP 6.0+, SYCL/oneAPI, OpenMP, TBB, std::thread, serial
- Performance: >94% of native CUDA on matrix operations
- Production: CMS Run 3 HLT (high-level trigger), Patatrack project
- Limitation: Backend fixed at compile time (CMake flag). Template complexity.
- **Source:** `alpaka-sofie-analysis.md`, `new/cern-cms-alpaka-production.md`

**Kokkos** (Sandia/DOE):
- Polymorphic memory layouts, execution policies
- P3 scores 0.75-0.99 — highest among all portability frameworks
- <5% overhead vs native
- **Source:** `papers-runtime-dispatch.md` A1

**RAJA** (LLNL):
- Lambda-based execution policies
- P3 scores 0.47-1.00; <3% overhead vs native CUDA
- **Source:** `papers-runtime-dispatch.md`

**HIP** (AMD):
- Thin header-swap model — zero runtime indirection on NVIDIA, direct ROCclr on AMD
- hipify-clang: ~90-95% automatic CUDA→HIP translation
- Limitation: Compile-time target. Untranslatable: inline PTX, tensor cores, warp size differences
- **Source:** `hip-rocm.md`

### 3.2 Runtime Portability Layers (Flexibility, Performance Cost)

**SYCL** (Khronos Standard):
- DPC++ (Intel): multi-pass fat binary, SPIR-V + native backends
- AdaptiveCpp: single-pass SSCP + runtime JIT — the most portable SYCL implementation
- Device selection: runtime via `device_selector` with custom scoring
- Performance: P3 0.46-0.65 — "SYCL appears to often perform worse" [Davis ICS 2025]
- On A100: ALPAKA and Kokkos matched native CUDA while SYCL was ~10x slower
- SYCL-MLIR compiler (CGO 2024): up to 4.3x speedup over DPC++ on Intel GPUs
- **Source:** `sycl-ecosystem.md`, `new/sycl-mlir-cgo2024.md`, `new/adaptivecpp-sscp.md`

**SPIR-V as Universal IR:**
- chipStar: HIP/CUDA → SPIR-V → OpenCL/Level Zero. 0.75x geometric mean vs native HIP.
- LLVM RFC: "SPIR-V IR as vendor-agnostic GPU representation" — proposes SPIR-V in LLVM's backend
- IREE SPIR-V backend: vendor-agnostic code given specified hardware features
- Fundamental limitation: SPIR-V lacks vendor-specific features (tensor cores, MFMA)
- **Source:** `spirv-analysis.md`, `new/chipstar-2026-spirv-portability.md`, `new/mlir-gpu-infrastructure-2026.md`

**OpenCL** (lessons learned):
- Source portability ≠ performance portability (15-year cautionary tale)
- No reference implementation → fragmentation. NVIDIA never moved beyond 1.2. Apple deprecated 2018.
- Legacy: SPIR-V, Platform/Device/Context model, ICD dispatch mechanism
- **Source:** `opencl-lessons.md`

**Vulkan Compute:**
- Broadest hardware coverage: NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno, Apple (MoltenVK)
- ~70-80% of CUDA on NVIDIA; matches or exceeds ROCm/HIP on AMD RDNA3
- Cooperative matrices (`VK_KHR_cooperative_matrix`) can exceed CUDA in LLM scenarios
- `vkGetPhysicalDeviceFeatures2` + Vulkan Profiles: adaptive kernel selection
- **Source:** `vulkan-webgpu.md`

### 3.3 Full-Stack ML Compilers

**IREE:**
- End-to-end MLIR compiler+runtime, FlatBuffer modules, bytecode VM, HAL
- Backends: llvm-cpu, vulkan-spirv, metal-spirv, cuda, rocm, vmvx
- Flow → Stream → HAL dialect pipeline for scheduling
- `#stream.affinity` attributes for device placement
- Multi-target compilation support but runtime selection is "sort of broken"
- **Source:** `iree-deep-dive.md`, `new/iree-2026-state.md`

**Triton** (OpenAI):
- Python DSL → MLIR (9 custom dialects) → NVPTX/AMDGCN
- Most MLIR-native ML compiler. Matches cuBLAS on standard shapes.
- CC 8.0+ only. Single-target at compile time. Linux only.
- **Source:** `triton-compiler-approaches.md`

**TVM** (Apache):
- relax::Function + tir::PrimFunc, MetaSchedule auto-tuning
- Broad target support: CUDA, ROCm, OpenCL, Vulkan, WebGPU, Metal
- Own IR stack, limited MLIR integration
- **Source:** `triton-compiler-approaches.md` §6, `papers-ml-compilation.md`

**XLA / OpenXLA:**
- StableHLO as portable IR between frameworks and compilers
- PJRT: uniform device runtime API with plugin model
- SPMD partitioning for multi-device. Shardy (successor to GSPMD).
- Target-agnostic compilation, but device selection at PJRT plugin level
- **Source:** `new/openxla-pjrt-2026.md`

### 3.4 Heterogeneous Runtime Systems (Closest Prior Art)

**IRIS** (ORNL, IEEE TPDS 2024):
- Wraps CUDA/HIP/Level Zero/OpenCL in unified task-based runtime
- Platform-independent task scheduling
- No cost-model-driven kernel selection
- **Source:** `new/iris-2024-task-dispatch.md`

**Helix** (ASPLOS 2025):
- Mixed GPU cluster LLM serving via max-flow optimization
- 3.3x throughput, 66%/24% latency reduction (prompt/decode)
- MILP-based model placement + per-request pipeline scheduling
- Node-level dispatch (whole layers), not kernel-level
- **Source:** `new/helix-2025-mixed-gpu.md`

**Proteus** (CGO 2025):
- Portable JIT for GPU kernels via LLVM IR embedding
- Runtime constant folding → up to 2.8x AMD, 1.78x NVIDIA
- JIT specialization, not cross-vendor dispatch
- **Source:** `papers-jit-gpu.md` §1

---

## 4. Key Technical Insights for Our Design

### 4.1 MLIR's gpu-module-to-binary CAN Produce Multi-Target Binaries
- The `gpu-module-to-binary` pass accepts multiple target attributes → produces `gpu.binary` with objects per target
- `gpu.select_object` selects which object to embed → but it's **compile-time only** (picks first object)
- `GPUOffloadingLLVMTranslationAttrInterface` is the extensibility point for a runtime-aware handler
- **No upstream mechanism does runtime hardware detection to choose among variants**
- **Source:** `mlir-jit-analysis.md`, `notes/novelty-gaps.md` Gap 1

### 4.2 Cost-Model-Driven Dispatch Is Proven Effective
- cuBLAS ML recommender: 93% of oracle performance at runtime
- Roofline model enables per-device ranking: compute-bound → pick highest FLOPS, memory-bound → pick highest BW
- Ridge point varies dramatically: H100 ~20 FLOP/byte, A100 ~9.7, MI300X ~30.8
- Dispatch overhead must be <10ns to be negligible vs 5-20us kernel launch
- **Source:** `new/cost-models-kernel-dispatch.md`, `production-ml-dispatch.md`

### 4.3 The "ld.so for GPUs" Analogy Is Sound
- Linux dynamic linker: discovers available shared libraries, resolves symbols, caches bindings
- CUDA fat binaries: driver selects best SASS, JIT-compiles PTX fallback → per-arch dispatch within one vendor
- ONNX Runtime: execution provider priority selection across vendors at graph level
- PyTorch dispatcher: per-operator runtime dispatch with backend fallback chains
- **Our contribution:** extend this pattern to MLIR-compiled kernel variants across vendors

### 4.4 Native Performance + Runtime Selection Is Achievable
- ALPAKA proves >94% native performance is achievable with proper abstraction
- Kokkos P3 0.75-0.99 shows the performance ceiling for portable code
- Our approach: pre-compile native binaries for each target (nvptx, amdgcn, x86), select at runtime
- No SPIR-V performance tax on the hot path; SPIR-V only as a portable fallback
- Dispatch overhead target: <10ns → amortized to zero over 5-20us kernel launch

### 4.5 PJRT Is the Closest Production Analog
- PJRT (Portable JAX Runtime) defines a plugin interface for device backends
- Device-specific code packaged as plugins; framework calls through uniform API
- Our layer sits BELOW PJRT — at the MLIR compilation output level, not the framework level
- Complementary: PJRT selects backend; our layer selects kernel variant within a backend
- **Source:** `new/openxla-pjrt-2026.md`

---

## 5. Open Research Questions

1. **Granularity of dispatch:** Individual kernels vs fused subgraphs vs whole model regions?
   - IREE dispatches fused regions. ONNX RT dispatches subgraphs. cuBLAS dispatches individual kernels.
   - Different granularities have different overhead/benefit ratios.

2. **Cost model accuracy vs complexity:** Simple device matching → roofline → learned model?
   - cuBLAS shows learned models can reach 93% optimal. Is a simple roofline sufficient for our use case?

3. **Data movement awareness:** Dispatch must account for data residency and transfer costs.
   - Dispatching to a faster GPU may lose if data transfer dominates kernel compute time.
   - PCIe 4.0: ~32 GB/s bidirectional. NVLink 4.0: 900 GB/s. This matters.

4. **Integration with existing frameworks:** torch.compile backend? ONNX RT EP? Standalone?
   - PyTorch dispatcher architecture is explicitly designed for backend extensibility.
   - ONNX RT's EP model already handles multi-vendor dispatch at graph level.

5. **Caching and warmup:** How to minimize first-invocation overhead?
   - chipStar: ~15% first launch overhead from JIT, near-zero cached
   - AdaptiveCpp SSCP: similar pattern — JIT at first use, cache for subsequent calls
   - Our approach: pre-compiled binaries (no JIT), only routing table construction at first call

---

## 6. Positioning Statement

**What we are NOT doing:**
- Building a new compiler (MLIR already compiles to multi-target)
- Building a new runtime (vendor runtimes already execute kernels)
- Competing with IREE (our layer is complementary, not a replacement)
- Proposing SPIR-V as the universal solution (SPIR-V is a fallback, not the primary path)

**What we ARE doing:**
- Building the **thin glue layer** between MLIR's multi-target compilation output and vendor runtimes
- Providing **hardware introspection** (unified capability model across CUDA/HIP/Vulkan/CPU)
- Implementing **cost-model-driven kernel routing** (roofline-based, with extensibility to learned models)
- Demonstrating that **<500 LOC** is sufficient for useful vendor-agnostic dispatch
- Filling a gap that **IREE has documented for 6+ years** but approaches with 100K+ LOC

---

## 7. Comparative Positioning Table (for Paper §3)

| Criterion | libkdl | IREE | SYCL | ALPAKA | chipStar | IRIS | Proteus |
|-----------|:------:|:----:|:----:|:------:|:--------:|:----:|:-------:|
| MLIR-native | **Yes** | Yes | No | No | No | No | No |
| Runtime dispatch | **Yes** | Yes | Yes | No | JIT | Yes | JIT |
| Cost-model selection | **Yes** | Planned | No | N/A | No | No | No |
| Lightweight (<1K LOC) | **Yes** | No (100K+) | No | ~header | No | No | ~1K |
| Native perf target | **>94%** | >90% | 46-65% P3 | >94% | 75% | varies | native+JIT |
| Requires ecosystem buy-in | **No** | Yes | Yes | Yes | Yes | No | No |
| Vendor coverage | **NVIDIA+AMD+CPU** | 6 backends | Intel-best | 8 backends | SPIR-V | 4 backends | CUDA+HIP |

---

## 8. References Quick List

### Primary References (for BibTeX)

```
@inproceedings{lattner2021mlir,
  title={MLIR: Scaling Compiler Infrastructure for Domain Specific Computation},
  author={Lattner, Chris and others},
  booktitle={CGO},
  year={2021}
}

@inproceedings{davis2025perf-portability,
  title={Taking GPU Programming Models to Task for Performance Portability},
  author={Davis, Joshua H. and others},
  booktitle={ICS '25},
  year={2025}
}

@inproceedings{alpay2023adaptivecpp,
  title={One Pass to Bind Them -- First Single-Pass SYCL Compiler with Unified Code Representation},
  author={Alpay, Aksel and Heuveline, Vincent},
  booktitle={IWOCL '23},
  year={2023}
}

@article{zenker2016alpaka,
  title={Alpaka -- An Abstraction Library for Parallel Kernel Acceleration},
  author={Zenker, Erik and others},
  journal={arXiv:1602.08477},
  year={2016}
}

@article{trott2022kokkos,
  title={Kokkos 3: Programming Model Extensions for the Exascale Era},
  author={Trott, Christian R. and others},
  journal={IEEE TPDS},
  year={2022}
}

@inproceedings{ivanov2024retargeting,
  title={Retargeting and Respecializing GPU Workloads for Performance Portability},
  author={Ivanov, Ivan R. and others},
  booktitle={CGO '24},
  year={2024}
}

@inproceedings{georgakoudis2025proteus,
  title={Proteus: Portable Runtime Optimization of GPU Kernel Execution with JIT Compilation},
  author={Georgakoudis, Giorgis and others},
  booktitle={CGO '25},
  year={2025}
}

@inproceedings{pennycook2016metric,
  title={A Metric for Performance Portability},
  author={Pennycook, S. J. and Sewall, J. D. and Lee, V. W.},
  booktitle={PMBS, SC '16},
  year={2016}
}

@inproceedings{chen2018tvm,
  title={TVM: An Automated End-to-End Optimizing Compiler for Deep Learning},
  author={Chen, Tianqi and others},
  booktitle={OSDI '18},
  year={2018}
}

@article{velesko2026chipstar,
  title={chipStar: Making HIP/CUDA Applications Cross-Vendor Portable via Open Standards},
  author={Velesko, P. and others},
  journal={IJHPCA},
  year={2026}
}

@inproceedings{helix2025asplos,
  title={Helix: Serving Large Language Models over Heterogeneous GPUs and Networks via Max-Flow},
  author={CMU Parallel Data Laboratory},
  booktitle={ASPLOS '25},
  year={2025}
}
```
