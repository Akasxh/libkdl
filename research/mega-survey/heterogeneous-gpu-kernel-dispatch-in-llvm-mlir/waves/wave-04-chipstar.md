# Wave 04 — chipStar CUDA-on-SPIR-V Portability

**Angle:** chipStar CUDA/HIP applications compiled to SPIR-V and dispatched via Level Zero/OpenCL
**Query:** "chipStar CUDA HIP applications SPIR-V Level Zero OpenCL portability layer"
**Date:** 2026-04-06

---

## Summary

chipStar is the primary production-ready system for running unmodified HIP/CUDA code on non-NVIDIA/non-AMD hardware via SPIR-V. It achieves 0.75x geometric mean performance vs. native AMD HIP across HeCBench (~220 benchmarks), demonstrating that SPIR-V-mediated portability has a measurable but bounded compile-time cost. It is the most direct prior-work comparison for libkdl: where chipStar commits to SPIR-V dispatch at compile time, libkdl selects pre-compiled per-vendor binaries at runtime.

---

## Sources

### Source 1

**Title:** chipStar: Making HIP/CUDA applications cross-vendor portable by building on open standards
**URL:** https://journals.sagepub.com/doi/10.1177/10943420261423001
**Date:** 2026 (published IJHPCA)
**Type:** peer-reviewed paper
**Relevance/Novelty:** 10/10
**Summary:** Primary 2026 IJHPCA paper by Velesko, Jääskeläinen et al. (16 authors, Argonne + Intel + Tampere University). Describes the full chipStar compilation stack from HIP/CUDA source through SPIR-V to OpenCL/Level Zero runtime. Key data: 0.75x geometric mean on HeCBench vs. native AMD HIP; GAMESS-GPU-HF ported to Aurora (Intel Ponte Vecchio) without source changes; v1.1 ~30% faster than v1.0; v1.1 up to 2x faster than v1.0 on some workloads.
**Key detail for libkdl:** Establishes 0.75x as the empirical cost floor for SPIR-V portability when the dispatch method is static compilation + driver JIT. libkdl's architecture targets better-than-0.75x by using pre-compiled vendor binaries that bypass driver JIT quality variance entirely.

---

### Source 2

**Title:** chipStar GitHub Repository (CHIP-SPV/chipStar)
**URL:** https://github.com/CHIP-SPV/chipStar
**Date:** Active, latest tag v1.2.1 (Nov 2024)
**Type:** open-source project / reference implementation
**Relevance/Novelty:** 9/10
**Summary:** Source code and documentation for chipStar. Compilation pipeline: (1) Clang/LLVM 18/19/20 compiles HIP/CUDA to LLVM IR, (2) chipStar bitcode library linked, (3) 14 custom LLVM passes applied, (4) SPIRV-LLVM-Translator converts to SPIR-V binary, (5) offload bundle assembled into fat binary, (6) host code compiled, (7) final object linked. Runtime backend selected by `CHIP_BE` environment variable (opencl or level0), allowing per-invocation selection between backends on the same binary.
**Key detail for libkdl:** The 14 custom LLVM passes document every semantic gap between HIP/CUDA and OpenCL/SPIR-V — these are precisely the transformations that any SPIR-V-targeting dispatch system must handle. The `CHIP_MODULE_CACHE_DIR` mechanism demonstrates that production SPIR-V stacks require kernel binary caching to amortize first-execution JIT overhead.

---

### Source 3

**Title:** ChipStar 1.2 Released For Compiling & Running HIP/CUDA On SPIR-V/OpenCL Hardware
**URL:** https://www.phoronix.com/news/ChipStar-1.2-Released
**Date:** Sep 25, 2024 (v1.2 release)
**Type:** release announcement / blog
**Relevance/Novelty:** 7/10
**Summary:** v1.2 introduced `cucc`, a drop-in replacement for nvcc that enables direct CUDA compilation without nvcc, hipBLAS/hipFFT/rocRAND integration via Intel MKL backends, OpenCL buffer device address extension support, Level Zero memory-leak and thread-safety fixes, and initial RISC-V testing (Starfive Visionfive 2). v1.2.1 added module caching, JIT timing instrumentation, and Level Zero copy queues.
**Key detail for libkdl:** The `cucc` wrapper shows how a portability layer can intercept at the compiler driver level — analogous to how libkdl intercepts at the loader level. Module caching in v1.2.1 is chipStar's acknowledgment that driver JIT latency is a real production problem.

---

### Source 4

**Title:** ChipStar 1.1 Released For Compiling & Running HIP/CUDA On SPIR-V
**URL:** https://www.phoronix.com/news/ChipStar-1.1-HIP-CUDA-SPIR-V
**Date:** Jan 22, 2024
**Type:** release announcement / blog
**Relevance/Novelty:** 6/10
**Summary:** v1.1 added LLVM 17 support, Intel Unified Shared Memory (USM) extension via OpenCL, optimized atomics via OpenCL 3.0, Immediate Command Lists for low-latency Level Zero dispatch, and achieved 30% average improvement on HeCBench with up to 2x gains on some workloads.
**Key detail for libkdl:** Immediate Command Lists in Level Zero are a near-zero-latency kernel submission path — they reduce dispatch overhead by removing command list construction and submission. This is the Level Zero analog to CUDA's stream approach. libkdl's runtime layer can exploit the same path.

---

### Source 5

**Title:** chipStar Development Documentation (CHIP-SPV/chipStar/docs/Development.md)
**URL:** https://github.com/CHIP-SPV/chipStar/blob/main/docs/Development.md
**Date:** Active / continuously updated
**Type:** technical documentation
**Relevance/Novelty:** 8/10
**Summary:** Documents all 14 LLVM transformation passes applied between LLVM IR and SPIR-V output. Passes address: dynamically-sized shared memory (HipDynMem), global device variable access via shadow kernels (HipGlobalVariable), warp-sensitive intrinsics using `cl_intel_reqd_sub_group_size` (HipWarps), texture object lowering to OpenCL image+sampler (HipTextureLowering), printf translation (HipPrintf), zero-length array elimination (HipLowerZeroLengthArrays), freeze instruction removal for SPIRV-LLVM Translator compatibility (HipDefrost), large kernel argument spilling (HipKernelArgSpiller).
**Key detail for libkdl:** The warp-to-subgroup mapping (HipWarps) exposes the fundamental semantic mismatch: CUDA warp width is 32 and fixed, while OpenCL subgroup width is driver-defined and variable. `cl_intel_reqd_sub_group_size` forces width=32, but this is Intel-specific. On non-Intel OpenCL platforms without this extension, warp-sensitive kernels are silently incorrect — a limitation libkdl avoids by dispatching to vendor-native kernels that never cross this gap.

---

### Source 6

**Title:** HIP on Aurora: HIP for Intel GPUs — chipStar: A HIP Implementation for Aurora (ALCF Developer Session)
**URL:** https://www.alcf.anl.gov/sites/default/files/2024-08/HIPonAurora-ALCF-Dev-Session-2024-08-21_0.pdf
**Date:** Aug 21, 2024
**Type:** technical presentation (Argonne/ALCF)
**Relevance/Novelty:** 7/10
**Summary:** Argonne's Brice Videau presents chipStar's role as the HIP implementation for Intel's Aurora supercomputer (Ponte Vecchio GPUs). Covers dual OpenCL/Level Zero backend architecture, known limitations and workarounds in production HPC use, and instructions for compiling HIP applications against chipStar on Aurora/Sunspot. Demonstrates that chipStar is used in production at a top-10 HPC site (Aurora = 1-exaflop Intel machine at Argonne).
**Key detail for libkdl:** Aurora deployment confirms that SPIR-V-based portability is not a research toy — it is production HPC infrastructure. The coexistence of OpenCL and Level Zero backends on the same hardware (Aurora has both) maps directly to the libkdl multi-backend dispatch problem.

---

### Source 7

**Title:** HIPCL: Tool for Porting CUDA Applications to Advanced OpenCL Platforms Through HIP
**URL:** https://dl.acm.org/doi/10.1145/3388333.3388641
**Date:** 2020 (IWOCL/SYCLcon proceedings)
**Type:** academic paper (conference)
**Relevance/Novelty:** 6/10 (historical)
**Summary:** Original HIPCL paper by Babej et al. (Tampere University / Intel). First demonstration of HIP-over-OpenCL using SPIR-V. Established the architecture that chipStar inherits: Clang + SPIRV-LLVM-Translator pipeline, OpenCL runtime as the execution layer. chipStar supersedes HIPCL and extends it to Level Zero.
**Key detail for libkdl:** HIPCL's 2020 results pre-chipStar establish the baseline overhead of SPIR-V mediation. The progression HIPCL (2020) → HIPLZ (2021-2023) → chipStar (2024+) shows ~5 years of incremental improvement to reach 0.75x, suggesting that the SPIR-V-JIT approach has a fundamental overhead floor that pre-compiled dispatch avoids.

---

### Source 8

**Title:** HIPLZ: Enabling Performance Portability for Exascale Systems
**URL:** https://link.springer.com/chapter/10.1007/978-3-031-31209-0_15
**Date:** 2023 (Springer LNCS)
**Type:** academic paper (book chapter)
**Relevance/Novelty:** 6/10 (historical / merged into chipStar)
**Summary:** HIPLZ paper by Zhao et al. (Argonne). Describes the Level Zero backend for HIP that was later merged into chipStar. Key architectural contribution: direct Level Zero API calls for kernel submission bypass the OpenCL abstraction layer, reducing dispatch latency for Intel GPU targets. The merger with HIPCL to form chipStar unified both backends under one frontend.
**Key detail for libkdl:** The performance argument for Level Zero over OpenCL on Intel hardware (lower overhead, better exposing hardware capabilities) is the same argument libkdl makes for per-vendor native dispatch over SPIR-V: the JIT abstraction layer has cost, and direct native paths are faster.

---

### Source 9

**Title:** POSTER: Performance Comparison of GPU Programming Models Using HeCBench Benchmarks (ACM Computing Frontiers 2025)
**URL:** https://dl.acm.org/doi/10.1145/3719276.3727956
**Date:** 2025 (ACM CF proceedings)
**Type:** academic poster paper
**Relevance/Novelty:** 7/10
**Summary:** Compares HIP, CUDA, SYCL, OpenCL, and chipStar/SPIR-V across HeCBench benchmarks on multiple platforms. Assesses both pass rate (functionality) and performance relative to native baselines. Provides cross-model performance data that contextualizes chipStar's 0.75x result in the broader portability landscape.
**Key detail for libkdl:** Cross-model comparison provides the empirical backdrop for claiming libkdl's dispatch overhead is in a different category from SPIR-V-JIT overhead — HeCBench is the standard benchmark, and the 0.75x chipStar number is now a community-known reference point.

---

### Source 10

**Title:** Evaluating CUDA Portability with HIPCL and DPCT
**URL:** https://www.osti.gov/servlets/purl/1838992
**Date:** 2021 (IEEE publication via OSTI)
**Type:** academic paper
**Relevance/Novelty:** 5/10 (historical)
**Summary:** Compares HIPCL and Intel's DPCT (source-translation tool) for porting CUDA applications. Finds HIPCL preserves source code while DPCT requires code modification. Early performance data predates chipStar's improvements but establishes the HIPCL approach as superior for portability-without-modification.
**Key detail for libkdl:** The HIPCL vs. DPCT comparison foreshadows the fundamental trade-off between runtime portability (modify nothing, accept overhead) and ahead-of-time translation (modify code, potentially achieve native performance) — the same trade-off libkdl navigates by selecting pre-compiled variants at load time.

---

## Technical Deep-Dive: chipStar's Dispatch Mechanism

### Compilation-Time vs. Runtime Dispatch

chipStar's dispatch is fundamentally compile-time: a single SPIR-V binary is produced and then JIT-compiled by the driver at first execution. The runtime selection is limited to:
1. Backend choice (`CHIP_BE=opencl` or `CHIP_BE=level0`) — set before execution, not dynamically switched
2. Device selection (`CHIP_PLATFORM`, `CHIP_DEVICE`, `CHIP_DEVICE_TYPE`) — filters available hardware

There is no mechanism in chipStar v1.2.1 for a single binary to dynamically select between OpenCL on machine A and Level Zero on machine B, or between two different kernel implementations based on detected hardware features. This is the exact dispatch gap that libkdl fills.

### The 14 LLVM Passes as a Feature-Gap Map

The 14 chipStar LLVM passes directly enumerate every semantic incompatibility between HIP/CUDA and SPIR-V/OpenCL. Each pass is a workaround for something CUDA assumes that OpenCL does not provide:

| Pass | Gap addressed |
|------|---------------|
| HipWarps | CUDA fixed warp=32 vs. OpenCL variable subgroup width |
| HipDynMem | CUDA `extern __shared__` vs. OpenCL kernel argument model |
| HipGlobalVariable | CUDA `__device__` globals vs. OpenCL no-global-mutable-state |
| HipTextureLowering | CUDA texture objects vs. OpenCL image+sampler model |
| HipPrintf | CUDA printf vs. OpenCL printf (different format) |
| HipLowerSwitch | CUDA allows non-power-of-2 integers; SPIR-V does not |
| HipDefrost | LLVM freeze instructions not representable in SPIR-V |
| HipKernelArgSpiller | CUDA allows large arg lists; OpenCL has per-arg size limits |

For libkdl: the SPIR-V format path in a kernel registry must either apply these same transformations or accept SPIR-V that was already transformed by a chipStar-like pipeline.

### Module Caching: Acknowledging JIT Overhead

v1.2.1 added `CHIP_MODULE_CACHE_DIR` to cache compiled kernel binaries on disk. This is chipStar's production acknowledgment that first-execution JIT latency (100ms-1s+ per program per device) is a real deployment problem. The cache stores the driver's native binary output (Intel GEN ISA, etc.) keyed by SPIR-V hash + device fingerprint.

libkdl avoids this problem entirely: pre-compiled binaries are loaded directly into device memory without any JIT step.

---

## Relevance to "Runtime Kernel Format Translation as Alternative to Pre-compiled Multi-Vendor Binaries"

**Rating: 9/10 (primary prior work, contrasting approach)**

chipStar is the most mature implementation of the "single portable binary + runtime translation" alternative to pre-compiled multi-vendor binaries. Its 0.75x geometric mean result is the best-known quantification of the cost of this approach at scale.

The key contrast for libkdl's positioning:

| Dimension | chipStar (SPIR-V + JIT) | libkdl (pre-compiled + dynamic dispatch) |
|-----------|------------------------|------------------------------------------|
| Binary size | 1x (single SPIR-V) | Nx (one object per vendor target) |
| First-kernel latency | 100ms–1s+ (driver JIT) | Near-zero (load pre-compiled .so) |
| Peak throughput | 0.75x native (driver JIT quality ceiling) | ~1.0x native (vendor-compiled code) |
| Hardware novelty | Runs on any SPIR-V-capable device | Only runs on registered targets |
| ML tensor core use | No (SPIR-V has no tensor core ops) | Yes (vendor kernels can use intrinsics) |
| Dynamic hardware switch | No (build-time backend selection) | Yes (runtime kdl_open on new device) |
| Warp primitive correctness | Intel-only guarantee (cl_intel_reqd_sub_group_size) | Full correctness per vendor |

chipStar proves the SPIR-V approach works. libkdl argues the pre-compiled approach is better when (a) the target hardware set is known, (b) peak performance matters, and (c) ML-specific hardware features (tensor cores) are required.

---

## Risks / Inconsistencies Discovered

1. **SPIR-V native backend vs. SPIRV-LLVM-Translator split:** chipStar documents a planned switch from SPIRV-LLVM-Translator to LLVM's built-in SPIR-V backend "as it becomes stable." As of LLVM 20 (released 2024), the built-in backend exists but chipStar still uses the Translator. This creates a moving target for anyone building on chipStar's compilation path.

2. **Warp correctness on non-Intel OpenCL:** The `cl_intel_reqd_sub_group_size` workaround is Intel-specific. On ARM Mali or PowerVR, warp-sensitive kernels may produce silent incorrect results. The 0.75x benchmark number may be optimistic if it excludes warp-sensitive workloads on non-Intel hardware.

3. **FP64 gap on mobile:** ARM Mali and PowerVR lack native double precision. Scientific applications (GAMESS, libCEED) that use FP64 cannot run on these platforms via chipStar. This restricts the "any SPIR-V platform" claim for HPC workloads.

4. **OpenCL SVM requirement filters out many OpenCL platforms:** The OpenCL backend requires coarse-grained SVM (system virtual memory), which is an OpenCL 2.0 feature not exposed by all OpenCL 1.2 implementations. Platforms like Raspberry Pi GPU (VideoCore, OpenCL 1.2 only) cannot use chipStar's OpenCL path.

5. **Library portability depends on MKL:** The AMD library ports (hipBLAS, hipFFT, rocRAND, etc.) are routed to Intel MKL on Aurora. This means "portable" library calls work only where MKL is available — not on ARM or RISC-V chipStar targets.

---

## Citation Block

```bibtex
@article{velesko2026chipstar,
  author    = {Velesko, Paulius and Jääskeläinen, Pekka and Linjamäki, Henry and
               Babej, Michal and Tu, Peng and Sarkar, Sarbojit and Ashbaugh, Ben and
               Bertoni, Colleen and Chen, Jenny and Roth, Philip C. and
               Elwasif, Wael and Gayatri, Rahulkumar and Zhao, Jisheng and
               Herbst, Karol and Harms, Kevin and Videau, Brice},
  title     = {chipStar: Making HIP/CUDA applications cross-vendor portable by
               building on open standards},
  journal   = {International Journal of High Performance Computing Applications},
  year      = {2026},
  doi       = {10.1177/10943420261423001}
}

@inproceedings{babej2020hipcl,
  author    = {Babej, Michal and others},
  title     = {HIPCL: Tool for Porting CUDA Applications to Advanced OpenCL Platforms Through HIP},
  booktitle = {Proceedings of the International Workshop on OpenCL (IWOCL/SYCLcon)},
  year      = {2020},
  doi       = {10.1145/3388333.3388641}
}

@incollection{zhao2023hiplz,
  author    = {Zhao, Jisheng and others},
  title     = {HIPLZ: Enabling Performance Portability for Exascale Systems},
  booktitle = {Lecture Notes in Computer Science},
  year      = {2023},
  doi       = {10.1007/978-3-031-31209-0_15}
}
```
