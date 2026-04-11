# chipStar: Making HIP/CUDA Applications Cross-Vendor Portable via Open Standards
## Literature Review — LLVM Dublin 2026 Poster

**Citation:** Velesko, P., Jääskeläinen, P., Linjamäki, H., Babej, M., Tu, P., Sarkar, S., Ashbaugh, B., Bertoni, C., Chen, J., Roth, P.C., Elwasif, W., Gayatri, R., Zhao, J., Herbst, K., Harms, K., Videau, B. (2026). "chipStar: Making HIP/CUDA applications cross-vendor portable by building on open standards." *International Journal of High Performance Computing Applications (IJHPCA)*, DOI: 10.1177/10943420261423001.

**Date reviewed:** 2026-04-06
**Paper URL:** https://journals.sagepub.com/doi/10.1177/10943420261423001
**GitHub:** https://github.com/CHIP-SPV/chipStar

---

## Relevance Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Technical relevance | 9/10 | Directly solves the vendor portability problem via SPIR-V; closest compile-time analog to libkdl's runtime approach |
| Approach overlap | 7/10 | Shares the SPIR-V abstraction layer strategy; differs fundamentally in when dispatch decisions are made |
| Citation priority | 10/10 | 2026 IJHPCA publication, peer-reviewed, directly comparable; must cite as primary prior work |

---

## Problem

HIP and CUDA programs are binary-incompatible across GPU vendors. The dominant deployment path — compile once for AMD with HIP, recompile for NVIDIA with CUDA, separately for Intel with SYCL — requires maintaining multiple toolchain configurations and build systems. Platforms like Argonne's Aurora (Intel Ponte Vecchio) receive no first-class HIP support from AMD's ROCm stack, forcing scientific software teams (GAMESS, libCEED, HeCBench) to either maintain separate codebases or forego non-AMD/non-NVIDIA targets entirely.

The core question chipStar answers: can a HIP/CUDA source program be compiled once, producing a SPIR-V binary, that runs correctly on any platform with a SPIR-V-capable driver (OpenCL 2.0+ or Level Zero)?

---

## Contribution

chipStar is an open-source compilation stack — born from merging the HIPCL (HIP over OpenCL) and HIPLZ (HIP over Level Zero) projects — that translates HIP and CUDA source code to SPIR-V and executes on any conformant OpenCL or Level Zero runtime. The primary contributions are:

1. **Unified HIP-to-SPIR-V compilation pipeline** using Clang + SPIRV-LLVM-Translator (versions 18/19/20), with chipStar-patched LLVM branches for fixes not yet upstreamed.
2. **Dual backend dispatch**: the compiled SPIR-V binary is submitted to either an OpenCL 2.0+ backend (requiring coarse-grained SVM and `cl_khr_spir` extension) or a Level Zero backend (Intel Compute Runtime / oneAPI).
3. **Broad platform reach**: Intel GPUs (Aurora/Ponte Vecchio, Arc, Xe), ARM Mali, RISC-V + PowerVR, AMD CPUs (via OpenCL), with experimental RISC-V hardware testing in v1.2.
4. **HPC application validation**: GAMESS-GPU-HF (quantum chemistry), libCEED (finite element), HeCBench (heterogeneous compute benchmark suite) all execute correctly with competitive performance.
5. **Library ecosystem**: hipBLAS, hipFFT, rocRAND, rocPRIM, hipCUB, rocThrust, rocSPARSE all supported via MKL-based backends on Intel hardware.

---

## Methodology

### Compilation Pipeline (detailed)

```
HIP/CUDA source (.hip / .cu)
    │
    ▼
Clang (chipStar-patched LLVM 18/19/20)
    │  - HIP device builtins remapped to abstract operations
    │  - CUDA __global__ / __device__ functions lowered to LLVM IR
    │
    ▼
LLVM IR (device bitcode)
    │
    ▼
SPIRV-LLVM-Translator
    │  - Converts device LLVM IR to SPIR-V binary
    │  - Future: switch to LLVM's native SPIR-V backend as it stabilizes
    │
    ▼
SPIR-V binary
    │
    ├──▶ OpenCL backend (clCreateProgramWithIL, JIT per device)
    │       Requires: OpenCL 2.0+, coarse-grained SVM, generic address space
    │
    └──▶ Level Zero backend (zeModuleCreate, JIT per device)
            Requires: Intel Compute Runtime, oneAPI Level Zero Loader
```

The host-side HIP runtime APIs (`hipLaunchKernel`, `hipMalloc`, `hipMemcpy`, etc.) are implemented by chipStar's runtime library, which intercepts calls and routes to whichever backend is active. This is structurally analogous to how CUDA's runtime translates host API calls to PTX kernel launches — chipStar occupies the same layer but targets SPIR-V + OpenCL/L0 rather than CUDA's proprietary stack.

### Evaluation Methodology

- Benchmark suite: HeCBench (heterogeneous compute benchmark, ~220 kernels), GAMESS-GPU-HF, libCEED
- Reference baseline: native AMD HIP on ROCm (AMD RX 6000 series or equivalent)
- Platforms evaluated: Intel GPUs (Ponte Vecchio/Aurora), ARM Mali, RISC-V + PowerVR, AMD CPUs via OpenCL
- Metric: normalized execution time relative to native AMD HIP; reported as geometric mean across benchmarks

---

## Results

**Primary performance headline:** Geometric mean 0.75x relative to native AMD HIP across HeCBench benchmarks. This means chipStar on non-AMD hardware achieves approximately 75% of the performance of the AMD native baseline — a 25% overhead for full cross-vendor portability without source changes.

**Version-to-version improvement:** chipStar v1.1 is approximately 2x faster than v1.0 on some workloads; average improvement over v1.0 is ~30%. This confirms the stack is actively maturing.

**Platform-specific observations:**
- Intel GPU (Level Zero backend): competitive; close to native OneAPI DPC++ performance for memory-bandwidth-bound workloads
- ARM Mali: limited by missing native FP64 support; double-precision workloads require emulation or are unsupported
- RISC-V + PowerVR: experimental in v1.2; portability demonstrated, performance not yet characterized
- AMD CPUs via OpenCL: functional but not the primary target

**Application validation:** GAMESS-GPU-HF (a production HPC quantum chemistry application) runs correctly on Aurora (Intel Ponte Vecchio) via chipStar, which the authors describe as demonstrating that chipStar is "mature enough for wider testing and even production use."

**Key quote:** "A comparison against the original AMD HIP platform provides a geometric mean of 0.75, a reasonable price to pay for the enhanced portability."

---

## Architecture Details

### Runtime Layer
chipStar implements the full HIP runtime API surface in terms of OpenCL or Level Zero:
- `hipMalloc` → `clCreateBuffer` (OpenCL) or `zeMemAllocDevice` (L0)
- `hipMemcpy` → `clEnqueueCopyBuffer` or `zeCommandListAppendMemoryCopy`
- `hipLaunchKernel` → `clEnqueueNDRangeKernel` or `zeCommandListAppendLaunchKernel`
- `hipEventRecord/Sync` → profiling queue events
- HIP streams → OpenCL command queues / Level Zero command lists

### SPIR-V as the Portability Layer
SPIR-V serves as a vendor-neutral binary IR accepted by all conformant OpenCL 2.1+ and Level Zero drivers. Drivers then JIT-compile SPIR-V to device-native ISA (GEN12 assembly for Intel, AMDGCN for AMD OpenCL, native scalar ISA for RISC-V targets). This mirrors what a JVM bytecode does for software: the SPIR-V is the portable "bytecode," and the driver is the per-platform JIT.

### chipStar-Patched LLVM
The project maintains its own LLVM branches with patches not yet accepted upstream. This is both a maintenance burden (rebasing against new LLVM versions) and a correctness requirement (upstream LLVM lacks some HIP-specific transformations needed for correct SPIR-V output). The SPIRV-LLVM-Translator project (separate from LLVM's built-in SPIR-V backend) is the current translation tool; the paper notes a planned future switch to LLVM's native SPIR-V backend as it stabilizes.

---

## Limitations

1. **JIT latency at kernel first-use:** SPIR-V compilation by the driver is deferred to runtime. First kernel execution can incur 100ms–1s+ JIT delay depending on driver and kernel complexity. Kernel caching mitigates repeat invocations.
2. **FP64 gaps on mobile/embedded hardware:** ARM Mali and PowerVR GPUs lack native double-precision; FP64 workloads require emulation or are unsupported. This restricts scientific computing use on these platforms.
3. **Subgroup (warp) primitive limitations:** Warp-level intrinsics (`__shfl`, `__ballot`, etc.) have incomplete or inconsistent support across OpenCL implementations. OpenCL subgroup extensions are optional and driver-dependent.
4. **Compile-time backend selection:** While chipStar enables portability across vendors, the backend (OpenCL vs. Level Zero) must be configured at build time. There is no runtime selection between backends based on available hardware — a single binary cannot dynamically route to OpenCL on one machine and Level Zero on another.
5. **LLVM maintenance overhead:** The chipStar-patched LLVM forks must track upstream LLVM releases. Each major LLVM version upgrade requires re-applying patches, creating a continuous maintenance tax.
6. **No ML-specific kernel optimization:** chipStar makes no attempt to optimize for tensor/matrix operations. A GEMM kernel compiled through chipStar will not leverage hardware tensor cores (NVIDIA Tensor Cores, AMD Matrix Cores, Intel AMX) unless the original HIP source explicitly used those intrinsics — which then become non-portable.
7. **OpenCL SVM requirement:** The OpenCL backend requires coarse-grained SVM, which not all OpenCL implementations expose. Platforms without SVM cannot use the OpenCL path.

---

## Connection to Our Work (libkdl)

chipStar and libkdl solve adjacent but distinct problems, making chipStar the primary prior-art reference for the compiler layer while libkdl addresses the runtime dispatch layer.

**Where they overlap:**
- Both use SPIR-V as a cross-vendor portable binary representation
- Both target OpenCL and/or Level Zero as backend runtimes
- Both aim to run HIP/CUDA-origin code on non-NVIDIA/non-AMD hardware

**Where libkdl diverges:**
| Property | chipStar | libkdl |
|----------|----------|--------|
| Dispatch granularity | Whole-program compilation | Per-kernel dynamic selection |
| Runtime hardware selection | No (build-time backend choice) | Yes (runtime detection + dispatch) |
| ML kernel optimization | None (passes through source logic) | Yes (kernel registry with tuned variants per vendor) |
| Binary model | SPIR-V binary, driver JITs once | Multiple pre-compiled vendor objects, kdl selects at load time |
| Live hardware switching | No | Yes (reload different .kdl module for new device) |

**Key insight for poster:** chipStar demonstrates that 0.75x geometric mean is the cost of SPIR-V-mediated portability via static compilation. libkdl's dynamic dispatch with pre-tuned per-vendor kernels avoids this overhead by maintaining optimized implementations per target rather than relying on vendor driver JIT quality. The 25% chipStar overhead is precisely the gap that libkdl's approach closes — at the cost of a larger binary (one kernel object per target) versus a single SPIR-V object.

**Key quote for poster use:** "A comparison against the original AMD HIP platform provides a geometric mean of 0.75, a reasonable price to pay for the enhanced portability." — use as motivation for why libkdl's architecture targets better than 0.75x.

---

## Related Work Connections

- **HIPCL (2020):** chipStar's predecessor; first demonstration of HIP-over-OpenCL. chipStar supersedes it.
- **HIPLZ (2021):** HIP-over-Level-Zero; merged into chipStar.
- **DPCT (Intel):** Source-translation tool (CUDA → SYCL/DPC++) rather than a runtime portability layer. Requires code modification; chipStar does not.
- **SYCL/AdaptiveCpp:** Higher-level abstraction (C++ standard). chipStar targets HIP/CUDA source directly without requiring a rewrite.
- **libkdl (our work):** Where chipStar operates at compile time (SPIR-V translation), libkdl operates at runtime (kernel object selection). They are complementary: chipStar could feed the SPIR-V path in libkdl's registry while libkdl provides the runtime dispatch logic chipStar lacks.

---

## Citation

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
```

---

## Sources

- [chipStar IJHPCA 2026 paper](https://journals.sagepub.com/doi/10.1177/10943420261423001)
- [chipStar GitHub repository](https://github.com/CHIP-SPV/chipStar)
- [ChipStar 1.2 Release — Phoronix](https://www.phoronix.com/news/ChipStar-1.2-Released)
- [ALCF HIP on Aurora Developer Session (August 2024)](https://www.alcf.anl.gov/sites/default/files/2024-08/HIPonAurora-ALCF-Dev-Session-2024-08-21_0.pdf)
- [chipStar — CASS Software Catalog](https://cass.community/software/chipStar.html)
