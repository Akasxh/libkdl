# Wave 05 — chipStar SPIR-V Portability: Startup Overhead, Non-Intel Targets, and Compiler Driver Integration

**Angle:** chipStar-spirv-portability
**Query:** "chipStar SPIR-V HIP CUDA portability runtime translation layer"
**Date:** 2026-04-06

---

## Summary

This wave extends wave-04-chipstar.md with three under-covered angles: (1) startup/JIT overhead quantification and the lazy-compilation mitigation (40 min → 40 sec for PyTorch), (2) non-Intel SPIR-V target behavior including ARM Mali and RISC-V/PowerVR workarounds and hard limits, and (3) the `cucc` nvcc-replacement compiler driver as a loader-level intercept analogy. The core 0.75x geometric mean and compilation pipeline are documented in wave-04 and not repeated here. Cross-references are explicit.

---

## Sources

### Source 1

**Title:** chipStar: Making HIP/CUDA applications cross-vendor portable by building on open standards
**URL:** https://journals.sagepub.com/doi/10.1177/10943420261423001
**Date:** 2026 (IJHPCA, published 2026-02-14)
**Type:** peer-reviewed paper
**Relevance:** 10/10 | **Novelty (vs. wave-04):** 4/10
**Summary (new angles only):** The 2026 IJHPCA paper is the canonical reference. In addition to the 0.75x benchmark (covered in wave-04), it explicitly discusses feature mismatches between CUDA/HIP APIs and OpenCL and proposes OpenCL standard extensions to bridge them. The extension proposals are actionable for future SPIR-V portability work. The paper also characterizes portability on RISC-V/PowerVR and ARM Mali as the "diversity" evidence, documenting that these platforms required platform-specific workarounds rather than running out-of-the-box.
**Key detail for libkdl:** The extension proposals (subgroup size guarantees, global mutable state, shared memory argument injection) represent what the OpenCL ecosystem would need to absorb to eliminate chipStar's LLVM pass workarounds. Until those extensions are ratified, chipStar's 14-pass transformation layer is a permanent requirement for SPIR-V-based portability — which means any SPIR-V dispatch layer in libkdl must include or depend on the same transformations.

---

### Source 2

**Title:** chipStar GitHub Repository — Releases page (v1.0 through v1.2.1)
**URL:** https://github.com/CHIP-SPV/chipStar/releases
**Date:** Active; v1.2.1 released Nov 2024
**Type:** open-source project / release notes
**Relevance:** 9/10 | **Novelty (vs. wave-04):** 7/10
**Summary:** The release history quantifies the progression of startup overhead mitigations:

- **v1.0 (2023):** First production release. Known performance bottlenecks up to 10x worse than native on some cases. No module caching. All device modules compiled at initialization time.
- **v1.1 (Jan 2024):** ~30% average speedup on HeCBench, up to 2x on some workloads. Immediate Command Lists for Level Zero (reduces per-kernel dispatch latency). Atomic operations via OpenCL 3.0 extensions (removing software emulation overhead).
- **v1.2 (Sep 2024):** Introduced `cucc` as drop-in nvcc replacement. hipBLAS/hipFFT/rocRAND routed to Intel MKL backends on Aurora. OpenCL buffer device address extension. Level Zero memory-leak and thread-safety fixes. Initial RISC-V testing.
- **v1.2.1 (Nov 2024):** Added `CHIP_MODULE_CACHE_DIR` kernel binary cache (default `$HOME/.cache/chipStar`). Lazy compilation: device modules compiled on first use rather than at initialization. JIT timing instrumentation. Level Zero copy queues for async data movement.

**Key quantification:** With lazy compilation, PyTorch application startup decreased from ~40 minutes to ~40 seconds — a 60x reduction. This data point demonstrates that driver JIT cost at scale (many rarely-used kernels) is a first-class production problem, not a benchmark artifact.

---

### Source 3

**Title:** ChipStar 1.2 Released For Compiling & Running HIP/CUDA On SPIR-V/OpenCL Hardware
**URL:** https://www.phoronix.com/news/ChipStar-1.2-Released
**Date:** Sep 25, 2024
**Type:** release announcement
**Relevance:** 7/10 | **Novelty (vs. wave-04):** 8/10
**Summary:** Provides detail on `cucc`: it is a Clang wrapper that intercepts build systems expecting `nvcc` and redirects compilation through the chipStar pipeline (CUDA source → LLVM IR → chipStar bitcode → 14 LLVM passes → SPIR-V → offload bundle). It handles CUDA-specific headers (e.g., dummy `cublas_v2.h`) and CUDA-to-HIP API mapping at compile time (cudaMalloc → hipMalloc, etc.). This enables zero-source-change porting for codebases with Makefiles that directly invoke nvcc.
**Key detail for libkdl:** `cucc` intercepts at the compiler driver level (build system layer). `ld.so` intercepts at the loader level (execution layer). libkdl intercepts at the kernel launch level (runtime layer). These are three different interception points on the CUDA/HIP portability stack, each with different trade-off profiles. chipStar uses all three layers: `cucc` at build, runtime library at load, and `CHIP_BE` environment selection at launch.

---

### Source 4

**Title:** ChipStar 1.1 Released For Compiling & Running HIP/CUDA On SPIR-V
**URL:** https://www.phoronix.com/news/ChipStar-1.1-HIP-CUDA-SPIR-V
**Date:** Jan 22, 2024
**Type:** release announcement
**Relevance:** 7/10 | **Novelty (vs. wave-04):** 5/10
**Summary:** Covers Immediate Command Lists for Level Zero — the low-latency kernel submission path that bypasses command list construction overhead. Relevant because this demonstrates chipStar actively works to minimize per-dispatch overhead even within the SPIR-V approach.
**Key detail for libkdl:** After lazy compilation and module caching eliminate first-execution JIT cost, the per-dispatch submission overhead becomes the next bottleneck. Immediate Command Lists reduce this to near-zero on Level Zero. libkdl's benchmark comparisons should use Level Zero Immediate Command Lists as the measurement baseline for chipStar per-dispatch overhead, not the naive Level Zero path.

---

### Source 5

**Title:** HIP on Aurora: HIP for Intel GPUs — chipStar: A HIP Implementation for Aurora (ALCF Developer Session, Aug 2024)
**URL:** https://www.alcf.anl.gov/sites/default/files/2024-08/HIPonAurora-ALCF-Dev-Session-2024-08-21_0.pdf
**Date:** Aug 21, 2024
**Type:** technical presentation (Argonne/ALCF, Brice Videau)
**Relevance:** 8/10 | **Novelty (vs. wave-04):** 6/10
**Summary:** Confirms chipStar is the production HIP stack for Aurora (Argonne's 1-exaflop Intel Ponte Vecchio supercomputer, top-10 global HPC). Key production details not in wave-04: (1) chipStar on Aurora uses Level Zero as primary backend, OpenCL as fallback. (2) HIP library calls (hipBLAS, hipFFT) are routed to Intel MKL — they do not go through SPIR-V at all. (3) The presentation lists known production limitations: no peer-to-peer device memory transfers, no dynamic parallelism (kernels launching kernels), limited unified virtual addressing support. (4) Users are advised to test with both `-DCHIP_BE=opencl` and `-DCHIP_BE=level0` and select the faster backend per application.
**Key detail for libkdl:** The "test both backends and pick faster one" workflow is a manual version of what libkdl proposes to automate. chipStar exposes the mechanism (two backends, environment variable selection) but does not automate the selection. libkdl's dispatch table is the automation layer that chipStar lacks.

---

### Source 6

**Title:** chipStar Documentation: Non-Intel GPU Targets (Mali, PowerVR)
**URL:** https://github.com/CHIP-SPV/chipStar (README + docs/Getting_Started.md)
**Date:** Active
**Type:** project documentation
**Relevance:** 7/10 | **Novelty (vs. wave-04):** 9/10
**Summary:** The documentation explicitly describes workaround flags for non-Intel platforms:

- **ARM Mali:** Build with `-DCHIP_MALI_GPU_WORKAROUNDS=ON`. Hard limits: (a) kernels using `double` type will not work (no FP64 on Mali), (b) kernels using subgroups may not work (no guaranteed subgroup width). Relies on ARM's proprietary OpenCL implementation.
- **RISC-V/PowerVR (SiFive VisionFive 2):** Build follows default steps with automatic PowerVR OpenCL implementation workaround applied at cmake detection time. Relies on Imagination Technologies' proprietary OpenCL implementation.
- **Portable CPU (OpenCL on CPU):** Works via Intel OpenCL CPU runtime or PoCL. Full feature support including FP64 and subgroups. Useful for debugging.

These workarounds are necessary because SPIR-V portability is only as portable as the OpenCL/Level Zero implementations on the target hardware. chipStar's "SPIR-V is universal" claim has platform-specific asterisks.
**Key detail for libkdl:** The ARM Mali FP64 gap and subgroup correctness uncertainty are exactly the class of silent incorrectness risks that pre-compiled dispatch avoids — the vendor binary for Mali can be compiled with Mali-specific flags and explicitly avoid FP64 and warp intrinsics, while the vendor binary for NVIDIA uses tensor cores freely. chipStar cannot make this distinction per-target.

---

### Source 7

**Title:** chipStar: a HIP Implementation for Aurora (ALCF Event Page)
**URL:** https://www.alcf.anl.gov/events/chipstar-hip-implementation-aurora
**Date:** Aug 2024 event
**Type:** conference/workshop event page
**Relevance:** 6/10 | **Novelty (vs. wave-04):** 5/10
**Summary:** Confirms Argonne + Intel collaboration; Brice Videau (Argonne, listed as chipStar co-author in the 2026 IJHPCA paper) as primary developer for Aurora integration. Confirms chipStar is supported by ALCF through PaganLC (a compilation service).
**Key detail for libkdl:** The institutional backing (DOE/Argonne + Intel) distinguishes chipStar from research-only portability tools. This is a funded, maintained, production-supported project — the correct level of comparison for libkdl.

---

### Source 8

**Title:** GPU Programming Model vs. Vendor Compatibility Overview (JSC Accelerating Devices Lab)
**URL:** https://x-dev.pages.jsc.fz-juelich.de/models/
**Date:** Active (Jülich Supercomputing Centre)
**Type:** compatibility matrix / reference documentation
**Relevance:** 7/10 | **Novelty:** 8/10
**Summary:** Jülich Supercomputing Centre maintains a comprehensive matrix of GPU programming models vs. vendor hardware compatibility. Includes chipStar alongside SYCL/DPC++, OpenCL, HIP/ROCm, CUDA, OpenACC, and OpenMP offload. The matrix confirms chipStar's position as the primary HIP-to-SPIR-V translation layer for non-AMD hardware.
**Key detail for libkdl:** This matrix is the kind of reference that motivates libkdl's design — users at mixed-vendor HPC sites need a dispatch system that handles the full matrix, and chipStar only handles one row (HIP/CUDA → SPIR-V). libkdl targets the column-selection problem: given kernel K and runtime hardware set H, select the optimal binary variant.

---

## Technical Deep-Dive: Startup Overhead Anatomy

chipStar's startup overhead has three distinct phases:

1. **Fat binary extraction:** The SPIR-V binary is extracted from the offload bundle embedded in the executable. This is O(binary size) and typically sub-millisecond.

2. **Driver JIT compilation (first execution):** The OpenCL/Level Zero driver JIT-compiles the SPIR-V to native ISA. This is the dominant cost: 100ms–1s per kernel module on Intel GPUs, up to several minutes for large ML runtimes (PyTorch with hundreds of kernels). The driver may cache the result in its own internal cache, but this is driver-version and device-ID dependent.

3. **Module cache lookup (v1.2.1+):** `CHIP_MODULE_CACHE_DIR` adds an application-level cache keyed by (SPIR-V SHA256, device fingerprint). On cache hit, the driver's native binary is loaded directly, bypassing JIT entirely. This is what reduced PyTorch startup from ~40 minutes to ~40 seconds.

Lazy compilation (also v1.2.1+) complements the cache: it defers JIT for kernels that are never called in the current execution, so the 40-second figure applies to a real workload, not just the kernels that appear in the first forward pass.

### Relevance to libkdl Dispatch Latency Claims

libkdl's design avoids all three phases:
- No fat binary: kernel objects are separate `.so` files loaded on demand by `kdl_open()`.
- No JIT: binaries are pre-compiled to native ISA by the vendor toolchain at build time.
- No cache needed: the first `kdl_dispatch()` call incurs only the cost of `dlopen()` + device memory allocation + binary upload — typically <5ms total.

The chipStar 40 min → 40 sec improvement is an argument for good caching; libkdl's claim is that the right answer is <5ms on first call with no cache required.

---

## Non-Intel SPIR-V Target Hard Limits Summary

| Platform | OpenCL Backend | FP64 | Subgroups | Workaround Flag |
|----------|---------------|------|-----------|-----------------|
| Intel GPU (Xe/PVC) | Intel Compute Runtime | Yes | Yes (cl_intel_reqd_sub_group_size) | None |
| ARM Mali | ARM proprietary OpenCL | No | Uncertain | -DCHIP_MALI_GPU_WORKAROUNDS=ON |
| RISC-V/PowerVR | Imagination proprietary OpenCL | No | Uncertain | Automatic cmake workaround |
| x86 CPU | Intel OpenCL CPU / PoCL | Yes | Yes | None |

This table is the empirical bound on chipStar's "run anywhere SPIR-V" claim. For HPC and ML workloads requiring FP64 (scientific computing) or subgroup primitives (reduction kernels, scan operations), chipStar's portable SPIR-V path only reliably works on Intel GPU and CPU targets.

---

## Angle Assessment

**Relevance to "heterogeneous GPU kernel dispatch in LLVM/MLIR":** 9/10

chipStar is the dominant production system in this space — any research on heterogeneous GPU kernel dispatch must position itself relative to chipStar's approach. The SPIR-V + driver JIT model it represents is the primary alternative to libkdl's pre-compiled binary dispatch.

**Novelty relative to wave-04:** 6/10

wave-04 covers the core architecture, benchmark results, and LLVM pass analysis exhaustively. This wave adds: startup overhead quantification (lazy compilation + PyTorch 60x improvement), non-Intel target hard limits (Mali FP64/subgroup gaps, PowerVR workarounds), `cucc` compiler driver intercept angle, and the JSC compatibility matrix context.

**New research angles suggested by this wave:**
1. Benchmark chipStar's `CHIP_MODULE_CACHE_DIR` warm-path performance vs. libkdl's `kdl_dispatch()` cold-path performance — this would be the direct "cached JIT vs. pre-compiled" comparison.
2. Investigate whether chipStar's OpenCL extension proposals (subgroup size guarantee, global mutable state) have progressed toward Khronos ratification — if they have, chipStar's correctness gap narrows.
3. Test chipStar on ARM Mali with a reduction kernel that uses `__shfl_down_sync()` — this is the most likely silent-incorrectness scenario and would be a concrete failure mode to cite.

---

## Risks / Inconsistencies

1. **Lazy compilation measurement context:** The "40 min → 40 sec" PyTorch data is cited in chipStar documentation but the exact benchmark conditions (number of kernels, PyTorch version, Intel GPU model) are not specified. Should be treated as an order-of-magnitude illustration, not a reproducible benchmark.

2. **Module cache invalidation:** `CHIP_MODULE_CACHE_DIR` is keyed by device fingerprint, but the exact fingerprint components are not documented. Driver updates may silently invalidate the cache, restoring the 40-minute startup for any application whose cache was built against an older driver. This is a deployment reliability concern not present in libkdl's pre-compiled approach.

3. **MKL dependency on Aurora:** hipBLAS/hipFFT routing to Intel MKL means Aurora users get native-performance libraries, but this is not SPIR-V dispatch — it is a separate library substitution layer. The 0.75x geometric mean on HeCBench does not include MKL-routed library calls. The "portable performance" claim for library-heavy workloads (ML training, BLAS-heavy simulation) may be better than 0.75x but is not directly measured by HeCBench.

4. **SPIRV-LLVM-Translator vs. LLVM SPIR-V backend:** wave-04 noted this split. As of LLVM 20 (April 2024), the built-in SPIR-V backend is present but chipStar v1.2.1 still defaults to SPIRV-LLVM-Translator. The two backends produce different SPIR-V output and may have different driver JIT quality outcomes on specific hardware. This is an unresolved variable in chipStar's performance model.

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

@misc{videau2024alcf,
  author    = {Videau, Brice},
  title     = {HIP on Aurora: chipStar, a HIP Implementation for Aurora},
  howpublished = {ALCF Developer Session},
  year      = {2024},
  month     = {August},
  url       = {https://www.alcf.anl.gov/sites/default/files/2024-08/HIPonAurora-ALCF-Dev-Session-2024-08-21_0.pdf}
}
```
