# Wave 07: Proteus Deep Dive — LLNL CGO 2025 Runtime Specialization
## Heterogeneous GPU Kernel Dispatch in LLVM/MLIR

**Date:** 2026-04-06
**Focus:** Proteus (CGO 2025, doi:10.1145/3696443.3708939) — closest academic competitor to libkdl
**Sources:** 12 primary sources (ACM DL abstract, GitHub repo, LLNL newsroom, HPCwire, CGO proceedings, SC25 P3HPC workshop, Zenodo artifact, Olympus-HPC org page, RAJAPerf benchmark branch, RAJA feature branch, proteus-benchmarks repo)

---

## 1. Paper Identification

**Title:** Proteus: Portable Runtime Optimization of GPU Kernel Execution with Just-in-Time Compilation
**Authors:** Giorgis Georgakoudis, Konstantinos Parasyris, David Beckingsale — all Lawrence Livermore National Laboratory (LLNL), Center for Applied Scientific Computing (CASC)
**Venue:** 23rd ACM/IEEE International Symposium on Code Generation and Optimization (CGO 2025)
**Presented:** Tuesday, March 4, 2025, 14:40–15:00 PT, Casuarina Ballroom (Level 2), "GPU & Parallelism" session
**Session Chair:** Bastian Hagedorn
**DOI:** https://doi.org/10.1145/3696443.3708939
**Artifact:** Zenodo 10.5281/zenodo.14087064 (tag v1.0.0-cgo25), Apache 2.0 license
**Repository:** https://github.com/Olympus-HPC/proteus (338 commits, 46 stars, 7 forks as of 2026-04)
**Latest release:** v2026.03.0 (March 2026) — active development

**Companion paper (SC25 P3HPC workshop):**
Bowen J., Parasyris K., Beckingsale D., Ben-Nun T., Stitt T., Georgakoudis G.
"Extending RAJA Parallel Programming Abstractions with Just-In-Time Optimization"
SC '25 Workshops, P3HPC session — doi:10.1145/3731599.3767492

---

## 2. Core Problem Proteus Solves

AOT compilation optimizes GPU kernels using only information statically available at compile time. Many GPU kernel parameters — loop bounds, array sizes, launch dimensions, algorithmic flags — are only known at runtime. Without these values, the compiler cannot:

- Unroll loops (unknown trip count)
- Eliminate dead branches (unknown flags)
- Constant-fold arithmetic expressions (unknown multipliers)
- Specialize register allocation (unknown launch bounds)

Proteus's thesis: if you expose runtime values to the compiler at the moment of kernel launch, you recover optimization opportunities that AOT permanently forecloses. The gap between AOT-compiled and hand-specialized HPC GPU kernels is often 2–3x, and Proteus narrows it automatically.

---

## 3. Architecture: Two-Phase Pipeline

### Phase 1 — AOT Extraction (compile-time LLVM pass)

When an application is compiled with Clang + Proteus, the **ProteusPass** LLVM IR plugin runs as a compiler pass. It:

1. Identifies annotated functions/kernels (via `__attribute__((annotate("jit", arg_idx_1, arg_idx_2, ...)))`)
2. Extracts the LLVM IR of those functions
3. Records which IR values correspond to which function arguments (the "specialization variables")
4. Embeds the extracted LLVM IR into the application binary alongside the AOT-compiled binary
5. Inserts a runtime call wrapper that routes annotated kernel invocations through the Proteus runtime instead of directly to the GPU driver

The embedded LLVM IR is the "portable substrate" — vendor-neutral, architecture-neutral, and optimizable. This is what makes Proteus portable across CUDA/HIP backends: the same LLVM IR is JIT-compiled by the appropriate backend at runtime.

### Phase 2 — JIT Specialization (runtime)

At kernel launch time, the Proteus runtime:

1. **Intercepts** the kernel launch call (via the wrapper inserted by ProteusPass)
2. **Reads** the actual runtime values of the annotated arguments
3. **Checks the cache**: if a specialization for this exact set of values exists (in-memory hash table, then persistent disk cache), load and launch it immediately
4. **On cache miss**: clone the stored LLVM IR, inject the runtime values as IR constants (`ConstantInt`, `ConstantFP`, etc.), run LLVM optimization passes (inline, loop unroll, SROA, GVN, etc.), and call the appropriate backend (NVPTX for CUDA, AMDGPU for HIP) to produce a device binary
5. **Cache** the new specialization (in-memory first, flushed to disk)
6. **Launch** the specialized binary via the vendor driver API

### Two-Level Caching Strategy

- **In-memory cache:** hash(kernel_name || runtime_values) → loaded GPU module handle. O(1) lookup, in-process lifetime
- **Persistent disk cache:** binary files per specialization keyed by the same hash. Survives process restarts; eliminates JIT on second run of the application with the same inputs
- **Cache invalidation:** application must clear disk cache when source code changes (no automatic invalidation)

### Background JIT

The JIT compilation (step 4 above) runs on a **background thread** to minimize wall-clock impact. The first invocation of an uncached kernel runs the AOT-compiled version while the JIT compiles in parallel. Subsequent invocations switch to the specialized binary once available. This hides compilation latency in iterative HPC workloads where kernels execute hundreds to thousands of times per run.

---

## 4. Three Integration APIs

### 4.1 Code Annotation Interface (requires Clang)

```cpp
__attribute__((annotate("jit", 1, 2)))  // specialize on args 1 and 2
__global__
void daxpy(double A, size_t N, double *X, double *Y) {
    // N gets constant-folded → loop unrolling possible
    // A gets constant-folded → eliminates multiply in hot path
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += ...)
        Y[i] = A * X[i] + Y[i];
}
```

ProteusPass activates only when compiled with Clang. The annotation index `1, 2` refers to positional argument indices (0-based) to specialize.

### 4.2 C++ Frontend API

Accepts C++ source code as strings, embedding runtime values via `std::format` or similar. No AOT compiler plugin required; the source string is compiled entirely at JIT time. Higher flexibility, higher JIT compilation cost (full compilation, not just constant substitution into existing IR).

### 4.3 DSL API

High-level C++ constructs build kernels programmatically. No string manipulation. Backend-agnostic. Does not require Clang. Intended for users who want to generate kernels algorithmically rather than annotate existing kernels.

---

## 5. Runtime Constant Folding: What Gets Folded

The key compiler optimization enabled by constant folding:

| Runtime Value Folded | Classical Optimizer Exploit |
|---------------------|----------------------------|
| Loop trip count (array size N) | Full loop unrolling, vectorization |
| Launch bounds (blockDim.x) | Register allocation, warp specialization |
| Algorithmic flags (bool) | Dead branch elimination, specialization |
| Scaling factors (float A) | Constant-fold multiplications |
| Stride parameters | Memory access coalescing hints |
| Launch dimensions (gridDim.x) | Reduction specialization |

**Dimension specialization** (introduced in SC25 RAJA extension): replaces runtime queries for launch configuration (`blockDim.x`, `gridDim.x`) with constants. This enables the compiler to unroll the innermost loop of reduction kernels, producing 23x speedup on EDGE3D (AMD MI250X) — the highest reported in any Proteus evaluation.

---

## 6. Performance Numbers (Complete)

### CGO 2025 paper (Georgakoudis et al.):

| Metric | AMD | NVIDIA |
|--------|-----|--------|
| Peak end-to-end speedup vs AOT | **2.8x** | **1.78x** |
| Average speedup vs Jitify (CUDA-specific) | — | **1.23x better** |
| Test systems | Tioga (MI250X) | Lassen (V100/A100) |
| Benchmark domains | ML, weather, hydrodynamics (HECBench + LBANN + RAJAPerf) | Same |

"Up to 2.8×" means the peak across benchmarks, not the geomean. The paper reports that Proteus "recuperates" the JIT overhead by end of run — end-to-end includes first-run JIT compilation cost.

### SC25 P3HPC workshop (Bowen et al.) — RAJA integration:

| Kernel | AMD MI250X | NVIDIA V100 |
|--------|-----------|-------------|
| EDGE3D | **23x speedup** | — |
| REDUCE_STRUCT | — | **15x speedup** |
| Range (geomean) | **1.2x – 23x** | **1.1x – 15x** |
| Regressions | Zero | Zero |

The 23x and 15x are outlier cases where dimension specialization completely unrolls a loop that dominates kernel runtime. The geomean is not reported in accessible text, but the stated range implies it is substantially lower (likely 1.5x–3x based on typical JIT specialization results).

### Jitify comparison:

Proteus outperforms NVIDIA's own Jitify with 1.23x average speedup. The mechanism: Proteus operates on LLVM IR (lower-level, more optimization-amenable) while Jitify operates on CUDA C++ source strings (must go through full parsing, preprocessing, and compilation from scratch each time). Proteus's partial compilation (substituting constants into pre-extracted IR) is faster and generates better code.

---

## 7. Supported Backends and Hardware

| Backend | Runtime | Tested Hardware | LLVM Versions (CI) |
|---------|---------|-----------------|---------------------|
| CUDA | NVRTC | V100, A100, H100 (sm_90 documented) | 19, 20, 22 |
| HIP/ROCm | HIPRTC | MI250X | 6.4.3, 7.1.1, 7.2.0 |
| Host C/C++ | LLVM ORC JIT | x86 CPU | Same |

Install via Spack: `spack install proteus +cuda cuda_arch=90`
CMake integration: `find_package(proteus CONFIG REQUIRED)` + `add_proteus(<target>)`

**Key build constraint:** ProteusPass (annotation interface) requires Clang as the AOT compiler. The DSL and C++ frontend APIs work with any compiler that supports linking against the proteus runtime library.

---

## 8. RAJA Integration (SC25 P3HPC)

**Repository:** https://github.com/Olympus-HPC/RAJA/tree/feature/proteus
**Benchmark repro:** https://github.com/Olympus-HPC/proteus-benchmarks/tree/p3hpc-sc25-paper-repro (includes HECBench, LBANN, RAJAPerf benchmark configurations)

The SC25 paper introduces "indirect kernel launching" — Proteus wraps RAJA's kernel launch mechanism so that RAJA's lambda-based kernels (which hide the underlying `__global__` function from the annotation interface) are still JIT-specializable. The integration adds:

1. **Automatic dimension specialization**: RAJA's `TeamsPolicies` carry launch configuration at the abstraction level; Proteus intercepts and specializes `blockDim`/`gridDim` automatically without developer annotation
2. **Zero annotation burden for RAJA users**: portability abstraction and JIT specialization compose without user-visible API changes
3. **No performance regressions**: tested on full RAJAPerf benchmark suite with zero regressions across 100+ kernels

This is the most sophisticated production integration of Proteus to date. Two unspecified "mission-critical LLNL applications" use Proteus+RAJA in production (per LLNL newsroom, 2025).

---

## 9. Positioning vs. libkdl: Detailed Comparison

### 9.1 Problem Statement Overlap and Divergence

Both Proteus and libkdl address the observation that GPU kernels can perform better with runtime information than with static AOT compilation. But they address different sub-problems:

**Proteus problem:** "A single GPU vendor target exists; the AOT-compiled kernel for it is suboptimal because runtime values were unknown at compile time."

**libkdl problem:** "Multiple GPU vendor targets exist; the correct pre-compiled kernel for this device has not been selected yet."

These are orthogonal: Proteus improves performance within a chosen target; libkdl selects among targets. They are also composable: Proteus-specialized kernels could be stored as variants in a libkdl MTB bundle.

### 9.2 Architecture Comparison Matrix

| Dimension | Proteus | libkdl |
|-----------|---------|--------|
| **Core mechanism** | LLVM IR constant folding + JIT recompilation | AOT multi-variant bundle + runtime selection |
| **IR substrate** | LLVM IR (embedded in binary by ProteusPass) | Vendor-native binaries (PTX/HSACO/x86) in MTB |
| **Cross-vendor dispatch** | No — one CUDA OR one HIP binary per run | Yes — CUDA + HIP + CPU in single bundle |
| **Dispatch decision time** | Each kernel launch (specialization check) | Load time (device enumeration + capability match) |
| **First-launch overhead** | JIT compilation (background thread, but non-zero) | <5 ms (hash lookup + driver load) |
| **Steady-state overhead** | Near-zero (cache hit) | Near-zero (hash lookup) |
| **Performance source** | Specialization enables new optimizations (up to 23x) | Pre-tuned vendor-native binary (no translation overhead) |
| **Max performance ceiling** | Potentially above AOT (JIT sees runtime values) | Equal to best AOT per vendor (no specialization) |
| **User annotation required** | Yes (arg indices annotated) | No (transparent to kernel code) |
| **Programming model** | C++ / CUDA / HIP (no SYCL requirement) | Programming-model-agnostic (dispatches binaries) |
| **MLIR awareness** | None | MLIR gpu.module / gpu.binary compatible design |
| **Cost model** | None — JIT decides by compiling | Roofline analytical cross-vendor cost model |
| **Vendor neutrality at runtime** | One vendor per process | Multiple vendors simultaneously |
| **HPC adoption** | Two LLNL production apps, RAJA integration | Prototype (~5100 LOC), not yet deployed |
| **ML ecosystem integration** | None documented | Designed for PyTorch/ONNX RT integration |
| **Binary size** | AOT binary + embedded LLVM IR overhead | N vendor binaries + routing table |
| **Cache persistence** | Yes (disk cache, per specialization) | Yes (disk cache, per device fingerprint) |

### 9.3 The Key Asymmetry: JIT vs. AOT

Proteus can **exceed** what AOT achieves for a single vendor because it sees runtime values the AOT compiler cannot. libkdl **never exceeds** the best AOT binary for that vendor — it just ensures the right vendor's best binary is loaded.

Implication for the poster: Proteus and libkdl are not competing on the same performance axis. Proteus makes "the kernel you already have" better. libkdl makes "the right kernel for this hardware" available. They are complementary, and the paper should say so explicitly.

### 9.4 What Proteus Cannot Do That libkdl Does

1. **Select among GPU vendors at runtime.** A Proteus binary is compiled for CUDA _or_ HIP, not both. Deploying on an AMD cluster vs NVIDIA cluster requires separate binaries or recompilation. libkdl selects the appropriate pre-compiled variant without recompilation.

2. **CPU fallback with zero additional JIT.** Proteus can compile for host (CPU) but this is a separate binary artifact. libkdl's MTB bundle includes a CPU fallback as a ranked variant, automatically selected if no GPU is found.

3. **Cross-vendor cost-model-driven selection.** Proteus has no mechanism to say "for this GEMM shape, the RTX 4090 is faster than the MI300X." libkdl's roofline cost model does this (prototype level).

4. **Heterogeneous multi-device dispatch.** Proteus targets one device per kernel launch. libkdl's architecture supports routing different kernels in a DAG to different devices simultaneously.

5. **Binary format portability.** Proteus produces CUDA fatbin or HIP code object — vendor-native formats. libkdl MTB is a single multi-vendor artifact. For cloud deployment where device type is unknown at build time, libkdl MTB is the correct artifact format; Proteus's output is not.

### 9.5 What Proteus Does That libkdl Does Not

1. **Dynamic performance improvement beyond AOT.** Proteus's 2.8x AMD / 1.78x NVIDIA speedups represent genuine performance gains that AOT cannot achieve. libkdl does not improve individual kernel performance — it selects the best pre-existing binary.

2. **Annotation-based specialization interface.** For HPC kernels where the developer knows which arguments are performance-critical, Proteus provides a principled way to expose that knowledge to the compiler. libkdl has no equivalent.

3. **JIT adaptation to changing inputs.** If loop bounds vary across different calls (e.g., N=1024 then N=4096), Proteus generates and caches both specializations automatically. libkdl's variants are fixed at bundle-build time.

4. **HPC production validation.** Two LLNL production applications + RAJA integration constitute more deployment evidence than libkdl's GTX 1650 prototype.

5. **Fortran support.** LLVM IR as the common substrate enables JIT for Fortran GPU offload, which libkdl (targeting MLIR/C) does not address.

### 9.6 Addressing the "Why Not Just Use Proteus?" Objection

The anticipated poster reviewer question: if Proteus already does portable GPU JIT via LLVM IR, why is libkdl needed?

**Answer (three-part):**

(a) **Different target problem.** Proteus solves intra-vendor kernel specialization. libkdl solves inter-vendor kernel selection. An ML deployment scenario where the production server may be AWS (NVIDIA) or Azure (AMD) or CPU-only cannot be solved by Proteus — it requires either recompilation per vendor or a dispatch layer like libkdl.

(b) **JIT overhead is non-trivial for first invocation.** Proteus runs AOT-compiled code on the first kernel call and switches to JIT-specialized code on subsequent calls. In inference serving (often single-pass for new request types), this JIT warmup never amortizes. libkdl's <5ms load-time dispatch has no warmup penalty.

(c) **Proteus has no cross-vendor cost model.** Even if Proteus could theoretically compile for both CUDA and HIP (it cannot in a single binary), it has no mechanism to decide which target to use based on the connected hardware's compute and memory bandwidth characteristics. libkdl's roofline cost model is explicitly designed for this.

---

## 10. Positioning in the Broader Landscape

Proteus's closest relatives by technique:

| System | Technique | Cross-vendor | Peak speedup |
|--------|-----------|:---:|:---:|
| **Proteus** (CGO 2025) | LLVM IR constant folding, JIT | No | 2.8x AMD |
| AdaptiveCpp SSCP (IWOCL 2023) | LLVM IR → JIT per target | Yes | +30% over CUDA |
| Jitify (NVIDIA) | CUDA source string JIT | NVIDIA only | Baseline |
| NVRTC (NVIDIA) | CUDA source string JIT | NVIDIA only | Baseline |
| HIPRTC (AMD) | HIP source string JIT | AMD only | Baseline |
| ORC JIT (LLVM) | IR → host only | CPU only | N/A |
| **libkdl** | Pre-compiled AOT + runtime selection | Yes | Native per vendor |

Proteus is the only system in this table that:
- Operates at LLVM IR level (not source string, not PTX)
- Works on both AMD and NVIDIA GPU targets
- Has published speedups exceeding 2x

This makes it the single strongest academic precedent for LLVM-IR-level GPU optimization. The paper MUST be cited in the libkdl poster as the state-of-the-art for intra-vendor JIT, which libkdl then extends to inter-vendor selection.

---

## 11. Gaps Proteus Does Not Address (Confirmed by Research)

From analysis of all accessible Proteus materials (paper, GitHub, LLNL newsroom, SC25 paper, conference page):

**G1 — No cross-vendor dispatch.** Proteus is explicitly tested on "AMD and NVIDIA GPUs" as separate evaluation tracks. There is no mechanism to produce a single binary that runs on both NVIDIA and AMD, or to select between vendors at runtime. The LLVM IR embedded by ProteusPass is vendor-agnostic at the IR level, but the JIT backend is selected at compile time (NVPTX for CUDA, AMDGPU for HIP).

**G2 — No selection policy.** Proteus does not implement any cost-model-driven selection of which GPU to use for a given workload. Selection is implicit: the binary is compiled for a specific vendor, so the GPU of that vendor gets the kernel.

**G3 — JIT warmup exists.** The first execution of an uncached specialization runs the AOT binary while JIT compiles in background. For latency-critical first-request scenarios (inference serving, cold deployment), this warmup is non-trivial. Magnitude not published in accessible text; estimated at hundreds of milliseconds for complex kernels (standard LLVM optimization + NVPTX/AMDGPU backend).

**G4 — ML ecosystem integration absent.** No PyTorch operator, torch.compile backend, or ONNX RT execution provider integration is documented. Proteus targets HPC codes (RAJA, BOUT++, hydrodynamics, weather). The ML inference dispatch use case is outside Proteus's current scope.

**G5 — Requires annotation or source modification.** Developers must add `__attribute__((annotate("jit", ...)))` or rewrite kernels using the DSL/frontend API. libkdl's dispatch is transparent to kernel source code.

**G6 — Single device per launch.** Proteus dispatches to one GPU. Heterogeneous DAG dispatch (kernel A on NVIDIA, kernel B on AMD) is not addressed.

---

## 12. Risk Assessment for libkdl Positioning

### Risk: Proteus appears to "solve" the JIT GPU problem

**Mitigation:** Proteus solves intra-vendor kernel quality; libkdl solves inter-vendor kernel selection. Frame as complementary in a single comparison table row. The existing `findings.md` table already does this correctly.

### Risk: Reviewer cites Proteus's 2.8x speedup and asks "why not just use Proteus?"

**Mitigation:** Three-part answer at section 9.6 above. Rehearse this as the poster's most likely hard question.

### Risk: Proteus adds cross-vendor dispatch in a future release

**Timeline check:** Current repository (March 2026, v2026.03.0) has no cross-vendor branch or issue. The SC25 paper (November 2025) only extends RAJA integration. No indication of cross-vendor dispatch in roadmap. Proteus is HPC-focused; ML inference deployment is not in scope.

### Risk: The "1.23x better than Jitify" claim minimizes libkdl's AOT advantage

**Mitigation:** Jitify comparison is within NVIDIA only. The correct comparison for libkdl is: libkdl dispatches a vendor-native pre-compiled binary with <5ms latency vs. Proteus JIT which compiles a new binary on first launch. These are not the same problem. Use the PT2/Triton 843-second cold start data to motivate why JIT-first architectures are costly.

---

## 13. Recommended Citations and Framing for the Poster

**Primary citation:**
Georgakoudis, G., Parasyris, K., Beckingsale, D. "Proteus: Portable Runtime Optimization of GPU Kernel Execution with Just-in-Time Compilation." CGO 2025. doi:10.1145/3696443.3708939

**Secondary citation:**
Bowen, J., Parasyris, K., Beckingsale, D., Ben-Nun, T., Stitt, T., Georgakoudis, G. "Extending RAJA Parallel Programming Abstractions with Just-In-Time Optimization." SC '25 Workshops, P3HPC. doi:10.1145/3731599.3767492

**Framing for poster comparison table row:**
"Proteus [CGO25]: LLVM-IR JIT specialization. Improves single-vendor kernel quality (2.8x AMD, 1.78x NVIDIA). No cross-vendor dispatch, no selection policy, JIT warmup on first call. Complementary to libkdl: libkdl selects which vendor-tuned binary to run; Proteus improves that binary's quality for the chosen vendor."

**Framing for novelty claim:**
"Unlike Proteus, which specializes a kernel for its runtime environment after the vendor is chosen, libkdl selects the pre-compiled optimal kernel for the available hardware — enabling deployment-time portability without any JIT overhead."

---

## 14. Summary

**Finding:** Proteus (LLNL, CGO 2025) is the closest academic competitor to libkdl's approach but addresses a strictly different sub-problem. Proteus performs runtime kernel specialization within a single vendor target using LLVM IR constant folding; libkdl performs runtime vendor selection among pre-compiled multi-vendor binaries. Both use LLVM IR as a central technical substrate but at different phases.

**Evidence:**
- Proteus performance: 2.8x AMD (Tioga/MI250X), 1.78x NVIDIA (Lassen/V100), 1.23x better than Jitify
- RAJA+Proteus: 1.2x–23x on AMD MI250X, 1.1x–15x on NVIDIA V100, zero regressions
- Architecture: ProteusPass extracts IR at AOT time; runtime folds constants into IR clone and JIT-compiles per specialization
- Cross-vendor: explicitly absent — separate CUDA and HIP binary tracks in all evaluations
- ML integration: absent — HPC-only (RAJA, BOUT++, hydrodynamics, weather)

**Related:** AdaptiveCpp SSCP (closest to Proteus in embedding LLVM IR for JIT), Jitify (outperformed by Proteus on NVIDIA), liboffload PR #186088 (infrastructure Proteus could integrate with), ROCR InterceptQueue (cleaner AMD dispatch interception than Proteus uses)

**Risks:**
- Proteus may add cross-vendor dispatch in future versions (no current indication)
- The 2.8x speedup is a powerful claim — libkdl's claim is qualitatively different (selection, not specialization) and must be clearly distinguished
- Poster reviewers familiar with Proteus will ask "why not just use Proteus?" — section 9.6 provides the answer

---

## Sources

1. ACM DL — CGO 2025 paper abstract: https://dl.acm.org/doi/10.1145/3696443.3708939
2. CGO 2025 conference page: https://2025.cgo.org/details/cgo-2025-papers/24/Proteus-Portable-Runtime-Optimization-of-GPU-Kernel-Execution-with-Just-In-Time-Comp
3. GitHub repository: https://github.com/Olympus-HPC/proteus
4. LLNL newsroom: https://computing.llnl.gov/about/newsroom/proteus
5. Olympus-HPC organization: https://olympus-hpc.github.io/
6. Zenodo artifact (v1.0.0-cgo25): https://zenodo.org/records/14087064
7. RAJA feature branch: https://github.com/Olympus-HPC/RAJA/tree/feature/proteus
8. RAJAPerf benchmark branch: https://github.com/Olympus-HPC/RAJAPerf/tree/benchmark-proteus
9. Proteus-benchmarks SC25 repro: https://github.com/Olympus-HPC/proteus-benchmarks/tree/p3hpc-sc25-paper-repro
10. SC25 P3HPC paper: https://dl.acm.org/doi/10.1145/3731599.3767492
11. ACM DL DOI redirect: https://doi.org/10.1145/3696443.3708939
12. Mneme companion project: https://github.com/Olympus-HPC/Mneme
