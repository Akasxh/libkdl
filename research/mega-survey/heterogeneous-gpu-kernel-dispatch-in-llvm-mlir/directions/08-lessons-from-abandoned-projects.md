# Direction 08: Lessons from Abandoned GPU Portability Standards — Design Rationale via Negative Evidence

**Composite Score: 6.75/10**
**Rank: 8 of 8**

---

## Title

**Why HSA, GPU Ocelot, and OpenCL Failed at Heterogeneous Dispatch — And How libkdl Avoids Their Fate**

## One-Sentence Description

A systematic taxonomy of five failure modes from abandoned GPU portability projects (HSA Foundation, GPU Ocelot, OpenCL compute, HSAIL, C++ AMP), showing how libkdl's architectural decisions explicitly mitigate each one.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| HSA Foundation collapse | wave-05-abandoned-hetero-dispatch | Twitter silent since Feb 2020; GCC dropped 21,000 lines of HSA offloading (Aug 2020) |
| GPU Ocelot dormancy | wave-05-abandoned-hetero-dispatch | PTX interception broke repeatedly on CUDA ABI changes; AMD CAL backend discontinued |
| OpenCL ecosystem failure | wave-05-abandoned-hetero-dispatch | NVIDIA deliberate underperformance; no cuDNN/cuBLAS equivalent; Apple deprecated |
| SPIR-V semantic fragmentation | wave-05-abandoned-hetero-dispatch | Kernel vs GLCompute execution models incompatible; vendor extensions create de facto dialects |
| GCC HSA + Intel MIC removal | wave-05-abandoned-hetero-dispatch | Zero documented real-world usage → compiler backend deletion |
| AQL dispatch survived in ROCm | wave-05-abandoned-hetero-dispatch | Technical mechanism preserved inside AMD's walled garden |
| ACM HPDC 2025 cross-vendor paper | wave-05-abandoned-hetero-dispatch | Confirms no runtime cross-vendor kernel dispatch exists as of 2025 |

---

## Novelty Argument

The novelty here is not in the individual historical facts but in the systematic failure-mode taxonomy and the explicit mapping to libkdl design decisions:

| Failure Mode | Historical Victim | libkdl Mitigation |
|-------------|-------------------|-------------------|
| FM-1: Single-vendor dependency | HSA Foundation (AMD only) | Three independent backends; no single backend is load-bearing |
| FM-2: Proprietary ABI fragility | GPU Ocelot (PTX interception) | Dispatches via stable vendor driver APIs; does not intercept vendor binary formats |
| FM-3: Performance underdelivery | OpenCL (0.3x vs CUDA for ML) | Routes to vendor-native pre-compiled binaries; no translation overhead |
| FM-4: Semantic fragmentation | SPIR-V (Kernel vs GLCompute) | Capability-query phase validates kernel-backend compatibility before dispatch |
| FM-5: Zero demonstrated usage | GCC HSA, Intel MIC | Prototype verified on real hardware; targets PyTorch ecosystem for adoption |

This taxonomy has not been published as a structured analysis. The poster benefits from showing that libkdl is not repeating history.

---

## Feasibility Plan

Already complete — wave-05-abandoned-hetero-dispatch.md and wave-05-abandoned-hsa.md provide all necessary historical data. The contribution is analytical, not implementation-based.

For poster: a single "lessons learned" box with the 5-row failure mode table above.

---

## Poster Potential

**Limited as standalone; strong as a poster narrative element.**

Best used as a motivation panel: "Why prior approaches failed and how libkdl differs."
- 5-row failure mode table (compact, high information density)
- GPU Ocelot as the closest historical precedent: intercepted below the runtime; libkdl dispatches above it
- Survivors table: AQL in ROCm, SPIR-V in Khronos, PTX-to-LLVM in NVPTX — mechanisms survive when adopted into dominant stacks

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **6/10** | Historical analysis is useful context but not a new research contribution. |
| **Feasibility** | **10/10** | All data already collected across wave-05 files. |
| **Evidence** | **8/10** | HSA collapse, GPU Ocelot dormancy, GCC deletion all well-documented. |
| **Impact** | **5/10** | Good narrative element but does not advance the technical state of the art. |
| **Composite** | **6.75/10** | |

---

## Key Takeaway for Poster

The single most important lesson: **a dispatch layer must use vendor runtimes as backends, not replace them.** GPU Ocelot tried to replace vendor runtimes by intercepting PTX. OpenCL tried to replace vendor APIs with a standard one. Both failed. libkdl delegates to cuModuleLoadData, hipModuleLoadData, and dlopen — it adds a selection layer above vendor APIs, not a replacement layer beside them.
