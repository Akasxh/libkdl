# Wave 03 — AdaptiveCpp SSCP Generic Offloading

**Angle:** AdaptiveCpp SSCP Generic Offloading
**Query:** AdaptiveCpp hipSYCL SSCP single-source single-compiler-pass generic offloading
**Date:** 2026-04-06

---

## Source Index

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | One Pass to Bind Them: The First Single-Pass SYCL Compiler with Unified Code Representation Across Backends | https://dl.acm.org/doi/abs/10.1145/3585341.3585351 | IWOCL 2023 | Paper | 10/10 |
| S2 | Adaptivity in AdaptiveCpp: Optimizing Performance by Leveraging Runtime Information During JIT-Compilation | https://dl.acm.org/doi/10.1145/3731125.3731127 | IWOCL 2025 | Paper | 9/10 |
| S3 | AdaptiveCpp Stdpar: C++ Standard Parallelism Integrated Into a SYCL Compiler | https://dl.acm.org/doi/fullHtml/10.1145/3648115.3648117 | IWOCL 2024 | Paper | 8/10 |
| S4 | hipSYCL: The first single-pass SYCL implementation (SSCP blog post) | https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/ | 2023 | Blog/Docs | 10/10 |
| S5 | AdaptiveCpp Compilation Model (official docs) | https://adaptivecpp.github.io/AdaptiveCpp/compilation/ | Current | Docs | 10/10 |
| S6 | AdaptiveCpp GitHub README and doc/compilation.md | https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/compilation.md | Current | Source/Docs | 9/10 |
| S7 | AdaptiveCpp DeepWiki architecture overview | https://deepwiki.com/AdaptiveCpp/AdaptiveCpp | 2024-2025 | Docs | 7/10 |
| S8 | AdaptiveCpp 24.02 Release — persistent kernel cache | https://www.phoronix.com/news/AdaptiveCpp-24.02-Released | 2024 | Blog | 7/10 |
| S9 | Semantic Scholar: One Pass to Bind Them | https://www.semanticscholar.org/paper/One-Pass-to-Bind-Them:-The-First-Single-Pass-SYCL-Alpay-Heuveline/849f4d7ae753873f531bbb914f9e86ae5fb820cb | 2023 | Index | 8/10 |
| S10 | HCF (hipSYCL Container Format) Discussion | https://github.com/OpenSYCL/OpenSYCL/discussions/760 | 2022-2023 | Discussion | 7/10 |

---

## Source Summaries

### S1 — One Pass to Bind Them (IWOCL 2023) [10/10]

**Authors:** Aksel Alpay, Vincent Heuveline (Heidelberg University / IWR)
**DOI:** https://doi.org/10.1145/3585341.3585351

The foundational paper introducing SSCP in AdaptiveCpp (then hipSYCL). Prior SYCL implementations required one compiler invocation per backend target (SPIR-V pass, PTX pass, amdgcn pass), making universal binaries impractical. This work presents the first single-source, single-compiler-pass design where device IR is extracted during the host compilation pass and stored in a backend-independent LLVM IR form. At runtime, the stored IR is JIT-lowered to whatever backend is present.

**Key detail:** Universal binaries targeting any NVIDIA, Intel, or AMD ROCm GPU achieve only ~20% additional compilation time compared to a plain clang host build — vs. 2x or more for multi-target AOT compilation.

**Relevance to libkdl thesis:** This is the closest prior art to libkdl's "compile once, dispatch anywhere" model. SSCP is the compiler-side analog; libkdl is the runtime-side analog for pre-existing kernel binaries.

---

### S2 — Adaptivity in AdaptiveCpp (IWOCL 2025) [9/10]

**Venue:** 13th International Workshop on OpenCL and SYCL
**DOI:** https://doi.org/10.1145/3731125.3731127

Extends SSCP with runtime adaptivity: the JIT stage automatically specializes kernels using observed runtime information — work-group sizes, pointer alignment, argument values. This is the first SYCL implementation to do this automatically across all backends (CPU, NVIDIA, AMD, Intel) using a single unified JIT infrastructure.

**Key detail:** With the adaptivity framework, AdaptiveCpp outperforms CUDA by 30% in geometric mean, HIP by 44%, oneAPI by 23% on a suite of mini-apps and benchmarks across NVIDIA/AMD/Intel hardware. Gains up to 5x in extreme cases.

**Relevance to libkdl thesis:** Demonstrates that deferring compilation to runtime is not just neutral in performance but can be actively superior to AOT compilation — directly validates the dynamic dispatch value proposition.

---

### S3 — AdaptiveCpp Stdpar (IWOCL 2024) [8/10]

**DOI:** https://doi.org/10.1145/3648115.3648117

Demonstrates using the SSCP infrastructure to offload C++ Standard Parallelism (`std::for_each`, `std::transform_reduce`) to GPUs without any SYCL-specific annotations. The SSCP single-pass model makes this tractable: because the compiler only parses once and does not require device-specific attribute markup, arbitrary C++ parallel algorithms can be intercepted and offloaded.

**Key detail:** AdaptiveCpp is described as the first compiler able to offload standard C++ to "any" GPU — reusing the same generic LLVM IR container and llvm-to-backend runtime infrastructure.

**Relevance to libkdl thesis:** Shows SSCP's IR container is general enough to hold non-SYCL workloads, a stepping stone toward a general-purpose kernel dispatch layer.

---

### S4 — SSCP Blog Post / Original Announcement (2023) [10/10]

**URL:** https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/

The primary public-facing technical explanation of how SSCP works. Documents the two-stage pipeline, the unified LLVM IR representation, the llvm-to-backend runtime infrastructure, and the performance results from BabelStream.

**Key detail (compilation overhead):** SSCP compilation of BabelStream takes roughly 15% longer than host-only compilation while producing a binary that runs on any NVIDIA GPU, any Intel GPU, and 38 distinct AMD ROCm GPUs. The prior per-target approach was 2x slower for only 3 AMD targets. Kernel runtime performance is within 10% in either direction vs. per-target binaries.

**Key detail (runtime translation):** The llvm-to-backend step roughly doubles the driver-level JIT compilation time, but does not change its order of magnitude — so the absolute overhead remains in the same range as existing driver-level compilation costs.

---

### S5 — Official Compilation Model Documentation [10/10]

**URL:** https://adaptivecpp.github.io/AdaptiveCpp/compilation/

Current authoritative documentation. Describes all three compilation flows:

1. **Generic SSCP** (default): single parse, LLVM IR embedded, runtime JIT via llvm-to-backend
2. **Multipass interop flows** (`cuda.integrated-multipass`, `hip.integrated-multipass`, etc.): clang toolchain-based AOT, enables SYCL+CUDA/HIP mixing
3. **Library-only flows** (`omp.library-only`, `cuda-nvcxx`): third-party compiler, deployment simplicity

SSCP supports: CUDA (PTX), ROCm/HIP (amdgcn), Level Zero/Intel (SPIR-V), OpenCL (SPIR-V), CPU (OpenMP/LLVM), Apple Metal (experimental).

**Key detail:** The doc explicitly states the three advantages of unified IR: (1) enables backend-independent JIT features like kernel fusion, (2) single parse dramatically reduces compile time for template-heavy code, (3) binaries run on all supported hardware without user-side precompilation.

---

### S6 — doc/compilation.md (GitHub, develop branch) [9/10]

**URL:** https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/compilation.md

Source-controlled documentation. Confirms that `--acpp-targets=generic` is the current default target flag that activates SSCP. Introduces the Heterogeneous Container Format (HCF) — a flexible binary container holding arbitrary numbers of kernel code images plus hierarchical text metadata. The embedded LLVM IR blob is wrapped in HCF and linked into the host binary; `HIPSYCL_HCF_DUMP_DIRECTORY` env var causes the runtime to dump HCF data for inspection; `acpp-hcf-tool` is provided for examining/modifying HCF files.

**Key detail:** The generic target's JIT overhead is mitigated by a persistent on-disk kernel cache. On subsequent runs, the JIT result is read from cache, eliminating first-launch latency for warm applications.

---

### S7 — DeepWiki AdaptiveCpp Architecture Overview [7/10]

**URL:** https://deepwiki.com/AdaptiveCpp/AdaptiveCpp

Third-party architectural synthesis. Documents the modular runtime: four subsystems (compilation, runtime, memory, API layers). The runtime uses a task DAG model: user submits command groups → handler creates DAG nodes → DAG manager schedules by data dependency → backend executors dispatch. Each backend implements a common `backend` interface and `backend_manager` registers all available backends. SSCP binaries carry all targets' IR; the runtime backend_manager selects the executor matching the discovered hardware.

**Key detail:** "A dynamic plugin architecture where backends are loaded at runtime, enabling a single binary that targets all supported hardware, adapting to the available resources at runtime." This is the exact model libkdl proposes at the kernel-binary level.

---

### S8 — AdaptiveCpp 24.02 Release Notes [7/10]

**URL:** https://www.phoronix.com/news/AdaptiveCpp-24.02-Released

Reports that version 24.02 makes SSCP the default compilation model and ships persistent on-disk kernel cache and automatic runtime kernel specialization as production features. Also confirms SSCP as "one of the best SYCL compilers for performance" in external benchmarks at this point.

**Key detail:** Persistent cache + automatic specialization together mean JIT overhead is a one-time cost per hardware configuration, not per-run overhead.

---

### S9 — Semantic Scholar Index Entry: One Pass to Bind Them [8/10]

**URL:** https://www.semanticscholar.org/paper/.../849f4d7ae753873f531bbb914f9e86ae5fb820cb

Confirms paper metadata: Alpay & Heuveline, IWOCL 2023, Article 7, pp. 1-12. Abstract confirms: "the very first SYCL implementation with a single-source, single compiler pass (SSCP) design and a unified code representation across backends."

---

### S10 — HCF Tool Discussion Thread [7/10]

**URL:** https://github.com/OpenSYCL/OpenSYCL/discussions/760

Community discussion revealing that HCF (hipSYCL Container Format) is designed to be easy to parse, support both binary code images and text-based hierarchical metadata, and be flexible enough for multiple code images (different backends) within one container. The `hipsycl-hcf-tool` / `acpp-hcf-tool` enables inspection and manipulation.

**Key detail:** HCF is the binary container format that embeds multiple kernel code images (one per target backend) in a single file — analogous to what libkdl's `.kdl` format aims to achieve.

---

## Technical Architecture: SSCP in Depth

### Two-Stage Compilation Pipeline

```
Source (.cpp with SYCL kernels)
          |
          v
[Stage 1 — Compile time: single clang pass]
  - Host AST parsed once
  - Device kernels identified and extracted as LLVM IR
  - Backend-independent builtin representations inserted
  - LLVM IR blob wrapped in HCF container
  - HCF embedded as data section in host ELF/binary
          |
          v
Host binary with embedded HCF (LLVM IR)
          |
          v [first kernel launch on target device]
[Stage 2 — Runtime: llvm-to-backend JIT]
  - Runtime discovers hardware (CUDA/HIP/Level Zero/OpenCL)
  - Extracts LLVM IR blob from HCF
  - Runs llvm-to-backend lowering:
      LLVM IR → PTX (NVIDIA)     OR
      LLVM IR → amdgcn (AMD)    OR
      LLVM IR → SPIR-V (Intel/OpenCL)
  - Result cached on-disk (persistent kernel cache)
  - Subsequent launches: load from cache
          |
          v
Backend-specific kernel binary submitted to driver
```

### llvm-to-backend Infrastructure

The `llvm-to-backend` component is the runtime translation layer that knows how to lower the generic LLVM IR to each vendor format. Key responsibilities:
- Insert backend-specific intrinsics replacing generic builtin stubs (`get_global_id`, etc.)
- Apply backend-specific calling conventions and address space handling
- Run target-specific optimization passes
- Invoke backend code generator (NVPTX, AMDGPU, or SPIR-V translator)

This is architecturally equivalent to what a "kernel dynamic linker" does at load time — translating a portable IR into a runnable format.

### HCF Binary Format

- Flexible container: N binary images + hierarchical text metadata
- Supports multiple code formats in one container (LLVM IR blob + optional pre-compiled images)
- Can be inspected/modified with `acpp-hcf-tool`
- Runtime dumps via `HIPSYCL_HCF_DUMP_DIRECTORY` env var
- Designed for easy parsing (not a complex format like ELF fat binaries)

### Runtime Backend Selection

AdaptiveCpp's `backend_manager` dynamic plugin system:
- Loads backend plugins at startup
- Discovers available hardware (CUDA driver present? ROCm? Level Zero?)
- Registers all functional backends
- When a kernel is submitted, the runtime selects the backend executor matching the target device's type
- SSCP JIT is invoked lazily on first kernel dispatch per device

### Persistent Kernel Cache

- On-disk cache keyed by (kernel IR hash, device hardware fingerprint, optimization flags)
- First run: JIT compiles + stores result
- Subsequent runs: cache hit, load pre-compiled binary
- Runtime adaptivity: cache entries can be specialized versions (e.g., for observed work-group size of 256)

---

## Performance Data

| Metric | Value | Source |
|--------|-------|--------|
| Compilation overhead vs. host-only | +15-20% | S1, S4 |
| Compilation overhead vs. 3-target AOT | 2x faster | S4 |
| Target coverage from single binary | 38 AMD GPUs + all NVIDIA + all Intel | S4 |
| Kernel runtime vs. per-target AOT | within ±10% | S4 |
| Runtime JIT translation overhead | ~2x driver JIT (same order of magnitude) | S4 |
| Adaptivity framework vs. CUDA | +30% geometric mean | S2 |
| Adaptivity framework vs. HIP | +44% geometric mean | S2 |
| Adaptivity framework vs. oneAPI | +23% geometric mean | S2 |
| Max single-kernel speedup (adaptivity) | 5x+ | S2 |

---

## Relevance to libkdl: "Compile Once, Dispatch Anywhere"

AdaptiveCpp SSCP is the **strongest prior art** for libkdl's core thesis, operating at the compiler level:

| Dimension | AdaptiveCpp SSCP | libkdl |
|-----------|-----------------|--------|
| Portable IR | LLVM IR embedded in HCF | `.kdl` container with LLVM IR / SPIR-V |
| Translation point | Runtime, on first kernel launch | Load time / first dispatch |
| Backend selection | backend_manager + JIT per device type | kdl_dispatch() selects registered handler |
| Persistent cache | On-disk kernel cache | (potential analog) |
| Compilation model | Full SYCL compiler | Agnostic — works with any pre-compiled kernel |
| Scope | SYCL programs only | Any ML kernel (PyTorch, IREE, custom) |

**Critical differentiator:** SSCP requires recompiling with AdaptiveCpp's toolchain. libkdl targets kernels that already exist as compiled binaries or IR blobs from arbitrary frameworks (PTX from PyTorch, SPIR-V from IREE, HIP from ROCm). The libkdl value proposition is decoupled from compilation toolchain choice.

**Shared insight:** Both SSCP and libkdl validate that deferred, runtime-driven lowering from a portable IR is (a) feasible without prohibitive overhead when persistent caching is used, and (b) can outperform AOT because runtime information (hardware, workload shape) enables better specialization.

---

## Risks / Open Questions

1. **SSCP is SYCL-specific:** The IR representation uses SYCL builtins and annotations. A libkdl-style system targeting PyTorch or IREE-compiled kernels would need a different (or more neutral) portable IR — SPIR-V or raw LLVM IR with fewer SYCL assumptions.

2. **JIT latency on first run:** Even with caching, cold-start latency exists. The IWOCL 2025 paper notes this is "generally no longer of concern for most applications" — but ML inference serving (latency-critical, frequent model swaps) may require careful cache warming strategies.

3. **llvm-to-backend is not public API:** The translation infrastructure is internal to AdaptiveCpp. libkdl would need to either reuse this (as a library dependency) or reimplement equivalent per-backend lowering.

4. **Apple Metal support is experimental:** The SPIR-V → Metal path is not production-ready. libkdl faces the same challenge for Apple Silicon targets.

5. **Runtime specialization vs. portability tension:** The adaptivity framework (S2) achieves 30-44% gains by specializing per-run — but this means the cached kernel is not portable across machines with different hardware fingerprints. A shared kernel cache (e.g., in a containerized ML serving scenario) may not benefit.

---

## Key Citations for Poster

- Alpay & Heuveline, "One Pass to Bind Them," IWOCL 2023 — cite as foundational proof that compile-once-dispatch-anywhere is achievable with only 20% compilation overhead.
- Alpay et al., "Adaptivity in AdaptiveCpp," IWOCL 2025 — cite as evidence that runtime-deferred JIT is not a performance compromise but a performance advantage (+30% vs. CUDA).
- AdaptiveCpp Compilation Docs — cite for the concrete two-stage SSCP architecture as a reference design for libkdl's runtime dispatch pipeline.
