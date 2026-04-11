# SYCL-MLIR: Experiences Building an MLIR-Based SYCL Compiler
## Literature Note — LLVM Dublin 2026

**Paper:** "Experiences Building an MLIR-Based SYCL Compiler"
**Venue:** CGO 2024 (IEEE/ACM International Symposium on Code Generation and Optimization)
**Authors:** Ettore Tiotto, Víctor Pérez, Whitney Tsang, Lukas Sommer, Julian Oppermann, Victor Lomüller, Mehdi Goli, James Brodman (Intel / Codeplay)
**arXiv:** https://arxiv.org/abs/2312.13170
**ACM DL:** https://dl.acm.org/doi/10.1109/CGO57630.2024.10444866
**Artifact:** https://zenodo.org/records/10410758
**Relevance Score:** 9/10

---

## Finding

SYCL-MLIR demonstrates that routing SYCL compilation through MLIR dialects — rather than lowering directly from Clang AST to LLVM IR — yields a **geometric mean 1.18x speedup** over DPC++ and **up to 4.3x speedup** on individual Polybench benchmarks on Intel Data Center GPU Max 1100 (Ponte Vecchio). The speedup comes primarily from **host-device co-analysis** made possible by preserving SYCL semantics in MLIR across the host/device boundary.

---

## Core Technical Contribution

### The problem with traditional SYCL lowering

Both DPC++ (Intel) and AdaptiveCpp lower device kernels to LLVM IR early in the compilation pipeline, immediately after Clang AST processing. This discards two critical classes of information:

1. **High-level SYCL semantics:** Buffer accessor aliasing, work-item position in the ND-range, reduction structure — all become opaque after lowering to LLVM IR loads/stores.
2. **Host context:** Constants computed on the host side (kernel arguments, work-group sizes set by the host call site) are invisible to the device compilation unit because host and device modules are separate.

Result: the device compiler optimizes a kernel in isolation, with no knowledge of what arguments it will receive or what the call pattern looks like.

### The SYCL MLIR dialect

SYCL-MLIR introduces a **new MLIR dialect** capturing core SYCL programming model concepts:

| MLIR Op | SYCL concept |
|---------|--------------|
| `sycl.id` / `sycl.range` | Work-item global/local ID, ND-range dimensions |
| `sycl.accessor.get` | Buffer access with mode (read/write/read_write) |
| `sycl.kernel` | Kernel function with launch parameters |
| `sycl.barrier` | Work-group barrier |
| `sycl.reduction` | Built-in reduction pattern |

Crucially, the **device code module is nested inside the host code module** as an MLIR nested region. This means host-side call sites (with concrete argument values, work-group sizes, accessor bindings) are visible during device-side optimization.

### Compilation pipeline

```
Clang AST (host + device, single parse)
         ↓  Clang-to-MLIR frontend
    Host MLIR (std + affine + SYCL dialect)
         │   ╔══════════════════════╗
         ├── ║  Device MLIR (SYCL   ║  ← nested region
         │   ║  dialect + linalg)   ║
         │   ╚══════════════════════╝
         ↓  Host-device joint optimization passes
    Optimized MLIR (both modules)
         ↓  Device lowering
    LLVM IR (device) → SPIR-V → DPC++ backend → GPU binary
         ↓  Host lowering
    LLVM IR (host) → link + clang-offload-wrapper → fat binary
```

Key optimization passes enabled by the joint representation:

1. **Constant propagation across boundary:** Host-computed kernel arguments (e.g., a work-group size set as a variable) are propagated into device code, enabling loop unrolling and vectorization that LLVM IR-level compilation cannot perform.

2. **Accessor aliasing analysis:** MLIR's type system preserves accessor access modes. Device memory operations on `read_only` accessors are provably non-aliasing, enabling aggressive load motion and CSE.

3. **Affine loop transformations:** Work-item loops expressed in SYCL's ND-range model are represented as affine loops in MLIR. The polyhedral optimizer can then apply tiling and interchange — not possible after lowering to LLVM scalar IR.

---

## Performance Results

**Hardware:** Intel Data Center GPU Max 1100 (Ponte Vecchio, 128 Xe-HPC cores)
**Benchmark suite:** Polybench/SYCL (17 benchmarks: linear algebra, data mining, stencils)
**Baselines:** DPC++ (Intel LLVM), AdaptiveCpp (hipSYCL)

| Metric | Result |
|--------|--------|
| Geometric mean vs DPC++ | **1.18x speedup** |
| Geometric mean vs AdaptiveCpp | ~1.2x speedup |
| Best single benchmark | **4.3x speedup** (matrix computation with host-constant propagation) |
| Benchmarks with regression | None reported |

The 4.3x peak comes from a benchmark where the kernel's main loop bound was a host-side constant, enabling full loop unrolling in MLIR that DPC++ missed because the constant was invisible to the device compiler.

---

## Why MLIR Wins Here

The mechanism mirrors MLIR's core design thesis: **higher-level representations enable higher-quality optimization**. Specifically:

- LLVM IR has lost knowledge that two loads alias different buffers → conservative alias analysis, no vectorization
- MLIR SYCL dialect preserves that they come from accessors with disjoint modes → precise aliasing, vectorized
- LLVM IR has an opaque function call for the kernel launch → no inlining of constants from call site
- MLIR's nested regions expose the call site → constants propagate, loops unroll

This is the same argument libkdl makes at the dispatch level: by preserving kernel identity and capability metadata at the binary level (rather than discarding it after compilation), the runtime can make better dispatch decisions.

---

## Limitations Noted by Authors

1. MLIR-based C/C++ frontends (Polygeist, ClangIR) still have limitations for complex C++ patterns (templates, exceptions, virtual dispatch)
2. The joint host-device optimization potential is not fully exploited — only constant propagation and alias analysis were demonstrated; full interprocedural optimization would require complete host IR analysis
3. Evaluated only on Intel GPU Max; no results for NVIDIA or AMD targets (the SPIR-V path could target these via oneAPI plugins, but not demonstrated)
4. No dynamic shape support evaluated
5. Prototype implementation — not yet integrated into production DPC++ toolchain

---

## Relationship to Existing Literature Notes

This paper is already cited in `/home/akash/PROJECTS/LLVM/literature/sycl-ecosystem.md` (Section 6.1) but treated briefly. Key additional detail from this note:

- The **host-device nesting in MLIR** is the architectural novelty — not just "use MLIR for SYCL"
- The **4.3x peak** comes from a specific mechanism (host-constant propagation), not general MLIR magic
- The **1.18x geometric mean** is the honest overall result — the 4.3x is a cherry-picked best case

---

## Relevance to libkdl

### Direct connections

1. **Proves MLIR-level representation > LLVM-level for GPU kernels:** The 1.18x–4.3x speedup from preserving semantics in MLIR dialects directly supports our argument that MLIR-based kernel dispatch tables can carry richer capability metadata than post-LLVM-lowering artifacts.

2. **Host-device co-analysis precedent:** SYCL-MLIR shows that crossing the host/device boundary in MLIR enables optimizations impossible below. libkdl's dispatch descriptor (compiled into the binary by the MLIR pipeline) is the runtime analog — it carries host-context information (detected GPU capabilities) across the dispatch boundary to select the right kernel.

3. **SPIR-V as the portable target:** The paper's pipeline ultimately lowers to SPIR-V for cross-vendor execution. This validates our use of SPIR-V as a portable intermediate in the libkdl kernel store.

4. **Nested region architecture:** The device-code-nested-in-host-code structure in MLIR is exactly how `gpu.module` works in the MLIR GPU dialect. Our multi-target `gpu.binary` approach follows the same nesting principle.

### Key difference from libkdl

SYCL-MLIR is a **compiler optimization** — it produces better static binaries. libkdl is a **runtime dispatch mechanism** — it selects which binary to run. The two are orthogonal and composable: SYCL-MLIR could generate the optimized kernel variants that libkdl then dispatches at runtime.

---

## Notes for Poster

- Primary citation for "MLIR-based compilation outperforms LLVM-level SYCL" — use 1.18x geo-mean and 4.3x peak
- Cite alongside AdaptiveCpp SSCP to show compiler + runtime perspectives on the same problem
- The Codeplay blog post (https://codeplay.com/portal/blogs/2024/02/09/experiences-building-an-mlir-based-sycl-compiler) is a readable summary for reviewers unfamiliar with MLIR
- The LLVM Dev Meeting 2023 lightning talk slides (https://llvm.org/devmtg/2023-05/slides/Lightning-Talks/02-Lomuller-SYCL-MLIR.pdf) are useful for the LLVM-audience context of our poster
- Honest framing: the 4.3x is real but requires favorable conditions (host-constant propagation path); cite both figures
