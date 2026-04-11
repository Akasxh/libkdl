# Heterogeneous GPU Kernel Dispatch via LLVM/MLIR

**Venue:** LLVM Dublin 2026 (Poster/Lightning Talk)
**Deadline:** Poster ready by 2026-04-07

## Submitted Abstract

LLVM and MLIR enable portable code generation for machine learning kernels, abstracting operations like GEMM through dialects such as linalg to facilitate efficient lowering to target-specific binaries. However, a key runtime gap persists — the absence of dynamic hardware introspection and vendor-agnostic dispatch, forcing per-target recompilation or manual orchestration in heterogeneous NVIDIA, AMD, and CPU environments.

Existing runtimes only partially address these challenges. The MLIR ExecutionEngine supports JIT invocation but fixes targets during compilation. Similarly, the IREE hardware abstraction layer utilizes modular backends yet still requires internal stacks. Broader abstractions like SYCL provide portability at the cost of significant runtime overhead but promises portability.

We explore existing (e.g. SYCL) implementations alongside different implementations for vendor-agnostic rerouting for HPC workloads while keeping machine learning oriented kernels in mind.

Inspiration from ALPAKA and CERN's ROOT TMVA-SOFIE, which uses it for heterogeneous acceleration of ONNX-derived ML models which is being developed using ALPAKA's header-only library for GPU-agnostic behavior.
