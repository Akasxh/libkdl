# Wave 01: PyTorch 2 Compiler Multi-Device Dispatch
Search query: PyTorch inductor triton dispatch multi-device GPU CPU backend selection torch.compile
Sources found: 10
Date: 2026-04-06

## Sources

### 1. TorchInductor: A PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes
- URL: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- Type: docs/RFC (PyTorch Developer Mailing List)
- Date: 2022 (foundational design post, still authoritative)
- Relevance: 9/10
- Novelty: 7/10
- Summary: The original design document for TorchInductor describes a pythonic define-by-run loop-level IR that maps PyTorch models into Triton kernels on GPUs and C++/OpenMP on CPUs. The two-backend model — C++ for CPU, Triton for GPU — is explicitly device-dispatched at the scheduling layer. This is the canonical reference for understanding how Inductor routes to different backends based on device placement metadata.
- Key detail: Inductor augments its IR with operand metadata including tensor shapes and device placement (CPU or GPU), and uses this metadata to route lowering to either CppScheduling or TritonScheduling. The codegen path is therefore determined statically per-subgraph by device type, not dynamically at kernel launch.

---

### 2. Extend TorchInductor to Support More Backends (GitHub Issue #99419)
- URL: https://github.com/pytorch/pytorch/issues/99419
- Type: RFC/issue
- Date: 2023 (ongoing through 2025)
- Relevance: 10/10
- Novelty: 9/10
- Summary: This issue proposes and tracks a dynamic registration mechanism for new Inductor backends, introducing `register_scheduling_for_device` and `get_scheduling_for_device` runtime APIs. New backends (e.g., Intel XPU, Habana Gaudi, AMD via ROCm) can register their Scheduling/Kernel/WrapperCodegen triad at runtime without upstream changes. This is the closest thing PyTorch has to a "kernel dynamic linker" concept — backend selection keyed by device type, resolved at compile/codegen time rather than launch time.
- Key detail: Three structures must be customized per backend: Scheduling (loop fusion and tiling), Kernel (emitting code for a single fused kernel), and WrapperCodegen (wrapper code that calls kernels). Registration is done via `register_backend_for_device(device_str, scheduling_cls)`.

---

### 3. [RFC] A Device-Agnostic Python Runtime API Design for Stream-Based Accelerators (GitHub Issue #128403)
- URL: https://github.com/pytorch/pytorch/issues/128403
- Type: RFC
- Date: 2024
- Relevance: 8/10
- Novelty: 9/10
- Summary: This RFC proposes a unified `torch.accelerator` interface abstracting device-specific stream, event, guard, and allocator operations across CUDA, XPU, MPS, MTIA, and custom (PrivateUse1) devices. It is a runtime-layer analog to Inductor's codegen-time backend dispatch — the goal is that user code can be written once and routed to any registered accelerator. A companion RFC (#134978) extends the design to device-agnostic memory management APIs.
- Key detail: The proposed APIs (`torch.current_stream(device_type)`, `torch.Stream(device_type)`, `torch.DeviceGuard`) accept a device_type string, enabling fully vendor-neutral Python code at the PyTorch runtime layer — a key building block for heterogeneous execution.

---

### 4. [TAC] Follow Up: Inductor HW Backend Implementation (dev-discuss.pytorch.org)
- URL: https://dev-discuss.pytorch.org/t/tac-follow-up-inductor-hw-backend-implementation/2455
- Type: RFC/discussion thread
- Date: September–November 2024
- Relevance: 10/10
- Novelty: 9/10
- Summary: A TAC (Technical Advisory Council) discussion involving engineers from Graphcore, AMD, Intel, and Meta about the challenges of registering new Inductor backends. Key findings: (1) graph-level fusion passes in Inductor are largely device-agnostic; (2) the Halide backend landing proved the pluggable backend model works in practice; (3) Intel Gaudi uses a mid-layer Dynamo backend approach (registers as a torch.compile backend, converts FX graphs to its own executable format) rather than Inductor-level codegen. The thread reveals that device type and Inductor backend are currently conflated, causing friction for new hardware.
- Key detail: Jansel (Inductor lead) notes that the graph fusion portion of Inductor is device-backend-agnostic, and what differs per hardware is only the lowering and codegen stages — directly relevant to libkdl's goal of separating dispatch from execution.

---

### 5. PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch (arXiv 2503.19779)
- URL: https://arxiv.org/abs/2503.19779
- Type: paper
- Date: March 2025
- Relevance: 7/10
- Novelty: 8/10
- Summary: PyGraph integrates into PyTorch 2 to maximize CUDA Graph coverage via three optimizations: automatic code transformation to make models CUDA-Graph-compatible, elimination of parameter copy overhead, and cost-benefit-driven selective deployment. The paper demonstrates that CPU launch overhead is now a dominant bottleneck as GPU throughput scales, achieving 29% geomean speedup over baseline PyTorch 2 CUDA Graph support on 25 ML applications.
- Key detail: Up to 3.36x speedup on H100 by eliminating per-kernel CPU-side launch overhead — directly quantifies the problem that heterogeneous kernel dispatch infrastructure (including libkdl) must solve efficiently.

---

### 6. GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2 (arXiv 2509.16248)
- URL: https://arxiv.org/abs/2509.16248
- Type: paper
- Date: September 2025
- Relevance: 6/10
- Novelty: 7/10
- Summary: GraphMend applies AST-level transformations (Predicated Dynamic Control Flow, Graph-Epilogue Deferred Side Effects) to eliminate TorchDynamo graph breaks caused by dynamic control flow and Python side effects. Evaluated on 8 Hugging Face models on RTX 3090 and A40, it eliminates graph breaks in 6/8 models and achieves up to 75% latency reduction. Relevant because graph breaks force fallback to eager mode — defeating any dispatch optimization.
- Key detail: Graph breaks are a first-order obstacle to heterogeneous dispatch: a model that falls back to eager mode at a graph break loses all FX-level routing opportunities. GraphMend's source transformation approach is a complementary technique to backend dispatch.

---

### 7. State of PyTorch Hardware Acceleration 2025
- URL: https://tunguz.github.io/PyTorch_Hardware_2025/
- Type: blog/survey
- Date: 2025
- Relevance: 7/10
- Novelty: 6/10
- Summary: A comprehensive survey of PyTorch hardware backend maturity as of 2025, covering CUDA (H100, Blackwell), ROCm (ROCm 7.0 with Triton support), and XLA (TPU, custom accelerators). The analysis finds that ROCm's torch.compile+Triton path is functional but lacks CUDA's autotuning maturity, and that engineering debug costs for non-CUDA backends often exceed hardware cost savings. The survey establishes the practical state of heterogeneous dispatch from a practitioner viewpoint.
- Key detail: AMD ROCm 7.0 + torch.compile + Triton is functional for many workloads but lacks aggressive autotuning parity with CUDA — showing the "last-mile" problem in vendor-agnostic dispatch is autotuning, not code generation.

---

### 8. PyTorch 2.9 Wheel Variant Support Expands to ROCm (AMD Developer Blog)
- URL: https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html
- Type: blog
- Date: 2025
- Relevance: 6/10
- Novelty: 6/10
- Summary: PyTorch 2.9 introduces variant wheel architecture for ROCm, enabling hardware-agnostic installation scripts across NVIDIA and AMD GPUs. This is a packaging-layer solution to heterogeneous dispatch: the same Python code targets different silicon by switching the wheel. It does not change runtime dispatch but lowers the barrier for multi-vendor deployments.
- Key detail: PyTorch's wheel variant system is a static (install-time) vendor selection mechanism — contrasting with libkdl's goal of dynamic (runtime) kernel dispatch across vendors on the same system.

---

### 9. [RFC] Intel GPU Inductor Backend Upstreaming (GitHub Issue #114856)
- URL: https://github.com/pytorch/pytorch/issues/114856
- Type: RFC/issue
- Date: 2023–2024
- Relevance: 8/10
- Novelty: 7/10
- Summary: Intel's upstreaming RFC for the XPU Inductor backend demonstrates the `register_backend_for_device` pattern in production: the XPU backend reuses TritonScheduling from the CUDA path while registering to a different device string. This is a concrete example of runtime backend registration enabling one codegen path (Triton/SPIR-V) to service multiple device types, which is architecturally analogous to libkdl's multi-target dispatch.
- Key detail: Intel XPU reuses upstream TritonScheduling by registering under the "xpu" device key — showing Triton+SPIR-V as a convergence point for heterogeneous codegen across CUDA and non-CUDA devices.

---

### 10. [RFC] MPMD+SPMD Pipeline Parallelism in PyTorch/XLA (GitHub Issue #9019)
- URL: https://github.com/pytorch/xla/issues/9019
- Type: RFC/issue
- Date: April 2025
- Relevance: 7/10
- Novelty: 9/10
- Summary: This XLA RFC proposes Multi-Program Multi-Data (MPMD) + SPMD hybrid parallelism, splitting global device pools into heterogeneous SPMD sub-programs that execute across different pipeline stages. This is the first PyTorch-ecosystem RFC explicitly addressing heterogeneous execution across physically distinct device pools (e.g., CPU stages + GPU stages in one graph), making it the closest analog to runtime heterogeneous dispatch in the PyTorch compiler stack.
- Key detail: The RFC allows different pipeline stages to use different hardware (and potentially different SPMD configurations), which is a runtime dispatch decision at the graph boundary — the same layer libkdl targets.

---

## Angle Assessment

- **Coverage:** This angle is moderately well-explored in upstream PyTorch via RFCs and TAC discussions, but the actual heterogeneous *runtime* dispatch (selecting between vendors at kernel-launch time on a live system) remains largely unsolved. Inductor's backend selection is compile-time/codegen-time, not launch-time.

- **Surprise findings:**
  1. The `register_backend_for_device` API (Issue #99419) is a direct runtime registration mechanism for dispatch — closer to libkdl's model than expected, but it operates at codegen time, not launch time.
  2. Intel XPU reusing TritonScheduling under a different device key shows Triton/SPIR-V as a practical convergence ISA across vendors — this strengthens the case for SPIR-V as the portable kernel IR in libkdl's design.
  3. Graph breaks (GraphMend paper) are an underappreciated obstacle to heterogeneous dispatch — they force eager fallback, defeating FX-level routing.
  4. CPU-side launch overhead (PyGraph paper) is quantified at 29% geomean overhead — a strong motivator for batched/graph-level dispatch that libkdl should reference.

- **Gaps:**
  - No existing PyTorch mechanism dispatches the *same compiled kernel* to multiple hardware targets at runtime (e.g., choosing CUDA vs. ROCm vs. CPU at launch based on availability/load). All current dispatch is compile-time.
  - No runtime cost model for backend selection in Inductor — selection is fully static (device type of tensor determines codegen path).
  - No heterogeneous mixed-vendor execution within a single inference pass (e.g., operator A on NVIDIA, operator B on AMD) — the MPMD XLA RFC is a first step but not yet implemented.

- **Suggested follow-up angles:**
  1. **MLIR GPU dialect and backend lowering** — how does MLIR's GPU dialect compare to Inductor's device-keyed backend registration for heterogeneous targets?
  2. **SPIR-V as convergence IR** — Intel XPU + ROCm both targeting Triton/SPIR-V; is this viable as a universal portable kernel format for libkdl?
  3. **Runtime cost models for kernel dispatch** — literature on auto-selecting GPU vs. CPU execution based on tensor shape/size at launch time (complements libkdl's dispatch policy).
  4. **IREE multi-backend dispatch** — how does IREE's HAL compare to PyTorch's `register_backend_for_device` for the same heterogeneous problem?
  5. **CPU-side kernel launch overhead quantification** — extend PyGraph's numbers to heterogeneous (multi-vendor) settings.
