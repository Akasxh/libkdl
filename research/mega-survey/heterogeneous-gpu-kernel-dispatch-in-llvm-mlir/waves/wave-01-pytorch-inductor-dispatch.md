# Wave 01: PyTorch Inductor Dispatch
Search query: "PyTorch Inductor device dispatch backend selection TorchDynamo compilation"
Sources found: 9
Date: 2026-04-06

## Sources

### 1. TorchInductor: A PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes
- URL: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- Type: RFC (PyTorch Developer Mailing List)
- Date: 2022
- Relevance: 9/10
- Novelty: 6/10
- Summary: The founding design document for TorchInductor describes a loop-level IR that maps PyTorch FX graphs into either Triton kernels (GPU) or C++/OpenMP (CPU). Device-type metadata on each tensor determines at codegen time which of the two scheduling paths (CppScheduling vs TritonScheduling) is invoked — this is the canonical two-backend dispatch point. The design explicitly anticipates extension to additional hardware via the scheduling abstraction.
- Key detail: Backend selection is fully static per subgraph — device type of the input tensors determines the entire codegen path; there is no runtime cost-model or load-aware selection at kernel launch time.

---

### 2. Extend TorchInductor to Support More Backends (GitHub Issue #99419)
- URL: https://github.com/pytorch/pytorch/issues/99419
- Type: RFC/issue
- Date: 2023 (tracked through 2025)
- Relevance: 10/10
- Novelty: 9/10
- Summary: Proposes and tracks a runtime registration mechanism for new Inductor backends via `register_backend_for_device(device_str, scheduling_cls, wrapper_codegen_cls)`. New backends (Intel XPU, Habana Gaudi, AMD) can plug in their Scheduling/Kernel/WrapperCodegen triad without upstream source changes, making Inductor backend dispatch pluggable at load time. This is the closest structural analog in PyTorch to a kernel dynamic linker — selection keyed by device string, resolved at codegen time.
- Key detail: Three classes must be provided per backend: Scheduling (loop fusion/tiling), Kernel (per-fused-kernel code emitter), WrapperCodegen (Python wrapper that glues and benchmarks kernels). All three can live out-of-tree and register at runtime.

---

### 3. [RFC] Intel GPU Inductor Backend Upstreaming (GitHub Issue #114856)
- URL: https://github.com/pytorch/pytorch/issues/114856
- Type: RFC/issue
- Date: 2023–2024
- Relevance: 9/10
- Novelty: 8/10
- Summary: Intel's production upstreaming of an XPU Inductor backend demonstrates `register_backend_for_device` in practice: the XPU path reuses upstream TritonScheduling (Triton supports Intel GPUs via SPIR-V) while registering under the "xpu" device key. Identifies the key friction point — WrapperCodegen hard-codes CUDA-specific calls like `synchronize()` — and proposes a `DeviceOpOverrides` interface to abstract device-biased code generation.
- Key detail: `DeviceOpOverrides` is the proposed abstraction layer for device-specific wrapper operations; without it, adding a new device requires patching core Inductor classes rather than pure out-of-tree registration.

---

### 4. [RFC] A Device-Agnostic Python Runtime API for Stream-Based Accelerators (GitHub Issue #128403)
- URL: https://github.com/pytorch/pytorch/issues/128403
- Type: RFC
- Date: 2024
- Relevance: 8/10
- Novelty: 9/10
- Summary: Proposes a unified `torch.accelerator` interface abstracting stream, event, guard, and allocator operations across CUDA, XPU, MPS, MTIA, and PrivateUse1 devices. This is a runtime-layer complement to Inductor's codegen-time dispatch — user code can be device-neutral by calling `torch.current_stream(device_type)` and related APIs rather than `torch.cuda.*`. A companion RFC (#134978) extends this to memory management APIs.
- Key detail: `torch.Stream(device_type)`, `torch.DeviceGuard`, and `torch.current_accelerator()` form the proposed runtime dispatch surface — vendor-neutral Python that routes through registered device backends at runtime.

---

### 5. [TAC] Follow Up: Inductor HW Backend Implementation (PyTorch Developer Mailing List)
- URL: https://dev-discuss.pytorch.org/t/tac-follow-up-inductor-hw-backend-implementation/2455
- Type: discussion thread (TAC)
- Date: September–November 2024
- Relevance: 10/10
- Novelty: 9/10
- Summary: TAC discussion involving engineers from Graphcore, AMD, Intel, and Meta examining the practical challenges of new Inductor hardware backends. Key finding: Inductor's graph-level fusion passes are already device-agnostic; only the lowering and codegen stages are device-specific. The Halide backend demonstrated the pluggable model works. Intel Gaudi takes a different approach — registering as a Dynamo-level backend (above Inductor) and converting FX graphs to its own executable format.
- Key detail: Jason Ansel (Inductor lead) confirms graph fusion in Inductor is device-backend-agnostic; the dispatch fork occurs only at the lowering step — directly relevant to designs where a shared optimization pass precedes device-specific kernel dispatch.

---

### 6. Generating State-of-the-Art GEMMs for Heterogeneous Hardware with TorchInductor (PyTorch Conference 2025)
- URL: https://pytorchconference.sched.com/event/2A7dg/generating-state-of-the-art-gemms-for-heterogeneous-hardware-with-torchinductor-michael-lazos-and-henry-tsang-meta
- Type: conference talk (PyTorch Conference 2025)
- Date: 2025
- Relevance: 8/10
- Novelty: 8/10
- Summary: Meta engineers describe integrating CUTLASS (NVIDIA) and Composable Kernel/CK (AMD) as GEMM backends within TorchInductor. Inductor automates autotuning by precompiling kernel variants, caching them locally/globally, and benchmarking at PT2 compile time to select the optimal kernel per shape. Backends support torch.compile, AOTInductor, and all GEMM ops (mm, addmm, bmm), achieving up to 10% over Triton/cuBLAS on production workloads.
- Key detail: This is a within-device heterogeneous selection: for a given CUDA device, Inductor benchmarks Triton, CUTLASS, and cuBLAS kernels for each GEMM shape and caches the winner — a shape-aware runtime dispatch cache operating at compile time.

---

### 7. AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models (PyTorch Docs)
- URL: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html
- Type: docs
- Date: 2024–2025 (current)
- Relevance: 7/10
- Novelty: 6/10
- Summary: AOTInductor compiles torch.export-ed models ahead of time, producing shared libraries (.so) that can be deployed without the Python runtime. The compiled artifact embeds device-specific code (CUDA kernels, CPU SIMD code) selected at AOT compile time — the deployment target is fixed at build time, not at inference time. Torch-TensorRT integrates by embedding TRT engines in the same package alongside Inductor-compiled fallback subgraphs.
- Key detail: AOTInductor's device selection is compile-time only — a deployed .so targets exactly one device type, making it the opposite of dynamic heterogeneous dispatch. Runtime multi-device would require separate .so artifacts per target.

---

### 8. Introduction to torch.compile and How It Works with vLLM (vLLM Blog)
- URL: https://blog.vllm.ai/2025/08/20/torch-compile.html
- Type: blog (engineering)
- Date: August 2025
- Relevance: 7/10
- Novelty: 7/10
- Summary: Describes vLLM's integration of torch.compile with Inductor, including custom Inductor compiler passes for kernel fusion (attention fusion, MoE routing fusion) and the interaction with CUDA Graphs for low-overhead dispatch. vLLM adds its own shape-static autotuning path on top of Inductor, benchmarking Triton template variants per shape. The 2025 improvements include stopping recompilation of identical artifacts and MoE cold-start optimization.
- Key detail: vLLM's custom Inductor passes apply operator fusion at the FX graph level before lowering — demonstrating that the pre-lowering (device-agnostic) Inductor pass stage is an insertion point for application-level dispatch policy, not just vendor-level dispatch.

---

### 9. KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents (PyTorch Blog)
- URL: https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/
- Type: blog/paper (Meta)
- Date: 2025
- Relevance: 6/10
- Novelty: 9/10
- Summary: KernelFalcon is an agentic system (GitHub: meta-pytorch/KernelAgent) that generates verified Triton kernels for arbitrary subgraph specifications via a parallel worker pool with early-exit on first passing candidate. A Dispatcher coordinates parallel TritonKernelAgents per subgraph; failed candidates trigger isolated error feedback without polluting other workers' contexts. Achieves 100% correctness on all 250 KernelBench L1/L2/L3 tasks.
- Key detail: The Dispatcher's "parallel exploration with isolated contexts and early exit" pattern is a dispatch architecture primitive — generating and selecting among competing kernel implementations at compile time, which is structurally analogous to libkdl's runtime multi-target selection but operating pre-deployment.

---

## Angle Assessment

- **Coverage:** PyTorch Inductor's dispatch architecture is actively documented via RFCs and TAC discussions, with the `register_backend_for_device` API being the primary extension point. However, *runtime* (launch-time) heterogeneous dispatch across multiple vendor backends on the same live system is an acknowledged gap with no current solution in the PyTorch ecosystem.

- **Surprise findings:**
  1. The TAC thread (source 5) confirms that Inductor's graph fusion passes are already device-agnostic — the only device-specific code is at the lowering/codegen stage. This is a stronger separation of concerns than expected and aligns with libkdl's design premise.
  2. Intel Gaudi chose to register at the Dynamo level rather than the Inductor level, suggesting the Inductor backend API has friction that drives some vendors to a higher-level integration point.
  3. KernelFalcon's parallel-exploration-with-early-exit pattern (source 9) is an emergent kernel dispatch architecture — selecting among competing kernel variants by parallel compilation and correctness verification, not by static policy.
  4. AOTInductor's fixed-at-compile-time device selection (source 7) creates a hard boundary: deployed artifacts are single-target, requiring separate binaries per hardware — the exact problem libkdl proposes to solve at runtime.

- **Gaps:**
  - No PyTorch mechanism dispatches the *same compiled kernel* to multiple hardware targets at runtime (e.g., CUDA vs. ROCm vs. CPU based on availability or load). All current dispatch is compile-time.
  - No runtime cost model or load-aware selector in Inductor — backend selection is fully static, driven only by tensor device type.
  - The `DeviceOpOverrides` abstraction (source 3) is proposed but not yet widely adopted, meaning CUDA-specific assumptions still leak into nominally device-agnostic Inductor code.
  - No publicly documented design for dispatching across mixed-vendor GPU pools within a single inference pass (e.g., operator A on NVIDIA, operator B on AMD in one forward pass).

- **Suggested follow-up angles:**
  1. **Triton/SPIR-V as convergence IR** — Intel XPU reuses TritonScheduling via SPIR-V; assess whether SPIR-V is a viable universal portable kernel IR for libkdl (search angle: "Triton SPIR-V backend Intel AMD portable kernel IR").
  2. **Runtime cost models for kernel selection** — literature on shape-aware, load-aware backend selection at dispatch time; complements libkdl's dispatch policy layer (search angle: "GPU kernel selection cost model shape aware dispatch").
  3. **Halide backend in Inductor** — the first third-party Inductor backend to land upstream; study its architecture for lessons on clean out-of-tree backend registration (search angle: "Inductor Halide backend cpu_backend cuda_backend implementation").
  4. **Dynamo-level vs. Inductor-level integration tradeoffs** — why Gaudi registered at Dynamo vs. Inductor; implications for libkdl's integration point in the compiler stack.
  5. **AOTInductor multi-target packaging** — proposals or workarounds for shipping one .so targeting multiple hardware (search angle: "AOTInductor multi-device fat binary portable deployment").
