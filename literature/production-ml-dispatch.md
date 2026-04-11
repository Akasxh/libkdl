# Production ML Framework Kernel Dispatch — Research Findings

*For LLVM Dublin 2026 poster: "Heterogeneous GPU Kernel Dispatch via MLIR"*
*Addresses reviewer objection: "ML kernels are well known at compile time"*
*Compiled: 2026-04-02*

---

## Executive Summary

The "ML is static" argument fails in a large and growing class of real deployments. This document establishes, with citations, that production ML systems perform substantial runtime dispatch — hardware introspection, capability-conditioned kernel selection, graph partitioning across execution providers, and request-level GPU routing. The gap our work addresses (a principled, vendor-agnostic runtime dispatch layer) is real and unresolved.

---

## 1. PyTorch Dispatcher: Runtime Kernel Selection Is the Default

### 1.1 Architecture

The PyTorch dispatcher is a mandatory routing layer that runs on *every* operator call. It maintains a 2D table indexed by (operator, DispatchKey) and selects kernels by computing the current active DispatchKeySet at runtime.

The DispatchKeySet is computed by unioning four independent inputs at call time:
1. **Tensor contributions** — each tensor argument contributes its backend bits (CPU, CUDA, HIP, XLA, MPS, HPU, etc.)
2. **Local include set** — thread-local state for modal transforms (tracing, vmap)
3. **Global set** — always-active keys
4. **Local exclude set** — masks applied by RAII guards to prevent reprocessing

The dispatcher then picks the highest-priority active bit and transfers control to that handler. This means dispatch is **not** compile-time: it depends on the live tensor metadata and thread-local state at the moment of the call.

Source: [E. Yang, "Let's talk about the PyTorch dispatcher," ezyang's blog, Sept 2020](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)

### 1.2 DispatchKey Priority and the Backend Component System

DispatchKeys are organized into two tiers encoded in a `uint8_t` bitset:

- **Lower ~12 bits — BackendComponent**: CPU, CUDA, HIP, XLA, MPS, IPU, XPU, HPU, VE, Lazy, MTIA, MAIA, ...
- **Upper bits — Functionality**: Autograd, Dense, Sparse, Quantized, AutogradFunctionality, Batched (vmap), Tracer, ...

When a functionality bit is "customizable per backend," the dispatcher additionally inspects the BackendComponent bits and uses the highest set bit to resolve the backend. This allows, e.g., `AutogradCUDA` to be distinct from `AutogradCPU`.

The typical dispatch sequence for a CUDA tensor operation is:
```
Autograd → [guard excludes Autograd] → BackendSelect (if needed) → CUDA kernel
```

For tracing or vmap transforms, the sequence inserts additional handlers before the backend step using thread-local guards (`IncludeDispatchKeyGuard`).

Source: [PyTorch Dispatcher Tutorial, docs.pytorch.org](https://docs.pytorch.org/tutorials/advanced/dispatcher.html); [c10/core/DispatchKey.h source](https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h)

### 1.3 Fallback and Boxing Mechanisms

Three registration levels fill the dispatch table, in descending priority:
1. **Exact registration** — specific operator + specific dispatch key
2. **Catch-all kernel** — covers entire operator row (all dispatch keys)
3. **Boxed fallback** — covers entire dispatch key column (all operators)

Boxed fallbacks use an `IValue`-based calling convention that allows a single kernel to handle any operator without code generation. This is how PyTorch's built-in transforms (autograd, tracing, vmap) are implemented: once globally, applied universally. A new backend that registers some kernels automatically inherits all boxed fallbacks for unregistered ops — those fall through to CPU execution.

Key insight for our work: PyTorch's dispatch architecture is *explicitly designed* to be extensible to new backends at runtime without recompilation of the framework. The mechanism we want to build (vendor-agnostic dispatch) is exactly what the lower levels of PyTorch already do — we would be extending it upward to the MLIR compilation boundary.

Source: [E. Yang, "Let's talk about the PyTorch dispatcher"](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)

---

## 2. torch.compile and TorchInductor: Compile-Time Meets Runtime

### 2.1 The Compilation Stack

`torch.compile` adds a JIT compilation layer above the dispatcher:

- **TorchDynamo** (frontend): intercepts Python bytecode, traces through the model, captures `torch.fx` graphs. Extracts static computation structure while noting all "graph breaks" (Python control flow depending on tensor values).
- **TorchInductor** (backend): takes fx graphs, runs fusion passes, calls Triton to generate GPU kernels, and optionally wraps the result in CUDA Graphs.

The default backend is `inductor`. Available backends (per `torch.compiler.list_backends()`): `inductor`, `cudagraphs`, `onnxrt`, `openxla`, `openxla_eval`, `tvm`. Third-party backends (TensorRT, HPU) register via an out-of-core plugin mechanism.

Source: [torch.compiler documentation, PyTorch 2.11](https://docs.pytorch.org/docs/stable/torch.compiler.html); [torch.compile Inductor/Triton backend on AMD ROCm, rocm.blogs.amd.com](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)

### 2.2 Why torch.compile Does NOT Eliminate Runtime Dispatch

Despite performing ahead-of-time compilation, torch.compile preserves multiple runtime dispatch points:

**Dynamic shapes recompilation**: When `dynamic=False` (default), the system installs *guards* — assertions on tensor shapes and strides. If a guard fails (e.g., a new batch size arrives), Dynamo recompiles. The cache lookup is a runtime dispatch: the system checks guards at every invocation and selects the matching compiled artifact. Cache size is bounded by `torch._dynamo.config.recompile_limit` (default: 8); beyond that, execution falls back to eager mode.

**`dynamic=True` mode**: Dynamo promotes tensor dimensions to symbolic SymInts (tracked by a ShapeEnv using Sympy). Triton kernels are generated with dynamic numel arguments (`xnumel`, `rnumel` as runtime values). A guard-based mechanism installs divisibility checks — if `numel % 16 == 0` can be proven statically, Triton emits vectorized loads (`LDG.E.128`); if not, it falls back to scalar loads with 1.5-3.4× overhead. This is runtime-driven kernel variant selection.

Source: [Dynamic Shapes Core Concepts, PyTorch docs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html); [PR #176653, pytorch/pytorch](https://github.com/pytorch/pytorch/pull/176653)

### 2.3 Backend Selection by Users

Backend selection is a runtime configuration decision:

```python
torch.compile(model, backend="inductor")        # default: Triton kernels
torch.compile(model, backend="onnxrt")          # ONNX Runtime EP chain
torch.compile(model, backend="openxla")         # XLA (TPU/GPU)
torch_tensorrt.compile(model, ...)              # TensorRT engine
```

Critically, `torch.compile` itself does not perform hardware introspection to select the best backend automatically. Users must make this choice manually based on the target hardware. This is exactly the gap our dispatch layer would fill.

Source: [torch.compiler documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html); [Torch-TensorRT backend docs](https://docs.pytorch.org/TensorRT/dynamo/torch_compile.html)

---

## 3. CUDA Graphs: Aggressive Optimization That Creates Static Constraints

### 3.1 What CUDA Graphs Do

CUDA Graphs batch kernel launches into a static DAG that is captured once and replayed with zero CPU launch overhead. PyTorch integrates CUDA Graphs through two mechanisms:

- **Standard CUDA Graphs**: Full graph capture for static-shape inference
- **CUDAGraph Trees**: PyTorch's `torch.compiler_cudagraph_trees` — a more sophisticated system that handles partial graph invalidation and replay

For CUDA Graphs to function correctly, all tensor *addresses* must remain fixed across replays (memory is pre-allocated). This means CUDA Graphs fundamentally require fixed:
1. Tensor shapes (determines allocation sizes)
2. Memory addresses (captured into the graph)
3. Kernel launch parameters

Source: [CUDAGraph Trees, PyTorch 2.9 docs](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html); [PyGraph: Robust Compiler Support for CUDA Graphs, arXiv:2503.19779](https://arxiv.org/html/2503.19779v1)

### 3.2 Why CUDA Graphs Break for Dynamic ML Workloads

CUDA Graphs require re-recording for *every unique input shape*. Re-recording costs:
- **64 KB of device memory per kernel launch** in the captured graph
- **CPU-side graph capture overhead** (proportional to graph complexity)
- **Loss of all batching benefits** during re-recording

For LLM inference specifically, the decode phase generates one token per step, with KV-cache lengths growing at every step — a continuously varying shape. vLLM's solution is "piecewise CUDA Graphs" that capture only the segments between attention operations, while attention itself runs in eager mode. This hybrid approach is necessary precisely because CUDA Graphs cannot handle the dynamic KV-cache shapes.

The practical workaround for serving systems is bucketing: padding inputs to a small set of fixed shapes (e.g., batch sizes 1, 2, 4, 8, 16, 32) and maintaining one CUDA Graph per bucket. This is a form of multi-version dispatch — selecting among pre-compiled static graphs based on runtime input characteristics.

Source: [vLLM torch.compile integration docs](https://docs.vllm.ai/en/latest/design/torch_compile/); [Grape: Practical CUDA Graph Execution for Dynamic DNNs, ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3613424.3614248); [NVIDIA Developer Forums on CUDA Graph best practices](https://forums.developer.nvidia.com/t/best-practices-for-optimizing-pytorch-models-with-cuda-graphs/351177)

### 3.3 Connection to Our Work

The CUDA Graphs limitation is a concrete argument that runtime dynamic dispatch is needed. When a model is served across mixed hardware (A100 + H100 + L4) with variable-length sequences, static graph capture is insufficient. Our proposed dispatch layer would operate *above* the CUDA Graph level, routing to the correct pre-compiled kernel variant based on hardware + shape, generalizing the bucketing pattern across vendor boundaries.

---

## 4. cuDNN and cuBLAS: The Industry Standard for Runtime Kernel Selection

### 4.1 cuBLAS: ML-trained Heuristic Dispatch

cuBLAS does not contain a single GEMM implementation — it contains **hundreds**. For square matrices up to 4096×4096, there are 16 different SGEMM kernel implementations. The selection mechanism is a *trained recommender system*:

> "cuBLAS library leverages a recommender system at runtime to dispatch the fastest configuration possible for any user-requested matmuls. Each configuration includes implementations (kernels) and runtime launch parameters. This recommender system is trained on actual timing data from running a large number of problems (including multiple precisions, matrix shapes, layouts and epilogues) with several available configurations on the GPU."

The recommender achieves **93% of optimal performance** (geometric mean) across the full problem space. This is explicit, documented, production-grade runtime kernel dispatch happening inside what appears to be a simple BLAS call.

Hardware introspection in cuBLAS:
- **`cublasSetSmCountTarget()`**: Override the number of streaming multiprocessors for the library's dispatch heuristic. Used when cuBLAS kernels run concurrently with other CUDA streams — e.g., on an A100 with 108 SMs, if another kernel occupies 8 SMs, calling `cublasSetSmCountTarget(100)` causes cuBLAS to select kernels optimized for 100 SMs.
- **Architecture detection**: Certain algorithm modes "have no effect on NVIDIA Ampere architecture GPUs and newer" — the library detects SM version and branches at runtime.
- **Multi-stream adaptation**: "When multiple concurrent CUDA streams are active, the library may optimize total performance by picking different internal implementations."
- **Auto-tuning mode** (`CUBLAS_GEMM_AUTOTUNE`): Benchmarks available algorithms and caches results per problem size.

Source: [cuBLAS documentation, NVIDIA](https://docs.nvidia.com/cuda/cublas/index.html); [GPU Glossary: cuBLAS, modal.com](https://modal.com/gpu-glossary/host-software/cublas); [Introducing Grouped GEMM APIs in cuBLAS, NVIDIA blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)

### 4.2 cuDNN v9: Runtime Compilation and Heuristics

cuDNN v9 introduces a graph API with explicit runtime kernel selection:

**Three heuristic modes**:
- `CUDNN_HEUR_MODE_A` — Fast heuristic, low CPU overhead, covers most patterns. Returns engine configs ranked by predicted performance.
- `CUDNN_HEUR_MODE_B` — Higher accuracy, higher CPU latency. Falls back to Mode A when Mode A is known to be better.
- `CUDNN_HEUR_MODE_FALLBACK` — Prioritizes correctness over performance; functional guarantee without performance expectation.

**Runtime fusion engines** (`CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION`): These generate CUDA kernels *at runtime* using NVRTC, compiling specialized fused kernels based on the specific graph pattern and the live hardware's compute capability. The cuDNN JIT configuration supports only these runtime-compiled engines, drastically reducing binary size at the cost of a compilation step at first use.

**Hardware capability-based support surfaces**:
- Support Surface 90: compute capability ≥ 9.0 (H100/Blackwell)
- Support Surface 80: compute capability ≥ 8.0 (Ampere)
- Support Surface 70: compute capability ≥ 7.0 (Volta/Turing)

The runtime selects the appropriate support surface by querying the device's SM version, enabling the same cuDNN binary to dispatch correctly across generations. Newer architectures (e.g., Blackwell) use exact target architecture flags (`-arch=sm_90a`) when NVRTC-compiling kernels.

Source: [cuDNN v9 Graph API developer guide](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.0/developer/graph-api.html); [cuDNN v9.0.0 release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.0.0/release-notes.html)

### 4.3 Implication for the "ML is Static" Argument

The reviewer's argument assumes kernels are selected at compile time. cuBLAS and cuDNN demonstrate that even for a *single operator* (GEMM, convolution), production systems perform runtime hardware introspection and dispatch to one of hundreds of kernel variants. Our work generalizes this to the *cross-vendor* case.

---

## 5. vLLM: Multi-GPU Dispatch for LLM Serving

### 5.1 Core Parallelism Architecture

vLLM decomposes multi-GPU dispatch into three orthogonal strategies:

- **Tensor Parallelism (TP)**: Weight matrices split across GPUs; column-parallel GEMM, row-parallel GEMM with an AllReduce. Requires high-bandwidth intra-node interconnect. Rule of thumb: TP size = GPUs per node.
- **Pipeline Parallelism (PP)**: Model layers split sequentially across GPU groups. Reduces memory per device but adds inter-stage latency; vLLM uses micro-batch scheduling to fill pipeline bubbles.
- **Expert Parallelism (EP)** for MoE models: Tokens dispatched to expert-holding GPUs via All-to-All communication. vLLM implements FP4 quantization of dispatch activations, reducing All-to-All traffic by 4× versus FP16.

The distributed runtime uses either Ray (multi-node) or Python native multiprocessing (single-node).

Source: [vLLM Distributed Inference docs v0.8.0](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html); [vLLM MoE Playbook, ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html); [vLLM blog on DeepSeek-R1 WideEP](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html)

### 5.2 torch.compile Integration in vLLM V1

vLLM V1 enables `torch.compile` by default. The integration exposes several dispatch decisions made at runtime:

**Piecewise CUDA Graphs**: The computation graph is split at `torch.ops.vllm.unified_attention_with_output` boundaries. Segments between attention ops are captured as CUDA Graphs (static, efficient). Attention ops run in eager mode (dynamic KV-cache length). This is a runtime dispatch: at each decode step, vLLM selects the CUDA Graph artifact matching the current batch size bucket.

**Dynamic Shape Modes**:
- `BACKED` (default): Installs guards; drops guards when unsafe to preserve performance.
- `UNBACKED`: Maximum guard safety, conservative.
- `BACKED_SIZE_OBLIVIOUS`: Experimental.

Only batch size varies at runtime; weights are static. This insight drives the compilation strategy: compile for dynamic batch, specialize weights.

**Compilation Caching**: Artifacts stored in `~/.cache/vllm/torch_compile_cache`, keyed by hashes of model config, compiler config, and model code. Enables reuse across autoscaling replicas (warm start benefit).

**Custom compiler passes** producing measurable gains:
- Attention + Quantization fusion: up to 7% improvement
- AllReduce + RMSNorm fusion: up to 15% improvement
- Sequence Parallelism + Async TP: up to 10% improvement

Source: [vLLM torch.compile integration design doc](https://docs.vllm.ai/en/latest/design/torch_compile/); [vLLM blog: Introduction to torch.compile and How It Works with vLLM](https://vllm.ai/blog/torch-compile)

### 5.3 vLLM and Heterogeneous Hardware

Standard vLLM assumes **homogeneous GPU clusters**. Heterogeneous serving (mixing GPU types) is an active research problem, not a solved one in the vLLM codebase.

User-facing evidence: The vLLM community forum has a thread titled "How to run a model use heterogeneous GPUs" where the answer is essentially: vLLM does not natively support this. Users must provision separate vLLM instances per GPU type and route requests externally.

Research systems built on top of vLLM to address this:
- **Helix** (ASPLOS 2025) — see Section 8
- **Hetis** (2025) — see Section 8

This is a direct argument for our work: the dominant production LLM serving system has no native heterogeneous dispatch, and the gap is being filled by research systems that require bespoke orchestration.

Source: [vLLM Forums: heterogeneous GPUs](https://discuss.vllm.ai/t/how-to-run-a-model-use-heterogeneous-gpus/1360)

---

## 6. TensorRT-LLM: Static GPU Binding as a Portability Problem

### 6.1 Engine Building

TensorRT-LLM converts models into TensorRT engines: GPU-specific, serialized computation graphs with kernels selected, fused, and parameterized for a specific GPU architecture. The build step:

1. Parses the model
2. Runs NVIDIA's graph optimizer (layer fusion, precision selection)
3. Profiles available kernel implementations against the target GPU
4. Selects the fastest configuration per operation
5. Serializes the result into a `.engine` file

The engine bakes in: target compute capability (sm_xx), tensor core availability, SM count, memory bandwidth assumptions, and quantization layout.

Source: [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM); [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

### 6.2 Portability Limitations

By default, **serialized engines are not portable across GPU architectures or operating systems**. An engine built for sm_89 (Ada/RTX 4090) will not run on sm_90 (Hopper/H100). Specific constraints:

- Engines require TensorRT version match
- Engines require the same SM version as build GPU
- Platform (Linux/Windows) must match
- Refitting (weight update without rebuild) requires identical SM version

**Hardware Compatibility Mode**: An opt-in feature that allows engines to run on multiple GPU generations. However, this sacrifices performance — the engine cannot use architecture-specific features (e.g., H100's Transformer Engine, FP8 Tensor Cores) not present on the lowest-common-denominator target.

**TensorRT for RTX** offers "build once, deploy anywhere" across Ampere, Ada, and Blackwell NVIDIA RTX GPUs — but this is explicitly a consumer-focused product with reduced optimization vs. data center TensorRT.

Source: [TensorRT architecture overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html); [TensorRT capabilities](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html); [CPU-only AOT and TensorRT-RTX engines](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/cpu-engines.html)

### 6.3 Implication

TensorRT-LLM is state of the art for NVIDIA-only deployments. It is architecturally unable to serve AMD GPU users, CPU-only nodes, or mixed-vendor clusters. Each new GPU generation requires a rebuild. This is the exact problem our dispatch layer addresses at a more fundamental (MLIR) level.

---

## 7. ONNX Runtime: The Runtime Dispatch Model We Should Learn From

### 7.1 Execution Provider Architecture

ONNX Runtime (ORT) implements runtime dispatch via **Execution Providers (EPs)**, a plugin interface that cleanly separates model semantics from execution backend. The EP ecosystem includes 14+ providers:

- GPU: CUDA EP (uses cuDNN/cuBLAS), TensorRT EP (uses TRT engine), ROCm EP, MIGraphX EP
- CPU: Default CPU EP (always present as fallback), oneDNN EP, XNNPACK EP
- Edge/Mobile: QNN (Qualcomm), NNAPI (Android), CoreML (Apple), ARM solutions
- Specialized: Azure, Vitis-AI (Xilinx), CANN (Huawei)

Source: [ORT Execution Providers docs](https://onnxruntime.ai/docs/execution-providers/)

### 7.2 Runtime Graph Partitioning via GetCapability

The dispatch mechanism is explicit and principled:

1. **Session initialization** — ORT converts the ONNX graph into its internal representation and runs provider-independent graph optimizations.
2. **Graph partitioning** — EPs are evaluated in priority order (user-specified or default). Each EP's `GetCapability()` method is called; the EP returns `ComputeCapability` objects describing which nodes/subgraphs it can handle. ORT assigns each EP the *maximal* subgraph it claims.
3. **Compilation** — Each partition is compiled by calling the EP's `Compile()` method, producing a fused custom operator.
4. **Default CPU fallback** — The CPU EP processes all unclaimed nodes. ORT guarantees that the CPU EP handles all ONNX operators.

This partitioning is performed **once at session creation**, not per-inference. The runtime graph is then a static sequence of subgraph calls, each routed to its assigned EP.

Source: [ORT Architecture / High Level Design](https://onnxruntime.ai/docs/reference/high-level-design.html); [ORT Choosing Execution Providers for Performance](https://pkreg101.github.io/onnxruntime/docs/performance/choosing-execution-providers.html)

### 7.3 Fallback Behavior and Known Pathologies

**Silent fallbacks** are a documented production problem in ORT. If a node cannot be handled by the preferred EP, it silently falls back to the next EP in the priority chain. Common symptom: "Some nodes were not assigned to the preferred execution providers which may or may not have a negative impact on performance."

This silent demotion has caused real-world performance bugs — GPU-accelerated sessions running large fractions of computation on CPU, with inference speed barely faster than CPU-only. The community has proposed making fallback opt-in (disabled by default), with explicit error on fallback.

**TensorRT EP fallback**: When TensorRT cannot parse a subgraph (unsupported op, dynamic shape beyond TRT's capabilities), it falls back to CUDA EP. If CUDA EP cannot handle the op, it falls back to CPU EP. Each fallback crosses a device boundary, causing PCIe transfers.

Source: [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html); [GitHub issue: fallback provider logic bug](https://github.com/microsoft/onnxruntime/issues/25145); [GitHub discussion: disabling fallback by default](https://github.com/onnx/onnx/discussions/6623)

### 7.4 How ORT Informs Our Design

ORT's GetCapability pattern is the closest existing analogue to what we propose. Key differences and improvements our system would make:

| Dimension | ONNX Runtime | Our Proposed System |
|---|---|---|
| Dispatch granularity | ONNX node / subgraph | MLIR operation / dialect region |
| When dispatch is decided | Session creation (once) | Potentially per-invocation with runtime hardware query |
| Vendor coverage | Plugin-based, not uniform | Uniform via MLIR lowering paths |
| Kernel origin | Vendor libraries (cuDNN, etc.) | MLIR-generated, JIT-compiled Triton/HIP/SPIR-V |
| Cross-vendor portability | No (each EP is vendor-specific) | Yes (single source, multiple lowering targets) |
| Hardware introspection | None (EP selection is manual) | Built into dispatch: query SM count, architecture, memory |

---

## 8. Heterogeneous Cluster Serving: Research Systems Proving the Need

### 8.1 The Homogeneity Assumption Is Breaking Down

> "Most existing LLM serving systems like Orca and vLLM target homogeneous GPU clusters where all GPUs are of the same type with identical memory capacity and compute resources."

This is the status quo. And it is increasingly untenable:

- **Hardware generation cycles** now outpace cluster refresh cycles. A 2-year-old cluster may mix V100, A100, and H100 GPUs.
- **Cloud spot instances** offer 60-91% cost savings but provide heterogeneous availability: the GPU type available depends on region, time, and bidding.
- **Cost efficiency varies radically by workload**: workstation GPUs (A40, A6000, L40) offer 1.2× higher memory bandwidth and 1.8× greater memory capacity per dollar vs. data center GPUs. Compute-bound prefill favors H100; memory-bound decode favors L40.

Source: [Helix ASPLOS 2025](https://arxiv.org/abs/2406.01566); ["Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs", arXiv:2502.00722](https://arxiv.org/html/2502.00722v1); [SkyServe arXiv:2411.01438](https://arxiv.org/abs/2411.01438)

### 8.2 Helix (ASPLOS 2025): Max-Flow Dispatch Over Heterogeneous GPUs

**Problem**: Serve LLMs over a cluster with 7+ different GPU types (A100, L4, T4, etc.) and heterogeneous network topology.

**Approach**: Model the cluster as a directed weighted graph (nodes = GPU instances; edge capacities = min(compute token/s, network bandwidth/transmission size)). Formulate LLM inference as a max-flow problem. Solve with MILP to find optimal model layer placement across heterogeneous GPUs. At runtime, use Interleaved Weighted Round-Robin to dispatch requests to per-request pipeline paths.

**Key mechanism**: Each inference request gets its *own* pipeline path through the cluster (per-request pipelines), as opposed to static pipeline partitioning.

**Results** (evaluated on 24–42 node clusters):
- Throughput: up to **3.3× improvement** over vLLM baseline
- Prompt latency: up to **66% reduction**
- Decode latency: up to **24% reduction**

Source: [Helix paper, arXiv:2406.01566](https://arxiv.org/abs/2406.01566); [Helix source code](https://github.com/Thesys-lab/Helix-ASPLOS25); [ACM DL](https://dl.acm.org/doi/10.1145/3669940.3707215)

### 8.3 Hetis (2025): Fine-Grained Dynamic Parallelism

**Problem**: Heterogeneous clusters mixing high-end (A100 80GB), mid-range (3090 24GB), and legacy (P100 12GB) GPUs.

**Observation**: The P100 is 24.5× slower than A100 on MLP computation. Naive uniform assignment wastes fast GPUs waiting for slow ones.

**Approach**: Two-tier dispatch:
1. **Primary Workers**: Run compute-intensive MLP and prefill Attention on high-end GPUs only.
2. **Dynamic Attention Parallelism**: Distribute Attention computation at *individual attention head* granularity across all GPUs including low-end, via a linear programming formulation solved per request batch.

The Dispatcher solves the LP optimization at runtime, accounting for network transfer overhead between primary/attention workers, per-device computation time (profiled), and KV cache memory constraints.

**Results** (A100 + 3090 + P100 cluster):
- Throughput: **2.25× vs. Splitwise**, **1.33× vs. Hexgen**
- P95 TTFT latency: **1.47× better**
- KV cache availability: **1.87× improvement**
- Prediction accuracy of computation time model: **93.8%**

Source: [Hetis paper, arXiv:2509.08309](https://arxiv.org/abs/2509.08309)

### 8.4 Demystifying Cost-Efficiency (2025)

Key quantitative finding: "Selecting the most appropriate GPU type for specific workloads and models can enhance cost-efficiency by up to **2.27×**."

Ablation results: Suboptimal deployment configuration reduces throughput by **34%**; suboptimal workload assignment reduces throughput by **32%**. Joint optimization (GPU type + parallelism config + request routing) achieves up to **41% throughput improvement** over homogeneous baselines.

This establishes that runtime routing of requests to matching GPU types has a first-order impact on efficiency — not a marginal effect.

Source: [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs, arXiv:2502.00722](https://arxiv.org/html/2502.00722v1)

---

## 9. torch.compile Backend Selection: The User-Facing Dispatch API

### 9.1 Current State

Users invoke backend selection via:

```python
# Using inductor (default) — generates Triton GPU kernels
model_opt = torch.compile(model, backend="inductor")

# Using ONNX Runtime — delegates to ORT EP chain
model_opt = torch.compile(model, backend="onnxrt")

# Using OpenXLA — for TPU or JAX-compatible backends
model_opt = torch.compile(model, backend="openxla")

# Using TensorRT (external plugin)
import torch_tensorrt
model_opt = torch.compile(model, backend="torch_tensorrt")
```

Backend choice is manual, static per process invocation, and requires user knowledge of target hardware. There is no automatic backend selection based on detected hardware.

### 9.2 Backend Comparison

| Backend | Target | Runtime Dispatch? | Hardware Introspection? |
|---|---|---|---|
| `inductor` | NVIDIA/AMD/Intel GPU, CPU | No (Triton target fixed at compile) | Partial (Triton tunes via autotuner) |
| `onnxrt` | Multi-EP fallback chain | Yes (GetCapability per session) | No (user specifies EP priority) |
| `openxla` | TPU, GPU via XLA | No | No |
| `torch_tensorrt` | NVIDIA GPU only | No (engine is GPU-specific) | No |
| `cudagraphs` | NVIDIA GPU | No (shape-bucketed replay) | No |

The absence of automatic cross-vendor backend selection is the precise gap our work addresses.

Source: [torch.compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html); [RFC: Moving torch.compile backends out of core, pytorch/pytorch #109687](https://github.com/pytorch/pytorch/issues/109687)

### 9.3 The "Inductor Assumption" Problem

The PyTorch 2025 ecosystem has converged on Inductor + Triton as the reference performance path. As one analysis notes: "The efficacy of a platform in 2025 is largely determined by how well it services the Inductor-Triton pipeline."

Consequence: AMD/ROCm uses a Triton-compatible compiler (HIP Triton). Intel GPU uses a similar path. But CPU-only inference, ARM NPUs, and AMD-only deployments all require different backends. When a model is deployed to unknown hardware, the user must manually select the backend — or accept suboptimal performance or breakage.

Source: [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)

---

## 10. Where Runtime Dispatch Adds Value: The Case Against "ML is Static"

This section directly rebuts reviewer 91B's argument.

### 10.1 Multi-GPU Serving With Mixed Hardware

**Evidence**: Helix (3.3× throughput), Hetis (2.25× throughput), cost-efficiency analysis (2.27× efficiency). These are peer-reviewed systems, deployed and benchmarked on real clusters. The dispatch problem is not hypothetical.

**Mechanism**: When GPU types differ, compute-optimal kernel configurations differ too. A matrix multiplication on A100 vs. P100 requires different tile sizes, different SM occupancy targets, different memory access patterns. A static kernel compiled for A100 runs at a fraction of peak on P100 and vice versa.

### 10.2 Heterogeneous Cloud Clusters

**Evidence**: Cloud providers (AWS, GCP, Azure) offer 15+ GPU instance types, and spot availability varies dynamically. SkyServe reduces serving cost by 43% on average using spot instances. But spot instances may deliver V100 when you expected A100 — requiring runtime kernel adaptation.

**ML pipeline reality**: A company running inference on AWS may have `p3.8xlarge` (V100), `p4d.24xlarge` (A100), and `p5.48xlarge` (H100) instances in the same autoscaling group, all running the same Docker image. A static compiled engine works on at most one of these.

Source: [SkyServe, arXiv:2411.01438](https://arxiv.org/abs/2411.01438); [SpotServe, arXiv:2311.15566](https://arxiv.org/abs/2311.15566)

### 10.3 Edge Deployment on Unknown Hardware

**Evidence**: PyTorch ExecuTorch 1.0 (released October 2025) targets 12+ hardware backends — CPU, GPU, NPU — across smartphones, embedded systems, and microcontrollers. The challenge is explicitly that the deploying engineer does not know, at model export time, which NPU will be present on the end user's device.

ExecuTorch's backend delegation mechanism allows partial graph delegation at export time, with fallback to CPU for unsupported subgraphs. However, the delegation is static per build: there is no runtime hardware query that adapts to the specific device variant.

**Concrete example**: Qualcomm QNN EP works on devices with Hexagon DSP. Samsung Exynos NPU EP works on Exynos chips. An app deployed to both device families needs either a per-device binary or a runtime dispatch layer.

Source: [ExecuTorch 1.0 announcement, PyTorch blog](https://pytorch.org/blog/introducing-executorch-1-0/); [Edge AI Vision Alliance, ExecuTorch 1.0 analysis](https://www.edge-ai-vision.com/2025/10/bringing-edge-ai-performance-to-pytorch-developers-with-executorch-1-0/)

### 10.4 Model Portability Across Cloud Providers

**The static argument's implicit assumption**: The user deploys to exactly one GPU type. This is valid for single-datacenter, single-GPU-vendor, dedicated-hardware deployments.

**When this breaks**: Multi-cloud strategies (AWS + GCP + Azure), org-level heterogeneity (ML team uses A100, inference team uses T4), model migration during GPU generation transitions, and open-source model distribution (model weights published to Hugging Face must run on every GPU type a user might have).

**Current workaround**: Maintain N separate compiled artifacts (N = number of GPU types in deployment). This is O(N) maintenance overhead and requires knowing N at model release time. Our dispatch layer reduces this to O(1) by selecting the appropriate compilation path at runtime.

### 10.5 LLM Inference Dynamic Shapes Are the Counter-Example

Reviewer 91B's claim ("ML kernels are well known at compile time along with tensor sizes") applies to training and offline batch inference with fixed batch sizes. It fails for:

- **Auto-regressive generation**: Each decode step has a different KV-cache length (grows by 1 per step). Variable sequence lengths in a batch.
- **Speculative decoding**: Draft and verification steps have different sequence lengths.
- **Continuous batching** (vLLM, TGI): Batch size changes at every scheduler tick as requests arrive and complete. No fixed batch size exists.
- **RAG pipelines**: Retrieved context length varies per query.

vLLM's piecewise CUDA Graphs and guard-based recompilation in torch.compile are engineering solutions to this fundamental dynamism. Our dispatch layer adds the cross-vendor dimension to this existing dynamism.

---

## 11. Synthesis: Where Our Contribution Fits

The production ML ecosystem reveals a clear pattern:

**Each layer does runtime dispatch, but only within one vendor:**

| Layer | Does Runtime Dispatch? | Cross-Vendor? |
|---|---|---|
| cuBLAS kernel selection | Yes (ML-trained heuristic, SM-aware) | No (NVIDIA only) |
| cuDNN heuristics (Mode A/B) | Yes (NVRTC compilation at runtime) | No (NVIDIA only) |
| PyTorch dispatcher | Yes (DispatchKeySet computed per call) | Yes (CPU/CUDA/XLA/MPS/HPU) but not at kernel level |
| torch.compile + Inductor | No (Triton target fixed at compile) | Partial (Triton supports AMD/Intel too) |
| ONNX Runtime GetCapability | Yes (partition at session init) | Yes (14+ EPs) but no cross-EP kernel fusion |
| vLLM + CUDA Graph buckets | Yes (bucket selection per call) | No (NVIDIA only) |
| TensorRT-LLM engine | No (baked in at build time) | No (NVIDIA only) |

**The gap**: No system performs MLIR-level kernel compilation with runtime hardware introspection to select among vendor-optimal lowering paths (PTX/SASS for NVIDIA, GCN/ISA for AMD, SPIR-V for cross-vendor, LLVM IR for CPU), within a single unified source representation.

**Our contribution** (as the poster will argue): A dispatch layer that:
1. Accepts kernels in MLIR linalg/tensor dialect (vendor-agnostic)
2. At deployment time, queries hardware capabilities (SM version, vendor, memory bandwidth, SM count)
3. Selects and JIT-compiles the appropriate lowering path (CUDA/PTX, HIP/GCN, SPIR-V, LLVM CPU)
4. Caches compiled artifacts per (kernel hash, hardware fingerprint) pair
5. Dispatches at runtime with O(1) lookup overhead

This is precisely the generalization that cuBLAS does per-operator (at kernel-selection level) extended to the full kernel compilation pipeline, and that ORT's GetCapability does at graph-partition level extended to per-kernel cross-vendor targeting.

---

## References

1. E. Yang. "Let's talk about the PyTorch dispatcher." ezyang's blog, September 2020. https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/

2. PyTorch Contributors. "Registering a Dispatched Operator in C++." PyTorch Tutorials 2.11. https://docs.pytorch.org/tutorials/advanced/dispatcher.html

3. PyTorch Contributors. "Dynamic Shapes Core Concepts." PyTorch 2.10 documentation. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html

4. PyTorch Contributors. "CUDAGraph Trees." PyTorch 2.9 documentation. https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html

5. NVIDIA. "cuBLAS documentation v13.2." https://docs.nvidia.com/cuda/cublas/index.html

6. NVIDIA. "cuDNN v9.5.0 Graph API Developer Guide." https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.0/developer/graph-api.html

7. vLLM Contributors. "torch.compile Integration Design." https://docs.vllm.ai/en/latest/design/torch_compile/

8. vLLM Contributors. "Introduction to torch.compile and How It Works with vLLM." vLLM Blog, August 2025. https://vllm.ai/blog/torch-compile

9. Mu et al. "Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow." ASPLOS 2025. https://arxiv.org/abs/2406.01566

10. Hetis Authors. "Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism." arXiv:2509.08309, 2025. https://arxiv.org/abs/2509.08309

11. "Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs." arXiv:2502.00722, 2025. https://arxiv.org/html/2502.00722v1

12. Microsoft. "ONNX Runtime Architecture / High Level Design." https://onnxruntime.ai/docs/reference/high-level-design.html

13. Microsoft. "ONNX Runtime Execution Providers." https://onnxruntime.ai/docs/execution-providers/

14. NVIDIA. "TensorRT Architecture Overview." https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html

15. NVIDIA. "TensorRT-LLM GitHub." https://github.com/NVIDIA/TensorRT-LLM

16. PyTorch Contributors. "Introducing ExecuTorch 1.0: Powering the next generation of edge AI." PyTorch Blog, October 2025. https://pytorch.org/blog/introducing-executorch-1-0/

17. Miao et al. "SkyServe: Serving AI Models across Regions and Clouds with Spot Instances." arXiv:2411.01438, 2024. https://arxiv.org/abs/2411.01438

18. C. Rand. "Maximizing AI/ML Model Performance with PyTorch Compilation." Medium / Towards Data Science. https://towardsdatascience.com/maximizing-ai-ml-model-performance-with-pytorch-compilation/

19. T. Unguz. "State of PyTorch Hardware Acceleration 2025." https://tunguz.github.io/PyTorch_Hardware_2025/

20. M. Ye et al. "PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch." arXiv:2503.19779, 2025. https://arxiv.org/html/2503.19779v1

21. GitHub Issue. "Fallback provider logic bug in ONNX Runtime." microsoft/onnxruntime #25145. https://github.com/microsoft/onnxruntime/issues/25145

22. GitHub RFC. "Moving most torch.compile backends out of core." pytorch/pytorch #109687. https://github.com/pytorch/pytorch/issues/109687
