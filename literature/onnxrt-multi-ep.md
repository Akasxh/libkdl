# ONNX Runtime Execution Provider Model — Research Findings

**Compiled:** 2026-04-02
**Purpose:** LLVM Dublin 2026 poster — heterogeneous GPU kernel dispatch
**Scope:** EP architecture, selection, partitioning, performance, MLIR integration

---

## 1. Execution Provider Architecture Overview

ONNX Runtime (ORT) uses an **Execution Provider (EP) framework** as its primary abstraction for hardware-specific acceleration. Each EP encapsulates a hardware backend's capabilities — operator support, memory management, compilation/optimization — behind a common C++ interface (`IExecutionProvider`).

The key principle: ORT guarantees every ONNX operator has at least one implementation (via the CPU EP), while specialized EPs accelerate subsets of ops on faster hardware. The framework was designed to allow a single model to run across diverse hardware without application-level changes.

### 1.1 Registered EPs (as of ORT 1.23+)

**NVIDIA:**
- **CUDA EP** — Uses cuDNN; granular per-kernel GPU execution; fast setup. Best when setup time matters or the model has dynamic shapes.
- **TensorRT EP** — Whole-graph optimization via TensorRT; evaluates all execution paths and profiles them; can take minutes for engine build but delivers best throughput for stable shapes and FP16/INT8. Falls back to CUDA EP for unsupported subgraphs.
- **TensorRT RTX EP** — Added in 2025; RTX-specific optimizations (sparse tensor cores, etc.).

**AMD:**
- **ROCm EP** — Deprecated and **removed in ORT 1.23** (last supported: ROCm 7.0). Users must migrate to MIGraphX EP.
- **MIGraphX EP** — AMD's current recommended EP. Uses MIGraphX graph optimization engine on AMD GPUs (ROCm 5.2–7.2+). Supports FP16, BF16, INT8, FP8 via calibration tables. No explicit supported-ops list; users check via `ORT_MIGRAPHX_DUMP_MODEL_OPS`. Lost `.mxr` save/load in ROCm 6.4.

**Cross-vendor (Windows):**
- **DirectML EP** — DirectX 12 ML library; supports any DX12-capable GPU (NVIDIA, AMD, Intel). Recommended for Windows deployments. Limitation: DirectML does not expose DX12 command lists directly, complicating GPU-side synchronization between preprocessing and inference.

**CPU:**
- **Default CPU EP** — Reference implementation supporting all ONNX operators. Final fallback. Uses common data types only; uncommon types may lack kernels.
- **Intel oneDNN EP** — CPU-optimized via oneDNN (MKL-DNN).
- **XNNPACK EP** — Lightweight CPU inference (mobile/edge focus).
- **Intel OpenVINO EP** — Multi-platform; CPU, Intel GPU, Intel NPU; heterogeneous plugin internal to OpenVINO.

**Edge/Mobile:**
- Android NNAPI, Apple CoreML (preview), Qualcomm QNN, Arm Compute Library (preview), Xilinx Vitis-AI (preview), Huawei CANN (preview), Apache TVM (preview).

**Source:** [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
**Source:** [ROCm EP Docs](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
**Source:** [MIGraphX EP Docs](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

---

## 2. EP Selection at Runtime — Priority-Based Fallback

### 2.1 Registration Order as Priority

EPs are registered on `SessionOptions` as an ordered list. Example:

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

The list is **strictly priority-ordered**: ORT attempts to assign each graph node to the highest-priority EP that claims it. If CUDA EP cannot handle a node, it falls to the next EP in the list. The CPU EP, always last, guarantees complete coverage.

### 2.2 Capability Query: `GetCapability()`

Before graph partitioning, ORT queries each registered EP via `GetCapability()`:

```cpp
virtual std::vector<std::unique_ptr<ComputeCapability>>
GetCapability(const onnxruntime::GraphViewer& graph_viewer,
              const IKernelLookup& kernel_lookup,
              const GraphOptimizerRegistry& graph_optimizer_registry,
              IResourceAccountant* resource_accountant = nullptr) const;
```

Each EP returns a collection of `ComputeCapability` objects — subgraph specifications for nodes it can execute. EPs are queried in priority order. A node not claimed by any specialized EP falls to the CPU EP.

**Nuance:** Even if an EP has a kernel for an op, it may decline to claim that node in `GetCapability()` for performance reasons (e.g., if the op would create too many small subgraphs causing excessive host-device data transfers).

**Source:** [ORT Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html)
**Source:** [execution_provider.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/execution_provider.h)

---

## 3. Graph Partitioning Algorithm

### 3.1 The Partitioning Pipeline

ORT applies graph partitioning in this sequence during session initialization:

1. **Load ONNX model** → in-memory graph representation
2. **Provider-independent optimizations** — constant folding, node fusion, shape inference (these run before partitioning and apply uniformly)
3. **Priority-ordered EP query** — for each EP (highest priority first), call `GetCapability()` to collect claimed subgraphs
4. **Assign maximal subgraphs** — each EP receives the largest contiguous subgraphs it can handle; one EP may receive multiple disconnected subgraphs
5. **Remainder to CPU EP** — any unclaimed node is assigned to the CPU fallback
6. **Fuse partitions** — each assigned subgraph is compiled via the EP's `Compile()` method and wrapped as a **single fused custom operator** in the execution graph

### 3.2 The `Compile()` Step

```cpp
virtual common::Status Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs);
```

Each partition becomes a fused node. The EP's `Compile()` transforms the subgraph into an EP-native execution plan (e.g., a TensorRT engine, a MIGraphX program). At inference time, `NodeComputeInfo::compute_func` is called to run the partition.

### 3.3 Cross-EP Memory Management

Each EP exposes its own memory allocator:
```cpp
virtual AllocatorPtr GetAllocator(OrtMemType mem_type) const;
```

When a tensor produced by EP-A (e.g., CUDA) is consumed by EP-B (e.g., CPU), ORT inserts **implicit data-copy nodes** using `GetDataTransfer()`:
```cpp
virtual std::unique_ptr<IDataTransfer> GetDataTransfer() const;
```

This host-device copy is a known performance penalty — the primary cost of mixed-EP execution. CUDA Graphs (`enable_cuda_graph`) can reduce CPU kernel launch overhead within a single CUDA EP subgraph, but do not eliminate cross-EP transfer overhead.

**Source:** [ORT High-Level Design](https://onnxruntime.ai/docs/reference/high-level-design.html)
**Source:** [NVIDIA CUDA/TensorRT EP Blog](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/)

---

## 4. Performance Comparison: NVIDIA vs AMD vs CPU

### 4.1 GPU vs CPU (March 2026 Benchmark)

**Hardware:** NVIDIA L4 (24 GB, 300 GB/s) vs AMD EPYC 9965 (192 cores, 2-socket, 24 NUMA nodes, AVX-512)
**Model:** Tessera satellite imagery embedding model (128-dim output)
**ORT Version:** 1.24.1

| Configuration | Per-Batch Latency | Notes |
|---|---|---|
| GPU (L4) baseline | 900 ms | Single stream |
| CPU (8 threads, untuned) | 11,360 ms | 12.6x slower than GPU |
| CPU (NUMA-optimized, 24 parallel jobs) | 5.21 ms per job | 2.0x **faster** aggregate throughput than GPU |

**Critical finding:** Thread scaling on multi-socket CPUs plateaus at ~16 threads due to NUMA latency. Treating each NUMA node as an independent inference unit (24 parallel jobs, each pinned to one node) reversed the GPU advantage for this embarrassingly parallel workload.

**Source:** [GPU vs CPU ONNX Inference - Tunbury.ORG](https://www.tunbury.org/2026/03/11/gpu-vs-cpu/)

### 4.2 TensorRT EP vs CUDA EP vs PyTorch

**ResNet-50 (TensorRT, INT8, batch=4):**
- Throughput: ~507 inferences/sec (2028 images/sec)
- Median latency: 1.969 ms
- TensorRT generally outperforms CUDA EP by selecting globally optimal graph execution path

**BERT-Large (TensorRT 8.0+, INT8):**
- Inference latency: as low as 1.2 ms

**General pattern (from multiple 2024–2025 sources):**
- TensorRT EP > CUDA EP > CPU EP for large transformer models on NVIDIA hardware
- CUDA EP advantage over CPU EP becomes decisive at >100M parameters or long sequences
- CPU thread scaling: 1→8 threads = 4.2x speedup; 8→16 threads ≈ +2% only

**Source:** [OpenBenchmarking.org ONNX RT](https://openbenchmarking.org/test/pts/onnx)
**Source:** [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
**Source:** [ORT Transformer INT8 Blog](https://opensource.microsoft.com/blog/2022/05/02/optimizing-and-deploying-transformer-int8-inference-with-onnx-runtime-tensorrt-on-nvidia-gpus)

### 4.3 AMD MIGraphX EP — Known Gaps

- No public operator support matrix (users must use `ORT_MIGRAPHX_DUMP_MODEL_OPS` to discover coverage)
- Exhaustive tuning option (`migraphx_exhaustive_tune`) trades compile time for runtime speedup
- Removed model caching (`.mxr` format) in ROCm 6.4 — users must recompile on each session start
- No published head-to-head benchmark against CUDA EP under comparable hardware conditions (RTX-class vs RX-class) found in literature

**Source:** [MIGraphX EP Docs](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

### 4.4 CPU ARM64 (Azure Cobalt 100)

**Model:** SqueezeNet 1.0 INT8
- Average inference: 1.86 ms
- Peak throughput: 538 inferences/sec
- Peak memory: <37 MB
- CPU utilization: ~96%

**Source:** [Arm Azure ONNX Benchmark](https://learn.arm.com/learning-paths/servers-and-cloud-computing/onnx-on-azure/benchmarking/)

---

## 5. Lessons: What Works and What Doesn't

### 5.1 What Works Well

**Priority-ordered fallback is pragmatically correct.** The design ensures no model is ever un-runnable. The CPU EP as universal backstop is a key correctness guarantee. In practice, most production models run cleanly on a single EP (usually CUDA or TensorRT).

**EP Context caching (ORT 1.18+, formalized 1.23+)** solves the TensorRT engine build problem. Pre-compiled graphs serialized to disk reduce multi-minute session init to near-zero. Multiple EPs can each embed `EPContext` nodes in the same model file, enabling true multi-EP cached deployment.

**Plugin EP API (ORT 1.23+)** decouples EP development from ORT core. EPs are now built as shared libraries exporting `CreateEpFactories()` / `ReleaseEpFactory()`, allowing third-party EPs without patching ORT itself. TensorRT EP was refactored to this model as the reference implementation.

**Operator kernel registration via custom op domain** allows EPs to expose non-standard ops (e.g., TensorRT plugins) as custom ONNX operators, integrating them into the graph partitioning flow.

### 5.2 What Doesn't Work Well

**Greedy maximal-subgraph partitioning is suboptimal.** ORT's documented partitioning is "a simple graph partitioning technique" — greedy, priority-ordered, maximally extending each EP's claimed region. It does not globally optimize the partition boundary placement to minimize host-device transfers. A model with scattered unsupported ops can fragment into many small GPU subgraphs interspersed with CPU fallbacks, each boundary costing a PCIe data copy.

**Cross-EP memory copies are implicit and hard to control.** Developers cannot directly observe or control where copy nodes are inserted. The only mitigation is eliminating the need for mixed EPs (i.e., ensuring full operator coverage on the target EP). Partial coverage of a specialized EP can be worse than running entirely on CPU.

**Operator coverage opacity for non-CUDA EPs.** CUDA EP has kernel-level documentation; MIGraphX EP has none. Users discover coverage gaps at model load time. This asymmetry makes AMD deployment significantly harder to validate upfront.

**ROCm EP deprecation and community fragmentation.** AMD's trajectory — ROCm EP → MIGraphX EP — broke existing deployments. The removal of `.mxr` model caching in ROCm 6.4 further regressed AMD usability. No equivalent to TensorRT's model serialization exists in the AMD stack as of ORT 1.23.

**Static priority list requires user knowledge of model structure.** Users must know in advance which EPs to register and in what order. ORT provides no automatic EP selection based on model analysis or hardware profiling. This puts the burden on the user to know their model's operator distribution.

**TensorRT EP limitations with dynamic shapes.** Dynamic input shapes prevent TensorRT from optimizing the full graph — it must either use optimization profiles (predefined shape ranges) or fall back to CUDA EP. This is a critical limitation for transformer inference with variable sequence lengths.

**Source:** [ORT Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html)
**Source:** [Choosing EP Guide](https://pkreg101.github.io/onnxruntime/docs/performance/choosing-execution-providers.html)
**Source:** [EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
**Source:** [ORT Troubleshooting](https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html)

---

## 6. Implementing an MLIR-Based Custom ONNX Runtime EP

### 6.1 Feasibility Assessment

**Yes, an MLIR-based dispatch layer can be implemented as an ORT custom EP.** ORT's plugin EP API (stable since 1.23) is specifically designed for this pattern. The EP would:

1. Implement `GetCapability()` to claim ONNX ops that MLIR can handle (initially all, or a defined subset)
2. Implement `Compile()` to lower claimed subgraphs through an MLIR pipeline to native code
3. Expose the compiled function via `NodeComputeInfo::compute_func`
4. Implement `GetDataTransfer()` for device memory management

### 6.2 The Plugin EP Interface (ORT 1.23+)

The shared library must export:
```c
OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    OrtEpFactory*** factories,
    size_t* num_factories);

void ReleaseEpFactory(OrtEpFactory* factory);
```

`OrtEpFactory` creates `OrtEp` instances per session. The EP instance implements:
- `GetCapability()` / equivalent — node claiming
- `Compile()` — subgraph lowering to executable
- Memory allocation and data transfer

**Source:** [Plugin EP Libraries](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html)
**Source:** [Add New EP Guide](https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html)
**Source:** [TensorRT Plugin EP (reference impl)](https://github.com/onnxruntime/onnxruntime-ep-tensorrt)

### 6.3 Discussion on LLVM Forums (May 2025)

A thread on the LLVM Discourse forums directly asked: "Is There Existing Work to add ONNX Runtime Execution Provider based on MLIR or LLVM?" — indicating the idea is actively being explored by the community but no complete implementation has been published as of the thread date.

Key technical challenge identified: ONNX Runtime's graph representation must be converted to MLIR IR within `Compile()`, then lowered through dialect passes to code-gen. The ONNX-MLIR project (see Section 7) provides the dialect and lowering infrastructure that an MLIR-EP could reuse.

**Source:** [LLVM Discourse: MLIR-based ORT EP](https://discourse.llvm.org/t/is-there-existing-work-to-add-onnx-runtime-execution-provider-based-on-mlir-or-llvm/86383)

### 6.4 Architectural Sketch for an MLIR-EP

```
ORT Session
  └─ MLIR EP (plugin shared library)
       ├─ GetCapability(): claim all supported ONNX ops
       ├─ Compile(subgraph):
       │    ├─ Convert ONNX subgraph → onnx MLIR dialect
       │    ├─ Apply ONNX dialect optimizations
       │    ├─ Lower: onnx dialect → linalg/arith/vector dialects
       │    ├─ Target selection:
       │    │    ├─ NVIDIA: lower → nvvm dialect → PTX → cubin
       │    │    ├─ AMD:    lower → rocdl dialect → amdgcn → hsaco
       │    │    └─ CPU:    lower → llvm dialect → native object
       │    └─ JIT compile and cache
       └─ compute_func(): invoke compiled kernel
```

This architecture enables **vendor-agnostic dispatch at the MLIR level** — a single EP handles all hardware targets by selecting the appropriate MLIR backend, rather than requiring separate EPs per vendor.

**Advantages over the current ORT EP model:**
- Single EP eliminates cross-EP memory copy overhead for heterogeneous GPU models
- MLIR's progressive lowering enables target-specific optimizations without duplicate op implementations
- JIT compilation enables dynamic specialization (batch size, dtype, hardware capability)

**Disadvantages / open problems:**
- MLIR-to-ONNX subgraph conversion within `Compile()` must handle all ORT graph structures (including control flow ops)
- Cold-start compile latency (mitigated by EP Context caching)
- MLIR GPU backends for AMD (rocdl → amdgcn) are less mature than CUDA (nvvm → PTX)

---

## 7. ONNX-MLIR Project

### 7.1 Overview

ONNX-MLIR is an **independent project** (not part of ORT) that compiles ONNX models using MLIR infrastructure. It is not an ORT EP but a standalone compiler producing shared libraries with minimum runtime support.

**Origin:** IBM Research, 2019. Now multi-company: AMD, ByteDance, Groq, Microsoft contributors. 100+ contributors, ~2,540 commits.

**IBM use case:** zDLC compiler for Telum server deployment.

**Source:** [ONNX-MLIR GitHub](https://github.com/onnx/onnx-mlir)

### 7.2 Dialect Pipeline

ONNX-MLIR introduces two custom dialects:

1. **ONNX Dialect** — encodes ONNX standard semantics directly in MLIR IR; one-to-one with the ONNX opset
2. **Krnl Dialect** — loop-based intermediate dialect sitting above MLIR's affine dialect; exposes `krnl.define_loops`, `krnl.iterate`, `krnl.block` (tiling), `krnl.permute` (loop permutation)

**Lowering stages (selectable at compile time):**
- `--EmitONNXIR` — Ingest ONNX protobuf, emit ONNX dialect MLIR
- `--EmitMLIR` — Lower to MLIR built-in dialects (linalg, affine, etc.)
- `--EmitLLVMIR` — Lower to LLVM dialect, then LLVM IR
- Produces: ONNX IR, MLIR files, LLVM bytecodes, shared libraries (.so), JNI jars, C/Python/Java runtimes

### 7.3 Supported Targets

- **CPU (x86, Power, s390x, ARM)** — primary target; complete
- **IBM Telum NNPA** — accelerator-specific path
- **GPU** — not officially supported as of 2025; the project focuses on CPU and IBM-specific accelerators

### 7.4 Relationship to ORT

ONNX-MLIR and ORT are **complementary but architecturally separate**:

| Aspect | ORT | ONNX-MLIR |
|---|---|---|
| Approach | Runtime dispatch via EP framework | AOT/JIT compilation via MLIR |
| Heterogeneous support | Multiple EPs at graph-partition granularity | Single compiled artifact per target |
| GPU support | CUDA, TensorRT, MIGraphX, DirectML (mature) | CPU/NNPA (GPU: not mature) |
| Custom ops | Custom op domain in ORT | Limited |
| Integration | Session-based inference API | C/Java/Python library from compiled ONNX |

**Proposed integration path for our poster:** Use ONNX-MLIR's dialect infrastructure (ONNX dialect, lowering passes) as the compiler component inside an ORT custom EP. The EP wraps the MLIR compiler pipeline; ORT handles the session, graph partitioning, and API surface.

**Source:** [Compiling ONNX with MLIR (arXiv 2008.08272)](https://arxiv.org/abs/2008.08272)
**Source:** [ONNX-MLIR Docs](https://onnx.ai/onnx-mlir/)

---

## 8. EP Selection and Our Heterogeneous Dispatch Problem

### 8.1 Direct Relevance

ORT's EP model is the **closest production analog** to the heterogeneous dispatch problem our poster addresses. Key parallels:

| ORT EP Mechanism | Our Problem |
|---|---|
| Priority-ordered EP list | Runtime hardware selection policy |
| `GetCapability()` per EP | Operator-to-device capability query |
| Greedy maximal-subgraph partitioning | Kernel-to-device assignment strategy |
| Cross-EP memory copy insertion | Inter-device data movement scheduling |
| EP Context caching | AOT-compiled kernel cache per target |
| Plugin EP API (shared library) | Pluggable dispatch backend architecture |

### 8.2 Gaps ORT EP Model Exposes (Opportunities for Our Work)

**Gap 1: Static, user-specified priority.** ORT requires the user to know and specify EP priority at session creation. Our system should perform **automatic hardware discovery and capability-driven selection** without user intervention.

**Gap 2: Greedy partition placement is transfer-unaware.** ORT's partitioner does not model the cost of host-device data copies when deciding partition boundaries. An MLIR-based dispatcher could use **cost modeling** (transfer bandwidth, kernel latency) to optimize boundary placement.

**Gap 3: No cross-vendor unified EP.** The NVIDIA/AMD split forces separate EPs with different coverage, APIs, and reliability levels. An MLIR-based EP unifies both targets behind a single `Compile()` that selects nvvm vs. rocdl as a backend detail — vendors become implementation choices, not API-level distinctions.

**Gap 4: Dynamic shape limitations block full TensorRT optimization.** MLIR's progressive lowering supports specialization at JIT time — a compiled kernel variant can be generated per observed input shape, cached, and dispatched, addressing the dynamic shape problem without TensorRT's static optimization profile constraint.

**Gap 5: Operator coverage opacity for AMD.** Our MLIR-EP approach would emit coverage information as a first-class output of the MLIR pipeline (which ops successfully lower for which target), making coverage transparent and testable — in contrast to MIGraphX's opaque coverage.

### 8.3 IREE as the State-of-the-Art Reference

**IREE** (Intermediate Representation Execution Environment) is the closest existing system to what we propose:

- MLIR-based end-to-end compiler and runtime
- Supports ONNX as an import frontend (`iree-import-onnx` via torch-mlir)
- Targets: CPU (LLVM), GPU via CUDA, ROCm, Vulkan, Metal
- Both AOT and JIT compilation
- Broad operator coverage "is an active investment area"

IREE's approach is **IR-first** (compile to MLIR, then to hardware) vs ORT's **runtime-first** (dispatch to hardware-specific library at runtime). Our poster's contribution sits between these: an MLIR-based ORT EP that brings IR-first compilation discipline to the ORT ecosystem without abandoning ORT's mature session API and operator coverage.

**Source:** [IREE ONNX Guide](https://iree.dev/guides/ml-frameworks/onnx/)
**Source:** [IREE Homepage](https://iree.dev/)

---

## 9. Summary Table: EP Comparison

| EP | Vendor | Status | Op Coverage | Dynamic Shapes | Strengths | Weaknesses |
|---|---|---|---|---|---|---|
| CPU EP | Any | Stable | Complete | Full | Guaranteed fallback, all ops | Slowest |
| CUDA EP | NVIDIA | Stable | High (documented) | Full | Fast setup, broad coverage | Not globally optimal |
| TensorRT EP | NVIDIA | Stable | High (with CUDA fallback) | Limited (profiles) | Best NVIDIA throughput | Long engine build, shape restrictions |
| TensorRT RTX EP | NVIDIA | New (2025) | Subset | TBD | RTX tensor core optimizations | Narrow hardware target |
| ROCm EP | AMD | **Removed** (1.23) | Was partial | N/A | — | Removed; use MIGraphX |
| MIGraphX EP | AMD | Active | Undocumented | Partial | AMD GPU acceleration | No op list, lost model caching |
| DirectML EP | Multi (Windows) | Stable | Moderate | Partial | Cross-vendor Windows GPU | DX12 sync limitations |
| OpenVINO EP | Intel | Stable | High | Partial | Edge/NPU, heterogeneous internal | Intel-focused |
| ONNX-MLIR EP (proposed) | Any | Research | Compiler-defined | Full (JIT) | Vendor-agnostic, cost-aware | Maturity, MLIR GPU backends |

---

## 10. Key Citations

1. Microsoft. "ONNX Runtime Execution Providers." https://onnxruntime.ai/docs/execution-providers/
2. Microsoft. "ONNX Runtime High-Level Design." https://onnxruntime.ai/docs/reference/high-level-design.html
3. Microsoft. "EP Context Design." https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html
4. Microsoft. "Add a New Execution Provider." https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html
5. Microsoft. "MIGraphX Execution Provider." https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
6. Microsoft. "ROCm Execution Provider." https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html
7. Microsoft / ONNX Runtime. "IExecutionProvider C++ interface." https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/execution_provider.h
8. ONNX Runtime TensorRT Plugin EP (reference implementation). https://github.com/onnxruntime/onnxruntime-ep-tensorrt
9. NVIDIA. "End-to-End AI for NVIDIA-Based PCs: CUDA and TensorRT EPs in ONNX Runtime." https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/
10. Microsoft Open Source. "Optimizing transformer INT8 inference with ONNX Runtime-TensorRT." https://opensource.microsoft.com/blog/2022/05/02/optimizing-and-deploying-transformer-int8-inference-with-onnx-runtime-tensorrt-on-nvidia-gpus
11. Tunbury.org. "GPU vs CPU for ONNX Inference: NVIDIA L4 vs AMD EPYC 9965." https://www.tunbury.org/2026/03/11/gpu-vs-cpu/
12. ARM. "Benchmarking ONNX Runtime on Azure Cobalt 100 (ARM64)." https://learn.arm.com/learning-paths/servers-and-cloud-computing/onnx-on-azure/benchmarking/
13. OpenBenchmarking.org. "ONNX Runtime Benchmark." https://openbenchmarking.org/test/pts/onnx
14. LLVM Discourse. "Is There Existing Work to add ONNX Runtime Execution Provider based on MLIR or LLVM?" https://discourse.llvm.org/t/is-there-existing-work-to-add-onnx-runtime-execution-provider-based-on-mlir-or-llvm/86383
15. ONNX / IBM. "ONNX-MLIR: Representation and Reference Lowering of ONNX Models in MLIR." https://github.com/onnx/onnx-mlir
16. Zhang et al. "Compiling ONNX Neural Network Models Using MLIR." arXiv:2008.08272. https://arxiv.org/abs/2008.08272
17. IREE Project. "ONNX Support in IREE." https://iree.dev/guides/ml-frameworks/onnx/
18. AMD GPUOpen. "ONNX and DirectML Execution Provider Guide." https://gpuopen.com/learn/onnx-directlml-execution-provider-guide-part1/
19. Graiphic. "ONNX Runtime Node Coverage Project." https://graiphic.io/onnx-runtime-node-coverage/
20. Microsoft. "Choosing Execution Providers for Performance." https://pkreg101.github.io/onnxruntime/docs/performance/choosing-execution-providers.html
