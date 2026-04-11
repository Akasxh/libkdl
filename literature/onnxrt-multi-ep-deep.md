# ONNX Runtime Multi-EP: Deep Dive on Priority-Based Selection and Graph Partitioning

**Compiled:** 2026-04-06
**Relevance Score:** 9/10 — ORT's EP model is the production analog to our dispatch problem; the Plugin EP API (1.23+) is the direct template for implementing a libkdl-backed ORT EP; the gap analysis maps directly to our poster's contribution framing
**Connection to our work:** This is a deep companion to `onnxrt-multi-ep.md` (filed 2026-04-02). This document covers: the EP selection algorithm in detail, multiple simultaneous EP activation, runtime device selection logic, and the architectural sketch for a libkdl-backed EP. Cross-reference with `onnxrt-multi-ep.md` for benchmark data and EP catalog.
**Note:** `onnxrt-multi-ep.md` already covers the EP catalog, GetCapability/Compile interfaces, benchmark data, and MLIR-EP architectural sketch extensively. This document focuses on aspects not covered there: EP selection algorithm internals, multi-EP concurrent activation, the Windows ML EP selection API, and updated 2025–2026 developments.

---

## 1. What's New Since the April 2 Deep Dive

The `onnxrt-multi-ep.md` document (2026-04-02) covers ORT 1.23 architecture comprehensively. This companion focuses on:
- The priority-ordered selection algorithm at the graph node level
- How multiple EPs are simultaneously active (not mutually exclusive)
- The EP selection API in Windows ML (new in late 2024)
- Recent 2025 ORT developments
- Updated framing of how ORT relates to libkdl

---

## 2. Priority-Based EP Selection: Algorithm Detail

### 2.1 The Two-Phase Selection Process

ORT's EP selection happens in two phases during `InferenceSession::Initialize()`:

**Phase 1: Capability Collection**

```
For each EP in priority order (index 0 = highest priority):
    ep.GetCapability(graph_viewer, kernel_lookup) → [ComputeCapability, ...]

    For each ComputeCapability returned:
        Mark claimed node_ids as "claimed by ep_index"

    # An EP may claim overlapping nodes; later EPs cannot claim already-claimed nodes
    # EPs are queried even if lower priority — they can "see" the full graph
    # but claimed nodes are skipped
```

**Phase 2: Subgraph Formation and Partition**

```
For each claimed set of node_ids per EP:
    Form maximal connected subgraphs
    # "Maximal" = not splittable by inserting an unclaimed node
    # Multiple disconnected subgraphs per EP are normal

    ep.Compile([FusedNodeAndGraph...]) → [NodeComputeInfo...]

    # Each compiled subgraph becomes a single fused custom op in the execution graph
    # The fused op's compute_func calls the EP's compiled binary

Unclaimed nodes → CPU EP (guaranteed fallback)
```

**Key behavior:** The priority list is strictly respected — a higher-priority EP's claimed nodes are never re-evaluated by lower-priority EPs. If CUDA EP claims a Conv2D node, MIGraphX EP never sees it.

**Source:** [ORT High-Level Design](https://onnxruntime.ai/docs/reference/high-level-design.html), [execution_provider.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/execution_provider.h)

### 2.2 Can Multiple EPs Be Active Simultaneously?

**Yes.** Multiple EPs in the provider list are all active simultaneously during inference. The graph is partitioned so that:
- Different subgraphs execute on different EPs in the same inference pass
- The execution graph is a DAG where each node is assigned to exactly one EP
- ORT's executor traverses the DAG, dispatching each node to its assigned EP's `compute_func`

**Example:** `['TensorRTExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- TensorRT EP handles large fused subgraphs it can optimize whole
- CUDA EP handles ops TensorRT declined (e.g., ops with unsupported ONNX opset)
- CPU EP handles any remaining ops (e.g., Gather, TopK if CUDA EP doesn't claim them)

All three EPs are simultaneously active in the same inference session. This is the defining feature of ORT's multi-EP model.

**Practical constraint:** Cross-EP boundaries incur data copy overhead (PCIe transfers for device ↔ host). The implicit copy nodes inserted by ORT have measurable latency (see `onnxrt-multi-ep.md`, Section 3.3). Minimizing EP boundary crossings is the main performance consideration in multi-EP deployment.

### 2.3 Runtime Device Selection Logic — Is There Any?

ORT's priority-ordered EP list is **user-specified at session creation time**. There is **no automatic runtime device selection** — ORT does not:
- Probe installed GPU drivers to determine what EPs are available
- Automatically add CUDA EP if an NVIDIA GPU is detected
- Fall back to CPU EP if the CUDA EP initialization fails (silently) — this requires explicit error handling

**What does happen:**
- EP initialization is attempted in priority order
- If an EP fails to initialize (e.g., CUDA not available), ORT raises an exception unless the session is configured to silently skip unavailable EPs (`SessionOptions.SetLogSeverityLevel` + error handling)
- The actual node-level device selection (which EP claims which node) happens in `GetCapability()` calls, which do run device-side queries (e.g., TensorRT probes the engine builder's layer support)

**Implication for our work:** ORT's lack of automatic hardware discovery is Gap #1 in `onnxrt-multi-ep.md` Section 8.2. Our libkdl approach performs hardware discovery at library load time, eliminating the need for the user to specify target hardware.

---

## 3. EP Selection API in Windows ML (New, 2024–2025)

Windows ML (the Windows 11 ML runtime) adds a **declarative EP selection API** that partially addresses ORT's manual configuration problem:

```cpp
// Windows ML EP selection (as of late 2024)
LearningModelSessionOptions options;
options.SetCloseModelOnSessionCreation(true);

// Declare desired EP capabilities instead of specific EP names
options.PreferredExecutionDevices({
    LearningModelDevice(LearningModelDeviceKind::DirectXHighPerformance),
    LearningModelDevice(LearningModelDeviceKind::Cpu)
});
```

Windows ML maps `LearningModelDeviceKind` to available EPs based on the system configuration:
- `DirectXHighPerformance` → maps to best available DX12-capable GPU EP (DirectML EP or TensorRT EP on NVIDIA systems)
- `Cpu` → CPU EP

This is a capability-based (rather than name-based) EP selection. It allows the application to express "use the fastest GPU available" without knowing the specific GPU vendor.

**Source:** [Select Execution Providers Using Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers)

**Limitation:** This API is Windows-specific and maps to DirectML as the multi-vendor GPU EP. It does not expose CUDA/TensorRT vs DirectML selection or Linux/cross-platform deployment.

---

## 4. TensorRT EP: Partition Iteration Limit

TensorRT EP adds a unique wrinkle to multi-EP partitioning that illustrates the complexity of graph-level EP selection:

```
ORT_TENSORRT_MAX_PARTITION_ITERATIONS (default: 1000)
```

TensorRT EP's partitioning is iterative: it builds a TensorRT engine for a candidate subgraph, and if the engine build fails (unsupported op combination), it removes the failing node and retries. This can require many iterations for complex models with scattered unsupported ops.

When `MAX_PARTITION_ITERATIONS` is exceeded, **the entire model falls back to CUDA EP** (not just the failing subgraph). This is an all-or-nothing fallback at the model level, not the subgraph level — a behavior that can cause surprising performance regressions.

**Source:** [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

---

## 5. Plugin EP API (ORT 1.23+): Summary for libkdl Integration

The Plugin EP API (introduced in ORT 1.23, reference implementation: TensorRT EP refactored to this model) is the mechanism by which libkdl could be integrated into ORT as a custom EP.

### 5.1 Required Exports

A plugin EP shared library must export:

```c
// Entry points ORT calls at plugin load time
OrtStatus* CreateEpFactories(
    const char* registration_name,   // "LibKDLExecutionProvider"
    const OrtApiBase* ort_api_base,  // ORT API handle
    OrtEpFactory*** factories,       // [out] array of factory objects
    size_t* num_factories);          // [out] count

void ReleaseEpFactory(OrtEpFactory* factory);
```

`OrtEpFactory` then creates `OrtEp` instances per session. Each `OrtEp` implements:
- `GetCapabilityFunc` — claim supported subgraphs
- `CompileFunc` — lower subgraphs to executables
- Allocator/data transfer implementations

**Source:** [ORT Plugin EP Libraries](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html), [TensorRT Plugin EP Reference](https://github.com/onnxruntime/onnxruntime-ep-tensorrt)

### 5.2 Registration at Runtime

```python
# Python: register plugin EP
import onnxruntime as ort

# Load plugin library
ort.preload_dlls(["libkdl_ort_ep.so"])

# Register with environment
env = ort.OrtEnvironment()
env.register_execution_provider_library("LibKDL", "libkdl_ort_ep.so")

# Use in session
session = ort.InferenceSession("model.onnx",
    providers=["LibKDLExecutionProvider", "CPUExecutionProvider"])
```

### 5.3 libkdl as an ORT Plugin EP: Design Sketch

```
ORT Session (user code)
  └─ LibKDL Plugin EP (libkdl_ort_ep.so)
       │
       ├─ GetCapability():
       │    └─ Claim all ONNX ops in our supported set
       │       (determined by MLIR lowering coverage)
       │
       ├─ Compile(subgraph):
       │    ├─ Convert ONNX subgraph → Linalg/MLIR IR (via onnx-mlir dialect)
       │    ├─ Detect hardware at compile time:
       │    │    ├─ NVIDIA: emit NVVM → PTX → cubin variant
       │    │    ├─ AMD:    emit ROCDL → AMDGCN → hsaco variant
       │    │    └─ CPU:    emit LLVM → native object variant
       │    ├─ Pack all variants into .kdl section table
       │    └─ Register dispatch table with libkdl runtime
       │
       └─ compute_func(inputs, outputs):
            └─ libkdl_dispatch(kernel_id, inputs, outputs)
                 └─ Select variant via ELF section header query
                    (hardware detected at library load time)
```

**Key advantage over separate CUDA/MIGraphX EPs:** The libkdl EP provides **single-EP multi-vendor coverage**, eliminating all cross-EP memory copy overhead for GPU models. The GPU stays as the execution device throughout; vendor selection is an internal libkdl detail invisible to ORT's graph partitioner.

---

## 6. Graph Optimization Interaction with EP Selection

### 6.1 Optimization Timing

ORT runs graph optimizations in three levels:

| Level | Name | Timing | EP Awareness |
|-------|------|--------|--------------|
| 0 | None | Disabled | N/A |
| 1 | Basic | Before partitioning | EP-independent |
| 2 | Extended | Before partitioning + per-EP | Some EP-specific fusions |
| 3 | All | Before + after | Includes EP-specific layout transforms |

**Critical point:** Provider-independent optimizations (constant folding, CSE, shape inference) run **before** EP partitioning. EP-specific layout transforms (e.g., NCHW → NHWC for GPU EPs) run **after** the EP claims subgraphs. This means the graph presented to `GetCapability()` is already partially optimized but not yet EP-specialized.

**Source:** [ORT Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

### 6.2 EP Context (Pre-compiled Graph Caching)

EP Context (stable in ORT 1.23+) enables serializing compiled EP subgraphs to disk:
- Each EP embeds `EPContext` nodes in the ONNX model after first compilation
- On subsequent loads, the EP rehydrates compiled state from `EPContext` nodes rather than recompiling
- Multiple EPs can each embed their `EPContext` in the same model file
- This eliminates the TensorRT engine build time (minutes → zero) on subsequent session starts

**For libkdl integration:** Our `.kdl` compiled fat binary maps naturally to the `EPContext` serialization — the pre-compiled multi-vendor artifact can be embedded as an `EPContext` node, enabling zero-compile-time deployment.

**Source:** [ORT EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)

---

## 7. What Multi-EP Does Not Solve (Confirmed Gaps)

These gaps are documented in `onnxrt-multi-ep.md` Section 8 and confirmed by 2025 ORT documentation:

1. **No automatic hardware-driven EP selection.** The user must enumerate EPs manually. ORT provides no `SessionOptions.SetAutoEP()` that detects installed hardware and configures the provider list.

2. **No transfer-aware partitioning.** The greedy maximal-subgraph partitioner does not model data-copy cost. A model with scattered unsupported ops can produce many small GPU subgraphs interspersed with CPU fallback — each boundary crossing is a PCIe transfer. ORT has no mechanism to identify this situation and consolidate partitions.

3. **No single-EP multi-vendor coverage.** Each GPU vendor requires a separate EP. NVIDIA users use CUDA/TensorRT EP; AMD users use MIGraphX EP. There is no "GPU EP" that automatically selects the correct vendor backend. A system with both NVIDIA and AMD GPUs (heterogeneous workstation) requires careful manual configuration.

4. **MIGraphX EP coverage opacity.** AMD's MIGraphX EP has no published operator support matrix. Users discover unsupported ops at model load time via `ORT_MIGRAPHX_DUMP_MODEL_OPS` debug output.

5. **Static priority list.** The EP priority list cannot be updated between inference calls. If the CUDA EP is registered first, it will always claim GPU-capable nodes — even if user code wants to temporarily route to CPU for debugging.

---

## 8. Summary: ORT Multi-EP vs libkdl Dispatch Model

| Dimension | ORT Multi-EP | libkdl Approach |
|-----------|-------------|-----------------|
| Granularity | Subgraph (multi-node) | Kernel (single op) |
| Vendor selection timing | Session creation (user-specified) | Library load (hardware detected) |
| Multi-vendor single binary | No | Yes (.kdl fat binary) |
| Cross-vendor memory copy | Yes (implicit copy nodes at EP boundaries) | No (same device throughout) |
| Operator coverage transparency | CUDA EP: documented. MIGraphX EP: opaque | MLIR lowering success = coverage |
| AOT kernel cache | EPContext (per-EP, composable) | .kdl ELF section table |
| Custom EP integration | Plugin EP API (ORT 1.23+) | libkdl as a plugin EP |
| Transfer-aware partitioning | No | Future work (cost-model-based) |
| Auto hardware discovery | No | Yes (CUDA/HIP/CPU probe at dlopen) |

---

## 9. Key Citations (Supplemental to onnxrt-multi-ep.md)

1. Microsoft. "ONNX Runtime Architecture." https://onnxruntime.ai/docs/reference/high-level-design.html
2. Microsoft. "Plugin EP Libraries." https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html
3. Microsoft. "Using a Plugin Execution Provider Library." https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html
4. ONNX Runtime TensorRT Plugin EP (reference implementation). https://github.com/onnxruntime/onnxruntime-ep-tensorrt
5. Microsoft. "Add a New Execution Provider." https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html
6. Microsoft. "EP Context Design." https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html
7. Microsoft. "Graph Optimizations in ONNX Runtime." https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
8. Microsoft. "TensorRT Execution Provider." https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
9. Microsoft Learn. "Select Execution Providers using Windows ML." https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers
10. NVIDIA. "End-to-End AI for NVIDIA-Based PCs: CUDA and TensorRT EPs in ONNX Runtime." https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/
11. LLVM Discourse. "Is There Existing Work to add ONNX Runtime Execution Provider based on MLIR or LLVM?" https://discourse.llvm.org/t/is-there-existing-work-to-add-onnx-runtime-execution-provider-based-on-mlir-or-llvm/86383

*(See `onnxrt-multi-ep.md` for full citation list including benchmark sources and MIGraphX EP documentation.)*
