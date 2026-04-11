# Wave 04 — ONNX Runtime Multi-EP Dispatch
**Angle:** ONNX Runtime Execution Provider Multi-Backend Dispatch
**Query:** ONNX Runtime execution provider multi-backend dispatch CUDA TensorRT ROCm selection
**Date:** 2026-04-06

---

## Summary

ONNX Runtime's Execution Provider (EP) framework is the most widely deployed
production system for heterogeneous ML kernel dispatch across NVIDIA, AMD, Intel,
Qualcomm, and CPU backends. It implements a greedy priority-ordered graph
partitioning scheme where each EP claims the maximal subgraph it can handle, with
automatic MemcpyFromHost/MemcpyToHost insertion at device boundaries. The system
is production-validated at massive scale (Windows ML, Azure inference, mobile
Copilot+ PCs) and provides the clearest real-world evidence that multi-backend
dispatch is not just feasible but the dominant deployment pattern for ML inference.
The EP context caching design and its session initialization time data (384s raw →
1.9s with embedded engine) are directly quantifiable data points for the libkdl
dispatch overhead argument.

---

## Sources

### Source 1
**Title:** ONNX Runtime Architecture — High Level Design
**URL:** https://onnxruntime.ai/docs/reference/high-level-design.html
**Date:** Maintained 2024-2025 (live doc)
**Type:** Official documentation
**Relevance/Novelty:** 9/10
**Summary:** Canonical description of ORT's graph partitioning pipeline. Covers the
PartitionGraph() algorithm, GetCapability() interface, fused operator compilation via
Compile(), memory allocator exposure, and MemcpyFromHost/MemcpyToHost auto-insertion
at EP boundaries.
**Key Detail:** "The available execution providers will be considered in a specific
order, and each will be assigned the maximal subgraphs (possibly more than one) that
it is able to handle." Each partition is then reduced to a single fused operator via
the EP's `Compile()` method, wrapped as a custom op. Memory allocation follows
provider preferences — the EP exposes its allocator which is used to pre-allocate
input tensors for that partition. This greedy, ordered, maximal-subgraph approach is
the core algorithm driving all multi-EP dispatch in ORT.

---

### Source 2
**Title:** ONNX Runtime Execution Providers — Overview
**URL:** https://onnxruntime.ai/docs/execution-providers/
**Date:** Live doc, updated through ORT 1.22 (2025)
**Type:** Official documentation
**Relevance/Novelty:** 8/10
**Summary:** Lists all EPs with production/experimental status. Priority ordering
semantics. Confirms ROCm EP deprecation as of ORT 1.23 (ROCm 7.0 was the final
supported version), migration to MIGraphX. Introduces TensorRT RTX EP for RTX-class
consumer hardware.
**Key Detail:** Priority is purely positional: `['CUDAExecutionProvider',
'CPUExecutionProvider']` means CUDA is tried first, CPU is fallback. No runtime
capability negotiation beyond the initial partitioning pass. Production-stable EPs:
CUDA, TensorRT, TensorRT RTX, DirectML, CoreML, XNNPACK, oneDNN, OpenVINO, NNAPI,
QNN. Experimental/preview: TVM, CANN, Vitis-AI, Rockchip NPU. ROCm explicitly
deprecated.

---

### Source 3
**Title:** NVIDIA TensorRT Execution Provider — ORT Docs
**URL:** https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
**Date:** Updated for ORT 1.22 / TensorRT 10.9 (2025)
**Type:** Official documentation
**Relevance/Novelty:** 9/10
**Summary:** Most complete single-source description of production EP behavior. Covers
subgraph partitioning iteration limits, fallback triggering conditions, and three tiers
of engine caching with concrete session initialization timing data.
**Key Detail:** Session initialization benchmark with ResNet-class model:
- No cache: **384 seconds**
- Timing cache only: **42 seconds**
- Engine cache: **9 seconds**
- Embedded engine (EPContext model): **1.9 seconds**

Partitioning controls: `trt_max_partition_iterations` (default 1000) — if partitioning
fails to converge, the entire model falls back to CUDA/CPU. `trt_min_subgraph_size`
(default 1) — subgraphs below this threshold fall back automatically. Mandatory to
register CUDAExecutionProvider alongside TRT EP to handle incompatible nodes.
Known performance regression in TRT 10.0–10.5 for data-dependent shape ops
(NonMaxSuppression, NonZero, RoiAlign) — workaround via `trt_op_types_to_exclude`.

---

### Source 4
**Title:** EP Context Design
**URL:** https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html
**Date:** 2024-2025 (live doc)
**Type:** Official design spec
**Relevance/Novelty:** 9/10
**Summary:** Defines the EPContext ONNX node format in the `com.microsoft` domain
that encapsulates pre-compiled EP subgraphs. Solves the session creation latency
problem (especially for LLM-scale models where NPU compilation can take tens of
minutes) by enabling compiled binaries to be cached and reloaded without
recompilation.
**Key Detail:** EPContext node attributes: `main_context` (primary vs secondary
reference), `ep_cache_context` (binary payload or file path), `embed_mode` (1 =
embedded in model, 0 = external file), `source` (EP identifier string),
`ep_sdk_version`, `hardware_architecture` (version validation). EP implements
`GetEpContextNodes()` — partitioned subgraphs compile into backend SDK format, then
EPContext nodes wrap the binaries. Weight sharing across sessions via
`ep.share_ep_contexts` session option. For N models, outputs N+1 files: N context
models + 1 shared binary. This is effectively a "pre-linked kernel binary" approach —
directly analogous to what libkdl proposes at the dispatch level.

---

### Source 5
**Title:** End-to-End AI for NVIDIA-Based PCs: CUDA and TensorRT EPs in ONNX Runtime
**URL:** https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/
**Date:** 2023 (NVIDIA Technical Blog)
**Type:** Blog / technical deep-dive
**Relevance/Novelty:** 7/10
**Summary:** Practical architecture comparison of CUDA EP vs TensorRT EP from an
application developer perspective. CUDA EP = cuDNN kernel-per-op with exhaustive
kernel search on first run. TensorRT EP = whole-graph optimization, operation
reordering, path profiling, engine serialization.
**Key Detail:** The two providers are "quite interchangeable because they're based on
CUDA." Application code can template over the EP choice, suggesting a clean runtime
substitution interface. TensorRT EP has "multiple minutes" engine build time for large
models but then delivers superior inference speed. Key production recommendation:
cache TensorRT engines to disk keyed by GPU architecture (engines are device-specific),
pre-build at install time, not on first user inference.

---

### Source 6
**Title:** Core Architecture — DeepWiki microsoft/onnxruntime
**URL:** https://deepwiki.com/microsoft/onnxruntime/3-architecture
**Date:** 2024 (community doc)
**Type:** Architecture reference
**Relevance/Novelty:** 8/10
**Summary:** Code-level walkthrough of PartitionGraph() in `graph_partitioner.cc`.
Identifies the exact partitioning sequence and how MemcpyFromHost/MemcpyToHost nodes
are inserted at boundaries.
**Key Detail:** Partitioner iterates EPs in registered order → calls `GetCapability()`
to get supported node subgraphs → assigns nodes to first EP that claims them →
inserts `MemcpyFromHost`/`MemcpyToHost` at EP boundaries → CPU EP catches all
unclaimed nodes. Registration order is performance-critical: earlier-registered EPs
get priority on contested nodes. `IExecutionProvider` interface defined at
`execution_provider.h:116-627`. The greedy first-fit nature means EP ordering in
`SessionOptions` is the primary tuning lever for multi-EP workloads.

---

### Source 7
**Title:** Addressing the Need for Disabled Fallback by Default in ONNX Runtime
**URL:** https://github.com/onnx/onnx/discussions/6623
**Date:** 2024 (GitHub Discussion)
**Type:** Community discussion / design debate
**Relevance/Novelty:** 7/10
**Summary:** Community debate about making CPU fallback opt-in rather than silent
default. Documents three real-world failure modes of silent fallback: misdiagnosed
profiling, OOM crashes from unexpected CPU memory allocation, and silent GPU
under-utilization.
**Key Detail:** Core problem: fallback is silent by default. Users allocate GPU
infrastructure, observe expected latency, but significant compute is actually running
on CPU because of unsupported ops. Proposed fixes: make `disable_cpu_ep_fallback=True`
the default; add verbose logging with op names + EP resolution path. No overhead
measurements in the discussion — qualitative impact only. This is a direct design
tension in production multi-EP dispatch: correctness (all ops run) vs performance
(silent fallback can 10-100x degrade latency).

---

### Source 8
**Title:** Performance Investigation — microsoft/onnxruntime Wiki
**URL:** https://github.com/microsoft/onnxruntime/wiki/Performance-Investigation
**Date:** Maintained (live wiki)
**Type:** Developer guide
**Relevance/Novelty:** 7/10
**Summary:** ORT's recommended methodology for profiling multi-EP execution. Uses nvprof
and Visual Profiler for GPU kernel identification. Key diagnostic: search generated
ONNX graph for `memcpy` nodes — zero should be the target for fully GPU-resident
models.
**Key Detail:** MemcpyToHost + MemcpyFromHost sandwiching a node is the canonical
indicator of a missing CUDA kernel. In a reported issue (ORT #16625), iterative
application of a neural network via CUDA EP was dominated by Memcpy time — graph
contained O(N) memcpy boundaries proportional to the number of loop iterations. The
wiki also identifies Cast node proliferation (bad mixed-precision handling) and
unfused LayerNorm/GELU/MatMul as common secondary bottlenecks in multi-EP graphs.

---

### Source 9
**Title:** Choosing Execution Providers for Performance
**URL:** https://pkreg101.github.io/onnxruntime/docs/performance/choosing-execution-providers.html
**Date:** ORT documentation mirror
**Type:** Official guidance doc
**Relevance/Novelty:** 6/10
**Summary:** Decision matrix for EP selection. Quantifies the conceptual cost of EP
switching ("significant performance impact"). No absolute latency numbers. Windows GPU
→ DirectML, NVIDIA maximum perf → TensorRT, NVIDIA compatibility → CUDA, cross-platform
→ CPU.
**Key Detail:** "Switching between CPU and GPU can cause significant performance
impact." When TensorRT cannot handle subgraphs it automatically falls back to CUDA
(not to CPU), preserving on-device execution. This TRT→CUDA fallback path is
important: it is a GPU-to-GPU fallback that avoids PCIe data movement, qualitatively
different from GPU-to-CPU fallback.

---

### Source 10
**Title:** Unlocking the Power of Qualcomm QNN Execution Provider GPU Backend for ONNX Runtime
**URL:** https://www.qualcomm.com/developer/blog/2025/05/unlocking-power-of-qualcomm-qnn-execution-provider-gpu-backend-onnx-runtime
**Date:** May 2025
**Type:** Vendor blog
**Relevance/Novelty:** 7/10
**Summary:** Announces QNN EP preview with Adreno GPU backend alongside existing HTP
(Hexagon NPU) backend. Snapdragon X Elite can now dispatch to NPU (HTP), GPU (Adreno
via QNN EP), or GPU (DirectML EP) depending on operator support and configured
priority.
**Key Detail:** QNN EP supports configurable backends: `htp` (Hexagon NPU), `cpu`,
`gpu` (Adreno). The EP context caching design is critical here — NPU compilation of
LLM-scale models can take tens of minutes; EPContext nodes amortize this. Windows ML
distributes QNN EP updates as OS components (KB5067994 updated QNN EP to v1.8.13.0
for Copilot+ PCs). This is the most complex heterogeneous dispatch case in ORT: a
single model session can partition across NPU + GPU + CPU on a single Snapdragon SoC,
each with different memory spaces and compilation requirements.

---

## Synthesis

### The ORT Multi-EP Dispatch Model (How It Actually Works)

1. **Session initialization** — `SessionOptions` receives an ordered EP list.
2. **Graph partitioning** — `PartitionGraph()` in `graph_partitioner.cc` iterates EPs
   in priority order. Each EP's `GetCapability()` returns `ComputeCapability` objects
   describing claimable subgraphs. The partitioner assigns each node to the first EP
   that claims it (greedy first-fit, not globally optimal).
3. **Boundary insertion** — Where adjacent partitions belong to different device types,
   `MemcpyFromHost` and `MemcpyToHost` nodes are automatically inserted. These are the
   primary runtime dispatch overhead.
4. **Compilation** — Each partition's EP `Compile()` method is called, returning a fused
   custom op. TensorRT EP runs TRT engine build here; QNN EP runs HTP/Adreno
   compilation; CUDA EP resolves cuDNN kernels.
5. **Runtime execution** — The fused ops execute in their assigned provider. Memory is
   managed by each provider's exposed allocator.
6. **Caching** — EPContext nodes encode compiled binaries; subsequent sessions skip
   steps 2-4 entirely.

### Key Tensions Relevant to libkdl

| Tension | ORT Behavior | libkdl Opportunity |
|---|---|---|
| Dispatch at partitioning time vs runtime | Static: partitioned once at session init | Dynamic: per-call dispatch based on runtime load/availability |
| Silent fallback | Default-on, causes silent perf regression | Explicit: caller knows which backend ran |
| Granularity | Subgraph-level (multiple ops fused) | Kernel-level (individual dispatch) |
| Overhead amortization | EPContext caching amortizes compilation | `.kdl` binary amortizes linking |
| Cross-device data movement | Auto-inserted Memcpy nodes, not quantified | Explicit copy budget in dispatch decision |
| AMD support | ROCm deprecated; MIGraphX is successor | Backend-agnostic from design |

### Quantified Overhead Data Points

- TensorRT session initialization without caching: **~384 seconds** (large model)
- With engine cache: **~9 seconds** (42x reduction)
- With embedded EPContext: **~1.9 seconds** (200x reduction vs cold start)
- Memcpy-dominated inference: reported cases where PCIe copy time exceeds kernel time
  for iterative models (ORT issue #16625) — no exact numbers published

### Production Validation Evidence

- Windows ML ships EP dispatch as OS infrastructure (Copilot+ PCs, Windows 11 24H2)
- Azure inference uses ORT as the primary inference runtime across CUDA and CPU backends
- Qualcomm QNN EP distributed as Windows Update component (KB5067994)
- Immich open-source application updated to ORT 1.22 with OpenVINO + ROCm→MIGraphX migration (PR #23458)

### Gaps / Open Questions for libkdl

1. No published per-Memcpy-node overhead measurement (microseconds per H2D/D2H copy
   at inference batch boundaries).
2. No formal analysis of greedy-vs-optimal partitioning quality gap.
3. TRT→CUDA GPU-to-GPU fallback path overhead is undocumented.
4. No data on multi-EP session stability under device hot-plug or EP failure at runtime.

---

## Relevance to "Production-Validated Multi-Backend Dispatch Patterns"

**Rating: 9/10**

ORT is the production reference implementation for multi-backend ML kernel dispatch.
Every major claim in libkdl's design space (operator routing, fallback, caching,
heterogeneous memory management) has a direct ORT counterpart with years of production
deployment. The EPContext caching mechanism is the closest existing analogue to the
`.kdl` binary format proposed for libkdl. The silent-fallback design debate (source 7)
directly motivates libkdl's explicit dispatch contract. The Memcpy boundary overhead
problem is the quantitative gap this research should fill — ORT acknowledges it
qualitatively but never publishes numbers, which creates a specific measurement
opportunity for the libkdl prototype.
