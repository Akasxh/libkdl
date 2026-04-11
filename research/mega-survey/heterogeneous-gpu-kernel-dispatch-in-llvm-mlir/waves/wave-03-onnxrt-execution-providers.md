# Wave 03 — ONNX Runtime Execution Providers
**Angle:** onnx-runtime-execution-providers
**Query:** ONNX Runtime execution provider multi-backend dispatch CUDA TensorRT ROCm
**Date:** 2026-04-06

---

## Summary

ONNX Runtime's Execution Provider (EP) framework implements a greedy
priority-ordered graph partitioning scheme that is the most widely-deployed
production system for heterogeneous ML kernel dispatch today. Dispatch decisions
are made statically at session initialization via `GetCapability()` calls on each
registered EP in priority order. Subgraphs are fused and compiled into opaque
custom ops; device-boundary crossings generate auto-inserted Memcpy nodes.
Fallback is silent-by-default, a known production failure mode with an active
community push to invert the default. ROCm EP was deprecated (removed from tree
as of ORT 1.23); AMD dispatch now routes through MIGraphX EP. TensorRT RTX EP
(2025) introduces a two-phase AOT+JIT compilation model that is the most
hardware-portable design in the ORT EP ecosystem. ONNX-MLIR is a parallel
LLVM/MLIR-based compiler path for ONNX that represents the direct LLVM connection.

---

## Sources

### Source 1
**Title:** ONNX Runtime Execution Providers — Official Overview
**URL:** https://onnxruntime.ai/docs/execution-providers/
**Date:** Live doc, updated through ORT 1.22/1.23 (2025)
**Type:** Official documentation
**Relevance:** 9/10 | **Novelty:** 7/10
**Summary:** Canonical listing of all production and experimental EPs with hardware
targets, stability status, and priority-ordering semantics. Confirms ROCm EP
deprecation and MIGraphX as the AMD successor. Introduces TensorRT RTX EP for
consumer RTX hardware as distinct from datacenter TensorRT EP.
**Key Technical Details:**
- Priority is purely positional in the registered EP list: `['CUDAExecutionProvider', 'CPUExecutionProvider']` means CUDA is tried first, CPU is fallback.
- No dynamic re-prioritization occurs after session initialization.
- Production-stable EPs: CUDA, TensorRT, TensorRT RTX, DirectML, CoreML, XNNPACK, oneDNN, OpenVINO, NNAPI, QNN.
- ROCm EP explicitly deprecated; users redirected to MIGraphX or Vitis-AI EPs.
- Even if an op has a CUDA kernel, ORT may assign it to CPU for performance reasons (e.g., shape-related ops like `Shape`, `Squeeze`).

---

### Source 2
**Title:** ONNX Runtime Architecture — High Level Design
**URL:** https://onnxruntime.ai/docs/reference/high-level-design.html
**Date:** Live documentation (maintained 2024-2025)
**Type:** Official architecture reference
**Relevance:** 9/10 | **Novelty:** 8/10
**Summary:** Canonical description of the `PartitionGraph()` pipeline, the
`GetCapability()` EP interface contract, and the fused-op compilation model.
The greedy maximal-subgraph assignment algorithm is specified here.
**Key Technical Details:**
- `GetCapability()` returns `ComputeCapability` objects — each describes a claimable subgraph (set of node indices).
- Partitioner iterates EPs in priority order; each node is assigned to the first EP that claims it (greedy, first-fit, not globally optimal).
- When an EP claims a subgraph, ORT calls `EP::Compile()` on it, producing a fused custom op node.
- `MemcpyFromHost` / `MemcpyToHost` nodes are automatically inserted at boundaries between EPs with different device types.
- Each EP exposes an allocator; ORT pre-allocates input tensors for a partition using the EP's preferred allocator.
- The CPU EP is always last in the chain and accepts any unclaimed node — guaranteeing completeness.
- Nested subgraphs (e.g., control flow ops) are partitioned bottom-up (recursive descent).

---

### Source 3
**Title:** NVIDIA TensorRT Execution Provider
**URL:** https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
**Date:** Updated for ORT 1.22 / TensorRT 10.9 (2025)
**Type:** Official EP documentation
**Relevance:** 9/10 | **Novelty:** 8/10
**Summary:** Most detailed single-source description of production EP behavior at
the graph partitioning level. Contains concrete session initialization timing data
and documents the TRT→CUDA fallback path triggered by unsupported operators or
partitioning convergence failure.
**Key Technical Details:**
- TRT EP fallback is triggered when: (a) an op is not in TRT's supported op set, (b) `trt_max_partition_iterations` (default 1000) is exceeded, or (c) the subgraph is below `trt_min_subgraph_size` (default 1 node).
- On partitioning failure to converge, the entire model falls back to CUDA/CPU — not just the failing subgraph.
- Mandatory to register `CUDAExecutionProvider` alongside TRT EP; TRT subgraphs that fail fall to CUDA, not CPU.
- Session initialization timing for ResNet-class model:
  - No cache: ~384 seconds
  - Timing cache: ~42 seconds (TRT layer-timing data reuse)
  - Engine cache: ~9 seconds
  - EPContext embedded engine: ~1.9 seconds (200x reduction vs cold start)
- `trt_op_types_to_exclude` env var: explicitly banish specific op types from TRT (workaround for TRT 10.0-10.5 regression on `NonMaxSuppression`, `NonZero`, `RoiAlign`).
- All TRT subgraph inputs must have static shapes; dynamic shape models require explicit profile ranges via `trt_profile_min_shapes` / `trt_profile_max_shapes`.

---

### Source 4
**Title:** NVIDIA TensorRT RTX Execution Provider
**URL:** https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html
**Date:** 2025 (ORT 1.22+)
**Type:** Official EP documentation
**Relevance:** 8/10 | **Novelty:** 9/10
**Summary:** New EP for consumer RTX hardware that separates compilation into
Ahead-of-Time (AOT) and Just-in-Time (JIT) phases. AOT produces a portable binary
blob in an EPContext model; JIT compiles that blob to the exact GPU at inference
time. Directly relevant to portable kernel binary design.
**Key Technical Details:**
- AOT phase: `onnxruntime.compile()` API (ORT 1.22) converts ONNX model into an EPContext model storing optimized binary blobs — one blob per TRT partition.
- JIT phase: At inference time, TensorRT RTX dynamically compiles the blob into a hardware-specific engine tuned for the exact RTX generation (RTX 4090 vs RTX 5080).
- EPContext model is cross-GPU-generation portable at the blob level; hardware-specific tuning happens at load time with zero user overhead.
- `ModelCompilationOptions` API controls the AOT compilation pass.
- Tested with TensorRT 10.9; separate from datacenter TensorRT EP.
- This AOT→JIT split is the closest ORT analogue to a "compile-once, link-on-device" model akin to libkdl's `.kdl` format.

---

### Source 5
**Title:** MIGraphX Execution Provider (AMD)
**URL:** https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
**Date:** Live doc (ORT 1.20+)
**Type:** Official EP documentation
**Relevance:** 8/10 | **Novelty:** 7/10
**Summary:** AMD's primary GPU EP in ORT post-ROCm deprecation. Uses AMD's
MIGraphX deep learning graph optimization engine for AMD Radeon and Instinct GPUs.
**Key Technical Details:**
- MIGraphX EP uses AMD's graph optimizer to fuse ops and generate ROCm/HIP kernels.
- ROCm EP was removed from ORT source tree; MIGraphX is the current recommended AMD GPU path.
- Same `GetCapability()` / `Compile()` contract as CUDA EP, but output kernels target HIP/ROCm backend.
- Recommended priority pairing: `['MIGraphXExecutionProvider', 'CPUExecutionProvider']`.
- For unsupported ops, falls back to CPU EP (no equivalent of TRT→CUDA GPU-to-GPU fallback on AMD side).
- Vitis-AI EP is available as alternative for Xilinx FPGA/DPU targets.

---

### Source 6
**Title:** graph_partitioner.cc — ONNX Runtime Core Source
**URL:** https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/graph_partitioner.cc
**Date:** Active (mainline, 2024-2025)
**Type:** Source code
**Relevance:** 9/10 | **Novelty:** 8/10
**Summary:** Implementation of `PartitionGraph()` and the EP assignment algorithm.
Confirms the greedy, first-fit, maximal-subgraph assignment with bottom-up
recursion for nested graphs.
**Key Technical Details:**
- `IExecutionProvider` interface defined in `execution_provider.h` lines 116-627.
- `GetCapability()` returns `std::vector<std::unique_ptr<ComputeCapability>>`.
- Partitioner walks nodes in topological order; marks nodes with their assigned EP once claimed.
- CPU EP is registered last unconditionally; no node can remain unassigned after partitioning.
- Fused subgraph becomes a single `FunctionOp` node whose kernel calls `EP::Compile()` result.
- Memory copy insertion logic: if node N assigned to EP-A and its consumer assigned to EP-B where A.device != B.device, a `MemcpyToHost` (if A is GPU) or `MemcpyFromHost` (if B is GPU) node is injected between them.

---

### Source 7
**Title:** Addressing the Need for Disabled Fallback by Default in ONNX Runtime
**URL:** https://github.com/onnx/onnx/discussions/6623
**Date:** 2024 (GitHub Discussion)
**Type:** Community design discussion
**Relevance:** 8/10 | **Novelty:** 8/10
**Summary:** Active community debate proposing to invert the default fallback
behavior from silent-permissive to explicit-opt-in. Documents three real-world
production failure modes from silent CPU fallback.
**Key Technical Details:**
- Current default: any op not claimed by a registered EP silently falls to CPU.
- `session.disable_cpu_ep_fallback()` (added ORT 1.16.0) was intended to disable this — multiple bug reports (ORT #17801, ORT #23647) confirm it has no effect as of 2024-2025.
- Three documented failure modes: (1) profiling misdiagnosis — GPU telemetry looks fine but CPU is doing work; (2) OOM on CPU — model was sized for GPU VRAM, CPU fallback exhausts system RAM; (3) silent under-utilization — GPU cluster billed at full rate, effective GPU utilization 10-40%.
- Proposed fix: `disable_cpu_ep_fallback=True` as default, with structured logging emitting op name + EP resolution path for every node.
- No resolution as of research date; fallback remains silent-permissive by default.

---

### Source 8
**Title:** End-to-End AI for NVIDIA-Based PCs: CUDA and TensorRT EPs in ONNX Runtime
**URL:** https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/
**Date:** 2023 (NVIDIA Technical Blog)
**Type:** Technical blog post
**Relevance:** 7/10 | **Novelty:** 6/10
**Summary:** Developer-perspective comparison of CUDA EP vs TensorRT EP. Both target
NVIDIA GPUs but differ fundamentally in optimization scope — CUDA EP is per-op,
TensorRT EP is whole-subgraph.
**Key Technical Details:**
- CUDA EP: cuDNN kernel-per-op; exhaustive kernel search on first run (auto-tuning); predictable startup time (~seconds).
- TensorRT EP: whole-subgraph optimization with op reordering; multiple execution path candidates profiled; engine serialized to disk.
- TRT EP engine build time: "multiple minutes" for large models on first run.
- Key production recommendation: cache TRT engines keyed by `(model_hash, GPU_architecture)` — engines are not portable across GPU generations.
- Both EPs interchangeable at API level; application code can template over EP choice.
- TRT fallback path: unsupported TRT nodes → CUDA EP (GPU-to-GPU); no PCIe transfer needed for TRT→CUDA fallback within the same device.

---

### Source 9
**Title:** Is There Existing Work to Add an ONNX Runtime EP Based on MLIR or LLVM?
**URL:** https://discourse.llvm.org/t/is-there-existing-work-to-add-onnx-runtime-execution-provider-based-on-mlir-or-llvm/86383
**Date:** 2024 (LLVM Discourse)
**Type:** Community discussion
**Relevance:** 9/10 | **Novelty:** 9/10
**Summary:** LLVM Discourse thread directly exploring the intersection of ORT EP
architecture with MLIR/LLVM compiler infrastructure. Confirms that ONNX-MLIR is
a parallel but separate compiler path; no production MLIR-based ORT EP exists.
**Key Technical Details:**
- ONNX-MLIR (IBM Research, AMD, ByteDance contributors) is a standalone MLIR-based compiler for ONNX models — it does not plug into ORT as an EP.
- No MLIR-native EP exists in the ORT EP registry as of 2024.
- Community interest exists for an MLIR-based CPU EP that would use MLIR's lowering stack in place of ORT's current MLAS/Eigen CPU backend.
- ORT's internal CPU EP uses hand-tuned kernels (MLAS, Eigen) — an MLIR EP would enable automatic vectorization via LLVM's backend.
- ONNX-MLIR performs ONNX → MLIR → LLVM IR → native code; the resulting binary is monolithic (no EP dispatch layer).
- Direct relevance to libkdl: ORT's EP interface is the right abstraction boundary where an MLIR/LLVM-JIT backend could plug in — this gap is currently unfilled.

---

### Source 10
**Title:** ONNX-MLIR: An MLIR-based Compiler for ONNX AI Models (LLVM DevMtg 2025)
**URL:** https://llvm.org/devmtg/2025-06/slides/technical-talk/le-onnx.pdf
**Date:** June 2025 (LLVM Developers' Meeting)
**Type:** Conference slides
**Relevance:** 9/10 | **Novelty:** 9/10
**Summary:** LLVM DevMtg 2025 presentation on ONNX-MLIR's current state. The project
lowers ONNX models through MLIR dialects to LLVM IR, with GPU lowering via the MLIR
`gpu` and `nvvm`/`rocdl` dialects. Directly bridges the ORT EP dispatch world and
the MLIR GPU dialect research angle.
**Key Technical Details:**
- ONNX-MLIR lowering path: `onnx` dialect → `affine`/`linalg` → `scf`/`memref` → `llvm`/`nvvm`/`rocdl` dialects → LLVM IR → native binary.
- Over 100 operators lowered to MLIR; covers ResNet, BERT, LSTM workloads.
- Heterogeneous targeting: same ONNX model graph can lower partitions to LLVM IR (CPU) and NVVM/ROCDL (GPU), with explicit partition boundaries.
- Compiled binary is a single shared library with a runtime dispatch wrapper — closer to libkdl's design than ORT's EP system.
- Supports accelerator-specific passes (e.g., IBM Z14/Z16 NNPA backend) alongside standard CPU/GPU.
- Presented at the same venue (LLVM DevMtg) where libkdl poster would be shown — high audience overlap.

---

## Synthesis

### How ORT Decides Which EP Handles Which Operator

The dispatch decision is entirely static and made during session initialization:

1. `SessionOptions` receives an ordered EP list (user-specified priority).
2. `PartitionGraph()` iterates EPs in registration order.
3. Each EP's `GetCapability()` is called with the full graph; it returns the subgraph(s) it can handle.
4. Each node is assigned to the first EP that claims it (greedy first-fit).
5. Unclaimed nodes fall to the CPU EP unconditionally.
6. `Compile()` is called on each assigned subgraph, producing a fused custom op.
7. At inference time, each fused op dispatches to its pre-assigned backend — no runtime re-routing.

### The Fallback Chain (Concrete)

```
TensorRT EP
  |-- unsupported op or subgraph below min_size
  |-- partitioning iteration limit hit
  v
CUDA EP (GPU-to-GPU, no PCIe transfer)
  |-- op not in CUDA EP kernel list
  |-- shape ops (Shape, Squeeze) explicitly routed CPU-ward for perf
  v
CPU EP (guaranteed catch-all, but silent)
```

On AMD: `MIGraphX EP → CPU EP` (no GPU-to-GPU fallback tier between them).

### Key Design Differences vs libkdl

| Dimension | ORT EP System | libkdl Target |
|-----------|--------------|---------------|
| Dispatch timing | Static (session init) | Dynamic (per-call) |
| Granularity | Subgraph (multi-op fused) | Single kernel |
| Fallback visibility | Silent by default | Explicit contract |
| Backend portability | EP-specific binaries; device-specific engines | Hardware-agnostic `.kdl` binary |
| AMD path | MIGraphX EP (no ROCm) | Backend-agnostic |
| LLVM/MLIR integration | None (separate ONNX-MLIR project) | Native LLVM IR substrate |
| Overhead data | 384s→1.9s caching; per-Memcpy cost unpublished | Benchmark opportunity |

### Quantified Data Points for libkdl Paper

- TensorRT session init without caching: **~384s** (large model)
- With engine cache: **~9s** (42x reduction)
- With EPContext embedded: **~1.9s** (200x reduction)
- `disable_cpu_ep_fallback` API: added ORT 1.16.0, broken as of 2024-2025 (ORT #23647)
- Per-Memcpy-node overhead: **unpublished** — direct measurement opportunity for libkdl prototype

### LLVM/MLIR Connection (Key for Dublin Poster Framing)

ONNX-MLIR (presented at LLVM DevMtg 2025, same venue as libkdl) is the only
LLVM/MLIR-based ONNX compilation path. It compiles monolithically and does not
support runtime EP dispatch. The gap between ONNX-MLIR's static compilation model
and ORT's dynamic EP dispatch model is exactly the space libkdl occupies — a
runtime kernel linker that enables post-compilation dispatch decisions. Framing
libkdl as "what ONNX-MLIR needs to become a proper ORT EP replacement" could be a
compelling angle for the Dublin audience.

---

## Angle Assessment

**Relevance to heterogeneous GPU kernel dispatch:** 9/10

ORT is the dominant production reference for this problem domain. Its EP framework
is the concrete prior art that libkdl must position against. The static dispatch
vs dynamic dispatch distinction, the silent fallback problem, and the EPContext
caching mechanism all map directly to design decisions in libkdl. The ONNX-MLIR
connection discovered via LLVM Discourse adds a previously unlinked angle: the
only LLVM-native ONNX compilation path explicitly lacks the runtime dispatch layer
that libkdl provides, and this was presented at LLVM DevMtg 2025.

**Novelty relative to existing wave-04-onnxrt-ep.md:** 7/10

`wave-04-onnxrt-ep.md` covered the same territory with high quality. This report adds:
- Concrete `disable_cpu_ep_fallback` bug report references (ORT #17801, #23647)
- TensorRT RTX AOT/JIT two-phase design detail
- ROCm EP removal timeline precision
- ONNX-MLIR LLVM DevMtg 2025 source (new — not in wave-04)
- LLVM Discourse thread on MLIR-based ORT EP (new — not in wave-04)
- Concrete fallback chain diagram for AMD path

**Recommended action:** Merge the ONNX-MLIR LLVM DevMtg 2025 source and the
MLIR-based ORT EP gap finding into the synthesis document as a new research
direction. The `disable_cpu_ep_fallback` bug data strengthens the silent-fallback
argument in the libkdl motivation section.
