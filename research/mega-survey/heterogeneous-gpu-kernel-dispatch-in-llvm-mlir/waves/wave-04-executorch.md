# Wave 04 — ExecuTorch Edge Device Multi-Backend Dispatch

**Angle:** ExecuTorch Edge Device Multi-Backend Dispatch
**Search query:** "ExecuTorch delegate backend dispatch edge device multi-target XNNPACK CoreML GPU"
**Date:** 2026-04-06

---

## Sources

### Source 1
- **Title:** Understanding Backends and Delegates — ExecuTorch 1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html
- **Date:** 2025 (v1.1, stable)
- **Type:** Official docs
- **Relevance/Novelty:** 10/10
- **Summary:** Core reference for ExecuTorch's delegation model. Defines the `to_backend` API, partitioner protocol, `call_delegate` instruction, and two composition strategies for multi-backend graphs. The most architecturally dense document in the ExecuTorch corpus.
- **Key detail:** Multi-backend composition is achieved via either (a) sequential `to_backend()` calls — each partitioner claims nodes before the next runs — or (b) a unified partitioner that assigns distinct `DelegationSpec` tags to different node groups in a single pass. Both approaches produce a graph where some subgraphs become opaque `call_delegate` nodes dispatched to named backend strings at runtime.

---

### Source 2
- **Title:** Backend Overview — ExecuTorch 1.0/1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/backends-overview.html
- **Date:** 2025 (v1.0/1.1)
- **Type:** Official docs
- **Relevance/Novelty:** 9/10
- **Summary:** Enumerates all 14 supported backends with hardware targets and operator coverage. Defines the compile-time selection model: one `.pte` file per backend target, with multi-partitioner priority ordering for intra-file fallback.
- **Key detail:** 14 backends as of v1.1: XNNPACK (CPU/all), CUDA (NVIDIA GPU), Core ML (Apple NPU/GPU/CPU), MPS (Apple GPU), Vulkan (Android GPU), Qualcomm (Hexagon NPU), MediaTek NPU, Arm Ethos-U, Arm Cortex-M, Arm VGF, OpenVINO (Intel), NXP, Cadence (DSP), Samsung Exynos. Backend selection is compile-time; the recommendation for multi-platform is separate `.pte` files per target platform.

---

### Source 3
- **Title:** Architecture and Components — ExecuTorch 1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/getting-started-architecture
- **Date:** 2025 (v1.1)
- **Type:** Official docs
- **Relevance/Novelty:** 9/10
- **Summary:** Documents the full AOT-to-runtime pipeline. The runtime is deliberately minimal (kernel + backend registry, memory management, platform abstraction, profiling) with all optimization work pushed to AOT. The C++ runtime contains a static "kernel and backend registry" resolved at build link time.
- **Key detail:** The execution pipeline is: `torch.export()` → ATen Dialect EXIR → decompose to Core ATen → Edge Dialect → backend delegation (graph partitioning + `preprocess()` to binary blob) → flatbuffer `.pte` serialization → C++ runtime `call_delegate` dispatch. The runtime does NOT perform backend discovery or dynamic selection — everything is resolved AOT.

---

### Source 4
- **Title:** XNNPACK Backend — ExecuTorch 1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html
- **Date:** 2025 (v1.1)
- **Type:** Official docs / tutorial
- **Relevance/Novelty:** 7/10
- **Summary:** Concrete walkthrough of XnnpackPartitioner usage. Partitioner iterates graph nodes, tags compatible nodes with `delegation_tag` metadata, then `to_backend()` extracts tagged subgraphs and calls XNNPACK's `preprocess()` to serialize into XNNPACK flatbuffer blobs embedded in the `.pte`.
- **Key detail:** Each XNNPACK-delegated subgraph becomes a separate blob initialized independently at runtime via `XNNExecutor::init()`. The XNNPACK backend's `execute()` function runs the XNNPACK graph in-place. Unsupported nodes are left in the main graph and fall back to the Portable Kernel Library.

---

### Source 5
- **Title:** Vulkan Backend — ExecuTorch 0.7/1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/0.7/backends-vulkan.html
- **Date:** 2024–2025
- **Type:** Official docs
- **Relevance/Novelty:** 8/10
- **Summary:** Documents VulkanPartitioner which identifies GPU-compatible ops and creates GPU-dispatched subgraphs. Explicitly designed for partial delegation: a model can be lowered even when it contains unsupported ops — those remain on CPU Portable Kernels while GPU-compatible subgraphs run via Vulkan.
- **Key detail:** Vulkan delegate is cross-platform (Android primary, but any Vulkan-capable device). The partitioner marks "compatible sections" — the resulting `.pte` can have interleaved CPU (portable) and GPU (Vulkan) `call_delegate` subgraphs, with the runtime orchestrating handoff.

---

### Source 6
- **Title:** Core ML Backend — ExecuTorch 0.7/1.0 documentation
- **URL:** https://docs.pytorch.org/executorch/0.7/backends-coreml.html
- **Date:** 2024–2025
- **Type:** Official docs
- **Relevance/Novelty:** 7/10
- **Summary:** Core ML backend enables unified CPU/GPU/ANE dispatch on Apple hardware. CoreML itself performs internal hardware selection (CPU vs GPU vs Neural Engine) based on availability and model characteristics — ExecuTorch delegates the entire partitioned subgraph to CoreML, which then dispatches internally.
- **Key detail:** This represents a two-level dispatch hierarchy: ExecuTorch selects the subgraph partition for CoreML at AOT time; CoreML performs hardware dispatch (CPU/GPU/ANE) at runtime. This nested model is distinct from the flat backend registry model used by XNNPACK and Vulkan.

---

### Source 7
- **Title:** Qualcomm AI Engine Backend — ExecuTorch 1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- **Date:** 2025
- **Type:** Official docs
- **Relevance/Novelty:** 8/10
- **Summary:** QNN backend targets Qualcomm Hexagon DSP/NPU. Supports partial delegation — QNN-compatible ops go to Hexagon, remainder fall back to portable CPU kernels. Provides evidence that the partial-delegation pattern is the standard across all ExecuTorch backends.
- **Key detail:** Issue #4082 confirms partial QNN delegation is a production use case. The QNN backend uses the same `to_backend()` + partitioner interface as all other backends — architectural uniformity is a design principle.

---

### Source 8
- **Title:** [RFC] Multi-backend recipes for easy target-focused model lowering — GitHub Issue #13732
- **URL:** https://github.com/pytorch/executorch/issues/13732
- **Date:** 2025-08-27
- **Type:** RFC / GitHub issue
- **Relevance/Novelty:** 10/10
- **Summary:** RFC proposing `ExportRecipe` and `LoweringRecipe` classes to encapsulate backend-specific configuration (partitioner, transformation passes, compile specs) into composable objects. The `combine()` method merges recipes for multi-backend targets. Filed August 2025, represents the current direction for ExecuTorch's dispatch configuration API.
- **Key detail:** The RFC explicitly acknowledges that the existing API (manual `to_backend()` orchestration) has too much cognitive friction. The recipe approach pre-packages partitioner + transformation passes + compile specs into one object, then `combine()` merges multiple recipes for scenarios like "GPU-first with CPU fallback". This is the proposed high-level API over the existing low-level delegate infrastructure.

---

### Source 9
- **Title:** Kernel Registration — ExecuTorch 1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/kernel-library-custom-aten-kernel
- **Date:** 2025
- **Type:** Official docs
- **Relevance/Novelty:** 7/10
- **Summary:** Documents how custom kernels register into the ExecuTorch kernel registry at build time via YAML-based kernel spec files. At build time, a kernel resolver maps (operator name + tensor metadata) → kernel symbol, codegen generates C++ bindings.
- **Key detail:** Kernel selection in the non-delegated path is NOT runtime-dynamic — it is resolved at build-link time by the selective build system. The kernel registry is static. This is a sharp contrast to frameworks like TVM or IREE that perform runtime kernel dispatch.

---

### Source 10
- **Title:** Accelerating On-Device ML Inference with ExecuTorch and Arm SME2 — PyTorch Blog
- **URL:** https://pytorch.org/blog/accelerating-on-device-ml-inference-with-executorch-and-arm-sme2/
- **Date:** 2025
- **Type:** Blog / benchmark
- **Relevance/Novelty:** 6/10
- **Summary:** Concrete benchmark showing XNNPACK backend performance uplift from Arm SME2 hardware features. INT8: 1.83x speedup (556ms → 304ms), FP16: 3.9x speedup (1163ms → 298ms) on SqueezeSAM inference.
- **Key detail:** Performance gains are realized without changing the dispatch model — XNNPACK backend transparently uses SME2 when available. This demonstrates ExecuTorch's backend abstraction successfully hides hardware-specific microarchitecture from the dispatch layer.

---

## Synthesis

### ExecuTorch's Dispatch Architecture

ExecuTorch implements a **compile-time partitioning + runtime opaque-blob dispatch** model. The fundamental design choice is to push all dispatch decisions to AOT, producing a serialized program (`.pte` flatbuffer) where subgraphs are already assigned to backends. The runtime simply executes a sequence of operations, some of which are `call_delegate(backend_id, blob, inputs)` instructions.

**Dispatch pipeline:**
```
nn.Module
  → torch.export() [ATen Dialect EXIR]
  → to_edge() [Edge Dialect — dtype+dim_order annotations]
  → to_backend(partitioner_list) [graph partitioning]
      Each partitioner:
        1. Tags nodes with delegation_tag metadata
        2. to_backend() extracts tagged subgraphs
        3. Calls backend.preprocess(subgraph, compile_specs) → binary blob
        4. Replaces subgraph with call_delegate(backend_id, blob)
  → Serialization [flatbuffer .pte with embedded backend blobs]

Runtime:
  → Load .pte
  → For each instruction:
      if op: lookup kernel in static registry → execute
      if call_delegate: lookup backend by string ID in backend registry
                       → backend.init(blob) [first time]
                       → backend.execute(handle, inputs, outputs)
```

### Multi-Backend Composition

Two patterns in current code, one proposed in RFC:

1. **Sequential partitioners** (current): `to_backend([VulkanPartitioner(), XNNPACKPartitioner()])` — Vulkan claims GPU ops first, XNNPACK claims remaining CPU-compatible ops, unsupported ops fall back to portable kernels.

2. **Unified partitioner** (current): Single partitioner assigns different `DelegationSpec` tags to different node groups — gives finer control over partition boundaries but requires custom partitioner code.

3. **ExportRecipe + combine()** (RFC #13732, Aug 2025): High-level API that encapsulates partitioner + passes + compile specs into a recipe object; `combine()` merges recipes for multi-target deployment. Reduces setup to ~3 lines.

### Backend Registry at Runtime

The runtime backend registry is populated at **library load time** via static initializers (`register_backend()` called in global constructors). There is no dynamic discovery, no dlopen-based loading, and no runtime capability probing. Backends are linked into the binary at build time. This is architecturally analogous to a static symbol table — correct backend availability is enforced at build/link time, not at runtime.

### Nested Dispatch Hierarchy (CoreML)

CoreML introduces a two-level dispatch pattern: ExecuTorch partitions to CoreML at AOT; CoreML independently selects CPU/GPU/ANE at runtime based on device capabilities. This nested model provides hardware-transparent dispatch within a hardware-specific backend — a form of intra-backend heterogeneous dispatch not present in XNNPACK or Vulkan.

---

## Relevance to libkdl Dispatch Design

| ExecuTorch Pattern | Relevance to libkdl | Score |
|--------------------|---------------------|-------|
| String-keyed backend registry with static registration | Direct parallel: libkdl could use a similar `kdl_register_backend(name, vtable)` API with compile-time or dlopen-time registration | 9/10 |
| `call_delegate(backend_id, blob, inputs)` opaque subgraph dispatch | Maps to libkdl's per-kernel dispatch: dispatch by capability key, pass opaque compiled kernel blob | 8/10 |
| Sequential partitioner priority chain (GPU-first → CPU fallback) | Direct model for libkdl's fallback chain: CUDA → ROCm → SPIR-V → CPU | 10/10 |
| Compile-time backend selection → AOT `.pte` serialization | Contrasts with libkdl's goal of runtime dispatch — ExecuTorch deliberately avoids runtime backend selection, libkdl must solve this | 8/10 (as counter-example) |
| Portable Kernel Library as universal fallback | Direct parallel: libkdl's portable C/CPU path as guaranteed fallback when no accelerated backend matches | 9/10 |
| RFC #13732 ExportRecipe compose pattern | API design inspiration: libkdl could expose a "dispatch recipe" that composes capability requirements + fallback chain declaratively | 7/10 |
| CoreML nested dispatch (ExecuTorch → CoreML → ANE/GPU/CPU) | Important pattern: libkdl could delegate to vendor runtimes (e.g., CUDA, ROCm) that perform their own internal dispatch | 8/10 |
| Static kernel registry (build-link time, no runtime discovery) | Key gap libkdl must solve: ExecuTorch cannot select backends at runtime; libkdl's value proposition is doing exactly this via runtime capability probing | 10/10 (gap = libkdl opportunity) |

### Critical Insight for libkdl

ExecuTorch's architecture makes a deliberate tradeoff: maximum performance via AOT compilation at the cost of runtime flexibility. The absence of any runtime backend discovery or dynamic dispatch is a **design choice**, not a limitation — it enables a 50KB runtime. libkdl targets the opposite end of the tradeoff: it accepts the overhead of runtime dispatch in exchange for true heterogeneous adaptability (e.g., selecting between CUDA and ROCm based on what is actually present at execution time on a multi-GPU node).

The ExecuTorch `preprocess()` → binary blob model is directly relevant: libkdl could store pre-compiled kernel variants as named blobs (CUBIN, HSACO, SPIR-V) and select among them at dispatch time using a capability vector — achieving the best of both worlds: AOT compilation of variants, runtime selection of the correct variant.

---

## Risks / Gaps

- ExecuTorch's runtime backend dispatch has **no capability probing** — if the target device lacks a registered backend, it silently falls back to portable kernels. For libkdl, capability mismatch should be explicit (error or logged fallback), not silent.
- The static registry model **prevents hot-swapping backends** — a known limitation for deployments where GPU drivers or hardware availability changes between runs. libkdl should address this.
- No cross-backend **tensor format negotiation** at runtime — each backend independently determines its memory layout requirements during AOT. In libkdl's heterogeneous scenario (e.g., handing a CUDA tensor to a SPIR-V backend), format conversion overhead must be accounted for.
- ExecuTorch's `.pte` format **embeds backend-specific blobs opaquely** — this is a portability concern for multi-backend distribution (shipping one `.pte` for all targets requires all backend blobs, increasing file size). libkdl's kernel bundle format faces the same design tension.
