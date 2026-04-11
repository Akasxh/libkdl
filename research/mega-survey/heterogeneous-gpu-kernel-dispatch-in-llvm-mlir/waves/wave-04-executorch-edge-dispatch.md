# Wave 04 — ExecuTorch Edge Inference Dispatch: Backend Delegation and NPU/GPU Heterogeneous Execution

**Angle:** edge-inference-dispatch-executorch
**Search query:** "ExecuTorch edge inference dispatch backend delegation heterogeneous CPU GPU NPU"
**Date:** 2026-04-06

> **Note:** `wave-04-executorch.md` covers the core delegation protocol and multi-backend composition API in depth. This report focuses on complementary angles: edge-specific NPU/DSP backends introduced in 2025, the OpenVINO heterogeneous runtime integration, the comparison with IREE and ONNX Runtime dispatch strategies, and the design tension between AOT-only vs. runtime-dynamic dispatch at the edge.

---

## Sources

### Source 1
- **Title:** Backends Overview — ExecuTorch 1.0/1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/backends-overview.html
- **Date:** 2025 (v1.0/1.1)
- **Type:** Official docs
- **Relevance:** 10/10 | **Novelty:** 8/10
- **Summary:** Enumerates all 14 hardware backends as of v1.1 across the full edge spectrum: mobile CPU (XNNPACK), Android GPU (Vulkan), Apple (Core ML, MPS), Qualcomm (Hexagon DSP/NPU), MediaTek NPU, Arm Ethos-U NPU, Arm Cortex-M, Arm VGF, Intel (OpenVINO), NXP, Cadence DSP, Samsung Exynos DSP/NPU, CUDA. Distinguishes backends by their partitioning fidelity (full-graph vs. partial) and hardware targeting.
- **Key detail:** Backend selection is compile-time. For multi-target deployment, the recommended pattern is separate `.pte` files per hardware target — there is no single binary that dynamically selects among CUDA, Vulkan, and Ethos-U at runtime. This is the central gap libkdl addresses.

---

### Source 2
- **Title:** Arm Ethos-U NPU Backend — ExecuTorch 1.0 documentation
- **URL:** https://docs.pytorch.org/executorch/1.0/backends-arm-ethos-u.html
- **Date:** 2025
- **Type:** Official docs
- **Relevance:** 8/10 | **Novelty:** 8/10
- **Summary:** Ethos-U backend demonstrates ExecuTorch's NPU delegation model for deeply embedded targets (Cortex-M + Ethos-U55/U65). The AOT partitioner identifies TOSA-lowerable subgraphs; `preprocess()` generates Vela-compiled command streams. At runtime, a cortex-m-specific backend executes the Vela command stream on the NPU, with the remaining ops executing on the Cortex-M CPU via portable kernels.
- **Key detail:** Arm blog (2025, Ethos-U and Beyond) confirms this pattern extends across the full Arm IP stack — the same ExecuTorch export workflow targets Ethos-U NPUs, Cortex-A CPUs, and Cortex-M MCUs by selecting the appropriate partitioner. The heterogeneous CPU+NPU model is idiomatic, not exceptional.

---

### Source 3
- **Title:** Optimizing ExecuTorch on Intel AI PCs with OpenVINO Backend
- **URL:** https://www.intel.com/content/www/us/en/developer/articles/community/optimizing-executorch-on-ai-pcs.html
- **Date:** 2025
- **Type:** Intel developer blog
- **Relevance:** 9/10 | **Novelty:** 9/10
- **Summary:** OpenVINO backend for ExecuTorch uses a two-level heterogeneous dispatch architecture. ExecuTorch's partitioner delegates supported subgraphs to the OpenVINO backend; OpenVINO itself then performs runtime hardware selection across Intel CPU, iGPU, and NPU (on Core Ultra Series 2 AI PCs). Supports 4-bit INT4 LLM weight compression and INT8 vision model quantization via `OpenVINOQuantizer`.
- **Key detail:** The OpenVINO backend introduces **runtime hardware dispatch inside the backend** — the ExecuTorch delegate boundary is opaque, and OpenVINO performs its own device selection (CPU/iGPU/NPU) based on runtime capability probing. This is the same two-level nested dispatch pattern as CoreML (OpenVINO independently implements the same pattern on the Intel stack). The backend guarantees a full model execution via CPU fallback for unsupported ops.

---

### Source 4
- **Title:** Samsung Exynos Backend — ExecuTorch GitHub / Release Notes
- **URL:** https://github.com/pytorch/executorch/releases
- **URL:** https://github.com/pytorch/executorch/issues/16395
- **Date:** 2025
- **Type:** GitHub release notes + issue tracker
- **Relevance:** 7/10 | **Novelty:** 9/10
- **Summary:** Samsung Exynos backend introduced in 2025, enabling ExecuTorch delegation to Samsung SoC DSP/NPU via the Samsung Neural Processing SDK. Supports 60 operators. Follows the standard ExecuTorch `to_backend(ExynosPartitioner())` + `preprocess()` protocol. Issue #16395 documents real-device integration debugging — early production use cases on Galaxy hardware.
- **Key detail:** The Exynos backend joins MediaTek NPU as a second Samsung-family hardware target, highlighting the ecosystem expansion: ExecuTorch's uniform delegate API now covers 14+ distinct hardware execution units, from ARM MCUs to NVIDIA CUDA GPUs, using the same `Partitioner` + `preprocess()` + `call_delegate` interface.

---

### Source 5
- **Title:** Vulkan GPU Backend (DeepWiki / ExecuTorch docs)
- **URL:** https://deepwiki.com/pytorch/executorch/5.2-vulkan-backend
- **URL:** https://docs.pytorch.org/executorch/0.6/backends-vulkan.html
- **Date:** 2024–2025
- **Type:** Official docs + community analysis
- **Relevance:** 8/10 | **Novelty:** 7/10
- **Summary:** VulkanPartitioner identifies GPU-compatible nodes based on operator support, tensor representation, and hardware constraints. Operator coverage is currently limited (binary arithmetic, reductions, convolution with fusions), so models typically split: GPU (Vulkan) + CPU (Portable Kernels). Samsung GPU compute partnership added operator fusions (clamp-to-convolution) and optimized compute shaders in 2025. Cross-platform: Android primary, but any Vulkan-capable device including embedded Linux targets.
- **Key detail:** The interleaving of Vulkan `call_delegate` and portable-kernel instructions in a single `.pte` graph is the canonical ExecuTorch heterogeneous dispatch pattern — it makes the CPU/GPU boundary explicit in the serialized graph rather than hidden inside a backend runtime. This is architecturally different from IREE's HAL abstraction, which hides the CPU/GPU split inside the compiled artifact.

---

### Source 6
- **Title:** RFC: Multi-Backend Recipes for Easy Target-Focused Model Lowering — Issue #13732
- **URL:** https://github.com/pytorch/executorch/issues/13732
- **Date:** 2025-08-27
- **Type:** RFC / GitHub issue
- **Relevance:** 10/10 | **Novelty:** 10/10
- **Summary:** Proposes `ExportRecipe` + `LoweringRecipe` classes as a high-level API over the existing low-level `to_backend()` orchestration. Each `LoweringRecipe` encapsulates: (a) pre-edge transformation passes, (b) a partitioner instance, (c) compile specs. `combine(recipes)` merges multiple recipes for multi-target scenarios. Represents the current trajectory for ExecuTorch's dispatch configuration API.
- **Key detail:** The RFC explicitly acknowledges that the current API requires users to manually orchestrate partitioner ordering, transformation passes, and compile specs — too much cognitive friction for the target audience (mobile ML developers). The `combine()` pattern directly supports priority-ordered fallback: `combine([QNNRecipe(), XNNPACKRecipe()])` produces "Hexagon NPU first, CPU fallback" in ~3 lines. Filed August 2025, likely landing in v1.2 or v1.3.

---

### Source 7
- **Title:** ExecuTorch Concepts and Architecture — ExecuTorch 1.1 documentation
- **URL:** https://docs.pytorch.org/executorch/stable/concepts.html
- **URL:** https://docs.pytorch.org/executorch/stable/getting-started-architecture
- **Date:** 2025
- **Type:** Official docs
- **Relevance:** 9/10 | **Novelty:** 7/10
- **Summary:** Defines the full AOT pipeline: ATen Dialect → Core ATen decomposition → Edge Dialect (dtype + dim_order annotations) → backend delegation → flatbuffer `.pte`. The Edge Dialect is the crucial pre-delegation IR: scalars are converted to tensors, dtype/dim_order are explicit, and the operator set is constrained to enable selective build. Backend delegation happens after Edge lowering, so all backends see the same strongly-typed, tensor-only graph.
- **Key detail:** The Edge Dialect's dtype and dim_order information allows partitioners to make hardware-aware decisions (e.g., skip FP64 nodes on NPUs that only support FP16/INT8) without inspecting the full graph semantics. This is a clean separation between type-system constraints and hardware capability matching.

---

### Source 8
- **Title:** ExecuTorch vs IREE vs ONNX Runtime — ML Runtimes Comparison
- **URL:** https://aman.ai/primers/ai/ml-runtimes/
- **URL:** https://medium.com/@Modexa/eight-torchscript-alternatives-for-the-pytorch-2-x-era-34dcb68f2940
- **Date:** 2025
- **Type:** Community analysis / blog
- **Relevance:** 8/10 | **Novelty:** 7/10
- **Summary:** Comparison of three heterogeneous dispatch strategies across ExecuTorch, IREE, and ONNX Runtime. ONNX Runtime uses a sub-graph partition + Execution Provider (EP) model where EPs register capability constraints and the runtime assigns nodes at graph load time — closest to runtime dispatch. IREE uses a multi-target compiled artifact (HAL executables) with runtime device selection among pre-compiled variants. ExecuTorch uses AOT-only partitioning, maximizing performance at the cost of runtime flexibility.
- **Key detail:** Three distinct dispatch philosophies, all solving the same heterogeneous operator assignment problem at different points on the compile-time/runtime tradeoff axis: ONNX Runtime (most runtime-dynamic), IREE (pre-compiled variants, runtime selection), ExecuTorch (fully AOT, least runtime overhead). libkdl targets the IREE point on this spectrum — AOT compilation of multiple variants, runtime selection among them.

---

### Source 9
- **Title:** Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend — ExecuTorch docs
- **URL:** https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- **Date:** 2025
- **Type:** Official docs
- **Relevance:** 7/10 | **Novelty:** 6/10
- **Summary:** QNN backend supports Qualcomm Kryo CPU, Adreno GPU, and Hexagon DSP/NPU through a unified API (QNN SDK). The ExecuTorch `QnnPartitioner` delegates to whichever Qualcomm accelerator is configured at AOT time. This is a three-way heterogeneous accelerator inside a single vendor's SoC — all mediated by QNN's abstraction layer, then wrapped in ExecuTorch's delegate interface.
- **Key detail:** Qualcomm's QNN SDK performs its own hardware selection (CPU/GPU/NPU) below the ExecuTorch delegate boundary — another instance of the nested dispatch pattern. This makes the dispatch hierarchy three levels deep for Qualcomm: ExecuTorch partitioner → QNN backend → Hexagon/Adreno/Kryo.

---

### Source 10
- **Title:** Deploy YOLO11 on Mobile & Edge with ExecuTorch — Ultralytics Docs
- **URL:** https://docs.ultralytics.com/integrations/executorch/
- **Date:** 2025
- **Type:** Integration guide / use case
- **Relevance:** 6/10 | **Novelty:** 5/10
- **Summary:** Real-world use case: YOLO11 object detection model exported to ExecuTorch for mobile/edge deployment with XNNPACK delegation. Illustrates the end-to-end workflow from production PyTorch model to on-device inference and shows how the abstraction layers (torch.export → to_edge → to_backend) work in practice for non-trivial vision models.
- **Key detail:** YOLO11 on ExecuTorch: model is partitioned, XNNPACK handles conv/linear/pool subgraphs, non-XNNPACK-compatible ops (e.g., complex control flow) fall back to portable kernels. Demonstrates that partial delegation (not full-model delegation) is the practical production mode for real models, not just toy examples.

---

## Synthesis

### The Edge Inference Dispatch Problem

ExecuTorch's fundamental design question is: *how do you run a PyTorch model efficiently across 14 heterogeneous hardware targets (from MCUs to NPUs to mobile GPUs) while keeping the runtime under 50KB?* The answer is: **push all dispatch decisions to AOT, make the runtime a pure executor of a pre-partitioned instruction stream.**

This produces the cleanest possible runtime dispatch path — a `call_delegate(backend_id, blob, inputs)` instruction is O(1) overhead; no capability probing, no kernel selection, no format negotiation at inference time. The cost is a separate `.pte` file per hardware target and no runtime adaptability.

### The Nested Dispatch Hierarchy Pattern

Three ExecuTorch backends (CoreML, OpenVINO, QNN) implement a two-level dispatch hierarchy where ExecuTorch assigns subgraphs to the backend at AOT, and the backend itself performs runtime hardware selection (ANE/GPU/CPU, iGPU/NPU/CPU, Hexagon/Adreno/Kryo). This is the pragmatic approach for platforms where vendor SDKs already solve intra-SoC dispatch (Apple, Intel, Qualcomm). ExecuTorch acts as the inter-backend orchestrator; vendor SDKs act as intra-backend dispatchers.

```
ExecuTorch partitioner (AOT)
  → CoreML subgraph blob
      CoreML runtime (device runtime)
        → ANE / GPU / CPU  [runtime selection, opaque to ExecuTorch]
  → XNNPACK subgraph blob
      XNNPACK runtime (CPU portable)
  → Portable kernels (fallback)
```

### Dispatch Strategy Comparison (ExecuTorch vs. IREE vs. ONNX Runtime)

| Dimension | ExecuTorch | IREE | ONNX Runtime |
|-----------|-----------|------|--------------|
| Dispatch point | AOT only | AOT compile + runtime select | Runtime (graph load) |
| Backend discovery | Build-link time (static) | Runtime (HAL device enum) | Runtime (EP registration) |
| Kernel variants | One per `.pte` file | Multiple variants per artifact | One per EP |
| Fallback model | Portable Kernel Library (linked in) | CPU HAL device | CPU EP |
| Runtime overhead | Minimal (O(1) call_delegate) | Low (HAL dispatch table) | Higher (EP capability query) |
| Adaptability | None (requires recompile) | Medium (selects pre-compiled variant) | High (can swap EPs at load time) |
| Binary size | ~50KB runtime | Medium | Large |
| Edge suitability | Highest | Medium | Lowest |

libkdl's target position: **between IREE and ONNX Runtime** — runtime dispatch among AOT-compiled variants (like IREE), but with explicit capability probing and fallback chain semantics (like ONNX Runtime's EP model), at a binary footprint closer to ExecuTorch.

### RFC #13732 — The Direction of ExecuTorch Dispatch API

The `ExportRecipe` / `LoweringRecipe` / `combine()` pattern (RFC #13732, August 2025) is the most directly relevant ExecuTorch development for libkdl API design:

```python
# RFC #13732 proposed API (simplified)
qnn_recipe = QNNRecipe(compile_specs=[...])
xnnpack_recipe = XNNPACKRecipe()
combined = combine([qnn_recipe, xnnpack_recipe])  # priority-ordered
exported = export_for_edge(model, inputs, recipe=combined)
```

This maps cleanly to a libkdl dispatch recipe:
```c
kdl_recipe_t r = kdl_recipe_combine(
    kdl_recipe_cuda(cubin_blob),    // GPU-first
    kdl_recipe_hip(hsaco_blob),     // AMD fallback
    kdl_recipe_spirv(spv_blob),     // cross-vendor fallback
    kdl_recipe_cpu(native_blob)     // guaranteed fallback
);
kdl_dispatch(handle, r, inputs, outputs);
```

---

## Angle Assessment

- **Relevance to heterogeneous GPU kernel dispatch in LLVM/MLIR:** 9/10
  ExecuTorch is the most mature production system for heterogeneous multi-backend ML kernel dispatch at the edge. Its architecture directly defines what libkdl must either replicate (uniform delegate API, priority-chain fallback) or improve upon (runtime backend discovery, dynamic dispatch among AOT-compiled variants).

- **Novelty relative to existing literature:** 7/10
  The core delegation protocol is well-documented. Novel angles here: (a) the nested dispatch hierarchy pattern (ExecuTorch → vendor SDK → hardware) as a reusable design pattern, (b) the RFC #13732 recipe API as the emerging high-level dispatch configuration abstraction, (c) the three-way dispatch philosophy comparison (AOT-only vs. compiled-variants vs. runtime-dynamic).

- **Actionability for libkdl design:** 10/10
  Highest-scoring source set for direct API and architecture decisions. The `preprocess() → binary blob → call_delegate` pipeline, priority-chain partitioner ordering, and nested dispatch hierarchy are all directly transferable patterns.

---

## Risks / Gaps Identified

1. **No cross-backend tensor format negotiation at ExecuTorch runtime** — each backend independently manages memory layout during AOT preprocessing. libkdl's runtime dispatch must handle tensor format conversion (CUDA NCHW → Vulkan NCHW4, etc.) on-the-fly; ExecuTorch avoids this by resolving formats AOT.

2. **Static backend registry prevents hot-swap** — ExecuTorch cannot switch backends without recompilation. In cloud or server heterogeneous scenarios (GPU availability changes between jobs), this is a hard limitation. libkdl must solve this with a runtime-loadable backend registry (analogous to `dlopen` for kernel backends).

3. **Separate `.pte` per target creates distribution complexity** — shipping a model to a heterogeneous fleet requires maintaining multiple files. libkdl's kernel bundle format should address this: one bundle containing multiple compiled variants (CUBIN + HSACO + SPIR-V), with runtime selection.

4. **Vendor SDK nested dispatch is opaque** — for platforms like Qualcomm and Intel, ExecuTorch delegates dispatch to closed SDKs. libkdl cannot use this model for open multi-vendor dispatch; it must implement its own runtime capability probing at the kernel selection layer.

5. **RFC #13732 not yet merged (as of April 2026)** — the high-level recipe API is still a proposal. The current ExecuTorch API requires manual partitioner orchestration, which is error-prone at scale. Watch for v1.2/v1.3 landing to see the final API shape.

---

## Sources (URL Reference List)

- [Understanding Backends and Delegates — ExecuTorch 1.1](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)
- [Backends Overview — ExecuTorch 1.0/1.1](https://docs.pytorch.org/executorch/stable/backends-overview.html)
- [Arm Ethos-U NPU Backend — ExecuTorch 1.0](https://docs.pytorch.org/executorch/1.0/backends-arm-ethos-u.html)
- [Ethos-U and Beyond: How ExecuTorch 1.0 Powers AI at the Edge — Arm Blog](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/ethos-u-and-beyond-how-executorch-1-0-powers-ai-at-the-edge)
- [Optimizing ExecuTorch on Intel AI PCs with OpenVINO Backend — Intel](https://www.intel.com/content/www/us/en/developer/articles/community/optimizing-executorch-on-ai-pcs.html)
- [Building and Running ExecuTorch with OpenVINO Backend — ExecuTorch 1.0](https://docs.pytorch.org/executorch/1.0/build-run-openvino.html)
- [Vulkan GPU Backend — DeepWiki](https://deepwiki.com/pytorch/executorch/5.2-vulkan-backend)
- [Vulkan Backend — ExecuTorch 0.6](https://docs.pytorch.org/executorch/0.6/backends-vulkan.html)
- [RFC: Multi-Backend Recipes — Issue #13732](https://github.com/pytorch/executorch/issues/13732)
- [Qualcomm AI Engine Backend — ExecuTorch 1.0](https://docs.pytorch.org/executorch/1.0/backends-qualcomm.html)
- [Architecture and Components — ExecuTorch 1.1](https://docs.pytorch.org/executorch/stable/getting-started-architecture)
- [Concepts — ExecuTorch 1.1](https://docs.pytorch.org/executorch/stable/concepts.html)
- [Samsung Exynos Backend Issue #16395](https://github.com/pytorch/executorch/issues/16395)
- [ExecuTorch Releases](https://github.com/pytorch/executorch/releases)
- [Deploy YOLO11 on Mobile & Edge with ExecuTorch — Ultralytics](https://docs.ultralytics.com/integrations/executorch/)
- [ML Runtimes Comparison — Aman AI Primers](https://aman.ai/primers/ai/ml-runtimes/)
- [Eight TorchScript Alternatives for PyTorch 2.x — Medium/Modexa](https://medium.com/@Modexa/eight-torchscript-alternatives-for-the-pytorch-2-x-era-34dcb68f2940)
- [ExecuTorch XNNPACK Backend — ExecuTorch 1.1](https://docs.pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)
- [Core ML Backend — ExecuTorch 0.7](https://docs.pytorch.org/executorch/0.7/backends-coreml.html)
