# ExecuTorch: Edge AI with Runtime Backend Delegation
## Literature Note — LLVM Dublin 2026

**System:** ExecuTorch 1.0 (Meta / PyTorch ecosystem)
**Release:** ExecuTorch 1.0 announced October 2025; 1.2 current as of early 2026
**Docs:** https://executorch.ai / https://docs.pytorch.org/executorch/stable/
**Code:** https://github.com/pytorch/executorch
**Blog:** https://pytorch.org/blog/introducing-executorch-1-0/
**Relevance Score:** 8/10

---

## Finding

ExecuTorch implements the most production-complete example of **compile-time graph partitioning with runtime backend delegation** in the ML deployment ecosystem. It handles unknown or heterogeneous hardware through a two-layer mechanism: (1) a partitioner selects which subgraphs go to which backend at export time, and (2) a graceful CPU fallback handles any operator the selected backend cannot execute at runtime. This is the closest existing production system to libkdl's dispatch model at the ML framework level.

---

## Architecture

### Export pipeline (compile time)

```
PyTorch model (eager)
        ↓  torch.export()
    ExportedProgram (ATen dialect)
        ↓  to_backend(partitioner)
    Edge Dialect (lowered ops + delegated subgraphs)
        ↓  ExecutorchBackendConfig + emit
    .pte file (binary, portable execution format)
```

The `.pte` file embeds:
- Operator graph in ExecuTorch's serialization format
- Pre-compiled backend blobs (one per delegated subgraph)
- Metadata for runtime dispatch routing

### Partitioner (the dispatch decision)

The partitioner implements the hardware-specific dispatch logic at compile time:

```python
# Example: Vulkan partitioner for GPU subgraphs
partitioner = VulkanPartitioner(
    config=VulkanPartitionerConfig(
        require_dynamic_shapes=False,
    )
)
program = to_backend(exported_program, [partitioner])
```

The partitioner:
1. Traverses the exported graph
2. Tags each node with `backend_id` if the backend supports the operator
3. Groups tagged nodes into delegated subgraphs
4. Calls `backend.preprocess(subgraph, compile_specs)` → compiled binary blob
5. Embeds the blob in the `.pte`

**Multiple partitioners can be chained** in priority order. For example: `[CoreMLPartitioner, XNNPackPartitioner]` — CoreML gets GPU ops on Apple Silicon, XNNPACK handles CPU fallback for remainder.

### Runtime execution

At runtime, `ExecuTorch Runtime` loads the `.pte` and processes the graph node-by-node:

- ATen op node → dispatch to portable CPU kernel (via `kernel_registry`)
- `call_delegate` node → invoke backend's `execute(context, inputs, outputs)` method

The backend's `execute()` call is entirely opaque to the runtime — the backend owns the compiled blob and its own execution path.

### Handling unknown hardware

This is the key mechanism for deployment to uncharacterized edge devices:

1. **Pre-delegation (compile time):** The developer exports with the most specific backend(s) available in their test environment
2. **Unknown backend at deployment:** If the expected backend is unavailable (e.g., CoreML on a non-Apple device, QNN on a non-Qualcomm device), the backend registration check fails
3. **Fallback:** Undelegated ops (any op not covered by a successful backend) fall through to **portable CPU kernels** — always present, no hardware assumptions
4. **Partial acceleration:** If only some backends fail, the graph runs partially accelerated; only missing-backend subgraphs fall to CPU

The runtime does NOT perform dynamic hardware detection to re-route GPU ops — it relies on compile-time decisions. This is the critical limitation relative to libkdl.

---

## Backend Ecosystem (as of ExecuTorch 1.0–1.2)

| Backend | Hardware | Mechanism |
|---------|----------|-----------|
| XNNPACK | ARM NEON / x86 SSE (CPU) | Compile-time graph lowering |
| CoreML | Apple Neural Engine (ANE), Apple GPU, CPU | Apple's CoreML framework |
| MPS (Metal) | Apple GPU | Metal Performance Shaders |
| QNN (Qualcomm) | Qualcomm Hexagon NPU | Qualcomm AI Engine Direct SDK |
| Vulkan | Any Vulkan-conformant GPU | SPIR-V via Vulkan compute |
| OpenVINO | Intel CPU / integrated GPU / NPU | Intel OpenVINO backend |
| VGF | Arm 2026 GPU (new in ExecuTorch 1.0) | Arm Neural Technology via ExecuTorch |
| TOSA | Arm Cortex-A / Cortex-M | TOSA ML operator spec |
| MediaPipe | Mobile CPU / GPU | Google MediaPipe Tasks |

Runtime footprint: **50 KB minimum** (portable kernels only). Full deployment with one backend delegate: ~2–5 MB.

---

## Compile-Spec System

Backends accept compile-time configuration via `CompileSpec` (key-value pairs). These encode:
- Target hardware parameters (memory budget, precision constraints)
- Quantization settings (int8, fp16 preferences)
- Dynamic shape support flags

This is analogous to libkdl's dispatch descriptor specifying capability requirements, but resolved at compile time rather than runtime.

---

## What ExecuTorch Does NOT Do

ExecuTorch does **not** perform:

1. **Runtime hardware detection to select backends:** The backend selection is frozen at `.pte` export time. A `.pte` file compiled for QNN will not automatically use CoreML on a different device.
2. **Multi-target kernel selection:** There is no mechanism to embed multiple compiled backend blobs for the same subgraph and select one at runtime based on detected hardware.
3. **Cross-vendor GPU dispatch from a single binary:** A `.pte` file targeting Vulkan will use Vulkan everywhere, not switch to CUDA on NVIDIA or HIP on AMD.

**This is exactly the gap libkdl fills for the GPU compute layer.** ExecuTorch solves the problem at the framework/model level by compiling one backend per deployment; libkdl would solve it at the kernel level by carrying multiple compiled variants and selecting at runtime.

---

## Relevance to libkdl

### Direct connections

1. **Backend delegation API pattern:** ExecuTorch's `to_backend(partitioner)` + `preprocess()` + `execute()` lifecycle is a clean blueprint for how a kernel dispatch system should separate compile-time preparation from runtime invocation. libkdl's `kdl_dispatch_t` struct and dispatch table follow the same principle.

2. **Fallback chain:** ExecuTorch's multi-partitioner priority chain (GPU backend → CPU XNNPACK → portable) is the model-level analog of libkdl's fallback: if no GPU variant matches detected hardware, fall back to CPU. The architecture validates the approach.

3. **Portable kernel baseline:** ExecuTorch ships a fully portable CPU kernel set as a zero-assumption fallback. libkdl's CPU kernels (via OpenMP or plain C) play the same role. Both guarantee correct execution regardless of detected hardware.

4. **Compile-time overhead, runtime zero cost:** For the common case (known hardware at export), ExecuTorch's delegation adds zero runtime overhead — the compiled blob is directly executed. libkdl's dispatch table lookup is O(1) for the common case (exact capability match).

### Key difference from libkdl

ExecuTorch targets **mobile/edge deployment** (single device, known hardware class). libkdl targets **heterogeneous clusters** (unknown GPU vendor at deployment, must adapt at process startup). ExecuTorch solves the portability problem by compiling multiple `.pte` variants (one per deployment target); libkdl solves it with a single binary embedding multiple compiled kernel variants.

The two systems are complementary: ExecuTorch could use libkdl as its Vulkan/GPU backend to achieve true single-binary heterogeneous GPU dispatch.

---

## Performance Data

- **XNNPACK delegate:** 2–4x speedup vs portable CPU kernels on ARM Cortex-A (Apple M1-class: up to 10x)
- **QNN delegate on Snapdragon:** 5–8x speedup vs CPU for vision models
- **50 KB runtime binary:** enables MCU-class deployment (Cortex-M55 with Ethos-U NPU)
- **12+ backends** as of ExecuTorch 1.0 (October 2025)

---

## Risks / Gaps

- ExecuTorch 1.0 was released October 2025, cutting it close to the poster submission deadline (April 2026); cite the stable release version.
- The multi-partitioner chaining is documented but underspecified for conflict resolution (what happens if two backends tag the same node).
- No public benchmark comparing cross-vendor deployment scenarios (the gap libkdl addresses).
- Vulkan backend is present but less optimized than vendor-native paths — the same performance portability problem libkdl aims to solve exists within ExecuTorch.

---

## Notes for Poster

- Cite in "Related Work: Production Edge Inference" and "Motivation: Why Static Backend Selection Fails"
- The ExecuTorch architecture diagram (partitioner → .pte → runtime delegate chain) is a useful visual reference for explaining libkdl's analogous design
- Key contrast: ExecuTorch requires re-export per target device class; libkdl requires no re-compilation for new GPU vendors
- Quote: ExecuTorch 1.0 supports "12+ hardware backends" — use this to illustrate the fragmentation problem our work addresses
