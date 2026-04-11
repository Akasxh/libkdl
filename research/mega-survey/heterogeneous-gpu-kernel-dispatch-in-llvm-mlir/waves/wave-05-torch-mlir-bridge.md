# Wave 05 — Torch-MLIR Bridge
**Angle:** torch-mlir-bridge
**Query:** "torch-mlir PyTorch MLIR bridge backend lowering linalg tensor"
**Date:** 2026-04-06

---

## Executive Summary

Torch-MLIR is a translation layer (not an end-to-end compiler) that bridges PyTorch's IR into MLIR dialects, where downstream compilers (IREE, XLA, custom backends) take over. For libkdl, torch-mlir's Linalg-on-Tensors lowering is the natural **entry point**: libkdl would consume Linalg IR after torch-mlir normalization and produce a vendor-fat `.kdl` binary with runtime dispatch across NVIDIA, AMD, and CPU. Torch-MLIR explicitly does not solve multi-vendor runtime dispatch — that gap is exactly libkdl's contribution.

---

## Sources

1. [torch-mlir GitHub — llvm/torch-mlir](https://github.com/llvm/torch-mlir)
2. [torch-mlir architecture.md](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md)
3. [torch-mlir roadmap.md](https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md)
4. [DeepWiki: llvm/torch-mlir pipeline analysis](https://deepwiki.com/llvm/torch-mlir)
5. [FOSDEM 2025 — An Introduction to Torch-MLIR (Marius Brehler)](https://archive.fosdem.org/2025/schedule/event/fosdem-2025-6643-an-introduction-to-torch-mlir/)
6. [FOSDEM 2025 slides PDF](https://archive.fosdem.org/2025/events/attachments/fosdem-2025-6643-an-introduction-to-torch-mlir/slides/237934/An_Introd_nnOMKYo.pdf)
7. [Torch-MLIR — Bridging PyTorch and MLIR ecosystems (PyTorch Forums)](https://discuss.pytorch.org/t/torch-mlir-bridging-pytorch-and-mlir-ecosystems/133151)
8. [IREE PyTorch Integration Guide](https://iree.dev/guides/ml-frameworks/pytorch/)
9. [RFC: Integrate torch-mlir into IREE](https://groups.google.com/g/iree-discuss/c/geG1O1E4820)
10. [Towards a high-performance AI compiler with upstream MLIR (arXiv 2404.15204)](https://arxiv.org/html/2404.15204v1)
11. [StableHLO migration issue #1835](https://github.com/llvm/torch-mlir/issues/1835)
12. [Local literature: /home/akash/PROJECTS/LLVM/literature/torch-mlir-bridge.md]

---

## Key Findings

### 1. Pipeline Architecture (Three-Phase)

Torch-MLIR's compilation is divided into three sequential phases:

```
Phase 1: Frontend Import
  torch.export / torch.fx   →  FxImporter  →  Torch dialect (modern, recommended)
  torch.jit.script/trace    →  TorchScript →  Torch dialect (legacy)
  ONNX model                →  OnnxImporter→  Torch dialect (cross-framework)

Phase 2: Torch Dialect Normalization (Backend Contract)
  DecomposeComplexOps    — aten.matmul → aten.mm / aten.bmm, etc.
  MaximizeValueSemantics — !torch.tensor → !torch.vtensor (no aliasing)
  RefineTypes            — shape + dtype propagation via ~1000 shape functions
  LowerToBackendContract — validate: value-semantics only, known ranks/dtypes

Phase 3: Backend Lowering (three pipelines, independently applicable)
  torch-backend-to-linalg-on-tensors-backend-pipeline  → linalg + arith + tensor
  torch-backend-to-stablehlo-backend-pipeline           → stablehlo
  torch-backend-to-tosa-backend-pipeline                → tosa
```

The **backend contract** is the hard boundary: it guarantees value semantics, completed shape inference, and decomposed ops before any dialect lowering. It is the place where PyTorch semantics end and structural MLIR ops begin.

### 2. Import Path Details

**FxImporter (primary path for PyTorch 2.x)**
- Entry: `torch_mlir.fx.export_and_import(exported_program)`
- Preserves symbolic shape expressions from `torch.export` as MLIR symbolic dimensions
- Supports dynamic sequence lengths — critical for transformer inference
- Can handle mutations, aliasing, and complex control flow via torch.export's normalized form

**TorchScript (legacy)**
- Entry: `torch_mlir.compile(jit_module, ...)`
- Weaker dynamic-shape support; TorchScript itself in maintenance mode upstream
- Still tested but not the recommended path for new work

**OnnxImporter**
- Entry: `torch_mlir.extras.onnx_importer`
- Converts ONNX → Torch dialect → same lowering pipeline
- Enables frameworks (TensorFlow, Keras) to feed torch-mlir without PyTorch

### 3. Multi-Backend Lowering (Three Targets)

| Backend | Dialect | Primary Consumers | Dispatch Capability |
|---------|---------|-------------------|---------------------|
| Linalg-on-Tensors | `linalg` + `arith` + `tensor` | IREE, custom backends | Most complete; structured ops enable tiling/vectorization |
| TOSA | `tosa` | Edge inference, TFLite tooling | Hardware-operator-level; inference-optimized |
| StableHLO | `stablehlo` | XLA, IREE (via StableHLO path), PyTorch/XLA | XLA/JAX ecosystem |

Coverage asymmetry: Linalg is the most complete backend. TOSA and StableHLO have op coverage gaps for uncommon PyTorch operations (e.g., topk, complex reductions). Each backend is implemented independently, leading to some divergence — a known pain point in the roadmap.

### 4. Integration with IREE (iree-turbine)

The production deployment path is `iree-turbine` (formerly SharkTurbine):

```
torch.compile(model, backend="turbine_cpu")
  or
aot.export(model, args)
  ↓
TorchDynamo → FxImporter → Torch dialect
  ↓
Linalg lowering (torch-mlir)
  ↓
IREE compiler: Flow → Stream → HAL → target codegen
  ↓
IREE CUDA / ROCm / Vulkan / CPU target
  ↓
Compiled .vmfb (target-specific artifact)
```

Key limitation: each `.vmfb` is compiled for a single target. IREE's variant mechanism exists but **runtime selection between NVIDIA and AMD** (in the same process, on the same model) is not fully productionized in iree-turbine as of 2026. The `TurbineDevice` is a native PyTorch device backed by IREE's HAL — a production-quality runtime interface, but single-vendor per session.

### 5. Where Heterogeneous Dispatch Is Missing

Torch-MLIR's architecture.md states explicitly: "Torch-MLIR does not attempt to provide a production end-to-end flow." Multi-vendor runtime dispatch is out of scope for torch-mlir itself. It provides:
- A normalized Linalg IR for any downstream to compile
- No kernel caching, no runtime dispatch tables, no fat-binary format

This is the exact gap libkdl fills.

---

## Angle Assessment

**Relevance to libkdl:** 9/10

Torch-MLIR is the standard PyTorch → MLIR translation layer. If libkdl targets PyTorch-ecosystem models (the largest ML workload category), torch-mlir is the mandatory upstream. The Linalg-on-Tensors output is the natural libkdl input: structured ops with explicit iteration spaces, target-agnostic, directly lowerable to NVVM/ROCDL/LLVM.

**Novelty:** 5/10

Torch-MLIR itself is well-documented and not novel. The novelty is in the **gap it exposes**: torch-mlir terminates at dialect-level IR and leaves multi-vendor runtime dispatch completely unsolved. No existing project (torch-mlir, iree-turbine, OpenXLA) produces a single compiled artifact that selects NVIDIA vs. AMD vs. CPU at runtime based on hardware availability. That is libkdl's contribution.

---

## libkdl Integration Point

The proposed integration slot:

```
PyTorch model (torch.export)
  ↓  torch-mlir FxImporter
Torch dialect → Linalg-on-Tensors dialect    ← libkdl entry point
  ↓  libkdl MLIR-EP compilation pipeline
  ├─ NVIDIA path:  Linalg → gpu.launch → NVVM → PTX → cubin
  ├─ AMD path:     Linalg → gpu.launch → ROCDL → AMDGCN → hsaco
  └─ CPU path:     Linalg → SCF/vector → LLVM → native object
Single .kdl fat binary with runtime dispatch table
  ↓  kdl_dispatch() selects variant at load time
```

Key properties enabled by entering at Linalg:
1. **Structured iteration spaces** — each `linalg.generic` encodes reduction/parallel/window dims, directly enabling tiling and vectorization per target
2. **Symbolic shapes from FxImporter** — enable JIT specialization for dynamic sequence lengths; libkdl can cache compiled variants keyed on shape constraints
3. **Target-agnostic IR** — no vendor ops until libkdl's lowering passes; same source IR drives all variant compilations
4. **Reference implementation** — IREE's Linalg-to-GPU lowering is public and battle-tested; libkdl can reuse/adapt it

Entering at StableHLO is also viable (broader ecosystem: JAX, TF, PyTorch/XLA all produce it) but requires additional lowering passes (StableHLO → Linalg) before codegen, adding pipeline complexity.

---

## Risks and Issues

1. **Op coverage gaps** — Linalg lowering for uncommon ops (complex reductions, sparse ops, some attention variants) is incomplete. Models relying on these ops will fail at torch-mlir before reaching libkdl. Mitigation: define a supported-ops surface and document fallback to eager PyTorch.

2. **Mutation overhead** — In-place ops (scatter, index_put) require copy-materialization during TorchConversion. This inflates memory usage vs. native PyTorch. Not a correctness issue but a performance regression for mutation-heavy models.

3. **TorchScript deprecation trajectory** — The legacy import path is maintenance-mode. Any tooling built on TorchScript import will have a shortened lifespan. All new work should use FxImporter.

4. **nod.ai/AMD team continuity** — The AMD team (former nod.ai) co-maintained torch-mlir; post-acquisition priorities may shift torch-mlir contribution levels. Google (IREE team) is the primary maintainer today.

5. **iree-turbine GPU parity lag** — The `torch.compile` backend in iree-turbine started CPU-only; GPU support was added later. Documentation lags implementation, causing confusion about actual GPU capability. Verify with direct testing rather than documentation.

---

## Related Angles for Follow-up

- `iree-hal-runtime-dispatch` — downstream of torch-mlir; how IREE's HAL handles multi-device dispatch after compilation
- `mlir-gpu-dialect` — the `gpu.launch` ops that torch-mlir's Linalg output targets via IREE
- `tvm-unity-multi-target` — TVM's analogous compilation pipeline; comparable Linalg-equivalent (TE/TIR) input
- `stablehlo-portability` — the alternative entry point for JAX/TF ecosystem models
- `multi-versioned-kernels` — the fat-binary format question that torch-mlir sidesteps but libkdl must answer
