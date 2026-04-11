# Torch-MLIR: Bridging PyTorch to MLIR Backends

**Compiled:** 2026-04-06
**Relevance Score:** 8/10 — Torch-MLIR is the standard gateway from PyTorch to IREE/XLA/custom MLIR backends; its FxImporter and multi-backend lowering architecture is directly relevant to our system's input surface
**Connection to our work:** Any libkdl integration with PyTorch would flow through torch-mlir's Torch dialect or its FxImporter path; iree-turbine demonstrates the production deployment pattern we are targeting; the multi-backend lowering chain (Torch → Linalg/TOSA/StableHLO → vendor target) is the architecture template for our MLIR-EP sketch

---

## 1. Project Overview and Status

Torch-MLIR is an LLVM project providing first-class MLIR compiler support for PyTorch. It is not an end-to-end compiler — it is a **translation layer** from PyTorch's IR (FX graph, TorchScript) into MLIR dialects.

**Organizational home:** LLVM monorepo subproject (`llvm/torch-mlir`)
**Primary maintainers:** Google (IREE team), AMD (nod.ai team, now AMD after acquisition)
**FOSDEM 2025 presentation:** "An Introduction to Torch-MLIR" — confirms active project with recent improvements

**Source:** [torch-mlir GitHub](https://github.com/llvm/torch-mlir), [FOSDEM 2025 slides](https://archive.fosdem.org/2025/events/attachments/fosdem-2025-6643-an-introduction-to-torch-mlir/slides/237934/An_Introd_nnOMKYo.pdf)

---

## 2. Architecture: How Torch-MLIR Bridges PyTorch to MLIR

### 2.1 Two Import Paths

**Path 1: FxImporter (modern, recommended)**
```
torch.export(model, args)  →  ExportedProgram (torch.fx graph)
  ↓
torch_mlir.fx.export_and_import()
  ↓
Torch MLIR dialect (mlir::torch::TorchDialect)
```

- Handles `torch.export` models with dynamic shapes, mutations, and symbolic shape expressions
- Symbolic shapes from PyTorch's shape analysis are preserved as MLIR symbolic dimensions
- Supports complex models including transformers, vision models, and graph neural networks
- Required for `torch.export`-based workflows (the recommended PyTorch 2.x export path)

**Path 2: TorchScript importer (legacy)**
```
torch.jit.script(model)  →  TorchScript module
  ↓
torch_mlir.compile()
  ↓
Torch MLIR dialect
```

- Older path, still supported but TorchScript itself is in maintenance mode
- Less capable with dynamic shapes than FxImporter

**Source:** [torch-mlir architecture.md](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md), [torch-mlir roadmap.md](https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md)

### 2.2 The Torch Dialect

The `torch` MLIR dialect is torch-mlir's canonical intermediate representation:

- Models PyTorch's type system: `!torch.tensor`, `!torch.vtensor`, `!torch.list`, `!torch.dict`
- Encodes PyTorch semantics faithfully (including mutation, aliasing)
- `TorchConversion` dialect handles the type boundary when lowering to strict-value dialects
- Both mutable (`!torch.tensor`, value semantics via copy-on-write) and immutable (`!torch.vtensor`) tensor types

The Torch dialect is the lingua franca — all import paths target it, all export paths lower from it.

### 2.3 Backend Contract

Before lowering to target dialects, torch-mlir runs a **backend contract** normalization:
- All dynamic dispatch resolved
- Mutations lowered to explicit copy operations
- Shape inference completed (where static)
- Output: `func.func` with `!torch.vtensor` types (value semantics, no aliasing)

This backend-contract IR is the input to backend-specific lowering passes.

---

## 3. Multi-Target Compilation via Backend Lowering Passes

### 3.1 Three Supported Backend Dialects

Torch-MLIR provides three production lowering pipelines from the backend-contract Torch IR:

| Backend | MLIR Dialect | Primary Users | Notes |
|---------|-------------|---------------|-------|
| **Linalg-on-Tensors** | `linalg` + `arith` + `tensor` | IREE, custom backends | Most general; structured ops |
| **TOSA** | `tosa` | Edge inference, TFLite-compatible tools | Operator-level target |
| **StableHLO** | `stablehlo` | XLA, IREE (via StableHLO path) | XLA ecosystem compatibility |

Each pipeline is a sequence of MLIR passes that rewrites Torch ops into target-dialect ops. The three backends are **not mutually exclusive** — a project can use different backends for different parts of a model.

**Source:** [torch-mlir architecture.md](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md)

### 3.2 Lowering to StableHLO (2025 State)

The StableHLO lowering path is actively maintained and used for PyTorch/XLA integration:

```python
from torch_mlir.extras.fx_importer import FxImporter
import torch_mlir

# torch.export → StableHLO
exported = torch.export.export(model, example_inputs)
mlir_module = torch_mlir.compile(exported, example_inputs,
                                  output_type="stablehlo")
```

PyTorch/XLA also exposes this directly:
```python
from torch_xla.stablehlo import exported_program_to_stablehlo
stablehlo_program = exported_program_to_stablehlo(exported_program)
```

**Active issues (2025):** Some ops have incomplete StableHLO lowering (e.g., `topk` lowering gaps reported October 2025 in Issue #4337). Coverage is high for standard transformer ops but sporadic for uncommon ops.

**Source:** [PyTorch StableHLO export docs](https://docs.pytorch.org/xla/master/features/stablehlo.html), [torch-mlir Issue #4337](https://github.com/llvm/torch-mlir/issues/4337)

### 3.3 Lowering to Linalg (for IREE)

The Linalg lowering is torch-mlir's primary path for IREE compilation:

```
Torch dialect (backend contract)
  ↓ torch-backend-to-linalg-on-tensors-backend-pipeline
linalg + arith + tensor + memref ops
  ↓  (handed to IREE or custom codegen)
```

The Linalg representation is "structured" — each operation has explicit iteration spaces (reduction, parallel, window dimensions) that downstream compilers can exploit for tiling, vectorization, and parallelization.

---

## 4. Connection to IREE: iree-turbine

### 4.1 iree-turbine Architecture

`iree-turbine` (formerly SharkTurbine) is IREE's official PyTorch frontend. It uses torch-mlir internally and provides:

- `torch.compile` backend — integrates into PyTorch's compilation infrastructure via TorchDynamo
- AOT export API — ahead-of-time compilation to IREE `.vmfb` deployment artifacts
- `TurbineDevice` — native PyTorch device backed by IREE's HAL

**Compilation flow:**

```
torch.compile(model, backend="turbine_cpu")  or  AOT export
  ↓  TorchDynamo captures FX graph
torch-mlir FxImporter
  ↓  Torch dialect
torch-mlir Linalg lowering pipeline
  ↓  Linalg IR
IREE compiler (iree-compile)
  ↓  Flow → Stream → HAL lowering
  ↓  Target codegen (CUDA / ROCm / Vulkan / CPU)
Compiled VMFB module
  ↓  IREE runtime execution
```

**Source:** [iree-turbine GitHub](https://github.com/iree-org/iree-turbine), [IREE PyTorch guide](https://iree.dev/guides/ml-frameworks/pytorch/), [iree-turbine walkthrough](https://jysh1214.github.io/pytorch/2024/10/08/A-Walkthrough-Example-of-torch.compile-with-IREE-Turbine.html)

### 4.2 Multi-Target Support in iree-turbine

IREE supports multiple hardware targets; iree-turbine inherits this. Compilation target is specified at `iree-compile` time:

```python
import iree.turbine.aot as aot

# Compile for NVIDIA GPU
exported = aot.export(model, args)
exported.save_mlir("model.mlir")
# Then: iree-compile model.mlir --iree-hal-target-device=cuda -o model.vmfb
```

The compiled `.vmfb` is target-specific. For multi-target deployment, IREE's variant mechanism applies (see `iree-deep-dive.md`, Section 4) — but runtime selection between NVIDIA and AMD GPU variants remains incomplete in IREE's stack.

### 4.3 Note on "Initial CPU-only" Documentation

Some iree-turbine documentation states "initial integration only supports CPU." This reflects the torch.compile backend specifically (which started CPU-only for stability). The AOT export path and direct IREE compilation support all IREE targets (CUDA, ROCm, Vulkan, CPU). The documentation lag is a project communication issue, not a technical limitation.

---

## 5. Does Torch-MLIR Support Multi-Target Compilation?

**Yes, via the multi-backend lowering pipelines.** Torch-MLIR produces three dialects (Linalg, TOSA, StableHLO), each of which can be further compiled to multiple hardware targets by downstream compilers (IREE, XLA, custom backends).

**No, not in a single artifact.** Torch-MLIR itself does not produce a single compiled artifact that selects between NVIDIA and AMD GPU variants at runtime. It is a translation layer, not a runtime. The multi-target dispatch problem falls to the downstream compiler (IREE's variant model, PJRT plugins, etc.).

---

## 6. Implications for Our Poster

### 6.1 Torch-MLIR as Our Input Path

If libkdl targets integration with PyTorch-ecosystem models, torch-mlir provides the input surface:

```
PyTorch model (torch.export)
  ↓  torch-mlir FxImporter
Torch dialect → Linalg dialect
  ↓  libkdl's MLIR-EP compilation pipeline
  ├─ NVIDIA variant: Linalg → NVVM → PTX → cubin
  ├─ AMD variant:   Linalg → ROCDL → AMDGCN → hsaco
  └─ CPU variant:   Linalg → LLVM → native object
Single .kdl fat binary with runtime dispatch
```

This positions libkdl as a **downstream consumer of torch-mlir's Linalg IR**, fitting naturally into the existing compilation toolchain without requiring framework modifications.

### 6.2 Linalg Is the Right Entry Point

The Linalg-on-Tensors representation is the optimal entry point for our compilation pipeline because:
- Structured iteration spaces enable target-specific tiling and vectorization
- Target-agnostic — no vendor-specific operations
- Directly lowerable to NVVM (GPU) or LLVM (CPU) via standard MLIR passes
- Already integrated into the IREE codegen pipeline (reference implementation available)

Entering at StableHLO is also viable (broader framework compatibility) but requires more lowering passes before reaching codegen-ready IR.

### 6.3 FxImporter Handles Dynamic Shapes

For models with dynamic sequence lengths (the primary transformer inference challenge), FxImporter's symbolic shape preservation means our compilation pipeline receives shape constraints as MLIR symbolic dimensions. This enables:
- JIT specialization: compile specialized kernels for observed shapes
- Caching: reuse compiled variants for repeated shapes
- Fallback: CPU kernel for shapes that exceed GPU variant coverage

---

## 7. Known Gaps and Issues (2025)

1. **StableHLO coverage gaps** — some PyTorch ops do not have complete StableHLO lowering (topk, some reduction variants, complex number ops). Linalg coverage is broader.

2. **Mutation handling overhead** — the TorchConversion passes that handle in-place ops (scatter, index_put) can introduce extra copies when lowering to value-semantic dialects. This adds latency not present in the original PyTorch eager execution.

3. **No upstream multi-vendor dispatch** — torch-mlir's architecture.md explicitly states it "does not attempt to provide a production end-to-end flow." The multi-target story requires downstream systems (IREE, XLA, libkdl).

4. **AMD nod.ai team absorption** — nod.ai (the AMD team that was the second major torch-mlir contributor alongside Google) was absorbed into AMD after AMD's acquisition of nod.ai in 2022. Team continuity has been maintained but priorities shift with AMD's roadmap.

---

## 8. Key Citations

1. LLVM/torch-mlir project. GitHub. https://github.com/llvm/torch-mlir
2. Wang, Chi et al. (LLVM/AMD). "An Introduction to Torch-MLIR." FOSDEM 2025. https://archive.fosdem.org/2025/events/attachments/fosdem-2025-6643-an-introduction-to-torch-mlir/slides/237934/An_Introd_nnOMKYo.pdf
3. torch-mlir. "Architecture Overview." https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
4. torch-mlir. "Roadmap." https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md
5. iree-org. "iree-turbine: IREE's PyTorch Frontend." https://github.com/iree-org/iree-turbine
6. IREE. "PyTorch Integration Guide." https://iree.dev/guides/ml-frameworks/pytorch/
7. PyTorch/XLA. "Torch Export to StableHLO." https://docs.pytorch.org/xla/master/features/stablehlo.html
8. OpenXLA. "PyTorch Export to StableHLO Tutorial." https://openxla.org/stablehlo/tutorials/pytorch-export
9. torch-mlir. "Issue #4337: topk export to stablehlo error." https://github.com/llvm/torch-mlir/issues/4337
10. IREE. "IREE/MLIR/Linalg Tutorial." https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/
11. Brehler, Marius. LLVM Discourse: "Is there a way to externalize weights in torch-mlir output?" https://discourse.llvm.org/t/is-there-a-way-to-externalize-weights-in-torch-mlir-output/89842
12. PyPI. `iree-turbine` package. https://pypi.org/project/iree-turbine/
13. Kuderski, Jakub (AMD). "The Long Tail of AI: SPIR-V in IREE and MLIR." Vulkanised 2025. https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf
