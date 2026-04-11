# Direction 06: Torch-MLIR to libkdl — The PyTorch Ecosystem Entry Point

**Composite Score: 7.5/10**
**Rank: 6 of 8**

---

## Title

**From torch.export to Cross-Vendor Dispatch: libkdl as the Missing Runtime Layer Below Torch-MLIR**

## One-Sentence Description

Torch-MLIR's Linalg-on-Tensors output is the natural entry point for libkdl's multi-vendor compilation pipeline — torch-mlir produces target-agnostic structured IR; libkdl lowers it to per-vendor native variants and selects at runtime.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| Torch-MLIR architecture.md | wave-05-torch-mlir-bridge | "Torch-MLIR does not attempt to provide a production end-to-end flow" — explicitly out of scope |
| Torch-MLIR Linalg lowering | wave-05-torch-mlir-bridge | Three-phase pipeline terminates at Linalg-on-Tensors; no vendor codegen, no runtime dispatch |
| iree-turbine single-target VMFB | wave-05-torch-mlir-bridge | Each .vmfb is single-target; no runtime cross-vendor selection in same process |
| FxImporter symbolic shapes | wave-05-torch-mlir-bridge | Dynamic sequence lengths from torch.export preserved as MLIR symbolic dims — enables shape-keyed variant caching |
| IREE Linalg-to-GPU lowering | wave-05-torch-mlir-bridge | Public, battle-tested lowering passes reusable by libkdl |
| PT2 843s cold start | wave-04-kernel-caching | torch.compile → Triton JIT is the current path; libkdl AOT eliminates this |
| MLIR gpu.binary multi-target | wave-01-mlir-gpu-dialect-dispatch | gpu.binary can carry NVVM+ROCDL+XeVM objects simultaneously |

---

## Novelty Argument

The PyTorch ecosystem currently has two paths to GPU execution:
1. **torch.compile → Triton → vendor-specific kernel** (843s cold start, single vendor)
2. **torch-mlir → IREE → single-target VMFB** (no cross-vendor runtime selection)

Neither produces a single artifact that dispatches across NVIDIA, AMD, and CPU at runtime. The proposed path:

```
torch.export → FxImporter → Torch dialect → Linalg-on-Tensors
  → libkdl MLIR pipeline:
    ├─ NVVM → PTX → CUBIN
    ├─ ROCDL → AMDGCN → HSACO  
    └─ LLVM → native CPU object
  → single .kdl bundle
  → kdl_dispatch() selects at runtime
```

This is architecturally novel: torch-mlir provides the portable IR; libkdl provides the multi-vendor codegen and runtime dispatch.

---

## Feasibility Plan

1. **Demonstrate Linalg consumption**: Take a simple model (MLP, attention layer), export via torch-mlir to Linalg, show the IR
2. **Dual-target compilation**: Lower same Linalg to NVVM (CUDA) and LLVM (CPU) using existing MLIR passes; package both in MTB
3. **Runtime dispatch**: Load MTB via libkdl, detect hardware, dispatch to appropriate variant
4. **Comparison**: Same model via torch.compile (Triton path) vs. libkdl path — measure cold start and steady-state throughput

**For poster deadline**: Steps 1-3 with MLP on GTX 1650 + CPU. Step 4 as comparison number.

---

## Poster Potential

**Moderate — strong ecosystem story but requires MLIR pipeline work.**

- Pipeline diagram: torch.export → torch-mlir → libkdl → multi-vendor dispatch
- Cold start comparison: PT2/Triton (843s) vs. libkdl (<5ms)
- Fills the "how does this connect to PyTorch?" question reviewers will ask

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **7/10** | Novel combination (torch-mlir + multi-vendor AOT + runtime dispatch) but each component exists separately. |
| **Feasibility** | **7/10** | Requires MLIR pipeline setup; Linalg → NVVM/LLVM lowering exists but integration effort non-trivial. |
| **Evidence** | **8/10** | Torch-MLIR docs explicitly state runtime dispatch is out of scope. PT2 843s cold start quantifies the alternative. |
| **Impact** | **8/10** | PyTorch is the dominant ML framework; connecting libkdl to it via torch-mlir maximizes adoption potential. |
| **Composite** | **7.5/10** | |
