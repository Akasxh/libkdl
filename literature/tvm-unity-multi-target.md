# Apache TVM / TVM Unity: Multi-Target Compilation and Runtime Dispatch

**Compiled:** 2026-04-06
**Relevance Score:** 7/10 — TVM provides the most direct prior art on multi-target ML kernel compilation; MLC-LLM demonstrates runtime device dispatch in practice; MetaSchedule auto-tuning is the reference point for specialization overhead
**Connection to our work:** TVM's Relax VM heterogeneous execution RFC defines the problem we solve at a lower level; TVM's lack of MLIR integration is a key differentiator for our MLIR-native approach; OctoAI/NVIDIA acquisition affects project trajectory

---

## 1. TVM Architecture Overview (2025 State)

Apache TVM is a general-purpose ML compiler with a two-level IR, active since 2017 (OSDI 2018 paper). Key architectural evolution:

| Generation | IR Layer | Status |
|------------|----------|--------|
| TVM Classic | Relay (graph IR) + TIR (tensor IR) | Maintenance mode |
| TVM Unity / Relax | Relax (graph IR with control flow + dynamic shapes) + TIR | Active development |

**Current version:** Apache TVM v0.19.0 (released 2025-01-28), v0.24 release scheduled 2025.

**Project governance:** Apache Software Foundation Top-Level Project. Community-maintained committer model. TVM Unity is the strategic direction.

**Source:** [TVM Apache](https://tvm.apache.org/), [TVM Monthly May 2025](https://discuss.tvm.apache.org/t/tvm-monthly-may-2025/18362)

---

## 2. Relax VM and Dynamic Shape Support

### 2.1 What Relax Introduces

Relax (Relax is a Language for eXpressive ML computation) is TVM Unity's graph-level IR. Compared to Relay, Relax adds:

- **First-class dynamic shapes** — shape values are symbolic variables tracked through the IR; shape functions compute output shapes at runtime
- **Dataflow regions** — explicit separation of pure dataflow (optimizable) from effectful code
- **Composable abstractions** — IR, VM, and runtime are designed to be individually replaceable; no monolithic compilation pipeline
- **Control flow** — proper if/then/else, loops expressible in the IR (not just unrolled graphs)

**Paper:** "Relax: Composable Abstractions for End-to-End Dynamic Machine Learning." arXiv:2311.02103. [PDF](https://arxiv.org/pdf/2311.02103)

### 2.2 Dynamic Shape Execution Model

When a Relax program has dynamic shapes:
1. Symbolic shape variables are propagated through the IR during compilation
2. Shape functions are compiled as separate TIR functions that compute output shapes from input shapes at runtime
3. The Relax VM executes shape functions before kernel dispatch to allocate output buffers of the correct size
4. Kernels may be compiled for a single symbolic shape or for multiple specialized variants with a dispatch table

**Key difference from XLA:** XLA prefers static traced shapes; Relax is designed for dynamic shapes as a first-class concern.

**Source:** [Relax paper](https://arxiv.org/pdf/2311.02103), [TVM arch docs](https://tvm.apache.org/docs/arch/index.html)

---

## 3. Multi-Target Compilation in TVM Unity

### 3.1 Target Specification

TVM compiles `tvm.IRModule` → per-target artifacts. A target is specified as a `tvm.target.Target` object:

```python
# CPU target
target_cpu = tvm.target.Target("llvm -mtriple=x86_64-linux-gnu -mcpu=core-avx2")

# NVIDIA GPU target
target_gpu = tvm.target.Target("cuda -arch=sm_80")

# AMD GPU target
target_amd = tvm.target.Target("rocm -mcpu=gfx906")
```

Each target generates a separate compiled module. For heterogeneous execution, multiple targets are compiled together and a dispatcher routes ops to the appropriate target.

### 3.2 Compilation Pipeline (TVM Unity)

```
PyTorch / ONNX / TensorFlow model
  ↓  (import via torch.fx, onnx, etc.)
Relax IRModule
  ↓  Graph-level optimizations
     ├─ Constant folding, DCE
     ├─ Operator fusion
     └─ Layout optimization (memory access pattern)
  ↓  TIR lowering
     ├─ Library dispatch (cuBLAS, cuDNN, oneDNN calls)
     └─ TensorIR schedule optimization
  ↓  MetaSchedule / DLight auto-tuning (optional)
  ↓  Target-specific codegen
     ├─ LLVM IRBuilder (x86, ARM, RISC-V)
     ├─ CUDA C generation → nvcc
     └─ OpenCL/Vulkan/Metal for other GPU targets
  ↓  Compiled module (.so / .tar)
```

**Source:** [TVM Design and Architecture](https://tvm.apache.org/docs/arch/index.html)

### 3.3 Portability Scope

TVM supports the broadest target range of any ML compiler:

| Target | Backend | Notes |
|--------|---------|-------|
| NVIDIA GPU | CUDA → nvcc | Primary GPU target; best performance |
| AMD GPU | ROCm / OpenCL | HIP backend less mature than CUDA |
| Intel GPU | OpenCL | Via Intel OpenCL runtime |
| x86 CPU | LLVM | AVX2/AVX-512, MKL dispatch |
| ARM CPU | LLVM | NEON, SVE, Cortex-M |
| Apple Silicon | Metal (via MLC-LLM) | Mobile/edge focus |
| WebGPU | Vulkan / WebGPU | Experimental |
| Qualcomm | BYOC (Bring Your Own Codegen) | Via QNN external codegen |

---

## 4. Does TVM Do Runtime Device Selection?

### 4.1 Standard TVM: No

In standard TVM compilation, the target is specified at compile time and the compiled artifact runs on exactly one device type. There is no mechanism analogous to ORT's EP priority list for runtime backend selection within a compiled TVM module.

### 4.2 Heterogeneous Execution in Relax: RFC Status (2023–2025)

An RFC titled "Heterogeneous Execution for Relax" was filed in mid-2023 (GitHub Issue #15101, TVM Discuss thread). This is the most detailed plan in TVM for runtime multi-device execution.

**What the RFC proposes:**
- `VDevice` — a virtual device descriptor that can be resolved to a physical device at compile time or runtime
- `hint_on_device`, `to_vdevice` — ops that annotate tensors with intended device placement
- `to_device` builtin — explicit cross-device tensor copy
- `UpdateVDevice` pass — compiler pass to propagate device placement through the IR
- `RealizeVDevice` pass — lower virtual device annotations to concrete device assignments

**What was implemented:**
- `VDevice` data structure added ✓
- TVMScript parser/printer support for VDevice ✓
- `hint_on_device` / `to_vdevice` / `to_device` ops ✓
- `UpdateVDevice` and `RealizeVDevice` passes ✓

**What remains incomplete (as of 2025):**
- Full end-to-end heterogeneous execution with automatic placement
- Cost-model-driven device assignment
- Runtime device selection between alternative devices of the same type (e.g., NVIDIA vs AMD GPU)

**Source:** [RFC Heterogeneous Relax](https://discuss.tvm.apache.org/t/rfc-unity-relax-heterogeneous-execution-for-relax/14670), [Tracking Issue #15101](https://github.com/apache/tvm/issues/15101)

### 4.3 PackedFunc Runtime Dispatch

TVM's runtime does support **multi-device deployment** via its PackedFunc abstraction:
- A compiled module exports functions as `PackedFunc` callables
- The caller specifies which device to execute on at call time
- Multiple compiled modules (one per target) can be loaded simultaneously
- The caller manually selects which module to invoke per input

This is **user-managed dispatch** — it requires application code to select the appropriate module. TVM provides no automatic hardware discovery or fallback policy.

---

## 5. MetaSchedule Auto-Tuning as a Form of Specialization

### 5.1 What MetaSchedule Does

MetaSchedule (NeurIPS 2022) is TVM's current auto-tuner. For a given operator (e.g., a matrix multiply), it:
1. Defines a parameterized schedule space (tile sizes, vectorization, parallelism, tensor core usage)
2. Searches the space using evolutionary search + a learned cost model
3. Selects the best schedule for the **specific target hardware**

The tuned schedule is recorded and used in subsequent compilations via a `database` (JSON file or SQLite).

### 5.2 Specialization Implications

MetaSchedule produces **hardware-specific specialized kernels** — a schedule optimized for an NVIDIA A100 is not optimal for a GTX 1650 or an AMD RX 6800 XT. This is analogous to libkdl's concept of per-hardware kernel variants, but at the auto-tuning level rather than the runtime dispatch level.

**Key limitation:** MetaSchedule tuning is restricted to **static shapes**. DLight (tuning-free mode) provides reasonable performance for dynamic shapes but cannot match tuned performance.

**Source:** [MetaSchedule paper](https://openreview.net/forum?id=nyCr6-0hinG), [existing `papers-ml-compilation.md` P6]

### 5.3 DLight: Tuning-Free Specialization

DLight provides heuristic-based schedule generation without measurement:
- Uses device properties (compute capability, memory bandwidth) to select tile sizes
- No search required; near-zero compile overhead
- Trades ~20-40% performance for zero tuning time
- Primary use case: LLM inference with dynamic shapes

---

## 6. MLC-LLM: TVM Unity's Production Runtime Dispatch Story

### 6.1 What MLC-LLM Is

MLC-LLM (Machine Learning Compilation for Large Language Models) is TVM Unity's primary production deployment system for LLMs. It compiles LLMs using TVM and deploys them across diverse hardware without backend-specific changes.

**Supported platforms via single codebase:**
- NVIDIA GPU (CUDA)
- AMD GPU (ROCm, Vulkan)
- Apple Silicon (Metal)
- Intel GPU (Vulkan)
- Android (Vulkan, OpenCL)
- iOS (Metal)
- WebGPU (in-browser)
- CPU (x86, ARM)

### 6.2 How MLC-LLM Achieves Cross-Platform Dispatch

MLC-LLM's portability comes from **compile-time target selection**, not runtime dispatch:

1. User runs `mlc_llm compile model.mlc --target cuda` (or `metal`, `vulkan`, etc.)
2. TVM Unity compiles model to target-specific `.so` / `.dylib`
3. Deployment uses the pre-compiled target artifact

**Runtime dispatch within a single artifact** is not the mechanism — separate artifacts are compiled per target. The "cross-platform" story is that the same Python compilation script produces correct results on all platforms.

**However,** for Vulkan targets, MLC-LLM achieves **cross-vendor execution** at runtime: a SPIR-V compiled artifact runs on NVIDIA, AMD, Intel, and mobile GPU Vulkan drivers without recompilation. This is the nearest thing to runtime vendor dispatch in the TVM ecosystem, but it comes with the Vulkan performance trade-off (see `competitive-landscape.md`, Section 2.3).

**Source:** [MLC-LLM GitHub](https://github.com/mlc-ai/mlc-llm), [MLC-LLM intro docs](https://llm.mlc.ai/docs/get_started/introduction)

---

## 7. OctoAI / NVIDIA Acquisition — Project Trajectory

### 7.1 Acquisition Details

NVIDIA acquired OctoAI (formerly OctoML, the TVM commercialization company) on **September 30, 2024** for ~$250M. OctoML/OctoAI was co-founded by Tianqi Chen (TVM creator), Luis Ceze, and others from University of Washington. OctoAI wound down its commercial cloud services on October 31, 2024; key team members joined NVIDIA.

**Source:** [OctoAI acquired by NVIDIA](https://www.hpcwire.com/bigdatawire/2024/09/30/octoai-snapped-up-by-nvidia/), [GeekWire](https://www.geekwire.com/2024/chip-giant-nvidia-acquires-octoai-a-seattle-startup-that-helps-companies-run-ai-models/)

### 7.2 Impact on Apache TVM

Apache TVM is an independent Apache Software Foundation project — it does not belong to OctoML/OctoAI. The acquisition did not change ownership of TVM's codebase. However:

- The commercial entity that was driving TVM production adoption is now NVIDIA-owned
- NVIDIA has no incentive to invest in AMD/Intel GPU backends for TVM
- Community activity remains active but the original core committer team has bifurcated
- Tianqi Chen (CMU) continues maintaining TVM through MLC-LLM and the `mlc-ai` org

**Current project health:**
- v0.19.0 released Jan 2025; v0.24 release scheduled
- Active focus areas: LLM workloads (KVCache, MLA/DeepSeek), FP8/BF16/F4 data types, Torch importer improvements
- Broader heterogeneous execution roadmap deprioritized in favor of LLM-specific features

**Risk assessment for our poster:** Citing TVM as the state-of-the-art multi-target ML compiler requires noting the community fragmentation risk. The project is active but its strategic future is uncertain given NVIDIA's interest.

---

## 8. MLIR Integration Status

From our existing `competitive-landscape.md` research (confirmed by 2025 search):

TVM **does not use MLIR as its core IR**. TVM predates MLIR and has its own IR stack (Relax + TIR). An experimental MLIR ingestion path exists, but:
- TVM's optimization pipeline operates on Relax/TIR, not MLIR dialects
- There is no MLIR lowering path in production TVM
- The project has discussed MLIR integration but it is not implemented

**This is a key differentiator for our poster:** A libkdl-style dispatch system built on MLIR is fundamentally different from TVM's approach. MLIR enables us to leverage the entire LLVM ecosystem (NVVM, ROCDL, SPIR-V dialects, LLVM codegen) without maintaining a parallel compiler stack.

---

## 9. Comparison: TVM vs Our Approach

| Dimension | TVM Unity / Relax | Our libkdl Approach |
|-----------|-------------------|---------------------|
| IR foundation | Relax + TIR (custom) | MLIR dialects |
| Dynamic shapes | First-class (symbolic) | Specialization at load time |
| Multi-target | Compile-time selection | Runtime dispatch from single binary |
| Runtime dispatch | Manual (PackedFunc) or Vulkan | Automatic via ELF section header + capability query |
| Auto-tuning | MetaSchedule (static), DLight (dynamic) | Pre-compiled variants (no search at deployment) |
| NVIDIA support | CUDA (primary) | NVVM → PTX |
| AMD support | ROCm (secondary, less maintained post-acquisition) | ROCDL → AMDGCN |
| CPU fallback | Yes (LLVM backend) | Yes (LLVM CPU codegen) |
| Deployment artifact | Target-specific `.so` per compile | Single `.kdl` fat binary |
| Framework integration | PyTorch, ONNX, TF frontends | Library-level (framework agnostic) |

---

## 10. Key Citations

1. Chen, Tianqi et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018. https://arxiv.org/abs/1802.04799
2. Shao, Junru et al. "Relax: Composable Abstractions for End-to-End Dynamic Machine Learning." arXiv:2311.02103. https://arxiv.org/pdf/2311.02103
3. Shao, Junru et al. "MetaSchedule: Unified Machine Learning-Based Tensor Program Optimization." NeurIPS 2022. https://openreview.net/forum?id=nyCr6-0hinG
4. Apache TVM. "Design and Architecture." https://tvm.apache.org/docs/arch/index.html
5. Apache TVM. "[RFC][Unity][Relax] Heterogeneous Execution for Relax." TVM Discuss 2023. https://discuss.tvm.apache.org/t/rfc-unity-relax-heterogeneous-execution-for-relax/14670
6. Apache TVM. "[Unity] Tracking Issue: Heterogeneous execution for Relax. #15101." GitHub. https://github.com/apache/tvm/issues/15101
7. MLC-LLM. "Introduction to MLC LLM." https://llm.mlc.ai/docs/get_started/introduction
8. MLC-LLM GitHub. https://github.com/mlc-ai/mlc-llm
9. Apache TVM. "TVM Monthly May 2025." https://discuss.tvm.apache.org/t/tvm-monthly-may-2025/18362
10. BigDATAwire. "OctoAI Snapped Up by Nvidia." September 2024. https://www.hpcwire.com/bigdatawire/2024/09/30/octoai-snapped-up-by-nvidia/
11. GeekWire. "Chip giant Nvidia acquires OctoAI." https://www.geekwire.com/2024/chip-giant-nvidia-acquires-octoai-a-seattle-startup-that-helps-companies-run-ai-models/
