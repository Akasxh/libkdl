# OpenXLA / XLA: Multi-Device Compilation, StableHLO, and PJRT

**Compiled:** 2026-04-06
**Relevance Score:** 9/10 — PJRT is the closest production analog to libkdl's dispatch goal; StableHLO is the portable IR layer our system must be aware of
**Connection to our work:** PJRT defines the interface pattern we are implementing at the MLIR/compiler layer; XLA's SPMD partitioner shows the state of the art for multi-device placement; our contribution fills the "runtime selection without framework recompilation" gap below the framework layer

---

## 1. OpenXLA Project Overview (2024–2026 State)

OpenXLA is Google's open-source umbrella for ML compiler infrastructure, founded March 2023. Primary components:

- **XLA** — the core ML compiler (HLO → target code)
- **StableHLO** — versioned, serializable MLIR dialect serving as portable IR between frameworks and compilers
- **Shardy** — MLIR-based partitioning system (successor to GSPMD, 2024–2025)
- **IREE** — full end-to-end MLIR compiler/runtime (see `iree-deep-dive.md`)
- **PJRT** — uniform device runtime API (plugin model)

Backed by AMD, Arm, Apple, AWS, Google, Intel, NVIDIA, Samsung. Target: every major hardware platform from TPU to Apple Silicon to AWS Trainium.

**Source:** [OpenXLA Project](https://openxla.org/), [Google OSS Blog 2024](https://opensource.googleblog.com/2024/12/a-robust-open-ecosystem-accelerating-ai-infrastructure.html)

---

## 2. StableHLO as the Portable IR Layer

### 2.1 What StableHLO Is

StableHLO is a versioned MLIR dialect providing a **stable contract** between ML frameworks and ML compilers. It is based on MHLO (MLIR HLO) but adds:

- **Backward-compatibility guarantee** — MLIR bytecode serialization enables safe cross-version communication
- **Versioned opset** — dialect version pinned per release; older serialized programs remain loadable
- **Forward compatibility window** — 6 months minimum (frameworks → compiler version gap)
- **Comprehensive coverage** — supports dynamism, quantization (via `stablehlo.uniform_quantize`), and sparsity annotations
- **MHLO migration target** — XLA's internal MHLO dialect is being migrated to StableHLO

**Entry paths into StableHLO:**

| Framework | Mechanism |
|-----------|-----------|
| JAX | `jax.export` → StableHLO bytecode |
| TensorFlow | `tf2xla` → XLA HLO → StableHLO |
| PyTorch | `torch.export` → torch-mlir FxImporter → StableHLO |
| PyTorch/XLA | `torch_xla.stablehlo.exported_program_to_stablehlo` |

**Source:** [StableHLO Spec](https://openxla.org/stablehlo/spec), [StableHLO GitHub](https://github.com/openxla/stablehlo), [PyTorch StableHLO Export Tutorial](https://openxla.org/stablehlo/tutorials/pytorch-export)

### 2.2 Multi-Target Role

StableHLO is **target-agnostic by design**. The same StableHLO bytecode can be:
- Compiled by XLA for NVIDIA GPU (NVPTX path), AMD GPU (ROCm path), TPU, CPU
- Compiled by IREE for Vulkan, Metal, CUDA, ROCm, CPU
- Distributed across devices via SPMD partitioning annotations before lowering

This makes StableHLO the **dispatch-neutral interchange format** our poster identifies as missing. It represents a computation without committing to a device — device selection happens at the compiler (PJRT plugin) or runtime layer.

**Gap for our work:** StableHLO is designed for AOT portability. Using StableHLO as an interchange for deferred, runtime-dispatched kernel selection is unexplored territory (see our `papers-ml-compilation.md`, research gap #4).

---

## 3. XLA Multi-Device Compilation

### 3.1 XLA:GPU Architecture

XLA's GPU compilation pipeline (as of 2025 XLA:GPU architecture docs):

```
StableHLO input
  ↓
XLA HLO (after StableHLO → HLO lowering)
  ↓
Target-independent optimizations
  ├─ Algebraic simplification
  ├─ Operation fusion (horizontal + vertical)
  ├─ CSE, DCE, buffer analysis
  └─ Sharding annotation injection (for SPMD)
  ↓
Target-specific backend
  ├─ NVIDIA: NVPTX → PTX → cubin (via LLVM)
  ├─ AMD:    AMDGPU → AMDGCN → hsaco (via LLVM/ROCm)
  ├─ TPU:    custom backend
  └─ CPU:    LLVM → native object
```

**Key property:** Each compilation is to a **single target**. XLA does not produce fat binaries or dispatch across GPU vendors in a single artifact. Vendor selection is determined at framework/PJRT layer, not within the XLA compiler itself.

**Source:** [XLA:GPU Architecture](https://openxla.org/xla/gpu_architecture)

### 3.2 Device Placement: GSPMD and Shardy

For multi-device (multi-GPU, multi-node) execution, XLA uses **SPMD (Single Program, Multiple Data)** partitioning:

**GSPMD (General and Scalable Parallelization for ML):**
- Takes HLO with user-provided sharding annotations (`jax.pjit`)
- Produces a sharded HLO program executable by N identical devices
- Optimizes execution schedule, overlaps computation with inter-device communication
- Key paper: [GSPMD (arXiv 2105.04663)](https://arxiv.org/abs/2105.04663)

**Shardy (2024–2025, successor to GSPMD):**
- MLIR-based, operates on StableHLO dialect directly
- Axis-based sharding representation (mesh axes)
- Includes: sharding annotation propagation, SPMD partitioner, compiler APIs
- Built from collaboration of GSPMD and PartIR teams
- Enables finer-grained sharding control than GSPMD
- Status: available in JAX via `jax.sharding`

**Source:** [Shardy GitHub](https://github.com/openxla/shardy), [Shardy JAX Guide](https://openxla.org/shardy/getting_started_jax), [OpenXLA Dev Lab 2024](https://opensource.googleblog.com/2024/05/openxla-dev-lab-2024-building-grouundbreaking-systems-together.html)

**Critical distinction for our poster:** GSPMD/Shardy partitions across **homogeneous devices** (N identical GPUs). They do not solve **heterogeneous dispatch** (selecting between NVIDIA GPU vs AMD GPU vs CPU at runtime). Homogeneous SPMD is a solved, production-deployed problem. Heterogeneous runtime selection is the gap we address.

### 3.3 Does XLA Do Runtime Dispatch Across GPU Vendors?

**No.** XLA selects a backend at compile time. PJRT (see Section 4) provides runtime backend selection at the framework layer, but within a PJRT plugin invocation, XLA compiles for exactly one hardware target.

There is no mechanism in XLA to compile a StableHLO program into a single artifact that selects between NVIDIA and AMD GPU kernels at deployment time.

---

## 4. PJRT — The Plugin-Based Device Abstraction

### 4.1 Architecture

PJRT (Pretty much Just a Runtime) defines a **uniform device API** — a stable C/C++ interface that hardware vendors implement as plugins, consumed by ML frameworks without framework-level modification.

**The long-term vision (from PJRT docs):**
```
ML Framework (JAX, TF, PyTorch/XLA)
  ↓  calls PJRT C API
PJRT Plugin (hardware-specific, opaque to framework)
  ├─ Compiles StableHLO using XLA or vendor compiler
  ├─ Manages device memory
  ├─ Executes compiled programs
  └─ Reports device capabilities
```

**Key design principles:**
- C stable API — ABI-stable, framework can be a different binary than plugin
- Plugin is a shared library loaded at runtime via `PJRT_PLUGIN_PATH` environment variable
- Frameworks need zero modification to support new hardware; vendors implement the plugin

**Source:** [PJRT Uniform Device API](https://openxla.org/xla/pjrt), [PJRT C++ API Overview](https://openxla.org/xla/pjrt/cpp_api_overview)

### 4.2 C API Interface

```c
// Key PJRT C API functions (stable ABI)
PJRT_Api_Version PJRT_Api_Version_Major/Minor;
PJRT_Client_Create         // Initialize device connection
PJRT_Client_Devices        // Enumerate available devices
PJRT_Program_Create        // Wrap StableHLO/HLO program
PJRT_Client_Compile        // Compile program for device
PJRT_LoadedExecutable_Execute  // Run compiled program
PJRT_Buffer_ToHostBuffer   // Device → host transfer
```

Both C API (direct) and C++ API (with C→C++ wrapper) are supported.

**Source:** [PJRT Integration Guide](https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md), [PJRT C README](https://github.com/openxla/xla/blob/main/xla/pjrt/c/README.md)

### 4.3 Active Vendor Plugin Ecosystem (2024–2025)

| Hardware | Plugin | Notes |
|----------|--------|-------|
| NVIDIA GPU (CUDA) | `jax-cuda12-plugin` | Extracted from JAX; installable from PyPI separately |
| Google TPU | Built-in JAX `jax-tpu` | Original PJRT target |
| Apple Silicon + AMD GPU (macOS) | Apple Metal PJRT plugin | Enables JAX on M-series chips and AMD GPUs via Metal |
| Intel GPU (Data Center Max, Flex, Arc) | Intel XPU PJRT plugin | Compiles StableHLO → dispatches to Intel GPU via oneAPI |
| AWS Trainium/Inferentia | JAX Neuron plugin | Latest addition; enables JAX natively on AWS accelerators |
| AMD GPU (ROCm, Linux) | AMD ROCm PJRT plugin | Via OpenXLA |

**Source:** [PJRT Plugin Blog 2024](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html), [Intel JAX on Intel GPU via PJRT](https://opensource.googleblog.com/2023/06/accelerate-jax-models-on-intel-gpus-via-pjrt.html)

### 4.4 Does PJRT Enable Runtime Dispatch Across Vendors?

**Partially, at the framework level.** A JAX user can switch between NVIDIA PJRT plugin and AMD PJRT plugin by changing environment variables or plugin registration, without changing their JAX code. However:

1. **One plugin active at a time per device type** — the framework selects one PJRT plugin per logical device. There is no single PJRT plugin that internally dispatches between NVIDIA and AMD kernel variants based on runtime-detected hardware.

2. **No sub-plugin dispatch** — PJRT doesn't define an API for "try NVIDIA first, fall back to AMD." Plugin selection is resolved before compilation.

3. **No multi-vendor fat binary** — a PJRT `LoadedExecutable` is compiled for exactly one device type; it cannot be executed on a different hardware vendor.

**This is the gap libkdl fills:** PJRT provides vendor-agnostic framework integration but requires advance knowledge of target hardware. libkdl provides runtime selection of pre-compiled vendor-specific kernel variants *within a single deployed artifact*, below the PJRT layer.

---

## 5. XLA Heterogeneous GPU Support — Current State

### 5.1 AMD ROCm Path Maturity

From competitive landscape research and OpenXLA sources:
- AMD GPU support via ROCm is **production quality as of 2024** (noted in `competitive-landscape.md` XLA section)
- LLVM AMDGPU backend generates AMDGCN code
- AMD and NVIDIA paths are separate compilation backends, not interoperable at runtime

### 5.2 XLA AutoTuning (Persisted Autotuning)

XLA supports AOT autotuning via XProf/persisted autotuning, where kernel tile configurations are profiled and cached per hardware target. This is a form of compile-time specialization — the autotuned binary is hardware-specific. No equivalent to MetaSchedule's hardware-agnostic tuning database.

### 5.3 Dynamic Shape Handling

XLA prefers static shapes (traced, JIT-compiled programs). Dynamic shapes require **dynamism handling passes** and are less optimized than static-shape programs. This is a known limitation for transformer inference with variable sequence lengths, where TorchDynamo + TorchInductor (Triton backend) competes more favorably.

---

## 6. Relationship to Our Poster Contribution

### 6.1 What XLA/PJRT Establishes

- **PJRT** proves that a stable C API for hardware plugins is viable and widely adopted (5+ major vendors)
- **StableHLO** provides the portable IR layer our system needs for framework → compiler communication
- **GSPMD/Shardy** shows that compiler-driven multi-device placement is solved for homogeneous cases

### 6.2 Gaps PJRT Does Not Fill (Our Contribution Space)

| Gap | PJRT Status | Our Approach |
|-----|------------|--------------|
| Runtime vendor selection without framework recompilation | Plugin selection is pre-dispatch; no single-artifact multi-vendor binary | libkdl fat binary with runtime ELF header query |
| Kernel-level dispatch (not graph-level) | PJRT dispatches full programs | libkdl dispatches individual kernel variants |
| MLIR IR-level dispatch (below framework) | PJRT operates above compilation | libkdl operates in the loaded-module dispatch layer |
| CPU/GPU fallback within one binary | Not supported; separate PJRT plugins | Single `.kdl` artifact with CPU fallback variant |
| Zero-dependency deployment | Requires framework + PJRT plugin installation | libkdl is a standalone shared library |

### 6.3 Poster Framing

PJRT should be cited as "the production state of the art for framework-level device abstraction" — it shows the industry direction toward hardware-agnostic APIs. Our contribution is complementary: we push this dispatch capability down to the kernel-library layer, enabling vendor-agnostic execution **without requiring the full XLA/JAX/PJRT stack**.

---

## 7. Key Citations

1. OpenXLA Project. "PJRT — Uniform Device API." https://openxla.org/xla/pjrt
2. OpenXLA Project. "PJRT C++ API Overview." https://openxla.org/xla/pjrt/cpp_api_overview
3. OpenXLA Project. "PJRT Plugin Integration." https://openxla.org/xla/pjrt/pjrt_integration
4. Google Open Source Blog. "PJRT Plugin to Accelerate Machine Learning." March 2024. https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html
5. Google Open Source Blog. "PJRT: Simplifying ML Hardware and Framework Integration." May 2023. https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html
6. Google Open Source Blog. "A Robust Open Ecosystem for All: Accelerating AI Infrastructure." December 2024. https://opensource.googleblog.com/2024/12/a-robust-open-ecosystem-accelerating-ai-infrastructure.html
7. OpenXLA Project. "XLA:GPU Architecture Overview." https://openxla.org/xla/gpu_architecture
8. OpenXLA Project. "StableHLO Specification." https://openxla.org/stablehlo/spec
9. OpenXLA Project. "Shardy: MLIR-based partitioning system." https://github.com/openxla/shardy
10. OpenXLA Project. "Shardy Guide for JAX Users." https://openxla.org/shardy/getting_started_jax
11. Lepikhin, Dmitry et al. "GSPMD: General and Scalable Parallelization for ML Computation Graphs." arXiv:2105.04663. https://arxiv.org/abs/2105.04663
12. Google Open Source. "Accelerate JAX models on Intel GPUs via PJRT." June 2023. https://opensource.googleblog.com/2023/06/accelerate-jax-models-on-intel-gpus-via-pjrt.html
13. OpenXLA. "PJRT C API Integration Guide." https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md
14. OpenXLA. "PyTorch Export to StableHLO Tutorial." https://openxla.org/stablehlo/tutorials/pytorch-export
