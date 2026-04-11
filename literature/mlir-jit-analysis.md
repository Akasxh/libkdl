# MLIR JIT Compilation and Heterogeneous GPU Dispatch: Technical Analysis

*Research compiled 2026-04-02 for LLVM Dublin 2026 poster on vendor-agnostic GPU kernel dispatch.*

---

## Table of Contents

1. [MLIR JIT Compilation Pipeline](#1-mlir-jit-compilation-pipeline)
2. [GPU-Relevant MLIR Dialects](#2-gpu-relevant-mlir-dialects)
3. [The Complete Lowering Pipeline](#3-the-complete-lowering-pipeline)
4. [Multi-Target Compilation](#4-multi-target-compilation)
5. [Host vs Device Code Generation](#5-host-vs-device-code-generation)
6. [GPU Dialect Operations in Detail](#6-gpu-dialect-operations-in-detail)
7. [IREE: The Closest Existing Runtime Solution](#7-iree-the-closest-existing-runtime-solution)
8. [Triton: Multi-Backend via MLIR](#8-triton-multi-backend-via-mlir)
9. [Relevant RFCs and Discourse Discussions](#9-relevant-rfcs-and-discourse-discussions)
10. [What MLIR Can and Cannot Do Today](#10-what-mlir-can-and-cannot-do-today)
11. [Implications for Our Poster](#11-implications-for-our-poster)

---

## 1. MLIR JIT Compilation Pipeline

### 1.1 ExecutionEngine Architecture

MLIR's `mlir::ExecutionEngine` is a utility wrapper around LLVM's ORC JIT (specifically LLJIT) that accepts MLIR IR as input. It assumes the IR can be converted to LLVM IR before JIT compilation.

**Internal architecture:**
- ExecutionEngine wraps LLVM's `LLJIT` class, which uses an `IRCompileLayer` and `RTDyldObjectLinkingLayer`
- LLJIT performs eager compilation: a symbol's definition is compiled as soon as its address is looked up
- The translation path: MLIR LLVM Dialect -> LLVM IR (via `translateModuleToLLVMIR()`) -> machine code (via ORC)

**Key API:**
- `ExecutionEngine::create(Operation *op, ExecutionEngineOptions &options)` -- static factory, eagerly JIT-compiles the module
- `engine->invoke("funcName", args...)` -- calls a JIT-compiled function with typed arguments
- `engine->invokePacked("funcName", packedArgs)` -- calls using opaque void pointers
- `engine->lookup("funcName")` -- returns raw function pointer

**ExecutionEngineOptions fields:**
| Field | Type | Description |
|-------|------|-------------|
| `llvmModuleBuilder` | `function_ref<unique_ptr<Module>(Operation*, LLVMContext&)>` | Custom MLIR-to-LLVM-IR translation |
| `transformer` | `function_ref<Error(Module*)>` | LLVM IR optimization/transformation callback |
| `jitCodeGenOptLevel` | `optional<CodeGenOptLevel>` | Target code generation optimization level |
| `sharedLibPaths` | `ArrayRef<StringRef>` | Shared libraries to open and link for symbol resolution |
| `sectionMemoryMapper` | `SectionMemoryManager::MemoryMapper*` | Custom memory mapper |
| `enableObjectDump` | `bool` (default: false) | Store generated object files |
| `enableGDBNotificationListener` | `bool` (default: true) | Notify GDB of JIT events |
| `enablePerfNotificationListener` | `bool` (default: true) | Notify perf of JIT events |

**Sources:**
- [MLIR ExecutionEngine class reference](https://mlir.llvm.org/doxygen/classmlir_1_1ExecutionEngine.html)
- [ExecutionEngineOptions struct reference](https://mlir.llvm.org/doxygen/structmlir_1_1ExecutionEngineOptions.html)
- [MLIR Toy Tutorial Ch. 6 -- Lowering to LLVM](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)

### 1.2 ORC JIT (On-Request Compilation)

ORC is LLVM's modular JIT compilation framework. Key properties:

- **Layered architecture**: Each layer wraps a part of the compiler pipeline (e.g., `IRTransformLayer` for custom passes, `IRCompileLayer` for LLVM IR to object code)
- **Lazy compilation**: Unlike MCJIT, ORC supports true lazy compilation where functions compile only when first called (though MLIR's ExecutionEngine uses LLJIT which is eager by default)
- **JITDylib**: Symbol tables that act as dynamic libraries within the JIT
- **ExecutionSession**: Manages string pool, error reporting, synchronization, and symbol lookup

The MLIR ExecutionEngine creates wrapper functions for each exported function with a fixed interface: the only argument is interpreted as a list of pointers to actual arguments, followed by a pointer to the result.

**Sources:**
- [ORC Design and Implementation](https://llvm.org/docs/ORCv2.html)
- [Building a JIT: ORC Layers Tutorial](https://llvm.org/docs/tutorial/BuildingAJIT2.html)

### 1.3 GPU JIT via ExecutionEngine

The ExecutionEngine's relationship with GPU code is indirect. For GPU execution:

1. GPU kernel code in `gpu.module` is compiled to a binary blob (cubin, hsaco, or SPIR-V) during the `gpu-module-to-binary` pass
2. The binary is embedded in the host LLVM IR as a global string
3. Host-side GPU operations (`gpu.launch_func`) are lowered to CUDA/HIP/Vulkan runtime API calls
4. The ExecutionEngine JIT-compiles the *host* code, which at runtime loads and launches the pre-compiled GPU binary

**Critical limitation**: The ExecutionEngine itself does not JIT-compile GPU device code at runtime. Device code must be compiled ahead of time into the binary blob. The only "JIT" happening is on the host side.

**Sources:**
- [MLIR GPU Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR GPU execution without runtime load/unload (Discourse)](https://discourse.llvm.org/t/mlir-gpu-execution-without-runtime-load-unload/61712)

---

## 2. GPU-Relevant MLIR Dialects

### 2.1 Dialect Taxonomy

| Dialect | Level | Vendor | Purpose |
|---------|-------|--------|---------|
| `linalg` | High | Agnostic | Named operations (matmul, conv), tiling, fusion |
| `tensor` | High | Agnostic | Tensor-level operations (immutable SSA tensors) |
| `vector` | Mid | Agnostic | Fixed-size vector operations, vectorization target |
| `affine` | Mid | Agnostic | Polyhedral loop representation, affine analysis |
| `scf` | Mid | Agnostic | Structured control flow (for, while, if) |
| `memref` | Mid | Agnostic | Memory reference types and operations |
| **`gpu`** | Mid | **Agnostic** | GPU execution model: launch, alloc, barrier, modules |
| **`spirv`** | Low | **Agnostic** | SPIR-V representation (Vulkan/OpenCL), ~500 ops |
| **`nvvm`** | Low | NVIDIA | NVVM IR intrinsics for NVIDIA GPUs |
| **`nvgpu`** | Mid | NVIDIA | Higher-level NVIDIA ops (tensor cores, TMA, wgmma) |
| **`rocdl`** | Low | AMD | ROCm Device Library intrinsics |
| **`amdgpu`** | Mid | AMD | Higher-level AMD ops (MFMA, buffer ops, barriers) |
| **`xegpu`** | Mid | Intel | Intel Xe GPU operations (2D block load, MMA) |
| `llvm` | Low | Agnostic | LLVM IR representation in MLIR |
| `arith` | Low | Agnostic | Arithmetic operations |
| `func` | Low | Agnostic | Function definitions and calls |
| `cf` | Low | Agnostic | Unstructured control flow (branch, switch) |

### 2.2 gpu Dialect (Target-Agnostic Layer)

The gpu dialect is the critical abstraction layer. It provides:

- **Kernel execution**: `gpu.launch`, `gpu.launch_func`
- **Memory management**: `gpu.alloc`, `gpu.dealloc`, `gpu.memcpy`, `gpu.memset`
- **Synchronization**: `gpu.barrier`, `gpu.wait`, async tokens
- **Module structure**: `gpu.module`, `gpu.func`, `gpu.binary`
- **Compilation**: Target attributes, `gpu-module-to-binary` pass
- **Reductions**: `gpu.all_reduce`, `gpu.subgroup_reduce`
- **Thread indexing**: `gpu.thread_id`, `gpu.block_id`, `gpu.block_dim`, `gpu.grid_dim`

**Source:** [gpu Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/)

### 2.3 spirv Dialect (Vendor-Agnostic Low-Level)

The SPIR-V dialect provides a separate lowering path that does NOT go through LLVM. Key properties:

- ~500 operations covering arithmetic, memory, control flow, atomics, group ops, images
- Supports both Vulkan (Shader capability) and OpenCL (Kernel capability) execution environments
- `spirv.target_env` attribute specifies version (v1.0--v1.6), capabilities, extensions, resource limits
- Full round-trip serialization via `spirv::serialize()` and `spirv::deserialize()`
- GPU dialect conversion: `gpu.module` -> `spirv.module`, `gpu.func` -> entry function
- Memref lowering avoids LLVM's descriptor model: directly maps to `!spirv.ptr<!spirv.array>`

**Limitations**: Non-32-bit scalars unconditionally converted to 32-bit. Vectors must be 2/3/4 elements. No implicit module-level SSA value references.

**Source:** [SPIR-V Dialect Documentation](https://mlir.llvm.org/docs/Dialects/SPIR-V/)

### 2.4 nvvm + nvgpu Dialects (NVIDIA)

- **nvvm**: Direct wrappers around NVVM/PTX intrinsics (thread IDs, barriers, memory operations, MMA)
- **nvgpu**: Higher-level bridge between `gpu`/`vector` and `nvvm`. Provides:
  - `nvgpu.warpgroup.mma` -- warpgroup-level (128 threads) matrix multiply-accumulate via wgmma
  - `nvgpu.warpgroup.generate.descriptor` -- descriptor generation for shared memory matrices
  - Tensor Memory Accelerator (TMA) operations
  - Async copy operations
  - Support for Hopper architecture features (sm_90+)

**Source:** [nvgpu Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/), [nvvm Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)

### 2.5 rocdl + amdgpu Dialects (AMD)

- **rocdl**: Direct wrappers around AMD GPU LLVM intrinsics (analogous to nvvm for NVIDIA)
- **amdgpu**: ~40+ higher-level operations including:
  - `amdgpu.mfma` -- Matrix Fused Multiply-Add (M/N: 4/16/32, K: 1-128)
  - `amdgpu.scaled_mfma` -- scaled fp4/fp6/fp8 operations
  - `amdgpu.sparse_mfma` -- 2:4 structured sparsity
  - `amdgpu.raw_buffer_load/store` -- buffer operations with bounds checking
  - Synchronization barriers (`ds_barrier_*`, `lds_barrier`, `sched_barrier`)
  - Lane operations (`dpp`, `permlane_swap`, `swizzle_bitmode`)
  - Uses `gfx` numbers for architecture identification (gfx90a, gfx942, gfx1250, etc.)

**Source:** [amdgpu Dialect](https://mlir.llvm.org/docs/Dialects/AMDGPU/), [rocdl Dialect](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/)

### 2.6 xegpu Dialect (Intel)

- Counterpart of nvgpu/amdgpu for Intel Xe GPUs
- Operations: 2D block load/store, DPAS (MMA equivalent), atomic, scattered load, named barrier
- Lowering path: XeGPU -> GEN dialect -> LLVM IR/bitcode -> SPIR-V binary
- **Status**: Upstream in MLIR but lacks a complete lowering pipeline as of 2025
- Wrapper libraries available for Level Zero and SYCL runtimes

**Source:** [xegpu Dialect](https://mlir.llvm.org/docs/Dialects/XeGPU/), [RFC: Add XeGPU dialect](https://discourse.llvm.org/t/rfc-add-xegpu-dialect-for-intel-gpus/75723)

---

## 3. The Complete Lowering Pipeline

### 3.1 General Pipeline: linalg -> ... -> GPU Backend

```
linalg.matmul / linalg.generic
        |
        v  (convert-linalg-to-loops / convert-linalg-to-affine-loops)
    affine.for / scf.for
        |
        v  (convert-affine-for-to-gpu / convert-parallel-loops-to-gpu)
      gpu.launch { gpu.terminator }
        |
        v  (gpu-kernel-outlining)
    gpu.launch_func @module::@kernel
        |
        +---> gpu.module @module { gpu.func @kernel { ... } }
        |           |
        |           +---> [NVIDIA path]  convert-gpu-to-nvvm -> nvvm-attach-target -> gpu-module-to-binary
        |           +---> [AMD path]     convert-gpu-to-rocdl -> rocdl-attach-target -> gpu-module-to-binary
        |           +---> [SPIR-V path]  convert-gpu-to-spirv -> gpu-module-to-binary
        |
        v  (gpu-to-llvm)
    Host-side LLVM dialect (runtime API calls to load/launch kernel binary)
        |
        v  (translate to LLVM IR)
    LLVM IR -> machine code (JIT via ExecutionEngine or AOT via llc)
```

### 3.2 Concrete NVIDIA Pipeline (gpu-lower-to-nvvm-pipeline)

This is the only fully consolidated default pipeline in upstream MLIR. The pass sequence:

```
1.  canonicalize
2.  one-shot-bufferize (tensors -> buffers)
3.  canonicalize
4.  convert-linalg-to-affine-loops (or convert-linalg-to-loops)
5.  affine-loop-invariant-code-motion
6.  convert-affine-for-to-gpu (map loops to GPU grid)
7.  gpu-kernel-outlining (extract device code)
8.  lower-affine
9.  gpu-decompose-memrefs
10. expand-strided-metadata
11. normalize-memrefs
12. gpu.module(convert-gpu-to-nvvm) (GPU ops -> NVVM intrinsics)
13. nvvm-attach-target{chip=sm_90 O=3} (attach target attributes)
14. convert-nvvm-to-llvm
15. reconcile-unrealized-casts
16. gpu-to-llvm (host-side GPU API calls)
17. gpu-module-to-binary (serialize device code to cubin/fatbin)
```

**Invocation:**
```bash
mlir-opt example.mlir -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
```

**Source:** [gpu Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/), [Stephen Diehl: GPU Compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)

### 3.3 AMD Pipeline (Manual Assembly Required)

**There is no `gpu-lower-to-rocdl-pipeline` equivalent.** Users must manually compose passes:

```
gpu-kernel-outlining
-> rocdl-attach-target{chip=gfx90a O=3}
-> gpu.module(convert-gpu-to-rocdl)
-> gpu-to-llvm
-> gpu-module-to-binary
```

This asymmetry between NVIDIA and AMD support is a significant gap. The `convert-gpu-to-rocdl` pass exists and the `rocdl.target` attribute works, but there is no single consolidated pipeline pass.

**Source:** [Discourse: How to Generate AMDGPU Code from MLIR?](https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627)

### 3.4 SPIR-V Pipeline

The SPIR-V path is fundamentally different -- it does NOT go through LLVM:

```
gpu.module { gpu.func { ... } }
    |
    v  (convert-gpu-to-spirv)
spirv.module { spirv.func { ... } }
    |
    v  (spirv serialization)
SPIR-V binary blob
    |
    v  (consumed by Vulkan/OpenCL driver at runtime)
Native GPU ISA (vendor driver responsibility)
```

In LLVM 19, a generic MLIR-to-SPIR-V pass was merged (contributed by AMD), providing better coverage of upstream compilation to SPIR-V.

**Source:** [SPIR-V Dialect](https://mlir.llvm.org/docs/Dialects/SPIR-V/), [Phoronix: Generic MLIR to SPIR-V Pass](https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V)

---

## 4. Multi-Target Compilation

### 4.1 gpu-module-to-binary: The Multi-Target Mechanism

This is the key pass for multi-target compilation. It:

1. Searches for all nested `gpu.module` operations
2. Reads attached target attributes (implementing GPU Target Attribute Interface)
3. Serializes the module for *each* target attribute
4. Produces a `gpu.binary` operation containing one object per target

**Example -- single module, multiple targets:**
```mlir
// Input: module with two NVIDIA targets and one AMD target
gpu.module @kernels <#gpu.select_object<1>> [
    #nvvm.target<chip = "sm_90">,
    #nvvm.target<chip = "sm_60">,
    #rocdl.target<chip = "gfx90a">
] {
  gpu.func @matmul(...) kernel { ... }
}

// Output after gpu-module-to-binary:
gpu.binary @kernels [
    #gpu.object<#nvvm.target<chip = "sm_90">, "sm_90 cubin blob">,
    #gpu.object<#nvvm.target<chip = "sm_60">, "sm_60 cubin blob">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, "gfx90a hsaco blob">
]
```

### 4.2 gpu.select_object: Runtime Object Selection

The `#gpu.select_object` offloading attribute selects which binary object to use at runtime:

```mlir
gpu.binary @myobject <#gpu.select_object<#rocdl.target>> [
    #gpu.object<#nvvm.target, "NVPTX binary">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, "AMDGPU binary">
]
```

**Critical observation**: `gpu.select_object` takes a *compile-time* index or target specification. It does NOT perform runtime hardware detection. The selection is baked into the compiled binary during LLVM IR translation.

### 4.3 Fat Binary Support

The `gpu-module-to-binary` pass supports output format `fatbin`, which generates a fat binary containing:
- A cubin for the specified `gpuChip`
- Embedded PTX for JIT compilation at runtime

This enables architecture mismatch recovery: if the cubin doesn't match the running GPU, the CUDA driver can JIT-compile the PTX to the actual architecture. However, JIT can only target a *higher* compute capability than `gpuChip`.

### 4.4 Cross-Vendor Fat Binaries: Not Supported

While MLIR can produce `gpu.binary` with objects for both NVIDIA and AMD targets, the offloading handler (`gpu.select_object`) selects one object at compile time during LLVM IR translation. There is **no mechanism** to:

1. Detect the available GPU vendor at runtime
2. Dynamically select between NVIDIA cubin and AMD hsaco
3. Fall back from GPU to CPU execution

This is the fundamental gap our poster addresses.

**Source:** [gpu Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/), [D154149: Add gpu-module-to-binary pass](https://reviews.llvm.org/D154149)

---

## 5. Host vs Device Code Generation

### 5.1 Code Separation Model

MLIR separates host and device code structurally:

- **Device code**: Lives inside `gpu.module { gpu.func @kernel kernel { ... } }` -- isolated from host
- **Host code**: Everything outside `gpu.module` -- standard `func.func` operations
- **Outlining**: `gpu-kernel-outlining` pass extracts inline `gpu.launch` regions into `gpu.func` within `gpu.module`, converting to `gpu.launch_func` references

### 5.2 Device Code Compilation

Device code follows a separate compilation path within the same pipeline:
1. Operations inside `gpu.module` are lowered to target-specific dialect (nvvm/rocdl/spirv)
2. The `gpu-module-to-binary` pass serializes to binary blob
3. The binary is embedded in the host module as a global string constant

### 5.3 Host Code Compilation

Host-side GPU operations are lowered to runtime API calls:
- `gpu.launch_func` -> CUDA Driver API calls (`cuModuleLoadData`, `cuLaunchKernel`) or equivalent
- `gpu.alloc` -> `cuMemAlloc` / `hipMalloc`
- `gpu.memcpy` -> `cuMemcpy` / `hipMemcpy`
- `gpu.wait` -> stream synchronization

The `gpu-to-llvm` pass handles this conversion through a thin wrapper interface.

### 5.4 Runtime Module Load/Unload Overhead

A known performance issue: the host code calls `cuModuleLoadData` before kernel launch and `cuModuleUnload` after. For small kernels in loops, this overhead is significant. The Discourse thread on "GPU execution without runtime load/unload" discusses mitigation strategies, but no upstream solution exists for persistent module caching.

**Source:** [Discourse: GPU execution without runtime load/unload](https://discourse.llvm.org/t/mlir-gpu-execution-without-runtime-load-unload/61712)

---

## 6. GPU Dialect Operations in Detail

### 6.1 gpu.launch

Inline kernel definition with grid/block dimensions:

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
           [dynamic_shared_memory_size %s]
           workgroup(%workgroup: memref<32xf32, 3>)
           private(%private: memref<1xf32, 5>) {
  // kernel body (min 12 block arguments: block/thread IDs + dims)
  gpu.terminator
}
```

Supports: async execution via `!gpu.async.token`, cluster dimensions, memory attributions for workgroup (address space 3) and private (address space 5) memory.

### 6.2 gpu.launch_func

Reference-based kernel launch from host code:

```mlir
gpu.launch_func async [%t0] @kernels::@kernel_1
    clusters in (%c1, %c1, %c1)
    blocks in (%bx, %by, %bz)
    threads in (%tx, %ty, %tz)
    dynamic_shared_memory_size %s
    args(%arg0 : f32, %arg1 : memref<?xf32, 1>)
```

Requirements: kernel must have `gpu.kernel` attribute, must be in `gpu.module`, parent must have `gpu.container_module`.

### 6.3 gpu.alloc / gpu.memcpy

```mlir
// Allocate device memory (supports async, host_shared for unified memory)
%mem, %token = gpu.alloc async [%dep] host_shared (%width) : memref<64x?xf32, 1>

// Copy between host and device (async capable)
%token2 = gpu.memcpy async [%token] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```

### 6.4 gpu.module and gpu.binary

```mlir
// Module with target attributes
gpu.module @kernels [#nvvm.target<chip = "sm_90">, #rocdl.target<chip = "gfx90a">] {
  gpu.func @matmul(%a: memref<...>, %b: memref<...>, %c: memref<...>) kernel {
    // ...
    gpu.return
  }
}

// After gpu-module-to-binary:
gpu.binary @kernels <#gpu.select_object<0>> [
    #gpu.object<#nvvm.target<chip = "sm_90">, bin = "...">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, bin = "...">
]
```

**Source:** [gpu Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/)

---

## 7. IREE: The Closest Existing Runtime Solution

### 7.1 Architecture

IREE (Intermediate Representation Execution Environment) is the most complete MLIR-based system for heterogeneous execution:

- **Compiler**: MLIR-based, lowers ML models to HAL (Hardware Abstraction Layer)
- **Runtime**: Minimal, aligned with Vulkan execution model
- **HAL**: Consistent interface across execution resources

### 7.2 Multi-Backend Support

IREE supports multiple GPU backends simultaneously:
- `LLVMGPU/CUDA` -- NVIDIA via PTX/cubin
- `LLVMGPU/HIP` -- AMD via hsaco
- `SPIR-V/Vulkan` -- Vendor-agnostic via Vulkan compute
- `SPIR-V/Metal` -- Apple via Metal Shading Language

### 7.3 Device Selection Model

IREE uses URI-based device selection:

```bash
# Heterogeneous: CPU + Vulkan GPU
--device=local-task --device=vulkan

# Multi-GPU: two CUDA devices
--device=cuda://GPU-abcd0 --device=cuda://GPU-abcd1

# Specific Vulkan device by UUID
--device=vulkan://GPU-uuid-here
```

Operations can be attributed with device categories for constraint-based placement ("big GEMMs go on the accelerator").

### 7.4 Limitations Relevant to Our Work

- Device selection is largely compile-time or CLI-driven, not true runtime introspection
- The HAL is IREE-specific, not reusable as a general MLIR component
- SPIR-V backend generates vendor-agnostic code but still requires Vulkan runtime
- Issue #50 (since project start): multi-device heterogeneous execution remains an open challenge

**Sources:**
- [IREE Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)
- [IREE GPU Vulkan Guide](https://iree.dev/guides/deployment-configurations/gpu-vulkan/)
- [IREE Issue #50](https://github.com/iree-org/iree/issues/50)
- [IREE Issue #15334](https://github.com/iree-org/iree/issues/15334)
- [IREE Issue #12230](https://github.com/iree-org/iree/issues/12230)

---

## 8. Triton: Multi-Backend via MLIR

### 8.1 Architecture

Triton is an MLIR-based compiler for GPU kernels. Its compilation pipeline:

```
Python DSL -> Triton-IR (MLIR dialect) -> Triton-GPU IR (MLIR dialect) -> LLVM IR -> PTX/AMDGCN
```

### 8.2 Multi-Backend Support

- NVIDIA: via NVPTX backend with vendor-specific passes (TMA, Async Dot)
- AMD: via AMDGPU backend with vendor-specific passes (OptimizeLDSUsage, BlockPingpong)
- Backend selection happens at compile time based on the detected/specified GPU

### 8.3 ML-Triton (2025)

ML-Triton extends Triton with a hierarchical compilation pipeline:
- Three-stage lowering with layout-encoding propagation
- Applicable to NVIDIA (WMMA), AMD (MFMA), and potentially other hardware
- Achieves within 5% of expert-tuned C++ and assembly across GEMM, memory-bound, and attention kernels

### 8.4 Relevance

Triton demonstrates that MLIR-based multi-backend GPU compilation works in practice, but its dispatch model is still compile-time (you choose NVIDIA or AMD when compiling). No runtime switching.

**Sources:**
- [ML-Triton Paper (arXiv:2503.14985)](https://arxiv.org/pdf/2503.14985)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [vllm-triton-backend for PyTorch Conference 2025](https://research.ibm.com/publications/vllm-triton-backend-how-to-get-state-of-the-art-performance-on-nvidia-and-amd-with-just-triton)

---

## 9. Relevant RFCs and Discourse Discussions

### 9.1 [RFC] An MLIR Dialect for Distributed Heterogeneous Computing (June 2025)

- **URL**: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- **Proposal**: New dialect with `schedule` operation grouping `task` operations, each annotated with a target (e.g., cpu, gpu)
- **Key ideas**: Explicit orchestration, static analysis, lowering to MPI dialect
- **Status**: RFC stage, not merged
- **Relevance**: Closest upstream proposal to our research direction, but focused on distributed systems (MPI) rather than runtime GPU vendor dispatch

### 9.2 [RFC] Extending MLIR GPU Device Codegen Pipeline (May 2023)

- **URL**: https://discourse.llvm.org/t/rfc-extending-mlir-gpu-device-codegen-pipeline/70199
- **Key problems identified**:
  - Cannot link to bytecode libraries (e.g., libdevice)
  - No device linking support
  - Requires building MLIR on the target system
- **Relevance**: These limitations directly affect runtime compilation scenarios

### 9.3 [RFC] SPIR-V IR as a Vendor Agnostic GPU Representation (2025)

- **URL**: https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- **Proposal**: Use SPIR-V IR as a universal GPU intermediate representation in LLVM
- **Relevance**: Directly aligned with vendor-agnostic dispatch -- if SPIR-V can serve all vendors, runtime dispatch simplifies to "emit SPIR-V, let the driver handle it"

### 9.4 [RFC] Cleaning the GPU Dialect (September 2025)

- **URL**: https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- **Problem**: GPU dialect contains operations that "don't really belong" -- vendor-specific ops mixed with generic abstractions
- **Relevance**: Reflects the structural tension between vendor-agnostic and vendor-specific code in the gpu dialect

### 9.5 [RFC] Add GEN Dialect for Intel GPUs / XeVM Dialect (2024-2025)

- **URLs**: https://discourse.llvm.org/t/rfc-add-gen-dialect-for-intel-gpus/76753, https://discourse.llvm.org/t/rfc-proposal-for-new-xevm-dialect/86955
- **Relevance**: Shows Intel actively building MLIR infrastructure, adding a third major GPU vendor to the ecosystem

### 9.6 GPU Stream/Queue Proposal (2023)

- **URL**: https://discourse.llvm.org/t/proposal-to-add-stream-queue-as-an-optional-argument-to-few-gpu-dialect-ops/67920
- **Problem**: Intel's SYCL/DPC++ runtime requires stream/queue creation with explicit context and device information, but upstream GPU dialect ops don't support this
- **Relevance**: Shows vendor-specific runtime requirements leaking into the "generic" GPU dialect

---

## 10. What MLIR Can and Cannot Do Today

### 10.1 What MLIR CAN Do

1. **Write once, lower to multiple targets**: A single `gpu.module` can have multiple target attributes and produce binaries for NVIDIA, AMD, and SPIR-V simultaneously via `gpu-module-to-binary`

2. **Embed multi-target binaries**: `gpu.binary` can contain objects for multiple architectures in a single binary

3. **Abstract GPU execution model**: The `gpu` dialect provides vendor-agnostic kernel launch, memory management, and synchronization primitives

4. **Fat binary support**: NVIDIA fat binaries with PTX for JIT fallback across compute capability versions

5. **SPIR-V as vendor-agnostic path**: The SPIR-V dialect provides a complete path that avoids vendor-specific LLVM backends entirely

6. **JIT compile host code**: ExecutionEngine can JIT the host-side code at runtime

7. **Transform dialect for tuning**: Parameterized compilation strategies that can be tuned per-target without rebuilding the compiler

### 10.2 What MLIR CANNOT Do Today

1. **Runtime hardware detection**: No mechanism to detect available GPU vendor/architecture at runtime within the MLIR compilation pipeline

2. **Dynamic backend selection**: `gpu.select_object` selects at compile time, not runtime. There is no "if NVIDIA then use cubin, else if AMD then use hsaco" logic

3. **Cross-vendor fat binaries**: While `gpu.binary` can hold both cubin and hsaco, the offloading handler picks one during LLVM IR translation -- not at runtime

4. **GPU device code JIT**: The ExecutionEngine JITs host code only. Device code must be pre-compiled. There is no "JIT compile a CUDA kernel on demand" path (except via NVIDIA's own PTX JIT in fat binaries)

5. **Unified AMD pipeline**: No `gpu-lower-to-rocdl-pipeline` equivalent to the NVIDIA pipeline -- AMD requires manual pass composition

6. **Device management**: The GPU dialect does not yet provide abstractions for device enumeration, capability querying, or context management

7. **Persistent module caching**: No built-in mechanism to avoid repeated `cuModuleLoadData`/`cuModuleUnload` for the same kernel

8. **Link to device libraries**: Cannot link to external bitcode libraries (e.g., libdevice) at the MLIR level without external tooling

9. **Intel GPU pipeline completion**: XeGPU dialect exists upstream but lacks a complete lowering pipeline

10. **Runtime cost model**: No mechanism to decide at runtime which device (CPU vs GPU, or which GPU) would be faster for a given workload

---

## 11. Implications for Our Poster

### 11.1 The Gap We Address

MLIR provides the *compilation* infrastructure for multi-target GPU code (via `gpu-module-to-binary` with multiple target attributes), but completely lacks the *runtime* infrastructure for:
- Hardware introspection (what GPU is available?)
- Dynamic object selection (which binary to load?)
- Fallback chains (GPU not available? Use CPU)
- Cost-model-driven dispatch (which device is fastest for this kernel?)

### 11.2 Where Our Contribution Fits

```
                    MLIR TODAY                    |           OUR CONTRIBUTION
                                                  |
  linalg -> gpu -> {nvvm, rocdl, spirv}           |
  gpu-module-to-binary (compile-time)             |   Runtime dispatch layer:
  gpu.select_object (compile-time selection)      |   - Hardware detection
  gpu.binary [nvvm_obj, rocdl_obj, spirv_obj]     |   - Dynamic object selection
                                                  |   - Fallback chain management
  ExecutionEngine (host JIT only)                 |   - Cost-model integration
                                                  |   - Device code JIT (optional)
```

### 11.3 Key Technical Insight

The `gpu.binary` operation already supports multi-target object storage. The missing piece is a runtime-aware offloading handler that replaces `gpu.select_object`'s compile-time selection with runtime hardware detection and dynamic dispatch. This could be implemented as:

1. A new offloading attribute (e.g., `#gpu.runtime_select`) implementing `GPUOffloadingLLVMTranslationAttrInterface`
2. At LLVM IR translation time, it emits runtime detection code (vendor ID query, capability check)
3. The emitted code selects the appropriate binary object and loads it via the corresponding runtime API

### 11.4 Design Considerations

- **IREE already does this** (partially) via HAL -- our contribution should be lighter weight and composable with the existing MLIR ecosystem without requiring IREE's full stack
- **SPIR-V simplifies the problem**: If all targets accept SPIR-V (via Vulkan), only one binary is needed. But performance may suffer vs native (PTX/AMDGCN)
- **Triton's approach**: Shows multi-backend works but with compile-time selection. Runtime selection would enable serving infrastructure where GPU type varies per machine
- **The "ML kernels are static" argument** (from reviewer 91B): Our value proposition is strongest for inference serving with heterogeneous fleet, edge deployment with variable hardware, and cloud environments with mixed GPU pools

### 11.5 Relevant Existing Components to Build On

1. `gpu.binary` multi-object storage (already exists)
2. `GPUOffloadingLLVMTranslationAttrInterface` (extensible for new handlers)
3. `gpu-module-to-binary` pass (already produces multi-target binaries)
4. ORC JIT's lazy compilation (could enable deferred device code compilation)
5. Transform dialect (could encode per-target optimization strategies)

---

## Appendix: Source Index

### Official MLIR Documentation
- [gpu Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [SPIR-V Dialect](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
- [nvgpu Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [nvvm Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)
- [amdgpu Dialect](https://mlir.llvm.org/docs/Dialects/AMDGPU/)
- [rocdl Dialect](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/)
- [xegpu Dialect](https://mlir.llvm.org/docs/Dialects/XeGPU/)
- [linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [ExecutionEngine Class](https://mlir.llvm.org/doxygen/classmlir_1_1ExecutionEngine.html)
- [ExecutionEngineOptions](https://mlir.llvm.org/doxygen/structmlir_1_1ExecutionEngineOptions.html)
- [MLIR Toy Tutorial Ch. 6](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)

### LLVM Discourse Discussions
- [RFC: Distributed Heterogeneous Computing Dialect](https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960)
- [RFC: Extending GPU Device Codegen Pipeline](https://discourse.llvm.org/t/rfc-extending-mlir-gpu-device-codegen-pipeline/70199)
- [RFC: SPIR-V IR as Vendor-Agnostic GPU Representation](https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115)
- [RFC: Cleaning the GPU Dialect](https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170)
- [RFC: Add XeGPU Dialect](https://discourse.llvm.org/t/rfc-add-xegpu-dialect-for-intel-gpus/75723)
- [RFC: XeVM Dialect Proposal](https://discourse.llvm.org/t/rfc-proposal-for-new-xevm-dialect/86955)
- [How to Generate AMDGPU Code from MLIR](https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627)
- [GPU Execution Without Runtime Load/Unload](https://discourse.llvm.org/t/mlir-gpu-execution-without-runtime-load-unload/61712)
- [GPU Code Generation Status: NVidia, OpenCL](https://discourse.llvm.org/t/gpu-code-generation-status-nvidia-opencl/2080)
- [Lowering GPU Dialect to SPIR-V](https://discourse.llvm.org/t/how-to-correctly-lower-gpu-dialect-to-spirv/90006)
- [GPU Stream/Queue Proposal](https://discourse.llvm.org/t/proposal-to-add-stream-queue-as-an-optional-argument-to-few-gpu-dialect-ops/67920)

### External References
- [Stephen Diehl: GPU Compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)
- [ORC JIT Design and Implementation](https://llvm.org/docs/ORCv2.html)
- [IREE Project](https://iree.dev/)
- [IREE Issue #50: Multi-device](https://github.com/iree-org/iree/issues/50)
- [IREE Issue #15334](https://github.com/iree-org/iree/issues/15334)
- [ML-Triton Paper](https://arxiv.org/pdf/2503.14985)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [Phoronix: Generic MLIR to SPIR-V Pass in LLVM 19](https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V)
- [D154149: Add gpu-module-to-binary pass](https://reviews.llvm.org/D154149)
- [D154104: Add GPU target attribute interface](https://reviews.llvm.org/D154104)
- [Targeting NVIDIA Hopper in MLIR (LLVM Dev Meeting 2024)](https://llvm.org/devmtg/2024-03/slides/nvidia-hopper-in-mlir.pdf)
- [MLIR Transform Dialect (CGO 2025)](https://2025.cgo.org/details/cgo-2025-papers/7/The-MLIR-Transform-Dialect-Your-compiler-is-more-powerful-than-you-think)
- [rocMLIR](https://github.com/ROCm/rocMLIR)
- [FlyDSL: Python-Native DSL on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html)
