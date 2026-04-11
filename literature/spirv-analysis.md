# SPIR-V as a Universal GPU Intermediate Representation
## Critical Analysis for Heterogeneous GPU Kernel Dispatch

**Prepared for:** LLVM Developers' Meeting, Dublin 2026 Poster
**Topic:** Vendor-agnostic runtime dispatch for ML kernels
**Date:** 2026-04-02
**Status:** Literature review — Phase 1

---

## Table of Contents

1. [SPIR-V Overview: The Khronos Standard](#1-spir-v-overview)
2. [Toolchain Ecosystem: clspv, Vulkan Compute, WebGPU](#2-toolchain-ecosystem)
3. [Vendor Support Matrix](#3-vendor-support-matrix)
4. [What SPIR-V Cannot Express](#4-limitations-expressiveness-gaps)
5. [SPIR-V in MLIR: The `spirv` Dialect](#5-spir-v-in-mlir)
6. [SPIR-V Extensions for Compute](#6-spir-v-extensions-for-compute)
7. [SPIR-V vs PTX vs AMDGCN as Compilation Targets](#7-comparison-spir-v-vs-ptx-vs-amdgcn)
8. [IREE's SPIR-V Backend: Vendor-Agnostic Code Generation](#8-iree-spir-v-backend)
9. [Synthesis: Can SPIR-V Truly "Compile Once, Run Anywhere"?](#9-synthesis)
10. [Implications for the Poster Contribution](#10-poster-implications)

---

## 1. SPIR-V Overview

### Standard Definition

SPIR-V (Standard Portable Intermediate Representation — Version) is a binary intermediate language for graphical shader stages and compute kernels, standardized by the Khronos Group. Version 1.0 was released November 16, 2015, directly replacing the earlier text-based SPIR. The current revision (as of research date) is SPIR-V 1.6, Revision 7 (March 12, 2026).

**Key structural properties:**

- **Binary format**: Word-stream encoding, not text. Each instruction is a sequence of 32-bit words. This enables stable serialization and fast parsing, contrasting with LLVM IR's text-based approach.
- **SSA-based**: Functions contain a CFG of basic blocks with SSA-form instructions, mapping conceptually to LLVM IR.
- **Multiple execution models**: Vertex, Fragment, Compute, RayGeneration, etc. ML kernels target the `GLCompute` (Vulkan) or `Kernel` (OpenCL) execution model.
- **Capability system**: Rather than a fixed feature set, modules declare required capabilities (e.g., `Shader`, `Kernel`, `GroupNonUniform`, `CooperativeMatrixKHR`). Drivers validate capability sets against supported features.
- **Extension system**: Additional semantic sets beyond the core specification via named extensions (e.g., `SPV_KHR_cooperative_matrix`, `SPV_INTEL_inline_assembly`).

### The Two Disjoint Environments

A critical and frequently overlooked architectural reality: SPIR-V is not one thing — it is two disjoint subsets that share a format but have fundamentally different semantics:

| Dimension | Vulkan / GLCompute | OpenCL / Kernel |
|---|---|---|
| Execution model | `GLCompute` | `Kernel` |
| Memory model | Vulkan (logical addressing) | OpenCL (physical addressing) |
| Pointer arithmetic | Not allowed in baseline | Allowed |
| Casts between address spaces | Not allowed | Allowed |
| Driver requirement | Any Vulkan 1.1+ driver | OpenCL CL_DEVICE_IL_VERSION |
| Interoperability | No — Vulkan won't run Kernels | No — OpenCL won't run Shaders |

**Implication for our poster**: Any system claiming "vendor-agnostic dispatch via SPIR-V" must choose one of these environments. IREE, clspv, and most ML frameworks target the Vulkan/GLCompute path due to broader hardware availability and the absence of OpenCL deprecation concerns.

### Industry Momentum (2024–2026)

- **September 2024**: Microsoft announced plans to adopt SPIR-V as the Direct3D 12 interchange format, replacing DXIL, starting from Shader Model 7. This is a multi-year transition.
- **Vulkan 1.4**: Requires SPIR-V versions 1.0–1.6.
- **WebGPU / Chrome 141**: Rolling out SPIR-V 1.4 support on Android and ChromeOS, used by the Tint WGSL compiler for improved Vulkan code generation.

**Sources:**
- [SPIR-V Khronos Page](https://www.khronos.org/spirv/)
- [SPIR-V Specification — Khronos Registry](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [DirectX Adopting SPIR-V — Microsoft DirectX Blog](https://devblogs.microsoft.com/directx/directx-adopting-spir-v/)
- [Wikipedia: Standard Portable Intermediate Representation](https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation)

---

## 2. Toolchain Ecosystem

### 2.1 clspv — OpenCL C → SPIR-V for Vulkan

**Repository**: https://github.com/google/clspv (Google, Apache 2.0, not an official product)

clspv compiles a subset of OpenCL C 1.2 into SPIR-V for use with Vulkan compute pipelines. It operates by:
1. Using Clang to parse OpenCL C → LLVM IR
2. Running a series of LLVM Module passes to transform LLVM IR into a Vulkan-compatible SPIR-V module

**Technical limitations:**
- Input is OpenCL C 1.2 only (not full OpenCL C 3.0)
- Covers a subset of OpenCL C (no dynamic libraries, limited builtins)
- Generates Vulkan `GLCompute` shaders, not `Kernel` capability — meaning the output is not compatible with the OpenCL runtime

**Reflection mechanism**: clspv embeds kernel argument metadata via the `NonSemantic.ClspvReflection` extended instruction set, requiring `SPV_KHR_non_semantic_info`. This allows the host runtime to determine buffer layouts without parsing C source.

**Relevance for poster**: clspv demonstrates that OpenCL C source → Vulkan SPIR-V is achievable but requires bridging two different semantic universes. It is not a clean passthrough — it is a source dialect translation.

### 2.2 Vulkan Compute Shaders

Vulkan's compute pipeline accepts SPIR-V modules with `GLCompute` execution model. The computational model is:
- Workgroups → SubGroups → Invocations (direct analog to CUDA: Blocks → Warps → Threads)
- Shared memory via `Workgroup` storage class
- Global memory via `StorageBuffer` (SSBO)
- Push constants for small uniform data

**Important constraint**: Vulkan's logical addressing model prohibits raw pointer arithmetic by default. Physical addressing requires `SPV_EXT_physical_storage_buffer` and `PhysicalStorageBuffer64` capability — which is optional and not universally supported on mobile.

### 2.3 WebGPU / WGSL / Tint

WebGPU uses WGSL (WebGPU Shading Language) as its primary shader language. The Tint compiler (part of Dawn, Google's WebGPU implementation) provides:
- SPIR-V → WGSL translation (for importing Vulkan shaders)
- WGSL → SPIR-V compilation (for generating Vulkan shaders)
- WGSL → HLSL / MSL / GLSL (for other backends)

As of Chrome 141, Tint benefits from SPIR-V 1.4 features when compiling to Vulkan on Android/ChromeOS, including improved subgroup operations and relaxed memory access.

**Key issue**: WebGPU explicitly chose WGSL over SPIR-V as its surface language, citing security concerns (SPIR-V validation complexity, potential for driver exploits). The gpuweb issue tracker (#847) contains the debate — the decision against accepting SPIR-V directly was deliberate.

**Implication**: SPIR-V is a compiler target/interchange format in the WebGPU ecosystem, not a developer-facing language. For ML dispatch systems, this means SPIR-V works as an IR but needs translation for WebGPU consumption.

**Sources:**
- [google/clspv GitHub](https://github.com/google/clspv)
- [OpenCL C on Vulkan spec](https://chromium.googlesource.com/chromiumos/third_party/clspv/+/HEAD/docs/OpenCLCOnVulkan.md)
- [What is SPIR-V — Vulkan Documentation](https://docs.vulkan.org/guide/latest/what_is_spirv.html)
- [WebGPU Chrome 141 — Chrome Dev Blog](https://developer.chrome.com/blog/new-in-webgpu-141)

---

## 3. Vendor Support Matrix

### 3.1 Intel — Native First-Class SPIR-V Consumer

Intel is the strongest native SPIR-V supporter. Intel's GPU compiler stack (IGC — Intel Graphics Compiler) accepts SPIR-V as its primary input format for both OpenCL and Level Zero workloads.

- **Level Zero**: Intel's low-level GPU API accepts SPIR-V modules directly; IGC compiles them JIT to the hardware ISA (GEN ISA, Xe ISA)
- **oneAPI DPC++**: Compiles SYCL → LLVM IR → SPIR-V → IGC → native binary
- **Intel ISPC for Xe**: Outputs SPIR-V by default, consumed by ISPCRT → IGC

The Intel document "SPIR-V: Default Interface to Intel Graphics Compiler for OpenCL Workloads" explicitly confirms SPIR-V is the designated interface to IGC.

**Assessment**: On Intel, SPIR-V is not a portability layer — it is the native interface. Performance is first-class.

### 3.2 AMD — Vulkan Native, ROCm via Vendor-Flavored SPIR-V

**Vulkan path (standard SPIR-V)**: AMD RDNA/CDNA GPUs support Vulkan fully, consuming standard SPIR-V `GLCompute` shaders. This works and is production-quality.

**ROCm compute path**: AMD does not have a device-independent code representation for compute. ROCm's traditional flow compiles to AMDGCN (specific per-GPU ISA). There is **no standard SPIR-V `Kernel` capability support in ROCm**.

**AMD's "vendor-flavored SPIR-V"** (landed in LLVM as of ~LLVM 19+): AMD introduced `amdgcnspirv` — a SPIR-V variant that includes:
- Inline AMDGCN assembly via `SPV_INTEL_inline_assembly` extension
- AMDGCN target-specific built-in functions
- A feature set matching the union of AMDGCN GPU target capabilities

This is invoked as:
```
clang++ -x hip --offload-arch=amdgcnspirv main.cpp
```

**Critical limitation of amdgcnspirv**: GPU architecture macros (`__gfx1100__`, etc.) are undefined at compile time. Wavefront size is not a compile-time constant. Runtime detection via `__builtin_amdgcn_processor_is` is required. This means vendor-flavored SPIR-V is portable across AMD devices but still locked to AMD hardware — it explicitly "forfeits absolute genericity."

**Assessment**: AMD support for SPIR-V as a compute target is functional but requires either the Vulkan path (generic but limited feature set) or the vendor-flavored extension path (AMD-only). Standard portable SPIR-V for ROCm compute does not exist.

### 3.3 NVIDIA — Vulkan Only, No Native SPIR-V Compute Driver

NVIDIA's compute stack (CUDA/PTX) does not consume SPIR-V. NVIDIA's Vulkan driver accepts SPIR-V `GLCompute` shaders and compiles them internally to the hardware ISA, but:
- There is no public PTX-via-SPIR-V path
- CUDA compute jobs cannot be submitted as SPIR-V to the CUDA driver
- The Vulkan driver path exists but is optimized for graphics workloads

**Control flow tension**: NVIDIA Volta+ GPUs use entirely unstructured control flow with explicit barrier primitives (`bssy`, `bsync`). SPIR-V mandates structured control flow via `OpSelectionMerge`/`OpLoopMerge`. NVIDIA's Vulkan driver must perform CFG restructuring when consuming SPIR-V — this introduces correctness challenges (documented Genshin Impact rendering bug from improper re-convergence) and potential performance overhead.

**ChipStar**: A research project compiling HIP/CUDA to SPIR-V for execution on Intel/AMD via OpenCL. ChipStar 1.1 was released in 2024, demonstrating that CUDA→SPIR-V translation is feasible but requires significant workarounds.

**Assessment**: NVIDIA does not natively support SPIR-V as a compute target. Vulkan compute is functional but carries structural mismatch costs.

### 3.4 ARM Mali — Good Vulkan SPIR-V Support

ARM Mali (Valhall+ architecture) supports Vulkan 1.1+ and therefore consumes SPIR-V `GLCompute` shaders. IREE explicitly lists Mali as a supported target with "Good" performance rating for its SPIR-V/Vulkan backend.

ARM's GPU architecture is tile-based deferred rendering (TBDR), which means memory access patterns optimized for NVIDIA/AMD may be suboptimal on Mali. However, the SPIR-V abstraction is honored at the driver level.

**Subgroup support**: Mali Valhall supports `VK_KHR_shader_subgroup_extended_types` and related subgroup operations, enabling portable subgroup-level parallelism via SPIR-V.

### 3.5 Qualcomm Adreno — Reasonable Vulkan SPIR-V Support

Qualcomm Adreno (640+ architectures) supports Vulkan and SPIR-V. IREE lists Adreno as a supported target with "Reasonable" performance through its SPIR-V/Vulkan backend.

Adreno has its own QCOM extensions in the SPIR-V registry, including `SPV_QCOM_cooperative_matrix_conversion` (which requires `SPV_KHR_cooperative_matrix`). This demonstrates that even mobile vendors are extending SPIR-V for ML acceleration.

**Assessment**: Adreno is a functional SPIR-V target, but performance tuning still requires architecture-specific configuration (tile sizes, memory access patterns).

### 3.6 Vendor Support Summary Table

| Vendor | SPIR-V Vulkan Compute | SPIR-V OpenCL Kernel | Native SPIR-V Compute | ML Kernel Quality |
|---|---|---|---|---|
| Intel | Yes (via Vulkan) | Yes (IGC primary) | **Yes** — IGC native | First-class |
| AMD | Yes (RDNA/CDNA Vulkan) | No | Vendor-flavored only | Good (Vulkan path) |
| NVIDIA | Yes (Vulkan driver) | No | No | Reasonable (overhead) |
| ARM Mali | Yes (Valhall+) | No | No | Good |
| Qualcomm Adreno | Yes (640+) | No | No | Reasonable |
| Apple (Metal) | No direct SPIR-V | No | No | Via MoltenVK only |

**Sources:**
- [AMD AMDGCN Flavored SPIR-V — Phoronix](https://www.phoronix.com/news/LLVM-AMDGCN-Flavored-SPIR-V)
- [ROCm SPIR-V Support — AMD ROCm Docs](https://rocm.docs.amd.com/projects/llvm-project/en/develop/conceptual/spirv.html)
- [Mali processor — Wikipedia](https://en.wikipedia.org/wiki/Mali_(processor))
- [Adreno — Wikipedia](https://en.wikipedia.org/wiki/Adreno)
- [Intel SPIR-V for OpenCL — Intel PDF](https://www.intel.com/content/dam/develop/external/us/en/documents/spirv-for-publishing.pdf)
- [Re-converging Control Flow on NVIDIA — Collabora Blog](https://www.collabora.com/news-and-blog/blog/2024/04/25/re-converging-control-flow-on-nvidia-gpus/)

---

## 4. Limitations: What SPIR-V Cannot Express

This is the most critical section for poster framing. SPIR-V's portability comes at a price in expressiveness.

### 4.1 Structured Control Flow Mandate

SPIR-V enforces **structured control flow** via `OpSelectionMerge` and `OpLoopMerge` instructions. All branches must be structured — no unstructured jumps, even when the control flow is provably uniform.

**Why this is a serious problem:**

1. **Compiler-generated code**: High-level ML compilers (including MLIR-lowered code from Linalg or SCF dialects) may produce unstructured CFGs that require structurization passes before SPIR-V emission. These passes are lossy — they inflate code size and reduce optimization opportunities.

2. **GPU hardware mismatch**: NVIDIA Volta+ uses unstructured control flow internally. The Vulkan driver must:
   - Accept structured SPIR-V
   - Convert to unstructured native ISA
   This double conversion introduces correctness risks and optimization loss. The Collabora blog documented a real rendering bug caused by incorrect re-convergence.

3. **Subgroup divergence semantics**: SPIR-V's specification of divergent subgroup operations is underspecified. The 2022 blog post "The Trouble with SPIR-V" describes this as "shockingly ill-defined." Subgroup operations (shuffles, reductions) executed in divergent control flow have implementation-defined behavior — a portability fiction.

### 4.2 No Unstructured Pointer Arithmetic (Vulkan Path)

In the Vulkan environment, SPIR-V uses logical addressing. Raw pointer arithmetic is prohibited by default. This prevents:
- Pointer-based kernel interfaces common in OpenCL/CUDA
- Custom memory allocators within kernels
- Efficient implementation of sparse data structures

Workaround: `SPV_EXT_physical_storage_buffer` enables physical addressing, but this extension is optional and not supported on all mobile GPUs. Using it breaks portability to Mali and older Adreno.

### 4.3 Tensor Cores / Matrix Operations — Incomplete Standardization

NVIDIA's tensor cores expose a hierarchy of matrix-multiply instructions in PTX:
- `wmma.*` — warp-cooperative, older (Volta/Turing)
- `mma.sync.*` — synchronous warp MMA (Ampere), supports more types and shapes
- `wgmma.mma_async.*` — warp-group cooperative, asynchronous (Hopper), 4th-gen tensor cores

**SPIR-V's equivalent**: `SPV_KHR_cooperative_matrix` (core extension, approved 2023, Revision 10 as of April 2025). This provides `OpCooperativeMatrixMulAddKHR` with:
- Flexible matrix dimensions (defined by implementation, not spec)
- Floating-point and integer component types
- Saturation support for integer accumulation

**Gap between SPIR-V cooperative matrix and PTX MMA:**

| Feature | PTX wgmma.mma_async (Hopper) | SPV_KHR_cooperative_matrix |
|---|---|---|
| Asynchronous execution | Yes — async issue, barrier-sync | No — synchronous only |
| Warp-group scope (128 threads) | Yes | Subgroup scope (implementation-defined) |
| Direct register file tensors | Yes (RF accumulation) | No — memory-mapped |
| Specific tile sizes (M=64,N=8...) | Hardware-defined, exposed | Implementation-defined, opaque |
| TMA (Tensor Memory Accelerator) | Hopper-specific, PTX-native | No equivalent in SPIR-V |

**Vendor-specific extensions filling the gap:**
- `SPV_NV_cooperative_matrix` / `SPV_NV_cooperative_matrix2`: NVIDIA's pre-standard extensions, expose more Turing/Ampere-specific features
- `SPV_INTEL_subgroup_matrix_multiply_accumulate`: Intel's subgroup-level matrix multiply
- `SPV_QCOM_cooperative_matrix_conversion`: Qualcomm's mobile cooperative matrix extension

**Reality**: Writing ML kernels that use tensor cores portably via SPIR-V means accepting the lowest common denominator of `SPV_KHR_cooperative_matrix`, which misses Hopper's async warp-group MMA, AMD's WMMA native ISA optimizations, and Intel's XMX engine-specific tile sizes.

### 4.4 No Asynchronous Memory Operations

CUDA exposes `cp.async` (Ampere+) for asynchronous global→shared memory copies, enabling compute-memory overlap (the basis of "async pipeline" kernels in Triton and CUTLASS). SPIR-V has no equivalent. Achieving memory-compute overlap via SPIR-V requires relying on the driver's prefetcher — which is hardware and driver-revision dependent.

### 4.5 No Direct Access to Hardware Scheduling Primitives

- **NVIDIA**: `__syncthreads()`, `cooperative_groups::sync()`, `bar.sync` in PTX — fine-grained barrier control within a thread block
- **AMD**: `s_barrier`, wavefront-level operations in AMDGCN
- **SPIR-V equivalent**: `OpControlBarrier` with scope and semantics — abstract, driver-interpreted

The SPIR-V barrier model is expressive enough for correctness but does not guarantee any particular microarchitectural scheduling behavior. Peak-performance kernels on NVIDIA (e.g., FlashAttention, CUTLASS) rely on barrier piggybacking and producer-consumer warp coordination that has no SPIR-V expression.

### 4.6 Limited Integer Type Support in Practice

Non-32-bit integer types (i8, i16, i64) require explicit capability declarations (`Int8`, `Int16`, `Int64`). MLIR's SPIR-V dialect converts index types unconditionally to `i32` and lacks full support for `i8`/`i16` in the type converter. IREE's SPIR-V backend includes an explicit pass `-iree-spirv-emulate-i64` that implements 64-bit integer operations using 32-bit primitives — a performance compromise required for mobile GPU portability.

**Sources:**
- [The Trouble with SPIR-V, 2022 Edition — Gob's Blog](https://xol.io/blah/the-trouble-with-spirv/)
- [SPIR-V Relationship with gpu.subgroup_mma_compute — LLVM Discourse](https://discourse.llvm.org/t/relationship-between-gpu-subgroup_mma_compute-and-spirv-khr-cooperativematrixmuladd-in-mlir/88531)
- [SPV_KHR_cooperative_matrix Specification](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html)
- [SPV_NV_cooperative_matrix](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_matrix.html)
- [NVIDIA PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Re-converging Control Flow on NVIDIA — Collabora](https://www.collabora.com/news-and-blog/blog/2024/04/25/re-converging-control-flow-on-nvidia-gpus/)

---

## 5. SPIR-V in MLIR: The `spirv` Dialect

### 5.1 Design Philosophy

The MLIR `spirv` dialect is defined in `mlir/lib/Dialect/SPIRV/` and serves as "a proper intermediate representation to facilitate compiler transformations" while maintaining serializability to SPIR-V binary. Three core design principles:

1. **Semantic parity**: One-to-one mapping with SPIR-V specification concepts
2. **MLIR-idiomatic representation**: Uses MLIR regions, block arguments, and attribute system where cleaner than literal binary format
3. **Straightforward (de)serialization**: `mlir::spirv::serialize()` / `mlir::spirv::deserialize()`

### 5.2 Module Structure

`spirv.module` encapsulates the entire SPIR-V program. Key constraints:
- Closed region — only `func` ops and `spirv.*` ops permitted inside
- No implicit capturing of external SSA values
- Capabilities and extensions declared as module-level attributes (e.g., `#spirv.target_env`)

The `#spirv.target_env` attribute specifies:
- SPIR-V version (`v1.0` through `v1.6`)
- Required extensions (`SPV_KHR_16bit_storage`, etc.)
- Required capabilities (`Shader`, `Kernel`, `GroupNonUniform`, etc.)
- Resource limits (workgroup dimensions, invocation counts)

### 5.3 Type System

SPIR-V dialect types include:
- `!spirv.ptr<i32, Function>` — pointer with explicit storage class
- `!spirv.array<4 x i32, stride = 4>` — fixed-size array
- `!spirv.rtarray<i32>` — runtime-sized array (for SSBOs)
- `!spirv.struct<f32 [0], i32 [4]>` — struct with layout decorations

`memref` is lowered by `SPIRVTypeConverter` to `!spirv.ptr<spirv.struct<spirv.array<...>>>`. Notably, the type converter avoids LLVM-style `MemrefDescriptor` structs because:
- Vulkan logical addressing prohibits pointer load/store
- Embedding shape metadata wastes descriptor space
- Separate `OpVariable` declarations enable push constants for shape data

**Known limitation (documented)**: Index type is hardcoded to `i32`. Non-32-bit scalar types convert unconditionally to 32-bit. Vectors beyond length 4 are unsupported (must convert to scalars or arrays).

### 5.4 Operation Naming Convention

- `CamelCase` ops (e.g., `spirv.FMul`): Direct mirrors of SPIR-V instructions, 1:1 serialization
- `snake_case` ops (e.g., `spirv.module`, `spirv.mlir.selection`): MLIR-idiomatic, may map to multiple instructions
- `mlir.snake_case` ops (e.g., `spirv.mlir.merge`): Structural requirements with no binary equivalent

### 5.5 Control Flow Representation

Rather than literal `OpSelectionMerge`/`OpLoopMerge` instructions, the dialect uses regions:

- `spirv.mlir.selection`: Region containing header, case blocks, merge block
- `spirv.mlir.loop`: Region with entry, header, body, continue, merge blocks

During serialization, these regions materialize into SPIR-V basic blocks with merge instructions. During deserialization, merge instructions are reconstructed into regions.

**Implication for compilation**: Lowering from MLIR dialects that use unstructured CFGs (e.g., SCF dialect after loop unrolling) requires explicit structurization before SPIR-V emission. This is a known friction point.

### 5.6 GPU Dialect → SPIR-V Lowering Path

The lowering chain from GPU-level MLIR to SPIR-V binary:

```
Linalg (named ops: matmul, conv, etc.)
  ↓ tiling + distribution passes
SCF / Affine dialects (loops, parallel regions)
  ↓ gpu-map-parallel-loops / convert-parallel-loops-to-gpu
GPU dialect (gpu.module, gpu.func, gpu.block_id, gpu.thread_id)
  ↓ convert-gpu-to-spirv pass
SPIR-V dialect (spirv.module, spirv.func, spirv.GlobalVariable)
  ↓ mlir::spirv::serialize()
SPIR-V binary (.spv)
```

Key conversion patterns (from `MLIRGPUToSPIRV` library):
- `gpu.module` → `spirv.module`
- `gpu.func` → entry function with `spirv.entry_point_abi`
- `gpu.block_id` / `gpu.thread_id` → `spirv.GlobalVariable` with `BuiltIn` decoration
- `gpu.barrier` → `spirv.ControlBarrier`

**Missing lowerings (active development areas as of 2026):**
- `gpu.subgroup_broadcast` → `spirv.GroupNonUniformBroadcast` (recently added, per LLVM bug #157940)
- `gpu.subgroup_mma_*` → `spirv.KHR.CooperativeMatrix*` (partially implemented, IREE extends this)

The `SPIRVConversionTarget` performs dynamic legality checks against the declared `#spirv.target_env`, preventing illegal ops from appearing in the output module.

### 5.7 ABI Passes

Two passes finalize the SPIR-V ABI:
- `LowerABIAttributesPass`: Converts `spirv.entry_point_abi` and `spirv.interface_var_abi` into `spirv.GlobalVariable` definitions and `spirv.EntryPoint` ops
- `UpdateVCEPass`: Computes the minimum required Version/Capabilities/Extensions from actual module content and updates `spirv.target_env`

**Sources:**
- [SPIR-V Dialect — MLIR Official Docs](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
- [SPIR-V Dialect — LLVM Googlesource](https://llvm.googlesource.com/llvm-project/+/refs/heads/main/mlir/docs/Dialects/SPIR-V.md)
- [AMD AI Engineer Generic MLIR→SPIR-V Pass (LLVM 19) — Phoronix](https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V)
- [GPU Dialect → SPIR-V Discourse](https://discourse.llvm.org/t/how-to-correctly-lower-gpu-dialect-to-spirv/90006)
- [SPIR-V Dialect to LLVM Conversion Manual — MLIR](https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/)

---

## 6. SPIR-V Extensions for Compute

### 6.1 SPV_KHR_cooperative_matrix

**Status**: Approved 2023-05-03 (Khronos SPIR-V WG), 2023-06-16 (Khronos Board). Revision 10 as of 2025-04-24. Requires SPIR-V 1.3 minimum. Requires `VulkanMemoryModel` capability.

**Core capability**: `CooperativeMatrixKHR`

**Types**: `OpTypeCooperativeMatrixKHR` — declares matrix type with:
- Component type (any scalar numeric type)
- Scope (which invocation group shares the matrix)
- Rows, Columns (parameterized — not fixed by spec, defined per client API)
- Use (MatrixAKHR / MatrixBKHR / MatrixAccumulatorKHR)

**Operations**:
- `OpCooperativeMatrixMulAddKHR`: A×B+C with optional saturation
- `OpCooperativeMatrixLoadKHR` / `OpCooperativeMatrixStoreKHR`: Memory access in row-major or column-major layout
- `OpCooperativeMatrixLengthKHR`: Number of elements accessible to current invocation
- Standard arithmetic on elements (add, sub, mul, negate)

**Storage restriction**: Cooperative matrix types can only be allocated in `Function` or `Private` storage class — they cannot be passed through workgroup memory or SSBOs. This restricts multi-stage pipelines that want to share matrix fragments across compute dispatches.

**Vendor comparison**:
- Replaces `SPV_NV_cooperative_matrix` (NVIDIA pre-standard extension)
- Qualcomm adds `SPV_QCOM_cooperative_matrix_conversion` on top
- Intel uses `SPV_INTEL_subgroup_matrix_multiply_accumulate` for XMX engines (different scope semantics)

**Gap relative to PTX**: `SPV_KHR_cooperative_matrix` is synchronous and register-file bound. NVIDIA's Hopper `wgmma.mma_async` is fundamentally different — it operates on 128-thread warp groups, accesses shared memory directly (not register file), and executes asynchronously. There is no SPIR-V equivalent for async tensor core operation.

### 6.2 Subgroup Operations

SPIR-V's `GroupNonUniform*` instructions provide warp/wavefront-level operations:
- `spirv.GroupNonUniformBroadcast` — broadcast value from specific lane
- `spirv.GroupNonUniformBroadcastFirst` — broadcast from lowest-active lane
- `spirv.GroupNonUniformShuffle` / `ShuffleXor` / `ShuffleUp` / `ShuffleDown`
- `spirv.GroupNonUniformIAdd` / `FAdd` / `IMul` / `FMul` (subgroup reductions)
- `spirv.GroupNonUniformElect` — leader election

**Capability requirements**: `GroupNonUniform`, `GroupNonUniformShuffle`, `GroupNonUniformArithmetic` — declared in `spirv.module`.

**Divergence problem**: As documented in "The Trouble with SPIR-V," the behavior of `GroupNonUniform*` ops under divergent control flow is implementation-defined. Different drivers may produce different results when the invocation set is not fully active within the scope. This is a portability hole in the specification.

### 6.3 Other Relevant Extensions

| Extension | Purpose | Notes |
|---|---|---|
| `SPV_KHR_16bit_storage` | i16/f16 in StorageBuffer/Uniform | Required for FP16 ML workloads |
| `SPV_KHR_8bit_storage` | i8 in StorageBuffer | Required for INT8 quantization |
| `SPV_KHR_float_controls` | Rounding mode, NaN behavior control | Important for numerically stable ML |
| `SPV_KHR_vulkan_memory_model` | Proper Vulkan memory ordering | Required by cooperative matrix |
| `SPV_EXT_physical_storage_buffer` | Pointer arithmetic in Vulkan | Breaks mobile portability |
| `SPV_INTEL_inline_assembly` | Inline ISA code (repurposed by AMD) | Vendor-specific, non-portable |
| `SPV_KHR_non_semantic_info` | Debug/reflection metadata (NonSemantic.*) | Used by clspv for kernel reflection |

**Sources:**
- [SPV_KHR_cooperative_matrix — Khronos Registry](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html)
- [SPV_NV_cooperative_matrix — Khronos Registry](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_matrix.html)
- [SPV_QCOM_cooperative_matrix_conversion — Khronos Registry](https://github.khronos.org/SPIRV-Registry/extensions/QCOM/SPV_QCOM_cooperative_matrix_conversion.html)
- [SPV_INTEL_subgroup_matrix_multiply_accumulate — Khronos Registry](https://github.khronos.org/SPIRV-Registry/extensions/INTEL/SPV_INTEL_subgroup_matrix_multiply_accumulate.html)
- [VK_KHR_cooperative_matrix — Vulkan Docs](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
- [SPIR-V Dialect — MLIR](https://mlir.llvm.org/docs/Dialects/SPIR-V/)

---

## 7. Comparison: SPIR-V vs PTX vs AMDGCN as Compilation Targets

### 7.1 Design Philosophy

| Property | SPIR-V | PTX (NVIDIA) | AMDGCN |
|---|---|---|---|
| Controlled by | Khronos (committee, multi-vendor) | NVIDIA (sole authority) | AMD (sole authority) |
| Abstraction level | High — API-level IR | Medium — virtual ISA | Low — hardware ISA |
| Binary format | Stable, versioned, portable | Versioned, NVIDIA-only | Per-generation, AMD-only |
| Target | Any SPIR-V driver | NVIDIA GPUs ≥ specified compute capability | Specific AMD GFX target |
| JIT strategy | Driver-side (vendor black box) | CUDA driver (NVIDIA-controlled) | ROCm runtime |
| Forward compatibility | Extension-based | Explicit (compute_XX) | Per-ISA generation |

### 7.2 PTX Strengths over SPIR-V

**Forward compatibility**: PTX is designed for compute capability versioning. `compute_70` PTX JIT-compiles to any NVIDIA GPU of capability 7.0 or higher, including future generations. NVIDIA has maintained this contract for many years.

**Tensor core exposure**: PTX directly exposes `wmma`, `mma.sync`, and `wgmma.mma_async` — the full tensor core ISA including Hopper's 4th-generation hardware. There is no intermediary abstraction.

**Unstructured control flow**: PTX allows arbitrary control flow. No structurization overhead.

**Shared memory control**: PTX exposes `ld.shared`, `st.shared`, `bar.sync`, and `cp.async` directly, enabling precise control over the shared memory pipeline (critical for software pipelining in CUTLASS, FlashAttention).

**Warp-level primitives**: `shfl.*`, `vote.*`, `match.*` — fine-grained warp lane manipulation with no abstraction.

**PTX weakness**: Vendor lock-in. PTX code runs only on NVIDIA hardware. There is no cross-vendor equivalent.

### 7.3 AMDGCN Strengths and Weaknesses

**Strength**: Direct hardware access. AMD publishes ISA reference guides. Performance-critical kernels can be hand-tuned at the ISA level.

**Weakness**: No device-independent representation. Every target GFX generation (gfx906, gfx1100, etc.) has a distinct ISA. Shipping a single binary for all AMD GPUs requires shipping multiple AMDGCN variants — the ROCm toolchain does exactly this (compiling separately for each GPU in the support matrix).

**AMDGCN vs SPIR-V for portability**: AMDGCN is strictly less portable than standard SPIR-V. AMD's "vendor-flavored SPIR-V" partially addresses this by providing a single AMDGCN-family SPIR-V variant, but it remains AMD-only.

### 7.4 SPIR-V's Unique Position

SPIR-V is the only intermediate representation that is:
- Accepted by multiple GPU vendors (Intel natively, AMD/NVIDIA via Vulkan, mobile via Vulkan)
- Standardized by a neutral body (Khronos)
- Extensible without breaking older consumers (capability + extension system)

**The fundamental tension**: To achieve portability across NVIDIA, AMD, and Intel, a system must either:
1. Use standard SPIR-V → accept performance floor at ~50–80% of native (documented in the arxiv paper 2603.28793)
2. Use vendor-specific extensions → lose portability
3. Multi-target compile → generate PTX + AMDGCN + SPIR-V separately (what SYCL implementations do)

### 7.5 AdaptiveCpp's SSCP Model as a Reference Architecture

AdaptiveCpp (formerly hipSYCL) implements Single-Source, Single Compiler Pass (SSCP) by:
1. Compiling source once to LLVM IR (backend-independent, with SYCL builtins annotated)
2. Embedding LLVM IR in the application binary
3. At runtime, JIT-compiling LLVM IR → PTX (NVIDIA) or AMDGCN (AMD) or SPIR-V (Intel)

Performance results: kernels run "typically within 10% performance in both directions" of native. The runtime JIT overhead roughly doubles the existing driver-JIT time, which is already present in all SPIR-V consumers.

**Key insight for our poster**: AdaptiveCpp shows that a unified IR (LLVM IR in their case) with runtime specialization can achieve near-native performance across all vendors. SPIR-V could serve a similar role but is constrained by its Vulkan/GLCompute semantic restrictions.

**Sources:**
- [Understanding PTX — NVIDIA Technical Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
- [AdaptiveCpp SSCP — AdaptiveCpp Docs](https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/)
- [ROCm SPIR-V — AMD Docs](https://rocm.docs.amd.com/projects/llvm-project/en/develop/conceptual/spirv.html)
- [Toward a Universal GPU ISA — arXiv 2603.28793](https://arxiv.org/html/2603.28793)
- [hipSYCL Single-Pass SYCL — IWOCL/SYCLcon 2023 Paper](https://cdrdv2-public.intel.com/786536/Heidelberg_IWOCL__SYCLCon_2023_paper_2566-1.pdf)

---

## 8. IREE's SPIR-V Backend: Vendor-Agnostic Code Generation

### 8.1 What "Vendor-Agnostic" Means in IREE

IREE's claim of vendor-agnostic SPIR-V code generation means: **IREE generates standard Vulkan-compatible SPIR-V that, in principle, runs on any Vulkan-compliant driver without vendor-specific extensions**. The vendor-agnosticism is at the SPIR-V level, not the performance level.

The mechanism: IREE's `iree-spirv-select-lowering-strategy-pass` selects a dispatch pipeline based on runtime-queried device capabilities (subgroup size, cooperative matrix support, available extensions) at runtime. The SPIR-V binary that gets dispatched is pre-compiled, but multiple variants are compiled and the appropriate one is selected at runtime via IREE's VM dispatch mechanism.

### 8.2 Complete IREE SPIR-V Codegen Pass Pipeline

IREE implements 21 distinct SPIR-V codegen passes, organized in this conceptual order:

**Strategy Selection:**
- `-iree-spirv-select-lowering-strategy-pass`: Selects among `SPIRVSubgroupReduce`, `SPIRVCooperativeMatrixVectorize`, `LLVMGPUTileAndFuse`, etc.
- `-iree-spirv-convert-gpu-target`: Transforms GPU target spec into SPIR-V environment descriptor

**Tiling and Distribution:**
- `-iree-spirv-tile-and-distribute`: Partitions Linalg ops across invocations
- `-iree-spirv-tile-and-promote`: Elevates tiled ops to workgroup memory
- `-iree-spirv-tile-to-cooperative-ops`: Subgroup-targeted tiling for cooperative matrix

**Vectorization:**
- `-iree-spirv-initial-vector-lowering`: Early vector adaptations
- `-iree-spirv-vectorize-to-cooperative-ops`: Subgroup vectorization → cooperative ops
- `-iree-spirv-vector-to-gpu-subgroup-mma-ops`: Vector → GPU MMA primitives
- `-iree-spirv-vectorize-load-store`: Memory access vectorization
- `-iree-spirv-breakdown-large-vector`: Decomposes unsupported vector sizes
- `-iree-spirv-final-vector-lowering`: Terminal vector transforms

**Memory and Type Handling:**
- `-iree-spirv-map-memref-storage-class`: Memref space → SPIR-V storage class
- `-iree-spirv-erase-storage-buffer-static-shape`: Static → dynamic buffer conversion
- `-iree-spirv-emulate-i64`: 64-bit integer emulation via 32-bit (for mobile portability)
- `-iree-spirv-annotate-winograd-loops`: Winograd convolution loop distribution

**Finalization:**
- `-iree-convert-to-spirv`: Final dialect conversion
- `-iree-spirv-lower-executable-target-pass`: Primary lowering with dispatch pipeline
- `-iree-spirv-lower-executable-using-transform-dialect`: Transform dialect-driven lowering
- `-iree-spirv-link-executables`: Consolidates SPIR-V executables
- `-iree-spirv-trim-executable-target-env`: Minimizes required capabilities
- `-iree-spirv-materialize-executable-conditions`: Target requirement validation

### 8.3 Supported Hardware and Performance Tiers

IREE's Vulkan/SPIR-V backend explicitly supports (from official docs):

| Target | Architecture | Performance Rating | Notes |
|---|---|---|---|
| ARM Mali | Valhall+ | Good | Mobile primary target |
| Qualcomm Adreno | 640+ | Reasonable | Mobile secondary |
| AMD RDNA+ | RDNA2/3 | Good | Desktop/server |
| NVIDIA Turing+ | RTX 20xx+ | Reasonable | Not primary target |

The `--iree-vulkan-target` flag accepts architecture names (`rdna3`, `valhall4`, `ampere`, `adreno`) or product names (`rx7900xtx`, `a100`). This translates to SPIR-V target environment configurations.

**Critical statement from IREE docs**: "We don't support the full spectrum of GPUs" and target specifications are "just an approximation for usage given Vulkan implementation variances across extensions and properties."

### 8.4 Limitations of IREE's "Vendor-Agnostic" Claim

**Limitation 1 — Vulkan-only**: IREE's SPIR-V codegen path exclusively targets Vulkan. It does not support OpenCL (no `Kernel` capability SPIR-V). This means it cannot run on OpenCL-only environments (some HPC nodes, some Intel iGPUs in restricted configurations).

**Limitation 2 — Multiple compiled variants**: To achieve "vendor-agnostic" execution, IREE compiles multiple SPIR-V variants at model compilation time (one per target configuration). This is not "compile once" — it is "compile for each target class and dispatch at runtime."

**Limitation 3 — No μkernel support on SPIR-V**: IREE issue #17788 documents that "GPU backends like SPIR-V and ROCm currently don't have compilers/tools for generating the core implementation of a μkernel." μkernels — tightly hand-tuned compute kernels — are the mechanism for achieving peak performance on CPU (IREE uses LLVM-generated μkernels). For SPIR-V/GPU, there is no equivalent: performance-critical operations fall back to compiled SPIR-V rather than optimized microcode.

**Limitation 4 — Performance gap from abstraction**: The IREE SPIR-V backend achieves "vendor-agnostic" semantics by staying within standard SPIR-V, but this means not using `wgmma.mma_async` on Hopper, not using AMD-specific memory access patterns, and not using Intel XMX engine specifics. The performance cost is real.

**Limitation 5 — Extension dependency**: The cooperative matrix path (`SPIRVCooperativeMatrixVectorize`) requires `VK_KHR_cooperative_matrix` — which is an optional extension, not universally supported. Runtime variant selection handles this but adds binary size overhead.

### 8.5 The Vulkanised 2025 Presentation (AMD/IREE)

A February 2025 presentation "The Long Tail of AI: SPIR-V in IREE and MLIR" by Jakub Kuderski (AMD) at Vulkanised 2025 addressed SPIR-V's role in ML compilation. The presentation covered:
- Codegen strategy selection based on device queries
- Variant selection at runtime for different capability tiers
- The challenge of the "long tail" — many GPU configurations exist beyond the top-tier targets

This confirms IREE's approach: runtime variant selection rather than true single-binary portability.

**Sources:**
- [SPIRV Passes — IREE Reference](https://iree.dev/reference/mlir-passes/CodegenSPIRV/)
- [GPU Vulkan Deployment — IREE](https://iree.dev/guides/deployment-configurations/gpu-vulkan/)
- [IREE μkernel SPIR-V Issue #17788](https://github.com/iree-org/iree/issues/17788)
- [Vulkanised 2025 SPIR-V in IREE and MLIR (PDF)](https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf)
- [IREE Codegen Passes Design Doc](https://github.com/iree-org/iree/blob/4546315d004d4cfc4079f3d13164bc92fddf6061/docs/developers/design_docs/codegen_passes.md)

---

## 9. Synthesis: Can SPIR-V Truly "Compile Once, Run Anywhere"?

### 9.1 The Honest Answer

**No — not for peak-performance ML kernels.** Yes — for functional correctness at a performance floor.

SPIR-V achieves genuine portability in the following sense: a SPIR-V `GLCompute` module written against the Vulkan 1.1 core spec plus `SPV_KHR_16bit_storage` will run correctly on ARM Mali Valhall, Qualcomm Adreno 640+, AMD RDNA, Intel Xe, and NVIDIA Turing+ without modification. This is real and valuable.

What it does not guarantee:
- Equivalent performance across vendors
- Access to vendor-specific acceleration hardware (tensor cores, XMX engines, matrix units)
- Predictable memory access latency or scheduling behavior
- Convergent subgroup behavior under divergent control flow

### 9.2 The Performance Gap

The arXiv paper "Toward a Universal GPU ISA" (arXiv:2603.28793) provides the most rigorous quantification: portable interfaces achieve **50–80% of native performance** because they abstract rather than formalize the underlying execution model. Five of six benchmark configurations met the 80% threshold, with four exceeding 100% on specific workloads. The key missing primitive for full portability: intra-wave shuffle (adding it brings the mandatory primitive count to 11).

### 9.3 The Structural Mismatch Problem

The three major compilation targets represent fundamentally different execution models:

- **PTX**: Per-thread virtual ISA, warp-implicit parallelism, forward-compatible, JIT-stable
- **AMDGCN**: Wavefront-64 hardware ISA, per-generation, maximum hardware exposure
- **SPIR-V**: Invocation-group model, structured control flow, capability-gated, driver-compiled

These cannot be unified at the IR level without semantic loss. The most promising approach (demonstrated by AdaptiveCpp SSCP and planned in our poster work) is to unify at the **source/compiler IR level** (LLVM IR or MLIR) and lower to each target separately at runtime.

### 9.4 What SPIR-V Does Well

1. **Broad hardware availability**: Vulkan is the most widely available compute API on heterogeneous hardware, especially mobile
2. **Neutral standards body**: No single vendor controls the roadmap
3. **Toolchain integration**: Strong integration with MLIR, clang, DXC, Tint, SPIRV-Tools
4. **Offline compilation**: SPIR-V enables pre-compilation and binary distribution without source
5. **Rapidly expanding ecosystem**: DirectX SM7 adoption (2024), WebGPU adoption, SPIR-V 1.6

### 9.5 The Runtime Dispatch Angle (Our Poster's Contribution)

The insight this research supports: SPIR-V is best understood not as a "compile once run anywhere" target but as a **lowest-common-denominator portable IR that enables runtime variant selection**. The dispatch architecture should:

1. Compile ML kernels to multiple representations (SPIR-V for portability, PTX for NVIDIA peak, AMDGCN for AMD peak)
2. At runtime, probe device capabilities (via Vulkan device properties, CUDA driver, ROCm driver)
3. Dispatch to the highest-performance available variant

IREE demonstrates this pattern but couples it tightly to the Vulkan runtime. Our poster's contribution is to surface this dispatch decision into MLIR's compilation pipeline via a lightweight, modular dispatch layer — not assuming Vulkan as the only path.

**Sources:**
- [Toward a Universal GPU ISA — arXiv:2603.28793](https://arxiv.org/html/2603.28793)
- [AdaptiveCpp Compilation Model](https://adaptivecpp.github.io/AdaptiveCpp/compilation/)
- [SPIR-V Performance Analysis — apxml.com](https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/spirv-heterogeneous-execution)
- [SYCL vs OpenCL vs Vulkan Compute — Till Code](https://tillcode.com/sycl-vs-opencl-vs-vulkan-compute-cross-platform-gpu-api/)

---

## 10. Implications for the Poster Contribution

### Key Technical Claims Supported by This Research

1. **SPIR-V is a portability layer, not a performance layer.** Any design claiming SPIR-V alone suffices for ML kernels across NVIDIA/AMD/Intel is either accepting significant performance loss or limiting scope to non-peak workloads.

2. **The dispatch problem is real and unsolved.** There is no existing system that compiles ML kernels once and dispatches across NVIDIA (PTX), AMD (AMDGCN), and Intel (SPIR-V) at near-native performance from a unified source. AdaptiveCpp comes closest but is SYCL-specific.

3. **MLIR is the right integration point.** The MLIR SPIR-V dialect provides the necessary infrastructure for SPIR-V emission from the MLIR GPU dialect. The missing piece is the runtime dispatch mechanism and the multi-target lowering strategy selection.

4. **The cooperative matrix gap is actionable.** `SPV_KHR_cooperative_matrix` covers the basics but misses async Hopper MMA. For the poster, this gap is a concrete example of where vendor-specific dispatch (PTX wgmma) is needed and SPIR-V is insufficient.

5. **IREE's approach validates the architecture but shows its limits.** IREE's 21-pass SPIR-V codegen pipeline, multi-variant compilation, and runtime selection is the right pattern. The poster should extend this into a more modular, vendor-aware dispatch framework that is not tied to the Vulkan stack.

### Risks to Flag

- **IREE already does much of this**: Must clearly differentiate our contribution from IREE's existing work. Our angle: modular MLIR pass pipeline that integrates dispatch decisions into the compilation IR itself, not a separate runtime system.
- **SPIR-V structured control flow**: If our prototype generates kernels with complex control flow, structurization overhead may dominate and invalidate performance claims.
- **Cooperative matrix vendor fragmentation**: Using SPV_KHR_cooperative_matrix means accepting the extension gap relative to native PTX. Be explicit about this tradeoff in the poster.

---

## Appendix: Reference Index

| Source | URL | Type |
|---|---|---|
| SPIR-V Khronos Page | https://www.khronos.org/spirv/ | Standard |
| SPIR-V Specification | https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html | Standard |
| SPIR-V Wikipedia | https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation | Reference |
| Vulkan What is SPIR-V | https://docs.vulkan.org/guide/latest/what_is_spirv.html | Guide |
| MLIR SPIR-V Dialect | https://mlir.llvm.org/docs/Dialects/SPIR-V/ | Technical Docs |
| MLIR SPIR-V to LLVM | https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/ | Technical Docs |
| google/clspv | https://github.com/google/clspv | Tool |
| OpenCL C on Vulkan (clspv) | https://chromium.googlesource.com/chromiumos/third_party/clspv/+/HEAD/docs/OpenCLCOnVulkan.md | Docs |
| DirectX Adopting SPIR-V | https://devblogs.microsoft.com/directx/directx-adopting-spir-v/ | Blog |
| AMD AMDGCN Flavored SPIR-V | https://www.phoronix.com/news/LLVM-AMDGCN-Flavored-SPIR-V | News |
| AMD ROCm SPIR-V Docs | https://rocm.docs.amd.com/projects/llvm-project/en/develop/conceptual/spirv.html | Docs |
| Re-converging Control Flow NVIDIA | https://www.collabora.com/news-and-blog/blog/2024/04/25/re-converging-control-flow-on-nvidia-gpus/ | Blog |
| The Trouble with SPIR-V | https://xol.io/blah/the-trouble-with-spirv/ | Blog |
| SPV_KHR_cooperative_matrix | https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html | Standard |
| SPV_NV_cooperative_matrix | https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_matrix.html | Standard |
| SPV_QCOM_cooperative_matrix | https://github.khronos.org/SPIRV-Registry/extensions/QCOM/SPV_QCOM_cooperative_matrix_conversion.html | Standard |
| SPV_INTEL_subgroup_matrix | https://github.khronos.org/SPIRV-Registry/extensions/INTEL/SPV_INTEL_subgroup_matrix_multiply_accumulate.html | Standard |
| NVIDIA PTX ISA 9.2 | https://docs.nvidia.com/cuda/parallel-thread-execution/ | Standard |
| NVIDIA PTX Blog | https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/ | Blog |
| IREE SPIR-V Passes Reference | https://iree.dev/reference/mlir-passes/CodegenSPIRV/ | Docs |
| IREE Vulkan Deployment | https://iree.dev/guides/deployment-configurations/gpu-vulkan/ | Docs |
| IREE μkernel SPIR-V Issue | https://github.com/iree-org/iree/issues/17788 | Issue |
| Vulkanised 2025 IREE/MLIR Talk | https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf | Talk |
| AdaptiveCpp SSCP | https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/ | Docs |
| AdaptiveCpp Compilation Model | https://adaptivecpp.github.io/AdaptiveCpp/compilation/ | Docs |
| Toward Universal GPU ISA | https://arxiv.org/html/2603.28793 | Paper |
| Intel SPIR-V for OpenCL | https://www.intel.com/content/dam/develop/external/us/en/documents/spirv-for-publishing.pdf | Docs |
| LLVM 19 MLIR→SPIR-V Pass | https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V | News |
| WebGPU Chrome 141 | https://developer.chrome.com/blog/new-in-webgpu-141 | Blog |
| VK_KHR_cooperative_matrix | https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html | Standard |
| ChipStar 1.1 HIP/CUDA→SPIR-V | https://www.phoronix.com/news/ChipStar-1.1-HIP-CUDA-SPIR-V | News |
