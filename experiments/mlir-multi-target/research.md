# MLIR Multi-Target GPU Compilation: Deep Research

**Date:** 2026-04-02
**Context:** LLVM Dublin 2026 poster -- Heterogeneous GPU Kernel Dispatch
**Scope:** Compiling a single MLIR module to NVPTX, AMDGCN, SPIR-V, and x86 simultaneously

---

## 1. Compiling a Single MLIR Module to Multiple Targets Simultaneously

### The Core Mechanism: `gpu.module` Target Attributes

MLIR's GPU dialect natively supports multi-target compilation through the `targets` attribute on `gpu.module`. A single GPU module can be annotated with an **array** of target attributes, each implementing the `GPUTargetAttrInterface`. The `gpu-module-to-binary` pass iterates over all attached targets, invoking `target.serializeToObject()` for each, and producing a `gpu.binary` containing one `gpu.object` per target.

```mlir
// Single module, multiple targets
gpu.module @kernels <#gpu.select_object<1>> [
    #nvvm.target<chip = "sm_90">,
    #rocdl.target<chip = "gfx90a">,
    #spirv.target<...>
] {
  gpu.func @my_kernel(...) kernel { ... }
}
```

After `gpu-module-to-binary`, this becomes:

```mlir
gpu.binary @kernels [
  #gpu.object<#nvvm.target<chip = "sm_90">, "...cubin data...">,
  #gpu.object<#rocdl.target<chip = "gfx90a">, "...hsaco data...">,
  #gpu.object<#spirv.target<...>, "...spirv binary...">
]
```

### What This Means

MLIR already has first-class support for multi-target compilation at the `gpu.module` level. This is not a hack -- it is the **designed architecture**. The `gpu.module` acts as the fork point: a single device IR is lowered through target-specific conversion passes, then serialized to multiple binary objects within a single `gpu.binary` operation.

### Limitation: The Fork Must Happen Before Target-Specific Lowering

The challenge is that `convert-gpu-to-nvvm`, `convert-gpu-to-rocdl`, and `convert-gpu-to-spirv` are **destructive** passes -- they replace GPU dialect ops with target-specific ops (NVVM intrinsics, ROCDL intrinsics, SPIR-V ops). You cannot run `convert-gpu-to-nvvm` and then also run `convert-gpu-to-rocdl` on the same module.

**Implication:** To compile for multiple targets, you must either:
1. Clone the `gpu.module` before target-specific lowering and run separate pipelines on each clone, OR
2. Use the `gpu-module-to-binary` pass which internally handles this by operating on the pre-lowered GPU module and invoking per-target serialization

The `gpu-module-to-binary` pass handles option (2) internally. The target attributes define their own serialization pipelines, so the pass can serialize the same GPU module to multiple targets without the user needing to manually fork.

---

## 2. Forking at the GPU Dialect Level

### Architecture of the Fork Point

The GPU dialect is the natural fork point in MLIR's compilation hierarchy:

```
linalg/tensor (target-agnostic)
    |
    v
affine/scf (target-agnostic loops)
    |
    v
gpu.launch / gpu.module (target-agnostic device code)  <-- FORK POINT
    |            |            |           |
    v            v            v           v
  nvvm         rocdl        spirv       llvm (CPU)
    |            |            |           |
    v            v            v           v
  NVPTX        AMDGCN      SPIR-V      x86-64
  (cubin)      (hsaco)     (binary)    (object)
```

### How `gpu-module-to-binary` Implements the Fork

Source: `mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp` and the `gpu-module-to-binary` pass implementation.

The pass iterates through `op.getTargetsAttr()` and for each target:
1. Calls `target.serializeToObject(op, targetOptions)`
2. The target attribute interface implementation handles the entire lowering pipeline internally
3. Each target returns a serialized byte array (cubin, hsaco, SPIR-V binary)
4. All serialized objects are collected into a `gpu.BinaryOp`

This means the serialization infrastructure **clones the module internally** for each target. The user-facing IR only sees the GPU dialect before serialization and the binary after.

### Can We Fork Manually?

Yes, but it requires a custom pass. The approach would be:
1. After `gpu-kernel-outlining`, clone each `gpu.module`
2. Run target-specific pipelines on each clone
3. Merge results into a single `gpu.binary`

This is essentially what `gpu-module-to-binary` already does, but a custom approach could allow more control over per-target optimization.

---

## 3. `mlir-opt` Multi-Target Lowering

### Current Support

`mlir-opt` supports multi-target lowering through the pipeline infrastructure:

```bash
# NVIDIA target
mlir-opt input.mlir \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"

# For multi-target, use target attachment + module-to-binary
mlir-opt input.mlir \
  -gpu-kernel-outlining \
  -nvvm-attach-target="chip=sm_90" \
  -rocdl-attach-target="chip=gfx90a" \
  -gpu-module-to-binary
```

### The `*-attach-target` Passes

These passes add target attributes to `gpu.module` operations:
- `-nvvm-attach-target`: Adds `#nvvm.target` with chip, features, triple, opt-level
- `-rocdl-attach-target`: Adds `#rocdl.target` with chip, features, triple
- `-spirv-attach-target`: Adds `#spirv.target` with version, extensions, capabilities
- `-xevm-attach-target`: Adds Intel XeVM target (newer addition)

Multiple attach passes can be chained to add multiple targets to the same `gpu.module`:

```bash
mlir-opt input.mlir \
  -gpu-kernel-outlining \
  -nvvm-attach-target="chip=sm_90" \
  -nvvm-attach-target="chip=sm_80" \
  -rocdl-attach-target="chip=gfx90a" \
  -gpu-module-to-binary
```

This produces a `gpu.binary` with THREE objects: sm_90 cubin, sm_80 cubin, and gfx90a hsaco.

### What Is NOT Supported Natively

There is **no** built-in `gpu-lower-to-rocdl-pipeline` equivalent to `gpu-lower-to-nvvm-pipeline`. The AMDGPU path requires manual pass pipeline construction. This was confirmed in LLVM Discourse (thread: "How to Generate AMDGPU Code from MLIR?").

There is no single `gpu-lower-to-all-targets-pipeline` that handles everything. Multi-target compilation requires either:
- Using attach-target passes + gpu-module-to-binary (which internally handles per-target lowering), OR
- Building a custom pipeline that clones and forks

---

## 4. Exact Pass Pipelines Per Target

### 4.1 NVIDIA (linalg -> NVPTX cubin)

The `gpu-lower-to-nvvm-pipeline` as implemented in `GPUToNVVMPipeline.cpp`:

```
Phase 1: Common Pipeline (buildCommonPassPipeline)
  1.  convert-nvgpu-to-nvvm
  2.  gpu-kernel-outlining
  3.  convert-vector-to-scf
  4.  convert-scf-to-cf
  5.  convert-nvvm-to-llvm
  6.  convert-func-to-llvm
  7.  expand-strided-metadata           (memref pass)
  8.  nvvm-attach-target                (chip, features, triple, opt-level)
  9.  lower-affine
  10. convert-arith-to-llvm
  11. convert-index-to-llvm             (index-bitwidth param)
  12. canonicalize
  13. cse

Phase 2: GPU Module Pipeline (nested on gpu.module)
  14. convert-gpu-to-nvvm               (useBarePtrCallConv, indexBitwidth)
  15. canonicalize
  16. cse
  17. reconcile-unrealized-casts

Phase 3: Host Post-Pipeline (buildHostPostPipeline)
  18. gpu-to-llvm                       (host/kernel barePtrCallConv)
  19. gpu-module-to-binary              (compilationTarget format)
  20. convert-math-to-llvm
  21. canonicalize
  22. cse
  23. reconcile-unrealized-casts
```

**Preceding passes (user must run before this pipeline):**
```
one-shot-bufferize
convert-linalg-to-affine-loops  (or convert-linalg-to-loops)
convert-affine-for-to-gpu       (or convert-parallel-loops-to-gpu)
```

Full end-to-end: `linalg.op` -> `one-shot-bufferize` -> `convert-linalg-to-affine-loops` -> `convert-affine-for-to-gpu` -> `gpu-lower-to-nvvm-pipeline`

### 4.2 AMD (linalg -> AMDGCN hsaco)

No built-in pipeline exists. Manual construction required:

```
Phase 0: Pre-GPU (same as NVIDIA)
  one-shot-bufferize
  convert-linalg-to-affine-loops
  convert-affine-for-to-gpu

Phase 1: GPU Outlining
  gpu-kernel-outlining

Phase 2: GPU Module Pipeline (nested on gpu.module)
  strip-debuginfo
  convert-gpu-to-rocdl              (chipset=gfx90a, index-bitwidth=32)

Phase 3: Target and Serialization
  rocdl-attach-target               (chip=gfx90a)
  convert-scf-to-cf
  convert-func-to-llvm
  expand-strided-metadata
  convert-arith-to-llvm
  convert-index-to-llvm
  lower-affine
  gpu-to-llvm
  gpu-module-to-binary
  reconcile-unrealized-casts
```

The `convert-gpu-to-rocdl` pass maps GPU ops to ROCDL intrinsics:
- `gpu.thread_id` -> `rocdl.workitem.id.x/y/z`
- `gpu.block_id` -> `rocdl.workgroup.id.x/y/z`
- `gpu.barrier` -> `rocdl.barrier`

Serialization produces HSACO (HSA Code Object) binaries via the ROCm toolchain.

### 4.3 SPIR-V (linalg -> SPIR-V binary)

The SPIR-V path diverges from NVVM/ROCDL because SPIR-V is a separate dialect, not a lowering to LLVM:

```
Phase 0: Pre-GPU (same as above)
  one-shot-bufferize
  convert-linalg-to-affine-loops
  convert-affine-for-to-gpu

Phase 1: GPU Outlining
  gpu-kernel-outlining

Phase 2: GPU to SPIR-V Conversion
  convert-gpu-to-spirv              (GPU ops -> SPIR-V ops)
  convert-arith-to-spirv
  convert-scf-to-spirv
  convert-func-to-spirv
  convert-index-to-spirv
  convert-vector-to-spirv
  convert-memref-to-spirv

Phase 3: SPIR-V Optimization and Serialization
  spirv-canonicalize
  spirv-lower-abi-attrs
  spirv-update-vce
  spirv-attach-target
  gpu-module-to-binary
```

**Key difference:** The SPIR-V path converts `gpu.module` -> `spirv.module` directly, rather than going through LLVM IR. The `spirv.module` enforces SPIR-V-specific constraints (no external SSA captures, restricted op set).

**Alternative path (LLVM 19+):** A newer `convert-gpu-to-llvm-spv` pass generates LLVM IR intended for the SPIR-V backend (LLVM's SPIR-V target), rather than using the SPIR-V dialect. This is useful for Intel GPUs and OpenCL runtimes.

```
  convert-gpu-to-llvm-spv           (GPU -> LLVM for SPIR-V backend)
  # Then use LLVM's SPIR-V target backend
```

### 4.4 CPU/x86 (linalg -> LLVM IR -> native)

No GPU dialect involved. Direct lowering through LLVM:

```
Phase 0: Bufferization
  one-shot-bufferize

Phase 1: Linalg Lowering
  convert-linalg-to-loops           (or convert-linalg-to-affine-loops)
  lower-affine                      (if affine loops used)

Phase 2: SCF and Control Flow
  convert-scf-to-cf

Phase 3: LLVM Lowering
  convert-func-to-llvm
  convert-arith-to-llvm
  convert-math-to-llvm
  convert-index-to-llvm
  expand-strided-metadata
  finalize-memref-to-llvm
  convert-cf-to-llvm
  reconcile-unrealized-casts

Phase 4: Translation
  mlir-translate --mlir-to-llvmir
  llc -march=x86-64 -mcpu=...
```

---

## 5. GPU Fat Binaries in LLVM

### 5.1 Clang Offload Bundler (`clang-offload-bundler`)

**Purpose:** Bundles multiple device code objects (+ host code) into a single file.

**Binary format:**
```
Offset  Size    Field
0       24      Magic: "__CLANG_OFFLOAD_BUNDLE__"
24      8       Number of bundle entries (N)
32      N*...   Entry descriptors:
                  8 bytes: offset to code object
                  8 bytes: size of code object
                  8 bytes: length of bundle entry ID string
                  L bytes: bundle entry ID string (not NUL-terminated)
...     ...     Code object data sections
```

**Bundle Entry ID format:** `<offload-kind>-<target-triple>[-<target-id>]`

Examples:
- `host-x86_64-unknown-linux-gnu`
- `openmp-amdgcn-amd-amdhsa-gfx906:sramecc+:xnack+`
- `hip-amdgcn-amd-amdhsa-gfx90a`

### 5.2 LLVM Offload Binary (`llvm-offload-binary`)

**Purpose:** Newer, more structured format for embedding device images in host objects.

**Binary format:**
```
Magic:   0x10FF10AD
Version: uint32_t (currently 1)
Size:    uint64_t (total binary size)
Entry:   offset + size for entry table
         offset + size for string table
         offset + size for device image

Entry fields:
  - image_kind:   uint16_t (object, bitcode, cubin, fatbinary, PTX)
  - offload_kind: uint16_t (OpenMP, CUDA, HIP, SYCL)
  - flags:        uint32_t
  - string table: key-value metadata (triple, arch, etc.)
  - image data:   raw device binary
```

**Multi-target:** Multiple offloading images are concatenated. Each has its own header (self-describing), allowing tools to locate entries even after linker operations.

**Embedding:** Device images go into the `.llvm.offloading` ELF section with `SHF_EXCLUDE` flag (stripped from final binary by linker, extracted during linking phase).

### 5.3 CUDA Fat Binary Format (NVIDIA)

NVIDIA's `fatbin` format bundles PTX and SASS (cubin) for multiple compute capabilities:

```bash
nvcc kernel.cu -o kernel \
  --gpu-architecture=compute_70 \
  --gpu-code=sm_70,sm_80,sm_90
```

This produces a fatbin containing:
- PTX for compute_70 (forward-compatible fallback)
- cubin for sm_70
- cubin for sm_80
- cubin for sm_90

The CUDA runtime selects the best match at load time via `cuModuleLoadFatBinary`.

### 5.4 Relationship to MLIR

MLIR's `gpu.binary` with multiple `gpu.object` entries is conceptually equivalent to a fat binary. The `#gpu.select_object` attribute determines which object to embed/use:

```mlir
gpu.binary @kernels <#gpu.select_object<#rocdl.target>> [
  #gpu.object<#nvvm.target<chip = "sm_90">, "...cubin...">,
  #gpu.object<#rocdl.target<chip = "gfx90a">, "...hsaco...">,
]
```

During LLVM translation, `SelectObjectAttr` selects one object and embeds it. This is a **compile-time** selection, not runtime dispatch. For runtime dispatch, a custom offloading handler would be needed.

---

## 6. Custom Multi-Target MLIR Binary Format

### What Exists Today

MLIR's `gpu.binary` already serves as a multi-target container at the IR level. However, during LLVM translation, the `#gpu.select_object` handler selects **one** object to embed. The final executable contains only one target's binary.

### Proposed Custom Format

For true runtime multi-target dispatch, we would need a custom format that:

1. **Embeds all target binaries** in the final executable (not just one)
2. **Includes a dispatch table** mapping target triples to binary offsets
3. **Runtime probes** available hardware and selects the appropriate binary

A possible design:

```
MLIR Multi-Target Binary Format (proposed)
==========================================
Header:
  magic:          "MLIR_MTB\0"        (8 bytes)
  version:        uint32_t
  num_targets:    uint32_t
  num_kernels:    uint32_t

Target Table (repeated num_targets times):
  target_id:      uint32_t            (enum: NVPTX=0, AMDGCN=1, SPIRV=2, X86=3)
  triple:         offset into string table
  chip:           offset into string table
  binary_offset:  uint64_t
  binary_size:    uint64_t

Kernel Table (repeated num_kernels times):
  kernel_name:    offset into string table
  per_target[]:   { target_idx, entry_point_offset }

String Table:
  NUL-terminated strings

Binary Data:
  Concatenated device binaries (cubin, hsaco, spirv, x86 object)
```

### Implementation Path

To implement this in MLIR:
1. **Create a custom `OffloadingLLVMTranslationAttrInterface`** that replaces `#gpu.select_object`
2. The custom handler embeds ALL objects (not just one) as global constant arrays
3. Generate a dispatch function that queries device type and returns the correct binary pointer
4. The dispatch function calls `mgpuModuleLoad()` with the selected binary

This is the **core contribution** for the poster: demonstrating that MLIR's existing infrastructure can be extended to support runtime multi-target dispatch with minimal changes.

### Alternatively: Use LLVM's Offload Binary Format

Instead of a custom format, extend MLIR's serialization to produce an `llvm-offload-binary` compatible output. This reuses existing tooling:

```bash
# After gpu-module-to-binary, extract objects and bundle them:
llvm-offload-binary -o fat.bin \
  --image=file=kernel.cubin,triple=nvptx64-nvidia-cuda,arch=sm_90 \
  --image=file=kernel.hsaco,triple=amdgcn-amd-amdhsa,arch=gfx90a \
  --image=file=kernel.spv,triple=spirv64-unknown-unknown
```

---

## 7. GPU-to-LLVM Conversion and gpu-launch-func Lowering

### The `gpu-to-llvm` Pass

This pass converts host-side GPU operations to LLVM dialect + runtime calls. It does NOT touch device code inside `gpu.module`. Key conversions:

| GPU Operation | LLVM Translation |
|---|---|
| `gpu.alloc` | `mgpuMemAlloc()` |
| `gpu.dealloc` | `mgpuMemFree()` |
| `gpu.memcpy` | `mgpuMemcpy()` |
| `gpu.launch_func` | Module load + kernel launch sequence |
| `gpu.wait` | `mgpuStreamSynchronize()` |

### gpu.launch_func Lowering Detail

The `SelectObjectAttr` implementation in `SelectObjectAttr.cpp` generates this sequence:

```
1. Binary Embedding (at module init, priority 123):
   - mgpuModuleLoad(global_binary_ptr, size) -> module_handle
   - Store module_handle in global variable

2. Kernel Launch (at each gpu.launch_func):
   - mgpuModuleGetFunction(module_handle, "kernel_name") -> func_handle
   - mgpuStreamCreate() -> stream
   - Allocate argument struct on stack
   - Store each kernel argument into struct
   - mgpuLaunchKernel(func_handle,
                       gridX, gridY, gridZ,
                       blockX, blockY, blockZ,
                       dynSharedMem, stream,
                       &arg_struct, /*extra=*/nullptr)
   - mgpuStreamSynchronize(stream)
   - mgpuStreamDestroy(stream)

3. Module Unload (at program exit, priority 123):
   - mgpuModuleUnload(module_handle)
```

### The `mgpu*` Runtime API

These are **wrapper functions** (not direct CUDA/HIP/OpenCL calls) that provide a stable ABI. The actual runtime library maps them:
- For CUDA: `mgpuModuleLoad` -> `cuModuleLoadData`
- For ROCm: `mgpuModuleLoad` -> `hipModuleLoadData`
- For Vulkan/SPIR-V: Would need a custom implementation

This abstraction layer is key for multi-target dispatch -- the runtime library selection determines which GPU API is called.

### JIT vs AOT Loading

Two loading modes:
- `mgpuModuleLoadJIT(data, optLevel)`: For PTX assembly (NVIDIA JIT compiles at load time)
- `mgpuModuleLoad(data, size)`: For pre-compiled binaries (cubin, hsaco)

---

## 8. gpu.module and Device Code Mapping

### gpu.module Structure

```mlir
gpu.module @module_name [<target_attributes>] {
  // GPU functions
  gpu.func @kernel1(%arg0: memref<...>) kernel {
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    // ... kernel body ...
    gpu.return
  }

  // Non-kernel device functions (callable from kernels)
  gpu.func @helper(%x: f32) -> f32 {
    // ...
    gpu.return %result : f32
  }

  // Global/shared memory
  memref.global @shared_mem : memref<256xf32, 3>  // address space 3 = shared
}
```

### Key Properties

1. **Symbol table semantics**: `gpu.module` acts as a symbol table, enabling `gpu.launch_func` to reference kernels by name (`@module::@kernel`)

2. **Isolation**: GPU modules cannot capture SSA values from the surrounding context. All data must be passed as kernel arguments or reside in global memory.

3. **Address spaces**: Memory references use address spaces:
   - 0: Global memory
   - 1: Constant memory (NVIDIA)
   - 3: Shared/workgroup memory
   - 5: Private/scratch memory

4. **gpu.container_module attribute**: The top-level `module` must have `{gpu.container_module}` to indicate it contains GPU code.

### Mapping to Device Code

| MLIR Concept | CUDA Equivalent | HIP Equivalent | SPIR-V Equivalent |
|---|---|---|---|
| `gpu.module` | Translation unit | Translation unit | `spirv.module` |
| `gpu.func kernel` | `__global__` function | `__global__` function | `OpEntryPoint` |
| `gpu.func` (non-kernel) | `__device__` function | `__device__` function | Regular function |
| `gpu.thread_id` | `threadIdx.x` | `threadIdx.x` | `GlobalInvocationId` |
| `gpu.block_id` | `blockIdx.x` | `blockIdx.x` | `WorkgroupId` |
| `gpu.block_dim` | `blockDim.x` | `blockDim.x` | `WorkgroupSize` |
| `gpu.grid_dim` | `gridDim.x` | `gridDim.x` | `NumWorkgroups` |
| `gpu.barrier` | `__syncthreads()` | `__syncthreads()` | `OpControlBarrier` |
| `gpu.shuffle` | `__shfl_sync()` | `__shfl()` | Subgroup ops |

### gpu.module Lifecycle

```
gpu.module @M           (IR-level device code container)
    |
    | gpu-kernel-outlining (gpu.launch body -> gpu.func in gpu.module)
    |
    | *-attach-target (annotate with compilation targets)
    |
    | convert-gpu-to-{nvvm,rocdl,spirv} (lower to target dialect)
    |
    | gpu-module-to-binary (serialize to gpu.binary)
    |
gpu.binary @M           (compiled device code blob)
    |
    | gpu-to-llvm (lower host code, generate runtime calls)
    |
    | LLVM translation (SelectObjectAttr embeds binary, generates load/launch)
    |
    v
Final executable with embedded device binary
```

---

## Summary: Multi-Target Compilation Architecture

### What MLIR Provides Today

1. **Target-agnostic device IR** via the GPU dialect (`gpu.module`, `gpu.func`, `gpu.launch_func`)
2. **Multiple target attributes** on a single `gpu.module`
3. **Per-target serialization** via `gpu-module-to-binary` (produces one object per target)
4. **Fat binary representation** via `gpu.binary` with multiple `gpu.object` entries
5. **Target attachment passes** (`nvvm-attach-target`, `rocdl-attach-target`, `spirv-attach-target`, `xevm-attach-target`)

### What Is Missing (Poster Contribution Opportunity)

1. **Runtime dispatch**: `#gpu.select_object` does compile-time target selection. No built-in runtime dispatch exists.
2. **Unified pipeline**: No single pipeline handles all targets. The NVIDIA path has `gpu-lower-to-nvvm-pipeline`; AMD and SPIR-V require manual pipeline construction.
3. **ROCDL pipeline**: No `gpu-lower-to-rocdl-pipeline` exists (confirmed on LLVM Discourse).
4. **Cross-vendor binary format**: The `gpu.binary` multi-object approach is IR-level only. No standard on-disk format bundles all targets for runtime selection.
5. **CPU fallback integration**: The CPU path bypasses GPU dialect entirely; integrating a CPU fallback into the `gpu.binary` dispatch would require custom work.

### Proposed Poster Contribution

A custom `OffloadingLLVMTranslationAttrInterface` implementation (replacing `#gpu.select_object`) that:
- Embeds ALL target binaries in the final executable
- Generates a runtime dispatch function that probes available hardware
- Falls back to CPU if no GPU is available
- Uses the existing `mgpu*` runtime abstraction layer

This would demonstrate that MLIR's GPU compilation infrastructure is **architecturally ready** for heterogeneous multi-target dispatch, requiring only a thin runtime dispatch layer on top.

---

## References

- [GPU Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR Passes Reference](https://mlir.llvm.org/docs/Passes/)
- [SPIR-V Dialect Documentation](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
- [GPU Compilation with MLIR (Stephen Diehl)](https://www.stephendiehl.com/posts/mlir_gpu/)
- [Clang Offload Bundler Documentation](https://clang.llvm.org/docs/ClangOffloadBundler.html)
- [Clang Offloading Design & Internals](https://clang.llvm.org/docs/OffloadingDesign.html)
- [llvm-offload-binary Documentation](https://llvm.org/docs/CommandGuide/llvm-offload-binary.html)
- [D154149: gpu-module-to-binary pass review](https://reviews.llvm.org/D154149)
- [SelectObjectAttr.cpp source](https://mlir.llvm.org/doxygen/SelectObjectAttr_8cpp_source.html)
- [GPUToNVVMPipeline.cpp source](https://mlir.llvm.org/doxygen/GPUToNVVMPipeline_8cpp_source.html)
- [AMDGPU Pipeline Discussion (Discourse)](https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627)
- [SPIR-V Lowering Discussion (Discourse)](https://discourse.llvm.org/t/how-to-correctly-lower-gpu-dialect-to-spirv/90006)
- [LLVM 19 Generic MLIR-to-SPIR-V Pass (Phoronix)](https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V)
- [IREE Compiler and Runtime](https://iree.dev/)
- [GPU Compilation (vectorfold.studio)](https://vectorfold.studio/blog/gpu-compilation)
