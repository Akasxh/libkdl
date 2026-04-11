# MLIR GPU Compilation Infrastructure: Deep Technical Analysis
*Research compiled 2026-04-06 for LLVM Dublin 2026 poster on vendor-agnostic GPU kernel dispatch.*

---

## Summary Table

| Topic | Finding | Relevance to libkdl |
|-------|---------|---------------------|
| `gpu-module-to-binary` | Compiles one `gpu.module` to N objects (one per target), all stored in `gpu.binary` | This is the AoT multi-target compilation step libkdl sits after |
| `gpu.select_object` | Selects ONE object at LLVM-IR translation time (compile-time, not runtime) | **The gap**: no runtime hardware detection; libkdl fills this |
| `GPUOffloadingLLVMTranslationAttrInterface` | Interface for custom binary embedding + kernel launch codegen | libkdl could implement this as `#gpu.runtime_select` |
| SPIR-V RFC | Proposes SPIR-V as universal GPU IR in LLVM; RFC active, not merged | If merged, vendor-agnostic compilation simplifies; runtime dispatch still needed |
| ExecutionEngine | JITs host code only; device code must be pre-compiled blobs | No device-side JIT; our runtime dispatch layer operates on pre-compiled blobs |

---

## 1. `gpu-module-to-binary` Pass

### Problem

`gpu.module` operations contain device code that must be compiled to vendor-specific
binary blobs (cubin, hsaco, SPIR-V binary) before the host program can run.
The compilation pipeline must support multiple targets simultaneously.

### Pass Definition

From `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td`:

```tablegen
def GpuModuleToBinaryPass : Pass<"gpu-module-to-binary", ""> {
  let summary = "Transforms a GPU module into a GPU binary.";
  let description = [{
    This pass searches for all nested GPU modules and serializes the module
    using the target attributes attached to the module, producing a GPU binary
    with an object for every target.

    The `format` argument can have the following values:
    1. `offloading`, `llvm`: produces an offloading representation.
    2. `assembly`, `isa`: produces assembly code.
    3. `binary`, `bin`: produces binaries.
    4. `fatbinary`, `fatbin`: produces fatbinaries.
  }];
  let options = [
    Option<"toolkitPath", "toolkit", "std::string", [{""}], "Toolkit path.">,
    ListOption<"linkFiles", "l", "std::string", "Extra files to link to.">,
    Option<"cmdOptions", "opts", "std::string", [{""}], "Command line options.">,
    Option<"compilationTarget", "format", "std::string", [{"fatbin"}],
      "The target representation of the compilation process.">,
    Option<"elfSection", "section", "std::string", [{""}],
      "ELF section where binary is to be located.">
  ];
}
```

### Target Attachment Passes (run before `gpu-module-to-binary`)

| Pass | Default chip | Key options |
|------|-------------|-------------|
| `-nvvm-attach-target` | `sm_75` | chip, O, features (PTX version), triple, fast, ftz |
| `-rocdl-attach-target` | `gfx900` | chip, O, features, triple, abi, wave64, fast, daz |
| `-spirv-attach-target` | (none) | spirvVersion, spirvCapabilities, spirvExtensions, clientApi, deviceVendor, deviceType, deviceId |
| `-xevm-attach-target` | `bmg` | triple (`spirv64-unknown-unknown`), chip, optLevel |

### How It Works: Implementation Detail

From `mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp`:

```cpp
// 1. Walk top-level op, find all gpu.module ops
for (Region &region : op->getRegions())
  for (Block &block : region.getBlocks())
    for (auto module : llvm::make_early_inc_range(block.getOps<GPUModuleOp>()))
      if (failed(moduleSerializer(module, handler, targetOptions)))
        return failure();

// 2. Per module: iterate every target attribute
for (auto targetAttr : op.getTargetsAttr()) {
  auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);
  auto serializedModule = target.serializeToObject(op, targetOptions);
  if (!serializedModule) {
    op.emitError("An error happened while serializing the module.");
    return failure();    // fail-fast: any target failure aborts
  }
  objects.push_back(target.createObject(op, *serializedModule, targetOptions));
}

// 3. Resolve handler: module's own handler takes precedence if none provided
if (auto moduleHandler = dyn_cast_or_null<OffloadingLLVMTranslationAttrInterface>(
    op.getOffloadingHandlerAttr()); !handler && moduleHandler)
  handler = moduleHandler;

// 4. Build gpu.binary op with all collected objects
gpu::BinaryOp::create(builder, op.getLoc(), op.getName(), handler,
    builder.getArrayAttr(objects));
op.erase();  // replace gpu.module with gpu.binary
```

**Key design decision**: Any single target failure aborts the entire pass.
There is no partial-success path.

### Multiple Targets: Example Transformation

```mlir
// Input: one gpu.module, three targets
gpu.module @kernels [
    #nvvm.target<chip = "sm_90">,
    #nvvm.target<chip = "sm_60">,
    #rocdl.target<chip = "gfx90a">
] {
  gpu.func @matmul(...) kernel { ... }
}

// Output after gpu-module-to-binary:
gpu.binary @kernels [
    #gpu.object<#nvvm.target<chip = "sm_90">,  bin = "...cubin-sm90-blob...">,
    #gpu.object<#nvvm.target<chip = "sm_60">,  bin = "...cubin-sm60-blob...">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, bin = "...hsaco-blob...">
]
```

The `gpu.module` is erased; the `gpu.binary` with all objects replaces it.

### Output Formats

| Format string | Meaning | Runtime behaviour |
|--------------|---------|-------------------|
| `offloading` / `llvm` | LLVM bitcode blob | Passed to driver/JIT for final compilation |
| `assembly` / `isa` | PTX / GCN assembly text | Null-terminated; driver JITs to native ISA at load time |
| `binary` / `bin` | Pre-compiled native binary | `mgpuModuleLoad` (cubin / hsaco) |
| `fatbinary` / `fatbin` | NVIDIA fat binary | Contains cubin + PTX; driver falls back to JIT if cubin version mismatches |

**Default is `fatbin`** — chosen because it enables architecture fall-through.

### NVVM Target Attribute Parameters

Constructed in `NVVMAttachTarget.cpp` via `builder.getAttr<NVVMTargetAttr>()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optLevel` | 2 | LLVM/PTXAS optimization level (0-3) |
| `triple` | `nvptx64-nvidia-cuda` | Target triple |
| `chip` | `sm_75` | Compute capability: sm_50 … sm_90a |
| `features` | `""` | PTX feature flags: `+ptx80`, etc. |
| flags dict: `fast` | false | Fast math |
| flags dict: `ftz` | false | Flush-to-zero |
| `filesToLink` | (empty) | Device bitcode libraries (libdevice.10.bc) |

### ROCDL Target Attribute Parameters

Constructed in `ROCDLAttachTarget.cpp`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optLevel` | 2 | Optimization level |
| `triple` | `amdgcn-amd-amdhsa` | Target triple |
| `chip` | `gfx900` | GPU architecture: gfx90a, gfx942, gfx1250, etc. |
| `features` | `""` | Feature flags |
| `abiVersion` | HIP default | ABI version |
| flags dict: `wave64` | false | Use 64-thread wavefronts |
| flags dict: `fast` | false | Fast math |
| flags dict: `daz` | false | Denormals-as-zero |
| flags dict: `finite_only` | false | Finite-only math |
| flags dict: `unsafe_math` | false | Unsafe math |
| `filesToLink` | (empty) | ROCm device bitcode libraries |

### SPIR-V Target Attribute Parameters

From `SPIRVAttachTarget.cpp` — encodes a full `spirv.target_env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spirvVersion` | `v1.0` | SPIR-V spec version |
| `clientApi` | `Unknown` | Vulkan / OpenCL |
| `deviceVendor` | `Unknown` | AMD / NVIDIA / Intel / etc. |
| `deviceType` | `Unknown` | DiscreteGPU / IntegratedGPU / CPU |
| `deviceId` | `kUnknownDeviceID` | PCI device ID |
| `spirvCapabilities` | `[]` | Shader, Kernel, Int64, Float16, etc. |
| `spirvExtensions` | `[]` | SPV_KHR_*, SPV_AMD_*, etc. |

The SPIR-V target is fundamentally different from NVVM/ROCDL: it does not go
through the LLVM backend at all. Compilation produces SPIR-V bytecode consumed
directly by a Vulkan or OpenCL driver.

---

## 2. `gpu.binary` Op and `gpu.select_object`

### `gpu.binary` Op Definition

From `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td`:

```tablegen
def GPU_BinaryOp : GPU_Op<"binary", [Symbol]>,
    Arguments<(ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<OffloadingTranslationAttr>:$offloadingHandler,
      ConfinedAttr<GPUObjectArrayAttr, [ArrayMinCount<1>]>:$objects
    )>
```

Three components:
1. **sym_name**: symbol name (referenced by `gpu.launch_func`)
2. **offloadingHandler**: optional attribute implementing `OffloadingTranslationAttrTrait`; defaults to `#gpu.select_object<0>` (first object)
3. **objects**: array of `#gpu.object<target_attr, binary_data>` — at least 1 required

### `#gpu.object` Attribute Format

```mlir
#gpu.object<#nvvm.target<chip = "sm_90">, offload = "...binary-blob...">
#gpu.object<#rocdl.target<chip = "gfx90a">, bin = "...binary-blob...">
```

The second field is keyed by the format type (`offload` for LLVM IR, `bin` for
pre-compiled binary, `isa` for assembly text).

### `#gpu.select_object`: The Default Offloading Handler

`SelectObjectAttr` is the *only* built-in offloading handler in upstream MLIR.

**Selection logic** (from `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp`):

```cpp
// getSelectedObject():
// 1. If handler has IntegerAttr index -> use it directly as array index
// 2. Otherwise, iterate objects, compare each object's target attr against
//    the handler's target attr until match is found
// 3. If handler has no target -> default to index 0
// 4. Out-of-bounds index -> error
```

Example uses:
```mlir
// Default: selects first object (index 0)
gpu.binary @kernels [#gpu.object<#nvvm.target, "...">, ...]

// Select by index (0-based integer)
gpu.binary @kernels <#gpu.select_object<1>> [obj0, obj1, obj2]

// Select by target attribute match
gpu.binary @kernels <#gpu.select_object<#rocdl.target<chip = "gfx90a">>> [
    #gpu.object<#nvvm.target, "...">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, "...">
]
```

### What `SelectObjectAttr::embedBinary` Produces

From `SelectObjectAttr.cpp` — the `embedBinary` method generates LLVM IR globals at
translation time:

```
1. @serializedObj  : global constant i8 array containing the chosen binary blob
   (null-terminated if format == assembly, for driver JIT)

2. @modulePtr      : global i8* initialized to null
   (will hold the loaded CUmodule/hipModule_t at runtime)

3. @loadFn         : constructor function (priority 123)
   - calls mgpuModuleLoadJIT(data, optLevel)  [if format == assembly]
   - OR calls mgpuModuleLoad(data, size)      [if format == binary]
   - stores result into @modulePtr
   - registered via llvm.global_ctors

4. @unloadFn       : destructor function (priority 123)
   - calls mgpuModuleUnload(@modulePtr)
   - registered via llvm.global_dtors
```

Runtime functions called by generated code:

| Function | Purpose | Wraps |
|----------|---------|-------|
| `mgpuModuleLoad(void*, size_t)` | Load pre-compiled binary | `cuModuleLoadData` / `hipModuleLoadData` |
| `mgpuModuleLoadJIT(void*, int)` | Load and JIT-compile assembly | `cuModuleLoadDataEx` with JIT options |
| `mgpuModuleUnload(CUmodule)` | Release loaded module | `cuModuleUnload` / `hipModuleUnload` |
| `mgpuModuleGetFunction(module, name)` | Get kernel handle | `cuModuleGetFunction` / `hipModuleGetFunction` |
| `mgpuLaunchKernel(...)` | Launch kernel | `cuLaunchKernel` / `hipModuleLaunchKernel` |
| `mgpuLaunchClusterKernel(...)` | Launch with cluster dims | `cuLaunchKernelEx` (Hopper) |

### Why `gpu.select_object` Is Compile-Time Only

The entire `embedBinary` method runs during MLIR-to-LLVM-IR translation, which
happens at compile time (or JIT time for the host). By the time `embedBinary`
is called, it must commit to a single binary blob to embed as a global string.
The choice is irreversible at that point.

There is no mechanism to:
- Embed multiple binary blobs with runtime switching logic
- Query the GPU vendor/device at IR-translation time
- Generate if-else dispatch code across vendor APIs (CUDA vs HIP vs Vulkan)

This is the **fundamental architectural gap** for cross-vendor runtime dispatch.

---

## 3. `GPUOffloadingLLVMTranslationAttrInterface`

### Interface Definition

From `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td`:

```tablegen
def OffloadingLLVMTranslationAttrInterface : AttrInterface<
    "OffloadingLLVMTranslationAttrInterface",
    [OffloadingTranslationAttrTrait]> {
  let methods = [
    InterfaceMethod<
      "Translates a gpu.binary Op into LLVM IR target-specific instructions.",
      "llvm::LogicalResult", "embedBinary",
      (ins "Operation*":$binaryOp,
           "llvm::IRBuilderBase&":$hostBuilder,
           "LLVM::ModuleTranslation&":$hostModuleTranslation)>,
    InterfaceMethod<
      "Translates a gpu.launch_func Op into LLVM IR instructions.",
      "llvm::LogicalResult", "launchKernel",
      (ins "Operation*":$launchFunc,
           "Operation*":$binaryOp,
           "llvm::IRBuilderBase&":$hostBuilder,
           "LLVM::ModuleTranslation&":$hostModuleTranslation)>
  ];
}
```

### Supporting Interfaces

**`GPUTargetAttrInterface`** — implemented by `#nvvm.target`, `#rocdl.target`, `#spirv.target_env`:

```tablegen
def GPUTargetAttrInterface : AttrInterface<"GPUTargetAttrInterface"> {
  let methods = [
    InterfaceMethod<"Serializes a GPU module to binary",
      "std::optional<SerializedObject>", "serializeToObject",
      (ins "Operation*":$module, "const TargetOptions&":$options)>,
    InterfaceMethod<"Creates gpu.object attribute from binary data",
      "Attribute", "createObject",
      (ins "Operation*":$module, "const SerializedObject&":$object,
           "const TargetOptions&":$options)>
  ];
}
```

**`TargetOptions`** — opaque configuration passed to `serializeToObject`:

| Method | Returns | Notes |
|--------|---------|-------|
| `getToolkitPath()` | `StringRef` | Path to CUDA toolkit / ROCm |
| `getLibrariesToLink()` | `ArrayAttr` | Device library bitcode files |
| `getCmdOptions()` | `StringRef` | Extra flags for ptxas / hipcc |
| `getELFSection()` | `StringRef` | ELF section for binary placement |
| `getCompilationTarget()` | `CompilationTarget` | offloading/isa/binary/fatbin enum |
| IR callbacks | `function_ref` | Inspect LLVM IR at various pipeline stages |

### What Would Need to Change for Runtime Selection

Currently `OffloadingLLVMTranslationAttrInterface` has two methods:
- `embedBinary`: called once per `gpu.binary` op, must commit to one binary
- `launchKernel`: called per `gpu.launch_func`, generates launch code

A **new attribute** implementing this interface could override `embedBinary` to:

1. Embed **all** binary blobs as separate globals (one per vendor/arch)
2. Generate a runtime detection function that queries:
   - `cuInit` / `cuDeviceGet` (NVIDIA presence)
   - `hipInit` / `hipGetDeviceCount` (AMD presence)
   - OpenCL / Vulkan device enumeration
3. Emit if-else dispatch: call `mgpuModuleLoad` for NVIDIA blob, or HIP equivalent for AMD blob
4. Store a function pointer for the appropriate kernel launcher

This would be a new attribute, e.g., `#gpu.runtime_select`, implementing the same
interface but emitting multi-path LLVM IR.

**Key constraint**: This requires linking against multiple GPU runtimes simultaneously,
which typically requires runtime dynamic library loading (dlopen), not static linking.

---

## 4. MLIR ExecutionEngine Limitations

### Architecture

`mlir::ExecutionEngine` wraps LLVM's `LLJIT` (ORC-based eager JIT):

```
MLIR module
    |
    v  (translateModuleToLLVMIR via registered translators)
LLVM IR module (with embedded GPU binary globals)
    |
    v  (TMOwningSimpleCompiler, target machine = host CPU)
Host machine code
    |
    v  (ORC RTDyldObjectLinkingLayer, runtime symbol resolution)
Executable in memory
```

Key `ExecutionEngineOptions`:

| Field | Type | Description |
|-------|------|-------------|
| `llvmModuleBuilder` | callback | Custom MLIR→LLVM translation |
| `transformer` | callback | LLVM IR optimization pass |
| `jitCodeGenOptLevel` | `CodeGenOptLevel` | Host code optimization |
| `sharedLibPaths` | `ArrayRef<StringRef>` | Runtime `.so` files to link (e.g., `libcuda.so`) |
| `enableObjectDump` | bool | Persist compiled objects |

### Target Triple Fixation

The target triple is taken directly from the `TargetMachine`, which defaults to
`detectHost()` — the machine running the compilation. This means:

- No cross-compilation from this API
- No multi-target host code generation
- Target is always fixed at `ExecutionEngine::create()` call time

### GPU Code Path in ExecutionEngine

GPU device code is **not JIT-compiled by ExecutionEngine**. The flow is:

1. `gpu-module-to-binary` (AoT pass) serializes device code to binary blobs
2. Blobs are embedded as global string constants in the LLVM IR host module
3. `ExecutionEngine` JIT-compiles **only the host** LLVM IR
4. The JIT-compiled host code, when run, calls `mgpuModuleLoad` to load the pre-compiled blobs
5. Runtime GPU JIT (e.g., NVIDIA PTX-to-cubin) may occur inside the driver, transparent to MLIR

**Consequence**: "JIT GPU kernel dispatch" in MLIR means JIT-compiling the host
dispatch code, not the kernel code. Device code is always AoT compiled.

### No GPU Support in ExecutionEngine Itself

The `ExecutionEngine.cpp` contains:
- AArch64-specific workaround for GOT/TEXT section spacing
- COFF binary format handling
- No architecture-specific GPU handling at all

GPU execution is entirely delegated to the runtime wrapper functions
(`mgpu*`) loaded via `sharedLibPaths`.

### Implications for Runtime Dispatch

Because ExecutionEngine is a CPU host JIT only, runtime dispatch must be
implemented as **CPU-side dispatch logic** that:
- Selects the appropriate pre-compiled binary at runtime
- Calls the appropriate vendor runtime API
- Is compiled into the JIT-compiled host code

This is exactly what a new `#gpu.runtime_select` offloading attribute
(implementing `OffloadingLLVMTranslationAttrInterface`) would generate.

---

## 5. The SPIR-V RFC: "SPIR-V IR as Vendor-Agnostic GPU Representation"

### RFC Details

- **URL**: https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- **Status**: Active RFC (as of early 2026); not merged
- **Proposers**: Intel/Khronos contributors

### What Is Proposed

Use SPIR-V IR as the primary intermediate representation for GPU code in LLVM,
analogous to how LLVM IR serves CPUs. The proposal would:

1. Establish SPIR-V as the canonical "GPU assembly" within the LLVM toolchain
2. Allow GPU code to be compiled to SPIR-V and distributed in SPIR-V form
3. Vendors would provide SPIR-V-to-native-ISA backends (NVIDIA already has one for OpenCL/Vulkan)
4. Eliminate the need for vendor-specific LLVM backends (NVPTX, AMDGPU) as primary paths

### Current SPIR-V Backend Status (LLVM 19+)

- **Stability improvements**: LLVM 2024-10 talk: "Advancing SPIR-V Backend Stability:
  Navigating GlobalISel Compromises" (Michal Paszkowski, Vyacheslav Levytskyy)
- **Generic MLIR-to-SPIR-V pass**: Merged in LLVM 19 (AMD contribution), providing
  broader coverage of `gpu.module` → `spirv.module` lowering
- **GlobalISel-based**: The SPIR-V backend uses Global Instruction Selection,
  unlike most other backends (SelectionDAG)
- **Client APIs supported**: OpenCL, SYCL/DPC++, and partial Vulkan support

### Community Reception

Key tensions in the RFC:

| Perspective | Position |
|-------------|----------|
| Intel / Khronos | Strong support; SPIR-V solves the "write once, run anywhere" GPU problem |
| NVIDIA camp | Skeptical; NVIDIA's SPIR-V-to-CUDA path adds latency; PTX ecosystem is mature |
| AMD | Partially supportive; ROCm has SPIR-V support but prefers native AMDGPU path for performance |
| MLIR maintainers | Open to discussion; concerned about duplication with existing NVPTX/AMDGPU backends |

The RFC remains in discussion without a clear resolution timeline.

### Why SPIR-V Alone Doesn't Solve Runtime Dispatch

Even if SPIR-V becomes the universal GPU IR:
- Vulkan/OpenCL runtime still needed at runtime (not CUDA/HIP)
- Different Vulkan features required per vendor (Intel, AMD, NVIDIA support different extensions)
- NVIDIA's Vulkan/SPIR-V performance still lags behind native CUDA on many kernels
- ML-critical operations (tensor cores, MFMA) require vendor-specific SPIR-V extensions

**Conclusion**: SPIR-V reduces the problem but does not eliminate runtime dispatch complexity.
A dispatch layer (like libkdl) remains necessary to select between SPIR-V-Vulkan,
native CUDA, and native HIP paths.

---

## 6. Recent MLIR GPU Dialect Developments (2025-2026)

### New Operations (2025-2026)

| Operation | Added | Description |
|-----------|-------|-------------|
| `gpu.ballot` | March 2026 | ArgMax/ArgMin-style reduction; ROCDL + NVVM + SPIR-V lowerings |
| Constant address space | April 2026 | `gpu.constant` address space across all GPU backends |
| `SymbolUserOpInterface` on `gpu.launch_func` | 2025 | Enables symbol-usage analysis passes |
| `ValueBoundsOpInterface` on `gpu.subgroup_broadcast` | 2025 | Affine analysis through subgroup ops |

### Pass Improvements (2025-2026)

| Pass | Change | Impact |
|------|--------|--------|
| `gpu-eliminate-barriers` | Made address-space-aware | Removes unnecessary workgroup barriers in global-memory-only loops |
| XeGPU pipeline | Replaced full canonicalization with targeted folding | Better Intel GPU compilation efficiency |
| SPIR-V backend | Stability improvements (GlobalISel) | More reliable OpenCL/SYCL/Vulkan codegen |

### New Targets

| Target | Added | Notes |
|--------|-------|-------|
| `XeVMAttachTarget` | 2024-2025 | Intel Xe GPU target (`spirv64-unknown-unknown`, chip `bmg`) |
| Intel XeGPU dialect | Upstream 2024 | Operations for 2D block load/store, DPAS (Intel MMA equivalent) |

### Active RFCs (2025)

| RFC | URL | Status | Relevance |
|-----|-----|--------|-----------|
| SPIR-V as vendor-agnostic GPU IR | .../85115 | Active discussion | High: vendor-agnostic compilation |
| Distributed heterogeneous computing dialect | .../86960 | Active RFC | Medium: targets distributed/multi-device, not runtime dispatch |
| Cleaning the GPU dialect | .../88170 | Active RFC | Low: refactoring, removes vendor-specific ops from generic dialect |
| XeVM dialect | .../86955 | Accepted | Medium: Intel third vendor, increases multi-target complexity |
| MLIR GPU codegen pipeline extension | .../70199 | Older RFC | High: device linking gap directly limits runtime compilation |

### Asymmetry Between NVIDIA and AMD Support

As of 2026, a significant asymmetry persists:

- **NVIDIA**: Full consolidated pipeline `gpu-lower-to-nvvm-pipeline`; single command to compile
- **AMD**: No equivalent `gpu-lower-to-rocdl-pipeline`; must manually compose passes
- **Intel XeGPU**: Dialect exists upstream; complete lowering pipeline absent

This asymmetry makes multi-vendor support harder to build on top of raw MLIR.

---

## 7. The Full Compilation Pipeline (End-to-End)

### NVIDIA Path

```
linalg.matmul / linalg.generic
    |
    v  convert-linalg-to-affine-loops
affine.for / scf.for
    |
    v  convert-affine-for-to-gpu
gpu.launch { ... }
    |
    v  gpu-kernel-outlining
gpu.launch_func @module::@kernel
    +---> gpu.module @module { gpu.func @kernel { ... } }
              |
              v  convert-gpu-to-nvvm
           nvvm IR (in gpu.module)
              |
              v  nvvm-attach-target{chip=sm_90 features=+ptx80 O=3}
           gpu.module [#nvvm.target<chip="sm_90">] { ... }
    |
    v  gpu-module-to-binary (format=fatbin)
gpu.binary @module [#gpu.object<#nvvm.target<chip="sm_90">, "fatbin-blob">]
    |
    v  gpu-to-llvm
Host LLVM dialect with runtime API calls:
  - mgpuModuleLoad(fatbin_blob) -> modulePtr
  - mgpuModuleGetFunction(modulePtr, "kernel") -> funcPtr
  - mgpuLaunchKernel(funcPtr, grid, block, smem, stream, args)
    |
    v  translate-to-llvm-ir
LLVM IR -> machine code (via ExecutionEngine JIT or mlir-cpu-runner)
```

### AMD Path (Manual Composition Required)

```
gpu.module @module { gpu.func @kernel { ... } }
    |
    v  rocdl-attach-target{chip=gfx90a O=3}
    v  (module nested) convert-gpu-to-rocdl
    v  gpu-to-llvm
    v  gpu-module-to-binary (format=binary)
    v  translate-to-llvm-ir
```

No `gpu-lower-to-rocdl-pipeline` equivalent exists.

### SPIR-V Path (Does Not Use LLVM Backend)

```
gpu.module { gpu.func { ... } }
    |
    v  spirv-attach-target{spirvVersion=v1.3 clientApi=Vulkan}
    v  convert-gpu-to-spirv
spirv.module { spirv.func { ... } }
    |
    v  spirv serialization (spirv::serialize())
SPIR-V binary (.spv bytecode)
    |
    v  loaded at runtime by Vulkan/OpenCL driver
    v  driver compiles SPIR-V to native ISA
Native GPU ISA
```

---

## 8. Implications for libkdl

### Where libkdl Operates

```
MLIR compilation time:
  gpu-module-to-binary produces gpu.binary with [nvvm_obj, rocdl_obj, spirv_obj]
  gpu.select_object picks ONE at LLVM-IR translation -> hardcoded into binary

libkdl operates AFTER this:
  Runtime dispatch layer on pre-compiled gpu.binary contents
  Detects: NVIDIA present? AMD present? Only CPU?
  Selects: appropriate blob from the multi-object gpu.binary
  Loads:   via appropriate runtime API (CUDA / HIP / Vulkan / CPU)
  Launches: via uniform kernel invocation interface
```

### The Specific Gap libkdl Fills

MLIR already provides:
- Multi-target compilation (one `gpu.module` -> many `gpu.object` entries in `gpu.binary`)
- Serialization pipeline per vendor (`serializeToObject` per target)
- Binary embedding infrastructure (`OffloadingLLVMTranslationAttrInterface`)

MLIR does NOT provide:
- Hardware detection at runtime (which vendor/device is present)
- Dynamic binary selection from multi-object `gpu.binary`
- Cross-vendor runtime API abstraction (CUDA vs HIP vs Vulkan)
- CPU fallback from failed GPU dispatch
- Persistent module caching (avoids per-launch `mgpuModuleLoad`)

### libkdl as a New Offloading Attribute (Future Direction)

A `#gpu.runtime_select` attribute implementing `OffloadingLLVMTranslationAttrInterface`
could integrate libkdl's dispatch logic into the MLIR compilation pipeline:

```cpp
class RuntimeSelectAttrImpl : public OffloadingLLVMTranslationAttrInterface::Concept {
  LogicalResult embedBinary(Operation* binaryOp, IRBuilderBase& builder,
                            ModuleTranslation& modTrans) override {
    // 1. Embed ALL objects as separate LLVM globals (not just one)
    // 2. Generate runtime vendor detection code (libkdl_detect())
    // 3. Emit dispatch table: vendor_id -> (binary_ptr, runtime_load_fn)
    // 4. Generate wrapper that calls appropriate load function at runtime
  }
  LogicalResult launchKernel(Operation* launchFunc, Operation* binaryOp,
                             IRBuilderBase& builder,
                             ModuleTranslation& modTrans) override {
    // Emit indirect call through dispatch table
    // -> cuda path or hip path or vulkan path or cpu path
  }
};
```

This is a concrete upstream contribution target for the libkdl project.

---

## 9. Summary of Findings

### Finding 1: gpu-module-to-binary Is Already Multi-Target

The pass fully supports N targets → N objects in a single `gpu.binary`. The
multi-target infrastructure is complete at the compilation layer.

**Evidence**: `ModuleToBinary.cpp` target iteration loop; test files
`module-to-binary-nvvm.mlir` and `module-to-binary-rocdl.mlir`.

### Finding 2: gpu.select_object Is Compile-Time Only — By Design

`SelectObjectAttr::embedBinary` is called at LLVM-IR translation time. It commits
to a single binary blob that is embedded as a compile-time global constant. No
runtime selection mechanism exists in the current implementation.

**Evidence**: `SelectObjectAttr.cpp` implementation; global variable creation for
`serializedObj` and `modulePtr` at translation time.

### Finding 3: OffloadingLLVMTranslationAttrInterface Is the Extension Point

The two-method interface (`embedBinary`, `launchKernel`) is the correct hook for
implementing runtime dispatch. Any attribute implementing this interface with a
new strategy (embedding multiple blobs, emitting dispatch code) would be a valid
upstream contribution.

**Evidence**: `CompilationAttrInterfaces.td` interface definition.

### Finding 4: ExecutionEngine Has No GPU Awareness

ExecutionEngine JITs only host (CPU) code. GPU device code must be pre-compiled
into binary blobs. The "runtime" in "runtime dispatch" refers to CPU-side dispatch
logic selecting among pre-compiled GPU binaries.

**Evidence**: `ExecutionEngine.cpp` target triple detection (`detectHost()`), no
GPU-specific handling.

### Finding 5: SPIR-V RFC Is Active But Unresolved

The RFC proposes SPIR-V as universal GPU IR but faces resistance from NVIDIA and
performance concerns. Even if accepted, runtime dispatch logic would still be needed
because vendor-specific extensions and performance differences remain.

**Evidence**: RFC URL .../85115; LLVM 2024 Dev Meeting SPIR-V stability talk.

### Finding 6: Intel XeVM/XeGPU Adds Urgency to Multi-Vendor Dispatch

With Intel Xe GPU support now upstream (XeGPU dialect, XeVMAttachTarget pass,
`bmg` chip default), the ecosystem now has three major GPU vendors in MLIR:
NVIDIA (nvvm), AMD (rocdl), Intel (xevm/xegpu). The multi-target compilation
story becomes more pressing as Intel GPU hardware proliferates in datacenters.

**Evidence**: `XeVMAttachTarget.cpp`; XeGPU dialect documentation.

---

## 10. Relevance Scores

| Topic | Relevance to libkdl poster | Score (1-5) |
|-------|---------------------------|-------------|
| `gpu-module-to-binary` pass mechanics | Direct prerequisite understanding | 5 |
| `gpu.binary` op format | Data structure libkdl reads at runtime | 5 |
| `gpu.select_object` limitation | The exact gap libkdl fills | 5 |
| `OffloadingLLVMTranslationAttrInterface` | Extension point for libkdl upstream integration | 5 |
| ExecutionEngine GPU limitations | Confirms device-code-is-AoT assumption | 4 |
| SPIR-V RFC | Long-term direction; doesn't change current reality | 3 |
| XeVM/XeGPU dialect | Increases multi-vendor urgency | 3 |
| NVVM/ROCDL/SPIRV target attribute parameters | Needed for test harness design | 4 |
| AMD pipeline asymmetry | Highlights manual composition requirement | 3 |
| Barrier/reduction improvements 2025-2026 | Not directly relevant | 1 |

---

## References

- MLIR GPU Dialect documentation: https://mlir.llvm.org/docs/Dialects/GPU/
- `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td` — pass definitions
- `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td` — interface definitions
- `mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp` — pass implementation
- `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp` — select_object implementation
- `mlir/lib/ExecutionEngine/ExecutionEngine.cpp` — host JIT, no GPU support
- `mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp` — CUDA runtime API wrappers
- `mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp` — ROCm runtime API wrappers
- RFC: SPIR-V as vendor-agnostic GPU IR: https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- RFC: Distributed heterogeneous computing dialect: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- RFC: Cleaning the GPU dialect: https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- Discourse: GPU execution without runtime load/unload: https://discourse.llvm.org/t/mlir-gpu-execution-without-runtime-load-unload/61712
- Discourse: How to generate AMDGPU code from MLIR: https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627
- Existing local analysis: `/home/akash/PROJECTS/LLVM/literature/mlir-jit-analysis.md`
