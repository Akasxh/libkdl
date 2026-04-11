# Wave 06 — Level Zero Runtime: DDI Architecture, Mutable Dispatch, Multi-Device, and IMEX Integration

**Angle:** level-zero-oneapi-runtime
**Query:** Level Zero API GPU kernel dispatch zeModuleCreate zeKernelCreate Level Zero multi-device Intel multi-device
**Date:** 2026-04-06
**Note:** `wave-04-level-zero.md` covered Level Zero's basic dispatch pipeline and UR adapter model in depth. This report focuses on distinct angles: the DDI loader architecture, Mutable Command List extension, multi-tile/multi-card dispatch mechanics, Level Zero SPIR-V capability surface, and IMEX (Intel MLIR Extensions) integration.

---

## Source Index

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | Level Zero Loader Architecture — DeepWiki | https://deepwiki.com/oneapi-src/level-zero/1-level-zero-overview | 2025 | Architecture Reference | 10/10 |
| S2 | Mutable Command List Extension — Level Zero Spec v1.11 | https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_MutableCommandList.html | Current | Spec | 10/10 |
| S3 | Multi-Tile and Multi-Card with Level Zero — intel/llvm | https://intel.github.io/llvm/MultiTileCardWithLevelZero.html | Current | Docs | 9/10 |
| S4 | SPIR-V Programming Guide — Level Zero Spec v1.11 | https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/SPIRV.html | Current | Spec | 9/10 |
| S5 | Intel MLIR Extensions (IMEX) — GitHub | https://github.com/intel/mlir-extensions | Active (2024–2025) | Project | 8/10 |
| S6 | Level Zero Immediate Command Lists — Intel Developer Guide | https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html | 2023–2025 | Docs | 8/10 |
| S7 | Considerations for Multi-Tile/Card — oneAPI DPC++ Compiler Docs | https://intel.github.io/llvm/MultiTileCardWithLevelZero.html | Current | Docs | 8/10 |
| S8 | Level Zero Immediate Command List Append Extension — Spec v1.11 | https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_ImmediateCommandListAppend.html | Current | Spec | 7/10 |
| S9 | Level Zero — oneAPI GPU Optimization Guide 2025 | https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/level-zero.html | 2025 | Docs | 7/10 |
| S10 | OpenCL 3.0.12 Mutable Dispatch Extension — Khronos | https://www.khronos.org/blog/opencl-3.0.12-released-with-command-buffers-mutable-dispatch-extension-and-enhanced-layers-support | 2024 | Blog/Announcement | 6/10 |

---

## Source Summaries

### S1 — Level Zero Loader DDI Architecture — DeepWiki [10/10]

**URL:** https://deepwiki.com/oneapi-src/level-zero/1-level-zero-overview
**Type:** Architecture reference synthesized from source
**Date:** 2025

The Level Zero loader uses a dual-component design:

- **Static loader (`ze_loader.a`)**: Links directly into the application, provides API entry points, can optionally dlopen the dynamic loader.
- **Dynamic loader (`ze_loader.so`)**: System-wide shared object containing driver management, DDI dispatch, handle routing.

**DDI (Device Driver Interface) table mechanism:** The loader builds a hierarchy of function-pointer tables at initialization — `ze_global_dditable_t`, `ze_driver_dditable_t`, device/context/memory tables. Each discovered driver populates its DDI tables via `zeGetGlobalProcAddrTable()` and friends. API calls route through loader wrapper → DDI table lookup → driver function pointer. This is zero-overhead after initialization: all paths are pointer dereferences, no string dispatch or hash lookups.

**Driver discovery:**
- Linux: scans `/usr/lib`, `/usr/local/lib`, `LD_LIBRARY_PATH` for `ze_intel_*` pattern libraries.
- Windows: queries `HKLM\SOFTWARE\Intel\IGFX\ZE` registry and display adapter info.
- Override: `ZEL_ALT_DRIVERS` env var for non-standard driver locations.

**Multi-driver/multi-device:** Handle factories (`ze_device_factory_t`) map loader-level handles to driver-specific handles transparently. Driver priority order: discrete GPU > integrated GPU > NPU (overridable via `ZEL_DRIVERS_ORDER` or `ZE_ENABLE_PCI_ID_DEVICE_ORDER`). Heterogeneous multi-driver scenarios (e.g., Intel dGPU + Intel iGPU) handled by handle translation functions `zerTranslateDeviceHandleToIdentifier` / `zerTranslateIdentifierToDeviceHandle`.

**Optimization: DDI Driver Extension Path.** When drivers declare `ZE_DRIVER_DDI_HANDLES_EXT`, the loader bypasses handle wrapping entirely, reducing multi-driver overhead to near-zero — the driver's native handles are used directly.

**Relevance to libkdl:** The DDI table pattern is directly applicable. libkdl's backend plugin dispatch could adopt an identical function-pointer table loaded at `dlopen` time, achieving the same zero-overhead routing as the Level Zero loader itself. The `ZEL_ALT_DRIVERS` pattern maps to libkdl's `KDL_BACKEND_PATH`.

---

### S2 — Mutable Command List Extension — Level Zero Spec v1.11 [10/10]

**URL:** https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_MutableCommandList.html
**Type:** Official specification, experimental extension
**Date:** Current (spec v1.11, production use in 2024+)

The Mutable Command List (MCL) extension enables modifying a closed command list between executions without rebuilding it — a critical optimization for iterative ML inference loops.

**Mutable parameters per command:**
- Kernel arguments (individual parameter values)
- Group count (workgroup grid dimensions X/Y/Z)
- Group size (local workgroup dimensions)
- Global offset (dispatch offset)
- Signal/wait events (synchronization primitives)
- Kernel instructions (swap to a different pre-registered kernel implementation)

**API flow:**
```c
// At command list build time:
ze_mutable_command_id_exp_desc_t cmdIdDesc = {
    .flags = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
             ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT
};
uint64_t cmdId;
zeCommandListGetNextCommandIdExp(cmdList, &cmdIdDesc, &cmdId);
zeCommandListAppendLaunchKernel(cmdList, kernel, &groupCount, ...);
zeCommandListClose(cmdList);

// At update time (after synchronization):
ze_mutable_kernel_argument_exp_t argMutation = {
    .commandId = cmdId,
    .argIndex = 0,
    .argSize = sizeof(ptr),
    .pArgValue = &newBufPtr
};
ze_mutable_commands_exp_desc_t mutDesc = { .flags = 0 };
zeCommandListUpdateMutableCommandsExp(cmdList, &mutDesc);
// Re-submit without rebuild:
zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
```

**Kernel swapping via `zeCommandListGetNextCommandIdWithKernelsExp`:** Pre-register multiple kernel alternatives at ID assignment time, then switch between them at update time via `zeCommandListUpdateMutableCommandKernelsExp`. Enables operator dispatch (e.g., select between FP16/FP32 kernel variant) without command list rebuild.

**Constraint:** Mutation flags must be declared at ID assignment — the driver allocates resources for exactly those mutations. This is analogous to Vulkan's `VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT` pattern: opt-in pre-declaration enables driver optimization.

**Relevance to libkdl:** MCL is libkdl's primary mechanism for iterative inference dispatch on Intel GPUs. The operator-dispatch pattern — build once, update arguments per inference step, re-submit — matches exactly the ML inference hot path. The kernel-swapping capability enables libkdl's multi-versioned kernel selection (e.g., pick tiled-GEMM-v1 vs tiled-GEMM-v2) without rebuild overhead.

---

### S3 — Multi-Tile and Multi-Card Dispatch — intel/llvm [9/10]

**URL:** https://intel.github.io/llvm/MultiTileCardWithLevelZero.html
**Type:** DPC++ compiler documentation
**Date:** Current

**Sub-device partitioning:** Root devices (e.g., Intel Data Center GPU Max with 2 tiles) expose tiles as sub-devices via `zeDeviceGetSubDevices`. Partitioning uses `ZE_DEVICE_PARTITION_BY_AFFINITY_DOMAIN` exclusively — no count-based or equal-size partitioning supported for Intel GPUs.

**Four dispatch strategies (performance-ranked):**

| Strategy | Mechanism | Performance | Use Case |
|----------|-----------|-------------|----------|
| Explicit scaling (per-tile queue) | Separate queue per sub-device in shared context | Best per-tile | Custom load balancing |
| Implicit scaling (root-device queue) | Single queue to root-device, driver distributes | Good, less control | Uniform workloads |
| Implicit scaling (affinity mask) | `ZE_AFFINITY_MASK` selects sub-device at process level | Good | Process-level isolation |
| Multi-card (separate contexts) | Separate root-device contexts, data via host memory | Lowest (PCIe transfers) | True multi-GPU |

**Memory placement matters:** Device-allocated memory (`zeMemAllocDevice`) stays on the target tile and is fastest for kernel execution. Host-allocated memory (`zeMemAllocHost`) incurs PCIe transfers. For sub-devices in the same root-device, device memory is shareable within the same context — no copy needed for cross-tile kernel inputs.

**`ZE_AFFINITY_MASK` env var:** Controls which sub-devices the Level Zero UMD exposes at the process level. E.g., `ZE_AFFINITY_MASK=0` exposes only tile 0. This is the coarse-grained isolation mechanism — no code changes needed for tile selection.

**Relevance to libkdl:** libkdl's multi-device dispatch must handle these four strategies. The explicit-scaling (per-tile queue) approach is the right model for libkdl's heterogeneous dispatch — each "device slot" in libkdl maps to a Level Zero sub-device with its own command list. The `ZE_AFFINITY_MASK` pattern suggests libkdl should expose a similar process-level device filter environment variable.

---

### S4 — Level Zero SPIR-V Capability Surface [9/10]

**URL:** https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/SPIRV.html
**Type:** Official spec
**Date:** Current (spec v1.11)

Level Zero imposes specific SPIR-V constraints that differ from generic Vulkan compute SPIR-V:

**Required execution model:** `Kernel` (not `GLCompute` as in Vulkan/OpenGL). This means Level Zero SPIR-V modules use OpenCL-derived kernel semantics.

**Required memory model:** `OpenCL` (not `Vulkan`). SPIR-V for Level Zero must be compiled with OpenCL memory semantics — important for cross-compilation from CUDA/HIP via chipStar or similar.

**Required addressing model:** `Physical64` — 64-bit device pointers mandatory. 32-bit addressing not supported.

**Recursion:** Explicitly prohibited. Kernel IR must be acyclic in the call graph.

**Required core capabilities:**
- `Addresses`, `Float16Buffer`, `Int64`, `Int16`, `Int8`, `Kernel`, `Linkage`, `Vector16`, `GenericPointer`, `Groups`
- Optional (device-dependent): `Float16`, `Float64`, `Int64Atomics`

**Intel-specific extensions via `OpExtension`:**
- `SPV_INTEL_subgroups` — Intel subgroup operations (warp-like)
- `SPV_INTEL_float_controls2` — fine-grained FP rounding/exception control
- `SPV_KHR_linkonce_odr` — link-once-ODR linkage for C++ templates across modules
- `SPV_INTEL_function_pointers` — function pointer support on GPU

**`ZE_extension_linkonce_odr`:** When declared, environments accept `SPV_KHR_linkonce_odr`, enabling the `LinkOnceODR` linkage type — essential for C++ template instantiations split across separately compiled modules.

**Relevance to libkdl:** The `Kernel` execution model + `OpenCL` memory model requirement means libkdl's Intel SPIR-V kernels must be compiled via a SPIR-V path that targets OpenCL semantics (Clang with SPIR target, or SPIRV-LLVM-Translator). Vulkan-targeted SPIR-V (from e.g. glslang) is not compatible. The `SPV_INTEL_subgroups` extension is required for efficient GEMM/convolution kernels on Intel hardware.

---

### S5 — Intel MLIR Extensions (IMEX) — XeGPU/XeTile Dialects [8/10]

**URL:** https://github.com/intel/mlir-extensions
**Type:** Open-source project (Intel, actively maintained)
**Date:** Active development 2024–2025 (148 stars, 43 forks)

IMEX is Intel's staging ground for MLIR dialects targeting Intel silicon, with GPU focus via XeGPU and XeTile dialects.

**Dialect stack:**
- **XeTile:** Tile-based programming model, decomposes GEMM kernels to large pre-defined tile sizes at subgroup/workgroup level. Targets the "GEMM at MLIR level" use case.
- **XeGPU:** Models Xe hardware instructions directly — DPAS (dot-product accumulate systolic), 2D block load/store. Closest-to-metal MLIR abstraction for Intel GPUs.
- **NDArray, Dist:** Experimental dialects for array and distributed computing abstractions.

**Lowering pipeline:**
```
Linalg / affine / SCF
    ↓ (xetile-pipeline)
XeTile ops
    ↓ (xegpu-pipeline)
XeGPU ops (DPAS, 2D block load)
    ↓
SPIR-V dialect
    ↓ (spirv-serialize)
SPIR-V binary
    ↓
zeModuleCreate() → Intel GEN ISA
```

**Two-step GPU execution (mlir-runner path):**
1. `mlir-opt --pass-pipeline="gpu-lower-to-xevm-pipeline"` lowers to LLVMIR/SPIR-V
2. `mlir-runner --shared-libs=libmlir_levelzero_runtime.so` dispatches via Level Zero

**Level Zero runtime wrapper:** IMEX ships `libmlir_levelzero_runtime.so` — a thin C shared library that wraps Level Zero calls (zeModuleCreate, zeKernelCreate, zeCommandListAppendLaunchKernel) for use as an MLIR JIT runner backend. This is the `gpu-runner` pattern adapted for Level Zero.

**Build requirement:** `"-DLLVM_TARGETS_TO_BUILD=X86;SPIRV"` — the SPIRV LLVM target must be enabled for the pipeline to function.

**Relevance to libkdl:** IMEX's `libmlir_levelzero_runtime.so` is structurally identical to what libkdl's Intel backend implements. IMEX's XeGPU → SPIR-V → Level Zero pipeline is the MLIR-native route for generating kernels that libkdl would dispatch. libkdl could integrate with IMEX by consuming its SPIR-V output and handling the dispatch layer.

---

### S6 — Immediate Command Lists: Submission Mode Architecture [8/10]

**URL:** https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html

Key addendum to Wave-04 findings: the **L0 v2 adapter (2025.3)** moves to immediate-in-order-only mode. This makes immediate command lists the de facto standard path for all Xe2+ hardware. The `zeCommandListCreateImmediate` call creates a combined queue+list where each `zeCommandListAppendLaunchKernel` immediately submits to device hardware — no explicit `zeCommandQueueExecuteCommandLists` call needed.

The **Immediate Command List Append Extension** (`ZE_extension_immediate_command_list_append`) in spec v1.11 formalizes the ability to append from multiple threads concurrently to an immediate command list — enabling parallel kernel submission without queue-level serialization.

---

### S7 — OpenCL 3.0.12 Mutable Dispatch — Khronos [6/10]

**URL:** https://www.khronos.org/blog/opencl-3.0.12-released-with-command-buffers-mutable-dispatch-extension-and-enhanced-layers-support
**Type:** Khronos announcement
**Date:** 2024

OpenCL adopted a parallel mutable dispatch concept in `cl_khr_command_buffer_mutable_dispatch`. The extension enables updating kernel arguments, workgroup dimensions, and global offsets in a recorded command buffer without reconstruction. This mirrors Level Zero's MCL extension exactly, suggesting cross-API convergence on mutable command patterns — driven by the same ML inference loop requirement.

**Significance:** Both Level Zero and OpenCL now have mutable dispatch APIs, indicating this is an emerging standard pattern for iterative ML inference dispatch, not an Intel-specific feature. CUDA's equivalent is the CUDA Graph update API (`cudaGraphExecKernelNodeSetParams`). All three major dispatch APIs converged on the same pattern.

---

## Synthesis

### Architectural Cross-Section: What Makes Level Zero Distinct

Level Zero's architecture relative to CUDA and HIP can be characterized across three axes:

**1. Driver Interface: DDI Tables vs. CUDA's monolithic driver**

Level Zero uses a loader + DDI function-pointer tables that are built at init time and used for zero-overhead dispatch. This is explicitly multi-driver capable — discrete GPU + iGPU + NPU can coexist with transparent handle routing. CUDA's driver API is monolithic and single-vendor. HIP mirrors CUDA's model. Level Zero's multi-driver architecture is unique and directly maps to libkdl's plugin model.

**2. SPIR-V as first-class IR, not a compatibility layer**

Unlike ROCm/HIP's HSACO or CUDA's PTX/CUBIN, SPIR-V is Level Zero's *primary* binary format (`ZE_MODULE_FORMAT_IL_SPIRV`). The `ZE_MODULE_FORMAT_NATIVE` format is the post-JIT output for caching. The direction is SPIR-V → native, not native-is-first-class. This makes Level Zero the only major GPU runtime where SPIR-V holds first-class status, with robust required capabilities (`Kernel`, `OpenCL` memory model, `Physical64`).

**3. Mutable Command Lists: structural parity with CUDA Graphs**

Level Zero's MCL extension provides the same capabilities as CUDA Graphs (argument update, kernel swap, dispatch dimension change without rebuild). All three ecosystems (CUDA Graphs, Level Zero MCL, OpenCL command buffer mutable dispatch) converged on this pattern, confirming it as the standard solution for low-overhead iterative dispatch in ML inference.

### Multi-Device Dispatch Decision Matrix

```
Q: Single Intel GPU with multiple tiles?
   → Use explicit scaling: one sub-device handle + queue per tile
   → zeDeviceGetSubDevices() → array of ze_device_handle_t
   → Separate zeCommandListCreate per sub-device
   → Manual work partitioning; driver does NOT auto-distribute in explicit mode

Q: Multiple Intel discrete GPUs (separate PCIe cards)?
   → Separate root-device contexts per card
   → Data exchange via host pinned memory (PCIe bandwidth bound)
   → Consider NVLink equivalent? Not available for Intel dGPU; rely on explicit copies

Q: Intel dGPU + Intel iGPU (heterogeneous Intel mix)?
   → Two drivers in same process; loader's DDI handles routing transparently
   → ZEL_DRIVERS_ORDER controls which runs first
   → libkdl creates two backend instances, one per driver
```

### IMEX Pipeline: MLIR-Native Intel GPU Kernel Generation

The IMEX XeGPU → SPIR-V → Level Zero pipeline is the MLIR-native route for Intel GPU ML kernels:

```
MLIR (linalg.matmul)
    ↓ xetile-pipeline
XeTile (32x32 tile, GEMM decomposed)
    ↓ xegpu-pipeline
XeGPU (dpas instruction, 2D block.load)
    ↓ spirv-serialize
.spv binary
    ↓ libkdl Intel backend
zeModuleCreate() → GEN ISA
zeKernelCreate("matmul")
zeCommandListAppendLaunchKernel()
```

This is the highest-performance ML kernel path for Intel GPUs via MLIR. Triton-Intel (via SPIRV) is an alternative route using similar lowering.

### Gaps and Risks Specific to This Wave

1. **MCL extension is experimental.** The `EXT_Exp_` prefix means the API can change. Production adoption requires waiting for promotion to core spec — no confirmed timeline found.

2. **SPIR-V execution model lock-in.** The `Kernel` + `OpenCL` memory model requirement means Level Zero SPIR-V is not interchangeable with Vulkan SPIR-V (`GLCompute` + `Vulkan` model). Cross-ecosystem portability requires two distinct SPIR-V compilation paths.

3. **IMEX is pre-upstream.** IMEX dialects (XeGPU, XeTile) are not yet in mainline MLIR/LLVM. Any libkdl integration with IMEX carries a maintenance burden until upstream.

4. **No public dispatch latency numbers.** Despite the L0 v2 adapter's "significantly reduces host runtime overhead" claim, no specific microsecond measurements are publicly available for Intel-vs-NVIDIA-vs-AMD dispatch latency comparison. This is a measurable gap libkdl's benchmarks could fill.

5. **Sub-device partitioning is affinity-only.** No count-based or uniform partitioning for Intel GPUs — limits automatic work partitioning strategies.

---

## Angle Assessment

**Relevance to heterogeneous GPU kernel dispatch in LLVM/MLIR:** 9/10

Level Zero is the foundational Intel GPU dispatch API. Its DDI loader architecture, MCL extension, and SPIR-V-first design are directly applicable to libkdl's Intel backend. The convergence of MCL/CUDA-Graphs/OpenCL-mutable-dispatch confirms mutable dispatch as a cross-vendor standard pattern.

**Novelty vs. Wave-04 coverage:** 7/10

Wave-04 covered the basic dispatch pipeline and UR adapter model well. This wave adds unique material on: DDI table architecture (loader internals), Mutable Command List extension (with API detail), multi-tile/card dispatch strategies (4-strategy matrix), SPIR-V capability surface (execution model constraints), and IMEX (XeGPU/XeTile MLIR pipeline). The MCL extension and cross-API convergence finding are new.

**Key actionable finding for libkdl:**

The Mutable Command List extension + `zeModuleGetNativeBinary` AOT caching together enable the full "build once, dispatch many times" pattern libkdl needs for Intel GPUs. The DDI table pattern should be adopted verbatim for libkdl's plugin dispatch table. The IMEX `libmlir_levelzero_runtime.so` is the closest existing analog to libkdl's Intel backend and should be studied as a reference implementation.

---

## Sources

- [Level Zero Loader Architecture — DeepWiki](https://deepwiki.com/oneapi-src/level-zero/1-level-zero-overview)
- [Mutable Command List Extension — Level Zero Spec v1.11](https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_MutableCommandList.html)
- [Multi-Tile and Multi-Card with Level Zero — intel/llvm](https://intel.github.io/llvm/MultiTileCardWithLevelZero.html)
- [SPIR-V Programming Guide — Level Zero Spec v1.11](https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/SPIRV.html)
- [Intel MLIR Extensions (IMEX)](https://github.com/intel/mlir-extensions)
- [Level Zero Immediate Command Lists — Intel](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)
- [Level Zero — oneAPI GPU Optimization Guide 2025](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/level-zero.html)
- [Immediate Command List Append Extension — Spec v1.11](https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_ImmediateCommandListAppend.html)
- [OpenCL 3.0.12 Mutable Dispatch — Khronos](https://www.khronos.org/blog/opencl-3.0.12-released-with-command-buffers-mutable-dispatch-extension-and-enhanced-layers-support)
- [Get Started Using Level Zero API Backend — Intel](https://www.intel.com/content/www/us/en/developer/articles/technical/zero-in-on-level-zero-oneapi-open-backend-approach.html)
