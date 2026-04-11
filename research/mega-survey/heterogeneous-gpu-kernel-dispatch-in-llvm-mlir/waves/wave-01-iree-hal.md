# Wave 01: IREE HAL Multi-Backend Dispatch
Search query: IREE HAL hardware abstraction layer multi-backend dispatch CUDA Vulkan ROCm
Sources found: 9
Date: 2026-04-06

## Sources

### 1. IREE HAL Dialect Reference — Official MLIR Dialect Docs
- URL: https://iree.dev/reference/mlir-dialects/HAL/
- Type: docs
- Date: 2024 (continuously updated)
- Relevance: 10/10
- Novelty: 7/10
- Summary: Canonical reference for the HAL MLIR dialect. The dialect is described as "a Vulkan-like model with all of the graphics bits chopped out," exposing buffers, semaphores, command buffers, and dispatch ops. Push constants are 4-byte opaque values; dispatch uses 3D workgroup counts following Vulkan conventions.
- Key detail: `hal.command_buffer.dispatch` accepts a deferred workgroup count read from a buffer at a given offset as three uint32 XYZ values — enabling runtime-computed dispatch sizes without host round-trips.

### 2. IREE CUDA HAL Driver Design Doc
- URL: https://iree.dev/developers/design-docs/cuda-hal-driver/
- Type: docs
- Date: 2023–2024
- Relevance: 9/10
- Novelty: 7/10
- Summary: Details how the CUDA HAL driver maps IREE's HAL API onto CUDA driver API (not runtime API) to reduce dependency surface and stay close to hardware. Kernels are stored as PTX in a FlatBuffer alongside entry-point metadata; CUDA JIT compiles PTX to native ISA at first load and caches it for process lifetime. Two command buffer implementations exist: CUDA Graphs (default, maps naturally to HAL command buffers) and CUDA Streams (retained for comparison).
- Key detail: PTX is the serialization format — the CUDA driver handles last-mile ISA compilation. CUDA Graphs (`cuGraphLaunchKernel`) provide lower per-launch overhead vs. per-kernel `cuLaunchKernel` on CUDA Streams. One-shot command buffers may incur CUDA Graph construction overhead.

### 3. IREE HIP HAL Driver Design Doc (ROCm)
- URL: https://iree.dev/developers/design-docs/hip-hal-driver/
- Type: docs
- Date: 2024
- Relevance: 9/10
- Novelty: 8/10
- Summary: Describes the HIP HAL driver that extends ROCm support in IREE. The HIP driver was derived from the CUDA driver's code structure but adapted for HIP semantics where no direct concept mapping exists. The driver was productionized as part of the GPU backend rework epic (#16341), replacing the earlier experimental ROCm path contributed by Nod.ai.
- Key detail: HIP HAL driver is now the default path for AMD GPUs (not the older "rocm" target). The LLVMGPU backend serves both CUDA and HIP, sharing codegen infrastructure while differentiating at the HAL runtime layer.

### 4. Metal HAL Driver Design Doc
- URL: https://iree.dev/developers/design-docs/metal-hal-driver/
- Type: docs
- Date: 2022–2024
- Relevance: 7/10
- Novelty: 6/10
- Summary: Describes how Metal (Apple GPU) is supported via the HAL abstraction. Kernels are compiled to MSL (Metal Shading Language) and undergo pipeline caching at runtime similar to Vulkan's pipeline cache mechanism. This driver shows the HAL's portability: the same dispatch interface covers NVIDIA (PTX/CUDA Graphs), AMD (HIP), Vulkan (SPIR-V), and Metal (MSL).
- Key detail: For backends requiring runtime preprocessing (SPIR-V, MSL), IREE HAL uses a pipeline cache/batch compilation mechanism where each executable is compiled on-demand and cached for the process lifetime — directly analogous to what libkdl would need for its kernel binary cache.

### 5. [Epic] Rework GPU Compiler Backends — GitHub Issue #16341
- URL: https://github.com/iree-org/iree/issues/16341
- Type: PR/issue
- Date: 2023–2024
- Relevance: 8/10
- Novelty: 8/10
- Summary: Tracks the multi-backend GPU compiler rework effort that aimed to regularize CUDA, ROCm/HIP, Metal, and Vulkan backends, which had diverged due to different development cycles. This epic produced the productionized HIP driver, the cuda2 rewrite, and unified codegen infrastructure under LLVMGPU.
- Key detail: Historical divergence between GPU backends was acknowledged as a technical debt item. The rework explicitly unified SPIR-V (Vulkan/Metal/WebGPU) and LLVMGPU (CUDA/ROCm) as the two codegen paths, with distinct HAL drivers per runtime API.

### 6. RFC — IREE Compiler Plugin Mechanism (Issue #12520)
- URL: https://github.com/iree-org/iree/issues/12520
- Type: RFC
- Date: 2022–2023
- Relevance: 8/10
- Novelty: 8/10
- Summary: Defines the compiler plugin system that allows in-tree and out-of-tree HAL target backends to be registered via a stable C API. Plugins hook into the main compilation pipeline at defined pass hook points. This is the mechanism enabling custom HAL targets (custom hardware, experimental backends) without forking IREE.
- Key detail: `StaticLinkedPlugins.inc` calls `iree_register_compiler_plugin_<id>()` during `PluginManager::initialize()` — this is the registration surface analogous to what libkdl would use if built as an IREE compiler plugin. HAL target dependencies must be limited to LLVM core libs and vendor SDK fragments manageable on CI.

### 7. IREE HAL Executable Variant / Fat Binary Target Selection
- URL: https://iree.dev/reference/mlir-dialects/HAL/
- Type: docs
- Date: 2024
- Relevance: 10/10
- Novelty: 9/10
- Summary: IREE's multi-targeting mechanism compiles each executable into one or more target-specific variants (fat binary style). At runtime, variants are selected by evaluating an optional condition op against the runtime `!hal.device`; the first valid variant wins. This is compile-time multi-versioning — not dynamic post-compilation selection.
- Key detail: Variant selection uses a condition region returning a boolean — not hardware capability queries or performance counters. If no variant matches, module loading fails. This design makes IREE's target selection **static after compile time** with no runtime performance-based re-dispatch, which is the primary gap libkdl would fill.

### 8. IREE Multi-GPU Issue #8435 — Multi-GPU Training Limitation
- URL: https://github.com/iree-org/iree/issues/8435
- Type: issue
- Date: 2022–2023
- Relevance: 7/10
- Novelty: 8/10
- Summary: Reports that while IREE initializes all available GPUs, only one GPU performs computation. The issue reveals that true multi-GPU dispatch (work split across devices in a single module execution) was not yet supported. Queue affinity (`iree_hal_queue_affinity_t`) is the intended mechanism but was not fully exercised for multi-GPU workloads at that time.
- Key detail: Queue affinity is a bitmask specifying which logical queues may execute a command buffer — logically ordered. This is the kernel routing primitive IREE exposes, but its use for load-balancing across heterogeneous devices was a gap as of 2022–2023.

### 9. IREE CUDA Backend Blog Post — October 2021
- URL: https://scottamain.github.io/iree/blog/2021-10-15-cuda-backend/
- Type: blog
- Date: 2021-10-15
- Relevance: 7/10
- Novelty: 6/10
- Summary: Describes the original CUDA HAL bring-up: HAL API was a natural fit for CUDA given its inspiration from Vulkan/Metal. PTX stored in FlatBuffer, JIT-compiled by CUDA driver at load, `cuLaunchKernel` used for dispatch. BERT training used as validation workload.
- Key detail: CUDA driver API (not runtime API) was chosen from the start to avoid unnecessary abstraction layers — demonstrating that IREE deliberately stays thin at the hardware interface layer, preferring direct driver-level calls.

---

## Angle Assessment

### Coverage
This angle is **well-documented** at the architectural level. IREE's HAL design is clearly specified with dedicated design docs per driver (CUDA, HIP, Metal, Vulkan). The compiler-side (target backend registration, plugin mechanism) and runtime-side (executable variants, queue affinity) are both covered in public docs.

### Surprise Findings
1. **IREE's fat-binary multi-targeting is purely static** — variant selection is a boolean condition op evaluated at module load time, not a runtime performance-based dispatch. This is a significant architectural gap relative to what libkdl proposes (dynamic post-dispatch selection based on runtime state, profiling, or load).
2. **Two distinct codegen paths** (LLVMGPU for CUDA/ROCm vs. SPIR-V for Vulkan/Metal/WebGPU) coexist at the HAL level with a single unified runtime dispatch interface — the abstraction works but the codegen split reveals backend fragmentation.
3. **CUDA Graphs are the default command buffer** implementation, not CUDA Streams — this is a performance-conscious choice that reduces per-kernel launch overhead via graph replay, directly relevant to dispatch overhead modeling.
4. **Queue affinity as a bitmask** is the only runtime kernel routing primitive IREE exposes — no capability scoring, no NUMA-awareness, no fallback cascade logic.

### Gaps
1. **No measured overhead numbers** for IREE HAL dispatch vs. direct `cuLaunchKernel` — the docs reference CUDA Graph benefits but give no microsecond-level latency data.
2. **No runtime-adaptive variant selection** — condition ops are evaluated at load, not per-dispatch. There is an experimental `#hal.device.optimal` attribute for allocation affinity but it does not extend to dispatch routing.
3. **Multi-device heterogeneous scheduling** (CPU + GPU + accelerator within one module execution) is described in the design roadmap as future work using constraint solving, but no production implementation exists.
4. **No preemption or migration mechanism** — once a kernel is dispatched to a device, there is no HAL-level concept of redirecting it mid-flight.

### Suggested Follow-Up Angles
1. **IREE Stream Dialect** — the `stream` dialect layer above HAL handles partitioning, async scheduling, and resource allocation; this is where multi-device work splitting decisions are made before lowering to HAL.
2. **IREE vs. OpenCL/oneAPI dispatch overhead** — need empirical numbers comparing HAL dispatch latency to direct API calls to quantify the abstraction cost.
3. **IREE design roadmap — constraint-based device assignment** — the roadmap mentions ML-guided dispatch ("big GEMMs go on the accelerator") but there is no public implementation; this is a direct research contribution opportunity.
4. **CUDA context-independent module loading** — NVIDIA's `cuLibraryLoadFromFile` / `cuKernelGetFunction` API (2023+) enables kernel loading without bound CUDA contexts, which is directly relevant to libkdl's need to manage kernel binaries across device contexts.
5. **IREE Plugin API for custom HAL targets** — exploring how libkdl could register as an IREE HAL driver plugin to intercept dispatch calls and inject dynamic routing logic.
