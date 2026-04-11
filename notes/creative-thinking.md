# Creative Thinking for Research: Heterogeneous GPU Kernel Dispatch via MLIR

*Generated 2026-04-02 using cognitive science-backed creative strategies.*
*Grounded in current state of MLIR (2025-2026), IREE, AdaptiveCpp, Kokkos, torch.compile.*

---

## Framework Application

### 1. Combinatorial Creativity -- Cross-Domain Fusion

**OS Scheduling + GPU Dispatch:**
Linux's CFS scheduler doesn't know at compile time which core a process will run on -- it decides at runtime based on load, affinity, and NUMA topology. GPU dispatch today is the opposite: the target is baked in at compile time. What if we built a "GPU CFS" that treats compiled kernel variants as runnable entities and dispatches them based on device load, memory pressure, and affinity (vendor-optimized vs. portable)?

**Database Query Planning + Kernel Compilation:**
PostgreSQL's query planner estimates cost for multiple execution strategies and picks the cheapest at runtime. The same principle applies: given a GEMM kernel and 3 available devices (NVIDIA A100, AMD MI300, x86 CPU), estimate execution cost on each using a cost model (FLOPS, memory bandwidth, launch overhead) and dispatch to the cheapest. The "query plan" is the set of pre-compiled kernel variants; the "statistics" are hardware capabilities discovered at runtime.

**Network Routing + Kernel Dispatch:**
BGP routers maintain routing tables with multiple paths and select the best one based on metrics. Similarly, a kernel could have a "routing table" mapping (kernel_id, device_type, device_capability) -> compiled_binary, with the runtime acting as a router that selects the best path. Fallback routes (CPU) exist if preferred paths (GPU) are unavailable.

**Package Managers + Kernel Versions:**
npm resolves `lodash@^4.0.0` to the best matching installed version. What if kernels had version constraints like `gemm@{sm_80+, gfx90a+, avx512+}`? A "kernel registry" stores multiple compiled variants, and the runtime "resolves" the best match for available hardware, with a dependency graph for composed operations.

### 2. Constraint Manipulation

**Drop "same binary format":**
Currently, GPU binaries are vendor-locked (PTX/SASS for NVIDIA, AMDGCN for AMD). MLIR already has `gpu-module-to-binary` which can produce objects for every target attribute attached to a module. The constraint to drop: "we must choose ONE target at compile time." Instead, emit ALL targets and let runtime pick. This is already partially possible -- the creative leap is making it seamless and lightweight.

**Drop "zero overhead":**
AdaptiveCpp's generic SSCP compilation shows ~15% JIT overhead on first kernel launch, reduced to near-zero on subsequent launches via persistent cache. If we accept 1-5% overhead for first-launch JIT specialization but guarantee near-native performance thereafter, the portability/performance tradeoff becomes very favorable. For long-running ML inference (minutes to hours), amortized overhead is negligible.

**Drop "full portability":**
Instead of "run anywhere," target the 3 scenarios that cover 95%+ of real ML deployments: (1) NVIDIA datacenter (A100/H100), (2) AMD datacenter (MI250/MI300), (3) x86 CPU fallback. This dramatically reduces the engineering surface while covering the practical need. Mobile/edge GPUs can be a future extension.

### 3. Inversion

**"Compile for no target, specialize lazily":**
Instead of choosing a target, compile to MLIR's `gpu` dialect (or `linalg`) and stop. At runtime, when the first kernel launch happens, JIT-lower from `gpu` -> target-specific (nvvm/rocdl/spirv/llvm) based on discovered hardware, then cache. This is the AdaptiveCpp SSCP model applied to MLIR.

**"Backend advertises capabilities, kernels select themselves":**
Flip the dispatch direction. Instead of a central dispatcher picking a backend for a kernel, each backend registers its capabilities (SM version, memory bandwidth, supported operations), and the kernel itself has selection logic: "I run best on backends with warp_size=32 AND shared_memory >= 48KB." This is analogous to Linux's device-driver model where drivers register capabilities and the kernel matches devices to drivers.

### 4. SCAMPER on IREE

| SCAMPER | Application | Assessment |
|---------|-------------|------------|
| **Substitute** | Replace IREE's full HAL with a ~500-line minimal dispatch layer that only does: device discovery, kernel launch, memory copy, synchronization | High feasibility. IREE's HAL is powerful but requires buying into the full IREE ecosystem. A minimal HAL that works with plain MLIR-compiled binaries has standalone value. |
| **Combine** | ALPAKA's compile-time type-level abstraction + MLIR's code generation. Use ALPAKA-style C++ templates to generate MLIR `gpu` dialect code, getting both type safety and multi-target compilation. | Medium feasibility. Requires C++/MLIR interop work. Novel combination. |
| **Adapt** | cuDNN's heuristic-based kernel selection to multi-vendor context. cuDNN tries multiple algorithms at runtime (GEMM strategies) and picks the fastest. Do the same across VENDORS, not just algorithms. | High feasibility. Well-understood technique applied to a new axis. |
| **Modify** | Modify MLIR's `gpu` dialect to carry target-preference metadata: `gpu.launch { targets = [#nvvm, #rocdl, #spirv], preference = "highest_flops" }` | Medium feasibility. Requires upstream MLIR RFC, but the `gpu-module-to-binary` pass already supports multiple target attributes per module. |
| **Put to other use** | Use LLVM's existing `clang-offload-bundler` fat binary infrastructure (designed for CUDA/HIP offloading) to bundle cross-vendor GPU kernels. The bundler already handles multiple architectures -- extend it to handle {nvptx, amdgcn, spirv, x86}. | High feasibility. Reuses proven infrastructure. |
| **Eliminate** | Eliminate the need for a full compiler at runtime. Pre-compile to ALL targets at build time (via `gpu-module-to-binary`), ship a multi-target bundle, and at runtime only do: discover hardware -> select binary -> launch. No JIT needed. | Very high feasibility. Simplest approach. Trades binary size for runtime simplicity. |
| **Reverse** | "Many pre-compiled targets, one dispatch decision." Instead of the traditional "one source, compile for target at deploy time," compile for all targets at build time. The poster's contribution becomes the dispatch decision logic + the tooling to make multi-target compilation ergonomic via MLIR. | Very high feasibility. This is the strongest candidate direction. |

### 5. First Principles Decomposition

| Step | What's Needed | Existing Solutions | Gap Assessment |
|------|---------------|-------------------|----------------|
| 1. Device discovery | Know what hardware is available | CUDA/HIP/Vulkan APIs, `lspci`, `hwloc` | **Well-served.** Every runtime has this. A thin unified wrapper is straightforward. |
| 2. Multi-target code | Have code ready for each target | MLIR `gpu-module-to-binary` can compile to multiple targets; IREE does this internally | **Partially served.** MLIR has the infrastructure but no ergonomic end-to-end flow. IREE requires full buy-in. |
| 3. Target selection | Pick the best target | cuDNN does this for algorithms; no equivalent for cross-vendor dispatch | **UNDER-SERVED.** This is the biggest gap. No lightweight, standalone cost-model-driven target selector exists. |
| 4. Execution | Launch on chosen target with correct memory management | Each vendor runtime handles this; no unified minimal API outside IREE/SYCL | **Partially served.** Unified APIs exist (SYCL, IREE HAL) but are heavyweight. |
| 5. Failure handling | Fallback chain | ONNX Runtime's EP fallback model | **Under-served** for lightweight use cases. |

**Key insight from first principles: Step 3 (target selection / dispatch decision) is the most under-served.** MLIR already handles step 2 via `gpu-module-to-binary`. Steps 1 and 4 have vendor APIs. But nobody has built a lightweight, cost-model-driven target selector that sits outside a full runtime like IREE or SYCL.

---

## The 10 Most Promising Ideas

### Idea 1: `mlir-dispatch` -- Multi-Target AOT + Lightweight Runtime Selector

**Concept:** Use MLIR's existing `gpu-module-to-binary` pass to compile a single `gpu` module to multiple target objects (NVPTX, AMDGCN, SPIR-V, x86). Bundle them into a single artifact. At runtime, a thin C library (~300 lines) discovers available hardware, selects the best binary, and dispatches.

**Reasoning:** This is the "SCAMPER Reverse + Eliminate" idea. It avoids JIT entirely by pre-compiling all targets. The novelty is NOT in compilation (MLIR already does this) but in the glue: (a) an ergonomic CLI/pass pipeline that makes multi-target compilation a single command, (b) a minimal runtime that selects and launches the right binary, (c) benchmark evidence that this approach has near-zero dispatch overhead.

**Why it addresses reviewers:**
- 91A: Concrete mechanism (the runtime selector + pass pipeline)
- 91B: Shows value for model serving across heterogeneous clouds (AWS with NVIDIA, Azure with AMD, CPU fallback)
- 91C: Clear proposal, not a survey
- 91D: Broader than SYCL -- uses MLIR natively, compares to IREE/SYCL/ALPAKA

**Feasibility (2 days):** HIGH. `gpu-module-to-binary` exists. Device discovery APIs are well-documented. The runtime selector is small. Main risk: getting MLIR's GPU compilation pipeline to work end-to-end for a non-trivial kernel on the available hardware.

**Implementation sketch:**
1. Write a GEMM in MLIR `linalg` dialect
2. Lower to `gpu` dialect, attach `#nvvm.target`, `#rocdl.target`, `#spirv.target` attributes
3. Run `gpu-module-to-binary` to produce multi-target binary
4. Write a C runtime: `discover_devices()` -> `select_best_target(binary)` -> `launch_kernel()`
5. Benchmark dispatch overhead vs. direct CUDA/HIP launch

---

### Idea 2: Capability-Aware Cost Model for Target Selection

**Concept:** Build a simple cost model that, given a kernel's computational profile (FLOPS, memory bandwidth requirement, parallelism) and a device's capabilities (SM count, memory bandwidth, clock speed), estimates execution time on each device and picks the fastest.

**Reasoning:** This is the "database query planner" analogy. cuDNN does this for algorithm selection within NVIDIA; nobody does it for cross-vendor target selection. The cost model doesn't need to be perfect -- even a simple roofline-model-based estimator beats random selection.

**Why it's novel:** Existing multi-target systems (IREE, SYCL) either use fixed priority ordering or let the user choose. A cost-model-driven automatic selection based on kernel characteristics + device capabilities is genuinely new for cross-vendor dispatch.

**Feasibility (2 days):** HIGH. A roofline-model cost estimator is ~100 lines of Python. Device capability queries are standard APIs. The challenge is calibrating the model, but even a rough model demonstrates the concept for a poster.

---

### Idea 3: Fallback Chain with Graceful Degradation

**Concept:** Define a priority-ordered fallback chain: try CUDA first, then HIP, then Vulkan/SPIR-V, then CPU. If the preferred backend fails (device OOM, driver error, missing hardware), automatically fall back to the next option. Inspired by ONNX Runtime's Execution Provider model but applied at the kernel level rather than the model level.

**Reasoning:** Production ML systems crash when the expected GPU isn't available or runs out of memory. A fallback chain makes the system resilient. This is the "network routing" analogy -- multiple paths, automatic rerouting on failure.

**Feasibility (2 days):** HIGH. This is mostly runtime logic with try/catch around kernel launches. Pairs naturally with Idea 1.

---

### Idea 4: MLIR `gpu` Dialect Extension for Multi-Target Metadata

**Concept:** Propose (as an RFC sketch) extending MLIR's `gpu` dialect operations with target-preference metadata. Example:
```mlir
gpu.launch_func @kernel
    targets = [#nvvm.target<chip="sm_90">,
               #rocdl.target<chip="gfx942">,
               #llvm.target<cpu="x86-64-v4">]
    selection = "cost_model"  // or "priority" or "round_robin"
```

The `gpu-module-to-binary` pass would consume these attributes to produce a multi-target binary, and a runtime pass would generate dispatch code.

**Reasoning:** This embeds multi-target dispatch into MLIR's own abstraction layer, making it a first-class MLIR feature rather than an external tool. This is the "Modify" SCAMPER idea.

**Feasibility (2 days):** MEDIUM. Designing the attribute schema and writing an RFC sketch is feasible. Actually modifying MLIR upstream is not a 2-day task, but a design proposal + proof-of-concept pass is.

---

### Idea 5: Lazy JIT Specialization with Persistent Cache

**Concept:** Compile to MLIR's `gpu` dialect and stop. At first kernel launch, detect hardware, JIT-compile to the native target using MLIR's ORC JIT, and cache the result to disk. Subsequent launches (even across program restarts) use the cached binary.

**Reasoning:** This is the AdaptiveCpp SSCP model applied to bare MLIR. The advantage over AOT multi-target compilation (Idea 1) is smaller binary size -- you only compile what you actually use. The advantage over AdaptiveCpp is working within pure MLIR without requiring the SYCL programming model.

**Why it addresses 91D:** Directly implements "multi-versioned kernels specialized at JIT time by querying hardware features."

**Feasibility (2 days):** MEDIUM-LOW. MLIR's JIT infrastructure works but is complex to set up. Getting the full pipeline (linalg -> gpu -> nvvm -> PTX -> launch) working via JIT in 2 days is ambitious. Better as a design proposal with partial implementation.

---

### Idea 6: Cross-Vendor Fat Binary via `clang-offload-bundler`

**Concept:** Repurpose LLVM's `clang-offload-bundler` tool (which already bundles CUDA/HIP fat binaries for multiple GPU architectures) to create cross-VENDOR fat binaries containing {NVPTX, AMDGCN, x86} objects. Write a thin unbundler/selector runtime.

**Reasoning:** The infrastructure already exists in LLVM for bundling multiple GPU architectures. The creative leap is using it across vendors, not just across architecture versions of the same vendor. This is "Put to other use" from SCAMPER.

**Feasibility (2 days):** MEDIUM-HIGH. The bundler tool exists. The question is whether its format supports cross-vendor targets (it's designed for same-vendor multi-arch). May need minor modifications or a custom bundling format.

---

### Idea 7: Kernel Routing Table -- Static Dispatch Map with Runtime Resolution

**Concept:** At compile time, produce a JSON/binary "routing table" that maps:
```
(kernel_name, device_vendor, min_compute_capability) -> binary_offset
```
At runtime, query hardware, look up the routing table, and jump to the right binary. This is a hash-map lookup -- effectively zero overhead.

**Reasoning:** The "network routing" analogy made literal. The routing table is generated by the compiler, consumed by the runtime. It's dead simple, easy to debug (the table is human-readable), and extensible (add new targets by adding rows).

**Feasibility (2 days):** VERY HIGH. This is essentially a lookup table + device query. The engineering is trivial; the value is in demonstrating it works end-to-end with MLIR-compiled kernels.

---

### Idea 8: Self-Selecting Kernels via Capability Contracts

**Concept:** Invert the dispatch model. Each compiled kernel variant declares its requirements as a "capability contract":
```
gemm_nvptx: requires {cuda >= 11.0, sm >= 80, shared_mem >= 48KB}
gemm_amdgcn: requires {hip >= 5.0, gfx >= 90a}
gemm_cpu: requires {avx512}
gemm_generic: requires {} // always matches
```
At runtime, the device advertises its capabilities, and kernel variants "self-select" by checking if the device satisfies their contract. The first matching variant (in priority order) wins.

**Reasoning:** This is the "Inversion" idea -- backends advertise, kernels select. It's modular: adding a new kernel variant or a new device type requires no changes to the dispatch logic, only a new contract. It mirrors Linux's device-driver model.

**Feasibility (2 days):** HIGH. Capability contracts are just metadata. Device capability queries are standard. The matching logic is trivial.

---

### Idea 9: Integration Shim for torch.compile / ONNX Runtime

**Concept:** Build a thin "backend shim" that plugs into torch.compile's backend API or ONNX Runtime's Execution Provider API. The shim intercepts compiled graphs and redirects them through our multi-target dispatch layer. This directly answers Reviewer 91B's concern about PyTorch/TF integration.

**Reasoning:** The poster's biggest weakness per reviewers is lack of connection to real ML frameworks. Showing that the dispatch layer can sit behind `torch.compile(model, backend="mlir_hetero")` makes it immediately relevant to practitioners.

**Feasibility (2 days):** LOW for full implementation, HIGH for design + proof-of-concept. Writing a torch.compile backend that emits MLIR and dispatches is a substantial project. But sketching the architecture and showing a minimal example (even if it only handles a single op) is feasible and sufficient for a poster.

---

### Idea 10: Benchmark-Driven Auto-Tuning at Install Time

**Concept:** At "install time" (when the library is first used on a new machine), automatically benchmark each kernel variant on each available device, record results, and build an optimized dispatch table. Subsequent runs use the pre-tuned dispatch table for near-zero overhead and near-optimal performance.

**Reasoning:** This is the "cuDNN heuristic selection" idea applied cross-vendor. cuDNN benchmarks multiple GEMM algorithms on first use and caches results. We do the same across vendors. It sidesteps the need for a perfect cost model (Idea 2) by using empirical measurement instead.

**Feasibility (2 days):** MEDIUM. The benchmarking harness is straightforward. The challenge is having multiple GPU vendors available for testing. On a single-GPU machine, this can be demonstrated for GPU vs. CPU selection.

---

## Recommended Combination for Poster

The strongest poster contribution combines Ideas 1 + 2 + 7 + 8, which together form a coherent system called **`mlir-hetero-dispatch`**:

1. **Multi-target AOT compilation** via MLIR's `gpu-module-to-binary` (Idea 1) -- uses existing MLIR infrastructure
2. **Kernel routing table** with capability contracts (Ideas 7 + 8) -- the novel dispatch mechanism
3. **Cost-model-driven selection** (Idea 2) -- the intelligent decision layer
4. **Fallback chain** (Idea 3) -- resilience

This combination:
- Has a **concrete implementation** (addresses 91A, 91C)
- Is **broader than SYCL** -- works at the MLIR level (addresses 91D)
- Shows **value beyond compile-time dispatch** via cost-model selection and fallback (addresses 91B)
- Can be **sketched as a torch.compile integration** (addresses 91B's PyTorch concern)
- Is **implementable in 2 days** as a proof-of-concept
- Sits in an **under-served niche**: lightweight, standalone, MLIR-native dispatch without requiring IREE/SYCL/ALPAKA buy-in

### Architecture Sketch

```
                    Build Time                          Runtime
              +--------------------+           +---------------------+
              |                    |           |                     |
  MLIR Source |  linalg.matmul     |           |  1. discover_devices()
  (linalg)   |        |           |           |     -> [A100, MI300, CPU]
              |        v           |           |                     |
              |  gpu.launch_func   |           |  2. load routing table
              |   + target attrs   |           |     (from binary bundle)
              |        |           |           |                     |
              |        v           |           |  3. match capabilities
              | gpu-module-to-     |           |     kernel.contract vs
              |   binary           |           |     device.capabilities
              |   |    |    |      |           |                     |
              |   v    v    v      |           |  4. cost_model_rank()
              | nvptx amdgcn x86  |           |     -> A100 wins for GEMM
              |   |    |    |      |           |                     |
              |   v    v    v      |           |  5. launch(nvptx_binary)
              | [routing table]    |           |     fallback -> amdgcn
              | [bundled binary]   |           |     fallback -> x86
              +--------------------+           +---------------------+
```

### 2-Day Sprint Plan

**Day 1 (8 hours):**
- Hours 1-2: Set up MLIR build, write GEMM kernel in linalg dialect
- Hours 3-4: Compile to gpu dialect, attach multi-target attributes, run `gpu-module-to-binary`
- Hours 5-6: Write the routing table generator (Python script that parses the binary and creates the dispatch map)
- Hours 7-8: Write the runtime selector in C (~300 lines): device discovery + table lookup + dispatch

**Day 2 (8 hours):**
- Hours 1-2: Write capability contract schema and matching logic
- Hours 3-4: Implement simple roofline cost model for target ranking
- Hours 5-6: Benchmark: measure dispatch overhead, compare native vs. dispatched kernel performance
- Hours 7-8: Generate benchmark plots, write poster content, create architecture diagram

---

## Feasibility Summary

| Idea | Novelty | Feasibility (2 days) | Reviewer Impact | Recommended |
|------|---------|---------------------|-----------------|-------------|
| 1. Multi-target AOT + selector | Medium | High | High (91A,C) | YES - core |
| 2. Cost model for selection | High | High | Medium (91B) | YES - differentiator |
| 3. Fallback chain | Low | High | Medium (91B) | YES - resilience |
| 4. MLIR dialect extension | High | Medium | High (91D) | Partial - RFC sketch only |
| 5. Lazy JIT specialization | Medium | Medium-Low | High (91D) | Design only |
| 6. Fat binary via offload-bundler | Medium | Medium-High | Medium | Backup approach |
| 7. Kernel routing table | Medium | Very High | High (91A) | YES - core |
| 8. Self-selecting capability contracts | High | High | High (91D) | YES - core |
| 9. torch.compile integration shim | Medium | Low (full) / High (sketch) | Very High (91B) | Architecture sketch |
| 10. Install-time auto-tuning | Medium | Medium | Medium | Future work |

---

## Key Technical Grounding

### What MLIR Already Provides (no need to reinvent)

- **`gpu-module-to-binary` pass**: Compiles GPU modules to binaries for every target attribute attached. Supports NVVM, ROCDL, SPIR-V targets. This is the compilation backbone.
- **`gpu` dialect**: Vendor-agnostic GPU operations (`gpu.launch_func`, `gpu.alloc`, `gpu.memcpy`). The natural IR level for multi-target dispatch.
- **Target attributes**: `#nvvm.target<chip="sm_90">`, `#rocdl.target<chip="gfx942">`, `#spirv.target<...>`. Already multi-target aware.
- **`gpu-lower-to-nvvm-pipeline`**: Complete lowering from arith/memref/scf/gpu -> NVVM. Similar pipelines for ROCDL and SPIR-V.
- **ORC JIT via ExecutionEngine**: JIT compilation support, though it fixes targets at JIT time.

### What Exists Externally (must differentiate from)

- **IREE**: Full compiler + runtime with HAL. Supports CUDA, HIP, Vulkan, CPU backends. But requires buying into the full IREE stack -- not composable with arbitrary MLIR programs.
- **AdaptiveCpp (hipSYCL)**: Generic SSCP compiles to vendor-independent LLVM IR, JIT-compiles at runtime. ~15% first-launch overhead, cached thereafter. But requires SYCL programming model.
- **SYCL (oneAPI DPC++)**: Broad hardware support via Intel's oneAPI. Heavyweight runtime.
- **Kokkos/RAJA**: C++ performance portability frameworks. Compile-time dispatch via templates. No MLIR integration.
- **OCCA**: JIT compilation for portable GPU kernels. Runtime target selection. No MLIR integration.
- **torch.compile + Inductor**: Backend API allows custom compilation backends. Multi-GPU dispatch still evolving.
- **ONNX Runtime EPs**: Execution Provider model with CUDA/TensorRT/ROCm/CPU EPs. Model-level dispatch, not kernel-level.

### The Actual Gap Our Poster Fills

**No existing tool provides a lightweight, standalone, MLIR-native multi-target dispatch mechanism.** IREE requires its full stack. SYCL/AdaptiveCpp requires the SYCL programming model. Kokkos/RAJA are C++ template-based. OCCA is its own thing.

What's missing: a thin layer that takes MLIR-compiled multi-target binaries and dispatches them at runtime with minimal overhead, without requiring any specific programming model or full runtime stack. This is the `mlir-hetero-dispatch` contribution.

---

## Sources

- [MLIR GPU Dialect Documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
- [gpu-module-to-binary pass review](https://reviews.llvm.org/D154149)
- [MLIR Part 8 - GPU Compilation](https://www.stephendiehl.com/posts/mlir_gpu/)
- [AdaptiveCpp/hipSYCL GitHub](https://github.com/AdaptiveCpp/AdaptiveCpp)
- [AdaptiveCpp JIT Optimization Paper](https://dl.acm.org/doi/10.1145/3731125.3731127)
- [AMD Vendor-Flavored SPIR-V in LLVM](https://www.phoronix.com/news/LLVM-AMDGCN-Flavored-SPIR-V)
- [IREE Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)
- [IREE Deployment Configurations](https://iree.dev/guides/deployment-configurations/)
- [Kokkos/RAJA Performance Portability Comparison](https://arxiv.org/html/2402.08950v1)
- [torch.compile Documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Intel GPU Inductor Backend RFC](https://github.com/pytorch/pytorch/issues/114856)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/)
- [Mojo GPU Performance Portability](https://arxiv.org/html/2509.21039v1)
- [MLIR SPIR-V Dialect Discussion](https://discourse.llvm.org/t/spirv-dialect-and-spir-v-target-design-decision/66646)
- [LLVM Clang Offload Bundler](https://llvm.org/docs/CompileCudaWithLLVM.html)
