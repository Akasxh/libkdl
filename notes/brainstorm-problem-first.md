# Brainstorming: Problem-First Research Directions

**Project:** Heterogeneous GPU Kernel Dispatch via MLIR
**Venue:** LLVM Dublin 2026
**Date:** 2026-04-02
**Author:** Akash (IIT Patna, CERN GSoC / ALPAKA, vLLM contributor)

---

## Landscape Summary (grounding the brainstorm)

Before generating ideas, here is what actually exists today:

| System | Approach | Multi-target? | Runtime dispatch? | MLIR integration? | Limitation |
|--------|----------|---------------|--------------------|--------------------|------------|
| IREE | Full-stack compiler+runtime, FlatBuffer fat binaries | Yes (SPIR-V, CUDA, HIP, CPU) | Yes, via HAL | Deep (owns the MLIR pipeline) | Requires buying into IREE's entire stack |
| Triton | Python DSL -> MLIR -> PTX/AMDGCN | Yes (NVIDIA, AMD) | No runtime selection -- compile-time target | Uses MLIR internally | Single-target at compile time |
| MLIR gpu dialect | `gpu-module-to-binary` can emit objects for multiple targets | Yes (multi-target attributes) | Only `#gpu.select_object` (selects first) | IS MLIR | No intelligent runtime selection logic |
| ALPAKA | C++ header-only, template metaprogramming | Yes (CUDA, HIP, SYCL, OpenMP, serial) | Compile-time backend selection | None | No IR-level approach, C++ templates only |
| SYCL/AdaptiveCpp | Source-level portability, SPIR-V/PTX/AMDGCN | Yes | Runtime device selection | Minimal | Heavy runtime, performance overhead |
| Kokkos/RAJA | C++ abstraction layers | Yes | Compile-time | None | No IR-level, similar to ALPAKA |
| ONNX Runtime | Execution Provider model | Yes (CUDA EP, TRT EP, ROCm EP, CPU EP) | Yes (priority-based EP selection) | None | Graph-level, not kernel-level dispatch |
| TVM | Compiler + AutoTVM/Ansor tuning | Yes | Limited runtime scheduling | Separate from MLIR | Own IR (Relay/TIR), not MLIR |
| CUDA fat binaries | PTX + SASS for multiple SM versions | Yes (same vendor, different arch) | Yes (driver selects best SASS, JIT-compiles PTX) | N/A | NVIDIA-only |

**The gap:** MLIR already has `gpu-module-to-binary` that can compile to multiple targets. But the runtime selection is trivial (`select_object` picks the first one). Nobody has built an intelligent, capability-aware runtime dispatch layer that sits between MLIR's multi-target compilation and actual execution. IREE does this but requires its entire stack. There is no lightweight, composable alternative.

---

## Framework 1: Problem Decomposition

Breaking "heterogeneous GPU dispatch" into subproblems:

### Subproblem 1: Multi-target compilation
**Status:** Partially solved. MLIR's `gpu-module-to-binary` can attach multiple target attributes and produce objects per target. IREE does this with its compiler. Triton compiles to one target at a time.
**Gap:** The pass exists but is under-documented and under-used. No standard way to specify "compile for sm_80 + gfx90a + x86_64" in a single pipeline invocation from Python.

### Subproblem 2: Hardware discovery at runtime
**Status:** Well-solved individually (CUDA `cudaGetDeviceProperties`, HIP `hipGetDeviceProperties`, Vulkan `vkEnumeratePhysicalDevices`). No unified API.
**Gap:** No lightweight, vendor-neutral hardware discovery library that works across CUDA, HIP, Vulkan, and CPU.

### Subproblem 3: Kernel-to-device matching
**Status:** Poorly solved. CUDA fat binaries do this within NVIDIA (match SM version). IREE's HAL does it within IREE. ONNX Runtime's EP model does it at graph level.
**Gap:** No generic "match a compiled kernel variant to the best available device" mechanism outside of vendor stacks.

### Subproblem 4: Dispatch with minimal overhead
**Status:** Kernel launch overhead is well-understood (~5-15us for CUDA, similar for HIP). Dispatch indirection adds negligible overhead if done once per kernel invocation.
**Gap:** Nobody has measured the overhead of a cross-vendor dispatch layer on top of MLIR.

### Subproblem 5: Integration with ML frameworks
**Status:** PyTorch's `torch.compile` has a backend system. ONNX Runtime has EPs. TF/XLA has its own pipeline.
**Gap:** No `torch.compile` backend that does vendor-agnostic dispatch via MLIR multi-target compilation.

### **Most impactful, least solved:** Subproblem 3 (kernel-to-device matching) combined with Subproblem 5 (framework integration). The compilation infrastructure exists. The device APIs exist. The missing piece is the intelligent glue layer with a clean integration path.

---

## Framework 2: Analogical Transfer

### Analogy 1: OS Process Scheduling -> GPU Kernel Scheduling
Linux CFS doesn't know at compile time which CPU core a process will run on. It discovers cores, measures load, and dispatches dynamically. The scheduler uses per-core run queues, load balancing, and CPU affinity hints.

**Transfer:** Build a "GPU kernel scheduler" that maintains a registry of compiled kernel variants, discovers available devices, and dispatches based on a cost model (not just "pick the first one"). Support affinity hints ("prefer NVIDIA for this GEMM, but fall back to AMD").

**Idea 1: `mlir-ksched` -- A kernel scheduler modeled on Linux CFS for GPU dispatch**
- Novelty: 4/5 -- the analogy is fresh, the mechanism is new in MLIR
- Feasibility: 3/5 -- needs runtime + MLIR pass integration
- Impact: 4/5 -- compelling story for poster
- LLVM-relevance: 5/5

### Analogy 2: Database Query Optimization -> Kernel Plan Selection
Postgres doesn't execute a query as-is. It generates multiple execution plans, estimates costs, and picks the cheapest. The cost model uses table statistics (hardware = statistics, kernel = query).

**Transfer:** For each kernel, generate multiple "execution plans" (one per target), estimate cost based on hardware capabilities (memory bandwidth, compute throughput, occupancy), and select the optimal plan at runtime.

**Idea 2: `cost-model-dispatch` -- Query-optimizer-style kernel plan selection**
- Novelty: 4/5 -- cost-model-driven dispatch for MLIR is unexplored
- Feasibility: 3/5 -- cost model is the hard part
- Impact: 4/5 -- intellectually elegant
- LLVM-relevance: 4/5

### Analogy 3: Network Routing -> Kernel Routing
BGP/OSPF doesn't hardcode packet paths. Routers discover topology, measure link costs, and compute shortest paths. Routes adapt when links fail.

**Transfer:** Treat each device as a "node" in a network. Kernels are "packets." Route each kernel to the device with the lowest "latency" (dispatch cost + execution time). When a device is busy or unavailable, reroute.

**Idea 3: `kernel-routing` -- Topology-aware kernel routing across heterogeneous devices**
- Novelty: 3/5 -- interesting framing but complex
- Feasibility: 2/5 -- topology discovery + routing is heavy
- Impact: 3/5 -- may be over-engineered for a poster
- LLVM-relevance: 3/5

### Analogy 4: Dynamic Linking -> Dynamic Kernel Linking
`ld.so` resolves shared library symbols at runtime. It searches `LD_LIBRARY_PATH`, checks ELF headers, handles versioning (`SONAME`), and lazy-binds symbols on first call.

**Transfer:** Treat compiled kernel variants as "shared libraries." At runtime, a "kernel linker" resolves the best variant for the current hardware, lazy-loads it, and caches the binding. Symbol = kernel name, version = target architecture.

**Idea 4: `mlir-kernel-linker` -- Dynamic linking semantics for GPU kernel dispatch**
- Novelty: 5/5 -- nobody has framed GPU dispatch as dynamic linking
- Feasibility: 4/5 -- conceptually clean, implementable
- Impact: 5/5 -- immediately resonates with systems programmers at LLVM Dublin
- LLVM-relevance: 5/5 -- directly extends LLVM/ELF concepts to GPUs

### Analogy 5: Web Browser Rendering -> GPU/CPU Rendering Decision
Chrome decides per-element whether to use GPU compositing or CPU rendering based on feature support, GPU memory, and heuristics. Falls back gracefully.

**Transfer:** Per-operation decision: "Can this op run on GPU? Which GPU? Fall back to CPU if needed." Graceful degradation with performance monitoring.

**Idea 5: `graceful-dispatch` -- Per-operation GPU/CPU dispatch with automatic fallback**
- Novelty: 3/5 -- ONNX Runtime EPs do something similar at graph level
- Feasibility: 4/5 -- straightforward
- Impact: 3/5 -- incremental over existing systems
- LLVM-relevance: 4/5

---

## Framework 3: Constraint Removal

### Assumption 1: "Kernels must target a specific GPU at compile time"
**Remove it:** MLIR's `gpu-module-to-binary` already supports multiple targets. But nobody uses it with a runtime selector beyond `select_object`. What if the default MLIR pipeline made multi-target the norm, not the exception?

**Idea 6: `multi-target-by-default` -- Make MLIR's multi-target compilation the default, not opt-in**
- Extend the `gpu-lower-to-nvvm-pipeline` to also produce AMDGCN and SPIR-V objects alongside NVPTX
- Add a `#gpu.capability_select` offloading handler that replaces `#gpu.select_object`
- Novelty: 4/5 -- extending existing MLIR infrastructure in a natural direction
- Feasibility: 4/5 -- builds on existing pass infrastructure
- Impact: 4/5 -- directly improves MLIR upstream
- LLVM-relevance: 5/5

### Assumption 2: "The runtime must know the exact GPU model"
**Remove it:** What if the runtime only needs to know broad capability categories? (e.g., "has tensor cores," "supports f16," "shared memory >= 48KB"). Map kernels to capability requirements, not specific GPU models.

**Idea 7: `capability-descriptors` -- Abstract hardware capability descriptors for kernel matching**
- Define a capability descriptor format: `{compute_units: >=64, shared_mem: >=48KB, has_tensor_cores: true}`
- Each compiled kernel variant is tagged with minimum requirements
- Runtime matches requirements to available hardware
- Novelty: 4/5 -- capability-based matching exists in Vulkan but not in MLIR compilation
- Feasibility: 4/5 -- capability descriptors are simple to define
- Impact: 4/5 -- clean abstraction
- LLVM-relevance: 5/5

### Assumption 3: "Performance portability requires source-level changes"
**Remove it:** ALPAKA requires writing kernels using ALPAKA's API. SYCL requires SYCL syntax. What if portability is achieved purely at the IR level, requiring zero source changes?

**Idea 8: `zero-change-portability` -- IR-level portability via MLIR dialect transformations**
- Take standard `linalg.generic` or `linalg.matmul` ops
- Apply target-specific transformations (tiling, vectorization) as MLIR passes
- Emit multi-target binaries from the same high-level IR
- User writes nothing GPU-specific
- Novelty: 3/5 -- this is what IREE does, but with its own stack
- Feasibility: 3/5 -- tiling/vectorization passes exist but need per-target tuning
- Impact: 4/5 -- strong demo value
- LLVM-relevance: 5/5

### Assumption 4: "Multi-vendor dispatch requires a heavyweight runtime"
**Remove it:** IREE has a full VM+HAL. SYCL has a complex runtime. What if the dispatch layer is ~200 lines of C with no dependencies beyond vendor driver APIs?

**Idea 9: `micro-dispatch` -- A <500-line C runtime for cross-vendor kernel dispatch**
- Minimal runtime: device discovery + kernel selection + launch
- No scheduler, no memory management, no graph compiler
- Just: "here are N compiled kernels, here is the hardware, pick one and run it"
- Novelty: 4/5 -- the minimalism IS the novelty (contrast with IREE's complexity)
- Feasibility: 5/5 -- very implementable in 2 days
- Impact: 4/5 -- simplicity sells at a poster
- LLVM-relevance: 4/5

---

## Framework 4: Negation

### Negation 1: "Kernels DON'T need to target a specific GPU at compile time"
This is the SPIR-V thesis. Compile to a portable IR, let the driver/runtime translate.

**Idea 10: `mlir-to-spirv-universal` -- MLIR -> SPIR-V as the universal dispatch target**
- Compile everything to SPIR-V via MLIR's `spirv` dialect
- Use Vulkan compute as the universal execution layer
- Measure performance gap vs native CUDA/HIP
- Honest assessment: where does SPIR-V fall short?
- Novelty: 3/5 -- IREE already does SPIR-V, but measuring the gap honestly is valuable
- Feasibility: 4/5 -- MLIR has spirv dialect and Vulkan runner
- Impact: 3/5 -- incremental unless the perf numbers are surprising
- LLVM-relevance: 4/5

### Negation 2: "We DON'T try to match native performance -- aim for 90% with zero porting effort"
**Idea 11: `90-percent-portable` -- Quantify the "portability tax" across MLIR backends**
- Compile the same GEMM kernel via MLIR to NVPTX, AMDGCN, SPIR-V, x86
- Measure performance relative to hand-tuned cuBLAS/rocBLAS
- Plot the "portability frontier": effort vs performance across targets
- Answer: "What percentage of native perf do you get for free with MLIR?"
- Novelty: 4/5 -- this specific measurement hasn't been published
- Feasibility: 5/5 -- compile + benchmark, no new infrastructure needed
- Impact: 5/5 -- every attendee wants to know this number
- LLVM-relevance: 5/5

### Negation 3: "The dispatch decision is made by the COMPILER, not the runtime"
What if the compiler pre-computes a dispatch table based on known target profiles, and the runtime just does a table lookup?

**Idea 12: `compiled-dispatch-tables` -- Compiler-generated dispatch tables for known hardware profiles**
- Compiler enumerates common GPU profiles (A100, H100, MI250, MI300, Arc A770...)
- Generates a dispatch table: {profile -> best kernel variant + tiling params}
- Runtime: detect hardware, lookup table, launch
- Analogy: like CUDA's architecture-specific SASS in fat binaries, but cross-vendor
- Novelty: 4/5 -- cross-vendor dispatch tables are new
- Feasibility: 4/5 -- table generation is straightforward
- Impact: 4/5 -- practical and demonstrable
- LLVM-relevance: 4/5

### Negation 4: "We dispatch at the OPERATION level, not the KERNEL level"
Current systems dispatch whole kernels. What if we dispatch individual operations within a kernel to different devices?

**Idea 13: `op-level-dispatch` -- Fine-grained per-operation device placement**
- Analyze a computation graph: some ops are better on GPU A, others on GPU B, others on CPU
- Partition and dispatch at the operation level
- Similar to TF's device placement, but at the MLIR level with cost-model guidance
- Novelty: 3/5 -- TF/IREE do graph-level partitioning
- Feasibility: 2/5 -- data movement costs dominate; hard to make this win
- Impact: 3/5 -- interesting but may not show clear benefits
- LLVM-relevance: 4/5

---

## Framework 5: Gap Analysis

### Gap 1: MLIR's compilation power <-> Runtime execution
MLIR can compile to NVPTX, AMDGCN, SPIR-V, x86. But `ExecutionEngine` fixes the target. The `gpu-module-to-binary` pass can produce multi-target binaries, but the offloading handler (`#gpu.select_object`) is trivial. There is no "smart" handler that queries hardware and selects the best object.

**Idea 14: `#gpu.capability_select` -- A new MLIR offloading handler for capability-aware dispatch**
- Implement a new offloading attribute that replaces `#gpu.select_object`
- At translation time, emits runtime code that:
  1. Queries available GPU devices (CUDA, HIP, Vulkan APIs)
  2. Matches device capabilities to compiled kernel requirements
  3. Selects and loads the best matching binary object
- This is a direct, upstreamable contribution to MLIR's gpu dialect
- Novelty: 5/5 -- this specific handler does not exist
- Feasibility: 4/5 -- extends existing MLIR infrastructure
- Impact: 5/5 -- directly usable by anyone using MLIR's gpu dialect
- LLVM-relevance: 5/5 -- this IS an LLVM/MLIR contribution

### Gap 2: IREE's full-stack approach <-> A lightweight alternative
IREE is powerful but monolithic. You must use IREE's compiler, VM, HAL, and runtime together. What if you want multi-target dispatch without IREE?

**Idea 15: `mlir-dispatch-lite` -- Lightweight multi-target dispatch without IREE**
- Position: "IREE is Kubernetes; we are a shell script"
- Same multi-target compilation via MLIR, but runtime is 500 lines of C
- No VM, no FlatBuffers, no HAL -- just device detection + kernel selection + launch
- Novelty: 4/5 -- lightweight alternative to IREE is underexplored
- Feasibility: 5/5 -- small scope, very achievable
- Impact: 4/5 -- attractive to people who don't want IREE's complexity
- LLVM-relevance: 5/5

### Gap 3: ALPAKA's template abstraction <-> MLIR's IR-level approach
ALPAKA achieves portability via C++ template metaprogramming at compile time. MLIR achieves portability via IR transformations. Nobody has bridged these: using MLIR to generate ALPAKA-style portable code, or using ALPAKA's abstraction model to inform MLIR's lowering choices.

**Idea 16: `alpaka-meets-mlir` -- Bridging ALPAKA's portability model with MLIR code generation**
- Define ALPAKA's parallelism hierarchy (grid/block/thread/element) as MLIR attributes
- Use these attributes to guide the `gpu` dialect lowering to different targets
- Effectively: ALPAKA's abstraction model, but at the IR level instead of C++ templates
- Leverages Akash's CERN GSoC experience with ALPAKA
- Novelty: 5/5 -- nobody has connected ALPAKA's model to MLIR
- Feasibility: 3/5 -- requires understanding both ALPAKA and MLIR deeply
- Impact: 4/5 -- bridges two communities (HPC + compiler)
- LLVM-relevance: 4/5

### Gap 4: Dynamic shapes <-> Static dispatch
Reviewer 91B says "ML kernels are static." But dynamic shapes (variable batch sizes, sequence lengths in LLMs) are increasingly common. TensorRT-RTX compiles "fallback" + "shape-specialized" kernels. Nobody does this across vendors via MLIR.

**Idea 17: `shape-aware-dispatch` -- Multi-target kernels specialized per shape at JIT time**
- At compile time: generate kernel variants for multiple targets AND shape ranges
- At runtime: observe actual tensor shapes, JIT-specialize or select best pre-compiled variant
- Addresses "ML kernels are static" by showing shapes are NOT static in modern LLM serving
- Novelty: 4/5 -- combining shape specialization with multi-target is new
- Feasibility: 3/5 -- JIT specialization adds complexity
- Impact: 4/5 -- directly answers Reviewer 91B
- LLVM-relevance: 4/5

---

## Framework 6: User-Centric

### User 1: ML engineer deploying to heterogeneous cloud
**Scenario:** Deploying a model on AWS with a mix of A100, H100, and T4 instances. Different instances have different GPUs. Model should work optimally on all.
**Need:** Compile once, deploy everywhere. Runtime picks the best kernel for each instance's GPU.

**Idea 18: `cloud-portable-inference` -- Demo: one MLIR binary, deployed across 3 GPU types**
- Compile a GEMM or attention kernel to sm_70 (T4), sm_80 (A100), sm_90 (H100) + CPU fallback
- Show: same binary, different performance characteristics, all near-optimal
- This is within NVIDIA's ecosystem but demonstrates the dispatch mechanism
- Novelty: 3/5 -- CUDA fat binaries do this, but doing it via MLIR is novel
- Feasibility: 5/5 -- straightforward benchmarking
- Impact: 4/5 -- immediately relatable to cloud ML engineers
- LLVM-relevance: 4/5

### User 2: HPC researcher on mixed-GPU cluster
**Scenario:** University cluster has nodes with V100s, A100s, and some AMD MI250s donated. Code should run on any node without recompilation.
**Need:** True cross-vendor portability at the binary level.

**Idea 19: `cross-vendor-fat-binary` -- MLIR-generated fat binaries containing NVPTX + AMDGCN + x86**
- Extend MLIR's fat binary format to include objects for multiple vendors (not just multiple SM versions)
- Runtime detects vendor and architecture, selects appropriate object
- This goes beyond CUDA's fat binary (which is NVIDIA-only) to a truly cross-vendor format
- Novelty: 5/5 -- cross-vendor fat binaries via MLIR don't exist
- Feasibility: 4/5 -- MLIR's `gpu-module-to-binary` already produces per-target objects; packaging them is engineering
- Impact: 5/5 -- this is the dream of heterogeneous computing
- LLVM-relevance: 5/5

### User 3: Edge ML developer targeting unknown hardware
**Scenario:** Shipping an ML model to edge devices where the GPU vendor/model is unknown at compile time.
**Need:** A binary that works on NVIDIA Jetson, Qualcomm Adreno, ARM Mali, or falls back to CPU.

**Idea 20: `edge-portable-dispatch` -- SPIR-V + CPU fallback for unknown edge hardware**
- Compile to SPIR-V (Vulkan) + x86/ARM CPU
- Runtime: try Vulkan first, fall back to CPU
- Measure: "How much performance do you sacrifice for total portability?"
- Novelty: 3/5 -- the question is interesting but IREE targets this
- Feasibility: 4/5 -- MLIR has both spirv and llvm backends
- Impact: 3/5 -- edge is less exciting for LLVM Dublin audience (more server/HPC focused)
- LLVM-relevance: 4/5

### User 4: ML framework maintainer adding new backend support
**Scenario:** Adding AMD GPU support to a PyTorch-based system. Currently CUDA-only.
**Need:** Minimize the effort to add a new backend. Ideally, add a single compilation target and get dispatch for free.

**Idea 21: `zero-effort-backend` -- Adding a new GPU target to an MLIR-based pipeline in <100 lines**
- Show: starting from an MLIR pipeline that targets NVPTX, add AMDGCN support by adding target attributes + a dispatch handler
- Measure the lines of code and engineering effort required
- Contrast with the effort of adding AMD support to PyTorch (thousands of lines of HIPification)
- Novelty: 3/5 -- the measurement/comparison is the contribution
- Feasibility: 5/5 -- mostly documentation + measurement
- Impact: 4/5 -- very practical
- LLVM-relevance: 5/5

---

## Framework 7: Minimum Viable Research

What is the smallest contribution that would be novel, impressive, and achievable in 2 days?

**Idea 22 (MVP): `#gpu.capability_select` handler + micro-dispatch runtime + GEMM benchmark**

The minimum viable research contribution combines:
1. A new MLIR offloading handler (`#gpu.capability_select`) that replaces the trivial `#gpu.select_object` with capability-aware selection (~200 lines of C++)
2. A micro-dispatch runtime in C (~300 lines) that:
   - Detects available GPUs via CUDA/HIP/Vulkan APIs
   - Reads compiled kernel capabilities from metadata
   - Selects the best matching variant
   - Launches with timing instrumentation
3. A benchmark showing a single `linalg.matmul` compiled to sm_80 + gfx90a + x86, dispatched correctly on available hardware
4. Performance comparison: our dispatch vs native cuBLAS/rocBLAS (showing the "portability tax")

This is:
- **Novel:** The `#gpu.capability_select` handler does not exist in MLIR
- **Concrete:** Working code, not a survey
- **Demonstrable:** Benchmark numbers on a poster
- **LLVM-relevant:** Directly extends MLIR's gpu dialect
- **Addresses all reviewers:**
  - 91A: concrete mechanism (the handler + runtime)
  - 91B: shows value beyond static dispatch (different hardware at runtime)
  - 91B: integration path via `torch.compile` backend or ONNX Runtime EP
  - 91C: clear proposal, not a survey
  - 91D: broader than SYCL; compares SPIR-V, HIP, multi-versioned JIT
  - 91D: acknowledges IREE's capabilities; positions as lightweight alternative

---

## Additional Creative Ideas

### Idea 23: `dispatch-as-linking` -- GPU kernel dispatch as a dynamic linking problem
Frame the entire problem using ELF/linking terminology:
- Compiled kernel variants = shared objects (`.so`)
- Kernel name = symbol
- Target architecture = SONAME/version
- Device capability = loader compatibility check
- Dispatch = `dlopen` + `dlsym` + call

This framing is deeply natural for LLVM Dublin audience. Implement a proof-of-concept "GPU kernel linker" that uses actual `dlopen` semantics.
- Novelty: 5/5
- Feasibility: 4/5
- Impact: 5/5
- LLVM-relevance: 5/5

### Idea 24: `profile-guided-dispatch` -- Use profiling data to optimize dispatch decisions
First run: execute on all available devices, measure performance per kernel.
Subsequent runs: dispatch to the fastest device per kernel based on profile data.
Analogous to profile-guided optimization (PGO) in compilers.
- Novelty: 4/5
- Feasibility: 3/5
- Impact: 3/5
- LLVM-relevance: 4/5

### Idea 25: `mlir-dispatch-dialect` -- A new MLIR dialect for expressing dispatch decisions
Define operations like:
```mlir
%device = dispatch.discover_devices {vendors = ["nvidia", "amd", "cpu"]}
%best = dispatch.select_device %device {min_compute_units = 64, min_shared_mem = 48KB}
dispatch.launch @my_kernel on %best (%arg0, %arg1) : (memref<1024x1024xf32>, memref<1024x1024xf32>)
```
This makes dispatch a first-class concept in MLIR.
- Novelty: 5/5 -- no dispatch dialect exists
- Feasibility: 3/5 -- designing a dialect is substantial
- Impact: 4/5 -- architecturally interesting
- LLVM-relevance: 5/5

### Idea 26: `vllm-hetero-dispatch` -- Heterogeneous GPU dispatch for LLM serving
Extend vLLM's architecture to use MLIR-compiled kernels that dispatch across mixed GPU types. Show: serving an LLM on a node with both A100 and MI250 simultaneously.
Leverages Akash's vLLM contributor experience.
- Novelty: 4/5
- Feasibility: 2/5 -- deep vLLM integration is heavy
- Impact: 5/5 -- LLM serving is the hottest topic
- LLVM-relevance: 3/5

### Idea 27: `sofie-mlir-bridge` -- Connecting ROOT TMVA-SOFIE's ONNX inference to MLIR dispatch
SOFIE generates C++ code from ONNX. Currently uses ALPAKA for GPU portability. What if SOFIE generated MLIR instead of C++, enabling multi-target compilation + runtime dispatch?
Leverages Akash's CERN GSoC experience directly.
- Novelty: 5/5 -- this bridge does not exist
- Feasibility: 2/5 -- requires significant SOFIE changes
- Impact: 3/5 -- niche audience (HEP + LLVM intersection)
- LLVM-relevance: 4/5

---

## Scoring Summary

| # | Idea | Novelty | Feasibility (2 days) | Impact | LLVM-relevance | **Total** |
|---|------|---------|----------------------|--------|----------------|-----------|
| 14 | `#gpu.capability_select` handler | 5 | 4 | 5 | 5 | **19** |
| 19 | Cross-vendor fat binary via MLIR | 5 | 4 | 5 | 5 | **19** |
| 23 | Dispatch-as-dynamic-linking | 5 | 4 | 5 | 5 | **19** |
| 4 | `mlir-kernel-linker` (dynamic linking) | 5 | 4 | 5 | 5 | **19** |
| 22 | MVP: handler + micro-dispatch + benchmark | 4 | 5 | 4 | 5 | **18** |
| 11 | `90-percent-portable` (portability tax) | 4 | 5 | 5 | 5 | **19** |
| 6 | Multi-target-by-default MLIR | 4 | 4 | 4 | 5 | **17** |
| 7 | Capability descriptors | 4 | 4 | 4 | 5 | **17** |
| 9 | `micro-dispatch` (<500 lines) | 4 | 5 | 4 | 4 | **17** |
| 15 | `mlir-dispatch-lite` (lightweight IREE alt) | 4 | 5 | 4 | 5 | **18** |
| 25 | `mlir-dispatch-dialect` | 5 | 3 | 4 | 5 | **17** |
| 16 | ALPAKA-meets-MLIR | 5 | 3 | 4 | 4 | **16** |
| 12 | Compiled dispatch tables | 4 | 4 | 4 | 4 | **16** |
| 2 | Cost-model dispatch | 4 | 3 | 4 | 4 | **15** |
| 1 | `mlir-ksched` (CFS-style) | 4 | 3 | 4 | 5 | **16** |
| 17 | Shape-aware dispatch | 4 | 3 | 4 | 4 | **15** |
| 18 | Cloud-portable inference demo | 3 | 5 | 4 | 4 | **16** |
| 21 | Zero-effort backend addition | 3 | 5 | 4 | 5 | **17** |
| 8 | Zero-change portability | 3 | 3 | 4 | 5 | **15** |
| 10 | MLIR-to-SPIR-V universal | 3 | 4 | 3 | 4 | **14** |
| 5 | Graceful dispatch (browser analogy) | 3 | 4 | 3 | 4 | **14** |
| 24 | Profile-guided dispatch | 4 | 3 | 3 | 4 | **14** |
| 26 | vLLM hetero-dispatch | 4 | 2 | 5 | 3 | **14** |
| 27 | SOFIE-MLIR bridge | 5 | 2 | 3 | 4 | **14** |
| 20 | Edge-portable dispatch | 3 | 4 | 3 | 4 | **14** |
| 3 | Kernel routing (network analogy) | 3 | 2 | 3 | 3 | **11** |
| 13 | Op-level dispatch | 3 | 2 | 3 | 4 | **12** |

---

## Top 5 Recommendations

### Rank 1: **Composite -- "Dynamic Kernel Linking for Heterogeneous GPUs via MLIR"** (Ideas 4+14+19+23)

**Elevator pitch:** "We treat GPU kernel dispatch as a dynamic linking problem. MLIR compiles kernels into cross-vendor fat binaries containing NVPTX, AMDGCN, and SPIR-V objects. At runtime, a lightweight kernel linker -- analogous to `ld.so` -- discovers available GPUs, matches kernel capability requirements to device features, and dispatches to the optimal variant. The dispatch overhead is <1us. We implement this as a new `#gpu.capability_select` offloading handler for MLIR's gpu dialect and a ~500-line C runtime."

**Why this wins:**
- The dynamic linking framing is novel, memorable, and natural for the LLVM audience
- Extends MLIR's existing `gpu-module-to-binary` infrastructure (not reinventing the wheel)
- Positions clearly against IREE ("IREE is a full OS; we are `ld.so`")
- Addresses every reviewer concern
- Akash's systems background (vLLM, ALPAKA) makes this credible
- Implementable in 2 days as a proof-of-concept

**Components:**
1. `#gpu.capability_select` -- new MLIR offloading handler (C++, ~200 lines)
2. `libkdl` (kernel dynamic linker) -- C runtime (~300 lines): device discovery, capability matching, dispatch
3. Cross-vendor fat binary format -- container for {NVPTX, AMDGCN, SPIR-V, x86} objects with capability metadata
4. Benchmark: GEMM on N targets, showing dispatch overhead + portability tax

---

### Rank 2: **"Quantifying the MLIR Portability Tax"** (Idea 11)

If implementation of Rank 1 proves too ambitious, this is a strong fallback that requires no new infrastructure -- just careful measurement.

**Pitch:** "We compile the same GEMM, elementwise, and reduction kernels via MLIR to NVPTX, AMDGCN, SPIR-V, and x86. We measure performance relative to hand-tuned vendor libraries (cuBLAS, rocBLAS). We answer the question every GPU programmer has: how much performance do you actually sacrifice for MLIR portability?"

---

### Rank 3: **"ALPAKA's Portability Model Meets MLIR's Code Generation"** (Idea 16)

Unique to Akash's experience. Nobody else at LLVM Dublin would have both ALPAKA (CERN) and MLIR backgrounds.

**Pitch:** "ALPAKA achieves GPU portability via C++ templates. MLIR achieves it via IR transformations. We formalize ALPAKA's parallelism hierarchy (grid/block/thread/element) as MLIR attributes and show how they can guide target-specific lowering, getting the best of both worlds."

---

### Rank 4: **"Shape-Aware Multi-Target Dispatch for LLM Serving"** (Idea 17 + 26)

Connects to the hottest topic (LLMs) and answers Reviewer 91B's "kernels are static" objection.

**Pitch:** "In LLM serving, batch sizes and sequence lengths vary per request. We show that runtime dispatch must consider both hardware capabilities AND input shapes, compiling shape-specialized kernel variants per target and selecting at inference time."

---

### Rank 5: **"A Dispatch Dialect for MLIR"** (Idea 25)

Most architecturally ambitious. High risk, high reward. Would be a memorable poster if executed well.

**Pitch:** "We propose `dispatch` -- a new MLIR dialect that makes device discovery, capability matching, and kernel routing first-class IR concepts, enabling the compiler to reason about dispatch decisions alongside computation."

---

## Addressing Reviewer Concerns (mapped to top recommendation)

| Reviewer Concern | How "Dynamic Kernel Linking" Addresses It |
|------------------|-------------------------------------------|
| 91A: Need concrete mechanism | The `#gpu.capability_select` handler + `libkdl` runtime IS the mechanism |
| 91B: ML kernels are static | Fat binaries enable runtime selection when deploying to unknown/mixed hardware; LLM serving has dynamic shapes requiring runtime decisions |
| 91B: Connect to PyTorch/TF | `libkdl` can serve as a `torch.compile` custom backend or ONNX Runtime EP; show integration sketch |
| 91C: Survey vs proposal | This is a proposal with working code |
| 91C: Too specific/jargony | Frame using dynamic linking analogy -- every systems programmer understands `ld.so` |
| 91D: Why SYCL specifically? | We DON'T focus on SYCL. We compare NVPTX, AMDGCN, SPIR-V, x86 as targets within MLIR |
| 91D: IREE SPIR-V is vendor-agnostic | Yes, and IREE is excellent. We provide a lightweight alternative for those who don't need IREE's full stack |
| 91D: Multi-versioned kernels at JIT time | Our fat binary IS multi-versioned; can extend with JIT specialization as future work |

---

## Research Sources

- [MLIR GPU Dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
- [IREE deployment configurations](https://iree.dev/guides/deployment-configurations/)
- [The Long Tail of AI: SPIR-V in IREE and MLIR (Vulkanised 2025)](https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf)
- [CUDA fat binary understanding (NVIDIA blog)](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
- [Mojo: MLIR-Based Performance-Portable HPC Science Kernels](https://arxiv.org/html/2509.21039v1)
- [GPU Compilation with MLIR (Stephen Diehl)](https://www.stephendiehl.com/posts/mlir_gpu/)
- [ALPAKA documentation](https://alpaka.readthedocs.io/en/latest/basic/intro.html)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [Adaptive Runtime Selection for GPU (Dollinger et al.)](https://ieeexplore.ieee.org/document/6687340/)
- [KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization](https://arxiv.org/html/2603.12440)
- [Dynamic Kernel Selection for Real-Time ML Inference](https://www.preprints.org/manuscript/202510.0674/v1/download)
- [DISC: Dynamic Shape Compiler for ML Workloads](https://arxiv.org/pdf/2103.05288)
- [vLLM Triton backend for NVIDIA and AMD (PyTorch blog)](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/)
- [Evaluative Comparison of Performance Portability across GPU Programming Models](https://arxiv.org/html/2402.08950v1)
- [IREE design roadmap](https://iree.dev/developers/design-docs/design-roadmap/)
- [gpu-module-to-binary MLIR pass review](https://reviews.llvm.org/D154149)
- [NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/pdf/CUDA_Binary_Utilities.pdf)
- [Triton language and compiler (GitHub)](https://github.com/triton-lang/triton)
- [PyTorch torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Single-node ML Runtime Foundation (Lei.Chat)](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)
