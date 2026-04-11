# Novelty Gap Analysis: Heterogeneous GPU Kernel Dispatch via MLIR

**Date:** 2026-04-02
**Purpose:** Identify what does NOT exist for LLVM Dublin 2026 poster positioning.

---

## Gap 1: MLIR Pass Emitting Multi-Versioned Kernels with Runtime Selection

### Status: PARTIALLY EXISTS (in IREE), NOT in upstream MLIR

**What exists:**
- MLIR's `gpu-module-to-binary` pass (D154149) serializes GPU modules with multiple target attributes, producing a `gpu.binary` containing objects for every target (e.g., NVPTX sm_70 + AMDGCN gfx90a).
- `gpu.select_object` selects which target object to embed -- but this is a **compile-time/link-time** selection mechanism, NOT runtime hardware detection. The default handler "selects the first object from the array and embeds it as a string."
- IREE issue #15334 is an **open epic** titled "Support for target and strategy multi-versioning." It explicitly states: "Currently, the only well-worn path for the compiler today requires the user to specify precisely the target device they are compiling for." The issue is open and lists significant unfinished work.
- IREE issue #12230 tracks "kernel specialization and multi-versioning" with planned step 2b: "Generate runtime kernel selection logic" -- described as "sort of broken" currently.
- IREE issue #50 (open since project inception) requests target configuration with runtime matching: "At runtime we should then match against those to select the best suited for the given runtime configuration."

**What does NOT exist:**
- No upstream MLIR pass that emits multi-versioned kernels AND generates runtime selection logic.
- No mechanism in upstream MLIR to do runtime hardware capability querying to choose among pre-compiled kernel variants.
- IREE's planned runtime selection logic (issue #12230 step 2b) is explicitly unfinished. They note: "right now this part is sort of broken."

**Why the gap exists:** IREE is approaching this incrementally within their full-stack runtime. The upstream MLIR infrastructure provides the building blocks (`gpu.binary`, target attributes) but stops short of runtime dispatch logic. This is technically hard because it requires coupling compiler output format with a runtime query protocol.

**Assessment: STRONG NOVELTY OPPORTUNITY.** A lightweight pass + runtime shim that bridges the gap between MLIR's multi-target binary support and actual runtime dispatch would be novel. IREE wants this but hasn't built it. Upstream MLIR doesn't have it.

---

## Gap 2: Combining ALPAKA's Abstraction Model with MLIR Code Generation

### Status: DOES NOT EXIST

**What exists:**
- ALPAKA is a header-only C++20 abstraction library supporting CUDA, HIP, SYCL, OpenMP, std::thread backends for portable kernel acceleration.
- MLIR has its own GPU dialect, linalg dialect, etc. for code generation.
- CERN's TMVA-SOFIE uses ALPAKA for heterogeneous GPU inference of ONNX models (GSoC 2025 project: "TMVA SOFIE - GPU Support for Machine Learning Inference"). SOFIE generates C++ code with ALPAKA calls for cuBLAS/rocBLAS backends.
- No search results, papers, or projects combine ALPAKA's abstraction layer with MLIR's compiler infrastructure.

**What does NOT exist:**
- No MLIR dialect or pass that lowers to ALPAKA abstractions.
- No compiler pipeline that uses MLIR for optimization and ALPAKA for portable backend dispatch.
- No academic paper exploring ALPAKA + MLIR integration.

**Why the gap exists:** These are fundamentally different approaches to the same problem. ALPAKA is a library-level abstraction (C++ templates), while MLIR is a compiler-level abstraction (IR transformations). They operate at different layers of the stack. Nobody has tried to bridge them because MLIR's own GPU dialect already targets the same backends ALPAKA does, making the integration seem redundant -- but the dispatch model is different. ALPAKA's runtime portability model (single source, runtime backend selection) is architecturally distinct from MLIR's compile-time target selection.

**Assessment: MODERATE NOVELTY but possibly low impact.** The question is whether bridging these layers provides value beyond what each does independently. The interesting angle: could MLIR-compiled kernels be wrapped in ALPAKA's runtime dispatch model to get both compiler optimization AND runtime portability?

---

## Gap 3: Lightweight Runtime (~500 LOC) for GPU Capability Query + MLIR Kernel Routing

### Status: DOES NOT EXIST as a standalone component

**What exists:**
- MLIR's ExecutionEngine provides JIT invocation but fixes targets at compilation time.
- IREE's HAL (Hardware Abstraction Layer) provides runtime device queries and dispatch, but it is a heavyweight runtime (~100K+ LOC) deeply coupled to IREE's compilation model.
- IREE issue #12230 step 3 envisions "query the device parallelism and then using host logic to compute tile/workgroup sizes" but notes "cross-vendor parallelism querying extensions is non-existing for Vulkan right now."
- chipStar creates fat binaries that can run on any SPIR-V-capable device, but it's a full HIP/CUDA compatibility layer, not a lightweight dispatch shim.

**What does NOT exist:**
- No standalone, minimal runtime that: (1) queries GPU vendor/capabilities at startup, (2) selects from pre-compiled MLIR kernel variants, (3) dispatches to the appropriate backend.
- No thin shim between MLIR's `gpu.binary` multi-target output and actual device-specific invocation.
- The gap between MLIR's compilation and execution is currently filled only by full-stack solutions (IREE, PyTorch, etc.) or manual glue code.

**Why the gap exists:** Building a lightweight runtime requires defining a clear interface contract between the compiler's output format and the runtime's dispatch logic. IREE chose to build a comprehensive runtime instead. Nobody has attempted the minimal version because MLIR's current users are either full-stack projects (IREE, Torch-MLIR) or research prototypes that target a single backend.

**Assessment: STRONGEST NOVELTY OPPORTUNITY.** This is the most defensible contribution for the poster. A ~500 LOC runtime that: queries CUDA/HIP/Vulkan/CPU capabilities, loads the appropriate object from a `gpu.binary`, and dispatches -- would be genuinely novel and immediately useful. It fills a gap that IREE acknowledges but approaches with a heavyweight solution.

---

## Gap 4: SPIR-V as Universal Intermediate for MLIR -> {CUDA, HIP, CPU} Dispatch

### Status: PARTIALLY EXISTS but with significant limitations

**What exists:**
- MLIR has a SPIR-V dialect and can lower GPU operations to SPIR-V.
- MLIR's `mlir-spirv-cpu-runner` prototype (D86108) runs SPIR-V on CPU via GPU->SPIRV->LLVM conversion.
- chipStar compiles HIP/CUDA to SPIR-V, runs via OpenCL or Level Zero. Performance: geometric mean 0.75x vs native HIP. Published in IJHPCA 2026.
- IREE uses SPIR-V via Vulkan as one backend, generating "vendor-agnostic code given specified hardware features" (reviewer 91D's point).
- clvk provides OpenCL 3.0 on top of Vulkan using clspv compiler.
- PoCL supports SPIR-V for CPU and CUDA drivers.
- LLVM RFC: "SPIRV IR as a vendor agnostic GPU representation" (discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115) proposes using SPIR-V IR in LLVM as a convergence point.

**What does NOT exist:**
- No single system that takes MLIR input and produces SPIR-V that dispatches to CUDA, HIP, AND CPU through a unified runtime.
- chipStar goes HIP/CUDA -> SPIR-V -> OpenCL/Level Zero (reverse direction from what we want).
- IREE's SPIR-V path goes through Vulkan only, not directly to CUDA/HIP.
- The SPIR-V CPU runner is a prototype, not production-ready.
- No unified "SPIR-V as universal dispatch IR" pipeline exists end-to-end.

**Why the gap exists:** SPIR-V was designed for Vulkan/OpenCL consumption, not as a universal GPU IR. Using it to target CUDA requires translation layers (chipStar approach) with performance penalties (~25% overhead). The LLVM RFC for SPIR-V as vendor-agnostic GPU representation is active but not implemented. The fundamental tension: SPIR-V lacks CUDA/HIP-specific features (tensor cores, shared memory semantics), making it lossy as a universal intermediate.

**Assessment: GAP EXISTS BUT TECHNICALLY CONSTRAINED.** The 25% performance overhead from chipStar shows the cost of SPIR-V universality. A poster contribution here would need to honestly address performance tradeoffs. Better angle: use SPIR-V for the "portable fallback" tier while keeping native NVPTX/AMDGCN for performance-critical paths.

---

## Gap 5: Capability-Aware JIT Specializing MLIR Kernels Based on Runtime Hardware Queries

### Status: PARTIALLY EXISTS in concept, NOT implemented end-to-end

**What exists:**
- MLIR's ExecutionEngine uses LLVM's ORC JIT. JIT specialization to target hardware is "fully supported" in principle.
- MLIR documentation mentions: "JIT'ing at runtime or install-time once the full characteristics of the target are known."
- libxsmm uses a two-stage JIT (dispatch shape -> compile -> cache function pointer) for micro-kernels.
- IREE issue #12230 step 3 envisions runtime parallelism querying to adjust tile sizes, but acknowledges "cross-vendor parallelism querying extensions is non-existing for Vulkan."
- Triton JIT-compiles Python-decorated kernels to GPU code with hardware-aware tuning.

**What does NOT exist:**
- No system that: (1) takes MLIR IR at runtime, (2) queries the actual GPU's compute capability/SIMD width/memory hierarchy, (3) specializes the MLIR pipeline (tile sizes, vectorization widths, memory hierarchy usage) based on those queries, (4) JIT-compiles to the detected target.
- Triton does hardware-aware JIT but from its own DSL, not from MLIR IR.
- IREE's vision for this (issue #12230 step 3) is explicitly future work.
- No "plug in your MLIR module, get hardware-optimized kernel" system exists.

**Why the gap exists:** This requires coupling three hard problems: hardware capability detection across vendors, parameterized MLIR compilation pipelines, and efficient JIT with caching. Each piece exists in isolation (CUDA's `cudaGetDeviceProperties`, MLIR's transform dialect for parameterized pipelines, ORC JIT) but nobody has wired them together. IREE is the closest but plans to embed hardware knowledge in a static database rather than dynamic queries.

**Assessment: GENUINE GAP, HIGH IMPACT.** This is the reviewer 91D suggestion ("multi-versioned kernels specialized at JIT time by querying hardware features"). A prototype demonstrating even a subset (e.g., query compute capability -> select tile size -> JIT compile MLIR matmul) would be novel and directly responsive to reviewer feedback.

---

## Gap 6: "GPU Fat Binary" Format Bundling NVPTX + AMDGCN + SPIR-V + x86 from MLIR

### Status: PARTIALLY EXISTS in MLIR, NOT complete

**What exists:**
- MLIR's `gpu.binary` can contain multiple `gpu.object` entries for different targets (NVVM, ROCDL demonstrated).
- CUDA fat binaries (fatbin) bundle multiple PTX/CUBIN for different SM architectures -- but NVIDIA-only.
- HIP fat binaries exist for AMD -- but AMD-only.
- The `gpu-module-to-binary` pass produces a binary "with an object for every target."

**What does NOT exist:**
- No demonstrated `gpu.binary` bundling NVPTX + AMDGCN + SPIR-V + x86 together. Tests only show NVVM + NVVM (different SM) or NVVM + ROCDL combinations.
- No standard format for a cross-vendor fat binary with runtime selection across GPU vendors AND CPU fallback.
- No x86/CPU object bundled alongside GPU objects in `gpu.binary` (CPU execution goes through MLIR's standard LLVM lowering, separate from GPU binary flow).
- IREE's `.vmfb` format bundles compiled artifacts but is IREE-specific and not a general-purpose fat binary format.
- IREE issue #15334 explicitly asks for single-artifact multi-target deployment but this is unfinished.

**Why the gap exists:** MLIR's `gpu.binary` was designed for GPU targets. CPU execution follows a completely different path (LLVM lowering, not GPU module serialization). Unifying these into a single artifact requires bridging two separate compilation pipelines. Additionally, SPIR-V targets in `gpu.binary` require Vulkan runtime, adding another dimension. Nobody has unified all four targets because each has different runtime requirements.

**Assessment: MODERATE NOVELTY.** Defining and demonstrating a true cross-vendor+CPU fat binary format from MLIR would be useful but is more of an engineering contribution than a research one. The interesting research question: what is the minimal runtime protocol to select and launch from such a bundle?

---

## Gap 7: MLIR-Compiled Kernels as PyTorch Custom Backend or ONNX Runtime Execution Provider

### Status: PARTIAL -- PyTorch bridge exists, ONNX Runtime EP does NOT

**What exists for PyTorch:**
- Torch-MLIR provides first-class support bridging PyTorch and MLIR ecosystems.
- `torch.compile` custom backend API allows registering external compilers (gm: GraphModule, inputs) -> Callable.
- iree-turbine integrates IREE + torch-mlir + PyTorch for deployment.
- TorchFuser (proposed May 2025) is an MLIR-based compiler targeting transformer fusion, designed as a torch.compile backend. Early-stage proposal, no public implementation.
- Cerebras, Tenstorrent, and others use torch-mlir as their PyTorch integration layer.

**What exists for ONNX Runtime:**
- ONNX-MLIR compiles ONNX models to native code via MLIR, but is a standalone compiler, NOT an ONNX Runtime execution provider.
- LLVM Discourse thread (May 2025): "Is There Existing Work to add ONNX Runtime Execution Provider based on MLIR or LLVM?" -- indicating this is an open question.
- IREE can import ONNX via torch-mlir's `iree-import-onnx` but doesn't plug into ONNX Runtime as an EP.
- A separate Discourse thread (2026): "Early evaluators for a portable compute IR (ONNX -> CPU/GPU via OpenCL)" -- another angle at this problem.

**What does NOT exist:**
- No ONNX Runtime execution provider that uses MLIR/LLVM for code generation (the Discourse thread confirms this gap).
- No system that takes MLIR-compiled multi-target kernels and exposes them through PyTorch's dispatcher with runtime device selection.
- TorchFuser targets multi-GPU (NVIDIA, AMD, Apple MLX) but is vaporware as of April 2026.

**Why the gap exists:**
- For PyTorch: torch-mlir and iree-turbine bridge compilation, but don't address runtime multi-target dispatch. Each deployment targets a single backend.
- For ONNX Runtime: ONNX-MLIR is a standalone compiler; wrapping it as an EP requires implementing the EP interface (~significant engineering). Nobody has prioritized this because ONNX Runtime already has vendor-specific EPs (CUDA EP, TensorRT EP, etc.).

**Assessment: ONNX Runtime EP is a CLEAR GAP. PyTorch integration is crowded but the multi-target dispatch angle is novel.** An MLIR-based ONNX Runtime EP that dynamically selects between CPU/CUDA/HIP backends would be novel and directly connects to reviewer 91B's feedback about connecting to real ML frameworks.

---

## Summary: Novelty Ranking for Poster

| Gap | Novelty | Impact | Feasibility | Poster Score |
|-----|---------|--------|-------------|-------------|
| 3. Lightweight runtime for MLIR kernel routing | HIGH | HIGH | HIGH | **Best** |
| 5. Capability-aware JIT for MLIR kernels | HIGH | HIGH | MEDIUM | **Strong** |
| 1. Multi-versioned MLIR pass + runtime selection | HIGH | HIGH | MEDIUM | **Strong** |
| 7. ONNX Runtime EP via MLIR | HIGH | MEDIUM | MEDIUM | Good |
| 4. SPIR-V universal dispatch | MEDIUM | MEDIUM | LOW | Moderate |
| 6. Cross-vendor fat binary format | MEDIUM | MEDIUM | MEDIUM | Moderate |
| 2. ALPAKA + MLIR integration | MEDIUM | LOW | LOW | Weak |

## Recommended Poster Contribution

**Primary:** Gap 3 -- A lightweight (~500 LOC) runtime that queries GPU capabilities and routes MLIR-compiled kernels. This:
- Is genuinely novel (IREE does this with 100K+ LOC; nobody has done it minimally)
- Addresses reviewer 91A's demand for a "concrete mechanism"
- Addresses reviewer 91D's suggestion about multi-versioned kernel JIT dispatch
- Is feasible to prototype before the deadline

**Supporting:** Gap 1 + Gap 5 -- Show the runtime working with MLIR's `gpu.binary` multi-target support and demonstrate capability-aware selection. This connects the compiler infrastructure (which exists) to runtime dispatch (which doesn't).

**Framework connection (reviewer 91B):** Demonstrate the runtime dispatching kernels invoked from PyTorch via torch-mlir, addressing "connect to PyTorch/TF ecosystem."

---

## Key Sources

- IREE issue #50: Target configuration and runtime matching (open since project start)
  https://github.com/iree-org/iree/issues/50
- IREE issue #15334: Epic for target and strategy multi-versioning (open)
  https://github.com/iree-org/iree/issues/15334
- IREE issue #12230: Kernel specialization and multi-versioning (open)
  https://github.com/iree-org/iree/issues/12230
- MLIR GPU dialect docs: gpu.binary, gpu.select_object, gpu-module-to-binary
  https://mlir.llvm.org/docs/Dialects/GPU/
- D154149: gpu-module-to-binary pass implementation
  https://reviews.llvm.org/D154149
- chipStar: HIP/CUDA on SPIR-V (0.75x geometric mean vs native)
  https://github.com/CHIP-SPV/chipStar
- LLVM RFC: SPIR-V IR as vendor-agnostic GPU representation
  https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- LLVM Discourse: ONNX Runtime EP via MLIR/LLVM (open question, May 2025)
  https://discourse.llvm.org/t/is-there-existing-work-to-add-onnx-runtime-execution-provider-based-on-mlir-or-llvm/86383
- Torch-MLIR project
  https://github.com/llvm/torch-mlir
- ONNX-MLIR compiler
  https://github.com/onnx/onnx-mlir
- TorchFuser proposal (May 2025)
  https://dev-discuss.pytorch.org/t/torchfuser-a-plug-and-play-mlir-based-compiler-and-optimized-runtime-integration/2972
- CERN TMVA-SOFIE GPU support with ALPAKA (GSoC 2025)
  https://hepsoftwarefoundation.org/gsoc/2025/proposal_TMVA-SOFIE-GPU.html
- MLIR SPIR-V CPU runner prototype
  https://reviews.llvm.org/D86108
- chipStar IJHPCA 2026 paper
  https://journals.sagepub.com/doi/10.1177/10943420261423001
- IREE deployment configurations
  https://iree.dev/guides/deployment-configurations/
- Vulkanised 2025: SPIR-V in IREE and MLIR (Jakub Kuderski, AMD)
  https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf
