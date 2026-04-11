# IREE 2025-2026 State: Multi-Target Dispatch, Runtime Kernel Selection, and GPU Backend Unification

**Research date:** 2026-04-06
**Context:** LLVM Dublin 2026 poster — Heterogeneous GPU Kernel Dispatch via libkdl
**Researcher:** Akash
**Sources:** GitHub issue tracker, official IREE docs, Vulkanised 2025 talk (Kuderski/AMD), Roofline.AI case study, MLIR talks archive, release changelogs v3.6–v3.10

---

## Quick Reference Scores

| Dimension | Score (1–10) | Rationale |
|-----------|-------------|-----------|
| Technical relevance | 10 | IREE is the direct prior art for every core claim in our poster |
| Approach overlap | 7 | IREE's HAL variant model is analogous to libkdl's dispatch table but at a higher abstraction level |
| Citation priority | 10 | Must cite; three open issues are the textbook gap statement for our contribution |
| Gap-filling value | 9 | Our work directly addresses what Issues #50, #12230, #15334 describe as unsolved |

---

## 1. Problem (What IREE Is Trying to Solve)

IREE (Intermediate Representation Execution Environment) is an MLIR-based end-to-end compiler and runtime for ML model deployment. It targets a fundamentally hard problem: producing a single deployable artifact that runs efficiently across heterogeneous hardware — CPU, GPU (NVIDIA, AMD, ARM Mali, Apple, Qualcomm), and NPU — without requiring users to recompile per target.

The core tension IREE exposes, and which our work directly inherits, is:

> "Portability via a common IR (SPIR-V/Vulkan) sacrifices peak performance; peak performance (PTX/AMDGPU) sacrifices portability. A system that provides both must carry multiple compiled variants and make runtime dispatch decisions — but neither the compiler infrastructure nor the runtime policy for doing this is complete."

This has been the recognized goal since Issue #50 was filed in October 2019 by Ben Vanik (IREE co-creator), and it remains open as of April 2026 — six years and four months later.

---

## 2. Contribution (What IREE Has Built)

### 2.1 The Dialect Pipeline

IREE's compiler passes programs through three dialects in sequence:

**Flow dialect** — partitions tensor computations into dispatch regions (groups of dense ops), deduplicates structurally identical dispatches, and outlines them into executables. Key pass: `DeduplicateExecutables`.

**Stream dialect** — converts to explicitly scheduled asynchronous programs. Assigns `#stream.affinity` attributes to operations. This is where device placement decisions live. Tensors are encoded into target-specific resources. Key ops: `stream.cmd.dispatch`, `stream.async.execute`, `stream.async.concurrent`. Every op that carries affinity implements the `Stream_AffinityOp` interface.

**HAL dialect** — models a low-level hardware abstraction (Vulkan-like minus graphics). Manages buffers, command buffers, executables, semaphores. Each `hal.executable` contains one or more `hal.executable.variant` ops, each compiled for a specific target. This is the fat-binary model.

### 2.2 Executable Variant Model (The Fat Binary)

This is IREE's most mature mechanism for multi-target support. Structure:

- One `hal.executable` contains N `hal.executable.variant` ops (one per target, e.g., SPIR-V, CUDA PTX, CPU ELF)
- Each variant has an optional **condition op** that queries the runtime `!hal.device` for compatibility
- At module load time: variants are evaluated in declared order; **the first valid variant wins**
- Variant conditions can call `hal.device.query` to check extensions, architecture features, capability bits

The `.vmfb` (VM FlatBuffer) file format encodes this structure as a zip-like archive. Opening a `.vmfb` as a zip reveals files like `module.fb` (VM bytecode) plus target-specific executables such as `abs_dispatch_0_system_elf_x86_64.so` or `shader_0_vulkan_spirv.spv`. Multiple GPU/CPU variants coexist in a single artifact.

### 2.3 HAL Compiler Passes for Devices (2024–2025 additions)

The HAL pass pipeline now includes explicit multi-device management:

| Pass | Role |
|------|------|
| `-iree-hal-assign-target-devices` | Assigns `hal.device.targets` from CLI spec; supports multiple named devices |
| `-iree-hal-materialize-target-devices` | Creates global device handle ops per target |
| `-iree-hal-initialize-devices` | Builds runtime initializers that enumerate and select devices |
| `-iree-hal-resolve-device-aliases` | Expands `#hal.device.alias` to concrete targets |
| `-iree-hal-resolve-device-promises` | Maps `#hal.device.promise<@name>` to materialized globals |
| `-iree-hal-memoize-device-selection` | Hoists repeated device-selection decisions into globals (one-time evaluation) |
| `-iree-hal-memoize-device-queries` | Hoists `hal.device.query` ops into startup-initialized globals |

The `#hal.device.promise` mechanism (new in 2024) allows a program to declare that a named device handle will be provided at runtime without knowing its concrete type at compile time — enabling runtime device injection into pre-compiled artifacts.

### 2.4 Shape-Based Deduplication (Implemented, Highly Effective)

The most successful multi-versioning work in IREE is Phase 1 of Issue #12230: erasing static shape information after dispatch region formation, then deduplicating dispatches with identical operation sequences. Concrete results from Lei Zhang's benchmark (February 2023):

| Model | Executables (no dedup) | Executables (after dedup) | Reduction |
|-------|----------------------|--------------------------|-----------|
| TF ResNet50 f32 | 75 | 20 | 3.75x |
| TF EfficientNetV2 Small f32 | 278 | 30 | 9.27x |
| TF BERT Large f32 | 413 | 14 | 29.5x |
| TF MobileBERT i8 | 1452 | 162 | 8.96x |
| TF MobileBERT f32 | 752 | 20 | 37.6x |

These reductions are purely compile-time — they do not add any runtime dispatch capability. The residual counts are limited by fusion variants and load/store pairing, not by the approach itself.

### 2.5 GPU Backend Unification (Issue #16341, In Progress)

Opened February 2024, this epic tracks making LLVMGPU (CUDA/ROCm) and SPIR-V (Vulkan/Metal/WebGPU) more architecturally consistent. Status as of April 2026:

**Completed (~43% of tasks):**
- PR #17217: Thread through common target description across backends
- PR #17451: Support LLVMGPU backend target features under `#iree_gpu.target`
- PR #17623: Switch SPIR-V backend to use the common `#iree_gpu.target` attribute
- PR #17710: Make CUDA/HIP/Vulkan target CLI options consistent (unified flag surface)
- PR #17816: Push GPU target conversion before SPIR-V conversion in pipeline
- PR #16342: Create basic ROCDL CodeGen directory structure (CUDA/ROCm decoupling)

**Remaining (not started):**
- Support GPU multi-targeting for HIP, CUDA, SPIR-V (0/6 subtasks complete)
- Change SPIR-V/HIP/CUDA default target to null (prerequisite for multi-targeting)
- Port ROCDL pipelines; rename LLVMGPU to NVVM configurations
- Avoid fixed subgroup size for SPIR-V
- Replace hardcoded heuristics with runtime probing

The unified `#iree_gpu.target` attribute is now the canonical target descriptor across all GPU backends. It contains canonical architecture (e.g., `sm_80`, `gfx942`), workgroup processor attributes, and optional chip-level features. This replaces the previous fragmented per-backend representations (`vk.target_env`, CUDA triple strings, ROCm GFX identifiers).

### 2.6 Heterogeneous Execution Infrastructure (New in 2025)

Two significant developments in 2025:

**Roofline.AI upstream contributions (PRs #20885, #21005, #21029 — December 2025):** Infrastructure for portable asynchronous cross-device (CPU + GPU + NPU) execution built and upstreamed. Key capabilities:
- Shared CPU-GPU memory with hardware-architecture-aware compiler support
- Zero-copy data handoff between device kernels
- Unified async coordination layer using `!stream.timepoint` respecting data dependencies
- Validated on NXP, Apple, and Qualcomm edge SoCs running Qwen3-0.6B/1.7B

**`#hal.device.optimal` attribute:** An experimental attribute on allocation operations (documented in HAL pass reference 2025) that defers device selection to runtime. The compiler either statically resolves these using topology information and DeviceAnalysis (checking if two affinities refer to the same device) or leaves them unmodified for runtime resolution. This is the first step toward truly dynamic device selection at the allocation granularity.

### 2.7 Multi-GPU Tensor Parallelism (Shortfin LLM Server, 2025)

The Shortfin project (SHARK's inference serving layer atop IREE) implemented multi-GPU tensor parallelism for LLM inference. Compilation targets multiple devices explicitly:

```
--iree-hal-target-device=hip[0]
--iree-hal-target-device=hip[1]
```

Issue #19639 tracked a bufferization failure in decode with `tensor_parallelism==2` on AMD GPUs. Resolved via PR #19670 (`[LLVMGPU] Use LLVMGPUDistribute for small input scatters`). The fix addresses how small input scatters are distributed across workgroups, preventing `tensor.extract_slice` from folding away at the wrong time during bufferization.

This demonstrates that IREE's multi-device path is operational for homogeneous multi-GPU setups (two AMD HIPs), though it requires explicit device index specification at compile time rather than runtime adaptation.

---

## 3. Methodology (How IREE Does Runtime Dispatch)

### 3.1 Variant Selection Protocol

The runtime variant selection algorithm is:

1. At module load, iterate through all `hal.executable.variant` entries in declaration order
2. For each variant, evaluate its condition op (if present) against the bound `!hal.device`
3. First variant whose condition returns `true` is selected; all subsequent variants are ignored
4. If zero variants match, module loading fails with an error (no fallback, no retry)

This is a **first-match, static-order** policy. There is no:
- Cost model ranking between multiple valid variants
- Runtime profiling or adaptive selection
- Problem-size-aware dispatch (variant choice is independent of input shapes/sizes)
- Cross-device load balancing

### 3.2 Device Affinity in the Stream Dialect

`#stream.affinity` is an attribute on stream-level ops that propagates through the compiler:

```
stream.cmd.dispatch @executable::@entry on(%device_affinity)
stream.async.execute on(#stream.affinity<@gpu>) { ... }
```

Lowering to HAL maps affinities to `!hal.device` + queue affinity bitmask. Queue affinity selects a specific execution queue within a device. The compiler's job is to assign coherent affinities so that buffer ownership and synchronization are well-defined.

The key design choice: a logical `hal.device` may map to multiple physical devices with a unified address space. Multi-queue on one device is "functionally equivalent from the compiler's perspective to multiple physical devices exposed as a single logical device." This simplifies the compiler but means IREE's heterogeneous dispatch is still fundamentally a compile-time partitioning problem, not a runtime scheduling problem.

### 3.3 The VMFB Multi-Target Bundle Format

A `.vmfb` file is a zip archive containing:
- `module.fb`: FlatBuffer-encoded VM bytecode (host scheduling logic)
- One or more target-specific binary blobs per executable variant (`.so`, `.spv`, `.ptx`, `.hsaco`, etc.)
- Metadata: function signatures, exported entry points, module name

The structure enables bundling N targets in one artifact. The runtime reads the FlatBuffer index at load time, evaluates variant conditions, and memory-maps only the selected variant binary. Unselected variants consume space in the archive but zero runtime memory. This is analogous to a universal binary (Apple's fat Mach-O) or a CUDA fatbin.

**Known limitation:** SPIR-V variants are not designed to be compact. "SPIR-V is not designed to be a small at-rest format" (IREE design roadmap). Fat binaries with N variants explode in archive size. No compression or variant-level deduplication is applied.

---

## 4. Results (What Actually Works as of April 2026)

### 4.1 Confirmed Working

- Compiling a single `.vmfb` for multiple backends (e.g., `--iree-hal-target-device=local --iree-hal-target-device=vulkan`) produces a fat binary with variants for each target
- CPU multi-versioning (ISA-level: SSE4.2 vs AVX2 vs AVX-512) works via `hal.executable.variant` + condition ops (Issue #3768 closed November 2023)
- RISC-V runtime hardware detection for f16 kernels (PR #22231, v3.9.0 November 2024): `mmt4d` ukernel with `zvfh/zvfhmin` support, using `hal.device.query` to probe ISA extensions
- Multi-GPU homogeneous tensor parallelism (AMD HIP[0] + HIP[1]) works for LLM inference with Shortfin (PR #19670, early 2025)
- CPU-GPU-NPU asynchronous heterogeneous execution on edge SoCs (PRs #20885, #21005, #21029, December 2025)
- MLPerf Inference v5.0 (April 2025): AMD submitted SDXL on MI325X using IREE + Shortfin, demonstrating production-grade dispatch via the ROCm backend with the SHARK tuner achieving ~10% improvement over IREE's default codegen heuristics

### 4.2 Confirmed Not Working (as of April 2026)

The following remain explicitly open and unimplemented:

**Issue #50** (filed October 2019, OPEN): Unified cross-backend target configuration with runtime best-match selection. After 6.5 years, still no implementation beyond per-backend partial solutions.

**Issue #12230** (filed February 2023, OPEN, deprioritized P2): Multi-pipeline codegen (Phase 2a/2b) and dynamic runtime adjustment (Phase 3). Only Phase 1 (shape deduplication) landed.

**Issue #15334** (filed October 2023, OPEN): All 15+ task checkboxes remain unchecked. This includes:
- Specifying `hal.device.targets` as explicit CLI args for multi-versioning (not the same as existing `--iree-hal-target-device`)
- `MaterializeExecutableVariantsPass` for strategy expansion
- Trinary SPIR-V capability states (present/not present/unknown)
- Multiple SPIR-V modules per variant with a linking stage
- Specialization constant support in SPIR-V path
- Strategy-specific codegen nesting on functions (not exports)

**Issue #16341** (filed February 2024, OPEN, 43% complete): Multi-targeting support for HIP, CUDA, SPIR-V not plumbed; default targets not set to null (prerequisite). ROCDL/NVVM separation incomplete.

**Issue #22147** (filed September 2025, OPEN): CUDA sm_90+ architectures (Hopper H100, H200, B200, Ada RTX 40xx+) have missing GPU target support. Compilation with `--iree-cuda-target=sm_100` fails with "error: missing GPU target in #hal.executable.target." LLVM pattern selection for tcgen05 tensor operations missing.

---

## 5. Limitations

### 5.1 Structural Limitations of the Current Design

**First-match variant selection is not performance-aware.** The runtime evaluates variants in declaration order and picks the first valid one. If a `spir-v-sm80` variant is listed before a `spir-v-sm90` variant and both are valid on an H100, the sm80 variant runs — even if the sm90 variant would be 2x faster. This is Issue #15334's "strategy multi-versioning" gap.

**No runtime strategy selection.** For a single dispatch on a single device, IREE cannot choose between codegen strategies (e.g., SIMT vs cooperative matrix for matmul) based on problem size, occupancy, or other dynamic parameters. The strategy is fixed at compile time per variant. Issue #12230 Phase 2 was specifically designed to address this but has been stalled since May 2023.

**No cross-vendor parallelism probing.** Lei Zhang acknowledged in Issue #12230 that "cross-vendor parallelism querying extensions is non-existing for Vulkan right now." IREE cannot query SM count, CU count, warp/wavefront size, or shared memory parameters in a cross-vendor way via Vulkan/SPIR-V. Phase 3 of #12230 (dynamic tile sizing) is therefore blocked at a standards level, not just an IREE implementation level.

**SPIR-V capability fragmentation.** SPIR-V has two incompatible capability trees: Shader (for Vulkan/OpenGL) and Kernel (for OpenCL). IREE targets Shader exclusively. This means IREE's Vulkan/SPIR-V path has fundamental tension with OpenCL-style compute kernels. The Khronos standardization process for cross-vendor compute queries has not advanced to close this gap.

**Binary size scaling.** A fat binary with k GPU targets × m strategy variants per dispatch grows as O(k × m × dispatch_count). With 30+ dispatches per model and 3–5 GPU targets, archives grow into hundreds of megabytes before compression. SPIR-V's verbosity (it is not designed as a compact format) exacerbates this.

**Static compile-time partitioning.** True heterogeneous dispatch in IREE (CPU + GPU + NPU in one program) still requires compile-time affinity annotation. The `#stream.affinity` system provides the mechanism, but automatic partitioning based on workload characteristics (the "big GEMMs go on accelerator" heuristic) is rudimentary. Learned or profiled device placement is documented as a future goal.

**Hopper-class GPU gap (critical, 2025).** Issue #22147 reveals that IREE's CUDA backend as of late 2025 does not support sm_90+ (H100, H200, B200) properly. The `--iree-cuda-target=sm_100` flag fails at compile time. For LLVM dev meeting attendees working on cutting-edge GPU hardware, this is a significant current gap.

### 5.2 Ecosystem Gaps

**TVM alternative lost momentum.** TVM, previously the main alternative for multi-target ML compilation, had its core team acquired by NVIDIA via OctoAI in late 2024. Its future direction for heterogeneous dispatch is uncertain.

**PyTorch integration is incomplete.** iree-turbine provides a Torch-MLIR-to-IREE bridge, but GPU vendor coverage through this path lags behind direct IREE compilation paths.

**SHARK/Shortfin is AMD-centric.** While Shortfin uses the IREE runtime C API, in practice its tuning infrastructure (SHARK tuner, AMDGPU kernel optimization guide) is optimized for AMD Instinct and Radeon hardware. NVIDIA CUDA support in Shortfin is not a primary use case.

---

## 6. Gap Analysis: What Is Explicitly Unsolved

The following table maps confirmed open problems to our libkdl contributions:

| Open Problem | IREE Issue | libkdl Address |
|---|---|---|
| Runtime cost-model variant selection | #15334, #50 | Dispatch table with scoring function based on device capability vector |
| Dynamic strategy selection per problem size | #12230 Phase 2 | Runtime tile-size selection using device occupancy oracle |
| Cross-vendor device capability probing | #12230 Phase 3 | Lightweight device database (kdl_device_db) with fallback heuristics |
| Unified cross-backend target spec | #50, #16341 | KDL target descriptor (kdl_target_t) as common abstraction |
| SPIR-V multi-module per variant with linking | #15334 | KDL's multi-ISA linking: SPIR-V + PTX + HSA as first-class citizens |
| Specialization constants runtime injection | #15334 | KDL pipeline constant API for dispatch-time parameter injection |
| sm_90+ CUDA support | #22147 | libkdl targets sm_89/sm_90 via LLVM NVVM directly |

---

## 7. Key Findings About Specific Issues (Updated as of April 2026)

### Issue #50 — "Add target configuration to compiler for specifying assumed/required features"
- **Status:** OPEN (6.5 years)
- **Last substantive activity:** March 2020 (two comments total)
- **Current relevance:** This issue is the foundational problem statement. The `#iree_gpu.target` effort from Issue #16341 is the closest thing to a solution, but it only handles target description (what to compile for), not runtime best-match selection (which pre-compiled variant to use). The issue remains the canonical statement that unified target specification across backends is unsolved.
- **Cite as:** "IREE's foundational unresolved issue — vendor-agnostic target specification with runtime selection — opened October 2019 and still open."

### Issue #12230 — "Towards kernel specialization and multi-versioning"
- **Status:** OPEN, P2 (deprioritized May 2023)
- **Last activity:** February 2023 (issue opened and initial discussion; no updates since P2 labeling)
- **Phase 1** (shape deduplication): Landed. Delivers 3.75–37.6x reduction in executable count across models.
- **Phases 2 and 3** (multi-pipeline codegen + runtime selection): Zero progress. Still listed in the issue description as "sort of broken" — the runtime kernel selection logic that would support multi-pipeline dispatch was never fixed. This is the most specific statement in the IREE tracker about the runtime dispatch gap.
- **Cite as:** "Multi-pipeline codegen and runtime strategy selection explicitly planned in February 2023, explicitly deprioritized in May 2023, zero implementation progress."

### Issue #15334 — "[Epic] Support for target and strategy multi-versioning"
- **Status:** OPEN, all checkboxes unchecked
- **Last activity:** November 2, 2023 (initial comments only)
- **What it plans:** Two orthogonal improvements: (1) target multi-versioning — one artifact, N target environments, host code selects variant at runtime; (2) strategy multi-versioning — within one dispatch, multiple codegen strategies selectable via specialization constants + problem size. First comment: "pseudo-clone of #50."
- **Critical blocker identified:** The `MaterializeExecutableVariantsPass` that would generate the variant expansion does not exist. Strategy-specific codegen currently operates at the export granularity (whole kernel), not the function granularity needed for per-strategy specialization.
- **Cite as:** "IREE's complete technical roadmap for multi-versioning, filed October 2023, all tasks still pending as of April 2026."

### Issue #16341 — "[Epic] Rework various GPU compiler backends"
- **Status:** OPEN, ~43% complete
- **Last activity:** Early 2024 (PRs #17217–#17816, #16342)
- **What completed:** Unified `#iree_gpu.target` attribute across LLVMGPU and SPIR-V; consistent CLI flag names for CUDA/HIP/Vulkan targets; basic ROCDL codegen structure.
- **What remains:** The actual multi-targeting (null defaults, per-device-index compilation for HIP/CUDA/SPIR-V), ROCDL pipeline porting, NVVM renaming.
- **Cite as:** "GPU backend unification 43% complete — shared target description achieved but multi-targeting not yet plumbed (April 2026)."

---

## 8. IREE Talks and Papers (2024–2026)

### Vulkanised 2025 — Jakub Kuderski (AMD), February 2025

**"The Long Tail of AI: SPIR-V in IREE and MLIR"**
- Venue: Vulkanised 2025, Cambridge UK, February 11–13, 2025
- Slides: https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf
- Coverage by Phoronix: https://www.phoronix.com/news/AMD-Vulkan-SPIR-V-Wide-AI

Key claims from the talk (via Phoronix writeup, February 28, 2025):
- AMD's strategy uses MLIR as common IR targeting Radeon GPUs, Instinct accelerators, Ryzen AI NPUs, and CPUs
- A generic MLIR-to-SPIR-V conversion layer enables AI acceleration on any Vulkan-supporting hardware
- **Ongoing work** (not yet shipped): "Support for heterogeneous executables, for example GPU and NPU kernels used within the same program binary" — confirming this was still future work as of the talk date
- **Ongoing work**: "One click" whole-model tuning using profile-guided optimizations

This talk is directly relevant to our poster: the speaker is an AMD IREE contributor explicitly listing multi-device heterogeneous executables as unfinished work, matching the gap our libkdl prototype addresses.

### EuroLLVM 2025 — Kunwar Grover (AMD), April 2025

Grover is listed as a tutorial speaker at 2025 EuroLLVM (Berlin, April 14, 2025) with affiliation described as "IREE, an MLIR-based end-to-end compiler." His tutorial (title not publicly listed in speaker list) covers IREE codegen paths. This is the canonical venue for IREE practitioner exposure in the LLVM community.

Other relevant 2025 EuroLLVM speakers for our context:
- Ivan Butygin (AMD) — GPU kernels in MLIR
- Mahesh Ravishankar (AMD, Sr. Manager) — core MLIR committee member, IREE architect
- Alex Zinenko (AMD) — LLVM contributor, MLIR affine/GPU
- Matthias Springer (NVIDIA) — MLIR dialect development

The co-presence of AMD's IREE team and NVIDIA's MLIR contributors at EuroLLVM 2025 Berlin is the precise academic community for which our Dublin 2026 poster needs to be positioned.

### LLVM US Dev Meeting GPU Offloading Workshop, October 2025

A dedicated GPU/Offloading Workshop was held at the 2025 US LLVM Developer Meeting. This is where vendor-specific GPU compilation progress (OpenMP offload, CUDA, HIP, SPIR-V) is discussed. Slides were posted to the LLVM Discourse. No confirmed IREE-specific talk at this workshop was found in our search, but the workshop audience is the exact community we target.

### Roofline.AI Case Study — "Asynchronous Heterogeneous Execution for Edge SoCs" (December 2025)

- URL: https://www.roofline.ai/case-studies/asynchronous-heterogeneous-execution
- Organization: Roofline (AI infrastructure startup, former IREE contributors)
- Date: December 10, 2025

Key contributions:
- Three PRs upstreamed to IREE (#20885, #21005, #21029)
- Built portable infrastructure for CPU-GPU-NPU asynchronous execution
- Demonstrated on NXP, Apple, and Qualcomm edge SoCs with Qwen3-0.6B/1.7B models
- Key claim: "Zero-copy data handoff between device kernels" via shared memory and async semaphores

This is the most recent practical demonstration (December 2025) of IREE enabling heterogeneous execution across unlike compute units. Critically, it validates that the `#stream.affinity` + `!stream.timepoint` infrastructure scales to real NPU deployments, not just CUDA/HIP.

### TinyIREE (2022, still cited)

- arXiv:2205.14479
- "TinyIREE: An ML Execution Environment for Embedded Systems from Compilation to Deployment"
- Relevant for the FlatBuffer module format description and embedded deployment model

The FlatBuffer format shown in Figure 1 of TinyIREE (module.fb + variant-specific executables in a zip structure) has not changed fundamentally, though the dialect names and pass names have evolved. The paper remains the canonical citation for IREE's deployment artifact format.

### Lei Zhang's "Single-Node ML Runtime Foundation" (April 2023)

- URL: https://www.lei.chat/posts/single-node-ml-runtime-foundation/
- Author: Lei Zhang (antiagainst), IREE core contributor

Critical insight from this post: "N to 1 is easy, 1 to N is hard." Building for the most constrained edge/client case (multi-vendor GPU, dynamic OS throttling, minimal binary size) naturally produces a foundation that scales to server/cloud. IREE's architecture choices — HAL abstraction, fat-binary format, async-first runtime — are motivated by this principle.

Also contains the canonical description of IREE's fat binary strategy: "kernels for multiple vendors and architectures" with "runtime hardware probing for dynamic dispatch selection." This is the design intention; our research establishes that the implementation is incomplete.

---

## 9. Relevance to libkdl

### 9.1 Direct Positioning

libkdl (Kernel Dynamic Linker) operates at a lower abstraction level than IREE's HAL but solves the same core problem: given a set of pre-compiled kernel variants for N GPU targets, select and dispatch the correct variant at runtime.

IREE solves this for full ML models with MLIR-native tooling. libkdl solves this for individual kernels with a C runtime and minimal dependency footprint. The distinction matters for:

- **Embedded/constrained deployment** (no MLIR/LLVM runtime, no Python, bare CUDA/HIP/Vulkan API calls)
- **Hybrid systems** (PyTorch operators that need vendor-agnostic dispatch without rewriting in MLIR)
- **Runtime-only integration** (accepting pre-compiled PTX/SPIR-V/HSA binaries, no recompilation)

### 9.2 What IREE Gets Right (to acknowledge)

- The `hal.executable.variant` + condition op pattern is the most mature multi-target mechanism in any MLIR compiler — cite as prior art
- Shape-based deduplication (Issue #12230 Phase 1) proves 3.75–37.6x reduction in kernel count is achievable — cite as motivation for our dispatch table compactness
- The `#stream.affinity` model in stream dialect provides a principled framework for device placement — our KDL affinity labels mirror this design
- Fat-binary `.vmfb` format proves the deployment model (one artifact, N targets) is both necessary and achievable

### 9.3 What IREE Does Not Solve (our contribution)

The following are explicitly documented IREE gaps that libkdl directly addresses:

1. **Cost-model variant selection** (Issues #15334, #50): Our `kdl_dispatch_select()` uses a scoring function over device capability vectors rather than first-valid-match order.

2. **Dynamic strategy selection** (Issue #12230 Phase 2): Our `kdl_pipeline_t` abstraction allows multiple pre-compiled strategies per entry point, with a lightweight runtime cost estimator based on problem dimensions and device occupancy.

3. **Cross-vendor device capability probing** (Issue #12230 Phase 3): The `kdl_device_db` provides a fallback database for Vulkan devices that lack standardized parallelism queries, supplemented by runtime probing where available.

4. **Specialization constant injection** (Issue #15334): Our pipeline constant API supports SPIR-V specialization constants and PTX parameter specialization at dispatch time, enabling the variant to receive problem-size information without recompilation.

5. **sm_90+ CUDA coverage** (Issue #22147): libkdl's PTX/CUBIN loading path targets sm_89/sm_90 directly via LLVM NVVM, tested on our GTX 1650 (sm_75) with planned sm_89 validation.

### 9.4 Critical Framing for Poster

Per reviewer feedback: "Acknowledge IREE SPIR-V correctly."

Correct framing (based on this research):
- IREE has the **infrastructure** for multi-target dispatch (variant model, HAL, VMFB format) but the **runtime selection logic is explicitly broken** (Issue #12230: "sort of broken") and the **strategy multi-versioning roadmap is entirely unimplemented** (Issue #15334: all checkboxes open)
- SPIR-V/Vulkan provides **broad vendor reach** but not **peak performance** — NVIDIA CUDA/PTX and AMD HIP/ROCm consistently outperform SPIR-V on the same hardware
- IREE's heterogeneous dispatch (CPU+GPU+NPU) only reached edge SoC validation in December 2025 (Roofline.AI PRs) — it is a very recent milestone, not a mature capability
- The multi-GPU tensor parallelism in Shortfin works for **homogeneous** multi-GPU (AMD HIP × N) with **explicit compile-time indexing**, not vendor-agnostic runtime selection

Do NOT claim: "IREE solves multi-target dispatch — we just add X."
DO claim: "IREE defines the architecture (variant model + HAL + stream affinity) that our work adopts and extends with a cost-model runtime, cross-vendor probing, and a C-API deployment path that works without MLIR toolchain dependency."

---

## 10. Sources

### Primary: GitHub Issues (with current status)
- [Issue #50 — Add target configuration (2019, OPEN)](https://github.com/iree-org/iree/issues/50)
- [Issue #12230 — Towards kernel specialization and multi-versioning (2023, OPEN P2)](https://github.com/iree-org/iree/issues/12230)
- [Issue #15334 — [Epic] Support for target and strategy multi-versioning (2023, OPEN)](https://github.com/iree-org/iree/issues/15334)
- [Issue #16341 — [Epic] Rework various GPU compiler backends (2024, OPEN, ~43%)](https://github.com/iree-org/iree/issues/16341)
- [Issue #19639 — Multi-GPU tensor_parallelism==2 error (2025, CLOSED via PR #19670)](https://github.com/iree-org/iree/issues/19639)
- [Issue #22147 — CUDA sm_90+ missing support (2025, OPEN)](https://github.com/iree-org/iree/issues/22147)

### IREE Release Changelogs
- [v3.6.0 (July 21, 2025)](https://github.com/iree-org/iree/releases/tag/v3.6.0) — AMDGPU executable impl; ElideAsyncTransfersPass; group_any thread affinity
- [v3.9.0 (November 25, 2024)](https://github.com/iree-org/iree/releases/tag/v3.9.0) — RISC-V mmt4d ukernel with runtime ISA detection; timeline-aware async execution; stream canonicalizations
- [v3.10.0 (February 2, 2025)](https://github.com/iree-org/iree/releases/tag/v3.10.0) — Producer-Consumer Fusion (PCF) infrastructure; linear-scan register allocator; AMDGPU 3-stage pipelining

### Talks and Case Studies
- [Jakub Kuderski (AMD), "The Long Tail of AI: SPIR-V in IREE and MLIR," Vulkanised 2025 (February 2025)](https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf)
- [Phoronix writeup of Kuderski talk (February 28, 2025)](https://www.phoronix.com/news/AMD-Vulkan-SPIR-V-Wide-AI)
- [Roofline.AI — "Asynchronous Heterogeneous Execution for Edge SoCs" (December 2025)](https://www.roofline.ai/case-studies/asynchronous-heterogeneous-execution)
- [AMD ROCm Blog — MI325X MLPerf v5.0 SDXL submission (April 2025)](https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html)

### Official IREE Documentation
- [Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)
- [HAL Passes Reference](https://iree.dev/reference/mlir-passes/HAL/)
- [Stream Dialect Reference](https://iree.dev/reference/mlir-dialects/Stream/)
- [HAL Dialect Reference](https://iree.dev/reference/mlir-dialects/HAL/)
- [GPU Vulkan Deployment](https://iree.dev/guides/deployment-configurations/gpu-vulkan/)
- [GPU CUDA Deployment](https://iree.dev/guides/deployment-configurations/gpu-cuda/)
- [GPU ROCm Deployment](https://iree.dev/guides/deployment-configurations/gpu-rocm/)
- [Tuning Reference](https://iree.dev/reference/tuning/)

### External Analysis
- [Lei Zhang — "Single-Node ML Runtime Foundation" (April 2023)](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)
- [TinyIREE arXiv:2205.14479](https://arxiv.org/pdf/2205.14479) — canonical FlatBuffer format reference
- [2025 EuroLLVM Speaker List (Berlin, April 2025)](https://llvm.swoogo.com/2025eurollvm/speakers)
- [LLVM GPU/Offloading Workshop 2025 — US DevMtg](https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832)
- [RFC: SPIR-V IR as vendor-agnostic GPU representation (LLVM Discourse, March 2025)](https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115)
