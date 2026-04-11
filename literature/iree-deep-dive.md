# IREE Deep Dive: Architecture, HAL, and Multi-Target Dispatch

**Research date:** 2026-04-02
**Context:** LLVM Dublin 2026 poster -- Heterogeneous GPU Kernel Dispatch via MLIR
**Researcher:** Akash

---

## Table of Contents

1. [IREE Architecture Overview](#1-iree-architecture-overview)
2. [The Dialect Pipeline: Flow, Stream, HAL](#2-the-dialect-pipeline-flow-stream-hal)
3. [HAL Architecture and Design](#3-hal-architecture-and-design)
4. [Multi-Target Dispatch: Current State](#4-multi-target-dispatch-current-state)
5. [SPIR-V Backend: Capabilities and Limitations](#5-spir-v-backend-capabilities-and-limitations)
6. [GPU Backend Unification Effort](#6-gpu-backend-unification-effort)
7. [Analysis of Three Key GitHub Issues](#7-analysis-of-three-key-github-issues)
8. [Kernel Specialization and Multi-Versioning](#8-kernel-specialization-and-multi-versioning)
9. [Device Affinity and Heterogeneous Execution](#9-device-affinity-and-heterogeneous-execution)
10. [SPIR-V as Vendor-Agnostic GPU IR: Broader Context](#10-spir-v-as-vendor-agnostic-gpu-ir-broader-context)
11. [Comparison with Other Approaches](#11-comparison-with-other-approaches)
12. [Open Problems and Research Gaps](#12-open-problems-and-research-gaps)
13. [Implications for Our Poster](#13-implications-for-our-poster)

---

## 1. IREE Architecture Overview

IREE (Intermediate Representation Execution Environment) is an end-to-end MLIR-based compiler and runtime for ML model deployment. It compiles from high-level frameworks (PyTorch, TensorFlow, JAX via StableHLO/TOSA) down to optimized native code for CPUs, GPUs, and accelerators.

**Key architectural properties:**
- Ahead-of-time (AOT) compilation with optional JIT paths
- FlatBuffer-based module encoding for deployment artifacts (executables + scheduling logic)
- Bytecode VM for host-side orchestration
- Hardware Abstraction Layer (HAL) separating device-specific details from scheduling
- Part of the OpenXLA ecosystem (alongside XLA and StableHLO)
- Joined LF AI & Data Foundation as sandbox project (May 2024)

**Supported backends:**
- `llvm-cpu` -- CPU via LLVM codegen
- `vulkan-spirv` -- GPU via Vulkan/SPIR-V
- `metal-spirv` -- Apple GPU via Metal/SPIR-V
- `cuda` -- NVIDIA GPU via PTX/CUBIN
- `rocm` -- AMD GPU via AMDGPU/HSA
- `vmvx` -- Portable reference backend (VM-based vector extensions)
- `vmvx-inline` -- Embedded/bare-metal variant

Source: [IREE Developer Overview](https://iree.dev/developers/general/developer-overview/), [IREE Deployment Configurations](https://iree.dev/guides/deployment-configurations/)

---

## 2. The Dialect Pipeline: Flow, Stream, HAL

IREE's compiler transforms programs through three core dialects in sequence, each at a different abstraction level:

### 2.1 Flow Dialect
- Operates on MLIR value-semantic tensors
- Ingests high-level linear algebra from StableHLO/TOSA/linalg
- Partitions computations into **dispatch regions** -- groups of compatible dense operations
- Outlines dispatch regions into **executables**
- Runs `DeduplicateExecutables` pass to merge structurally identical dispatches

Source: [Flow Dialect Reference](https://iree.dev/reference/mlir-dialects/Flow/)

### 2.2 Stream Dialect
- Converts tensor programs to **explicitly scheduled asynchronous programs**
- Assigns **affinity attributes** to operations (specifying target device/queue)
- Partitions work between targets
- Encodes tensors into target-specific resources with symbolic sizes
- Schedules work for concurrency using timepoints (`!stream.timepoint`)
- Key ops: `stream.cmd.dispatch`, `stream.async.execute`, `stream.async.concurrent`

The stream dialect is where **device placement decisions** are made. Almost all operations carry an optional `#stream.affinity` attribute. The compiler's role is to assign affinities so that lowering to HAL becomes a simple mapping to `!hal.device` + queue affinity.

Source: [Stream Dialect Reference](https://iree.dev/reference/mlir-dialects/Stream/)

### 2.3 HAL Dialect
- Models a low-level hardware abstraction layer (Vulkan-like, minus graphics)
- Manages buffers, command buffers, executables, and synchronization
- Command buffers follow Vulkan's flat recording model
- Operations execute via `iree_hal_device_queue_execute()` with semaphore-based ordering
- Each executable may contain **multiple target-specific variants** (fat binary model)

Key HAL passes (in compilation order):
1. `AssignTargetDevices` -- assigns HAL devices from target specs
2. `MaterializeInterfaces` -- creates `hal.executable` + `hal.variant` per target
3. `ConfigureExecutables` -- attaches target-specific codegen configuration
4. `TranslateExecutables` -- lowers from generic MLIR (linalg) to target dialects (spirv, llvm)
5. `LinkExecutables` -- links/deduplicates executables per backend
6. `SerializeExecutables` -- converts to target-specific binary format

Source: [HAL Dialect Reference](https://iree.dev/reference/mlir-dialects/HAL/), [HAL Passes Reference](https://iree.dev/reference/mlir-passes/HAL/)

---

## 3. HAL Architecture and Design

### 3.1 Core Abstractions

The HAL provides a uniform interface across all execution targets:

| Concept | Description |
|---------|-------------|
| `hal.device` | Logical device (may map to multiple physical devices) |
| `hal.executable` | Container for one or more compiled variants |
| `hal.executable.variant` | Target-specific compiled code within an executable |
| `hal.executable.export` | Entry point within an executable (dispatch target) |
| `hal.command_buffer` | Recorded sequence of dispatch/transfer commands |
| `hal.semaphore` | Timeline semaphore for cross-queue synchronization |
| `hal.buffer` | Device memory allocation |

### 3.2 Executable Variant Model

This is central to IREE's multi-target story:

- Each `hal.executable` may contain **one or more `hal.executable.variant`** ops
- Variants are lowered independently during compilation
- At runtime, variants are selected based on:
  1. Their declared target compatibility
  2. An **optional condition op** that queries the runtime `!hal.device`
- If multiple variants are valid, **the first valid one is used** (priority ordering)
- If no variant is valid, **loading fails at runtime**
- This is analogous to **fat binaries** in CUDA/HIP

### 3.3 Device Query Mechanism

`hal.device.query` enables runtime capability probing. The runtime can check for features, extensions, and device properties. This feeds into variant condition ops for selection.

### 3.4 Limitations of Current HAL Design

1. **First-match selection**: No cost-model-based selection between valid variants. The first valid variant wins, not the "best" one.
2. **Static compilation**: Variant set is fixed at compile time. No runtime JIT of new variants.
3. **No cross-device dispatch**: A single dispatch cannot span multiple devices. Device affinity is per-dispatch.
4. **Queue affinity != device affinity**: Multi-queue support is a stepping stone to multi-device, but true heterogeneous dispatch across different device types in a single program is still WIP.

Source: [HAL Dialect Reference](https://iree.dev/reference/mlir-dialects/HAL/), [Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)

---

## 4. Multi-Target Dispatch: Current State

### 4.1 What Works Today

- Compiling a module for **multiple target backends** (e.g., `--iree-hal-target-device=local --iree-hal-target-device=vulkan`) is supported
- Each dispatch gets compiled into variants for each specified target
- Runtime selects the variant matching the available device
- **CPU multi-versioning** via `hal.executable.variant` conditional enablement was implemented and closed as solved (Issue #3768, November 2023)

### 4.2 What Does NOT Work Today

- **No single-artifact multi-GPU-vendor deployment**: You cannot compile one artifact that runs optimally on both NVIDIA and AMD GPUs via Vulkan alone. SPIR-V variants must target specific capability sets.
- **No dynamic strategy selection within a dispatch**: Cannot choose between e.g., SIMT vs cooperative-matrix codegen at runtime based on problem size
- **No specialization constants**: SPIR-V specialization constants are not yet leveraged for runtime tuning
- **Runtime kernel selection logic is "sort of broken"** (per Issue #12230): The connection between kernel capability requirements and runtime checks needs fixing for multi-versioning to work properly
- **No cross-vendor parallelism querying**: Vulkan has no cross-vendor extension for querying device parallelism (number of SMs, CUs, etc.), making runtime tile-size adaptation impossible without a hardcoded device database

Source: Issues #15334, #12230, [Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)

---

## 5. SPIR-V Backend: Capabilities and Limitations

### 5.1 What SPIR-V Can Do

- **Broad vendor reach via Vulkan**: AMD, ARM Mali, Intel, NVIDIA, Qualcomm Adreno all support Vulkan + SPIR-V
- **Portable binary format**: SPIR-V modules are driver-consumable without LLVM dependency at runtime
- **Capability declaration**: SPIR-V declares required capabilities upfront, enabling driver fast-paths
- **Extension mechanism**: Vendor, EXT, and KHR tiers allow incremental feature exposure
- **Stability**: Designed for long-lived drivers that may never update (critical for embedded/mobile)

### 5.2 What SPIR-V Cannot Do (for Vendor-Agnostic Dispatch)

1. **Shader vs Kernel capability split**: SPIR-V has two fundamentally different capability trees -- Shader (for Vulkan/OpenGL) and Kernel (for OpenCL). They are "quite different" and switching between them is not trivial. IREE targets Shader capability exclusively for its Vulkan path.

2. **Too coupled to Vulkan**: IREE's SPIR-V backend historically had separate target management tightly bound to Vulkan concepts (e.g., `vk.target_env`). The unified `#iree_gpu.target` effort aims to fix this (Issue #16341).

3. **No vendor-specific intrinsics**: SPIR-V cannot express NVIDIA tensor core ops or AMD matrix ops natively the way PTX/AMDGPU IL can. Performance-critical kernels require vendor-specific extensions that break portability.

4. **Binary size**: "SPIR-V is not designed to be a small at-rest format" (per IREE design roadmap). Compression (e.g., SMOL-V) is needed for deployment. Fat binaries with multiple SPIR-V variants exacerbate this.

5. **Missing cross-vendor compute queries**: No standardized way to query compute unit count, warp/wavefront size, shared memory size across vendors via Vulkan/SPIR-V alone. IREE would need a hardcoded device database.

6. **Target triple is approximate**: IREE docs acknowledge "we don't support the full spectrum of GPUs...the target triple is just an approximation." Real performance requires vendor-specific tuning.

7. **Trinary capability state unsolved**: Issue #15334 notes the need for trinary capability states (present / not present / unknown) on SPIR-V capabilities and extensions, which is not yet implemented.

### 5.3 Supported GPU Targets via Vulkan/SPIR-V

| Vendor | Architectures | Performance Level |
|--------|--------------|-------------------|
| AMD Desktop | RDNA1-RDNA4 (gfx1010-gfx1200) | Good |
| ARM Mali | Valhall-Valhall4 | Good |
| NVIDIA | Turing+ (sm_75-sm_89) | Reasonable |
| Qualcomm | Adreno 640+ | Reasonable |

Note: "Reasonable" for NVIDIA means SPIR-V/Vulkan significantly underperforms CUDA/PTX on the same hardware.

Source: [GPU Vulkan Deployment Guide](https://iree.dev/guides/deployment-configurations/gpu-vulkan/), Issue #16341, [Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)

---

## 6. GPU Backend Unification Effort

Issue #16341 ("Rework various GPU compiler backends") tracks a major restructuring:

### 6.1 Problem Statement

IREE has two GPU codegen paths that evolved independently:
- **LLVMGPU**: targets CUDA (NVIDIA) and ROCm (AMD) via LLVM IR -> PTX/AMDGPU
- **SPIRV**: targets Vulkan/Metal/WebGPU via MLIR SPIR-V dialect

These share significant codegen logic but have inconsistent:
- Target description mechanisms
- Compiler pass organization
- Pipeline configurations
- Hardcoded settings

### 6.2 Planned Solution

1. **Unified `#iree_gpu.target` attribute**: Replaces separate `vk.target_env`, CUDA triples, etc. Contains:
   - Canonical target architecture (e.g., `sm_80`, `gfx942`)
   - Workgroup processor attributes (compute/storage capabilities)
   - Optional chip-level attributes for product-specific features
2. **Decouple CUDA and ROCm**: Create separate NVVM and ROCDL codegen subdirectories
3. **Replace hardcoded settings** with heuristic-based deduction and runtime probing

### 6.3 Status (as of research date)

Five target-description PRs have merged. Remaining work includes full multi-targeting support and the CUDA/ROCm backend separation.

Source: [Issue #16341](https://github.com/iree-org/iree/issues/16341)

---

## 7. Analysis of Three Key GitHub Issues

### 7.1 Issue #50: "Add target configuration to compiler for specifying assumed/required features"

- **URL**: https://github.com/iree-org/iree/issues/50
- **Created**: 2019-10-13 (one of IREE's earliest issues)
- **State**: OPEN (6+ years old)
- **Author**: benvanik (IREE co-creator)

**What was proposed:**
The ability to specify target capabilities (Vulkan/SPIR-V extensions, device limits, etc.) as compiler flags, producing a variety of executables. At runtime, the system should match against available capabilities to select the best-suited executable, with override support for benchmarking.

**What was discussed:**
- Ben Vanik noted early work in the Vulkan dialect with `target_env` attributes but called for "something more general that wraps that so we have a unified way of specifying targets across all backends."
- Lei Zhang (antiagainst) volunteered to work on it for a demo case.

**What is still unsolved:**
This issue remains open after 6+ years. While partial solutions exist (`hal.executable.variant` with conditions, `vk.target_env`, the new `#iree_gpu.target`), a fully unified, cross-backend target configuration system with runtime best-match selection is still incomplete. Issue #15334 is described as a "pseudo-clone" of this issue, indicating the problem has been reframed but not resolved.

**Significance for our poster:** This is the foundational issue. The fact that it has been open since October 2019 demonstrates that vendor-agnostic target specification and runtime selection is a hard, unsolved problem even within the most sophisticated MLIR-based compiler.

### 7.2 Issue #12230: "Towards kernel specialization and multi-versioning"

- **URL**: https://github.com/iree-org/iree/issues/12230
- **Created**: 2023-02-16
- **State**: OPEN (deprioritized to P2 as of May 2023)
- **Author**: antiagainst (Lei Zhang, IREE core contributor)

**What was proposed:**
A three-phase plan for kernel specialization and multi-versioning:

1. **Reduce specialization toward input shapes**: Erase static shape information after dispatch region formation to deduplicate dispatches with identical op sequences but different shapes. This alone achieved 2-30x reduction in executable count across models.

2. **Run multiple codegen pipelines per dispatch**: Compile the same dispatch region with different target environments and pipeline selections (e.g., SIMT vs cooperative matrix for matmul on GPU). Emit runtime kernel selection logic via host code that queries the target environment.

3. **Dynamic runtime adjustment**: Query device parallelism (SM count, etc.) at runtime and compute tile/workgroup sizes dynamically. Acknowledged that "cross-vendor parallelism querying extensions is non-existing for Vulkan right now."

**What was discussed (with quantitative results):**

Lei Zhang provided concrete deduplication statistics:

| Model | L0 (no dedup) | L1 (basic dedup) | L2 (+dynamic shapes) | L0/L2 ratio |
|-------|--------------|-------------------|----------------------|-------------|
| TF ResNet50 f32 | 75 | 39 | 20 | 3.75x |
| TF EfficientNetV2 Small f32 | 278 | 69 | 30 | 9.27x |
| TF BERT Large f32 | 413 | 15 | 14 | 29.5x |
| TFL MobileBERT i8 | 1452 | 168 | 162 | 8.96x |

With additional dynamic slice offset/size handling:
- MobileBERT f32: 752 -> 20 (37.6x reduction)
- MobileBERT i8: 1452 -> 117 (12.4x reduction)

Ben Vanik commented: "In my mind deduplication is compression, and 10x compression wins are fantastic let alone 30x."

**What is still unsolved:**
- Phases 2 and 3 (multi-pipeline codegen + runtime selection) have NOT been implemented
- The issue was deprioritized to P2 in May 2023 and has seen no further updates
- Runtime kernel selection logic remains "sort of broken" per the issue description
- The quantized model case (MobileBERT i8) still has high residual count due to fusion variants
- No mechanism exists to query device parallelism cross-vendor

**Significance for our poster:** This issue contains the most detailed technical plan for multi-versioning in IREE. The fact that Phase 1 (deduplication) succeeded but Phases 2-3 (runtime selection) stalled demonstrates that the compiler-side deduplication is tractable but the runtime dispatch problem is much harder.

### 7.3 Issue #15334: "[Epic] Support for target and strategy multi-versioning"

- **URL**: https://github.com/iree-org/iree/issues/15334
- **Created**: 2023-10-28
- **State**: OPEN
- **Author**: qedawkins (Quinn Dawkins, IREE contributor)

**What was proposed:**
Two orthogonal improvements:

1. **Target multi-versioning**: User specifies a range of target environments; IREE compiles efficient device code for all targets plus host code to switch between them at runtime.

2. **Strategy multi-versioning**: Within a single dispatch, compile multiple codegen strategies (e.g., different tile sizes, different algorithms) and swap between them based on specialization constants and dynamic problem sizes.

**Concrete task list (all unchecked as of research date):**

General:
- Allow specifying explicit `hal.device.targets` as command-line arguments
- Allow `hal.dispatch.extern` to specify minimum required features (not exact target)
- Create `MaterializeExecutableVariantsPass` to expand/contract variants based on lowering strategies

Codegen:
- Nest strategy-specific codegen on functions (not exports) for specialization
- Allow multiple strategies per entry point
- Add interface to `DispatchLoweringPipeline` for specifying required/unneeded/unknown capabilities

SPIR-V specific:
- Trinary capability state (present / not present / unknown)
- Multiple SPIR-V modules per variant with linking stage
- Specialization constant support

**What is still unsolved:**
All tasks remain unchecked. This is essentially a redesign of how IREE handles the target-to-codegen mapping, and it has not progressed beyond the planning stage. The first comment notes it is a "pseudo-clone of #50", linking back to the 2019 foundational issue.

**Significance for our poster:** This epic represents IREE's most complete vision for what heterogeneous multi-target dispatch should look like. The fact that all tasks remain open makes it a clear research gap that our poster can address.

---

## 8. Kernel Specialization and Multi-Versioning

### 8.1 Shape Deduplication (Implemented)

IREE's most successful multi-versioning work is shape-based executable deduplication:
- Erase static shapes after dispatch region formation
- Deduplicate dispatches with identical op sequences
- Pass concrete shape values as `hal.interface.constant.load` ops
- Codegen can inspect all possible values to decide versioning strategy

This is purely a compile-time optimization. It does not add runtime dispatch capability.

### 8.2 LLVM CPU Multi-Versioning (Solved)

Issue #3768 proposed LLVM function multi-versioning (compiling the same function for multiple ISA levels, e.g., SSE4.2 vs AVX2 vs AVX-512). This was closed as solved in November 2023 via `hal.executable.variant` conditional enablement -- each ISA level gets a separate variant with a condition checking CPU feature flags.

### 8.3 GPU Strategy Multi-Versioning (Unsolved)

The harder problem -- choosing between codegen strategies (SIMT vs tensor core, different tile sizes) at runtime for the same dispatch on the same device -- remains fully unsolved. Key blockers:
- No runtime cost model for strategy selection
- No specialization constant infrastructure in SPIR-V path
- No mechanism to probe device parallelism cross-vendor

Source: Issues #12230, #15334, #3768

---

## 9. Device Affinity and Heterogeneous Execution

### 9.1 Current Model

IREE models heterogeneous execution through **affinity attributes** in the stream dialect:

```
stream.cmd.dispatch @executable::@entry on(%device_affinity)
```

The compilation pipeline:
1. Stream dialect assigns affinities (compiler heuristics, user annotations, or profile-guided)
2. Lowering to HAL maps affinities to `!hal.device` + queue affinity bitmask
3. At runtime, HAL routes commands to the appropriate device queue

### 9.2 Logical vs Physical Devices

A key IREE design choice: a logical `hal.device` may map to **multiple physical devices** with unified address space. This simplifies the compiler -- executables, pipeline layouts, and buffers are handled once regardless of physical device count. Multi-queue on one device is "functionally equivalent from the compiler's perspective to multiple physical devices exposed as a single logical device."

### 9.3 Limitations

- True heterogeneous execution (CPU + GPU in one program with automatic partitioning) is WIP
- Device placement heuristics are rudimentary -- "big GEMMs go on the accelerator" level
- No learned/profiled device placement in practice
- Multi-device support requires explicit affinity annotation; automatic partitioning is a future goal

Source: [Issue #10765](https://github.com/iree-org/iree/issues/10765), [Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)

---

## 10. SPIR-V as Vendor-Agnostic GPU IR: Broader Context

### 10.1 LLVM Discourse RFC (March 2025)

An RFC titled "SPIR-V IR as a vendor-agnostic GPU representation" was posted on LLVM Discourse in March 2025. The proposal explores using SPIR-V (via the LLVM SPIR-V backend, promoted to official in LLVM 20) as a common GPU target that can be lowered to vendor-specific ISAs downstream.

Key tensions in the discussion:
- SPIR-V's stability is a strength for deployment but limits expressiveness
- The Shader vs Kernel capability split fragments the ecosystem
- MLIR-to-SPIR-V can skip LLVM IR entirely (as IREE does), questioning whether LLVM IR -> SPIR-V is the right path
- Vendor-specific intrinsics (tensor cores, matrix ops) have no portable SPIR-V representation

### 10.2 Lei Zhang's Analysis (IREE/MLIR contributor)

Lei Zhang's blog post "Compilers and IRs: LLVM IR, SPIR-V, and MLIR" provides a foundational analysis:
- SPIR-V was designed because "LLVM IR is not really designed for this kind of task" (cross-vendor GPU distribution)
- SPIR-V prioritizes **driver consumption** over compiler convenience
- Native GPU concepts (decorations, builtins) make it domain-appropriate
- But tight coupling to Khronos standards (Vulkan, OpenCL, OpenGL) limits applicability beyond those ecosystems

Source: [LLVM Discourse RFC](https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115), [Lei Zhang's blog](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/)

---

## 11. Comparison with Other Approaches

### 11.1 IREE vs XLA

| Aspect | IREE | XLA |
|--------|------|-----|
| IR foundation | MLIR (flow/stream/hal) | HLO -> LLVM IR |
| Multi-target | Variant-based fat binaries | Separate compilation per target |
| Runtime dispatch | First-valid-variant selection | Static target binding |
| GPU backends | SPIR-V + LLVM (CUDA/ROCm) | CUDA, ROCm, SPIR-V (limited) |
| Deployment | Single artifact, multiple targets | Per-target artifacts |

### 11.2 IREE vs TVM

| Aspect | IREE | TVM |
|--------|------|-----|
| Multi-target | Compiler-driven variants | Schedule-based autotuning |
| Runtime | HAL + VM bytecode | Graph runtime / Relay VM |
| Portability | SPIR-V/Vulkan for GPUs | OpenCL, Vulkan, CUDA, Metal |
| Heterogeneous | Affinity-based placement | Heterogeneous execution via relay |
| Status | Active, well-funded (Google) | Uncertain future (core team acquired by NVIDIA via OctoAI, late 2024) |

### 11.3 Other Relevant Systems

- **ALPAKA**: C++ abstraction layer for heterogeneous accelerators. Compile-time backend selection, no runtime dispatch.
- **SYCL/DPC++**: Single-source C++ with runtime device selection. Closer to IREE's vision but at the programming model level rather than compiler IR level.
- **HIP**: AMD's CUDA-compatible layer. Single-vendor focus despite portability claims.

---

## 12. Open Problems and Research Gaps

### 12.1 Unsolved in IREE (confirmed by issue tracker)

1. **Runtime strategy selection**: No mechanism to choose between codegen strategies at runtime based on problem size, device occupancy, or other dynamic factors.

2. **Cross-vendor device capability probing**: Vulkan lacks standardized extensions for querying compute unit counts, warp sizes, etc. IREE would need a hardcoded device database.

3. **Specialization constants for SPIR-V**: Not yet integrated into the multi-versioning pipeline.

4. **Multi-variant cost model**: Current first-valid-match selection has no performance-aware ranking.

5. **True heterogeneous dispatch**: Automatic work partitioning across CPU + GPU + accelerator based on workload characteristics.

6. **SPIR-V binary size**: Fat binaries with N variants x M targets explode in size. No compression/deduplication at the variant level.

7. **Unified target specification**: The `#iree_gpu.target` effort is in progress but incomplete.

### 12.2 Fundamental Tensions

- **Portability vs Performance**: SPIR-V provides reach but cannot match vendor-specific backends (CUDA PTX, AMDGPU) in performance. The multi-variant approach trades binary size for coverage.
- **Compile-time vs Runtime decisions**: IREE heavily favors AOT compilation. Adding runtime decision-making (strategy selection, dynamic tiling) conflicts with the "compile once, deploy everywhere" philosophy.
- **Deduplication vs Specialization**: Erasing shape information for deduplication conflicts with the need for shape-specialized high-performance kernels.

---

## 13. Implications for Our Poster

### 13.1 What IREE Gets Right

- The `hal.executable.variant` + condition op pattern is the most mature mechanism for multi-target dispatch in any MLIR-based compiler
- The flow -> stream -> HAL pipeline cleanly separates concerns
- Shape-based deduplication proves that significant executable reduction is achievable (up to 30x)
- The affinity model in stream dialect provides a principled framework for device placement

### 13.2 What Our Poster Can Contribute

Given IREE's open gaps, our poster could address:

1. **Runtime dispatch policy**: Propose a cost-model-based variant selection mechanism (beyond first-valid-match) that considers problem size, device occupancy, and historical performance data.

2. **Dynamic strategy selection**: Design a lightweight runtime that selects between pre-compiled codegen strategies (e.g., SIMT vs cooperative matrix) based on dispatch parameters -- exactly what Issue #12230 Phase 2 describes but hasn't implemented.

3. **Cross-vendor device database**: Prototype the "simple database to embed" that Lei Zhang mentions in Issue #12230 for device parallelism querying.

4. **Hybrid SPIR-V + native approach**: Compile SPIR-V as portable fallback + vendor-native (PTX/AMDGPU) as performance path, with runtime selection -- a practical form of Issue #15334's vision.

### 13.3 Critical Positioning

The reviewer feedback said to "Acknowledge IREE SPIR-V correctly." Based on this research:
- Do NOT claim IREE solves multi-target dispatch -- it has the *infrastructure* but the *runtime selection logic* is incomplete
- Do NOT claim SPIR-V enables vendor-agnostic high-performance GPU code -- it provides reach but not peak performance
- DO cite the specific issues (#50, #12230, #15334) as evidence that the problem is recognized but unsolved
- DO acknowledge IREE's variant model as the state of the art while identifying the gaps our work fills

---

## Sources

### IREE Official Documentation
- [Developer Overview](https://iree.dev/developers/general/developer-overview/)
- [Design Roadmap](https://iree.dev/developers/design-docs/design-roadmap/)
- [HAL Dialect Reference](https://iree.dev/reference/mlir-dialects/HAL/)
- [Stream Dialect Reference](https://iree.dev/reference/mlir-dialects/Stream/)
- [Flow Dialect Reference](https://iree.dev/reference/mlir-dialects/Flow/)
- [HAL Passes Reference](https://iree.dev/reference/mlir-passes/HAL/)
- [GPU Vulkan Deployment Guide](https://iree.dev/guides/deployment-configurations/gpu-vulkan/)
- [Metal HAL Driver Design Doc](https://iree.dev/developers/design-docs/metal-hal-driver/)

### GitHub Issues (Primary Sources)
- [Issue #50: Add target configuration to compiler](https://github.com/iree-org/iree/issues/50) -- 2019, OPEN
- [Issue #12230: Towards kernel specialization and multi-versioning](https://github.com/iree-org/iree/issues/12230) -- 2023, OPEN (P2)
- [Issue #15334: Support for target and strategy multi-versioning](https://github.com/iree-org/iree/issues/15334) -- 2023, OPEN
- [Issue #16341: Rework various GPU compiler backends](https://github.com/iree-org/iree/issues/16341) -- 2024, OPEN
- [Issue #3768: LLVM function multiversioning with LTO](https://github.com/iree-org/iree/issues/3768) -- CLOSED (solved via variant conditions)
- [Issue #10765: Initial affinity support for multiple queues](https://github.com/iree-org/iree/issues/10765) -- OPEN
- [Issue #12520: RFC - IREE Compiler Plugin Mechanism](https://github.com/iree-org/iree/issues/12520)

### External References
- [RFC: SPIR-V IR as a vendor-agnostic GPU representation (LLVM Discourse, March 2025)](https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115)
- [Compilers and IRs: LLVM IR, SPIR-V, and MLIR -- Lei Zhang](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/)
- [LLVM 20 Promotes SPIR-V to Official Backend](https://www.phoronix.com/forums/forum/software/programming-compilers/1521770-llvm-20-promotes-spir-v-to-official-backend-enabled-by-default)
