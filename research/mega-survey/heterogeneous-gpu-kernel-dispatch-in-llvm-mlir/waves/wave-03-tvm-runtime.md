# Wave 03 — TVM Unified Runtime Multi-Target Dispatch

**Angle:** TVM Unified Runtime Multi-Target
**Date:** 2026-04-06
**Sources surveyed:** 14 primary sources (docs, RFCs, papers, release notes, GitHub issues)

---

## Executive Summary

TVM is the most complete prior art for multi-target ML kernel compilation, but it is not a runtime dispatch system in the sense libkdl targets. TVM selects targets at compile time and produces per-target artifacts. The closest approximations to runtime vendor dispatch within TVM are: (1) Vulkan/SPIR-V targets, which produce artifacts runnable across any Vulkan-compliant vendor at runtime, and (2) the VDevice/Relax heterogeneous execution RFC (2023), which defines compile-time device placement rather than runtime selection. There is no mechanism in TVM that mirrors libkdl's "single fat binary, hardware-queried dispatch at load time" model.

**Relevance to vendor-agnostic kernel dispatch runtime:** 8/10 — TVM is the primary prior art to position against. Its gaps are the motivation for libkdl.

---

## Sources

### S1 — TVM Runtime System (Official Docs, v0.24.dev0)
- **URL:** https://tvm.apache.org/docs/arch/runtime.html
- **Type:** Official documentation
- **Date:** Current (mirrors dev HEAD ~2025-2026)
- **Relevance/Novelty:** 8/10
- **Summary:** Defines the PackedFunc abstraction as TVM's universal dispatch primitive. A `PackedFunc` is type-erased — caller and callee can be in different languages (Python, C++, CUDA C). The Module system provides device-typed compiled objects: `ModuleNode` is an abstract base class with per-backend implementations (CUDA, Metal, OpenCL, Vulkan). Function handles are cached after first retrieval from a Module. The runtime documentation explicitly states that calling PackedFunc versus a direct function call has overhead of "only saving a few values on the stack" — negligible for kernel-granularity calls.
- **Key detail:** PackedFunc is the granular dispatch unit. Compiled modules are separate per-target objects; the caller selects which module to invoke. There is no built-in module-level fallback or hardware-capability-driven selection.

### S2 — TVM Device/Target Interactions (Official Docs, v0.21.dev0)
- **URL:** https://tvm.apache.org/docs/arch/device_target_interactions.html
- **Type:** Official documentation
- **Date:** Current
- **Relevance/Novelty:** 9/10
- **Summary:** Defines the `DeviceAPI` class — one subclass per hardware backend (e.g., `CUDADeviceAPI`, `VulkanDeviceAPI`). Device selection uses `tvm.runtime.device('cuda', 0)` at call time — explicit, not automatic. `GetAttr` queries device properties (`DeviceAttrKind` enum): device name, thread count, warp size, clock rate. Not all parameters are supported by all devices (unsupported returns `nullptr`). Code generators register under `"target.build.foo"` in `target_kind.cc` via `TVM_REGISTER_TARGET_KIND`, mapping each target kind to a `DLDeviceType`. Multiple target kinds can map to the same device (e.g., both `"llvm"` and `"c"` target `kDLCPU`).
- **Key detail:** `DeviceAPI::GetAttr` is TVM's capability-query primitive — it exists, but it is used for codegen decisions at compile time, not for runtime dispatch selection between alternative backends.

### S3 — Vulkan Runtime Architecture (Official Docs, v0.24.dev0)
- **URL:** https://tvm.apache.org/docs/arch/runtimes/vulkan.html
- **Type:** Official documentation
- **Date:** Current
- **Relevance/Novelty:** 9/10
- **Summary:** TVM compiles all GPU kernels to SPIR-V compute shaders for Vulkan targets. Device capabilities (subgroup operations, 8/16-bit storage, float16/int8/int64 support) are queried at compile time via `-from_device=N`, which reads all Vulkan device parameters from the physical device and specializes SPIR-V for that capability set. Generated shaders declare the minimum SPIR-V capability set required. This means a SPIR-V artifact compiled for a low-capability device will run on any higher-capability Vulkan implementation — but artifacts compiled for a specific device may use extensions unavailable on other vendors' drivers. SPIR-V validation via `spvValidate` is available.
- **Key detail:** TVM's Vulkan target is the closest thing to cross-vendor runtime portability in TVM. A SPIR-V shader produced with conservative capability assumptions runs on NVIDIA, AMD, Intel, Qualcomm, and mobile Vulkan drivers. This is static portability, not dynamic dispatch — the capability envelope is fixed at compile time.

### S4 — RFC: Heterogeneous Execution for Relax (TVM Discuss, April 2023)
- **URL:** https://discuss.tvm.apache.org/t/rfc-unity-relax-heterogeneous-execution-for-relax/14670
- **Type:** RFC / design document
- **Date:** April 2023
- **Relevance/Novelty:** 9/10
- **Summary:** Defines `VDevice` — a virtual device descriptor with three fields: `Target` (compilation target), `vdevice_id` (distinguishes multiple devices of same type), `MemoryScope`. `hint_on_device` annotates an expression's intended device; `to_vdevice` emits an explicit cross-device tensor copy. The RFC's `UpdateVDevice` and `RealizeVDevice` compiler passes propagate and materialize device placement. Crucially, a community commenter (Lunderberg) raised whether `vdevice_id` could support dynamic dispatch using `PrimExpr` instead of a static integer — this was not addressed in the RFC. The RFC does not include performance cost estimates for `to_device` cross-device copies.
- **Key detail:** RFC explicitly frames this as compile-time device assignment, not runtime selection. The design has no mechanism for "try GPU 0, fall back to GPU 1 of a different vendor." This is the gap libkdl fills.

### S5 — Tracking Issue: Heterogeneous Execution for Relax (GitHub #15101)
- **URL:** https://github.com/apache/tvm/issues/15101
- **Type:** GitHub issue / implementation tracker
- **Date:** Opened 2023; last activity late 2023
- **Relevance/Novelty:** 7/10
- **Summary:** Tracks phased implementation of RFC S4. Phase 1 (VDevice data structure, TVMScript parser/printer, `hint_on_device`/`to_vdevice`/`to_device` builtins, `UpdateVDevice`/`RealizeVDevice` passes) was merged. Phase 2 (cost-model-driven device assignment, end-to-end heterogeneous execution) has no merged PRs as of late 2023. Activity dropped after the OctoAI acquisition closed (September 2024).
- **Key detail:** The feature is structurally partially implemented but not end-to-end functional. Post-acquisition, the core TVM commercial team is now inside NVIDIA, which has no interest in advancing AMD/Intel heterogeneous backends.

### S6 — Relax: Composable Abstractions for End-to-End Dynamic ML (arXiv:2311.02103)
- **URL:** https://arxiv.org/pdf/2311.02103
- **Type:** Academic paper (arXiv preprint, 2023)
- **Relevance/Novelty:** 8/10
- **Summary:** Formalizes Relax IR design. First-class dynamic shapes via symbolic shape variables propagated through the IR; shape functions are compiled as separate TIR functions that execute at runtime before kernel dispatch to allocate correct-sized output buffers. Dataflow regions make optimization boundaries explicit. "Composable abstractions" means IR, VM, and runtime can be individually substituted — the paper cites this as enabling third-party backends without modifying TVM's compiler core. This composability is what BYOC (Bring Your Own Codegen) exploits.
- **Key detail:** Shape function execution before kernel dispatch adds latency proportional to IR graph depth. For dynamic-batch LLM inference (the primary workload), DLight trades ~20-40% throughput for zero tuning time — this is the measured specialization cost.

### S7 — MetaSchedule: Unified ML-Based Tensor Program Optimization (NeurIPS 2022)
- **URL:** https://openreview.net/forum?id=nyCr6-0hinG
- **Type:** Academic paper (conference)
- **Date:** NeurIPS 2022
- **Relevance/Novelty:** 7/10
- **Summary:** TVM's auto-tuner searches parameterized schedule spaces (tile sizes, vectorization, tensor core usage) using evolutionary search + learned cost model. Produces hardware-specific specialized kernels: a schedule tuned for A100 is suboptimal on GTX 1650 or RX 6800. Tuning is restricted to static shapes. The database of tuned schedules is per-target — there is no shared schedule database across vendors. This is the reference system for "per-hardware kernel specialization cost."
- **Key detail:** MetaSchedule makes explicit that TVM's performance story depends on per-target tuning. A single deployment cannot be optimally tuned for multiple vendors without maintaining multiple tuning databases. This is the core cost TVM accepts for performance; libkdl instead relies on pre-compiled per-hardware variants shipped in the fat binary.

### S8 — DLight: Hardware-Aware Tuning-Free Schedule Generation (TVM Discuss, 2024)
- **URL:** https://discuss.tvm.apache.org/t/dlight-enabling-fast-and-efficient-kernel-generation-by-hardware-information/16273
- **Type:** Design document / development thread
- **Date:** 2024
- **Relevance/Novelty:** 7/10
- **Summary:** DLight is a set of heuristic schedule rules that query device properties (compute capability, memory bandwidth) to select tile sizes and parallelism without measurement. Targets dynamic-shape LLM inference. Achieves reasonable performance (~20-40% below fully tuned MetaSchedule) with zero tuning time. Device property queries at schedule-rule-selection time — this is a compile-time query, not a runtime dispatch mechanism.
- **Key detail:** DLight's device-property-queried schedule selection is architecturally analogous to libkdl's capability query for kernel variant selection, but happens at compile time inside the auto-scheduler rather than at library load time.

### S9 — BYOC: Bring Your Own Codegen to TVM (TVM Blog + Docs, 2020/current)
- **URL:** https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm
- **Type:** Blog post + official documentation
- **Date:** 2020, documentation current
- **Relevance/Novelty:** 6/10
- **Summary:** BYOC allows hardware vendors to inject external codegen (e.g., TensorRT, NNAPI, Qualcomm QNN) into TVM's compilation pipeline. Annotated subgraphs are partitioned and handed off to the external codegen; unsupported ops fall back to TVM's native backends. This is the mechanism by which heterogeneous execution across specialized accelerators works in production today. However, BYOC is also a compile-time dispatch — the partition is fixed in the compiled artifact.
- **Key detail:** BYOC + graph partitioning is TVM's production heterogeneous dispatch story. It is powerful but static: the fallback hierarchy (e.g., TensorRT → CUDA → CPU) is baked in at compile time, not resolved at runtime.

### S10 — MLC-LLM: Universal LLM Deployment (GitHub + Docs, 2024-2025)
- **URL:** https://github.com/mlc-ai/mlc-llm
- **Type:** Open-source project / documentation
- **Date:** Active 2024-2025
- **Relevance/Novelty:** 8/10
- **Summary:** MLC-LLM is TVM Unity's production deployment system for LLMs. Supports CUDA, ROCm, Vulkan, Metal, OpenCL, WebGPU, CPU. Portability is achieved by compile-time target selection (`mlc_llm compile model --target vulkan`), not runtime dispatch. The Vulkan artifact achieves cross-vendor runtime portability (runs on NVIDIA, AMD, Intel, mobile) as a side effect of SPIR-V portability, but performance is below native CUDA for NVIDIA hardware. January 2025 blog post introduces "microserving" (multi-engine prefill/decode splitting) — no new multi-vendor dispatch mechanism.
- **Key detail:** MLC-LLM is the primary production system validating TVM Unity's portability. Its Vulkan story is the nearest real-world analog to vendor-agnostic dispatch, but it is a portability layer, not a dispatch layer — there is no performance-based selection between CUDA and Vulkan at runtime.

### S11 — MicroTVM: Edge Device Dispatch (ACM EDGEAI 2023, arXiv:2304.04842)
- **URL:** https://dl.acm.org/doi/10.1145/3615338.3618125
- **Type:** Workshop paper (ACM EDGEAI 2023)
- **Date:** 2023
- **Relevance/Novelty:** 5/10
- **Summary:** MicroTVM compiles ML models to AOT (ahead-of-time) C code for bare-metal MCUs (ARM Cortex-M, RISC-V, ESP32). No OS, no dynamic linker — the entire model is a static C binary. "Dispatch" in microTVM means partitioning computation between MCU cores and optional hardware NPUs via BYOC. The AoT executor replaces TVM's standard graph executor, eliminating runtime IR interpretation overhead.
- **Key detail:** MicroTVM's AOT executor is the extreme end of static dispatch — zero runtime overhead, zero flexibility. Relevant to libkdl only as the opposite design point: libkdl introduces controlled dynamic dispatch overhead to gain portability that AoT cannot provide.

### S12 — Apache TVM v0.19.0 Release Notes (January 2025)
- **URL:** https://github.com/apache/tvm/issues/17575
- **Type:** Release notes / GitHub issue
- **Date:** January 2025
- **Relevance/Novelty:** 6/10
- **Summary:** v0.19.0 focus areas: Relax (especially PyTorch frontend improvements), OpenCL backend improvements, MetaSchedule stability fixes. No mention of heterogeneous execution advancement. Confirms the trajectory: LLM workloads and PyTorch interop are the development priority; broader multi-vendor dispatch work is stalled.
- **Key detail:** v0.19.0 release confirms that as of early 2025, TVM's heterogeneous execution RFC work has not progressed to a user-facing feature.

### S13 — Apache TVM v0.20.0 Release Notes (April 2025)
- **URL:** https://www.mail-archive.com/commits@tvm.apache.org/msg113118.html
- **Type:** Release notes
- **Date:** April 2025
- **Relevance/Novelty:** 5/10
- **Summary:** v0.20.0 focus areas: Relax with PyTorch frontend, CUDA improvements. No new multi-device dispatch capability. The release trajectory confirms: TVM is converging on NVIDIA-centric LLM optimization following the OctoAI/NVIDIA acquisition. AMD/Intel/cross-vendor work is community-maintained with limited core-team investment.
- **Key detail:** Post-acquisition trajectory is clear from two consecutive releases. The window for TVM to develop vendor-agnostic dispatch is closing; this is where libkdl's MLIR-native approach differentiates.

### S14 — MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge (arXiv:2410.08855, 2024)
- **URL:** https://arxiv.org/html/2410.08855v1
- **Type:** Academic paper (arXiv, October 2024)
- **Date:** October 2024
- **Relevance/Novelty:** 7/10
- **Summary:** MATCH extends TVM with hardware-aware deployment for heterogeneous edge SoCs (MCU + NPU). Uses BYOC to dispatch operators to NPU when supported, falling back to CPU TIR otherwise. Generates optimized hardware-specific C code while dispatching matched operators to external vendor libraries. Demonstrates TVM's BYOC extensibility for custom accelerator dispatch — but the dispatch graph is static, produced at compile time from hardware capability XML descriptions.
- **Key detail:** MATCH is the closest to a "capability-driven dispatch" workflow in TVM's ecosystem. The capability description is a static XML file, not a runtime query. Contrast: libkdl queries ELF section headers and GPU capability registers at load time.

---

## Synthesis: TVM's Dispatch Architecture

### What TVM does well

1. **Broadest target coverage** of any ML compiler: CUDA, ROCm, Vulkan, Metal, OpenCL, WebGPU, x86, ARM, RISC-V, Qualcomm, custom NPUs via BYOC.
2. **PackedFunc** provides a clean, low-overhead (~stack-save cost) universal function call interface across language and device boundaries.
3. **DeviceAPI/GetAttr** provides capability querying infrastructure — the primitives exist, they are used at codegen time.
4. **Vulkan/SPIR-V** achieves genuine cross-vendor runtime portability at the cost of ~15-30% performance vs native CUDA on NVIDIA hardware.
5. **BYOC** enables heterogeneous operator dispatch within a single compiled model, with static fallback hierarchies.

### What TVM does not do

1. **No runtime backend selection** between alternative vendors (e.g., "prefer CUDA, fall back to ROCm, fall back to Vulkan"). This must be implemented in application code.
2. **No hardware discovery at load time** — there is no mechanism that inspects available GPUs and selects the compiled module variant for the best available device.
3. **No fat binary dispatch** — there is no single artifact format that contains CUDA + ROCm + Vulkan variants and selects at runtime.
4. **No automatic cross-device fallback** — if a CUDA device is unavailable, a CUDA-compiled TVM module fails; it does not retry with Vulkan.
5. **No dynamic vendor selection** — the VDevice RFC proposes compile-time placement; dynamic `PrimExpr`-based device selection was explicitly noted as out-of-scope by the RFC authors.

### Architectural gap libkdl fills

```
TVM model:
  compile-time: select target → produce target-specific .so
  runtime:      load .so → call PackedFunc → CUDA/Vulkan/ROCm runs

libkdl model:
  compile-time: compile all targets → pack into .kdl fat binary
  load-time:    query hardware capabilities → select best kernel variant
  runtime:      call dispatch table → selected variant executes
```

The gap is the load-time capability query + variant selection step. TVM has the infrastructure (DeviceAPI::GetAttr), the target abstraction, and the per-target codegen. What it lacks is the fat binary format and the selection logic that binds them at deployment. libkdl is that missing layer.

---

## Relevance Ratings to Vendor-Agnostic Kernel Dispatch Runtime

| Source | Relevance | Primary Value |
|--------|-----------|---------------|
| S2 — Device/Target Interactions | 9/10 | DeviceAPI::GetAttr as capability query primitive |
| S3 — Vulkan Runtime | 9/10 | SPIR-V cross-vendor portability; closest TVM analog |
| S4 — Relax Heterogeneous RFC | 9/10 | VDevice design; explicit gap in runtime selection |
| S1 — Runtime System | 8/10 | PackedFunc overhead characterization |
| S6 — Relax paper | 8/10 | Shape function dispatch overhead; DLight cost |
| S10 — MLC-LLM | 8/10 | Production evidence that Vulkan portability works at LLM scale |
| S14 — MATCH | 7/10 | Capability-driven dispatch via static XML; design contrast |
| S7 — MetaSchedule | 7/10 | Per-hardware specialization cost model |
| S8 — DLight | 7/10 | Tuning-free device-property-queried scheduling |
| S5 — Tracking Issue #15101 | 7/10 | Implementation status evidence; stalled post-acquisition |
| S9 — BYOC | 6/10 | Static heterogeneous dispatch in production |
| S12, S13 — Release Notes | 6/10 | Post-acquisition trajectory evidence |
| S11 — MicroTVM | 5/10 | Opposite design point (fully static AoT) |

---

## Key Quotable Claims (for poster)

1. "TVM's PackedFunc overhead is approximately the cost of saving a few values on the stack" — confirming that kernel-granularity dispatch overhead in a properly designed runtime is negligible.

2. "TVM requires the target to be specified at compile time; there is no mechanism for runtime vendor selection between alternative backends" — framing the problem libkdl solves.

3. "MLC-LLM's Vulkan target achieves cross-vendor execution at runtime through SPIR-V, but the capability envelope is fixed at compile time from a single target device's query" — the static vs dynamic portability distinction.

4. "The VDevice RFC's community discussion explicitly raised the need for `PrimExpr`-based dynamic device IDs for runtime selection but this was not implemented" — evidence that the research community identified the gap.

5. "MATCH demonstrates capability-driven dispatch via static XML hardware descriptors — libkdl moves this to a runtime ELF section header query, eliminating the need for ahead-of-time hardware knowledge."

---

## Risks and Concerns for Poster

1. **TVM is cited as prior art, not a strawman.** Reviewers may be TVM contributors. The comparison must be precise: libkdl does not replace TVM's compilation pipeline; it fills the runtime dispatch gap that TVM explicitly left unfilled.

2. **Vulkan nuance.** Vulkan on NVIDIA works but with ~15-30% performance penalty vs CUDA for dense linear algebra. Any claim about "Vulkan as a portable backend" must acknowledge this tradeoff.

3. **Post-acquisition trajectory.** Citing TVM as "the state-of-the-art" requires a footnote: NVIDIA acquisition of OctoAI (Sept 2024) has concentrated TVM's commercial roadmap toward NVIDIA-only optimization. This is a risk to TVM's vendor-agnostic future, not to the validity of its current design.

4. **DLight ~20-40% overhead** is a compilation trade-off, not a runtime dispatch overhead. Conflating the two would be an error in the poster narrative.

---

## Cross-References to Other Waves

- Wave 01 (IREE HAL): Compare IREE's HAL device abstraction vs TVM's DeviceAPI — both query capabilities, but HAL uses dynamic dispatch at the HAL layer; TVM queries at codegen time.
- Wave 01 (SPIR-V Portable IR): TVM's Vulkan/SPIR-V approach is the concrete embodiment of SPIR-V portability discussed there.
- Wave 02 (Fat Binaries): TVM explicitly lacks fat binary dispatch — this is libkdl's direct contribution.
- Wave 02 (LLVM Offloading): TVM uses LLVM's codegen backends but not LLVM's offloading infrastructure; this is an architectural contrast point.
