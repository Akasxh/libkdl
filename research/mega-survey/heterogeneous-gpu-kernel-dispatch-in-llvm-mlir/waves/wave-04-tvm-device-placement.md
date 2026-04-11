# Wave 04 — TVM Relay/Relax Device Placement and Heterogeneous Execution Scheduling

**Angle:** tvm-relay-device-placement
**Query:** TVM device annotation, VirtualDevice, PlanDevices, Relax VDevice, BYOC partitioning
**Date:** 2026-04-06
**Sources surveyed:** 12 primary sources (RFCs, official docs, GitHub issues, papers, forum threads)

---

## Executive Summary

TVM has two distinct heterogeneous device placement architectures — one per IR generation. In Relay (classic), device placement is expressed via `on_device()` annotation ops and resolved by the `PlanDevices` constraint-propagation pass, which inserts `device_copy` nodes at device boundaries. In Relax (TVM Unity), the mechanism is redesigned around first-class `VDevice` descriptors on `TensorStructInfo`, with `hint_on_device`/`to_vdevice` ops and two dedicated compiler passes (`UpdateVDevice`, `RealizeVDevice`). Both systems share the same fundamental limitation relative to libkdl: device placement is decided statically at compile time, and no runtime hardware-capability query selects between compiled variants. The `PlanDevices` pass has a known algorithmic deficiency — it cannot handle the case where a single variable is consumed by operators assigned to different devices, requiring manual user workarounds. The Relax redesign fixes this at the data-structure level (VDevice is a first-class IR field) but leaves the runtime dispatch gap unchanged.

**Relevance to vendor-agnostic kernel dispatch runtime:** 8/10 — provides detailed prior-art grounding for the static-vs-dynamic dispatch argument; the PlanDevices deficiency is a concrete case study motivating libkdl's load-time approach.

**Novelty vs. wave-03 (TVM unified runtime):** Wave-03 covered what TVM's overall dispatch architecture achieves and lacks at a system level. This wave documents the internal mechanism by which TVM performs device placement — the algorithmic specifics of PlanDevices, the SEScope/VirtualDevice constraint lattice, the annotation API, and the Relax redesign. Not duplicative.

---

## Sources

### S1 — TVM Relay RFC: Compilation for Heterogeneous Execution (GitHub Issue #2296, 2018)
- **URL:** https://github.com/apache/tvm/issues/2296
- **Type:** Original design RFC
- **Date:** 2018
- **Relevance:** 7/10 | **Novelty:** 6/10
- **Summary:** The original RFC establishing TVM's heterogeneous execution model. Introduced the two core mechanisms that persist to this day: (a) a modified `build()` API accepting a `Dict[device_type, target_string]` instead of a single target, with a `fallback_device` parameter; (b) an `annotate_ops(expr, op_name_dev_map, fallback_device)` pass that assigns device attributes to operators. Device communication was acknowledged to require "copy ops" that bypass standard `fcompute`/`fschedule` requirements. The RFC also introduced the alternative of wrapping ops in synthetic `on_device()` annotations rather than modifying the AST directly — the latter approach won.
- **Key detail:** The original RFC explicitly contemplated "cost-function-based strategies considering communication overhead" for automated device placement, but this was never implemented. The gap between the RFC's aspiration and what shipped is the compile-time placement constraint.

### S2 — TVM Device/Target Interactions Architecture Doc (Official Docs, v0.21.dev0)
- **URL:** https://tvm.apache.org/docs/arch/device_target_interactions.html
- **Type:** Official architecture documentation
- **Date:** Current (mirrors dev HEAD ~2025-2026)
- **Relevance:** 8/10 | **Novelty:** 6/10
- **Summary:** Defines the three-layer abstraction: `DeviceAPI` (hardware handle — memory, streams, sync), `Target` (property lookup table — thread count, warp size, capability flags), and target code generators (registered under `"target.build.foo"` in `target_kind.cc` via `TVM_REGISTER_TARGET_KIND`). `DeviceAPI::GetAttr` is the capability-query primitive: it accepts a `DeviceAttrKind` enum (device name, max threads, warp size, clock rate, compute version) and returns a queried value or `nullptr` if unsupported. Multiple target kinds can map to the same `DLDeviceType` (e.g., both `"llvm"` and `"c"` map to `kDLCPU`). This infrastructure exists for capability queries; it is used at compile time in auto-schedulers (DLight reads compute capability via `GetAttr` to select tile sizes), not at load time for variant selection.
- **Key detail:** `DeviceAPI::GetAttr` is the right primitive for a runtime capability-query dispatch system; TVM's design uses it only at compile time. libkdl's load-time query is architecturally the same call, made at a different program phase.

### S3 — RFC 0045: First-Class Virtual Device (tvm-rfcs, 2021–2022)
- **URL:** https://github.com/apache/tvm-rfcs/blob/main/rfcs/0045-first-class-virtual-device.md
- **Type:** RFC
- **Date:** 2021–2022
- **Relevance:** 9/10 | **Novelty:** 8/10
- **Summary:** Proposes adding a `virtual_device_` field directly to `RelayExprNode` (analogous to `checked_type_`), so that device information is stored on every node rather than inferred from surrounding `on_device()` wrapper ops. The motivating problem: existing `on_device()` annotations require `DeviceAware` visitors that maintain device context during traversal — passes that introduce or reorganize expressions must manually propagate device context or re-run `PlanDevices`. The RFC's thesis is that virtual device should be as first-class as type — readable directly from a node without context. The `PlanDevices` pass would then maintain this field rather than inserting/removing `on_device` wrapper nodes.
- **Key detail:** The RFC deferred details of the PlanDevices algorithm itself, pointing to source code. The important architectural point: the previous design stored device information *outside* the expression tree (in wrapper nodes and function attributes); the RFC moves it *inside* the expression tree as a typed field, enabling passes to be device-aware without special DeviceAware visitors.

### S4 — How Does the PlanDevices Pass Work? (TVM Discuss, Thread #13184, 2022)
- **URL:** https://discuss.tvm.apache.org/t/how-does-the-plandevices-pass-work/13184
- **Type:** Technical forum discussion
- **Date:** 2022
- **Relevance:** 9/10 | **Novelty:** 9/10
- **Summary:** Community thread asking for algorithmic details of `PlanDevices`. Reveals that:
  - `PlanDevices` runs a `DeviceAnalyzer` that traverses the expression tree and tracks "initial function domain" and "function body domain" for each function being processed.
  - The pass runs after `SimplifyInference`, which expands batch normalization into composite operators (add, sqrt, etc.) — so `PlanDevices` sees an expanded graph that doesn't match the user's source IR.
  - Device constraint resolution uses a union-find based approach to group expressions that must share the same device.
  - The pass inserts `device_copy` nodes at device boundaries after resolving constraints.
  - There is no public documentation of the constraint lattice or join operation semantics; these must be read from `src/relay/transforms/device_planner.cc`.
- **Key detail:** The pass operates on the post-`SimplifyInference` graph, which can surprise users who annotated ops by name in the original model — the annotation may not match the expanded form that `PlanDevices` sees.

### S5 — PlanDevices Bug: Shared Input Across Different Target Devices (GitHub Issue #15019, 2023)
- **URL:** https://github.com/apache/tvm/issues/15019
- **Type:** Bug report
- **Date:** 2023
- **Relevance:** 9/10 | **Novelty:** 9/10
- **Summary:** Critical bug demonstrating a fundamental limitation in `PlanDevices`' constraint propagation. When an expression `(a+b) - (b+c)` is compiled with the first `+` on CPU and the second on GPU, the pass:
  1. Visits `(a+b)` and assigns variable `b` to CPU.
  2. Visits `(b+c)` and attempts to assign `b` to GPU.
  3. Detects a conflict and throws: `"Function parameters and result VirtualDevices do not match those of call."`
  The pass has no backtracking mechanism and cannot automatically resolve the conflict by inserting `device_copy` for `b` (one copy feeding the CPU op, another feeding the GPU op). The expected behaviors — either automatic `device_copy` insertion or input replication — were not implemented.
- **Key detail:** This is not a corner case; shared activations between operator groups on different devices are common in residual networks (skip connections feeding both local and remote paths). The bug means heterogeneous execution is practically restricted to models where the device boundary is a clean cut — each activation used by exactly one device tier.

### S6 — Heterogeneous Execution API: CPU/GPU Setup (TVM Discuss, Thread #11561, 2021)
- **URL:** https://discuss.tvm.apache.org/t/how-to-do-heterogeneous-execution-on-cpu-and-gpu/11561
- **Type:** Community Q&A
- **Date:** 2021
- **Relevance:** 7/10 | **Novelty:** 6/10
- **Summary:** Demonstrates the concrete user-facing API for Relay heterogeneous execution:
  ```python
  # Annotate ops with device targets
  relay.op.annotation.on_device(y + z, tvm.cpu())

  # Build with multi-target dict
  lib = relay.build(relay_mod,
                    target={"cpu": "llvm", "gpu": "cuda"},
                    params=params)

  # Instantiate runtime with both device handles
  m = graph_executor.GraphModule(lib["default"](tvm.cpu(), tvm.gpu()))
  ```
  The multi-target `build()` triggers the heterogeneous compilation pipeline: `PlanDevices` runs, `device_copy` nodes are inserted, and the graph executor is configured to dispatch to both CPU and GPU device APIs.
- **Key detail:** The user must provide device handles in the correct order corresponding to the `target` dict. There is no runtime negotiation — the executor is initialized against a static set of devices specified at module creation time.

### S7 — RFC: Heterogeneous Execution for Relax (TVM Discuss, Thread #14670, 2023)
- **URL:** https://discuss.tvm.apache.org/t/rfc-unity-relax-heterogeneous-execution-for-relax/14670
- **Type:** RFC
- **Date:** April 2023
- **Relevance:** 9/10 | **Novelty:** 8/10
- **Summary:** Redesigns device placement for Relax IR. Key changes vs Relay:
  - **`VDevice`**: Three-field descriptor — `Target`, `vdevice_id` (integer, distinguishes multiple devices of same type), `MemoryScope`. Annotation syntax: `R.Tensor((2, 3), "float32", "cuda:0")`.
  - **`TensorStructInfo`**: Gets an optional `vdevice` member — device placement is part of the tensor's type, not a wrapper op.
  - **`hint_on_device(data, dst_vdevice)`**: Soft annotation — hints intended device, resolved by `RealizeVDevice` pass. Removed from IR after pass completes.
  - **`to_vdevice(data, dst_vdevice)`**: Hard annotation — emits an explicit cross-device copy. Survives the pass unchanged.
  - **Normalization**: `InferStructInfo` propagates device info forward; partially constrained VDevices default to the first vdevice in `global_infos`.
  - **`RealizeVDevice` pass**: Backward propagation; resolves all device constraints across function boundaries and emits `to_vdevice` nodes at boundaries.
  - Community commenter (Lunderberg) explicitly asked: "Can `vdevice_id` be a `PrimExpr` to support runtime device selection?" — the RFC authors responded that dynamic selection was out of scope for this RFC.
- **Key detail:** The Relax redesign fixes the structural fragility of Relay's `on_device()` wrapper approach by making device a first-class type attribute. But it preserves the same fundamental property: device selection is compile-time. The `PrimExpr` vdevice_id question identifies the exact design extension that would enable runtime dispatch; the decision to defer it is TVM's explicit choice to remain static.

### S8 — Tracking Issue: Heterogeneous Execution for Relax (GitHub Issue #15101, 2023)
- **URL:** https://github.com/apache/tvm/issues/15101
- **Type:** Implementation tracker
- **Date:** Opened June 2023; closed December 20, 2023
- **Relevance:** 8/10 | **Novelty:** 7/10
- **Summary:** Phased implementation plan, all marked complete:
  - P1: `VDevice` data structure; TVMScript parser/printer support. ✓
  - P2: `hint_on_device`, `to_vdevice`, `to_device` builtin ops. ✓
  - P3: `InferStructInfo` updates; Relax VM codegen. ✓
  - P4: `UpdateVDevice` pass. ✓ (PR #15570, merged 2023-08-15)
  - P5: `RealizeVDevice` pass. ✓ (PR #15636, merged 2023-08-28)
  - End-to-end multi-device test cases: PR #15447 (merged 2023-08-01).
  The feature was structurally completed in Q3 2023. Subsequent development activity dropped significantly following the OctoAI/NVIDIA acquisition announcement (September 2024). Phase 2 goals — cost-model-driven device assignment and automated placement without user annotation — were never implemented.
- **Key detail:** "Users don't need to pass multiple targets to `relax.build`; targets are defined in the vdevice list of `global_infos` of the IRModule." This is an improvement over Relay's API but the underlying model is unchanged: targets are compiled and resolved at build time.

### S9 — Relax Op Documentation: `to_vdevice` and `hint_on_device` (TVM Docs, v0.24.dev0)
- **URL:** https://tvm.apache.org/docs/reference/api/python/relax/op.html
- **Type:** Official API documentation
- **Date:** Current
- **Relevance:** 7/10 | **Novelty:** 6/10
- **Summary:**
  - `tvm.relax.op.to_vdevice(data: Expr, dst_vdevice: VDevice) → RelaxExpr` — explicit cross-device tensor transfer. Marked as pure, no in-place operation; can appear in DataFlow blocks. Survives `RealizeVDevice`.
  - `tvm.relax.op.hint_on_device(data: Expr, dst_vdevice: Device, memory_scope: str = 'global') → RelaxExpr` — soft hint consumed and removed by `RealizeVDevice`. Provides device information to the pass without committing to a specific transfer implementation.
  - The distinction: `to_vdevice` is a hard cross-device copy (analogous to Relay's `device_copy`); `hint_on_device` is a compile-time annotation that disappears after device planning.
- **Key detail:** The separation of hint (soft, compile-time) from explicit copy (hard, persists to runtime) is an improvement over Relay's single `on_device` + post-processing model. It allows the compiler to make smarter placement decisions without the user having to specify every transfer point.

### S10 — MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge (arXiv:2410.08855, 2024)
- **URL:** https://arxiv.org/html/2410.08855v1
- **Type:** Academic paper (arXiv, October 2024)
- **Date:** October 2024
- **Relevance:** 8/10 | **Novelty:** 8/10
- **Summary:** MATCH extends TVM's BYOC framework for heterogeneous edge SoCs (MCU + NPU). The graph partitioning mechanism: Pattern Matcher scans the operator graph and tags node groups with hardware module references; for each positive match, MATCH's DSE tool estimates latency/energy via ZigZag's LOMA engine; the pattern is assigned to the hardware module minimizing the cost metric. This is an iterative exploration through hardware modules, unlike TVM's standard BYOC which does a single pass. For GAP9 (9-core RISC-V + NE16 accelerator + 8-core cluster), the per-operator assignment considers all three compute units. Achieves 60.88x speedup over plain TVM on DIANA, 67.83x on GAP9.
- **Key detail:** MATCH demonstrates capability-driven device placement using a static XML hardware descriptor (not a runtime query). This is the closest existing system to libkdl's load-time capability query — the design is proven valid; MATCH just performs the query at compile time from a spec file rather than at load time from hardware registers. Migrating MATCH's cost-model selection to load time is conceptually what libkdl does.

### S11 — Compass Apache TVM: Arm NPU Heterogeneous Execution (GitHub, 2024)
- **URL:** https://github.com/Arm-China/Compass_Apache_TVM
- **Type:** Vendor extension / open-source project
- **Date:** Active 2024
- **Relevance:** 7/10 | **Novelty:** 7/10
- **Summary:** Arm China's production extension of Apache TVM for their Zhouyi NPU. Uses BYOC to partition NN operators: NPU-supported ops form NPU subgraphs; remaining ops form CPU subgraphs. The BYOC graph partition split is static (compile-time). The runtime executes the compiled model heterogeneously and automatically — the heterogeneity is transparent to the end user at inference time, but fixed at compile time. Key design insight: the partition is expressed as a static annotation in the compiled graph, and the runtime executor simply dispatches to the correct device handle for each subgraph.
- **Key detail:** Production evidence that BYOC-based heterogeneous execution works in the field. The system is static but the user experience is seamless — the partition complexity is hidden in the compiler. libkdl can match this user experience while adding vendor selection flexibility.

### S12 — Relay VM RFC: Heterogeneous Execution with Union-Find Device Analysis (GitHub Issue #4178, 2019)
- **URL:** https://github.com/apache/tvm/issues/4178
- **Type:** RFC
- **Date:** 2019
- **Relevance:** 7/10 | **Novelty:** 7/10
- **Summary:** Original proposal for porting Relay heterogeneous execution to the Relay VM (which at the time supported only a single device). Proposed: (a) union-find-based context analysis pass to group IR nodes by device context; (b) `DeviceCopy` instruction in the VM bytecode to copy data across devices; (c) memory allocation and planning passes made context-aware. The VM's memory planning (`memory_alloc`, `memory_plan`) would track which device each allocation targets. Static vs. dynamic device selection was flagged as an open design question at this stage.
- **Key detail:** The union-find approach for constraint grouping is the algorithmic ancestor of `PlanDevices`. The bug in S5 (shared input between two different devices) represents a case where union-find fails: unioning `b` with "CPU" and `b` with "GPU" simultaneously creates a constraint contradiction, and the system has no conflict resolution strategy.

---

## Synthesis: The TVM Device Placement Mechanism in Detail

### Relay Pipeline (Classic TVM)

```
User IR (relay.Module)
  ↓  relay.annotation.on_device(expr, device)
     [wraps target ops in synthetic on_device() nodes]
  ↓  relay.transform.SimplifyInference()
     [expands BatchNorm → add + sqrt + etc.]
  ↓  relay.transform.PlanDevices(fallback_device)
     [DeviceAnalyzer: union-find constraint grouping]
     [resolves on_device() annotations to SEScope/VirtualDevice labels]
     [inserts device_copy nodes at device boundaries]
     [removes on_device() wrapper nodes]
  ↓  relay.build(mod, target={"cpu": "llvm", "gpu": "cuda"})
     [compiles per-device subgraphs]
     [generates device_copy → TVMArrayCopyFromTo calls]
  ↓  graph_executor.GraphModule(lib["default"](tvm.cpu(), tvm.gpu()))
     [runtime dispatches each op to its device handle]
     [sees __copy ops → calls TVMArrayCopyFromTo]
```

**Constraint propagation algorithm:** Union-find grouping of IR nodes. An `on_device(expr, device_A)` annotation unifies `expr` with `device_A`. If `expr` is later also transitively constrained to `device_B ≠ device_A`, the unification fails (throws). There is no join or meet operation that could resolve conflicts — the system is strictly conflict-detection-only, not conflict-resolution.

**device_copy mechanics:** At device boundaries, `PlanDevices` inserts `relay.device_copy(src, src_device, dst_device)` nodes. During codegen these map to `__copy` packed functions. At runtime, `graph_executor` calls `TVMArrayCopyFromTo(src_tensor, dst_tensor)` which dispatches to the appropriate `DeviceAPI::CopyDataFromTo` for the involved device pair.

### Relax Pipeline (TVM Unity)

```
Relax IRModule (with VDevice global_infos)
  ↓  User annotation (optional):
     hint_on_device(expr, vdevice)   — soft, compile-time only
     to_vdevice(expr, vdevice)       — hard, generates actual copy
     R.Tensor((n,), "float32", "cuda:0")  — in-type device annotation
  ↓  InferStructInfo (normalization)
     [propagates VDevice forward through IR]
     [partially-constrained → defaults to first vdevice in global_infos]
  ↓  UpdateVDevice pass (PR #15570)
     [backward propagation of VDevice constraints]
     [cross-function device information flow]
  ↓  RealizeVDevice pass (PR #15636)
     [resolves all hint_on_device annotations]
     [removes hint_on_device nodes; inserts to_vdevice where needed]
     [VDevice now stable: every TensorStructInfo has concrete vdevice]
  ↓  relax.build(mod)  — targets from global_infos, no separate target dict
     [TIR lowering per prim_func based on target attribute]
     [Relax VM codegen: multiple device handles in VM state]
  ↓  Relax VM executes with multiple device contexts
```

**Improvement over Relay:** VDevice is part of `TensorStructInfo` (the tensor's type), not a wrapper node. Passes see device information directly on expressions, removing the DeviceAware visitor requirement. `hint_on_device` is not committed until `RealizeVDevice`, allowing the pass to make globally consistent choices before inserting copies.

### The Fundamental Static Constraint

Both pipelines share the same property: device assignment is decided no later than `relay.build()` / `relax.build()`. The Relax VM can hold multiple device contexts simultaneously, but which context executes which function is determined at compile time. There is no mechanism for:
- Querying available hardware at module load time and selecting variant accordingly.
- Falling back to a different vendor if the preferred device is absent.
- Selecting between CUDA and ROCm variants of the same kernel at runtime.

The community recognized this: the Relax RFC discussion explicitly raised `PrimExpr`-typed `vdevice_id` to enable runtime device selection, and the design decision was to defer this. The tracking issue (S8) closed with this gap intact.

---

## Angle Assessment

**Relevance to libkdl / vendor-agnostic dispatch:** 8/10

TVM's device placement system is the richest prior art for heterogeneous kernel dispatch in the ML compiler ecosystem. The specific mechanisms — `PlanDevices`, VDevice, `to_vdevice` — define the conceptual vocabulary for the problem. The gaps are well-documented and explicit: TVM chose static placement for architectural simplicity. libkdl's contribution is exactly the missing layer: a load-time capability query driving selection among pre-compiled device variants.

**Novelty vs. existing wave coverage:** 8/10

Wave-03 covered TVM's dispatch architecture at system level. This wave provides the internal algorithmic detail — the union-find constraint engine, the annotation API pipeline, the specific failure modes (`PlanDevices` shared-variable bug), and the Relax redesign's improvements and remaining limitations. The PlanDevices bug (S5) is new and directly useful for the poster narrative: it shows that even TVM's internal heterogeneous dispatch is fragile at common residual-network patterns.

**Poster relevance:** High. The `PrimExpr`-vdevice_id gap (S7, Lunderberg comment) is a quotable statement from TVM's own community identifying the exact problem libkdl solves. The MATCH 60x speedup from proper device placement (S10) quantifies what correct placement decisions are worth.

---

## Key Quotable Claims (for poster)

1. "TVM's `PlanDevices` pass uses union-find constraint grouping with no conflict-resolution mechanism — it fails when a tensor is consumed by operators on two different devices, a pattern common in residual networks." (S5, #15019)

2. "The Relax RFC discussion explicitly asked whether `vdevice_id` could be a `PrimExpr` to support runtime device selection. The decision was to defer this." (S7, RFC #14670, Lunderberg comment) — this is the exact gap libkdl fills.

3. "TVM Relay stores device information in synthetic `on_device()` wrapper nodes — a design so fragile that RFC 0045 was filed specifically to move device to a first-class IR field." (S3, RFC 0045)

4. "MATCH achieves 60x speedup over plain TVM by replacing static operator-to-device assignment with cost-model-driven placement — demonstrating that dynamic device selection decisions matter at 2-3 orders of magnitude." (S10)

5. "At runtime, TVM's `graph_executor` dispatches cross-device copies via `TVMArrayCopyFromTo` — the same operation libkdl's runtime uses for device staging. The copy primitive is shared; what differs is whether the routing decision was made at compile time or load time." (S2, S6)

---

## Risks and Concerns

1. **PlanDevices bug scope:** Bug #15019 may have been fixed post-2023 without a linked closing commit. Should verify against current TVM main before citing as an open issue. If fixed, the argument still stands for historical correctness — the design had this limitation for years and it constrained practical heterogeneous use.

2. **Relax VDevice completeness:** The tracking issue closed December 2023 with "COMPLETED," but post-acquisition development slowdown means end-to-end heterogeneous execution may not be tested against real multi-GPU workloads. Claims about Relax heterogeneous capabilities should be verified against a running system, not just issue closure status.

3. **MATCH cost-model comparison:** MATCH's 60x speedup is over "plain TVM" (no BYOC, no custom scheduling) — not over TVM with BYOC-based dispatch. The speedup is from proper accelerator utilization, not purely from device placement decisions. The framing must be precise to avoid reviewer challenge.

4. **Union-find algorithm source:** The union-find characterization of `PlanDevices` comes from community discussions, not official documentation. The actual source (`src/relay/transforms/device_planner.cc`) should be checked if a precise algorithmic claim is made in the poster.

---

## Cross-References to Other Waves

- **Wave 03 (TVM Unified Runtime):** System-level dispatch architecture — what TVM does and doesn't do. This wave provides the internals of *how* TVM attempts heterogeneous dispatch and where it breaks algorithmically.
- **Wave 01 (IREE HAL):** IREE's HAL device abstraction is the direct competitor to TVM's `DeviceAPI` + `VDevice`. HAL does runtime device selection; TVM does compile-time selection. Compare S2 (`DeviceAPI::GetAttr`) with IREE HAL's `iree_hal_device_query_i64`.
- **Wave 02 (Fat Binaries):** TVM produces separate per-target `.so` artifacts. The argument that this is insufficient for deployment (no single artifact, no hardware discovery) is supported by S1, S6, and S8 in this wave.
- **Wave 04 (ONNX Runtime EP):** ORT's execution provider priority list is a runtime device selection mechanism — the analog of what TVM explicitly declined to implement in the Relax RFC. This cross-reference strengthens the argument that runtime dispatch is a proven, deployable pattern that TVM simply chose not to adopt.
- **Wave 05 (ld.so analogy):** The libkdl/ld.so framing is directly supported by the observation that TVM's `PlanDevices` is a compile-time linker pass for device assignment — libkdl moves the equivalent decision to load time, exactly as dynamic linking moved symbol resolution from link time to load time.

---

## Sources (URLs for citation)

- [S1] https://github.com/apache/tvm/issues/2296
- [S2] https://tvm.apache.org/docs/arch/device_target_interactions.html
- [S3] https://github.com/apache/tvm-rfcs/blob/main/rfcs/0045-first-class-virtual-device.md
- [S4] https://discuss.tvm.apache.org/t/how-does-the-plandevices-pass-work/13184
- [S5] https://github.com/apache/tvm/issues/15019
- [S6] https://discuss.tvm.apache.org/t/how-to-do-heterogeneous-execution-on-cpu-and-gpu/11561
- [S7] https://discuss.tvm.apache.org/t/rfc-unity-relax-heterogeneous-execution-for-relax/14670
- [S8] https://github.com/apache/tvm/issues/15101
- [S9] https://tvm.apache.org/docs/reference/api/python/relax/op.html
- [S10] https://arxiv.org/html/2410.08855v1
- [S11] https://github.com/Arm-China/Compass_Apache_TVM
- [S12] https://github.com/apache/tvm/issues/4178
