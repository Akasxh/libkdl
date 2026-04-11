# Topic 03: Cost Model Annotations Surviving Through MLIR Lowering

**Topic ID:** 03
**Config key:** `gpu-dispatch-cost-attr`
**Persona:** MLIR optimization researcher
**Date:** 2026-04-07
**Research depth:** Exhaustive — 7 specific questions investigated, 8 wave files + 3 literature files + kdl.c primary source cross-referenced

---

## Gap

Static analysis results — FLOPs, memory bytes, arithmetic intensity — are computed during MLIR
compilation but are discarded before the runtime dispatcher sees them. The runtime must either
re-derive them (expensive, sometimes impossible) or receive them from the user (error-prone).

The specific failure chain is:

1. MLIR computes or could compute cost information during optimization passes (affine analysis,
   linalg tiling decisions, the KernelInfo pass at LTO time).
2. The `gpu-module-to-binary` pass serializes the GPU module to a `gpu.binary`, discarding all
   module-level metadata not preserved by an explicit convention.
3. The `#gpu.select_object` attribute picks one binary object at MLIR lowering time — always
   compile-time, never runtime — and the surviving binary blob contains zero cost metadata.
4. At runtime, the dispatcher (libkdl, liboffload, or any policy layer) has no access to the
   compile-time analysis results. It operates either on manually-specified contract JSON (libkdl's
   current approach, kdl.c:962–969) or on raw capability matching with no performance prediction.

This is not a hypothetical gap. The LLVM KernelInfo pass (PR #102944, Joel Denny, ORNL, merged
January 2025) emits `AllocasStaticSizeSum`, `FlatAddrspaceAccesses`, `omp_target_thread_limit`,
and launch-bound hints as optimization remarks — but explicitly does NOT emit `flops`,
`bytes_total`, or `arithmetic_intensity`. Its stated design goal is "bad code pattern detection,"
not roofline input generation. The remarks are emitted to stderr/YAML at compile time and then
lost. No downstream consumer ingests them into a binary format readable by a runtime dispatcher.

The result: the three numbers that tritonBLAS (arXiv:2512.04226) proves are sufficient for 94.7%
dispatch quality — `flops`, `bytes_total`, `arithmetic_intensity` — have no standard propagation
path from MLIR analysis to runtime consumption.

---

## Proposal

**Title:** `kdl.cost` — A Three-Layer Protocol for Surviving Cost Annotations in GPU Kernel Binaries

**One-sentence pitch:** Attach FLOPs/bytes/AI as a `DictionaryAttr` on `gpu.func`, preserve it
through `gpu-module-to-binary` into an `OffloadBinary` property string, and consume it in the
runtime dispatcher — closing the static-analysis-to-dispatch loop without profiling.

### Layer 1 — MLIR Attribute Annotation

Introduce a structured attribute `#kdl.cost` (or generalized `#gpu.cost_hint`) as a
`DictionaryAttr` on `gpu.func` operations:

```mlir
gpu.func @matmul_kernel(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
    kernel
    attributes {
      #kdl.cost<flops = 2097152.0,
                bytes_total = 786432.0,
                arithmetic_intensity = 2.667,
                min_shared_mem_bytes = 16384>
    }
```

The `DictionaryAttr` mechanism is the correct MLIR primitive: it is the standard representation
for structured key-value metadata on operations and survives through any pass that uses
`addPass`-style pass management as long as the attribute key is not explicitly removed.

Preservation rule: any pass that creates a new `gpu.func` or clones a `gpu.module` must forward
the `#kdl.cost` attribute. The `gpu-module-to-binary` pass should be patched to extract this
attribute and embed it in the serialized binary object (see Layer 2).

**Population paths** (three options, composable):

a. **Static FLOP/byte counting pass** — count `arith.mulf`, `arith.addf`, `arith.fma`, and
   `memref.load`/`memref.store` instructions per function in a new MLIR analysis pass
   (`KernelCostAnnotator`). For loop nests with known trip counts (analyzable by affine/SCEV),
   multiply per-iteration counts by trip count. Gate on KernelInfo's `IndirectCalls == 0` and
   `InlineAssemblyCalls == 0` flags as a validity pre-filter (if either is nonzero, static
   counting is unreliable — leave `has_compute_profile = false`).

b. **KernelInfo-bridge tool** (`kdl-contract-gen`) — consume KernelInfo remarks YAML at LTO time,
   extract `AllocasStaticSizeSum` (lower-bound on `min_shared_mem_bytes`) and
   `omp_target_thread_limit`, and write them back as MLIR attributes via a post-LTO annotation
   pass. This is a ~200-LOC Python or C tool wrapping `opt -passes=kernel-info -pass-remarks-output`.

c. **Profile-guided annotation** — use the ORNL GPU PGO infrastructure (PR #93365, PR #94268,
   McDonough/Denny/Doerfert, IWOMP 2025) to measure actual FLOPs and bytes on a reference input,
   then encode the measured values as `#kdl.cost` attributes in a post-profiling compilation step,
   analogous to LLVM's `!prof` mechanism for branch weights.

### Layer 2 — OffloadBinary Property Embedding

The `OffloadBinary` class (`llvm/include/llvm/Object/OffloadBinary.h`) stores binary metadata as
a `StringMap<StringRef>` of named properties. Existing properties include `"triple"`, `"arch"`,
`"producer"`, `"kind"`. The class is written specifically to support extension: any `StringMap`
key not recognized by the reader is preserved and round-tripped.

Embed the cost annotation as a structured property during binary serialization:

```
"kdl.cost.flops"       = "2097152.0"
"kdl.cost.bytes_total" = "786432.0"
"kdl.cost.ai"          = "2.667"
"kdl.cost.shared_mem"  = "16384"
```

The `gpu-module-to-binary` pass (in `mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp`)
produces `gpu.object` attributes with a `properties` DictionaryAttr field. Patching it to carry
the extracted `#kdl.cost` dictionary into this field costs approximately 20 lines of C++ and zero
new MLIR ops.

At link time, the `clang-linker-wrapper` creates the final `.llvm.offloading` ELF section from
`OffloadBinary` objects. The cost properties survive in the binary's metadata block because the
linker wrapper round-trips unknown string properties unchanged.

This is the exact analog of LLVM `!prof` metadata: just as `!prof` nodes attach branch weights to
IR instructions and survive through most optimization passes to inform PGO-aware lowering, the
`OffloadBinary` property strings attach cost data to GPU binary objects and survive to inform the
runtime dispatcher.

**Critical difference from `!prof`:** `!prof` metadata lives in the IR and is consumed by IR-level
optimization passes. The `OffloadBinary` property embedding proposed here carries annotations
across the IR-to-binary boundary, enabling consumption at runtime. This is a novel contribution
with no upstream precedent.

### Layer 3 — Runtime Dispatcher Consumption

The runtime dispatcher (libkdl, or any system reading `OffloadBinary` objects) reads the
`kdl.cost.*` properties and populates the `kdl_contract` struct fields at `kdl_load_bundle()` time:

```c
// kdl-contract-gen path (from OffloadBinary properties):
contract->flops              = atof(props["kdl.cost.flops"]);
contract->bytes_total        = atof(props["kdl.cost.bytes_total"]);
contract->arithmetic_intensity = atof(props["kdl.cost.ai"]);
contract->min_shared_mem_kb  = atoi(props["kdl.cost.shared_mem"]) / 1024;
contract->has_compute_profile = 1;
```

With `has_compute_profile = 1`, `kdl_estimate_cost_weighted()` (kdl.c:1013–1088) switches from
the fallback path (return 1e9, dispatch by priority) to the full roofline cost model using the
embedded values. The dispatcher then selects the best variant across NVIDIA, AMD, and CPU using:

```c
double T_compute = contract->flops / (device->peak_tflops_f32 * 1e12);
double T_memory  = contract->bytes_total / (device->peak_bw_gbps * 1e9);
double cost      = fmax(T_compute, T_memory) / efficiency + overhead;
```

This is the `max(T_compute, T_memory)` formulation validated by tritonBLAS at 94.7% dispatch
accuracy. The correction from libkdl's current weighted sum (`w.compute * T_compute + w.memory *
T_memory`) to the roofline-correct `fmax` is a separate but complementary improvement.

**What data the dispatcher actually needs** (from literature synthesis and kdl.c analysis):

| Field | Source | Required for |
|-------|--------|-------------|
| `flops` | Static count or PGO | Compute-bound regime detection |
| `bytes_total` | Static count or PGO | Memory-bound regime detection |
| `arithmetic_intensity` | `flops/bytes_total` | Ridge-point comparison |
| `min_shared_mem_bytes` | KernelInfo `AllocasStaticSizeSum` | Hardware capability gate |
| `omp_target_thread_limit` | KernelInfo direct | Block-size compatibility pre-filter |
| `flat_addrspace_risk` | KernelInfo `FlatAddrspaceAccesses > 0` | CPU fallback preference signal |

The first three drive the cost model. The last three are capability gates that filter out
structurally incompatible devices before scoring. KernelInfo provides all three capability fields
today; only `flops` and `bytes_total` require the new static analysis pass.

---

## Evidence

### On MLIR attribute survival through lowering

- **`gpu-module-to-binary` pass** (`mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp`):
  The pass iterates `gpu.module` ops and calls `target.serializeToObject(op, targetOptions)` per
  target attribute. The `gpu.object` attribute produced has a `properties` DictionaryAttr field
  that currently carries nothing from the kernel body. The hook for cost annotation is the
  `properties` field — it is already designed as an extension point.

- **`#gpu.select_object` compile-time limitation** (MLIR GPU dialect docs, wave-01-mlir-gpu-dialect):
  "There is no runtime target selection mechanism in the GPU dialect itself. The `#gpu.select_object`
  attribute picks one object at compile time (by index)." This confirms that runtime cost-driven
  selection requires out-of-band metadata, not an MLIR op.

- **`GPUOffloadingLLVMTranslationAttrInterface`** (literature/mlir-gpu-infrastructure-2026.md):
  An existing interface for custom binary embedding and kernel launch codegen. A `#gpu.cost_hint`
  attribute implementing this interface could route cost metadata into the final LLVM IR as a
  `!gpu.cost` metadata node, analogous to `!prof`.

### On KernelInfo pass (PR #102944)

- **Merged:** commit `18f8106f`, January 29, 2025. Author: Joel E. Denny (ORNL).
  Source: `llvm/lib/Analysis/KernelInfo.cpp` (326 lines).

- **Complete metric list** (wave-08-kernel-info-pass.md):
  `AllocasStaticSizeSum`, `FlatAddrspaceAccesses`, `IndirectCalls`, `InlineAssemblyCalls`,
  `omp_target_thread_limit`, `maxntidx/y/z`, `amdgpu-waves-per-eu`, `amdgpu-flat-work-group-size`.
  **Not present:** `flops`, `bytes_total`, `arithmetic_intensity`.

- **Design intent** (PR #102944 description, Denny): "The ultimate goal of these statistics is to
  help identify bad code patterns and ways to mitigate them." Explicitly not a roofline input
  generator.

- **libkdl alignment** (wave-08-kernel-info-pass.md, lines 122–138): `AllocasStaticSizeSum` maps
  to a lower-bound on `min_shared_mem_kb`; `omp_target_thread_limit` directly feeds block-size
  compatibility; `FlatAddrspaceAccesses > 0` is a CPU-fallback preference signal absent from the
  current `kdl_contract` schema. All three are available today with no new LLVM pass.

### On tritonBLAS roofline model

- **Citation:** arXiv:2512.04226, December 2025. tritonBLAS: analytical roofline for Triton GEMM
  tile selection.

- **Result:** `argmin_config max(T_compute, T_memory)` achieves **94.7%** of exhaustive autotuning
  quality on 150,000 GEMM shapes. AMD ROCm (MI250X): 91%. NVIDIA (A100, H100): 94.7%.

- **Direct libkdl relevance** (wave-03-cost-model-selection, source S3): tritonBLAS proves that
  the `fmax` formula — not a weighted sum — is the correct basis for GEMM selection. libkdl's
  current cost model (kdl.c:1061–1064) uses `w.compute * compute_time + w.memory * memory_time`;
  replacing this with `fmax(compute_time, memory_time)` aligns with tritonBLAS's 94.7% ceiling.

- **Selection time:** 50–80 microseconds (Triton JIT context). libkdl operates at <10 ns (kdl.c
  `bench_dispatch` output) — 3–4 orders of magnitude faster because it operates on pre-compiled
  variants rather than JIT-time parameterization.

### On NeuSight (ASPLOS 2025)

- **Citation:** "Forecasting GPU Performance for Deep Learning Training and Inference."
  ASPLOS 2025. ACM DL: https://dl.acm.org/doi/10.1145/3669940.3707265. GitHub: sitar-lab/NeuSight.

- **Method:** Tile-granularity decomposition + 5 specialized MLPs, each bounded by per-GPU
  roofline ceiling. Hardware features: `peak_flops`, `peak_bw`, L1/L2 cache sizes.

- **Result:** 2.3% mean error on H100 (not in training set). 8.9% mean error across all
  evaluated GPUs vs 60.8% (linear regression).

- **Relevance to cost annotation propagation:** NeuSight's hardware feature vector is exactly
  the `kdl_device_info` fields. NeuSight demonstrates that tile-decomposed per-GPU roofline
  bounding is sufficient for cross-architecture transfer — supporting the claim that the three
  numbers (`flops`, `bytes_total`, `peak_flops`, `peak_bw`) embedded in the binary are enough for
  near-optimal dispatch without profiling.

- **Gap for AMD transfer:** NeuSight is trained on NVIDIA-only hardware; AMD transfer accuracy
  is not validated. For AMD, the analytical roofline (libkdl's approach) is the only
  published cross-vendor predictor.

### On LLVM `!prof` metadata as architectural precedent

- **Mechanism:** `!prof` is a named metadata node (`MDNode` with string "branch_weights" or
  "function_entry_count") attached to IR instructions and function definitions. It is generated
  during PGO instrumentation compilation and consumed by optimization passes (branch prediction,
  inlining heuristics, loop unrolling decisions).

- **Survival path:** `!prof` nodes are preserved through all standard optimization passes (O1/O2/O3
  pipelines) as long as passes that remove basic blocks also remove associated metadata. The LLVM
  pass manager has explicit logic to propagate metadata through dominator-tree-altering transforms.

- **GPU cost analog:** A `!gpu.cost` metadata node on `define ... @kernel_name()` in the device IR
  would follow the same propagation logic through LTO, surviving into the final device binary's
  debug/annotation section. This is technically sound; no upstream infrastructure exists for it.

- **OffloadBinary as the binary-boundary crossing:** Where `!prof` stays in IR (consumed before
  binary emission), `!gpu.cost` must cross the IR-to-binary boundary. The `OffloadBinary`
  `StringMap<StringRef>` property bag is the correct structure: it is explicitly designed for
  forward-compatible property extension and is preserved through the `clang-linker-wrapper`
  fat-object pipeline.

### On OffloadBinary StringMap extension

- **Source:** `llvm/include/llvm/Object/OffloadBinary.h`, `llvm/lib/Object/OffloadBinary.cpp`.
  The `OffloadBinary::create()` / `OffloadBinary::write()` API uses a `StringMap<StringRef>` for
  properties. The reader (`OffloadBinary::readBinary()`) round-trips unknown keys unchanged.

- **Existing property keys** (`offload/include/llvm/Offload/OffloadBinary.h`):
  `OFFLOAD_PROPERTY_TARGET` (`"triple"`), `OFFLOAD_PROPERTY_ARCH` (`"arch"`),
  `OFFLOAD_PROPERTY_PRODUCER` (`"producer"`).

- **Proposed addition:** `kdl.cost.flops`, `kdl.cost.bytes_total`, `kdl.cost.ai`,
  `kdl.cost.shared_mem`, `kdl.cost.thread_limit`, `kdl.cost.flat_addrspace_risk`.
  These are plain ASCII double strings, zero parsing overhead, zero schema validation required.

- **Embedding path:** The `gpu-module-to-binary` pass serializes to `gpu.object`; the
  `GPUTargetAttrInterface::serializeToObject()` method returns a byte array + properties dict.
  Patching NVVM and ROCDL target serializers to include cost properties from the GPU function's
  attribute dict requires approximately 30 lines of C++ per target (extract `#kdl.cost` dict,
  serialize each entry to the properties map).

### On runtime dispatcher data requirements

- **kdl_contract struct** (kdl.c:112–122): `flops`, `bytes_total`, `arithmetic_intensity`,
  `min_shared_mem_kb`, `has_compute_profile`. The `has_compute_profile` gate at kdl.c:1016
  (`if (!c->has_compute_profile) return 1e9`) is the exact injection point where embedded cost
  data changes dispatch behavior from priority-based to roofline-based.

- **Cost model** (kdl.c:1013–1088): Uses `peak_tflops_f32` and `peak_bw_gbps` from
  `kdl_device_info`, which are measured at first load via `kdl_calibrate()`. The per-kernel
  cost annotation is the missing input; device parameters are already correctly collected.

- **IREE has no runtime cost model** (iree-kernel-benchmark, wave-03-cost-model-selection):
  IREE's HAL selects among pre-compiled targets by capability string only, no roofline or ML
  cost estimation. ORT uses static EP priority order (wave-04-cost-models, S12 equivalent).
  Neither system uses compile-time cost annotations at runtime.

---

## Feasibility

**High feasibility. Three independent implementation paths, each de-risked.**

### Path A: KernelInfo bridge (zero new LLVM passes, works today)

1. Compile with `clang -foffload-lto -Rpass=kernel-info -pass-remarks-output=remarks.yml`
2. Run `kdl-contract-gen remarks.yml` — a ~200-LOC Python script parsing the remarks YAML,
   extracting `AllocasStaticSizeSum`, `omp_target_thread_limit`, `FlatAddrspaceAccesses` per
   kernel name
3. Emit these fields into the MTB contract JSON for each kernel variant
4. Result: `min_shared_mem_kb`, `flat_addrspace_risk`, `omp_target_thread_limit` fields
   populated; `flops`/`bytes_total` still require user input or Path B

This path demonstrates the end-to-end infrastructure with zero MLIR/LLVM modifications.
Deliverable for poster: `kdl-contract-gen` tool + a table showing which KernelInfo fields
map to which `kdl_contract` fields.

### Path B: Static FLOP/byte counting MLIR pass

New analysis pass `KernelCostAnnotator` running after bufferization and before `gpu-module-to-binary`:

- Walk `gpu.func` bodies
- Count FP operation instructions (`arith.mulf`, `arith.addf`, `arith.fmaf`, etc.) and
  memory operations (`memref.load`, `memref.store`) with byte widths
- For affine loop nests with constant trip counts, multiply per-iteration counts by trip count
  (affine trip count analysis is already in upstream MLIR)
- Emit `#kdl.cost` DictionaryAttr on the function
- Gate on KernelInfo's `IndirectCalls == 0` for validity

**Scope:** ~500 LOC new MLIR pass. Patch to `gpu-module-to-binary` to preserve the attribute:
~30 LOC. Patch to NVVM/ROCDL serializers to embed as OffloadBinary properties: ~30 LOC each.

**Total implementation:** ~600 LOC of new code + ~90 LOC of patches to existing files.
This is within the scope of a single GSoC or LFX Mentorship project.

**Accuracy:** For compute-bound kernels with known trip counts (GEMM, Conv2D, LSTM cells),
static FLOP counting is exact. For memory-bound kernels with input-variable access patterns
(sparse, attention with variable seq_len), accuracy degrades — the TACO 2021 baseline
(arXiv:2001.07104) shows 52% MAPE for memory-bound kernels with static analysis alone. This is
acceptable for the dispatch problem (we only need to identify the right regime, not predict
exact latency).

### Path C: PGO-guided annotation (production quality, ORNL infrastructure)

Use ORNL's GPU PGO infrastructure (PR #93365, PR #94268) to instrument a kernel for one
reference input, measure actual FLOPs and bytes executed, then write the measured values
back as `#kdl.cost` attributes in a post-profiling compilation step. This produces
ground-truth values rather than static estimates.

The `!prof` precedent in CPU PGO is the exact architectural template: profile once, annotate
once, use forever. The `has_compute_profile` flag in `kdl_contract` acts as the "PGO data
present" marker, mirroring LLVM's PGO "profdata present" check.

---

## Upstream Path

**Conservative estimate: 18–24 months to full upstream acceptance. Staged milestones.**

### Stage 1 — Non-invasive tooling (0–6 months, no upstream changes needed)

- Publish `kdl-contract-gen` as a standalone tool consuming KernelInfo remarks
- Demonstrate end-to-end: KernelInfo → contract JSON → libkdl dispatch decision change
- Present at LLVM Dublin 2026 poster session as proof-of-concept
- Target audience: GPU offloading community, ORNL (KernelInfo authors), nod-ai/AMD

### Stage 2 — MLIR attribute proposal (6–12 months)

- File RFC on LLVM Discourse: "`#gpu.cost_hint` attribute for kernel cost annotations"
- Anchor to existing discussions: GPU dialect cleanup RFC (#88170, September 2025),
  MLIR cost model RFC (https://discourse.llvm.org/t/rfc-target-description-and-cost-model-in-mlir/76990)
- The cost model RFC (Intel PCL, 2024) concluded that "MLIR lacks a standard cross-target cost
  model interface" — the `#gpu.cost_hint` proposal fills part of this gap at the kernel level
- Implement `KernelCostAnnotator` pass as a child project of the RFC

### Stage 3 — OffloadBinary property standardization (12–18 months)

- File PR to add named cost properties to OffloadBinary: `OFFLOAD_PROPERTY_COST_FLOPS`, etc.
- Anchor to liboffload's `ol*` C API (PR #118614, PR #122106) — the API could expose cost
  properties via `olGetKernelInfoSize(kernel, OL_KERNEL_INFO_FLOPS, ...)` as a new `KernelInfo`
  enum value in the unified `info` pattern already used for device info
- Joseph Huber (AMD, liboffload primary author) has explicitly framed liboffload as "ld.so for
  GPU code" (LLVM DevMtg 2025 talk) — cost-aware dispatch is a natural extension of this vision

### Stage 4 — KernelInfo extension (18–24 months)

- Extend `llvm/lib/Analysis/KernelInfo.cpp` with static FLOP/byte counting
- Requires careful scoping: the pass is intentionally narrow ("bad patterns only"); extension
  needs a new remark category that does not bloat existing outputs
- Alternative: contribute as a separate companion pass (`KernelCostInfo` vs `KernelInfo`)
  registered separately, avoiding modification of existing pass behavior

**Key upstream stakeholders:**
- Joel E. Denny, ORNL (KernelInfo author, GPU PGO work) — natural collaborator
- Joseph Huber, AMD (liboffload, `llvm.gpu` intrinsics) — framing alignment
- Fabian Mora (GPU dialect cleanup RFC) — timing alignment for dialect restructuring
- Intel PCL team (cost model RFC authors) — shared motivation

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **9/10** | No system carries compile-time cost annotations from MLIR analysis into GPU binary format for runtime dispatch. `!prof` analogy is novel as a cross-boundary metadata propagation mechanism. KernelInfo is upstream but produces remarks, not binary-embedded metadata. |
| **Feasibility** | **8/10** | Three implementation paths at different risk levels. Path A (KernelInfo bridge) works today with no MLIR changes. Path B (static pass) is ~600 LOC. The `OffloadBinary` StringMap extension has a clear implementation path. Prototype on GTX 1650 + CPU is achievable before poster deadline. |
| **Evidence Strength** | **9/10** | tritonBLAS (94.7% with 3 numbers), NeuSight (2.3% error), KernelInfo (PR #102944, upstream), OffloadBinary StringMap (source-verified), `!prof` precedent (LLVM PGO docs), kdl.c cost model (live prototype). All evidence from primary sources. |
| **Impact** | **8/10** | Solves the "annotation gap" that prevents any static-analysis-informed runtime dispatch. Applicable to libkdl, liboffload policy layers, IREE HAL dispatch, any multi-variant GPU kernel system. Upstream path to LLVM/MLIR ecosystem is viable. |
| **Composite** | **8.50/10** | |

---

## Pitch

**The three-sentence poster pitch:**

GPU compilers compute FLOPs and memory bytes during optimization but discard them before the
binary is written. Runtime dispatchers — choosing between NVIDIA, AMD, and CPU kernel variants
— need exactly these three numbers to apply a roofline cost model that achieves 94.7% of
optimal (tritonBLAS, arXiv:2512.04226). We close this gap with a three-layer protocol: static
analysis annotates `gpu.func` with a `#kdl.cost` DictionaryAttr, `gpu-module-to-binary`
embeds it as an `OffloadBinary` property string (the GPU analog of LLVM's `!prof` metadata),
and libkdl reads it at runtime — enabling zero-profiling cost-model dispatch across vendors.

**Poster panel structure:**

1. The gap diagram: MLIR analysis → (annotation discarded) → binary blob → runtime guessing
2. The three-layer protocol: attribute → binary property → runtime consumption
3. KernelInfo alignment table: which fields are available today vs. what needs a new pass
4. tritonBLAS 94.7% result cited as the quality ceiling achievable with the embedded data
5. Dispatch accuracy comparison: contract-annotated vs. priority-only vs. exhaustive profiling
6. Future upstream path: `#gpu.cost_hint` RFC, `olGetKernelInfo` extension, `KernelCostInfo` pass

---

## Risks

1. **`DictionaryAttr` survival is pass-dependent.** If any intermediate pass creates a new
   `gpu.func` without forwarding attributes, the annotation is silently lost. MLIR has no
   "mandatory attribute preservation" mechanism — the implementation must audit all passes in
   the GPU lowering pipeline. This is a correctness risk that must be addressed with tests.

2. **Static FLOP counting is wrong for memory-bound kernels with variable access patterns.**
   The TACO 2021 baseline (arXiv:2001.07104) shows up to 52% MAPE for memory-bound kernels.
   The dispatch problem only requires regime identification (compute-bound vs. memory-bound),
   not exact latency, so 52% error in time prediction does not necessarily translate to 52%
   error in dispatch decisions. But a clear accuracy qualification is needed for the poster.

3. **KernelInfo is intentionally narrow.** Adding FLOP counting to KernelInfo risks scope creep
   that may be rejected upstream. A separate companion pass (`KernelCostInfo`) is the safer
   proposal, at the cost of requiring users to run two passes.

4. **OffloadBinary property round-trip is untested for non-string data.** The `StringMap<StringRef>`
   stores double values as ASCII strings. Floating-point serialization precision (how many digits
   of `flops` survive `sprintf`/`strtod`) must be validated. `%.17g` format preserves full
   double precision; the implementation must enforce this.

5. **NeuSight AMD transfer gap.** NeuSight is trained on NVIDIA GPUs only; the claim that
   tile-decomposed roofline transfers to AMD must be qualified. The analytical roofline (no ML)
   is vendor-agnostic by construction and is the safer claim for the poster.

6. **Upstream priority competition.** The GPU dialect cleanup RFC (#88170) is actively ongoing
   as of April 2026. If it restructures `gpu.func` semantics significantly, the `#kdl.cost`
   attribute definition may need revision. Timing the RFC to align with the cleanup resolution
   is important for community reception.

---

## Cross-References

- `kdl.c:112–122` — `kdl_contract` struct (fields this proposal populates)
- `kdl.c:1013–1088` — `kdl_estimate_cost_weighted()` (the runtime consumer)
- `kdl.c:1016` — `has_compute_profile` gate (the exact injection point)
- `wave-08-kernel-info-pass.md` — complete KernelInfo field analysis and libkdl alignment table
- `wave-03-cost-model-selection.md` — tritonBLAS 94.7% result, NeuSight ASPLOS 2025
- `wave-04-cost-models.md` — cuBLAS/CUTLASS production precedents, dispatch overhead table
- `wave-01-mlir-gpu-dialect-dispatch.md` — `#gpu.select_object` compile-time limitation confirmed
- `wave-08-mlir-async-llvm-gpu.md` — `gpu.binary` architecture, OffloadBinary embedding path
- `wave-02-llvm-offload-runtime.md` — OffloadBinary format, liboffload `ol*` API, `GenericKernelTy`
- `wave-06-kernel-binary-abi.md` — binary cache key requirements; sm_90a PTX forward-compat break
- `literature/cost-models-kernel-dispatch.md` — roofline theory, dispatch overhead table (<10 ns)
- `literature/mlir-gpu-infrastructure-2026.md` — `gpu-module-to-binary` pass internals
- `research/mega-survey/.../directions/03-roofline-cross-vendor-cost-model.md` — sister direction
- `research/mega-survey/.../waves/wave-03-cost-model-selection.md` — NeuSight, tritonBLAS, SparseX
