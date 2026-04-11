# Direction 02: MLIR-Native Runtime Dispatch Op

**Final Score: 8.50/10** (Rank #2 of 6)
**Scoring History:** Round 1: 8.25 → Round 2: 8.50 → Final: 8.50

---

## One-Sentence Description

Propose a new MLIR operation (`gpu.dispatch_select`) that replaces the compile-time `gpu.select_object` with runtime hardware query + variant selection, making MLIR the first compiler IR with native runtime multi-target dispatch semantics.

---

## Score Breakdown

| Criterion | Score | Justification |
|-----------|------:|---------------|
| Novelty | 10/10 | No MLIR dialect contains a runtime variant selection operation; all current paths are compile-time |
| Feasibility | 5/10 | Requires MLIR tablegen, new op semantics, LLVM translation integration; too heavy for poster deadline |
| Evidence Strength | 9/10 | GPU dialect cleanup RFC (Sep 2025) separates containers from policy; Distributed Computing RFC (Jun 2025) proposes compile-time analog with no runtime equivalent |
| Impact | 10/10 | Upstream MLIR contribution; highest long-term influence if accepted |

---

## Evidence Summary

### The Gap in MLIR

| Evidence | What It Shows | Source |
|----------|--------------|--------|
| `gpu.select_object` semantics | Compile-time first-object-wins selection; no runtime query | wave-01-mlir-gpu S1 |
| `gpu-module-to-binary` pass (D154149) | Produces one `gpu.object` per target inside `gpu.binary`; multi-target packaging exists | wave-01-mlir-gpu S2 |
| GPU Target Attribute Interface (D154104) | Extensible: any target attribute registers `serializeToObject()` | wave-01-mlir-gpu S2 |
| GPU dialect cleanup RFC (Sep 2025) | Separates binary containers (`gpu.binary`) from dispatch policy — the exact boundary for a new op | wave-05-discourse S11 |
| Distributed Heterogeneous Computing RFC (Jun 2025) | Proposes `schedule` + `task` + `target` for compile-time placement; no runtime equivalent | wave-05-discourse S10 |
| MLIR cost model RFC | Confirmed as open gap; no existing MLIR pass performs cost-model-driven selection | wave-04-cost-models S8 |
| IREE HAL variant conditions | Closest analog: boolean condition evaluated at load-time, not per-dispatch | wave-03-multi-versioned-kernels S7 |

### What Currently Happens

The MLIR GPU compilation pipeline is statically target-determined:

```
gpu.module @kernels {
  gpu.func @matmul(...) { ... }
}
        ↓ gpu-module-to-binary (compile-time target selection)
gpu.binary @kernels [
  #gpu.object<#nvvm.target, "...cubin...">,
  #gpu.object<#rocdl.target, "...hsaco...">
]
        ↓ gpu.select_object (compile-time first-match)
Selected: one gpu.object (e.g., cubin if NVVM target was first)
```

The `gpu.select_object` operation has no runtime device query. The selected object is determined by the order of targets in the binary, fixed at compile time.

### What Should Happen

```
gpu.binary @kernels [
  #gpu.object<#nvvm.target<sm_80>, "...cubin...">,
  #gpu.object<#nvvm.target<sm_90>, "...cubin...">,
  #gpu.object<#rocdl.target<gfx942>, "...hsaco...">,
  #gpu.object<#spirv.target, "...spv...">
]
        ↓ gpu.dispatch_select (runtime device query + cost model)
Selected: best gpu.object for detected hardware
```

---

## Novelty Argument

### What exists in MLIR

- `gpu.binary`: container for multi-target compiled objects (since D154149)
- `gpu.select_object`: compile-time static selection (no runtime query)
- `gpu.launch_func`: dispatch a kernel (takes a single binary, no variant choice)
- IREE HAL: variant conditions evaluated at module load, not per-dispatch

### What does NOT exist

- **No MLIR op queries runtime hardware.** All device-specific decisions are made at compile time or by the host runtime (outside MLIR's representation).
- **No MLIR op selects among compiled variants.** `gpu.select_object` is not a selection mechanism — it is a static extraction that always returns the same object.
- **No cost model in MLIR for dispatch.** The MLIR cost model RFC (wave-04-cost-models S8) proposes analytical cost estimation for transformations, not for runtime dispatch decisions.

### Why this matters

The GPU dialect cleanup RFC (Sep 2025) is clarifying what `gpu.binary` should and should not do: it should represent a compiled binary artifact, not a dispatch policy. This creates a clean separation:
- `gpu.binary` = kernel container (already exists)
- `gpu.dispatch_select` = runtime selection policy (**new**)

This separation mirrors the mechanism/policy split in liboffload (mechanism) vs. libkdl (policy), but at the IR level rather than the runtime level.

---

## Proposed Design

### Op Semantics

```mlir
// New operation: gpu.dispatch_select
%kernel = gpu.dispatch_select @kernels::@matmul
    on %device : !gpu.dispatch_handle
    {cost_model = "roofline", fallback = "first_compatible"}
```

**Semantics:**
1. At runtime, query `%device` for architecture capabilities
2. Iterate over `gpu.object` entries in the binary
3. For each entry, check capability contract compatibility
4. Score compatible entries using the specified cost model
5. Return a handle to the highest-scoring compatible entry
6. Cache the result for subsequent calls with the same (kernel, device) pair

### Lowering

The op lowers to a runtime library call:

```c
// Generated lowering target
void* __mlir_gpu_dispatch_select(
    void* binary_data,      // gpu.binary serialized data
    int device_index,       // runtime device ID
    const char* kernel_name // symbol name within binary
);
```

The runtime library implements the same resolver algorithm as libkdl:
1. Bloom filter elimination (<100 ns)
2. Roofline analytical scoring (100 ns - 10 us)
3. Cache result in dispatch table (O(1) subsequent lookups)

### Integration with Existing Passes

- `gpu-module-to-binary` remains unchanged (produces `gpu.binary` with all targets)
- `gpu.dispatch_select` replaces `gpu.select_object` in the lowering pipeline
- `gpu.launch_func` accepts the `!gpu.dispatch_handle` returned by `gpu.dispatch_select`

---

## Feasibility Assessment

### Why score is 5/10

| Challenge | Effort | Notes |
|-----------|--------|-------|
| MLIR TableGen for new op | 2-3 days | Straightforward but requires MLIR build system familiarity |
| Runtime library implementation | 1-2 weeks | Same algorithm as libkdl; needs to be packaged as MLIR-consumable runtime |
| Integration with LLVM translation | 1 week | Lower `gpu.dispatch_select` to LLVM IR calling convention |
| Testing across CUDA + HIP + SPIR-V | 2 weeks | Multi-target testing infrastructure needed |
| Upstream review process | 2-6 months | MLIR RFC + review cycle; not achievable before poster deadline |

**Total estimated effort:** 4-6 weeks of focused development + 2-6 months of upstream review.

### Why this is a future-work item for the poster

- The poster deadline is 2026-04-07 — insufficient time for MLIR upstream review
- The prototype (libkdl as a standalone library) demonstrates the algorithm without requiring MLIR changes
- Presenting the MLIR op as future work signals upstream ambition without overcommitting

---

## Poster Role

### How this appears on the poster

**Section 8: Future Work** (150 words):

> "libkdl's resolver algorithm is designed to be liftable into MLIR as a native `gpu.dispatch_select` operation. The MLIR GPU dialect's `gpu.binary` already stores multi-target compiled objects (since D154149); `gpu.select_object` performs compile-time extraction. A new `gpu.dispatch_select` op would replace this static selection with runtime hardware query + cost-model ranking, making MLIR the first compiler IR with native runtime multi-target dispatch semantics. This would enable MLIR-compiled programs to carry vendor-native kernels for all targets in a single module and select the optimal variant at deployment time — the same capability libkdl provides at the binary level, elevated to the IR level. We plan to propose this as an MLIR RFC following the poster presentation."

### Why this strengthens the poster

1. Shows the contribution has legs beyond a standalone library
2. Signals to MLIR developers that libkdl is designed for upstream integration
3. Creates a natural discussion point for poster session conversations
4. The GPU dialect cleanup RFC (Sep 2025) provides the architectural opening

---

## Relationship to Direction A

Direction A (libkdl as policy layer) and Direction D (MLIR op) are complementary, not competing:

| Aspect | Direction A (libkdl) | Direction D (MLIR op) |
|--------|---------------------|----------------------|
| Layer | Runtime binary dispatch | Compiler IR |
| Input | Pre-compiled MTB bundles | MLIR `gpu.binary` modules |
| User | Application developer | MLIR toolchain user |
| Timeline | Now (prototype exists) | Future (requires MLIR RFC) |
| Dependency | None (standalone) | Depends on Direction A's algorithm |

The poster presents Direction A as the contribution and Direction D as the roadmap. The algorithm is the same; the integration point differs.

---

## Key References

1. MLIR `gpu-module-to-binary` (D154149) — multi-target binary infrastructure
2. MLIR GPU Target Attribute Interface (D154104) — extensible target system
3. GPU dialect cleanup RFC (Sep 2025, wave-05-discourse S11) — binary vs. policy separation
4. Distributed Heterogeneous Computing RFC (Jun 2025, wave-05-discourse S10) — compile-time analog
5. IREE HAL variant conditions (wave-03-multi-versioned-kernels S7) — closest existing analog
6. MLIR cost model RFC (wave-04-cost-models S8) — cost model gap
7. `gpu.select_object` semantics (wave-01-mlir-gpu S1) — current static selection
