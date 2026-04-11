# Topic 04: Impact of GPU Dialect Cleanup RFC on Multi-Target Compilation

**Topic ID:** 04 — `mlir-gpu-dialect-cleanup-impact`
**Persona:** MLIR ecosystem analyst
**Date:** 2026-04-07
**Status:** Research complete, proposal drafted

---

## Research Log

### Question

The September 2025 "RFC: Cleaning the GPU Dialect" proposes significant structural changes to the
MLIR `gpu` dialect. What opportunities does this create for better multi-target support, given
that XeVM was just upstreamed (August 2025) making MLIR a genuine tri-vendor GPU platform?

### Primary Sources Consulted

| Source | URL | Type | Date |
|--------|-----|------|------|
| RFC: Cleaning the GPU Dialect | discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170 | RFC | Sep 2025 |
| Intel XeVM Upstreaming (PR #148286) | phoronix.com/news/Intel-XeVM-MLIR-In-LLVM | News/PR | Aug 19 2025 |
| RFC: Distributed Heterogeneous Computing | discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960 | RFC | Jun 2025 |
| D154104: GPUTargetAttrInterface | reviews.llvm.org/D154104 | PR | 2023 |
| D154149: gpu-module-to-binary pass | reviews.llvm.org/D154149 | PR | 2023 |
| PR #119440: ELF section in gpu-module-to-binary | github.com/llvm/llvm-project/pull/119440 | PR | Dec 2024 |
| RFC: SPIR-V IR as vendor-agnostic GPU repr | discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115 | RFC | Mar 2025 |
| GPU Dialect official docs | mlir.llvm.org/docs/Dialects/GPU/ | Docs | current |
| MLIR GPU Infrastructure deep analysis | literature/mlir-gpu-infrastructure-2026.md | Local | 2026-04-06 |
| Wave 01: MLIR GPU Dialect Dispatch | mega-survey/.../wave-01-mlir-gpu-dialect-dispatch.md | Local | 2026-04-06 |
| Wave 05: LLVM Discourse RFCs | mega-survey/.../wave-05-llvm-discourse-rfcs.md | Local | 2026-04-06 |
| Wave 08: MLIR Async + llvm.gpu intrinsics | mega-survey/.../wave-08-mlir-async-llvm-gpu.md | Local | 2026-04-06 |
| EuroLLVM 2026 CFP | llvm.org/devmtg/2026-04/ | Docs | current |
| LLVM DevMtg 2025 Schedule | llvm.org/devmtg/2025-10/ | Docs | Oct 2025 |

---

## Background: The Current State of MLIR Multi-Target GPU Compilation

### Three-Vendor Reality (as of August 2025)

MLIR now has three production GPU target attribute implementations following Intel's XeVM
upstreaming on August 19, 2025 (PR #148286):

| Target Attribute | Vendor | Binary Format | Chip Default |
|-----------------|--------|---------------|-------------|
| `#nvvm.target` | NVIDIA | PTX / cubin / fatbin | `sm_75` |
| `#rocdl.target` | AMD | HSACO (ELF) | `gfx900` |
| `#xevm.target` | Intel | SPIR-V binary (via ocloc) | `bmg` |

A single `gpu.module` can carry all three target attributes. The `gpu-module-to-binary` pass
calls `serializeToObject()` on each via the `GPUTargetAttrInterface` and produces:

```mlir
gpu.binary @matmul [
  #gpu.object<#nvvm.target<chip = "sm_90">,  bin = "...cubin-sm90...">,
  #gpu.object<#rocdl.target<chip = "gfx942">, bin = "...hsaco-gfx942...">,
  #gpu.object<#xevm.target<chip = "bmg">,    bin = "...spirv-bmg...">
]
```

The infrastructure to *package* three-vendor kernel binaries in one MLIR op already exists and
works. The infrastructure to *select* among them at runtime does not.

### The Selection Gap: `#gpu.select_object` is Compile-Time Only

The only built-in offloading handler in upstream MLIR is `SelectObjectAttr`, which resolves
to a single object at LLVM-IR translation time:

```
Selection logic in SelectObjectAttr.cpp:
1. If handler has integer index  → use as array index (compile-time constant)
2. If handler has target attr    → match against objects array (compile-time match)
3. Default                       → index 0 (always first object)
```

The selected object becomes a `@serializedObj` global byte array in LLVM IR, loaded by a
constructor function that calls `mgpuModuleLoad()` at program startup. There is no runtime
device query, no hardware detection, no fallback chain.

**The structural gap:** A program compiled against a tri-vendor `gpu.binary` can only ever run
one kernel variant — the one chosen at MLIR→LLVM translation time. To deploy the same binary
across a NVIDIA datacenter and an AMD workstation, you must produce two separate host binaries,
each with a different compile-time `#gpu.select_object` choice.

### The RFC: Cleaning the GPU Dialect (September 2025)

Author: Fabian Mora (fabianmcg) — the same engineer who designed `gpu-module-to-binary` and
`GPUTargetAttrInterface`.

**Core diagnosis:** The GPU dialect has accumulated operations from multiple contributors with
overlapping and incoherent semantics:
- Device-programming model ops (`gpu.launch`, `gpu.func`, thread ID intrinsics) — genuinely
  belong in the GPU dialect as the vendor-neutral programming abstraction
- Binary management ops (`gpu.module`, `gpu.binary`, `#gpu.select_object`) — compilation
  pipeline artifacts, not a programming model
- Host-side orchestration ops (`gpu.host_register`, `gpu.host_unregister`, `gpu.wait`) — really
  runtime/HAL concerns that have leaked into the dialect

**Proposed restructuring direction:**
- Tighten the dialect's semantic contract: it should represent the *GPU programming model*
  (kernel functions, thread hierarchy, synchronization, memory), not compiler pipeline state
- Relocate or refactor binary management ops: `gpu.binary` and its offloading handler
  infrastructure may move to a dedicated binary/compilation dialect or lowering layer
- Clarify the boundary between abstract dispatch (`gpu.launch_func` as a vendor-neutral
  operation specifying what to run) and concrete dispatch (the lowering of that op to
  `mgpuLaunchKernel` or equivalent driver calls)
- Three pages of community discussion indicate active engagement; the RFC is not yet merged
  as a formal design decision as of April 2026

**Why this creates an opportunity:** If `gpu.binary` / `gpu.module` / `#gpu.select_object`
are refactored into a cleaner binary management layer, the `GPUOffloadingLLVMTranslationAttrInterface`
becomes the principled extension point for alternative dispatch strategies. A
`#gpu.runtime_select` attribute implementing this interface could perform runtime device
detection and hardware-capability-based object selection — filling the gap the cleanup RFC
creates architectural space for.

---

## What Multi-Target Opportunities Open Up

### Opportunity 1: `#gpu.runtime_select` Offloading Handler

The `GPUOffloadingLLVMTranslationAttrInterface` has two methods:
- `embedBinary(gpu::BinaryOp, LLVM::GlobalOp)` — generates LLVM IR for storing the binary
- `translateLaunchFuncToLLVMCalls(gpu::LaunchFuncOp, ...)` — generates the launch sequence

A `#gpu.runtime_select` attribute implementing this interface would:
1. In `embedBinary`: embed *all* objects from the `gpu.binary` as separate globals, plus
   embed a runtime dispatch table (target string → binary pointer)
2. In `translateLaunchFuncToLLVMCalls`: emit a runtime call to a query function
   (`kdl_detect_vendor()` or equivalent) that checks available devices and selects the
   matching binary at the first kernel launch

This is a one-attribute implementation enabling drop-in runtime dispatch for any existing
`gpu.binary` without changing upstream MLIR infrastructure. The cleanup RFC clarifies which
layer owns this logic (the offloading handler, not the dialect core), making the contribution
clean and well-scoped.

### Opportunity 2: Tri-Vendor Dispatch Benchmarks

With XeVM upstreamed, a `gpu.module` compiled to `gpu.binary` with NVVM + ROCDL + XeVM targets
is now expressible in upstream MLIR. The missing runtime layer is the only barrier to
end-to-end NVIDIA → AMD → Intel dispatch from a single compiled module. Benchmarking this gap:
- Measuring the overhead of `#gpu.select_object` (compile-time, requires rebuild per vendor)
  vs. a `#gpu.runtime_select` handler (one binary, runtime detection)
- Quantifying the binary size cost of embedding three vendor objects vs. one
- Profiling driver-call overhead for Level Zero (XeVM), CUDA, and HIP dispatch paths

This data does not exist in the literature and would be the first tri-vendor MLIR dispatch
overhead characterization.

### Opportunity 3: `GPURuntimeDispatchInterface` — A New Interface Proposal

Analogous to `GPUTargetAttrInterface` (which abstracts compile-time serialization across
vendors), a `GPURuntimeDispatchInterface` could abstract runtime dispatch across vendors. It
would expose:
- `detectCompatibleTargets(BinaryOp) → SmallVector<TargetAttrInterface>` — query runtime for
  available devices and filter the binary's objects to compatible ones
- `rankTargets(SmallVector<TargetAttrInterface>) → TargetAttrInterface` — select the best
  match by capability score
- `createKernelHandle(selected, LaunchFuncOp) → KernelHandle` — return a cached handle

This interface mirrors the design of `TargetAttrInterface` in the serialization direction and
would be a natural complement to the cleanup RFC's separation of compilation concerns from
dispatch concerns. Filing this as a follow-up RFC after the cleanup lands would be a clean
upstream contribution path.

### Opportunity 4: AMDGPU Pipeline Parity

The cleanup RFC's focus on semantic clarity exposes a concrete asymmetry: `gpu-lower-to-nvvm`
is a well-documented pipeline helper; no equivalent `gpu-lower-to-rocdl` pipeline exists in
upstream MLIR. AMD users must use IREE or manually assemble passes. Documenting this parity
gap and proposing a `gpu-lower-to-rocdl-pipeline` (or noting that IREE's pipeline should be
factored out) is a complementary contribution that the cleanup RFC creates space for.

### Opportunity 5: RFC: Distributed Heterogeneous Computing Dialect as Compile-Time Counterpart

The June 2025 RFC from IIT Madras (PLDI 2025 SRC) proposes a `schedule`/`task`/`target`
dialect that annotates computation with target devices at the MLIR IR level — a compile-time
heterogeneous dispatch representation. If this lands, it would be the natural source-level
counterpart to a runtime dispatch layer: the IR encodes *intent* (run this on any GPU), and
the runtime selects *mechanism* (pick NVIDIA or AMD or Intel based on availability). A poster
connecting the compile-time RFC to the runtime dispatch gap makes the full stack visible.

---

## Key Technical Evidence

### `gpu-module-to-binary` Already Produces Multi-Vendor Binaries

From `ModuleToBinary.cpp` (confirmed in local literature analysis):

```cpp
for (auto targetAttr : op.getTargetsAttr()) {
  auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);
  auto serializedModule = target.serializeToObject(op, targetOptions);
  objects.push_back(target.createObject(op, *serializedModule, targetOptions));
}
gpu::BinaryOp::create(builder, op.getLoc(), op.getName(), handler,
    builder.getArrayAttr(objects));
```

This already runs for NVVM, ROCDL, and now XeVM. The binary is there. The runtime selection
is not.

### `#gpu.select_object` Selection Logic (Confirmed Compile-Time)

```cpp
// SelectObjectAttr.cpp:
// 1. Integer index → compile-time array subscript
// 2. Target attr match → compile-time scan
// 3. Default → index 0
// No device query. No runtime. No fallback.
```

### ELF Section Embedding Infrastructure (PR #119440, Dec 2024)

The `gpu-module-to-binary` pass now accepts `--section <name>` to embed GPU object bytes in a
named ELF section. This means a host binary can contain per-target GPU objects in separate
sections named `gpu.nvvm.sm_90`, `gpu.rocdl.gfx942`, `gpu.xevm.bmg` — discoverable at runtime
by a dispatch library that iterates ELF sections and matches device capability. The storage
infrastructure for runtime dispatch already exists; the dispatcher does not.

### XeVM Binary Format: SPIR-V (not PTX/HSACO)

XeVM serializes to SPIR-V binary via Intel's `ocloc` offline compiler. This means a runtime
dispatcher selecting XeVM objects must load via Level Zero (`zeModuleCreate` with SPIR-V), not
via `cuModuleLoad` or `hipModuleLoad`. The dispatch path diverges at the module-load level, not
just at the kernel-launch level — a `#gpu.runtime_select` handler must be vendor-aware at least
at the module-load stage.

### llvm.gpu Intrinsics: The Future Fallback Path

PR #131190 (Mar 2025) + PR #174910 (Jan 2026) establish `llvm.gpu.*` intrinsics that abstract
NVPTX/AMDGCN/SPIRV64 thread hierarchy and lane operations. A `gpu.binary` entry of type
`BINARY_LLVM_GPU_IR` (hypothetical extension) storing portable IR with `llvm.gpu.*` intrinsics
could serve as a JIT-compiled fallback for novel targets not covered by pre-compiled objects.
The cleanup RFC's separation of binary management from dialect core makes this extension point
cleaner to define.

---

## EuroLLVM 2026 Context

EuroLLVM 2026 is April 14-15, 2026 in Dublin, with pre-conference workshops on April 13.
CFP closed January 11, 2026. No session list is public as of the poster deadline (April 7).

**Likely community interest:** The GPU dialect cleanup RFC was filed in September 2025, XeVM
landed August 2025, and the SPIR-V portability RFC is from March 2025. All three are active or
recently active threads. The LLVM/Offload workshop (annual since 2023, half-day in 2025) is
likely to continue at EuroLLVM 2026 in some form. GPU/MLIR infrastructure is one of the most
active areas in the LLVM community.

**Poster positioning:** A poster presenting (a) the dispatch gap exposed by the cleanup RFC,
(b) a `#gpu.runtime_select` implementation, and (c) tri-vendor overhead benchmarks would
directly engage the community discussion initiated by the RFC and by XeVM's landing. It fills
the gap the community has identified but not yet addressed.

---

## Gap

The MLIR GPU dialect can compile a single `gpu.module` to a tri-vendor `gpu.binary` containing
one `gpu.object` per target (NVVM, ROCDL, XeVM) — but the only built-in object selector
(`#gpu.select_object`) is compile-time only. To deploy compiled kernels across NVIDIA, AMD, and
Intel hardware, users must produce separate host binaries or use IREE's runtime HAL. The
September 2025 RFC to clean the GPU dialect creates explicit architectural space for a
runtime-selecting offloading handler, but no RFC proposes implementing one.

---

## Proposal

**Title:** `#gpu.runtime_select`: Runtime Vendor Detection for MLIR Multi-Target GPU Binaries

**One-sentence pitch:** Implement a `#gpu.runtime_select` attribute satisfying
`GPUOffloadingLLVMTranslationAttrInterface` that embeds all `gpu.object` entries from a
`gpu.binary` into the host executable and selects among them at program startup by querying
available GPU vendors — enabling a single MLIR-compiled binary to dispatch correctly on NVIDIA,
AMD, and Intel hardware without recompilation.

**Scope:**
1. Implement `#gpu.runtime_select` attribute (C++, ~300 LOC + tests)
2. Implement a vendor detection helper (`mgpuDetectVendor()` or equivalent, ~200 LOC)
3. Benchmark: tri-vendor binary deployment overhead vs. separate per-vendor binaries
4. Measure: binary size, startup latency, per-launch dispatch overhead
5. Demonstrate: single `mlir-opt` + `mlir-cpu-runner` invocation dispatching to whichever
   GPU vendor is present on the test machine

**Connection to cleanup RFC:** The cleanup RFC's restructuring of `gpu.binary` / offloading
handler separation is a prerequisite or parallel contribution. The poster can cite the RFC as
the community-recognized architectural motivation and present `#gpu.runtime_select` as the
concrete realization of the gap it identifies.

---

## Evidence

| Claim | Evidence | Source |
|-------|----------|--------|
| Three production target attrs in MLIR | `#nvvm.target`, `#rocdl.target`, `#xevm.target` all implement `GPUTargetAttrInterface` | PR #148286 (XeVM), D154104 (interface), local lit analysis |
| `#gpu.select_object` is compile-time only | SelectObjectAttr.cpp: integer index or attribute match, no device query | mlir-gpu-infrastructure-2026.md §2 |
| `GPUOffloadingLLVMTranslationAttrInterface` is the extension point | `embedBinary` + `translateLaunchFuncToLLVMCalls` methods; `SelectObjectAttr` is the only upstream impl | GPU Dialect docs, wave-01-mlir-gpu-dialect-dispatch |
| Cleanup RFC identifies binary management / dispatch boundary | RFC #88170: ops "don't belong to the dialect"; 3-page discussion | wave-01-mlir-gpu-dialect-dispatch §5, wave-05-llvm-discourse-rfcs §11 |
| ELF section embedding exists | PR #119440: `--section` flag in `gpu-module-to-binary` | wave-01-mlir-gpu-dialect-dispatch §3, local lit §1 |
| XeVM uses SPIR-V → Level Zero path | PR #148286: `Assembly/Binary/Offload` modes via ocloc; runtime via `zeModuleCreate` | WebFetch: phoronix XeVM article |
| llvm.gpu intrinsics cover thread/lane/barrier ops for NVPTX+AMDGCN+SPIRV64 | PR #131190 (Mar 2025), PR #174910 (Jan 2026) | wave-08-mlir-async-llvm-gpu §2 |
| IIT Madras heterogeneous computing dialect RFC | RFC #86960: `schedule`/`task`/`target` ops, PLDI 2025 SRC | wave-01-mlir-gpu-dialect.md §7, wave-05-llvm-discourse-rfcs §10 |
| No `gpu-lower-to-rocdl-pipeline` exists upstream | Community thread discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir/88627 | wave-01-mlir-gpu-dialect-dispatch §8 |
| MLIR async runtime is CPU-only; no GPU dispatch hook | mlirAsyncRuntime contains zero CUDA/HIP references; `@mgpuLaunchKernel` is the only GPU dispatch point | wave-08-mlir-async-llvm-gpu §1 |

---

## Feasibility

**Time:** 4-6 weeks for a first implementation + benchmarks.

**Code scope:**
- ~300 LOC: `GpuRuntimeSelectAttr` in `mlir/lib/Target/LLVMIR/Dialect/GPU/`
- ~200 LOC: `mgpuDetectVendor()` runtime helper in `mlir/lib/ExecutionEngine/`
- ~150 LOC: lit tests in `mlir/test/Target/LLVMIR/GPU/`
- ~100 LOC: integration test demonstrating single-binary tri-vendor dispatch

**Skills required:** C++, MLIR TableGen, LLVM IR generation (the `embedBinary` method generates
LLVM IR globals and constructor functions — directly analogous to `SelectObjectAttr.cpp` which
is well-documented in local literature). Fabian Mora's `SelectObjectAttr.cpp` implementation
serves as a complete reference.

**Hardware required:** GTX 1650 (NVIDIA, already available) + any AMD GPU for
ROCDL benchmarks. Intel GPU for XeVM is optional — the SPIR-V path can be tested via
software emulation (`ocloc` on CPU).

**Risk:** The cleanup RFC is not yet merged. If the RFC's restructuring moves `gpu.binary` to a
different dialect, the implementation location changes but the interface contract remains. The
`GPUOffloadingLLVMTranslationAttrInterface` is the stable hook regardless of where the op lives.

**Prototype overlap:** The libkdl prototype (`experiments/prototype/src/kdl.c`, ~5100 LOC)
implements runtime vendor detection and capability-based kernel selection at the C runtime level.
The MLIR contribution wraps this logic behind the MLIR extension interface — the detection
algorithm is already proven.

---

## Upstream Path

1. **Now (April 2026):** Implement `#gpu.runtime_select` as out-of-tree attribute, demo on
   existing MLIR test suite. Write benchmarks showing tri-vendor dispatch overhead.

2. **RFC (May 2026):** Post to LLVM Discourse as response to the cleanup RFC (#88170) — "here
   is a concrete runtime dispatch handler that the cleanup's cleaner boundary enables." This
   framing positions the contribution as downstream of community consensus, not ahead of it.

3. **PR (June 2026):** Upstream `GpuRuntimeSelectAttr` to `mlir/lib/Target/LLVMIR/Dialect/GPU/`
   with full tests. Simultaneously propose `GPURuntimeDispatchInterface` as a TableGen interface
   in `mlir/include/mlir/Dialect/GPU/IR/GPUInterfaces.td` — the serialization counterpart.

4. **EuroLLVM 2026 follow-up:** The poster at EuroLLVM (April 14-15, Dublin) seeds community
   discussion; the upstream PR closes the loop by the June MLIR open meeting.

---

## Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | 9/10 | `#gpu.runtime_select` does not exist; no RFC proposes it; `SelectObjectAttr` is the only upstream impl and is compile-time only |
| Feasibility | 8/10 | ~800 LOC; clear reference impl in `SelectObjectAttr.cpp`; prototype detection logic exists in libkdl; hardware available |
| Community fit | 9/10 | Directly answers cleanup RFC's implicit question; tri-vendor (NVVM+ROCDL+XeVM) topic is current and community-urgent |
| Upstream viability | 8/10 | Implements existing interface; non-breaking; cleanup RFC creates alignment; risk is RFC delay |
| Benchmark novelty | 10/10 | First tri-vendor single-binary MLIR dispatch overhead characterization in the literature |
| Connection to libkdl | 9/10 | MLIR-layer wrapper around libkdl's runtime detection logic; positions libkdl as infrastructure, not one-off prototype |
| **Composite** | **8.8/10** | |

---

## Pitch

> MLIR can already compile a single GPU kernel to a fat binary containing NVIDIA, AMD, and
> Intel objects — but it cannot run that binary on whichever GPU happens to be installed.
> The `#gpu.select_object` handler that ships with MLIR picks one vendor at compile time, burning
> the choice into the host executable. The September 2025 GPU dialect cleanup RFC identifies
> this boundary as the wrong place to resolve vendor selection, but proposes no alternative.
>
> This poster presents `#gpu.runtime_select`: a drop-in replacement for `#gpu.select_object`
> that queries available GPU drivers at program startup, matches them against the objects in the
> `gpu.binary`, and loads the best-fit variant. The implementation satisfies
> `GPUOffloadingLLVMTranslationAttrInterface` — the existing MLIR extension point — without
> modifying any upstream infrastructure. Benchmarks characterize, for the first time, the
> tri-vendor dispatch overhead (NVVM vs ROCDL vs XeVM) from a single MLIR-compiled host binary.
> The work directly extends libkdl's runtime detection logic into the MLIR abstraction layer,
> closing the gap between the dialect's packaging capability and its deployment reality.

---

## Related Angles (Cross-References to Other Topics)

| Topic | Connection |
|-------|-----------|
| Topic 01: `gpu.select_variant` op | That topic proposes a new dialect op for runtime variant selection; this topic implements it via the existing offloading handler interface — complementary approaches |
| Topic 05: offload-ld dynamic linker | Both implement runtime binary selection; offload-ld operates at the LLVM offloading layer, `#gpu.runtime_select` at the MLIR offloading handler layer — different levels of abstraction with the same goal |
| Topic 07: OffloadBinary capability metadata | The metadata format needed to tag `gpu.object` entries with queryable capability strings is a prerequisite for `#gpu.runtime_select`'s ranking logic |
| Topic 20: portable GPU IR via llvm.gpu | The `llvm.gpu.*` intrinsics provide the JIT fallback path for novel targets not in the `gpu.binary`; together they define the complete dispatch strategy |
| libkdl main direction | `#gpu.runtime_select` is libkdl's detection and selection logic wrapped in an MLIR attribute — the prototype's C runtime becomes the backend of the MLIR extension |

---

## Inconsistencies and Risk Flags

1. **Cleanup RFC status is undefined:** The RFC has three pages of discussion but no formal
   resolution as of April 2026. If the community decides `gpu.binary` should be removed rather
   than refactored, the implementation target changes. Monitor discourse.llvm.org/t/88170.

2. **XeVM runtime path is Level Zero only:** XeVM's `#xevm.target` produces SPIR-V binary that
   must be loaded via Level Zero (`zeModuleCreate`), not via `mgpuModuleLoad`. The `mgpu*`
   runtime wrappers do not cover Level Zero. A `#gpu.runtime_select` implementation targeting
   XeVM must either (a) add Level Zero wrappers to `mlir/lib/ExecutionEngine/` or (b) restrict
   tri-vendor demo to NVVM + ROCDL + SPIR-V-via-OpenCL. This is a bounded scope question but
   must be resolved before upstreaming.

3. **AMDGPU pipeline parity gap:** `gpu-lower-to-rocdl-pipeline` does not exist in upstream
   MLIR (confirmed: discourse.llvm.org/t/88627). Benchmarking ROCDL dispatch requires either
   IREE or manual pass assembly. This is a real friction point for reproducing benchmarks.

4. **`GPUOffloadingLLVMTranslationAttrInterface` is not TableGen-generated:** The interface is
   defined in C++ headers, not in `GPUInterfaces.td`. Adding a new attribute that implements it
   is straightforward but requires understanding the manual interface registration, which is
   less documented than the TableGen-generated `GPUTargetAttrInterface`.

5. **No EuroLLVM 2026 GPU sessions confirmed:** As of the CFP close (Jan 11, 2026), no accepted
   talks are public. This poster may be the primary GPU-dialect multi-target contribution at the
   event — a positioning advantage, but also a risk if the RFC community chooses a different
   resolution path before April 14.
