# Topic 02: Multi-Target Fallback Chain Lowering in MLIR

**Persona:** MLIR pass engineer
**Topic name:** gpu-fallback-chain-pass
**Research date:** 2026-04-07
**Status:** Complete

---

## Finding

MLIR cannot produce a single artifact with ordered fallback (native PTX > HSACO > SPIR-V > CPU) where the runtime tries each in order. The infrastructure for *packaging* multi-target binaries exists (`gpu-module-to-binary`, `gpu.binary`, `gpu.object`), but the infrastructure for *runtime selection* among them does not. Every existing selection mechanism in MLIR is static and compile-time. The gap is confirmed independently by six separate evidence sources from three distinct ecosystems.

---

## Title

**Multi-Target Fallback Chain Lowering in MLIR: A `gpu.fallback_select` Pass and Runtime Dispatch Table**

---

## Gap

### The core problem: packaging exists, dispatch does not

The `gpu-module-to-binary` pass (landed in LLVM 17, Phabricator D154149) is the MLIR equivalent of `nvcc`'s fatbin packaging step. A single `gpu.module` can carry multiple target attributes — `#nvvm.target`, `#rocdl.target`, `#xevm.target` (Intel, landed August 2025) — and `gpu-module-to-binary` serializes each via the `GPUTargetAttrInterface::serializeToObject()` hook, producing a `gpu.binary` containing one `gpu.object` per target. This is complete and production-quality.

The dispatch side has no equivalent. `#gpu.select_object` is the only op in the GPU dialect that can retrieve an object from a `gpu.binary`. Its semantics are:

> "Select the first `gpu.object` in the binary whose index matches the compile-time integer attribute."

There is no capability query. There is no runtime device probe. There is no fallback iteration. The selected object is determined at MLIR compilation time and never revisited at kernel launch time.

The result: a `gpu.binary` containing `{#nvvm cubin, #rocdl hsaco, #spirv spv, #cpu elf}` is technically representable in MLIR IR today, but there is no lowering path that produces executable code which tries the variants in order at runtime.

### What currently happens (documented pipeline)

```
gpu.module @matmul [#nvvm.target, #rocdl.target, #spirv.target] {
  gpu.func @sgemm(...) { ... }
}
     |
     v  gpu-module-to-binary
gpu.binary @matmul [
  #gpu.object<#nvvm.target,  "...sm_75 cubin..."},
  #gpu.object<#rocdl.target, "...gfx1030 hsaco..."},
  #gpu.object<#spirv.target, "...spv blob..."}
]
     |
     v  gpu.select_object { object_index = 0 }  // compile-time constant
%obj = gpu.object (the cubin blob — always, regardless of runtime device)
     |
     v  gpu-to-llvm
@mgpuModuleLoad(%obj_ptr) -> %module
@mgpuLaunchKernel(%module, "sgemm", ...) -> void
```

Any deployment that calls `mgpuModuleLoad` with an NVIDIA cubin blob on an AMD device fails at the driver level. There is no retry, no fallback, no ordered trial. The compile-time selection is the only selection.

### What IREE does (closest prior art and its limitation)

IREE's `hal.executable.variant` mechanism is the most mature analog. A `hal.executable` carries N variants; the runtime evaluates each variant's `condition` op in declaration order against the bound `!hal.device` and selects the first valid one. This is evaluated at **module load time, not per-dispatch**. IREE Issue #12230 explicitly notes "the runtime kernel selection logic [for multi-pipeline dispatch] is sort of broken" and was deprioritized at P2 in May 2023. All 15 checkboxes in Issue #15334 (strategy multi-versioning epic, filed October 2023) remain unchecked as of April 2026. IREE Issue #50, the foundational issue requesting vendor-agnostic target specification with runtime selection, has been open since October 2019 — 6.5 years.

### What CUDA does (the reference model)

CUDA's fatbin embeds multiple cubins and one PTX blob. The CUDA driver selection algorithm at `cuModuleLoad`/`cuLibraryLoad` time:

1. Enumerate all cubin entries in the fatbin
2. For each cubin, check if `sm_XY` matches executing device's compute capability (same-major check)
3. Select the highest matching cubin; if none, JIT-compile the PTX blob for the executing device

This is ordered fallback within the NVIDIA ecosystem. It crosses the cubin-to-PTX boundary (binary to portable IR), but it never crosses the NVIDIA/AMD/Intel vendor boundary. There is no mechanism in CUDA to fall back to an AMD HSACO or a SPIR-V binary. The fallback chain is intra-vendor only.

### What LLVM's `llvm.gpu.*` intrinsics provide (and their SPIR-V gap)

PR #131190 (March 2025, Joseph Huber) introduced `llvm.gpu.*` LLVM IR intrinsics — `llvm.gpu.thread.id.x`, `llvm.gpu.num.lanes`, `llvm.gpu.ballot`, `llvm.gpu.shuffle.idx.u32`, etc. — that lower to the correct NVPTX/AMDGCN/SPIRV64 instruction sequences at code generation time. The explicit design goal: "postpone choosing a target architecture for libc until JIT time."

PR #174910 (merged January 9, 2026) added `spirvintrin.h`, explicitly acknowledged as "intentionally inefficient — this is just to start." The SPIR-V lowering for `llvm.gpu.read_first_lane` does not use the dedicated `OpGroupNonUniformBroadcastFirst` SPIR-V instruction; it uses a scalar fallback. For compute-intensive ML kernels (matrix multiply, attention) the performance gap is material, not cosmetic.

This means the `llvm.gpu.*` path produces correct cross-vendor code but not peak-performance cross-vendor code. A practical fallback chain still needs native PTX and HSACO as performance-first entries, with the portable `llvm.gpu.*` IR as the correctness-first fallback — and that is precisely the structure this poster proposes.

---

## Proposal

### A `gpu-fallback-chain-lower` MLIR pass

The pass consumes a `gpu.binary` containing multiple `gpu.object` entries and produces LLVM IR implementing ordered fallback selection at runtime.

**Input IR:**

```mlir
%binary = gpu.binary @matmul [
  #gpu.object<#nvvm.target<sm_75>,   "...ptx/cubin blob...">
  #gpu.object<#rocdl.target<gfx1030>, "...hsaco blob...">
  #gpu.object<#spirv.target,          "...spirv blob...">
  #gpu.object<#cpu.target,            "...native elf...">
] {offloading_handler = #gpu.select_object<0>}
```

**Output LLVM IR (structural sketch):**

```llvm
; Generated fallback chain for @matmul::@sgemm
; Invoked once at program init; result cached.
define ptr @__kdl_resolve_matmul_sgemm(i32 %device_id) {
entry:
  %vendor = call i32 @__kdl_get_device_vendor(i32 %device_id)
  %arch   = call i32 @__kdl_get_device_arch(i32 %device_id)
  switch i32 %vendor, label %try_spirv [
    i32 KDL_VENDOR_NVIDIA, label %try_ptx
    i32 KDL_VENDOR_AMD,    label %try_hsaco
  ]
try_ptx:
  %ok_ptx = call i1 @__kdl_can_load_ptx(i32 %arch, ptr @matmul_ptx_blob, i64 ptx_size)
  br i1 %ok_ptx, label %use_ptx, label %try_spirv
use_ptx:
  %h = call ptr @__kdl_load_and_cache(ptr @matmul_ptx_blob, i64 ptx_size, i32 %device_id)
  ret ptr %h
try_hsaco:
  %ok_hsa = call i1 @__kdl_can_load_hsaco(i32 %arch, ptr @matmul_hsaco_blob, i64 hsaco_size)
  br i1 %ok_hsa, label %use_hsaco, label %try_spirv
use_hsaco:
  %h = call ptr @__kdl_load_and_cache(ptr @matmul_hsaco_blob, i64 hsaco_size, i32 %device_id)
  ret ptr %h
try_spirv:
  ; SPIR-V is the cross-vendor fallback; attempt via OpenCL or Vulkan compute
  %ok_spv = call i1 @__kdl_can_load_spirv(i32 %device_id, ptr @matmul_spirv_blob, i64 spirv_size)
  br i1 %ok_spv, label %use_spirv, label %use_cpu
use_spirv:
  %h = call ptr @__kdl_load_and_cache(ptr @matmul_spirv_blob, i64 spirv_size, i32 %device_id)
  ret ptr %h
use_cpu:
  %h = call ptr @__kdl_load_cpu_impl(ptr @matmul_cpu_blob, i64 cpu_size)
  ret ptr %h
}
```

The dispatch table (`__kdl_resolve_*`) is invoked once per (kernel, device) pair; results are memoized in a lock-free hash table keyed by `(binary_ptr, device_id)`. Subsequent dispatches call through the cached handle with zero selection overhead.

### `#gpu.fallback_chain` offloading handler attribute

Rather than modifying `gpu.select_object` (which is currently used for extract-one semantics), add a new offloading handler attribute:

```tablegen
def GPU_FallbackChainAttr : GPU_Attr<"FallbackChain", "fallback_chain",
    [DeclareAttrInterfaceMethods<GPU_OffloadingTranslationAttrInterface>]> {
  let parameters = (ins
    "ArrayAttr":$priority_order,   // ordered list of target attribute classes
    "StringAttr":$cost_model       // "first_valid" | "roofline" | "pgo"
  );
  let assemblyFormat = "`<` $priority_order (`,` $cost_model^)? `>`";
}
```

Usage:

```mlir
gpu.binary @matmul [...]
  {offloading_handler = #gpu.fallback_chain<[#nvvm.target, #rocdl.target, #spirv.target, #cpu.target], "first_valid">}
```

The `translateToLLVMIR()` method on this attribute generates the fallback chain LLVM IR shown above, driven by the `priority_order` list.

### Integration points with existing passes

- `gpu-module-to-binary`: unchanged — continues to produce all `gpu.object` entries
- `gpu.select_object`: unchanged — continues to work for its existing use case (compile-time extraction)
- `gpu.fallback_chain` handler: new lowering path, selected when the offloading_handler attribute is set to `#gpu.fallback_chain`
- `gpu-to-llvm` conversion: detects `#gpu.fallback_chain` attribute and emits the runtime resolver instead of a static object extraction

### CUDA fatbin analogy: replicating the pattern generically

CUDA's fatbin selection is embedded inside the driver as opaque behavior. The proposed MLIR lowering makes this algorithm explicit and inspectable at the MLIR level. Three design decisions mirroring CUDA's approach:

| CUDA fatbin | MLIR gpu.fallback_chain |
|------------|------------------------|
| Cubin entries with `sm_XY` tags | `gpu.object` entries with typed target attributes |
| PTX as forward-compat fallback | `#spirv.target` or `#gpu.object<llvm.gpu IR>` as cross-vendor fallback |
| Driver-internal selection at `cuModuleLoad` | `__kdl_resolve_*()` function called once at init, result cached |
| NVIDIA-only | All vendors: NVPTX, AMDGCN, SPIRV, CPU |
| Closed-source | Explicit LLVM IR, auditable, PGO-instrumentable |

The key generalization: CUDA's algorithm selects across architectures within one vendor. The proposed mechanism selects across vendors, architectures, and ISA levels (binary → portable IR → CPU scalar), with the CPU ELF as the guaranteed-correct fallback.

### ExecuTorch fallback model: applicability

ExecuTorch's RFC #13732 (August 2025) proposed `combine([QNNRecipe(), XNNPACKRecipe()])` as a priority-ordered fallback API at the AOT partitioning level. The structural analogy to the proposed MLIR pass:

| ExecuTorch | MLIR gpu.fallback_chain |
|-----------|------------------------|
| `combine([backend_A, backend_B, ...])`  | `#gpu.fallback_chain<[#nvvm, #rocdl, #spirv, #cpu]>` |
| AOT partitioning priority | Runtime object selection priority |
| Per-subgraph delegation | Per-kernel binary selection |
| `.pte` file per target | Single `gpu.binary` with all objects |
| Portable Kernel Library = CPU fallback | `#cpu.target gpu.object` = CPU fallback |

ExecuTorch's model differs in a key dimension: its fallback is decided at AOT time by the partitioner (which subgraph goes to which backend). The MLIR proposal decides at runtime (which pre-compiled binary gets loaded for the detected device). The MLIR model is more flexible for heterogeneous cloud deployments where the target device is unknown at compile time.

---

## Evidence

### Direct evidence for the gap

| Evidence item | Source | What it shows |
|--------------|--------|---------------|
| `#gpu.select_object` semantics: index-based, compile-time | MLIR GPU Dialect docs (https://mlir.llvm.org/docs/Dialects/GPU/), wave-01-mlir-gpu-dialect-dispatch Source 1 | No runtime query; always compile-time |
| `gpu-module-to-binary` produces all objects, selects none | Phabricator D154149, wave-01-mlir-gpu-dialect-dispatch Source 2 | Packaging complete; dispatch absent |
| `#gpu.select_object` reason: "only index-based" | MLIR GPU Dialect docs, wave-01-mlir-gpu-dialect-dispatch Source 1 | No `GPURuntimeDispatchInterface` exists |
| Intel XeVM landed August 2025 | Phoronix news, wave-01-mlir-gpu-dialect-dispatch Source 7 | MLIR now has 3 vendor targets; dispatch urgency increased |
| IREE Issue #50: open since October 2019 | https://github.com/iree-org/iree/issues/50 | Runtime best-match selection = 6.5-year open problem |
| IREE Issue #15334: all checkboxes unchecked | https://github.com/iree-org/iree/issues/15334 | Strategy multi-versioning: 0% implemented |
| IREE Issue #12230: Phase 2/3 "sort of broken", deprioritized | https://github.com/iree-org/iree/issues/12230 | Multi-pipeline runtime selection explicitly stalled |
| LLVM Issue #75356: name-based kernel loading absent | https://github.com/llvm/llvm-project/issues/75356 | libomptarget cannot dispatch unlisted kernels; dynamic selection blocked |
| PR #174910 SPIR-V gpuintrin.h "intentionally inefficient" | commit 5c4324326d, wave-08-mlir-async-llvm-gpu | SPIR-V as sole fallback is not performance-competitive |
| Joseph Huber LLVM DevMtg 2025: "ld.so for GPU code" framing | LLVM DevMtg 2025 slides (huber.pdf), wave-02-llvm-offload-runtime Source 9 | Community validates the problem; nobody has implemented selection policy |
| No `GPURuntimeDispatchInterface` in MLIR | MLIR source scan, wave-01-mlir-gpu-dialect-dispatch | Compile-time only; confirmed absence |

### Evidence for the CUDA fatbin analogy

| Evidence item | Source |
|--------------|--------|
| CUDA fatbin two-tier model (cubin + PTX JIT) | CUDA Programming Guide (CUDA 13.2) §2, wave-02-fat-binary-multiversioning Source 1 |
| PTX as forward-compatibility mechanism (not workaround) | NVIDIA Technical Blog "Understanding PTX" (2025), wave-02-fat-binary-multiversioning Source 2 |
| `cuLibraryLoad` / `cudaLibraryLoad` (CUDA 12.0, context-independent) | NVIDIA blog "CUDA Context-Independent Module Loading" (2023), wave-06-dynamic-linking-gpu Source 2 |
| AMD HSA: 6-step load pipeline vs. CUDA 2-step | HSA Runtime AMD, LLVM AMDGPU docs, wave-06-dynamic-linking-gpu Source 3 |
| Vendor load APIs compared: `cuModuleGetFunction`, `hsa_executable_get_symbol_by_name`, `clCreateKernel` | wave-06-dynamic-linking-gpu Pattern 1 table |
| HetGPU virtual ISA (arxiv 2506.15993): single binary JIT across all vendors | wave-02-fat-binary-multiversioning Source 8 |

### Evidence for ExecuTorch dispatch analogy

| Evidence item | Source |
|--------------|--------|
| RFC #13732 `combine([QNN, XNNPACK])` priority API | https://github.com/pytorch/executorch/issues/13732, wave-04-executorch-edge-dispatch Source 6 |
| OpenVINO backend: nested dispatch within ExecuTorch delegate | Intel developer blog, wave-04-executorch-edge-dispatch Source 3 |
| ExecuTorch vs IREE vs ONNX Runtime dispatch philosophy table | wave-04-executorch-edge-dispatch Source 8 |
| ExecuTorch AOT-only, no runtime backend switching | wave-04-executorch-edge-dispatch Source 1 |

### Evidence for `llvm.gpu.*` SPIR-V path limitations

| Evidence item | Source |
|--------------|--------|
| PR #131190: NVPTX + AMDGCN lowering production-ready | wave-08-mlir-async-llvm-gpu §2.7 |
| PR #174910: SPIR-V lowering "intentionally inefficient" | wave-08-mlir-async-llvm-gpu §2.3 |
| Warp-size abstraction (32 vs. 64 lanes) not covered by `llvm.gpu.*` | wave-08-mlir-async-llvm-gpu §2.5 |
| MMA/tensor-core operations outside `llvm.gpu.*` scope | wave-08-mlir-async-llvm-gpu §2.5 |
| SPIR-V capability fragmentation (Shader vs. Kernel trees) | literature/iree-2026-state.md §5.1 |

---

## Feasibility

### Score: 7/10

**Why this is achievable:**

1. All the infrastructure for multi-target binary packaging already exists and is production-quality. The `gpu-module-to-binary` pass, `GPUTargetAttrInterface`, and the tri-vendor target attributes (NVVM, ROCDL, XeVM) are merged and tested. The pass engineer's job is to add a lowering path, not rebuild the packaging system.

2. The runtime resolver function (`__kdl_resolve_*`) is a well-understood algorithm. The libkdl prototype at `experiments/prototype/src/kdl.c` (~5100 LOC) implements the same resolver over a flat dispatch table and has been validated on GTX 1650 + CPU. The LLVM IR generation target is a subset of what libkdl already does.

3. Adding a new `GPU_OffloadingTranslationAttrInterface` implementation for `#gpu.fallback_chain` follows the exact same pattern as `#gpu.select_object`. The attribute definition, tablegen, and `translateToLLVMIR()` method are mechanical; the algorithmic complexity is in the runtime library, not in the MLIR pass.

**Why the score is not 10/10:**

1. The runtime library (`libkdl_mlir_runtime.so`) required to resolve symbols like `@__kdl_get_device_vendor`, `@__kdl_can_load_ptx`, etc., does not yet exist as a standalone MLIR-linkable artifact. Packaging it requires decisions about how it links with existing MLIR runtime wrappers (`mlir_cuda_runtime.so`, `mlir_rocm_runtime.so`).

2. Testing tri-vendor fallback chains requires access to NVIDIA + AMD + Intel GPU hardware in the same CI environment. LLVM's CI infrastructure has NVIDIA GPU runners (nvptx CI) but not a tri-vendor setup. The poster demo on GTX 1650 + CPU validates the 2-step chain (GPU → CPU fallback) but not the 4-step chain (PTX > HSACO > SPIR-V > CPU).

3. The `#gpu.fallback_chain` attribute requires MLIR review consensus on whether "runtime dispatch semantics" belongs in the GPU dialect or in a separate dialect. The GPU dialect cleanup RFC (September 2025, discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170) is actively reshaping the dialect boundary — the RFC may either create a clean home for this attribute or make it architecturally premature.

### Implementation stages for a poster demo

| Stage | Effort | Output |
|-------|--------|--------|
| 1. Add `#gpu.fallback_chain` attr to GPU dialect | 2-3 days | Parses, round-trips, appears in IR |
| 2. Implement `translateToLLVMIR()` for 2-step chain (GPU → CPU) | 3-5 days | Executable PTX → CPU fallback demo |
| 3. Implement `__kdl_mlir_runtime` shim (vendor probe + load/cache) | 1 week | Backed by libkdl prototype code |
| 4. Extend to 4-step chain (PTX > HSACO > SPIR-V > CPU) | 1-2 weeks | Full fallback chain |
| 5. MLIR lit tests, CI, RFC | 4-8 weeks | Upstreamable patch |

Stage 2-3 is feasible before the poster deadline and produces the demo results needed.

---

## Upstream Path

### Near-term (with the poster)

File a Discourse RFC titled: "RFC: `#gpu.fallback_chain` — Runtime Variant Selection for MLIR GPU Binaries." Reference the GPU dialect cleanup RFC (September 2025) as the architectural motivation (separation of binary containers from dispatch policy). Attach the poster as context. Expect discussion about:
- Whether this belongs in `gpu` dialect or as a new `gpu_dispatch` dialect
- Whether the fallback chain semantics should be expressed as an op or as an attribute
- Integration with IREE's HAL variant model

### Dependencies to track

- GPU dialect cleanup RFC (discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170) — resolution determines the dialect home for any new attribute
- `#xevm.target` dispatch path completion — needed for the Intel leg of a tri-vendor chain
- SPIR-V `llvm.gpu.*` optimization PRs (follow-on from PR #174910) — needed before SPIR-V can be a performance-viable fallback

### IREE integration note

IREE's `hal.executable.variant` with condition ops is the HAL-level analog of this proposal. The poster should acknowledge IREE's model and position the MLIR-level proposal as complementing IREE: MLIR encodes the dispatch intent in the IR; IREE's HAL implements a production-quality runtime that could consume it. The two are not competing.

---

## Novelty / Interest Scores

| Dimension | Score | Rationale |
|-----------|------:|-----------|
| Technical novelty | 9/10 | No existing MLIR mechanism performs runtime variant selection; the gap is confirmed by 10+ independent sources spanning 3 years |
| LLVM community interest | 8/10 | GPU dialect cleanup RFC (active), XeVM landing (August 2025), Joseph Huber's "ld.so" framing (DevMtg 2025) — the community is actively looking for this |
| Implementation feasibility | 7/10 | Building on production-ready infrastructure; 2-step demo achievable before deadline |
| Differentiation from prior work | 8/10 | CUDA fatbin exists (intra-vendor), IREE HAL exists (load-time, not per-dispatch) — this proposal is cross-vendor + per-dispatch + IR-native |
| Downstream impact | 9/10 | Any MLIR-compiled ML workload targeting heterogeneous hardware would benefit; composable with libkdl, IREE, ExecuTorch |
| Poster fit | 9/10 | Answers "why does this matter?" with a concrete, demonstrable compilation pipeline change |

**Composite: 8.3/10**

---

## Draft Pitch

> GPU machines are heterogeneous. A cloud VM may present NVIDIA GPUs; a workstation may have AMD; a laptop may have only an Intel iGPU with SPIR-V compute. MLIR can already compile a single kernel for all three targets — the `gpu-module-to-binary` pass packages NVVM, ROCDL, and XeVM objects into one `gpu.binary` — but it cannot yet select the right binary at runtime. The only existing selector, `#gpu.select_object`, is a compile-time constant: you name the winning object's index at lowering time and it never changes.
>
> This proposal adds `#gpu.fallback_chain`, an offloading handler attribute that lowers a `gpu.binary` into a chain of runtime probes: try native PTX if the device is NVIDIA and the compute capability matches; fall back to HSACO if the device is AMD and the gfx ID matches; fall back to SPIR-V if either vendor supports Vulkan compute; finally fall back to a CPU ELF implementation. Each probe is tried in priority order; the first to succeed is cached for the lifetime of the process, imposing zero overhead on subsequent dispatches.
>
> The pattern replicates — generically and transparently — what CUDA's fatbin driver has done within the NVIDIA ecosystem since CUDA 4: try the cubin, fall back to PTX JIT. We extend it across vendor boundaries, and we express it in MLIR IR so that the selection logic is auditable, PGO-instrumentable, and composable with the existing GPU dialect passes.
>
> We present: (1) the `#gpu.fallback_chain` attribute design, (2) the lowered LLVM IR pattern it produces, (3) a demo on GTX 1650 showing GPU-to-CPU fallback when the CUDA driver is absent, and (4) a path to extending the 2-step demo to the 4-step cross-vendor chain once tri-vendor hardware is available in LLVM CI.

---

## Related

- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/directions/02-mlir-native-dispatch-op.md` — complementary proposal for a `gpu.dispatch_select` op (IR-level policy op, same problem from the op side)
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-01-mlir-gpu-dialect-dispatch.md` — primary evidence for `#gpu.select_object` limitations, XeVM landing, GPU dialect cleanup RFC
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-08-mlir-async-llvm-gpu.md` — `llvm.gpu.*` intrinsics coverage, SPIR-V inefficiency confirmed, async dispatch architecture
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-06-dynamic-linking-gpu.md` — cross-vendor load API comparison (cuModuleLoad, hsa_executable, clCreateProgramWithIL)
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-executorch-edge-dispatch.md` — ExecuTorch RFC #13732 fallback model, nested dispatch hierarchy
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-01-iree-hal-runtime-dispatch.md` — IREE Issue #50, #12230, #15334; HAL variant selection limitations
- `/home/akash/PROJECTS/LLVM/literature/iree-2026-state.md` — complete IREE issue tracker analysis, all open gaps confirmed
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-fat-binary-multiversioning.md` — CUDA fatbin mechanics, HetGPU virtual ISA
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` — libkdl prototype implementing the runtime resolver algorithm

---

## Risks

### Risk 1: GPU dialect cleanup RFC may block or redirect this work

The RFC "Cleaning the GPU Dialect" (September 2025, Fabian Mora) is restructuring what belongs in the `gpu` dialect. If the RFC concludes that runtime dispatch policy should live outside `gpu`, the `#gpu.fallback_chain` attribute may need to be a new dialect. This is not fatal (a `gpu_dispatch` dialect would be a reasonable home) but adds MLIR review friction.

**Mitigation:** Monitor the RFC thread. File the `#gpu.fallback_chain` proposal as explicitly dependent on the RFC's architectural conclusion. Frame it as "wherever the RFC says policy lives, this attribute implements it."

### Risk 2: SPIR-V performance gap makes the fallback chain degenerate

If SPIR-V lowering via `llvm.gpu.*` intrinsics remains "intentionally inefficient" (PR #174910's framing), the 4-step chain degenerates to a 2-step chain in practice: PTX (NVIDIA fast path) > HSACO (AMD fast path) > CPU scalar (correctness fallback), with SPIR-V skipped because its performance is worse than the CPU ELF for compute-intensive kernels.

**Mitigation:** Frame SPIR-V as the correctness-bridge for edge cases (ARM Mali, Intel integrated, Qualcomm Adreno), not the performance path. The poster acknowledges this explicitly and positions `llvm.gpu.*` SPIR-V optimization as future work that libkdl/fallback-chain will benefit from automatically when it lands.

### Risk 3: Runtime vendor probing is not zero-overhead at cold start

The first dispatch through a fallback chain requires calling `__kdl_get_device_vendor()`, which calls into the driver (e.g., `cuCtxGetDevice` + `cuDeviceGetAttribute` for CUDA). On cold start with the CUDA context not yet initialized, this could add 10-100ms. For ML inference serving with thousands of calls after warm-up, this is irrelevant. For single-kernel invocations, it matters.

**Mitigation:** The pass should generate both a "lazy init" version (probe on first dispatch, cache result) and an "eager init" version (probe at module load, before first kernel). The eager path amortizes the probe cost over module initialization, matching IREE's load-time variant evaluation pattern. Expose both as options on the `#gpu.fallback_chain` attribute (`init = "lazy" | "eager"`).

### Risk 4: Inconsistency with libomptarget entry table model

LLVM's `libomptarget` dispatches kernels via a compiler-generated `omp_offloading_entries` table. The `#gpu.fallback_chain` lowering emits explicit load/symbol-resolve calls that bypass this table entirely, using the vendor driver APIs directly. Programs mixing OpenMP offload regions with `gpu.fallback_chain`-lowered kernels would have two parallel dispatch systems with no interoperability.

**Mitigation:** The proposal targets the MLIR GPU dialect path, not the OpenMP offload path. These are architecturally separate lowering pipelines. The poster makes this clear: `#gpu.fallback_chain` is for MLIR-native GPU code, not for OpenMP `#pragma target` regions. Long-term, liboffload's `olEnqueueKernelLaunch` could be used as the execution layer, but that is future integration work.
