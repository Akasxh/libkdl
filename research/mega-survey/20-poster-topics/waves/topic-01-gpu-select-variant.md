# Topic 01: Runtime Variant Selection Op for MLIR GPU Dialect

---

## The Gap (what's missing in LLVM/MLIR today)

MLIR's `gpu-module-to-binary` pass can already compile one `gpu.module` into a
`gpu.binary` carrying multiple `#gpu.object` entries — one per vendor target
(`#nvvm.target`, `#rocdl.target`, `#xevm.target`).  The *only* built-in mechanism
to consume that bundle is `#gpu.select_object`, which resolves the choice **at
LLVM-IR translation time** (compile time) by index or by static target-attribute
match, embedding a single binary blob as a global string constant with no
runtime switching logic whatsoever [mlir-gpu-infrastructure-2026.md §2].  With Intel
XeVM landing upstream in August 2025 (PR #119440, phoronix.com/news/Intel-XeVM-MLIR-In-LLVM),
MLIR now supports tri-vendor GPU targets in a single `gpu.binary`, but has zero
MLIR-native mechanism to dispatch among them at runtime; the RFC "Cleaning the GPU
Dialect" (discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170, September 2025)
explicitly separates `gpu.binary` (kernel container) from dispatch policy — and that
policy slot is empty in mainline.

---

## The Proposal (what the poster would present)

The poster proposes **`gpu.select_variant`**, a new MLIR op that defers binary
selection from compile time to kernel-launch time, replacing `#gpu.select_object`
as the default offloading handler for multi-target `gpu.binary` ops.

At the IR level the op is a value-returning op that accepts a `!gpu.binary`-typed
value (or symbol reference) and returns a `!gpu.module_handle` through a runtime
callback:

```mlir
%handle = gpu.select_variant @kernels
            attributes {strategy = #gpu.rank_by_device}
%_ = gpu.launch_func %handle::@matmul ...
```

Under the hood, `gpu.select_variant` lowers through the existing
`OffloadingLLVMTranslationAttrInterface` by implementing a new attribute
`#gpu.runtime_select` that overrides `embedBinary` to:

1. Embed **all** `#gpu.object` binary blobs as separate LLVM global constants
   (one array per vendor/arch), rather than committing to a single one.
2. Emit a runtime detection stub that calls `cuInit` / `hipInit` / Level Zero
   `zeInit` via `dlopen`-loaded symbols, returning a vendor enum.
3. Construct a **dispatch table** global (array of `{vendor_id, binary_ptr, size,
   load_fn_ptr}`) populated at image load via `llvm.global_ctors`.
4. Emit `launchKernel` as an indirect call through the dispatch table's selected
   `load_fn_ptr`, routing to `mgpuModuleLoad` (CUDA), `hipModuleLoad` (AMD), or
   the Level Zero kernel-create path (Intel).

A companion pass `--gpu-mark-select-variant` walks `gpu.binary` ops that carry two
or more `#gpu.object` entries and replaces the implicit `#gpu.select_object<0>`
handler with `#gpu.runtime_select`, making opt-in trivial.  The type system impact
is minimal: `gpu.binary` already models a multi-object container; the change is
confined to the offloading attribute and the LLVM translation of launch sites.
No new dialect types are required.

The poster also demonstrates the design on the GTX 1650 + CPU prototype
(`experiments/prototype/src/kdl.c`): libkdl's existing `kdl_select()` function
is the runtime half of `gpu.select_variant`, already verified on hardware.  The
poster maps libkdl's MTB dispatch loop directly to the LLVM IR that
`#gpu.runtime_select::embedBinary` would emit, making the connection between the
prototype and the upstream contribution explicit.

---

## Evidence (sources found)

### 1. MLIR GPU Dialect Official Documentation — https://mlir.llvm.org/docs/Dialects/GPU/ — `#gpu.select_object` picks one object at *compile time* by index; no runtime selection mechanism exists in mainline.

### 2. `mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp` (LLVM monorepo) — `embedBinary` creates a single `@serializedObj` global at translation time, hardcodes the chosen blob; no multi-path dispatch code is emitted [mlir-gpu-infrastructure-2026.md §2].

### 3. `mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp` — iterates all target attributes, calls `serializeToObject` per target, stores all results in `gpu.binary`; multi-target compilation is complete; downstream dispatch is not [mlir-gpu-infrastructure-2026.md §1].

### 4. `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td` — `OffloadingLLVMTranslationAttrInterface` defines `embedBinary` + `launchKernel` as the two-method extensibility point; a new attribute implementing both with multi-blob logic is a complete and self-contained upstream contribution [mlir-gpu-infrastructure-2026.md §3].

### 5. RFC: Cleaning the GPU Dialect — https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170 — September 2025 RFC (Fabian Mora) proposes that `gpu.binary` = container only, explicitly leaving the dispatch-policy slot vacant; wave-05-llvm-discourse-rfcs.md §11 and wave-01-mlir-gpu-dialect-dispatch.md §5.

### 6. Intel XeVM upstreamed August 2025 — https://www.phoronix.com/news/Intel-XeVM-MLIR-In-LLVM — MLIR now has `#nvvm.target`, `#rocdl.target`, `#xevm.target` as first-class GPU targets; a single `gpu.binary` can carry objects for all three vendors today, making runtime dispatch urgent [wave-01-mlir-gpu-dialect-dispatch.md §7].

### 7. RFC: llvm-project/offload (Johannes Doerfert, LLNL) — https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302 — liboffload provides mechanism (`olCreateProgram`, `olEnqueueKernelLaunch`) but explicitly excludes multi-version selection policy from its roadmap; wave-05 §1-2 confirm the policy gap is intentional.

### 8. RFC: SPIR-V IR as a vendor-agnostic GPU representation — https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115 — March 2025; even if SPIR-V becomes the portable IR, vendor-specific extensions (tensor cores, MFMA) and performance differences mean runtime selection among pre-compiled binaries remains necessary [mlir-gpu-infrastructure-2026.md §5].

### 9. IREE issues #50 (2019, OPEN), #12230 (2023, OPEN P2), #15334 (2023, OPEN) — runtime kernel selection logic for HAL variants is "sort of broken"; Phases 2-3 of multi-versioning plan (runtime selection, dynamic tile adaptation) remain unimplemented after 6 years [iree-deep-dive.md §7].

### 10. `experiments/prototype/src/kdl.c` (this repo) — ~5100 LOC prototype implementing vendor detection, MTB dispatch table, and roofline cost model; verified on GTX 1650 + CPU; directly demonstrates the runtime half of the proposed `gpu.select_variant` lowering.

### 11. PR #119440: ELF section option for gpu-module-to-binary — https://github.com/llvm/llvm-project/pull/119440 — December 2024; GPU binary objects can now be embedded in named ELF sections, enabling runtime linker traversal; structurally the same mechanism that `#gpu.runtime_select` would use for multi-blob global emission [wave-01-mlir-gpu-dialect-dispatch.md §3].

### 12. GPU/Offloading Workshop 2025 (LLVM DevMtg, October 27) — https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832 — "Not-Compiler Runtime Library GPUs" talk explicitly targets user-space GPU dispatch from application code, not compiler infrastructure; the community is actively seeking this contribution [wave-05-llvm-discourse-rfcs.md §12].

---

## Feasibility (can a student demo this?)

**Yes, with appropriate scoping.**

The prototype (`kdl.c`) already implements the runtime half: vendor detection, dispatch
table, and binary loading via the appropriate vendor API.  The MLIR half — the new
`#gpu.runtime_select` attribute implementing `OffloadingLLVMTranslationAttrInterface`
— requires:

- ~300-500 LOC C++ for the attribute implementation (model: `SelectObjectAttr.cpp` is
  ~200 LOC; the new version embeds N blobs + emits detection code, roughly 2-2.5x)
- Familiarity with MLIR's TableGen + LLVM IR builder API (non-trivial but documented)
- One integration test: `gpu.binary` with `#nvvm.target` + `#rocdl.target` objects →
  `gpu.select_variant` → verify the correct binary is loaded on a given machine

The demo scope for the poster is achievable: show that on a machine with an NVIDIA GPU,
the NVVM object is loaded; on CPU-only, fall back gracefully.  The AMD path requires
a ROCm machine for the demo, but the code path is symmetric and can be shown via unit
tests without hardware.

The main risk is the `dlopen`-based multi-vendor API linking: calling both `libcuda.so`
and `libamdhip64.so` from the same process requires careful symbol isolation (lazy
`dlopen`, not link-time linking).  This is a known pattern (used by JAX, PyTorch) and
is documented.

**Overall feasibility: Medium.**  Prototypable in 4-6 weeks of focused work; upstream
contribution requires additional testing, review negotiation, and integration with
the cleanup RFC's outcome.

---

## Upstream Path (where would this land in llvm-project?)

| Artifact | Location in llvm-project |
|----------|--------------------------|
| `#gpu.runtime_select` attribute definition | `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td` |
| Attribute implementation (LLVM IR emission) | `mlir/lib/Target/LLVMIR/Dialect/GPU/RuntimeSelectAttr.cpp` |
| Runtime detection helper (vendor probe) | `mlir/lib/Target/LLVMIR/Dialect/GPU/GPUVendorDetect.cpp` |
| `--gpu-mark-select-variant` pass | `mlir/lib/Dialect/GPU/Transforms/MarkSelectVariant.cpp` |
| Pass registration | `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td` |
| Integration tests | `mlir/test/Target/LLVMIR/gpu-runtime-select.mlir` |
| Lit tests for the marking pass | `mlir/test/Dialect/GPU/mark-select-variant.mlir` |

The RFC "Cleaning the GPU Dialect" (#88170) is the natural review context — coordinate
with Fabian Mora to land `#gpu.runtime_select` as the dispatch-policy half of the
cleanup, alongside whatever structural changes the RFC drives.

The `OffloadingLLVMTranslationAttrInterface` is already the correct extension point
and has precedent in the existing `SelectObjectAttr`; no new interfaces are needed.
The LLVM IR patterns emitted (global arrays, `global_ctors`, indirect calls) are
standard and well-reviewed in the NVVM lowering path.

---

## Novelty Score: 9/10

No MLIR-native runtime variant selection mechanism exists today.  `#gpu.select_object`
is the only upstream handler and is compile-time-only by design.  The three-vendor
landscape (NVVM + ROCDL + XeVM) as of August 2025 creates a concrete, visible gap.
The proposal is the direct, minimal answer to a question the RFC "Cleaning the GPU
Dialect" leaves explicitly open.  It is not a survey paper — it is a new op with a
concrete lowering strategy.

## Community Interest Score: 9/10

- The GPU/Offloading Workshop themes for two consecutive years (2024, 2025) are
  "where are we going?" — runtime dispatch is the open question.
- The RFC #88170 (September 2025) has three pages of active discussion.
- The creative-brainstorm synthesis (this repo) ranked `gpu.select_variant` as #1
  of 20 poster ideas with the note "Fills 3-year gap in GPU/Offloading workshop topics."
- Joseph Huber (liboffload maintainer, LLNL) is likely at Dublin; the `rankImage()`
  callback gap (wave-05 §1-2) is the liboffload-side analogue of this proposal.
- Intel XeVM maintainers have immediate motivation: without runtime dispatch, their
  new upstream target attribute cannot be used in heterogeneous deployments.

## Implementation Effort: Medium

The prototype runtime logic is already written (`kdl.c`).  The MLIR C++ work is
bounded by the size of `SelectObjectAttr.cpp` as a template.  The main unknowns are:
(1) integration with the cleanup RFC's final shape for `gpu.binary` semantics, and
(2) the `dlopen` vendor-detection design review (linking policy is politically
sensitive in the LLVM community).

---

## Draft One-Sentence Pitch

"We add `gpu.select_variant`, an MLIR op that defers the choice among a
`gpu.binary`'s pre-compiled NVVM, ROCDL, and XeVM objects to kernel-launch time
via a five-line runtime vendor-detection stub — closing the only remaining gap
between MLIR's already-capable multi-target compilation pipeline and actual
heterogeneous GPU deployment."
