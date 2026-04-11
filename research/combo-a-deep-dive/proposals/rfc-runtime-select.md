# [RFC] `#gpu.runtime_select` — Runtime variant selection for `gpu.binary`

---

## Summary

`gpu-module-to-binary` can produce multi-vendor binaries (NVIDIA, AMD, Intel since XeVM merged in PR #148286), but `#gpu.select_object` — the only `OffloadingLLVMTranslationAttrInterface` implementation — commits to a single binary at compile time. RFC #88170 (Fabian Mora) separates `gpu.binary` as a container from dispatch policy but leaves the policy slot empty. This RFC fills that slot with `#gpu.runtime_select`, a new attribute that embeds all variants and emits host-side dispatch logic to select among them at module-load time.

**CC:** @fabianmcg (RFC #88170) @jhuber6 (liboffload) @jdenny-ornl (KernelInfo)

---

## The Gap

`#gpu.select_object` resolves the binary choice at LLVM IR translation time by index or static target match. With tri-vendor GPU targets now in MLIR (`#nvvm.target`, `#rocdl.target`, `#xevm.target`), a `gpu.binary` can hold objects for multiple vendors — but no mechanism dispatches among them at runtime. RFC #88170 explicitly envisions dispatch-policy attributes on `gpu.binary`; none exist today.

---

## Proposal: `#gpu.runtime_select`

A new attribute implementing `OffloadingLLVMTranslationAttrInterface` (the same interface `#gpu.select_object` implements):

```mlir
gpu.binary @kernels <#gpu.runtime_select<
    strategy = "rank_by_priority", fallback = "cpu">> [
  #gpu.object<#nvvm.target<chip = "sm_75">, bin = "...cubin...">,
  #gpu.object<#rocdl.target<chip = "gfx90a">, bin = "...hsaco...">,
  #gpu.object<#nvvm.target<chip = "sm_90">, bin = "...cubin-sm90...">
]
```

**`embedBinary()` emits:**

1. N LLVM global constants (`@kernels_blob_0`, `@kernels_blob_1`, ...) — one per `#gpu.object`
2. A `%RuntimeSelectEntry` dispatch table: `{vendor_id, blob_ptr, size}`
3. A `global_ctors` vendor-detection stub that `dlopen`-probes `cuInit`/`hipInit`/`zeInit`
4. A `@kernels_module_ptr` global populated at constructor time with the selected GPU module

**`launchKernel()` emits identical code to `SelectObjectAttr`** — loads from `@kernels_module_ptr`, calls `mgpuModuleGetFunction` + `mgpuLaunchKernel`. Zero hot-path overhead after one-time selection.

**Parameters:**
- `strategy`: `first_compatible` (default), `rank_by_priority`, or `rank_by_capability`
- `fallback`: `error` (default) or `cpu`

---

## Relation to Metadata Vocabulary RFC

The companion RFC proposes five standard OffloadBinary metadata keys. `#gpu.runtime_select` consumes `variant_priority` when `strategy = "rank_by_priority"`. Without the metadata RFC, only `first_compatible` works. The two RFCs compose but can land independently.

---

## Implementation

~780 LOC across five components:

| Component | LOC | Description |
|-----------|-----|-------------|
| TableGen attribute definition | ~30 | `GPU_RuntimeSelectAttr` in `GPUOps.td` |
| `RuntimeSelectAttr.cpp` | ~400 | `embedBinary` + `launchKernel` implementations |
| `GPURuntimeSelectWrappers.cpp` | ~200 | `__gpu_runtime_select_detect_vendor()` and `__gpu_runtime_select_rank()` runtime helpers |
| `--gpu-mark-runtime-select` pass | ~50 | Walks `gpu.module` ops with 2+ targets, sets handler |
| Integration tests | ~100 | Two `.mlir` test files |

No changes to existing passes. Purely additive — dispatched through the existing `OffloadingLLVMTranslationAttrInterface` extension point.

---

## Questions for Community

1. **Where should the attribute live?** Alongside `SelectObjectAttr` in `GPU/IR/GPUOps.td`, or in a separate `GPU/IR/GPURuntimeSelectAttr.td`?
2. **How should vendor detection compose with liboffload?** The current design uses `dlopen`-based probing (matching CPU FMV's resolver pattern). If liboffload gains a `rankImage()` callback, `embedBinary` could emit a call to that instead of inline selection logic. Should we design for that composability now?
3. **Should the selection strategy be configurable or fixed?** Three strategies are defined (`first_compatible`, `rank_by_priority`, `rank_by_capability`). Would a single default strategy reduce complexity without losing value?
