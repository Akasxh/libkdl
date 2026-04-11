# LLVM IR Design Proof: #gpu.runtime_select

This directory contains `runtime_select_example.ll` -- a hand-written LLVM IR file
demonstrating the exact output that `RuntimeSelectAttr::embedBinary()` would produce.

## What it shows

A `gpu.binary @kernels` with two variants (NVIDIA cubin + CPU fallback) compiled
through `#gpu.runtime_select<strategy = "first_compatible", fallback = "error">`.

The IR maps 1:1 to the five phases of `RuntimeSelectAttr::embedBinary()` in
`RuntimeSelectAttr.cpp.sketch`:

| Phase | Sketch Lines | IR Globals/Functions |
|-------|-------------|---------------------|
| 1. Blob embedding | 171-204 | `@kernels_blob_0`, `@kernels_blob_1` |
| 2. Dispatch table | 226-270 | `@kernels_dispatch_table` ([2 x %RuntimeSelectEntry]) |
| 3. Mutable state | 280-310 | `@kernels_selected_idx`, `@kernels_module_ptr` |
| 4. Constructor | 320-450 | `@kernels_ctor` (detect + rank + load) |
| 5. Destructor | 460-515 | `@kernels_dtor` (null-guarded unload) |

Plus `@launch_kernel` from `launchKernel()` (sketch Section 3, lines 522-610).

## How to verify (requires LLVM tools)

```bash
# Parse and verify the IR is well-formed:
opt -verify runtime_select_example.ll -o /dev/null

# Or assemble to bitcode:
llvm-as runtime_select_example.ll -o runtime_select_example.bc
```

## Mapping to C++ sketch

See `RuntimeSelectAttr.cpp.sketch` for the C++ that generates this IR.
Every IR instruction is annotated with the corresponding sketch line number
and the `IRBuilderBase` call that would emit it.

## Key design property

The launch function (`@launch_kernel`) is IDENTICAL to what `SelectObjectAttr`
produces -- a load from a module pointer global + `mgpuModuleGetFunction` +
`mgpuLaunchKernel`. The ONLY difference is that the module pointer is populated
at constructor time by the vendor detection stub, not at compile time. This
proves **zero hot-path overhead**.

## Control flow in the constructor

```
@kernels_ctor:
  entry:
    %vendor = call @__gpu_runtime_select_detect_vendor()
    %idx    = call @__gpu_runtime_select_rank(table, 2, %vendor, 0)
    br (%idx < 0) ? fallback : load

  fallback:                          ; no compatible GPU found
    puts("RuntimeSelectAttr: ...")
    abort()
    unreachable

  load:                              ; success path
    store %idx -> @kernels_selected_idx
    GEP table[%idx] -> blob_ptr, size
    %mod = call @mgpuModuleLoad(blob_ptr, size)
    store %mod -> @kernels_module_ptr
    ret void
```

## What this proves for the poster

1. **Well-defined IR emission** -- no hand-waving; every phase produces concrete IR
2. **Zero launch overhead** -- `@launch_kernel` is byte-for-byte identical to `SelectObjectAttr`
3. **Clean separation** -- all dispatch logic is in `@kernels_ctor` (runs once before `main()`)
4. **Composability** -- runtime helpers are plain C functions, independently testable
5. **Extensibility** -- adding a third vendor variant requires only one blob global + one table entry
