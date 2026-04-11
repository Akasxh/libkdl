# Real CUBIN Test Results — runtime_select_poc

**Date:** 2026-04-09
**Host:** NVIDIA GeForce GTX 1650 (sm_75), CUDA 13.1 (V13.1.80, driver: libcuda.so.1)
**PoC:** `/home/akash/PROJECTS/LLVM/experiments/prototype/src/runtime_select_poc.c`

---

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce GTX 1650 |
| SM version | 7.5 (sm_75) |
| CUDA Toolkit | 13.1 (nvcc V13.1.80) |
| Min supported SM | sm_75 (CUDA 13.1 dropped sm_50/60/70) |
| OS | Linux 6.17.0-20-generic |
| Build | `gcc -O2 -Wall -std=c11 -o runtime_select_poc runtime_select_poc.c -ldl` |

---

## Step 1 — CUDA Kernel Source

```cuda
// /tmp/test_kernel.cu
extern "C" __global__ void null_kernel() {
    // Empty kernel -- measures pure dispatch overhead
}

extern "C" __global__ void add_kernel(float* __restrict__ a,
                                       const float* __restrict__ b,
                                       const float* __restrict__ c,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + c[i];
}
```

Both kernels use `extern "C"` to suppress C++ name mangling. Symbol names in the cubin are exactly `null_kernel` and `add_kernel` (confirmed via `cuobjdump -symbols`).

---

## Step 2 — Compilation Results

| SM Target | Command | Exit | Size | Notes |
|-----------|---------|------|------|-------|
| sm_75 | `nvcc -arch=sm_75 -cubin` | 0 | 4328 bytes | Native for GTX 1650 |
| sm_80 | `nvcc -arch=sm_80 -cubin` | 0 | 4712 bytes | Ampere — incompatible with GTX 1650 |
| sm_89 | `nvcc -arch=sm_89 -cubin` | 0 | 4712 bytes | Ada Lovelace — incompatible with GTX 1650 |
| sm_50 | `nvcc -arch=sm_50 -cubin` | 1 | — | **CUDA 13.1 dropped sm_50 support** |
| sm_60 | `nvcc -arch=sm_60 -cubin` | 1 | — | **CUDA 13.1 dropped sm_60 support** |
| sm_70 | `nvcc -arch=sm_70 -cubin` | 1 | — | **CUDA 13.1 dropped sm_70 support** |

**Note:** CUDA 13.1 minimum architecture is sm_75. The "generic fallback" role that sm_50 played historically is now filled by sm_75 on modern toolchains.

### Cubin structure (sm_75, via `cuobjdump -elf`):
```
64-bit ELF: type=ET_EXEC, ABI=8, sm=75, toolkit=13.1
Symbols:
  add_kernel   STT_FUNC  STB_GLOBAL  STO_ENTRY
  null_kernel  STT_FUNC  STB_GLOBAL  STO_ENTRY
Text sections:
  .text.add_kernel   (0x100 bytes, align=128)
  .text.null_kernel  (0x80 bytes, align=128)
```

---

## Step 3 — Test Directory Setup

```
/tmp/test_cubins/
  kernel_sm75.cubin  (4328 bytes, min_sm=75)
  kernel_sm80.cubin  (4712 bytes, min_sm=80)
  kernel_sm89.cubin  (4712 bytes, min_sm=89)
```

---

## Step 4 — PoC Run: Full Dispatch Pipeline

```
$ ./runtime_select_poc /tmp/test_cubins

=== #gpu.runtime_select Proof-of-Concept ===
Demonstrates RuntimeSelectAttr::embedBinary() mechanism in C

[Phase 1] Vendor detection (dlopen probe)
  vendor:      NVIDIA (id=1)
  device:      NVIDIA GeForce GTX 1650
  sm_version:  75 (sm_75)
  detect_ns:   196386923

[Phase 2] Dispatch table construction
  source: directory /tmp/test_cubins
  [0] loaded kernel_sm80.cubin (4712 bytes, min_sm=80, priority=5)
  [1] loaded kernel_sm89.cubin (4712 bytes, min_sm=89, priority=5)
  [2] loaded kernel_sm75.cubin (4328 bytes, min_sm=75, priority=5)
  entries:     3
  table_ns:    162413

[Phase 3] Variant selection (strategy=rank_by_priority)
  selected:    [2] sm_75 (min_sm=75, priority=5)
  select_ns:   261

[Phase 4+5] Module load + kernel launch
  module_load_ns: 85104
  get_function_ns: 51429
  launch_ns: 27603
  sync_ns:   9689

=== Timing Summary ===
  detect_ns:   196386923
  table_ns:    162413
  select_ns:   261
  total_overhead_ns: 196549597  (detect + table + select)

=== Selection Microbenchmark (100,000 iterations) ===
  per_select_ns: 4
```

---

## Step 5 — Verification Results

### Q1: Does it correctly load the sm_75 cubin?
**YES.** Phase 4 succeeded: `cuModuleLoadData` returned 0, `cuModuleGetFunction("null_kernel")` returned 0, `cuLaunchKernel` returned 0 with synchronization completing cleanly.

Module load: **85 µs**, function lookup: **51 µs**, kernel launch: **28 µs**, sync: **10 µs**.

### Q2: Does it correctly reject sm_89 cubin?
**YES.** The rejection happens in Phase 3 selection (`select_best_entry`): entry `min_sm=89 > device_sm=75` → filtered out. Neither sm_80 nor sm_89 entries are loaded by `cuModuleLoadData` — rejection is purely at the dispatch table selection layer, before any GPU memory is touched.

**Rejection test** (only sm_80 + sm_89 in directory, no sm_75):
```
$ ./runtime_select_poc /tmp/test_cubins_too_new

[Phase 3] Variant selection (strategy=rank_by_priority)
  NO COMPATIBLE ENTRY FOUND
  (device sm_75, vendor=1, 2 entries checked)
Exit code: 1
```

### Q3: Does it select sm_75 over sm_80/sm_89 (higher SM compatibility priority)?
**YES — by compatibility filter, then tiebreak.** The selection algorithm:
1. Filters out entries where `min_sm > device_sm` (sm_80 and sm_89 both rejected here)
2. Among compatible entries, ranks by `variant_priority` (all three have priority=5 in this config)
3. Tiebreak: highest `min_sm` among compatible — sm_75 is the only compatible entry, so it wins

If multiple compatible entries had the same priority, the one with the highest `min_sm` (most specialized) would be chosen, which is the correct semantic (most optimized variant that the device supports).

### Q4: Can it actually launch the null_kernel on the GPU?
**YES.** Full GPU execution confirmed. The kernel launched and completed synchronously via the default CUDA stream:
- `cuLaunchKernel`: grid=1x1x1, block=1x1x1, shared_mem=0, stream=NULL
- `cuStreamSynchronize(NULL)` returned successfully
- No CUDA errors at any phase

---

## Timing Analysis

| Phase | Time | Notes |
|-------|------|-------|
| Vendor detection (`dlopen` + cuInit + cuDeviceGet) | ~196 ms | One-time startup cost; dominated by `dlopen` + driver initialization |
| Dispatch table construction (3 file reads) | ~162 µs | File I/O for 3 cubins |
| Variant selection | **261 ns** | Pure in-memory comparison loop |
| Module load (`cuModuleLoadData`) | ~85 µs | JIT compile from cubin — one-time per kernel family |
| Function lookup (`cuModuleGetFunction`) | ~51 µs | One-time handle resolution |
| Kernel launch (`cuLaunchKernel`) | ~28 µs | Per-launch overhead for null kernel |
| GPU sync | ~10 µs | `cuStreamSynchronize` for null kernel |

### Selection microbenchmark (hot path, 100k iterations):
**4 ns per selection call** — this is the steady-state runtime overhead of `#gpu.runtime_select` vs. compile-time `#gpu.select_object` (0 ns overhead).

At 4 ns per call, a kernel launched at 1 MHz frequency incurs **0.4% selection overhead** in the dispatch path. For real ML workloads (kernel launch at ~10 kHz), this is negligible.

---

## Key Observations for Poster

1. **The PoC implements the full pipeline end-to-end:** dlopen probe → dispatch table → selection → `cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel`. No stubs in the hot path.

2. **Selection is O(n) over dispatch table entries, entirely in CPU cache.** At 4 ns for n=3, it scales to ~40 ns for n=30 variants — still well within the noise floor of a CUDA kernel launch (~28 µs).

3. **The architecture matches the proposed LLVM IR emission exactly:**
   - Each cubin becomes an `@kernels_blob_{idx}` global constant (here: file-loaded blob)
   - The `RuntimeSelectEntry` table mirrors `%RuntimeSelectEntry` struct layout
   - `detect_vendor()` mirrors `__gpu_runtime_select_detect_vendor()`
   - `select_best_entry()` mirrors `__gpu_runtime_select_rank()`
   - The global_ctors stub is the one-time `cuModuleLoadData` call

4. **Rejection is zero-cost for incompatible variants.** sm_80 and sm_89 cubins are loaded into RAM by `load_file()` but never passed to `cuModuleLoadData`. In the embedded-binary model (LLVM IR globals), the blobs are always in the binary but `cuModuleLoadData` is only called on the selected entry.

5. **CUDA 13.1 minimum SM is 75**, not 50/60/70. This is a toolchain reality: any production deployment of libkdl targeting modern CUDA will have sm_75 as the lowest sensible fallback.

---

## Cubin Symbol Naming

`extern "C" __global__` kernels in CUDA have **no underscore prefix** in the cubin ELF symbol table:
- Source: `extern "C" __global__ void null_kernel()` → ELF symbol: `null_kernel`
- The PoC tries `_null_kernel` first (fails silently), then `null_kernel` (succeeds)

For the LLVM `RuntimeSelectAttr` emission, the mangled kernel name used in `cuModuleGetFunction` should match the ELF symbol, which for `extern "C"` kernels is the bare function name.

---

## Files

| File | Description |
|------|-------------|
| `/tmp/test_kernel.cu` | CUDA source with null_kernel + add_kernel |
| `/tmp/kernel_sm75.cubin` | 4328 bytes, native for GTX 1650 |
| `/tmp/kernel_sm80.cubin` | 4712 bytes, Ampere — rejected on sm_75 |
| `/tmp/kernel_sm89.cubin` | 4712 bytes, Ada Lovelace — rejected on sm_75 |
| `/tmp/test_cubins/` | Directory with all 3 cubins for PoC testing |
| `/tmp/test_cubins_too_new/` | Only sm_80 + sm_89 — tests rejection path |
| `/home/akash/PROJECTS/LLVM/experiments/prototype/src/runtime_select_poc.c` | PoC source (no changes needed) |
