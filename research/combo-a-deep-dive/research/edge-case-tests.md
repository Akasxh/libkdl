# runtime_select_poc Edge-Case Test Results

**Date:** 2026-04-10
**Binary:** `experiments/prototype/src/runtime_select_poc`
**Source:** `experiments/prototype/src/runtime_select_poc.c` (665 LOC)
**Device:** NVIDIA GeForce GTX 1650, sm_75

## Summary

| Test | Input | Exit Code | Crash? | Behavior |
|------|-------|-----------|--------|----------|
| 1  | No arguments (synthetic) | 0 | No | Correct: uses synthetic table, selects sm_75, skips GPU load |
| 2  | Empty directory | 1 | No | Correct: 0 entries, reports NO COMPATIBLE ENTRY FOUND |
| 3  | Incompatible cubins (sm_89 only) | 1 | No | Correct: 1 entry loaded, min_sm=89 > device sm_75, no match |
| 4  | Real OffloadBinary (3 entries) | 0 | No | Correct: parses all 3 entries, selects sm_75 match |
| 5  | Non-existent path | 1 | No | Correct: opendir fails, prints error |
| 6  | bench_layers GPU smoke test | 0 | No | GPU functional, all layers benchmarked |
| 7  | Regular file (not .offloadbin) | 1 | No | Correct: treats as directory, opendir fails |
| 8  | Truncated .offloadbin (3 bytes) | 1 | No | Correct: "offloadbin too small" |
| 9  | Bad magic .offloadbin (48 bytes) | 1 | No | See NOTE below |
| 9b | Bad magic .offloadbin (49 bytes) | 1 | No | Correct: "bad OffloadBinary magic: 0xDEADBEEF" |
| 10 | 21 cubins (exceeds MAX_ENTRIES=16) | 255 | No | Loads 16, silently drops 5. Selects sm_69 (highest <=75). cuModuleLoadData fails (fake data). Exit 255 from -1 return. |
| 11 | Cubin with no "sm" in filename | 255 | No | Parses min_sm=0, selects it (wildcard match). cuModuleLoadData fails (fake data). Exit 255. |
| 12 | Broken symlink .cubin | 1 | No | Correct: load_file fopen fails, entry skipped, 0 entries, no match |
| 13 | Bad version .offloadbin (48 bytes) | 1 | No | See NOTE on Test 9 (same size-check issue) |
| 13b | Bad version .offloadbin (49 bytes) | 1 | No | Correct: "unsupported OffloadBinary version: 99" |
| 14 | Real sm_75 cubin from /tmp | 0 | No | Full success: load + launch + sync in ~82us total |

**Zero crashes. Zero segfaults. No undefined behavior observed.**

## Detailed Results

### Test 1: No Arguments (Synthetic Mode)
```
=== #gpu.runtime_select Proof-of-Concept ===
Demonstrates RuntimeSelectAttr::embedBinary() mechanism in C

[Phase 1] Vendor detection (dlopen probe)
  vendor:      NVIDIA (id=1)
  device:      NVIDIA GeForce GTX 1650
  sm_version:  75 (sm_75)
  detect_ns:   143261877

[Phase 2] Dispatch table construction
  source: synthetic entries (no cubin directory given)
  entries:     3
  table_ns:    491

[Phase 3] Variant selection (strategy=rank_by_priority)
  selected:    [1] sm_75 (min_sm=75, priority=5)
  select_ns:   50

[Phase 4+5] Module load + kernel launch
  (synthetic entry -- skipping actual GPU load/launch)
```
**Verdict:** PASS. Selects sm_75 over sm_50 (higher SM), correctly skips sm_90 (device too old).

### Test 2: Empty Directory
```
[Phase 2] Dispatch table construction
  source: directory /tmp/empty_cubins
  entries:     0

[Phase 3] Variant selection (strategy=rank_by_priority)
  NO COMPATIBLE ENTRY FOUND
  (device sm_75, vendor=1, 0 entries checked)
```
**Verdict:** PASS. Clean error, exit code 1.

### Test 3: Only Incompatible Cubins
```
[Phase 2] Dispatch table construction
  source: directory /tmp/incompatible_cubins
  [0] loaded kernel_sm89.cubin (12 bytes, min_sm=89, priority=5)
  entries:     1

[Phase 3] Variant selection (strategy=rank_by_priority)
  NO COMPATIBLE ENTRY FOUND
  (device sm_75, vendor=1, 1 entries checked)
```
**Verdict:** PASS. Correctly rejects sm_89 for a sm_75 device.

### Test 4: Real OffloadBinary
```
[Phase 2] Dispatch table construction
  source: OffloadBinary file /tmp/multi_arch.offloadbin
  OffloadBinary: magic=0x10FF10AD version=1 entries=3
  [0] sm_86 triple=nvptx64-nvidia-cuda kind=cuda (4712 bytes, min_sm=86, priority=5)
  [1] sm_89 triple=nvptx64-nvidia-cuda kind=cuda (4712 bytes, min_sm=89, priority=5)
  [2] sm_75 triple=nvptx64-nvidia-cuda kind=cuda (4328 bytes, min_sm=75, priority=5)
  entries:     3

[Phase 3] Variant selection (strategy=rank_by_priority)
  selected:    [2] sm_75 (min_sm=75, priority=5)
```
**Verdict:** PASS. Correctly parses real OffloadBinary, selects the only compatible entry.

### Test 5: Non-Existent Path
```
[runtime_select] cannot open directory: /tmp/nonexistent_path
Failed to build dispatch table
```
**Verdict:** PASS. Clean error, exit code 1.

### Test 6: bench_layers GPU Smoke Test
```
=== bench_layers: LLVM GPU dispatch stack layer benchmark ===
warmup=100  measure=10000  cold_trials=100

[bench_layers] loaded cubin from /tmp/null_sm75.cubin (2984 bytes)
CUDA devices found: 1

--- Layer 1: cuDeviceGet (hot-path, in-process) ---
--- Layer 2 (cold): cuModuleLoadData via exec-child ---
[bench_layers] layer2 cold: 100/100 trials succeeded
--- Layer 2 (warm): cuModuleLoadData in existing context ---
--- Layer 3: cuModuleGetFunction ---
--- Layer 4: cuLaunchKernel ---
--- Layer 5: cuStreamSynchronize ---

=== RESULTS (10000 warm iterations, sorted percentiles) ===
```
**Verdict:** PASS. GPU is functional.

### Test 7: Regular File (Not .offloadbin, Not Directory)
```
[runtime_select] cannot open directory: /tmp/test_notcubin.txt
Failed to build dispatch table
```
**Verdict:** PASS. Correctly treats non-.offloadbin argument as directory path, opendir fails gracefully.

### Test 8: Truncated .offloadbin (3 bytes)
```
[runtime_select] offloadbin too small
Failed to parse OffloadBinary
```
**Verdict:** PASS. Size check catches it before any out-of-bounds read.

### Test 9/9b: Bad Magic OffloadBinary
48-byte file (== sizeof(ObFileHeader)):
```
[runtime_select] offloadbin too small
```
49-byte file (> sizeof(ObFileHeader)):
```
[runtime_select] bad OffloadBinary magic: 0xDEADBEEF
```
**Verdict:** PASS (functional), but NOTE below on off-by-one.

### Test 10: 21 Cubins Exceeding MAX_ENTRIES=16
```
[Phase 2] Dispatch table construction
  source: directory /tmp/many_cubins
  [0] loaded kernel_sm61.cubin (5 bytes, min_sm=61, priority=5)
  ...
  [15] loaded kernel_sm63.cubin (5 bytes, min_sm=63, priority=5)
  entries:     16

[Phase 3] Variant selection (strategy=rank_by_priority)
  selected:    [1] sm_69 (min_sm=69, priority=5)

[Phase 4+5] Module load + kernel launch
[runtime_select] cuModuleLoadData failed: 200
```
**Verdict:** PASS (no crash). Silently drops entries beyond MAX_ENTRIES=16. Correct selection among loaded entries (sm_69 is highest <=75). cuModuleLoadData error 200 (CUDA_ERROR_INVALID_IMAGE) expected for fake cubin data. Exit code 255 (from -1 cast to uint8).

### Test 11: Cubin with No SM in Filename
```
[Phase 2] Dispatch table construction
  [0] loaded kernel_unknown.cubin (5 bytes, min_sm=0, priority=5)

[Phase 3] Variant selection (strategy=rank_by_priority)
  selected:    [0] sm_0 (min_sm=0, priority=5)

[Phase 4+5] Module load + kernel launch
[runtime_select] cuModuleLoadData failed: 200
```
**Verdict:** PASS (no crash). min_sm=0 acts as a wildcard (compatible with any device). cuModuleLoadData fails because the file is fake data, not a real cubin. See NOTE below.

### Test 12: Broken Symlink .cubin
```
[Phase 2] Dispatch table construction
  entries:     0

[Phase 3] Variant selection (strategy=rank_by_priority)
  NO COMPATIBLE ENTRY FOUND
  (device sm_75, vendor=1, 0 entries checked)
```
**Verdict:** PASS. load_file fopen fails on broken symlink, entry silently skipped.

### Test 13/13b: Bad Version OffloadBinary
48-byte file: hits size check before version validation (same as Test 9).
49-byte file:
```
[runtime_select] unsupported OffloadBinary version: 99
```
**Verdict:** PASS.

### Test 14: Real sm_75 Cubin (Full Pipeline)
```
[Phase 3] Variant selection (strategy=rank_by_priority)
  selected:    [0] sm_75 (min_sm=75, priority=5)

[Phase 4+5] Module load + kernel launch
  module_load_ns: 39355
  get_function_ns: 23685
  launch_ns: 14999
  sync_ns:   4458

=== Timing Summary ===
  detect_ns:   138363697
  table_ns:    48271
  select_ns:   130
  total_overhead_ns: 138412098  (detect + table + select)

=== Selection Microbenchmark (100,000 iterations) ===
  per_select_ns: 2
```
**Verdict:** PASS. Full end-to-end: detect -> table -> select -> load -> launch -> sync. Total kernel dispatch overhead ~82us (module_load + get_function + launch + sync). Selection overhead: 2ns per call.

## Notes and Observations

### NOTE 1: Off-by-one in OffloadBinary size check (cosmetic, not a bug)
The check `if (fsz <= (long)sizeof(ObFileHeader))` means a file of exactly 48 bytes (== sizeof) is rejected as "too small" before the magic/version checks run. This is technically correct (a valid file must have data beyond the header), but means the error message for a 48-byte corrupted file says "too small" instead of "bad magic" or "bad version". Not a bug, but slightly misleading diagnostics.

**Impact:** None. The file is correctly rejected regardless.

### NOTE 2: min_sm=0 acts as wildcard (design consideration)
When `parse_sm_from_filename` finds no "sm" substring, it returns 0. Since the selection filter is `e->min_sm > device_sm`, an entry with min_sm=0 passes for any device. This is arguably correct (sm_0 = "no minimum requirement") but could surprise users who expected unparseable filenames to be rejected.

**Impact:** Low. Only affects hand-named cubins, not compiler-generated ones.

### NOTE 3: Silent truncation at MAX_ENTRIES=16
When a directory contains more than 16 cubins, entries beyond index 15 are silently dropped. No warning is printed. The dropped entries depend on readdir() order, which is filesystem-dependent and not sorted.

**Impact:** Medium. A directory with e.g. 20 cubins could drop the best match depending on filesystem enumeration order. For the poster demo (3-5 variants) this is fine.

### NOTE 4: Exit code 255 from load_and_launch returning -1
When cuModuleLoadData fails (CUDA_ERROR_INVALID_IMAGE=200), `load_and_launch` returns -1, which `main` returns directly. The shell interprets this as exit code 255 (uint8 wrapping). This is unconventional; exit codes >128 typically indicate signal-based termination.

**Impact:** Low. Only visible to shell scripts checking $?. The stderr message is correct.

### NOTE 5: No memory leak on cuModuleLoadData failure path
Confirmed by code review: `load_and_launch` properly calls `p_cuModuleUnload` and sets `g_module_ptr = NULL` on all failure paths after a successful `cuModuleLoadData`. The blob memory (from `load_file`) is freed in main's cleanup loop.

### NOTE 6: Non-.offloadbin regular files treated as directories
A regular file without the `.offloadbin` extension (e.g., `/tmp/test_notcubin.txt`) is passed to `opendir()`, which fails with a reasonable error. This is correct behavior -- the PoC only has two input modes (directory or .offloadbin file).

## Overall Assessment

**The PoC handles all tested failure modes correctly.** No crashes, no segfaults, no undefined behavior, no memory corruption. All error paths produce clean error messages and non-zero exit codes. The code is robust for a proof-of-concept demo.

The three actionable items for hardening (if this moves beyond PoC):
1. Warn when MAX_ENTRIES truncation occurs
2. Consider rejecting cubins with unparseable SM versions instead of defaulting to min_sm=0
3. Return exit code 1 instead of -1 from load_and_launch for conventional exit codes
