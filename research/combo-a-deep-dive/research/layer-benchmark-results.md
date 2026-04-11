# Layer Benchmark Results — LLVM GPU Dispatch Stack

**Date:** 2026-04-09
**Hardware:** NVIDIA GTX 1650, SM 7.5
**Benchmark:** `bench_layers.c` (Approach B from proposal-v2.md)
**Methodology:** CUDA driver API via dlopen, no liboffload build required

## Setup

```
Cubin: /tmp/null_sm75.cubin (4328 bytes, ELF CUBIN SM 7.5)
Warmup: 100 iterations (discarded)
Measure: 10,000 iterations per warm layer
Cold trials: 100 exec-child processes (clean address space per trial)
```

**Cold path note:** CUDA is not fork-safe — `cuInit` in a forked child that
inherits a parent's CUDA state returns `CUDA_ERROR_NOT_INITIALIZED` (code 3).
The cold path uses `execve(/proc/self/exe)` per trial so each child starts
with a pristine address space. All 100/100 cold trials succeeded.

## Raw Results

```
layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             53.1       50.0       70.0       40.0     2655.0
layer2:cuModuleLoadData (cold/exec-child)     54633.2    42670.0   111269.0    31539.0   111269.0
layer2:cuModuleLoadData (warm/same-ctx)       10102.6    10069.0    16311.0     7374.0    59061.0
layer3:cuModuleGetFunction                       57.4       60.0       61.0       50.0      121.0
layer4:cuLaunchKernel (submit)                 1682.7     1573.0     3496.0     1343.0    16391.0
layer5:cuStreamSynchronize (GPU RTT)           2573.9     2475.0     3647.0      381.0    17332.0
```

## Flame-Graph (folded, mean_ns)

```
cuInit;cuDeviceGet;cuCtxCreate 53
cuModuleLoadData 10103
cuModuleGetFunction 57
cuLaunchKernel 1683
cuStreamSynchronize 2574
```

## Summary Statistics

| Metric | Value |
|--------|-------|
| Hot-path dispatch (launch + sync) | **4,257 ns** (4.26 µs) |
| Cold module load (exec-child) | **54,633 ns** (54.6 µs) |
| Warm module load (same context) | **10,103 ns** (10.1 µs) |
| Symbol lookup (cuModuleGetFunction) | **57 ns** (0.06 µs) |
| Module load / launch overhead ratio | **6.0 x** |

## Layer-by-Layer Interpretation

### Layer 1 — cuDeviceGet warm (53 ns median: 50 ns)

Proxy for the per-call driver overhead once init is complete.
This is the floor for any CUDA driver API call: a simple integer
copy through the driver shim. The `cuInit` + `cuCtxCreate` sequence
is a one-time cost and is not re-measurable in-process; the cold
exec-child trials (Layer 2 cold) implicitly include it.

**Poster claim:** Driver shim overhead per call ≈ 50 ns.

### Layer 2 — cuModuleLoadData cold (median: 42,670 ns / 42.7 µs)

Each trial uses `execve(/proc/self/exe)` to get a fresh CUDA driver
state. Includes: process startup, dlopen(libcuda.so.1), cuInit,
cuDeviceGet, cuCtxCreate, and cuModuleLoadData.

The p99 of 111 µs vs. median of 43 µs shows high variance — likely
from OS scheduler jitter and GPU driver initialization variability.
The min of 31.5 µs gives a lower bound on pure cuModuleLoadData
after the driver is already initialized (the exec overhead is fixed
overhead amortized differently per run).

**To isolate cuModuleLoadData only:** warm measurement below is cleaner.

### Layer 2 — cuModuleLoadData warm (median: 10,069 ns / 10.1 µs)

Re-loading the same 4328-byte cubin into an already-initialized context,
10,000 times. The tight IQR (7.4–16.3 µs at min/p99) shows this is
dominated by the driver's ELF parsing and GPU memory allocation, not
OS scheduling noise.

**Poster claim:** Binary selection adds ≤ X µs if the module is cached;
the selection logic itself (a table scan over 2-4 entries) is < 1 µs.
The 10 µs cuModuleLoadData cost is unavoidable if selection requires
loading a different module at dispatch time.

### Layer 3 — cuModuleGetFunction (median: 60 ns)

Symbol lookup by name in an already-loaded module. Essentially a
hash table lookup in the driver's module symbol table. Cost is
negligible (< 1% of dispatch latency).

**Implication for libkdl:** Caching function handles after first lookup
eliminates this cost entirely on subsequent dispatches.

### Layer 4 — cuLaunchKernel submit (median: 1,573 ns / 1.57 µs)

Host-side cost of submitting the launch command to the CUDA stream.
This is the CPU-side push into the command buffer — the GPU has not
yet received or executed the work when this returns.

Variance (min 1.3 µs, p99 3.5 µs) comes from the HW command buffer
and driver lock contention.

**Poster claim:** Raw CUDA launch submission ≈ 1.6 µs (CPU-side only).

### Layer 5 — cuStreamSynchronize (median: 2,475 ns / 2.5 µs)

Time from after launch submission until GPU completion of a 1-thread
null kernel. Includes: GPU kernel scheduling, fetch-decode-execute of
`ret`, completion interrupt or polling, and the driver's
synchronization check.

The minimum of 381 ns is remarkably low — this is the GPU's best-case
round-trip for a no-op. The median 2.5 µs includes driver polling
overhead and PCIe latency (GTX 1650 is on PCIe 3.0 x16).

**Poster claim:** GPU round-trip for a null kernel ≈ 2.5 µs on GTX 1650
(PCIe 3.0). This is the floor for any compute dispatch measurement.

## Overhead Analysis (Approach B Subtraction)

Given these baselines, we can bound liboffload's overhead by subtraction:

```
Direct CUDA hot-path:          launch(1.68µs) + sync(2.57µs) = 4.26 µs
liboffload dispatch overhead:  measured_total - 4.26 µs
```

The binary selection step (OffloadBinary parse + variant table scan)
must fit in the gap between a measured liboffload dispatch time and
the 4.26 µs CUDA baseline. Our theoretical model (from
theoretical-verification.md) predicts ≈ 150-300 ns for the selection
logic itself, which is below the measurement noise floor here.

**Key finding:** The dominant per-dispatch cost is `cuStreamSynchronize`
(2.5 µs), not the selection logic. Optimizing the selection algorithm
matters only if the selection path is on the critical path of the
synchronization — which it is not in the current design (selection
runs before launch, not during sync).

## Comparison With Previous Benchmark (bench_dispatch)

`bench_dispatch.c` measures the kdl abstraction layer (CPU-only path).
These measurements provide the pure CUDA driver baseline for comparison:

| Metric | bench_dispatch (CPU kdl) | bench_layers (raw CUDA) |
|--------|--------------------------|-------------------------|
| Module load (warm) | N/A (CPU-only) | 10.1 µs |
| Launch overhead | < 1 µs (CPU NOP) | 1.7 µs (GPU submit) |
| Total hot-path | < 1 µs | 4.3 µs |

The 4.3 µs hot-path figure should be used as the CUDA baseline when
comparing against liboffload measurements.

## Reproducibility

```bash
# Build
cd /home/akash/PROJECTS/LLVM/experiments/prototype/src
make bench_layers

# Run (requires /tmp/null_sm75.cubin and libcuda.so.1)
./bench_layers

# Regenerate cubin if needed (requires nvcc):
nvcc -arch=sm_75 -cubin -o /tmp/null_sm75.cubin /dev/stdin <<'EOF'
__global__ void null_kernel() {}
EOF
```

## Raw Output (verbatim)

```
=== bench_layers: LLVM GPU dispatch stack layer benchmark ===
warmup=100  measure=10000  cold_trials=100

[bench_layers] loaded cubin from /tmp/null_sm75.cubin (4328 bytes)
CUDA devices found: 1

--- Layer 1: cuDeviceGet (hot-path, in-process) ---
--- Layer 2 (cold): cuModuleLoadData via exec-child ---
[bench_layers] layer2 cold: 100/100 trials succeeded
--- Layer 2 (warm): cuModuleLoadData in existing context ---
--- Layer 3: cuModuleGetFunction ---
--- Layer 4: cuLaunchKernel ---
--- Layer 5: cuStreamSynchronize ---

=== RESULTS (10000 warm iterations, sorted percentiles) ===

layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             53.1       50.0       70.0       40.0     2655.0
layer2:cuModuleLoadData (cold/exec-child)     54633.2    42670.0   111269.0    31539.0   111269.0
layer2:cuModuleLoadData (warm/same-ctx)       10102.6    10069.0    16311.0     7374.0    59061.0
layer3:cuModuleGetFunction                       57.4       60.0       61.0       50.0      121.0
layer4:cuLaunchKernel (submit)                 1682.7     1573.0     3496.0     1343.0    16391.0
layer5:cuStreamSynchronize (GPU RTT)           2573.9     2475.0     3647.0      381.0    17332.0

=== FLAME-GRAPH (folded format, mean_ns) ===
cuInit;cuDeviceGet;cuCtxCreate 53
cuModuleLoadData 10103
cuModuleGetFunction 57
cuLaunchKernel 1683
cuStreamSynchronize 2574

=== SUMMARY ===
Hot-path dispatch (launch+sync):           4257 ns  (  4.26 us)
Cold module load (exec-child/trial):      54633 ns  ( 54.63 us)
Warm module load (same context):          10103 ns  ( 10.10 us)
Symbol lookup (cuModuleGetFunction):         57 ns  (  0.06 us)
ModuleLoad/Launch overhead ratio:           6.0 x

done.
```
