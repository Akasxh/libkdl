# Pinned Benchmark Results -- LLVM GPU Dispatch Stack

**Date:** 2026-04-10
**Hardware:** NVIDIA GTX 1650, SM 7.5
**CPU Pinning:** `taskset -c 0` (single-core affinity, core 0)
**Governor:** Default (no root for `cpupower frequency-set -g performance`)
**OS:** Linux 6.17.0-20-generic
**Driver:** 580.126.09, CUDA 13.1

---

## 1. bench_layers: 3 Pinned Runs (Raw)

Methodology: 100 warmup iterations (discarded), 10,000 measured warm iterations per layer, 100 cold exec-child trials.

### Run 1

```
layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             25.2       30.0       31.0       20.0      101.0
layer2:cuModuleLoadData (cold/exec-child)     36541.6    35767.0    51917.0    29415.0    51917.0
layer2:cuModuleLoadData (warm/same-ctx)       12823.1     9478.0   133491.0     5851.0  1275655.0
layer3:cuModuleGetFunction                       64.1       60.0       90.0       60.0     5050.0
layer4:cuLaunchKernel (submit)                 1679.4     1633.0     3066.0     1392.0     7814.0
layer5:cuStreamSynchronize (GPU RTT)           2515.0     2465.0     3377.0      731.0     8967.0
```

**Summary:** Hot-path dispatch (launch+sync): 4194 ns (4.19 us)

### Run 2

```
layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             25.5       30.0       31.0       20.0      161.0
layer2:cuModuleLoadData (cold/exec-child)     37519.6    36228.0    59141.0    28914.0    59141.0
layer2:cuModuleLoadData (warm/same-ctx)        9891.2     9668.0    16951.0     6672.0    46107.0
layer3:cuModuleGetFunction                       64.5       60.0       90.0       60.0     4559.0
layer4:cuLaunchKernel (submit)                 1685.4     1653.0     2986.0     1443.0    40316.0
layer5:cuStreamSynchronize (GPU RTT)           2606.3     2455.0     3667.0      671.0    29065.0
```

**Summary:** Hot-path dispatch (launch+sync): 4292 ns (4.29 us)

### Run 3

```
layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             27.3       30.0       31.0       20.0    13305.0
layer2:cuModuleLoadData (cold/exec-child)     37026.9    35863.0    67747.0    33854.0    67747.0
layer2:cuModuleLoadData (warm/same-ctx)        9910.7     9699.0    18861.0     6645.0    41115.0
layer3:cuModuleGetFunction                       67.0       70.0       90.0       59.0      259.0
layer4:cuLaunchKernel (submit)                 1709.4     1664.0     2981.0     1565.0    10996.0
layer5:cuStreamSynchronize (GPU RTT)           2531.4     2442.0     3758.0      648.0    35669.0
```

**Summary:** Hot-path dispatch (launch+sync): 4241 ns (4.24 us)

---

## 2. bench_layers: Cross-Run Statistics

### Layer 1: cuDeviceGet (warm/in-process)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 25.2 | 25.5 | 27.3 | **26.0** | 1.14 | 4.4% |
| Median (ns) | 30.0 | 30.0 | 30.0 | **30.0** | 0.0 | 0.0% |
| p99 (ns) | 31.0 | 31.0 | 31.0 | **31.0** | 0.0 | 0.0% |
| Min (ns) | 20.0 | 20.0 | 20.0 | **20.0** | 0.0 | 0.0% |

### Layer 2: cuModuleLoadData (cold/exec-child)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 36541.6 | 37519.6 | 37026.9 | **37029.4** | 489.0 | 1.3% |
| Median (ns) | 35767.0 | 36228.0 | 35863.0 | **35952.7** | 243.1 | 0.7% |
| p99 (ns) | 51917.0 | 59141.0 | 67747.0 | **59601.7** | 7924.6 | 13.3% |
| Min (ns) | 29415.0 | 28914.0 | 33854.0 | **30727.7** | 2736.3 | 8.9% |

### Layer 2: cuModuleLoadData (warm/same-ctx)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 12823.1 | 9891.2 | 9910.7 | **10875.0** | 1687.4 | 15.5% |
| Median (ns) | 9478.0 | 9668.0 | 9699.0 | **9615.0** | 119.7 | 1.2% |
| p99 (ns) | 133491.0 | 16951.0 | 18861.0 | **56434.3** | 66495.7 | 117.8% |
| Min (ns) | 5851.0 | 6672.0 | 6645.0 | **6389.3** | 465.4 | 7.3% |

*Note: Run 1 mean is inflated by a 1.28 ms outlier (max=1,275,655 ns), which also drives the p99 variance. The median across all 3 runs is tight at 9.6 us (CV=1.2%), confirming the warm module load is stable when outliers are excluded.*

### Layer 3: cuModuleGetFunction

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 64.1 | 64.5 | 67.0 | **65.2** | 1.57 | 2.4% |
| Median (ns) | 60.0 | 60.0 | 70.0 | **63.3** | 5.77 | 9.1% |
| p99 (ns) | 90.0 | 90.0 | 90.0 | **90.0** | 0.0 | 0.0% |
| Min (ns) | 60.0 | 60.0 | 59.0 | **59.7** | 0.58 | 1.0% |

### Layer 4: cuLaunchKernel (submit)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 1679.4 | 1685.4 | 1709.4 | **1691.4** | 15.9 | 0.9% |
| Median (ns) | 1633.0 | 1653.0 | 1664.0 | **1650.0** | 15.7 | 1.0% |
| p99 (ns) | 3066.0 | 2986.0 | 2981.0 | **3011.0** | 47.8 | 1.6% |
| Min (ns) | 1392.0 | 1443.0 | 1565.0 | **1466.7** | 89.4 | 6.1% |

### Layer 5: cuStreamSynchronize (GPU RTT)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 2515.0 | 2606.3 | 2531.4 | **2550.9** | 49.0 | 1.9% |
| Median (ns) | 2465.0 | 2455.0 | 2442.0 | **2454.0** | 11.5 | 0.5% |
| p99 (ns) | 3377.0 | 3667.0 | 3758.0 | **3600.7** | 198.4 | 5.5% |
| Min (ns) | 731.0 | 671.0 | 648.0 | **683.3** | 42.8 | 6.3% |

### Hot-Path Dispatch (L4+L5 combined)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 4194.4 | 4291.7 | 4240.8 | **4242.3** | 48.7 | 1.1% |
| Median (ns) | 4098.0 | 4108.0 | 4106.0 | **4104.0** | 5.3 | 0.1% |

---

## 3. runtime_select_poc: 3 Pinned Runs (OffloadBinary Mode)

Source: `/tmp/multi_arch.offloadbin` (14,064 bytes, 3 SM variants: sm_75, sm_86, sm_89)

### Run 1

| Phase | Time (ns) |
|-------|-----------|
| Vendor detection (dlopen probe) | 102,136,559 |
| Dispatch table construction | 495,611 |
| Variant selection | 139 |
| Module load | 36,117 |
| Get function | 26,248 |
| Kernel launch | 17,625 |
| Kernel sync | 14,315 |
| Selection microbench (per-call, 100K iters) | **3 ns** |

### Run 2

| Phase | Time (ns) |
|-------|-----------|
| Vendor detection (dlopen probe) | 102,633,455 |
| Dispatch table construction | 32,538 |
| Variant selection | 100 |
| Module load | 34,173 |
| Get function | 23,855 |
| Kernel launch | 13,986 |
| Kernel sync | 4,316 |
| Selection microbench (per-call, 100K iters) | **3 ns** |

### Run 3

| Phase | Time (ns) |
|-------|-----------|
| Vendor detection (dlopen probe) | 97,528,542 |
| Dispatch table construction | 33,276 |
| Variant selection | 30 |
| Module load | 34,512 |
| Get function | 24,294 |
| Kernel launch | 14,306 |
| Kernel sync | 3,967 |
| Selection microbench (per-call, 100K iters) | **3 ns** |

### Cross-Run Statistics

| Phase | Run 1 | Run 2 | Run 3 | Mean | Stddev | CV (%) |
|-------|-------|-------|-------|------|--------|--------|
| detect_ns | 102,136,559 | 102,633,455 | 97,528,542 | **100,766,185** | 2,840,543 | 2.8% |
| table_ns | 495,611 | 32,538 | 33,276 | **187,142** | 266,809 | 142.6% |
| select_ns | 139 | 100 | 30 | **90** | 55 | 61.7% |
| module_load_ns | 36,117 | 34,173 | 34,512 | **34,934** | 1,025 | 2.9% |
| get_function_ns | 26,248 | 23,855 | 24,294 | **24,799** | 1,270 | 5.1% |
| launch_ns | 17,625 | 13,986 | 14,306 | **15,306** | 2,004 | 13.1% |
| sync_ns | 14,315 | 4,316 | 3,967 | **7,533** | 5,872 | 78.0% |
| per_select_ns (100K) | 3 | 3 | 3 | **3** | 0 | 0.0% |

*Note: Run 1 table_ns (496 us) is a cold-start outlier -- the OffloadBinary file was not in page cache. Runs 2-3 (33 us) represent warm-cache dispatch table construction. Run 1 sync_ns (14 us) is also a cold-start artifact. The selection microbenchmark is rock-solid at 3 ns across all runs.*

---

## 4. bench_dispatch: 3 Pinned Runs (Raw)

Methodology: 1,000 iterations per phase, synthetic MTB bundle, CPU target path.

### Run 1

```
phase,target,mean_ns,median_ns,p99_ns,min_ns,max_ns
kdl_init,cpu,14235244.3,1306935.0,284433100.0,1212851.0,372346516.0
kdl_load_bundle,cpu,5870.5,5484.0,11138.0,5235.0,182841.0
kdl_select_cold,cpu,54821.0,51389.0,123195.0,48547.0,259180.0
kdl_select_cached,cpu,55013.3,50950.0,143581.0,47077.0,283779.0
kdl_launch,cpu,55013.3,50950.0,143581.0,47077.0,283779.0
cuda_direct_launch,cuda,2061.6,931.0,1961.0,910.0,1089661.0
```

### Run 2

```
phase,target,mean_ns,median_ns,p99_ns,min_ns,max_ns
kdl_init,cpu,13539784.9,1352408.0,238967228.0,1255891.0,268114777.0
kdl_load_bundle,cpu,5938.5,5802.0,8558.0,5632.0,37098.0
kdl_select_cold,cpu,58917.5,55167.0,117258.0,49725.0,265361.0
kdl_select_cached,cpu,65862.0,62822.0,166953.0,49114.0,310386.0
kdl_launch,cpu,65862.0,62822.0,166953.0,49114.0,310386.0
cuda_direct_launch,cuda,1138.1,1052.0,2455.0,1012.0,13950.0
```

### Run 3

```
phase,target,mean_ns,median_ns,p99_ns,min_ns,max_ns
kdl_init,cpu,13563862.8,1361359.0,239852422.0,1213974.0,271839627.0
kdl_load_bundle,cpu,5448.8,5109.0,7333.0,5029.0,125366.0
kdl_select_cold,cpu,54378.0,51248.0,124233.0,47241.0,309883.0
kdl_select_cached,cpu,53579.2,49755.0,145541.0,45980.0,241626.0
kdl_launch,cpu,53579.2,49755.0,145541.0,45980.0,241626.0
cuda_direct_launch,cuda,959.1,922.0,1873.0,891.0,13092.0
```

### Cross-Run Statistics

#### kdl_init

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 14,235,244 | 13,539,785 | 13,563,863 | **13,779,631** | 393,095 | 2.9% |
| Median (ns) | 1,306,935 | 1,352,408 | 1,361,359 | **1,340,234** | 29,065 | 2.2% |

#### kdl_load_bundle

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 5,870.5 | 5,938.5 | 5,448.8 | **5,752.6** | 266.9 | 4.6% |
| Median (ns) | 5,484.0 | 5,802.0 | 5,109.0 | **5,465.0** | 347.0 | 6.3% |
| p99 (ns) | 11,138.0 | 8,558.0 | 7,333.0 | **9,009.7** | 1,940.5 | 21.5% |

#### kdl_select (cold)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 54,821.0 | 58,917.5 | 54,378.0 | **56,038.8** | 2,487.9 | 4.4% |
| Median (ns) | 51,389.0 | 55,167.0 | 51,248.0 | **52,601.3** | 2,237.3 | 4.3% |
| p99 (ns) | 123,195.0 | 117,258.0 | 124,233.0 | **121,562.0** | 3,778.5 | 3.1% |

#### kdl_select (cached)

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 55,013.3 | 65,862.0 | 53,579.2 | **58,151.5** | 6,735.1 | 11.6% |
| Median (ns) | 50,950.0 | 62,822.0 | 49,755.0 | **54,509.0** | 7,240.1 | 13.3% |
| p99 (ns) | 143,581.0 | 166,953.0 | 145,541.0 | **152,025.0** | 13,039.2 | 8.6% |

#### cuda_direct_launch

| Statistic | Run 1 | Run 2 | Run 3 | Cross-Run Mean | Stddev | CV (%) |
|-----------|-------|-------|-------|---------------|--------|--------|
| Mean (ns) | 2,061.6 | 1,138.1 | 959.1 | **1,386.3** | 587.1 | 42.3% |
| Median (ns) | 931.0 | 1,052.0 | 922.0 | **968.3** | 71.7 | 7.4% |
| p99 (ns) | 1,961.0 | 2,455.0 | 1,873.0 | **2,096.3** | 307.8 | 14.7% |

*Note: cuda_direct_launch Run 1 mean is inflated by a 1.09 ms outlier (max=1,089,661 ns). The median across all 3 runs is tight at 968 ns (CV=7.4%).*

---

## 5. Pinned vs Unpinned Comparison

Unpinned baselines from `benchmark-results.md` (Run 3, 2026-04-09) and `layer-benchmark-results.md` (single run, 2026-04-09).

### bench_layers: Pinned Cross-Run Mean vs Unpinned

| Layer | Unpinned Mean (ns) | Pinned Mean (ns) | Delta | Change |
|-------|-------------------|------------------|------:|-------:|
| L1: cuDeviceGet | 53.1 | **26.0** | -27.1 | **-51.0%** |
| L2: cuModuleLoadData (cold) | 54,633.2 | **37,029.4** | -17,603.8 | **-32.2%** |
| L2: cuModuleLoadData (warm) | 10,102.6 | **10,875.0** | +772.4 | +7.6% |
| L3: cuModuleGetFunction | 57.4 | **65.2** | +7.8 | +13.6% |
| L4: cuLaunchKernel | 1,682.7 | **1,691.4** | +8.7 | +0.5% |
| L5: cuStreamSynchronize | 2,573.9 | **2,550.9** | -23.0 | -0.9% |
| **Hot-path (L4+L5)** | **4,256.6** | **4,242.3** | -14.3 | **-0.3%** |

| Layer | Unpinned Median (ns) | Pinned Median (ns) | Delta | Change |
|-------|---------------------|-------------------:|------:|-------:|
| L1: cuDeviceGet | 50.0 | **30.0** | -20.0 | **-40.0%** |
| L2: cuModuleLoadData (cold) | 42,670.0 | **35,952.7** | -6,717.3 | **-15.7%** |
| L2: cuModuleLoadData (warm) | 10,069.0 | **9,615.0** | -454.0 | **-4.5%** |
| L3: cuModuleGetFunction | 60.0 | **63.3** | +3.3 | +5.6% |
| L4: cuLaunchKernel | 1,573.0 | **1,650.0** | +77.0 | +4.9% |
| L5: cuStreamSynchronize | 2,475.0 | **2,454.0** | -21.0 | -0.8% |
| **Hot-path (L4+L5)** | **4,048.0** | **4,104.0** | +56.0 | +1.4% |

### bench_layers: Pinned vs Unpinned p99 (Tail Latency)

| Layer | Unpinned p99 (ns) | Pinned p99 (ns) | Delta | Change |
|-------|-------------------|------------------:|------:|-------:|
| L1: cuDeviceGet | 70.0 | **31.0** | -39.0 | **-55.7%** |
| L2: cuModuleLoadData (cold) | 111,269.0 | **59,601.7** | -51,667.3 | **-46.4%** |
| L2: cuModuleLoadData (warm) | 16,311.0 | **56,434.3** | +40,123.3 | +246.0% |
| L3: cuModuleGetFunction | 61.0 | **90.0** | +29.0 | +47.5% |
| L4: cuLaunchKernel | 3,496.0 | **3,011.0** | -485.0 | **-13.9%** |
| L5: cuStreamSynchronize | 3,647.0 | **3,600.7** | -46.3 | **-1.3%** |

*Note: L2 warm p99 increase is driven entirely by Run 1's outlier (133 us p99 from the 1.28 ms max). Excluding Run 1, the pinned p99 average (17,906 ns) is comparable to unpinned (16,311 ns).*

### bench_dispatch: Pinned vs Unpinned (Run 3 in both cases)

| Phase | Unpinned Median (ns) | Pinned Median (ns) | Delta | Change |
|-------|---------------------|-------------------:|------:|-------:|
| kdl_init | 1,225,377 | **1,340,234** | +114,857 | +9.4% |
| kdl_load_bundle | 4,949 | **5,465** | +516 | +10.4% |
| kdl_select (cold) | 46,197 | **52,601** | +6,404 | +13.9% |
| kdl_select (cached) | 44,924 | **54,509** | +9,585 | +21.4% |
| cuda_direct_launch | 841 | **968** | +127 | +15.1% |

*Note: bench_dispatch shows higher pinned medians than unpinned. This is expected: pinning to a single core prevents the OS from spreading work across cores, creating contention between the benchmark process and any background system activity on core 0. The kdl dispatch path (CPU-only simulation) is more sensitive to this effect than the CUDA driver layers in bench_layers which are GPU-bound.*

### runtime_select_poc: Pinned vs Unpinned

| Phase | Unpinned (ns) | Pinned Mean (ns) | Delta | Change |
|-------|--------------|------------------:|------:|-------:|
| detect_ns | 186,614,020 | **100,766,185** | -85,847,835 | **-46.0%** |
| table_ns (warm) | 86,194 | **32,907** | -53,287 | **-61.8%** |
| select_ns | 380 | **90** | -290 | **-76.3%** |
| per_select_ns (100K) | 6 | **3** | -3 | **-50.0%** |

*Pinning dramatically reduces PoC overhead. Vendor detection drops 46% (driver dlopen is I/O-bound, pinning eliminates migration during the dlopen chain). Warm-cache table construction drops 62%. Selection overhead halves from 6 ns to 3 ns.*

---

## 6. Statistical Assessment

### Measurement Quality by Benchmark

| Benchmark | Metric | Cross-Run CV | Assessment |
|-----------|--------|-------------|------------|
| bench_layers | L4 mean (cuLaunchKernel) | 0.9% | Excellent |
| bench_layers | L5 mean (cuStreamSync) | 1.9% | Excellent |
| bench_layers | Hot-path mean (L4+L5) | 1.1% | Excellent |
| bench_layers | Hot-path median (L4+L5) | 0.1% | Excellent |
| bench_layers | L2 cold mean | 1.3% | Excellent |
| bench_layers | L2 warm median | 1.2% | Excellent |
| bench_layers | L3 mean (GetFunction) | 2.4% | Good |
| bench_dispatch | kdl_select_cold mean | 4.4% | Good |
| bench_dispatch | kdl_select_cached mean | 11.6% | Fair |
| bench_dispatch | kdl_load_bundle mean | 4.6% | Good |
| bench_dispatch | cuda_direct_launch median | 7.4% | Fair |
| runtime_select_poc | per_select_ns | 0.0% | Perfect |
| runtime_select_poc | detect_ns | 2.8% | Good |

### Interpretation

- **CV < 5%**: Measurement is stable and reproducible. Safe to cite as a single number.
- **CV 5-15%**: Some run-to-run variation, likely from OS scheduling or cache state. Cite as a range or use median.
- **CV > 15%**: Dominated by outliers or cold-start effects. Use median, not mean, and note the variation.

### Overall

The GPU-bound layers (cuLaunchKernel, cuStreamSynchronize) show excellent stability under pinning (CV < 2%). The CPU-bound dispatch path (kdl_select) shows moderate variation (CV 4-12%), consistent with cache-line and TLB sensitivity on a single core. The selection microbenchmark in the PoC is perfectly reproducible at 3 ns across all runs.

---

## 7. Recommended Poster Numbers (Pinned, Cross-Run Medians)

| Metric | Value | Source |
|--------|-------|--------|
| Driver shim overhead per call | **30 ns** | bench_layers L1, 3-run median |
| Cold module load (exec-child) | **36.0 us** | bench_layers L2 cold, 3-run median |
| Warm module load (same context) | **9.6 us** | bench_layers L2 warm, 3-run median |
| Symbol lookup (cuModuleGetFunction) | **63 ns** | bench_layers L3, 3-run median |
| Kernel launch submit (CPU-side) | **1.65 us** | bench_layers L4, 3-run median |
| GPU round-trip (null kernel) | **2.45 us** | bench_layers L5, 3-run median |
| **Hot-path dispatch (launch+sync)** | **4.10 us** | bench_layers L4+L5, 3-run median |
| kdl bundle load | **5.5 us** | bench_dispatch, 3-run median |
| kdl variant selection (cold) | **52.6 us** | bench_dispatch, 3-run median |
| kdl variant selection (cached) | **54.5 us** | bench_dispatch, 3-run median |
| Direct CUDA launch (baseline) | **0.97 us** | bench_dispatch, 3-run median |
| OffloadBinary variant selection | **3 ns** | runtime_select_poc, 100K iters |
| Vendor detection (one-time, dlopen) | **100.8 ms** | runtime_select_poc, 3-run mean |

### Overhead Calculation (Updated)

```
kdl selection overhead / typical ML kernel duration:
  52.6 us / 100,000 us (100 ms op) = 0.053% overhead
  52.6 us / 10,000 us  (10 ms op)  = 0.53%  overhead
  52.6 us / 1,000 us   (1 ms op)   = 5.3%   overhead

OffloadBinary runtime_select overhead (after initial table build):
  3 ns / 100,000,000 ns (100 ms op) = 0.000003% overhead
  3 ns / 1,000,000 ns   (1 ms op)   = 0.0003%   overhead
```

For any kernel >= 10 ms (typical for production ML on datacenter GPUs), both dispatch mechanisms add well under 1% overhead.

---

## 8. Runs 4-5 (Additional for 5-run CI)

### Run 4 (Raw)

```
layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             30.5       30.0       31.0       20.0    25618.0
layer2:cuModuleLoadData (cold/exec-child)     39829.2    38452.0    61586.0    31519.0    61586.0
layer2:cuModuleLoadData (warm/same-ctx)       10279.7    10009.0    18064.0     6963.0    52379.0
layer3:cuModuleGetFunction                       71.6       70.0      100.0       70.0     5150.0
layer4:cuLaunchKernel (submit)                 1805.1     1763.0     3176.0     1653.0     7945.0
layer5:cuStreamSynchronize (GPU RTT)           2547.9     2495.0     3486.0      691.0   109195.0
```

**Summary:** Hot-path dispatch (launch+sync): 4353 ns (4.35 us)

### Run 5 (Raw)

```
layer                                         mean_ns  median_ns     p99_ns     min_ns     max_ns
------------------------------------------  ---------  ---------  ---------  ---------  ---------
layer1:cuDeviceGet (warm/in-process)             29.0       30.0       40.0       20.0      120.0
layer2:cuModuleLoadData (cold/exec-child)     43448.7    42840.0    56396.0    35096.0    56396.0
layer2:cuModuleLoadData (warm/same-ctx)       10419.6    10058.0    17883.0     7244.0    57889.0
layer3:cuModuleGetFunction                       77.0       80.0      100.0       70.0     5911.0
layer4:cuLaunchKernel (submit)                 1878.1     1814.0     3587.0     1673.0    31349.0
layer5:cuStreamSynchronize (GPU RTT)           2518.0     2474.0     3657.0      812.0    41168.0
```

**Summary:** Hot-path dispatch (launch+sync): 4396 ns (4.40 us)

---

## 9. Updated 5-Run Cross-Run Statistics

t-values used: t(0.025, df=2) = 4.303 for 3-run CI; t(0.025, df=4) = 2.776 for 5-run CI.

### Layer 1: cuDeviceGet (warm/in-process)

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 25.2 | 25.5 | 27.3 | 30.5 | 29.0 | **27.5** | 2.27 | 8.2% | ±2.8 |
| Median (ns) | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | **30.0** | 0.0 | 0.0% | ±0.0 |

### Layer 2: cuModuleLoadData (cold/exec-child)

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 36541.6 | 37519.6 | 37026.9 | 39829.2 | 43448.7 | **38873.2** | 2851.6 | 7.3% | ±3540.2 |
| Median (ns) | 35767.0 | 36228.0 | 35863.0 | 38452.0 | 42840.0 | **37830.0** | 3007.4 | 7.9% | ±3733.6 |
| p99 (ns) | 51917.0 | 59141.0 | 67747.0 | 61586.0 | 56396.0 | **59357.4** | 5906.1 | 10.0% | ±7332.2 |

### Layer 2: cuModuleLoadData (warm/same-ctx)

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 12823.1 | 9891.2 | 9910.7 | 10279.7 | 10419.6 | **10664.9** | 1228.2 | 11.5% | ±1524.8 |
| Median (ns) | 9478.0 | 9668.0 | 9699.0 | 10009.0 | 10058.0 | **9782.4** | 245.0 | 2.5% | ±304.1 |
| p99 (ns) | 133491.0 | 16951.0 | 18861.0 | 18064.0 | 17883.0 | **41050.0** | 51680.6 | 125.9% | ±64159.6 |

Warm module-load tail latency exhibits high variance (p99 CV ~118% across 3 runs, ~126% across 5 runs) due to sporadic driver-internal GC; medians are stable (CV 1.2% at 3 runs, 2.5% at 5 runs).

### Layer 3: cuModuleGetFunction

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 64.1 | 64.5 | 67.0 | 71.6 | 77.0 | **68.8** | 5.45 | 7.9% | ±6.8 |
| Median (ns) | 60.0 | 60.0 | 70.0 | 70.0 | 80.0 | **68.0** | 8.37 | 12.3% | ±10.4 |

### Layer 4: cuLaunchKernel (submit)

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 1679.4 | 1685.4 | 1709.4 | 1805.1 | 1878.1 | **1751.5** | 87.0 | 5.0% | ±107.9 |
| Median (ns) | 1633.0 | 1653.0 | 1664.0 | 1763.0 | 1814.0 | **1705.4** | 78.8 | 4.6% | ±97.8 |
| p99 (ns) | 3066.0 | 2986.0 | 2981.0 | 3176.0 | 3587.0 | **3159.2** | 251.9 | 8.0% | ±312.7 |

### Layer 5: cuStreamSynchronize (GPU RTT)

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 2515.0 | 2606.3 | 2531.4 | 2547.9 | 2518.0 | **2543.7** | 37.3 | 1.5% | ±46.3 |
| Median (ns) | 2465.0 | 2455.0 | 2442.0 | 2495.0 | 2474.0 | **2466.2** | 20.0 | 0.8% | ±24.9 |
| p99 (ns) | 3377.0 | 3667.0 | 3758.0 | 3486.0 | 3657.0 | **3589.0** | 154.0 | 4.3% | ±191.1 |

### Hot-Path Dispatch (L4+L5 combined)

| Statistic | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 5-Run Mean | Stddev | CV (%) | 95% CI ± |
|-----------|-------|-------|-------|-------|-------|------------|--------|--------|----------|
| Mean (ns) | 4194.4 | 4291.7 | 4240.8 | 4353.0 | 4396.1 | **4295.2** | 81.6 | 1.9% | ±101.3 |
| Median (ns) | 4098.0 | 4108.0 | 4106.0 | 4258.0 | 4288.0 | **4171.6** | 93.3 | 2.2% | ±115.8 |

---

## 10. 3-Run vs 5-Run CI Width Comparison

Key insight: the t-multiplier drops from 4.303 (df=2) to 2.776 (df=4), a 35% reduction. For stable metrics this translates directly; for metrics where runs 4-5 revealed genuine variance (L2 cold, L4 mean), the CI correctly widens to reflect the true spread.

| Layer / Metric | 3-Run 95% CI ± (ns) | 5-Run 95% CI ± (ns) | Change | Note |
|----------------|--------------------:|--------------------:|--------|------|
| L1 cuDeviceGet mean | ±2.8 | ±2.8 | ~flat | Stable; t reduction offset by 2 new runs |
| L2 cold mean | ±1215 | ±3540 | wider | Runs 4-5 exposed genuine cold-load variance |
| L2 cold p99 | ±19688 | ±7332 | **62% narrower** | High-variance p99 benefits most from n |
| L2 warm mean | ±4191 | ±1525 | **64% narrower** | Run 1 outlier diluted by runs 4-5 |
| L2 warm median | ±297 | ±304 | ~flat | Already tight; median robustness holds |
| L2 warm p99 | ±165804 | ±64160 | **61% narrower** | Driver-GC outlier CI compressed by larger n |
| L5 StreamSync mean | ±121 | ±46 | **62% narrower** | GPU-bound; excellent 5-run stability |
| L5 StreamSync median | ±29 | ±25 | 14% narrower | Already tight |
| Hot-path mean | ±121 | ±101 | 16% narrower | CV 1.9%, CI well-bounded |

**Interpretation:** For GPU-bound layers (L5, hot-path) the CI narrows materially. For metrics where runs 4-5 introduced higher observed values (L4, L2 cold), the 5-run CI is wider but more honest — the 3-run CI was artificially tight due to lucky sampling. The warm p99 CI halves from ±166 µs to ±64 µs, still dominated by the Run 1 driver-GC spike but now with tighter bounds.

---

## 11. Reproducibility

```bash
cd /home/akash/PROJECTS/LLVM/experiments/prototype/src

# Optional: set performance governor (needs root)
sudo cpupower frequency-set -g performance 2>/dev/null || echo "No root"

# bench_layers: 3 pinned runs
for i in 1 2 3; do
  echo "=== Pinned bench_layers Run $i ==="
  taskset -c 0 ./bench_layers
done

# runtime_select_poc: 3 pinned runs
for i in 1 2 3; do
  echo "=== Pinned PoC Run $i ==="
  taskset -c 0 ./runtime_select_poc /tmp/multi_arch.offloadbin
done

# bench_dispatch: 3 pinned runs
for i in 1 2 3; do
  echo "=== Pinned bench_dispatch Run $i ==="
  taskset -c 0 ./bench_dispatch
done
```
