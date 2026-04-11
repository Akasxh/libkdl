# Statistical Appendix: 95% Confidence Intervals

**Source:** `pinned-benchmark-results.md`, Section 2 (Cross-Run Statistics)
**Method:** 3 independent pinned runs, cross-run medians as summary statistic
**Formula:** 95% CI = mean +/- t(0.025, df=2) x (stddev / sqrt(3)), where t(0.025, 2) = 4.303

---

## Layer-by-Layer CIs (bench_layers, cross-run medians)

### cuModuleLoadData cold (exec-child)

| Run | Median (ns) |
|-----|-------------|
| 1   | 35,767      |
| 2   | 36,228      |
| 3   | 35,863      |

- Mean of medians: 35,952.7 ns
- Stddev: 243.1 ns
- SE (stddev/sqrt(3)): 140.3 ns
- 95% CI half-width: 4.303 x 140.3 = 604 ns
- **95% CI: 35,953 +/- 604 ns (35,349 -- 36,557 ns)**

### cuLaunchKernel (submit)

| Run | Median (ns) |
|-----|-------------|
| 1   | 1,633       |
| 2   | 1,653       |
| 3   | 1,664       |

- Mean of medians: 1,650.0 ns
- Stddev: 15.7 ns
- SE (stddev/sqrt(3)): 9.07 ns
- 95% CI half-width: 4.303 x 9.07 = 39.0 ns
- **95% CI: 1,650 +/- 39 ns (1,611 -- 1,689 ns)**

### cuStreamSynchronize (GPU RTT)

| Run | Median (ns) |
|-----|-------------|
| 1   | 2,465       |
| 2   | 2,455       |
| 3   | 2,442       |

- Mean of medians: 2,454.0 ns
- Stddev: 11.5 ns
- SE (stddev/sqrt(3)): 6.64 ns
- 95% CI half-width: 4.303 x 6.64 = 28.6 ns
- **95% CI: 2,454 +/- 29 ns (2,425 -- 2,483 ns)**

### Hot-path total (L4 + L5)

| Run | Median (ns) |
|-----|-------------|
| 1   | 4,098       |
| 2   | 4,108       |
| 3   | 4,106       |

- Mean of medians: 4,104.0 ns
- Stddev: 5.3 ns
- SE (stddev/sqrt(3)): 3.06 ns
- 95% CI half-width: 4.303 x 3.06 = 13.2 ns
- **95% CI: 4,104 +/- 13 ns (4,091 -- 4,117 ns)**

---

## Summary Table

| Layer                    | Mean (ns) | 95% CI (ns) | Relative CI |
|--------------------------|-----------|-------------|-------------|
| Cold init + cuModuleLoad | 35,953    | +/- 604     | +/- 1.7%    |
| cuLaunchKernel           | 1,650     | +/- 39      | +/- 2.4%    |
| cuStreamSynchronize      | 2,454     | +/- 29      | +/- 1.2%    |
| **Hot-path (L4+L5)**     | **4,104** | **+/- 13**  | **+/- 0.3%**|

All CIs are narrow (under 2.5% relative), confirming high measurement stability.
The hot-path CI of +/- 13 ns (0.3%) is particularly tight, reflecting sub-microsecond
run-to-run reproducibility of the GPU dispatch floor.

---

## Notes

- t-distribution used because n=3 is too small for normal approximation
- t(0.025, df=2) = 4.303 (two-tailed 95% interval)
- Cross-run medians used rather than means to reduce outlier sensitivity
- CPU pinning (taskset -c 0) eliminates core-migration variance
- All runs on same hardware: GTX 1650, SM 7.5, CUDA 13.1, Linux 6.17.0-20
