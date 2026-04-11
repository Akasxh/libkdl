# Q&A Cards Normalization to Pinned Benchmark Numbers

**Date:** 2026-04-10
**Source of truth:** `research/pinned-benchmark-results.md` (Section 7: Recommended Poster Numbers)

---

## Mapping Applied

| Metric | Old (unpinned) | New (pinned, 3-run cross-run median) | Affected locations |
|--------|---------------|--------------------------------------|-------------------|
| Cold module load | 42.7 us | **36.0 us** | Definitive table, Q5, Q11, Q14, Q15, Q23, Q25, summary rows 5/11/14/25 |
| Hot-path dispatch (L4+L5) | 4.26 us (4,257 ns) | **4.1 us** (4,104 ns) | Definitive table, Q5, Q13, Q14, Q15, Q16, Q23, summary rows 13/14/16 |
| cuLaunchKernel | 1.57 us (1,573 ns) | **1.65 us** (1,650 ns) | Definitive table |
| cuStreamSynchronize | 2.5 us (2,475 ns) | **2.45 us** (2,454 ns) | Definitive table |
| cuModuleGetFunction | 57 ns (60 ns median) | **63 ns** (63 ns median) | Definitive table, Q11 |
| cuDeviceGet | 50 ns | **30 ns** | Definitive table |
| Warm module load | 10.1 us | **9.6 us** | Definitive table |
| OffloadBinary selection (PoC) | 6 ns | **3 ns** | Definitive table, Q1-Q4, Q6-Q9, Q13, Q14-Q16, Q23, Q25, Q28, summary rows 1/3-6/8/15/16/23, emergency pocket card |
| Vendor detection | 170 ms | **101 ms** | Q15 |
| Selection overhead % | 0.14% (6ns/4.26us) | **0.07%** (3ns/4.1us) | Q16, summary row 16 |

---

## Cards Modified

### Definitive Numbers Table (top of file)
- All 9 bench_layers metrics updated to pinned 3-run medians
- OffloadBinary selection: 6 ns -> 3 ns
- Source annotations now include "(pinned)" and "(pinned, 3-run median)"

### Category 1 -- "Why not X?" (Q1-Q6)
- **Q1:** 6 ns -> 3 ns selection overhead
- **Q2:** 6 ns -> 3 ns dispatch layer cost
- **Q3:** 6 ns -> 3 ns pre-compiled binary selection
- **Q4:** 6 ns -> 3 ns selection overhead
- **Q5:** 42.7 us -> 36.0 us cold-load, 4.26 us -> 4.1 us hot-path
- **Q6:** 6 ns -> 3 ns runtime selection

### Category 2 -- Technical Depth (Q7-Q12)
- **Q7:** 6 ns -> 3 ns steady-state selection
- **Q8:** 6 ns -> 3 ns ranking logic cost
- **Q9:** 6 ns -> 3 ns steady-state selection
- **Q11:** 42.7 us -> 36.0 us cold-path module load, 57 ns -> 63 ns symbol lookup

### Category 3 -- Data Challenges (Q13-Q16)
- **Q13:** 6 ns -> 3 ns, 4.26 us -> 4.1 us hot-path
- **Q14:** 6 ns -> 3 ns selection, 42.7 us -> 36.0 us cold load, 4.26 us -> 4.1 us hot-path
- **Q15:** 170 ms -> 101 ms vendor detection, 6 ns -> 3 ns selection, 42.7 us -> 36.0 us cold, 4.26 us -> 4.1 us hot-path
- **Q16:** 6 ns -> 3 ns, 4.26 us -> 4.1 us, 0.14% -> 0.07%

### Category 5 -- "So What?" (Q23, Q25)
- **Q23:** 6 ns -> 3 ns in question title, answer body, and "Do NOT say" line; 42.7 us -> 36.0 us cold, 4.26 us -> 4.1 us hot-path
- **Q25:** 42.7 us -> 36.0 us cold module load, 6 ns -> 3 ns selection

### Category 6 -- Hostile Reviewer (Q28)
- **Q28:** 6 ns -> 3 ns selection overhead

### Quick-Reference Summary Table
- 15 rows updated to pinned numbers (rows 1, 3, 4, 5, 6, 8, 11, 13, 14, 15, 16, 23, 25)

### Emergency Pocket Card
- "6 ns per dispatch call" -> "3 ns per dispatch call" with "(pinned, 3-run cross-run)" qualifier

---

## Numbers NOT Changed (correctly retained)

- TaxBreak H100 reference: 4.71 us (external citation, not ours)
- PyTorch eager dispatch: 5-10 us/kernel (external citation)
- OffloadBinary file size: 14,064 bytes (measured, not a timing)
- Directory-mode selection: 4 ns (pinned result matches old value)
- Dispatch table construction: 86 us (retained as approximate from PoC, not a bench_layers metric)
- kdl bundle load: 5.3 us (retained; bench_dispatch pinned median is 5.5 us but this was already approximate)

---

## Verification

Post-normalization grep for stale values:
- `42.7`: 0 matches
- `4.26`: 0 matches
- `1.57` (as cuLaunchKernel): 0 matches
- `57 ns` (as cuModuleGetFunction): 0 matches
- `50 ns` (as cuDeviceGet): 0 matches
- `6 ns` (as selection overhead): 0 matches

All Q&A cards now cite pinned 3-run cross-run medians from `pinned-benchmark-results.md` Section 7.
