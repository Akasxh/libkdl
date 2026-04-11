# Direction 03: Roofline-Based Cross-Vendor Cost Model for Variant Selection

**Composite Score: 8.25/10**
**Rank: 3 of 8**

---

## Title

**Cross-Vendor Kernel Variant Selection via Analytical Roofline Scoring**

## One-Sentence Description

An analytical cost model that scores pre-compiled kernel variants across NVIDIA, AMD, and CPU using hardware-agnostic roofline parameters (`peak_flops`, `peak_bandwidth`, `ridge_point`), validated against exhaustive autotuning on GEMM configurations.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| tritonBLAS (arXiv:2512.04226) | wave-03-cost-model-selection | `max(T_compute, T_memory)` achieves 94.7% of exhaustive tuning; 50-80 us selection time; works on NVIDIA and AMD |
| Stream-K++ Bloom filter (arXiv:2408.11417) | wave-04-cost-models | 95.8% unsuitable variant elimination in <100 ns via Bloom filter |
| NeuSight (ASPLOS 2025) | wave-03-cost-model-selection | Tile-granularity ML prediction transfers to unseen GPUs at <9% error; 2.3% on H100 |
| Seer (compiled decision trees) | wave-03-cost-model-selection | Sub-microsecond overhead via pre-compiled decision trees for variant selection |
| SparseX (CGO 2026) | wave-03-cost-model-selection | Runtime library selection via classifier accepted at top venue; validates contribution class |
| ML-MLIR cost model | wave-03-cost-model-selection | Build-time contract population via MLIR analysis; enables static metadata in MTB |
| Universal GPU ISA (arXiv:2603.28793) | wave-01-spirv-portable-ir | 6 irreducible architectural divergences define minimum set of vendor-specific parameters |
| CMS Alpaka 30-40% penalty | wave-03-alpaka-portability | Quantifies cost of not having per-device tuning; roofline model eliminates this |

### Key Quantitative Evidence

- tritonBLAS analytical model: **94.7%** of exhaustive tuning quality
- Stream-K++ Bloom filter: **95.8%** variant pruning, **<100 ns** latency
- NeuSight cross-GPU transfer: **2.3%** error (NVIDIA), **<9%** mean (unseen architectures)
- All existing cost models are vendor-specific (cuBLAS heuristic, Ansor, Fasor, MIOpen); no cross-vendor model exists

---

## Novelty Argument

Every existing kernel selection model operates within a single vendor ecosystem. cuBLAS uses a handcrafted heuristic for NVIDIA. MIOpen uses a SQLite autotuning database for AMD. Ansor, Fasor, and NeuSight train on vendor-specific hardware profiles.

No published system scores pre-compiled kernel variants across vendors using a unified analytical model. The roofline framework (introduced by Williams et al., 2009) is well-understood but has never been applied as a runtime variant selector in a multi-vendor dispatch system.

The correction from tritonBLAS is critical: replace weighted sum `w.compute * T_compute + w.memory * T_memory` (current libkdl prototype) with `fmax(T_compute, T_memory)` (roofline-correct). This simple change achieves 94.7% quality.

---

## Feasibility Plan

1. **Roofline parameter collection** (existing): `peak_flops` and `peak_bandwidth` measured per device at first load via micro-benchmark (STREAM-like bandwidth test + FMA throughput test). Cache results persistently.

2. **Per-variant metadata** (build-time): Each kernel variant in the MTB carries `arithmetic_intensity` (ops/byte) and `estimated_flops`. These are computed by the MLIR cost model or measured during offline autotuning.

3. **Selection algorithm** (runtime):
   ```
   for each variant v compatible with device d:
       T_compute = v.flops / d.peak_flops
       T_memory = v.bytes / d.peak_bandwidth
       score[v] = 1.0 / fmax(T_compute, T_memory)
   select argmax(score)
   ```
   Add Bloom filter fast-elimination (Stream-K++ pattern) to prune 95%+ of incompatible variants before scoring.

4. **Validation** (poster): Compare roofline selection against exhaustive profiling on 3+ GEMM configurations (M/N/K variants) across GTX 1650 + CPU. Report selection accuracy percentage.

---

## Poster Potential

**Yes — fills a poster panel as the "how it selects" story.**

- Roofline diagram with variant points plotted across vendors
- Selection accuracy table: roofline vs. exhaustive autotuning
- Bloom filter throughput number (<100 ns per elimination)
- Comparison with tritonBLAS (94.7%) and NeuSight (2.3%) as reference points

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **8/10** | No cross-vendor cost model exists in the literature. The roofline application to variant selection is novel. |
| **Feasibility** | **9/10** | Roofline parameters are simple to measure. tritonBLAS validates the `fmax` formula. Bloom filter is off-the-shelf. |
| **Evidence** | **8/10** | tritonBLAS (94.7%), NeuSight (<9% error), Stream-K++ (95.8% pruning) all validate components. No cross-vendor empirical results yet — must generate for poster. |
| **Impact** | **8/10** | Directly improves libkdl's dispatch quality. Applicable beyond libkdl to any multi-variant selection system. |
| **Composite** | **8.25/10** | |

---

## Limitations

- [LIMITATION] NeuSight AMD transfer accuracy unvalidated; all cross-vendor ML prediction claims must be qualified
- [LIMITATION] Roofline model assumes compute-bound or memory-bound regime; transition-region kernels (around ridge point) may have higher selection error
- [LIMITATION] Arithmetic intensity of arbitrary kernel variants requires either MLIR analysis or offline measurement; not always available
