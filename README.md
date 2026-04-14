# libkdl — Kernel Dynamic Linker

**Bridging Runtime Gaps in LLVM: Vendor-Agnostic Dispatch for ML Kernels**

*EuroLLVM Developers' Meeting, Dublin 2026 — Poster Session*

## Overview

MLIR can compile a single `gpu.module` to multiple GPU vendors (NVIDIA, AMD, Intel) and pack them into one `OffloadBinary`. But at runtime, the offload stack picks the **first compatible image** and stops — there is no metadata vocabulary, no published measurement, and no "best-compatible" selection mechanism.

This project addresses three gaps:

1. **OffloadBinary Metadata Vocabulary** — 5 new standard keys (`min_sm`, `min_gfx`, `requires_features`, `variant_priority`, `variant_tag`) extending `isMetadataCompatible()`
2. **First Dispatch Stack Flame Graph** — per-layer latency decomposition of the LLVM GPU dispatch path, measured on real hardware
3. **`#gpu.runtime_select` Design** — an MLIR attribute that defers binary selection to runtime, inspired by CPU Function Multi-Versioning (IFunc)

## Key Results

| Metric | Value |
|--------|-------|
| Selection overhead | **3–6 ns** per dispatch |
| Cold module load (`cuModuleLoadData`) | 36.0 µs (90% of cold path) |
| Hot-path total (launch + sync) | 4.1 µs |
| Overhead vs 10ms ML kernel | < 0.0001% |
| Prototype LOC | 5,100 (libkdl) + 664 (PoC) |

> At 3–6 ns per dispatch, selection overhead is faster than a single L2 cache access.

## Repository Structure

```
├── poster/                     # Conference poster (A0 HTML, slides)
│   ├── poster-combo-a.html     # Main poster — open in browser, print to PDF
│   ├── slides.tex              # Beamer slides
│   └── slides.pdf
├── experiments/
│   └── prototype/
│       ├── src/
│       │   ├── kdl.c / kdl.h           # libkdl runtime library
│       │   ├── runtime_select_poc.c    # PoC: real OffloadBinary dispatch
│       │   ├── bench_layers.c          # Per-layer latency benchmark
│       │   ├── RuntimeSelectAttr.cpp.sketch  # MLIR attribute design
│       │   └── Makefile
│       ├── benchmarks/                 # Benchmark drivers + plotting
│       └── results/                    # Benchmark figures
├── research/
│   ├── combo-a-deep-dive/
│   │   ├── proposals/                  # RFCs, extended abstract, Q&A cards
│   │   ├── research/                   # Benchmark data, statistical analysis
│   │   └── critiques/                  # Review feedback
│   └── mega-survey/                    # Literature survey (~450 sources)
├── literature/                         # 40+ annotated paper summaries
└── findings.md                         # Core research findings
```

## Building the Prototype

```bash
cd experiments/prototype/src
make                    # Builds libkdl.so + all benchmarks
./runtime_select_poc    # Run the PoC dispatcher
./bench_layers          # Run per-layer latency benchmark
```

**Requirements:** CUDA Toolkit 12+, GCC/Clang, Linux (tested on GTX 1650 sm_75, CUDA 13.1)

## Viewing the Poster

```bash
# Open in browser
xdg-open poster/poster-combo-a.html

# Print to PDF (A0)
# Chrome → Ctrl+P → Paper: Custom 841×1189mm → Margins: None → Background graphics: ON
```

## Upstream Path

1. **Metadata RFC** on [discourse.llvm.org](https://discourse.llvm.org) — 5 keys, ~30 LOC header patch
2. **Flame graph benchmark** in `llvm-test-suite`
3. **`#gpu.runtime_select` RFC** — ~780 LOC, implements `OffloadingLLVMTranslationAttrInterface`

## Related Work

| System | Runtime Select? | Cross-Vendor? | MLIR-Native? | Ranked? |
|--------|:-:|:-:|:-:|:-:|
| IREE HAL | Yes | Yes | Yes | Partial |
| chipStar | Yes (SPIR-V) | Yes | No | No |
| Proteus (LLNL) | Yes (JIT) | Partial | No | No |
| HetGPU | Yes (IR translate) | Yes | No | No |
| liboffload #186088 | Yes | Yes | No | No (first-wins) |
| CPU FMV (target_clones) | Yes | N/A | No | Yes (IFunc) |
| **This Work** | **Metadata + Measurement + Design** | **Yes** | **Yes** | **Yes** |

## References

- IREE HAL — [iree.dev](https://iree.dev)
- chipStar — [github.com/CHIP-SPV](https://github.com/CHIP-SPV)
- Proteus — CGO 2025
- HetGPU — [arXiv:2506.15993](https://arxiv.org/abs/2506.15993)
- Universal GPU ISA — [arXiv:2603.28793](https://arxiv.org/abs/2603.28793)
- KernelEvolve — ISCA 2026, [arXiv:2512.23236](https://arxiv.org/abs/2512.23236)
- AdaptiveCpp — IWOCL 2025

## Author

**S. Akash** — IIT Patna | CERN GSoC | vLLM contributor

## License

Research project — see individual files for applicable licenses.
