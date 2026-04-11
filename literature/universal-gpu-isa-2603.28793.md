# Toward a Universal GPU ISA: Cross-Vendor Analysis of Hardware-Invariant Primitives

**Source:** arXiv:2603.28793 (March 2026)
**Type:** Research paper
**Added:** 2026-04-10

## Summary
First systematic cross-vendor analysis of GPU instruction set architectures spanning all four major GPU vendors: NVIDIA (PTX ISA v1.0–v9.2, Fermi–Blackwell), AMD (RDNA 1–4, CDNA 1–4), Intel (Gen11–Xe-HPC), and Apple (G13, reverse-engineered).

## Key Findings
- **10 hardware-invariant primitives** appearing across all 4 vendors
- **6 parameterizable dialects** — identical concepts with different vendor parameters
- **6 true architectural divergences** — fundamental design disagreements between vendors
- Abstract model matches or exceeds native performance on 5/6 benchmark-platform pairs (NVIDIA T4, Apple M1)
- SPIR-V could distribute programs targeting the universal ISA

## Relevance to Our Work (3-axis score)
- **Direct relevance:** 4/5 — validates capability contract design
- **Novelty differentiation:** 4/5 — theoretical ISA unification vs practical dispatch
- **Citation priority:** HIGH — strongest academic validation of our approach

## How It Validates Our Design
- Our `requires_features` metadata maps to their "parameterizable dialects"
- The 6 true divergences confirm vendor-specific optimized kernels will outperform universal ones
- Our dispatch layer bridges the gap between portable fallback and vendor-optimized variants

## Key Quote
"The distinction is that existing standards specify how to talk to a GPU; the universal ISA specifies what a GPU is."
