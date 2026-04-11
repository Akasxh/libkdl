# KernelEvolve: Adaptive Kernel Generation for Heterogeneous Hardware

**Source:** arXiv:2512.23236 (ISCA 2026)
**Type:** Industry paper (Meta Platforms)
**Added:** 2026-04-10

## Summary
Agentic kernel generation system deployed at Meta for production ML inference. Uses search-based optimization to generate kernels targeting NVIDIA, AMD, MTIA, and CPU simultaneously. Achieves 60% inference throughput improvement in production deployments.

## Key Technical Details
- Multi-target kernel generation through iterative search-based refinement
- Targets 4 hardware families simultaneously
- Production deployment at Meta scale
- Design-time system — generates variants, does not address runtime selection

## Relevance to Our Work (3-axis score)
- **Direct relevance:** 3/5 — validates multi-target kernel need but different pipeline stage
- **Novelty differentiation:** 5/5 — complementary (generation vs dispatch)
- **Citation priority:** HIGH — strongest industry validation of the problem space

## Differentiation
KernelEvolve is upstream in the pipeline: it creates optimized kernel variants for multiple targets. Our work is downstream: given a fat binary with N target-specific kernels, select the right one at runtime. The two compose naturally.

## Key Insight
Meta investing in multi-target kernel generation at this scale validates that heterogeneous GPU deployment is a real production need, not just an academic concern.
