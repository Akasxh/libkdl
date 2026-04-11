# Adaptivity in AdaptiveCpp: Optimizing via Runtime Information During JIT-Compilation

**Source:** IWOCL 2025 (ACM DL: 10.1145/3731125.3731127)
**Type:** Conference paper
**Added:** 2026-04-10

## Summary
Presents AdaptiveCpp's SSCP (Single-Source, Single Compilation Pass) mode that JIT-specializes kernels at first launch using runtime information: work-group sizes, pointer alignments, kernel argument values. Achieves near-native performance after initial JIT cost (~15% first-launch overhead, near-zero cached).

## Key Technical Details
- JIT specialization from single SYCL source
- Uses runtime information not available at compile time
- Near-native performance after first execution (caching)
- Requires SYCL programming model adoption

## Relevance to Our Work (3-axis score)
- **Direct relevance:** 3/5 — different approach to same multi-target problem
- **Novelty differentiation:** 5/5 — JIT from source vs AOT variant selection
- **Citation priority:** MEDIUM — important for completeness

## Differentiation
| Aspect | AdaptiveCpp SSCP | Our Work |
|--------|-----------------|----------|
| Mechanism | JIT specialization | AOT dispatch |
| First-launch cost | ~15% overhead | 3 ns |
| Programming model | SYCL required | None (OffloadBinary) |
| Source requirement | Runtime SYCL source | Pre-compiled binaries |
| MLIR integration | None | Native |
