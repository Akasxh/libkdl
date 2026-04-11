# HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs

**Source:** arXiv:2506.15993 (June 2025)
**Type:** Research paper
**Added:** 2026-04-10

## Summary
HetGPU is a system (compiler + runtime + abstraction layer) enabling a single GPU binary to execute on NVIDIA, AMD, Intel, and Tenstorrent hardware. The compiler emits architecture-agnostic GPU IR with execution state metadata; the runtime dynamically translates this IR to native code on the target GPU.

## Key Technical Details
- Addresses divergent SIMT (NVIDIA/AMD) vs MIMD (Tenstorrent RISC-V) execution models
- Handles varied instruction sets, scheduling models, and memory hierarchies
- Supports state serialization for live GPU migration across vendors
- Preliminary evaluation shows minimal overhead for unmodified binary migration

## Relevance to Our Work (3-axis score)
- **Direct relevance:** 4/5 — closest prior art for cross-vendor GPU binary portability
- **Novelty differentiation:** 5/5 — fundamentally different mechanism (translation vs selection)
- **Citation priority:** HIGH — must cite and explicitly differentiate

## Differentiation
| Aspect | HetGPU | Our Work |
|--------|--------|----------|
| Mechanism | Runtime IR translation | AOT variant selection |
| JIT cost | Yes (translation overhead) | No (3 ns lookup) |
| Peak perf | Limited by translation | Native per-vendor |
| Programming model | Transparent | Transparent |
| MLIR integration | None | Native (OffloadBinary) |

## Key Quote
"Unmodified GPU binaries compiled with hetGPU can be migrated across disparate GPUs with minimal overhead."
