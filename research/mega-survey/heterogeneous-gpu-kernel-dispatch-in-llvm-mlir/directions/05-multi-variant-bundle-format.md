# Direction 05: Multi-Variant Kernel Bundle Format (MTB) Aligned with OffloadBinary

**Composite Score: 7.75/10**
**Rank: 5 of 8**

---

## Title

**MTB: A Cross-Vendor Kernel Archive Format Extending LLVM OffloadBinary with Capability Contracts and Dispatch Metadata**

## One-Sentence Description

A standardized kernel archive format that carries CUBIN + HSACO + SPIR-V + CPU ELF + roofline metadata + Bloom filters for variant elimination in a single file, designed as a superset of LLVM's OffloadBinary container.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| OffloadBinary v2 format (OffloadBinary.h) | wave-06-llvm-offload-new-driver | Extensible StringMap metadata; MTB capability contracts map to existing key-value pairs |
| CUDA fatbin (nvFatbin API) | wave-06-kernel-binary-abi | NVIDIA-only multi-arch container; programmatic construction via nvFatbinAddCubin/PTX/LTOIR |
| AOTriton AKS2 | wave-02-triton-multibackend, wave-04-kernel-caching | AMD-only LZMA-compressed HSACO archive with SQLite autotuning DB |
| clang-offload-bundler v3 | wave-06-rocm-code-objects | Compressed per-entry payloads (zlib/zstd); under-documented format detail |
| CUDA sm_90a cache key requirements | wave-06-kernel-binary-abi | Cache key must include arch-accelerated variant flag; PTX forward-compat broken |
| AMD xnack/sramecc feature flags | wave-06-kernel-binary-abi | V4+ relaxed constraints; V2/V3 require exact match — cache key must differ |
| AMD .amdhsa.kd section transition | wave-06-rocm-code-objects | PR #122930 stalled; parser must handle descriptors in both .rodata and .amdhsa.kd |
| ExecuTorch separate .pte files | wave-04-executorch-edge-dispatch | 14 backends = 14 separate deployment artifacts; motivates single-file format |
| Cross-framework cache comparison | wave-04-kernel-caching | No cross-vendor archive exists; every framework invented its own cache |
| Stream-K++ Bloom filter | wave-04-cost-models | 95.8% variant pruning in <100 ns; embeddable in MTB header |

---

## Novelty Argument

No cross-vendor kernel archive format exists:
- CUDA fatbin: NVIDIA-only (CUBIN + PTX)
- AMD AKS2: AMD-only (per-gfx HSACO + LZMA + SQLite)
- clang-offload-bundler: vendor-neutral syntax but single-vendor per entry, no capability metadata
- OffloadBinary: multi-image capable but no dispatch metadata (no roofline parameters, no Bloom filters, no autotuning DB)

MTB extends OffloadBinary with three additions:
1. **Capability contracts**: per-variant key-value metadata (min SM, xnack requirement, required extensions) stored as OffloadBinary StringMap entries — zero format changes needed
2. **Bloom filter index**: pre-computed filter for fast variant elimination at dispatch time
3. **Roofline metadata**: per-variant arithmetic intensity and estimated flops for cost-model scoring

The recommended approach: MTB is OffloadBinary with a richer metadata schema, not a competing format. libkdl auto-detects OffloadBinary (magic 0x10FF10AD) and MTB-extended OffloadBinary.

---

## Feasibility Plan

1. Define MTB as OffloadBinary + extended StringMap keys (e.g., `kdl.min_sm`, `kdl.arithmetic_intensity`, `kdl.bloom_filter`)
2. Implement `kdl_pack` tool that takes N vendor binaries + metadata YAML and produces an OffloadBinary-compatible MTB
3. Implement `kdl_info` tool that dumps MTB contents (like `llvm-offload-binary` but with capability metadata)
4. Demonstrate round-trip: LLVM toolchain → OffloadBinary → kdl_pack enrichment → libkdl consumption

---

## Poster Potential

**Moderate — best as a supporting element of Direction 01, not standalone.**

- Format diagram showing OffloadBinary header + MTB extensions
- Side-by-side: CUDA fatbin (NVIDIA-only) | AKS2 (AMD-only) | MTB (cross-vendor)
- Binary size comparison for a GEMM kernel: single SPIR-V vs. CUBIN + HSACO + CPU

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **7/10** | OffloadBinary already carries multi-image; MTB adds richer metadata but format novelty reduced. |
| **Feasibility** | **8/10** | StringMap extension requires no upstream format changes. kdl_pack tool is straightforward. |
| **Evidence** | **8/10** | Every framework's independent cache validates the need. ExecuTorch's 14 .pte files quantifies the distribution problem. |
| **Impact** | **8/10** | Enables single-artifact deployment across vendors. Complements liboffload ecosystem. |
| **Composite** | **7.75/10** | |
