<!--
  HOW TO POST:
  1. Go to https://discourse.llvm.org/c/runtimes/
  2. Click "New Topic"
  3. Title: [RFC] Standard capability metadata keys for OffloadBinary
  4. Category: Runtimes
  5. Paste everything below this comment block
-->

Hi all, I'm preparing a poster for EuroLLVM Dublin on runtime variant selection for GPU offloading. This RFC proposes a standard metadata vocabulary for OffloadBinary that the community has discussed but never formalized.

---

# [RFC] Standard capability metadata keys for OffloadBinary

## Summary

OffloadBinary (D122069, 2022) carries fat binaries across GPU targets but standardizes only two metadata keys: `triple` and `arch`. Four years later, these remain the only standard keys. A consumer hook (`isMetadataCompatible()`, PR #185663, merged March 2026) now exists but has no vocabulary to act on. This RFC proposes five standardized keys to enable runtime selection beyond triple/arch matching.

**CC:** @jhuber6 @jdenny-ornl @fabianmcg Yury Plyakhin (yury) Saiyedul Islam (saiyedul)

---

## The Gap

OffloadBinary's string-map was designed to be extensible, but no vocabulary was defined. A `feature=` key was prototyped in D127686 for LTO but never standardized. The `kernel-resource-usage` pass (D123878) extracts register/occupancy at compile time; PR #185663 added the consumer hook. The consumer exists; the vocabulary does not.

---

## Proposal: 5 Standard Keys

**Tier 1 — Runtime Must-Check (if present):**

| Key | Type | Example | Semantics |
|-----|------|---------|-----------|
| `min_sm` | uint | `80` | Min CUDA compute capability. Reject if device SM < value. |
| `min_gfx` | family-tagged arch | `gfx90a:cdna2` | Min AMD GFX target within ISA family. Format: `<gfx_target>:<family_tag>`. Valid family tags: `cdna1`, `cdna2`, `cdna3`, `rdna1`, `rdna2`, `rdna3`. Comparison valid only within the same family; cross-family images are rejected at the `triple`/`arch` level. Family tag is required. |
| `requires_features` | comma-list | `tensor_core_nv,bf16` | Per-vendor capability tokens. Reject if any token absent. Vendor-specific tokens: `tensor_core_nv` (NVIDIA Tensor Cores), `mfma_amd` (AMD Matrix Fused Multiply-Add), `bf16` (bfloat16 support). Vendor-neutral tokens (e.g., `tensor_core`) reserved for future use pending formal cross-vendor capability equivalence. |

**Tier 2 — Runtime May-Use (ranking):**

*(A resource-usage tier covering keys like `sgpr_count` and `vgpr_count` is deferred to a follow-up RFC.)*

| Key | Type | Example | Semantics |
|-----|------|---------|-----------|
| `variant_priority` | uint | `10` | Higher = preferred when multiple images satisfy Tier 1 requirements. |
| `variant_tag` | string | `optimized` | Label: `generic`, `optimized`, `fallback`, `debug`. |

---

## Composition with `isMetadataCompatible()`

Current (PR #185663):
```cpp
bool isMetadataCompatible(image, device) {
  return image.Triple == device.Triple &&
         areArchsCompatible(image.Arch, device.Arch);
}
```
*(Pseudo-code simplified for clarity; actual `isMetadataCompatible` is a virtual method on the plugin interface.)*

Extended (opt-in when keys present):
```cpp
bool isMetadataCompatible(image, device) {
  if (image.Triple != device.Triple) return false;
  if (!areArchsCompatible(image.Arch, device.Arch)) return false;
  if (auto minSm = image.getString("min_sm"))
    if (parseSmFromArch(device.Arch) < std::stoul(*minSm)) return false;
  if (auto features = image.getString("requires_features"))
    for (auto &tok : split(*features, ','))
      if (!device.hasCapability(tok)) return false;
  return true;
}
```

**Backward compat:** Missing keys = no constraint; old runtimes ignore unknown keys. Additive string-key extension -- no format version bump, no ABI break. Images compiled without the new keys retain current behavior; keys are opt-in for producers.

**Note on `device.hasCapability()`:** This method requires a capability query API that does not yet exist in liboffload. Initial implementation hardcodes known capabilities per arch string. A follow-up patch will propose the formal query API.

---

## Implementation Plan

Five patches: (1) header constants in `OffloadBinary.h`, (2) `isMetadataCompatible()` extension, (3) AMDGPU writer, (4) NVPTX writer, (5) `llvm-offload-binary --annotate`. Steps 2--5 are independent after step 1.

---

## Questions for Community

1. Should the initial patch include only Tier 1 keys (`min_sm`/`min_gfx`/`requires_features`), or also the Tier 2 ranking keys (`variant_priority`, `variant_tag`)?
2. Should `requires_features` reserve vendor-neutral tokens (e.g., `tensor_core`) now, or define them only when a formal cross-vendor equivalence exists?
3. Should `min_gfx` family-tag syntax (`gfx90a:cdna2`) be a separate metadata key rather than an embedded colon-delimited format?
