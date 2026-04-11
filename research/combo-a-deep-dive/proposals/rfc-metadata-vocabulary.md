# [RFC] Standard capability metadata keys for OffloadBinary

---

## Summary

OffloadBinary (D122069, 2022) carries fat binaries across GPU targets but standardizes only two metadata keys: `triple` and `arch`. Four years later, these remain the only standard keys — a consumer hook (`isMetadataCompatible()`, PR #185663, merged March 2026) now exists but has no vocabulary to act on. This RFC proposes five standardized keys to enable runtime selection beyond triple/arch matching.

**CC:** @jhuber6 @jdenny-ornl Yury Plyakhin Saiyedul Islam

---

## The Gap

OffloadBinary's string-map was designed to be extensible, but no vocabulary was defined. A `feature=` key was prototyped in D127686 for LTO but never standardized. The `kernel-resource-usage` pass (D123878) extracts register/occupancy at compile time; PR #185663 added the consumer hook — but there is nothing to consume. This metadata is produced and then dropped.

---

## Proposal: 5 Standard Keys

**Tier 1 — Runtime Must-Check (if present):**

| Key | Type | Example | Semantics |
|-----|------|---------|-----------|
| `min_sm` | uint | `80` | Min CUDA compute capability. Reject if device SM < value. |
| `min_gfx` | family-tagged arch | `gfx90a:cdna2` | Min AMD GFX within ISA family. Format: `<gfx_target>:<family_tag>`; tags: `{cdna1,cdna2,cdna3,rdna1,rdna2,rdna3}`. Comparison valid only within same family. |
| `requires_features` | comma-list | `tensor_core_nv,bf16` | Per-vendor capability tokens. Reject if any token absent. Vendor-neutral tokens reserved for future equivalence mapping. |

**Tier 2 — Runtime May-Use (ranking):**[^tier]

| Key | Type | Example | Semantics |
|-----|------|---------|-----------|
| `variant_priority` | uint | `10` | Higher = preferred when multiple images satisfy requirements. |
| `variant_tag` | string | `optimized` | Label: `generic`, `optimized`, `fallback`, `debug`. |

[^tier]: Tier numbering follows the full vocabulary design; Tier 2 (resource-usage keys) is deferred.

---

## Composition with `isMetadataCompatible()`

Current (PR #185663):
```cpp
bool isMetadataCompatible(image, device) {
  return image.Triple == device.Triple &&
         areArchsCompatible(image.Arch, device.Arch);
}
```
*(Simplified; actual signature is a virtual method on the plugin interface.)*

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

**Backward compat:** Missing keys = no constraint; old runtimes ignore unknown keys. Additive — no ABI break. `device.hasCapability()` requires a follow-up capability query API; initial implementation hardcodes known capabilities per arch string.

---

## Implementation Plan

Five patches: (1) header constants, (2) `isMetadataCompatible()` extension, (3) AMDGPU writer, (4) NVPTX writer, (5) `llvm-offload-binary --annotate`. Steps 2–5 are independent after step 1.

---

## Questions for Community

1. Are these five keys sufficient for your variant-selection use cases?
2. Should `requires_features` reserve vendor-neutral tokens (e.g., `tensor_core`) for future equivalence mapping, or use per-vendor tokens only?
3. Should `min_gfx` family-tag syntax (`gfx90a:cdna2`) be separate metadata?

