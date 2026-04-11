# RFC Review: Standard capability metadata keys for OffloadBinary

**File Reviewed:** `proposals/rfc-metadata-vocabulary.md`
**Reviewed:** 2026-04-09
**Total Issues:** 10 (1 CRITICAL, 3 HIGH, 4 MEDIUM, 2 LOW)

---

## Severity Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 3 |
| MEDIUM | 4 |
| LOW | 2 |

---

## Issues

### [CRITICAL] Tier numbering gap: Tier 1 jumps to Tier 3, Tier 2 is missing

**File:** `rfc-metadata-vocabulary.md:30,38`

The RFC labels the two tiers as "Tier 1" and "Tier 3" with no explanation of what happened to Tier 2. Proposal-v2 explicitly fixed this — line 490 states `"Tier 3" renamed to "Tier 2". Footnote added.` But this fix was never backported to the RFC draft. An LLVM Discourse reader seeing "Tier 1... Tier 3" will immediately wonder if Tier 2 was accidentally deleted or the author cannot count. This is the single most damaging credibility issue for a standards-vocabulary RFC.

**Fix:** Rename "Tier 3" to "Tier 2" on line 38. Add footnote: *"Tier numbering follows the full vocabulary design; Tier 2 (resource-usage keys: `sgpr_count`, `vgpr_count`) is deferred to a follow-up patch."*

---

### [HIGH] Word count exceeds the 400-word target: 549 words

**File:** `rfc-metadata-vocabulary.md` (entire file)

The task requirement is "under 400 words." The file is 549 words. LLVM Discourse RFCs that are concise get more engagement; long posts lose busy maintainers. The RFC has redundant phrasing in The Gap section and the Upstream Patch Sequence table adds ~80 words that could be deferred to the patch series.

**Fix:** Cut The Gap section to 2 bullets instead of 4 (HSACO/cubin details belong in comments/discussion). Target 350–380 words.

---

### [HIGH] `min_gfx` comparison semantics are underspecified

**File:** `rfc-metadata-vocabulary.md:35`

The RFC says `gfx90a:cdna2` and "Family tag prevents cross-family comparison" but does not define the actual ordering within a family. What does "min" mean? Is `gfx90a < gfx942`? Is the comparison lexicographic on the numeric suffix? Any LLVM reviewer will ask "how does the runtime compare two GFX strings?" and the RFC has no answer.

**Fix:** Add: "Format: `<gfx_target>:<family_tag>`. Family tags: `{cdna1, cdna2, cdna3, rdna1, rdna2, rdna3}`. Comparison valid only within same family."

---

### [HIGH] D127686 is referenced nowhere in the RFC but should be

**File:** `rfc-metadata-vocabulary.md:9` (The Gap section)

The RFC's narrative is "the vocabulary never grew in four years." But proposal-v2 documents that D127686 *did* try to add a `feature=` key for LTO and failed to standardize. This is the strongest evidence for why a formal RFC is needed — a prior attempt existed and stalled. Omitting D127686 makes the "no one tried" claim weaker and leaves the RFC vulnerable to "did you know about D127686?" questions in the thread.

**Fix:** Add to The Gap section: "A third key (`feature=`) was prototyped in D127686 for LTO but never standardized."

---

### [MEDIUM] CC list is missing Discourse `@` handles for two reviewers

**File:** `rfc-metadata-vocabulary.md:11`

"Yury Plyakhin" and "Saiyedul Islam" without `@` handles will not ping them on Discourse. Also, Fabian Mora (RFC #88170 author) is absent from the CC list entirely.

**Fix:** Look up Discourse handles for Yury Plyakhin, Saiyedul Islam, and Fabian Mora before posting. Add `@fabianmcg` if that is his handle.

---

### [MEDIUM] `isMetadataCompatible()` pseudo-code assumes APIs that do not exist

**File:** `rfc-metadata-vocabulary.md:62-66`

The extended code calls `device.hasCapability(tok)`, which implies a device capability query API that does not exist anywhere in liboffload or the offload plugins. The RFC hand-waves the hardest implementation question with a function signature.

**Fix:** Add note: "`device.hasCapability()` requires a capability query API (follow-up). Initial implementation hardcodes known capabilities per arch string."

---

### [MEDIUM] `isMetadataCompatible()` signature does not match actual PR #185663 code

**File:** `rfc-metadata-vocabulary.md:51-54`

The "Current" pseudo-code shows a free function `bool isMetadataCompatible(image, device)`. Per expert verification, PR #185663 introduced it as a virtual method in `PluginInterface.cpp`, not a standalone function. Any reviewer of PR #185663 will notice.

**Fix:** Add note: "Pseudo-code simplified for clarity; actual `isMetadataCompatible` is a virtual method on the plugin interface."

---

### [MEDIUM] `min_gfx` syntax has no formal grammar

**File:** `rfc-metadata-vocabulary.md:35`

The colon-separated `arch:family` format is proposed without defining: is the family tag required or optional? What are valid family names? Case sensitivity?

**Fix:** Covered by the `min_gfx` ordering fix above — add explicit format definition and valid tag enumeration.

---

### [LOW] "Questions for Community" section could be stronger

**File:** `rfc-metadata-vocabulary.md:91-93`

Q1 ("Are these five keys sufficient?") is too open-ended. Missing: a question about Tier 1 vs Tier 2 inclusion scope, and where constants should live (`OffloadBinary.h` vs new `OffloadMetadata.h`).

**Fix (optional):** Replace Q1 with: "Should the initial patch include only Tier 1 keys (`min_sm`/`min_gfx`/`requires_features`), or also the ranking keys (`variant_priority`, `variant_tag`)?"

---

### [LOW] Placeholder link at end of document

**File:** `rfc-metadata-vocabulary.md:95`

`Discourse thread: [Link to RFC discussion]` is a placeholder that should not appear in the posted RFC.

**Fix:** Remove the line entirely.

---

## PR Number Verification

| Reference | Claimed Context | Verified | Notes |
|-----------|----------------|----------|-------|
| PR #185663 | `isMetadataCompatible()` hook, merged March 2026 | CORRECT | Confirmed merged March 10, 2026 |
| PR #186088 | Generalize OffloadBinary images | CORRECT | Open as of April 9, 2026 (not referenced in RFC, which is fine) |
| D122069 | OffloadBinary original, 2022 | CORRECT | Consistent across all project files |
| D127686 | Feature key prototype for LTO | NOT IN RFC | Should be added — see HIGH issue above |
| D123878 | `kernel-resource-usage` pass, Joel Denny | CORRECT | Confirmed in proposal-v2 and expert verification |

---

## Backward Compatibility Assessment

The claim is **technically correct but incompletely argued**:

1. Missing keys = no constraint — correct. The `if (auto minSm = ...)` pattern passes old images unchanged.
2. Old runtimes ignore unknown keys — correct. The string-map stores arbitrary key-value pairs.
3. No ABI break — correct. Wire format is additive.

**Gap in the argument:** The RFC should state explicitly: "Images compiled without the new keys retain current behavior; the keys are opt-in for producers. Only explicitly annotated images are subject to the new constraints."

---

## Tone Assessment

**Appropriate** overall with two adjustments:

- Line 9: "Four years later, despite..." reads slightly editorial. Change to: "Four years later, these remain the only standard keys, though a consumer hook (`isMetadataCompatible()`) now exists."
- Line 17: "Why two keys for four years?" is rhetorically confrontational. Reframe as: "The string-map design was intentionally extensible, but no standard vocabulary beyond triple/arch has been defined."

---

## Positive Observations

- Correctly identifies a real, well-evidenced gap.
- Backward-compatibility argument is structurally sound. The opt-in pattern is correct.
- 5-step upstream patch sequence is well-scoped with credible review targets.
- Separation of must-check keys from ranking keys matches how runtime selection works.
- CC list targets the right people — liboffload maintainer and KernelInfo pass author.

---

## Recommendation

**REQUEST CHANGES**

The Tier 1/Tier 3 numbering gap alone would undermine the proposal's credibility. Fix CRITICAL and HIGH issues before posting. After those fixes, the RFC is strong enough for Discourse — the metadata vocabulary is the most defensible of the Combo A contributions, and PR #185663 just merged the consumer hook, so the community is primed for the vocabulary discussion.
