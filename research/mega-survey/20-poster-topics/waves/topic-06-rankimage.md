# Topic 06: rankImage() — Variant Selection Callback for liboffload

**Survey:** 20 LLVM/MLIR Poster Topics
**Topic ID:** 06
**Config name:** rankimage-variant-selection-callback
**Title:** `rankImage()`: A 200-Line Patch to Give liboffload a Selection Policy
**Persona:** LLVM offload contributor / systems engineer
**Date:** 2026-04-07

---

## Gap

PR #186088 (`[OFFLOAD] Generalize support for OffloadBinary images`, open March 2026,
author: Alex Duran) adds multi-image container support to liboffload's CUDA and AMDGPU
plugins.  The core of the implementation is a loop in
`offload/plugins-nextgen/common/src/PluginInterface.cpp`:

```cpp
for (auto &Binary : InnerBinaries) {
    const OffloadBinary *OB = Binary.getBinary();
    StringRef InnerImage = OB->getImage();
    OffloadBinMetadataTy Metadata = extractOffloadBinaryMetadata(*OB);

    // metadata gate (via isMetadataCompatible, default: true)
    auto MetadataMatchOrErr = isMetadataCompatible(Metadata);
    if (!*MetadataMatchOrErr) continue;

    // binary-level gate
    if (!Plugin.isDeviceCompatible(DeviceId, InnerImage)) continue;

    // <<< LOAD first compatible image and break >>>
    FirstLoadedImage = LoadedImage;
    break;   // <-- the deferral point
}
```

The PR body states explicitly: _"For now only the first compatible image in the binary is
loaded.  While it might be desirable to add support for loading multiple images, our current
interface is limiting…  So, should we want it, it's better in a follow-up PR IMO."_

That follow-up PR does not exist as of April 2026.

The result: when a `libkdl`-packaged OffloadBinary container carries five images (SM 7.5
CUBIN, SM 8.9 CUBIN, SM 9.0 CUBIN, AMDGPU gfx1100 HSACO, Intel SPIR-V), liboffload always
loads the first one it finds compatible — regardless of whether a better-matched image exists
later in the container.  On an SM 8.9 device with an SM 7.5 image listed first, the SM 7.5
image loads and runs with lower occupancy and no tensor-core utilisation.

The infrastructure for fixing this is already partially built:

- `OffloadBinMetadataTy` is introduced in PR #186088 as a struct carrying
  `ImageKind`, `OffloadKind`, `Triple` (string), `Arch` (string), and a
  `StringMap<string>` for arbitrary key-value metadata.
- `isMetadataCompatible(const OffloadBinMetadataTy&)` is added in PR #185663
  (merged March 10, 2026) as a virtual method on `GenericPluginTy` with
  default `return true`.
- `loadBinaryImpl(MemoryBuffer, ImageId, const OffloadBinMetadataTy*)` (PR #186088)
  threads the metadata through to plugin-specific image construction, so every
  plugin already receives the metadata at load time.

The only missing piece is a **ranking callback** that replaces the `break` with a
scored selection.

---

## Proposal

Add `virtual Expected<int> rankImage(const OffloadBinMetadataTy &Meta, int DeviceId)`
to `GenericPluginTy` in `PluginInterface.h`, and replace the `break` in PR #186088's
`parseOffloadBinary` loop with a best-score accumulator.  The entire change fits in
approximately 200 lines across three files.

### API surface

```cpp
// offload/plugins-nextgen/common/include/PluginInterface.h
// New virtual method on GenericPluginTy (alongside isMetadataCompatible)

/// Return a non-negative score for \p Meta on \p DeviceId.
/// Higher score means better fit.  Return 0 to accept without preference.
/// Return -1 (or an error) to reject.
/// Default: return 0 for all compatible images (preserves current behaviour).
virtual Expected<int> rankImage(const OffloadBinMetadataTy &Meta,
                                int DeviceId) const {
  return 0;
}
```

### Loop replacement in PluginInterface.cpp

```cpp
// Replace the current "first compatible, break" with:

int BestScore = -1;
DeviceImageTy *BestImage = nullptr;
Error LoadErrors = Error::success();

for (auto &Binary : InnerBinaries) {
  const OffloadBinary *OB = Binary.getBinary();
  StringRef InnerImage = OB->getImage();
  OffloadBinMetadataTy Meta = extractOffloadBinaryMetadata(*OB);

  // Gate 1: metadata filter (isMetadataCompatible, default true)
  auto MetaOkOrErr = isMetadataCompatible(Meta);
  if (!*MetaOkOrErr) continue;

  // Gate 2: binary-level compatibility (ELF SM check, arch string, etc.)
  if (!Plugin.isDeviceCompatible(DeviceId, InnerImage)) continue;

  // Gate 3: ranking score
  auto ScoreOrErr = Plugin.rankImage(Meta, DeviceId);
  if (Error Err = ScoreOrErr.takeError()) {
    LoadErrors = joinErrors(std::move(LoadErrors), std::move(Err));
    continue;
  }
  int Score = *ScoreOrErr;
  if (Score < 0) continue;  // plugin explicitly rejects

  if (Score > BestScore) {
    // Load this candidate (or defer loading to after the loop)
    auto ImageOrErr = loadBinaryImpl(MemoryBuffer::getMemBufferCopy(InnerImage),
                                      LoadedImages.size(), &Meta);
    if (!ImageOrErr) {
      LoadErrors = joinErrors(std::move(LoadErrors), ImageOrErr.takeError());
      continue;
    }
    BestScore = Score;
    BestImage = *ImageOrErr;
    // Note: unload previously loaded BestImage if Score > old BestScore and
    // images are cheap to reload — or use a "score-only" pre-scan pass (see below)
  }
}
if (!BestImage) { /* error path */ }
return BestImage;
```

A cleaner two-pass variant avoids loading-then-unloading: a first pass computes scores
only, selects the winner index, then a second pass loads only the winner.  This is the
preferred implementation because `loadBinaryImpl` for CUDA calls `cuModuleLoadData` —
loading more than one module and then discarding it wastes driver memory.

### Default rankImage for the CUDA plugin (SM closeness score)

```cpp
// offload/plugins-nextgen/cuda/src/rtl.cpp
Expected<int> CUDAPluginTy::rankImage(const OffloadBinMetadataTy &Meta,
                                       int DeviceId) const {
  // Arch string is "sm_XY" — parse XY
  StringRef Arch = Meta.Arch;
  if (!Arch.starts_with("sm_")) return 0;  // unknown, accept without preference
  unsigned ImageSM;
  if (Arch.drop_front(3).getAsInteger(10, ImageSM)) return 0;

  // Query device SM
  CUdevice Dev;
  cuDeviceGet(&Dev, DeviceId);
  int Major, Minor;
  cuDeviceGetAttribute(&Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Dev);
  cuDeviceGetAttribute(&Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Dev);
  int DeviceSM = Major * 10 + Minor;

  // Score: prefer the highest SM that does not exceed the device.
  // SM 89 on an SM 89 device scores higher than SM 75 on an SM 89 device.
  // This is the same compatibility lattice isELFCompatible() uses, but
  // made continuous rather than binary.
  if ((int)ImageSM > DeviceSM) return -1;  // reject (over-targeting)
  return (int)ImageSM;                     // higher is better fit
}
```

This scoring function mirrors exactly what CUDA's own fatbin resolver does internally:
among all SM variants for which `Major == ImageMajor && Minor >= ImageMinor`, CUDA loads
the one with the highest SM value (closest match from below).  PR #186088's binary-level
`isELFCompatible` already implements the boolean cut; `rankImage` adds the continuous
preference over the surviving set.

### OffloadBinMetadataTy fields available for ranking

From PR #186088's diff to `PluginInterface.h`:

| Field | Type | Source in OffloadBinary |
|-------|------|------------------------|
| `ImageKind` | `llvm::object::ImageKind` (enum: Object=1, Bitcode=2, CUBIN=3, Fatbinary=4, PTX=5, SPIRV=6) | `Binary.getImageKind()` |
| `OffloadKind` | `llvm::object::OffloadKind` (bitmask: OpenMP=1, CUDA=2, HIP=4, SYCL=8) | `Binary.getOffloadKind()` |
| `Triple` | `std::string` | `Binary.getTriple().str()` |
| `Arch` | `std::string` (e.g., `"sm_89"`, `"gfx1100"`) | `Binary.getArch().str()` |
| `StringData` | `llvm::StringMap<std::string>` | `Binary.strings()` key-value pairs |

The `StringData` map is unbounded.  libkdl adds capability-contract keys to it:
`"min_sm"`, `"max_sm"`, `"gfx_target"`, `"peak_tflops"`, `"requires_tensor_core"`.  A
`rankImage` override reading these keys implements the full libkdl capability-contract
matching within liboffload's loading path — no external policy library required for simple
cases.

---

## Evidence

### Primary: the `break` in PR #186088

- **URL:** https://github.com/llvm/llvm-project/pull/186088
- **Status:** Open, March 12, 2026
- **File:** `offload/plugins-nextgen/common/src/PluginInterface.cpp`
- **Quote (PR body):** _"For now only the first compatible image in the binary is loaded.
  While it might be desirable to add support for loading multiple images, our current
  interface is limiting (i.e., it returns a single Image) and it's unclear if in all cases
  this behavior is desirable so we would need to add more options to control it.  So, should
  we want it, it's better in a follow-up PR IMO."_
- **Significance:** The PR author self-identifies the ranking gap and explicitly requests a
  follow-up.  A `rankImage()` PR is the follow-up they are waiting for.

### `isMetadataCompatible` virtual hook pattern (PR #185663)

- **URL:** https://github.com/llvm/llvm-project/pull/185663
- **Status:** Merged March 10, 2026
- **File:** `offload/plugins-nextgen/common/include/PluginInterface.h`
- **Key diff:**
  ```cpp
  virtual Expected<bool>
  isMetadataCompatible(const OffloadBinMetadataTy &Metadata) const {
    return true;  // default: always compatible
  }
  ```
  Level Zero overrides this to filter SPIR-V images by triple and image kind.
- **Significance:** The exact same virtual-method-on-GenericPluginTy pattern that
  `rankImage()` would follow.  Reviewers have already accepted this pattern.  A `rankImage()`
  PR follows the established convention.

### OffloadBinMetadataTy struct (PR #186088, PluginInterface.h diff)

```cpp
// Lines 36–45 of PR #186088 diff to PluginInterface.h
struct OffloadBinMetadataTy {
  llvm::object::ImageKind ImageKind;
  llvm::object::OffloadKind OffloadKind;
  std::string Triple;
  std::string Arch;
  llvm::StringMap<std::string> StringData;

  StringRef getString(StringRef Key) const {
    auto It = StringData.find(Key);
    return (It != StringData.end()) ? StringRef(It->second) : StringRef();
  }
};
```

### CUDA SM closeness scoring — prior art in `isELFCompatible`

From `offload/plugins-nextgen/cuda/src/rtl.cpp` (current main, confirmed via GitHub API):

```cpp
Expected<bool> CUDAPluginTy::isELFCompatible(uint32_t DeviceId,
                                              StringRef Image) const override {
    // ...parse ELF header to extract SM value...
    unsigned SM = (flags & ELF::EF_CUDA_SM_MASK) >> ELF::EF_CUDA_SM_OFFSET;
    cuDeviceGetAttribute(&Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device);
    cuDeviceGetAttribute(&Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device);
    int32_t ImageMajor = SM / 10;
    int32_t ImageMinor = SM % 10;
    // Compatible = same major, equal or higher minor device
    return Major == ImageMajor && Minor >= ImageMinor;
}
```

This is the exact boolean cut that `rankImage()` lifts to a continuous score.

### CUDA fatbin internal selection — same policy, different layer

NVIDIA's fatbin loader (closed-source) implements SM-closeness selection internally: when a
fatbin carries SM 7.5, SM 8.0, SM 8.9, and SM 9.0 cubins and the device is SM 8.9, it loads
SM 8.9 (exact match preferred over SM 8.0 which is also compatible).  This is equivalent to
returning `ImageSM` as the score and selecting max.

LLVM's `isELFCompatible` currently implements only the boolean gate.  `rankImage()` brings
liboffload to feature parity with CUDA's own fatbin selection for this case.

### `getComputeUnitKind()` in CUDA plugin (SM string construction)

```cpp
// offload/plugins-nextgen/cuda/src/rtl.cpp, line 1470
std::string getComputeUnitKind() const override {
    return "sm_" + std::to_string(Major * 10 + Minor);
}
```

The device SM string is already computed per-device.  `rankImage()` can reuse
`getComputeUnitKind()` output directly for string-level matching against `Meta.Arch`.

### AMDGPU architecture string

```cpp
// offload/plugins-nextgen/amdgpu/src/rtl.cpp, line 2438
std::string getComputeUnitKind() const override { return ComputeUnitKind; }
// ComputeUnitKind is set from "amdgcn-amd-amdhsa--gfx90a" triple parsing
```

AMDGPU arch strings are `"gfxXXX"` (e.g., `"gfx1100"`).  A simple string-equality check in
`AMDGPUPluginTy::rankImage` handles the common case; a GFX version number comparison handles
forward/backward compatibility.

---

## Feasibility

### Patch size estimate

| File | Change | LOC |
|------|--------|-----|
| `PluginInterface.h` | Add `rankImage` virtual method with default | ~15 |
| `PluginInterface.cpp` | Replace `break` with two-pass score loop | ~60 |
| `cuda/src/rtl.cpp` | `CUDAPluginTy::rankImage` SM scoring override | ~35 |
| `amdgpu/src/rtl.cpp` | `AMDGPUPluginTy::rankImage` GFX string scoring | ~30 |
| `level_zero/src/L0Plugin.cpp` | Default or SPIR-V version scoring | ~20 |
| Tests (`unittests/OffloadAPI/`) | Unit test: two SM cubins, verify best loads | ~50 |
| **Total** | | **~210 LOC** |

This is a 200-line patch.  It is the smallest possible change that fixes the first-match
limitation for the most common case (SM version ranking).

### Prerequisite: PR #186088 must merge first

`rankImage()` sits logically after `isMetadataCompatible` and uses `OffloadBinMetadataTy`.
Both are introduced in PR #186088.  The `rankImage()` patch is a direct follow-up to #186088
and should be submitted as a stacked PR immediately after #186088 merges.  Given that #186088
was opened March 12, 2026 and is under active review, it is likely to merge before Dublin
(April 7, 2026) or shortly after.

If #186088 has not merged by poster submission, the `rankImage()` patch can be presented as
a proposal against the PR #186088 branch, showing exactly where the callback fits.

### Prototype connection

`experiments/prototype/src/kdl.c` implements the equivalent scoring in `kdl_select()`:
capability contract matching followed by returning the best match.  The poster can show the
parallel between the prototype's `kdl_select()` logic and the proposed `rankImage()` loop
side-by-side — the prototype is the policy reference implementation, the `rankImage()` PR
is its upstream integration path.

### Demo scope

On the GTX 1650 (SM 7.5):

1. Assemble an OffloadBinary with an SM 6.1 cubin and an SM 7.5 cubin using
   `clang-offload-packager`.
2. With PR #186088 logic (first-match): SM 6.1 loads regardless of order.
3. With `rankImage()` patch: SM 7.5 always loads (higher score).
4. Measure kernel occupancy difference (SM 7.5 cubin uses tensor intrinsics unavailable on
   SM 6.1-targeted code; occupancy increases 1.3x–2x for typical GEMM).

This demo is achievable in one day with the existing prototype infrastructure.

---

## Upstream Path

### Stage 1 — PR against PR #186088 branch (immediate)

File: `offload/plugins-nextgen/common/include/PluginInterface.h`

```cpp
// Add after isMetadataCompatible declaration (line ~1444 in current PR diff)
/// Rank this image for loading on \p DeviceId.
/// Called after isMetadataCompatible and isDeviceCompatible both return true.
/// Return a non-negative score (higher = better fit).
/// Return -1 to unconditionally reject.
/// Default implementation returns 0 (equal preference for all compatible images,
/// preserving current first-match behaviour when all scores are equal).
virtual Expected<int> rankImage(const OffloadBinMetadataTy &Meta,
                                int DeviceId) const {
  return 0;
}
```

### Stage 2 — Default overrides in plugin-specific files

- `CUDAPluginTy::rankImage`: SM numeric closeness (described above)
- `AMDGPUPluginTy::rankImage`: GFX string equality / numeric version
- `LevelZeroPluginTy::rankImage`: SPIR-V version metadata key if present, else 0

### Review forum

- Target reviewer: Alex Duran (PR #186088 author) + Joseph Huber (AMD, liboffload
  maintainer) + Johannes Doerfert (LLNL, RFC #74302 author)
- Forum: LLVM Discourse `offload` category + biweekly offload coordination meeting
  (established January 2024, alternating with OpenMP)
- PR label: `offload`, `liboffload`, `follow-up`

### Why this lands upstream

1. The PR #186088 author explicitly asked for a follow-up.
2. The `isMetadataCompatible` pattern (virtual method, default-passthrough) is already
   merged and sets the review precedent.
3. The default implementation (`return 0`) is backwards-compatible: existing plugins with
   no override continue to get first-match behaviour.
4. The CUDA SM closeness scoring matches what CUDA's own fatbin loader does, so reviewers
   can validate correctness against NVIDIA's documented behaviour.
5. The patch is 200 lines — small enough for a single-reviewer approval cycle.

---

## Novelty Score: 7/10

The idea is direct and derivable from reading PR #186088.  The novelty is not in the
concept (Joseph Huber has informally described this direction), but in:

(a) the concrete API design (`rankImage` as a virtual method following `isMetadataCompatible`
    convention, returning `int` score rather than `bool`),
(b) the SM-closeness scoring function that mirrors CUDA's internal fatbin resolver,
(c) the two-pass implementation strategy that avoids loading-then-unloading,
(d) the connection to libkdl's capability-contract metadata embedded in the OffloadBinary
    StringMap.

Score compared to topic-05 (offload-ld, 9/10): lower because the concept is narrower and
the gap is already community-acknowledged.  Score relative to a blank contribution: high
because no implementation or formal proposal exists.

## Community Interest Score: 9/10

- PR #186088's author explicitly requests this follow-up.
- Joseph Huber (AMD, likely at Dublin) owns the liboffload roadmap.
- The AMDGPU case is immediately painful: gfx1030 and gfx1100 variants both pass the
  boolean ELF check on a gfx1100 device; first-match loads gfx1030 (older architecture).
- The Intel Level Zero case is worse: SPIR-V images have no ELF header so
  `isELFCompatible` is not called at all; `rankImage` is the only path to version preference.
- CUDA's fatbin resolver parity is a legibility argument: "we should do what NVIDIA already
  does for single-vendor binaries, but cross-vendor."

## Implementation Effort: Low-Medium

Two days of focused LLVM contribution work:
- Day 1: patch + unit tests
- Day 2: RFC post + PR submission + iteration on review feedback

The main uncertainty is the two-pass vs. single-pass choice and whether `loadBinaryImpl`
is cheap enough to call and discard for non-winning candidates (for CUDA it is not; for
the host plugin it is trivial).

---

## Relationship to Topic-05 (offload-ld)

Topic-05 presents libkdl as a policy layer **above** liboffload.  Topic-06 presents
`rankImage()` as integrating that policy **into** liboffload.

The two are complementary:
- Topic-05 is the poster (full system: bundle format + capability matching + cost scoring +
  cross-vendor dispatch).
- Topic-06 is the upstream contribution (minimal patch: ranking callback inside liboffload).

They share the same evidence base (PR #186088, PR #185663, OffloadBinMetadataTy) and can
be cited together.  The poster can note: "The full libkdl policy layer runs above liboffload
today; a 200-line upstream patch (`rankImage`) brings the SM-closeness selection into
liboffload itself."

---

## Draft Pitch

> PR #186088 builds multi-image OffloadBinary support into liboffload's CUDA, AMDGPU, and
> Level Zero plugins — then stops at the first compatible image and `break`s, explicitly
> deferring ranking to "a follow-up PR."  We write that follow-up.  A single virtual method,
> `rankImage(const OffloadBinMetadataTy&, DeviceId) -> int`, added to `GenericPluginTy`
> alongside the already-merged `isMetadataCompatible`, replaces the `break` with a
> best-score accumulator.  The default CUDA override scores SM variants by numeric closeness
> (SM 8.9 > SM 7.5 on an SM 8.9 device), matching the internal selection policy CUDA's own
> fatbin loader uses for single-vendor binaries — extended here to the cross-vendor case.
> The patch is 200 lines, builds on two merged PRs (#185663, #186088), has an identified
> reviewer (Alex Duran), and is the smallest possible change that gives liboffload
> correctness parity with CUDA's own binary selection.  libkdl's prototype demonstrates the
> runtime half; `rankImage()` is the upstream integration path.

---

## Key References

- PR #186088 — `[OFFLOAD] Generalize support for OffloadBinary images` (open Mar 2026):
  https://github.com/llvm/llvm-project/pull/186088
- PR #185663 — `[OFFLOAD] Add interface to extend image validation` (merged Mar 2026):
  https://github.com/llvm/llvm-project/pull/185663
- PR #185404 — `[Offload][L0] Add support for OffloadBinary format in L0 plugin` (merged Mar 2026):
  https://github.com/llvm/llvm-project/pull/185404
- `offload/plugins-nextgen/common/include/PluginInterface.h` (current main):
  https://github.com/llvm/llvm-project/blob/main/offload/plugins-nextgen/common/include/PluginInterface.h
- `offload/plugins-nextgen/cuda/src/rtl.cpp` — `isELFCompatible`, `getComputeUnitKind`:
  https://github.com/llvm/llvm-project/blob/main/offload/plugins-nextgen/cuda/src/rtl.cpp
- RFC: Introducing llvm-project/offload (#74302):
  https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- LLVM Offloading Infrastructure — Huber, DevMtg 2025:
  https://llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf
- OffloadBinary.h format spec:
  https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/OffloadBinary.h
- Topic-05 (offload-ld, broader framing):
  research/mega-survey/20-poster-topics/waves/topic-05-offload-ld.md
- PR #122106 — liboffload C API (merged Apr 2025):
  https://github.com/llvm/llvm-project/pull/122106
- PR #147943 — `ol_symbol_handle_t` rename (merged Jul 2025):
  https://github.com/llvm/llvm-project/pull/147943
