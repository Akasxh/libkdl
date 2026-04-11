# Topic 07 — Standard Capability Metadata Keys for OffloadBinary

**Proposal class:** Format extension + tooling + documentation
**Effort estimate:** Medium (1–2 weeks implementation, RFC ~2 weeks review)
**Novelty:** Fills a documented gap; no competing RFC found as of April 2026

---

## Gap

`OffloadBinary` (introduced in D122069, `llvm/include/llvm/Object/OffloadBinary.h`) is
LLVM's canonical fat-binary container for GPU/offload images.  Its wire format is a
versioned header plus a per-entry string table — essentially a `MapVector<StringRef,
StringRef>` serialised to disk.  The format was intentionally left open-ended:

> "The format intentionally uses a flexible string map to facilitate future extensibility
> without requiring format redesign." — D122069 review thread

Yet as of LLVM 19 / version 2 of the format (bumped by PR #169425), only **two standard
string keys** are documented and used by the toolchain:

| Key       | Accessor         | Semantics                          |
|-----------|------------------|------------------------------------|
| `triple`  | `getTriple()`    | LLVM target triple, e.g. `nvptx64-nvidia-cuda` |
| `arch`    | `getArch()`      | Architecture string, e.g. `sm_89` |

A third key, `feature=`, was proposed in D127686 to carry target-feature strings for LTO
(e.g. `+ptx85`), motivated by removing a link-time clang driver dependency — but it was
never standardised into a documented vocabulary.

**Consequence:** A runtime choosing among multiple images in a fat binary (e.g. an
`sm_80` cubin vs. an `sm_90a` cubin that requires tensor-core Warp Specialization) has no
machine-readable way to know:

- What the minimum required SM level is beyond the `arch` string.
- Whether the image requires hardware features absent on the dispatch target (tensor cores,
  BF16, FP8, warp specialization, AGPR accumulation).
- How many SGPRs / VGPRs / LDS bytes the kernel consumes (occupancy planning).
- Whether a fallback image exists and how its quality compares.

The existing `areTargetsCompatible()` in `OffloadBinary.cpp` hard-codes only AMD
`xnack+/-` and `sramecc+/-` flag checking — all other selection logic is either absent or
pushed into vendor-specific runtimes with no shared vocabulary.

---

## Proposal

Define a **standard vocabulary of OffloadBinary string-map keys** that any compiler
backend can emit and any conforming runtime can consume.  The vocabulary is grouped into
four tiers:

### Tier 1 — Minimum-Requirement Keys (runtime MUST honour)

| Key                    | Value type     | Example                      | Semantics |
|------------------------|----------------|------------------------------|-----------|
| `min_sm`               | decimal uint   | `80`                         | Minimum CUDA compute capability (SMXY → 10*X+Y). Runtime rejects image if device SM < this. |
| `min_gfx`              | GFX arch string | `gfx90a`                    | Minimum AMD GFX target. Runtime rejects image if device ISA is older. |
| `requires_features`    | comma-list     | `tensor_core,bf16,fp8`       | Set of named capability tokens the image requires. Runtime rejects image if any token is absent on device. |

### Tier 2 — Resource-Usage Keys (runtime SHOULD use for occupancy)

| Key                    | Value type  | Example  | Semantics |
|------------------------|-------------|----------|-----------|
| `sgpr_count`           | decimal uint | `32`    | Scalar registers per wavefront (AMD). From `kernel-resource-usage` remark. |
| `vgpr_count`           | decimal uint | `128`   | Vector registers per work-item (AMD). |
| `agpr_count`           | decimal uint | `0`     | Accumulator VGPRs (AMD CDNA/MI series). |
| `registers_per_thread` | decimal uint | `64`    | NVIDIA registers per thread (from `EIATTR_REGCOUNT` in `.nv.info`). |
| `shared_mem_bytes`     | decimal uint | `49152` | Static shared memory in bytes. From `EIATTR_SHMEM_PARAM_SIZE` / AMD `.lds_size`. |
| `scratch_bytes`        | decimal uint | `0`     | Private/scratch memory per lane in bytes. |
| `warp_size`            | `32` or `64` | `32`    | Wavefront/warp width. 32 for NVIDIA, 32 or 64 for AMD. |

### Tier 3 — Quality/Variant Keys (runtime MAY use for selection ranking)

| Key                   | Value type   | Example       | Semantics |
|-----------------------|--------------|---------------|-----------|
| `variant_priority`    | decimal uint | `10`          | Higher = preferred when multiple images satisfy requirements. |
| `variant_tag`         | string       | `optimized`   | Human-readable tag: `generic`, `optimized`, `fallback`, `debug`. |
| `reqd_workgroup_size` | `X,Y,Z`      | `256,1,1`     | Required launch dimensions (mirrors AMD `.reqd_workgroup_size`). |
| `max_workgroup_size`  | decimal uint | `1024`        | Maximum flat workgroup size hint. |

### Tier 4 — Provenance Keys (tooling / diagnostics)

| Key                   | Value type  | Example                 | Semantics |
|-----------------------|-------------|-------------------------|-----------|
| `producer_version`    | semver str  | `18.1.0`                | Compiler version that produced this image. |
| `opt_level`           | `0`–`3`     | `3`                     | Optimisation level at which the image was compiled. |
| `lto`                 | `0` or `1`  | `1`                     | Whether this image was produced with LTO. |

### Proposed Canonical Token Vocabulary for `requires_features`

Tokens are lowercase, underscore-separated, and vendor-neutral where possible:

| Token              | Maps to CUDA           | Maps to AMD              | Maps to SPIR-V extension     |
|--------------------|------------------------|--------------------------|------------------------------|
| `tensor_core`      | `sm_75+`, Tensor Core  | `mfma` / `wmma`          | `SPV_KHR_cooperative_matrix` |
| `bf16`             | `sm_80+`               | gfx90a+                  | —                            |
| `fp8`              | `sm_89+` (e8m0)        | gfx940+                  | —                            |
| `warp_spec`        | `sm_90a` ws intrinsics | —                        | —                            |
| `dp4a`             | `sm_61+`               | gfx906+ dp4               | —                            |
| `xnack`            | —                      | demand paging            | —                            |
| `sramecc`          | —                      | SRAM ECC                 | —                            |
| `wavefront64`      | —                      | 64-wide wavefront        | —                            |
| `cooperative_groups` | `sm_60+`             | —                        | —                            |

---

## Evidence

### OffloadBinary format: only two standard keys exist

- `llvm/include/llvm/Object/OffloadBinary.h` (main): `getTriple()` → `getString("triple")`,
  `getArch()` → `getString("arch")`. No other named-constant keys are defined in the
  header.
- `llvm/lib/Object/OffloadBinary.cpp` (main): `areTargetsCompatible()` parses only AMDGPU
  target-ID xnack/sramecc flags from the arch string — no general capability matching.
- D127686: `feature=` key was prototyped for LTO target-feature propagation but never
  standardised.
- PR #169425: Version 2 format bump adds `EntriesCount` field and per-entry
  `ValueSize`-aware string entries — purely structural, adds no new semantic keys.

### What AMD already encodes in HSACO ELF (NT_AMDGPU_METADATA note, msgpack)

AMD Code Object V4/V5 kernel descriptors carry (per `AMDGPUUsage.html`):

```
.sgpr_count         uint32    scalar registers per wavefront
.vgpr_count         uint32    vector registers per work-item
.agpr_count         uint32    accumulator VGPRs (CDNA)
.lds_size           uint32    LDS allocation in bytes
.wavefront_size     uint32    32 or 64
.reqd_workgroup_size [X,Y,Z]  required launch dims
.max_flat_workgroup_size uint32
.private_segment_fixed_size uint32   scratch per lane
```

Target-ID feature flags (`xnack`, `sramecc`, `tgsplit`, `cumode`) are appended to the
arch string (e.g., `gfx908+xnack-sramecc`).  The `areTargetsCompatible()` function
already parses these from the arch string — but they are not first-class OffloadBinary
keys that a vendor-agnostic runtime can consume without AMD-specific string parsing.

### What CUDA already encodes in cubin `.nv.info` (EIATTR attributes)

CUDA cubin ELF carries per-kernel capability metadata in the `.nv.info` section:

```
EIATTR_REGCOUNT        (0x1F)  registers per thread
EIATTR_MAX_THREADS     (0x05)  max threads per block
EIATTR_REQNTID         (0x10)  required thread count (reqntid directive)
EIATTR_MIN_STACK_SIZE  (0x12)  minimum private stack
EIATTR_SHMEM_PARAM_SIZE         static shared memory
EIATTR_CBANK_PARAM_SIZE         constant bank param size
```

NVIDIA's `compute_90a` flag embeds architecture-specific variant tags (non-forward-
compatible cubins with warp-specialization intrinsics) — but these capability signals
never surface outside the CUDA driver's fatbin selector.

### What LLVM's KernelInfo pass already extracts (from IR)

`llvm/lib/Analysis/KernelInfo.cpp` emits `MachineOptimizationRemarkAnalysis` under pass
name `kernel-resource-usage`:

```
NumSGPR          scalar registers
NumVGPR          vector registers
NumAGPR          accumulator registers
ScratchSize      bytes per lane
Occupancy        waves per SIMD
SGPRSpill        spilled scalar registers
VGPRSpill        spilled vector registers
BytesLDS         LDS bytes (entry functions)
```

These values are emitted as remarks and printed to YAML diagnostics — but they are
**never written into the OffloadBinary string table**.  This is the direct pipeline gap.

---

## Feasibility

### Implementation path (end-to-end)

1. **Header constants** (`llvm/include/llvm/Object/OffloadBinary.h`):
   Add `constexpr StringLiteral` constants for each standard key (20 lines, no ABI
   impact since the format uses string equality for key lookup).

2. **Writer integration — AMDGPU backend**:
   In `llvm/lib/Target/AMDGPU/AMDGPUTargetObjectFile.cpp` (or the offload wrapper), after
   HSACO ELF emission, parse the NT_AMDGPU_METADATA msgpack note and populate
   `OffloadingImage::StringData` with the Tier 1 + Tier 2 keys.

3. **Writer integration — NVPTX backend**:
   In `clang/tools/clang-offload-wrapper/ClangOffloadWrapper.cpp`, after cubin emission,
   invoke `cuobjdump --dump-elf-raw` (or link against libcublas reader) to extract
   EIATTR values and emit them into the string table.  Alternatively, emit from the NVPTX
   MC layer before cubin generation is finalised.

4. **Runtime consumer** (`llvm/lib/Object/OffloadBinary.cpp`):
   Extend `areTargetsCompatible()` to accept a `DeviceCapabilities` struct and check
   `min_sm`, `min_gfx`, and `requires_features` tokens against it.

5. **llvm-offload-binary tooling** — new `--annotate` flag:
   Add `--annotate` to `llvm/tools/llvm-offload-binary/llvm-offload-binary.cpp` that,
   given a fat binary, prints all entries with their full string-table metadata in human-
   readable form (similar to `readelf --notes`).

6. **Documentation patch**:
   Add a "Standard Metadata Keys" table to
   `llvm/docs/CommandGuide/llvm-offload-binary.rst`.

### Prototype effort

The KernelInfo → OffloadBinary bridge (steps 2–3 above) is well-scoped:
- AMDGPU side: msgpack parsing of NT_AMDGPU_METADATA is already implemented in
  `llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp`; reuse that parser.
- NVPTX side: EIATTR values can be read from the cubin ELF after MC emission using
  `llvm::object::ELFFile<ELF64LE>` — no external dependencies.

The prototype in `experiments/prototype/src/kdl.c` (the KDL dispatcher) already performs
manual capability matching from a hand-rolled metadata struct; this proposal standardises
exactly that interface upstream.

---

## Upstream Path

### Is this an RFC or a documentation patch?

This requires a **short RFC on discourse.llvm.org** (Runtimes category) before the first
implementation patch, for two reasons:

1. The key names become a **stable ABI contract** — fat binaries built with LLVM 20 must
   be readable by LLVM 22's runtime.  Name bikeshedding should happen before code.
2. It touches at least three backends (AMDGPU, NVPTX, SPIR-V) and the offload runtime —
   the RFC lets each backend owner sign off on the shared vocabulary.

A documentation-only patch naming the two existing keys (`triple`, `arch`) and
reserving the namespace for the proposed vocabulary would be a viable first step and
could merge without an RFC, building community awareness.

### RFC outline

```
[RFC] Standard capability metadata keys for OffloadBinary

Background: OffloadBinary has had an extensible string table since D122069
but no documented vocabulary beyond "triple" and "arch".

Motivation:
  - Runtime variant selection requires machine-readable capability data.
  - areTargetsCompatible() currently hard-codes AMD xnack/sramecc parsing.
  - KernelInfo/kernel-resource-usage already extracts this data but drops it.

Proposal: Four-tier vocabulary (min-requirement, resource-usage, quality, provenance).
Compatibility: Additive string-key extension; old binaries are silently missing keys.
Tooling: --annotate flag for llvm-offload-binary.
Prototype: Exists in libkdl (LLVM Dublin 2026 poster).
```

### Patch sequence

| Step | Patch | Reviewer group |
|------|-------|----------------|
| 0 | RFC on discourse | offload, AMDGPU, NVPTX owners |
| 1 | Header constants + docs | llvm/Object owners |
| 2 | AMDGPU writer | AMDGPU backend |
| 3 | NVPTX writer | NVPTX/CUDA backend |
| 4 | Runtime consumer extension | offload runtime |
| 5 | llvm-offload-binary --annotate | tools |

All patches are independent and can be reviewed in parallel after step 1.

---

## Scores

| Dimension               | Score | Rationale |
|-------------------------|-------|-----------|
| **Novelty**             | 7/10  | No competing RFC; gap is well-known but unaddressed |
| **Upstream impact**     | 9/10  | Touches the foundational format used by every offload-capable LLVM backend |
| **Prototype alignment** | 9/10  | KDL already implements this at prototype level; proposal formalises it |
| **Implementation risk** | 3/10  | Additive string keys; no binary-compat break; well-precedented pattern |
| **Reviewer friction**   | 5/10  | Multi-backend RFC; naming bikeshedding is expected but manageable |
| **Poster fit**          | 9/10  | Concrete, visual, self-contained; table of keys + pipeline diagram tells the story cleanly |

**Composite:** 8/10 — strong poster candidate.

---

## Pitch

> LLVM's `OffloadBinary` carries fat binaries for every GPU target — NVIDIA, AMD, CPU —
> yet its metadata vocabulary ends at `triple` and `arch`.  A runtime trying to select the
> best image from a multi-variant fat binary has no standard way to ask: "does this cubin
> need tensor cores?  How many registers does it burn?  Is there a fallback?"  AMD's
> HSACO already encodes `.sgpr_count`, `.lds_size`, and ISA feature flags in msgpack
> notes; CUDA's cubin encodes EIATTR register and shared-memory attributes in `.nv.info`;
> LLVM's own `kernel-resource-usage` remark pass produces occupancy data at compile time.
> None of it reaches the string table.  This poster proposes a 4-tier standard vocabulary
> of OffloadBinary keys — **min_sm / requires_features** for capability gating,
> **sgpr_count / vgpr_count / shared_mem_bytes** for occupancy planning, and
> **variant_priority / variant_tag** for quality ranking — with a concrete pipeline from
> KernelInfo remark → OffloadBinary writer → runtime selector, a new
> `llvm-offload-binary --annotate` flag, and a prototype already running on GTX 1650 +
> MI300X under libkdl.

---

## Related Work / Prior Art

- D122069 — original OffloadBinary format (Joseph Huber, 2022)
- D127686 — `feature=` key prototype for LTO target features (2022, not merged to standard)
- PR #169425 — format version 2, multiple entries, ValueSize field (Yury Plyakhin, 2025)
- AMDGPU Code Object V5 msgpack metadata spec (`llvm/docs/AMDGPUUsage.rst`)
- CUDA Binary Utilities: EIATTR attributes in `.nv.info` (NVIDIA, 2026)
- D123878 — `kernel-resource-usage` remark pass (SGPR/VGPR/LDS/occupancy, 2022)
- libkdl KDL dispatcher (`experiments/prototype/src/kdl.c`) — proof-of-concept runtime
  using hand-rolled capability metadata

---

*Generated: 2026-04-07 | Research basis: OffloadBinary.h/cpp (main), AMDGPUUsage.rst,
KernelInfo.cpp, CUDA Binary Utilities 13.2, D122069/D127686/PR#169425 review threads*
