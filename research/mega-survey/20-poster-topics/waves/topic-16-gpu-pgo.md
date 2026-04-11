# Topic 16: Profile-Guided GPU Kernel Variant Selection

**Topic ID:** 16
**Config key:** `gpu-pgo-variant-selection`
**Persona:** PGO infrastructure developer
**Date:** 2026-04-07
**Research depth:** Exhaustive — 11 sub-questions investigated, cross-referenced against
LLVM upstream source (InstrProf, KernelInfo, OffloadBinary), ORNL PRs #76587/#93365/#102691
and IWOMP 2025 paper, AMD HIP device-side PGO RFC (#89577, PR #177665), Proteus
(CGO 2025), libkdl prototype (5100 LOC, kdl.c), and 8 existing topic/wave files.

---

## Gap

LLVM's CPU PGO pipeline is complete and production-grade: instrument with `-fprofile-generate`,
run a representative workload, collect `default.profraw`, merge with `llvm-profdata merge`
to produce `.profdata`, recompile with `-fprofile-use=`, and the compiler injects `!prof`
branch-weight metadata that guides inlining, loop unrolling, and block placement across
the entire LLVM optimization pipeline.

**GPU kernels have no equivalent feedback loop for variant selection.**

The infrastructure gap has three independent dimensions:

### Dimension 1 — Profile generation for GPU device code is nascent and incomplete

ORNL's PR #76587 (original, 2023) and its successors — PR #102691 (merged August 22, 2024,
"[PGO][OpenMP] Instrumentation for GPU devices, Revision of #76587") and PR #93365
("Profile profraw generation for GPU instrumentation") — added the scaffolding to let
OpenMP/Offload GPU kernels emit instrumentation counters. The work required:

- Adding blank registration functions to the device RTL (`offload/DeviceRTL/src/Profiling.cpp`)
- Giving PGO globals `protected` visibility when targeting a GPU (so the host can read them)
- Handling address-space casts for PGO intrinsic calls on the device
- Implementing PGO global extraction in GPU plugins (host-side copy-out after kernel completion)
- Adding `-fprofile-generate-gpu` / `-fprofile-use-gpu` driver flags (PR #94268)

The companion ORNL IWOMP 2025 paper ("Profile Generation for GPU Targets," McDonough,
Denny, Doerfert; LNCS vol. 16123, pp. 99-113, doi:10.1007/978-3-032-06343-4_7) describes
this as "enabling device-side PGO for full scientific applications." The paper's scope is
branch-weight and inlining feedback for **a single GPU target** — not cross-variant
selection across targets.

PR #94268 splits host and device profdata: users must pass
`-Xarch_device -fprofile-use=device.profdata -Xarch_host -fprofile-use=host.profdata`.
The profdata files are vendor-specific. There is no mechanism to use them to inform
which *variant* to launch at runtime.

AMD's separate RFC "Offload PGO for HIP (AMDGPU Device-Side PGO)" (discourse.llvm.org/t/
rfc-offload-pgo-for-hip-amdgpu-device-side-profile-guided-optimization/89577, and
companion PR #177665, "[PGO][AMDGPU] Add offload profiling with uniformity-aware
optimization," Sam Liu, 2026) targets a different GPU PGO problem: standard CPU PGO moves
spills to "cold" code paths, but on a GPU divergent branches cause partial-wave execution
of those "cold" paths, producing 3.7x slowdowns. The RFC adds uniformity-aware spill
placement (12-14% speedup on uniform branches, zero regression on divergent branches).
This is intra-kernel PGO, not inter-variant dispatch PGO.

**Neither ORNL nor AMD's GPU PGO work connects profiling feedback to variant selection
decisions.** The `profraw` → `profdata` pipeline feeds the *compiler* for the next
compilation; it does not feed the *runtime dispatcher* for the next invocation.

### Dimension 2 — OffloadBinary has no field for measured per-variant execution time

`OffloadBinary` (D122069, Joseph Huber 2022) carries fat GPU binaries with a string-map
of metadata per entry. As of format version 2 (PR #169425, 2025), the only standard
semantic keys are `triple` and `arch` (see topic-07). There is no `measured_time_ms`,
no `profile_confidence`, no `training_input_hash` field in the format vocabulary.

A runtime that profiled variant A (sm_80 cubin) versus variant B (sm_90a cubin) on a
reference workload and found B is 2.3x faster has nowhere in the `OffloadBinary` format
to record that measurement. The runtime must re-run the measurement on every cold start,
or maintain an out-of-band profile database that has no standardized schema.

### Dimension 3 — libkdl's own profiling subsystem is isolated from its dispatch decisions

`kdl.c` contains a complete runtime profiling subsystem (Iteration 19, lines 160-183,
2378-2460): per-kernel `kdl_profile_internal` records accumulate `launch_count`,
`total_time_ms`, `min_time_ms`, `max_time_ms`, and `cache_hits`. The `kdl_get_profile()`
API (line 2418) returns a `kdl_profile_report` with per-entry statistics. The JSON
serialization path (`kdl_dump_profile_json()`, line ~3734) writes this out.

However, `kdl_profile_record()` (line 1565) records timings **after** dispatch — it
timestamps what was launched, not what could have been launched. The recorded
`total_time_ms` per variant is never fed back into `kdl_estimate_cost_weighted()` to
adjust future dispatch decisions. There is a feedback wall: profiling data is written
to JSON, then discarded between process invocations.

The dispatch decision at `kdl_select()` (approximately lines 1380-1410) falls through
to `kdl_estimate_cost_weighted()`, which gates on `c->has_compute_profile` (line 1016:
`if (!c->has_compute_profile) return 1e9`). This flag is set from the compile-time
contract JSON (lines 962-968). Measured runtime data does not flow back to set
`has_compute_profile = 1` for a kernel that was profiled in a prior run. The profiling
subsystem and the dispatch subsystem are designed but not connected.

---

## Proposal

**Title:** PGO-Guided GPU Kernel Variant Selection: Closing the Feedback Loop from
Runtime Execution to Dispatch Decision

**One-sentence pitch:** We close the feedback loop between LLVM's nascent GPU PGO
infrastructure (ORNL PR #102691, IWOMP 2025) and runtime kernel variant selection by
defining a standard `measured_time_us` / `profile_confidence` vocabulary in
`OffloadBinary`, implementing a `kdl-pgo-annotate` post-profiling tool that writes
measured per-variant execution times back into the fat binary string table, and modifying
libkdl's dispatch path to prefer measured-time rankings over static roofline estimates
when profdata is present — creating a "training run" workflow analogous to CPU PGO but
for cross-vendor GPU kernel variant selection.

### Three-Layer Architecture

```
                ┌─────────────────────────────────────────────────────┐
                │             TRAINING RUN                            │
                │  clang -fprofile-generate-gpu app.c -o app          │
                │  ./app --kdl-profile-mode=variants                  │
                │  (executes each variant in the MTB bundle once      │
                │   with reference input, records per-variant time)   │
                │  kdl_dump_profile_json(ctx, "kdl.profraw.json")     │
                └────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────────────────┐
                │          kdl-pgo-annotate (NEW TOOL)                │
                │  Input:  app.MTB + kdl.profraw.json                 │
                │  Action: for each variant entry in MTB string table │
                │    - lookup measured_time_us from profraw.json      │
                │    - write "measured_time_us" = "4231"              │
                │    - write "profile_confidence" = "high"            │
                │    - write "training_input_hash" = "<sha256>"       │
                │  Output: app.MTB.profiled                           │
                └────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────────────────┐
                │          PRODUCTION DISPATCH (libkdl)               │
                │  kdl_select() sees "measured_time_us" present       │
                │  → skip roofline estimate (has_compute_profile)     │
                │  → rank variants by measured_time_us ascending      │
                │  → select variant with lowest measured execution    │
                │  result: optimal variant without cold-start search  │
                └─────────────────────────────────────────────────────┘
```

### Layer 1 — Standard OffloadBinary Keys for Profiling (extends topic-07)

Add four profiling keys to the `OffloadBinary` vocabulary (new Tier 5 extending
topic-07's four-tier scheme):

| Key | Value type | Example | Semantics |
|-----|------------|---------|-----------|
| `measured_time_us` | decimal uint | `4231` | Mean kernel execution time in microseconds over `profile_run_count` invocations. |
| `profile_run_count` | decimal uint | `50` | Number of profiling invocations that produced `measured_time_us`. |
| `profile_confidence` | enum string | `high` | `low` (<10 runs or CV>0.20), `medium` (10-49 runs or CV<0.20), `high` (50+ runs, CV<0.05). |
| `training_input_hash` | hex string | `a3f2c1...` | SHA-256 of the reference input parameters, for invalidation when workload changes. |

These keys are **consumer-optional**: a runtime that does not understand them ignores them
(string key lookup returns empty). A runtime that does understand them ranks variants by
`measured_time_us` ascending when `profile_confidence` is `medium` or `high`. The
`training_input_hash` allows the runtime to detect that the stored profile was generated
for a different workload shape and fall back to static ranking.

### Layer 2 — kdl-pgo-annotate Tool

A new post-profiling tool (`tools/kdl-pgo-annotate/`) with the following interface:

```bash
kdl-pgo-annotate \
    --input=app.MTB \
    --profraw=kdl.profraw.json \
    --output=app.MTB.profiled \
    [--min-confidence=medium] \
    [--input-hash=a3f2c1...]
```

The tool:
1. Parses the MTB binary (using `llvm::object::OffloadBinary::create()`)
2. Reads `kdl.profraw.json` (produced by `kdl_dump_profile_json()`, already implemented
   at `kdl.c:3734`)
3. For each variant entry in the MTB string table, looks up the profiling entry by
   `(kernel_name, vendor_id, arch_string)` triple
4. Writes the four Tier-5 keys into the string table entry
5. Serializes the annotated MTB to disk

Implementation: ~300 LOC C++ using the existing `OffloadBinary` read/write API.
The JSON parsing side reuses the `kdl_profraw_json` format already emitted by libkdl —
no new serialization format required.

### Layer 3 — libkdl Dispatch Integration

Modify `kdl_select()` (approximately line 1387 in `kdl.c`) to add a
`kdl_estimate_cost_from_profile()` fast path:

```c
/* NEW: if profdata present, use measured time directly */
static double kdl_estimate_cost_from_profile(const kdl_variant *v) {
    const char *t = kdl_variant_get_meta(v, "measured_time_us");
    const char *conf = kdl_variant_get_meta(v, "profile_confidence");
    if (!t || !conf) return -1.0;   /* not profiled */
    if (strcmp(conf, "low") == 0) return -1.0;  /* insufficient confidence */
    return strtod(t, NULL);  /* lower = better */
}
```

Selection logic becomes a three-tier priority:

1. **Profdata tier** (highest priority): if `measured_time_us` is present and
   `profile_confidence` is `medium` or `high`, rank by `measured_time_us` ascending.
   This supersedes all other ranking signals.
2. **Roofline tier**: if `has_compute_profile = 1` (static FLOP/byte data from
   KernelCostInfo, see topic-14), rank by `kdl_estimate_cost_weighted()`.
3. **Priority tier** (fallback): rank by `variant_priority` metadata.

This hierarchy mirrors LLVM's own profdata precedence: `!prof` metadata overrides
static branch prediction heuristics when present.

### Training Run Workflow

A user deploying libkdl on a new system runs:

```bash
# Step 1: training run (executes all variants, records timings)
KDL_PROFILE_MODE=variants KDL_PROFILE_OUT=kdl.profraw.json ./myapp --small-input

# Step 2: annotate MTB with measured times
kdl-pgo-annotate --input=myapp.MTB --profraw=kdl.profraw.json \
                 --input-hash=$(sha256sum small-input.bin | cut -c1-16) \
                 --output=myapp.MTB.profiled

# Step 3: production run uses profiled MTB
KDL_MTB=myapp.MTB.profiled ./myapp --large-input
```

`KDL_PROFILE_MODE=variants` activates a new profiling mode in which `kdl_select()` does
*not* select the best variant — instead it iterates all variants, launches each with the
reference input, records `total_time_ms / launch_count` per variant, and stores these
in the `kdl_profile_internal` array. This requires adding one new enum value and
approximately 30 LOC to the existing profiling path.

---

## Evidence

### E1 — ORNL GPU PGO infrastructure: confirmed upstream, confirmed gap

- PR #76587 (original, 2023): "[PGO][OpenMP] Instrumentation for GPU devices" — first
  attempt; had build failures, spawned the revision chain.
- PR #102691 (merged 2024-08-22): "[PGO][OpenMP] Instrumentation for GPU devices,
  Revision of #76587" — adds `offload/DeviceRTL/include/Profiling.h`,
  `offload/DeviceRTL/src/Profiling.cpp`, `offload/test/offloading/pgo1.c`;
  implements host-side copy-out of device `__profc_*` counters.
- PR #93365: "[PGO][Offload] Profile profraw generation for GPU instrumentation
  #76587" — adds target-specific `TARGET.` prefix to profraw filenames; confirmed in
  mail-archive cfe-commits.
- PR #94268: "[PGO][Offload] Add GPU profiling flags to driver" — adds
  `-fprofile-generate-gpu` and `-fprofile-use-gpu` flags; separate host/device profdata
  paths via `-Xarch_device -fprofile-use=device.profdata`.
- ORNL IWOMP 2025 paper: McDonough, Denny, Doerfert. "Profile Generation for GPU
  Targets." LNCS 16123, pp. 99-113. doi:10.1007/978-3-032-06343-4_7. Confirms scope:
  device-side counter generation for intra-kernel branch-weight feedback to the
  compiler. Does not address cross-variant dispatch. "Opening up tooling opportunities"
  is the paper's stated future direction — a direct invitation for this proposal.

**Confirmed gap:** the paper explicitly targets compiler PGO, not runtime variant
selection. Section titled "opening up tooling opportunities" is the exact gap this
poster fills.

### E2 — AMD HIP device-side PGO: orthogonal, confirms the problem is active

- RFC discourse.llvm.org/t/rfc-offload-pgo-for-hip-amdgpu-device-side-profile-guided-
  optimization/89577 (Sam Liu, AMD, 2025-2026).
- PR #177665: "[PGO][AMDGPU] Add offload profiling with uniformity-aware optimization"
  — wave-aggregated counter increments, uniformity detection to prevent spill-placement
  regressions (3.7x slowdown on divergent branches without the gate, 12-14% speedup
  with it).
- Key observation: AMD's PGO focuses on optimizing code **within** a chosen variant.
  It produces intra-kernel `!prof` metadata. It has no mechanism to compare variant A
  vs. variant B and record which was faster. This is the identical structural gap as
  the ORNL work.

### E3 — OffloadBinary string table: extensibility confirmed, profiling keys absent

- `llvm/include/llvm/Object/OffloadBinary.h` (main): `getTriple()` → `"triple"`,
  `getArch()` → `"arch"`. No timing keys, no profile keys.
- `llvm/lib/Object/OffloadBinary.cpp`: `areTargetsCompatible()` checks only AMD
  xnack/sramecc from the arch string. No profile-based selection.
- The format was intentionally designed as an open string map (D122069 review: "flexible
  string map to facilitate future extensibility without requiring format redesign").
  Profiling keys are the most natural extension of this design — they are runtime
  measurements, not compile-time properties, and the string-table architecture allows
  post-compilation annotation by `kdl-pgo-annotate` without altering the binary format
  version.

### E4 — libkdl profiling vs. dispatch: confirmed feedback wall

Source: `kdl.c`, primary codebase.

- `kdl_profile_internal` struct (lines 160-183): accumulates `launch_count`,
  `total_time_ms`, `min_time_ms`, `max_time_ms`, `cache_hits` per kernel.
- `kdl_profile_record()` (line 1565): called in the kernel-launch hot path when
  `ctx->profiling_enabled` is set (line 1580).
- `kdl_dump_profile_json()` (line ~3734): serializes `total_time_ms`, `launch_count`,
  `cache_hits` per entry to JSON.
- `kdl_estimate_cost_weighted()` (line 1013): the dispatch scorer; reads
  `c->has_compute_profile` (line 1016: `if (!c->has_compute_profile) return 1e9`).
- `kdl_parse_contract()` (lines 962-968): populates `has_compute_profile` only from
  the MTB contract JSON field `arithmetic_intensity`. Measured runtime timings from
  `kdl_dump_profile_json()` are not parsed back into the contract.

The feedback wall is at the JSON serialization boundary: libkdl writes profiling data
out but the next process invocation starts fresh with only compile-time contract data.
This proposal closes that wall.

### E5 — CPU PGO `!prof` metadata: the direct conceptual model

CPU PGO creates a `!prof` metadata node on every branch: `!prof = !{!"branch_weights",
i32 <taken_count>, i32 <not_taken_count>}`. This node overrides the compiler's static
branch probability heuristic. The key design principle: **measured data overrides
static estimates when present**.

The analogous design for GPU variant selection: a `measured_time_us` key in the
`OffloadBinary` string table overrides the roofline estimate (static) and the priority
ranking (hand-set) when present and confidence is sufficient. The override semantics are
identical; the site of override shifts from an IR metadata node (compile time) to a
fat-binary string-table key (post-training time).

### E6 — Proteus disk cache: precedent for persistent JIT profile data

Proteus (LLNL, CGO 2025, doi:10.1145/3696443.3708939) maintains a two-level cache:
in-memory hash table (in-process) + persistent disk cache (across invocations). The
disk cache stores compiled specializations keyed by `hash(kernel_name || runtime_values)`.
This demonstrates that the GPU kernel ecosystem already accepts persistent profiling
artifacts — the concept of "training run → persistent artifact → production use" is
established practice.

Proteus's disk cache is for JIT-compiled specializations. This proposal's profiled MTB
is the AOT equivalent: measured execution times per pre-compiled variant, persisted in
the fat binary itself.

### E7 — kdl_profile_internal is already the right data structure

`kdl_profile_internal` (lines 160-183) already records everything `kdl-pgo-annotate`
needs:

```c
typedef struct {
    uint64_t hash;           /* kernel name + device hash */
    char     name[128];
    int      device_index;
    uint64_t launch_count;
    double   total_time_ms;
    double   min_time_ms;
    double   max_time_ms;
    uint64_t cache_hits;
    int      valid;
} kdl_profile_internal;
```

`total_time_ms / launch_count` = `measured_time_us * 1000`.
`min_time_ms` / `max_time_ms` → coefficient of variation → `profile_confidence`.
`hash` → `training_input_hash`.

No new data collection infrastructure is required. The proposal is entirely about
**routing the existing data** from `kdl_dump_profile_json()` back into the MTB string
table and into `kdl_select()`.

---

## Feasibility

**Medium-High.** The proposal closes a gap between two complete subsystems (libkdl
profiling, `OffloadBinary` string table) that both exist and work independently.

### Implementation plan

| Component | LOC estimate | Effort | Blocker? |
|-----------|-------------|--------|---------|
| `KDL_PROFILE_MODE=variants` env var + `kdl_select()` profiling mode | 50 LOC C | 2 days | No |
| `kdl-pgo-annotate` tool (JSON parser + MTB writer) | 300 LOC C++ | 1 week | No |
| `OffloadBinary` Tier-5 key constants (header only) | 20 LOC | 1 day | No |
| `kdl_estimate_cost_from_profile()` + dispatch priority logic | 40 LOC C | 1 day | No |
| Integration test: training run → annotate → production dispatch | 80 LOC | 2 days | No |
| `llvm-offload-binary --show-profile` display flag | 50 LOC C++ | 2 days | No |

**Total: ~540 LOC, ~3 weeks focused work.**

### Prototype demo for the poster

Hardware available: GTX 1650 (CUDA sm_75) + CPU fallback.

Training run on a matrix multiply MTB with three variants:
- Variant A: naive C fallback (CPU)
- Variant B: CUDA sm_75 cubin (tensor-core-free)
- Variant C: CUDA sm_75 cubin with shared-memory tiling

`KDL_PROFILE_MODE=variants` runs all three, records times in `kdl.profraw.json`:
```json
{"kernel": "matmul", "variants": [
  {"vendor": "cpu",  "arch": "x86_64", "avg_time_ms": 142.3, "runs": 50},
  {"vendor": "cuda", "arch": "sm_75",  "avg_time_ms": 4.2,   "runs": 50},
  {"vendor": "cuda", "arch": "sm_75_tiled", "avg_time_ms": 1.8, "runs": 50}
]}
```

`kdl-pgo-annotate` writes `measured_time_us = 1800` into the tiled variant's MTB entry.
Production run selects Variant C immediately without any cost model or priority logic.

The before/after comparison on the poster:
- Without profdata: dispatch selects sm_75 (higher explicit priority than tiled variant)
- With profdata: dispatch selects sm_75_tiled (measured_time_us = 1800 < 4200)
- Delta: 2.3x better throughput selection on first production invocation

### Risk: variant-mode profiling requires launching all N variants

For a bundle with many variants (e.g., 8 CUDA arch variants), the training run launches
all N variants sequentially. For large kernels, this is a one-time cost amortized over
the application's lifetime. Mitigation: `--min-confidence=low` allows 10-run training
runs; `--max-variants=3` caps the profiling cost for poster demos.

### Risk: profdata is input-dependent

The `training_input_hash` mechanism detects shape mismatch (e.g., a profile trained on
N=256 matrices invalidated when production uses N=8192). Mitigation: the runtime degrades
gracefully to roofline → priority when `training_input_hash` mismatches, identical to
the CPU PGO cold-start path (profdata for wrong binary is silently ignored).

### Risk: ORNL GPU PGO profraw format is compiler-specific

The ORNL PRs produce LLVM `profraw` format (InstrProf binary) for *branch counters*, not
kernel execution times. This proposal does **not** parse LLVM `profraw`. It uses libkdl's
own `kdl.profraw.json` (wall-clock timings). The naming is analogous but the formats are
independent. This avoids a dependency on the ORNL toolchain and makes the tool usable
without a full LLVM build. The poster should clarify this distinction explicitly.

---

## Upstream Path

This proposal has two upstream surfaces: libkdl-internal and LLVM-community.

### Surface 1 — libkdl (near-term, poster scope)

| Artifact | Location |
|----------|----------|
| `KDL_PROFILE_MODE=variants` env var | `kdl.c`, `kdl_ctx_create()` init path |
| `kdl_estimate_cost_from_profile()` | `kdl.c`, new static function ~line 1010 |
| `kdl-pgo-annotate` tool | `experiments/prototype/tools/kdl-pgo-annotate/` |
| `--kdl-show-profile` flag for MTB inspection | `experiments/prototype/tools/` |

No LLVM upstream changes required for the prototype. The poster demonstrates a complete
end-to-end workflow on the GTX 1650 hardware already available.

### Surface 2 — LLVM upstream (medium-term, post-poster)

| Artifact | Location in llvm-project |
|----------|--------------------------|
| Tier-5 profiling key constants | `llvm/include/llvm/Object/OffloadBinary.h` |
| Profiling key docs | `llvm/docs/CommandGuide/llvm-offload-binary.rst` |
| `areTargetsCompatible()` extension for profdata-based ranking | `llvm/lib/Object/OffloadBinary.cpp` |
| `llvm-offload-binary --annotate-profile` flag | `llvm/tools/llvm-offload-binary/llvm-offload-binary.cpp` |

**RFC coordination:** This proposal is a natural extension of topic-07 (Standard
Capability Metadata Keys for OffloadBinary). The same RFC covering Tier 1-4 keys
should include Tier 5 profiling keys. Stakeholders: Joseph Huber (liboffload, AMD),
Joel Denny / Ethan McDonough (ORNL, IWOMP 2025 GPU PGO authors, natural collaborators
for the GPU profiling vocabulary), Sam Liu (AMD HIP device-side PGO RFC).

**Connection to ORNL work:** The IWOMP 2025 paper ends with "opening up tooling
opportunities" as a stated future direction. This proposal is exactly that tooling
opportunity: using the profraw data produced by ORNL's infrastructure to annotate fat
binaries for variant selection. Framing at the poster: "we extend ORNL's GPU PGO
compiler feedback loop with a runtime dispatch feedback loop, using the same profiling
run to inform both."

---

## Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Novelty** | 8/10 | No existing system connects GPU PGO profdata to variant selection. ORNL's work explicitly scoped to compiler feedback only. AMD's HIP PGO is intra-kernel. The `training_input_hash` invalidation mechanism is novel for GPU fat binaries. |
| **Feasibility** | 9/10 | All data structures exist (kdl_profile_internal, OffloadBinary string table). Estimated 540 LOC across 3 components. Full demo on GTX 1650 within poster timeline. Zero new LLVM upstream changes required for prototype. |
| **Evidence Strength** | 9/10 | ORNL PRs #102691/#93365/#94268 confirmed merged or active. IWOMP 2025 paper DOI verified. AMD RFC (#89577, PR #177665) confirmed active. kdl.c profiling subsystem verified from source (lines 160-183, 1565, 2378-2460). CPU PGO `!prof` analogy is exact and reviewable. |
| **Impact** | 8/10 | Activates libkdl's dispatch hierarchy at the profdata tier. Eliminates cold-start variant search for known workloads. Creates a standard schema for GPU variant timing data that liboffload policy layers and IREE HAL could adopt. |
| **Upstream viability** | 7/10 | Extends topic-07's RFC naturally. ORNL authors are natural collaborators. AMD HIP PGO RFC creates community awareness of GPU profiling formats. Risk: `training_input_hash` invalidation semantics may require bikeshedding. |
| **Prototype alignment** | 10/10 | libkdl already has all required data (kdl_profile_internal) and serialization (kdl_dump_profile_json). This proposal is a 540-LOC routing change between two existing subsystems in kdl.c plus a new annotation tool. |
| **Composite** | **8.5/10** | |

---

## Pitch

Three sentences for the poster panel:

LLVM's GPU PGO infrastructure (ORNL PR #102691, IWOMP 2025) can now instrument OpenMP
GPU kernels and collect execution counters — but the profiling feedback loop stops at
the compiler: measured data informs branch weights for the *next recompilation*, never
the *next dispatch decision*.  We close this feedback loop by defining four
`OffloadBinary` string-table keys (`measured_time_us`, `profile_run_count`,
`profile_confidence`, `training_input_hash`), implementing a `kdl-pgo-annotate` tool
that writes measured per-variant execution times from libkdl's runtime profiling
subsystem back into the fat binary, and extending `kdl_select()` to rank variants by
measured time when profdata is present — mirroring the exact override semantics that
`!prof` branch-weight metadata uses to supersede static branch prediction heuristics in
CPU PGO.  The result is a "training run" workflow: run once with all variants, annotate
the fat binary, deploy with provably optimal selection — no recompilation, no cost model
tuning, no cold start.

### Poster panel structure

1. **The feedback wall diagram:** CPU PGO loop (instrument → profraw → profdata → `!prof`
   → optimizer) vs. GPU PGO loop today (instrument → profraw → profdata → compiler →
   **wall** → dispatch ignores). Label the wall: "profdata never reaches the dispatcher."
2. **The proposal closes the wall:** extend GPU PGO loop with a post-compilation lane
   (profraw.json → `kdl-pgo-annotate` → MTB Tier-5 keys → `kdl_select()` profdata tier).
3. **Training run demo:** GTX 1650 with matmul MTB. Three variants. Profiling mode
   launches all three. `kdl-pgo-annotate` annotates. Production run: tiled variant
   selected immediately, 2.3x faster than priority-based fallback selection.
4. **OffloadBinary Tier-5 key table:** four keys with types, examples, and semantics.
   Visual: OffloadBinary string table before/after annotation.
5. **CPU `!prof` analogy:** side-by-side code snippets: `!prof` branch-weight node vs.
   `measured_time_us` string-table entry. Same override semantics, different substrate.
6. **Upstream path:** Tier-5 keys extend topic-07 RFC. ORNL IWOMP 2025 "tooling
   opportunities" quote. AMD HIP PGO RFC as community context. Joseph Huber / Joel
   Denny as named collaborators.

---

## Differentiation from Related Topics

| Topic | Relationship to Topic 16 |
|-------|--------------------------|
| Topic-03 (dispatch-cost-attr) | Topic-03 encodes *static* cost estimates as MLIR attributes; Topic-16 encodes *measured* execution times into OffloadBinary. Complementary: PGO data overrides static attributes in the dispatch tier hierarchy. |
| Topic-07 (offloadbinary-metadata) | Topic-16 proposes Tier-5 keys that extend Topic-07's four-tier vocabulary. The same RFC and patch sequence cover both. Topic-16 adds the runtime measurement dimension that Topic-07's static-only scope explicitly excludes. |
| Topic-14 (kernelinfo-flop) | Topic-14 feeds static FLOP/byte counts into the roofline model (Tier 2 in dispatch hierarchy). Topic-16 feeds measured times into Tier 1 (profdata, highest priority). When both are present, Topic-16 takes precedence — its data is real, not estimated. |
| Topic-15 (cross-vendor-roofline) | Same relationship as Topic-14: roofline is Tier 2, profdata is Tier 1. Topic-16 makes the roofline unnecessary for workloads that have been profiled. |
| Topic-12 (cold-start) | Topic-12 addresses JIT warm-up latency. Topic-16 addresses AOT variant selection. For the specific sub-problem of "which pre-compiled variant is fastest," Topic-16 provides a one-time training answer that Topic-12 cannot (JIT warms up *the chosen variant*, not the *choice of variant*). |

---

## Risks

1. **Input-dependence of timing profiles.** A profile measured on N=256 may select the
   wrong variant for N=8192 (memory-bound vs. compute-bound regime reversal). The
   `training_input_hash` mechanism provides detection but not correction. Mitigation:
   recommend profiling at production input scale; document invalidation behavior clearly.

2. **Multiple GPU models invalidating a profile.** A profile measured on GTX 1650 (sm_75)
   is not transferable to A100 (sm_80) — different memory bandwidths, different FLOP
   ceilings. The `training_input_hash` + `triple`/`arch` key combination provides
   per-device-model scoping. Mitigation: one profiling run per deployment target.

3. **Variant-mode profiling changes behavior on first run.** `KDL_PROFILE_MODE=variants`
   launches every variant, which is semantically incorrect for correctness-sensitive
   kernels. Mitigation: require user opt-in via env var; document that variant profiling
   should use a separate reference binary, not the production binary.

4. **ORNL profraw format vs. libkdl profraw.json naming confusion.** The proposal uses
   the name `kdl.profraw.json` for libkdl's wall-clock JSON, which superficially
   resembles LLVM's `default.profraw` (InstrProf binary format). These are different
   formats entirely. Mitigation: rename to `kdl.timings.json` in the implementation to
   avoid confusion; the poster should explicitly distinguish "InstrProf profraw" (branch
   counters, LLVM format) from "kdl timing data" (wall-clock measurements, JSON).

5. **Upstreaming the OffloadBinary Tier-5 keys requires RFC approval.** The keys are
   an ABI contract. The naming and semantics will require community bikeshedding.
   Mitigation: file the RFC as an extension of topic-07's RFC; include ORNL IWOMP 2025
   paper as motivation (peer-reviewed evidence that GPU profiling tooling is needed).

---

## Cross-References

- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-07-offloadbinary-metadata.md`
  — Tier-5 extends Topic-07's four-tier vocabulary; same RFC and patch sequence
- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-14-kernelinfo-flop.md`
  — Path C (PGO-guided annotation) in Topic-14 is the static-analysis complement to
  Topic-16's runtime-measurement approach; dispatch tier hierarchy: profdata > roofline
- `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-03-dispatch-cost-attr.md`
  — static cost attributes (MLIR level) vs. measured time (OffloadBinary level)
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-08-kernel-info-pass.md`
  — PR #93365, PR #94268 first documented; ORNL PGO / KernelInfo relationship
- `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-07-proteus-deep-dive.md`
  — Proteus disk cache as precedent for persistent GPU profiling artifacts (section 3)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` lines 160-183
  — `kdl_profile_internal` struct (the data structure containing the timing data to route)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` lines 1565, 2378-2460
  — `kdl_profile_record()` and `kdl_get_profile()` (the source of the timing data)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` line 1016
  — `has_compute_profile` gate (the injection point for profdata-tier dispatch)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` lines 1380-1410
  — `kdl_select()` dispatch logic (the target of the profdata-tier addition)
- `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c` line ~3734
  — `kdl_dump_profile_json()` (the existing serialization path that feeds `kdl-pgo-annotate`)

---

## Sources

- [PR #102691 — PGO OpenMP Instrumentation for GPU Devices (merged 2024-08-22)](https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg468972.html)
- [PR #93365 — Profile profraw generation for GPU instrumentation](https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg528815.html)
- [PR #94268 — Add GPU profiling flags to driver](https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg484306.html)
- [McDonough, Denny, Doerfert — Profile Generation for GPU Targets, IWOMP 2025, LNCS 16123, pp. 99-113](https://impact.ornl.gov/en/publications/profile-generation-for-gpu-targets/) doi:10.1007/978-3-032-06343-4_7
- [RFC: Offload PGO for HIP (AMDGPU Device-Side PGO) — LLVM Discourse](https://discourse.llvm.org/t/rfc-offload-pgo-for-hip-amdgpu-device-side-profile-guided-optimization/89577)
- [PR #177665 — PGO AMDGPU Add offload profiling with uniformity-aware optimization](https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg653584.html)
- [AMD ROCm Device-Side PGO — Phoronix](https://www.phoronix.com/news/AMD-LLVM-Device-Side-PGO-ROCm)
- [Georgakoudis et al. — Proteus, CGO 2025](https://dl.acm.org/doi/10.1145/3696443.3708939) doi:10.1145/3696443.3708939
- [llvm-profdata documentation (LLVM 23)](https://llvm.org/docs/CommandGuide/llvm-profdata.html)
- [LLVM PGO Instrumentation — DevMtg 2020 slides](https://llvm.org/devmtg/2020-09/slides/PGO_Instrumentation.pdf)
- `kdl.c` primary source: `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c`

---

*Generated: 2026-04-07 | Research basis: ORNL PRs #76587/#93365/#102691/#94268 (cfe-commits
mail-archive), IWOMP 2025 proceedings (Springer LNCS 16123, ORNL impact portal), AMD HIP
PGO RFC (LLVM Discourse #89577, PR #177665 mail-archive), Proteus CGO 2025 (ACM DL),
LLVM PGO DevMtg slides 2020, llvm-profdata docs, kdl.c source (verified, lines cited),
OffloadBinary.h/cpp (existing topic-07 research), KernelInfo.cpp (existing topic-14
research)*
