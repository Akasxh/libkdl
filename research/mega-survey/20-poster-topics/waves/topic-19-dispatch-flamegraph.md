# Topic 19: End-to-End Dispatch Overhead Flame Graph for LLVM GPU Stack

**Persona:** performance detective
**Config name:** dispatch-overhead-flamegraph
**Date:** 2026-04-07

---

## The Gap

Nobody has published a latency breakdown of the full LLVM GPU dispatch path from
OffloadBinary parse to GPU execution. The wall-clock time is well-measured at the
bottom (TaxBreak: 4.71 μs floor on H100) and at the top (PyTorch eager: 5–10 μs
per kernel), but the *interior* of the LLVM stack — how much time is spent in
OffloadBinary parse, `olCreateProgram`, plugin dispatch, `cuModuleLoad`,
`cuGetSymbol`, and `cuLaunchKernel` individually — has never been published.

The five layers of the LLVM GPU dispatch path, in order:

```
[1] clang-linker-wrapper / OffloadBinary parse
    .llvm.offloading section scan → OffloadBinary::create() iterator
    → magic check (0x10FF10AD), version, per-entry metadata decode

[2] olCreateProgram (liboffload → plugin)
    Binary blob handed to GenericPluginTy::loadBinary()
    → isMetadataCompatible() filter (PR #185663)
    → isDeviceCompatible() check
    → vendor cubin/hsaco load call

[3] cuModuleLoadData / hipModuleLoadData (driver API)
    ELF/CUBIN parsed by GPU driver
    → device memory allocation, code object upload

[4] olGetSymbol → cuModuleGetFunction (kernel handle lookup)
    Name-string hashed into module's symbol table
    → CUfunction / hipFunction_t returned

[5] olLaunchKernel → cuLaunchKernel (hot-path dispatch)
    Grid/block/args packaged → CUDA driver enqueues to stream
    → GPU hardware picks up from the command queue
```

The "first-compatible-wins" loop in PR #186088 (open March 2026) iterates all
images without any timing instrumentation. The OMPT device profiling hooks
(Dhruva Chakrabarti, LLVM DevMtg 2025 GPU Workshop; mentioned in wave-02-llvm-offload-runtime.md)
exist to observe kernel dispatch events but are wired to the OpenMP semantic layer,
not to the raw liboffload `ol*` path. Issue #110007 tracks expanding OMPT device
hooks, but as of April 2026 no one has published a microsecond-resolution breakdown
across all five layers in a single experiment.

**The gap in one sentence:** the community has the dispatch floor (TaxBreak) and
the framework ceiling (PyGraph), but zero published data on where the time goes
*inside* the LLVM abstraction stack between them.

---

## The Proposal

**Title:** "Where Does the Time Go? A Flame Graph of the LLVM GPU Dispatch Stack"

The poster presents the first end-to-end latency decomposition of the LLVM GPU
dispatch path, measured on a GTX 1650, using `clock_gettime(CLOCK_MONOTONIC)`
brackets around each layer boundary and Brendan Gregg's `flamegraph.pl` for
visualization.

### Measurement design

```c
/* Layer 1: OffloadBinary parse time */
clock_gettime(CLOCK_MONOTONIC, &t[0]);
OffloadBinaryRef ob = OffloadBinary::create(buf);   /* C++ API wrapper */
clock_gettime(CLOCK_MONOTONIC, &t[1]);

/* Layer 2: olCreateProgram (liboffload → plugin load) */
olCreateProgram(device, blob_data, blob_size, &prog);
clock_gettime(CLOCK_MONOTONIC, &t[2]);

/* Layer 3: implicitly inside olCreateProgram:
   cuModuleLoadData measured separately via direct CUDA driver call */
cuModuleLoadData(&cumod, blob_data);
clock_gettime(CLOCK_MONOTONIC, &t[3]);

/* Layer 4: olGetSymbol (name lookup) */
olGetSymbol(prog, "null_kernel", OL_SYMBOL_KIND_KERNEL, &sym);
clock_gettime(CLOCK_MONOTONIC, &t[4]);

/* Layer 5: olLaunchKernel (hot-path dispatch, N=10,000 iterations) */
for (int i = 0; i < N; i++) {
    clock_gettime(CLOCK_MONOTONIC, &t[5]);
    olLaunchKernel(queue, device, sym, NULL, &dims);
    olWaitQueue(queue);
    clock_gettime(CLOCK_MONOTONIC, &t[6]);
}
```

The null-kernel (1-thread, 0-work, 0-shared-mem) isolates dispatch overhead from
compute. Warm-up: 1,000 dispatches discarded. Measurement: 10,000 dispatches,
per-dispatch histogram (p50/p95/p99). The TaxBreak methodology (arXiv:2603.12465)
is the canonical reference for this approach on H100/H200; this work replicates it
on consumer hardware (GTX 1650) and extends it *up the stack* into liboffload.

### Flame graph format

Stack frames correspond to layers; frame width = fraction of total dispatch latency:

```
cuLaunchKernel (hardware floor ~4.71 μs on H100; estimated ~8–12 μs on GTX 1650)
└── cuModuleGetFunction / cuModuleLoadData (one-time; amortized to ~0 on hot path)
    └── plugin::loadBinary → isMetadataCompatible → isDeviceCompatible
        └── olCreateProgram (one-time path)
            └── OffloadBinary::create() → section iterate → metadata decode
                └── clang-linker-wrapper registration __tgt_register_lib()
```

Separate cold-path (first dispatch) and hot-path (subsequent dispatch) flame graphs,
because the amortization structure differs radically:

- **Cold path:** dominated by layers 1–4 (one-time load, parse, module init).
  Candidate bottleneck: `cuModuleLoadData` is known to JIT-link PTX on first use.
  Expected to take 10–100 ms on first call (driver JIT cost).
- **Hot path:** dominated by layer 5 alone. Layers 1–4 collapse to zero on a
  cached `CUfunction` handle. Expected: 8–15 μs on GTX 1650 vs 4.71 μs on H100
  (older hardware, slower PCIE bandwidth to driver).

### Connection to libkdl

libkdl already has `kdl_get_dispatch_latency_ns()` in `experiments/prototype/src/kdl.c`
(lines 4595–4649) which times `cuStreamSynchronize` as a proxy for hot-path dispatch
overhead. The poster extends this to full layer-by-layer decomposition, producing
the quantitative anchor that validates the libkdl thesis:

> "libkdl's O(1) hash-table variant lookup adds ~100–200 ns — measurably less
> than 2% of the GTX 1650 hardware floor, confirmed with per-layer instrumentation."

This turns a claim into a measurement.

### The accfg comparison

The accfg poster (Anton Lydike, Josse Van Delm — LLVM DevMtg 2024) addressed
accelerator dispatch setup overhead at the *compiler pass* level for ETH Zurich
SNitch cores (RISC-V based). This proposal addresses the same question at the
*runtime measurement* level for GPU dispatch on standard vendor hardware. The
differentiation is clean: accfg eliminates overhead at compile time; this work
*quantifies* it at runtime. If the accfg team is at Dublin, expect the question;
cite the distinction explicitly on the poster.

---

## Evidence

### Quantitative anchors (all from existing research waves)

| Measurement | Value | Source |
|------------|-------|--------|
| CUDA null-kernel floor (H100) | 4.71 μs avg (p50: 4.578 μs, p95: 5.396 μs) | TaxBreak arXiv:2603.12465, wave-03-dispatch-overhead.md |
| CUDA null-kernel floor (H200) | 4.50 μs avg | TaxBreak, wave-03-dispatch-overhead.md |
| cuLaunchKernel CPU-side latency | 2–4 μs (2019 hardware) | ICPP 2019 poster, Tsukuba HPCS |
| GPU-side kernel start latency | ~1 μs after API return | ICPP 2019 poster |
| PyTorch eager per-kernel overhead | 5–10 μs | PyGraph arXiv:2503.19779 |
| CUDA Graph per-node (Ampere, CUDA 12.6) | ~1 ns/node + 2.5 μs base | NVIDIA Blog, CUDA 12.6 |
| Dynamic dispatch table (O(1) lookup) | 1–2 μs, <0.8% e2e | arXiv:2601.00227, wave-03 |
| HIP baseline dispatch (ROCm 3.8) | 70 μs (25 μs optimized) | Kokkos #3670, wave-02-dispatch-overhead-benchmarks.md |
| Python-mediated CUDA dispatch (Numba) | ~200 μs | numba issue #3003, wave-02 |

### Stack architecture evidence

- `OffloadBinary.h` format: magic `{0x10,0xFF,0x10,0xAD}`, version=2, per-entry
  `ImageKind`/`OffloadKind`/triple/arch/StringMap. Full spec in wave-06-llvm-offload-new-driver.md.
- `olCreateProgram → olGetSymbol → olLaunchKernel` sequence: confirmed from
  PR #122106 (merged Apr 2025) and PR #147943 (merged Jul 2025, `olGetKernel` rename).
  Documented in wave-04-liboffload-multiversion.md.
- `parseOffloadBinary` loop in `PluginInterface.cpp` (PR #186088, open Mar 2026):
  iterates all inner images, calls `isMetadataCompatible` + `isDeviceCompatible`,
  breaks on first match — no timing instrumentation present.
- OMPT device hooks: "OMPT Device Support in LLVM" (Dhruva Chakrabarti, AMD,
  GPU/Offloading Workshop 2025) — exists at the OpenMP semantic layer but not
  wired to raw `ol*` API path. Issue #110007 tracks expansion.
  Source: wave-07-llvm-devmtg-gpu-landscape.md line 144.
- `kdl_get_dispatch_latency_ns()` prototype: `kdl.c` lines 4595–4649 —
  `clock_gettime(CLOCK_MONOTONIC)` bracketing `cuStreamSynchronize` over 100 reps.
  This is the measurement harness to extend.
- accfg poster: ETH Zurich Research Collection
  `https://www.research-collection.ethz.ch/server/api/core/bitstreams/1a209417-a8b9-42b6-9600-4031ced603b2/content`.
  Targets compile-time overhead elimination; different problem. Source: wave-07-llvm-poster-criteria.md.

### What is confirmed absent in the literature

1. No paper has published a per-layer microsecond breakdown of the LLVM offload
   dispatch path (OffloadBinary parse → plugin load → driver API → GPU).
2. PR #186088's `parseOffloadBinary` loop has zero instrumentation; the community
   does not know how long image-compatibility checking takes per variant.
3. Issue #110007 (OMPT device hooks) is open; OMPT does not yet reach the `ol*`
   raw dispatch path.
4. Level Zero dispatch latency: "significantly reduced overhead" claimed for the
   L0 v2 adapter (wave-04-level-zero.md) — zero microsecond figures published.

---

## Feasibility

**Hardware available:** GTX 1650 (CUDA 12.x), CPU (OpenMP/host plugin).

**Implementation cost: Low.** The measurement harness is ~200 LOC C extending
`kdl_get_dispatch_latency_ns()`. Key steps:

1. Extend `kdl.c` with explicit `clock_gettime` brackets around each `ol*` call.
2. Build liboffload from LLVM source (already done for the prototype context).
3. Compile a null CUDA kernel to CUBIN; embed in OffloadBinary container using
   `clang-offload-packager`.
4. Run 10,000 dispatches; record per-dispatch latency for each layer.
5. Pipe per-call stack-fold data to `flamegraph.pl` using the Brendan Gregg format:

```
# flamegraph input format (one sample per line, semicolon-separated frames):
OffloadBinary::create;olCreateProgram;cuModuleLoadData 45210
olGetSymbol;cuModuleGetFunction 1203
olLaunchKernel;cuLaunchKernel 128440
```

6. Generate cold-path and hot-path flame graphs as two SVG panels on the poster.

**Risk factors:**

- `olCreateProgram` and layer 1-4 timings are one-time costs; must separate cold
  (first dispatch) from hot (amortized) measurement explicitly — otherwise the cold
  PTX JIT cost (potentially 10–100 ms) swamps the graph.
- liboffload API is explicitly unstable (PR #122106 body warning); `olGetKernel`
  renamed to `olGetSymbol` in July 2025 (PR #147943). Pin to a specific LLVM commit.
- GTX 1650 is consumer hardware; H100 numbers from TaxBreak cannot be directly
  replicated. State clearly: "relative layer fractions, not absolute values,
  generalize across hardware."

**Overall feasibility: High.** The prototype timing infrastructure already exists
in `kdl.c`. The LLVM offload stack build is already set up. This is a measurement
study, not a compiler change. Estimated work: 1–2 weeks of focused effort.

---

## Upstream Path

This work does not require an upstream code change to land — the measurement is
external instrumentation. However, it creates a strong case for two upstream
contributions:

| Artifact | Location | Status |
|----------|----------|--------|
| Per-layer timing annotations in `parseOffloadBinary` | `offload/plugins-nextgen/common/PluginInterface.cpp` | Patch opportunity on top of PR #186088 |
| OMPT device hooks for `ol*` API path | `offload/liboffload/` | Extends Issue #110007 scope |
| Flame graph benchmark in LLVM test-suite | `llvm-test-suite/MicroBenchmarks/GPU/dispatch-overhead/` | New contribution |

The poster result (published latency numbers per layer) provides the quantitative
motivation for upstreaming the OMPT hook extension to Issue #110007. Citing Joseph
Huber's 2025 DevMtg framing — "ld.so for GPU code" — positions this measurement as
filling an observability gap his own team acknowledged.

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 8/10 | No per-layer latency breakdown published for LLVM offload stack. TaxBreak covers the floor; this covers the interior. |
| **Feasibility** | 9/10 | Measurement study only. Prototype harness already exists in `kdl.c`. ~200 LOC extension. GTX 1650 available. |
| **Evidence** | 9/10 | TaxBreak (floor), PyGraph (ceiling), ICPP 2019 (methodology), PR #186088 (gap in instrumentation), OMPT Issue #110007 all independently confirm the gap. |
| **Community fit** | 8/10 | LLVM DevMtg audience is performance-conscious. "Numbers that survive a sharp question" (wave-07-poster-criteria.md). Flame graphs are universally legible. |
| **Impact** | 7/10 | Measurement-only; no code change. High visibility as the first published number, but lower long-term impact than a patch. Becomes the motivation for a follow-up OMPT PR. |
| **Composite** | **8.2/10** | |

---

## Upstream Path (Summary)

The immediate deliverable is a published latency table + two flame graph SVGs
(cold path / hot path). The upstream story in three steps:

1. **Poster (Dublin 2026):** publish the numbers; cite PR #186088 gap explicitly.
2. **Patch (post-poster):** add per-layer `__llvm_offload_trace_*` hooks to
   `PluginInterface.cpp` following the same virtual-method pattern as
   `isMetadataCompatible` (PR #185663). Propose on the biweekly offload call.
3. **Test-suite addition:** add the null-kernel dispatch microbenchmark to
   `llvm-test-suite` so the community can track regressions across LLVM releases.

---

## One-Sentence Pitch

"We instrument every layer of the LLVM GPU dispatch path — from OffloadBinary
parse through `olCreateProgram`, plugin, `cuModuleLoadData`, and `cuLaunchKernel`
— producing the first published flame graph that shows where the 4.71 μs H100
dispatch floor actually comes from, and confirming that libkdl's O(1) variant
lookup contributes less than 2% of that budget."

---

## Cross-References

- wave-03-dispatch-overhead.md — TaxBreak measurements, ICPP 2019 baseline
- wave-02-dispatch-overhead-benchmarks.md — HIP dispatch gap, Kokkos #3670,
  OpenCL/Numba overhead table
- wave-04-liboffload-multiversion.md — `ol*` API sequence, PR #186088
  `parseOffloadBinary` loop, OMPT Issue #110007 reference
- wave-06-llvm-offload-new-driver.md — OffloadBinary format spec, layer 1 details
- wave-07-llvm-poster-criteria.md — accfg poster context, "numbers that survive"
  requirement, R7 accfg comparison readiness note
- wave-07-llvm-devmtg-gpu-landscape.md — OMPT Device Support talk (Dhruva Chakrabarti)
- directions/04-empirical-dispatch-overhead.md — head-to-head cross-runtime
  measurement direction; this topic is the LLVM-stack-specific sub-case of that
- `experiments/prototype/src/kdl.c` lines 4595–4649 — existing `clock_gettime`
  dispatch timing infrastructure to extend
