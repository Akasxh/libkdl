# Topic 11: MLIR-Native Kernel Autotuning Infrastructure

**Survey:** 20 LLVM/MLIR Poster Topics
**Topic ID:** 11
**Config name:** mlir-native-kernel-autotuning
**Title:** MLIR-Native Kernel Autotuning: Closing the Gap Between IR-Level Parameters and Search
**Persona:** Compiler architect / ML systems researcher
**Date:** 2026-04-07
**Research depth:** Exhaustive — Triton @autotune internals, TVM MetaSchedule, IREE SHARK Tuner,
  Transform dialect Tune Extension, MLIR cost model RFC, plus local literature cross-reference

---

## Gap

Triton, TVM, and Halide each have production autotuners that search over tile sizes, block
dimensions, pipeline stages, and other hardware parameters. MLIR — the compiler infrastructure
underlying Triton and much of modern ML compilation — has no equivalent.

The gap is not merely aesthetic. It manifests in three concrete failure modes:

### 1. Tunable Parameters Exist at the IR Level but Are Invisible to Search

MLIR's Transform dialect (as of LLVM 20) exposes:
- `transform.tune.knob` — a tunable parameter within a transform sequence
- `transform.tune.alternatives` — try successive transformation sequences, revert on failure
- `transform.structured.tile_using_for` / `transform.structured.tile_using_forall` — tiling
  with `!transform.param<T>`-typed sizes that flow as IR values

These building blocks allow a transformation author to *express* that a tile size is tunable.
They do not provide:
- A search driver that iterates over candidate parameter values
- A cost model or measurement harness that evaluates each candidate
- A caching layer that stores tuning results keyed by (kernel, hardware, shape)
- A cross-hardware database for transferring tuning results across devices

The Transform dialect tune extension is a *mechanism* for expressing alternatives, not an
*autotuner*. The actual parameter search must be wired externally and no standard wiring exists.

### 2. Cost Information Is Computed but Not Retained

MLIR computes optimization-relevant quantities during lowering:
- Linalg tiling decisions encode tile sizes (discarded after the pass)
- The KernelInfo analysis pass (PR #102944, ORNL, merged January 2025) computes
  `AllocasStaticSizeSum`, `FlatAddrspaceAccesses`, and thread-limit hints — as stderr remarks,
  not IR metadata
- The MLIR cost model RFC (discourse.llvm.org/t/rfc-target-description-and-cost-model-in-mlir/76990,
  Intel PCL, 2024) concluded that "MLIR lacks a standard cross-target cost model interface"
  and proposed one — but it was not accepted upstream and remains unimplemented

The net result: there is no standard way to (a) annotate a gpu.func with its tunable parameters,
(b) evaluate the quality of a given parameter assignment, or (c) record and reuse the result.

### 3. Every Downstream System Reimplements Autotuning Independently

| System | Autotuning mechanism | Portable? | Upstream? |
|--------|---------------------|-----------|-----------|
| Triton | `@triton.autotune` + per-target cache in `~/.triton/cache/` | No (per-vendor) | No |
| TVM MetaSchedule | Evolutionary search + XGBoost cost model + SQLite DB | No (per-device) | No |
| Halide 2019 autoscheduler | Beam search + gradient-boosted tree, 70K training programs | No (per-target) | No |
| IREE SHARK Tuner | Transform dialect spec + measurement on device, ~10% win on MI300X SDXL | No (IREE-only) | No |
| cuBLAS kernel selector | ML-trained heuristic, 93% optimal, ~5-20 μs selection latency | No (NVIDIA-only) | No |
| AMD MIOpen | Explicit benchmark + per-arch SQLite cache in `~/.config/miopen/` | No (AMD-only) | No |

Every autotuner reinvents: (a) the parameter space representation, (b) the measurement harness,
(c) the result cache format, and (d) the policy for when to retune. None of these are
interoperable. An NVIDIA-tuned kernel database cannot seed AMD tuning. A Triton cache cannot
inform an IREE dispatch decision.

The absence of a common MLIR-level autotuning abstraction forces this proliferation.

---

## Proposal

**Title:** `mlir-autotune` — A Tunable Parameter Dialect and Search Tool for GPU Kernels

**One-sentence pitch:** Introduce a `tunable` MLIR dialect that annotates `gpu.func` operations
with typed parameter spaces (tile sizes, block dims, pipeline stages), add a standalone
`mlir-autotune` tool that searches those spaces using measurement or cost model evaluation, and
emit the tuned parameters back into the IR as concrete values — making autotuning a first-class
compiler pass composable with any MLIR pipeline.

### Part A — The `tunable` Dialect (IR-Level Parameter Spaces)

Introduce a new dialect (or extend `gpu` dialect attributes) that lets a kernel author annotate
tunable parameters at the `gpu.func` or `linalg.generic` level:

```mlir
gpu.func @matmul_kernel(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
    kernel
    attributes {
      tunable.params = #tunable.param_space<[
        #tunable.int_range<"tile_m", min=16, max=256, step=16, default=128>,
        #tunable.int_range<"tile_n", min=16, max=256, step=16, default=128>,
        #tunable.int_range<"tile_k", min=8,  max=128, step=8,  default=32>,
        #tunable.int_range<"num_warps", values=[2, 4, 8], default=4>,
        #tunable.bool<"use_tensor_cores", default=true>
      ]>
    }
```

The `#tunable.param_space` attribute is a structured, queryable description of the search space.
Key design choices:
- Parameters are named, typed, and constrained (range/step or explicit set)
- Each parameter has a default that produces a valid (not necessarily optimal) kernel
- The attribute attaches to the kernel operation, not to a global configuration file
- Parameters can reference hardware features via symbolic queries:
  `#tunable.int_range<"tile_m", max=#target.warp_size * 8>` — the bound is computed
  from a target attribute at specialization time

This is directly analogous to Triton's `triton.Config` objects but expressed in IR rather than
Python, making them accessible to any MLIR pass or tool without invoking the Python runtime.

### Part B — The `mlir-autotune` Tool (Search + Evaluation)

A standalone tool in the MLIR toolchain, analogous to `mlir-opt` but for parameter search:

```bash
mlir-autotune input.mlir \
  --target=cuda:sm_90 \
  --search=evolutionary \
  --evaluator=measurement \
  --budget=200 \
  --output-db=tuning.json \
  --output-ir=tuned.mlir
```

**Evaluation backends** (pluggable via a `TuningEvaluator` interface):

1. **Measurement evaluator** — instantiates the kernel with candidate parameters, compiles it
   (via `mlir-opt` pipeline + driver API), runs it on the attached device, records wall time.
   This is what Triton's autotuner and TVM MetaSchedule do. Expensive but accurate.

2. **Cost model evaluator** — uses a roofline estimate or learned predictor to score candidates
   without compilation. The roofline model (max(T_compute, T_memory) / efficiency) achieves
   94.7% of exhaustive tuning quality for GEMM-class kernels (tritonBLAS, arXiv:2512.04226).
   Fast (<1 ms per candidate) but requires a hardware feature database.

3. **Hybrid evaluator** — cost model prunes the space, measurement evaluates the survivors.
   This is the pattern used by cuDNN (Mode_INSTANT for fast path, Mode_A for full ML inference).

**Search strategies** (pluggable via a `TuningSearcher` interface):

- Grid search (exhaustive, baseline)
- Evolutionary search (TVM MetaSchedule's approach: mutation + selection over parameter vectors)
- Bayesian optimization (Gaussian process surrogate, more sample-efficient)
- Beam search (Halide 2019 autoscheduler approach, beam width configurable)

The tool emits a `tuning.json` database (keyed by `{kernel_hash, device_id, input_shape_hash}`)
and optionally writes the tuned parameters back into the IR as concrete `arith.constant` values,
replacing the `#tunable.param_space` attribute.

### Part C — `gpu-module-to-binary` Integration

When `gpu-module-to-binary` encounters a `gpu.func` with `tunable.params`, it:
1. Checks the tuning database for a result matching `{kernel_hash, target_arch, input_shape}`
2. If found: instantiates the kernel with tuned parameters and compiles
3. If not found: uses default parameter values, emits a warning, tags the binary with
   `#tunable.not_tuned` for monitoring

This makes tuning database lookup part of the standard compilation pipeline, not a separate
manual step — the same relationship that TVM MetaSchedule has to TVM's compilation.

### Part D — Runtime Parameter Query (Future Extension)

For dynamic shapes, static tuning is insufficient. The proposal includes a hook for runtime
parameter selection:

```c
// In the generated host code, before gpu.launch_func:
kdl_params_t params = kdl_select_params(
    kernel_hash, device_id, M, N, K,  // input shape at runtime
    &tuning_db                         // loaded from tuning.json at init
);
// params.tile_m, params.tile_n, params.tile_k fed to the launch config
```

This mirrors IREE's SHARK Tuner + Transform dialect spec integration, but via a minimal C API
that does not require the full IREE runtime.

---

## Evidence

### Triton `@triton.autotune` — Mechanism

The `@triton.autotune` decorator (triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
works as follows:

1. The user supplies a list of `triton.Config` objects, each specifying values for meta-parameters
   (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_warps`, `num_stages`,
   `matrix_instr_nonkdim`) and a `key` list (problem dimensions like `M`, `N`, `K`).
2. On first call for a given `key` value combination, Triton benchmarks all configs by compiling
   and running the kernel, recording wall time per config.
3. The fastest config is cached, keyed by `{backend_hash, config_hash, key_values}`, in
   `$HOME/.triton/cache/`.
4. On subsequent calls with the same key values, the cached best config is used — zero search cost.
5. Cache is invalidated if `TRITON_CACHE_DIR` changes, backend changes, or source hash changes.

**Critical limitation:** The cache is per-target. A config optimized for `cuda:sm_90` is never
used for `hip:gfx942`. Each vendor requires a fresh search. The search is also purely empirical —
no cost model prunes the search space, so all configs in the list are benchmarked regardless of
how unlikely they are to perform well.

**What Triton does not have:**
- An IR-level representation of the parameter space (configs are Python objects, not IR attributes)
- A cost model that could rank configs before benchmarking
- A cross-architecture transfer mechanism (A100-optimal configs as priors for H100 search)
- Integration with the MLIR pass pipeline (configs are consumed by the Python JIT compiler,
  not by MLIR passes that could then lower to different targets)

URL: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
URL: https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py

### TVM AutoScheduler/MetaSchedule — Compilation-Level Autotuning

**Ansor (OSDI 2020):** Hierarchical search space construction + evolutionary search + learned
XGBoost cost model trained on hardware measurement data. Achieves 3.8x over Intel CPUs, 1.7x
over NVIDIA GPUs vs AutoTVM templates. The cost model is hardware-specific and must be retrained
per device (e.g., A100 model does not transfer to RX 7900 XTX).
URL: https://arxiv.org/abs/2006.06762

**MetaSchedule (NeurIPS 2022):** Introduces stochastic schedule primitives — the schedule space
is expressed as a probabilistic program where tile sizes, loop reordering, and memory hierarchy
choices are random variables. Evolutionary search samples this space; the cost model predicts
which samples are worth measuring. Key innovation: the cost model is decoupled from the schedule
representation, enabling fine-grained human guidance. The result is stored in a SQLite database
per (operator, target_hardware) pair and used in subsequent compilations.
URL: https://openreview.net/forum?id=nyCr6-0hinG (paywall); local summary: literature/tvm-unity-multi-target.md#5

**DLight (tuning-free mode):** Uses device properties (compute capability, memory bandwidth) to
select tile sizes heuristically without measurement. Trades ~20-40% performance for zero tuning
time. The primary use case is LLM inference with dynamic shapes where MetaSchedule's static-shape
assumption breaks down.

**MLIR integration:** TVM does not use MLIR as its core IR. TVM's optimization pipeline operates
on Relax/TIR, not MLIR dialects. There is no MLIR lowering path in production TVM.

### IREE SHARK Tuner — Dispatch Region Autotuning

IREE's SHARK Tuner (iree.dev/reference/tuning/) provides the closest existing example of
MLIR-integrated autotuning:

1. The IREE compiler creates "dispatch regions" — atomic blocks of computation.
2. Each dispatch exposes "knobs": subgroup tile sizes, workgroup thread counts, MMA layouts,
   reduction dimensions.
3. SHARK Tuner searches these knobs via trial-and-error across candidate values, measuring on
   the actual device.
4. The winning configuration is encoded as a Transform dialect spec (an MLIR file using
   `transform.tune.knob` and related ops).
5. The spec is loaded at recompile time via `--iree-codegen-tuning-spec-path`.
6. Result: ~10% improvement on SDXL (Stable Diffusion XL) inference on MI300X.

**Key limitation:** The knob encoding is IREE-specific. The tuning spec format (a Transform
dialect MLIR file) is not consumed by any other MLIR-based system. The SHARK Tuner itself is
an IREE project tool, not an upstream `mlir-autotune` tool. The approach is correct but
vertically integrated — it cannot tune kernels that do not go through the IREE compiler.

URL: https://iree.dev/reference/tuning/

### MLIR Transform Dialect Tune Extension

The Transform dialect (mlir.llvm.org/docs/Dialects/Transform/) as of LLVM 20 includes a Tune
Extension with:

- `transform.tune.alternatives` — n regions, each a transform sequence; attempts them in order,
  reverts via IR cloning on failure, returns after first success
- `transform.tune.knob` — a tunable parameter within a transform sequence

These operations allow a transformation author to express "try these tile sizes in this order,
use whichever compiles." However:
- The alternatives are tried in order, not searched — no measurement, no cost model
- The knob type is a compile-time constant, not a dynamic search variable
- There is no standard interface for an external search driver to populate knob values
- The alternatives mechanism is designed for robustness (handle compilation failures), not
  performance optimization (find the best config)

This is a useful foundation but is not an autotuner. The missing piece is a search driver that
(a) enumerates candidate parameter assignments, (b) evaluates each, and (c) commits the best.

URL: https://mlir.llvm.org/docs/Dialects/Transform/

### MLIR Cost Model RFC (Intel PCL, 2024)

discourse.llvm.org/t/rfc-target-description-and-cost-model-in-mlir/76990

Local reference: research/mega-survey/20-poster-topics/waves/topic-03-dispatch-cost-attr.md:380

The RFC concluded: "MLIR lacks a standard cross-target cost model interface." The proposal
attempted to define target description attributes and cost model interfaces for use during
optimization passes. It was not accepted upstream. The gap it identified — the absence of
a standard way to estimate operation cost in MLIR — remains open.

This RFC is directly relevant because an autotuner's cost model evaluator would plug into exactly
the interface the RFC proposed. The `mlir-autotune` proposal can cite this RFC as prior
motivation and propose a narrower, kernel-level cost interface rather than a general cost model.

### GPU Portability Needs Autotuning (arXiv:2505.03780)

Demonstrates that autotuned vendor-agnostic kernel implementations exceed hand-tuned vendor
implementations by >230% in throughput on AMD MI300X for LLM attention kernels. The key
mechanism: the tuner explores tile configurations unavailable in vendor libraries. The code size
reduction is 70x (eliminates manual per-shape specializations).

This paper is the strongest available evidence that autotuning is not a secondary optimization
but is often the only path to peak performance on non-primary hardware (e.g., AMD for kernels
designed on NVIDIA). An MLIR-native autotuner that operates before vendor-library selection
would generalize this finding across any MLIR-compiled kernel.
URL: https://arxiv.org/abs/2505.03780
Local reference: literature/cost-models-kernel-dispatch.md:282-290

### MLIR Transform Dialect TilingInterface

MLIR's linalg dialect defines parametric tiling as a core abstraction. The `TilingInterface`
(referenced in mlir.llvm.org docs) lets operations declare their tile-able loop dimensions.
The `transform.structured.tile_using_for` and `transform.structured.tile_using_forall` ops
accept `!transform.param<index>` values for tile sizes — IR-level parameters that can be
computed by preceding transforms. This means tile sizes can be parameterized at the MLIR level
today; the missing piece is the search infrastructure that selects parameter values.

---

## What `mlir-autotune` Would Look Like

A concrete tool design, grounded in the analysis above:

```
mlir-autotune [options] input.mlir

Required:
  input.mlir                     MLIR file with gpu.func ops annotated with tunable.params

Target selection:
  --target=<spec>                Target device (cuda:sm_90, hip:gfx942, cpu:x86_64)

Search:
  --search=grid|evolutionary|bayes|beam
                                 Search strategy (default: evolutionary)
  --budget=N                     Maximum number of configs to evaluate (default: 100)
  --seed=N                       Random seed for reproducibility

Evaluation:
  --evaluator=measurement|cost-model|hybrid
                                 Candidate scoring method (default: hybrid)
  --hardware-db=path             Device feature database for cost model evaluator
                                 (peak TFLOPS, bandwidth, cache sizes per device ID)
  --warmup=N                     Measurement warmup iterations (default: 3)
  --trials=N                     Measurement trials per config (default: 10)

Database:
  --db=path                      Tuning result database (JSON, read+write)
  --db-read-only                 Only read from db, don't write new results

Output:
  --output-ir=path               Write IR with tuned parameters instantiated
  --output-report=path           Write search result table (config, time, rank)
  --emit-untuned-warning         Warn when no db entry found for a kernel (default: on)
```

**Execution model:**

```
Input MLIR (with tunable.params attributes)
    |
    v
1. Parse tunable.params from each gpu.func → build parameter space per kernel
    |
    v
2. Generate candidate configs (search strategy chooses N configs from space)
    |
    v
3. For each candidate config:
   a. Instantiate parameters in a cloned module (replace tunable.params with constants)
   b. Run evaluator: measurement (compile + time) or cost model (roofline estimate)
   c. Record (config_vector, score) in result set
    |
    v
4. Select best config per kernel
    |
    v
5. Write to --db: {kernel_hash, target_arch, input_shape_hash} → best_config_vector
    |
    v
6. Emit --output-ir: original IR with tunable.params replaced by tuned constants
         original IR with #tunable.not_tuned attr preserved if evaluation failed
```

**Database schema (JSON, human-readable for easy inspection):**

```json
{
  "schema_version": 1,
  "entries": [
    {
      "kernel_hash": "sha256:abc...",
      "kernel_name": "matmul_kernel",
      "target_arch": "cuda:sm_90",
      "input_shape": {"M": 4096, "N": 4096, "K": 4096},
      "best_config": {"tile_m": 128, "tile_n": 128, "tile_k": 32, "num_warps": 4},
      "best_time_us": 217.3,
      "search_budget_used": 47,
      "tuned_at": "2026-04-07T12:00:00Z",
      "device_name": "NVIDIA H100 SXM5"
    }
  ]
}
```

The schema is intentionally simple and vendor-neutral. The same database format works for
CUDA, HIP, and CPU kernels. Cross-vendor seeding becomes possible: seed HIP tuning with CUDA
results by filtering for parameter constraints compatible with the target architecture.

---

## Feasibility

**High-confidence path (demo-ready for Dublin poster):**

The proof-of-concept requires no new MLIR dialect. It can be implemented as:

1. A Python script that:
   - Reads a `config.toml` specifying parameter spaces for named kernels
   - Instantiates candidates by invoking `mlir-opt` with different `-D` defines
   - Compiles via the existing GPU pipeline and times on the available hardware (GTX 1650 + CPU)
   - Outputs a `tuning.json` database and a "best config" summary

2. A demonstration showing:
   - A `linalg.matmul` lowered to `gpu.func` with tile parameters
   - Tuning over {tile_m, tile_n, tile_k, num_warps} with budget=50
   - Comparison: default config vs tuned config on GTX 1650
   - Runtime: roofline cost model scoring vs measurement scoring

This is a ~300-line Python + ~100-line MLIR file effort. No upstream MLIR changes required.

**Medium-confidence path (full proposal for discussion):**

The `tunable` dialect extension requires:
- A new `#tunable.param_space` attribute (registerable as a standalone attribute in a
  downstream project, no upstream MLIR change required for prototype)
- A `KernelParamAnnotator` pass that attaches param spaces from a TOML config
- The `mlir-autotune` binary (~500 LOC C++ driver wrapping existing MLIR infrastructure)
- Database I/O using nlohmann/json (already available in most MLIR builds via third-party)

Estimated effort: 2 weeks for a working prototype, 2 months for a clean implementation
suitable for RFC submission.

**Upstream path blockers:**

1. The Transform dialect tune extension is upstream but lacks a standard search driver interface.
   Adding one requires a new op (`transform.tune.run_search`) and a plugin mechanism for
   `TuningSearcher` implementations. This is RFC-level work, not a patch.

2. The MLIR cost model RFC (Intel PCL, 2024) was not accepted. The `mlir-autotune` proposal
   would revive a narrower version: a kernel-level cost model interface rather than a general
   cross-target cost model. The narrower scope may be more acceptable upstream.

3. The `tunable` dialect name collides with no existing MLIR dialect. However, it needs a
   clear specification of when `tunable.params` is consumed (at `mlir-autotune` time, replaced
   by constants before `gpu-module-to-binary`). This serialization contract needs an RFC.

---

## Upstream Path

### Stage 1 — Out-of-tree prototype (now → poster deadline)

- Implement `mlir-autotune.py` as a Python wrapper using subprocess + `mlir-opt`
- Demonstrate tuning a `linalg.matmul` lowered to GPU on GTX 1650 and CPU
- Publish tuning results (speedup table, budget curve, cost-model vs measurement comparison)
- Present at Dublin poster session with live demo on laptop

### Stage 2 — RFC: `transform.tune` search driver interface (post-poster)

- File RFC on LLVM Discourse: "Adding a search driver to transform.tune for autotuning"
- Propose `TuningEvaluator` and `TuningSearcher` C++ interfaces as extensible plugins
- Anchor to IREE SHARK Tuner as prior art (same problem, IREE-only solution)
- Anchor to MLIR cost model RFC (76990) as motivation for cost model evaluator
- Stakeholders: IREE team (nod-ai, Google), Intel XeGPU team (Morel/Karna — presenting at
  EuroLLVM 2026 MLIR Workshop on Intel GPU autotuning), AMD GPU compiler team

### Stage 3 — Upstream `tunable` dialect attributes (6-12 months post-RFC)

- Introduce `#tunable.param_space` as a `gpu` dialect attribute extension or a separate
  small dialect (following `amdgpu`/`nvgpu` pattern: vendor-neutral operations in `gpu`,
  vendor-specific in separate dialect)
- Path for acceptance: submit as an out-of-tree dialect with `mlir-autotune` driver, propose
  upstreaming after 3+ users demonstrate value (same pattern as `nvgpu`, `xegpu`)

### Stage 4 — Tuning database standardization (12-24 months)

- Propose a standard tuning database format (JSON schema + SQLite profile) as an MLIR project
  artifact, analogous to how LLVM has a standard profile format for PGO
- Cross-vendor seeding: AMD kernels seeded from NVIDIA tuning results for the same operator
- Anchor to MIOpen's per-arch SQLite cache as prior art for the database model

**Key upstream stakeholders:**
- Tuomas Karna, Rolf Morel (Intel) — presenting "Auto-tuning MLIR schedules for Intel GPUs"
  at EuroLLVM 2026 MLIR Workshop (April 13) — natural allies for Stage 2 RFC
- IREE/nod-ai team — SHARK Tuner is the direct prior art; they have motivation to generalize it
- Matthias Springer, Lorenzo Chelini (NVIDIA) — presenting "CUDA Tile IR" at EuroLLVM 2026 —
  Tile IR is the vendor-specific analog; cross-vendor generalization is differentiated
- AMD GPU compiler team — rocMLIR presenting at EuroLLVM 2026 main conference

---

## Novelty / Interest Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **9/10** | No upstream MLIR autotuning infrastructure exists. The Transform tune extension provides building blocks but not a search driver. Every existing autotuner (Triton, TVM, IREE SHARK) is vertically integrated and non-interoperable. A standard MLIR-level parameter space representation + standalone search tool would be genuinely new. |
| **Technical Interest** | **8/10** | The gap between "Transform dialect can express alternatives" and "there is an actual autotuner" is a clean, concrete problem with a well-scoped solution. The cost model vs. measurement evaluator trade-off is technically interesting and has quantified data (94.7% quality at 0 compile cost). |
| **Community Relevance** | **9/10** | Intel is presenting at the EuroLLVM 2026 MLIR Workshop on exactly this topic for Intel GPUs. This proposal generalizes their contribution across vendors. The audience for this poster includes every GPU compiler engineer at the conference. |
| **Feasibility** | **8/10** | The proof-of-concept is achievable in 1 week with no upstream changes. The full implementation is 4-8 weeks. The upstream path has clear stakeholders and a precedent (IREE SHARK Tuner as the existence proof). |
| **Impact** | **9/10** | Addresses a gap cited by three independent RFCs (cost model RFC, distributed hetero computing RFC, GPU dialect cleanup RFC). Enables cross-vendor tuning transfer, reduces per-deployment tuning cost, and provides the missing abstraction layer between MLIR compilation and production deployment. |
| **Composite** | **8.6/10** | Strong candidate. Especially strong given the conference timing — Intel GPU autotuning is being presented the day before the poster session. |

---

## Risks

1. **"IREE already does this"** objection. The counter: IREE's SHARK Tuner is vertically
   integrated (IREE compiler only, IREE runtime only). The proposal targets the MLIR level,
   not the IREE level. An `mlir-autotune` tool would work with any MLIR-based compiler
   (Triton, rocMLIR, the proposed Tile IR) without requiring the IREE runtime.

2. **"TVM MetaSchedule already does this"** objection. The counter: TVM does not use MLIR.
   MetaSchedule's search infrastructure is TVM-specific (Relax/TIR, not gpu.func operations).
   An MLIR-native autotuner would serve the majority of the ML compilation ecosystem that
   has moved to MLIR without adopting TVM's full stack.

3. **Scope creep.** The `mlir-autotune` proposal risks becoming a research project rather than
   a concrete tool. The Dublin poster must present measured results, not a design document.
   Mitigation: bound the prototype to a single operator (linalg.matmul → gpu.func) and a single
   target (GTX 1650), with a clear demo script. Results > design.

4. **Transform dialect Tune Extension instability.** The Transform dialect is marked as "more
   likely to change than others." Building on `transform.tune.knob` / `transform.tune.alternatives`
   risks API churn before upstream acceptance. Mitigation: prototype without the Tune Extension
   (use Python-level parameter substitution), and propose the Tune Extension integration as a
   future milestone.

5. **Intel GPU autotuning talk at MLIR Workshop overlaps directly.** Karna/Morel are presenting
   at the MLIR Workshop on April 13, one day before the poster session. If their talk is strong,
   it may pre-empt the novelty claim. Mitigation: position as cross-vendor generalization of
   their Intel-specific work, not as competition. Propose collaboration in the poster pitch.

---

## Draft Pitch

**Three-sentence poster pitch:**

Triton and TVM each have autotuners that search over tile sizes and block dimensions — but they
are vertically integrated, non-interoperable, and built outside MLIR. MLIR's Transform dialect
can express tunable parameters (`transform.tune.knob`, `transform.tune.alternatives`) but has no
search driver: the Intel-specific SHARK Tuner pattern exists, but no general tool produces and
consumes a tuning database across vendors. We propose `mlir-autotune`: a standalone tool that
reads `#tunable.param_space` annotations on `gpu.func` operations, searches tile size and
block dimension assignments using a roofline cost model pre-filter (achieving 94.7% of
exhaustive quality with <100 candidates), and writes a vendor-neutral tuning database — making
autotuning a first-class, composable step in any MLIR-based GPU compilation pipeline.

**Poster panel layout (6 panels):**

1. **The gap diagram** — shows Triton, TVM, IREE each with their own autotuning silo; no
   shared abstraction; cross-vendor transfer is impossible
2. **The `#tunable.param_space` attribute** — IR-level parameter space representation on
   `gpu.func`; comparison to Triton's Python `Config` objects
3. **`mlir-autotune` architecture** — search driver, pluggable evaluator (cost model vs.
   measurement), database I/O
4. **Cost model accuracy** — cites tritonBLAS 94.7% result; shows budget curve (quality vs.
   number of candidates evaluated) on GTX 1650 demo
5. **Cross-vendor transfer** — shows how a tuning DB entry for `cuda:sm_80` can seed
   `hip:gfx942` search (same operator, different architecture constraints)
6. **Upstream path** — Transform dialect Tune Extension, Intel GPU autotuning alignment,
   Stage 1-4 roadmap; call for collaboration

**Target audience at Dublin:**
- Intel GPU compiler engineers (Karna/Morel team — direct collaboration target)
- IREE contributors (generalize SHARK Tuner upstream)
- AMD rocMLIR team (cross-vendor tuning transfer directly relevant)
- Anyone who has written `@triton.autotune` and wished it worked on AMD too

---

## Key References

| Reference | Relevance | URL / Location |
|-----------|-----------|----------------|
| Triton @autotune decorator | Direct prior art — defines the user-facing API to generalize | https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html |
| TVM MetaSchedule (NeurIPS 2022) | Stochastic schedule search, cost model decoupling | https://openreview.net/forum?id=nyCr6-0hinG |
| Ansor (OSDI 2020) | Hierarchical search space, XGBoost cost model | https://arxiv.org/abs/2006.06762 |
| IREE SHARK Tuner | Direct prior art — MLIR-integrated autotuning, IREE-only | https://iree.dev/reference/tuning/ |
| Transform dialect Tune Extension | `transform.tune.knob`, `transform.tune.alternatives` | https://mlir.llvm.org/docs/Dialects/Transform/ |
| MLIR Cost Model RFC | Concluded MLIR lacks standard cost model interface | https://discourse.llvm.org/t/rfc-target-description-and-cost-model-in-mlir/76990 |
| tritonBLAS (arXiv:2512.04226) | 94.7% quality with roofline at 3 numbers | Local: research/mega-survey/20-poster-topics/waves/topic-03-dispatch-cost-attr.md |
| GPU Portability Needs Autotuning (arXiv:2505.03780) | >230% improvement, 70x code reduction | Local: literature/cost-models-kernel-dispatch.md:282-290 |
| Halide autoscheduler (SIGGRAPH 2019) | Beam search + learned cost model, 70K programs | https://halide-lang.org/papers/halide_autoscheduler_2019.pdf |
| cuBLAS kernel selector | 93% optimal ML heuristic, production-scale reference | https://docs.nvidia.com/cuda/cublas/ |
| MIOpen per-arch SQLite cache | Per-arch tuning DB pattern, production AMD reference | https://rocm.docs.amd.com/projects/MIOpen/ |
| EuroLLVM 2026 MLIR Workshop | Karna/Morel Intel GPU autotuning talk — convergence point | https://discourse.llvm.org/t/announcing-the-7th-mlir-workshop-eurollvm-2026-program/90119 |
| KernelInfo pass (PR #102944) | ORNL analysis pass — existing upstream cost analysis | https://github.com/llvm/llvm-project/pull/102944 |

---

*Cross-references:*
- `literature/cost-models-kernel-dispatch.md` — roofline model, Halide, TVM MetaSchedule, cuBLAS
- `literature/triton-compiler-approaches.md` — Triton @autotune internals, cache architecture
- `literature/tvm-unity-multi-target.md#5` — MetaSchedule, DLight, SQLite DB pattern
- `research/mega-survey/20-poster-topics/waves/topic-03-dispatch-cost-attr.md` — MLIR cost RFC,
  KernelInfo, tritonBLAS 94.7%, `#gpu.cost_hint` upstream path (complementary proposal)
- `notes/novelty-gaps.md#Gap5` — Capability-aware JIT for MLIR kernels (related gap)
- `research/poster-strategy-research.md#2.1` — EuroLLVM 2026 MLIR Workshop program context
