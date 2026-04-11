# Direction 01: libkdl as Policy Layer Above liboffload

**Final Score: 9.50/10** (Rank #1 of 6)
**Scoring History:** Round 1: 8.75 → Round 2: 9.00 → Final: 9.50

---

## One-Sentence Description

libkdl adds multi-version kernel selection, capability-contract matching, and roofline cost scoring on top of LLVM's liboffload mechanism API, filling the policy gap the LLVM community has explicitly identified and left unresolved.

---

## Score Breakdown

| Criterion | Score | Justification |
|-----------|------:|---------------|
| Novelty | 9/10 | Wave-05-discourse confirms no RFC proposes selection policy; wave-05-ld-so confirms no prior work uses the ld.so analogy explicitly; LLVM Issue #75356 remains open |
| Feasibility | 9/10 | Prototype exists at 5100 LOC; liboffload API stable enough for integration; GTX 1650 available for benchmarks |
| Evidence Strength | 10/10 | 25 waves converge from 16 ecosystems; quantitative data from TaxBreak, arXiv:2601.00227, chipStar, ALPAKA benchmarks |
| Impact | 10/10 | GPU/Offloading Workshop 2025 shows community actively seeking this; LLVM Dublin audience is exact target; fills gap LLVM itself identified |

---

## Evidence Summary

### Primary Evidence (directly validates the direction)

| Source | Type | What It Proves | Wave |
|--------|------|---------------|------|
| LLVM Issue #75356 | GitHub Issue | LLVM community identified `dlsym`-for-GPUs as missing; proposed `__tgt_get_kernel_handle`; no merged solution | wave-02-fat-binaries, wave-05-ld-so S1 |
| liboffload RFC (Discourse, Oct 2023) | RFC | "Basically unanimous support" for shared offloading runtime; mechanism layer only, no selection policy | wave-02-llvm-offloading, wave-05-discourse S1 |
| liboffload API PR #122106 | PR (merged) | `olCreateProgram` + `olCreateKernel` provides mechanism; no multi-version selection | wave-02-llvm-offloading, wave-05-discourse S2 |
| liboffload Roadmap RFC (Nov 2023) | RFC | Explicitly acknowledges API is "intentionally lower-level than any language model" — mechanism, not policy | wave-05-discourse S2 |
| GPU/Offloading Workshop 2025 | Workshop | "Not-Compiler Runtime Library GPUs" talk addresses user-space dispatch; community roadmap shows gap is active | wave-05-discourse S12 |

### Quantitative Evidence (supports performance claims)

| Metric | Value | Source | Wave |
|--------|-------|--------|------|
| CUDA dispatch floor (H100) | 4.71 us | TaxBreak (arXiv:2603.12465) | wave-03-dispatch-overhead |
| Dynamic dispatch overhead | <0.8% e2e | arXiv:2601.00227 | wave-03-dispatch-overhead |
| libkdl lookup overhead | ~100-200 ns | Prototype measurement | prototype |
| SPIR-V portability cost | 0.75x native | chipStar IJHPCA 2026 | wave-04-chipstar |
| ALPAKA tuning gap | 30-40% without per-device params | Kortelainen CHEP 2024 | wave-05-alpaka |
| AdaptiveCpp JIT vs. CUDA | +30% over CUDA | IWOCL 2025 | wave-03-adaptivecpp |
| Bloom filter variant elimination | 95.8% | Stream-K++ (arXiv:2408.11417) | wave-04-cost-models |
| MoE kernels per token | 9,305 | TaxBreak | wave-03-dispatch-overhead |

### Cross-Vendor Convergence Evidence

Seven independent ecosystems converged on `dlopen`/`dlsym` semantics for GPU kernels:

| Ecosystem | `dlopen` analog | `dlsym` analog |
|-----------|----------------|----------------|
| NVIDIA (CUDA 12.0) | `cuLibraryLoad` | `cuLibraryGetKernel` |
| AMD (HIP) | `hipModuleLoad` | `hipModuleGetFunction` |
| Intel (Level Zero) | `zeModuleCreate` | `zeKernelCreate` |
| LLVM (liboffload) | `olCreateProgram` | `olCreateKernel` |
| AMD (HSA/ROCR) | `hsa_code_object_deserialize` | `hsa_executable_get_symbol_by_name` |
| OpenXLA (PJRT) | `PJRT_Plugin_Initialize` | `PJRT_Client_Create` |
| Meta (ExecuTorch) | `call_delegate(backend_id, blob)` | — |

No system provides a single API that: (1) loads multi-vendor kernel bundles, (2) resolves kernel symbols across vendor backends, and (3) selects the best variant for detected hardware. libkdl fills all three.

---

## Novelty Argument

### What exists

- **Mechanism:** liboffload provides `olCreateProgram(binary_blob)` + `olCreateKernel(program, "name")` + `olEnqueueKernelLaunch()`. This is complete dispatch mechanism.
- **Single-vendor selection:** CUDA fat binary resolver (SM compatibility match, since 2014). AMD GFX target-ID matching. Both driver-internal, single-vendor.
- **Cross-vendor JIT:** AdaptiveCpp SSCP and chipStar achieve cross-vendor dispatch but via SPIR-V JIT, accepting 25% performance cost or requiring cold-start latency.

### What does NOT exist

- **Cross-vendor binary selection:** No system selects among pre-compiled vendor-native binaries at runtime.
- **Selection policy in LLVM:** The liboffload roadmap explicitly excludes it. No RFC proposes it. LLVM Issue #75356 has no merged resolution.
- **Explicit ld.so analogy:** No published work frames GPU dispatch as a dynamic linking problem (exhaustive search, wave-05-ld-so-analogy).
- **Cross-vendor cost model:** All existing cost models (nvMatmulHeuristics, cuBLAS, MIOpen find-and-cache) are intra-vendor. libkdl's roofline-based scorer is the first cross-vendor approach.

### Differentiators vs. closest prior art

| System | What it does | What libkdl adds |
|--------|-------------|-----------------|
| AOTriton | Pre-compiled multi-arch AMD dispatch with SQLite autotuning | Cross-vendor (AMD + NVIDIA + CPU); not AMD-only |
| AdaptiveCpp SSCP | Single-binary cross-vendor JIT | Pre-compiled native binaries (no JIT latency, no SPIR-V performance ceiling) |
| CUDA fat binary | SM-compatible variant selection | Cross-vendor; user-space; extensible cost model |
| IREE HAL | Multi-backend compiled executables | Per-kernel runtime selection (not load-time boolean); framework-agnostic |
| HetGPU | Cross-vendor via hetIR JIT | Pre-compiled (microsecond dispatch, not 50-200 ms cold start) |

---

## Feasibility Plan

### What already exists (prototype, ~5100 LOC)

- `kdl.c`: MTB bundle format parser, capability contract matching, roofline cost scorer
- CUDA backend: `cuModuleLoadData` + `cuLaunchKernel` path, verified on GTX 1650
- CPU fallback backend
- `kdl_bundle.py`: Python script for MTB creation
- Benchmark harness: dispatch overhead measurement

### What needs to be done for the poster (estimated effort)

| Task | LOC | Time | Priority |
|------|----:|------|----------|
| liboffload integration wrapper (`kdl_open` → `olCreateProgram`) | ~200 | 1 day | HIGH |
| Dispatch overhead micro-benchmark (CUDA Graph replay method) | ~300 | 1 day | HIGH |
| GEMM end-to-end benchmark (batch 1/16/64) | ~400 | 1 day | HIGH |
| Cost model quality benchmark (roofline vs. exhaustive, 10+ shapes) | ~200 | 0.5 day | MEDIUM |
| Architecture diagram for poster (SVG/TikZ) | — | 0.5 day | MEDIUM |
| Poster LaTeX/HTML layout | — | 1 day | MEDIUM |

### Hardware available

- GTX 1650 (CUDA, SM 7.5)
- CPU fallback (x86-64)
- No AMD GPU available — HIP path is design-validated but not benchmarked

### Risk mitigation

- If liboffload API changes before Dublin: libkdl wraps the vendor APIs directly (current prototype already does this)
- If GTX 1650 numbers are not competitive with H100 baselines: present as normalized overhead (% of hardware floor) rather than absolute latency

---

## Poster Potential

### Why this fills an A0 poster

1. **Immediately legible framing:** "ld.so for GPU kernels" requires zero explanation for the LLVM audience (systems compiler engineers who interact with dynamic linking daily)
2. **Architecture diagram:** Clean layered design (app → libkdl policy → vendor backends → liboffload mechanism) maps directly to a poster figure
3. **Comparison table:** 6-system comparison table with quantitative columns is the poster's centerpiece
4. **Benchmark numbers:** dispatch overhead table + GEMM throughput graph provide concrete evidence
5. **Historical context:** 5 GPU portability failure modes → 5 design constraints that libkdl satisfies — compelling narrative arc
6. **Future work:** MLIR `gpu.dispatch_select` op signals upstream ambition
7. **Related work:** 20 well-curated citations demonstrate survey depth

### Audience fit

The LLVM Developers' Meeting Dublin audience includes:
- liboffload/offload developers (direct stakeholders)
- MLIR GPU dialect contributors (interested in runtime dispatch gap)
- SYCL/DPC++ developers (competing approach, will engage critically)
- Vendor compiler engineers (NVIDIA, AMD, Intel) — will recognize the dlopen/dlsym convergence
- HPC/HEP community (CMS ALPAKA users) — will appreciate the production context

---

## Key References for This Direction

1. LLVM Issue #75356 — `dlsym`-for-GPUs gap
2. liboffload RFC + PR #122106 — mechanism layer
3. TaxBreak (arXiv:2603.12465) — dispatch floor
4. arXiv:2601.00227 — O(1) dispatch overhead
5. chipStar IJHPCA 2026 — SPIR-V cost baseline
6. AdaptiveCpp IWOCL 2023+2025 — compiler-side analog
7. Stream-K++ (arXiv:2408.11417) — Bloom filter cost model
8. GPU/Offloading Workshop 2025 — community roadmap
9. HetGPU (arXiv 2506.15993) — closest prior art
10. ALPAKA Kortelainen CHEP 2024 — tuning gap quantification
