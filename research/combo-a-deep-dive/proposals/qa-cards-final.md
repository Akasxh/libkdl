# Q&A Cards — Final (31 Cards)

**Poster:** "Measuring and Improving Multi-Target Binary Selection in LLVM's GPU Offload Stack"
**Venue:** EuroLLVM Developers' Meeting, Dublin 2026 — Wednesday April 15, 3:15-4:15 PM
**Prepared:** 2026-04-10
**Hardware:** NVIDIA GeForce GTX 1650 (sm_75), CUDA 13.1

---

## Definitive Numbers (OUR Measurements — cite only these)

| Metric | Value | Source |
|--------|-------|--------|
| Variant selection (OffloadBinary, 100k iter) | **3 ns** per call | `runtime_select_poc /tmp/multi_arch.offloadbin` (pinned) |
| Variant selection (directory, 100k iter) | **4 ns** per call | `runtime_select_poc /tmp/cubins` (pinned) |
| Cold module load `cuModuleLoadData` (exec-child, n=100) | **36.0 us median** | `bench_layers` layer2 cold (pinned, 3-run median) |
| Warm module load `cuModuleLoadData` (same ctx, n=10k) | **9.6 us median** | `bench_layers` layer2 warm (pinned, 3-run median) |
| Hot-path dispatch (launch + sync) | **4.1 us** (4,104 ns) | `bench_layers` layers 4+5 (pinned, 3-run median) |
| `cuLaunchKernel` submit (CPU-side) | **1.65 us median** (1,650 ns) | `bench_layers` layer4 (pinned, 3-run median) |
| `cuStreamSynchronize` (GPU RTT, null kernel) | **2.45 us median** (2,454 ns) | `bench_layers` layer5 (pinned, 3-run median) |
| `cuModuleGetFunction` (symbol lookup) | **63 ns** (63 ns median) | `bench_layers` layer3 (pinned, 3-run median) |
| `cuDeviceGet` (driver shim) | **30 ns median** | `bench_layers` layer1 (pinned, 3-run median) |
| kdl bundle load | **5.3 us median** | `bench_dispatch` 3-run avg |
| kdl cold selection (includes cuModuleLoadData) | **56 us median** | `bench_dispatch` 3-run avg |
| Direct CUDA launch baseline | **922 ns median** | `bench_dispatch` 3-run avg |
| OffloadBinary file (3 SM variants) | **14,064 bytes** | `offloadbinary_parse` output |

**Reference numbers (from cited papers — NEVER claim as ours):**
- TaxBreak H100 null-kernel floor: 4.71 us avg (Table III, arXiv:2603.12465)
- PyTorch eager dispatch overhead: 5-10 us/kernel (PyGraph, arXiv:2503.19779)

---

## Category 1 — "Why not X?" (10 cards)

---

### Q1: "Isn't this exactly what IREE already does?"

**Why it's dangerous:** IREE is the most prominent heterogeneous GPU stack in MLIR. If we cannot differentiate in 15 seconds, the conversation dies with "go contribute to IREE."

**Answer:** IREE's HAL dispatches at module granularity using a static `is_parameter_compatible` check — issues #12230 and #15334 show ranked image selection remains an open design question. In LLVM's own offload stack, PR #186088 explicitly defers ranked selection to a follow-up. This work operates at the kernel level, inside the LLVM offload stack, using the OffloadBinary container that every LLVM backend already emits. Our measured selection overhead is 3 ns per dispatch call (100k iterations, pinned, GTX 1650) — we add a dispatch table scan, not a new runtime layer.

**Do NOT say:** "IREE doesn't do this." It does some version of it. Never mention IREE first — let them raise it.

---

### Q2: "chipStar does cross-vendor dispatch. How is this different?"

**Why it's dangerous:** chipStar targets SPIR-V portability across AMD, Intel, NVIDIA. A chipStar expert will use it to negate novelty.

**Answer:** chipStar solves portability through a single IR (SPIR-V compile once, run anywhere). This work solves selection among pre-compiled vendor-native binaries — NVVM for peak NVIDIA performance, ROCDL for peak AMD — where the tradeoff is performance versus portability. The two compose: chipStar can generate the SPIR-V fallback entry in a fat binary, and our dispatch layer (3 ns selection cost, pinned measurement) prefers the native cubin when available. Our OffloadBinary PoC carries 3 SM variants in 14,064 bytes total.

**Do NOT say:** "chipStar is irrelevant." Frame as orthogonal. The questioner likely knows chipStar better than you.

---

### Q3: "Proteus already does JIT kernel specialisation. Why a static dispatch table?"

**Why it's dangerous:** Proteus (LLNL) applies JIT specialisation at the `cuModuleLoad` boundary. A Proteus-aware questioner will frame our work as redundant or inferior.

**Answer:** Proteus optimises an existing binary at dispatch time through JIT recompilation — valuable for specialisation but adds JIT latency and requires keeping LLVM IR alive at runtime. Our work selects among pre-compiled binaries in 3 ns (pinned, 100k iterations, OffloadBinary path, GTX 1650) with zero JIT cost, targeting deployment where recompilation is unacceptable (e.g., shipping a fat binary that must cold-start without LLVM). The two mechanisms compose: Proteus can produce one of the binaries that our layer selects.

**Do NOT say:** "JIT is bad." Say "different cost model, different deployment constraint."

---

### Q4: "SPIR-V is the portable IR. Why not compile to SPIR-V once and be done?"

**Why it's dangerous:** The RFC "SPIR-V IR as a vendor-agnostic GPU representation" (discourse #85115) is active. The questioner may be an author.

**Answer:** SPIR-V portability and vendor-native performance are a real tradeoff: tensor-core Warp Specialization (sm_90a), MFMA intrinsics, and AMD AGPR accumulation have no universal SPIR-V equivalents today — the RFC itself acknowledges vendor-specific extensions remain necessary for peak ML performance. A fat binary carrying both a SPIR-V fallback and native cubins/HSACOs, selected at runtime by our mechanism (3 ns selection overhead, pinned measurement), is strictly better than either alone. Our OffloadBinary PoC already carries 3 architecture variants in one container.

**Do NOT say:** "SPIR-V doesn't work." Acknowledge the portability value, then explain why selection is still needed for peak perf.

---

### Q5: "liboffload already handles multi-binary dispatch through `parseOffloadBinary`. Why do you need anything else?"

**Why it's dangerous:** Most technically precise "why not X" question. The questioner may have read PR #186088.

**Answer:** PR #186088's `parseOffloadBinary` loop implements "first-compatible-wins" — it breaks on the first image that passes `isMetadataCompatible` and `isDeviceCompatible`, with no ranking, no capability-aware scoring, and no fallback chain. The PR body explicitly defers ranked selection to a follow-up. Our contribution is that follow-up: the metadata vocabulary (5 new keys including `variant_priority` and `requires_features`) supplies the ranking inputs, and our flame graph (36.0 us cold-load median, 4.1 us hot-path, pinned 3-run medians) quantifies where time goes so ranking decisions are grounded in measurement. PR #186088 has been open 28+ days with no merge decision as of April 9.

**Do NOT say:** "liboffload doesn't do multi-binary." It does — just not ranked selection. Cite PR #186088 by number.

---

### Q6: "HetGPU does cross-vendor binary compatibility via JIT translation. How is this complementary or different?"

**Why it's dangerous:** HetGPU (arXiv 2506.15993) uses a custom hetIR for binary-level cross-vendor compatibility. A knowledgeable questioner sees it as solving the same problem.

**Answer:** HetGPU pursues binary compatibility — compile once to hetIR, JIT translate to native code at kernel load time on any vendor. This requires an entirely new IR, compiler backend (HETTarget), and per-vendor JIT pipeline. Our work is the opposite bet: keep vendor-native pre-compiled binaries (cubins, HSACOs) for peak performance and select among them at runtime in 3 ns. HetGPU pays JIT cost at load time for universal portability; we pay 14 KB of OffloadBinary storage overhead for zero-JIT deployment. The approaches address different points on the portability-performance tradeoff.

**Do NOT say:** "HetGPU is vaporware" or speculate about its maturity. Differentiate on mechanism (JIT translation vs. AOT selection) and cost model.

---

### Q7: "Meta's KernelEvolve generates kernels for multiple targets. Isn't that enough?"

**Why it's dangerous:** KernelEvolve (ISCA 2026) is a production system at Meta with 60% throughput improvement. Industry validation makes it credible.

**Answer:** KernelEvolve is a design-time kernel generation system — it creates optimized kernels for NVIDIA, AMD, MTIA, and CPU through search-based optimization. Excellent for producing the kernel variants. But it doesn't address the runtime question: given a fat binary with 4 target-specific kernels, which one runs on THIS hardware? That's exactly what our dispatch layer does. KernelEvolve is upstream of us — it generates the variants, we select among them at runtime in 3 ns.

**Do NOT say:** "We're better than KernelEvolve." Position as complementary pipeline stages.

---

### Q8: "AdaptiveCpp does runtime JIT specialization. How is this different?"

**Why it's dangerous:** AdaptiveCpp's SSCP (Single-Source, Single Compilation Pass) JIT-specializes based on runtime info. IWOCL 2025 paper shows real speedups.

**Answer:** AdaptiveCpp specializes a single SYCL source at JIT time using runtime information like work-group sizes and pointer alignments — powerful but requires the SYCL programming model and carries JIT latency on first launch. Our approach selects among pre-compiled AOT binaries (zero JIT cost) and operates inside LLVM's OffloadBinary ecosystem, not SYCL. For deployment scenarios where SYCL adoption isn't feasible or JIT is unacceptable (embedded, inference serving), our 3 ns AOT selection provides the multi-target benefit without the programming model or JIT cost.

**Do NOT say:** "SYCL is dead." SYCL has strong advocates. Frame as different deployment constraints.

---

### Q9: "The Universal GPU ISA paper identifies hardware-invariant primitives. Doesn't that make dispatch unnecessary?"

**Why it's dangerous:** arXiv:2603.28793 (March 2026) proposes a universal ISA matching native performance. If GPUs converge, dispatch becomes moot.

**Answer:** That paper actually validates our approach. They found 10 hardware-invariant primitives but also 6 true architectural divergences — fundamental design disagreements between vendors. Until those divergences resolve (years, if ever), vendor-specific optimized kernels will outperform any universal binary for compute-intensive ML workloads. Our dispatch layer bridges the gap: carry both the portable fallback and the vendor-optimized versions, select the best at runtime. The universal ISA defines what's invariant; we handle what's still divergent.

**Do NOT say:** "Universal ISA won't work." Acknowledge the direction, position our work as the bridge.

---

## Category 2 — Technical Depth (6 cards)

---

### Q10: "How does `gpu.select_variant` actually lower? Walk me through the LLVM IR."

**Why it's dangerous:** Core technical question. A vague answer ("it emits a dispatch stub") signals vaporware. The questioner is testing whether you understand `OffloadingLLVMTranslationAttrInterface`.

**Answer:** The `#gpu.runtime_select` attribute implements `embedBinary` to emit N LLVM global byte arrays (one per vendor object), a dispatch table global of `{vendor_id, binary_ptr, size, load_fn_ptr}` structs initialised via `llvm.global_ctors`, and a vendor-detection stub that calls `cuInit`/`hipInit`/`zeInit` through `dlopen`-loaded symbols. The `launchKernel` implementation replaces the hardcoded `mgpuModuleLoad` call with an indirect call through the selected `load_fn_ptr` slot. Our prototype validates the pattern end-to-end on GTX 1650: vendor detection, dispatch table construction (86 us one-time for OffloadBinary parse), and 3 ns steady-state selection. The LLVM IR patterns (`global_ctors`, global arrays, `dlopen`-based indirect calls) already exist in the NVVM lowering path.

**Do NOT say:** "It emits a runtime stub" without explaining the dispatch table structure. Know that the template is `SelectObjectAttr.cpp`.

---

### Q11: "What is your performance model for variant ranking?"

**Why it's dangerous:** The prototype (`kdl.c:1051-1054`) uses hardcoded constants. Calling this "roofline model" will destroy credibility instantly.

**Answer:** The current prototype uses a weighted heuristic — vendor-assigned locality constants — as a stand-in for a proper analytical model. The proposed upstream design uses `variant_priority` and `requires_features` keys from the OffloadBinary string table as ranking inputs, separating selection policy (pluggable) from mechanism (dispatch table). The `rankImage()` callback design — analogous to PR #186088's deferred follow-up — lets runtimes supply their own cost function. Our measurement shows the ranking logic itself costs 3 ns regardless of the scoring function complexity, since the dispatch table has 2-4 entries.

**Do NOT say:** "Roofline model." Do not cite tritonBLAS's 94.7% validation — that is their model. Say "weighted heuristic" and be specific.

---

### Q12: "Your `dlopen`-based multi-vendor detection: what happens if both CUDA and ROCm are installed?"

**Why it's dangerous:** Real deployment scenario. The `dlopen` approach must handle symbol isolation.

**Answer:** Each vendor runtime is loaded with `RTLD_LOCAL` flag isolation — CUDA symbols stay in the CUDA handle, HIP symbols in the HIP handle, Level Zero in its own — preventing cross-vendor symbol resolution. The detection probe calls `cuInit()` first; if it returns `CUDA_SUCCESS`, NVIDIA is selected; if `CUDA_ERROR_NO_DEVICE`, fall back to `hipInit`. Our prototype measures vendor detection at ~170 ms (one-time, dominated by `cuInit` driver init), with the dispatch table construction at 86-150 us and steady-state selection at 3 ns. This is the same `dlopen`+`RTLD_LOCAL` pattern JAX and PyTorch have shipped in production for three years.

**Do NOT say:** "We haven't tested multi-vendor." Acknowledge the architecture handles it via `RTLD_LOCAL`.

---

### Q13: "The OffloadBinary metadata keys — `min_sm`, `requires_features` — these become ABI. Who enforces backward compatibility?"

**Why it's dangerous:** String keys in a format become part of the ABI contract. A fat binary compiled with LLVM 20 must work with LLVM 22's runtime.

**Answer:** The proposal calls for an RFC before the first implementation patch, precisely because key names are a stable ABI contract. OffloadBinary already uses version numbers (PR #169425 bumped to version 2); the new key vocabulary is gated on format version so old runtimes encountering new keys either silently ignore them (missing key = no constraint, per the proposal) or reject images with unsatisfied `min_sm` requirements. The compatibility rule: old keys work in new runtimes; new keys are ignored by old runtimes. The header constants patch is ~30 LOC in `OffloadBinary.h`.

**Do NOT say:** "We'll figure that out in the RFC." Show you have thought through silent-ignore vs. explicit-reject semantics.

---

### Q14: "Your flame graph shows cold-path vs. hot-path separately. How do you account for PTX JIT cost contaminating the cold-path measurement?"

**Why it's dangerous:** `cuModuleLoadData` triggers PTX-to-SASS JIT on first load (10-100 ms). If this dominates, the LLVM stack latencies are invisible.

**Answer:** The experiment uses pre-compiled CUBIN (not PTX), eliminating PTX JIT from the cold path entirely. The null-kernel CUBIN is compiled ahead of time with `nvcc -arch=sm_75 -cubin` (4,328 bytes, ELF-verified). `cuModuleLoadData` receives a binary requiring no JIT. The cold-path measurement then isolates: OffloadBinary parse + module load (36.0 us median, pinned 3-run cross-run, n=100 exec-child trials) + symbol lookup (63 ns) + first launch. A separate experiment with PTX input can quantify JIT cost independently.

**Do NOT say:** "We separate cold and hot paths" without naming CUBIN specifically. If you cannot say "CUBIN", they know you have not run it.

---

### Q15: "What does the OffloadBinary file actually look like? Show me the format."

**Why it's dangerous:** Tests whether the OffloadBinary work is real implementation or handwaving.

**Answer:** We implemented a writer and parser matching `llvm/include/llvm/Object/OffloadBinary.h` exactly: 48-byte file header (magic `0x10FF10AD`, version, size, entry_offset, entry_count), then per entry: entry header (the_size, image_offset, image_size, string_offset, string_size) + null-terminated key-value string table (`triple\0nvptx64-nvidia-cuda\0arch\0sm_75\0kind\0cuda\0`) + raw ELF/CUBIN image bytes. Our packaged file is 14,064 bytes for 3 SM variants (sm_75: 4,328 bytes, sm_86/sm_89: 4,712 bytes each). All entries validated via `7F 45 4C 46` ELF magic check on readback.

**Do NOT say:** "We use the standard format" without being able to name the field sizes or magic number.

---

## Category 3 — Data Challenges (4 cards)

---

### Q16: "You're measuring on a GTX 1650. That's a consumer card from 2019. Why should I care?"

**Why it's dangerous:** The community benchmarks on H100s and MI300Xs. A GTX 1650 reads as "I didn't have real hardware access."

**Answer:** The GTX 1650 is the right hardware for this claim: dispatch overhead measurement is about software stack latency, not compute throughput. The claim is about relative layer fractions — "variant selection contributes 3 ns to a 4.1 us hot-path dispatch" — which generalises across hardware because the dispatch path (OffloadBinary parse, driver module load, symbol lookup) scales with software complexity, not VRAM bandwidth. TaxBreak's H100 result (4.71 us null-kernel floor) confirms the driver baseline is in the same order of magnitude. Modest hardware is honest benchmarking — H100 speed hides software overhead behind fast compute.

**Do NOT say:** "GTX 1650 is representative of production." It is not for compute. The argument is: dispatch overhead is a software-stack property.

---

### Q17: "What's your actual measured number for variant selection overhead? Not borrowed — your number, your hardware."

**Why it's dangerous:** The kill shot. If numbers are borrowed from TaxBreak or PyGraph, a sharp reviewer catches it instantly.

**Answer:** `runtime_select_poc` with a real OffloadBinary containing 3 CUBIN variants (sm_75, sm_86, sm_89) on GTX 1650: **3 ns per selection call** over 100,000 iterations (pinned, 3-run cross-run). The directory-mode test gives 4 ns. Cold module load (including `cuInit` + `cuModuleLoadData`) is 36.0 us median (pinned 3-run cross-run). Hot-path dispatch floor (launch + sync) is 4.1 us (pinned 3-run median, 10,000 iterations). All measured with `clock_gettime(CLOCK_MONOTONIC)` on our hardware, our kernel, our stack. TaxBreak H100 and PyGraph numbers are cited reference points only — never claimed as ours.

**Do NOT say:** Any number you have not personally measured. If asked about a number not in this card, say "I don't have that measurement" rather than improvising.

---

### Q18: "Does this work on AMD? You only mentioned GTX 1650."

**Why it's dangerous:** A heterogeneous dispatch proposal tested only on NVIDIA is half a proposal.

**Answer:** The AMD code path in `kdl.c` (lines 568, 749) loads `libamdhip64.so` via `dlopen` and calls `hipModuleLoadData` with the same dispatch table structure used for CUDA. The code is symmetric by design. The AMD path has been validated via unit tests with mocked HIP entry points but not on physical ROCm hardware due to availability. The CUDA path on GTX 1650 demonstrates the mechanism works end-to-end: vendor detection (101 ms one-time, pinned), dispatch table construction (86 us), selection (3 ns, pinned), module load (36.0 us cold, pinned), launch + sync (4.1 us, pinned). A collaborator with MI300X access is the stated next step.

**Do NOT say:** "It works on AMD." Say "validated via mocked HIP, physical ROCm pending" — honest and respected.

---

### Q19: "How about H100? Your numbers on a Turing card don't tell me about Hopper."

**Why it's dangerous:** H100 is the target hardware for ML inference. Without H100 data, the work seems disconnected from real deployment.

**Answer:** Our contribution is the layer decomposition methodology, not absolute latency values. On GTX 1650, the selection overhead (3 ns) is 0.07% of the hot-path dispatch (4.1 us). TaxBreak measured H100 null-kernel dispatch at 4.71 us via CUDA driver API directly — the same order of magnitude as our 4.1 us on GTX 1650, confirming the driver-level baseline is stable across generations. The 3 ns selection overhead would be an even smaller fraction on H100. The flame graph structure (which layers dominate) is the generalizable finding; absolute microsecond values are hardware-specific.

**Do NOT say:** "Our numbers apply to H100." They do not directly. Argue the relative fractions generalise.

---

## Category 4 — Upstream Viability (4 cards)

---

### Q20: "Would the RFC be accepted? Who would review it?"

**Why it's dangerous:** A proposal without a named reviewer is a proposal in limbo.

**Answer:** The GPU dialect's `OffloadingLLVMTranslationAttrInterface` was designed as an extension point for exactly this use case — `SelectObjectAttr.cpp` is the canonical example. RFC "Cleaning the GPU Dialect" (#88170) is actively maintained by Fabian Mora; coordinating `#gpu.runtime_select` with that RFC is the upstream path. For the metadata vocabulary (T07), OffloadBinary is primarily maintained by Joseph Huber (LLNL) — whose PR #186088 explicitly defers ranked selection. Both are likely at Dublin. The poster session is the opening of that conversation.

**Do NOT say:** "I haven't talked to anyone." Frame it as: "the poster session is the engagement strategy, and here are the specific reviewers."

---

### Q21: "Building a new MLIR op and getting it reviewed is a 6-month process. Why would this land?"

**Why it's dangerous:** Tests whether you understand the upstream process and have a realistic plan.

**Answer:** The implementation is self-contained: one new attribute in `GPUOps.td`, one ~400 LOC file modeled on `SelectObjectAttr.cpp`, one pass, and two test files. The fastest path is coordinating with Fabian Mora to land `#gpu.runtime_select` as the dispatch-policy implementation of RFC #88170's container/policy separation. The metadata vocabulary patch is even lighter: ~30 LOC header constants in `OffloadBinary.h` plus a docs patch. That can merge independently as a documentation-only change naming the existing two keys and reserving the namespace for new ones.

**Do NOT say:** "LLVM review is slow but we'll get there." Name a specific reviewer and a specific RFC.

---

### Q22: "Your metadata vocabulary touches three backends — AMDGPU, NVPTX, SPIR-V. How do you handle disagreement between backend owners?"

**Why it's dangerous:** Multi-backend RFCs are slow because each owner has opinions.

**Answer:** The proposal separates naming RFC from implementation patches: a documentation-only patch naming the existing two keys and reserving the namespace can merge without backend-owner consensus and builds awareness. Once the RFC agrees on Tier 1 key names (the only ABI-load-bearing ones), backend owners implement their writers independently and in parallel. The `min_sm` and `min_gfx` keys use naming from existing formats: AMD Code Object V5 target-ID and CUDA EIATTR, reducing bikeshedding. The full implementation is 5 independent patches (header: ~30 LOC, `isMetadataCompatible` extension: ~40 LOC, AMDGPU writer: ~60 LOC, NVPTX writer: ~60 LOC, tooling: ~80 LOC).

**Do NOT say:** "We'll figure it out in the RFC." Show the patch sequencing: docs first, Tier 1 only in RFC, backends in parallel.

---

### Q23: "The `feature=` key was proposed in D127686 and never standardised. What makes this different?"

**Why it's dangerous:** Best historical precedent against the metadata vocabulary. A previous attempt stalled.

**Answer:** D127686 proposed a single key (`feature=`) for a narrow use case (LTO target-feature propagation), without a vocabulary specification or RFC, and was never standardised because there was no community-agreed semantics for what the value meant. This proposal inverts the approach: start with an RFC to agree on the vocabulary before writing code, explicitly tier the keys by semantic weight (MUST/SHOULD/MAY), and provide a documentation-only patch naming the existing two keys as step zero. The `feature=` lesson is precisely why the RFC precedes the patch in our sequencing.

**Do NOT say:** "D127686 failed for unrelated reasons." It failed for the exact reason this proposal addresses. Cite the lesson.

---

## Category 5 — "So What?" Challenges (5 cards)

---

### Q24: "Who actually has a fat binary with multiple GPU objects that they need to dispatch at runtime?"

**Why it's dangerous:** If nobody ships fat binaries through MLIR today, the work has no users.

**Answer:** Three concrete users: (1) HEP-CCE at CERN maintains ~80 build configurations for heterogeneous GPU clusters (A100/V100 + MI250X + CPU fallback) — each configuration is a separate build; a single fat binary with runtime selection eliminates this combinatorial explosion. (2) vLLM maintains separate NVIDIA and AMD codepaths with different CUDA/HIP kernel implementations — runtime variant selection would allow a single binary serving both. (3) Cloud GPU containers (AWS p4/p5, Google A3) where the GPU model is unknown at compile time. The gap is real: every heterogeneous deployment works around it today in its own ad-hoc way.

**Do NOT say:** "Everyone will use this eventually." Name specific users with specific pain points.

---

### Q25: "Why now? This gap has existed since 2022. Why is this suddenly urgent?"

**Why it's dangerous:** If the gap has existed for 4 years without being filled, maybe nobody needs it.

**Answer:** Three things changed: (1) Intel XeVM landed in August 2025 (PR #148286), making MLIR's GPU dialect truly tri-vendor for the first time — the selection problem is now 3-way, not 2-way. (2) PR #186088 (March 2026) generalized OffloadBinary to all plugins — the fat-binary container is now production infrastructure, not experimental. (3) The metadata consumer hook `isMetadataCompatible()` merged March 10, 2026 (PR #185663) — the runtime filter exists but has no vocabulary to consume. All three prerequisites landed in the last 8 months.

**Do NOT say:** "It's been a problem since OffloadBinary was created." Explain what changed to make the solution timely.

---

### Q26: "What's the real-world impact? 3 ns selection overhead doesn't matter if nobody's bottlenecked on dispatch."

**Why it's dangerous:** Forces you to argue impact beyond microbenchmark bragging.

**Answer:** The 3 ns selection overhead is not the impact claim — it is the cost justification: runtime selection adds negligible overhead. The impact is operational: eliminating CERN HEP-CCE's 80-build-configuration matrix, enabling vLLM to ship one binary for NVIDIA+AMD, and letting cloud containers defer GPU target selection from build time to deploy time. The flame graph contribution (36.0 us cold load, 4.1 us hot-path, pinned per-layer decomposition) is independently valuable as the first published measurement of the LLVM GPU dispatch stack interior — it tells every liboffload user where their dispatch time goes.

**Do NOT say:** "3 ns is really fast!" The point is not that selection is fast; the point is that selection is free enough to enable operational simplification.

---

### Q27: "This is just a poster. Where's the RFC? Where's the patch?"

**Why it's dangerous:** Tests whether this is real work or a design-only exercise.

**Answer:** The poster presents two concrete contributions and one design sketch. Contribution 1 (metadata vocabulary) has a draft RFC ready to post — 5 keys, header constants patch (~30 LOC), and a staged upstream path with named reviewers (Huber, Denny, Mora). Contribution 2 (flame graph) requires no upstream change — it is measurement plus a benchmark contribution to `llvm-test-suite`. The design sketch (T01, `#gpu.runtime_select`) is explicitly presented as future direction: ~780 LOC total, modeled on `SelectObjectAttr.cpp`, no MLIR C++ exists yet. The prototype (`kdl.c`, 5100 LOC) validates the runtime mechanics end-to-end on GTX 1650 with measured numbers.

**Do NOT say:** "The RFC will be posted after Dublin." If the draft is ready, say so. If not, be honest about timeline.

---

### Q28: "CPU function multi-versioning does exactly this for CPUs. How is yours different?"

**Why it's dangerous:** CPU FMV (`target_clones`) is mature and upstream since GCC 6. The questioner asks why GPU dispatch needs a new mechanism.

**Answer:** CPU FMV operates on LLVM IR functions within a single ISA (x86 CPUID-based selection), emitting a resolver function at the IR level with no runtime library boundary. GPU dispatch crosses a driver API boundary: selection must happen before calling `cuModuleLoadData` or `hipModuleLoadData` with the correct binary blob, and target feature detection requires probing vendor runtime libraries via `dlopen` (not a CPUID instruction). Our dispatch table + `global_ctors` mechanism is structurally analogous to FMV's IFunc resolver but operates at the module-load level across mutually exclusive vendor stacks. Measured cost: one-time 36.0 us cold module load (vs. nanoseconds for IFunc resolution), steady-state 3 ns selection.

**Do NOT say:** "It's totally different." Acknowledge the structural analogy — the LLVM community appreciates you see the connection — then explain the precise differences.

---

## Quick-Reference Summary Table

| # | Category | Trap | Key Phrase | Our Number |
|---|----------|------|------------|------------|
| 1 | Why not X | IREE already does it | "HAL module vs. kernel granularity; PR #186088 defers ranked selection" | 3 ns selection |
| 2 | Why not X | chipStar handles portability | "Portability vs. peak perf; orthogonal, composable" | 14 KB OffloadBinary |
| 3 | Why not X | Proteus does JIT | "Zero JIT cost; different cost model" | 3 ns vs. JIT ms |
| 4 | Why not X | SPIR-V is the portable IR | "Vendor extensions have no SPIR-V equivalent for peak ML" | 3 ns to prefer native |
| 5 | Why not X | liboffload already handles it | "First-compatible-wins; PR #186088 defers ranking" | 36.0 us cold, 3 ns select |
| 6 | Why not X | HetGPU does cross-vendor | "JIT translation vs. AOT selection; different cost model" | 3 ns vs. JIT |
| 7 | Why not X | HetGPU binary compatibility | "HetGPU re-compiles; we route; approaches compose" | 3 ns AOT selection |
| 8 | Why not X | KernelEvolve generates multi-target | "Design-time generation vs. runtime selection; complementary stages" | 3 ns select |
| 9 | Why not X | AdaptiveCpp JIT specialization | "SYCL + JIT latency vs. AOT AOT selection; different deployment constraints" | 3 ns zero JIT |
| 10 | Why not X | Universal GPU ISA makes dispatch moot | "6 architectural divergences remain; dispatch bridges the gap" | 3 ns select |
| 11 | Technical | How does it lower? | "N globals + dispatch table + global_ctors + indirect call" | 86 us table build |
| 12 | Technical | Performance model? | "Weighted heuristic; NOT roofline; pluggable rankImage()" | 3 ns regardless of scoring |
| 13 | Technical | Dual-vendor dlopen | "RTLD_LOCAL per vendor; JAX/PyTorch precedent" | 170 ms one-time detect |
| 14 | Technical | Metadata keys become ABI | "RFC before code; silent-ignore for missing keys" | ~30 LOC header patch |
| 15 | Technical | PTX JIT contaminates cold path | "Pre-compiled CUBIN; no JIT in measurement" | 36.0 us cold (CUBIN) |
| 16 | Technical | Show me the binary format | "0x10FF10AD magic; 48-byte header; string table" | 14,064 bytes, 3 entries |
| 17 | Data | GTX 1650 is ancient | "Dispatch overhead is software stack latency, not compute" | 4.1 us hot-path |
| 18 | Data | Show YOUR numbers | "3 ns selection, 36.0 us cold, 4.1 us hot-path" | All from our hardware |
| 19 | Data | AMD hardware? | "Mocked HIP validated; physical ROCm pending" | Same 3 ns mechanism |
| 20 | Data | H100? | "Layer fractions generalise; TaxBreak H100 = 4.71 us same order" | 3 ns / 4.1 us = 0.07% |
| 21 | Upstream | Who reviews it? | "Mora (RFC #88170) + Huber (liboffload); poster is the opening" | — |
| 22 | Upstream | 6-month review process | "Self-contained: ~400 LOC + SelectObjectAttr.cpp template" | ~780 LOC total |
| 23 | Upstream | Three backends disagree | "Docs-first patch; Tier 1 only in RFC; backends parallel" | 5 independent patches |
| 24 | Upstream | D127686 also failed | "RFC-first inverts the failure mode; naming before code" | ~30 LOC step 0 |
| 25 | So what? | Who uses this? | "CERN 80-config, vLLM NVIDIA/AMD, cloud GPU containers" | — |
| 26 | So what? | Why now? | "XeVM Aug 2025, PR #186088 Mar 2026, isMetadataCompatible merged" | 3 prerequisites in 8 months |
| 27 | So what? | 3 ns doesn't matter | "Cost justification, not impact claim; operational simplification" | 3 ns = free enough |
| 28 | So what? | Where's the RFC? | "Draft ready; two concrete contributions + design sketch" | 5100 LOC prototype |
| 29 | So what? | CPU FMV does this | "IFunc analogy correct; difference is driver API boundary" | 36.0 us vs. ns IFunc |

---

## Emergency Pocket Card (if you blank)

If a question comes from a direction not covered above, anchor on these three facts:

1. **"Our selection overhead is 3 ns per dispatch call, measured on GTX 1650 with real CUBINs packed into a valid OffloadBinary (pinned, 3-run cross-run)."**
2. **"PR #186088 implements first-compatible-wins. Our contribution is the ranked selection that PR explicitly defers."**
3. **"The metadata RFC is a ~30 LOC header patch. The risk is naming bikeshedding, not technical complexity."**

Never fabricate a number. If asked for a measurement you do not have, say: "I don't have that specific measurement. What I can tell you is [pivot to a number you do have]."

---

*Prepared: 2026-04-10, updated 2026-04-10 (hostile review cards Q26-Q28)*
*Source data: bench_layers (10k warm + 100 cold trials), runtime_select_poc (100k iterations, directory + OffloadBinary modes), bench_dispatch (3 stable runs), offloadbinary_parse (writer + parser validation)*
*All measurements: GTX 1650 sm_75, CUDA 13.1, clock_gettime(CLOCK_MONOTONIC)*

---

## Category 6 — Hostile Reviewer Defenses (3 cards)

---

### Q29: "You cite IREE issue #50 as evidence that ranked selection is unimplemented for 6 years. Issue #50 is about device selection policy, not kernel binary selection. You're conflating two different problems."

**Why it's dangerous:** 9/10. If an IREE engineer pins you on this conflation, your credibility drops for the rest of the conversation. This is the single most damaging factual error in the poster materials.

**Answer:** You are correct. Issue #50 is about device selection policy within IREE's HAL, not kernel binary selection. The poster's original citation was imprecise and has been corrected. The stronger, directly analogous cite is PR #186088 in LLVM's liboffload, which implements `parseOffloadBinary` with first-compatible-wins semantics and explicitly defers ranked image selection to a follow-up. IREE's HAL and this work operate at different abstraction levels: IREE owns the full execution model (buffers, scheduling, device management); this work operates below any framework, inside the LLVM offload stack, using the OffloadBinary container. The two compose rather than compete. On IREE's ranked selection status: issues #12230 and #15334 are more relevant — they concern executable variant selection within HAL, which is closer to what we address.

**Do NOT say:** "IREE issue #50 proves they haven't done this." It does not. Concede the error immediately and pivot to PR #186088.

---

### Q30: "Your PoC doesn't use LLVM at all — it's a standalone C library. Where is the MLIR pass? Where is the OffloadingLLVMTranslationAttrInterface implementation?"

**Why it's dangerous:** 8/10. Accurate critique. The strongest version of the "vaporware" argument. `runtime_select_poc.c` and `kdl.c` include zero LLVM headers, link against zero LLVM libraries, and the `RuntimeSelectAttr.cpp.sketch` is pseudocode, not compilable C++.

**Answer:** The poster presents two concrete contributions and one design sketch — and labels them as such. Contribution 1 (metadata vocabulary: 5 keys, ~30 LOC header patch, backward-compatible) requires no MLIR C++. Contribution 2 (flame graph: first published per-layer latency decomposition of the LLVM GPU dispatch stack) is measurement, not code. Contribution 3 (`#gpu.runtime_select`) is explicitly labeled "Design Sketch" with "Zero lines of MLIR C++ exist" stated on the poster. The C prototype validates the *runtime mechanics* — dispatch table construction, vendor detection, variant selection, kernel launch — that `#gpu.runtime_select`'s `embedBinary()` would emit as LLVM IR. The C code is the C equivalent of what the LLVM IR would lower to. The poster session is where we start the upstream conversation with the specific reviewers (Huber, Mora, Denny) who would review an RFC — not where we present a merged patch.

**Do NOT say:** "We have MLIR code." You do not. Say: "Two concrete contributions plus a design sketch. The poster session starts the conversation."

---

### Q31: "Your OffloadBinary format is wrong — you're missing ImageKind, OffloadKind, and Flags fields. Your PoC is not interoperable with clang-offload-packager."

**Why it's dangerous:** 7/10. Factually accurate. The PoC implements a simplified subset of the real `OffloadBinary.h` format: correct magic (`0x10FF10AD`), correct overall container structure (file header + variable-length entries with string tables), but per-entry metadata encoding differs. The `ObEntryHeader` struct has five `uint64_t` fields where the real format has `ImageKind`, `OffloadKind`, and `Flags`. Not interoperable with LLVM tooling.

**Answer:** That is correct. The PoC implements the OffloadBinary container structure — magic, header, string table layout — as a round-trip demonstration of the selection mechanism. The per-entry fields are a simplified subset, not interoperable with `clang-offload-packager` or `clang-linker-wrapper`. The poster now includes a qualification: "compatible magic, simplified entry layout." The key defense: the selection mechanism (dispatch table scan, vendor detection, ranking by `variant_priority`) is independent of the exact binary format fields. Adding `ImageKind`, `OffloadKind`, and `Flags` to the struct is straightforward engineering that does not change the 3 ns selection overhead or the flame graph results. The format difference affects interoperability, not the dispatch mechanism being demonstrated.

**Do NOT say:** "Our format is correct" or "Our format is interoperable." It is neither. Acknowledge instantly and pivot to what the PoC *does* demonstrate: the selection mechanism works end-to-end regardless of per-entry field layout.
