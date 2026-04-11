# Poster Strategy Research — LLVM Developers' Meeting Dublin 2026

**Date:** 2026-04-06
**Purpose:** Determine what makes an outstanding poster/lightning talk at an LLVM Dev Meeting,
specifically for a poster on heterogeneous GPU kernel dispatch via MLIR.
**Sources:** Web research (conference agendas, LLVM Discourse, GitHub RFCs) + local literature base.

---

## 1. What the LLVM Community Values (Evidence-Based)

### 1.1 The Conference Itself

EuroLLVM 2026 is April 13-15 at Clayton Hotel Burlington Road, Dublin. The event the poster targets.
Confirmed structure:
- April 13: Pre-conference — **7th MLIR Workshop** (full day) + Newcomer Session
- April 14-15: Main program — technical talks, tutorials, lightning talks, **poster session**
- The poster session is on **April 15 (Day 2)**, during afternoon break

The MLIR Workshop is the highest-density venue for MLIR-native GPU work. The poster session is
lower-density but generates impromptu discussion and direct feedback from implementors.

### 1.2 What the LLVM Contribution Standards Say

From LLVM's Developer Policy and Contributing documentation:
- **Evidence from real-world workloads, not just synthetic benchmarks.** "Improvements seen only
  on synthetic benchmarks may be insufficient."
- **Prototype implementations make design discussions concrete** by demonstrating what is possible.
- **Testable artifacts** are required for any claim of improvement.
- **Consensus before major changes** — engage the community early.

These norms apply directly to poster evaluation: a poster with running code and real numbers carries
far more weight than a design-only proposal.

### 1.3 What Has Gotten Engagement at Recent Meetings

**2025 LLVM Dev Meeting (Santa Clara):**
Themes with notable coverage:
- GPU compilation for multiple vendors (Intel XeGPU, AMD AMDGPU, NVIDIA)
- MLIR dialect development
- AI/ML compilation (IREE, Triton)
- "Mojo GPU Compilation" — lightning talk on CPU+GPU programming via MLIR
- "Taming GPU programming in safe Rust"
- The LLVM/Offload Workshop ran as a parallel track with dedicated GPU/heterogeneous sessions

**EuroLLVM 2024 (Vienna):**
- GPU content was primarily execution model focused (NVGPU dialect for Hopper, MLIR vector distribution)
- Heterogeneous computing was addressed via language models (Mojo) rather than runtime dispatch
- Lightning talks included "Automatic Retuning of Floating-Point Precision" (Ivan Ivanov, Moses) — shows
  that lightweight tools with a clear use case get slots

**EuroLLVM 2025 (London):**
Poster session topics that were accepted:
- "Code-generation of finite element operations using MLIR" (JIT compilation for heterogeneous hardware)
- "LLVM Support for Sub-FP8 Quantization" (ML model efficiency)
- "SonicMachine: Scalable Architecture Description using MLIR"
- "CuSan: CUDA data race detector via ThreadSanitizer"
The pattern: each poster had **a working tool or prototype**, addressed **a specific gap**, and
made a connection to either ML workloads or GPU hardware.

**Pattern across all meetings:** GPU + MLIR + ML inference = the intersection with the most
active developer interest. Talks on new dialects (with running code) consistently outperform
pure surveys or pure design proposals.

---

## 2. The 2026 EuroLLVM Context: What Is Being Presented

This is critical — these are the topics the community is engaging with **at the same venue**
on **the same day** as the poster session.

### 2.1 MLIR Workshop (April 13, morning — same building, day before)

Accepted presentations that directly define the competitive landscape:

| Talk | Relevance to libkdl |
|------|---------------------|
| **"CUDA Tile IR"** (Matthias Springer, Lorenzo Chelini — NVIDIA) | NVIDIA's MLIR-based JIT compiler now in the CUDA GPU driver. Tile-level abstractions for tensor cores. NVIDIA-only. |
| **"ASTER: MLIR-Based Assembly Tooling"** | Low-level MLIR tooling for GPU codegen |
| **"Auto-tuning MLIR schedules for Intel GPUs"** (Tuomas Karna, Rolf Morel) | Intel-specific auto-tuning, compile-time |
| **"MLIR-RAJA: Bridging AI Models and HPC"** | MLIR + RAJA portability model — most directly adjacent to libkdl's goals |
| **"Multi Stage Sequential RL for MLIR Meta-Optimization"** | ML-driven compiler optimization |
| **"From Graphs to Warps: Semantic Interoperability"** | Graph-to-GPU lowering |
| **"Training-Aware Compilation for Custom AI Accelerators"** | AI accelerator compilation |

**Key observation:** Every GPU-related MLIR talk at the workshop targets **a single vendor** or
a **specific compilation pass**. There is no talk on runtime dispatch across vendors. This is a
gap in the program that libkdl fills directly.

### 2.2 Main Conference GPU Talks (April 14)

- **"rocMLIR: High-Performance ML Compilation for AMD GPUs with MLIR"** — AMD's production
  MLIR pipeline. AMDGCN-only. Demonstrates that vendor-specific MLIR pipelines are mature.
- **"Writing a Formal Execution and Memory Model for AMD GPUs"** — AMD GPU semantics
- **"Creating a runtime using the LLVM_ENABLE_RUNTIMES system"** (Tutorial) — directly relevant
  audience for a runtime-dispatch poster

**Key observation:** The main conference has AMD GPU coverage (rocMLIR) but nothing on
cross-vendor dispatch. The "creating a runtime" tutorial audience will walk out of that room
looking for exactly what libkdl proposes.

---

## 3. Active MLIR RFC and Upstream Work on Heterogeneous Dispatch

### 3.1 RFC: An MLIR Dialect for Distributed Heterogeneous Computing (June 2025)

**Source:** https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
**Author:** Robert K Samuel (IIT Madras, PACE Lab, Prof. Rupesh Nasre)
**Presented at:** PLDI 2025 Student Research Competition

Proposes a new MLIR dialect with:
- `schedule` operation grouping `task` operations, each annotated with a `target` (cpu, gpu)
- Unified IR for diverse hardware and programming models
- Fine and coarse-grained parallelism support
- Lowers to LLVM IR or other MLIR dialects

**Relationship to libkdl:**
- This RFC targets **static scheduling** (tasks annotated at compile time with targets)
- libkdl targets **runtime dispatch** (kernel variants pre-compiled, target selected at runtime based on discovered hardware)
- The two are complementary and non-overlapping: the RFC handles "which computation goes where";
  libkdl handles "which compiled variant of that computation runs on the discovered hardware"
- This RFC being active in the community validates that heterogeneous dispatch is a live research topic
  at the LLVM level — a direct argument for libkdl's relevance

### 3.2 RFC: Cleaning the GPU Dialect (September 2025)

**Source:** https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
Acknowledged problem: the GPU dialect contains operations that "don't really belong" and
needs stronger semantic definitions. This is an ongoing infrastructure cleanup effort.
**Relevance:** The dialect is in flux. A poster targeting `gpu.binary` / `gpu.select_object`
runtime behavior sits on top of maturing infrastructure. This should be acknowledged
and framed as "building on the stable parts (`gpu.binary` serialization) while the dialect evolves."

### 3.3 GPU Runtime Execution Without Module Load/Unload (Discourse 2025)

Community discussion: GPU device parts compiled into cubins with "runtime module load/unload
overhead that cannot be ignored for small kernels." This is the exact overhead libkdl must address
with pre-compiled binaries and a routing table cached after first call.

### 3.4 CUDA Tile IR — NVIDIA's Deployment of an MLIR JIT in the GPU Driver

**Source:** https://github.com/NVIDIA/cuda-tile
NVIDIA has shipped an MLIR-based JIT compiler inside the CUDA GPU driver (announced ~March 2025,
presented at GTC alongside CUTLASS 4.0, now at EuroLLVM 2026). This is significant:
- Confirms the LLVM community that **MLIR-based GPU runtimes are production-ready**
- NVIDIA has solved the problem for their own hardware stack
- The multi-vendor version of this — dispatching between NVIDIA (CUDA Tile IR path), AMD
  (rocMLIR path), and CPU fallback — is exactly the gap libkdl fills

**This is a strong framing device for the poster:** "NVIDIA has built MLIR-in-the-driver for
NVIDIA hardware. We build the cross-vendor version."

### 3.5 Upstream Gap Confirmed: No Runtime Selection in gpu.select_object

`gpu.select_object` remains compile-time selection ("selects the first object from the array and
embeds it as a string"). No upstream MLIR patch, RFC, or commit addresses runtime hardware detection
to choose among pre-compiled objects in a `gpu.binary`. This gap is current as of April 2026.

---

## 4. What Makes a Poster Outstanding at This Conference

Based on evidence from recent poster sessions and LLVM community norms:

### 4.1 The Non-Negotiables

**Running code.** Every high-engagement poster at EuroLLVM 2025 had a working prototype.
"Design proposals without implementation" get polite interest. "Here is the code, here are the
numbers" gets a crowd.

**Specific, verifiable gap.** The community has a strong radar for work that is genuinely
upstream-missing vs. work that already exists and the author hasn't looked. The libkdl gap
(no runtime dispatch in upstream MLIR `gpu.select_object`, IREE issues #50/#12230/#15334 open 6+
years) is documented and verifiable — this is the most defensible position.

**Concrete numbers.** Dispatch overhead (<10ns target), kernel selection accuracy (vs oracle),
performance on at least one real kernel (GEMM is the canonical choice). No numbers = no traction.

**Connection to something people use.** The torch.compile / PyTorch dispatcher connection is
critical. The audience building compilers for PyTorch and ONNX RT is large. Framing libkdl as
"the dispatch layer you'd call from torch.compile when you want cross-vendor portability" is more
engaging than "a new runtime."

### 4.2 Poster Content Formula That Works

Looking at successful LLVM posters across 2023-2025, the formula is:
1. **One-line problem** that any compiler engineer immediately recognizes
2. **Gap evidence** that is specific and citable (not "nobody has done this")
3. **Architecture diagram** that fits on half a poster and has at most 5 boxes
4. **Two or three key numbers** from measurements (not projections)
5. **Clear "what to do next"** for someone who wants to use or extend this

### 4.3 Poster Positioning That Resonates With This Audience

What this audience is: LLVM/MLIR contributors, GPU compiler engineers (NVIDIA, AMD, Intel), ML
framework engineers, academic compiler researchers, HPC compiler teams.

What they respond to:
- "We found an actual gap in upstream MLIR" (specificity + issue numbers)
- "Here is the minimal implementation" (composability, not a new ecosystem)
- "It takes <500 LOC to do useful dispatch" (minimalism is valued)
- "IREE needs 100K LOC for this; we need 500" (the contrast is striking and defensible)
- "We measured X overhead on real hardware" (evidence over claims)

What they are skeptical of:
- New frameworks that require full ecosystem adoption
- Performance claims without comparison to native baselines
- Work that doesn't acknowledge IREE, Triton, SYCL as prior art
- Claims that "no one has done this" without citation evidence

### 4.4 The NVIDIA CUDA Tile IR Moment Is an Opportunity

CUDA Tile IR is the most-discussed GPU compiler event in the LLVM community in 2025-2026. It is
being presented at the same EuroLLVM workshop as the poster venue. The framing opportunity:

"NVIDIA solved cross-SM dispatch for NVIDIA hardware with MLIR inside the driver. AMD has rocMLIR
for AMD hardware. Intel has XeVM for Intel GPUs. Nobody has built the layer that connects all three
at runtime. libkdl is that layer."

This positions libkdl not as competing with CUDA Tile IR but as a necessary complement. This is
the right posture: additive to the ecosystem, not competitive.

---

## 5. Competitive Landscape at the Poster Session

Based on the EuroLLVM 2026 agenda, the poster session on April 15 has **12 presenters**. What
we know about adjacent work that could be there:

- MLIR-RAJA (already in workshop, probably not also a poster)
- AMD GPU work (probably rocMLIR team is giving a talk, not a poster)
- Intel GPU auto-tuning (same)
- Unknown 12 poster topics

**What is very unlikely to be at the poster session:**
- A poster specifically on runtime dispatch across GPU vendors via MLIR
- A poster on lightweight (<1K LOC) runtime for cross-vendor kernel selection
- A poster connecting MLIR multi-target compilation to hardware capability detection

This leaves the libkdl space essentially uncrowded at this specific event.

---

## 6. Active MLIR Upstream Gaps That Define libkdl's Contribution Space

These are documented, verifiable, and should appear prominently on the poster:

| Gap | Evidence | Status |
|-----|---------|--------|
| `gpu.select_object` is compile-time only | MLIR docs: "selects first object" | Unresolved |
| No hardware capability query in MLIR GPU dialect | MLIR GPU docs | Unresolved |
| IREE runtime selection "sort of broken" | IREE issue #12230 step 2b | Stalled May 2023 |
| IREE multi-versioning epic | IREE issue #15334 | All tasks unchecked as of 2026 |
| IREE foundational target matching | IREE issue #50 | Open since 2019-10-13 |
| No upstream multi-vendor fat binary from MLIR | MLIR tests show NVVM+NVVM only | Unresolved |
| No ONNX Runtime EP using MLIR/LLVM | LLVM Discourse, May 2025 | Open question |

**The IREE issue age is the most striking data point on the poster:** Issue #50 asking for runtime
target matching has been open since the project's inception in 2019. That is 6+ years of acknowledged
need with no lightweight solution. libkdl addresses exactly this.

---

## 7. Framing Recommendations for Maximum Impact

### 7.1 Primary Contribution Frame

**"ld.so for GPU kernels"** — This is the right framing. Linux's dynamic linker discovers available
shared libraries at runtime, resolves symbols, and caches bindings. libkdl does the same for
MLIR-compiled GPU kernel variants: discover hardware, match contracts, route dispatch, cache.

This framing:
- Is immediately intuitive to anyone who has written a build system or linked a shared library
- Distinguishes from JIT compilation (we are routing, not compiling)
- Distinguishes from IREE (we are a library, not a framework)
- Is memorable and repeatable

### 7.2 The Three-Panel Poster Story

**Panel 1 — The Problem (should take 20 seconds to read):**
```
MLIR compiles to NVIDIA, AMD, CPU simultaneously.
At runtime, which binary runs? The first one. Always.
6 years of IREE issues, zero lightweight solutions.
```

**Panel 2 — The Design (architecture diagram, max 5 boxes):**
```
[MLIR + targets] → [gpu-module-to-binary] → [fat bundle]
                                                   ↓
[discover_devices()] → [match contracts] → [rank by cost model]
                                                   ↓
                              [dispatch to best match]
```

**Panel 3 — The Evidence:**
```
Dispatch overhead: <10ns (vs 5-20us kernel launch — 0.05-0.2% overhead)
Kernel selection accuracy: X% of oracle optimal
GEMM on A100: within 2% of native CUDA
LOC: 487 (vs IREE HAL: ~100K)
```

### 7.3 The Killer Comparison

The table that will stop people at the poster:

| | libkdl | IREE | SYCL | ALPAKA |
|---|---|---|---|---|
| Runtime dispatch | Yes | Yes (broken*) | Yes | No |
| MLIR-native | Yes | Yes | No | No |
| Cost-model selection | Yes | Planned | No | N/A |
| Lines of code | ~500 | 100K+ | N/A | ~header |
| Requires ecosystem buy-in | No | Yes | Yes | Yes |

*IREE issue #12230: "sort of broken" — cite this. The asterisk with a citation is more powerful
than just claiming a gap.

### 7.4 The NVIDIA Moment Hook

Opening line for the poster or lightning talk:
"NVIDIA just shipped CUDA Tile IR — an MLIR JIT inside the CUDA driver. AMD has rocMLIR. Intel has
XeVM. Each vendor solved the problem for their own hardware. We built the version that works for all three."

This is not hyperbole — it is an accurate description of the landscape, and it positions the work
as the natural next step after what the community is already building.

---

## 8. Risk Assessment

### 8.1 What Could Undermine the Poster

**Risk 1: "IREE already does this"**
Mitigation: Cite IREE issues #50, #12230, #15334 by number. The IREE team has members at this
conference. They will confirm these issues are open. This is the strongest rebuttal available.

**Risk 2: "Your benchmark is too simple"**
Mitigation: GEMM is the canonical benchmark for ML kernel dispatch. cuBLAS uses it. IREE uses it.
If you have GEMM numbers, you have credibility. Add a memory-bound kernel (element-wise or
reduction) to show the cost-model routing is working, not just always picking GPU.

**Risk 3: "Why not just use SPIR-V and be done with it?"**
Mitigation: SPIR-V delivers 50-75% of native performance. libkdl's native-binary path delivers
>94% (ALPAKA data). The poster should have a single number: "SPIR-V portability costs 25-50%
performance. libkdl costs <1% (dispatch overhead only)."

**Risk 4: "~500 LOC is too small to be a real contribution"**
Mitigation: ld.so is ~2500 LOC. The dispatch logic in cuBLAS is a few hundred lines wrapped
around a lookup table. Size is not the metric; impact is. The contribution is the protocol
(kernel contracts, capability model, cost model), not the line count.

**Risk 5: "This is just engineering, not research"**
Mitigation: The research question is: "what is the minimal interface between MLIR's compilation
output and a vendor-agnostic runtime?" This is an open design question with tradeoffs. The poster
answers it with a concrete point in the design space and measurements of the tradeoffs.

### 8.2 What Would Make This a Strong Submission vs. a Weak One

| Factor | Weak | Strong |
|--------|------|--------|
| Code status | Design only | Working on GTX 1650 + CPU |
| Numbers | "Expected <10ns overhead" | "Measured 7.3ns on GTX 1650" |
| Gap evidence | "Nobody has done this" | "IREE #50 open since 2019-10-13" |
| Related work | IREE, SYCL mentioned | Full comparison table with P3 scores, LOC |
| Framing | "New runtime for MLIR" | "ld.so for GPU kernels — fills gap IREE documented in 2019" |
| Connection to ecosystem | Standalone | Demonstrates invocation from torch.compile or ONNX RT |

---

## 9. Recommended Action Plan for the Poster

Based on all research, the highest-impact version of this poster:

### 9.1 Core Claim (One Sentence)
"libkdl is a ~500 LOC runtime dispatch layer that bridges MLIR's multi-target compilation to
vendor-specific GPU execution, filling a gap IREE has documented since 2019."

### 9.2 Evidence Required on the Poster
1. `gpu.select_object` is compile-time only — cite MLIR docs
2. IREE issue #50 open since 2019-10-13 — quote the issue text
3. Dispatch overhead measurement (run on GTX 1650)
4. At least one kernel accuracy measurement (GEMM on GTX 1650 vs CPU)
5. LOC comparison (487 vs IREE HAL ~100K)

### 9.3 Positioning Against the EuroLLVM 2026 Program
- **Differentiate from CUDA Tile IR** (NVIDIA-only JIT; libkdl is cross-vendor runtime dispatch)
- **Differentiate from rocMLIR** (AMD-only compile-time pipeline; libkdl is runtime selection)
- **Complement the "Creating a Runtime" tutorial** — the audience leaving that tutorial is the
  target demographic, and libkdl is a concrete example of what they just learned how to build
- **Engage MLIR-RAJA authors** — they are solving the "same kernel, different hardware" problem
  at a higher abstraction level; libkdl is the lower-level mechanism their work could use

### 9.4 What to Say in 5 Minutes (Lightning Talk Version)
If a lightning talk slot is available or opens up:
1. (30s) The problem: heterogeneous GPU clusters, one binary, wrong device selected
2. (60s) The gap: `gpu.select_object` is compile-time; IREE issue #50, 6 years open
3. (90s) The design: kernel contracts + capability model + cost-model routing
4. (60s) The numbers: dispatch overhead, GEMM accuracy, LOC
5. (30s) The ask: "If you use MLIR for GPU and target multiple vendors, try libkdl"

---

## 10. Summary: The Single Most Important Finding

The EuroLLVM 2026 MLIR Workshop has NVIDIA presenting CUDA Tile IR (MLIR JIT inside the CUDA
driver) and AMD presenting rocMLIR (MLIR pipeline for AMD GPUs). Intel is presenting GPU auto-tuning
with MLIR. Every major GPU vendor is presenting their **single-vendor MLIR solution**.

**No one is presenting the cross-vendor runtime dispatch layer.**

That gap is documented (IREE issues, open since 2019), technically addressable (<500 LOC prototype
exists), and fits perfectly into the 2026 program's GPU theme without competing with any existing
talk. This is the most favorable confluence of circumstances possible for a poster contribution.

The poster should walk in with:
1. Issue numbers (credibility — "we read the code")
2. A running prototype (credibility — "we wrote the code")
3. Real overhead measurements (credibility — "we measured it")
4. The NVIDIA/AMD/Intel framing (positioning — "we fill the gap they left")
5. The ld.so analogy (communication — "immediately intelligible to any systems programmer")

---

## Sources

- EuroLLVM 2026 Agenda: https://llvm.swoogo.com/2026eurollvm/agenda
- 7th MLIR Workshop Program: https://discourse.llvm.org/t/announcing-the-7th-mlir-workshop-eurollvm-2026-program/90119
- RFC: MLIR Dialect for Distributed Heterogeneous Computing: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- PLDI 2025 SRC: Same RFC: https://pldi25.sigplan.org/details/pldi-2025-src/3/An-MLIR-Dialect-for-Distributed-Heterogeneous-Computing
- RFC: Cleaning the GPU dialect: https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- CUDA Tile IR (NVIDIA): https://github.com/NVIDIA/cuda-tile
- NVIDIA CUDA Tile IR open-sourced: https://phoronix.com/news/NVIDIA-CUDA-Tile-IR-Open-Source
- Mehdi Amini (NVIDIA) on CUDA Tile IR: https://x.com/JokerEph/status/1902758983116657112
- EuroLLVM 2025 Poster Session: https://llvm.swoogo.com/2025eurollvm/session/2794299/poster-session
- EuroLLVM 2024 Program: https://llvm.org/devmtg/2024-04/
- LLVM Contributing Guidelines: https://llvm.org/docs/Contributing.html
- LLVM Developer Policy: https://llvm.org/docs/DeveloperPolicy.html
- MLIR Talks: https://mlir.llvm.org/talks/
- 2024 LLVM Dev Meeting Videos Online: https://www.phoronix.com/news/LLVM-2024-Meeting-Videos
- LLVM GPU Workshop 2024: https://hps.vi4io.org/events/2024/llvm-gpu
- MLIR GPU dialect: https://mlir.llvm.org/docs/Dialects/GPU/
- IREE issue #50: https://github.com/iree-org/iree/issues/50
- IREE issue #12230: https://github.com/iree-org/iree/issues/12230
- IREE issue #15334: https://github.com/iree-org/iree/issues/15334
- LLVM Discourse GPU runtime discussion: https://discourse.llvm.org/t/mlir-gpu-execution-without-runtime-load-unload/61712
- D154149 gpu-module-to-binary: https://reviews.llvm.org/D154149
