# Wave 07: LLVM Developers' Meeting GPU Landscape — 2023, 2024, 2025

**Search angles:**
- LLVM Developers' Meeting 2025 program schedule GPU talks
- LLVM DevMtg 2024 GPU MLIR offloading presentations
- GPU/Offloading Workshop 2023, 2024, 2025 agendas and slides
- "Not-Compiler Runtime Library GPUs" workshop talk 2025
- Where is libkdl positioned relative to recent community talks?

**Sources consulted:**
- llvm.org/devmtg/2025-10/ (official program)
- llvm.org/devmtg/2024-10/ (official program)
- llvm.swoogo.com/2024devmtg/agenda (full agenda with workshop details)
- llvm.swoogo.com/2025devmtg/agenda (full agenda with workshop details)
- discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832
- discourse.llvm.org/t/cfp-llvm-dev25-llvm-offload-workshop/88352
- discourse.llvm.org/t/announcing-the-preliminary-program-agenda-of-llvm-offload-workshop-llvm-developers-meeting-2024/82535
- discourse.llvm.org/t/pre-llvm-dev-23-gpu-offloading-pre-workshop-agenda/73775

**Date:** 2026-04-06

---

## Part 1: LLVM Dev Meeting 2025 — GPU/Offloading Landscape

### Conference Overview

**2025 US LLVM Developers' Meeting**
- Dates: October 28–29, 2025 (main), October 27 (pre-conference workshops)
- Location: Santa Clara Marriott, Santa Clara, California

---

### GPU/Offloading Workshop 2025

**Title:** "LLVM/Offload — Where are we, where are we going?"
**Time:** October 27, 8:30 AM – 12:30 PM, Grand Ballroom Salon E
**Format:** Mega-roundtables centered around introductory talks; open discussion interleaved with short presentations

**CFP solicited discussion topics** (from discourse.llvm.org/t/cfp-llvm-dev25-llvm-offload-workshop/88352):
- Runtime library support
- Offload API (API design, stability, feature gaps)
- Subregister handling in regalloc/scheduling
- SPIR-V
- Language support work

**Pre-workshop statement by jhuber6 (Joseph Huber, AMD), October 8, 2025:**
> "I'd like to talk a lot about where we came from and where we are going with GPU offloading in LLVM. Ideally we'll be able to have some longer discussions on the future of things like SPIR-V, SYCL, and MLIR runtime lowerings. I'll likely modify my conference slides."

**Slides released** (discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832, posted November 10, 2025):
1. "LLVM Offloading — Where are We Going?" — Google Slides link (53 views)
2. "SYCL status update.pdf" (104.9 KB)
3. "Not-Compiler Runtime Library GPUs.pdf" (633.4 KB)

**Critical observation:** The 633 KB "Not-Compiler Runtime Library GPUs" slide deck is the most directly relevant to libkdl. The size suggests a substantial presentation (~30+ slides). The title unambiguously addresses using LLVM's GPU infrastructure from non-compiler user-space code — the exact deployment model for libkdl. This talk's existence at the 2025 workshop confirms that the LLVM community is actively discussing how user-space programs access GPU kernel dispatch infrastructure, independently of compiler-driven code generation.

---

### Main Conference 2025 — GPU/Heterogeneous Talks

**Technical Talks:**

1. **The LLVM Offloading Infrastructure** — Joseph Huber (AMD)
   - Description: "Describes LLVM's offloading infrastructure enabling computation transfer to remote devices."
   - Slides: llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf (1.3 MB, image-only PDF)
   - Status: Huber's main conference talk is the paired companion to his workshop presentation. Together these are the two most important talks for situating libkdl.
   - Relevance: 10/10

2. **Taming GPU programming in safe Rust** — Manuel Drehwald
   - Description: "Exploring leveraging Rust's compiler guarantees for optimized, safe GPU programming in HPC and ML applications."
   - Slides: llvm.org/devmtg/2025-10/slides/technical_talks/drehwald.pdf
   - Context: Drehwald presented "Towards Rust (GPU) Offload" at the 2024 workshop. The 2025 main talk is a matured version.
   - Relevance to libkdl: Shows the community's interest in non-C++ GPU dispatch mechanisms. Rust GPU dispatch uses the same liboffload plugin infrastructure.
   - Relevance: 6/10

3. **CUTLASS Python DSL Infrastructure** — Guray Ozen
   - Slides: llvm.org/devmtg/2025-10/slides/technical_talks/ozen.pdf (CUTLASS Python DSL, Google, 5.3 MB)
   - Description: "Introduces Python DSL for writing high-performance GPU kernels with LLVM-based infrastructure."
   - Relevance: Demonstrates that the community is building high-level Python APIs over LLVM GPU infrastructure. CUTLASS kernels are prime candidates for a libkdl bundle — write once, dispatch to SM_80 or SM_89 or AMD or CPU based on runtime device detection.
   - Relevance: 7/10

4. **Building an LLVM-based Compiler Toolchain for Distributed Quantum Computing** — Vyacheslav Levytskyy
   - Focus: MLIR-based compiler infrastructure for quantum photonic networks
   - Note: Uses SPIR-V backend infrastructure, demonstrating SPIR-V's role beyond GPU targets
   - Relevance: 4/10 (context only)

**Quick Talks:**

5. **Extending ThinLTO Support for AMDGPU** — Shilei Tian
   - Slides: llvm.org/devmtg/2025-10/slides/quick_talks/tian.pdf
   - Focus: AMDGPU ThinLTO implementation challenges and new graph-based split scheme for parallel compilation
   - Relevance to libkdl: ThinLTO for AMDGPU improves AOT compilation quality for HIP kernels. Better ThinLTO outputs are better candidates for libkdl bundles (smaller, more optimized per-arch binaries). No overlap with dispatch runtime.
   - Relevance: 5/10

6. **Accelerating ML on Hexagon: Qualcomm's MLIR-Based Compiler** — Muthu Baskaran, Franck Slama
   - Focus: MLIR-based ML compiler targeting Hexagon DSP
   - Relevance: Heterogeneous dispatch extending beyond GPU to DSP; Hexagon as a potential libkdl target
   - Relevance: 5/10

7. **Optimizing IREE to Match llama.cpp** — Uiseop Eom
   - Focus: MLIR optimizations for LLM inference on IREE
   - Relevance: IREE's HAL dispatch is a competing runtime dispatch layer; performance data is relevant context
   - Relevance: 6/10

**Lightning Talks:**

8. **Mojo GPU Compilation** — Weiwei Chen, Abdul Dakkak
   - Description: "Mojo is a heterogeneous programming language...used extensively to unlock high performance on heterogeneous platforms."
   - Slides: llvm.org/devmtg/2025-10/slides/lightning_talks/chen_dakkak.pdf (9.8 MB)
   - Relevance: Mojo GPU compilation through MLIR/LLVM pipeline — another user of the offloading infrastructure libkdl builds on.
   - Relevance: 6/10

**Posters:**

9. **XeGPU: A High-Performance MLIR Dialect for Intel GPU Programming** — Chao Chen, Jianhui Li
   - Focus: Tile-based GPU kernel development with layout-guided programming model for Intel GPUs
   - Relevance: XeGPU is an Intel-specific MLIR dialect for kernel expression. If compiled through the SYCL/Level Zero path, XeGPU kernels are candidates for libkdl bundles targeting Intel GPUs. The dialect's existence confirms MLIR-native GPU kernel authoring is mainstream for Intel.
   - Relevance: 6/10

---

## Part 2: LLVM Dev Meeting 2024 — GPU/Offloading Landscape

### Conference Overview

**2024 US LLVM Developers' Meeting**
- Dates: October 23–24, 2024 (main), October 22 (pre-conference workshops)
- Location: Santa Clara Marriott, Santa Clara, CA

---

### GPU/Offloading Workshop 2024 — Complete Agenda

**Title:** "LLVM/Offload — Languages, Backends, and Features"
**Time:** 8:00 AM – 12:00 PM, Tuesday, October 22, 2024
**Location:** Hall of Cities — Newport/Santa Barbara
**Chair:** Johannes Doerfert (LLNL/AMD) and Shilei Tian

**Complete agenda** (from discourse.llvm.org/t/announcing-the-preliminary-program-agenda.../82535):

8:00–8:10 AM: **Welcome** — Johannes Doerfert (LLNL)

**Session 1** (8:10–10:00 AM):
- **OMPT Device Support in LLVM** — Dhruva Chakrabarti (AMD)
  - OpenMP tool-level observability of GPU kernel dispatch; runtime introspection infrastructure
- **Xbc: An Extensible Compiler for Heterogeneous Computing** — Fabian Mora (University of Delaware)
  - MLIR-based extensible compiler targeting X86, NVIDIA, AMD, and quantum. Reported comparable or better performance vs. vendor compilers on A100 and MI250x.
  - Note: Fabian Mora also presented "MLIR compiler for XBLang" at the 2023 pre-workshop, establishing this as ongoing work
- **Towards Rust (GPU) Offload** — Manuel Drehwald (University of Toronto)
  - Became the 2025 main conference technical talk
- **Thoughts and Results for an Offload-Specific Sanitizer** — Johannes Doerfert (LLNL)
  - GPU ASAN via software-managed virtual memory (paired with main conference talk)
- **Kernel-Info: An LLVM IR Pass for GPU Code Analysis** — Joel Denny (ORNL)
  - Static analysis pass extracting kernel resource requirements (registers, shared memory, occupancy)
  - Direct relevance: static kernel metadata extraction is a prerequisite for libkdl's capability contracts

10:00–10:30 AM: **Break**

**Session 2** (10:30–11:30 AM):
- **OpenMP Dispatch Support in LLVM** — Ravi Narayanaswamy (Intel)
  - Note: This is distinct from the main conference SYCL offloading talk — this is specifically about `omp declare variant` + dispatch directives
  - Direct relevance: OpenMP variant dispatch is the closest existing LLVM feature to libkdl's multi-version selection. The talk confirms that variant-based dispatch is being extended but remains OpenMP-scoped.
- **Automatic Parallelization and OpenMP Offloading of Fortran Array Notation** — Johannes Doerfert on behalf of Ivan Ivanov (Tokyo Institute of Technology)
  - Context: Fortran HPC workloads as GPU offloading consumers
- **Building GPU Runtimes With The LLVM Multi-Lib Infrastructure** — Joseph Huber (AMD)
  - High relevance: multi-lib infrastructure provides per-architecture library selection at link time. This is the static/link-time analog of libkdl's runtime selection. The talk confirms the community is aware of the multi-version problem at link time but has not extended the solution to runtime.

11:30–11:50 AM: **Open Discussion** — potential topic: LLVM/Offload API design

11:50 AM–12:00 PM: **Close** — Johannes Doerfert (LLNL)

---

### Main Conference 2024 — GPU/Heterogeneous Talks

**Technical Talks:**

1. **A C++ Toolchain for Your GPU** — Joseph Huber (AMD)
   - Slides: llvm.org/devmtg/2024-10/slides/techtalk/Huber-A-CPlusPlus-Toolchain-for-Your-GPU.pdf (1.4 MB, image-only)
   - Focus: Porting LLVM C library, runtime, and C++ runtime to GPU — enabling freestanding C++ compilation as standard GPU target
   - Direct relevance: This talk establishes that LLVM can now compile standard C/C++ for GPU natively (not just offload language extensions). libkdl's prototype in C is consistent with this trajectory — C programs that manage GPU kernels at the library level rather than via language annotations.
   - Relevance: 8/10

2. **Enhance SYCL Offloading Support to Use the New Offloading Model** — Ravi Narayanaswamy (Intel)
   - Slides: llvm.org/devmtg/2024-10/slides/techtalk/Narayanaswamy-EnhanceSYCL-offloading-support.pdf (323 KB, image-only)
   - Focus: clang-sycl-linker integration; per-kernel binary splitting; JIT/AOT compilation flows with metadata propagation
   - Key detail from wave-05: The per-kernel splitting model maps directly onto libkdl's kernel registry. SYCL's new flow confirms finer-grained kernel binary management is the direction.
   - Relevance: 9/10

3. **(Offload) ASAN via Software Managed Virtual Memory** — Johannes Doerfert (LLNL)
   - Focus: GPU address sanitizer avoiding memory overhead through virtual pointers
   - Relevance to libkdl: Demonstrates liboffload plugin infrastructure being extended for non-kernel-dispatch use cases (sanitizers). Shows the plugin layer is extensible.
   - Relevance: 5/10

4. **Advancing SPIR-V Backend Stability: Navigating GlobalISel Compromises** — Michal Paszkowski, Vyacheslav Levytskyy
   - Focus: SPIR-V backend development for OpenCL, SYCL/DPC++; pointer handling, extension integration
   - Relevance: SPIR-V backend stability directly affects libkdl's SPIR-V-as-portable-IR option. Ongoing GlobalISel compromises confirm that SPIR-V output quality is still being stabilized, supporting libkdl's multi-native-binary approach over single-SPIR-V.
   - Relevance: 7/10

**Quick Talks:**

5. **Instrumenting MLIR-Based ML Compilers for GPU Performance Analysis** — Corbin Robeck
   - Focus: GPU kernel performance bottleneck attribution across Python, C++, MLIR, LLVM, and ISA levels
   - Relevance: Performance attribution infrastructure needed to justify multi-version dispatch overhead. Relevant to libkdl's benchmarking methodology.
   - Relevance: 5/10

6. **Extending MLIR Dialects for Deep Learning Compilers** — Charitha Saumya, Jianhui Li
   - Focus: XeTile dialect for deep learning kernel compilation (Intel-specific)
   - Relevance: Another Intel-specific MLIR dialect producing kernels for the Level Zero plugin path
   - Relevance: 4/10

7. **Speeding Up Intel Gaudi Deep-Learning Accelerators Using MLIR-Based Compiler** — Jayaram Bobba
   - Focus: MLIR-based fusing compiler for Gaudi tensor cores, 54% model-level improvement
   - Relevance: Non-GPU accelerator dispatch via MLIR — demonstrates dispatch diversity beyond NVIDIA/AMD
   - Relevance: 5/10

**Posters:**

8. **accfg: Eliminating Setup Overhead for Accelerator Dispatch** — Anton Lydike, Josse Van Delm
   - Focus: Optimization dialect reducing accelerator configuration bottlenecks
   - Direct relevance: The "setup overhead for accelerator dispatch" problem is exactly the dispatch latency overhead that libkdl must minimize. The accfg work addresses this at the compiler-pass level; libkdl addresses it at the runtime selection level.
   - Relevance: 7/10

---

## Part 3: GPU/Offloading Workshop 2023 — Baseline

**Title:** "GPU Offloading with LLVM"
**Date:** October 10, 2023, 8:30 AM – 12:30 PM

**Talks** (from discourse.llvm.org/t/pre-llvm-dev-23-gpu-offloading-pre-workshop-agenda/73775):

Session 1:
- LLVM/libc on GPUs (Joseph Huber)
- LLVM/libm on GPUs (Anton Rydahl)
- libompx: portable wrappers for Thrust and friends (Mark Dewing)
- GPU offloading via LLVM/libcxx (Anton Rydahl)

Session 2:
- SPIRV update (Alexey Bader)
- HLSL in Clang (Justin Bogner)
- Flang offloading updates (Jan Sjödin)
- MLIR compiler for XBLang — extensible programming language targeting accelerators (Fabian Mora)

Session 3:
- CUDA-OMP, or Breaking the Vendor Lock (Johannes Doerfert)
- OpenMP Kernel language (Shilei Tian)
- GPU Kernel Compilation in Polygeist/MLIR (Ivan Radanov)
- GPU direct & LLVM-Test Suite on the GPU (Shilei Tian)

**Open discussion questions posed:**
- "Everyone hates language X, should we provide a generic 'LLVM offloading' API/runtime?"
- "Unifying drivers: is the 'new driver' good enough for everyone?"
- "Can we have a 'target independent' GPU IR?"

**Observation:** The 2023 open discussion question "should we provide a generic LLVM offloading API/runtime?" directly became the liboffload RFC (October 2023) — showing that workshop discussions feed directly into RFC and implementation work. The 2023 workshop is the origin point of the community trajectory that libkdl participates in.

---

## Part 4: Trend Analysis — Three Years of GPU Workshops

### Workshop Evolution (2023 → 2024 → 2025)

| Year | Title | Format | Scope |
|------|-------|--------|-------|
| 2023 | "GPU Offloading with LLVM" | Pre-workshop (half-day) | Bootstrapping: libc/libm on GPU, vendor lock-in, new driver |
| 2024 | "Languages, Backends, and Features" | Half-day (8:00–12:00) | Tooling: sanitizers, Rust offload, extensible compilers, multi-lib |
| 2025 | "Where are we, where are we going?" | Half-day (8:30–12:30) | Strategic: runtime APIs, SPIR-V future, MLIR runtime lowerings |

The theme progression is a maturity signal: 2023 was "build the infrastructure," 2024 was "extend the infrastructure," and 2025 was "decide on the direction." The 2025 theme — "where are we going?" — is the clearest possible signal that the community has not yet converged on answers to fundamental questions about runtime dispatch.

### What Has Been Presented (2023–2025)

**Covered exhaustively:**
- liboffload API design and roadmap
- New unified offloading driver (--offload-new-driver as default)
- Per-language runtime integration (SYCL, HIP, CUDA, OpenMP, Rust, Flang)
- SPIR-V backend stability
- GPU libc/libm/libcxx porting
- GPU sanitizers (ASAN, TSAN)
- OpenMP device tool support (OMPT)
- Extensible MLIR compilers for heterogeneous targets (xbc, XBLang)
- GPU runtime multi-lib selection (link-time)
- GPU kernel static analysis (Kernel-Info pass)
- CUTLASS/CUDA kernel expression via Python DSL
- Intel XeGPU MLIR dialect
- GPU programming safety (Rust)

**Covered partially or indirectly:**
- OpenMP variant dispatch (omp declare variant) — addressed as language feature, not runtime mechanism
- Dispatch overhead — mentioned but not benchmarked comprehensively
- Non-compiler runtime dispatch — "Not-Compiler Runtime Library GPUs" talk in 2025 is the only direct address, and its content (633 KB, presented October 27, 2025) is not yet analyzed

**Conspicuously absent across all three years:**
- Runtime multi-version kernel selection (selecting among pre-compiled variants at runtime based on device capability)
- Dynamic kernel loading (dlopen-style kernel discovery and loading — Issue #75356 from November 2023 is still open as of April 2026)
- Kernel bundle packaging format for multi-target dispatch (the MTB format libkdl defines)
- User-space dispatch policy layers (the "ranking" problem: given N compatible kernels, which to dispatch?)
- Fallback dispatch chains (cascade from vendor-native to SPIR-V to CPU)
- Cross-framework kernel portability (same kernel running in PyTorch context vs. standalone runtime context)
- Benchmark-informed dispatch (using measured performance data to drive selection, not just capability matching)

---

## Part 5: libkdl Positioning — Where It Stands

### Relationship to Main Conference Talks

**Direct complement to Huber's 2025 talk ("The LLVM Offloading Infrastructure"):**
Huber's talk describes the mechanism layer (liboffload, `ol*` API, plugin architecture). libkdl is the policy layer above it. The two talks are naturally paired: mechanism (Huber) + policy (libkdl). At Dublin 2026, this pairing is a natural positioning strategy — the poster can open with "Huber's 2025 talk at this meeting described where LLVM/Offload is going; libkdl demonstrates one concrete destination."

**Direct response to the 2025 workshop theme:**
The question "Where are we going?" on offloading received Huber's answer ("SPIR-V, SYCL, and MLIR runtime lowerings"). libkdl provides a complementary, concrete answer: "a policy layer for multi-version runtime selection, implementing the missing piece that liboffload explicitly excludes from its roadmap."

**Picks up where "Building GPU Runtimes With The LLVM Multi-Lib Infrastructure" (2024) leaves off:**
Huber's 2024 workshop talk addressed multi-lib selection at link time. libkdl addresses multi-version selection at runtime. The two are complementary: link-time selection handles cases where the target architecture is known at build time; runtime selection handles cases where it is not (cloud deployment, heterogeneous clusters, portable ML inference runtimes).

**Direct response to "Kernel-Info: An LLVM IR Pass for GPU Code Analysis" (ORNL, 2024):**
Joel Denny's Kernel-Info pass extracts static kernel resource metadata (registers, shared memory, occupancy) from LLVM IR. This is exactly the metadata that populates libkdl's capability contracts. A potential collaboration: Kernel-Info outputs → libkdl MTB bundle metadata → libkdl dispatch policy. The poster can cite this as "the community has the metadata extraction tool; libkdl provides the runtime consumer."

**Directly addresses the 2023 open discussion question:**
"Should we provide a generic 'LLVM offloading' API/runtime?" — The answer from the community over 2023–2025 was "yes, and liboffload is that API." libkdl addresses the follow-up question that was never asked: "Should we provide a policy layer above that API?" libkdl is a concrete existence proof that such a layer is implementable and useful.

### The "Not-Compiler Runtime Library GPUs" Gap

The 2025 workshop talk "Not-Compiler Runtime Library GPUs" (633 KB, October 27, 2025) is unanalyzed by the survey (PDF is image-only). This talk is the single most directly relevant community discussion to libkdl's contribution. Based on the title and context:
- It addresses using LLVM's GPU runtime infrastructure from user-space programs not generated by the LLVM compiler toolchain
- This is exactly the use case libkdl implements: a C library that any program (Python, Rust, C) can call to dispatch pre-compiled GPU kernels
- The talk almost certainly discusses the limitations of the compile-time registration model for user-space dispatch — the same limitations that motivate Issue #75356

**Action item:** If the Google Slides version of "LLVM Offloading — Where are We Going?" (linked from the workshop slides thread) is accessible, its content would be the highest-value external source for the Dublin poster.

### Community Appetite Signal

The annual GPU/Offloading Workshop has grown in scope and attendance:
- 2023: Pre-workshop (informal), ~50 attendees estimated
- 2024: Half-day workshop with formal agenda, ~100 attendees estimated (354 Discourse thread views)
- 2025: Half-day workshop with CFP, mega-roundtable format, 281 CFP views + 231 slides-post views

The growing workshop indicates an active, engaged community. A Dublin 2026 poster on libkdl enters this community at peak engagement on the topic. Huber's explicit statement that he wants "longer discussions on the future of SPIR-V, SYCL, and MLIR runtime lowerings" confirms that these are open questions — exactly the design space where libkdl's prototype provides concrete data.

---

## Angle Assessment

### What Has the LLVM DevMtg Community Presented About GPU Dispatch?

**Established (well-covered, consensus exists):**
- liboffload is the unified GPU dispatch mechanism for LLVM
- The new unified driver (--offload-new-driver) is the standard compilation path (LLVM 20+)
- Per-language runtimes (SYCL via Unified Runtime, HIP, CUDA, OpenMP) are all migrating to the liboffload plugin model
- SPIR-V backend stability is improving but not yet production-grade for all GPU targets
- Per-kernel binary splitting (SYCL clang-sycl-linker) is standard for fine-grained kernel management

**Discussed but unresolved (active community questions):**
- SPIR-V as the single portable GPU IR — RFC proposed March 2025, not yet implemented
- MLIR runtime lowerings for offloading — mentioned by Huber as a key 2025 discussion topic, no concrete design
- SYCL/Unified Runtime vs. liboffload unification — coexisting standards, no merger plan
- API stability of the `ol*` API (renamed once already, still evolving)
- The role of `offload/` as universal GPU runtime home vs. language-specific runtimes

**Not discussed, confirmed absent:**
- Runtime multi-version kernel selection policy
- Dynamic kernel loading (Issue #75356 from 2023, unresolved)
- User-space kernel dispatch libraries (partially addressed by "Not-Compiler Runtime Library GPUs" in 2025)
- Cross-vendor kernel bundle packaging format
- Dispatch fallback chains
- ML-inference-specific kernel dispatch patterns

### Where libkdl Stands in This Landscape

libkdl occupies a gap that is:
1. **Community-recognized** — The 2023 open discussion and Issue #75356 explicitly name the problem
2. **Officially out-of-scope for liboffload** — The roadmap RFC explicitly defers "multi-version selection policy" to future work
3. **Addressed by a 2025 workshop talk** — "Not-Compiler Runtime Library GPUs" shows the community is discussing user-space dispatch, but has no prototype
4. **Ready for a concrete prototype** — libkdl at ~5100 LOC is the first concrete implementation in the LLVM ecosystem

The Dublin 2026 poster should explicitly cite:
- Huber DevMtg 2025: "The LLVM Offloading Infrastructure" — as the mechanism layer libkdl builds on
- "Building GPU Runtimes With The LLVM Multi-Lib Infrastructure" (2024 workshop) — as the link-time analog libkdl extends to runtime
- "Kernel-Info: LLVM IR Pass for GPU Code Analysis" (2024 workshop, ORNL) — as the metadata extraction tool that feeds libkdl
- Issue #75356 "Name-Based Kernel Loading" — as the LLVM community's own articulation of the dynamic loading gap libkdl fills
- The 2025 workshop theme "Where are we going?" — as the open question libkdl answers concretely

### Relevance to libkdl (1–10): 10

The LLVM DevMtg GPU workshop series is the primary venue where libkdl's contribution must be positioned. The workshop is now annual, well-attended, and organized by the exact engineers (Huber, Doerfert, Tian, Narayanaswamy) who maintain the infrastructure libkdl builds on. A Dublin 2026 poster is presented to an audience that has attended these workshops — they understand the problem space. The poster does not need to explain what liboffload is; it needs to explain what libkdl adds on top.

### Novelty of this wave (1–10): 9

This wave provides the first systematic catalog of three years of LLVM DevMtg GPU talks with explicit mapping to libkdl's contribution. Prior waves established what the community built; this wave establishes what the community discussed and what is absent from those discussions.

---

## Cross-References to Prior Waves

- liboffload `ol*` API: wave-02-llvm-offloading.md, wave-06-llvm-offload-new-driver.md
- Issue #75356 (dynamic kernel loading): wave-06-llvm-offload-new-driver.md (Source 9)
- OffloadBinary format: wave-06-llvm-offload-new-driver.md (Source 7)
- GPU dialect cleanup RFC: wave-05-llvm-discourse-rfcs.md (Source 11)
- Mechanism/policy split: wave-05-llvm-discourse-rfcs.md (Sources 2, 4)
- SPIR-V-as-portable-IR RFC: wave-05-llvm-discourse-rfcs.md (Source 6)
- liboffload roadmap deferral: wave-05-llvm-discourse-rfcs.md (Source 2)
- XeGPU / Intel MLIR dialect: wave-02-sycl-multi-target.md

---

## Suggested Follow-Up Angles

1. **"Not-Compiler Runtime Library GPUs" slides** — The Google Slides link for "LLVM Offloading — Where are We Going?" (workshop key talk) appears to be accessible (the discourse post links to it). Fetch and read the slides to extract any direct discussion of user-space dispatch requirements that libkdl addresses.

2. **"Kernel-Info" pass source** — Search LLVM GitHub for the Kernel-Info LLVM IR pass from Joel Denny (ORNL). If upstreamed, its output format defines the vocabulary for libkdl's capability contracts.

3. **OpenMP `declare variant` dispatch semantics** — Ravi Narayanaswamy's 2024 workshop talk "OpenMP Dispatch Support in LLVM" covers the closest existing LLVM analog to multi-version kernel dispatch. Understanding its design constraints explains why libkdl is needed as a separate library rather than an extension of OpenMP dispatch.

4. **Xbc paper (Zenodo 2024)** — Fabian Mora's "Xbc: An Extensible Compiler for Heterogeneous Computing" reports performance on NVIDIA A100 and AMD MI250x. This is benchmarking data from a directly comparable system (MLIR-based heterogeneous compilation) that libkdl's benchmark section should acknowledge and distinguish from.

5. **LLVM Developers' Meeting 2026 (Dublin)** — The submission deadline for the Dublin meeting poster session. The wave-07-llvm-poster-criteria.md file already exists — cross-reference its criteria against the community topics identified in this wave to verify alignment.
