# LLVM Community Talks on GPU Dispatch, Offloading, and MLIR GPU Compilation
## Research for Combo A Poster — EuroLLVM 2026 Dublin

**Compiled:** 2026-04-09
**Purpose:** Catalog LLVM community talks (2023–2026) on GPU offloading, MLIR GPU, and kernel dispatch to position the Combo A contribution (libkdl policy layer + MTB format + dispatch benchmarks).
**Primary sources:** llvm.org/devmtg, LLVM Discourse, YouTube (@LLVMPROJ), EuroLLVM 2025 program, EuroLLVM 2026 program.

---

## 1. GPU/Offloading Workshop — 2023 US LLVM Developers' Meeting

**Event:** Pre-workshop day, October 10, 2023, Santa Clara
**Theme:** "GPU Offloading with LLVM"
**Discourse thread:** https://discourse.llvm.org/t/pre-llvm-dev-23-gpu-offloading-pre-workshop-agenda/73775

### Session 1: GPU Libraries

| Talk | Speaker | Relevance to Combo A |
|------|---------|----------------------|
| LLVM/libc on GPUs | Joseph Huber (AMD) | Establishes libc on GPU as foundation; libkdl's C-native design is compatible |
| LLVM/libm on GPUs | Anton Rydahl | Same trajectory — standard library availability on GPU |
| libompx: portable wrappers for Thrust | Mark Dewing | Shows portable abstraction layers being built over GPU runtimes |
| GPU offloading via LLVM/libcxx | Anton Rydahl | C++ runtime on GPU — same trajectory as Huber's 2024 main talk |

### Session 2: Language/IR

| Talk | Speaker | Relevance to Combo A |
|------|---------|----------------------|
| SPIR-V update | Alexey Bader (Intel) | SPIR-V backend status; informs Combo A's SPIR-V fallback slot in MTB |
| HLSL in Clang | Justin Bogner | Adds another language producing GPU binaries — all are candidates for MTB bundles |
| Flang offloading updates | Jan Sjödin | Fortran HPC workloads producing GPU kernels via new unified driver |
| MLIR compiler for XBLang | **Fabian Mora** (U. Delaware) | **Key:** Mora's extensible MLIR compiler for multi-target GPU; he is now at the EuroLLVM 2026 MLIR Workshop (ASTER talk) — direct community continuity |

### Session 3: OpenMP GPU / Portability

| Talk | Speaker | Relevance to Combo A |
|------|---------|----------------------|
| CUDA-OMP, or Breaking the Vendor Lock | Johannes Doerfert (LLNL) | Motivates vendor-agnostic dispatch; the exact problem Combo A solves at runtime |
| OpenMP Kernel Language | Shilei Tian (LLNL) | OpenMP as GPU kernel language; extension to llvm-hpc-2023 paper |
| GPU Kernel Compilation in Polygeist/MLIR | Ivan Radanov | MLIR-based GPU kernel compilation path |
| GPU direct & LLVM-Test Suite on GPU | Shilei Tian | Test infrastructure for GPU dispatch |

**Open discussion questions posed at this workshop:**
> "Everyone hates language X, should we provide a generic 'LLVM offloading' API/runtime?"
> "Can we have a 'target independent' GPU IR?"

**Combo A relevance:** These two questions are the founding motivation for both liboffload (2023 RFC that followed this discussion) and for Combo A's MTB format (a target-independent kernel bundle). The poster can open by citing these workshop questions as the problem statement that launched the liboffload trajectory — and then position Combo A as addressing the remaining unanswered piece (runtime selection policy).

---

## 2. Main Conference — 2023 US LLVM Developers' Meeting

**Event:** October 11–12, 2023, Santa Clara
**Program:** https://llvm.org/devmtg/2023-10/

### GPU-Related Technical Talks

**The LLVM C Library for GPUs**
- Speaker: Joseph Huber (AMD)
- Video: https://youtu.be/_LLGc48GYHc
- Slides: https://llvm.org/devmtg/2023-10/slides/techtalks/Huber-LibCforGPUs.pdf (1.7 MB)
- Key points: Porting LLVM libc to GPU targets; freestanding C++ on GPU; host service invocation from device code.
- Combo A relevance: libkdl is written in C and calls liboffload from user-space. Huber's libc-on-GPU work establishes that C-native GPU tooling is a community-endorsed direction. Direct precedent for libkdl's design choice.

**Mojo: A System Programming Language for Heterogeneous Computing** (Keynote)
- Speakers: Abdul Dakkak, Chris Lattner, Jeff Niu (Modular)
- Video: https://youtu.be/SEwTjZvy8vw
- Slides: https://llvm.org/devmtg/2023-10/slides/keynote/Mojo.pdf
- Key points: Python-family language built on MLIR+LLVM for heterogeneous computing (CPU+GPU). Compile-time heterogeneous dispatch via MLIR lowering.
- Combo A relevance: Mojo kernels are compile-time dispatched; Combo A addresses the runtime dispatch gap when the target is not known at compile time. Natural framing: "Mojo solves compile-time dispatch; libkdl solves runtime dispatch."

**Optimization of CUDA GPU Kernels and Translation to AMDGPU in Polygeist/MLIR** (Student)
- Speaker: Ivan Ivanov
- Video: https://youtu.be/W7-YIYb9ulc
- Key points: Target-agnostic parallel GPU kernel representations in MLIR; CUDA-to-AMDGPU translation at MLIR level.
- Combo A relevance: Shows MLIR-based cross-vendor kernel translation is feasible. Combo A's MTB format stores pre-compiled native binaries rather than translating on the fly, but the goal is the same — one authoring effort, multiple target execution.

**OpenMP Kernel Language Extensions for Performance Portable GPU Codes** (Student)
- Speaker: Shilei Tian
- Video: https://youtu.be/EhA4ZCDwzfI
- Key points: Transforming OpenMP into a kernel language; CUDA-to-OpenMP porting with minimal modification.
- Combo A relevance: OpenMP variant dispatch (`omp declare variant`) is the closest existing LLVM mechanism to Combo A's multi-version selection, but it is compile-time only. This talk confirms the community is pushing OpenMP toward kernel-language use, but runtime dynamic dispatch is not addressed.

**Leveraging MLIR for Loop Vectorization and GPU Porting of FFT Libraries** (Student)
- Speaker: Yifei He
- Video: https://youtu.be/8xDF4qku3AI
- Slides: https://llvm.org/devmtg/2023-10/slides/student-talks/He-LeveragingMLIRforLoopVectorization.pdf
- Key points: MLIR-based porting of FFT libraries across CPU and GPU targets.
- Combo A relevance: FFT libraries are exactly the kind of multi-architecture kernel suites that Combo A's MTB format is designed to package.

---

## 3. GPU/Offloading Workshop — 2024 US LLVM Developers' Meeting

**Event:** October 22, 2024, 8:00 AM–12:00 PM, Santa Clara
**Theme:** "LLVM/Offload — Languages, Backends, and Features"
**Chair:** Johannes Doerfert (LLNL/AMD), Shilei Tian
**Discourse:** https://discourse.llvm.org/t/announcing-the-preliminary-program-agenda-of-llvm-offload-workshop-llvm-developers-meeting-2024/82535

### Full Workshop Agenda

**OMPT Device Support in LLVM**
- Speaker: Dhruva Chakrabarti (AMD)
- Key points: OpenMP tool-level (OMPT) observability of GPU kernel dispatch. Runtime introspection infrastructure for profiling, tracing, and debugging GPU kernel launches.
- Combo A relevance: OMPT observability is the monitoring layer. Combo A's dispatch policy layer sits between OMPT and the application. If OMPT can observe what Combo A dispatches, the whole system is debuggable without modifying application code.

**Xbc: An Extensible Compiler for Heterogeneous Computing**
- Speaker: **Fabian Mora** (University of Delaware)
- Key points: MLIR-based extensible compiler targeting X86, NVIDIA GPU (A100), AMD GPU (MI250x), and quantum. Comparable or better performance vs. vendor compilers. Uses a domain-specific MLIR dialect (XBLang) with target-specific lowering passes.
- Combo A relevance (HIGH): Mora's xbc is the most directly comparable compiler-side project. His 2023 workshop talk introduced XBLang; this 2024 talk shows production results. **Mora is now presenting ASTER at EuroLLVM 2026 Dublin** (the same venue as the Combo A poster), making this a direct community continuity citation. Poster should cite xbc as establishing that MLIR-based multi-target compilation is feasible, while Combo A addresses the runtime dispatch side.
- Note: xbc paper published on Zenodo 2024 — reports 5x performance over Clang OpenMP on AMD MI250x in some workloads.

**Towards Rust (GPU) Offload**
- Speaker: Manuel Drehwald (University of Toronto)
- Key points: Rust safety guarantees applied to GPU offloading via the liboffload plugin infrastructure. Later became the 2025 main conference technical talk.
- Combo A relevance: Demonstrates that non-C++ user-space code can dispatch GPU kernels through the liboffload plugin stack. Combo A's C API is analogous — language-agnostic dispatch through the same plugin layer.

**Thoughts and Results for an Offload-Specific Sanitizer**
- Speaker: Johannes Doerfert (LLNL)
- Key points: GPU ASAN via software-managed virtual memory; paired with the main conference talk (Offload ASAN).
- Combo A relevance: Indirect. Shows the liboffload plugin infrastructure is being extended for non-dispatch use cases, confirming its architectural extensibility.

**Kernel-Info: An LLVM IR Pass for GPU Code Analysis**
- Speaker: **Joel Denny** (ORNL)
- Key points: Static analysis LLVM IR pass extracting GPU kernel resource statistics (register usage, shared memory allocation, theoretical occupancy). Operates at LLVM IR level — target-agnostic. Results are summary statistics + per-occurrence source locations.
- Combo A relevance (HIGH): The Kernel-Info pass extracts exactly the metadata that would populate Combo A's capability contracts in the MTB format. **This is a direct toolchain connection**: Kernel-Info (static extraction at compile time) → MTB bundle metadata field → libkdl dispatch policy (runtime selection using that metadata). The poster should explicitly propose this as a pipeline: LLVM produces kernels, Kernel-Info annotates them, kdl-pack bundles them, libkdl dispatches them at runtime.
- Documentation: https://llvm.org/docs/KernelInfo.html
- Test cases: `llvm/test/Analysis/KernelInfo`

**OpenMP Dispatch Support in LLVM**
- Speaker: Ravi Narayanaswamy (Intel)
- Key points: `omp declare variant` and dispatch directives for selecting kernel variants based on platform conditions. Compile-time variant selection, not runtime.
- Combo A relevance: This is the closest existing LLVM analog to Combo A's multi-version dispatch — but it operates at compile time and is OpenMP-scoped. The talk confirms the community understands the variant selection problem; it does not propose a runtime solution. Combo A's runtime selection is complementary.

**Building GPU Runtimes With The LLVM Multi-Lib Infrastructure**
- Speaker: Joseph Huber (AMD)
- Key points: Using LLVM's multi-lib infrastructure for per-architecture library selection at link time. Static/AOT selection among compiled-for-specific-architecture libraries.
- Combo A relevance (HIGH): This is the link-time analog to Combo A's runtime-time selection. **Key distinction**: multi-lib selects at link time (target architecture must be known at build time); Combo A selects at runtime (target architecture is discovered at process start). For cloud deployment, ML inference containers, and portable research software, the target is not known at build time — this is the gap Combo A fills. The poster should position Combo A as "extending multi-lib selection from link-time to runtime."

---

## 4. Main Conference — 2024 US LLVM Developers' Meeting

**Event:** October 23–24, 2024, Santa Clara
**Program:** https://llvm.org/devmtg/2024-10/

### GPU-Related Technical Talks

**A C++ Toolchain for Your GPU**
- Speaker: Joseph Huber (AMD)
- Video: https://youtu.be/4TxGWis1mws
- Slides: https://llvm.org/devmtg/2024-10/slides/techtalk/Huber-A-CPlusPlus-Toolchain-for-Your-GPU.pdf
- Key points: Porting LLVM C library, compiler runtime, and C++ runtime to GPU. Enabling freestanding C++ compilation as a standard GPU target (not through language extension annotations). Treating GPU as a hosted target.
- Combo A relevance: libkdl (C, ~5100 LOC) embeds pre-compiled GPU kernels and dispatches them from host C code. Huber's work establishes the C/C++ standard library on GPU; Combo A's approach is consistent — user-space C managing GPU kernels via library calls rather than compiler annotations.

**Enhance SYCL Offloading Support to Use the New Offloading Model**
- Speaker: Ravi Narayanaswamy (Intel)
- Video: https://youtu.be/4Qof7vtfhuk
- Slides: https://llvm.org/devmtg/2024-10/slides/techtalk/Narayanaswamy-EnhanceSYCL-offloading-support.pdf
- Key points: `clang-sycl-linker` integration; per-kernel binary splitting at finalization; JIT/AOT flows; AOT compilation for Intel, AMD, NVIDIA GPUs via unified fat-object.
- Combo A relevance (HIGH): The per-kernel binary splitting model maps directly onto Combo A's kernel registry design. After SYCL splitting, each kernel is individually addressable — exactly the granularity at which `kdl_load_kernel(bundle, "kernel_name")` operates. This talk confirms that finer-grained kernel binary management is the direction the whole LLVM ecosystem is moving.

**(Offload) ASAN via Software Managed Virtual Memory**
- Speaker: Johannes Doerfert (LLNL/AMD)
- Video: https://youtu.be/B60jp4khrvc
- Slides: https://llvm.org/devmtg/2024-10/slides/techtalk/Doerfert-Offload-ASAN.pdf
- Key points: GPU sanitizer leveraging virtual pointer indirection to avoid memory overhead.
- Combo A relevance: Demonstrates the liboffload plugin layer is extensible beyond basic kernel dispatch.

**Advancing SPIR-V Backend Stability**
- Speakers: Michal Paszkowski, Vyacheslav Levytskyy (Intel)
- Video: https://youtu.be/oLuTsD4mLXE
- Slides: https://llvm.org/devmtg/2024-10/slides/techtalk/Paszkowski-Levytskyy-AdvancingSPIR-V-BackendStability.pdf
- Key points: SPIR-V backend GlobalISel compromises; opaque pointer handling; type inference for OpenCL/SYCL/DPC++/Vulkan.
- Combo A relevance: SPIR-V backend quality directly affects Combo A's SPIR-V fallback slot in the MTB format. Ongoing stability work in 2024 confirms that SPIR-V output quality is improving but not yet production-grade for all GPU targets — this supports Combo A's strategy of keeping native binaries as primary dispatch targets with SPIR-V as a fallback.

### Quick Talks (2024)

**Instrumenting MLIR-Based ML Compilers for GPU Performance Analysis**
- Speaker: Corbin Robeck
- Key points: GPU kernel performance bottleneck attribution across Python, C++, MLIR, LLVM, and ISA levels.
- Combo A relevance: Performance attribution infrastructure needed to justify Combo A's dispatch overhead benchmarks. The methodology framework is directly applicable.

**accfg: Eliminating Setup Overhead for Accelerator Dispatch** (Poster)
- Speakers: Anton Lydike, Josse Van Delm
- Key points: Optimization dialect reducing accelerator configuration bottlenecks (setup overhead for accelerator dispatch).
- Combo A relevance (HIGH): The "setup overhead for accelerator dispatch" problem is exactly the dispatch latency overhead that Combo A measures in its benchmark suite. accfg addresses this at the compiler-pass level (reducing redundant setup instructions); Combo A addresses it at the runtime selection level (reducing policy overhead in the selection path). **The poster should cite accfg as establishing that dispatch setup overhead is a recognized compiler research problem**, validating Combo A's benchmark contribution.

---

## 5. GPU/Offloading Workshop — 2025 US LLVM Developers' Meeting

**Event:** October 27, 2025, 8:30 AM–12:30 PM, Grand Ballroom Salon E, Santa Clara
**Theme:** "LLVM/Offload — Where are we, where are we going?"
**Format:** Mega-roundtables centered on short introductory talks with open discussion interleaved
**CFP:** https://discourse.llvm.org/t/cfp-llvm-dev25-llvm-offload-workshop/88352
**Slides posted:** https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832 (November 2025)

### CFP-Solicited Topics (confirms community agenda)
- Runtime library support
- Offload API design, stability, feature gaps
- Subregister handling in regalloc/scheduling
- SPIR-V as portable IR
- Language support work

**Pre-workshop statement by Joseph Huber (jhuber6, AMD), October 8, 2025:**
> "I'd like to talk a lot about where we came from and where we are going with GPU offloading in LLVM. Ideally we'll be able to have some longer discussions on the future of things like SPIR-V, SYCL, and MLIR runtime lowerings. I'll likely modify my conference slides."

### Workshop Slides Released (November 10, 2025)

**"LLVM Offloading — Where are We Going?"**
- Speaker: Joseph Huber (AMD)
- Format: Google Slides (linked from Discourse thread, image-only PDF at `llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf`, 1.3 MB)
- Key claims (based on title + context): State-of-the-art survey of liboffload architecture; roadmap for SPIR-V integration; MLIR runtime lowering discussion.
- Combo A relevance (10/10): This is the single most important community talk for positioning Combo A. Huber's talk describes the mechanism layer (liboffload, `ol*` API, plugin architecture). Combo A is the policy layer above it. **The Dublin 2026 poster can open**: "Huber's 2025 talk at this meeting described where LLVM/Offload is going — libkdl demonstrates one concrete destination: a runtime policy layer for multi-version kernel selection."

**"SYCL status update"**
- Size: 104.9 KB PDF (suggests ~10 slides)
- Key points: libsycl + Unified Runtime integration status; SYCL's relationship to liboffload.
- Combo A relevance: Confirms SYCL is settling on the new offloading model; SYCL kernels compiled via new model are MTB-bundleable.

**"Not-Compiler Runtime Library GPUs"**
- Size: 633.4 KB PDF (suggests ~30+ slides) — significant presentation
- Key points (inferred from title + context): Using LLVM's GPU runtime infrastructure from user-space applications that are not generated by the LLVM compiler toolchain.
- Combo A relevance (**CRITICAL — 10/10**): This talk addresses the exact use case Combo A implements — a C library that any program (Python extension, Rust binary, C application) can call to dispatch pre-compiled GPU kernels without going through a compiler toolchain. **This talk is the community's first direct acknowledgment that this use case exists and is not yet served.** Combo A's libkdl is a concrete implementation of what this talk presumably describes as a gap. If the slide content can be accessed, it is the highest-value external source for the Dublin poster.

---

## 6. Main Conference — 2025 US LLVM Developers' Meeting

**Event:** October 28–29, 2025, Santa Clara
**Program:** https://llvm.org/devmtg/2025-10/

### GPU-Related Technical Talks

**The LLVM Offloading Infrastructure**
- Speaker: Joseph Huber (AMD)
- Video: https://youtu.be/pndPwVMouPg
- Slides: https://llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf (1.3 MB, image-only)
- Key points: Comprehensive description of LLVM's offloading infrastructure — liboffload, `ol*` API, plugin architecture, device/kernel/queue abstractions, fat-object format.
- Combo A relevance (10/10): **The mechanism layer that Combo A builds on.** libkdl calls `olCreateProgram()`, `olCreateKernel()`, `olEnqueueKernelLaunch()` via the liboffload plugin API. The poster should present this as a layered architecture diagram: Huber's infrastructure below, Combo A's policy layer above.
- Supporting claim: Huber's talk explicitly positions liboffload as enabling "developers to offload computations to a remote device" — generic enough to encompass user-space libraries, not just compiler-generated code.

**Taming GPU Programming in Safe Rust**
- Speaker: Manuel Drehwald (University of Toronto)
- Video: https://youtu.be/ASUek97s5P0
- Slides: https://llvm.org/devmtg/2025-10/slides/technical_talks/drehwald.pdf
- Key points: Rust safety guarantees for GPU offloading in HPC and ML; uses the same liboffload plugin infrastructure.
- Combo A relevance: Shows non-C++ languages calling through liboffload. Combo A's C API enables the same for Python extensions, Julia, etc. Community validation that liboffload as a universal GPU dispatch API is the right abstraction.

**CUTLASS Python DSL Infrastructure** (Guray Ozen, Google)
- Slides: https://llvm.org/devmtg/2025-10/slides/technical_talks/ozen.pdf (5.3 MB)
- Key points: Python DSL for writing high-performance GPU kernels using LLVM/CUTLASS infrastructure. Compile-time specialization per NVIDIA SM architecture.
- Combo A relevance: CUTLASS Python generates kernels specialized per GPU generation (SM_80, SM_89, SM_90a). These are prime candidates for Combo A MTB bundles — bundle the SM_80, SM_89, SM_90a variants together and let libkdl select at runtime based on detected CUDA capability. **This is a concrete example to include in the poster's "use case" section.**

**Mojo GPU Compilation** (Lightning Talk)
- Speakers: Weiwei Chen, Abdul Dakkak (Modular)
- Video: https://youtu.be/aY6I76b8rLY
- Slides: https://llvm.org/devmtg/2025-10/slides/lightning_talks/chen_dakkak.pdf (9.8 MB)
- Key points: Mojo as a heterogeneous CPU+GPU programming language; compilation through MLIR pipeline; LLVM JIT for GPU kernels.
- Combo A relevance: Mojo kernels compiled through MLIR produce GPU binary artifacts. These could feed into MTB bundles. Demonstrates that MLIR-to-GPU compilation pipeline is mature.

**Triton-San: Toward Precise Debugging of Triton Kernels via LLVM Sanitizers**
- Slides: https://llvm.org/devmtg/2025-10/slides/technical_talks/lu.pdf
- Key points: Sanitizer integration for Triton kernels at the LLVM IR level.
- Combo A relevance (low): Indirect. Shows Triton's LLVM-based compilation pipeline is mature enough to support sanitizer tooling.

### Quick Talks (2025)

**Extending ThinLTO Support for AMDGPU** (Shilei Tian)
- Slides: https://llvm.org/devmtg/2025-10/slides/quick_talks/tian.pdf
- Key points: AMDGPU ThinLTO implementation; parallel compilation for AMD GPU targets.
- Combo A relevance: Better ThinLTO outputs → smaller, more optimized per-arch AMD GPU binaries → smaller MTB bundles.

**Optimizing IREE to Match llama.cpp** (Uiseop Eom)
- Key points: MLIR optimizations for LLM inference on IREE. IREE's HAL as competing runtime dispatch layer.
- Combo A relevance: IREE's HAL dispatch is the closest competing runtime. The talk is about making IREE match llama.cpp performance — Combo A should benchmark against both.

**Accelerating ML on Hexagon: Qualcomm's MLIR-Based Compiler** (Muthu Baskaran, Franck Slama)
- Video: https://youtu.be/ozpJD2u_1ng
- Slides: https://llvm.org/devmtg/2025-10/slides/quick_talks/baskaran_slama.pdf
- Key points: MLIR-based ML compiler for Hexagon DSP. Heterogeneous dispatch beyond GPU.
- Combo A relevance: Hexagon as a potential libkdl target beyond NVIDIA/AMD/CPU. The MTB format's target field is extensible.

### Posters (2025)

**XeGPU: A High-Performance MLIR Dialect for Intel GPU Programming** (Chao Chen, Jianhui Li)
- Key points: Layout-guided tile-based GPU kernel development for Intel GPUs via MLIR.
- Combo A relevance: XeGPU-compiled kernels → Level Zero plugin path → candidate MTB bundle entries for Intel GPU targets.

---

## 7. EuroLLVM 2025 — GPU/Heterogeneous Talks

**Event:** April 15–16, 2025, Berlin (workshops April 14)
**Program:** https://llvm.org/devmtg/2025-04/

### Technical Talks

**Bringing NVIDIA Blackwell Support to LLVM and MLIR**
- Speakers: Guray Ozen, Durgadoss Ramanathan, Pradeep Kumar (NVIDIA)
- Video: https://youtu.be/qavOwgaieIo
- Slides: https://llvm.org/devmtg/2025-04/slides/technical_talk/ozen_blackwell.pdf
- Key points: Integrating NVIDIA Blackwell (SM_100a) architecture into LLVM and MLIR — new intrinsics, APFloat additions, NVGPU/NVVM dialect extensions for tensor compute.
- Combo A relevance: Each new NVIDIA architecture generation (Blackwell = SM_100a) requires a new MTB bundle entry. Huber's multi-lib infrastructure cannot handle this dynamically; Combo A's runtime selection can add SM_100a support without recompilation of the dispatch layer. **This is a concrete "future-proofing" argument for Combo A's runtime dispatch model.**

**Bridging LLVM and SPIR-V for Heterogeneous Computing**
- Speakers: Vyacheslav Levytskyy, Michal Paszkowski (Intel)
- Video: https://youtu.be/WYPqSVT8QBw
- Slides: https://llvm.org/devmtg/2025-04/slides/technical_talk/levytskyy_bridging.pdf
- Key points: SPIR-V backend enhancements for vendor-agnostic GPU programming; DPC++ + OpenAI Triton integration for Intel GPUs; neural network workloads via SPIR-V.
- Combo A relevance: Intel's SPIR-V pipeline produces SPIR-V from Triton and DPC++ — these SPIR-V binaries can occupy the SPIR-V slot in Combo A's MTB format. **This talk confirms that SPIR-V from LLVM is increasingly production-quality for Intel GPUs, validating Combo A's inclusion of SPIR-V as a dispatch target.**

**Optimizing FDTD Solvers Using MLIR Across Multiple Hardware Platforms** (Student)
- Speaker: Yifei He
- Video: https://youtu.be/Htr7n5eYQ-E
- Key points: MLIR optimizations for FDTD simulations across Intel, AMD, ARM CPUs, and GPUs.
- Combo A relevance: Demonstrates MLIR-based cross-hardware targeting is being applied in domain-specific compute contexts — exactly the use case for MTB bundles in scientific computing.

---

## 8. EuroLLVM 2026 (Dublin) — GPU/MLIR Talks

**Event:** April 14–15, 2026, Clayton Hotel Burlington Road, Dublin, Ireland
**MLIR Workshop:** April 13, 2026 (7th edition)
**This is the submission venue for the Combo A poster.**
**Program:** https://llvm.swoogo.com/2026eurollvm/agenda

### MLIR Workshop (April 13) — GPU Compilation Track

**CUDA Tile IR** (9:05–9:30 AM)
- Speakers: Matthias Springer (NVIDIA), Lorenzo Chelini
- Key points: MLIR-based intermediate representation for CUDA kernel tile-based optimization. Targets NVIDIA tensor cores. Vendor-specific (NVIDIA only). Produces CUDA cubins (AOT or JIT). GitHub: https://github.com/NVIDIA/cuda-tile
- Combo A relevance: CUDA Tile IR produces NVIDIA-specific cubins. These are exactly the NVIDIA-native binary entries in an MTB bundle. CUDA Tile IR → cubin → MTB slot. **The talk directly supports Combo A's claim that the community is generating increasingly specialized per-architecture GPU binaries that need a runtime dispatch layer.**

**ASTER: MLIR-Based Assembly Tooling** (9:30–10:00 AM)
- Speakers: Nicolas Vasilache, **Fabian Mora Corder**, Kunwar Grover
- Key points: MLIR-based assembly tooling framework (details TBD at talk time — slides not yet public).
- Combo A relevance (HIGH): **Fabian Mora is presenting at the same event as the Combo A poster.** He has a continuous research arc: XBLang (LLVM Workshop 2023) → xbc extensible compiler (LLVM Workshop 2024) → ASTER assembly tooling (EuroLLVM 2026). Combo A should be positioned as complementary to this work — Mora's toolchain produces multi-target binaries; Combo A dispatches them at runtime.

**Auto-tuning MLIR Schedules for Intel GPUs** (10:00–10:30 AM)
- Speakers: Tuomas Karna, Rolf Morel
- Key points: Auto-tuning framework for MLIR compilation schedules targeting Intel GPUs.
- Combo A relevance: Auto-tuned kernels for specific Intel GPU generations are candidates for MTB bundle entries.

**From Graphs to Warps: Semantic Interoperability** (1:00–1:30 PM)
- Key points: Interoperability across GPU programming models at the semantic level.
- Combo A relevance: Semantic interoperability across programming models is the higher-level problem that Combo A's binary-level dispatch addresses at the runtime level.

**MLIR-RAJA: Bridging AI Models and HPC** (3:00–3:30 PM)
- Key points: RAJA performance portability library integration with MLIR for AI+HPC workloads.
- Combo A relevance: RAJA is a performance portability layer for HPC — Combo A provides runtime dispatch for the same goal. Both are addressing portable execution; Combo A operates at the binary dispatch layer rather than the source abstraction layer.

### Main Conference (April 14–15)

**rocMLIR: High-Performance ML Compilation for AMD GPUs with MLIR**
- Speaker: Pablo Martinez
- Time: 1:45–2:15 PM, April 14
- Key points: MLIR-based kernel generation for AMD GPU GEMM, convolution, attention. Compiler infrastructure for ROCm.
- Combo A relevance: rocMLIR produces AMD GPU (HSACO/GCN) binary artifacts — these are AMD-native MTB bundle entries. The talk is at the same venue as the Combo A poster, confirming that AMD GPU kernel compilation is an active topic for this community.

**Writing a Formal Execution and Memory Model for Synchronization on AMD GPUs**
- Speaker: Pierre van Houtryve
- Time: 5:15–5:45 PM, April 14
- Key points: Formal model for AMD GPU memory synchronization primitives.
- Combo A relevance: Indirect — establishes AMD GPU runtime semantics, relevant to correctness of multi-version dispatch.

**HIVM: MLIR Dialect Stack for Ascend NPU Compilation**
- Time: 4:45–5:45 PM, April 15
- Key points: MLIR-based compilation for Huawei Ascend NPU.
- Combo A relevance: Confirms that heterogeneous dispatch increasingly includes non-GPU accelerators (NPU, DSP, Hexagon). MTB format extensibility to non-NVIDIA/AMD targets is relevant.

---

## 9. LLVM Performance Workshop at CGO 2025

**Event:** March 1, 2025, Las Vegas, NV
**Program:** https://llvm.org/devmtg/2025-03/

**The Proton Dialect: An MLIR Dialect for AI Compiler GPU Kernel Profiling**
- Speaker: Keren Zhou (Meta)
- Slides: https://llvm.org/devmtg/2025-03/slides/the_proton_dialect.pdf
- Key points: Multi-level instrumentation-based profiling for Triton kernels. MLIR dialect for expressing profiling operations at multiple IR levels (Triton, TritonGPU, PTX). Enables performance attribution across compilation layers.
- Combo A relevance: The Proton dialect instruments GPU kernels at the MLIR level. Combo A dispatches those kernels at runtime. A Proton-instrumented kernel bundle dispatched by libkdl would produce performance-attributed execution data — useful for benchmark-informed dispatch policy (the "cost model" direction that Combo A identifies as future work).

---

## 10. Jakub Kuderski / IREE — Vulkanised 2025

**Event:** Vulkanised 2025, February 2025
**Title:** "The Long Tail of AI: SPIR-V in IREE and MLIR"
- Speaker: Jakub Kuderski (AMD, formerly Google)
- Slides: https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf
- Key points: AI programming with Vulkan/SPIR-V in IREE+MLIR. IREE's GPU codegen from high-level fusions to target-specific optimizations. AMD's generic MLIR-to-SPIR-V target potentially enabling execution on any Vulkan/SPIR-V driver. Profile-guided optimization for "one-click" whole-model tuning. Ongoing work toward unified binaries for GPU and NPU kernels.
- Combo A relevance: IREE's HAL (Hardware Abstraction Layer) is the closest competing runtime dispatch system to Combo A's libkdl. Key distinction: IREE's HAL is compiler-coupled (IREE compiles the model) whereas Combo A is compiler-agnostic (accepts pre-compiled binaries from any toolchain). **The poster should acknowledge IREE's HAL as the state of the art and distinguish Combo A's contribution as compiler-agnostic user-space dispatch.**

---

## 11. Key RFC Threads (Not Talks, But Directly Cited Community Discourse)

These are not YouTube talks but are Discourse threads directly referenced in LLVM community GPU discourse and should be cited alongside the talks.

**[RFC] Introducing llvm-project/offload** (October 2023, Johannes Doerfert)
- URL: https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- "Right now each vendor is basically creating their own LLVM offloading run-time among a lot of other duplicated — and often downstream only — code."
- **This is the origin of the mechanism layer Combo A builds on.**

**[RFC] SPIR-V IR as a Vendor-Agnostic GPU Representation** (March 2025)
- URL: https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- Proposes SPIR-V as single portable GPU IR — "not yet achieved" for all GPU differences.
- Combo A relevance: Confirms multi-native-binary approach remains pragmatic in near-term.

**[RFC] Cleaning the GPU Dialect — MLIR** (September 2025)
- URL: https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- Proposes cleaning `gpu` dialect — separating `gpu.binary` (kernel container) from dispatch policy.
- Combo A relevance: This RFC creates the clean interface boundary at which Combo A operates: `gpu.binary` holds the compiled artifact; Combo A's dispatch policy selects among multiple `gpu.binary` instances at runtime.

**[RFC] An MLIR Dialect for Distributed Heterogeneous Computing** (June 2025, Robert K Samuel, IIT Madras)
- URL: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- Presented at PLDI 2025 Student Research Competition.
- Proposes `schedule(task @kernel target("gpu"))` compile-time dispatch in MLIR.
- Combo A distinction: This RFC handles compile-time dispatch (target known statically); Combo A handles runtime dispatch (target discovered at process start). **Complementary, not competing.**

---

## 12. Trend Analysis: What Has and Has Not Been Covered

### Covered Exhaustively (2023–2026)
- liboffload mechanism API (`ol*` API) — Huber 2023, 2024, 2025
- SPIR-V backend stability — Paszkowski/Levytskyy 2024, 2025
- Per-language runtime integration (SYCL, HIP, CUDA, OpenMP, Rust, Flang)
- GPU libc/libm/libcxx porting
- GPU sanitizers (ASAN, TSAN)
- MLIR-based extensible compilers for heterogeneous targets (xbc, ASTER, rocMLIR, XeGPU)
- GPU kernel static analysis (KernelInfo pass, ORNL)
- CUTLASS/CUDA Tile IR kernel expression
- Dispatch setup overhead at compiler level (accfg)

### Discussed But Unresolved (Active Community Questions)
- SPIR-V as single portable GPU IR (RFC March 2025, not implemented)
- MLIR runtime lowerings for offloading (Huber's stated 2025 workshop topic, no concrete design)
- SYCL/Unified Runtime vs. liboffload unification (coexisting standards, no merger plan)
- API stability of `ol*` API (still evolving)
- Whether `offload/` becomes universal GPU runtime home for all languages (governance RFC 2025)

### Conspicuously Absent (Confirmed Gap — Where Combo A Contributes)
- **Runtime multi-version kernel selection**: No talk proposes a policy layer for selecting among pre-compiled kernel variants at runtime based on dynamically discovered device capability.
- **Dynamic kernel loading** (dlopen-style): GitHub Issue #75356 "Name-Based Kernel Loading" open since November 2023, still unresolved as of April 2026.
- **Kernel bundle packaging format** for multi-target runtime dispatch: The `.llvm.offloading` format handles single-target fat objects; no format exists for multi-target bundles with selection metadata.
- **User-space dispatch policy libraries**: The 2025 workshop "Not-Compiler Runtime Library GPUs" acknowledges this use case exists, but no prototype has been presented.
- **Benchmark-informed dispatch** (cost model + measured performance → selection): Not addressed in any talk.
- **Fallback dispatch chains** (vendor-native → SPIR-V → CPU): Not proposed in any talk.
- **Cross-framework kernel portability**: Same kernel usable in PyTorch context vs. standalone runtime context — not addressed.

---

## 13. Combo A Positioning Summary

### The Narrative Arc (2023–2026)

1. **2023 Workshop question**: "Should we provide a generic LLVM offloading API?" → Led to liboffload RFC.
2. **2024 Workshop**: liboffload mechanism established; link-time multi-lib selection added; KernelInfo extracts static metadata; SYCL per-kernel splitting confirms fine-grained binary management.
3. **2025 Workshop**: "Where are we going?" — Huber identifies SPIR-V, SYCL, MLIR runtime lowerings as open; "Not-Compiler Runtime Library GPUs" talk acknowledges user-space dispatch gap.
4. **2026 Dublin (Combo A poster)**: Concrete prototype answering the 2023 question for the runtime case: libkdl is a policy layer above liboffload providing runtime multi-version selection, implemented as a C library (~5100 LOC), benchmarked on GTX 1650 + CPU.

### Direct Citations for the Poster

| Citation | Talk/RFC | What It Supports |
|----------|----------|-----------------|
| Mechanism layer | Huber DevMtg 2025 "The LLVM Offloading Infrastructure" | Combo A builds on liboffload |
| Link-time analog | Huber Workshop 2024 "Building GPU Runtimes with Multi-Lib" | Runtime gap Combo A fills |
| Static metadata | Denny Workshop 2024 "Kernel-Info LLVM IR Pass" | Pipeline: Kernel-Info → MTB metadata |
| Per-kernel granularity | Narayanaswamy DevMtg 2024 "Enhance SYCL Offloading" | Validates MTB kernel registry design |
| User-space gap | "Not-Compiler Runtime Library GPUs" Workshop 2025 | Direct community acknowledgment |
| SPIR-V not yet complete | RFC March 2025 + Paszkowski DevMtg 2024 | Multi-native-binary approach justified |
| Compile-time analog | Mora/xbc Workshop 2024 + ASTER Workshop 2026 | Complementary compiler side |
| Competing runtime | Kuderski/IREE Vulkanised 2025 | IREE HAL comparison |
| Community absence | No runtime dispatch policy in any talk | Combo A novelty confirmed |

---

## Sources

- https://llvm.org/devmtg/2023-10/
- https://llvm.org/devmtg/2024-10/
- https://llvm.org/devmtg/2025-10/
- https://llvm.org/devmtg/2025-04/
- https://llvm.org/devmtg/2025-03/
- https://llvm.org/devmtg/2026-04/
- https://llvm.swoogo.com/2024devmtg/agenda
- https://llvm.swoogo.com/2025devmtg/agenda
- https://llvm.swoogo.com/2026eurollvm/agenda
- https://discourse.llvm.org/t/pre-llvm-dev-23-gpu-offloading-pre-workshop-agenda/73775
- https://discourse.llvm.org/t/announcing-the-preliminary-program-agenda-of-llvm-offload-workshop-llvm-developers-meeting-2024/82535
- https://discourse.llvm.org/t/cfp-llvm-dev25-llvm-offload-workshop/88352
- https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832
- https://discourse.llvm.org/t/announcing-the-7th-mlir-workshop-eurollvm-2026-program/90119
- https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- https://llvm.org/docs/KernelInfo.html
- https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf
- https://www.phoronix.com/news/AMD-Vulkan-SPIR-V-Wide-AI
- https://github.com/NVIDIA/cuda-tile
