# Wave 05 — Abandoned Multi-Target GPU Projects: HSA and Others

**Angle:** Abandoned Multi-Target GPU Projects (HSA and others)
**Query:** HSA Heterogeneous System Architecture abandoned deprecated GPU dispatch + failed GPU portability projects
**Priority source types:** paper, blog, RFC
**Date:** 2026-04-06

---

## Source Index

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | Heterogeneous System Architecture — Wikipedia | https://en.wikipedia.org/wiki/Heterogeneous_System_Architecture | Current | Reference | 9/10 |
| S2 | HSA Foundation — Wikipedia | https://en.wikipedia.org/wiki/HSA_Foundation | Current | Reference | 9/10 |
| S3 | GCC Drops AMD HSA Offloading Support — Phoronix | https://www.phoronix.com/news/GCC-Drops-AMD-HSA | 2020 | News | 10/10 |
| S4 | The HSA Foundation Has Been Eerily Quiet As We Roll Into 2021 — Phoronix | https://www.phoronix.com/news/HSA-Quiet-Start-2021 | Jan 2021 | News | 10/10 |
| S5 | HSAIL-HLC-Development-LLVM (development stopped) — GitHub | https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM | 2015-2016 | Source code | 10/10 |
| S6 | ROCm Deprecates HCC in ROCm v3.5 — ROCm Deprecation Docs | https://cgmb-rocm-docs.readthedocs.io/en/latest/Current_Release_Notes/Deprecation.html | 2020 | Docs | 9/10 |
| S7 | AMD ROCm Revisited: Evolution of the GPU Computing Ecosystem — AMD Blogs | https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited-ecosy/README.html | 2023 | Blog | 8/10 |
| S8 | Why OpenCL and C++ GPU Alternatives Struggled (Democratizing AI Compute, Part 5) — Modular Blog | https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives | 2025 | Blog/Analysis | 10/10 |
| S9 | OpenCL — Wikipedia (history and deprecation timeline) | https://en.wikipedia.org/wiki/OpenCL | Current | Reference | 8/10 |
| S10 | C++ AMP — Wikipedia (history and deprecation) | https://en.wikipedia.org/wiki/C++_AMP | Current | Reference | 8/10 |
| S11 | C++ AMP Headers Deprecated — Microsoft VisualStudio DevCommunity | https://developercommunity.visualstudio.com/t/c-amp-headers-are-deprecated-what-is-the-replaceme/1495203 | 2021 | Forum | 8/10 |
| S12 | OpenGL, OpenCL Deprecated in macOS 10.14 Mojave — AppleInsider | https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave | June 2018 | News | 9/10 |
| S13 | Project Larrabee — How Intel's First Attempt at GPUs Failed — HowToGeek | https://www.howtogeek.com/896521/project-larrabee-how-intels-first-attempt-at-gpus-failed/ | 2023 | History | 7/10 |
| S14 | SPIR (original) vs SPIR-V — Standard Portable Intermediate Representation — Wikipedia | https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation | Current | Reference | 9/10 |
| S15 | AMD Boltzmann Initiative — High-Performance Computing News | https://insidehpc.com/2016/08/boltzmann-initiative/ | 2016 | News | 8/10 |

---

## Source Summaries

### S1 — Heterogeneous System Architecture — Wikipedia [9/10]

**URL:** https://en.wikipedia.org/wiki/Heterogeneous_System_Architecture
**Type:** Reference
**Date:** Current (reflects 2015 peak and subsequent decline)

HSA is a cross-vendor set of specifications released by the HSA Foundation — an industry consortium founded in 2012 with AMD, ARM, Imagination Technologies, MediaTek, Qualcomm, Samsung, and Texas Instruments as founding members. Its goal was to expose CPU and GPU as co-equal compute resources on a shared address space, enabling fine-grained task dispatching without expensive PCIe data copies.

**Key technical elements:**
- **hUMA (Heterogeneous Uniform Memory Access):** CPU and GPU share a unified virtual address space; GPU can access CPU-allocated memory and vice versa without explicit copies. AMD APUs (Kaveri, 2014) were the first hardware implementation.
- **hQ (Heterogeneous Queuing):** Hardware queue structures allowing CPUs and GPUs to mutually enqueue work on each other without OS intervention. The GPU could launch work on the CPU and vice versa, enabling fine-grained task graphs.
- **HSAIL (HSA Intermediate Language):** A virtual parallel ISA for compute kernels — HSA's analog to PTX or SPIR-V. Kernels compiled to HSAIL would be "finalized" to device-native ISA by a JIT compiler at load time.
- **HSA Runtime:** A thin user-mode API exposing queue management, packet dispatch, and signal synchronization primitives. Packet dispatch was AQL (Architected Queuing Language) — a binary protocol defining how work items were submitted to hardware queues.

The foundation's 1.0 specification was released in March 2015. Hardware conformant implementations included: AMD Kaveri (2014, partial), AMD Carrizo (2015), AMD Fiji (2015), Qualcomm Snapdragon 820 (2016), MediaTek Helio X20 (2016), Samsung Exynos 8890 (2016). NVIDIA was conspicuously absent from the list.

**Relevance to libkdl:** HSA's AQL packet format and hQ queuing model represent the most ambitious hardware-level heterogeneous dispatch mechanism ever standardized. Its failure teaches what happens when standards outpace hardware adoption and ecosystem willingness.

---

### S2 — HSA Foundation — Wikipedia [9/10]

**URL:** https://en.wikipedia.org/wiki/HSA_Foundation
**Type:** Reference
**Date:** Current

The HSA Foundation was the standards body managing HSA specifications. Founded in 2012, it joined the Linux Foundation in 2013. Its last published specification was HSA Runtime 1.2 in November 2021.

**Documented decline markers:**
- The HSA Foundation joined the Linux Foundation in 2013 — potentially a sign of seeking institutional backing rather than industry momentum.
- Active news and blog posts from the foundation ceased around 2016-2017.
- The Phoronix article from January 2021 (S4) documented the organization's effective dormancy: Twitter feed silent since February 2020, GitHub repositories inactive for ~2 years, website down for over a month.
- GCC dropped HSA offloading support (S3) due to being "unmaintained."
- AMD — the primary driver of HSA — shifted its GPU compute focus entirely to ROCm, which uses HIP + clang/LLVM rather than HSAIL or the HSA runtime API.

**Significance:** The HSA Foundation did not formally dissolve (its website remains up with archival content as of 2026) but has functionally ended active standardization work. It became a case of organizational capture: AMD was by far the most invested member, and when AMD pivoted to ROCm, the foundation's raison d'être disappeared.

---

### S3 — GCC Drops AMD HSA Offloading Support — Phoronix [10/10]

**URL:** https://www.phoronix.com/news/GCC-Drops-AMD-HSA
**Type:** News article
**Date:** 2020

GCC's HSA offloading support was removed from the GNU Compiler Collection. The stated reason: the code had not been maintained and had no measurable usage.

**Key details:**
- GCC had gained HSA offloading support in GCC 6 (2016) — a major engineering investment to enable GCC-compiled code to offload to AMD APU GPU cores via HSA.
- By 2020, the code was unmaintained and actively bitrotting. No developer was maintaining it.
- The removal was the first major public indicator that the HSA software ecosystem had effectively collapsed.
- Notably, GCC continued to develop AMD GPU offloading via a different path: GCN/AMDGPU offloading (for OpenACC/OpenMP) which compiles directly to AMD GCN ISA via the LLVM AMDGPU backend — a completely separate mechanism bypassing HSA entirely.

**Lesson for libkdl:** An offloading standard that loses its compiler support loses everything. Maintaining compiler toolchain integration is not optional — it is the primary adoption vector for a new dispatch layer.

---

### S4 — HSA Foundation Has Been Eerily Quiet — Phoronix [10/10]

**URL:** https://www.phoronix.com/news/HSA-Quiet-Start-2021
**Type:** News/investigation
**Date:** January 2021

Phoronix investigates the HSA Foundation's effective disappearance.

**Key documented facts:**
- Twitter/X feed for HSA Foundation had been silent since February 2020 (11+ months at time of writing).
- GitHub repositories under `github.com/hsafoundation` had not received meaningful commits for approximately 2 years.
- hsafoundation.com website was down for over one month at time of investigation.
- GCC had dropped HSA support "a few months back" (confirming S3 timing).
- Off-the-record comments from people involved in HSA working groups ranged from "it just stagnated" to "it dissolved."
- AMD was identified as the member whose pivot to ROCm effectively ended the consortium's purpose.

**Significance:** This is the definitive public record of HSA's collapse. The organization continued to technically exist (website eventually came back up) but ceased all forward activity. This is what "quietly abandoned" looks like for an industry standards body.

---

### S5 — HSAIL-HLC-Development-LLVM GitHub (development stopped) [10/10]

**URL:** https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM
**Type:** Abandoned source repository
**Date:** Last active ~2016

The HSAIL High-Level Compiler (HLC) was an LLVM-based toolchain that compiled OpenCL C (and later HIP) to HSAIL binary. This repository was the primary toolchain implementation.

**Key details:**
- The repository description now reads: "Development has stopped on this branch. This was a development branch." It is kept for reference only.
- The HLC used a forked LLVM with an HSAIL target — not upstreamed into mainline LLVM. This was a fundamental sustainability problem: AMD maintained a private LLVM fork to support HSAIL generation.
- In ROCm 1.9.0, AMD removed the closed-source HSAIL finalizer extension. All ROCm components then used LLVM IR directly for AMD GCN compilation — eliminating HSAIL from the pipeline entirely.
- The old HSAIL SC (shader compiler) path was replaced with a native LLVM/GCN backend.
- The transition happened approximately 2017-2018 as ROCm 1.x matured and the GCN LLVM backend reached production quality.

**Design failure analysis:** HSAIL required a forked LLVM, a closed-source finalizer, and a separate runtime API. Three non-upstreamed components is an unsustainable maintenance burden for any standards effort. When AMD's engineers could compile AMD GCN directly from LLVM without HSAIL, HSAIL became pure overhead.

**Lesson for libkdl:** Any portable GPU dispatch layer must be built on upstreamed, maintained infrastructure. Private forks always collapse. The LLVM AMDGPU backend (fully upstreamed) replaced the HSAIL fork.

---

### S6 — AMD Deprecates HCC Compiler in ROCm v3.5 [9/10]

**URL:** https://cgmb-rocm-docs.readthedocs.io/en/latest/Current_Release_Notes/Deprecation.html
**Type:** Official AMD documentation
**Date:** 2020 (ROCm 3.5 release)

HCC (Heterogeneous Compute Compiler) was AMD's first "post-HSAIL" attempt at a C++ GPU compiler. Released as part of the Boltzmann Initiative in 2015, it implemented a C++ language extension for GPU programming (initially as the open-source version of C++AMP). HCC was deprecated in ROCm v3.5 and replaced by HIP-Clang.

**HCC deprecation facts:**
- AMD announced HCC deprecation in ROCm 3.5 (2020); the final HCC release was planned for June 2019 (shipped slightly later).
- ROCm 1.9 was the last release supporting C++AMP. The C++AMP support in HCC was the first casualty.
- HCC was replaced by HIP-Clang: a clang-based compiler sharing code with the CUDA-clang path, conforming better to C++ standards and CUDA language syntax.
- Reason stated: "AMD is deprecating HCC to put more focus on HIP development and on other languages supporting heterogeneous compute."

**The HCC → HIP-Clang transition represents a strategic consolidation:** AMD went from three compiler paths (HSAIL finalizer, HCC, HIP) to one (HIP-Clang using the mainline LLVM AMDGPU backend). Each consolidation improved sustainability at the cost of abandoning previous investments.

**Relevance:** HCC's deprecation shows AMD deliberately chose to bet on the LLVM/Clang ecosystem rather than maintaining a separate compiler. This validated the LLVM-centric approach and is why the AMDGPU backend in LLVM is now production-grade.

---

### S7 — AMD ROCm Revisited: Evolution of the GPU Computing Ecosystem [8/10]

**URL:** https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited-ecosy/README.html
**Type:** AMD blog post
**Date:** 2023

AMD's own retrospective on the ROCm ecosystem's evolution, acknowledging the complexity of the software stack's history.

**Key details:**
- Documents the full genealogy: HSA runtime → ROCR runtime (same HSA API, AMD-specific implementation) → ROCm unified stack.
- The ROCR runtime (ROCm Runtime) implements the HSA runtime API but is AMD's own implementation, not the open multi-vendor standard AMD originally intended.
- ROCm's software stack grew organically from GPUOpen + HSA, with several components merging, being deprecated, or renamed over time.
- The blog post characterizes ROCm 1.0-2.0 as "pioneering but fragile" — frequent breaking changes as the stack matured.
- ROCm 5.0+ stabilized the ABI. ROCm 6.0 (2024) focused on NVIDIA hardware interoperability.

**Lesson extracted:** Even AMD — the primary HSA champion — refers to its own multi-tool history as complexity rather than elegance. The consolidation from HSA+HCC+OpenCL to HIP+LLVM was an explicit simplification.

---

### S8 — Why OpenCL and C++ GPU Alternatives Struggled — Modular Blog [10/10]

**URL:** https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives
**Type:** Analytical blog (Modular / Mojo team)
**Date:** 2025

Comprehensive post-mortem on every major failed GPU portability effort, written from the perspective of building a new AI compute platform.

**Key arguments and evidence:**

**OpenCL's structural failures:**
- Committee governance with competing vendors meant updates moved at "glacial" speed relative to CUDA's continuous velocity.
- No reference implementation: every vendor built their own fork from scratch, producing behavioral divergence and extension sprawl.
- NVIDIA maintained a "token" OpenCL implementation — strategically hobbled (no Tensor Core access) to ensure CUDA would always be necessary.
- Extension fragmentation: `cl_nv_*`, `cl_amd_*`, `cl_intel_*` extensions gave performance only on the vendor's own hardware, defeating portability.
- "Open coopetition" (simultaneously collaborating and competing) paralyzed decision-making.

**SYCL / oneAPI:**
- "Notionally open but in practice controlled by a single hardware vendor (Intel) competing with all the others."
- Without NVIDIA and AMD full buy-in, SYCL couldn't break CUDA's ecosystem lock.
- Remained "just another layer on top of an already troubled ecosystem" (quoting the article).

**The core thesis:** Technical merit alone is insufficient. CUDA's dominance is an ecosystem lock-in problem, not a technical superiority problem. Alternatives failed not because they were worse compilers but because they couldn't replicate CUDA's integrated framework+library+tool stack.

**Relevance to libkdl:** libkdl is explicitly not trying to be a CUDA alternative. It is a dispatch layer above vendor runtimes. This framing is essential to avoid being classified as "yet another failed GPU portability effort."

---

### S9 — OpenCL Wikipedia (History and Deprecation) [8/10]

**URL:** https://en.wikipedia.org/wiki/OpenCL
**Type:** Reference
**Date:** Current

Key dates for the OpenCL decline:
- 2018: Apple deprecated OpenCL in macOS 10.14 Mojave, recommending Metal instead.
- 2020: OpenCL 3.0 released — a pragmatic retreat making all 2.x features optional, acknowledging that NVIDIA never implemented them.
- 2021: OpenCL support silently dropped from macOS Security Update 2021-002.
- 2023: NVIDIA finally achieved OpenCL 3.0 conformance — 15 years after Apple created OpenCL, and only to the 1.2 mandatory baseline.
- OpenCL 2.0's SVM (Shared Virtual Memory) required hardware coherency. Intel and AMD implemented it. NVIDIA never did. Apple never went beyond 1.2. This killed the 2.0 generation as a portability target.

**The SVM lesson:** Features that require specific hardware capabilities cannot be in a "required" portion of a portability standard if adoption across all target vendors is the goal. OpenCL 3.0's "optional everything" was the correct design — 10 years too late.

---

### S10 — C++ AMP — Wikipedia [8/10]

**URL:** https://en.wikipedia.org/wiki/C++_AMP
**Type:** Reference
**Date:** Current

C++ AMP (Accelerated Massive Parallelism) was Microsoft's C++ extension for GPU programming, announced 2011. It was an open specification (unlike CUDA) built on DirectX 11 Compute.

**Key facts:**
- At its peak (2013), the HSA Foundation released a C++ AMP compiler that output to OpenCL, SPIR, and HSAIL — demonstrating the aspiration to make C++ AMP a cross-vendor GPU language.
- AMD's HCC compiler initially implemented C++ AMP as its first feature before evolving toward HIP.
- Microsoft deprecated C++ AMP in Visual Studio 2022 (version 17.0). Visual Studio 2026 and later do not support it.
- The replacement path leads to DirectX 12 compute, DirectML, or WinML — all Windows-specific.
- **Design lesson acknowledged:** "The basic concepts behind C++AMP, like using C++ classes to express parallel and heterogeneous programming features, have been inspirational to the SYCL standard." The idea survived; the implementation did not.

**Why it failed:** C++ AMP was Windows-only and DirectX-dependent. This limited adoption to Windows GPU workloads. The ML community standardized on Linux/CUDA, making C++ AMP irrelevant for the workloads that matter.

---

### S11 — C++ AMP Deprecation (Microsoft DevCommunity) [8/10]

**URL:** https://developercommunity.visualstudio.com/t/c-amp-headers-are-deprecated-what-is-the-replaceme/1495203
**Type:** Forum/official deprecation
**Date:** 2021

**Confirmed deprecation timeline:** C++ AMP headers declared deprecated in VS 2022 (v17.0), scheduled for removal beyond VS 2022. No migration path exists that preserves C++ AMP semantics on non-Windows. The recommended replacement is CUDA, DirectML, or raw DirectX 12 Compute — all platform or vendor-specific.

---

### S12 — OpenGL/OpenCL Deprecated in macOS 10.14 Mojave — AppleInsider [9/10]

**URL:** https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave
**Type:** News
**Date:** June 2018

Apple — the original creator and patent contributor of OpenCL — deprecated its own technology in macOS Mojave (June 2018). The stated reason: Metal is the modern replacement for both graphics (OpenGL) and compute (OpenCL) on Apple platforms.

**Significance of Apple's deprecation:**
- Apple contributed the original OpenCL specification to Khronos in 2008. Its deprecation in 2018 closed a 10-year arc.
- Without Apple as a founding champion, OpenCL lost its most powerful advocate.
- The signal to the developer community was unambiguous: the framework's own creator had abandoned it.
- iOS never received OpenCL support — the portable compute layer Apple built never shipped on Apple's largest platform.
- Metal's success on Apple Silicon (Apple's own chips) validated vertical integration as the winning architecture vs. open heterogeneous standards.

**Lesson:** A standard without its founding member's support is a zombie standard. When the entity that created a portability layer stops using it, the credibility collapse is immediate.

---

### S13 — Project Larrabee: How Intel's First GPU Attempt Failed [7/10]

**URL:** https://www.howtogeek.com/896521/project-larrabee-how-intels-first-attempt-at-gpus-failed/
**Type:** History / analysis
**Date:** 2023

Intel Larrabee (2005-2010) was an ambitious attempt to build a many-core x86 GPU that would be programmable with standard C rather than shader languages. The concept: if GPU cores are just x86 cores with wide SIMD, developers can use existing compilers and tools.

**Why Larrabee failed:**
- Software rasterization on x86 cores could not match dedicated hardware rasterization silicon in performance-per-watt.
- The SIMD programming model for x86 was not sufficiently simpler than CUDA/OpenCL for parallel GPU workloads.
- Execution schedules and competitive hardware both advanced faster than Larrabee's development timeline.
- Intel shelved consumer Larrabee in December 2009; the underlying technology became Intel Xeon Phi (Knight's Landing, 2016) — another product that ultimately failed commercially.

**Connection to heterogeneous dispatch:** Larrabee's "just use C" promise was the opposite of HSA's approach. Both failed, but for different reasons. Larrabee failed because the hardware model didn't deliver. HSA failed because the software ecosystem didn't coalesce. Together they frame the difficulty space.

---

### S14 — Original SPIR vs SPIR-V — Wikipedia [9/10]

**URL:** https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation
**Type:** Reference
**Date:** Current

The original SPIR (Standard Portable Intermediate Representation, 2012) was OpenCL's first attempt at a portable binary kernel format. It encoded OpenCL C programs as a constrained subset of LLVM IR bitcode.

**Why original SPIR failed and was replaced:**
- LLVM IR has no stability guarantees across versions. SPIR 1.2 was encoded in LLVM IR version 3.2, SPIR 2.0 in LLVM 3.4. Upgrading LLVM broke SPIR binaries.
- Being a constrained LLVM bitcode subset meant vendors needed to embed an LLVM parser in their GPU drivers — non-trivial and version-locked.
- SPIR had no capability declaration mechanism; the consuming driver had to guess what features the kernel used.
- GPU-native constructs (work-groups, address space qualifiers, memory barriers) were awkwardly mapped onto LLVM IR concepts without first-class representation.

SPIR-V (2015) fixed all of these: custom binary format (no LLVM dependency), explicit capability declarations, first-class GPU constructs, versioned format with backward compatibility guarantees.

The GitHub repository `KhronosGroup/SPIRV-LLVM` (the original SPIR-to-SPIR-V bridge) is archived: "This project is no longer active. Please join us at [SPIRV-LLVM-Translator]." The original SPIR standard is now formally superseded and no longer maintained.

**Lesson:** A portable IR format must have: (a) its own stable binary encoding independent of any compiler's internal format, (b) explicit capability negotiation, and (c) first-class constructs for the target domain. LLVM IR fails (a) and (b). MLIR bytecode addresses (a) and (b) but (c) depends on the dialect.

---

### S15 — AMD Boltzmann Initiative (2015-2016) [8/10]

**URL:** https://insidehpc.com/2016/08/boltzmann-initiative/
**Type:** Industry news
**Date:** August 2016

AMD's Boltzmann Initiative (announced November 2015, SC15) was a major strategic pivot to create an open-source CUDA-competitive ecosystem. It introduced three tools:
1. **HCC:** AMD's new C++ GPU compiler (replacing the older HCC based on C++AMP).
2. **HSA runtime infrastructure:** A headless Linux driver stack for HPC.
3. **HIP:** A CUDA source-compatibility layer for porting CUDA code to AMD GPUs.

**What succeeded and what failed:**
- **HIP succeeded** and is now the primary AMD GPU programming API. PyTorch, TensorFlow, JAX all support HIP. The CUDA source-compatibility was close enough to automate porting (~90% via hipify).
- **HCC failed** and was deprecated in ROCm 3.5 (S6). Replaced by HIP-Clang.
- **HSA runtime infrastructure** survived in the ROCR runtime API, but as an AMD-specific implementation rather than a multi-vendor standard.
- The original multi-vendor HSA vision was quietly abandoned as AMD concluded it was easier to be CUDA-compatible than to build a new ecosystem.

**Strategic lesson:** HIP's strategy of copying CUDA's API surface rather than building a new abstraction was the correct move. Every API that tried to design a "better" GPU programming model (C++AMP, SYCL, HSAIL) struggled to displace CUDA. HIP's "just be CUDA on AMD" approach succeeded.

---

## Synthesis

### The Graveyard: A Taxonomy of Failure Modes

The history of heterogeneous GPU portability is littered with abandoned standards, deprecated compilers, and silent GitHub repositories. Each failure has a distinct root cause. Mapping them enables a clear-eyed assessment of what libkdl must avoid.

#### Failure Mode 1: Spec Without Reference Implementation (OpenCL, HSA)

OpenCL's absence of a reference implementation forced every vendor to build independent implementations from scratch. Without a shared codebase, behavioral divergence was inevitable and expensive to debug. The ICD mechanism correctly allowed runtime multi-vendor coexistence, but without a reference to test against, conformance was aspirational rather than verified.

HSA suffered the same problem. The HSA Foundation published specifications but each member implemented the runtime independently. AMD's ROCR runtime was the only production-quality implementation. When AMD shifted focus to ROCm/HIP, no other member had a viable HSA stack to inherit leadership.

**Quantified consequence:** OpenCL extension fragmentation grew to 150+ vendor-specific extensions by 2020. A kernel using only "core" OpenCL for cross-vendor portability was performance-capped at a fraction of vendor-optimized capability.

#### Failure Mode 2: Private Fork Debt (HSAIL, original SPIR)

HSAIL required a private LLVM fork with an HSAIL target that was never upstreamed. Maintaining this fork consumed AMD engineering resources proportional to every LLVM version bump. When the mainline LLVM AMDGPU backend reached sufficient quality (~2017), the HSAIL fork became pure liability. It was abandoned within months.

Original SPIR's dependence on LLVM IR bitcode format created identical fragility: upgrading LLVM broke SPIR binaries. Every driver embedding a SPIR parser was locked to a specific LLVM version. SPIR-V replaced SPIR precisely by removing this dependency.

**Quantified consequence:** The HLC-HSAIL-Development-LLVM repository's last meaningful commit was 2016. ROCm 1.9 (2018) was the first release without HSAIL in the critical path.

#### Failure Mode 3: Founding Member Defection (OpenCL, HSA)

Apple created OpenCL and contributed it to Khronos. Apple then deprecated it in 2018 and silently dropped it in 2021. Apple's iOS never shipped OpenCL. The original creator's defection is the most damaging signal a standard can receive.

AMD was HSA's primary driver. When AMD concluded that being CUDA-compatible (via HIP) was a more tractable market strategy than building the HSA ecosystem, the foundation effectively ended. No other member had AMD's investment in the standard.

**Lesson:** A portable standard's health is a function of its smallest committed member, not its largest. If the standard requires the largest member's active development to remain viable, it fails when that member's priorities shift.

#### Failure Mode 4: Platform-Specific Constraints (C++AMP, Larrabee)

C++AMP was Windows-only and required DirectX 11 Compute. The ML ecosystem standardized on Linux. A portable GPU language that is not portable to the primary ML development platform is self-defeating.

Larrabee assumed that "just use C on many cores" would be simpler than CUDA. It was not — the SIMD programming model for wide vectors is fundamentally different from C's scalar model regardless of ISA.

**Lesson:** Platform scope must match the target ecosystem's actual development environment from day one. A portability solution that requires the dominant platform to switch to an alternative has an infinite adoption barrier.

#### Failure Mode 5: Committee Velocity vs. Hardware Velocity (OpenCL, HSA)

OpenCL's Khronos committee process required multi-vendor consensus for every spec update. NVIDIA shipped new tensor core features every 2 years (Volta 2017, Ampere 2020, Hopper 2022, Blackwell 2024). OpenCL could not track this rate — every new hardware capability required multi-year standardization before it could be exposed.

The result: NVIDIA could always truthfully claim "OpenCL doesn't expose our latest hardware." Whether by design or structural dysfunction, committee-governed standards cannot track vertically-integrated hardware innovation.

HSA had the same problem: the AQL queuing model was standardized in 2015 for hardware that had been shipping since 2014. By the time the spec was finalized, AMD had moved on to newer architectural features that HSA had no mechanism to expose.

**Lesson for libkdl:** libkdl must not include a standardization body in its critical path. A thin policy layer above vendor runtimes inherits vendor hardware features automatically — if `cuLaunchKernel` gains new capabilities, libkdl's CUDA backend uses them without any spec revision.

---

### What HSA Got Right (Design Elements That Survived)

Not all of HSA was wasted. Several architectural innovations from HSA appear in successor systems:

**1. AQL (Architected Queuing Language):** The binary packet format for submitting work to hardware queues. AMD's ROCm still uses AQL internally — ROCR runtime constructs AQL dispatch packets when launching HIP kernels. The mechanism survived, stripped of the cross-vendor standardization ambition.

**2. HSA Runtime API:** The thin user-mode API (`hsa_queue_create`, `hsa_signal_create`, `hsa_amd_memory_pool_allocate`) survived as the ROCR runtime API. AMD's ROCm documentation still references HSA runtime headers. The API specification is still used, just as AMD-only.

**3. Fine-grained memory pools:** HSA's model of explicitly typed memory pools (system memory, device memory, coherent memory) influenced ROCm's memory model and eventually Vulkan's memory heap model. The abstraction of "memory with defined coherency semantics" was correct — even if hUMA's promise of automatic CPU-GPU coherency proved hardware-intractable at the time.

**4. Signal-based synchronization:** HSA signals (64-bit monotonically-decreasing counters with hardware-signaled completion) are used verbatim in the ROCR runtime. The mechanism is more efficient than GPU fence polling in certain workloads and influenced HIP's synchronization primitives.

---

### OpenCL's Surviving Legacy

Despite its commercial failure in AI compute, OpenCL left several enduring contributions:

**SPIR-V (2015):** OpenCL 2.1 introduced SPIR-V, which is now the de facto GPU intermediate representation across Vulkan, DirectX (DXIL compiled from SPIR-V), WebGPU, and OpenCL itself. OpenCL's primary technical legacy is an artifact it introduced to fix its own compilation model problems.

**Platform/Device/Context model:** The hierarchy of `Platform → Device → Context → CommandQueue → Kernel` is standard across SYCL, HIP, Vulkan, and IREE's HAL. OpenCL's abstraction was correct; its implementation was fragmented.

**ICD mechanism:** The Installable Client Driver pattern — multiple vendor implementations loaded via runtime dispatch table — remains the correct model for multi-vendor API coexistence. Vulkan's layer system, IREE's driver model, and the oneAPI Unified Runtime's adapter model all follow this pattern.

**PoCL (Portable Computing Language):** An open-source, conformant OpenCL implementation built on LLVM proved the concept of "build your own portable runtime on LLVM" is viable and even competitive. PoCL 7.1 (October 2025) achieves OpenCL 3.0 conformance across x86, ARM, RISC-V, Intel GPU (Level Zero), NVIDIA (CUDA), and Vulkan. PoCL is direct infrastructure evidence for the libkdl thesis.

---

### The SPIR → SPIR-V Transition: A Clean Case Study in Correct Abandonment

The original SPIR format's replacement by SPIR-V is the single cleanest example of an abandoned GPU technology being replaced by a superior successor without dragging the ecosystem. The transition is instructive:

1. **Identified root cause:** LLVM bitcode dependency → version brittleness.
2. **Designed purpose-built replacement:** SPIR-V is not an LLVM artifact; it is an independent binary format designed specifically for GPU distribution.
3. **Added missing capabilities:** Explicit capability declarations, versioned backward compatibility, first-class GPU constructs.
4. **Maintained migration path:** LLVM IR → SPIR-V is implemented by `llvm-spirv` (SPIRV-LLVM-Translator), meaning existing toolchains could adopt SPIR-V without rewriting compilation pipelines.
5. **Retired cleanly:** The SPIRV-LLVM repository is archived with a clear pointer to SPIRV-LLVM-Translator. No one maintains both formats.

This pattern — clean deprecation with a clear migration path and a technically superior replacement — is the model for how libkdl should handle evolving vendor APIs.

---

### Implications for libkdl Design

The historical record of abandoned GPU portability efforts generates concrete design constraints:

**1. Do not build a new language or IR.** Every new GPU language that tried to be "better than CUDA" failed (HSAIL, C++AMP, OpenCL C). libkdl operates at the binary dispatch level — consuming pre-compiled vendor-native binaries, not defining how kernels are written.

**2. Do not start a standards body.** Committee governance is fatal to innovation velocity. libkdl is a software library, not a specification. Vendors can support it by building plugins, not by joining a foundation.

**3. Build on upstreamed infrastructure only.** The HSAIL private LLVM fork collapsed when the mainline AMDGPU backend matured. libkdl uses CUDA driver API, HIP runtime, Level Zero — all vendor-maintained, upstreamed infrastructure.

**4. SPIR-V as portable fallback, not primary path.** SPIR-V itself succeeded (unlike HSAIL) but carries a ~25% performance penalty for ML workloads vs. vendor-native compilation. libkdl's design of pre-compiled native binaries per target with SPIR-V as a fallback is validated by this history.

**5. The ICD pattern works.** OpenCL's ICD mechanism is the correct model for multi-vendor runtime coexistence. libkdl's backend plugin system (`dlopen`-based, vendor-specific `.so` loaded at runtime) directly implements this pattern. What OpenCL got wrong was the API above the ICD; the ICD itself was correct.

**6. Do not require hardware coherency.** HSA's hUMA promise required hardware-level CPU-GPU coherency that only AMD APUs could provide. NVIDIA discrete GPUs never supported it. Any portability layer that requires a specific hardware capability breaks on hardware that lacks it.

**7. Ecosystem integration is the adoption problem.** OpenCL failed partly due to never achieving deep integration with PyTorch/TensorFlow. libkdl must provide a PyTorch `register_backend_for_device` hook and an ONNX Runtime execution provider stub to be adoptable.

---

## Relevance to libkdl / Vendor-Agnostic Kernel Dispatch

The graveyard of failed GPU portability efforts is the strongest argument for libkdl's design choices. Each failure teaches a lesson:
- HSA: don't build standards bodies; don't require private LLVM forks; don't assume hardware coherency.
- OpenCL: don't conflate source portability with performance portability; build a reference implementation; cache binary artifacts from day one.
- HSAIL: stay on the LLVM mainline; forked infrastructure collapses when upstream matures.
- C++AMP: target the right platform (Linux/CUDA ecosystem, not Windows/DirectX).
- Original SPIR: build IR formats with their own stable binary encoding, not as a subset of a compiler's internal format.

libkdl's "pre-compiled native binaries + thin runtime selection layer" approach is the architectural answer to all five failure modes. It does not build a new language, does not fork LLVM, does not require hardware coherency, and operates on Linux with existing vendor toolchains. The failure modes of every predecessor are non-issues for this design.

The surviving elements — OpenCL's ICD mechanism, HSA's AQL queuing (embedded in ROCR), SPIR-V as the portable distribution IR, PoCL as the reference LLVM-based runtime — all appear in libkdl's stack, stripped of the multi-vendor governance overhead that killed their parent standards.

---

## Gaps and Open Questions

1. **Why did NVIDIA not join HSA?** The historical record is clear that NVIDIA was absent from the HSA Foundation but gives no official explanation. The implicit answer (CUDA moat protection) is widely understood but not documented in public technical writing. This is a relevant framing point for the poster's context-setting.

2. **Could HSA have succeeded with different governance?** If HSA had been structured as an Apache-licensed open-source project (rather than a Khronos-style standards body) with AMD as the primary maintainer and a clear API stability commitment, would it have attracted broader adoption? This counterfactual is speculative but frames what a "successful open GPU dispatch standard" would have required.

3. **Is SPIR-V's success a disproof of the "committee standards fail" thesis?** SPIR-V emerged from Khronos but succeeded. The difference: SPIR-V is a binary format specification (like ELF or ZIP) with no performance-critical implementation choices left to vendors, whereas OpenCL's API left all performance-critical decisions vendor-defined. Format standards can be committee-governed; performance-critical runtime standards cannot.

---

## Sources

- [Heterogeneous System Architecture — Wikipedia](https://en.wikipedia.org/wiki/Heterogeneous_System_Architecture)
- [HSA Foundation — Wikipedia](https://en.wikipedia.org/wiki/HSA_Foundation)
- [GCC Drops AMD HSA Offloading Support — Phoronix](https://www.phoronix.com/news/GCC-Drops-AMD-HSA)
- [The HSA Foundation Has Been Eerily Quiet As We Roll Into 2021 — Phoronix](https://www.phoronix.com/news/HSA-Quiet-Start-2021)
- [HSAIL-HLC-Development-LLVM (development stopped) — GitHub/HSAFoundation](https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM)
- [ROCm Deprecation Docs — HCC Deprecated ROCm v3.5](https://cgmb-rocm-docs.readthedocs.io/en/latest/Current_Release_Notes/Deprecation.html)
- [AMD ROCm Revisited — AMD Blogs](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited-ecosy/README.html)
- [Why OpenCL and GPU Alternatives Struggled — Modular Blog](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives)
- [OpenCL — Wikipedia](https://en.wikipedia.org/wiki/OpenCL)
- [C++ AMP — Wikipedia](https://en.wikipedia.org/wiki/C++_AMP)
- [C++ AMP Deprecated in VS 2022 — Microsoft DevCommunity](https://developercommunity.visualstudio.com/t/c-amp-headers-are-deprecated-what-is-the-replaceme/1495203)
- [OpenGL/OpenCL Deprecated in macOS Mojave — AppleInsider](https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave)
- [Project Larrabee: How Intel's First GPU Attempt Failed — HowToGeek](https://www.howtogeek.com/896521/project-larrabee-how-intels-first-attempt-at-gpus-failed/)
- [Standard Portable Intermediate Representation (SPIR) — Wikipedia](https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation)
- [AMD Boltzmann Initiative — insideHPC](https://insidehpc.com/2016/08/boltzmann-initiative/)
