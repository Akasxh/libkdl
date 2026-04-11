# Wave 05 — Abandoned Heterogeneous Dispatch: Dispatch-Layer Failures and Lessons for libkdl

**Survey:** Heterogeneous GPU Kernel Dispatch in LLVM/MLIR
**Angle:** abandoned-heterogeneous-dispatch
**Search queries:**
- "HSA heterogeneous system architecture abandoned GPU unified memory dispatch"
- "GPU Ocelot project abandoned CUDA PTX heterogeneous dispatch"
- "HSAIL ISA deprecated AMD HSA runtime history"
- "why HSA failed adoption NVIDIA never joined"
- "OpenCL decline failure adoption GPU compute heterogeneous"
- "AMD APU hUMA heterogeneous unified memory abandoned"
- "GCC HSA offloading dropped 2020 unmaintained"
- "SPIR-V trouble fragmentation vendor extensions 2022"
**Priority source types:** blog, news, historical analysis, academic
**Date:** 2026-04-06

**Note:** `wave-05-abandoned-hsa.md` covers HSA architecture internals (hUMA, hQ, HSAIL, AQL) and the consortium-level failure story in depth. This report is complementary — it focuses specifically on (a) the **dispatch-layer** failure mechanisms, (b) GPU Ocelot as a dynamic PTX dispatch precedent, (c) SPIR-V fragmentation as an ongoing portability risk, and (d) concrete failure-mode taxonomy applicable to libkdl design.

---

## Sources

### Source 1 — The HSA Foundation Has Been Eerily Quiet As We Roll Into 2021 — Phoronix
- **URL:** https://www.phoronix.com/news/HSA-Quiet-Start-2021
- **Date:** January 2021
- **Type:** News / analysis
- **Relevance:** 10/10
- **Novelty:** 8/10
- **Summary:** Phoronix reports HSA Foundation Twitter silent since February 2020, GitHub source trees with no meaningful commits for ~2 years. Informal comments from HSA working group members described the situation ranging from "stagnating" to "dissolving." The Foundation's website was down for over a month. AMD had redirected all effort to ROCm, which renders HSAIL and AQL-based dispatch obsolete. The GCC compiler had already dropped HSA offloading support (21,000 lines removed, August 2020).
- **Key dispatch lesson:** When a dispatch abstraction depends on a single champion vendor (AMD) and lacks a second major implementer (NVIDIA, Intel), the dispatch ABI becomes unmaintained and is eventually erased from toolchains. Neutrality in dispatch-layer design is an existential requirement, not a nice-to-have.

---

### Source 2 — AMD HSA Offloading Support Dropped From The GCC Compiler — Phoronix
- **URL:** https://www.phoronix.com/news/GCC-Drops-AMD-HSA
- **Date:** August 2020
- **Type:** News
- **Relevance:** 10/10
- **Novelty:** 9/10
- **Summary:** SUSE, which had done HSA bring-up for GCC under AMD contract, removed the HSA offloading backend (~21,000 lines) from GCC and libgomp. Reason: essentially zero documented real-world usage and no maintenance from AMD after ROCm became the preferred stack. Also notable: Intel MIC (Xeon Phi) offloading was similarly dropped from GCC 13 for the same reason (https://www.phoronix.com/news/Intel-MIC-Dropped-GCC-13), establishing a pattern of bespoke offload backends being dropped when usage cannot be demonstrated.
- **Key dispatch lesson:** An offload dispatch path with no measurable usage will be deleted from open-source compilers. libkdl must be deployable against real workloads (e.g., PyTorch custom ops) to survive — proof-of-use is a design requirement, not a stretch goal.

---

### Source 3 — GPU Ocelot: A Dynamic Compilation Framework for PTX — GitHub (gtcasl)
- **URL:** https://github.com/gtcasl/gpuocelot
- **Date:** 2008–2013 (active development), 2023 (community fork revived)
- **Type:** Source code / project history
- **Relevance:** 9/10
- **Novelty:** 9/10
- **Summary:** GPU Ocelot (Georgia Tech, Gregory Diamos, Andrew Kerr et al.) was the closest historical predecessor to libkdl. It intercepted CUDA runtime calls and JIT-compiled PTX to multiple backends: NVIDIA GPUs (pass-through), AMD GPUs (via CAL/ATI Stream), and x86 multicore CPUs (via LLVM IR translation). The core innovation was a transparent interposition layer — existing CUDA binaries ran unmodified and were redirected to whichever backend was available at load time. Development peaked at GTC 2012 (https://developer.download.nvidia.com/GTC/PDF/GTC2012/Posters/P0534_gpuOcelot-gtcSpring2012_V2.pdf) and the original repository went dormant circa 2013–2014, coinciding with CUDA 5.x ABI changes that broke interception. A community fork (gpuocelot/gpuocelot) revived it in 2023 with LLVM 15 and Ubuntu 22.04 support.
- **Failure causes identified:**
  1. CUDA ABI opacity — NVIDIA changed binary ABI without public specification, breaking the interception layer repeatedly.
  2. AMD backend dependency on the proprietary CAL/ATI Stream SDK, which AMD subsequently discontinued.
  3. No institutional maintainer after the Georgia Tech lab dispersed — a single-university project without industry backing.
  4. PTX is NVIDIA-controlled: new CUDA features (Tensor Cores, cooperative groups) required PTX extensions that AMD/CPU backends could not implement.
- **Key dispatch lesson for libkdl:** A dispatch layer that intercepts a proprietary binary format (PTX) is permanently at the mercy of the ISA owner. libkdl's architecture must target a vendor-neutral IR (SPIR-V or MLIR) at the dispatch boundary — not a vendor-owned bytecode.

---

### Source 4 — Why OpenCL and C++ GPU Alternatives Struggled — Medium / The Software Frontier
- **URL:** https://medium.com/the-software-frontier/making-ai-compute-accessible-to-all-part-5-why-opencl-and-c-gpu-alternatives-struggled-to-912dfb7baf3a
- **Date:** 2025
- **Type:** Technical analysis
- **Relevance:** 8/10
- **Novelty:** 7/10
- **Summary:** OpenCL's dispatch model failed not for technical reasons but for ecosystem integration reasons. NVIDIA built cuDNN, cuBLAS, TensorRT — optimized library layers on top of CUDA — and no equivalent existed for OpenCL. PyTorch and TensorFlow defaulted to CUDA dispatch and OpenCL was never a first-class execution provider. The article identifies the core pattern: open dispatch standards need not just a portable dispatch path but also portable high-performance implementations of all layers above the dispatch interface (BLAS, DNN, random number generation, etc.).
- **Key dispatch lesson:** A dispatch layer that is technically correct but produces 0.3x performance versus the vendor-native path will not be adopted. libkdl's dispatch must route to vendor-tuned kernels (cuBLAS, rocBLAS, oneMKL) not to a single portable fallback. The dispatch table must point to the best available implementation per target, not a lowest-common-denominator kernel.

---

### Source 5 — OpenCL vs CUDA — Why NVIDIA's Ecosystem Still Dominates in 2026 — ThunderCompute
- **URL:** https://www.thundercompute.com/blog/opencl-vs-cuda
- **Date:** 2026
- **Type:** Blog / competitive analysis
- **Relevance:** 7/10
- **Novelty:** 6/10
- **Summary:** Documents that OpenCL, despite ISO standardization and broad vendor backing (Apple, AMD, Intel, Qualcomm), was deprecated by Apple in macOS 10.14 (2018) in favor of Metal, and has progressively lost mindshare in AI/ML workloads. By 2026, OpenCL retains embedded/mobile foothold but is absent from major ML training pipelines. Highlights that NVIDIA's deliberate non-participation in OpenCL optimization (providing mediocre performance on NVIDIA hardware) actively undermined adoption — a competitive moating strategy.
- **Key dispatch lesson:** Dominant vendors can kneecap open dispatch standards by underperforming intentionally on their own hardware. libkdl must use vendor's own runtime paths (CUDA driver API, ROCm HIP, Level Zero) as its dispatch backends — not try to replace them — to avoid being sandbagged by deliberate performance regression.

---

### Source 6 — The Trouble with SPIR-V, 2022 Edition — Gob's Blog
- **URL:** https://xol.io/blah/the-trouble-with-spirv/
- **Date:** 2022
- **Type:** Technical blog / critical analysis
- **Relevance:** 9/10
- **Novelty:** 9/10
- **Summary:** Documents concrete fragmentation in SPIR-V as a "universal" dispatch IR. Key issues:
  1. **OpenCL vs Vulkan SPIR-V are disjoint subsets** — the Kernel execution model (OpenCL) and GLCompute execution model (Vulkan) share the SPIR-V binary format but have incompatible semantics for control flow, memory model, and subgroup operations. A kernel compiled for one environment cannot be dispatched through the other without re-compilation.
  2. **Control flow reconvergence is environment-defined** — OpenCL makes subgroup divergence undefined behavior; Vulkan requires structured control flow annotations. This means SPIR-V at the dispatch boundary does not constitute a portable kernel binary — it is a portable syntax with target-specific semantics.
  3. **Vendor extension proliferation** — Each GPU vendor (AMD, NVIDIA via `SPV_NV_*`, Intel via `SPV_INTEL_*`) adds capability extensions that create de facto incompatible dialects. A kernel using `SPV_NV_cooperative_matrix` cannot be dispatched on AMD hardware.
- **Key dispatch lesson:** SPIR-V is portable syntax, not portable semantics. A dispatch layer claiming SPIR-V as its universal kernel format must implement per-target semantic negotiation — checking extension availability, rejecting or retranslating kernels with unsupported capabilities before dispatch. This is precisely the finalization step libkdl must implement.

---

### Source 7 — Heterogeneous System Architecture — Wikipedia
- **URL:** https://en.wikipedia.org/wiki/Heterogeneous_System_Architecture
- **Date:** Current
- **Type:** Reference
- **Relevance:** 8/10
- **Novelty:** 5/10 (context only, not novel)
- **Summary:** Confirms HSA 1.0 specification released March 2015. Hardware implementations: AMD Kaveri (2014, partial), Carrizo (2015), Qualcomm Snapdragon 820 (2016). NVIDIA and Intel never joined. The AQL (Architected Queuing Language) packet-based dispatch protocol was the key innovation — kernels were dispatched by writing a 64-byte AQL dispatch packet into a hardware queue, bypassing the OS kernel entirely. This was a genuine advance in dispatch latency. AMD's transition to ROCm preserved AQL underneath: the ROCm HIP runtime still uses AQL packets internally via the HSA runtime API (ROCR). The Foundation-level effort collapsed; the technical mechanism survived inside AMD's walled garden.
- **Key dispatch lesson:** The right technical mechanism can survive inside a single vendor's stack even when the cross-vendor standard fails. libkdl can leverage AQL-based dispatch on AMD via ROCm HIP without requiring the HSA Foundation to exist — the mechanism is available through stable public APIs.

---

### Source 8 — Cross-Vendor GPU Programming: Extending CUDA Beyond NVIDIA — ACM HPDC 2025
- **URL:** https://dl.acm.org/doi/full/10.1145/3723851.3723860
- **Date:** 2025 (4th Workshop on Heterogeneous Composable and Disaggregated Systems)
- **Type:** Academic paper
- **Relevance:** 8/10
- **Novelty:** 8/10
- **Summary:** Explores current state of CUDA portability tools (HIP, SYCL via DPC++, chipStar) and identifies that cross-vendor dispatch still requires either source-level translation (HIP) or ahead-of-time recompilation (SYCL). There is no runtime that can take a CUDA binary and dispatch it on AMD hardware without a prior compilation step. Confirms the dispatch gap that libkdl targets: existing portability tools are compile-time, not runtime.
- **Key dispatch lesson:** The absence of runtime cross-vendor kernel dispatch (as opposed to compile-time translation) is confirmed as a genuine gap in the 2025 landscape. This validates libkdl's design premise.

---

## Synthesis

### Failure Mode Taxonomy for Heterogeneous Dispatch Projects

Analysis of the failed projects above yields five distinct failure modes applicable to libkdl risk assessment:

**FM-1: Single-vendor dependency (HSA Foundation, GPU Ocelot AMD backend)**
Standard or project depended on one vendor both as technical implementer and ecosystem driver. When that vendor redirected (AMD → ROCm), the project lost its only support. Mitigation for libkdl: implement CUDA, HIP, and Level Zero backends independently; no single backend should be load-bearing.

**FM-2: Proprietary ABI fragility (GPU Ocelot PTX interception)**
Dispatch layer intercepted a vendor-owned binary format that changed incompatibly without notice. Each CUDA version broke the interception machinery. Mitigation for libkdl: dispatch at the source-level kernel boundary (SPIR-V or MLIR), not at the binary level of any vendor's format.

**FM-3: Performance underdelivery (OpenCL vs CUDA)**
Portable dispatch layer produced correct but significantly slower results than the vendor-native path. Adoption collapsed when ML frameworks benchmarked both paths. Mitigation for libkdl: dispatch table must route to vendor-tuned implementations (cuBLAS, rocBLAS) per target, not to a generic fallback kernel.

**FM-4: Semantic fragmentation masquerading as portability (SPIR-V 2022)**
Universal IR that has vendor-incompatible subsets and extension-based divergence was claimed as a portable dispatch format but required per-target semantic negotiation. Mitigation for libkdl: include a capability-query phase before dispatch that validates kernel compatibility with the selected backend, with graceful fallback.

**FM-5: Zero demonstrated usage leading to toolchain removal (GCC HSA, Intel MIC)**
Offload paths with no documented real-world workloads were deleted from open-source compilers as maintenance burden. Mitigation for libkdl: poster must include at least one concrete benchmark demonstrating dispatch on a real ML kernel (matrix multiply, attention) on two distinct hardware targets.

---

### Positive Survivors: What Worked

Despite the failures above, several mechanisms from these projects survived:

| Mechanism | Original Project | Survived In |
|---|---|---|
| AQL packet dispatch | HSA Foundation | ROCm ROCR runtime |
| hUMA unified memory | AMD HSA/hUMA | MI300A UPM, CUDA Unified Memory |
| JIT-to-multiple-backends | GPU Ocelot | LLVM offloading, IREE HAL |
| PTX-to-LLVM IR translation | GPU Ocelot | NVPTX backend in LLVM mainline |
| Vendor-neutral IR | SPIR (original) | SPIR-V (succeeded by Vulkan/OpenCL 2.1) |

These survivors share a common pattern: they were adopted into either a standards body (Khronos SPIR-V) or a dominant vendor's stack (AMD ROCm) before the originating organization collapsed.

**Implication for libkdl:** To survive, libkdl needs to either (a) be absorbed into LLVM's offload runtime as a named feature, or (b) demonstrate performance on production workloads compelling enough for ROCm or CUDA teams to reuse the dispatch table design. The poster should explicitly frame the LLVM upstreaming path.

---

## Angle Assessment

- **Relevance:** 9/10 — Direct negative-space evidence for libkdl's design rationale. Every failure mode documented here corresponds to a design decision in kdl.c.
- **Novelty:** 8/10 — The FM-3 (performance underdelivery) and FM-4 (SPIR-V semantic fragmentation) angles are underrepresented in existing wave reports. FM-2 (GPU Ocelot PTX interception) is novel to this survey. `wave-05-abandoned-hsa.md` covers FM-1 and FM-5 more thoroughly.
- **Gap identified:** No wave report has yet examined GPU Ocelot in depth as a direct dispatch-layer precedent. A focused deep-dive on GPU Ocelot's architecture vs. libkdl architecture would be high value for the poster's related-work section.
- **Recommended follow-up angle:** `gpu-ocelot-architecture-comparison` — compare GPU Ocelot's dispatcher (PTX → backend IR → JIT) against libkdl's dispatcher (SPIR-V/PTX → kdl.c dispatch table → vendor runtime). Key differences: Ocelot operated below the runtime API; libkdl operates above it. Ocelot implemented backends; libkdl delegates to vendor runtimes. This distinction is the core argument for libkdl's superior robustness against FM-2.

---

## Sources (Full List)

- [The HSA Foundation Has Been Eerily Quiet As We Roll Into 2021 - Phoronix](https://www.phoronix.com/news/HSA-Quiet-Start-2021)
- [AMD HSA Offloading Support Dropped From The GCC Compiler - Phoronix](https://www.phoronix.com/news/GCC-Drops-AMD-HSA)
- [Intel MIC Offloading For Xeon Phi Dropped With GCC 13 - Phoronix](https://www.phoronix.com/news/Intel-MIC-Dropped-GCC-13)
- [GitHub - gtcasl/gpuocelot: GPUOCelot Dynamic Compilation Framework for PTX](https://github.com/gtcasl/gpuocelot)
- [GPU Ocelot GTC 2012 Poster (NVIDIA)](https://developer.download.nvidia.com/GTC/PDF/GTC2012/Posters/P0534_gpuOcelot-gtcSpring2012_V2.pdf)
- [GitHub - gpuocelot/gpuocelot: GPUOcelot (revived fork, LLVM 15)](https://github.com/gpuocelot/gpuocelot)
- [Why OpenCL and C++ GPU Alternatives Struggled - Medium / The Software Frontier](https://medium.com/the-software-frontier/making-ai-compute-accessible-to-all-part-5-why-opencl-and-c-gpu-alternatives-struggled-to-912dfb7baf3a)
- [OpenCL vs CUDA — Why NVIDIA's Ecosystem Still Dominates in 2026 - ThunderCompute](https://www.thundercompute.com/blog/opencl-vs-cuda)
- [The Trouble with SPIR-V, 2022 Edition - Gob's Blog](https://xol.io/blah/the-trouble-with-spirv/)
- [Heterogeneous System Architecture - Wikipedia](https://en.wikipedia.org/wiki/Heterogeneous_System_Architecture)
- [HSA Foundation - Wikipedia](https://en.wikipedia.org/wiki/HSA_Foundation)
- [Cross-Vendor GPU Programming: Extending CUDA Beyond NVIDIA - ACM 2025](https://dl.acm.org/doi/full/10.1145/3723851.3723860)
- [Heterogeneous Systems Architecture (HSA) - the TL;DR - StreamHPC](https://streamhpc.com/blog/2014-02-05/heterogeneous-systems-architecture-hsa-the-tldr/)
- [HSA Foundation Aims for Broader Adoption - hsafoundation.com 2016](https://hsafoundation.com/2016/07/04/hsa-foundation-aims-broader-adoption-coherent-memory-standard-heterogeneous-processors/)
