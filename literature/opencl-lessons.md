# OpenCL: History, Failures, and Lessons for Portable GPU Kernel Dispatch

*Research note for LLVM Dublin 2026 poster — Heterogeneous GPU Kernel Dispatch via MLIR*
*Authored: 2026-04-02*

---

## 1. OpenCL's Write-Once-Run-Anywhere Vision: History and Evolution

### Origins

OpenCL was conceived by Apple and submitted to the Khronos Group in 2008. The Khronos Compute Working Group was formed on June 16, 2008, and produced the OpenCL 1.0 specification by November 18, 2008 — a five-month sprint. OpenCL 1.0 shipped publicly with Mac OS X Snow Leopard on August 28, 2009.

From inception, OpenCL was framed as the open answer to NVIDIA's proprietary CUDA: a cross-platform, royalty-free standard enabling portable parallel computation across CPUs, GPUs, FPGAs, and DSPs from any vendor. The founding promise was "write once, run anywhere" for high-performance compute.

### Version Timeline

| Version | Year | Key Feature |
|---------|------|-------------|
| 1.0 | 2008 | Initial spec; basic platform/device/context/kernel model |
| 1.1 | 2010 | Sub-buffers, user events, 3D image writes |
| 1.2 | 2011 | Device fission, built-in kernels, offline compilation support |
| 2.0 | 2014 | SVM (shared virtual memory), pipes, device-side enqueue, atomics |
| 2.1 | 2015 | SPIR-V ingestion (core), C++ kernel language (preview) |
| 2.2 | 2017 | C++ kernel language finalized |
| 3.0 | 2020 | Radical restructuring: OpenCL 1.2 mandatory; all 2.x features optional |

### The "Write Once, Run Anywhere" Trap

The foundational promise broke down at the performance layer. OpenCL provided *source portability* — the same kernel source compiles on NVIDIA, AMD, Intel, ARM. But *performance portability* was never guaranteed or even addressed by the specification. Benchmarks repeatedly showed:

- A kernel tuned for AMD GCN underperformed by 3–5x on NVIDIA Fermi.
- An optimally tuned CUDA kernel on the same NVIDIA hardware outperformed the best OpenCL port by 30–50% in AI inference workloads [ThunderCompute 2026].
- Academic studies quantified the gap: proper per-device tuning could lift performance from ~15% to ~67% of native on Intel Ivy Bridge — but that tuning was entirely programmer-driven and not transferable across devices [ISC 2013, "Improving Performance Portability in OpenCL Programs"].

The core issue: tuning parameters such as work-group size, local memory tiling, loop unroll factors, and data layout are architecture-specific. OpenCL exposed control over these parameters but gave developers no tooling to make them portable.

---

## 2. Why OpenCL Lost to CUDA

### 2.1 Strategic NVIDIA Behavior

NVIDIA co-invented CUDA in 2006 and treated it as a platform moat. Their OpenCL engagement was deliberate underinvestment:

- NVIDIA maintained a minimal, "token" OpenCL implementation that exposed only OpenCL 1.2 features until OpenCL 3.0 conformance in 2023.
- Tensor Cores — the hardware units powering modern DNN GEMM — were never exposed through OpenCL, creating a 5–10x throughput gap on any DNN workload where cuDNN leverages them.
- NVIDIA co-designed CUDA's high-level libraries (cuDNN, cuBLAS, cuFFT, TensorRT) in lockstep with PyTorch and TensorFlow, ensuring these frameworks always ran fastest on NVIDIA/CUDA.
- NVIDIA began evaluation support of OpenCL 2.0 only in February 2017 (driver 378.66), and never completed it.

### 2.2 Committee-Driven Governance: "Coopetition Dysfunction"

OpenCL was managed by the Khronos Group — a consortium of competing hardware vendors. The structural problem:

- Every company (Intel, AMD, NVIDIA, Qualcomm, ARM) had different incentives and priorities.
- Consensus requirements meant spec updates moved at "committee consensus speed," which was described as "glacial" relative to CUDA's continuous shipping velocity.
- Companies withheld innovation announcements until after hardware shipped, preventing the standard from tracking hardware capability.
- Apple contributed only a specification, not a reference implementation. This forced every vendor to build their own fork from scratch. Without a shared evolving codebase, OpenCL became a patchwork of vendor-specific implementations with divergent edge-case behavior [Modular 2025].

### 2.3 No Reference Implementation → Fragmentation

The absence of a canonical implementation had cascading consequences:

1. **Behavioral divergence**: The same OpenCL kernel produced different results or crashed on different driver versions, making cross-vendor testing expensive.
2. **Extension sprawl**: Vendors introduced vendor-specific extensions (`cl_nv_*`, `cl_amd_*`, `cl_intel_*`) that developers needed to use for reasonable performance, defeating portability.
3. **Uneven feature support**: OpenCL 2.0's SVM required hardware coherency support; Intel and AMD shipped it, NVIDIA never did. Apple never went beyond 1.2. This meant writing code for OpenCL 2.0 features was futile if you needed cross-vendor coverage.
4. **ICD fragility**: The Installable Client Driver (ICD) mechanism — a dispatch table loaded at runtime via `/etc/OpenCL/vendors/*.icd` on Linux or registry keys on Windows — correctly allowed multiple vendor implementations to coexist. But it shifted debugging complexity onto the developer, who had to trace which ICD was being invoked and why behavior differed.

### 2.4 Ecosystem and Tooling Gap

CUDA arrived with:
- `nvcc` (tightly integrated compiler)
- `cuda-gdb`, `cuda-memcheck`, Nsight Systems/Compute (profiling)
- cuDNN, cuBLAS, cuSPARSE, cuFFT (tuned library layer)
- Seamless integration into PyTorch/TensorFlow from day one

OpenCL had:
- Runtime string compilation via `clBuildProgram` (driver-owned)
- Vendor-specific profilers with no shared standard
- No standardized high-level library for GEMM, convolution, or FFT
- Community tooling fragmented across vendors

The result: developing ML models with OpenCL required reinventing library infrastructure that CUDA provided out of the box, and that infrastructure would need to be tuned per-vendor.

### 2.5 Apple's Deprecation: A Symbolic Collapse

Apple — OpenCL's original architect — deprecated it in macOS 10.14 Mojave in June 2018, declaring Metal the successor for GPU compute. Active development ceased; the API was marked legacy. By 2021 (Security Update 2021-002), OpenCL support was silently dropped from macOS entirely. This sent a powerful signal to the developer community: the framework's own creator had abandoned it.

---

## 3. What Worked in OpenCL

Despite its commercial failure in the AI space, OpenCL established several durable architectural patterns worth preserving.

### 3.1 The Platform/Device/Context Abstraction Model

OpenCL's runtime model was conceptually sound and has been adopted by successors:

- **Platform**: A vendor implementation (one ICD entry = one platform)
- **Device**: A compute unit (GPU, CPU core, FPGA) within a platform
- **Context**: A shared execution environment binding devices + memory spaces
- **Command Queue**: Per-device, in-order or out-of-order execution queue
- **Kernel**: A compiled function dispatched via `clEnqueueNDRangeKernel`
- **Memory Objects**: Buffers and images managed with explicit `clEnqueueReadBuffer`/`clEnqueueWriteBuffer` or SVM pointers

This model cleanly separated the compute resource hierarchy from the execution model. SYCL (built on OpenCL concepts), HIP, and IREE's HAL all use analogous abstractions.

### 3.2 SPIR-V: The Key Success

OpenCL 2.1 (2015) introduced SPIR-V as the standard portable intermediate representation for kernel ingestion. This was the most architecturally significant contribution OpenCL made to the GPU ecosystem.

**What SPIR-V solved:**
- Removed the need for vendor drivers to include a full high-level language compiler (removing OpenCL C parser from the driver reduces driver complexity dramatically)
- Provided a stable binary encoding with explicit versioning and capability declarations, addressing LLVM IR's weak compatibility guarantees across driver generations
- Enabled independent toolchains to generate portable compute code; compilation once to SPIR-V, then vendor-specific native codegen at driver load time
- Introduced "capabilities" and "dialects": the Kernel dialect targets OpenCL compute via `llvm-spirv`, the Shader dialect targets Vulkan via `clspv`, enabling the same IR format to serve both the graphics and compute stacks

**SPIR-V vs LLVM IR for GPU targets:**

| Dimension | LLVM IR | SPIR-V |
|-----------|---------|--------|
| Stability guarantee | Weak (breaks across LLVM versions) | Strong (versioned binary format) |
| Driver compatibility | Requires matching LLVM version | Explicit capability negotiation |
| Optimization tooling | Rich (full LLVM pass pipeline) | Limited (spirv-opt, spirv-val) |
| GPU-native constructs | Generic, must be lowered | Native (work-groups, memory qualifiers) |
| MLIR integration | LLVM dialect (first-class) | SPIR-V dialect (first-class) |

**MLIR's SPIR-V dialect** (in-tree since LLVM 11) provides direct conversion from linalg, affine, and GPU dialects to SPIR-V. LLVM 19 added a generic MLIR-to-SPIR-V lowering pass, contributed by an AMD AI compiler engineer. This directly enables the research question: MLIR-compiled kernels can produce SPIR-V that routes to any vendor's OpenCL or Vulkan driver.

### 3.3 Runtime Compilation (clBuildProgram): Strengths and Limits

OpenCL's `clBuildProgram` performs JIT compilation of OpenCL C source (or SPIR-V) into device-native code at runtime. This was the original portability mechanism.

**Strengths:**
- Host binary is device-independent; the same `.so` runs on any conformant OpenCL device
- Adaptive compilation: kernels can be specialized with `-D TILE_SIZE=32` build flags at dispatch time, enabling runtime auto-tuning
- Eliminates the need to ship device-native binaries for every possible GPU SKU
- When using SPIR-V input, compilation is faster (no text parsing) and the kernel's IP is protected from trivial reverse engineering

**Weaknesses (empirically measured):**
- Compilation overhead per kernel: 0.1–1 second typical; worst-case >5 seconds for 64 kernels compiled individually on NVIDIA [Rupp 2016]
- Optimal strategy: pack related kernels into the same `cl_program` object; 2–4 programs is the sweet spot
- Early OpenCL: no standardized binary caching; each vendor implemented their own, or none at all. Libraries had to duplicate caching logic independently
- SPIR-V input reduces but does not eliminate compilation time; the native codegen phase still runs per device
- Not suited for hard real-time systems; embedded deployments need offline-compiled SPIR-V binaries

**For MLIR integration**: the `clBuildProgram` model is directly analogous to what an MLIR-based runtime dispatcher would need — a mechanism to take a portable IR (SPIR-V or MLIR blob), present it to the target device's driver, and receive a device-native executable. The key lesson is to build binary caching into the dispatch layer from day one, not retrofit it later.

---

## 4. OpenCL 3.0: The Pragmatic Retreat

### 4.1 What Changed

Released September 30, 2020 (provisional April 2020), OpenCL 3.0 represented a fundamental philosophical reversal.

**The core change:** OpenCL 1.2 functionality is the mandatory baseline. All OpenCL 2.x and 3.0 features are optional and must be queried via API before use.

This was an explicit acknowledgment that the OpenCL 2.x ambition — SVM, pipes, device-side enqueue, C++ kernels — had outpaced vendor willingness to implement. NVIDIA had refused to move beyond 1.2 entirely.

**Mandatory (all OpenCL 3.0 implementations must provide):**
- Full OpenCL 1.2 feature set
- API query mechanism for optional features
- Language macros indicating optional language feature availability

**Optional (queryable, not guaranteed):**
- All OpenCL 2.x features: SVM, pipes, device-side enqueue, program-scope global variables, OpenCL C 2.0 atomics
- OpenCL C 3.0 language features: generic address space, 3D image writes, subgroups, etc.

**Other changes:**
- OpenCL C++ kernel language formally removed; community-developed "C++ for OpenCL" (via Clang, outputting SPIR-V) recommended instead
- Unified specification: all versions described in one document (no more per-version spec splits)
- Two new extensions: UUID device queries (cross-API device identity), ordered DMA transactions
- New official Khronos OpenCL SDK released under Apache 2.0: ICD loader, headers, OpenCL Guide, samples

### 4.2 Analysis: Was OpenCL 3.0 Too Late?

The "optional everything" design was pragmatically correct but came a decade late. By 2020:
- PyTorch and TensorFlow were deeply CUDA-native
- cuDNN was the de facto standard for DNN primitives
- The ML research community had fully standardized on CUDA
- Apple had already deprecated OpenCL

OpenCL 3.0 stabilized the base but couldn't recover lost ecosystem momentum. Its importance shifted: it became the standard for embedded, mobile, and industrial compute (ARM Mali, Qualcomm Adreno, automotive SoCs) where CUDA was never relevant.

**Critical design lesson for MLIR-based dispatch:** The optional query mechanism of OpenCL 3.0 is the right model for a heterogeneous runtime. Any portable dispatch layer must be able to introspect hardware capability at runtime and select the appropriate kernel variant. Static feature assumptions at compile time are incompatible with genuinely heterogeneous deployment.

---

## 5. Lessons for Any New Portable GPU Runtime

The following lessons are directly distilled from OpenCL's 15-year trajectory and apply to the MLIR-based heterogeneous dispatch problem.

### Lesson 1: Source Portability ≠ Performance Portability

OpenCL's biggest misstep was conflating the two. The spec guaranteed identical semantics across vendors but said nothing about performance. A portable runtime for ML kernels must include:

- Automated kernel variant selection based on hardware profiling
- Runtime auto-tuning or ahead-of-time tuning database lookup (per device SKU)
- Work-group size heuristics exposed as tunable parameters, not hardcoded

This is precisely what JIT-based dispatch in MLIR enables: selecting among pre-compiled kernel variants based on hardware introspection at dispatch time.

### Lesson 2: A Reference Implementation Is Non-Negotiable

Without PoCL, OpenCL 1.x/2.x would have had zero open-source conformant implementations. PoCL (using Clang + LLVM) proved that the ICD model works when you have a solid open reference. Any new portable runtime must ship a working reference implementation, not just a specification. This creates a shared baseline for testing, avoids behavioral divergence, and builds community confidence.

### Lesson 3: Do Not Expose Vendor-Specific Features Through Portability API

OpenCL's extension mechanism (`cl_nv_cooperative_matrix`, `cl_amd_fp64`, etc.) created a tiered reality: programs using only core OpenCL were artificially limited, while "portable" programs that used vendor extensions for performance were not actually portable. The lesson: design optional features as first-class, queryable, documented capabilities within the standard — not side channels. OpenCL 3.0's optional-features architecture is the correct model; it arrived 10 years late.

### Lesson 4: Binary Caching From Day One

The multi-second `clBuildProgram` overhead was a known problem from OpenCL's first release. It took until OpenCL 3.0's SDK to standardize tooling around it. Any runtime that JIT-compiles kernels needs:

- Deterministic cache keys (device UUID + driver version + kernel IR hash)
- A vendor-neutral cache storage format
- Invalidation on driver update
- Warm-path fallback to precompiled SPIR-V or native binaries

The OpenCL ICD UUID extension (new in 3.0) specifically addresses the device identity problem for cache key construction.

### Lesson 5: Avoid Committee Governance for Core Performance

OpenCL failed to track hardware innovation because every spec change required multi-vendor consensus. The correct model is: a stable, minimal, royalty-free ABI standard for device discovery and command submission, with performance-critical paths implemented in fast-moving open-source libraries (like oneDNN for Intel, ROCm for AMD) that implement higher-level operations on top.

MLIR's dialect system is structurally superior here: it allows incremental dialect development (e.g., `linalg`, `nvgpu`, `amdgpu` dialects) without requiring spec-committee approval. The stability boundary is the MLIR bytecode format, not the dialect vocabulary.

### Lesson 6: First-Class Framework Integration Is Required

OpenCL never achieved deep integration with PyTorch or TensorFlow. CUDA's success was inseparable from being the "default compute backend" for both frameworks from their earliest releases. Any new portable dispatch layer must provide:

- A clean ATen operator override path (for PyTorch)
- An XLA custom call interface (for TensorFlow/JAX)
- A way to register MLIR-compiled kernels as dispatch targets

Without this, even a technically superior runtime remains academically interesting but practically invisible.

### Lesson 7: The ICD Pattern Is Worth Preserving

OpenCL's ICD mechanism — a function-pointer dispatch table loaded from a vendor-provided `.so` via a loader — is a clean solution to the multi-vendor runtime coexistence problem. It allows:

- Multiple vendor implementations on the same system simultaneously
- Runtime selection of the appropriate platform/device
- Application code unchanged across vendor switches

The MLIR ExecutionEngine today fixes the target at compilation time. A dispatch layer that replicates the ICD pattern — with runtime platform/device enumeration and deferred kernel compilation — would directly address the gap identified in the poster's abstract.

---

## 6. OpenCL's Runtime Compilation Model: Deep Dive

### The Compilation Pipeline

```
Source: OpenCL C text  --(clBuildProgram)-->  device-native binary
Source: SPIR-V binary  --(clBuildProgram)-->  device-native binary (faster)
```

`clBuildProgram` accepts either OpenCL C source string or SPIR-V binary. The vendor driver owns the second stage (SPIR-V → native). This split is architecturally important:

- The first stage (to SPIR-V) is done once, offline, with any conformant toolchain (Clang, DPC++, nvc++, etc.)
- The second stage is vendor-optimized and cannot be standardized
- MLIR already produces SPIR-V via its SPIR-V dialect; plugging into `clBuildProgram` is a straightforward integration point

### Binary Object Format

`clGetProgramInfo(CL_PROGRAM_BINARIES)` returns the compiled binary. The format is **vendor-specific** and not portable — an AMD binary will not load on NVIDIA. This was a persistent portability regression: applications caching compiled binaries lost cross-vendor portability.

OpenCL 3.0 did not standardize the binary format. The only portable cached representation is SPIR-V.

**Implication for MLIR dispatch**: Cache SPIR-V, not native binaries. Native compilation happens once per device at first dispatch; SPIR-V is the distributable artifact.

### Online vs Offline Compilation

| Mode | API | Timing | Use Case |
|------|-----|--------|----------|
| Online (JIT) | `clBuildProgram(source)` | Runtime | Adaptive kernels, development |
| Online (SPIR-V) | `clBuildProgram(spir-v)` | Runtime (fast) | Deployed applications |
| Offline | `clBuildProgram(binary)` | Load-time | Embedded, real-time |

All OpenCL drivers are required to support online compilation from OpenCL C source (where supported). SPIR-V ingestion is standard from 2.1 onward, optional in 3.0 but universally supported in practice.

---

## 7. PoCL: Portable Computing Language

### Overview and Significance

PoCL (Portable Computing Language) is an MIT-licensed, conformant open-source implementation of OpenCL 3.0. It is the primary evidence that "build your own portable GPU runtime on top of LLVM" is not only possible but can be competitive with vendor implementations.

PoCL is directly relevant to this poster's thesis: it demonstrates how LLVM can serve as the portable kernel compilation layer, with device-specific code generation delegated to LLVM backends.

### Architecture

```
OpenCL Host API (cl.h)
        |
   PoCL ICD Layer
        |
   +-----------+----------+-----------+-----------+
   |           |          |           |           |
  CPU        CUDA      Level Zero   Vulkan     Remote
 (LLVM)    (libcuda)   (Intel GPU)  (Any GPU)  (TCP/IP)
```

**Kernel compilation pipeline (CPU device):**
1. `clBuildProgram` called with OpenCL C source or SPIR-V
2. Clang parses OpenCL C; emits LLVM IR (or SPIR-V is translated to LLVM IR via `llvm-spirv`)
3. PoCL's kernel compiler applies target-independent passes:
   - Work-group function generation (barrier handling)
   - Parallel region formation (preserving SIMD/SIMT data-parallelism metadata)
   - Horizontal autovectorization of work-item loops
4. LLVM backend emits target-native code (x86, ARM, RISC-V, etc.)
5. Binary cached with automated kernel compiler caching

The key architectural insight from the pocl IJPP 2015 paper: "parallel region formation retains the information of the data parallelism using the LLVM IR and its metadata infrastructure." This preserves optimization opportunities that source-level lowering would lose, and allows generic LLVM vectorization passes to effectively SIMD-ize OpenCL work-groups.

### Supported Targets (PoCL 7.1, October 2025)

| Target | Status | Backend |
|--------|--------|---------|
| x86-64 CPU | Conformant (OpenCL 3.0) | LLVM |
| ARM 64-bit | Tested | LLVM |
| RISC-V | Supported | LLVM |
| Intel GPU (Level Zero) | Conformant | Level Zero + LLVM |
| NVIDIA GPU | Tested (SM5.0+) | libCUDA + LLVM/PTX |
| Vulkan | Supported (LLVM 18) | LLVM SPIR-V |
| Remote (PoCL-R) | Stable (5.0+) | TCP/IP transport |
| FPGA | Experimental | Built-in kernel library |

LLVM version support: LLVM 14–22; tested on 18–22 in CI. A new PoCL release is issued after each LLVM major release.

### PoCL-Remote: Distributed OpenCL Without MPI

PoCL 5.0 introduced PoCL-Remote (PoCL-R), a client-server backend enabling transparent OpenCL execution across networked machines.

**Architecture**: Standard OpenCL API on the client; a "smart proxy" driver that serializes commands to a remote server over TCP; peer-to-peer data transfer and event signaling between devices.

**Performance**: Up to 50x lower latency than prior-art OpenCL distribution layers in synthetic benchmarks.

**Recent research** (2024–2025):
- Agarwal 2024 (Eindhoven): "Dynamic Device Management and Automatic Network Resource Discovery in OpenCL for Multi-Access Edge Computing"
- Žádník et al., IEEE WCNC 2025: "Open Software Stack for Compression-Aware Adaptive Edge Offloading"
- Solanti & Jääskeläinen, IWOCL 2025: "Latency Reduction Potential of Server-Side Command Buffers in OpenCL-Based Edge Offloading"
- Solanti & Jääskeläinen, Int'l J. High Performance Computing Applications 2025: "PoCL-R: An Open Standard Based Heterogeneous Offloading Layer with Server Side Scalability"

**Relevance**: The PoCL-R model provides a blueprint for extending an MLIR-based dispatch layer to heterogeneous edge/cloud topologies without requiring distributed computing frameworks.

### PoCL Performance

The foundational 2015 paper showed PoCL was "faster or close to as fast as the best proprietary OpenCL implementation for the platform at hand" on CPU targets. The 2021 Intel CPU study showed a performance gap on Intel hardware, which a TBB-based CPU driver closed — achieving up to 1.3x faster than Intel's own OpenCL SDK on some benchmarks.

**Key performance technique**: horizontal autovectorization of work-groups. PoCL transforms the NDRange loop over work-items into SIMD vector operations, exploiting CPU SIMD units (AVX2, AVX-512) without requiring the programmer to write vectorized code explicitly.

### SYCL Backend Usage

PoCL 4.0 enabled use as a SYCL runtime backend via the Level Zero driver and chipStar integration (~800 SYCL tests). This positions PoCL as an open-source alternative to Intel's OpenCL runtime for SYCL workloads, directly relevant to the SYCL-vs-OpenCL comparison in the poster.

---

## 8. OpenCL and MLIR: The Connection

### ClangIR + OpenCL C → SPIR-V (GSoC 2024)

A Google Summer of Code 2024 project established OpenCL C kernel support in ClangIR — the MLIR-based IR layer for Clang. This work:

- Implemented a unified address space design aligned with ClangIR's MLIR objectives
- Added SPIR-V calling conventions (`SpirKernel`, `SpirFunction`)
- Emitted structured MLIR attributes for kernel metadata
- Validated against 20 polybenchGpu OpenCL C benchmarks

This is foundational: it means Clang can now compile OpenCL C → ClangIR (MLIR) → SPIR-V, with MLIR serving as the optimization layer. The same MLIR infrastructure that lowers linalg GEMM operations can also process OpenCL-sourced kernels.

### MLIR SPIR-V Dialect

MLIR contains a first-class `spirv` dialect that:
- Represents SPIR-V opcodes as MLIR ops
- Supports bidirectional conversion: MLIR → SPIR-V binary, SPIR-V binary → MLIR
- Is the official downstream target for GPU dialect lowering in MLIR
- Was extended in LLVM 19 with a generic MLIR-to-SPIR-V pass (from AMD AI compiler team)

This means an MLIR-compiled kernel can be serialized to SPIR-V and dispatched through any OpenCL or Vulkan driver — including via PoCL's CPU backend for testing.

---

## 9. Critical Analysis: OpenCL as a Design Anti-Pattern (and Pattern)

### What OpenCL Got Right (and Successors Copied)

1. **Device abstraction model** — Platform → Device → Context → Queue → Kernel. SYCL, HIP, IREE HAL, and Metal all use this hierarchy.
2. **SPIR-V** — Now the de facto standard portable GPU IR, used by Vulkan, OpenCL, DirectX (via DXIL-from-SPIR-V), and WebGPU (WGSL compiles through SPIR-V toolchain).
3. **ICD mechanism** — Runtime multi-vendor dispatch via a loader library. Vulkan's layer system and IREE's driver model follow this pattern.
4. **Runtime feature querying** — OpenCL 3.0's queryable optional features. Vulkan made this the default from day one; OpenCL arrived at it after a decade of painful optional-but-mandatory ambiguity.
5. **Offline SPIR-V + online native compilation** — The two-phase model (portable IR + driver-native codegen) is the correct architecture. IREE uses it. MLIR's compilation pipeline models it.

### What OpenCL Got Wrong (and New Systems Must Avoid)

1. **No performance portability mechanism** — Source portability without performance portability is a half-solution. Auto-tuning must be a first-class feature.
2. **Spec without reference implementation** — Created divergence immediately. Always ship a working reference.
3. **Committee governance for performance features** — Acceptable for stability guarantees, fatal for innovation velocity. Separate the ABI standard from the performance library.
4. **Extension fragmentation** — Vendor extensions that bypass the standard for performance nullify portability. Design optional capabilities into the core from the start.
5. **JIT compilation overhead unaddressed** — Known from day one, solved 12 years later. Build binary caching into the dispatch layer as a core design invariant.
6. **No ML ecosystem integration** — CUDA won by being the default backend for PyTorch and TensorFlow. A portable runtime that doesn't integrate with these frameworks has no path to adoption.
7. **Ignored hardware divergence** — OpenCL 2.0 SVM required hardware coherency that only Intel provided. Features that presuppose specific hardware capabilities violate the portability contract.

---

## 10. Citation Index

| Reference | Source |
|-----------|--------|
| OpenCL history, Khronos Working Group formation 2008 | [Wikipedia: OpenCL](https://en.wikipedia.org/wiki/OpenCL) |
| Why OpenCL lost to CUDA: committee governance, fragmentation | [Modular Blog, "Democratizing AI Compute Part 5"](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives) |
| 5–10x CUDA performance advantage on Tensor Core workloads | [ThunderCompute, "OpenCL vs CUDA 2026"](https://www.thundercompute.com/blog/opencl-vs-cuda) |
| OpenCL JIT compilation overhead benchmarks (0.1–5s range) | [Karl Rupp, "OpenCL JIT Compilation Benchmarks" 2016](https://www.karlrupp.net/2016/01/opencl-just-in-time-jit-compilation-benchmarks/) |
| OpenCL 3.0 specification finalized: mandatory/optional split | [Khronos Blog, OpenCL 3.0 Finalized 2020](https://www.khronos.org/blog/opencl-3.0-specification-finalized-and-initial-khronos-open-source-opencl-sdk-released) |
| OpenCL 3.0: optional features guide | [KhronosGroup/OpenCL-Guide, opencl_3.md](https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/opencl_3.md) |
| OpenCL 2.0 SVM overview; Intel/AMD adoption, NVIDIA non-adoption | [Intel Developer, "OpenCL 2.0 SVM Overview"](https://www.intel.com/content/www/us/en/developer/articles/technical/opencl-20-shared-virtual-memory-overview.html) |
| Apple deprecates OpenCL in macOS Mojave 2018 | [AppleInsider, June 2018](https://appleinsider.com/articles/18/06/04/opengl-opencl-deprecated-in-favor-of-metal-2-in-macos-1014-mojave) |
| OpenCL performance portability gap: 15% → 67% with tuning | ISC 2013: "Improving Performance Portability in OpenCL Programs" (Springer) |
| Performance portability investigation: per-device tuning required | [ScienceDirect: "An Investigation of the Performance Portability of OpenCL"](https://www.sciencedirect.com/science/article/abs/pii/S0743731512001669) |
| PoCL architecture, LLVM kernel compiler, device support | [portablecl.org](https://portablecl.org/) |
| PoCL: performance-portable OpenCL implementation (IJPP 2015) | [arXiv:1611.07083](https://arxiv.org/abs/1611.07083) |
| PoCL-Remote: distributed OpenCL, 50x latency improvement | [Khronos News: PoCL-Remote Backend](https://www.khronos.org/news/permalink/new-pocl-remote-backend-enables-distributed-computing-with-pure-opencl-no-mpi-needed) |
| PoCL 5.0: PoCL-Remote stabilization, SPIR-V improvements | [Phoronix: PoCL 5.0](https://phoronix.com/news/PoCL-5.0-Released) |
| PoCL Intel CPU performance study (TBB driver, 1.3x faster) | [ACM DL: "Performance Evaluation and Improvements of PoCL on Intel CPUs"](https://dl.acm.org/doi/fullHtml/10.1145/3456669.3456698) |
| PoCL-R: 2025 journal paper on open standard offloading | [SAGE Journals 2025](https://journals.sagepub.com/doi/10.1177/10943420251369350) |
| SPIR-V: design, OpenCL integration, Vulkan dialect | [Wikipedia: SPIR](https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation) |
| LLVM IR vs SPIR-V vs MLIR: technical comparison | [Lei.Chat: Compilers and IRs](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/) |
| Generic MLIR-to-SPIR-V pass in LLVM 19 (AMD) | [Phoronix: LLVM 19 MLIR to SPIR-V](https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V) |
| OpenCL C in ClangIR (GSoC 2024) | [LLVM Blog, August 2024](https://blog.llvm.org/posts/2024-08-29-gsoc-opencl-c-support-for-clangir/) |
| MLIR SPIR-V dialect documentation | [mlir.llvm.org/docs/Dialects/SPIR-V](https://mlir.llvm.org/docs/Dialects/SPIR-V/) |
| Intel SPIR-V as default OpenCL interface | [Intel Developer: SPIR-V for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/case-study/spir-v-default-interface-to-intel-graphics-compiler-for-opencl-workloads.html) |
| OpenCL ICD mechanism architecture | [KhronosGroup/OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader) |
| AdaptiveCpp: runtime multi-vendor SYCL | [GitHub: AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) |
| NVIDIA OpenCL 3.0 conformance announcement | [NVIDIA Blog, "NVIDIA is Now OpenCL 3.0 Conformant"](https://developer.nvidia.com/blog/nvidia-is-now-opencl-3-0-conformant/) |

---

*End of document. Use alongside `spirv-lessons.md`, `mlir-dispatch-survey.md`, and `iree-analysis.md` when composing the poster narrative.*
