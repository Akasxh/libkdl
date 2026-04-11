# Competitive Landscape: libkdl vs. All Identified Alternatives

**Date:** 2026-04-06
**Survey scope:** 55+ wave files, ~520 individual sources across 6 waves
**Purpose:** Definitive positioning of libkdl against every identified system in the heterogeneous GPU kernel dispatch space

---

## Full Comparison Matrix

| System | Runtime Dispatch | Cross-Vendor | Pre-Compiled Variants | Per-Kernel Granularity | Cost Model | Selection Policy | Format | Status |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|--------|--------|
| **libkdl** | **Yes (load-time)** | **CUDA+HIP+CPU** | **Yes (native per-device)** | **Yes** | **Roofline cross-vendor** | **Capability-scored ranking** | MTB / OffloadBinary | Prototype (~5100 LOC) |
| HetGPU | Yes (JIT) | CUDA+HIP+Tenstorrent | No (hetIR JIT) | Yes | None | Virtual ISA translation | hetIR fat binary | arXiv preprint |
| liboffload | Yes (load-time) | CUDA+HIP+L0+Host | Yes | Yes | None | **First compatible wins** | OffloadBinary (.llvm.offloading) | LLVM mainline (pre-1.0) |
| Unified Runtime (UR) | Yes | CUDA+HIP+L0+OpenCL | Yes | Yes | None | Per-adapter, no cross-adapter | UR binary | Intel/LLVM (production) |
| IREE HAL | Yes (module load) | CUDA+ROCm+Vulkan+CPU | Yes (VMFB) | No (whole-module) | None in dispatch | Static boolean conditions | VMFB flatbuffer | Google (production) |
| AdaptiveCpp SSCP | Yes (JIT) | CUDA+HIP+L0+OpenCL | No (LLVM IR → JIT) | Yes | SQLite adaptivity DB | JIT specialization | Embedded LLVM IR | Production (IWOCL 2025) |
| SparseX | Yes | CUDA only | Yes (library selection) | Yes (SpMM only) | Lightweight classifier | Intra-vendor library ranking | N/A (library call routing) | CGO 2026 accepted |
| chipStar | Yes (JIT) | Any OpenCL/L0 target | No (SPIR-V → JIT) | Yes | None | Backend env var (CHIP_BE) | SPIR-V offload bundle | Production (Aurora exascale) |
| GPU Ocelot | Yes (JIT) | CUDA+AMD(CAL)+CPU | No (PTX → JIT) | Yes | None | Backend selection at load | PTX interception | Abandoned (2013), revived fork |
| AOTriton | Yes (load-time) | AMD only | Yes (per-gfx HSACO) | Yes | SQLite autotuning DB | Architecture hierarchy match | AKS2 (LZMA compressed) | AMD production |
| ONNX RT EPs | Yes (session init) | CUDA+TRT+ROCm+DML+CoreML+... | Yes (per-EP) | No (per-session, greedy first-fit) | None | Priority-ordered greedy | ONNX model + EP binaries | Production |
| ExecuTorch delegates | AOT (partition) | CoreML+QNN+XNNPACK+Vulkan+... | Yes (per-delegate .pte) | No (per-partition) | None | AOT partitioner heuristic | .pte per delegate | Meta production |
| PJRT | Yes (init-time) | CUDA+TPU+CPU+custom | Yes | No (per-process) | None | Plugin selection at init | Plugin .so | Google/JAX production |
| Alpaka SwitchProducer | Yes (event-level) | CUDA+HIP+CPU | Yes (separate builds) | No (per-event) | None | CMS SwitchProducer routing | N builds for N vendors | CERN CMS production |
| TVM VDevice | AOT (compile) | CUDA+ROCm+Vulkan+Metal+CPU | Yes | No (per-tensor annotation) | Deferred (never implemented) | Static PlanDevices pass | Relay/Relax module | Apache (stalled heterogeneous) |

---

## Dimension-by-Dimension Analysis

### 1. Runtime vs. Compile-Time Dispatch

**Runtime dispatch (kernel-granularity):** libkdl, HetGPU, AdaptiveCpp, chipStar, AOTriton, SparseX
**Runtime dispatch (coarser granularity):** liboffload, UR, IREE HAL (module), ONNX RT (session), PJRT (process), Alpaka (event)
**Compile-time only:** ExecuTorch (AOT partition), TVM VDevice (static pass)

libkdl is one of only six systems offering true per-kernel runtime dispatch. Of those six, only libkdl and AOTriton use pre-compiled native binaries (zero JIT). AOTriton is AMD-only.

### 2. Cross-Vendor Coverage

| System | NVIDIA | AMD | Intel | CPU | Other |
|--------|:---:|:---:|:---:|:---:|:---:|
| libkdl | CUDA | HIP | Planned | Yes | - |
| HetGPU | hetIR→CUDA | hetIR→HIP | - | - | Tenstorrent (unvalidated) |
| liboffload | Plugin | Plugin | Plugin (L0) | Plugin | Extensible |
| UR | Adapter | Adapter | L0 native | Adapter | OpenCL generic |
| IREE HAL | CUDA | ROCm | Vulkan | LLVM CPU | WebGPU |
| AdaptiveCpp | CUDA | HIP | L0 | OpenMP | OpenCL generic |
| chipStar | - | - | L0/OpenCL | OpenCL CPU | Mali, PowerVR (limited) |
| ONNX RT | CUDA/TRT | MIGraphX/ROCm* | OpenVINO/DML | CPU EP | 14+ EPs |
| PJRT | CUDA | ROCm (community) | - | CPU | TPU native |

*ROCm EP deprecated in ORT 1.17 [wave-03-onnxrt-execution-providers.md]

### 3. Selection Policy Sophistication

| Tier | Systems | Policy |
|------|---------|--------|
| **No policy** | liboffload, UR, PJRT, TVM | First match / single target |
| **Static/heuristic** | IREE HAL, ExecuTorch, Alpaka | Boolean conditions, AOT partitioner |
| **Greedy** | ONNX RT | Priority-ordered first-fit at session init |
| **Learned (intra-vendor)** | AOTriton, SparseX, AdaptiveCpp | SQLite DB, classifier, adaptivity DB |
| **Analytical cross-vendor** | **libkdl (proposed)** | Roofline cost model + Bloom filter + capability contracts |

libkdl is the only system proposing an analytical cross-vendor selection policy. SparseX (CGO 2026) validates that runtime library selection via predictive model is a recognized contribution class, but is CUDA-only and SpMM-only.

### 4. Binary Format and Packaging

| System | Format | Multi-vendor in single artifact? | Metadata richness |
|--------|--------|:---:|--------|
| libkdl MTB | ELF-based multi-variant bundle | **Yes** | Capability contracts, Bloom filters, autotuning DB |
| LLVM OffloadBinary | .llvm.offloading section (magic 0x10FF10AD) | **Yes** | Triple, arch, arbitrary StringMap |
| CUDA fatbin | nvFatbin container | NVIDIA-only | SM version, PTX fallback |
| AMD AKS2 | LZMA-compressed HSACO archive | AMD-only | gfx target, autotuning SQLite |
| IREE VMFB | FlatBuffer | Single-target per artifact | Full module metadata |
| ExecuTorch .pte | FlatBuffer | Single-delegate per artifact | Delegate metadata |
| chipStar | clang-offload-bundler v3 | SPIR-V (vendor-neutral IR) | Target triple |

libkdl MTB and LLVM OffloadBinary are the only formats designed to carry vendor-native binaries for multiple vendors in a single file. The recommended path: libkdl should consume OffloadBinary containers AND support extended MTB metadata.

### 5. JIT vs. AOT Tradeoff

| System | Strategy | Cold Start | Peak Performance |
|--------|----------|-----------|-----------------|
| libkdl | AOT (pre-compiled variants) | <5 ms | Native (vendor-tuned binary) |
| HetGPU | JIT (hetIR → native) | 10-200 ms | 5-15% overhead |
| AdaptiveCpp | JIT (LLVM IR → native) | JIT latency | +30% over CUDA via specialization |
| chipStar | JIT (SPIR-V → native) | 100ms-40min (uncached) | 0.75x native (geometric mean) |
| Proteus | JIT (LLVM IR specialization) | Background thread | 2.8x AMD, 1.78x NVIDIA |
| AOTriton | AOT (per-gfx HSACO) | Zero | Native AMD |
| PT2/Triton | JIT (Triton → PTX/HSACO) | **843s** cold (Meta large model) | Native via autotuning |

libkdl's AOT approach eliminates the 843s cold start entirely. AdaptiveCpp and Proteus demonstrate that JIT can exceed AOT in specific cases; libkdl should support LLVM IR fallback variants for this scenario.

### 6. Failure Mode Resilience

Five failure modes identified from abandoned projects [wave-05-abandoned-hetero-dispatch.md]:

| Failure Mode | Systems Vulnerable | libkdl Mitigation |
|-------------|-------------------|-------------------|
| FM-1: Single-vendor dependency | HSA, GPU Ocelot | Three independent backends (CUDA, HIP, CPU) |
| FM-2: Proprietary ABI fragility | GPU Ocelot (PTX interception) | Dispatches via stable vendor driver APIs, not binary interception |
| FM-3: Performance underdelivery | OpenCL, chipStar (0.75x) | Routes to vendor-native binaries; no translation overhead |
| FM-4: Semantic fragmentation | SPIR-V (Kernel vs GLCompute) | Pre-compiled vendor-native variants; SPIR-V as fallback only |
| FM-5: Zero demonstrated usage | GCC HSA, Intel MIC | Prototype verified on GTX 1650 + CPU; targets PyTorch ecosystem |

### 7. LLVM Community Alignment

| System | LLVM Upstream Status | Community Signal |
|--------|---------------------|-----------------|
| liboffload | Mainline (llvm/offload) | Biweekly meetings, active RFC process |
| UR | Intel/LLVM fork → bridging to liboffload | SYCL upstreaming (Aug 2025) |
| IREE HAL | Separate project (Google) | MLIR-adjacent, not LLVM mainline |
| chipStar | External (Argonne/Intel) | Aurora production backing |
| **libkdl** | **Not yet proposed** | **Issue #75356 validates the gap; PR #186088 defers the policy; Huber DevMtg 2025 uses "ld.so for GPU code" metaphor** |

libkdl's strongest community alignment signal: the LLVM project's own issue tracker (#75356), its own offload roadmap RFC, and its own DevMtg keynote speaker all independently identify the exact gap libkdl fills.

---

## Key Differentiators Summary

libkdl occupies a unique position in this landscape along three axes:

1. **Pre-compiled + cross-vendor + per-kernel**: No other system combines all three. AOTriton has pre-compiled + per-kernel but is AMD-only. IREE HAL has cross-vendor + pre-compiled but dispatches at module granularity with static selection.

2. **Policy layer, not mechanism layer**: libkdl does not compete with liboffload, UR, or vendor driver APIs. It sits above them, providing the missing selection policy that PR #186088 explicitly defers.

3. **Zero JIT cold start**: In an ecosystem where PT2/Triton takes 843s and chipStar takes 40 minutes for uncached workloads, libkdl's <5ms first-dispatch latency is a qualitative difference, not just a quantitative one.
