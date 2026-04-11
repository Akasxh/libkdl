# Wave 03 — Dynamic Kernel Substitution Systems

**Angle:** dynamic-kernel-substitution-systems
**Query:** dynamic kernel substitution hot-swap runtime dispatch table O(1) GPU interposition NVBit
**Date:** 2026-04-06

---

## Summary

Runtime kernel interposition and substitution is an active, multi-layered area spanning binary instrumentation (NVBit, Luthier, gpu_ext), CUDA/ROCm API-level hooking (LD_PRELOAD, HSA InterceptQueue), and application-level O(1) dispatch (FlashInfer-Bench, HetGPU). The dominant cost of any dispatch indirection is the host-side CPU overhead (~4.5–10 μs hardware floor per launch), not the lookup itself — a well-implemented O(1) table adds only 1–2 μs and less than 0.8% end-to-end overhead. The primary risk for libkdl-style systems is the binary instrumentation path: NVBit's SASS rewriting carries 85–93% overhead, while the newer gpu_ext eBPF/trampoline approach reduces this to 3–14%. ROCm provides a first-class `InterceptQueue` mechanism in ROCR for packet-level kernel substitution with no publicly reported overhead floor, making it architecturally cleaner than the CUDA LD_PRELOAD path.

---

## Sources

### Source 1 — FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems

- **URL:** https://arxiv.org/abs/2601.00227
- **Date:** January 2026 (arXiv preprint)
- **Type:** Paper (ML systems / kernel dispatch)
- **Relevance:** 10/10 — Novelty: 9/10
- **Summary:** Introduces `flashinfer_bench.apply()`, a dynamic kernel substitution system for LLM inference that performs O(1) feature-key lookup over a prebuilt index to select optimal kernel implementations at runtime. The system supports both ahead-of-time and just-in-time compilation paths and integrates with SGLang and vLLM.
- **Key Technical Details:**
  - Dispatch mechanism: runtime constructs a shape/type feature key from input arguments, performs an **O(1) hash lookup** in a prebuilt kernel index
  - Per-kernel dispatch overhead: **1–2 microseconds**
  - End-to-end overhead when substituting identical kernels: **less than 0.8%** across all batch sizes (CUDA graph optimized, warmed up)
  - Fallback path: if no matching solution found, original function executes unchanged
  - Offline phase: AOT compiles high-frequency kernel variants; low-frequency variants JIT-compiled on first encounter
  - Dataset: 1,600 real LLM serving workloads across GEMM, attention, MoE, normalization kernel types
- **Connection to libkdl:** Direct empirical validation that O(1) dispatch table lookup adds negligible overhead vs direct kernel call — the core claim of the libkdl architecture.

---

### Source 2 — NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs

- **URL:** https://github.com/nvlabs/nvbit (code); https://dl.acm.org/doi/10.1145/3352460.3358307 (paper, MICRO 2019)
- **Date:** October 2019 (MICRO); actively maintained through 2025
- **Type:** Tool + Paper (systems / GPU binary instrumentation)
- **Relevance:** 9/10 — Novelty: 7/10 (foundational, well-known)
- **Summary:** NVBit uses SASS-level dynamic recompilation to inject instrumentation callbacks before or after arbitrary GPU instructions without recompilation of target applications. Uses `CUDA_INJECTION64_PATH` or `LD_PRELOAD` for injection; intercepts kernel load events to rewrite SASS in-place.
- **Key Technical Details:**
  - Injection entry point: functions intercepted "as they are loaded for the first time in the GPU" — one-time rewrite, persistent thereafter
  - Overhead: strongly workload-dependent due to per-thread state save/restore
    - Basic block counting: **2–5x** slowdown
    - Instruction counting / opcode histogram: **20–100x** slowdown
    - Memory tracing / register recording: **100–1000x** slowdown
  - Does not support AMD/Intel GPUs — NVIDIA-only (SASS is vendor-specific)
  - Critical CUDA 12.8+ note: tools must link with `g++` not `nvcc` to avoid device-linking conflicts
- **Limitation for libkdl:** NVBit is a profiling/tracing tool, not a substitution framework. High overhead from SIMT divergence in per-thread instrumentation callbacks makes it unsuitable as a production dispatch layer.

---

### Source 3 — gpu_ext: Extensible OS Policies for GPUs via eBPF

- **URL:** https://arxiv.org/abs/2512.12615 (arXiv, December 2025)
- **Date:** December 2025 (v1: Dec 14; v2: Dec 20, 2025)
- **Type:** Paper (systems / GPU OS / eBPF)
- **Relevance:** 10/10 — Novelty: 10/10
- **Summary:** gpu_ext treats the GPU driver and device as a programmable OS subsystem via eBPF. It dynamically intercepts CUDA runtime APIs to extract GPU kernel PTX, rewrites it with binary trampolines, and loads instrumented kernels back without restarting applications. SIMT-aware warp-leader execution pattern reduces overhead dramatically vs NVBit.
- **Key Technical Details:**
  - Interception mechanism: CUDA runtime API hook → PTX extraction → trampoline injection at kernel entry, memory instructions, and execution boundaries
  - Trampoline design: two-phase per-warp — (1) lane-local computation (no divergence), (2) warp leader aggregates via `__ballot_sync`/`__shfl_sync` and executes policy once
  - **Overhead vs NVBit (Table 2 from paper):**
    | Tool | gpu_ext overhead | NVBit overhead |
    |------|-----------------|----------------|
    | kernelretsnoop | 8% | 85% |
    | threadhist | 3% | 87% |
    | launchlate | 14% | 93% |
  - System-level gains: up to **4.8x throughput improvement**, **2x tail latency reduction** across LLM inference, GNN training, and vector search workloads
  - Authors: Yusheng Zheng, Tong Yu, Yiwei Yang et al.
- **Connection to libkdl:** gpu_ext's PTX interception + trampoline pattern is the closest existing mechanism to what libkdl needs for transparent kernel substitution on NVIDIA hardware. The 3–14% overhead bracket is the realistic floor for NVIDIA-side interposition.

---

### Source 4 — Luthier: A Dynamic Binary Instrumentation Framework Targeting AMD GPUs

- **URL:** https://ieeexplore.ieee.org/document/11096405/ (ISPASS 2025); https://github.com/matinraayai/Luthier (code)
- **Date:** May 2025 (ISPASS 2025, Ghent, Belgium, pp. 137–149)
- **Type:** Paper + Tool (systems / AMD GPU binary instrumentation)
- **Relevance:** 9/10 — Novelty: 9/10
- **Summary:** Luthier is the AMD-side counterpart to NVBit — a dynamic binary instrumentation framework for AMD GPUs that integrates with the ROCm software stack (HIP, OpenMP, OpenCL, direct ROCm runtime). Supports analyzing code objects loaded on GPU at runtime and inserting calls to device instrumentation functions.
- **Key Technical Details:**
  - Supports: HIP, OpenMP, OpenCL, native ROCm runtime applications on Linux
  - Capabilities: kernel analysis, device function analysis, static variable inspection, runtime instrumentation function injection
  - Published overhead analysis: "kernel runtime overhead incurred when running instrumented versions of benchmarks" — specific numbers in IEEE paper (not fully extracted from abstract)
  - Authors: Matin Raayai Ardakani, Andrew Nguyen, Ivan Rosales, Daoxuan Xu, Yuwei Sun, Yifan Sun, David Kaeli, Norman Rubin
  - Artifact publicly available on Zenodo (DOI: 10.5281/zenodo.15027182)
- **Connection to libkdl:** Luthier is the AMD-native equivalent tool. For libkdl to support AMD transparently, Luthier's instrumentation hooks are the relevant prior art. The two together (NVBit/gpu_ext for NVIDIA, Luthier for AMD) frame the binary-level interposition design space.

---

### Source 5 — HetGPU: Binary Compatibility Towards GPUs

- **URL:** https://arxiv.org/html/2506.15993v1
- **Date:** June 2025 (arXiv)
- **Type:** Paper (systems / cross-vendor GPU portability)
- **Relevance:** 8/10 — Novelty: 8/10
- **Summary:** HetGPU defines a portable intermediate representation (hetIR) compiled once and JIT-translated to vendor-native code (SASS, GCN, SPIR-V) at kernel load time. Runtime maintains a kernel table with unique identifiers; on `hetgpuLaunchKernel()`, checks cache before triggering backend translation.
- **Key Technical Details:**
  - Dispatch: kernel ID → cached native handle lookup → launch (O(1) for warm path)
  - JIT compilation cost (cold path): **10–200 ms per kernel** (cached thereafter)
  - Runtime overhead (warm path):
    - Vector Add: 0.11–0.22 ms added
    - Matrix Multiply: **3–8% slowdown** vs native
    - Reduction: **5–15% overhead**
  - Migration cost: 2.2 s downtime for 30-second job with 2 GB data transfer
  - Backend targets: NVIDIA (PTX→SASS via CUDA driver JIT), AMD (SPIR-V→GCN via ROCm), Intel (SPIR-V via Level Zero), Tenstorrent (Metalium assembly)
- **Connection to libkdl:** HetGPU is the most direct prior-art for libkdl's architecture: portable IR + runtime dispatch table + vendor JIT backend. The 3–8% overhead on matrix operations sets a reasonable performance target.

---

### Source 6 — ROCm ROCR HSA Runtime: InterceptQueue Mechanism

- **URL:** https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/ (docs); https://deepwiki.com/ROCm/ROCR-Runtime (architecture reference)
- **Date:** Active (ROCm 6.x / 7.x, 2024–2025)
- **Type:** Documentation / SDK
- **Relevance:** 9/10 — Novelty: 6/10 (architectural reference, not a paper)
- **Summary:** ROCR provides a first-class `InterceptQueue` proxy type specifically designed for tooling-level kernel substitution. Wraps any hardware-backed AQL queue, allowing inspection and modification of kernel dispatch packets before doorbell signaling reaches hardware.
- **Key Technical Details:**
  - Architecture: AQL (Architected Queuing Language) ring buffers with write/read index pairs; doorbell notification to hardware CP
  - `InterceptQueue` API: `AddInterceptor()` registers a callback; `HandleAsyncDoorbell()` fires before packet reaches hardware
  - Packet-level substitution: intercept `HSA_PACKET_TYPE_KERNEL_DISPATCH` packets, modify kernel object handle or arguments, then forward
  - `CoreApiTable` / `AmdExtTable`: structured function pointer tables organizing HSA API — canonical dispatch table design
  - No application modification required
  - rocprofv2 uses this mechanism for profiling; scheduled end-of-support: 2026 Q2 (successor: rocprofiler-sdk)
- **Connection to libkdl:** InterceptQueue is the AMD-native architectural hook for a "GPU ld.so" — it provides packet-level kernel substitution at the AQL layer, which is exactly what libkdl's dispatch layer would do for AMD targets. Notably cleaner than CUDA's LD_PRELOAD path because it is a supported API primitive, not a hooking hack.

---

### Source 7 — GPU API Interposer (CUDA LD_PRELOAD via pyxis-roc)

- **URL:** https://github.com/pyxis-roc/gpu-api-interposer
- **Date:** Active (2020–2024)
- **Type:** Tool / Research artifact
- **Relevance:** 7/10 — Novelty: 5/10
- **Summary:** Auto-generates LD_PRELOAD interposers from CUDA header files using a generator tool, producing shared libraries that transparently intercept all CUDA Driver API (`libcuda`) and Runtime API (`libcudart`) calls without modifying target applications.
- **Key Technical Details:**
  - Mechanism: `LD_PRELOAD` shared library overriding symbols from `libcuda.so` and `libcudart.so`
  - Important limitation: **CUDA Runtime API symbols cannot be directly hooked** via LD_PRELOAD (symbols are bound at link time in some configurations); only the CUDA Driver API layer is reliably interceptable
  - Workaround: interpose at driver API level (`cuLaunchKernel`, `cuLaunchKernelEx`) which all runtime paths eventually call
  - Includes `Blobstore` (SQLite3) for logging intercepted binary data and `Arghelper` for parsing kernel launch arguments
  - No specific overhead numbers published
- **Connection to libkdl:** This is the simplest NVIDIA-side interposition path, but the Runtime/Driver API limitation is a known pitfall. libkdl should target `cuLaunchKernel` at the driver layer rather than `cudaLaunchKernel` at the runtime layer.

---

### Source 8 — Judging a Type by Its Pointer: Optimizing GPU Virtual Functions (ASPLOS 2021)

- **URL:** https://engineering.purdue.edu/tgrogers/papers/zhang.asplos2021.pdf
- **Date:** April 2021 (ASPLOS 2021)
- **Type:** Paper (architecture / GPU virtual dispatch)
- **Relevance:** 7/10 — Novelty: 7/10
- **Summary:** Analyzes the overhead of GPU virtual function dispatch through vtable indirection (the closest in-kernel analog to a dispatch table). Shows that 87% of virtual call cost comes from the vTable pointer load, not the branch. Proposes TypePointer encoding (type info in unused address bits) to eliminate the memory load.
- **Key Technical Details:**
  - vTable dispatch overhead: **87% of cost is the load to the vTable pointer** (L1/L2 cache miss)
  - TypePointer optimization: encodes type info in unused 64-bit pointer bits → eliminates memory load
  - Performance improvement: **90% over baseline CUDA**, 56% over prior work, 12% over SharedOA
  - On NVIDIA V100, measured via PC sampling across GPU-enabled implementations
- **Connection to libkdl:** Direct relevance to the dispatch table design. If libkdl uses an in-kernel function pointer table (device-side dispatch), the vTable load cost is measurable and can be eliminated via TypePointer-style encoding. For host-side dispatch tables, the memory access pattern matters less (CPU cache is fast), but the same principle applies: avoid cache misses in the hot dispatch path.

---

### Source 9 — eGPU: Extending eBPF Programmability and Observability to GPUs

- **URL:** https://dl.acm.org/doi/10.1145/3723851.3726984 (HCDS '25); https://github.com/eunomia-bpf/eGPU
- **Date:** March 2025 (HCDS Workshop at EuroSys 2025, Rotterdam)
- **Type:** Workshop paper + Tool
- **Relevance:** 8/10 — Novelty: 9/10
- **Summary:** eGPU is the first framework to dynamically offload eBPF bytecode onto GPUs via dynamic PTX injection, enabling GPU-side observability and policy enforcement. Runs eBPF programs inside GPU kernels by attaching to application-level function calls or GPU-side hook points.
- **Key Technical Details:**
  - Dynamic PTX injection: eBPF bytecode is JIT-compiled and injected into PTX before SASS compilation
  - Attach points: application-level function calls, GPU kernel entry/exit, arbitrary instruction sites
  - No application modification required; works at CUDA runtime level
  - Leverages SIMT-aware execution model (same insight as gpu_ext)
  - Published at HCDS '25; code merged into bpftime project
  - Complementary to gpu_ext (which focuses on OS policy); eGPU focuses on observability/tracing
- **Connection to libkdl:** eGPU demonstrates that GPU-side eBPF policy hooks are feasible with low overhead. For libkdl's tracing and fallback dispatch logic, a similar PTX-injection approach could embed lightweight dispatch decision code directly into GPU kernels.

---

## Angle Assessment

**Angle Richness:** HIGH — This angle uncovered 9 distinct sources spanning the full stack: binary rewriting (NVBit, gpu_ext, Luthier), API-level hooking (ROCR InterceptQueue, CUDA LD_PRELOAD), application-level O(1) dispatch (FlashInfer-Bench, HetGPU), in-kernel dispatch optimization (TypePointer), and device-side eBPF (eGPU).

**Key Overhead Numbers for libkdl:**

| Layer | Mechanism | Overhead |
|-------|-----------|----------|
| Application O(1) lookup | FlashInfer-Bench hash table | 1–2 μs per call, <0.8% e2e |
| Host-side dispatch floor | CUDA kernel launch | 4.5–10 μs (hardware floor) |
| Binary instrumentation (new) | gpu_ext PTX trampolines | 3–14% |
| Binary instrumentation (legacy) | NVBit SASS rewrite | 85–93% |
| Cross-vendor JIT warm path | HetGPU kernel table | 3–8% compute slowdown |
| In-kernel vtable dispatch | CUDA virtual functions | 87% cost = vTable load (eliminable) |

**Critical Design Recommendations for libkdl:**

1. **AMD path**: Use ROCR `InterceptQueue` + `AddInterceptor()` for packet-level substitution. This is a first-class supported API, not a hack, and avoids binary rewriting entirely.

2. **NVIDIA path**: Target `cuLaunchKernel` (driver API), not `cudaLaunchKernel` (runtime API). Use gpu_ext's PTX trampoline pattern for transparent substitution with 3–14% overhead rather than NVBit's 85–93%.

3. **Dispatch table design**: Host-side O(1) hash over (device_id, kernel_name, shape_signature) → `cuFunction*` / HSA kernel object. Avoid device-side vtable indirection unless necessary (87% of cost is the pointer load).

4. **JIT cold path**: Accept 10–200 ms first-call JIT cost (as HetGPU does) and make it visible via telemetry. Warm-path overhead is negligible.

**Gaps / Open Questions:**
- No published overhead numbers for ROCR `InterceptQueue` pass-through cost (suspected < 1 μs, but unconfirmed)
- Luthier's specific overhead numbers are behind the ISPASS paywall; Zenodo artifact has benchmark code
- No paper covers the combination of ROCR InterceptQueue + O(1) kernel index + fallback dispatch — this is the design space libkdl occupies
- NVBit has no AMD equivalent at the SASS level; Luthier operates at a higher abstraction (GCN ISA via ROCm CodeObject)

**Novelty Score for libkdl positioning:** 9/10 — combining InterceptQueue-style packet interception with an O(1) multi-vendor dispatch table and transparent fallback is not covered by any single existing paper. The closest prior art is HetGPU (cross-vendor dispatch) + FlashInfer-Bench (O(1) substitution) + gpu_ext (transparent interposition), but none integrates all three for a "ld.so for kernels" abstraction.

---

## Sources List

- [FlashInfer-Bench arXiv:2601.00227](https://arxiv.org/abs/2601.00227)
- [NVBit GitHub (NVlabs)](https://github.com/nvlabs/nvbit)
- [NVBit MICRO 2019 (ACM DL)](https://dl.acm.org/doi/10.1145/3352460.3358307)
- [gpu_ext arXiv:2512.12615](https://arxiv.org/abs/2512.12615)
- [gpu_ext HTML paper](https://arxiv.org/html/2512.12615)
- [Luthier ISPASS 2025 (IEEE Xplore)](https://ieeexplore.ieee.org/document/11096405/)
- [Luthier GitHub](https://github.com/matinraayai/Luthier)
- [Luthier Zenodo artifact](https://zenodo.org/records/15027182)
- [HetGPU arXiv:2506.15993](https://arxiv.org/html/2506.15993v1)
- [ROCR HSA Runtime documentation](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/)
- [ROCR DeepWiki architecture reference](https://deepwiki.com/ROCm/ROCR-Runtime)
- [gpu-api-interposer GitHub (pyxis-roc)](https://github.com/pyxis-roc/gpu-api-interposer)
- [TypePointer ASPLOS 2021 PDF](https://engineering.purdue.edu/tgrogers/papers/zhang.asplos2021.pdf)
- [eGPU ACM DL (HCDS '25)](https://dl.acm.org/doi/10.1145/3723851.3726984)
- [eGPU GitHub](https://github.com/eunomia-bpf/eGPU)
- [NVBit Tutorial (eunomia.dev)](https://eunomia.dev/others/nvbit-tutorial/)
- [FlashInfer-Bench HTML](https://arxiv.org/html/2601.00227v1)
