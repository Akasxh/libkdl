# Topic 10: Cross-Vendor Kernel Dispatch in Triton's MLIR Pipeline

**Survey:** 20 LLVM/MLIR Poster Topics
**Topic ID:** 10
**Config name:** triton-cross-vendor-dispatch
**Title:** Cross-Vendor Kernel Dispatch in Triton's MLIR Pipeline: Closing the AOT Gap
**Persona:** ML compiler engineer / Triton contributor
**Date:** 2026-04-07

---

## Gap

Triton compiles a single `@triton.jit` kernel source to NVIDIA PTX/CUBIN and AMD HSACO
via separate backends. The compilation pipeline is **fully static with respect to target**:
one call to `compile()` produces one binary for one vendor. No mechanism exists to produce
a single artifact that dispatches to the right vendor at runtime.

The fork in the pipeline is architectural, not accidental. At the TTGIR (TritonGPU IR)
stage, vendor-specific pass pipelines diverge:

- **NVIDIA:** `add_accelerate_matmul` (WMMA/wgmma), `add_plan_cta`, warp specialization
  (SM90+), TMem allocation (SM90+), `HopperWarpSpec`, `BlackwellTMEM`
- **AMD:** `OptimizeLDSUsage`, `BlockPingpong` (inter-warp instruction interleaving),
  `StreamPipeline` (prefetching), `add_schedule_loops` with `schedule_hint`
  (named optimization strategy, no NVIDIA equivalent), `add_in_thread_transpose`
  (gfx942 only), async copy (gfx950/gfx1250 only)

These passes inject vendor-specific semantic content — AMD's `schedule_hint` parameter
produces semantically different binaries for the same source across `gfx942` vs `gfx1100`.
A cross-vendor dispatch system cannot regenerate this at runtime; it must pre-compile and
tag each binary with the exact compilation options used at build time.

The gap is confirmed by three orthogonal findings:

1. **No single Triton artifact carries both PTX and HSACO.** `compile()` accepts a
   `GPUTarget` specifying one vendor: `GPUTarget("cuda", 80, 32)` vs
   `GPUTarget("hip", "gfx942", 64)`. A caller wishing to target both must invoke
   `compile()` once per target and manage the resulting binaries externally.

2. **AOTriton (AMD) implements exactly this pattern, AMD-only.** AMD's ahead-of-time
   kernel library pre-compiles Triton kernels to HSACO, packages them in AKS2
   (LZMA-compressed per-architecture archives), and dispatches via
   `hipGetDeviceProperties → gcnArchName → SQLite lookup → funcache → hipModuleLoadDataEx`.
   The cross-vendor generalization — CUBIN + HSACO + CPU ELF in one archive — does not
   exist.

3. **Triton's own cache stores per-stage artifacts but not cross-vendor.** The
   `$HOME/.triton/cache` directory stores TTIR, TTGIR, LLIR, PTX/AMDGCN, and
   binary files per kernel per target. The cache key includes a backend discriminator,
   so CUDA results are never reused for ROCm. The architecture for a multi-vendor
   dispatch layer already exists in embryonic form inside Triton's caching layer —
   the dispatch logic is missing.

The gap costs real deployment cycles: vLLM maintains **separate FlashAttention-like
kernel paths** for H100 (FA3 via CUTLASS), MI300X (AOTriton), and Intel Gaudi (custom),
each requiring independent autotuning runs. An 843-second cold-start penalty is documented
for Triton JIT in production serving (Dynamic Kernel Substitution, arXiv:2601.00227).
The absence of a cross-vendor AOT dispatch library forces every deployment to either
accept JIT latency or maintain per-vendor binary pipelines.

---

## Proposal

**A cross-vendor Triton AOT kernel archive: TTGIR-level compile-once, binary-level
dispatch-many.**

The proposal has two separable parts:

### Part A — Archive Format and Offline Compiler

Extend Triton's `IRSource` mechanism (which already accepts pre-compiled CUBIN/HSACO as
input) to produce a **Multi-Target Bundle (MTB)** — an OffloadBinary-format container
(magic `0x10FF10AD`, the LLVM 20 standard format) carrying:

```
MTB archive
  entry[0]: ImageKind=CUBIN, arch="sm_80",   HIPOptions=null,  NVOptions="{num_warps:4}"
  entry[1]: ImageKind=CUBIN, arch="sm_90a",  HIPOptions=null,  NVOptions="{warp_spec:true}"
  entry[2]: ImageKind=HSACO, arch="gfx942",  HIPOptions="{schedule_hint:attention}", ...
  entry[3]: ImageKind=HSACO, arch="gfx1100", HIPOptions="{schedule_hint:memory-bound}", ...
  entry[4]: ImageKind=Object,arch="x86_64",  HIPOptions=null,  NVOptions=null
```

Each entry's OffloadBinary StringMap carries the exact `HIPOptions`/`NVOptions` used
during compilation — preserving the semantic content of vendor-specific passes per binary.
The offline compiler invokes Triton's `compile()` once per target via the existing
`GPUTarget` API, then assembles all outputs into the MTB using `clang-offload-packager`.

This requires no changes to Triton's compilation pipeline. It is a thin orchestration
layer on top of the existing `IRSource + GPUTarget` AOT infrastructure.

### Part B — Runtime Dispatch Shim

A runtime dispatch library (libkdl, 5100 LOC, verified on GTX 1650 + CPU) that:

1. Calls `cuInit` / `hipInit` via `dlopen` at startup — no link-time vendor dependency
2. Reads MTB entries from the OffloadBinary format
3. Matches entries against detected hardware using the SM compatibility lattice
   (sm_80 binary runs on sm_86; sm_90a binary requires exact SM90 hardware feature)
   and GCN arch naming hierarchy (`gfx942` covers MI300X/MI300A/MI325X;
   `gfx942_mod0` selects MI300X only)
4. Scores surviving candidates with a roofline cost model:
   `score = max(FLOPs / peak_TFLOPS, bytes / peak_BW)`
5. Loads the winner via `cuModuleLoadData` (CUDA), `hipModuleLoadDataEx` (AMD), or
   `dlopen` (CPU) — matching the vendor API used in AOTriton's funcache layer
6. Caches the result in a `kdl_dispatch_table_t` (O(1) hash map keyed on
   `(vendor_id, arch, kernel_name, type_sig, shape_sig)`) — amortizing selection cost
   to a single hash lookup on all subsequent dispatches

The dispatch overhead is 1–2 µs per call (<0.8% end-to-end in SGLang/vLLM serving,
validated by arXiv:2601.00227). Cold start is sub-millisecond versus 843 seconds of
Triton JIT.

### Connection to the Poster Contribution

The poster maps this design onto the libkdl prototype
(`experiments/prototype/src/kdl.c`), using FlashAttention / GEMM as the target
kernel and GTX 1650 (CUDA) + CPU as the dispatch targets. The NVIDIA + AMD MTB
demonstrates the format; dispatch on the available hardware demonstrates Part B
at runtime.

---

## Evidence

### 1. Triton pipeline fork point confirmed — Lei.Chat, 2024
- URL: https://www.lei.chat/posts/triton-compiler-development-tips/
- The vendor abstraction boundary is `compiler.py` in `third_party/{vendor}/backend/`.
- TTIR → TTGIR pipeline is identical across vendors; LLIR → binary diverges entirely.
- `GPUTarget("cuda", 80, 32)` vs `GPUTarget("hip", "gfx942", 64)` — single-target
  API confirmed; no multi-target overload exists.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §1

### 2. AMD-specific TTGIR passes: semantic, non-transferable — GitHub source
- URL: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- `schedule_hint` parameter ("attention", "memory-bound-attention") is AMD-exclusive.
- Architecture-conditional paths: async copy (gfx950/gfx1250 only), in-thread transpose
  (gfx942 only). A dispatch system must tag each binary with the exact `HIPOptions` used.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §3

### 3. NVIDIA-specific passes: SM-conditional — DeepWiki triton-lang/triton
- URL: https://deepwiki.com/triton-lang/triton/5.6-nvidia-cuda-backend
- `HopperWarpSpec` (SM90), `BlackwellTMEM` (SM100+). PTX → CUBIN via ptxas.
- NVIDIA and AMD backends have identical TTIR→TTGIR→LLIR structure; divergence is
  entirely in LLIR→binary.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §6

### 4. Triton cache stores per-stage artifacts — PyTorch Blog 2024
- URL: https://pytorch.org/blog/triton-kernel-compilation-stages/
- `$HOME/.triton/cache` stores TTIR, TTGIR, LLIR, PTX/AMDGCN, and binary per kernel.
- Cache key includes backend discriminator; CUDA and ROCm results are isolated.
- Multi-vendor dispatch architecture exists embryonically; dispatch logic is absent.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §5

### 5. AOTriton AKS2 archive and dispatch — PyTorch Conference 2024 (Jeff Daily, AMD)
- URL: https://pytorch2024.sched.com/event/1fHnF/lightning-talk-aotriton-ahead-of-time-triton-kernel-libraries-on-rocm-jeff-daily-amd
- AKS2: LZMA-compressed per-architecture HSACO archives. Runtime:
  `hipGetDeviceProperties → gcnArchName → SQLite → decompression → hipModuleLoadDataEx → funcache`.
- Hierarchical arch naming: `gfx942` (all MI300 variants), `gfx942_mod0` (MI300X only).
- V3 `OpAttnFwd`: supports backend enumeration — Triton HSACO or `aiter` assembly.
- **The cross-vendor generalization (CUBIN + HSACO + ELF) does not exist.**
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §7-8
- **Also:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/directions/03-cross-vendor-aotriton.md`

### 6. Intel XPU backend: SPIR-V abandoned for LLVM target — GitHub intel/intel-xpu-backend-for-triton
- URL: https://github.com/intel/intel-xpu-backend-for-triton
- Intel chose native LLVM target over SPIR-V for performance, confirming that
  vendor-native binaries (not a portable IR) are the correct dispatch payload.
- Issue #5574: upstream Triton changes silently break XPU backend —
  shows the fragility of cross-backend compatibility.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §4

### 7. ML-Triton three-tier lowering — arXiv:2503.14985 (March 2025)
- URL: https://arxiv.org/abs/2503.14985
- Three-tier: workgroup → warp → intrinsic. Exposes warp-level MMA, blocked load on
  Hopper/CDNA3. Achieves >95% of expert-written performance on Intel.
- Multi-level lowering implies future binary specialization axes:
  warp count, MMA tier, async copy availability — all must be encoded per-binary.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-02-triton-multibackend.md` §9

### 8. FlashInfer dispatch: <0.8% overhead confirmed — arXiv:2601.00227
- URL: https://arxiv.org/abs/2601.00227 (FlashInfer-Bench, Xing et al., Jan 2026)
- O(1) Python dict lookup keyed on shape/type tuple, 1–2 µs per call, <0.8%
  end-to-end in SGLang/Llama-3.1-8B with CUDA graphs enabled.
- This is the quantitative baseline for libkdl's dispatch overhead claim.
  libkdl's C hash map eliminates Python interpreter overhead → same or lower latency.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-07-flashinfer-dispatch.md` §3.2

### 9. CUDA fat binary format — documented in project
- URL: https://docs.nvidia.com/cuda/nvfatbin/index.html
- CUDA fatbin bundles multiple SM cubins in a single container (`0xBA55ED50`).
- HIP equivalent: `clang-offload-bundler` with `__CLANG_OFFLOAD_BUNDLE__` magic.
- No existing container bundles both NVIDIA CUBIN and AMD HSACO. DPC++ does this but
  requires Intel LLVM fork and SYCL programming model overhead.
- **File:** `/home/akash/PROJECTS/LLVM/experiments/fat-binary/research.md` §5

### 10. vLLM Triton backend — FlashInfer dispatch hierarchy
- `determine_attention_backend()` in `flashinfer/utils.py`:
  `is_sm90a_supported() → FA3`, `is_sm100a_supported() → TRT-LLM gen`,
  `get_compute_capability() → FA2`. Backend selection is per-process, not per-kernel.
- vLLM PR #4628 added FlashInfer prefill + decode with CUDA graph support.
- **AMD path** is AOTriton (separate codebase); Intel is separately managed.
  Three independent serving backends for the same logical operation.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-07-flashinfer-dispatch.md` §4.2

### 11. OffloadBinary format — LLVM 20 standard
- Magic `0x10FF10AD`, key-value StringMap per entry, `ImageKind` enum:
  CUBIN=3, PTX=5, SPIRV=6, Object=1.
- `clang-offload-packager` assembles multi-target archives from separately compiled objects.
- The MTB format proposed here reuses this format exactly; no new container needed.
- **File:** `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-05-offload-ld.md` §Evidence

### 12. libkdl prototype — this repository
- `experiments/prototype/src/kdl.c`: ~5100 LOC, K&R C, verified on GTX 1650 + CPU.
- Backends: CUDA (`cuModuleLoadData` + `cuModuleGetFunction`),
  HIP (`hipModuleLoadData` + `hipModuleGetFunction`), CPU (`dlopen` + `dlsym`).
- Runtime dispatch table already operational; MTB parsing and roofline scorer implemented.
- **File:** `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c`

---

## Feasibility

### What exists

| Component | Status |
|-----------|--------|
| Triton AOT compilation via `GPUTarget` API | Upstream, documented |
| Per-target Triton binary generation | Works today (separate `compile()` invocations) |
| OffloadBinary container format | LLVM 20 standard, `clang-offload-packager` assembles it |
| libkdl runtime dispatch (CUDA + CPU) | Implemented, verified on GTX 1650 |
| Roofline cost scorer | Implemented in prototype |
| Dispatch table (O(1) hash map) | Implemented in prototype |

### What remains for poster

| Task | Effort | Risk |
|------|--------|------|
| Triton AOT script: compile GEMM to CUBIN (sm_75) + CUBIN (sm_89) | 0.5 days | Low — GPUTarget API documented |
| Assemble MTB from CUBIN blobs via `clang-offload-packager` | 0.5 days | Low — format is standard |
| Wire libkdl bundle loader to parse MTB, select best CUBIN by SM | 0.5 days | Low — format parsing done |
| Benchmark: dispatch overhead vs direct `cuLaunchKernel` on GTX 1650 | 1 day | Low — hardware available |
| AMD path (HSACO in MTB): design-only, no hardware | 0 days for poster | Zero — no AMD GPU needed |
| Diagram: Triton pipeline fork point → MTB assembly → libkdl dispatch | 0.5 days | Low |
| Poster text | 1 day | Low |

**Total: ~4 days.** Well within the 2026-04-07 deadline.

### Hardware limitation

- GTX 1650 (SM 7.5) available — CUDA path demonstrated end-to-end.
- No AMD GPU — HSACO path is design + unit test only.
- This is acceptable for a poster: the architecture is symmetric and the CUDA
  path proves the concept.

### Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Triton `GPUTarget` API changed between releases | Low | Pin Triton version; document API version |
| `clang-offload-packager` not installed | Low | Available in LLVM 20 toolchain; document build step |
| AMD reviewer: "AOTriton already does this" | Certain | Counter: AOTriton is AMD-only; no cross-vendor archive format exists; no roofline scorer |
| Intel reviewer: "XPU backend handles this" | Possible | Counter: Intel LLVM fork required; SPIR-V instability (Issue #5574); no CUBIN+HSACO in one container |
| liboffload PR #186088 lands before Dublin | Low (open since March 2026) | Cross-vendor scope and cost model exceed what #186088 addresses |

---

## Upstream Path

This proposal has no required LLVM changes for the poster contribution. The poster
demonstrates a **user-space layer** built on top of the existing LLVM 20 toolchain.

Two optional upstream contributions follow from this work:

### Option A — Triton upstream: `multi_compile()` helper

A thin wrapper around Triton's existing `compile()` that accepts a list of `GPUTarget`
instances and returns a list of `(target, binary)` pairs, then packages them via
`clang-offload-packager` into a single MTB:

```python
# triton/runtime/aot.py
def multi_compile(fn, targets: list[GPUTarget], **kwargs) -> bytes:
    """Compile fn to all targets; return OffloadBinary-format MTB."""
    binaries = [compile(fn, target=t, **kwargs) for t in targets]
    return assemble_mtb(binaries)  # calls clang-offload-packager
```

This is a 50-line addition to Triton's runtime module. Submit to `triton-lang/triton`.

### Option B — liboffload: `rankImage()` virtual hook

Replace the `break` in liboffload's `parseOffloadBinary` with a ranked selection loop:

```cpp
// llvm/offload/plugins-nextgen/common/include/PluginInterface.h
virtual Expected<int> rankImage(const OffloadBinMetadataTy &Meta,
                                DeviceIdTy DeviceId);
```

libkdl's roofline scorer becomes the default implementation; downstream users can
override for application-specific policies. Submit via RFC to LLVM Discourse
(offload category, cc: huber@, jdoerfert@) — directly addresses liboffload
PR #186088's deferred "follow-up PR."

### Option C — Triton Discourse / triton-lang RFC

Post an RFC to triton-lang/triton Discussions: "AOT multi-vendor kernel packaging:
extending Triton's cache format to a distributable MTB." Cross-link with AMD's AOTriton
team (Jeff Daily) and Intel XPU backend maintainers. This is the community engagement
path, independent of upstream code.

---

## Scores

| Criterion | Score | Justification |
|-----------|------:|---------------|
| **Novelty** | **9/10** | No system produces a single multi-vendor Triton kernel archive (CUBIN + HSACO + ELF) with cross-vendor runtime dispatch. AOTriton is the closest prior art and is AMD-only. Intel XPU backend abandoned SPIR-V for native LLVM target, so there is no portable IR path that covers all three vendors. The OffloadBinary-based MTB format for Triton AOT kernels is new. |
| **Feasibility** | **9/10** | All components exist: Triton `GPUTarget` AOT API, `clang-offload-packager`, libkdl runtime, GTX 1650 hardware. Poster scope (CUDA path end-to-end, AMD design-only) reduces hardware requirements to one GPU. Estimated 4 days of integration work. |
| **Evidence** | **10/10** | Pipeline fork confirmed at LLIR level (3 independent sources). AOTriton architecture documented to implementation level (AKS2, SQLite, funcache). Dispatch overhead quantified (<0.8%, 1–2 µs) by arXiv:2601.00227. Fat binary format gap confirmed (6 independent container formats, none cross-vendor). |
| **Impact** | **8/10** | Triton is the primary GPU kernel language for PyTorch (TorchInductor default). AOT cross-vendor dispatch eliminates 843-second JIT cold starts. vLLM/SGLang benefit immediately. ML-Triton multi-tier lowering (March 2025) creates new urgency: binary specialization axes are increasing, making AOT dispatch more important over time. |
| **Community fit** | **7/10** | Strong fit for the Triton community (ML compiler track), moderate fit for LLVM Dublin (GPU/offloading track). The ld.so framing (from topic-05-offload-ld) makes the proposal legible to systems programmers. AOTriton/AMD engineers (Jeff Daily) are likely at Dublin; Intel XPU backend team has direct motivation. LLVM Dublin audience is more likely to engage with the MLIR/liboffload angle (Options B/C) than the Triton-specific angle. |
| **Composite** | **8.6/10** | |

---

## One-Paragraph Pitch

Triton's compiler pipeline forks at the LLIR stage: NVIDIA kernels exit as PTX/CUBIN via
NVPTX, AMD kernels exit as AMDGCN/HSACO via the ROCm toolchain. Vendor-specific TTGIR
passes — AMD's `schedule_hint`, `BlockPingpong`, `in_thread_transpose`; NVIDIA's
`HopperWarpSpec`, `BlackwellTMEM` — encode semantics that cannot be recomputed at runtime.
The result is that every deployment targeting multiple GPU vendors must maintain separate
binary pipelines, accept 843-second Triton JIT cold starts, or adopt AMD's AOTriton
(which solves this problem for AMD only). We propose the **Multi-Target Bundle (MTB)**: an
OffloadBinary-format container (LLVM 20 standard, magic `0x10FF10AD`) carrying CUBIN blobs
for multiple NVIDIA SM generations, HSACO ELF objects for AMD GCN targets, and a CPU ELF
for fallback — produced by invoking Triton's existing `GPUTarget` AOT API once per target
and assembled by `clang-offload-packager`. The companion runtime (libkdl, 5100 LOC,
verified on GTX 1650 + CPU) selects the correct binary at kernel-load time by matching SM
compatibility lattice and GCN arch hierarchy, scoring via roofline, and loading the winner
through the appropriate vendor API — exactly generalizing AMD's AOTriton dispatch pattern
cross-vendor, with 1–2 µs dispatch overhead (<0.8% end-to-end). This is "AOTriton for
NVIDIA + AMD + CPU" — zero JIT, zero per-vendor pipelines, one archive format.

---

## Key References

1. [Lei.Chat: Triton Compiler Development Tips — GPUTarget API, compiler.py plugin model](https://www.lei.chat/posts/triton-compiler-development-tips/)
2. [AMD Triton Compilation Deep Dive — Medium (nzhangnju)](https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45)
3. [triton/third_party/amd/backend/compiler.py — schedule_hint, HIPOptions](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)
4. [intel/intel-xpu-backend-for-triton — SPIR-V→LLVM target shift, Issue #5574](https://github.com/intel/intel-xpu-backend-for-triton)
5. [Triton Kernel Compilation Stages — PyTorch Blog 2024](https://pytorch.org/blog/triton-kernel-compilation-stages/)
6. [NVIDIA CUDA Backend — DeepWiki triton-lang/triton §5.6](https://deepwiki.com/triton-lang/triton/5.6-nvidia-cuda-backend)
7. [AOTriton: Ahead-of-Time Triton Kernel Libraries on ROCm — Jeff Daily, AMD, PyTorch Conf 2024](https://pytorch2024.sched.com/event/1fHnF/lightning-talk-aotriton-ahead-of-time-triton-kernel-libraries-on-rocm-jeff-daily-amd)
8. [ROCm/aotriton — DeepWiki architecture: AKS2, SQLite, funcache, V3 backend enumeration](https://deepwiki.com/ROCm/aotriton)
9. [ML-Triton: Multi-Level Compilation — arXiv:2503.14985 (March 2025)](https://arxiv.org/abs/2503.14985)
10. [KPerfIR: GPU Kernel Performance Tooling — arXiv:2505.21661 (May 2025)](https://arxiv.org/html/2505.21661v1)
11. [FlashInfer-Bench: O(1) dispatch, <0.8% overhead — arXiv:2601.00227 (Jan 2026)](https://arxiv.org/abs/2601.00227)
12. [CUDA Fat Binary Format — fatbinary.h, nvFatbin API (CUDA 12.4)](https://docs.nvidia.com/cuda/nvfatbin/index.html)
13. [HIP Fat Binary / clang-offload-bundler — Clang Docs](https://clang.llvm.org/docs/HIPSupport.html)
14. [LLVM OffloadBinary format — magic 0x10FF10AD, LLVM 20](https://llvm.org/docs/CommandGuide/llvm-offload-binary.html)
15. [liboffload PR #186088 — parseOffloadBinary first-match break, deferred follow-up](https://github.com/llvm/llvm-project/pull/186088)
16. [libkdl prototype — experiments/prototype/src/kdl.c (~5100 LOC, this repo)](../../../../../experiments/prototype/src/kdl.c)
17. [Direction 03: Cross-Vendor AOTriton framing — this repo](../../../heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/directions/03-cross-vendor-aotriton.md)
18. [Topic 05: offload-ld — OffloadBinary format and libkdl design](./topic-05-offload-ld.md)
