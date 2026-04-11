# Poster Pitch Script — mlir-hetero-dispatch
## LLVM Developers' Meeting, Dublin 2026

---

## 1. Two-Minute Elevator Pitch

Memorize the structure, not the exact words. Each block has a target duration.

---

### Hook (10 sec)

> "What if you could compile an ML kernel once and run it on any GPU — NVIDIA, AMD, or CPU
> — with near-native performance, and no recompilation?"

That is the problem we solved.

---

### Problem (20 sec)

> "Today, if you deploy an ML model to a cloud cluster, you need separate binaries for AWS
> NVIDIA A100s, Azure AMD MI300Xs, and CPU fallback nodes. MLIR's `gpu-module-to-binary`
> pass can actually compile to all three targets simultaneously — NVPTX, AMDGCN, x86 — but
> nothing in the stack decides *at runtime* which binary to run. `gpu.select_object` resolves
> at compile time. IREE acknowledges runtime variant selection as 'sort of broken' — it's been
> an open issue since 2019. There's a gap between MLIR's multi-target AOT compilation and
> actual runtime hardware dispatch, and nobody has closed it."

---

### Solution (30 sec)

> "We built `mlir-hetero-dispatch`: a two-part system. The build-time half uses the existing
> MLIR pass pipeline — `nvvm-attach-target`, `rocdl-attach-target`, `gpu-module-to-binary` —
> to compile a single `gpu.module` to NVPTX, AMDGCN, and x86 simultaneously, and packages
> the result into a Multi-Target Bundle with a routing table that maps kernel names to
> capability contracts.
>
> The runtime half is `libkdl` — the Kernel Dynamic Linker — about 500 lines of C. Think of
> it as `ld.so` for GPU kernels. On startup it dlopen-probes `libcuda.so.1` and
> `libamdhip64.so` with no link-time dependencies, discovers what hardware is present, matches
> kernel variants against their capability contracts — things like `sm >= 80`,
> `shared_mem >= 48KB` — and ranks candidates with a roofline cost model before dispatching.
> If the best GPU isn't available, it falls back down the chain to CPU."

---

### Results (20 sec)

> "The dispatch indirection itself adds less than 10 nanoseconds — against a CUDA kernel
> launch floor of 5 to 20 microseconds, that's under 0.05% overhead. For a multi-versioned
> SGEMM compiled with this approach, we land within 10% of the vendor theoretical max, which
> matches numbers from recent multi-versioning literature. The full runtime is around 500 lines
> of C — compared to IREE's 100K+ LOC full-stack buy-in. The key novelty is that this is the
> first MLIR-native dispatch layer with a cost model and no programming model requirement."

---

### Call to Action (10 sec)

> "We're working toward upstreaming the routing table pass and the MTB format into LLVM's
> existing offload binary infrastructure. If you're working on IREE, gpu dialect, or
> heterogeneous runtimes, I'd love to talk about where this fits. And if you're looking for
> a PhD project or internship at the intersection of ML compilers and systems, come find me."

---

### LLVM Audience Tailoring — Keep These Ready

- The extensibility point for a runtime-aware handler is `GPUOffloadingLLVMTranslationAttrInterface` — that is exactly where a production version of `libkdl` hooks in.
- The routing table pass (`kdl-routing-table-gen`) slots after `gpu-module-to-binary` in any existing pipeline.
- The MTB format is a prototype; the long-term target is LLVM's `llvm-offload-binary` infrastructure.
- The `gpu` dialect's `select_object` op is the right place to replace compile-time selection with a runtime call.

---

## 2. Anticipated Questions and Answers

---

### Q1: "How does this differ from IREE?"

IREE and `mlir-hetero-dispatch` are complementary, not competing. IREE provides a complete
end-to-end runtime — VM, HAL, FlatBuffer module format — at 100K+ LOC and requires full
ecosystem buy-in. What IREE explicitly does not do is lightweight runtime kernel variant
selection: Issue #12230 acknowledges that runtime strategy selection is "sort of broken" and
has been stalled since May 2023; Issue #15334 is a multi-versioning epic with every task
unchecked; Issue #50 has been open since 2019. `libkdl` fills that specific gap at ~500 LOC
and can slot into any existing MLIR pipeline without replacing the surrounding toolchain.

---

### Q2: "Why not just use SYCL or AdaptiveCpp?"

SYCL requires adopting the SYCL programming model — your kernels must be written in SYCL C++.
AdaptiveCpp's SSCP JIT mode achieves runtime dispatch but imposes ~15% overhead on first
launch and has no MLIR integration. Crucially, SYCL's performance portability in practice is
poor: Davis et al. (ICS 2025) measured P3 scores of 0.46--0.65 for SYCL versus 0.75--0.99
for Kokkos. `mlir-hetero-dispatch` requires no change to the kernel source — you write linalg
ops, the MLIR pipeline handles multi-target compilation, and `libkdl` dispatches at runtime.

---

### Q3: "What about tensor cores or other vendor-specific features?"

Capability contracts handle this directly. Each kernel variant declares its requirements in a
JSON contract embedded in the Multi-Target Bundle — for example, `requires_tensor_cores: true`
or `min_arch: sm_80`. During selection, `kdl_contract_matches()` filters out variants that
require hardware the current device does not provide. A kernel compiled with wmma intrinsics
for sm_80 simply will not be selected on a CPU or on a pre-Ampere GPU; the fallback chain
picks the next best variant. Vendor-specific features are a reason to have *more* variants in
the bundle, not a reason the dispatch model breaks.

---

### Q4: "How does the cost model work?"

It is a roofline model. Each capability contract includes a compute profile: FLOP count, bytes
read, bytes written, and arithmetic intensity. Given a device's peak TFLOP/s and peak memory
bandwidth, `kdl_estimate_cost()` computes `min(peak_compute, peak_bandwidth × arithmetic_intensity)`
to get an achieved throughput estimate, then divides FLOPs by that to get an estimated runtime.
Kernel launch overhead (20 µs for GPU, 1 µs for CPU) is added. The variant with the lowest
estimated cost wins. If no compute profile is provided, the cost model falls back to the
priority field in the routing table.

---

### Q5: "What's the overhead?"

Two numbers matter. The dispatch indirection through `ctx->backends[]` — the actual function
pointer call that replaces `cuLaunchKernel` — is under 10 nanoseconds. Cache-miss selection
(first call for a given kernel) takes around 200 microseconds, but this is amortized to zero
on all subsequent calls since results are cached keyed on kernel name and device index. The
irreducible floor for a CUDA kernel launch is 5--20 microseconds, so dispatch adds under
0.05% overhead relative to launch, and under 0.001% relative to a real ML kernel that runs
for 1--10 milliseconds.

---

### Q6: "Can this be upstreamed into MLIR?"

The routing table generation pass is a natural addition after `gpu-module-to-binary` and has
no dependencies outside the gpu dialect. The `GPUOffloadingLLVMTranslationAttrInterface` is
already designed as an extensibility point — a runtime-aware translation handler is exactly
what it was designed to accommodate. The MTB format is a prototype; production format would
align with `llvm-offload-binary`. The `libkdl` C library itself is intentionally standalone
with no LLVM dependencies, which makes it usable without the full LLVM build. The realistic
upstream path is: RFC on llvm-dev for the pass and format, `libkdl` as an out-of-tree runtime
initially.

---

### Q7: "How does it integrate with PyTorch?"

The integration sketch is a `torch.compile` backend registered via
`torch._dynamo.register_backend`. The backend receives a `torch.fx.GraphModule`, lowers it
through `torch-mlir` to linalg, runs the multi-target MLIR pipeline to produce an MTB bundle,
then returns a Python callable that calls `kdl_select_kernel` and `kdl_launch` on every
invocation. PyTorch's dispatcher already computes a `DispatchKeySet` on every operator call
from live tensor metadata — our dispatch fits naturally in that model, extending it to
hardware-level variant selection. The architecture sketch is in the ARCHITECTURE.md; a full
`torch.compile` backend is future work beyond this poster.

---

### Q8: "What about SPIR-V as a universal target?"

SPIR-V is a good idea with a performance ceiling. On NVIDIA hardware, Vulkan/SPIR-V trails
CUDA by roughly 20--30% for general LLM workloads (llama.cpp benchmarks). On AMD RDNA3,
Vulkan actually matches or exceeds ROCm/HIP in some workloads — so SPIR-V is viable there.
The fundamental problem is that vendor-specific features like CUDA tensor cores and AMD MFMA
instructions have no portable SPIR-V representation, so a SPIR-V-only target leaves
significant performance on the table. Our approach keeps SPIR-V as one possible variant in the
bundle — alongside native NVPTX and AMDGCN — and lets the cost model pick the right one per
device. SPIR-V is a valid fallback, not the universal solution.

---

### Q9: "Why not JIT instead of AOT?"

JIT and AOT are both valid; we chose AOT for two reasons. First, AOT compilation means
`libkdl` carries no compiler dependency at runtime — it is ~500 LOC C with no LLVM. Second,
the primary cost of JIT is first-call latency: AdaptiveCpp's SSCP JIT adds ~15% overhead on
first launch. AOT pushes that cost to build time, which is fine for deployment scenarios.
That said, a JIT variant of `libkdl` that calls back into MLIR to specialize for the exact
runtime device is a natural extension — the routing table and capability contracts are the
same; you just replace the pre-compiled blobs with an MLIR module and a JIT invocation. The
architecture document sketches this path.

---

### Q10: "How do you handle different memory models?"

The `kdl_device_info` struct captures `warp_size` (32 for NVIDIA, 64 for AMD CDNA, 1 for
CPU), `max_shared_mem`, and `vram_bytes`. Capability contracts can require a minimum shared
memory (`min_shared_mem_kb`) to filter out variants that rely on shared memory tiling that
will not fit. Warp size differences — one of the most common portability bugs — are handled
by compiling separate kernel variants for NVIDIA and AMD rather than trying to abstract over
them. The memory APIs (`kdl_malloc`, `kdl_memcpy_h2d`, `kdl_memcpy_d2h`) dispatch to
`cuMemAlloc` or `hipMalloc` through the same backend table, so memory operations are
consistent regardless of which GPU won selection.

---

### Q11: "What about graph-level dispatch versus kernel-level?"

`mlir-hetero-dispatch` operates at kernel level — a single `gpu.func` is the unit of
dispatch. Graph-level dispatch (routing an entire model graph to one accelerator vs. another)
is the problem ONNX Runtime's execution providers and Helix solve. Helix (ASPLOS 2025)
demonstrated 3.3x throughput improvement on heterogeneous clusters by routing entire inference
requests to optimal GPU types — but it does not address per-kernel variant selection within a
single device type. Our contribution is orthogonal: given that a request has been routed to a
specific node, `libkdl` ensures the right kernel binary runs on whatever GPU that node has.
Both levels of dispatch matter for production heterogeneous ML systems.

---

### Q12: "How is this different from ONNX Runtime Execution Providers?"

ONNX Runtime EPs dispatch at the operator level — each `MatMul` or `Conv` node is claimed by
a specific EP. But EPs are statically registered and compiled into the ORT binary; adding a
new hardware target requires writing a new EP in C++ and shipping a new ORT build. `libkdl`
operates below the EP layer, at the compiled kernel binary level. The ARCHITECTURE.md includes
a sketch of a `KdlExecutionProvider` that wraps `libkdl`: the EP's `Compile()` method runs
the multi-target MLIR pipeline and produces an MTB bundle, and `Compute()` calls
`kdl_select_kernel` + `kdl_launch`. The two systems are composable: ORT handles graph
partitioning, `libkdl` handles kernel variant selection.

---

### Q13: "What hardware have you tested on?"

The prototype targets SM 80 (A100-class) for NVPTX and x86-64 with AVX2 for the CPU
fallback. The `gpu-module-to-binary` pass has been validated for NVPTX and AMDGCN targets
(GFX90A / GFX942) at the MLIR level. The MTB bundle format and `libkdl` selection logic are
hardware-agnostic — they operate on metadata from `cuDeviceGetProperties` and
`hipGetDeviceProperties`. End-to-end dispatch on AMD hardware is the next benchmark target.
The poster presents dispatch overhead numbers from the NVIDIA path and cost model validation
against published roofline data.

---

### Q14: "What's the path to production?"

Three steps. First, the `kdl-routing-table-gen` pass and MTB format go upstream via an
llvm-dev RFC — this is the most composable piece and benefits the broader MLIR ecosystem
regardless of `libkdl`. Second, `libkdl` ships as an out-of-tree runtime library with a
stable C API, enabling integration with PyTorch, ONNX Runtime, and llama.cpp without
requiring LLVM build system changes. Third, a `GPUOffloadingLLVMTranslationAttrInterface`
implementation that calls `libkdl` at module load time replaces the current compile-time
`gpu.select_object` with a runtime-aware equivalent. The 500-LOC size is intentional — it
lowers the integration cost to the point where any project can vendor it.

---

### Q15: "How does this relate to your CERN GSoC work?"

At CERN I worked on ALPAKA, which is the compile-time portability side of this problem — same
kernel source, multiple backend targets, selected at CMake configuration time. ALPAKA achieves
>94% of native CUDA performance with zero runtime overhead, but requires recompilation per
target. That experience made the gap clear: ALPAKA solves portability for HPC centers where
you know the hardware at compile time, but fails for cloud deployment where hardware is
discovered at runtime. `mlir-hetero-dispatch` is the runtime complement — the same
compile-once / run-anywhere goal, achieved through a routing table and cost model instead of
C++ template metaprogramming.

---

## 3. Key Talking Points

Five things to have immediately available during any conversation:

---

**1. The gap is documented by IREE's own issues.**
IREE Issue #50 (open since October 2019), #12230 (runtime strategy selection "sort of
broken," stalled since May 2023), and #15334 (multi-versioning epic, every task unchecked).
This is not an academic gap — the most mature MLIR runtime explicitly acknowledges it.
Analogy: MLIR can bake all the breads, but has no waiter to bring the right loaf to the
right table.

---

**2. ML kernels are not static — cuBLAS, cuDNN, and PyTorch all dispatch at runtime.**
cuBLAS selects among hundreds of GEMM variants per precision using an ML-trained recommender.
cuDNN v9 has three heuristic modes plus NVRTC-compiled runtime fusion engines. PyTorch's
dispatcher computes a `DispatchKeySet` on every operator call from live tensor metadata.
The "ML is static" objection describes compilation graphs, not execution — execution has
always involved runtime selection.

---

**3. The `ld.so` analogy for non-experts.**
`libkdl` is to GPU kernels what the dynamic linker is to shared libraries. Just as `ld.so`
resolves `libmath.so` to the right file on your system at process start, `libkdl` resolves
`matmul.kernel` to the right binary (cubin, hsaco, or x86 ELF) based on what GPU is present.
The routing table is the `ldconfig` cache; capability contracts are `SONAME` version
requirements.

---

**4. Numbers to cite without looking them up.**
- Dispatch indirection: **< 10 ns**
- CUDA kernel launch floor: **5--20 µs** → dispatch overhead: **< 0.05%**
- `libkdl` size: **~ 500 LOC C**
- IREE size: **100K+ LOC**
- SYCL P3 portability scores: **0.46--0.65** (Kokkos: 0.75--0.99)
- Multi-versioned SGEMM vs. theoretical max: **within 10%**
- Helix mixed-GPU throughput gain: **3.3x**
- No published system performs runtime cross-vendor ML kernel dispatch from unified MLIR IR.

---

**5. The strongest evidence for novelty.**
The comparison table (findings.md §"Why This Is Novel") has six dimensions. The only system
with runtime dispatch + cross-vendor + MLIR-native + lightweight + no programming model
requirement + cost model is `mlir-hetero-dispatch`. IREE misses lightweight and cost model.
SYCL/AdaptiveCpp misses MLIR-native and requires the SYCL programming model. ALPAKA misses
runtime dispatch entirely. Proteus (CGO 2025) misses cross-vendor. Nobody has all six.

---

*Last updated: 2026-04-02. All numbers sourced from findings.md and ARCHITECTURE.md.*
