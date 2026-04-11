# AdaptiveCpp SSCP: Single-Source Single-Compiler-Pass JIT
## Literature Note — LLVM Dublin 2026

**System:** AdaptiveCpp (formerly hipSYCL / Open SYCL)
**Key Paper:** "Adaptivity in AdaptiveCpp: Optimizing Performance by Leveraging Runtime Information During JIT-Compilation"
**Venue:** IWOCL 2025 (13th International Workshop on OpenCL and SYCL)
**ACM DL:** https://dl.acm.org/doi/10.1145/3731125.3731127
**Docs:** https://adaptivecpp.github.io/AdaptiveCpp/compilation/
**Code:** https://github.com/AdaptiveCpp/AdaptiveCpp
**SSCP Architecture Docs:** https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/
**Relevance Score:** 10/10

---

## Finding

AdaptiveCpp's SSCP (Single-Source, Single Compiler Pass) pipeline is the **closest existing production system to libkdl's dispatch model**. It embeds backend-independent LLVM IR in the host binary at compile time, then JIT-compiles that IR to the detected GPU's native ISA at runtime. The 2025 IWOCL paper extends this with **adaptive kernel specialization**: the JIT uses runtime information (work-group sizes, pointer alignments, argument values) to generate specialized kernels, outperforming CUDA by **30% geometric mean**, HIP by **44%**, and oneAPI by **23%** across a set of mini-apps.

---

## SSCP Architecture: The Two-Stage Model

### Stage 1: Compile time (single pass)

AdaptiveCpp parses the source **once** for both host and device code. During host C++ compilation:

1. Clang processes the SYCL source (host + device in a single TU)
2. For kernel functions, the compiler extracts **backend-independent LLVM IR**
3. SYCL builtin calls (`get_global_id()`, `barrier()`, etc.) are lowered to abstract IR intrinsics — not to PTX or AMDGCN or SPIR-V
4. The resulting LLVM IR is **embedded into the host binary** (ELF section or equivalent)
5. Host code compilation continues normally; no separate device compilation pass

This is the key difference from DPC++:
- DPC++ runs a separate compilation for each target, producing target-specific code embedded in a fat binary
- AdaptiveCpp runs **one** compilation producing target-independent IR for lazy JIT

Compile-time overhead: **~15% over host-only compilation** (vs DPC++ multi-pass which can be 2x+ slower for multiple targets).

### Stage 2: Runtime JIT

When a kernel is first invoked on a given device:

1. AdaptiveCpp's runtime retrieves the embedded LLVM IR for the kernel
2. **Hardware detection:** queries the current device (CUDA device properties, HIP device capabilities, or SPIR-V target info via OpenCL)
3. Runs LLVM optimization passes on the IR with **target-specific features enabled** (tensor cores, subgroup size, memory hierarchy parameters)
4. Calls PTX (NVIDIA), AMDGCN (AMD), or SPIR-V (Intel) backend to generate native ISA binary
5. Loads and caches the binary on disk (persistent kernel cache)

On **subsequent runs**: the disk-cached binary is loaded directly — JIT cost is zero.

### The persistent kernel cache

```
~/.cache/adaptivecpp/  (or configured path)
    ├── kernels/
    │   ├── <hash(IR + device_id + optimization_config)>.ptx
    │   ├── <hash(IR + device_id + optimization_config)>.amdgcn
    │   └── <hash(IR + device_id + optimization_config)>.spv
    └── ...
```

Cache key = hash of (kernel IR + target GPU model + optimization flags). Same application, same GPU, second run: zero JIT overhead.

---

## Adaptive Specialization (IWOCL 2025)

The 2025 paper extends SSCP with a framework for **automatic kernel specialization at JIT time**.

### What information is available at JIT time (not at compile time)?

1. **Work-group size:** Often set programmatically; determines loop bounds, register pressure
2. **Pointer argument alignments:** Whether input data is 64-byte (cache-line) aligned enables vectorization
3. **Kernel argument values:** If an argument is a compile-time-constant in practice (e.g., a batch size of 512), it can be specialized as a literal
4. **Subgroup size:** Hardware-specific; 32 on NVIDIA, 64 on AMD GCN, variable on Intel

### Specialization mechanism

At JIT time, AdaptiveCpp:
1. Inspects actual kernel launch parameters (not known at compile time)
2. Substitutes concrete values as LLVM IR constants
3. Runs LLVM's constant propagation + LICM + loop unroll/vectorize with the specialized IR
4. Compiles the specialized kernel
5. Caches the specialization (additional cache key dimension: specialization hash)

This is **profile-guided optimization without a profiling run** — runtime values serve as the profile.

### Performance results (IWOCL 2025)

Evaluated on NVIDIA, AMD, and Intel hardware using mini-apps:

| Comparison | Geometric mean speedup |
|------------|------------------------|
| AdaptiveCpp SSCP+adaptive vs CUDA | **+30%** |
| AdaptiveCpp SSCP+adaptive vs HIP | **+44%** |
| AdaptiveCpp SSCP+adaptive vs oneAPI (Intel) | **+23%** |
| Best individual case | **>5x** (extreme specialization benefit) |

The gains come from LLVM generating better code when constants are known — loop bounds that unroll fully, alignment-enabled AVX/TensorCore instructions, etc.

---

## Single-Pass vs Multi-Pass: Detailed Comparison

| Property | DPC++ (multi-pass) | AdaptiveCpp SSCP (single-pass) |
|----------|-------------------|-------------------------------|
| Compilation passes | 1 per target + 1 host | 1 total |
| Output | Fat binary (target-specific blobs) | Single binary + generic LLVM IR |
| Target at compile time | Must know at build time | Not required |
| Target at runtime | Fixed to compiled targets | Any supported GPU present |
| First-run latency | None (AOT) | JIT cost (~100–500 ms per kernel, once) |
| Subsequent-run latency | None | None (persistent cache) |
| Binary size | O(n targets) | O(1) + LLVM IR overhead |
| Performance ceiling | Target-specific optimization | Runtime-specialized optimization |

---

## Supported Backends (AdaptiveCpp 24.x / 2025)

| Backend | Target ISA | JIT mechanism |
|---------|-----------|---------------|
| CUDA (NVIDIA) | PTX | LLVM NVPTX backend → `cuModuleLoadDataEx` |
| HIP (AMD) | AMDGCN | LLVM AMDGPU backend → `hipModuleLoadData` |
| SPIR-V (Intel / cross-vendor) | SPIR-V 1.4+ | LLVM SPIR-V backend → OpenCL/Level Zero |
| OpenMP (CPU) | Host ISA | LLVM host backend |

38+ AMD GPU architectures supported via ROCm 5.3+ automatic detection.

---

## Compile-Time Overhead Numbers

From AdaptiveCpp documentation and the sycl-ecosystem literature note:

- SSCP adds **~15% over host-only compilation** time
- SSCP is **>2x faster** than DPC++ multi-pass when targeting 3+ backends
- First kernel JIT: typically 100 ms – 1 s depending on kernel complexity
- Cached kernel load: < 1 ms

For ML inference at scale: the first-run JIT cost is amortized across thousands of forward passes. Not acceptable for latency-critical single requests.

---

## Relevance to libkdl

### This is the closest prior art

AdaptiveCpp SSCP is the closest production implementation to libkdl's core dispatch mechanism:

| Concept | AdaptiveCpp SSCP | libkdl |
|---------|-----------------|--------|
| Portable intermediate | Generic LLVM IR (embedded in binary) | MLIR `gpu.binary` with multiple target attributes |
| Hardware detection | CUDA/HIP/OpenCL device properties | Custom capability query (kdl_probe.c) |
| Target compilation | LLVM NVPTX/AMDGPU/SPIR-V backends (JIT) | Pre-compiled variants (AOT, no JIT latency) |
| Selection moment | First kernel invocation | Process startup (`kdl_init`) |
| Cache | Persistent on-disk kernel cache | In-memory dispatch table (no re-compilation) |
| Specialization | Runtime argument values → specialized kernels | Capability profile → pre-matched kernel variant |

### Key differences / libkdl advantages

1. **No first-run JIT latency:** libkdl carries pre-compiled kernel variants (AOT), not generic IR requiring JIT. Critical for latency-sensitive inference — no 100–500 ms first-invocation penalty.

2. **MLIR-level metadata:** libkdl's dispatch descriptor captures capability metadata at the MLIR level (before LLVM lowering), enabling richer dispatch logic than AdaptiveCpp's GPU capability query.

3. **Explicit multi-target design:** libkdl's dispatch table explicitly carries N variants (one per GPU family); AdaptiveCpp's JIT implicitly handles any target. libkdl's approach is more auditable and the performance of each variant is independently verifiable.

4. **No SYCL dependency:** libkdl works with any kernel source (CUDA, HIP, OpenCL, MLIR-generated) — it is a dispatch layer, not a programming model.

### AdaptiveCpp's advantage over libkdl

- **True target independence:** Can run on any GPU present, including future hardware not known at compile time. libkdl requires a pre-compiled variant for each target GPU family.
- **Adaptive specialization:** Runtime argument-value specialization (the IWOCL 2025 contribution) produces kernels more optimized than any fixed AOT compilation could achieve.
- **Production maturity:** AdaptiveCpp has years of production deployment across NVIDIA, AMD, and Intel hardware.

---

## Risks / Gaps

- AdaptiveCpp SSCP is the default in recent versions but the team is small; Intel DPC++ controls more industry resources
- The IWOCL 2025 30% CUDA speedup claim needs careful interpretation: it reflects specialization gains on specific benchmarks, not universal improvement; CUDA on complex kernels may still be faster
- JIT first-run overhead is a real barrier for latency-critical applications (embedded inference, real-time systems)
- Intel GPU performance via SPIR-V remains less optimized than DPC++ native path
- No MLIR integration — the generic IR is LLVM IR, not MLIR, so MLIR-level optimizations (affine, linalg) are not preserved

---

## Notes for Poster

- **Cite as primary prior art for runtime JIT dispatch** — "the only production SYCL implementation with unified JIT dispatch across all backends"
- Reference: IWOCL 2025 paper (ACM DL: https://dl.acm.org/doi/10.1145/3731125.3731127) for the adaptive specialization results
- Use the SSCP two-stage model figure as a comparison point for libkdl's architecture diagram
- Key contrast for poster: AdaptiveCpp JIT (generic IR → target-specific at first invocation, 15% compile overhead, disk cache) vs libkdl (pre-compiled variants → O(1) dispatch table lookup, zero JIT overhead, in-memory)
- The 30% CUDA speedup from adaptive specialization is a motivator for libkdl to eventually incorporate runtime-specialized variants alongside its pre-compiled ones (future work section)
- Honest framing: AdaptiveCpp SSCP is a stronger "single binary, any hardware" system than libkdl currently; libkdl's advantage is zero-latency dispatch and MLIR-level capability metadata
