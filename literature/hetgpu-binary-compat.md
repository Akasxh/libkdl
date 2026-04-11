# HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs
## Literature Review — LLVM Dublin 2026 Poster

**Citation:** (Authors not named in abstract.) "HetGPU: The pursuit of making binary compatibility towards GPUs." *arXiv preprint*, arXiv:2506.15993. Submitted June 19, 2025.

**Date reviewed:** 2026-04-06
**arXiv URL:** https://arxiv.org/abs/2506.15993
**Full paper:** https://arxiv.org/html/2506.15993v1
**Hacker News discussion:** https://news.ycombinator.com/item?id=46791479

> **Note:** This is a preprint, not peer-reviewed. Treat performance numbers as preliminary. The work targets a fundamentally different problem than libkdl (binary migration vs. dispatch optimization), but the contrast sharpens libkdl's contribution.

---

## Relevance Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Technical relevance | 7/10 | Addresses GPU vendor lock-in via a novel IR; directly comparable problem framing to libkdl |
| Approach overlap | 4/10 | Binary-level IR translation differs substantially from libkdl's pre-compiled kernel registry approach |
| Citation priority | 7/10 | Strong problem statement, useful contrast; cite to sharpen differentiation from libkdl |

---

## Problem

GPU vendor lock-in operates at the binary level. Programs compiled for NVIDIA GPUs (PTX → SASS) are incompatible with AMD (AMDGCN ISA), Intel (GEN/Xe assembly), or Tenstorrent (Tensix RISC-V). While source-level portability tools (SYCL, OpenCL, HIP, chipStar) enable recompilation across vendors, they require access to source and full toolchain reinvocation.

The gap HetGPU targets: can a binary compiled once for one GPU architecture execute — without recompilation, without source access — on a different vendor's GPU? This is the GPU equivalent of what Wine does for Windows binaries on Linux: binary translation and runtime abstraction.

Secondary motivation: **live kernel migration.** In large heterogeneous clusters, a running GPU kernel on one vendor's hardware should be checkpointable and resumable on a different vendor's GPU without restarting the job. This is not possible with any existing tool (CRIU-gpu and PhoenixOS target same-hardware checkpoint/restore; they do not support cross-vendor migration).

**Key quote:** "GPU binaries are not portable across vendors" due to "incompatible ISAs and architectures," creating vendor lock-in that "forces organizations with large CUDA codebases to expend significant effort porting."

---

## Contribution

1. **hetIR (Heterogeneous IR):** A custom architecture-agnostic GPU intermediate representation. Abstracts SIMT and MIMD execution models through explicit synchronization primitives and predication. Avoids committing to specific warp sizes or thread group topologies. Represents coordination via barriers rather than hardware-native warp intrinsics.
2. **HetGPU Compiler (frontend + backend):** Clang/LLVM-based frontend with CUDA support; custom LLVM backend (HETTarget) emitting hetIR. Defers device-specific optimization to the runtime JIT to enable per-device tuning.
3. **Multi-Target Runtime:** Dynamically translates hetIR to native code at kernel load time via JIT compilation. Routes to NVIDIA (cuModuleLoadDataEx → PTX), AMD (ROCm OpenCL / HIP path → AMDGCN), Intel (Level Zero / SPIR-V → GEN assembly), Tenstorrent (custom loader → Tensix RISC-V).
4. **Execution State Management:** IR-level checkpoint/restore at synchronization boundaries. Thread state is captured in a device-independent format (serialized register files + memory snapshot) enabling cross-vendor resumption.
5. **SIMT-to-MIMD Bridging:** Two strategies for executing warp-based SIMT code on Tenstorrent's 120-core RISC-V (MIMD) architecture: (a) vectorize warps within cores, (b) partition warps across multiple cores with software-coordinated synchronization.

---

## Methodology

### Compilation Pipeline

```
CUDA source (.cu)
    │
    ▼
Clang/LLVM frontend
    │  - vendor builtins → abstract hetIR operations
    │  - __global__ → hetIR kernel entry point
    │
    ▼
HETTarget LLVM backend
    │  - emits hetIR assembly
    │  - inserts suspension points at barrier/sync boundaries
    │  - embeds execution state metadata
    │
    ▼
hetIR binary (.het)
    │
    ├──▶ NVIDIA: JIT to PTX, then cuModuleLoadDataEx
    ├──▶ AMD:    JIT to SPIR-V/AMDGCN, OpenCL/HIP runtime
    ├──▶ Intel:  JIT to SPIR-V, Level Zero zeModuleCreate
    └──▶ Tenstorrent: JIT to Tensix RISC-V, custom loader
```

The JIT occurs at kernel first-use, not at program start. Results are cached in a per-process module cache, so repeated kernel invocations pay no JIT cost.

### SIMT-to-MIMD Abstraction

NVIDIA/AMD GPUs use SIMT: 32 (NVIDIA) or 64 (AMD) threads execute the same instruction each cycle (a warp/wavefront). Divergent branches are serialized via predication. Tenstorrent's BlackHole uses 120 Tensix cores in MIMD mode: each core independently fetches and executes instructions.

hetIR maps to MIMD via two strategies:
- **Vectorization:** Pack warp lanes into SIMD units within one Tensix core. Suitable for non-divergent workloads.
- **Partitioning:** Assign each logical warp lane to one physical Tensix core. Barriers become inter-core synchronization operations (spinlock on a shared memory location). Suitable for divergent workloads.

### Live Migration Mechanism

At each synchronization boundary (barrier, fence), hetIR instructs the runtime to potentially checkpoint execution state:
1. Serialize all live register values per thread to a device-independent buffer.
2. Copy buffer (and referenced memory regions) to host.
3. Transfer to destination device (PCIe / NVLink / network).
4. Deserialize register state on destination.
5. Resume execution from the barrier boundary.

### Validation Methodology

- 10 benchmark kernels: vector add, SAXPY, matrix multiply, reduction, scan, bitcount, Monte Carlo pi, neural network layer (2-layer MLP)
- Platforms: NVIDIA H100, AMD RX 9070 XT, Intel Iris Xe, Tenstorrent BlackHole (120 Tensix cores)
- Correctness: validated against reference CPU implementations; all results identical within floating-point rounding
- Migration: checkpointed matrix multiply mid-execution on H100, restored on RX 9070 XT and Tenstorrent BlackHole; verified numerical continuity

---

## Results

### Performance Overhead

| Kernel | Platform | hetGPU overhead vs native |
|--------|----------|--------------------------|
| Vector Add | H100 | ~0% (cached; ~18% first-run JIT) |
| Matrix Multiply (3.8 TFLOP baseline) | H100 | <8% overhead → ~3.5 TFLOPs |
| Reduction | H100 | ~6% overhead |
| Average (10 kernels) | H100 | 5–15% overhead at steady state |
| CUDA kernels → AMD RDNA | AMD | 8–20% overhead |
| Mixed precision workloads | Intel Arc | 10–25% overhead |

**JIT compilation latency:** 10–200 ms per kernel on first execution, depending on platform and kernel complexity. This is the dominant cost for short-running workloads.

**Live migration downtime:** 2.2 seconds total for a 30-second matrix multiply job:
- 0.5 s checkpoint on H100
- 0.6 s restore on AMD RX 9070 XT
- 1.1 s transfer to Tenstorrent (largest due to PCIe bandwidth)

These numbers represent preliminary prototype results, not production-optimized implementations.

---

## Architecture Details

### hetIR Design Principles

hetIR is designed to be "semantically complete" — every operation in the CUDA/OpenCL execution model has a hetIR equivalent, including:
- Warp-synchronous operations (ballot, shfl, reduce within warp)
- Shared/local memory operations with explicit synchronization
- Atomic operations (CAS, fetch-add, exchange)
- Texture/sampler operations (mapped to load + interpolate primitives)

The explicit synchronization model is key: rather than relying on hardware warp-coherence, hetIR makes all synchronization explicit. This enables the runtime to insert checkpoint boundaries without program modification.

### Runtime JIT Architecture

The runtime maintains a per-process hetIR module cache:
- On first `cuLaunchKernel` (intercepted), look up kernel in cache
- If miss: JIT-compile hetIR → target ISA, cache result
- If hit: dispatch cached native code directly

JIT compilation uses LLVM's standard optimization pipeline (O2 by default) targeting the detected hardware. Per-device optimization is thus applied at first use, not at build time — similar to how SPIR-V drivers work, but with richer IR semantics than SPIR-V.

---

## Limitations

1. **Tensor core / matrix accelerator gap:** hetIR cannot currently express NVIDIA Tensor Core (WMMA) operations, AMD Matrix Core (MFMA), or Intel XMX operations. ML workloads depending on these hardware units will not benefit from hetGPU — they will fall back to scalar implementations with 10–100x performance regression. The paper explicitly acknowledges this as a "future work" item.
2. **Dynamic parallelism unsupported:** CUDA's `cudaLaunchKernel` from device code (child grid launch) is not supported in hetIR. Kernels using cooperative groups are also excluded.
3. **JIT latency for first use:** 10–200 ms per kernel on cold start. For ML inference serving (where first-token latency matters), this is unacceptable without a pre-JIT / AOT compilation mode (not yet implemented).
4. **Migration overhead:** 2.2 s live migration downtime is impractical for latency-sensitive ML serving; more appropriate for long-running HPC batch jobs.
5. **MIMD mapping quality:** The SIMT-to-MIMD mapping for Tenstorrent is functional but unoptimized. Divergent workloads partitioned across Tensix cores incur significant synchronization overhead due to software-implemented barriers.
6. **Correctness coverage:** Validation covers 10 curated kernels. Coverage of arbitrary CUDA programs (dynamic shared memory sizes, indirect dispatch, tensor memory access, persistent kernels) is unclear in the preprint.
7. **Preprint status:** No peer review as of April 2026. Performance numbers should be treated as prototype baselines, not production benchmarks.

---

## Comparison to Related Approaches

| System | Approach | Source Required | Binary Portable | Live Migration | ML Optimization |
|--------|----------|----------------|-----------------|----------------|-----------------|
| chipStar | SPIR-V recompilation | Yes (HIP/CUDA) | No (SPIR-V per-compile) | No | No |
| SYCL/DPC++ | Source recompilation | Yes (SYCL C++) | No | No | No |
| OpenCL/SPIR-V | SPIR-V standard IR | Yes | Partial (SPIR-V binary) | No | No |
| ZLUDA | CUDA API interception | No | Partial (CUDA-to-HIP only) | No | No |
| GPU Ocelot | PTX-to-AMD translation | No | Partial (PTX only) | No | No |
| CRIUgpu / PhoenixOS | GPU checkpoint/restore | No | No (same-vendor only) | Same-vendor only | No |
| **HetGPU** | hetIR binary translation | No (binary-level) | Yes (NVIDIA/AMD/Intel/Tenstorrent) | Yes (cross-vendor) | No |
| **libkdl** (our work) | Per-vendor pre-compiled registry | Yes (source compile) | No (multi-object package) | No | Yes |

**Key contrast with ZLUDA:** ZLUDA intercepts CUDA API calls and emulates them via ROCm, achieving "2.5% of assembly patterns" correctly (as noted in hetGPU paper). hetGPU claims broader correctness through IR-level translation rather than API-level interception.

**Key contrast with chipStar:** chipStar requires source code and produces a SPIR-V binary that a driver JITs. hetGPU operates at the compiled binary level — no source required. chipStar is a compiler; hetGPU is a binary translator + runtime.

---

## Connection to Our Work (libkdl)

HetGPU and libkdl solve adjacent problems with fundamentally different strategies. The contrast is useful for poster positioning.

**Shared problem space:**
- Both address GPU vendor lock-in
- Both enable code written for one GPU vendor to run on another
- Both involve a runtime dispatch layer that routes to different GPU backends

**Fundamental divergence:**

| Property | HetGPU | libkdl |
|----------|--------|--------|
| Source availability | Not required (binary-level) | Required (compile kernel objects per vendor) |
| Kernel optimization | JIT-compiled from hetIR (generic) | Pre-compiled and tuned per vendor |
| Tensor core support | Not supported | First-class (Tensor Cores, MFMA, XMX) |
| ML performance | 5–25% overhead vs native; no tensor core path | Target: near-native (pre-tuned) |
| Live migration | Yes (key differentiator) | No |
| Dispatch trigger | Binary translation on first use | Symbol lookup in kernel manifest |
| Use case alignment | Legacy CUDA binary portability | ML kernel library deployment |

**Key insight for poster:** HetGPU prioritizes *binary* portability (run existing compiled code anywhere) at the cost of optimization opportunity. libkdl inverts this: it accepts source compilation per vendor to gain optimization quality, then provides transparent runtime dispatch. These are complementary strategies targeting different points on the source-access vs. optimization tradeoff:

```
Source required ◀──────────────────────────────▶ Binary-level
     │                                                    │
  libkdl                                              HetGPU
(optimized kernels,                           (arbitrary CUDA binary,
 runtime dispatch)                             runtime translation)
```

HetGPU's gap — inability to use tensor cores — is precisely libkdl's strength. Cite hetGPU to establish that the "run anywhere" problem is recognized, then argue that "run anywhere efficiently for ML" requires libkdl's approach.

**Key quote:** hetGPU's approach "decouples GPU binary code from the underlying hardware through a virtual GPU instruction set and runtime layer." libkdl decouples ML kernel selection from vendor-specific compilation through a kernel registry and dynamic linker.

---

## Citation

```bibtex
@misc{hetgpu2025,
  title         = {HetGPU: The pursuit of making binary compatibility towards GPUs},
  year          = {2025},
  eprint        = {2506.15993},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  url           = {https://arxiv.org/abs/2506.15993}
}
```

---

## Sources

- [HetGPU arXiv abstract](https://arxiv.org/abs/2506.15993)
- [HetGPU full HTML paper](https://arxiv.org/html/2506.15993v1)
- [Hacker News discussion thread](https://news.ycombinator.com/item?id=46791479)
- [Cross-Vendor GPU Programming — ResearchGate (related)](https://www.researchgate.net/publication/392803358_Cross-Vendor_GPU_Programming_Extending_CUDA_Beyond_NVIDIA)
- [Universal GPU ISA analysis (arXiv 2603.28793)](https://arxiv.org/abs/2603.28793)
