# HIP & ROCm — Technical Research Notes

*Compiled: 2026-04-02. For the LLVM Dublin 2026 poster on heterogeneous GPU kernel dispatch.*

---

## 1. AMD HIP: Compile-Time Portability Between CUDA and ROCm

### Core Mechanism

HIP (Heterogeneous-compute Interface for Portability) achieves compile-time portability via a thin header-swap model combined with a compiler-driver wrapper. There is no runtime translation layer — the abstraction is resolved entirely at compile time.

**The `hipcc` driver** inspects `HIP_PLATFORM` (or probes installed toolchains) and selects one of two paths:
- `HIP_PLATFORM=amd` → invokes `amdclang++` (Clang/LLVM with AMDGPU backend) + ROCclr runtime
- `HIP_PLATFORM=nvidia` → invokes `nvcc` (NVIDIA's CUDA compiler) + CUDA runtime

This means the same `.hip` or `.cu` source file produces native PTX on NVIDIA and native GCN ISA on AMD — but via entirely different compiler stacks. HIP is not a JIT layer; it is a header and API naming convention that maps onto whichever native stack is active.

**Header dispatch**: The `hip/hip_runtime.h` header resolves at preprocessing time to either `cuda_runtime.h` (on NVIDIA) or AMD's own runtime headers (on AMD). Every `hipMalloc`, `hipMemcpy`, `hipLaunchKernelGGL` call expands to either a `cuda*` or AMD-internal call depending on which header set is in scope. This is a zero-overhead mechanism — there is no indirection at runtime.

**ROCm 7.0 / `amdclang++`**: AMD ships two compiler front-ends. `hipcc` remains the legacy entry point familiar to CUDA developers. `amdclang++` is the production-grade alternative offering explicit control over build flags, offload-arch selection, and device library linking. Both compile via LLVM's Clang front-end and the upstream AMDGPU backend. As of ROCm 7.0 (released early 2025), the bundled clang/llvm version is AMD Clang 20.0.0 (based on LLVM 20.0.0 plus out-of-tree patches).

### Portability Model: What It Covers

HIP covers the **CUDA Runtime API** (thread management, memory allocation, streams, events, kernels, graphs) and a subset of the **CUDA Driver API** (exposed via `hipModule*` and `hipCtx*` prefixes). The `hipCtx` family is deprecated in modern HIP — AMD's runtime manages contexts implicitly, unlike CUDA's explicit context model.

Supported HIP features include: streams and events, asynchronous memory transfers, pinned/unified memory, cooperative groups (since ROCm 4.1), warp/wavefront-level intrinsics (with caveats), dynamic shared memory, atomics, and most of the CUDA math library.

**Sources**: [AMD HIP What Is HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html) | [HIP GitHub](https://github.com/ROCm/HIP) | [ROCm Compiler Reference](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4/reference/rocmcc.html)

---

## 2. Hipify Tool: Automated CUDA → HIP Translation

### Two Tools

**hipify-clang**: A Clang-based AST-level translator. It parses CUDA code using Clang's actual parser (requires a working CUDA installation with all headers), performs accurate semantic analysis, and rewrites API calls, kernel launch syntax, and type names. It is the recommended path for production codebases.

**hipify-perl**: A regex-based text substitution tool derived from hipify-clang's mapping tables. Does not require CUDA headers. Handles simple cases — keyword swaps like `cudaMalloc` → `hipMalloc` — but produces incorrect output for template-heavy code, overloaded functions, or context-dependent patterns. Intended as a quick first pass, not a final solution.

### What Gets Translated Automatically

- CUDA Runtime API calls (the bulk of typical application code)
- Kernel launch syntax: `<<<grid, block, sharedMem, stream>>>` → `hipLaunchKernelGGL(...)`
- Standard library function names: `cuda*` → `hip*`
- Most math intrinsics: `__sinf`, `__expf`, etc.
- Thread indexing built-ins: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- Preprocessor guards: `__CUDA_ARCH__` → `__HIP_DEVICE_COMPILE__`
- CUDA memory management: malloc, memcpy, memset, free variants

The `--print-stats` flag on hipify-clang emits a count of converted vs. unconverted constructs per file.

### Known Translation Failure Categories

**Category 1 — Hard failures (no HIP equivalent)**:
- Inline PTX assembly (`asm volatile("ptx instruction" ...)`) — PTX is NVIDIA-architecture-specific. There is no AMD equivalent that hipify can substitute. Code must be rewritten using HIP's GCN inline assembly or higher-level intrinsics.
- NVLink-specific APIs and CUDA peer-to-peer topology queries involving SMP domains.
- Graphics interoperability: CUDA's Direct3D 9/10/11 interop, CUDA-OpenGL interop, VDPAU, EGL interop. None have HIP equivalents.
- Texture memory and surface objects: HIP does not implement the CUDA texture/surface object model. Code relying on texture caches for hardware-accelerated interpolation requires architectural redesign.
- CUDA Error Log Management APIs (added in CUDA 13.0): `cudaLogsCurrent`, `cudaLogsDumpToFile` — no HIP mappings.
- CUDA 12.5+ Driver Entry Points: lagging by approximately 1-2 CUDA versions.

**Category 2 — Semantic mismatches requiring manual verification**:
- Warp shuffle instructions: `__shfl_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`. HIP provides `__shfl` variants but semantics differ in edge cases relating to wavefront size.
- Cooperative groups: available in HIP since ROCm 4.1 via `hip/hip_cooperative_groups.h`, but the API surface is narrower than CUDA's, and warp-wide cooperative group semantics are not officially vendor-supported.
- CUDA Graphs: HIP Graph support exists but lags CUDA's feature set (hipGraph APIs are present but not all graph node types are supported on all hardware).
- Build scripts: hipify never converts CMake, Makefiles, or Python build infrastructure. Must be done manually.

**Category 3 — Third-party library dependencies**:
Code that calls CUDA-only libraries (cuDNN, TensorRT, cuSPARSE, nvJPEG, etc.) cannot be automatically hipified. AMD provides ROCm equivalents (MIOpen, rocBLAS, hipBLAS, rocSPARSE) but they are separate libraries with different APIs in some areas, and not all features are covered.

### Real-World Success Rates

- Simple to moderate HPC codebases: reported ~90-95% of lines translate without modification (e.g., HACC: 95% automatic conversion)
- Complex ML infrastructure code: one academic study found approximately 43.9% of files fail conversion entirely — indicating hipify works well for API-heavy code but struggles with code that mixes PTX intrinsics, template metaprogramming, or library-layer CUDA
- Post-hipify, developers universally report needing a manual code review pass and architecture-specific performance tuning

**Sources**: [HIPIFY Docs](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) | [GPUOpen HIP Portability](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-hipify-readme/) | [LUMI Porting Guide](https://lumi-supercomputer.eu/preparing-codes-for-lumi-converting-cuda-applications-to-hip/) | [OLCF Porting Presentation](https://www.olcf.ornl.gov/wp-content/uploads/Porting-Applications-to-HIP.pdf)

---

## 3. HIP Runtime vs CUDA Runtime: API Compatibility and Performance Parity

### API Compatibility

The CUDA Runtime API has approximately 400+ functions across 36 categories. HIP covers the overwhelming majority of production-use categories:

| Category | HIP Coverage |
|---|---|
| Device Management | Full (35+ functions) |
| Stream Management | Full (25+ functions) |
| Memory Management | Full (60+ functions) |
| Event Management | Full (8 functions) |
| Graph Management | Partial (90+ functions; not all node types) |
| Texture/Surface | Absent |
| Graphics Interop (D3D/GL) | Absent |
| Error Log Management (CUDA 13+) | Absent |
| Driver Entry Points (CUDA 12.5+) | Partial (lagging ~1-2 versions) |

The HIP runtime also wraps the CUDA Driver API: `cuModule*` → `hipModule*`, `cuCtx*` → `hipCtx*` (deprecated). This means most intermediate-level code that bypasses the runtime and calls into the driver can also be ported, though `hipCtx` usage is discouraged in modern code.

### Performance Parity

HIP is designed to sit at the same hardware proximity as CUDA — it is not an abstraction layer that interposes at runtime. When compiled for NVIDIA GPUs via `HIP_PLATFORM=nvidia`, the overhead is literally zero: `hipMalloc` expands at compile time to `cudaMalloc`. When compiled for AMD via HIP-Clang, the calls go directly into AMD's runtime (ROCclr / HIP-ROCclr), which is also a thin layer over the AMD driver.

**Benchmark evidence** (Futhark compiler study, A100 vs MI100, 48 workloads):
- HIP and CUDA back-ends share the same compiler optimization pipeline through all stages except the final code generation
- Performance differences between CUDA and HIP (on their respective GPUs) are attributable to hardware differences, not runtime overhead
- API-level overhead (kernel launch latency, memory transfer setup) is statistically indistinguishable between CUDA and HIP for kernels longer than ~250 microseconds
- Sub-250µs kernels show some CPU-side overhead in OpenCL but not between CUDA and HIP

**Context management difference**: CUDA requires explicit context creation before device operations. HIP-Clang creates a primary context lazily when the first HIP API is called. This changes initialization patterns but does not affect steady-state kernel performance.

**Sources**: [Futhark CUDA/HIP/OpenCL Comparison](https://futhark-lang.org/blog/2024-07-17-opencl-cuda-hip.html) | [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html) | [CUDA Runtime API Supported by HIP](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html)

---

## 4. What HIP Cannot Translate: Tensor Core Ops, CUDA-Specific Intrinsics

This is the most important section for the poster's "dynamic dispatch adds value" argument.

### Tensor Core / WMMA

CUDA exposes tensor core operations via `nvcuda::wmma` (Warp Matrix-Multiply Accumulate). The API uses NVIDIA-specific fragment types and instruction semantics tied to SM >= 7.0.

HIP/AMD does not implement `nvcuda::wmma` directly. AMD instead provides:
- **rocWMMA** library: a portable C++ header library that wraps AMD's MFMA and WMMA instructions with an API compatible with `nvcuda::wmma`. This allows source-level portability for code that uses the high-level wmma API.
- **MFMA intrinsics** (Matrix Fused Multiply-Accumulate): exposed in LLVM's ROCDL dialect as `rocdl.mfma.*` and at the C level as `__builtin_amdgcn_mfma_*`. These exist for CDNA architectures (MI100, MI200, MI300 series — gfx908, gfx90a, gfx940+).
- **WMMA intrinsics** for RDNA 3+ (gfx11+): `__builtin_amdgcn_wmma_*`. Different tiling dimensions and data type support than NVIDIA's tensor cores.

The critical incompatibility: NVIDIA tensor cores operate on 16x16 warp-wide tiles with 32 threads; AMD MFMA operates on different tile shapes (4x4, 16x16, 32x32) with 64-thread wavefronts. Even if the API is matched at the C++ level (via rocWMMA), the performance characteristics, register file pressure, and optimal tile sizes differ significantly and require per-architecture tuning.

### Inline PTX Assembly

`asm volatile("ptx.instruction ...")` blocks in CUDA code are untranslatable. PTX is NVIDIA's virtual ISA; AMD uses its own GCN/AMDGCN assembly syntax. Code that uses inline PTX for:
- Warp voting (`vote.any.pred`, `vote.all.pred`)
- Shuffle instructions at PTX level
- Special register reads (`%laneid`, `%warpid`, `%smid`)
- Low-level memory operations (`.volatile`, cache bypassing)
- CUDA cooperative groups using PTX barriers
...must all be rewritten using HIP's AMD GCN intrinsics or higher-level HIP APIs.

### Warp Size: 32 vs 64

AMD hardware runs 64-thread wavefronts by default on CDNA/GFX9 and older. RDNA (GFX10+) added a 32-thread mode (`-mwavefrontsize32`), but default is still 64 on compute-oriented CDNA. This affects:
- Algorithms that assume `warpSize == 32` in thread-level bitmasks
- Warp-level reduction trees (require different loop bounds)
- CUDA's `__ballot_sync` returns a 32-bit mask; HIP's `__ballot` returns a 64-bit mask on CDNA
- `__shfl_sync` lane indices go 0-31 in CUDA; 0-63 in AMD wavefronts

This is not merely a porting inconvenience — it changes the algorithmic structure of warp-level operations. Portability frameworks like ALPAKA abstract over this but HIP requires developer awareness.

### NVLink and Multi-GPU Topology APIs

NVLink (`cudaDeviceGetNvlinkCapability`, NVSwitch topology queries, NVLink p2p bandwidth optimization) have no HIP equivalents. AMD uses Infinity Fabric and xGMI links — these are accessible via ROCm-specific APIs but not abstracted by HIP.

### CUDA-Specific Library Features

| CUDA Library | HIP/ROCm Equivalent | Gap |
|---|---|---|
| cuDNN | MIOpen | Missing: some conv algorithms, INT8 calibration modes |
| cuBLAS | rocBLAS / hipBLAS | API compatible; some precision modes lag |
| TensorRT | No direct equivalent | Significant gap for inference optimization |
| nvJPEG | No HIP equivalent | Image codec pipeline missing |
| cuSPARSE | rocSPARSE | Partial coverage |
| NCCL | RCCL | Feature parity improving; xGMI support added |
| CUDA Graphs (full) | hipGraph (partial) | Some node types missing |

### CUDA C++ Language Extensions Not in HIP

- `__device_builtin__` attribute
- `cudaLaunchCooperativeKernelMultiDevice` (multi-GPU cooperative launch) — no direct HIP equivalent
- `__noinline__` / `__forceinline__` — exist in HIP but behavior may differ
- `__trap()` / `__brkpt()` — partially supported

**Sources**: [AMD GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/) | [LUMI HIP Porting](https://lumi-supercomputer.eu/preparing-codes-for-lumi-converting-cuda-applications-to-hip/) | [ROCDL Dialect](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/) | [HIP C++ Language Extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)

---

## 5. ROCm's LLVM-Based Compiler Stack: Relationship to Upstream LLVM

### Architecture Overview

ROCm's compiler infrastructure is built on an AMD-maintained fork of LLVM: [github.com/ROCm/llvm-project](https://github.com/ROCm/llvm-project). The fork adds the following components in the `amd/` subdirectory:

- **`amd/comgr`** (Code Object Manager API): runtime compilation API; wraps the LLVM compilation pipeline for device code, used by the HIP runtime to JIT-compile kernels from IR or source.
- **`amd/device-libs`**: AMD-specific device-side runtime libraries (math, atomics, printf support) analogous to CUDA's libdevice (`.bc` bitcode files linked at compile time).
- **`amd/hipcc`**: The `hipcc` compiler driver source.

The ROCm compiler (rocmcc / `amdclang++`) compiles **only two target architectures**: x86 (host) and AMDGPU (device). This is distinct from upstream LLVM which supports 20+ targets.

### Relationship to Upstream

The relationship has two modes:
1. **Upstreamed features**: The AMDGPU backend itself (`llvm/lib/Target/AMDGPU/`) is fully upstream in LLVM. All major GPU architecture support (GFX9 through GFX12), the register allocator, instruction selector, GlobalISel, and most ISA-level features are maintained in upstream LLVM and receive contributions from AMD engineers.
2. **ROCm-specific out-of-tree features**: The `amd/` additions (comgr, device-libs, hipcc) and some experimental features are maintained in the ROCm fork and are in various stages of upstreaming.

**Known divergence areas** (from the ROCm/llvm-project issue #263 discussion):
- **Heterogeneous debugging** (debug-info for AMDGPU targets): A work-in-progress prototype in the ROCm fork; not yet upstreamed due to ongoing design changes.
- **Address Sanitizer for GPU**: Instrumentation changes partially upstreamed; sanitizer-common and asan library changes still diverged.
- **Reverted upstream patches**: AMD maintains a `revert_patches.txt` listing upstream patches that break internal ROCm testing and have been temporarily reverted in the AMD fork.
- **Version lag**: Features may enter ROCm's LLVM before or after upstream. A critical issue: ROCm 6.3 was documented as requiring LLVM 18 but internally used LLVM 19-specific intrinsics (e.g., `llvm.amdgcn.permlanex16.i32`). This caused breakage for distribution packagers trying to use upstream LLVM.

**ROCm 7.0 (2025)**: Bumped to AMD Clang 20.0.0 (LLVM 20 base). Enabled parallel code generation by default for full LTO (`-fgpu-rdc`). Added `__builtin_amdgcn_processor_is` for deferred target processor queries — useful for runtime dispatch scenarios.

**Practical impact for our poster**: An MLIR-based dispatch layer that emits AMDGCN code through the upstream AMDGPU backend is feasible and supportable without depending on AMD's proprietary fork — the AMDGPU backend is in upstream LLVM. The `rocdl` and `amdgpu` MLIR dialects lower to this upstream backend.

**Sources**: [ROCm/llvm-project GitHub](https://github.com/ROCm/llvm-project) | [ROCm Compiler Reference](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4/reference/rocmcc.html) | [ROCm upstream compatibility issue](https://github.com/ROCm/llvm-project/issues/263) | [ROCm 7.0 release notes](https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html) | [Phoronix: AMD Device-Side PGO](https://www.phoronix.com/news/AMD-LLVM-Device-Side-PGO-ROCm)

---

## 6. AMD's AMDGCN Backend in LLVM: Maturity, Features, Limitations

### Backend Maturity

The AMDGPU backend is one of LLVM's most actively maintained GPU targets. It supports GPU families from R600 (ancient, mostly legacy) through GFX13 (RDNA 5, 2025). For ML workloads, the relevant families are:

| Architecture | GPU Family | Notable GPU | Status |
|---|---|---|---|
| GFX9 (Vega/CDNA 1) | MI100 | gfx908 | Production-stable |
| GFX90a (CDNA 2) | MI200 | gfx90a | Production-stable |
| GFX940/941/942 (CDNA 3) | MI300X/MI300A | gfx940 | Production-stable |
| GFX10 (RDNA 1-2) | RX 6000 series | gfx1030 | Supported |
| GFX11 (RDNA 3) | RX 7000 series | gfx1100 | Supported |
| GFX12 (RDNA 4) | RX 9000 series | gfx1200 | Recent addition |

**Generic processors** (e.g., `gfx9-generic`, `gfx11-generic`) allow a single compiled code object to run across multiple GPU variants within the same generation — a form of compile-time portability within AMD's lineup.

### Key Features

**Address spaces**: 8 address spaces — Generic (flat), Global, Local (LDS), Private (scratch), Region (GDS), Constant, Buffer Resource (8-bit), Buffer Fat Pointer. Generic address space is supported on GFX7+ via hardware flat address support.

**Memory model**: HSA (Heterogeneous System Architecture) memory model with synchronization scopes: system, agent, cluster, workgroup, wavefront, singlethread.

**Matrix operations** (critical for ML):
- **MFMA** (Matrix Fused Multiply-Accumulate): GFX9 (gfx908+). 4x4, 16x16, 32x32 tile sizes. Supports f16, bf16, f32, f64, int8 element types. Exposed as `llvm.amdgcn.mfma.*` intrinsics.
- **WMMA** (Wave Matrix Multiply-Accumulate): GFX11 (gfx1100+). Tile sizes differ from MFMA. Exposed as `llvm.amdgcn.wmma.*` intrinsics.
- **SMFMAC** (sparse MFMA): GFX940+ for structured sparsity.
- Scaled-precision MFMA variants for FP8 and BF8 on MI300 series.

**Instruction-level features**: DPP (Data Parallel Primitives) for lane-level data sharing, buffer resource pointers (128-bit descriptors), cooperative atomics on GFX12.5+.

### Limitations and Known Issues

1. **GFX11 generic processor**: Applies "codegen pessimizations" — conservative hazard workarounds that reduce performance compared to targeting specific GFX11 sub-variants. A single binary for the generic target pays a performance tax.
2. **R600 family**: No generic address space, no HSA compatibility. Effectively legacy and should be ignored for modern work.
3. **Code object versioning**: Code object V2 cannot represent `_Any_` target features; treats them as `_On_`. This limits some feature-agnostic binary distribution.
4. **Flat access to scratch**: Requires hardware aperture setup in the kernel prologue. Flat access to LDS on GFX7-GFX8 requires M0 register setup.
5. **Debug info**: Heterogeneous debug information is still incomplete in both upstream LLVM and ROCm's fork (as noted above).
6. **Divergence handling**: The backend must model wavefront divergence for SGPR vs VGPR usage. Some algorithms that are efficient on SIMT-32 (NVIDIA) become inefficient on SIMT-64 (AMD CDNA) due to more threads active per instruction.

### MLIR Integration

The AMDGPU backend is the lowering target for both the `rocdl` dialect and the `amdgpu` dialect in MLIR:

**`rocdl` dialect**: 1:1 wrappers around LLVM AMDGPU intrinsics. Operations include:
- Thread/workgroup identification (workitem.id.x/y/z, workgroup.id.x/y/z)
- MFMA matrix operations (f32_16x16x1f32, f32_32x32x1f32, etc.)
- WMMA operations (bf16, f16, f32, i32)
- Buffer load/store with raw buffer atomics
- Lane permutation (permlane16, permlanex16, ds.swizzle)
- Barrier and scheduling primitives
- Low-precision type conversions (FP8, BF8)

**`amdgpu` dialect**: Higher-level wrappers that handle bitpacking, type conversions, and magic constants. Sits between linalg-level operations and ROCDL. Key operations: `amdgpu.mfma`, `amdgpu.wmma`, `amdgpu.sparse_mfma`, DPP lane sharing, raw buffer loads/stores, tensor DMA operations.

**Lowering pipeline** for ML kernels targeting AMDGCN:
```
linalg → affine/scf → gpu dialect → amdgpu dialect → rocdl dialect → LLVM IR → AMDGCN ISA
```

The `gpu` → `rocdl` lowering pass (`createLowerGpuOpsToROCDLOpsPass`) accepts a chipset string and runtime selector (`gpu::amd::Runtime::HIP`) to control kernel ABI and calling conventions.

**Sources**: [AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html) | [ROCDL Dialect](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/) | [AMDGPU Dialect](https://mlir.llvm.org/docs/Dialects/AMDGPU/) | [LLVM 20.1 release](https://www.webpronews.com/llvm-20-1-arrives-with-major-c-and-gpu-advances-setting-the-stage-for-the-next-era-of-compiler-infrastructure/)

---

## 7. HIP Fat Binaries: Bundling Code for Multiple GPU Architectures

### Mechanism

When compiling HIP code with `--offload-arch=gfx906 --offload-arch=gfx942`, Clang produces a **HIP fat binary** — a container embedding fully-linked device images for each target architecture.

**Technical format**:
- Device images (one per `--offload-arch` target) are generated as AMDGCN ELF objects
- The `clang-offload-bundler` tool packs these into a single bundled object, embedded as the global symbol `__hip_fatbin` in the `.hip_fatbin` ELF section
- Each bundle entry has a unique **bundle ID** string encoding: `hip-amdgcn-amd-amdhsa--gfx906`, `hip-spirv64-amd-amdhsa--amdgcnspirv`, etc.
- The fat binary is embedded in the host executable as a data blob

**Runtime registration**: The Clang front-end inserts `__hipRegisterFatBinary` calls into the global constructor (`@llvm.global_ctors`) of the compiled object. At program startup, the HIP runtime registers this fat binary, discovers what device architectures are present in the system, and loads the matching device image (or the closest compatible one). Cleanup is registered via `atexit`.

### Two Compilation Modes

**`-fno-gpu-rdc` (default — non-relocatable device code)**:
- Each compilation unit compiles to a fully-linked, self-contained device binary per architecture
- Device functions cannot call across translation units (no device-side linking)
- Simplest and most common mode

**`-fgpu-rdc` (relocatable device code)**:
- Device code compiles to LLVM bitcode (relocatable form)
- A separate device-link step combines bitcode from all translation units into a unified device image per architecture
- Required for cross-TU device function calls, separate compilation of GPU libraries
- ROCm 7.0 enables parallel code generation by default in this mode for improved build times

### Multi-Architecture Heterogeneity

ROCm's fat binary format supports heterogeneous images at multiple levels:
1. **Different ISA families**: A single binary can embed both AMD GCN and NVPTX images (though this requires building with both toolchains)
2. **Different GPU variants within a family**: `gfx906` + `gfx908` + `gfx942` in one binary
3. **Same GPU variant with different target features**: `gfx908:xnack+` vs `gfx908:xnack-` (XNACK = demand paging support), `gfx908:sramecc+` vs `gfx908:sramecc-`

**SPIR-V path**: When targeting `--offload-arch=amdgcnspirv`, the compiler emits SPIR-V instead of GCN ISA. Bundle ID: `hip-spirv64-amd-amdhsa--amdgcnspirv`. This SPIR-V is then translated to native ISA by the Vulkan/ROCm driver at runtime — a form of portable device code within AMD's ecosystem.

### Comparison to CUDA Fat Binaries (cubin/fatbin)

| Aspect | CUDA fatbin | HIP fat binary |
|---|---|---|
| Container format | `__cudaFatCubin` struct | ELF `.hip_fatbin` section |
| PTX inclusion | Yes (portable) + cubin (arch-specific) | AMDGCN ELF + optional SPIR-V |
| Architecture fallback | PTX JIT-compiled if no cubin match | No equivalent JIT fallback in default mode |
| Multi-arch | `sm_XX` compute capabilities | `gfxNNN` targets + feature flags |
| Cross-vendor | NVIDIA only | AMD primary; NVIDIA via nvcc path |

A notable gap: NVIDIA's fatbin includes PTX as a portable fallback — if no cubin matches the GPU's compute capability, the driver JIT-compiles the PTX. AMD's fat binary has no equivalent portable IR fallback in the default mode. The SPIR-V path (via `amdgcnspirv`) is the closest analog but requires the Vulkan driver path, not the standard HIP runtime.

**Sources**: [Clang HIP Support](https://clang.llvm.org/docs/HIPSupport.html) | [Clang Offload Bundler](https://clang.llvm.org/docs/ClangOffloadBundler.html) | [Clang Offloading Design](https://clang.llvm.org/docs/OffloadingDesign.html) | [HIP Compilers](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html)

---

## 8. Performance Comparisons: HIP on AMD vs CUDA on NVIDIA for ML Workloads

### Overall Picture (2024-2025)

ROCm/HIP has closed the performance gap significantly compared to 2022-2023 but CUDA retains a structural lead for ML training workloads. The gap is best characterized as:
- **Hardware gap**: MI300X has more HBM capacity (192GB vs H100 SXM 80GB) and higher memory bandwidth, but NVIDIA's tensor core utilization is more mature
- **Software gap**: cuDNN, cuBLAS, NCCL, and specialized kernels (Flash Attention, FlexAttention) have years of optimization lead over ROCm equivalents
- **Ecosystem gap**: TensorRT, Triton support on NVIDIA, and broader framework coverage

### GEMM (Matrix Multiplication — Core of ML Compute)

**BF16 GEMM** (from SemiAnalysis MI300X vs H100 benchmarks, mid-2024):
- H100 SXM: ~720 TFLOP/s achieved vs 989.5 TFLOP/s peak (73% utilization)
- MI300X: ~620 TFLOP/s achieved vs 1,307 TFLOP/s peak (47% utilization)
- **Result: MI300X is ~14% slower in BF16 GEMM despite higher theoretical peak**

**FP8 GEMM**:
- H100: ~1,280 TFLOP/s
- MI300X: ~990 TFLOP/s
- **Result: MI300X is ~22% slower in FP8 GEMM**

The low utilization percentage on MI300X is the key finding: AMD has more peak compute but less of it is reachable in practice due to library maturity.

**hipBLASLt improvements (2025)**: AMD's hipBLASLt now supports Stream K load balancing, which reduces GEMM tuning overhead and improves utilization on irregular tensor shapes. Supported data types: FP32, FP16, BF16, FP8, BF8 with fused activation functions (SiLU, GELU, Swish).

### LLM Training Throughput (from SemiAnalysis, 2024)

| Model | H100 (public release) | MI300X (public release) | Gap |
|---|---|---|---|
| GPT 1.5B | Baseline | 2.5x slower | Severe |
| Llama 8B | Baseline | Competitive only with custom builds | Moderate (software-bound) |
| Llama 70B | Baseline | Significantly behind | Large |
| Mistral 7B | Baseline | ~50% of H100 | Large |

Key insight: MI300X's superior memory capacity (192GB vs 80GB for H100 SXM) means it can serve larger models without tensor parallelism, but training throughput per GPU-dollar is worse on public stable releases.

### Flash Attention

Flash Attention is the canonical example of ROCm's software gap vs CUDA:
- NVIDIA: FlashAttention 2 native CUDA implementation, highly optimized, available since 2022
- AMD: Historically relied on Triton-based implementation; Flash Attention 2 Triton available via `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` flag
- PyTorch 2.2: FlashAttention 2 for NVIDIA; PyTorch 2.3: Extended ROCm support for SDPA
- PyTorch 2.6.0: FlexAttention available on ROCm
- The SemiAnalysis report noted flash attention achieving "<20 TFLOP/s" on ROCm during mid-2024; this was fixed in later releases

**Current state (early 2026)**: Flash Attention 2 Triton FP8 is the recommended path on ROCm. Performance is competitive with NVIDIA Flash Attention 2 on memory-bound sequences but lags on compute-bound (short sequences). A TileLang implementation on MI300X achieved 2.7x speedup over PyTorch SDPA, demonstrating that the hardware is capable but requires hand-tuned kernels.

### Network Bandwidth (Scale-Out)

AMD's RoCEv2 Ethernet (used by MI300X clusters) runs at approximately half the speed of NVIDIA's InfiniBand + SHARP for AllReduce collectives at the 16MiB-256MiB message sizes critical for model-parallel training. This is a system-level gap that impacts multi-node training more than single-node performance.

### General Compute (Non-ML)

In the Futhark compiler study (48 benchmarks, MI100 vs A100):
- OpenCL vs HIP on MI100: comparable performance; key differences attributable to numerical precision defaults and thread block size limits (OpenCL on AMD limited to 256 threads/block; HIP allows 1024)
- HIP itself introduces negligible overhead vs native AMDGCN code — the back-end compilation path is identical

### Cost-Performance

AMD hardware costs 15-40% less than comparable NVIDIA hardware on cloud and on-premise. For memory-bandwidth-bound inference workloads (LLM KV-cache serving, model with large weights), MI300X's superior memory capacity makes it more attractive despite compute gaps. For training, CUDA's training throughput per TCO was better on public ROCm releases as of mid-2024.

**Sources**: [SemiAnalysis MI300X vs H100 benchmark](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training) | [ThunderCompute ROCm vs CUDA 2026](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing) | [Flash Attention ROCm State](https://zdtech.substack.com/p/the-state-of-flash-attention-on-rocm) | [ROCm 7.0 Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.0-blog/README.html) | [Futhark OpenCL/CUDA/HIP](https://futhark-lang.org/blog/2024-07-17-opencl-cuda-hip.html)

---

## Cross-Cutting Analysis: Implications for Heterogeneous Dispatch

### The HIP Model as a Reference Point

HIP's compile-time portability model is the most mature example of "write once, run on NVIDIA or AMD" for GPU kernels. Its key design choices:
1. **Zero runtime overhead** — dispatch is resolved at compile time via header selection
2. **Native code generation** — no IR translation at runtime; each platform gets native ISA
3. **Single source** — one `.hip` file, two build configurations

**Limitation for our poster's problem**: HIP requires knowing the target at compile time. It cannot dispatch to NVIDIA and AMD from the same binary. A binary compiled with `HIP_PLATFORM=nvidia` will not run on an AMD GPU and vice versa.

### The Gap Our Work Addresses

HIP's model assumes **compile-time target selection**. In heterogeneous deployment scenarios — cloud instances with mixed GPU pools, edge deployments on unknown hardware, multi-tenant inference clusters — the target is not known at compile time. Our proposed MLIR-based dispatch layer fills this gap by:
1. Compiling to **multiple targets simultaneously** (NVPTX + AMDGCN + SPIR-V + x86) in a single fat binary
2. Querying hardware at runtime and selecting the appropriate pre-compiled kernel
3. Maintaining the zero-runtime-overhead property for the kernels themselves (dispatch overhead isolated to selection, not execution)

### AMDGCN Backend Maturity Assessment

The AMDGPU LLVM backend is **production-ready** for generating high-quality AMDGCN ISA from LLVM IR. The `rocdl` and `amdgpu` MLIR dialects provide complete lowering paths from high-level ML operations (linalg) to AMD hardware. An MLIR-based multi-target compiler that emits AMDGCN code is technically sound and does not require AMD's proprietary ROCm fork — the upstream LLVM AMDGPU backend is sufficient.

The fat binary bundling mechanism (`clang-offload-bundler`) provides a direct precedent and implementation reference for our multi-target fat binary design.

### Wavefront Size as a Dispatch Motivation

The 32 vs 64 thread wavefront difference is a strong argument for **runtime-specialized dispatch** rather than static portability. A warp-level reduction kernel optimized for 32-thread warps (NVIDIA) will have different optimal tile sizes, register allocation, and loop bounds than one optimized for 64-thread wavefronts (AMD). This is exactly the kind of hardware-specific optimization that cannot be expressed in a single static binary without sacrificing performance on one platform. A capability-aware JIT dispatch layer can specialize these parameters at load time.

---

## Bibliography

| Reference | URL | Relevance |
|---|---|---|
| AMD HIP What Is HIP | https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html | Architecture overview |
| HIP Porting Guide | https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html | API coverage, limitations |
| HIPIFY Documentation | https://rocm.docs.amd.com/projects/HIPIFY/en/latest/ | Translation tools |
| CUDA Runtime API Supported by HIP | https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html | API coverage table |
| Clang HIP Support | https://clang.llvm.org/docs/HIPSupport.html | Compilation pipeline |
| Clang Offload Bundler | https://clang.llvm.org/docs/ClangOffloadBundler.html | Fat binary format |
| Clang Offloading Design | https://clang.llvm.org/docs/OffloadingDesign.html | Multi-target bundling |
| AMDGPU Backend User Guide | https://llvm.org/docs/AMDGPUUsage.html | Backend features/limitations |
| ROCDL Dialect | https://mlir.llvm.org/docs/Dialects/ROCDLDialect/ | MLIR AMDGCN lowering |
| AMDGPU Dialect | https://mlir.llvm.org/docs/Dialects/AMDGPU/ | MLIR matrix ops |
| ROCm Compiler Reference | https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4/reference/rocmcc.html | Compiler architecture |
| ROCm/llvm-project GitHub | https://github.com/ROCm/llvm-project | Fork source |
| ROCm upstream compat issue | https://github.com/ROCm/llvm-project/issues/263 | Fork divergence |
| ROCm 7.0 Release Notes | https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html | Latest compiler |
| Futhark CUDA/HIP/OpenCL | https://futhark-lang.org/blog/2024-07-17-opencl-cuda-hip.html | HIP API overhead data |
| SemiAnalysis MI300X benchmark | https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training | ML perf data |
| ThunderCompute ROCm vs CUDA | https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing | 2025-2026 overview |
| Flash Attention ROCm State | https://zdtech.substack.com/p/the-state-of-flash-attention-on-rocm | Attention kernel analysis |
| AMD WMMA on RDNA3 | https://gpuopen.com/learn/wmma_on_rdna3/ | Tensor core portability |
| LUMI HIP Porting | https://lumi-supercomputer.eu/preparing-codes-for-lumi-converting-cuda-applications-to-hip/ | Porting guide |
| GPUOpen HIP Portability | https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-hipify-readme/ | HIPIFY guide |
| ROCm 7.0 Blog | https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.0-blog/README.html | ROCm 7 AI readiness |
| HIP FAQ | https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html | General FAQ |
