# Wave 06 — Kernel Binary Compatibility & ABI Stability

**Angle:** kernel-binary-compatibility-abi
**Search query:** "GPU kernel binary compatibility ABI stability versioning CUDA compute capability"
**Date:** 2026-04-06

---

## Executive Summary

GPU kernel binary compatibility is governed by a layered set of hardware-version rules, IR-level forward-compat guarantees, and runtime caching policies. The CUDA/PTX model and the AMD HSACO/code-object model differ sharply in their ABI versioning strategies. A critical break in CUDA's traditional forward-compatibility guarantee arrived in CUDA 12.0 with "architecture-accelerated features" (sm_90a, sm_100a), which fundamentally affects any runtime dispatch system that assumes PTX is always forward-portable. For libkdl, this means cached-binary validity checks must encode architecture family, driver version, and for AMD targets, xnack/sramecc feature flags.

---

## Sources

1. [CUDA Minor Version Compatibility — CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
2. [Forward Compatibility — CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
3. [Blackwell Architecture Compatibility Guide 13.2 — NVIDIA](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
4. [Hopper Architecture Compatibility Guide 13.2 — NVIDIA](https://docs.nvidia.com/cuda/hopper-compatibility-guide/)
5. [CUDA Compatibility PDF r590 — NVIDIA (Dec 2025)](https://docs.nvidia.com/deploy/pdf/CUDA_Compatibility.pdf)
6. [CUDA Pro Tip: Understand Fat Binaries and JIT Caching — NVIDIA Blog](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
7. [Understanding PTX, the Assembly Language of CUDA GPU Computing — NVIDIA Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
8. [nvFatbin 13.2 — NVIDIA](https://docs.nvidia.com/cuda/nvfatbin/index.html)
9. [User Guide for AMDGPU Backend — LLVM 23.0.0git](https://llvm.org/docs/AMDGPUUsage.html)
10. [ROCm-ComputeABI-Doc/AMDGPU-ABI.md — ROCm GitHub](https://github.com/ROCm/ROCm-ComputeABI-Doc/blob/master/AMDGPU-ABI.md)
11. [CUDA Compatibility — Lei Mao's Log Book](https://leimao.github.io/blog/CUDA-Compatibility/)
12. [GPU PTX/JIT Binaries — NASA LAVA](https://www.nas.nasa.gov/LAVA/setup_docs/gpu_binaries/)
13. [NVIDIA Blackwell and CUDA 12.9 Introduce Family-Specific Architecture Features — NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)
14. [SM90 Hopper Architecture Features — NVIDIA/cutlass DeepWiki](https://deepwiki.com/NVIDIA/cutlass/7.1-sm90-hopper-architecture)
15. [SPIR-V Specification — Khronos Registry](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
16. [What is SPIR-V — Vulkan Documentation Project](https://docs.vulkan.org/guide/latest/what_is_spirv.html)
17. [ROCm Code Object Format — ReadTheDocs](https://rocmdoc.readthedocs.io/en/latest/ROCm_Compiler_SDK/ROCm-Codeobj-format.html)

---

## Key Findings

### 1. CUDA Cubin Binary Compatibility: Same-Major Forward Only

Cubin (`.cubin`) binaries obey strict hardware-version rules:

- A cubin compiled for `sm_XY` runs on any GPU with `cc >= X.Y` within the **same major version** (same `X`).
- Cross-major is not supported: a `sm_7x` cubin will not run on `sm_8x` hardware.
- Example: `sm_86` cubin runs on `sm_89`, but not on `sm_80` or `sm_90`.

This means a cached cubin is invalidated by any major-architecture upgrade (Volta → Ampere → Hopper → Blackwell).

### 2. PTX as the Forward-Compatibility Vehicle

PTX (Parallel Thread Execution, NVIDIA's virtual ISA) provides the traditional escape hatch:

- PTX compiled for `compute_XY` JIT-compiles to any SM with `cc >= X.Y`, including cross-major jumps (e.g., `compute_80` PTX runs on `sm_90`).
- The CUDA runtime JIT-compiles PTX to cubin at first execution, then caches the result in `~/.nv/ComputeCache`.
- **Cache invalidation rule**: the compute cache is automatically invalidated whenever the device driver is upgraded, forcing re-JIT from stored PTX.
- Cache is controlled by env vars: `CUDA_CACHE_DISABLE`, `CUDA_CACHE_MAXSIZE` (default 256 MiB, max 4 GiB), `CUDA_CACHE_PATH`.

**Critical implication for libkdl**: any dispatch layer that caches compiled binaries must key cache entries on `(driver_version, sm_version)` — a driver upgrade silently invalidates all JIT-cached cubins.

### 3. Architecture-Accelerated Features: The PTX Forward-Compat Break (CUDA 12.0+)

CUDA 12.0 introduced "architecture-accelerated features" — a fundamental break in the PTX forward-compatibility contract:

- Kernels using Hopper's Tensor Memory Accelerator (TMA) or Warpgroup Matrix Multiply-Accumulate (WGMMA) must be compiled to `sm_90a` / `compute_90a` (note the `a` suffix).
- **PTX compiled for `sm_90a` runs only on devices with `cc == 9.0`** — it is neither forward nor backward compatible.
- Blackwell extends this: `sm_100a` / `sm_101a` features are similarly locked to their respective compute capabilities.
- This means: a heterogeneous dispatch system shipping a single PTX blob cannot transparently use TMA/WGMMA on Hopper while falling back gracefully to other hardware. Two separate code paths are required.

**Affected features (Hopper sm_90a):** WGMMA instructions, TMA async copy, barrier synchronization primitives.
**Affected features (Blackwell sm_100a/101a):** New family-specific instructions introduced in CUDA 12.9.

### 4. CUDA Minor Version Compatibility (Toolkit ↔ Driver)

Since CUDA 11, the toolkit-driver compatibility follows:

| CUDA Toolkit Major | Min Driver Version |
|---|---|
| CUDA 13.x | >= 580 |
| CUDA 12.x | >= 525 |
| CUDA 11.x | >= 450 |

Applications using features requiring a newer driver trigger `cudaErrorCallRequiresNewerDriver` at runtime. Before CUDA 11, every minor toolkit release had its own minimum driver requirement — making pre-CUDA-11 binaries significantly more fragile.

**Forward compatibility package**: for running CUDA 12.x apps on older drivers, NVIDIA ships `cuda-compat-12-x` packages. These are relevant for production deployments where driver upgrades lag toolkit releases.

### 5. Fat Binary Format and Runtime Selection Logic

NVIDIA fat binaries (`.fatbin`) bundle multiple code variants:

- Cubin variants for each targeted `sm_XX`
- PTX variant for the lowest targeted `compute_XX`

Runtime selection priority:
1. Find exact-match cubin for the present GPU's `cc`
2. Find compatible cubin (same major, lower-or-equal minor `cc`)
3. JIT-compile PTX, cache the result
4. Error if no compatible variant exists

The `nvFatbin` API (CUDA 13.2) allows programmatic fatbinary construction via `nvFatbinAddCubin`, `nvFatbinAddPTX`, `nvFatbinAddLTOIR`. This is directly relevant to libkdl's approach of constructing dispatch-ready binary bundles.

### 6. AMD HSACO Binary Compatibility: Code Object Versions and Feature Flags

AMD's HSACO (Heterogeneous System Architecture Code Object) uses ELF format with version-encoded ABI:

**Code object versions** (ELF ABI version field):
- V2: `ELFABIVERSION_AMDGPU_HSA_V2` — legacy, feature flags in EF flags and NT notes
- V3: `ELFABIVERSION_AMDGPU_HSA_V3`
- V4: `ELFABIVERSION_AMDGPU_HSA_V4`
- V5: `ELFABIVERSION_AMDGPU_HSA_V5` — **current default**
- V6: `ELFABIVERSION_AMDGPU_HSA_V6`

GPU target is encoded in the `EF_AMDGPU_MACH` bitfield of `e_flags` (V3+) or in `NT_AMD_HSA_ISA_VERSION` note record (V2).

**xnack and sramecc feature flags — the AMD binary compatibility critical path:**

| Feature | V2/V3 behavior | V4+ behavior |
|---|---|---|
| xnack unspecified | Loads only if process XNACK matches | Loads regardless of XNACK setting |
| sramecc unspecified | Loads only if process SRAMECC matches | Loads regardless of SRAMECC setting |

This means: V2/V3 HSACO binaries compiled without explicit xnack/sramecc annotations can fail to load at runtime if the device's hardware feature configuration differs. V4+ relaxed this, making cached binaries more portable across feature configurations.

**AMDGPU ABI versioning in kernel descriptor:**
- `amd_code_version_major`: not backward compatible across major versions
- `amd_code_version_minor`: backward compatible within same major
- `amd_machine_version_{major,minor,stepping}`: must match the GFX target family (gfx900, gfx906, gfx90a, gfx1100, etc.)

A cached HSACO becomes invalid when:
1. The GPU's GFX target differs from the `EF_AMDGPU_MACH` encoded in the ELF
2. For V2/V3: device xnack or sramecc setting differs from what was assumed at compile time
3. Code object major version changes
4. ROCm runtime version changes the HSA ABI major version

### 7. SPIR-V Binary Compatibility

SPIR-V uses a `Major.Minor.Revision` versioning scheme encoded directly in the binary header:
- Major has been fixed at 1 since inception (future reserved)
- Minor version additions are cumulative; consumers should support all features up to their stated version
- Revision changes: clarifications and deprecations only, no removals

SPIR-V is **not** directly hardware-executable — it requires offline or JIT compilation by the driver (Vulkan compute pipeline, OpenCL, SYCL). Binary compatibility therefore depends on:
- The target driver's declared `VkPhysicalDeviceProperties.apiVersion` (Vulkan)
- Which SPIR-V extensions are declared supported
- The `spirv-val` validator's target environment setting

SPIRV-Tools (KhronosGroup) versioning uses `vyear.index` scheme. The project has stated API stability: "we don't anticipate making a breaking change for existing features."

---

## Synthesis for libkdl / Heterogeneous Dispatch

### Binary Cache Key Requirements

For a correct heterogeneous kernel dispatch cache, the cache key must encode:

**CUDA targets:**
```
key = (sm_major, sm_minor, driver_version_major, cuda_arch_accelerated_variant)
```
- `cuda_arch_accelerated_variant` = `a`-suffix flag (sm_90a vs sm_90)
- Driver version major is needed because JIT cache is invalidated on driver upgrade

**AMD targets:**
```
key = (gfx_target_id, code_object_version, xnack_mode, sramecc_mode)
```
- For V4+ code objects, xnack/sramecc can be omitted from the key (both settings accepted)
- For V2/V3, xnack and sramecc must match the device's hardware configuration

**SPIR-V targets:**
```
key = (spirv_version, target_env, driver_version, extensions_used[])
```

### When Cached Binaries Become Invalid

| Platform | Invalidation Trigger |
|---|---|
| CUDA cubin | GPU major arch change (Ampere → Hopper), driver upgrade irrelevant for pre-compiled cubin |
| CUDA JIT cache | Any driver upgrade, GPU arch change |
| CUDA sm_90a PTX | Any hardware other than cc=9.0 |
| AMD HSACO (V2/V3) | GFX target change, xnack/sramecc mode change, ROCm ABI major bump |
| AMD HSACO (V4+) | GFX target change, ROCm ABI major bump (xnack/sramecc no longer invalidate) |
| SPIR-V | Driver SPIR-V version downgrade, unsupported extension on new target |

### The "Architecture-Accelerated Features" Problem for libkdl

This is a first-order concern for libkdl's design. If libkdl wants to dispatch optimized kernels using Hopper TMA/WGMMA or Blackwell sm_100a features, the dispatch table cannot use a single PTX blob with forward-compat fallback. The dispatch mechanism must:

1. Detect hardware at runtime (cc major.minor + `a`-suffix capability query)
2. Maintain separate binary variants: a standard `sm_90` cubin/PTX and a `sm_90a` cubin for the accelerated path
3. The selection logic is analogous to `ld.so` `IFUNC` resolvers — hardware-probed at load time, not at compile time

This is a direct argument for libkdl's multi-versioned kernel dispatch table approach.

---

## Angle Assessment

**Relevance: 9/10**
Core foundational material for libkdl's cache validity model and dispatch table design. The architecture-accelerated features break in CUDA 12.0+ is a direct design constraint for any heterogeneous dispatch system.

**Novelty: 7/10**
The individual facts are documented by NVIDIA and AMD, but the synthesis — particularly the contrast between CUDA's PTX forward-compat model breaking for sm_90a and AMD's V4+ relaxation of xnack/sramecc constraints — is not commonly presented together. The implications for a unified dispatch cache key design are novel in the context of this project.

**Gaps identified:**
- Need to investigate how PyTorch's inductor kernel cache (`~/.cache/torch`) handles sm_90a vs sm_90 variants
- How IREE's VMFB (VM flatbuffer) format encodes target metadata for cache validity
- Whether liboffload/OpenMP offload has formal ABI versioning for its code objects
- HIP's `clang-offload-bundler` fatbinary format and its version encoding rules
