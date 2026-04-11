# Wave 03: SYCL Multi-Target Compilation in LLVM
Search query: "SYCL DPC++ multi-target compilation LLVM SPIR-V CUDA HIP ahead-of-time"
Sources found: 9
Date: 2026-04-06

## Overview

SYCL provides two fundamentally different approaches to multi-target compilation, both highly relevant to
libkdl's vendor-agnostic dispatch model:

1. **DPC++ (Intel oneAPI):** Single-source, multiple-compiler-passes (SMCP). The Clang driver invokes
   separate device compilation passes per target, embeds all resulting binaries in a fat host object,
   and relies on the Unified Runtime (UR) layer to select the correct image at runtime.

2. **AdaptiveCpp (formerly hipSYCL):** Single-source, single-compiler-pass (SSCP). One compilation pass
   extracts backend-independent LLVM IR for all kernels, embeds it in the binary, and JIT-compiles to
   PTX/SPIR-V/AMDGCN at runtime depending on which GPU is found. Closest to libkdl's architectural goal.

---

## Sources

### 1. oneAPI DPC++ Compiler and Runtime Architecture Design — Intel/LLVM Official Docs
- URL: https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html
- Type: docs
- Date: 2024–2026 (continuously updated, mirrors intel/llvm `sycl` branch)
- Relevance: 10/10
- Novelty: 8/10
- Summary: Canonical reference for DPC++'s multi-target compilation pipeline. Describes the
  complete flow from `-fsycl-targets` flag through `sycl-post-link`, `llvm-spirv`, and
  `clang-offload-wrapper` to the final fat binary. Each target backend (SPIR-V, NVPTX, AMDGCN)
  follows a different code path: only NVPTX skips SPIR-V translation entirely, instead keeping
  bitcode all the way to PTX assembly via `ptxas`. Device images are registered with the runtime
  via auto-generated global constructors that call into `libsycl`'s registration functions.
- Key detail for libkdl: The `pi_device_binary_struct` is the runtime's unit of kernel packaging.
  It carries: target architecture string (e.g., `__SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64`), pointer
  to the binary blob, kernel symbol table, and specialization constant table. At dispatch time, the
  runtime iterates registered images, matches target string to available device capabilities, and
  selects the first compatible image. This is SYCL's equivalent of `ld.so` symbol resolution — and
  the exact data structure libkdl needs to generalize across vendors.

### 2. Proposed Offload Design — Intel/LLVM Official Docs (New Generation)
- URL: https://intel.github.io/llvm/design/OffloadDesign.html
- Type: docs
- Date: 2024–2025 (active development, aligns with LLVM upstream offload model)
- Relevance: 9/10
- Novelty: 9/10
- Summary: Documents the architectural shift away from `clang-offload-bundler`'s fat-object
  section labels (`__CLANG_OFFLOAD_BUNDLE__kind-triple`) toward a new `llvm-offload-binary` format.
  The new model moves device binary extraction from the Clang driver into a `clang-linker-wrapper`
  tool, matching LLVM upstream's offload design. Device images carry target triple, architecture,
  and offload-kind metadata as structured fields rather than embedded in section names. This enables
  third-party host compilers to participate in SYCL compilation.
- Key detail for libkdl: The `llvm-offload-binary` format encodes `{target_triple, arch, offload_kind,
  binary_blob}` per image — a clean schema that libkdl could adopt or extend as its "kernel descriptor"
  format. The transition from bundle-sections to structured metadata is ongoing in the 2024–2025
  Intel/LLVM codebase.

### 3. hipSYCL SSCP: Single-Pass SYCL Compilation — AdaptiveCpp Official Docs
- URL: https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/
- Type: docs/architecture
- Date: 2024 (AdaptiveCpp >= 23.10)
- Relevance: 10/10
- Novelty: 10/10
- Summary: Describes the SSCP compiler model that makes AdaptiveCpp fundamentally different from
  DPC++. Stage 1 (compile time): Clang parses the source once, extracts LLVM IR for kernels with
  all SYCL builtins lowered to abstract IR intrinsics (no PTX or SPIR-V yet), and embeds this IR
  in the host binary ELF. Stage 2 (runtime): `llvm-to-backend` JIT-compiles the stored IR to the
  appropriate native ISA (PTX for NVIDIA, SPIR-V for Intel, AMDGCN for AMD ROCm). The resulting
  binary supports not just 3 GPU vendors but 38 AMD architectures, any NVIDIA GPU, and any Intel GPU —
  all from a single compilation step that takes only ~15% longer than a plain C++ host compile.
- Key detail for libkdl: This is the purest form of "compile once, dispatch anywhere." The IR
  embedded in the ELF *is* libkdl's "universal kernel IR" — SSCP proves the model is production-
  viable. libkdl can treat SSCP-compiled binaries as first-class inputs, extracting the embedded
  IR and managing dispatch without the SYCL runtime layer.

### 4. Adaptivity in AdaptiveCpp: Runtime JIT Kernel Specialization — IWOCL 2025
- URL: https://dl.acm.org/doi/10.1145/3731125.3731127
- Type: paper
- Date: 2025 (IWOCL/SYCL '25, 13th International Workshop on OpenCL and SYCL)
- Relevance: 9/10
- Novelty: 9/10
- Summary: AdaptiveCpp's SSCP JIT is extended with a runtime adaptivity framework that
  auto-specializes kernels based on runtime information: work-group sizes, pointer argument
  alignments, kernel argument values. The specialization engine hooks into the JIT pipeline before
  backend lowering, generating optimized variants and caching them in a persistent kernel cache.
  This achieves 30% geometric mean speedup over CUDA, 44% over HIP, and 23% over oneAPI across
  a set of mini-apps. The 2025 paper marks AdaptiveCpp as the first SYCL implementation doing
  automatic JIT-time kernel specialization.
- Key detail for libkdl: The persistent kernel cache (JIT-compiled IR → native ISA, keyed by
  {kernel_name, target_arch, specialization_args}) is a design libkdl needs to replicate. The paper
  demonstrates that specializing at dispatch time (not compile time) closes the performance gap
  vs. vendor-tuned native code — validating libkdl's JIT-dispatch thesis.

### 5. Experiences Building an MLIR-Based SYCL Compiler — CGO 2024
- URL: https://arxiv.org/abs/2312.13170
- ACM DL: https://dl.acm.org/doi/10.1109/CGO57630.2024.10444866
- Artifact: https://zenodo.org/records/10410758
- Type: paper
- Date: 2024-03 (CGO 2024, IEEE/ACM)
- Relevance: 9/10
- Novelty: 9/10
- Summary: Intel/Codeplay collaboration (Tiotto, Pérez, Tsang, Sommer, Oppermann, Lomüller, Goli,
  Brodman) introduces an MLIR dialect for SYCL that preserves high-level semantics across the
  host/device boundary, deferring lowering to LLVM IR until after cross-boundary optimization. The
  key insight: traditional SYCL compilers (DPC++, AdaptiveCpp) lower to LLVM IR immediately after
  Clang AST, losing buffer aliasing semantics, ND-range structure, and host-side constants that
  are visible to the kernel call site. The MLIR-based compiler retains this information and achieves
  up to 4.3x speedup (geometric mean 1.18x) over DPC++ on Polybench benchmarks running on Intel
  Data Center GPU Max 1100 (Ponte Vecchio).
- Key detail for libkdl: The paper demonstrates that a dispatch infrastructure that preserves
  kernel call context (argument types, work-group sizes, access patterns) can dramatically improve
  generated code. libkdl's dispatch layer sits at the same semantic level as the SYCL runtime
  call site — it has access to all this information and could pass it to a JIT-specialization
  stage analogous to AdaptiveCpp's adaptivity framework or this MLIR optimization pass.

### 6. SYCL Fat Binary with PTX Backend — Intel Community Thread
- URL: https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/Fat-binary-with-PTX-backend/m-p/1287121
- Type: community/forum
- Date: 2023–2024
- Relevance: 7/10
- Novelty: 6/10
- Summary: Practical discussion of combining SPIR-V (JIT) + PTX (AOT) in a single DPC++ fat binary
  using `-fsycl-targets=spir64,nvptx64-nvidia-cuda`. Confirms that the DPC++ runtime selects between
  embedded device images based on runtime-discovered device type: NVIDIA devices get the PTX/CUBIN
  image, Intel/AMD OpenCL devices get the SPIR-V image. The `-fsycl-embed-ir` flag adds the LLVM IR
  bitcode to the fat binary alongside the SPIR-V for JIT re-use.
- Key detail for libkdl: The multi-target invocation `clang++ -fsycl -fsycl-targets=spir64,nvptx64-nvidia-cuda`
  is exactly the compilation model libkdl should support for ingesting DPC++ fat binaries. The runtime
  device-image selection logic (iterate registered images, match arch string to device caps) is the
  reference implementation libkdl must generalize and make framework-independent.

### 7. Unified Runtime (UR) — Intel/LLVM Official Docs
- URL: https://intel.github.io/llvm/design/UnifiedRuntime.html
- Type: docs
- Date: 2024–2025
- Relevance: 8/10
- Novelty: 7/10
- Summary: The Unified Runtime (UR) replaced DPC++'s original Plugin Interface (PI) as the
  abstraction layer between the vendor-neutral `libsycl` and device-specific adapter libraries.
  UR defines a C API; adapters implement it for OpenCL, Level Zero (Intel), CUDA (NVIDIA),
  HIP (AMD), and native CPU. Each Plugin object owns a `ur_adapter_handle_t`. Device discovery
  enumerates adapters, queries each for available devices, and builds a unified device list.
  Kernel launch goes through `urEnqueueKernelLaunch()` which dispatches to the adapter's native
  launch path. As of oneAPI 2025.3, CUDA/HIP adapters are no longer distributed in binary form
  and must be built from source.
- Key detail for libkdl: UR is the most complete existing reference implementation of "vendor-agnostic
  GPU kernel dispatch." Its adapter plugin architecture (dynamic `.so` discovery at runtime, common
  C API, per-adapter device enumeration) is the proven design pattern libkdl should adopt at the
  kernel-image level rather than the full programming-model level.

### 8. SYCL-Bench 2020: Benchmarking SYCL 2020 on AMD, Intel, and NVIDIA GPUs — IWOCL 2024
- URL: https://dl.acm.org/doi/10.1145/3648115.3648120
- Type: paper
- Date: 2024-04 (IWOCL/SYCL '24, 12th International Workshop on OpenCL and SYCL)
- Relevance: 7/10
- Novelty: 6/10
- Summary: Systematic benchmark of SYCL 2020 feature coverage across three GPU vendors. Evaluates
  six SYCL 2020 features: unified shared memory (USM), reduction kernels, specialization constants,
  group algorithms, in-order queues, and atomics. Results: Intel Gen-based iGPUs achieve up to 75.7%
  architectural efficiency; AMD RX 6700 XT dGPU reaches 51.7%; NVIDIA and Intel Xe-based GPUs show
  lower efficiency (23–52%). Highlights that SYCL 2020 feature support is uneven across vendor
  implementations, particularly for group algorithms on non-Intel hardware.
- Key detail for libkdl: The efficiency variance (75.7% vs 23.4% on different Intel GPU generations)
  shows that vendor-agnostic dispatch is not sufficient alone — the dispatch layer needs to select
  the *best* available kernel variant (or specialize at JIT time) rather than just *any* compatible
  kernel. This quantifies the performance gap that libkdl's dispatch heuristics must close.

### 9. Comparing Performance and Portability: CUDA vs SYCL on NVIDIA, AMD, Intel GPUs — arXiv 2023
- URL: https://arxiv.org/abs/2309.09609
- Type: paper
- Date: 2023-09 (published in Future Generation Computer Systems, 2025)
- Relevance: 7/10
- Novelty: 6/10
- Summary: Costanzo et al. compare a CUDA bioinformatics kernel (SW#, protein database search)
  with its SYCL/DPC++ port across NVIDIA (RTX 2070, RTX 3090), AMD (RX 6700 XT, Vega 6 iGPU),
  and Intel (UHD 630/770, Arc Xe) GPUs. Key finding: CUDA and SYCL achieve equivalent performance
  on NVIDIA (37–52% architectural efficiency). On AMD and Intel, SYCL reaches similar efficiency
  in 3 of 4 test cases. Intel Gen iGPUs outperform expected baseline (75.7% efficiency). Xe-based
  discrete GPUs underperform due to driver immaturity (23.4%). Hybrid CPU+GPU configurations
  show variable portability limited by workload distribution strategy, not language.
- Key detail for libkdl: Demonstrates that a single SYCL source compiled for all three vendors
  achieves near-native performance on 3 out of 4 tested configurations. The outlier (Intel Xe
  discrete) is a driver maturity issue, not an architectural flaw in SYCL's dispatch model. This
  gives libkdl a baseline: dispatching the same kernel to any of these GPUs via SYCL fat binary
  should lose at most ~10–15% vs native, except on immature drivers.

---

## Synthesis: What SYCL Multi-Target Compilation Means for libkdl

### Two Compilation Models, One Dispatch Problem

DPC++ and AdaptiveCpp expose two endpoints of the same design space:

| Property | DPC++ (SMCP) | AdaptiveCpp SSCP |
|---|---|---|
| Compilation passes | N passes (one per target) | 1 pass |
| IR in binary | Target-native (PTX/SPIR-V/CUBIN) | Backend-independent LLVM IR |
| Backend selection | Compile time (`-fsycl-targets`) | Runtime (JIT on first dispatch) |
| Binary portability | Fixed to compiled targets | Universal (any supported backend) |
| Cold-start overhead | None (AOT) | JIT latency on first kernel launch |
| Kernel cache | Not applicable | Persistent cache keyed by arch+args |
| Vendor coverage | Whatever `-fsycl-targets` lists | All AdaptiveCpp backends by default |

libkdl should support ingesting *both* formats:
- DPC++ fat binaries: extract device images by arch-string, dispatch to matching UR adapter
- AdaptiveCpp SSCP binaries: extract embedded LLVM IR, JIT-compile to target on demand

### Runtime Dispatch Architecture

The DPC++ runtime's dispatch path is:
1. At binary load: global ctor calls `__sycl_register_lib()` with a `pi_device_binary` array
2. At queue submission: runtime iterates registered images, scores each against available device
3. Selection: first image where `image.target_arch` matches device capabilities (exact or JIT-fallback)
4. Execution: selected image handed to UR adapter's `urEnqueueKernelLaunch()`

libkdl generalizes step 3: instead of SYCL's sequential scan with binary-compatible exact match,
libkdl can implement scored selection (capability overlap, arch generation distance, kernel
specialization score) and lazy JIT-compilation of IR-format images.

### Key Design Patterns to Adopt from SYCL

1. **`pi_device_binary_struct` schema**: The `{target_string, binary_blob, kernel_table,
   spec_constant_table}` tuple is a clean kernel packaging unit. libkdl's `kdl_image_t` should
   be structurally similar.

2. **UR adapter pattern**: UR's `ur_adapter_handle_t` + per-adapter device enumeration + common
   `urEnqueueKernelLaunch()` is the reference "vendor-agnostic dispatch" API. libkdl can be
   positioned as UR minus the SYCL programming model overhead.

3. **Persistent JIT cache**: AdaptiveCpp's kernel cache (keyed by arch + specialization args)
   eliminates repeated JIT latency. libkdl must implement an equivalent for LLVM IR images.

4. **ONEAPI_DEVICE_SELECTOR semantics**: The env-var-driven backend filter (`cuda:*`, `hip:*`,
   `level_zero:*`) is a user-facing override mechanism libkdl should expose as `KDL_DEVICE_SELECTOR`.

### Risks and Gaps

- **SSCP JIT latency**: First-launch JIT cost is measurable on cold runs. AdaptiveCpp mitigates
  with persistent cache. libkdl needs the same.
- **DPC++ CUDA/HIP adapters now source-only**: As of oneAPI 2025.3, CUDA and HIP UR adapters are
  not distributed as binaries — they must be compiled from source. This is a deployment friction
  point for multi-vendor dispatch in production.
- **Spec constant lowering**: SYCL specialization constants are lowered differently per backend
  (metadata-driven for SPIR-V, pragma-driven for NVPTX). A unified dispatch layer must handle
  all lowering paths.
- **SYCL 2020 feature coverage gaps**: SYCL-Bench shows uneven group algorithm support across
  vendors. libkdl cannot assume kernel semantic equivalence across backends even for "portable" code.
- **No standardized kernel naming**: SYCL uses name-mangled C++ class types as kernel identifiers.
  libkdl needs a stable, vendor-agnostic kernel naming scheme (e.g., hash of kernel IR).

---

## Connections to libkdl

| SYCL mechanism | libkdl analog |
|---|---|
| `pi_device_binary_struct` | `kdl_image_t` kernel descriptor |
| `clang-offload-wrapper` | `kdl_pack()` bundler |
| UR adapter plugin | `kdl_backend_t` dispatch plugin |
| `urEnqueueKernelLaunch()` | `kdl_dispatch()` |
| AdaptiveCpp SSCP IR embedding | libkdl universal IR image type |
| Persistent SSCP kernel cache | libkdl dispatch cache (`~/.kdl/cache/`) |
| `ONEAPI_DEVICE_SELECTOR` | `KDL_DEVICE_SELECTOR` env var |
| `sycl-post-link` kernel splitting | `kdl_split()` per-kernel image extraction |

SYCL is the most mature prior art for vendor-agnostic GPU kernel dispatch. libkdl's contribution
is stripping the SYCL programming model layer and exposing the dispatch infrastructure as a
standalone, framework-independent library — effectively making the UR + kernel-image machinery
available to kernels written in any language (CUDA C, HIP, ISPC, hand-written PTX, etc.).
