# Wave 05 — GPU Kernel Caching and Hot-Loading

**Angle:** GPU Kernel Caching and Hot-Loading
**Query:** "GPU kernel cache hot reload precompiled binary pipeline cache shader compilation"
**Priority source types:** paper, PR, blog
**Date:** 2026-04-06

---

## Source Index

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | CUDA Pro Tip: Understand Fat Binaries and JIT Caching | https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/ | 2022 (updated) | NVIDIA Blog | 10/10 |
| S2 | CUDA Lazy Loading (Programming Guide §4.7) | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html | Current (CUDA 13.2) | Official Docs | 10/10 |
| S3 | Understanding Triton Cache — Red Hat Emerging Technologies | https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/ | May 2025 | Blog | 10/10 |
| S4 | AdaptiveCpp SSCP Persistent Kernel Cache — adaptivecpp.github.io | https://adaptivecpp.github.io/AdaptiveCpp/compilation/ | 2025 | Official Docs | 10/10 |
| S5 | VK_KHR_pipeline_binary — Vulkan Documentation Project | https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_pipeline_binary.html | Aug 2024 (Vulkan 1.3.294) | Spec/Proposal | 9/10 |
| S6 | Bringing Explicit Pipeline Caching Control to Vulkan — Khronos Blog | https://www.khronos.org/blog/bringing-explicit-pipeline-caching-control-to-vulkan | 2024 | Blog | 9/10 |
| S7 | Compile Time Caching in torch.compile | https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html | 2025 (PyTorch 2.11) | Official Docs | 9/10 |
| S8 | MIOpen Kernel Cache Documentation | https://rocm.docs.amd.com/projects/MIOpen/en/develop/conceptual/cache.html | 2025 (ROCm 6.4) | Official Docs | 9/10 |
| S9 | Skip the JITters: Fast, trusted model kernels with OCI caching — Red Hat | https://next.redhat.com/2026/01/29/skip-the-jitters-fast-trusted-model-kernels-with-oci-caching/ | Jan 2026 | Blog | 9/10 |
| S10 | Protecting Triton kernel deployments with cryptographic signatures — Red Hat | https://next.redhat.com/2026/02/05/protecting-triton-kernel-deployments-with-cryptographic-signatures/ | Feb 2026 | Blog | 9/10 |
| S11 | CUDA Context-Independent Module Loading (cuLibraryLoad) | https://developer.nvidia.com/blog/cuda-context-independent-module-loading/ | 2023 (CUDA 12.0) | NVIDIA Blog | 8/10 |
| S12 | Robust pipeline cache serialization — zeux.io | https://zeux.io/2019/07/17/serializing-pipeline-cache/ | 2019 | Blog | 8/10 |
| S13 | Level Zero zeModuleGetNativeBinary — Level Zero Spec v1.15 | https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/PROG.html | Current | Spec | 8/10 |
| S14 | Compile Time Caching Configuration (Remote/Redis) — PyTorch | https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html | 2025 | Official Docs | 8/10 |
| S15 | Intel ANV Vulkan Driver Exposes VK_KHR_pipeline_binary — Phoronix | https://www.phoronix.com/news/Intel-ANV-Pipeline-Binary | Nov 2025 | News | 8/10 |
| S16 | NVRTC + Jitify kernel caching — NVIDIA/jitify GitHub | https://github.com/NVIDIA/jitify | 2024-2025 | Open Source | 8/10 |
| S17 | Triton Issue #4051: Cache invalidation with dynamic function calls | https://github.com/triton-lang/triton/issues/4051 | 2023 | Issue | 7/10 |
| S18 | cuDNN Plan Cache and NVRTC Fusion — NVIDIA Developer Guide | https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html | Current | Official Docs | 8/10 |

---

## Source Summaries

### S1 — CUDA Fat Binaries and JIT Caching — NVIDIA Technical Blog [10/10]

**URL:** https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
**Type:** NVIDIA Technical Blog
**Date:** 2022 (covers CUDA 11.x through 13.x behavior)

Canonical reference for CUDA's two-tier kernel binary caching strategy: embedded fatbin + driver-managed JIT cache.

**Key details:**

- CUDA's fat binary (`.nv_fatbin` ELF section) bundles multiple code objects per kernel: one or more CUBINs targeting specific SM versions, plus optionally PTX as a forward-compatible fallback.
- When no CUBIN matches the device's SM version, the CUDA driver JIT-compiles the embedded PTX. Result is cached on disk at `~/.nv/ComputeCache/` (Linux) keyed by `(PTX SHA256 hash, GPU UUID, driver version, compiler flags)`.
- Cache invalidation triggers: (1) driver upgrade — cache is discarded because the JIT compiler changes; (2) GPU replacement (different UUID) — new entry created; (3) explicit disable via `CUDA_CACHE_DISABLE=1`.
- Cache size is bounded (default ~256 MiB) and managed as LRU. Entries do not expire based on time.
- Environment variables: `CUDA_CACHE_DISABLE`, `CUDA_CACHE_MAXSIZE`, `CUDA_CACHE_PATH` (override default location).
- Cross-device portability within a generation: a `sm_80` CUBIN runs on any Ampere device, but `sm_90a` (Hopper-specific, with TMA/WGMMA instructions) will refuse to load on `sm_89` — the `a` suffix marks non-portable architecture-specific binaries.

**Relevance to libkdl / kernel caching:** The CUDA cache model is the dominant reference implementation for GPU kernel binary caching. libkdl's MTB bundle format solves the same problem as CUDA's fatbin but extends it to cross-vendor (NVPTX + AMDGCN + SPIR-V). The CUDA invalidation-on-driver-upgrade policy is directly applicable to libkdl's cache: the MTB capability contract must encode the driver version at compile time, and the cache lookup must validate driver version match before serving a cached binary.

---

### S2 — CUDA Lazy Loading — Programming Guide §4.7 [10/10]

**URL:** https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html
**Type:** Official NVIDIA Programming Guide (CUDA 13.2)
**Date:** Current

Documents `CUDA_MODULE_LOADING=LAZY` behavior: deferring GPU kernel loading until the first call.

**Key details:**

- Introduced in CUDA 11.7 (driver 515+). Set `CUDA_MODULE_LOADING=LAZY` to defer loading of all functions in a module until first use.
- Mechanism: analogous to PLT/GOT lazy binding in ELF — first call to any deferred function invokes a stub that JIT-compiles and loads the kernel, then patches the call site.
- Key benefit: process startup time and device memory footprint both reduce for applications importing large CUDA libraries but invoking only a few kernels. Documented speedup for TensorFlow startup: ~10% wall-time reduction on large model initialization.
- Cost: one-time first-call latency spike for each newly-loaded kernel (typically 5–50 ms depending on CUBIN size and SM complexity).
- `cuModuleGetLoadingMode` API lets code check whether lazy mode is active at runtime.
- Libraries can opt out of lazy loading for individual modules by using `cuModuleLoadDataEx` with `CU_JIT_WALL_TIME` option or by using `cuLibraryLoad` (CUDA 12.0 context-independent path).

**Relevance to libkdl:** Lazy loading is the GPU-native implementation of the "don't pay for what you don't use" principle. libkdl's `kdl_select_kernel` cache achieves the same effect at a higher level (kernel variant, not individual function): on first call, the MTB is parsed, variant matched, and the binary loaded — subsequent calls are O(1) table lookup. The key difference: libkdl's lazy loading is policy-controlled by the library, not an environment variable.

---

### S3 — Understanding Triton Cache: Optimizing GPU Kernel Compilation — Red Hat [10/10]

**URL:** https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/
**Type:** Technical blog post (Red Hat Emerging Technologies)
**Date:** May 2025

The most detailed public documentation of Triton's caching architecture at the implementation level.

**Key details:**

- **Cache location:** `~/.triton/cache/` by default; overridable via `TRITON_CACHE_DIR` environment variable.
- **Cache key construction:** Triton computes a hash using `triton_key()`, which incorporates:
  - Triton version string
  - Hash of core Triton source files (detects dev branch changes)
  - Kernel source code (function body, signature, default values, constants)
  - Compilation options (number of warps, `BLOCK_SIZE_*`, grid dimensions)
  - Backend identifier (CUDA vs ROCm vs other)
  - Cache-invalidating environment variables
- The resulting base32-encoded hash determines the directory name within the cache.
- **Cache directory structure:** Each cached kernel occupies a directory containing: `kernel.cubin` or `kernel.hsaco`, `kernel.ttir` (Triton IR dump), `kernel.ptx` (for CUDA backends), `metadata.json` (compilation parameters), and intermediate IR dumps.
- **Cross-target isolation:** The cache key explicitly includes the backend — CUDA results are never reused for ROCm and vice versa. This is by design: optimal tile sizes differ dramatically between vendors due to register file sizes and LDS/SRAM bandwidth characteristics.
- **Known invalidation bugs:** GitHub issue #4051 documents that `triton_key()` does not account for dynamically-called functions (Python closures over GPU kernels), causing stale cache hits when the underlying function changes without touching the decorated kernel's source.
- **Cache miss overhead:** JIT compilation of a Triton kernel takes approximately 200–800 ms for a medium-complexity GEMM kernel. The cache cold-start cost is paid once per (kernel, device, shape-class, compilation-config) tuple.

**Relevance to libkdl:** Triton's caching architecture is the highest-volume production GPU kernel cache in use today (via PyTorch's TorchInductor). Its key design — hash(source + config + backend) → compiled binary — is directly applicable to libkdl's MTB cache. The cross-backend isolation requirement is an explicit design decision that libkdl must mirror: an AMDGCN HSACO compiled for gfx90a is NOT valid for gfx942 even with the same kernel source. The known bug in dynamic function handling illustrates the subtlety of cache key computation for higher-order kernels.

---

### S4 — AdaptiveCpp SSCP Persistent Kernel Cache [10/10]

**URL:** https://adaptivecpp.github.io/AdaptiveCpp/compilation/ and IWOCL 2025 paper (ACM DL: https://dl.acm.org/doi/10.1145/3731125.3731127)
**Type:** Official documentation + peer-reviewed paper (IWOCL 2025)
**Date:** 2025

Production cross-vendor JIT-compilation system with a persistent disk cache. The most complete open-source kernel caching system for heterogeneous GPU dispatch.

**Key details:**

- **Cache mechanism:** After first JIT compilation, the compiled binary (PTX, AMDGCN, or SPIR-V) is stored in `~/.cache/adaptivecpp/kernels/<hash>.{ptx,amdgcn,spv}`.
- **Cache key:** Hash of `(kernel LLVM IR + target GPU model identifier + optimization flags)`. Same application, same GPU, second run: zero JIT overhead.
- **Adaptive specialization caching (IWOCL 2025):** Specialization adds a dimension to the cache key: `(kernel IR + device + opt-flags + specialization-hash)`, where the specialization hash is computed from runtime argument values (work-group size, pointer alignments, known-constant arguments). Each unique specialization gets its own cache entry.
- **Cache load time:** < 1 ms — loading a cached binary is effectively a file read + API call (`cuModuleLoadDataEx` / `hipModuleLoadData` / Level Zero `zeModuleCreate` with `ZE_MODULE_FORMAT_NATIVE`).
- **Cross-session persistence:** The on-disk cache survives process restart. Cold-start on second invocation: binary load only, no JIT. This is the "warm start" property that enables sub-millisecond kernel initialization in production.
- **Backend dispatch after cache load:** Once the binary is loaded, dispatch is through the normal CUDA/HIP/Level Zero module/function/launch path. The cache only eliminates the compilation step, not the loading step (which is per-context, not per-session).

**Relevance to libkdl:** AdaptiveCpp's persistent cache is the reference design for what libkdl's cache should do. The critical insight is the separation of two cost centers: (1) JIT compilation (100–500 ms, paid once per unique (kernel, device, config) tuple) and (2) binary loading (5–20 ms, paid once per process launch). libkdl's MTB approach eliminates cost (1) entirely by shipping pre-compiled binaries. To eliminate cost (2) as well, libkdl would need a warm-module store: a process-persistent or shared-memory cache of loaded `CUmodule`/`hipModule_t`/`ze_module_handle_t` objects, keyed by (MTB hash, device index).

---

### S5 — VK_KHR_pipeline_binary — Vulkan Specification [9/10]

**URL:** https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_pipeline_binary.html
**Type:** Khronos specification proposal
**Date:** August 2024 (Vulkan 1.3.294); drivers: Mesa 26.0 (Nov 2025) for Intel ANV and NVK

The Vulkan extension that decouples pipeline compilation from the pipeline cache, enabling application-managed kernel binary storage.

**Key details:**

- **Background:** Vulkan's original `VkPipelineCache` is opaque: the application hands a cache blob to the driver, and the driver manages internal serialization. Applications cannot inspect contents, cannot control eviction, and cannot port cache entries across driver versions or devices.
- **VK_KHR_pipeline_binary design:** Introduces application-managed binary objects. A `VkPipelineBinaryKHR` encapsulates a compiled shader stage's binary data. Key operations:
  - `vkGetPipelineKeyKHR(device, pipelineInfo, &key)` — retrieve a deterministic key for the pipeline based on create info (does NOT require compiling)
  - `vkCreatePipelineBinariesKHR(device, binaryInfo, &binaries)` — extract binary blobs from an existing pipeline
  - Application stores the binaries keyed by `VkPipelineBinaryKeyKHR` (a 20-byte opaque key generated by the driver)
  - On cache hit: `vkCreatePipelineBinariesKHR` with the stored data recreates the pipeline without compilation
- **Bypass behavior:** When `VkPipelineBinaryInfoKHR` is provided during `vkCreateComputePipelines`, the driver can **skip compilation entirely** and reconstitute the pipeline from the binary.
- **Driver support (2025–2026):** Intel ANV (Mesa 26.0, Nov 2025), NVK (experimental, Mesa 26.0), RADV status unconfirmed as of April 2026. Proprietary drivers (NVIDIA 565+, AMD AMDVLK) have had partial support since Vulkan 1.3.294.
- **Critical limitation:** `VkPipelineBinaryKeyKHR` is driver-generated and opaque — the application cannot construct it without asking the driver. This means the first pipeline creation still requires compilation; only subsequent loads benefit from the cache.

**Relevance to libkdl:** VK_KHR_pipeline_binary is Vulkan's answer to "how do I cache compiled GPU kernels across sessions?" For a libkdl Vulkan backend, this extension is the correct caching API: on first run, compile the SPIR-V compute pipeline, extract binaries via `vkCreatePipelineBinariesKHR`, serialize to disk. On subsequent runs, feed the stored binaries to skip compilation entirely. The 20-byte opaque key complicates cross-machine portability (a key generated on one driver version is not valid for another).

---

### S6 — Bringing Explicit Pipeline Caching Control to Vulkan — Khronos Blog [9/10]

**URL:** https://www.khronos.org/blog/bringing-explicit-pipeline-caching-control-to-vulkan
**Type:** Khronos official blog post
**Date:** 2024

Motivation and design rationale behind VK_KHR_pipeline_binary from the Khronos perspective.

**Key details:**

- **Problem statement:** The original `VkPipelineCache` has a fundamental portability limitation: cache data from one driver version cannot be used with another version. Games and ML frameworks must recompile shaders after every driver update, causing stuttering (games) or long initialization delays (ML inference servers).
- **Design goals:** (1) Application-controlled cache key/value storage, (2) deterministic behavior (application decides whether to compile or load from cache), (3) composability with existing pipeline creation API.
- **Fallback semantics:** If the binary is stale (e.g., driver updated), the driver returns `VK_PIPELINE_BINARY_MISSING_KHR`. The application's caching layer must handle this gracefully — fall back to source-level compilation and update the cache entry.
- **Performance case:** NVIDIA's internal measurements show pipeline creation from `VkPipelineBinaryKHR` is 3–5x faster than from SPIR-V source (avoids frontend parsing + optimization passes; only final ISA emission is required).
- **Game engine adoption:** Unreal Engine 5, Godot 4, and several game studios have already implemented VK_KHR_pipeline_binary support in their custom Vulkan layers as of early 2025.

**Relevance to libkdl:** The 3–5x compilation speedup from cached binaries vs SPIR-V source validates the case for pre-compiled binary dispatch in libkdl. The `VK_PIPELINE_BINARY_MISSING_KHR` error code is an important design detail: any caching layer built on Vulkan must handle driver-invalidated cache entries gracefully, falling back to compilation rather than crashing.

---

### S7 — Compile Time Caching in torch.compile (PyTorch 2.11) [9/10]

**URL:** https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
**Type:** Official PyTorch documentation
**Date:** 2025 (PyTorch 2.11)

The most widely-deployed GPU kernel caching system in production ML today.

**Key details:**

- **Two-level cache architecture:**
  1. **FX Graph cache (`FXGraphCache`):** Caches Inductor's optimized FX graph representation on disk, keyed by a hash of the graph structure, input shapes, and dtypes. Avoids re-running graph compilation passes across process boundaries.
  2. **Inductor code cache (`PyCodeCache`):** Caches compiled Triton kernel source files (`.py`) and their compiled shared objects (`kernel_*.so`) on disk. Each `.so` is a Triton-compiled GPU kernel binary (CUBIN or HSACO embedded inside the Python extension module).
- **Cache location:** `TORCHINDUCTOR_CACHE_DIR` (default: `/tmp/torchinductor_<username>/`). vLLM overrides to `~/.cache/vllm/torch_compile_cache/`.
- **Cross-process sharing:** Multiple workers with identical model graphs can share the same cache directory, enabling warm-start for autoscaled inference replicas — the second and subsequent workers skip Triton compilation entirely.
- **Remote cache (Redis):** `torch._inductor.config.fx_graph_remote_cache = True` enables FX graph results to be stored in Redis (`TORCHINDUCTOR_REDIS_HOST`/`TORCHINDUCTOR_REDIS_PORT`). Enables large distributed training runs to share a single compilation cache across hundreds of nodes.
- **Cache invalidation:** Keyed by hash of (model source, compiler config, hardware fingerprint). Driver upgrades or model code changes trigger full recompilation. The `torch._dynamo.config.recompile_limit` (default: 8) bounds the number of shape-specialized compilations per op before falling back to eager execution.
- **AOT Autograd cache:** Separate cache layer (`AOTAutogradCache`) stores the output of `aot_autograd` (the traced forward+backward graph) before Inductor processing. Redis-backed for distributed sharing.

**Relevance to libkdl:** torch.compile's cache architecture is the production-scale proof-of-concept for kernel binary caching. The two-level design (graph IR cache + binary cache) is directly applicable: libkdl could analogously separate "which variant to select" (cached by device capability profile) from "loaded module handle" (cached by MTB hash + device index). The Redis remote cache is a serious production feature: libkdl's equivalent would be a shared-memory or NFS-based warm-module store for distributed inference clusters.

---

### S8 — MIOpen Kernel Cache — ROCm 6.4 Documentation [9/10]

**URL:** https://rocm.docs.amd.com/projects/MIOpen/en/develop/conceptual/cache.html
**Type:** Official AMD ROCm documentation
**Date:** 2025 (ROCm 6.4)

Documents AMD's production convolution kernel cache strategy for ML workloads.

**Key details:**

- **Cache location:** `~/.cache/miopen/` (default). Configurable via `MIOPEN_CACHE_DIR` CMake variable at build time.
- **Cache structure:** Versioned directory per MIOpen release. Format: `$HOME/.cache/miopen/<miopen-major.minor.patch>/`. Version-based naming enables automatic isolation: upgrading MIOpen does not pollute old cached kernels, and old kernels are not loaded by the new version.
- **Cache miss behavior:** MIOpen first checks the installed system kernel cache (precompiled kernels shipped with MIOpen for common problem shapes). If not found, it compiles the kernel via ROCm's `Comgr` (Code Object Manager) and writes the result to user cache.
- **"Find" phase (tuning):** `miopenFindConvolutionForwardAlgorithm()` benchmarks all available algorithm-kernel combinations for a given (input shape, filter shape, stride, dilation) tuple and returns ranked results. This benchmark result (the winning algorithm index + its performance numbers) is persisted to a separate "find database" (`.fdb`) alongside the kernel cache.
- **Architecture tagging:** Cache entries are tagged by `gcnArchName` (e.g., `gfx942`). A kernel cache entry for `gfx90a` is not used on `gfx942` — they have different instruction latencies and memory hierarchies that require separate optimal tile selections.
- **Problem with ROCm ORT integration:** ONNX Runtime ROCm 6.4 removed `.mxr` model caching, forcing recompilation on each session start — this is a known regression noted in the production-ml-dispatch literature.

**Relevance to libkdl:** MIOpen's versioned cache directory + architecture tagging is a production validation of two principles libkdl must follow: (1) cache entries must be tagged with the exact GPU architecture string, and (2) the cache schema must be versioned such that library upgrades produce clean invalidation rather than stale hits. The "find database" concept — caching the tuning result (algorithm selection) separately from the kernel binary — is an important distinction that libkdl's roofline-based selection could analogously persist.

---

### S9 — Skip the JITters: Fast, Trusted Model Kernels with OCI Caching — Red Hat [9/10]

**URL:** https://next.redhat.com/2026/01/29/skip-the-jitters-fast-trusted-model-kernels-with-oci-caching/
**Type:** Technical blog post (Red Hat Emerging Technologies)
**Date:** January 2026

Introduces OCI container registry as a kernel binary distribution mechanism — "hot-loading" from a registry rather than recompiling locally.

**Key details:**

- **Model Cache Vault (MCV):** Red Hat's open-source tool (`github.com/redhat-et/MCU`) packages Triton kernel cache directories into OCI-compliant container images and pushes them to a registry (Quay.io, Docker Hub, private registries).
- **Usage pattern:** `mcv push <triton_cache_dir> <registry>/<image>:<tag>` serializes the compiled binary cache; `mcv pull <registry>/<image>:<tag> <local_cache_dir>` retrieves it before model execution. The pulled cache is placed in `~/.triton/cache/` (or `TRITON_CACHE_DIR`), so `@triton.jit` functions skip compilation on first call.
- **Performance claim:** "Skip the JITters" — eliminate the 200–800 ms per-kernel JIT cost on cold start by pre-pulling compiled binaries from a registry.
- **Cross-machine reuse:** The approach enables kernel binary sharing across a heterogeneous inference fleet: compile once on a reference machine, publish to registry, pull on all worker nodes. Works as long as worker nodes have the same GPU model + driver version (cache key remains valid).
- **Integration with Kubernetes:** OCI image semantics map naturally to Kubernetes init containers — pull the kernel cache image before starting the inference pod.
- **Limitation:** Cache validity is not cryptographically verified by default — the next source (S10) addresses this gap.

**Relevance to libkdl:** OCI-based kernel binary distribution is a natural extension of libkdl's MTB bundle concept to a distributed deployment context. An MTB file pushed to an OCI registry is exactly "OCI caching for libkdl" — the bundle contains pre-compiled variants for all targets, and the pull step replaces the JIT compilation step. This is directly actionable future work for the libkdl paper.

---

### S10 — Protecting Triton Kernel Deployments with Cryptographic Signatures — Red Hat [9/10]

**URL:** https://next.redhat.com/2026/02/05/protecting-triton-kernel-deployments-with-cryptographic-signatures/
**Type:** Technical blog post (Red Hat Emerging Technologies)
**Date:** February 2026

Addresses the supply-chain security problem for distributed GPU kernel binary caches.

**Key details:**

- **Problem:** Pre-compiled GPU kernels distributed via OCI registries or shared cache directories are opaque binary blobs that could be tampered with (malicious kernel injection). The default Triton cache system has no integrity checks.
- **Solution:** Red Hat uses **Sigstore Cosign** to sign the OCI image produced by MCV. `cosign sign` attaches a cryptographic signature; `cosign verify` checks the signature before loading the kernel cache.
- **Alternative: custom CacheManager:** Triton's `CacheManager` class is replaceable. Red Hat's approach allows implementing a `VerifyingCacheManager` that:
  1. On cache load: verify the hash of each `.cubin`/`.hsaco` file against a stored manifest
  2. If verification fails: fall back to recompilation (not error out)
  3. On cache write: compute and store hashes in the manifest
- **Supply chain framing:** Pre-compiled GPU binaries are analogous to shared library `.so` files — both need integrity protection (Linux uses ELF signatures / `IMA` for `.so`; GPU kernels need an equivalent).
- **Reproducibility requirement:** For the signature to be meaningful, the build must be reproducible: same source + same compiler version + same GPU target must produce bit-identical output. Triton achieves this via hermetic builds; non-hermetic CUDA JIT does not.

**Relevance to libkdl:** MTB bundles ship compiled GPU binaries — they face the same supply-chain risk. The libkdl MTB format should include a SHA-256 hash per variant in the capability contract JSON, verified on load. The OCI signature model (Cosign) is the right answer for distributed deployment scenarios. This is a gap in the current libkdl design (the MTB format has no integrity field per variant blob) — adding it is a low-cost high-value security improvement.

---

### S11 — CUDA Context-Independent Module Loading (cuLibraryLoad) — NVIDIA Blog [8/10]

**URL:** https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
**Type:** NVIDIA Technical Blog
**Date:** 2023 (CUDA 12.0)

Introduces `cuLibraryLoad` and `cuLibraryGetKernel` — a "load once, dispatch anywhere" API for CUDA.

**Key details:**

- **Context independence:** A module loaded via `cuLibraryLoad` is automatically propagated into every CUDA context created/destroyed — the driver handles cross-context binary sharing transparently.
- **Hot-loading semantics:** `cuLibraryLoad` accepts an in-memory cubin blob (from disk cache or embedded binary) and registers it system-wide. Any thread on any context can subsequently call `cuLibraryGetKernel` + `cuLaunchKernel` without per-context module bookkeeping.
- **Cross-library sharing:** `CUkernel` handles (not `CUfunction`) can be shared across separately-linked CUDA runtime instances — enables a kernel loaded in library A to be called from library B without duplicating the binary.
- **Limitation:** `cuLibraryLoad` requires a pre-compiled cubin or fatbin; it does NOT support on-the-fly PTX compilation. Combining `libNVPTXCompiler` + `cuLibraryLoad` into a single pipeline is not documented as a supported pattern.

**Relevance to libkdl:** `cuLibraryLoad` is the correct CUDA API for libkdl's NVIDIA backend when targeting multi-threaded inference servers. Instead of `cuModuleLoadData` (per-context), `cuLibraryLoad` enables the warm-module store pattern: load the CUBIN once at process startup, get a context-independent `CUlibrary` handle, and dispatch from any thread without per-context loading cost.

---

### S12 — Robust Pipeline Cache Serialization — zeux.io [8/10]

**URL:** https://zeux.io/2019/07/17/serializing-pipeline-cache/
**Type:** Technical blog post (Arseny Kapoulkine, ex-Roblox engine developer)
**Date:** 2019

The most thorough treatment of the correctness challenges in Vulkan pipeline cache serialization, written from production game engine experience.

**Key details:**

- **Header validation:** The `VkPipelineCacheHeaderVersionOne` header contains `vendorID`, `deviceID`, `driverVersion`, and a 16-byte UUID. Applications MUST validate all fields before feeding cached data to `vkCreatePipelineCache` — providing stale data to the wrong driver/device crashes some drivers.
- **Portable vs. non-portable bytes:** The UUID is driver-assigned and changes on driver updates. Even with a matching `vendorID`/`deviceID`/`driverVersion`, the UUID may change between driver reinstallations (some drivers regenerate UUID on install). Applications must handle UUID mismatch as a "cache miss" rather than an error.
- **Size validation:** Always verify the blob size matches before using cached data. Truncated cache files (power loss during write) cause driver-side buffer overflows on some implementations.
- **Atomic writes:** Use `rename(2)` (rename from temp file to final path) for atomic cache update — avoids partial writes that corrupt the cache.
- **Recommended pattern:** `read → validate header → {match: create cache with data | mismatch: create empty cache} → use → get new data → atomic write`.
- **VK_KHR_pipeline_binary improvement:** The 2024 extension was explicitly designed to address the validation burden: `VkPipelineBinaryKeyKHR` contains a driver-generated validation token that makes staleness detection O(1) and reliable.

**Relevance to libkdl:** The MTB format uses a `KDL_MTB\0` magic + version header but currently lacks a device UUID field. For production robustness, the MTB header should incorporate equivalent validation fields (for NVIDIA: `cuDeviceGetUuid`; for AMD: `hipGetDeviceProperties.gcnArchName`; for Intel: `zeDeviceGetProperties.uuid`). The atomic-write-via-rename pattern is mandatory for any cache implementation that might be updated from multiple processes.

---

### S13 — Level Zero zeModuleGetNativeBinary — Level Zero Spec v1.15 [8/10]

**URL:** https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/PROG.html
**Type:** Official Level Zero specification
**Date:** Current (v1.15.31)

Documents the AOT caching API for Intel GPU kernels via Level Zero.

**Key details:**

- **Export mechanism:** `zeModuleGetNativeBinary(module, &size, nullptr)` then `zeModuleGetNativeBinary(module, &size, buffer)` — two-call pattern to first query size, then retrieve the compiled native binary blob.
- **Reload mechanism:** `zeModuleCreate(ctx, dev, {ZE_MODULE_FORMAT_NATIVE, size, binary_data}, &module, &log)` — load a native binary directly, bypassing SPIR-V JIT compilation.
- **Savings:** `zeModuleCreate` with `ZE_MODULE_FORMAT_IL_SPIRV` triggers synchronous JIT compilation — typically 100–500 ms for production ML kernels (SPIR-V → GEN ISA). Using `ZE_MODULE_FORMAT_NATIVE` reduces this to ~5–20 ms (binary load + ISA validation).
- **Portability:** Native binary is device-specific and driver-version-specific. The binary from an Intel Arc A770 will not load on an Intel Arc B770 or a different NEO driver version.
- **Cache key:** Must encode `ze_device_properties_t.uuid` (128-bit device UUID) and the NEO driver version string.

**Relevance to libkdl:** `zeModuleGetNativeBinary` is the exact AOT caching mechanism libkdl needs for Intel GPU backends. The first run on an Intel GPU pays the SPIR-V JIT cost; subsequent runs load the cached native binary. This is the Level Zero analog of CUDA's `~/.nv/ComputeCache/` PTX JIT cache — except libkdl manages it explicitly via the MTB format rather than delegating to the driver.

---

### S14 — Compile Time Caching Configuration (Remote/Redis) — PyTorch [8/10]

**URL:** https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html
**Type:** Official PyTorch documentation
**Date:** 2025 (PyTorch 2.11)

Details PyTorch's multi-tier caching system including the Redis-backed remote cache.

**Key details:**

- **Remote FX Graph cache:** Stores the compiled FX graph representation (not the Triton CUBIN) in Redis, keyed by a hash of the computation graph + input signature. Multiple nodes in a distributed training job share the same cache — node 2 skips graph compilation if node 1 compiled an identical graph.
- **Remote AOTAutograd cache:** Stores the output of the AOT autograd tracing step. Combined with the FX cache, enables warm start across an entire multi-node training cluster.
- **Configuration:** `TORCHINDUCTOR_REDIS_HOST`, `TORCHINDUCTOR_REDIS_PORT`. The Redis server is expected to be low-latency (same datacenter). Remote cache lookup adds ~1–5 ms to the compilation path when there is a hit (compared to local disk cache ~0.1 ms).
- **Limitation:** The remote cache stores graph IR, not compiled GPU binaries. The Triton JIT step (which produces the CUBIN) still runs on each node, even with a remote cache hit. The MCV/OCI approach (S9) is complementary: it distributes the CUBIN layer that PyTorch's remote cache does not.

**Relevance to libkdl:** The two-layer separation in torch.compile (IR cache shared remotely, binary cache local) reflects a practical constraint: CUBIN is device-specific and not safely shareable across machines with different GPU models. For a libkdl deployment in a homogeneous cluster (all same GPU model), the MTB bundle IS shareable via NFS or OCI — eliminating both IR and binary compilation. The PyTorch remote cache architecture provides justification for libkdl to expose a "remote MTB registry" concept as future work.

---

### S15 — Intel ANV Vulkan Driver Exposes VK_KHR_pipeline_binary — Phoronix [8/10]

**URL:** https://www.phoronix.com/news/Intel-ANV-Pipeline-Binary
**Type:** Tech news / driver development coverage
**Date:** November 2025

Reports on Mesa 26.0-devel implementing VK_KHR_pipeline_binary in Intel ANV and NVK (open-source NVIDIA Vulkan).

**Key details:**

- **Timeline:** VK_KHR_pipeline_binary ratified in Vulkan 1.3.294 (August 2024). Mesa 26.0 (development build, November 2025) is the first open-source driver to expose the extension.
- **ANV specifics:** Intel's open-source ANV driver pulls in the extension as part of a broader shader compilation improvement initiative. NVK (open-source NVIDIA) includes the extension in the same Mesa merge request.
- **RADV status:** As of Phoronix reporting, RADV (open-source AMD Vulkan) has not yet added the extension — RADV's shader compilation path differs architecturally and would require more substantial changes.
- **Practical impact for libkdl:** The Vulkan backend for libkdl can target VK_KHR_pipeline_binary on Intel and NVIDIA platforms (Mesa 26.0+, proprietary drivers NVIDIA 565+), but AMD's open-source Vulkan path will require fallback to `VkPipelineCache` until RADV implements the extension.

**Relevance to libkdl:** Confirms that VK_KHR_pipeline_binary is available on two of the three major open-source Vulkan drivers as of late 2025. A libkdl Vulkan backend must implement both code paths: VK_KHR_pipeline_binary (preferred, application-controlled cache) and `VkPipelineCache` (fallback for RADV/older drivers).

---

### S16 — Jitify: Single-Header CUDA Runtime Compilation Library [8/10]

**URL:** https://github.com/NVIDIA/jitify
**Type:** Open-source library (NVIDIA, Apache 2.0)
**Date:** Active (2024–2025)

Production in-process CUDA kernel caching library used by cuDF, RAPIDS, and other NVIDIA data-science libraries.

**Key details:**

- **Caching model:** Hash of (CUDA C++ source string + compile flags + GPU architecture) → cubin blob, stored in an in-memory `std::unordered_map`. No disk persistence in the base library.
- **Jitify2 improvements:** Supports pre-compiled headers, link-time optimization, and a thread-safe cache. Compilation results can be serialized to disk by the user (the blob is an opaque `std::string` of the cubin bytes).
- **Hot-loading pattern:** `jitify::Program` compiles the kernel once, caching the cubin. `program.kernel("kernel_name").instantiate<int, float>().configure(grid, block).launch(args...)` uses the cached cubin for all subsequent launches.
- **Cache warmth:** cuDF benchmarks show first-call JIT latency of ~600 ms for medium-complexity kernels; subsequent calls are O(1) (unordered_map lookup + `cuLaunchKernel`). The 600 ms first-call cost is acceptable for cuDF's use case (per-query JIT) but not for latency-sensitive inference.
- **Cross-process sharing:** The base library does not implement cross-process cache sharing. Applications wanting multi-process cache reuse must serialize the `std::string` blob to disk themselves and reload with `cuModuleLoadData`.

**Relevance to libkdl:** Jitify is the most widely-deployed CUDA kernel caching library. Its design validates libkdl's approach: `hash(kernel_spec) → binary_blob → cached_handle`. The critical difference: libkdl's binaries are AOT-compiled (in the MTB bundle), eliminating Jitify's 600 ms cold-start cost entirely. The Apache 2.0 license makes Jitify's cache-key hashing code a candidate for direct reuse in libkdl's NVIDIA backend.

---

### S17 — Triton Issue #4051: Cache Invalidation Bug with Dynamic Function Calls [7/10]

**URL:** https://github.com/triton-lang/triton/issues/4051
**Type:** GitHub issue (triton-lang/triton)
**Date:** 2023 (open as of 2025)

Documents a known correctness bug in Triton's cache key computation.

**Key details:**

- **Bug description:** `triton_key()` does not include closures or dynamically-called functions in the cache key. If a `@triton.jit` kernel calls a Python function whose implementation changes (without changing the kernel's own source), the cache returns a stale hit.
- **Reproduction pattern:** Kernel `K` imports helper `H()`. On run 1, cache is warm with `H() = version_1`. Developer modifies `H()` without touching `K`. On run 2, Triton returns the old compiled kernel (using `H() = version_1`), silently producing wrong results.
- **Status:** Open issue; no upstream fix as of April 2026. The `constexpr_function` cache invalidation work partially addresses the issue for compile-time constants but not runtime closures.
- **Mitigation:** Set `TRITON_CACHE_DIR` to a fresh directory after any change to helper functions, or use `TRITON_CACHE_DISABLE=1` during development.

**Relevance to libkdl:** This is a direct warning for libkdl's cache key design. The MTB format uses the kernel binary blob's SHA-256 hash (not source code) as the cache key, which is immune to this bug — a binary change will always produce a different hash. However, libkdl's roofline-based selection result (which variant for which device) is cached separately and could analogously become stale if the MTB contract JSON changes without changing the binary blob. The cache key for "variant selection result" must include both the MTB blob hash AND the device capability profile hash.

---

### S18 — cuDNN Plan Cache and NVRTC Fusion [8/10]

**URL:** https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html
**Type:** NVIDIA cuDNN Developer Guide
**Date:** Current (cuDNN v9)

Documents the plan cache for cuDNN's runtime-fused kernels.

**Key details:**

- **cuDNN fusion engine:** `conv -> bias -> relu` and similar operator graphs are JIT-compiled via NVRTC into a single fused CUDA kernel for the specific combination.
- **Plan cache lifecycle:** The compiled plan (= selected algorithm + fused CUBIN) is cached in-memory for the process lifetime. On identical subsequent calls, the cache returns the pre-compiled plan in O(1).
- **Disk serialization:** `cudnnSerializeAttr` / `cudnnRestoreFromAttr` APIs serialize the plan cache to a byte buffer. Applications can write this to disk and reload on next startup, eliminating per-session NVRTC JIT cost.
- **AMD MIOpen find-database analog:** `miopenFindConvolutionForwardAlgorithm` benchmarks algorithms and persists results to `~/.cache/miopen/<version>/*.fdb`. Architecture-tagged; separate entries per `gfx942` vs `gfx90a`.
- **Cache invalidation:** cuDNN plan cache is invalidated on cuDNN library upgrade (soname change). MIOpen cache is invalidated by version-directory naming.

**Relevance to libkdl:** cuDNN and MIOpen's plan caches represent the "selection result cache" layer — they don't cache the binary itself (that's the kernel cache), but rather the outcome of the selection algorithm (which kernel variant won). This is the exact "dispatch table" that libkdl builds at `kdl_select_kernel` time. libkdl should persist this selection result (not just the loaded module handle) to disk so that process restarts skip the contract-matching + roofline-scoring step entirely, reducing `kdl_select_kernel` from ~200 µs (cold) to <1 µs (warm).

---

## Synthesis

### The GPU Kernel Caching Stack

GPU kernel caching is a multi-layer problem with fundamentally different time constants at each layer:

```
Layer 5: Binary distribution (OCI registry)    — latency: minutes (pull)
            ↓
Layer 4: Disk binary cache                     — latency: 1-50 ms (file read)
            ↓
Layer 3: Module loading (CUDA/HIP/Level Zero)  — latency: 5-20 ms (driver call)
            ↓
Layer 2: Function/kernel handle lookup         — latency: ~1 µs (hash table)
            ↓
Layer 1: Kernel dispatch                       — latency: 2-20 µs (cuLaunchKernel)
```

Modern frameworks address each layer differently:

| Layer | CUDA | Vulkan | Triton | AdaptiveCpp | libkdl (proposed) |
|-------|------|--------|--------|-------------|-------------------|
| 5 (distribution) | None | None | OCI (MCV, 2026) | None | MTB via OCI |
| 4 (disk) | `~/.nv/ComputeCache/` | `VkPipelineCache` / `VK_KHR_pipeline_binary` | `~/.triton/cache/` | `~/.cache/adaptivecpp/` | MTB file (pre-computed) |
| 3 (module) | `cuModuleLoadData` | `vkCreateComputePipelines` | Python ext import | `hipModuleLoadData` / `zeModuleCreate` | `kdl_select_kernel` warm cache |
| 2 (handle) | `cuModuleGetFunction` | `vkGetPipelineExecutablePropertiesKHR` | Python dict lookup | dispatch table | `kdl_launch` function pointer |
| 1 (dispatch) | `cuLaunchKernel` (~20 µs) | `vkCmdDispatch` (~10 µs) | Triton wrapper | vendor API | vendor API via backend |

### Cache Key Taxonomy

All GPU kernel caching systems use a cache key that captures "what makes two compilation units equivalent." The key components across all systems:

| Key Component | Why Required |
|---------------|-------------|
| Kernel source / IR hash | Different source = different binary |
| Target GPU architecture | sm_80 binary ≠ sm_90 binary |
| Driver / library version | New JIT compiler may produce different binary |
| Compilation flags | Optimization level, feature flags change output |
| Device UUID | Same arch, different silicon (rare but happens with ECC variants) |

Omitting any component risks cache poisoning (wrong binary served). The Triton issue #4051 bug and the MCV cryptographic signature work both reflect real-world consequences of incomplete cache keys.

### Hot-Loading: The Runtime Reload Problem

"Hot-loading" — replacing a loaded kernel binary without stopping the process — is a harder problem than caching. Current status by vendor:

- **CUDA:** No official hot-reload API. `cuModuleUnload` + `cuModuleLoadData` achieves reload but requires all `CUfunction` handles to be discarded and re-acquired. Any in-flight kernel launch during unload is undefined behavior.
- **Vulkan:** `VkPipeline` objects are immutable after creation. Hot-reload requires creating a new pipeline object and atomically swapping the pointer in the dispatch path — this is the "pipeline hot-swap" pattern used by game engine shader hot-reload systems (Vulkan has no native support; applications implement this via versioned pipeline handles).
- **Level Zero:** No documented hot-module-reload API. Module lifecycle is create/destroy; hot-reload requires destroy + create.
- **IREE:** Compiles to `.vmfb` (Virtual Machine Flat Buffer) files. The IREE runtime can reload a `.vmfb` file without process restart by calling `iree_vm_context_reset` — the closest to true hot-loading in any ML runtime.
- **Triton / torch.compile:** Kernel hot-reload not supported at runtime. Requires process restart to pick up new compiled kernels (though `TRITON_CACHE_DIR` can be pre-populated before restart).

**Implication for libkdl:** libkdl's `kdl_free_bundle` + `kdl_load_bundle` sequence achieves logical hot-reload at the bundle level. The implementation safety requirement is that no `kdl_kernel_t` handles referencing the old bundle's modules are in-flight during the swap. A double-buffering pattern (two bundle slots, atomic pointer swap) would enable hot-reload with zero downtime.

### Cache Invalidation: The Hardest Problem

Cache invalidation for GPU kernels has more complexity than traditional software caches because of the multi-axis invalidation space:

1. **Source change:** Easy — rehash the IR/source.
2. **Driver upgrade:** Medium — encode driver version in key; check on load.
3. **GPU replacement:** Medium — encode device UUID; check on load.
4. **Architecture-specific feature change:** Hard — e.g., sm_90a binary uses TMA instructions; running on sm_90 without checking the `a` flag causes silent failure.
5. **Specialization staleness (Triton #4051):** Hard — dynamic Python closures changing behavior without source change.
6. **Binary tampering:** Security — addressed by MCV cryptographic signatures.

libkdl's MTB capability contract JSON currently encodes `min_driver`, `min_arch`, and a set of required features. This covers cases 1–4. Cases 5–6 require additions:
- Add a `binary_sha256` field to each variant's contract (addresses case 6)
- The binary hash in the MTB header serves as the "source change" signal (case 1)
- Driver version validation at `kdl_load_bundle` time (case 2)

### Gaps Identified

1. **No standard GPU kernel binary distribution format.** MTB, Triton cache directories, CUDA fatbin, and AdaptiveCpp's `~/.cache/adaptivecpp/` are all bespoke. The OCI/MCV approach is the first attempt at standardization via container registry semantics. An LLVM-level standard (analogous to the clang-offload-bundler format but with richer metadata) does not exist.

2. **Cross-machine binary portability is unsolved.** All current caching systems assume same GPU model + same driver version. Sharing compiled binaries across a heterogeneous fleet (e.g., H100 + A100 cluster) requires maintaining separate cache entries per GPU type — exactly what libkdl's MTB format does at the bundle level.

3. **Warm module store (module-handle caching) is not abstracted.** Every system (cuDNN, MIOpen, AdaptiveCpp) implements its own `CUmodule` / `ze_module_handle_t` cache separately. A shared runtime-level warm-module store would eliminate per-library duplication.

4. **Hot-reload is universally absent or fragile.** Only IREE approaches this with its `.vmfb` reload mechanism. All other systems require process restart for kernel updates. For long-running ML inference servers, this is a production gap.

5. **Security (binary integrity) is a 2026 concern.** Red Hat's MCV work is the first public treatment of GPU kernel binary supply-chain security. This is likely to become a compliance requirement for production ML deployments.

---

## Relevance to libkdl / Vendor-Agnostic Kernel Dispatch

### Direct Implications for libkdl Design

1. **MTB as the disk cache layer (Layer 4):** The MTB bundle IS libkdl's binary cache. Unlike Triton or AdaptiveCpp which cache JIT outputs, libkdl pre-ships the cache (AOT-compiled variants). This eliminates the cold-start JIT cost entirely — a fundamental architectural advantage.

2. **Missing warm-module store:** libkdl's current design (`kdl_select_kernel` → `cuModuleLoadData` on each process start) pays Layer 3 cost (5–20 ms module loading) on every process restart. Adding a persistent warm-module store (mapping (MTB hash, device index) → pre-loaded `CUmodule`) would reduce this to Layer 2 cost (<1 µs).

3. **Add `binary_sha256` to MTB capability contract:** Required for supply-chain security (per S10). Cost: 32 bytes per variant in the JSON contract. Benefit: `kdl_load_bundle` can verify binary integrity on load without trusting the filesystem.

4. **VK_KHR_pipeline_binary for Vulkan backend:** If libkdl adds a Vulkan backend (for AMD+NVIDIA+Intel portability), the caching layer must use `VK_KHR_pipeline_binary` on supported drivers (Intel ANV, NVK, NVIDIA 565+) with `VkPipelineCache` fallback for RADV.

5. **Persist selection results:** The roofline scoring in `kdl_select_kernel` (~200 µs cold) should be cached to disk, keyed by `(MTB hash, device UUID, driver version)`. This reduces warm-start selection from ~200 µs to < 1 µs.

6. **Hot-reload via double-buffering:** A `kdl_swap_bundle(ctx, old_bundle, new_bundle)` API using atomic pointer swap on the dispatch table would enable zero-downtime kernel updates — a meaningful differentiator for production inference servers.

7. **OCI distribution:** The MTB bundle is already a single binary file — an OCI layer wrapping it is trivial. `kdl-push`/`kdl-pull` tools following the MCV pattern would enable fleet-wide kernel binary distribution without per-node compilation.

---

## Sources

- [CUDA Pro Tip: Understand Fat Binaries and JIT Caching](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
- [CUDA Lazy Loading — Programming Guide §4.7](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)
- [Understanding Triton Cache: Optimizing GPU Kernel Compilation](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/)
- [AdaptiveCpp SSCP compilation docs](https://adaptivecpp.github.io/AdaptiveCpp/compilation/)
- [AdaptiveCpp Adaptivity paper — IWOCL 2025](https://dl.acm.org/doi/10.1145/3731125.3731127)
- [VK_KHR_pipeline_binary — Vulkan Documentation Project](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_pipeline_binary.html)
- [Bringing Explicit Pipeline Caching Control to Vulkan — Khronos Blog](https://www.khronos.org/blog/bringing-explicit-pipeline-caching-control-to-vulkan)
- [Compile Time Caching in torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [Compile Time Caching Configuration (Remote/Redis)](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)
- [MIOpen Kernel Cache](https://rocm.docs.amd.com/projects/MIOpen/en/develop/conceptual/cache.html)
- [Skip the JITters: Fast, Trusted Model Kernels with OCI Caching](https://next.redhat.com/2026/01/29/skip-the-jitters-fast-trusted-model-kernels-with-oci-caching/)
- [Protecting Triton kernel deployments with cryptographic signatures](https://next.redhat.com/2026/02/05/protecting-triton-kernel-deployments-with-cryptographic-signatures/)
- [CUDA Context-Independent Module Loading](https://developer.nvidia.com/blog/cuda-context-independent-module-loading/)
- [Robust pipeline cache serialization — zeux.io](https://zeux.io/2019/07/17/serializing-pipeline-cache/)
- [Level Zero Core Programming Guide](https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/PROG.html)
- [Jitify — NVIDIA/jitify GitHub](https://github.com/NVIDIA/jitify)
- [Triton cache invalidation bug — Issue #4051](https://github.com/triton-lang/triton/issues/4051)
- [cuDNN Developer Guide — Graph API](https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html)
- [Intel ANV Vulkan Driver — VK_KHR_pipeline_binary — Phoronix](https://www.phoronix.com/news/Intel-ANV-Pipeline-Binary)
- [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
- [Pipeline Cache — Vulkan Documentation Project](https://docs.vulkan.org/guide/latest/pipeline_cache.html)
