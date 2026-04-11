# Wave 04 — Kernel Binary Caching Strategies: Systematic Cross-Framework Comparison

**Angle:** kernel-binary-caching-strategies — systematic comparison of kernel binary caching across frameworks
**Queries:**
- "GPU kernel cache invalidation persistent disk compute cache JIT compilation"
- "Triton cache ~/.triton/cache kernel hash"
- "CUDA compute cache ~/.nv/ComputeCache driver invalidation"
- "AdaptiveCpp persistent JIT cache"
**Date:** 2026-04-06

---

## Summary

Every major GPU compute framework independently invented a persistent kernel binary cache to amortize JIT compilation overhead. Despite being invented in isolation, all converge on the same two-axis design space: **what goes into the cache key** (correctness) and **when to evict** (storage management). The frameworks differ sharply on cross-process sharing, size management, and — critically — whether the cache is an afterthought or a first-class architectural component.

The most significant finding for libkdl: Meta's torch.compile data (PyTorch PT2) shows that Triton async compilation is the single largest bottleneck in model start-up — 843 seconds out of 1,825 seconds total cold-start time for a large foundation model. A well-designed cross-vendor binary cache that eliminates JIT entirely is therefore a significant production win, not a micro-optimization.

---

## Sources

| # | Title | URL | Date | Type | Relevance |
|---|-------|-----|------|------|-----------|
| S1 | CUDA Pro Tip: Understand Fat Binaries and JIT Caching | https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/ | 2022 | NVIDIA Blog | 10/10 |
| S2 | CUDA Environment Variables — Programming Guide §5.2 | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html | Current (CUDA 13.x) | Official Docs | 9/10 |
| S3 | Understanding Triton Cache: Optimizing GPU Kernel Compilation | https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/ | May 2025 | Red Hat Blog | 10/10 |
| S4 | JIT Compilation and Caching — triton-lang/triton DeepWiki | https://deepwiki.com/triton-lang/triton/2.2-core-operations | 2025 | Wiki / source analysis | 9/10 |
| S5 | Triton Issue #4051: Cache invalidation with dynamic function calls | https://github.com/triton-lang/triton/issues/4051 | 2023 | Bug report | 8/10 |
| S6 | AdaptiveCpp Performance Documentation | https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md | 2025 | Official Docs | 9/10 |
| S7 | AdaptiveCpp Adaptivity IWOCL/SYCL 2025 paper | https://dl.acm.org/doi/10.1145/3731125.3731127 | 2025 | Peer-reviewed | 9/10 |
| S8 | hipSYCL kernel_cache.hpp (AdaptiveCpp predecessor) | https://git.chalmers.se/hariv/AdaptiveCpp/-/blob/29fe4c1f/include/hipSYCL/runtime/kernel_cache.hpp | 2023 | Source code | 8/10 |
| S9 | Intel compute-runtime COMPILER_CACHE.md | https://github.com/intel/compute-runtime/blob/master/programmers-guide/COMPILER_CACHE.md | 2025 | Official Docs | 9/10 |
| S10 | chipStar GitHub — CHIP_MODULE_CACHE_DIR documentation | https://github.com/CHIP-SPV/chipStar/blob/main/docs/Using.md | 2024-2025 | Official Docs | 8/10 |
| S11 | AOTriton / ROCm — AKS2 format and funcache | https://deepwiki.com/ROCm/aotriton | 2025 | Wiki / source analysis | 8/10 |
| S12 | MIOpen Kernel Cache Documentation | https://rocm.docs.amd.com/projects/MIOpen/en/develop/conceptual/cache.html | 2025 (ROCm 6.4) | Official Docs | 9/10 |
| S13 | IREE HAL executable cache — design roadmap | https://iree.dev/developers/design-docs/design-roadmap/ | 2025 | Official Docs | 8/10 |
| S14 | CUTLASS Python DSL JIT Caching | https://deepwiki.com/NVIDIA/cutlass/3.3-jit-compilation-and-caching | 2025 | Wiki / source analysis | 8/10 |
| S15 | Numba on-disk kernel caching | https://nvidia.github.io/numba-cuda/user/caching.html | 2025 | Official Docs | 7/10 |
| S16 | Compile Time Caching in torch.compile | https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html | 2025 (PyTorch 2.11) | Official Docs | 9/10 |
| S17 | Reducing PT2 Compilation Time — Meta Internal Workloads | https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/ | 2025 | PyTorch Blog | 10/10 |
| S18 | Protecting Triton kernel deployments with cryptographic signatures | https://next.redhat.com/2026/02/05/protecting-triton-kernel-deployments-with-cryptographic-signatures/ | Feb 2026 | Red Hat Blog | 8/10 |

---

## Comparative Analysis

### 1. CUDA ComputeCache (`~/.nv/ComputeCache`)

**Scope:** NVIDIA driver-managed JIT cache for PTX-to-CUBIN compilation.
**Cache key schema:** `(PTX SHA-256 hash, GPU UUID, driver version, compiler flags)`
**Storage:** `~/.nv/ComputeCache/` on Linux; `%APPDATA%\NVIDIA\ComputeCache` on Windows.
**What is cached:** Native CUBIN binary produced by the driver's PTX JIT compiler. Not the PTX itself.

**Invalidation policy:**
- Driver upgrade: entire cache is discarded (the JIT compiler in the new driver may produce different code, so all entries are stale by definition).
- GPU replacement: new GPU UUID creates new entries; old entries remain but are never served.
- `CUDA_CACHE_DISABLE=1`: disables reads and writes.
- No time-based expiry — only LRU eviction when size limit is hit.

**Size management:** Default 256 MiB cap (increased from 32 MiB in driver release 334). Maximum configurable to 4 GiB via `CUDA_CACHE_MAXSIZE`. LRU eviction when adding a new entry would exceed the cap. Entries larger than the cap are never cached.

**Cross-process sharing:** Yes — the cache directory is shared across all processes run by the same user. File locking details are driver-internal (not documented publicly).

**Cold vs. warm performance:** Not quantified publicly in isolation. In practice, PTX-to-CUBIN JIT adds 100ms–1s+ per unique PTX stream on first execution. Warm cache hit is near-zero overhead (file read + mmap).

**Gaps:** Only covers the case where a CUBIN is absent from the fatbin (i.e., the SM version is newer than what was compiled in). If the fatbin already includes a matching CUBIN, the ComputeCache is bypassed entirely. The cache does not cover multi-vendor scenarios.

---

### 2. Triton Per-Kernel-Hash Cache (`~/.triton/cache`)

**Scope:** Triton compiler's cache for JIT-compiled GPU kernels (Python @triton.jit decorated functions).
**Cache key schema (SHA-256 of):**
```
key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
```
Where:
- `triton_key()` = Triton version string + SHA-256 of core Triton library files
- `src.hash()` = SHA-256 of the decorated Python function's source code (recursively includes all `@triton.jit` callees; this is the dependency-propagation step)
- `backend.hash()` = GPU architecture + warp size + SM version
- `options.hash()` = `num_warps`, `num_ctas`, `num_stages`, compilation flags
- `env_vars` = values of `LLVM_IR_ENABLE_DUMP`, `MLIR_ENABLE_DIAGNOSTICS`, and any other vars in `CACHE_INVALIDATING_ENV_VARS`

**Storage:** `~/.triton/cache/<base32(sha256(key))>/` containing:
- `metadata.json` — kernel signature, launch grid, shared memory requirements
- `kernel.ttir` — Triton IR
- `kernel.ttgir` — TritonGPU IR
- `kernel.llir` — LLVM IR
- `kernel.ptx` / `kernel.amdgcn` — ISA-level text
- `kernel.cubin` / `kernel.hsaco` — final device binary

**Invalidation policy:** Hash-deterministic. Any change to source code, Triton version, backend, options, or listed env vars produces a new hash → new directory. Old directories are never automatically deleted — the cache can grow unboundedly. No LRU, no size cap, no TTL.

**Size management:** None built-in. Users must manually delete or rotate `~/.triton/cache/`. This is a known operational pain point.

**Cross-process sharing:** Yes — the cache directory is shared across Python processes. The `metadata.json` sentinel check (`if metadata.json exists → load from cache`) provides a basic form of concurrent-safe loading, but there is no explicit file lock on write. Race condition possible on simultaneous first compilation of the same kernel (Issue #4051 documents incorrect invalidation on dynamic call sites, confirming cache logic bugs exist).

**Cold vs. warm performance:** Cold start for a single Triton kernel compilation (Python → cubin): typically 1–10 seconds depending on kernel complexity. Meta's PT2 benchmark shows `async_compile.wait` — which is pure Triton compilation — takes 843 seconds for a large model's first cold run. Warm cache hit: metadata.json parse + cubin read, typically <10ms.

**Dependency bug (Issue #4051):** When a `@triton.jit` function dynamically selects which sub-function to call (via a Python variable holding a function reference), the cache key does not include the callee's hash unless it is statically referenced. This causes stale cache hits when the dynamic callee changes. Unresolved as of early 2026.

**torch.compile integration (MegaCache):** PyTorch 2.x adds a layer above raw Triton caching: `FXGraphCache` (Inductor graph artifacts), `TritonBundlerCache` (Triton cubins), `AOTAutogradCache` (joint graph), `AutotuningCache` (autotune benchmark results). These are unified into a portable archive via `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()`. The cache validates `(PyTorch version, Triton version, GPU device)` before serving. Remote caching via Redis is supported for CI/CD pipelines.

---

### 3. AdaptiveCpp Persistent JIT Cache (`~/.acpp/apps`)

**Scope:** AdaptiveCpp's "generic compilation target" (SSCP — Single-Source Compilation Pass) which JIT-compiles SYCL kernels to device-specific binaries at first launch.

**Cache key schema:** A 128-bit hash covering:
- Kernel name (mangled)
- Target hardware identifier (vendor + architecture + device capabilities)
- Backend (HIP, CUDA, Level Zero, OpenCL)
- Compilation options and optimization flags
- JIT-time optimization decisions (inlining, loop unrolling choices)

This is a two-level cache:
1. **In-process cache:** `std::shared_mutex`-protected map from `(kernel_id, device_id)` to compiled `hipFunction_t` / CUDA function handle. Used within a single process execution.
2. **On-disk persistent cache:** `~/.acpp/apps/` directory, keyed by the 128-bit hash. Survives across process runs.

**What is cached:** The compiled device binary (HSACO on AMD, CUBIN on NVIDIA, or native binary for the Level Zero/OpenCL backend). Not the LLVM IR or intermediate stages.

**Adaptivity database (appdb):** A separate per-application SQLite database stores runtime profiling information (kernel launch frequencies, observed hardware counters). On subsequent runs, the adaptivity engine consults appdb to make better JIT-time optimization decisions. This means early runs may compile suboptimal variants, with optimization improving over multiple runs — the system prints a warning ("new binaries were JIT-compiled; run again for optimal performance") until the appdb stabilizes.

**Invalidation policy:** Explicit only. Changes to user application code do NOT automatically invalidate the cache (the binary is identified by hash, and if the kernel source changed, a new hash entry is created, but old entries linger). Infrastructure changes (AdaptiveCpp upgrade, driver upgrade) require manual cache clearing: `rm -rf ~/.acpp/apps/*`. No automatic driver-version invalidation.

**Cold vs. warm performance:** "Slight overhead" on first kernel launch per the documentation. For large SYCL programs, first-run JIT adds seconds. Warm cache eliminates JIT overhead entirely. The two-level cache means in-process reuse (same process, same kernel) hits the memory cache first; cross-process reuse hits the disk cache.

**Cross-process sharing:** Yes — `~/.acpp/apps/` is shared across processes. No documented locking mechanism.

**Size management:** None built-in. Old entries accumulate unless manually deleted.

---

### 4. Intel NEO Compiler Cache (OpenCL / Level Zero)

**Scope:** Intel's compute-runtime (NEO driver) persistent cache for OpenCL and Level Zero programs compiled from SPIR-V or OpenCL C.

**Cache key schema:** A hash computed from:
- SPIR-V binary content (or OpenCL C source)
- Target device identifier
- Compiler options / build flags
- compute-runtime version

**Note on gaps:** The cache does NOT include environment variables of external components in its hash. If a variable like `IGC_ShaderDumpEnable` changes (an Intel Graphics Compiler flag), the cache may serve stale binaries. This is documented as a known limitation that "can lead to unexpected behavior and difficult to debug errors."

**Storage:** Configurable via `NEO_CACHE_DIR` (default: `~/.cache/neo_compiler_cache/`). Each entry is stored as `<hash>.<extension>`.

**Invalidation policy:** Hash-deterministic for included fields. External env vars not covered. Manual clearing if external tools change. Driver upgrade: compute-runtime version is in the hash, so driver upgrades naturally create new entries (old entries remain but are never served — gradual LRU eviction).

**Size management:** Configurable via `NEO_CACHE_MAX_SIZE`. When adding a new entry would exceed the limit, LRU eviction is triggered. No default size is documented publicly.

**Cross-process sharing:** Yes — cache directory is shared across OpenCL/Level Zero applications from any process.

**Cold vs. warm performance:** chipStar documentation acknowledges first-execution JIT latency as a real production problem significant enough to warrant a dedicated `CHIP_MODULE_CACHE_DIR` environment variable. Intel's own driver-level cache reduces repeat-run overhead substantially.

---

### 5. chipStar Module Cache (`CHIP_MODULE_CACHE_DIR`)

**Scope:** chipStar's cache for SPIR-V binaries compiled to native device code (GEN ISA for Intel, OpenCL native code for other targets).

**Default location:** `$HOME/.cache/chipStar/`. Empty string disables caching.

**Cache key schema:** SPIR-V binary hash + device fingerprint (architecture, driver version). Exact hash algorithm not documented publicly; derived from chipStar v1.2.1 release notes.

**What is cached:** The driver's native binary output after JIT-compiling the SPIR-V. This avoids re-running the SPIR-V-to-native compilation on subsequent runs. PTX or LLVM IR intermediate stages are not cached (those are compile-time artifacts).

**Invalidation policy:** Hash-deterministic on SPIR-V content + device. Not documented whether driver upgrades are captured in the device fingerprint.

**Size management:** Not documented. Assumed unlimited accumulation.

**Cross-process sharing:** Yes — path is a single directory for all chipStar-using processes.

**Context:** v1.2.1 (Nov 2024) was the first release to include module caching, alongside JIT timing instrumentation. The fact that it took until v1.2.1 — after the system was production-deployed at Argonne's Aurora supercomputer — confirms that JIT latency was a real operational pain point rather than a theoretical concern.

---

### 6. AOTriton AKS2 + In-Process funcache

**Scope:** ROCm's Ahead-of-Time Triton library for Flash Attention and related math kernels. AOTriton ships pre-compiled HSACO binaries — there is no JIT at runtime.

**AKS2 format:** LZMA-compressed archives containing HSACO binaries organized by GPU architecture family and kernel variant. `PackedKernel` class manages a global registry mapping `(architecture, kernel_name, kernel_variant)` to compressed HSACO data.

**Decompression and loading lifecycle:**
1. HSACO is decompressed from `.aks2` lazily on first use.
2. `hipModuleLoadDataEx` loads raw HSACO bytes into a `hipModule_t`.
3. `hipModuleGetFunction` extracts `hipFunction_t`.
4. Result is stored in an in-process `funcache_` map keyed by `device_id`.

**Cache key schema (in-process funcache):** `device_id` only — since the binary is already architecture-specific (selected by the AKS2 registry), no further re-keying is needed at runtime.

**SQLite tuning database:** AOTriton ships per-architecture SQLite databases containing pre-profiled tuning parameters (tile sizes, pipeline depth, etc.) for each kernel variant. At runtime, the database is queried to select the optimal variant for the current device. This is not a JIT cache — it is a lookup table for pre-compiled variant selection.

**Cold vs. warm performance:** No JIT cold-start penalty. The decompression + `hipModuleLoad` overhead on first use is measured in single-digit milliseconds per kernel family. Subsequent calls hit the in-process `funcache_` and incur no loading overhead.

**Cross-process sharing:** No — the in-process funcache is per-process. Each process decompresses and loads from `.aks2` independently. No disk-level cache beyond the `.aks2` files themselves (which are read-only distribution artifacts).

**Size management:** Binary size is managed by LZMA compression at distribution time, not at runtime. The `funcache_` grows monotonically within a process.

---

### 7. MIOpen Kernel Cache (`~/.cache/miopen/`)

**Scope:** AMD's MIOpen (convolution, attention, etc.) kernel binary cache. Covers kernels that require OpenCL/HIP compilation at install time or first use.

**Two-level hierarchy:**
1. **System cache:** Installed by ROCm package manager into the MIOpen installation directory (architecture-specific precompiled kernel packages). Checked first.
2. **User cache:** `~/.cache/miopen/<version>/` — fallback when system cache misses. Kernels are compiled and stored here on first use.

**SQLite integration:** MIOpen uses SQLite for the user cache database. Schema stores `(kernel_signature_hash, architecture, compiled_binary)`. Warning `SQLiteBase: Missing system database file` appears when the precompiled system cache package is absent.

**Versioning:** Cache directories are versioned (introduced in MIOpen 2.4). An upgrade to MIOpen creates a new version directory; old entries are abandoned (not deleted automatically).

**Cache key schema:** `kernel_signature_hash` — a hash of the kernel source, compilation options, and target architecture. Exact fields not publicly documented; inferred from MIOpen source.

**Invalidation policy:** Version-based. Different MIOpen version = different cache directory = fresh start. No driver-version component (potential staleness on driver upgrade without MIOpen upgrade).

**Cold vs. warm performance:** MIOpen's documentation warns that missing system cache packages "affect network start-up time" significantly — implying minutes-scale compilation overhead for large convolution networks when the precompiled system cache is absent. Warm cache hit eliminates this entirely.

**Cross-process sharing:** Yes — `~/.cache/miopen/` is shared across MIOpen-using processes.

---

### 8. IREE Executable Cache (Process-Lifetime)

**Scope:** IREE's `iree_hal_executable_cache_t` — a per-device in-process cache of prepared GPU executables.

**Current behavior:** Process-lifetime only. Executables are compiled on-demand during module initialization and held in the cache for the duration of the process. No disk persistence.

**What is cached:** The compiled executable object (driver-native code). The cache avoids re-compiling the same executable if it is referenced multiple times within a module.

**Cache key schema:** Not documented publicly. Inferred to be based on executable binary identity (pointer equality or content hash).

**Persistent cache roadmap:** IREE's design roadmap acknowledges the limitation and describes a planned persistent caching mechanism: "The caches generated can be retrieved and saved by the hosting application, and upon the next execution the application can provide the caches and if still valid they will be used to avoid compilation." This implies an application-controlled (not automatic) persistence model, similar to Vulkan's `VkPipelineCache`.

**Cross-process sharing:** No — not supported in current design. The hosting application would need to implement serialization and sharing explicitly.

**Cold vs. warm performance:** Significant cold-start overhead for large models, since all kernels are compiled at module initialization. No quantitative data published.

**Implication for libkdl:** IREE's process-lifetime-only cache is the weakest caching model in this survey. libkdl's disk-persistent MTB bundle format is architecturally superior to IREE's current design and aligns with the stated roadmap direction.

---

### 9. CUTLASS Python DSL Cache (`CUTE_DSL_CACHE_DIR`)

**Scope:** CUTLASS Python (CuTe DSL) cache for JIT-compiled MLIR-to-CUBIN kernels.

**Cache key schema (composite hash of):**
- MLIR bytecode content (the generated intermediate representation)
- DSL package source file hashes (changes to CUTLASS Python implementation files)
- Shared library file hashes (runtime dependency versions)
- Environment variable values (e.g., `CUTE_DSL_ARCH`)

**Storage:** `{CUTE_DSL_CACHE_DIR}/cutedsl_{hash}.mlir`. Default: `{TMPDIR}/{user}/cutlass_python_cache` or `/tmp/cutlass_python_cache`. Note: `/tmp` is not persistent across reboots.

**Atomic writes:** Uses `os.replace()` for atomic file creation, preventing partial file reads by concurrent processes.

**CRC32 integrity verification:** Cached MLIR files include CRC32 checksums for corruption detection.

**In-memory cache:** Above the disk cache, a per-process in-memory cache provides instant retrieval for repeated kernel invocations within the same Python session.

**Invalidation policy:** Hash-deterministic. File deletion for manual invalidation. `CUTE_DSL_DISABLE_FILE_CACHING` disables disk persistence (in-memory only).

**Size management:** None. Manual deletion required.

**Cross-process sharing:** Yes — processes sharing the same `CUTE_DSL_CACHE_DIR` share cached artifacts.

---

### 10. Numba CUDA On-Disk Cache (`__pycache__`)

**Scope:** Numba's `@cuda.jit(cache=True)` mechanism for caching compiled GPU kernels.

**Storage:** `__pycache__/` subdirectory adjacent to the Python source file. Fallback to `$HOME/.cache/numba/` when the source directory is not writable.

**Cache key:** Per-compute-capability. Separate cache entries for cc=7.5, cc=8.0, cc=9.0, etc. The cache uses the Python function's bytecode hash as part of the key.

**Invalidation:** Source file modification time triggers recompilation. Compute capability mismatch creates a new entry.

**Cross-process sharing:** Yes — `__pycache__` is shared across processes accessing the same source.

**Size management:** None. Manual deletion only.

---

## Cross-Framework Comparison Table

| Framework | Cache Location | Key Schema | Auto-Invalidation | Size Management | Cross-Process | Persistence |
|-----------|---------------|------------|-------------------|-----------------|---------------|-------------|
| CUDA ComputeCache | `~/.nv/ComputeCache/` | PTX-SHA256 + GPU UUID + driver ver + flags | Driver upgrade | 256 MiB LRU (configurable to 4 GiB) | Yes (same user) | Persistent |
| Triton | `~/.triton/cache/<hash>/` | SHA-256(src + backend + options + env + triton ver) | Content-addressed; no TTL | None (unbounded growth) | Yes | Persistent |
| AdaptiveCpp | `~/.acpp/apps/` | 128-bit(kernel name + hw + backend + opts) | None (manual clear required) | None | Yes | Persistent |
| Intel NEO | `~/.cache/neo_compiler_cache/` | Hash(SPIR-V + device + compiler ver) — excludes some env vars | Content-addressed | Configurable LRU | Yes | Persistent |
| chipStar | `$HOME/.cache/chipStar/` | SPIR-V hash + device fingerprint | Unknown | None documented | Yes | Persistent |
| AOTriton | `.aks2` files (distribution) + in-process | device_id (in-process); architecture (AKS2) | N/A (AOT pre-compiled) | LZMA compression | No (per-process) | AOT at dist time |
| MIOpen | `~/.cache/miopen/<ver>/` (SQLite) | kernel signature hash | Version directory change | None auto | Yes | Persistent |
| IREE | In-process only | Executable identity | Process exit | Process exit | No | None |
| CUTLASS Python | `/tmp/cutlass_python_cache/` | Hash(MLIR + DSL source + shared libs + env) | Content-addressed | None | Yes (same dir) | Conditional (tmp may clear) |
| Numba | `__pycache__/` | Bytecode hash + compute capability | Source mtime | None | Yes | Persistent |

---

## Cold Start vs. Warm Cache Performance Data

The most concrete public dataset comes from Meta's PT2 (torch.compile) profiling of a large foundation model:

| Phase | Cold Start Time | Component |
|-------|----------------|-----------|
| Total compilation | 1,825.58 s (~30 min) | Full model |
| Triton async_compile.wait | 843.95 s | Largest single bottleneck (46.2% of total) |
| CachingAutotuner.benchmark | 238.00 s | Autotune overhead |
| AOTDispatch | 248.03 s | |
| Dynamo | 100.64 s | |

**Warm cache hit time:** <10 ms per kernel (metadata parse + disk read). The 843-second Triton compilation cost is eliminated entirely on warm cache hits with the full torch.compile MegaCache.

**Implication:** For production ML workloads, persistent kernel binary caches provide 2-3 orders of magnitude reduction in application start-up latency. A cross-vendor binary cache (as in libkdl) achieves this without any JIT at all — the compilation happened at build time.

Other data points (less precise):
- CUDA PTX JIT: 100ms–1s+ per unique PTX stream on cold start (NVIDIA documentation)
- AdaptiveCpp first-kernel JIT: "slight overhead" — estimated seconds for complex kernels
- MIOpen without precompiled system cache: "affects network start-up time" — implied minutes for large convolution networks

---

## Key Design Observations

### Observation 1: Everyone converges on SHA-256 content-addressing

Triton, CUTLASS, CUDA, and Intel NEO all use content-addressable caching: the cache key is derived from the inputs, so the cache never needs to be explicitly invalidated for content changes — a new key simply misses and recompiles. The only failure mode is when the key does not capture all relevant inputs (Triton Issue #4051; Intel NEO's external env var gap).

### Observation 2: Size management is universally neglected

Only CUDA and Intel NEO implement LRU eviction with configurable size caps. All other frameworks allow unbounded growth. In long-running developer environments (Triton `~/.triton/cache/` is particularly notorious), cache directories grow to tens of gigabytes.

### Observation 3: Cross-process sharing is assumed, not designed

All disk-backed caches share a directory across processes. None provide documented locking protocols. Race conditions on simultaneous first-compilation are possible in Triton, AdaptiveCpp, chipStar, and Numba. CUTLASS Python uses `os.replace()` for atomic writes, which is the only framework to address this explicitly.

### Observation 4: Driver-version invalidation is inconsistently handled

CUDA: explicit (driver upgrade discards cache). Intel NEO: implicit via runtime version in hash. AdaptiveCpp: not handled (manual clear required). MIOpen: library version directory change covers this indirectly. chipStar: unknown.

### Observation 5: AOT pre-compilation (AOTriton) eliminates all of these problems

By shipping LZMA-compressed pre-compiled HSACO binaries in the distribution package, AOTriton eliminates cold-start JIT latency entirely. The trade-off: binary size grows with the number of supported architectures, and new hardware requires a new release. libkdl's MTB bundle format follows this AOT model at the multi-vendor level.

### Observation 6: Security is an emerging concern

A 2026 Red Hat blog post (Feb 2026) discusses protecting Triton kernel deployments with cryptographic signatures — motivated by the observation that the shared cache directory (`~/.triton/cache/`) is a potential supply-chain attack vector: a malicious process could pre-populate the cache with tampered binaries. CUDA's ComputeCache has the same attack surface. This is an open problem for all framework caches.

---

## Angle Assessment

**Relevance to libkdl: 9/10**

Kernel binary caching is directly load-bearing for libkdl's production argument. The Meta PT2 data (843s Triton cold-start bottleneck) provides the most quantitative justification for eliminating JIT via pre-compiled dispatch. The comparison table above directly maps to the libkdl design space: libkdl's MTB bundle is the cross-vendor equivalent of CUDA's fatbin + ComputeCache combined, but designed from the ground up for multi-vendor deployment.

**Novelty: 7/10**

The individual caching mechanisms are individually documented but have not been systematically compared across frameworks. The cross-framework comparison table, the identification of universal design gaps (size management neglect, locking assumptions, driver-version handling inconsistency), and the quantitative cold-start data from Meta's PT2 benchmark are the novel contributions of this wave.

**Key gap surfaced:** No existing framework provides a *cross-vendor* persistent kernel binary cache — the problem that libkdl's MTB bundle + runtime cache layer solves. CUDA's ComputeCache is NVIDIA-only, Triton's cache is framework-specific and not shared across backend types, and AdaptiveCpp's cache covers a single device type per entry. A unified cache with entries keyed by `(kernel_name, vendor_id, architecture, capability_flags)` — as in the proposed libkdl registry — is not available in any production system today.

---

## Risks / Inconsistencies Discovered

1. **Triton Issue #4051 (unresolved, early 2026):** Dynamic callee selection produces stale cache hits. Any system building on Triton's cache for multi-vendor dispatch must be aware that the cache is not provably sound for dynamic dispatch patterns.

2. **Intel NEO external-env-var gap:** Compiler flags from tools outside compute-runtime (e.g., Intel Graphics Compiler debug flags) are not captured in the cache key. Developers who enable IGC debug options may get incorrect cached binaries on the next run without realizing it.

3. **AdaptiveCpp appdb stability warning:** The appdb optimization database requires multiple application runs to stabilize. Benchmarks taken on the first or second run may underestimate the JIT-optimized steady-state performance.

4. **IREE process-lifetime cache is a production bottleneck:** For server scenarios where the IREE runtime is restarted (e.g., after a crash), all kernel compilation overhead is repeated on every restart. The design roadmap acknowledges but has not yet shipped persistent caching.

5. **Security attack surface:** Shared cache directories are writable by any process running as the same user. Pre-population with malicious binaries is possible. None of the caching systems (except the proposed Red Hat OCI-layer approach from Jan 2026) implement signed binary verification on cache load.

6. **CUTLASS Python uses `/tmp` by default:** Cache contents are lost on reboot, making repeated cold starts the norm unless `CUTE_DSL_CACHE_DIR` is explicitly configured to a persistent location. This is a significant usability gap for production deployment.

---

## Citation Bibtex

```bibtex
@online{redhat2025triton,
  author    = {Sangiorgi, Alessandro},
  title     = {Understanding Triton Cache: Optimizing GPU Kernel Compilation},
  year      = {2025},
  url       = {https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/},
  month     = {May},
}

@online{pytorch2025pt2compilation,
  title     = {Experience in Reducing PT2 Compilation Time for Meta Internal Workloads},
  year      = {2025},
  url       = {https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/},
  publisher = {PyTorch Blog},
}

@online{pytorch2025compilecache,
  title     = {Compile Time Caching in torch.compile},
  year      = {2025},
  url       = {https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html},
  publisher = {PyTorch Documentation},
}

@inproceedings{adaptivecpp2025sscp,
  title     = {Adaptivity in AdaptiveCpp: Optimizing Performance by Leveraging Runtime Information During JIT-Compilation},
  booktitle = {Proceedings of the 13th International Workshop on OpenCL and SYCL},
  year      = {2025},
  doi       = {10.1145/3731125.3731127},
}

@online{redhat2026tritonsignatures,
  author    = {{Red Hat Emerging Technologies}},
  title     = {Protecting Triton kernel deployments with cryptographic signatures},
  year      = {2026},
  month     = {February},
  url       = {https://next.redhat.com/2026/02/05/protecting-triton-kernel-deployments-with-cryptographic-signatures/},
}
```
