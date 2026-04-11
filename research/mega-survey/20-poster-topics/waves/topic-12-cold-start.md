# Topic 12: Cold Start Elimination for JIT-Compiled ML Kernels

**Survey:** 20 LLVM/MLIR Poster Topics
**Topic ID:** 12
**Config name:** cold-start-elimination-jit-kernels
**Title:** Cold Start Elimination for JIT-Compiled ML Kernels: An LLVM-Level Persistent Kernel Object Cache
**Persona:** ML serving engineer / LLVM compiler infrastructure contributor
**Date:** 2026-04-07

---

## Gap

Production ML inference cold start is dominated not by model loading or weight transfer
but by GPU kernel JIT compilation. Meta's PT2 profiling of a large foundation model
(PyTorch Blog, 2025) shows:

| Phase | Latency | Share |
|-------|---------|-------|
| `async_compile.wait` (Triton JIT) | **843.95 s** | **46.2%** of total |
| `CachingAutotuner.benchmark` | 238.00 s | 13.0% |
| `AOTDispatch` | 248.03 s | 13.6% |
| `Dynamo` | 100.64 s | 5.5% |
| **Total cold start** | **1,825.58 s** | — |

The `async_compile.wait` bottleneck is entirely Triton JIT: Python `@triton.jit` decorated
functions compiling to CUBIN/HSACO on first invocation. This is not a framework bug — it
is the inherent cost of source-to-binary compilation at runtime for hundreds of unique
kernel specializations.

The current remediation paths are all framework-specific and incomplete:

1. **Triton's `~/.triton/cache/`** — keyed by
   `SHA256(triton_version || src_hash || backend_hash || options_hash || env_vars)`.
   Warm cache eliminates JIT (<10 ms per kernel). But: no LRU eviction, unbounded growth,
   no cross-process locking, unresolved stale-hit bug on dynamic callees (Issue #4051),
   no cross-vendor sharing, invalidated on any Triton version bump.

2. **CUDA ComputeCache (`~/.nv/ComputeCache`)** — driver-managed PTX→CUBIN cache, default
   256 MiB LRU. Discarded entirely on driver upgrade. Only covers NVIDIA. Only covers the
   PTX→CUBIN step, not the Python→PTX step that dominates Triton cost.

3. **AdaptiveCpp SSCP persistent cache (`~/.acpp/apps/`)** — two-level (in-process
   `shared_mutex` map + on-disk hash-addressed binary). +30% over CUDA, +44% over HIP via
   runtime JIT decisions. But: requires AdaptiveCpp's compilation model; no manual
   invalidation on driver upgrade; no cross-vendor format.

4. **torch.compile MegaCache** (`FXGraphCache` + `TritonBundlerCache` + `AOTAutogradCache`)
   — portable archive via `torch.compiler.save_cache_artifacts()`, validated on
   `(PyTorch version, Triton version, GPU device)`. Eliminates warm-path JIT but:
   Python-level, no LLVM IR persistence, no cross-framework sharing, only NVIDIA/AMD
   (no Intel, no CPU fallback path).

The structural gap: **no existing system persists GPU kernel objects at the LLVM IR level
across framework upgrades, vendor boundaries, and process restarts**. Every framework
operates at its own layer (source, PTX, CUBIN, Python bytecode) and fails at the same
failure modes: version-keyed invalidation discards warm caches on routine upgrades, and
no system shares pre-compiled binaries across vendor APIs.

The MLSys 2026 paper "Breaking the Ice: vLLM Cold Start" (referenced in arXiv:2601.00227,
Dynamic Kernel Substitution) documents the same 843-second phenomenon in the vLLM serving
context, confirming this is not a PyTorch-specific pathology but a structural problem for
any JIT-heavy ML inference stack.

---

## Proposal

**KernelObject Cache (KOC): a vendor-neutral, LLVM-level persistent kernel binary store
that decouples compiled GPU kernel objects from the framework that produced them.**

The proposal has three separable components.

### Component A — Standardized Kernel Object Identity (SKOI)

A content-addressed key scheme that is stable across framework upgrades:

```
SKOI = SHA256(
    kernel_ir_bitcode    ||   # LLVM IR of kernel body (framework-independent)
    target_triple        ||   # e.g., "nvptx64-nvidia-cuda-sm_75"
    compilation_flags    ||   # -O3, num_warps, num_stages, etc.
    driver_abi_version   ||   # CUDA driver major.minor, ROCm version
    capability_set            # SM capabilities, warp_size, max_shared_mem
)
```

The key element: `kernel_ir_bitcode` replaces framework source code (Python, CUDA C) with
the LLVM IR produced at the end of any compilation pipeline (Triton TTGIR→LLIR, Inductor
Triton, Proteus ProteusPass output, MLIR gpu.launch→NVPTX lowering). This makes the cache
key framework-agnostic: a kernel compiled via Triton and the same kernel compiled via
torch.inductor produce the same LLVM IR and therefore the same SKOI — sharing one cache
entry.

The `driver_abi_version` component replaces CUDA's "discard everything on driver upgrade"
policy with fine-grained invalidation: only entries whose ABI is incompatible with the
current driver are evicted, not the entire cache.

### Component B — Multi-Tier Storage with OffloadBinary Format

The persistent store uses LLVM's `OffloadBinary` container format
(magic `0x10FF10AD`, the LLVM 20 standard used by `clang-offload-packager`) as the
on-disk representation. Each cache entry is a valid `OffloadBinary` archive:

```
Cache entry (OffloadBinary):
  entry[0]: ImageKind=LLVM_BC,  triple="nvptx64-nvidia-cuda", data=<ir_bitcode>
  entry[1]: ImageKind=PTX,      triple="nvptx64-nvidia-cuda", sm="sm_75"
  entry[2]: ImageKind=CUBIN,    triple="nvptx64-nvidia-cuda", sm="sm_75"
  entry[3]: ImageKind=HSACO,    triple="amdgcn-amd-amdhsa",   arch="gfx942"
  entry[4]: ImageKind=Object,   triple="x86_64-unknown-linux", data=<cpu_elf>
```

The cache serves entries in a three-tier lookup hierarchy:

```
Tier 1 (zero latency): in-process hash map — kdl_dispatch_table_t lookup, O(1)
Tier 2 (~5 ms): on-disk OffloadBinary entry — file read + mmap, driver load via
                cuModuleLoadData / hipModuleLoadDataEx
Tier 3 (seconds): JIT compilation fallback — NVRTC / comgr / LLVM OrcJIT on
                embedded LLVM IR bitcode — then write result back to Tier 2
```

Tier 1 is what libkdl already implements: a C hash map keyed on
`(vendor_id, arch, kernel_name, type_sig, shape_sig)`.
Tier 2 is the new cross-framework persistent disk cache.
Tier 3 closes the "unseen architecture" case without requiring source distribution.

Storage layout:

```
~/.kdl/cache/
  <SKOI_hex>/
    metadata.json        # arch, driver_abi_version, compilation_flags, timestamp
    kernel.bc            # LLVM IR bitcode (portable, version-independent)
    kernel.ptx           # PTX text (NVIDIA, pinned to driver ABI)
    kernel.cubin         # CUBIN (NVIDIA, pinned to SM + driver)
    kernel.hsaco         # HSACO ELF (AMD, pinned to gfx arch)
    kernel.o             # CPU ELF (x86_64 / aarch64)
    kernel.sig           # Ed25519 signature over metadata + binary (Red Hat OCI model)
```

LRU eviction is implemented at the directory level (default 2 GiB cap, configurable via
`KDL_CACHE_MAXSIZE`). Size management is explicit: a background GC thread removes
least-recently-used directories when the cap is approached.

### Component C — LLVM Integration Points

Three LLVM/MLIR integration points that make this an LLVM-level (not just a user-space)
contribution:

**C1. MLIR `gpu.launch_func` attribute for cache hints**

Add a `#gpu.cache_hint` attribute to `gpu.launch_func` ops that carries the SKOI
(or its inputs) through MLIR lowering. This allows Triton/Inductor frontends to attach
the pre-computed cache key at the MLIR level, so the runtime can skip hash computation
on the hot path:

```mlir
gpu.launch_func @matmul_kernel::@kernel
    grid in (%gx, %gy, %gz) block in (%bx, %by, %bz)
    args(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>)
    {gpu.cache_key = "a3f7c2..."}  // SKOI pre-computed at compile time
```

**C2. liboffload `rankImage()` hook with cache writeback**

Replace the `break` in liboffload's `parseOffloadBinary` first-match loop
(PR #186088, open March 2026) with a ranked selection that also consults KOC:

```cpp
// llvm/offload/plugins-nextgen/common/src/PluginInterface.cpp
Expected<DeviceImageTy *> GenericPluginTy::loadNextBestImage(
    OffloadBinMetadataTy &Meta, KernelObjectCache &KOC) {
  // 1. Check KOC Tier 1 (in-process table)
  if (auto *Cached = KOC.lookupInProcess(Meta.SKOI))
    return Cached;
  // 2. Check KOC Tier 2 (disk cache)
  if (auto *Cached = KOC.lookupDisk(Meta.SKOI))
    return KOC.loadAndPromote(Cached);
  // 3. Score available images, select best, write back to KOC Tier 2
  auto *Best = rankImages(Meta.Images);
  KOC.store(Meta.SKOI, Best);
  return Best;
}
```

**C3. ThinLTO-style IR cache adaptation**

LLVM's ThinLTO cache (controlled by `-Oz-cache-path`, implemented in
`llvm/lib/LTO/ThinLTOCodeGenerator.cpp`) already implements content-addressed caching
of per-module LLVM IR → native object compilation results. KOC adapts this infrastructure
for GPU: the same SHA-256 addressable store, the same LRU eviction, but with
`OffloadBinary` as the output format rather than `.o` files, and with driver-ABI versioning
added to the cache key. This is an explicit reuse of existing LLVM cache machinery, not
a new invention — the novelty is applying it to GPU backend outputs.

---

## Evidence

### E1. Meta PT2 profiling: 843 s Triton cold start (primary motivation)
- Source: "Experience in Reducing PT2 Compilation Time for Meta Internal Workloads"
  (PyTorch Blog, 2025)
- URL: https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/
- `async_compile.wait` = 843.95 s = 46.2% of 1,825.58 s total cold start.
- Warm cache hit via MegaCache: <10 ms per kernel (framework-specific, not cross-vendor).
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-kernel-caching.md` §"Cross-Framework Comparison Table"

### E2. Triton cache architecture and Issue #4051 (cache unsoundness)
- Source: Red Hat Emerging Technologies, May 2025
- URL: https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/
- Cache key includes backend discriminator; no cross-vendor sharing.
- Issue #4051: dynamic callee selection causes stale hits. Unresolved early 2026.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-kernel-caching.md` §2

### E3. CUDA ComputeCache: driver-upgrade invalidation is the core fragility
- Source: CUDA Pro Tip: Understand Fat Binaries and JIT Caching (NVIDIA Blog, 2022)
- URL: https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
- Entire cache discarded on driver upgrade. Default 256 MiB LRU. NVIDIA-only.
- Only covers PTX→CUBIN step; not the Python→PTX step dominating Triton cost.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-kernel-caching.md` §1

### E4. AdaptiveCpp two-level cache: appdb stability warning, no driver-upgrade handling
- Source: AdaptiveCpp Performance Documentation (2025); IWOCL/SYCL 2025 paper
  (doi:10.1145/3731125.3731127)
- URL: https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md
- +30% over CUDA, +44% over HIP via runtime JIT — strong case for JIT specialization value.
- No automatic invalidation on driver upgrade; no cross-vendor format.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-kernel-caching.md` §3

### E5. Proteus (CGO 2025): LLVM IR as portable JIT substrate
- Source: Georgakoudis, Parasyris, Beckingsale — CGO 2025; doi:10.1145/3696443.3708939
- Repository: https://github.com/Olympus-HPC/proteus (v2026.03.0, active)
- ProteusPass embeds LLVM IR bitcode into the application binary at AOT time.
  At runtime, Proteus JIT-specializes from the embedded IR using NVRTC (CUDA) or
  comgr (AMD). 2.8x speedup on AMD, 1.78x on NVIDIA vs pure AOT.
- This validates both the embedded-IR approach (Component B Tier 3) and the thesis that
  IR-level caching is portable across vendor JIT backends.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-07-proteus-deep-dive.md` §3

### E6. MLIR `mgpuModuleLoadJIT` asymmetry (NVIDIA-only, PR #66220)
- Source: LLVM PR #66220 (fabianmcg)
- URL: https://github.com/llvm/llvm-project/pull/66220
- PTX-format GPU modules JIT-compiled by NVIDIA driver at runtime. No AMD equivalent.
- PR admits "significant runtime performance hit" vs fatbin — confirms JIT fallback
  should be Tier 3, not the primary path.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-05-gpu-kernel-jit.md` §3

### E7. LLVM ThinLTO cache: existing content-addressed IR→object cache infrastructure
- Source: `llvm/lib/LTO/ThinLTOCodeGenerator.cpp`, `-Oz-cache-path` flag
- LLVM already implements SHA-256 content-addressed caching of per-module IR→native
  compilation results with LRU eviction. KOC reuses this infrastructure for GPU outputs.
- This makes Component C3 a refactoring, not a greenfield build.

### E8. Red Hat OCI cache model (Jan 2026): container-layer GPU kernel distribution
- Source: "Skip the JITters: Fast, trusted model kernels with OCI caching" (Red Hat, Jan 2026)
- URL: https://next.redhat.com/2026/01/29/skip-the-jitters-fast-trusted-model-kernels-with-oci-caching/
- Proposes shipping pre-compiled Triton kernels as OCI image layers, pulled before
  model serving starts. Ed25519 signatures for supply-chain security.
- KOC Component A (SKOI) and the `kernel.sig` file in Component B adopt this signing model.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-05-kernel-caching.md` §S9

### E9. OffloadBinary format: LLVM 20 standard container
- Magic `0x10FF10AD`, StringMap per entry, ImageKind enum: Bitcode=2, CUBIN=3, PTX=5,
  SPIRV=6, Object=1. `clang-offload-packager` assembles multi-target archives.
- KOC uses this as the on-disk format — no new container format needed.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-05-offload-ld.md` §Evidence

### E10. libkdl three-tier dispatch table: existing Tier 1 implementation
- `experiments/prototype/src/kdl.c` (~5100 LOC): in-process O(1) hash map operational.
- KOC Tier 1 is what libkdl already implements. Tier 2 (disk) and Tier 3 (JIT) extend it.
- Verified on GTX 1650 (SM 7.5) + CPU; dispatch overhead <2 µs on warm path.
- **Local file:** `/home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c`

### E11. liboffload PR #186088: deferred "follow-up PR" for ranked image selection
- Source: https://github.com/llvm/llvm-project/pull/186088 (open March 2026)
- `parseOffloadBinary` uses a `break` after first-match — no ranking, no cache consulted.
- PR author explicitly deferred multi-image ranking to a follow-up. KOC's Component C2
  is that follow-up.

### E12. cuTENSOR JIT: 6.9x speedup for specialized plans confirms JIT value in Tier 3
- Source: cuTENSOR JIT documentation (NVIDIA)
- URL: https://docs.nvidia.com/cuda/cutensor/latest/just_in_time_compilation.html
- Cold: 1–8 seconds. `cutensorWriteKernelCacheToFile()` for persistence.
- 6.9x speedup vs pre-compiled (H100 PCIe, specific contraction). Validates that
  JIT-specialized variants can justify the one-time cold cost if cached correctly.
- **Local file:** `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-05-gpu-kernel-jit.md` §6

---

## Feasibility

### What exists today (no new work needed)

| Component | Status |
|-----------|--------|
| LLVM ThinLTO content-addressed cache (`ThinLTOCodeGenerator`) | Upstream, production |
| `OffloadBinary` format + `clang-offload-packager` | LLVM 20, upstream |
| Proteus IR embedding + JIT specialization pattern | CGO 2025, Apache 2.0 |
| libkdl Tier 1 in-process dispatch table | Implemented, verified |
| CUDA `mgpuModuleLoadJIT` (PTX JIT path) | LLVM upstream, NVIDIA only |
| NVRTC for Tier 3 JIT fallback (NVIDIA) | Mature, documented |
| comgr for Tier 3 JIT fallback (AMD) | ROCm 6.x, documented |

### What requires new implementation for poster demo

| Task | Effort | Risk |
|------|--------|------|
| SKOI hash function: adapt ThinLTO key scheme for GPU (driver ABI field) | 1 day | Low |
| Disk cache store/load using existing OffloadBinary serialization | 1.5 days | Low — format is standard |
| Wire Tier 2 disk cache into libkdl's `kdl_select_kernel` lookup path | 1 day | Low |
| Benchmark: cold start elimination demo — first-run vs second-run on GTX 1650 | 1 day | Low — hardware available |
| Component C1 MLIR attribute: prototype only, not upstream yet | 0.5 days | Low — attribute boilerplate |
| Poster diagram: three-tier lookup hierarchy + SKOI key construction | 0.5 days | Low |

**Total: ~5.5 days.** Feasible before 2026-04-07 deadline if scoped to NVIDIA path for demo.

### Hardware limitation

GTX 1650 (SM 7.5) available — CUDA path demonstrable end-to-end.
AMD path (comgr Tier 3 JIT): design-only for poster. Acceptable.

### Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LLVM ThinLTO cache API changed in LLVM 20 | Low | Read source before adapting; API is stable |
| "Triton already does this with MegaCache" objection | High | Counter: MegaCache is Python-level, PyTorch-version-keyed, CUDA+ROCm only. KOC is LLVM-IR-keyed (framework-agnostic), OffloadBinary (LLVM-standard format), covers CPU fallback, integrated into liboffload. |
| "CUDA ComputeCache already does this" objection | Medium | Counter: ComputeCache only covers PTX→CUBIN, is NVIDIA-only, discards entire cache on driver upgrade, has no LLVM IR tier. KOC covers the full Python→LLVM IR→binary pipeline, is cross-vendor, and has fine-grained ABI versioning. |
| Triton Issue #4051 (stale hits on dynamic callees) | Medium | SKOI's `kernel_ir_bitcode` key is derived from IR, not Python source — dynamic callee resolution is captured at the IR level, not the source level. KOC is immune to #4051 by design. |
| Security: shared cache directory as attack vector | Low for demo | Include `kernel.sig` Ed25519 field in format; note OCI model for production. |

---

## Upstream Path

### Path 1 — liboffload (shortest, highest-impact)

`parseOffloadBinary` PR #186088 deferred ranked selection. Submit Component C2 as the
promised follow-up:

1. Add `KernelObjectCache` class to `llvm/offload/src/` (reusing ThinLTO cache storage backend)
2. Replace `break` with `rankImages()` + cache lookup in `GenericPluginTy::loadNextBestImage()`
3. Wire `kdl_dispatch_table_t` as the in-process Tier 1 cache
4. RFC on LLVM Discourse (offload category, cc: jdoerfert, huber) — framed as
   "persistent kernel object cache for liboffload"

Expected reception: positive. liboffload is active; PR #186088 explicitly deferred this.

### Path 2 — MLIR GPU dialect (medium-term)

Add `#gpu.cache_key` attribute to `gpu.launch_func` (Component C1). This requires an
RFC to mlir-dev mailing list + Discourse (GPU dialect topic). The attribute carries the
SKOI through lowering so the runtime skips hash recomputation.

This is optional for the poster but creates a natural MLIR DevMtg discussion point.

### Path 3 — Triton upstream (community engagement)

File a Triton RFC: "Framework-agnostic kernel binary caching via LLVM IR bitcode keys."
Cross-link with AMD (AOTriton team), Intel XPU backend, and Meta (PyTorch inductor team).
Triton's existing `$HOME/.triton/cache/` stores `kernel.llir` already — the proposal
extends this to a cross-framework store keyed on that IR, not on the Python source.

This path does not require code changes to Triton; it requires community consensus on the
SKOI key format. The poster is the proof-of-concept that makes the RFC credible.

---

## Scores

| Criterion | Score | Justification |
|-----------|------:|---------------|
| **Novelty** | **8/10** | LLVM-IR-keyed cross-framework GPU kernel cache does not exist. ThinLTO cache adaptation for GPU outputs is unexplored. The SKOI design (IR + driver ABI) is new. Triton's IR-level cache is embryonic (stores kernel.llir but does not key on it). |
| **Feasibility** | **8/10** | All three tiers build on existing infrastructure. NVIDIA path demonstrable on GTX 1650. ThinLTO cache and OffloadBinary code are upstream and stable. Proteus validates embedded-IR approach. Estimated 5.5 days of integration work. |
| **Evidence** | **9/10** | 843-second Meta/vLLM cold start is quantified, public, and independently reproduced (MLSys 2026 vLLM paper). CUDA/Triton/AdaptiveCpp cache designs are documented to implementation level. Proteus 2.8x speedup validates Tier 3. cuTENSOR 6.9x validates caching specialized JIT outputs. |
| **Impact** | **9/10** | Eliminates the largest single ML serving cold start bottleneck (46% of total, 843 s). Framework-agnostic design means PyTorch, vLLM, SGLang, IREE, and custom stacks all benefit. Cross-vendor scope adds AMD/Intel to NVIDIA-only existing solutions. |
| **Community fit** | **8/10** | liboffload PR #186088 already opened the door. LLVM Dublin GPU track audience is exactly the right community (liboffload + MLIR GPU dialect contributors). The ThinLTO cache reuse framing makes this legible as "extending existing infrastructure" not "new project." Red Hat OCI model shows operator-community interest. |
| **Composite** | **8.4/10** | |

---

## One-Paragraph Pitch

A large foundation model's first cold start with torch.compile costs 843 seconds in
Triton JIT compilation — 46% of the total 1,825-second startup. Every major framework
independently invented a kernel binary cache (Triton's `~/.triton/cache/`, CUDA's
`~/.nv/ComputeCache/`, AdaptiveCpp's `~/.acpp/apps/`) but each operates at a different
layer (Python source, PTX, CUBIN), has no cross-vendor sharing, and catastrophically
invalidates on routine upgrades. We propose the **Kernel Object Cache (KOC)**: a
content-addressed persistent store for GPU kernel binaries, keyed on LLVM IR bitcode
(not Python source or PTX), so the cache key is stable across framework versions and
portable across vendor compilation paths. KOC integrates LLVM's existing ThinLTO cache
infrastructure (`llvm/lib/LTO/ThinLTOCodeGenerator.cpp`) with the `OffloadBinary` format
(LLVM 20, magic `0x10FF10AD`) to produce a three-tier lookup: in-process O(1) table
(libkdl Tier 1, <2 µs), on-disk OffloadBinary entry (Tier 2, <5 ms), and JIT fallback
via embedded LLVM IR bitcode (Tier 3, Proteus-style, one-time cost cached back to Tier 2).
The upstream contribution is a concrete follow-up to liboffload PR #186088: a
`KernelObjectCache` class that replaces liboffload's first-match `break` with a ranked,
cache-aware selection. Demonstrated on GTX 1650: second cold start after KOC warmup is
sub-millisecond vs 843 seconds without it.

---

## Key References

1. "Experience in Reducing PT2 Compilation Time for Meta Internal Workloads" — PyTorch Blog 2025
   https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/

2. "Understanding Triton Cache: Optimizing GPU Kernel Compilation" — Red Hat, May 2025
   https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/

3. CUDA Pro Tip: Understand Fat Binaries and JIT Caching — NVIDIA Blog
   https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/

4. AdaptiveCpp Performance Documentation — SSCP cache design
   https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md

5. Proteus: Portable Runtime Optimization of GPU Kernel Execution (CGO 2025)
   https://dl.acm.org/doi/10.1145/3696443.3708939

6. "Skip the JITters: Fast, trusted model kernels with OCI caching" — Red Hat, Jan 2026
   https://next.redhat.com/2026/01/29/skip-the-jitters-fast-trusted-model-kernels-with-oci-caching/

7. "Protecting Triton kernel deployments with cryptographic signatures" — Red Hat, Feb 2026
   https://next.redhat.com/2026/02/05/protecting-triton-kernel-deployments-with-cryptographic-signatures/

8. Compile Time Caching in torch.compile — PyTorch 2.11 Documentation
   https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

9. LLVM PR #66220: [mlir][gpu][NVPTX] Enable NVIDIA GPU JIT compilation path
   https://github.com/llvm/llvm-project/pull/66220

10. liboffload PR #186088: parseOffloadBinary first-match break, deferred follow-up
    https://github.com/llvm/llvm-project/pull/186088

11. cuTENSOR JIT documentation — 6.9x speedup, cutensorWriteKernelCacheToFile
    https://docs.nvidia.com/cuda/cutensor/latest/just_in_time_compilation.html

12. Triton Issue #4051: Cache invalidation with dynamic function calls (unresolved 2026)
    https://github.com/triton-lang/triton/issues/4051

13. LLVM OffloadBinary format — magic 0x10FF10AD, ImageKind enum, LLVM 20
    https://llvm.org/docs/CommandGuide/llvm-offload-binary.html

14. libkdl prototype — experiments/prototype/src/kdl.c (~5100 LOC, this repo)
    /home/akash/PROJECTS/LLVM/experiments/prototype/src/kdl.c

15. Local evidence base:
    - Wave 04 kernel caching: `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-04-kernel-caching.md`
    - Wave 05 GPU JIT: `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-05-gpu-kernel-jit.md`
    - Wave 07 Proteus deep dive: `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/waves/wave-07-proteus-deep-dive.md`
    - Direction 07 JIT-AOT hybrid: `/home/akash/PROJECTS/LLVM/research/mega-survey/heterogeneous-gpu-kernel-dispatch-in-llvm-mlir/directions/07-jit-aot-hybrid-dispatch.md`
    - Topic 10 Triton cross-vendor: `/home/akash/PROJECTS/LLVM/research/mega-survey/20-poster-topics/waves/topic-10-triton-cross-vendor.md`
