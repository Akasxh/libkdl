# MLIR Can Compile Your GPU Kernel to Three Vendors at Once. But Who Picks the Right Binary at Runtime?

*Akash -- IIT Patna, CERN GSoC alumnus, vLLM contributor*
*EuroLLVM Developers' Meeting, Dublin 2026*

---

Since August 2025, MLIR can target NVIDIA, AMD, and Intel GPUs from a single `gpu.module`. You write one kernel, attach three targets, run `gpu-module-to-binary`, and get a `gpu.binary` holding three device images. The `OffloadBinary` fat-binary format carries all three in one container.

That is the compilation story. It works. The runtime story is a different matter.

When your program runs on a machine with an NVIDIA GPU, something has to look inside that fat binary, find the right CUBIN, skip the HSACO and the SPIR-V, and hand the correct image to the driver. Today, LLVM does this with `#gpu.select_object` -- an MLIR attribute that resolves the choice at compile time, by index or static target match. If you compile with `#gpu.select_object<0>`, you get image zero, always. If your deployment target changes, you recompile.

The alternative is `liboffload`, LLVM's runtime offload layer. PR #186088 introduced a `parseOffloadBinary` loop that iterates over images, calls `isMetadataCompatible()` on each, and returns the first one that passes. First-compatible-wins. No ranking, no preference, no "this sm_90 CUBIN is better than the sm_75 one for your H100."

## The Gap

The gap is not that runtime selection is impossible. It is that the pieces do not connect.

`isMetadataCompatible()` (PR #185663, merged March 2026) is a runtime consumer hook. It checks whether an image is compatible with the current device. But it can only check two metadata keys: `triple` and `arch`. Those are the only two keys standardized in `OffloadBinary` since it was introduced in D122069 back in 2022. Four years, two keys. The flexible `StringMap<StringRef>` in every entry sits there waiting for vocabulary that never arrived.

Meanwhile, both AMD and NVIDIA embed rich per-kernel metadata in their native formats. AMD HSACOs carry register counts, LDS size, and target-ID feature flags. CUDA cubins carry `EIATTR_REGCOUNT` and `EIATTR_MAX_THREADS`. LLVM's own `kernel-resource-usage` remark pass (D123878, Joel Denny at ORNL) extracts register and occupancy data at compile time. None of it reaches the OffloadBinary string table. The pipeline drops it.

The consumer hook exists. The vocabulary does not. And with PR #186088's first-compatible-wins loop having zero timing instrumentation, nobody has measured what this approach actually costs.

## What We Measured

We built what we believe is the first published layer-by-layer latency breakdown of the LLVM GPU dispatch stack. The hardware is a GTX 1650 (Turing, sm_75, PCIe 3.0). The workload is a null kernel -- one thread, zero shared memory, zero computation -- compiled ahead of time to a CUBIN to eliminate PTX JIT noise.

The protocol: 100 fresh-process cold trials (each using `execve` to get a pristine CUDA driver state, because CUDA is not fork-safe), 10,000 warm iterations per layer with 100 discarded warmup rounds, and a separate 100,000-iteration microbenchmark for the selection path alone.

Here is where the time goes:

| Layer | Operation | Median | Notes |
|-------|-----------|--------|-------|
| 1 | `cuDeviceGet` (driver shim floor) | 50 ns | Per-call driver overhead once init is done |
| 2 | `cuModuleLoadData` (cold, fresh process) | 42.7 us | Includes driver init, context creation |
| 2 | `cuModuleLoadData` (warm, same context) | 10.1 us | ELF parsing + GPU memory allocation |
| 3 | `cuModuleGetFunction` (symbol lookup) | 60 ns | Hash table lookup in driver module table |
| 4 | `cuLaunchKernel` (submit to stream) | 1.57 us | CPU-side push into command buffer |
| 5 | `cuStreamSynchronize` (GPU round-trip) | 2.48 us | Kernel schedule + execute + completion |
| **Hot path total** | **launch + sync** | **4.26 us** | |

The dominant cost on the hot path is the GPU itself: launch submission plus synchronization. Module load, at 10.1 microseconds warm, is 6x more expensive than a launch -- but it is a one-time cost. Once the module is loaded and the function handle is cached, every subsequent dispatch goes straight to `cuLaunchKernel`.

The key finding for runtime variant selection: any selection logic that runs before module load is effectively free. The module load is 10 microseconds. A dispatch table scan over three CUBIN entries takes 4 nanoseconds. That is three orders of magnitude below the noise floor. Optimizing the selection algorithm is solving a problem that does not exist.

## What We Propose

**First: a metadata vocabulary.** Five new keys in `OffloadBinary`'s per-image string table, organized in two tiers. Tier 1 keys are hard constraints -- the runtime rejects the image if they are violated. `min_sm` (minimum CUDA compute capability), `min_gfx` (minimum AMD GFX target, family-tagged to prevent cross-family comparison between CDNA and RDNA), and `requires_features` (named capability tokens like `tensor_core_nv` or `mfma_amd`). Tier 2 keys are ranking hints: `variant_priority` (higher is preferred) and `variant_tag` (human-readable label: `optimized`, `generic`, `fallback`). The implementation is roughly 30 lines of header constants and 40 lines extending `isMetadataCompatible()`. Missing keys mean no constraint. Old runtimes ignore unknown keys. No format version bump, no ABI break.

**Second: a design sketch for `#gpu.runtime_select`.** This is a new MLIR attribute implementing `OffloadingLLVMTranslationAttrInterface` -- the same interface that `#gpu.select_object` implements. Its `embedBinary()` emits N global constants (one per `#gpu.object`), a dispatch table of `{vendor_id, blob_ptr, size}` entries, and a `global_ctors` vendor-detection stub that probes for CUDA/HIP/Level Zero via `dlopen`. At program startup, the stub scans the table, filters by Tier 1 constraints, ranks by Tier 2 priority, loads the winning module, and stores the handle. After that, `launchKernel()` emits the same code as `SelectObjectAttr` -- load the cached function handle, call `mgpuLaunchKernel`. Zero steady-state overhead. The pattern is inspired by CPU Function Multi-Versioning, where LLVM already emits IFunc resolvers at compile time for `target_clones`.

**Third: an upstream path.** The metadata vocabulary goes first as an RFC on Discourse, targeting the Runtimes category. The header constants and `isMetadataCompatible()` extension are small, independently landable patches. The dispatch flame graph stands on its own as a measurement contribution. `#gpu.runtime_select` comes last, consuming both. Each step is useful without the others; they do not form a monolithic dependency chain.

## What We Built

We built a proof-of-concept that dispatches from a real LLVM `OffloadBinary` on real hardware. Three CUBINs compiled with `nvcc` for sm_75, sm_86, and sm_89 -- packed into a single OffloadBinary file (14,064 bytes, magic `0x10FF10AD`, format matching `llvm/include/llvm/Object/OffloadBinary.h`). Since `clang-offload-packager` was not available on our test machine, we wrote our own writer and parser (~270 LOC) implementing the exact binary format. The PoC reads the file, parses the header and per-entry metadata, detects the local GPU via `dlopen`-probed `cuInit`, filters entries by compute capability, and selects the correct variant. On the GTX 1650: sm_75 selected, sm_86 and sm_89 rejected. Kernel loaded via `cuModuleLoadData`, launched via `cuLaunchKernel`, synchronized. Variant selection: 380 ns single-shot, 4-6 ns amortized over 100,000 iterations.

The broader prototype (`kdl.c`, ~5,100 LOC) uses a custom bundle format rather than OffloadBinary, and I want to be honest about that boundary. It validates the runtime mechanics -- vendor detection, dispatch table construction, capability-based selection, module caching -- but not LLVM format consumption. The OffloadBinary PoC closes that gap for the critical path: real format, real CUBINs, real driver calls, real hardware.

## What's Next

The metadata vocabulary RFC is ready to post on Discourse. The flame graph data is collected and reproducible. The design sketch for `#gpu.runtime_select` is detailed enough for community feedback. The next steps are engaging with the people who maintain the pieces this touches -- Joseph Huber on `isMetadataCompatible()` and the offload runtime, Fabian Mora on RFC #88170 and `gpu.binary` semantics, Joel Denny on the KernelInfo-to-OffloadBinary pipeline. If the vocabulary lands and the dispatch-policy slot in `gpu.binary` stabilizes, a ~780-line implementation of `RuntimeSelectAttr` can follow. One fat binary, any GPU, correct kernel, no recompilation.

---

**Links**

- Prototype source and benchmarks: [GitHub repository] *(link TBD)*
- Metadata vocabulary RFC: [Discourse thread] *(link TBD after posting)*
- EuroLLVM Dublin 2026 poster: [Conference page] *(link TBD)*
- LLVM OffloadBinary format: [`llvm/include/llvm/Object/OffloadBinary.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/OffloadBinary.h)
- PR #186088 (first-compatible-wins loop): [llvm/llvm-project#186088](https://github.com/llvm/llvm-project/pull/186088)
- PR #185663 (`isMetadataCompatible`): [llvm/llvm-project#185663](https://github.com/llvm/llvm-project/pull/185663)
- RFC #88170 (GPU dialect cleanup): [llvm/llvm-project#88170](https://github.com/llvm/llvm-project/issues/88170)
