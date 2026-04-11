# Elevator Pitch — EuroLLVM Dublin 2026

**Poster:** Measuring and Improving Multi-Target Binary Selection in LLVM's GPU Offload Stack
**Author:** Akash (IIT Patna, CERN GSoC, vLLM)

---

## 60-Second Script

**[0-10s] The Problem**

MLIR can compile one `gpu.module` to NVIDIA, AMD, and Intel targets simultaneously.
`OffloadBinary` can carry all three in a single fat binary.
But at runtime, LLVM picks the *first compatible image* and stops.
There is no "best compatible" -- no metadata to rank, no mechanism to choose.

**[10-25s] What We Measured**

We built the first published layer-by-layer flame graph of the LLVM GPU dispatch stack on a GTX 1650.
The hot-path dispatch floor is 4.26 microseconds -- that is launch plus sync on a null kernel.
Module load dominates: 10.1 microseconds warm, 54.6 microseconds cold.
Symbol lookup is 57 nanoseconds. The selection logic itself -- a dispatch table scan over three real CUBIN variants -- costs 4 nanoseconds per call. Measured, not estimated.

**[25-45s] What We Propose**

Two concrete contributions. First, a 5-key metadata vocabulary for OffloadBinary -- `min_sm`, `min_gfx`, `requires_features`, `variant_priority`, `variant_tag`. Additive, backward-compatible, about 30 lines in the header. It plugs directly into `isMetadataCompatible()` from PR #185663.

Second, a design sketch for `#gpu.runtime_select` -- a new MLIR attribute implementing `OffloadingLLVMTranslationAttrInterface`. It emits a dispatch table and a one-time vendor detection stub. After that, the hot path is identical to `SelectObjectAttr`. Zero steady-state overhead. The pattern is inspired by CPU Function Multi-Versioning with IFunc resolvers.

**[45-60s] Why It Matters**

CERN CMS maintains 80 build configurations to cover heterogeneous GPU clusters. vLLM maintains separate CUDA and HIP codepaths. Cloud containers do not know their GPU at compile time. Runtime variant selection eliminates this combinatorial explosion -- one binary, any GPU, correct kernel, no recompilation.

---

## 30-Second Script

`OffloadBinary` can carry NVIDIA, AMD, and Intel GPU images in one container, but LLVM's runtime picks the first compatible image and stops -- no ranking, no metadata. We measured the full dispatch stack layer by layer on a GTX 1650: hot-path is 4.26 microseconds, module load is 10 microseconds warm, and the dispatch table scan for variant selection costs 4 nanoseconds. We propose a 5-key metadata vocabulary for OffloadBinary and a `#gpu.runtime_select` MLIR attribute that emits a dispatch table at compile time with zero steady-state overhead. One fat binary, any GPU, correct kernel.

---

## 15-Second Script (Tweet-Length Hook)

LLVM can compile for 3 GPU vendors but picks the first compatible binary at runtime. We flame-graphed the dispatch stack -- 4.26 us hot-path, 4 ns for variant selection -- and propose a metadata vocabulary plus a `#gpu.runtime_select` MLIR attribute for best-compatible dispatch. One binary, any GPU.

---

## Back-Pocket Talking Points

### For Joseph Huber (liboffload maintainer)

**Open with:** "Your `isMetadataCompatible()` in PR #185663 is the consumer hook -- we are proposing the vocabulary it needs."

**Key points:**

- The OffloadBinary StringMap has had only 2 standardized keys (`triple`, `arch`) since D122069 in 2022. We propose 5 more, additive, no ABI break.
- `rankImage()` integration path: if liboffload gains a ranking callback, `#gpu.runtime_select`'s `embedBinary()` can emit a call to `rankImage()` instead of inline logic. The two layers compose; they do not compete.
- Our layer benchmarks show `cuModuleLoadData` at 10.1 us warm -- the dominant cost is in the driver, not the selection path. This means the metadata check (sub-microsecond) is effectively free relative to what the driver already does.
- The header constants patch is about 30 lines. The `isMetadataCompatible()` extension is about 40 lines. Small, reviewable, independently landable.

### For Fabian Mora (RFC #88170 author)

**Open with:** "Your RFC separates `gpu.binary` as container from dispatch policy -- we are filling the policy slot."

**Key points:**

- RFC #88170 proposes cleaning up `gpu.binary` semantics and separating the offloading handler. `#gpu.runtime_select` is designed to be a standalone `OffloadingLLVMTranslationAttrInterface` implementation -- it does not depend on the RFC landing but aligns with its direction.
- If the RFC changes `gpu.binary` structure, only TableGen definitions change; `embedBinary()`/`launchKernel()` logic stays the same.
- The PoC runs today: 3 real CUBINs (sm_75, sm_86, sm_89) packed into a real OffloadBinary (14,064 bytes), parsed, selected, launched on a GTX 1650. Variant selection: 380 ns single-shot, 4 ns amortized over 100,000 iterations.

### For Joel Denny (KernelInfo / ORNL)

**Open with:** "Your `kernel-resource-usage` remark pass extracts register counts and occupancy at compile time -- but that metadata never reaches the OffloadBinary."

**Key points:**

- KernelInfo (D123878) produces `sgpr_count`, `vgpr_count`, occupancy data. The pipeline drops it before the binary is packaged. Our deferred Tier 2 keys (`sgpr_count`, `vgpr_count`, `registers_per_thread`, `shared_mem_bytes`) would carry this data through to runtime.
- The KernelInfo-to-OffloadBinary writer integration is a separate patch series, scoped per backend (AMDGPU: ~60 LOC, NVPTX: ~60 LOC).
- Immediate value: even without Tier 2 keys, the Tier 1 `requires_features` key lets the runtime reject images that need hardware capabilities the device lacks (e.g., tensor cores, bf16). This prevents silent fallback to a suboptimal variant.

### For an IREE Person

**Open with:** "IREE's HAL solves dispatch at the framework level. We are solving it at the LLVM layer -- below the framework, above the driver."

**Key points:**

- IREE HAL has a mature device/executable abstraction with multi-target dispatch. That works for IREE users. But torch-mlir, ONNX-RT multi-EP, and anyone else going through MLIR's `gpu.binary` does not get that -- they get `#gpu.select_object`, which resolves at compile time.
- `#gpu.runtime_select` is not competing with HAL. It is filling the same gap at a different layer: MLIR-level emission of runtime dispatch logic that any `gpu.binary` consumer gets for free.
- Our numbers show the selection overhead (4 ns per dispatch) is negligible relative to any real kernel execution. IREE's HAL dispatch overhead is higher because it does more (buffer management, scheduling). Different tradeoff, complementary layers.

### For a Grad Student

**Open with:** "You know how LLVM can compile for NVIDIA and AMD GPUs? The problem is picking the right binary at runtime."

**Key points:**

- Think of it like `ld.so` for GPUs. When you run a program, the dynamic linker picks the right shared library. GPU binaries do not have that -- you compile for one target, or you ship multiple builds.
- We packed 3 different GPU binaries (for 3 NVIDIA architectures) into one file. At runtime, the system detects which GPU you have and picks the best match. The selection itself takes 4 nanoseconds -- basically free.
- The whole GPU dispatch -- sending a kernel to the GPU and waiting for it to finish -- takes about 4 microseconds on our test GPU (GTX 1650). Most of that is the hardware, not our code.
- This matters for places like CERN, where they have clusters with mixed GPU hardware, or for cloud deployments where you do not know which GPU you will get.

---

## Numbers Quick-Reference

All numbers from our benchmarks on GTX 1650 (sm_75), CUDA 13.1, 10,000 warm iterations / 100 cold trials.

| Metric | Value | Source |
|--------|-------|--------|
| Hot-path dispatch (launch + sync) | 4.26 us | bench_layers, warm, n=10,000 |
| Module load, warm | 10.1 us | bench_layers, cuModuleLoadData, n=10,000 |
| Module load, cold | 54.6 us | bench_layers, exec-child, n=100 |
| Symbol lookup | 57 ns | bench_layers, cuModuleGetFunction, n=10,000 |
| Driver shim per-call floor | 50 ns | bench_layers, cuDeviceGet, n=10,000 |
| Variant selection (amortized) | 4 ns | runtime_select_poc, n=100,000 |
| Variant selection (single-shot, 3 CUBINs) | 380-521 ns | runtime_select_poc, OffloadBinary mode |
| Bundle load (kdl) | 4.9 us | bench_dispatch Run 3, n=1,000 |
| Direct cuLaunchKernel baseline | 841 ns | bench_dispatch Run 3, median |
| OffloadBinary file size (3 entries) | 14,064 bytes | offloadbinary_parse output |
| CUBIN size (sm_75, null kernel) | 4,328 bytes | nvcc -arch=sm_75 |
