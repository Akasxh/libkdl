// Field Guide: Heterogeneous GPU Dispatch in LLVM/MLIR
// For EuroLLVM Dublin 2026 Poster Session

#set page(margin: (x: 2.5cm, y: 2cm), numbering: "1")
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true, leading: 0.65em)

#let green = rgb("#2e5339")
#show heading.where(level: 1): set text(fill: green, size: 18pt, weight: "bold")
#show heading.where(level: 2): set text(fill: green, size: 14pt, weight: "bold")
#show heading.where(level: 3): set text(fill: green, size: 12pt, weight: "bold")

#let callout(body) = block(
  width: 100%, inset: 10pt, radius: 2pt,
  stroke: 0.5pt + luma(180),
  fill: luma(248),
  text(style: "italic", body)
)

#align(center)[
  #v(3cm)
  #text(size: 28pt, weight: "bold", fill: green)[Heterogeneous GPU Dispatch\ in LLVM/MLIR]
  #v(0.5cm)
  #text(size: 14pt, style: "italic")[a field guide for your poster session]
  #v(1cm)
  Part I --- The problem space\
  Part II --- Your contributions\
  Part III --- What you'll be asked\
  Part IV --- Future directions
  #v(1cm)
  #text(style: "italic")[~ Dublin, April 2026 ~]
  #v(1cm)
  #line(length: 60%, stroke: 0.5pt + luma(180))
  #v(0.3cm)
  #text(size: 10pt, fill: luma(120))[Heterogeneous GPU Dispatch · a field guide for EuroLLVM]
]

#pagebreak()

= Part I --- The Problem Space

== 1. What multi-target GPU compilation actually means

MLIR can compile a single `gpu.module` to multiple GPU vendors in one pipeline. Since August 2025, three vendor targets coexist: *NVVM* (NVIDIA), *ROCDL* (AMD), and *XeVM* (Intel, via PR \#148286). The `gpu-module-to-binary` pass takes a `gpu.module` and produces a `gpu.binary` containing N embedded device objects --- one per target.

The result is packed into LLVM's *OffloadBinary* format (magic `0x10FF10AD`), which was introduced in 2022 (D122069). Each image in the binary carries a `StringMap` of metadata --- but only two keys are standardized: `triple` and `arch`.

#callout[
  The compile-time story is complete. You can write one kernel, compile it for three vendors, and pack the results into a single binary. The problem is what happens next.
]

The `#gpu.select_object` attribute resolves *at compile time* --- it picks one target by index during LLVM IR translation. There is no runtime path. The `GPUOffloadingLLVMTranslationAttrInterface` is the extensibility point where a runtime-aware handler would plug in, but nobody has implemented one.

== 2. The runtime gap nobody filled

When `liboffload` loads an OffloadBinary at runtime, it iterates through the images and picks the *first compatible one*. PR \#186088 (opened March 2026) makes this explicit:

#callout[
  "For now only the first compatible image in the binary is loaded. While it might be desirable to add support for loading multiple images, our current interface is limiting and it's unclear if in all cases this behavior is desirable."\ --- PR \#186088 author
]

This is not a bug --- it's a design deferral. The offload stack has no vocabulary for expressing "this image is better than that one." The `isMetadataCompatible()` hook (PR \#185663, merged March 2026) checks `triple` + `arch` but has no concept of features, priority, or ranking.

*Evidence the gap is real:*
- IREE \#12230 (kernel specialization): stalled since May 2023
- IREE \#15334 (multi-versioning epic): all tasks unchecked since Oct 2023
- IREE \#50 (target configuration): open since 2019
- Issue \#75356: Chapel team requesting "target-specific binary selection at runtime" since 2023
- RFC \#88170: GPU dialect cleanup separates container from dispatch policy --- policy slot is empty

== 3. Why "just pick the first binary" is a real problem

Three deployment scenarios where first-wins dispatch fails:

*HPC clusters (CERN):* ~80 build configurations for heterogeneous GPU clusters (A100 + MI250X + CPU). A single fat binary with runtime selection eliminates this combinatorial explosion.

*Cloud containers:* AWS p4/p5, Azure ND --- the GPU model is unknown at container image build time. You ship one binary and resolve at launch.

*ML frameworks:* torch-mlir and ONNX Runtime both produce `gpu.binary` with multiple targets. Each framework currently reimplements its own dispatch. Moving selection into LLVM eliminates per-framework duplication.

#callout[
  Runtime variant selection is not theoretical. cuBLAS selects from hundreds of GEMM variants at runtime. cuDNN v9 uses NVRTC JIT with 3 heuristic modes. PyTorch's dispatcher computes DispatchKeySet on every op call. MLIR's offload stack is the only major system without a runtime dispatch policy.
]

#pagebreak()

= Part II --- Your Contributions

== 4. C1: The metadata vocabulary (5 keys)

The OffloadBinary format includes a flexible `StringMap` per image, but the vocabulary hasn't grown in four years. You propose five new keys:

#table(
  columns: (auto, auto, 1fr),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header(
    text(weight: "bold")[Key],
    text(weight: "bold")[Tier],
    text(weight: "bold")[Purpose],
  ),
  [`min_sm`], [MUST], [Minimum CUDA compute capability (e.g. 75 for sm_75)],
  [`min_gfx`], [MUST], [Minimum AMD GFX version within ISA family],
  [`requires_features`], [MUST], [Named capability tokens (tensor_core, bf16, tma)],
  [`variant_priority`], [MAY], [Higher = preferred among compatible images],
  [`variant_tag`], [MAY], [Human-readable label: generic, optimized, fallback],
)

These keys are *backward-compatible*: missing keys mean no constraint. Old runtimes ignore unknown keys. No format version bump, no ABI break. The `isMetadataCompatible()` extension is ~30 lines:

```c
bool isMetadataCompatible(image, device) {
  if (image.Triple != device.Triple) return false;
  if (auto minSm = image.getString("min_sm"))
    if (parseSm(device.Arch) < stoul(*minSm))
      return false;
  if (auto feats = image.getString("requires_features"))
    for (auto &tok : split(*feats, ','))
      if (!device.hasCapability(tok)) return false;
  return true;
}
```

== 5. C2: The dispatch flame graph

Nobody has published a per-layer latency breakdown of the LLVM GPU dispatch path. You measured each layer individually on a GTX 1650 (sm_75), CUDA 13.1, using a null kernel CUBIN.

#table(
  columns: (1fr, auto, auto),
  inset: 8pt,
  stroke: 0.5pt + luma(200),
  table.header(
    text(weight: "bold")[Operation],
    text(weight: "bold")[Median],
    text(weight: "bold")[Share],
  ),
  [cuModuleLoadData (cold)], [*36.0 µs*], [*89.6%*],
  [cuModuleLoadData (warm)], [9.6 µs], [—],
  [cuModuleGetFunction], [63 ns], [0.2%],
  [cuLaunchKernel], [1.65 µs], [4.1%],
  [cuStreamSynchronize], [2.45 µs], [6.1%],
  text(weight: "bold")[Hot-path total], text(weight: "bold")[4.1 µs], text(weight: "bold")[launch+sync],
  text(weight: "bold", fill: rgb("#0d9488"))[Selection overhead], text(weight: "bold", fill: rgb("#0d9488"))[3--6 ns], text(weight: "bold", fill: rgb("#0d9488"))[< 0.02%],
)

#callout[
  Module loading dominates the cold path at 90%. Selection at 3--6 ns is faster than a single L2 cache access. The dispatch mechanism is essentially free --- the question is what information drives it.
]

*Methodology:* 100 cold trials via exec-child isolation, 10K warm iterations, 3-run cross-run medians. CPU-pinned (taskset -c 0). p99 values from pinned runs are tighter than unpinned.

*The GTX 1650 question:* You measure dispatch overhead, not kernel performance. The CUDA driver API path (cuModuleLoadData, cuLaunchKernel) has similar latency across GPU tiers. Selection (3--6 ns) is pure CPU work.

== 6. C3: The \#gpu.runtime_select design

A new MLIR attribute that attaches dispatch policy directly to `gpu.binary`:

```
gpu.binary @kernels <#gpu.runtime_select<
    strategy = "rank_by_priority",
    fallback = "cpu">> [
  #gpu.object<#nvvm.target<chip="sm_75">, bin="...cubin...">,
  #gpu.object<#rocdl.target<chip="gfx90a">, bin="...hsaco...">,
  #gpu.object<#nvvm.target<chip="sm_90">, bin="...cubin...">
]
```

This is a *proposed* extension, not upstream code. The design is inspired by CPU Function Multi-Versioning (IFunc resolvers via `target_clones`). The attribute implements `GPUOffloadingLLVMTranslationAttrInterface` and emits:
1. N global blobs (one per variant)
2. A dispatch table + vendor detection in `global_ctors`
3. A single `@kernels_module_ptr` that `launchKernel()` loads from

After one-time selection, the hot path is a single pointer load --- identical to compile-time `#gpu.select_object`. *Zero overhead after initialization.*

== 7. The prototype: libkdl

*libkdl* (Kernel Dynamic Linker) is a standalone C library:
- `kdl.c`: 5,157 lines --- device discovery, OffloadBinary parsing, metadata scoring, dispatch
- `runtime_select_poc.c`: 664 lines --- end-to-end PoC that reads real OffloadBinary, selects the right variant, and launches on GPU

The PoC packs 3 real CUBINs (sm_75 + sm_86 + sm_89) into OffloadBinary format, correctly selects sm_75 on GTX 1650, rejects sm_86/sm_89, and launches via cuModuleLoadData → cuLaunchKernel → cuStreamSynchronize.

#pagebreak()

= Part III --- What You'll Be Asked

== 8. Likely questions and crisp answers

*"How is this different from Triton/Helion?"*\
Triton and Helion recompile per target at JIT time. We select from pre-compiled variants at load time. Both are needed --- Triton for development iteration, libkdl for deployment where JIT latency is unacceptable.

*"ML kernels are known at compile time"* (Reviewer 91B's argument)\
cuBLAS selects from hundreds of GEMM variants at runtime. cuDNN v9 uses NVRTC JIT. PyTorch's DispatchKeySet fires on every op. Meta's KernelEvolve generates per-accelerator variants. ML is demonstrably not static.

*"Why GTX 1650? That's ancient."*\
We measure dispatch overhead, not kernel throughput. The CUDA driver API path has similar latency across GPU tiers. Selection at 3--6 ns is pure CPU work --- independent of GPU generation.

*"Is this a real system or just a proposal?"*\
5,157 + 664 LOC prototype with measured dispatch latency on real hardware. The `#gpu.runtime_select` MLIR attribute is a proposed design --- we believe it should be an upstream RFC.

*"Why not just extend liboffload directly?"*\
That's exactly what we propose. Step 1 of the upstream path is a metadata RFC for liboffload's existing hook.

*"What about IREE?"*\
IREE's HAL does multi-target compilation and dispatch, but ranked selection is unimplemented (issues \#12230 and \#15334 stalled since 2023). IREE requires the full IREE runtime (100K+ LOC). Our approach is lightweight and MLIR-native.

== 9. The upstream path

*Step 1: Metadata RFC* --- Propose 5 keys for OffloadBinary's StringMap. Low-risk, additive change. ~30 LOC header patch. PR \#185663's `isMetadataCompatible()` hook is already merged.

*Step 2: liboffload policy slot* --- Add a pluggable selection hook to liboffload's existing load path. The current "first-wins" becomes the default policy; ranked selection becomes an opt-in alternative.

*Step 3: \#gpu.runtime_select RFC* --- ~780 LOC implementing `OffloadingLLVMTranslationAttrInterface`. Emits dispatch table + vendor detection at compile time, resolves at runtime via `global_ctors`.

Each step is independently useful. Step 1 alone unlocks dispatch for any downstream consumer of OffloadBinary.

== 10. Related work you should know cold

*IREE HAL* --- Google's ML compiler runtime. Full-stack, 100K+ LOC. Ranked selection unimplemented.

*chipStar* --- HIP-over-SPIR-V portability layer. Runtime dispatch via SPIR-V translation, not pre-compiled variant selection.

*Proteus (CGO 2025)* --- LLNL's GPU JIT specializer. 2.8x AMD, 1.78x NVIDIA via constant specialization. Always JIT, no AOT fast path.

*AdaptiveCpp SSCP* --- Always-JIT from LLVM IR. +30% over static CUDA via runtime specialization. 100--600ms cold start.

*KernelEvolve (Meta, ISCA 2026)* --- LLM-generated kernel variants with UCB tree search for selection. Pre-computed routing table, not runtime adaptive.

*Helion (PyTorch Foundation, April 2026)* --- Portable kernel DSL. Autotuner-compiles to per-target Triton code. Different approach: recompile, not select.

*CPU FMV (target_clones)* --- The structural analogy. LLVM emits IFunc resolvers at compile time for `target_clones`. `#gpu.runtime_select` applies the same pattern to cross-vendor GPU binaries.

#pagebreak()

= Part IV --- Future Directions

== 11. Where this research goes next

The dispatch measurement proves the mechanism is free (3--6 ns). The interesting question is: *what information drives the selection?*

*Profile-guided dispatch:* After dispatching a variant, measure actual execution time via CUDA events. Use a multi-armed bandit (epsilon-greedy or UCB1) to explore alternatives. Update the dispatch table online. This is "PGO for dispatch decisions" --- nobody does it cross-vendor.

*Contention-aware dispatch:* On heterogeneous nodes (A100 + MI300X + CPU), the optimal dispatch depends on current device utilization and data locality --- not just kernel performance. This is a contextual bandit with non-stationary rewards.

*Dispatch-fusion co-optimization:* A GEMM+ReLU fusion is profitable on H100 but worse on CPU. The fusion boundary depends on the dispatch target. No system jointly optimizes these.

*Target venue:* CGO 2027 --- "Profiled Adaptive Dispatch for Cross-Vendor GPU Kernel Selection."

== 12. Glossary

#table(
  columns: (auto, 1fr),
  inset: 6pt,
  stroke: 0.5pt + luma(200),
  [`OffloadBinary`], [LLVM's container format for multi-target device images (magic 0x10FF10AD)],
  [`gpu.binary`], [MLIR op that holds compiled GPU code],
  [`gpu.select_object`], [MLIR attribute that picks one target at compile time],
  [`liboffload`], [LLVM's runtime library for GPU offloading],
  [`NVVM`], [MLIR target for NVIDIA (generates PTX/CUBIN)],
  [`ROCDL`], [MLIR target for AMD (generates HSACO)],
  [`XeVM`], [MLIR target for Intel Xe GPUs],
  [`IFunc`], [GNU indirect function --- runtime resolver for CPU FMV],
  [`global_ctors`], [C++ static initializers --- where one-time dispatch runs],
  [`FMV`], [Function Multi-Versioning via `target_clones`],
  [`cuModuleLoadData`], [CUDA driver API: load a compiled module into GPU context],
  [`cuLaunchKernel`], [CUDA driver API: submit a kernel to the GPU],
  [`StringMap`], [Key-value metadata attached to each OffloadBinary image],
)
