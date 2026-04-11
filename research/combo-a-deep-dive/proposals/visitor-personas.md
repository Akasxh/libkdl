# EuroLLVM Dublin Poster — Visitor Interaction Scripts

**Poster:** "Measuring and Improving Multi-Target Binary Selection in LLVM's GPU Offload Stack"
**Author:** Akash (IIT Patna, CERN GSoC alumnus, vLLM contributor)
**Date:** 2026-04-10
**Purpose:** Rehearsal scripts for 10 simulated poster visitors. Each includes an opening question, a 3-turn conversation, and responses grounded in our actual data.

---

## Persona 1: Joseph Huber (AMD/LLNL, liboffload maintainer)

**Background:** Maintains the liboffload runtime, authored PR #185663 (`isMetadataCompatible()`), deeply involved in PR #186088 (the `parseOffloadBinary` loop). Knows the offload stack better than anyone. Will immediately spot anything that misrepresents the runtime's architecture.

### Turn 1 — Opening

**Huber:** "So you want to add a `rankImage()` to the offload runtime? We deliberately kept `isMetadataCompatible()` as a boolean filter — adding ranking pushes policy into the runtime, which we avoided on purpose."

**Response:** "We are not proposing changes to liboffload's policy. We agree with the design decision — liboffload is mechanism, not policy, as the RFC explicitly states. Our proposal works at a different layer. The metadata vocabulary — `min_sm`, `min_gfx`, `requires_features`, `variant_priority`, `variant_tag` — those five keys go into the OffloadBinary string table that already exists but has only `triple` and `arch` after four years. Your `isMetadataCompatible()` hook is the consumer — we are proposing the vocabulary it consumes. The ranking logic itself lives in the MLIR-emitted code via `#gpu.runtime_select`, not in liboffload. If liboffload ever wants a `rankImage()` callback, the `embedBinary()` method can delegate to it instead of inlining the selection — the two layers compose, they do not compete."

### Turn 2 — Technical Probe

**Huber:** "The string-map in OffloadBinary is flexible by design, but standardizing keys means everyone has to agree on semantics. What happens when `min_sm=75` is in the string table but the runtime doesn't know what `min_sm` means? Old runtimes will just ignore it."

**Response:** "Exactly — and that is the backward compatibility model. Missing keys mean no constraint. An old runtime that does not understand `min_sm` ignores it and falls through to `triple`/`arch` matching, which is today's behavior. No ABI break, no format version bump. The keys are opt-in when present. The implementation is about 40 lines added to `isMetadataCompatible()` — check if the key exists, parse the value, compare against the device's capability. The header constants patch is about 20 lines in `OffloadBinary.h`. We have the pseudocode for the extended `isMetadataCompatible()` — it chains with the existing `triple`/`arch` checks, adding `min_sm` and `requires_features` as additional reject-if-violated gates."

### Turn 3 — The Hard Question

**Huber:** "Why five keys? Why not just `variant_priority` and let the compiler encode whatever policy it wants into a single integer?"

**Response:** "Because `variant_priority` alone is a compile-time decision that cannot adapt to runtime hardware. If I compile with three images — sm_75, sm_86, sm_90 — and assign priorities 1, 2, 3, that ranking is correct for an sm_90 device but wrong for an sm_75 device where the sm_90 image cannot even load. The Tier 1 keys (`min_sm`, `min_gfx`, `requires_features`) are *constraint* keys — they reject incompatible images before ranking. `variant_priority` is a Tier 2 *ranking* key that breaks ties among compatible images. You need both: filter then rank. Without the filter keys, priority alone either requires the compiler to enumerate all possible device combinations at compile time, or the runtime loads an image that the device cannot execute."

---

## Persona 2: Fabian Mora (U Delaware, GPU Dialect Cleanup RFC #88170 author)

**Background:** Authored the RFC proposing to separate `gpu.binary` as a container from dispatch policy. The RFC is active and unresolved. Directly affected by any new attribute that implements `OffloadingLLVMTranslationAttrInterface`.

### Turn 1 — Opening

**Mora:** "I saw you reference my RFC. Does `#gpu.runtime_select` assume the cleanup lands in a specific form? Because the RFC is still open and the direction could change."

**Response:** "We designed for exactly that uncertainty. `#gpu.runtime_select` implements `OffloadingLLVMTranslationAttrInterface` — the same two-method interface (`embedBinary` + `launchKernel`) that `#gpu.select_object` already uses. That interface is the stable extension point. If your RFC restructures `gpu.binary` — say, changes the container model or removes the `offloadingHandler` attribute — the TableGen definition changes, but the `embedBinary`/`launchKernel` implementation logic does not. If the RFC concludes without a dispatch-policy slot, `#gpu.runtime_select` can land independently as a standalone attribute. We are building on the interface, not on any particular RFC outcome."

### Turn 2 — Design Alignment

**Mora:** "The cleanup direction is about separating concerns — the container holds objects, the policy decides which one to use. Does your attribute respect that separation, or does it bake policy into the container?"

**Response:** "It respects it. The `gpu.binary` op holds the `#gpu.object` entries — that is the container. The `#gpu.runtime_select` attribute is the policy that sits in the `offloadingHandler` slot, separate from the container. What `embedBinary()` emits is: N separate LLVM global constants (one per object — the container data), plus a dispatch table and a vendor-detection constructor (the policy logic). The `launchKernel()` method is identical to `SelectObjectAttr` — it loads from a cached module pointer. So container data and dispatch policy are distinct in the emitted IR, and the MLIR-level separation between `gpu.binary` (container) and `#gpu.runtime_select` (handler) maps directly to that IR-level separation."

### Turn 3 — Community Fit

**Mora:** "If I wanted to prototype this on top of my cleanup patches, what would I need from you?"

**Response:** "Three things. First, the metadata vocabulary — the five keys in `OffloadBinary.h` that give the runtime something to filter on. That is an independent patch series with no dependency on your RFC. Second, the `RuntimeSelectAttr.cpp` implementation, which is about 400 lines of C++ implementing `embedBinary` and `launchKernel`. Third, the `--gpu-mark-runtime-select` pass that rewrites `#gpu.select_object` to `#gpu.runtime_select` when multiple objects are present. The total is about 780 lines. If your cleanup changes the attribute attachment point, we adapt the TableGen — the logic stays the same. We could coordinate on a shared branch if you are interested."

---

## Persona 3: Joel Denny (ORNL, KernelInfo pass author)

**Background:** Authored D123878, the `kernel-resource-usage` remark pass that extracts register counts and occupancy data at compile time. This data is produced today but never reaches OffloadBinary.

### Turn 1 — Opening

**Denny:** "You mention KernelInfo in your deferred keys section. How would register counts and occupancy data actually flow from the remark pass into the OffloadBinary string table?"

**Response:** "Today the pipeline drops that data. Your `kernel-resource-usage` pass extracts `sgpr_count`, `vgpr_count`, and occupancy at compile time — it emits remarks, but those remarks are not consumed by the OffloadBinary writer. The flow would be: your pass emits the data as remark metadata or as module-level attributes, then the backend writer (AMDGPU or NVPTX) picks up those attributes when constructing the OffloadBinary entry and writes them into the string-map as standard keys. We deferred `sgpr_count`, `vgpr_count`, `registers_per_thread`, `shared_mem_bytes` to a follow-up specifically because they require that writer integration in each backend — it is a separate patch series, roughly 60 lines per backend."

### Turn 2 — Value Proposition

**Denny:** "What would those keys actually be used for at runtime? The runtime does not have an occupancy calculator."

**Response:** "Two uses. First, constraint checking: if an image was compiled assuming 64 VGPRs per thread and the device only supports 48, that is a hard reject — the kernel will not execute correctly. That is a Tier 1 use, like `min_sm`. Second, ranking: given two compatible images for the same device — one using 32 VGPRs (higher occupancy, lower per-thread performance) and one using 64 VGPRs (lower occupancy, higher per-thread performance) — the runtime could use `variant_priority` combined with resource usage metadata to pick the better variant for the current workload. The runtime does not need a full occupancy calculator — it needs simple threshold checks and a comparison function. The heavy analysis stays at compile time in your pass; the runtime just reads the result."

### Turn 3 — Concrete Next Step

**Denny:** "Is FLOP counting the next step? We have been discussing adding compute intensity estimates to KernelInfo."

**Response:** "That would be extremely valuable for ranked selection. If the OffloadBinary carries both resource usage (VGPRs, shared memory) and compute intensity (FLOPs per byte), the runtime has enough information to make a roofline-informed choice between variants — pick the memory-bound variant on bandwidth-heavy devices, pick the compute-bound variant on devices with more ALUs. But that is beyond our current scope. The immediate step is getting the existing KernelInfo data — register counts and occupancy — into the OffloadBinary string table. That alone enables the constraint checking. Compute intensity ranking is a natural follow-on once the plumbing exists."

---

## Persona 4: Ben Vanik (Google, IREE lead)

**Background:** Leads IREE, which has its own HAL (Hardware Abstraction Layer) for multi-target dispatch. IREE has had runtime device selection since 2019 but incomplete ranked selection (issues #50, #12230, #15334 open for 6 years). Will ask why this is not just a contribution to IREE.

### Turn 1 — Opening

**Vanik:** "IREE already does multi-target dispatch at the HAL level. How is this different from what we have been doing for six years?"

**Response:** "IREE operates at the full-stack runtime level — HAL modules, device queues, the whole execution model. Your dispatch granularity is HAL module, not individual kernel. Our work is at a different layer: the LLVM OffloadBinary format and the MLIR `gpu.binary` op. The difference matters for projects that use MLIR's GPU dialect pipeline but do not use IREE's runtime — torch-mlir, ONNX-RT multi-EP, standalone OpenMP offload. Those projects compile through `gpu-module-to-binary`, produce `OffloadBinary` containers, and dispatch through liboffload. They have no access to IREE's HAL. The metadata vocabulary and `#gpu.runtime_select` give those pipelines the same capability at the LLVM layer, without requiring an IREE dependency."

### Turn 2 — Technical Comparison

**Vanik:** "Our HAL has device capability queries, queue selection, multi-device scheduling. Your five metadata keys seem very limited compared to that."

**Response:** "They are limited, and deliberately so. IREE's HAL is a full runtime abstraction. Our contribution is a thin metadata layer on an existing format — five keys in a string table, plus a selection attribute that emits a dispatch table. The design principle is: the minimum vocabulary that enables runtime selection for LLVM's existing offload pipeline, not a replacement for a full runtime like IREE. For projects that want IREE's capabilities, they should use IREE. For projects that want basic 'pick the right binary from a fat binary' without adopting an entire runtime framework, this is the lightweight alternative. The five keys map to the two things the runtime must know: can this image run on this device (Tier 1 constraints), and which compatible image is best (Tier 2 ranking)."

### Turn 3 — Composition

**Vanik:** "Could IREE consume your metadata keys? We already read OffloadBinary containers in some paths."

**Response:** "Yes, and that is a value of standardizing at the OffloadBinary level rather than inventing a new format. If IREE reads OffloadBinary containers, it gets the metadata keys for free — `variant_priority` and `requires_features` are immediately useful for your HAL module selection. The keys are additive to the existing format, so IREE can read them when present and ignore them when absent. The ranked selection that has been an open issue in IREE for six years — `variant_priority` solves the simple case without requiring changes to the HAL device query infrastructure. For the complex case, the deferred resource-usage keys (registers, occupancy) would give IREE's scheduler more information for device-queue assignment."

---

## Persona 5: AMD GPU Compiler Engineer

**Background:** Works on the AMDGPU backend. Cares about target-ID handling, ISA compatibility between GFX architectures, and whether this proposal understands the CDNA/RDNA distinction.

### Turn 1 — Opening

**AMD Engineer:** "Your `min_gfx` key — how do you handle the fact that gfx90a and gfx1100 are completely incompatible architectures? A simple >= comparison does not work for AMD."

**Response:** "We handle this explicitly. The `min_gfx` key uses a family-tagged format: `gfx90a:cdna2`. The family tag — `cdna1`, `cdna2`, `cdna3`, `rdna1`, `rdna2`, `rdna3` — scopes the comparison. An image tagged `gfx90a:cdna2` is only compared against devices in the CDNA2 family. A `gfx1100:rdna3` image is only compared within RDNA3. Cross-family images are rejected before the `min_gfx` comparison even runs — they fail at the `triple`/`arch` level or via the family tag mismatch. The key insight is that within a family, the arch numbers are monotonically ordered. Between families, they are not. The family tag makes that explicit."

### Turn 2 — Target-ID Features

**AMD Engineer:** "What about xnack and sramecc? Those are target-ID features that affect binary compatibility. gfx90a with xnack+ is not the same as gfx90a without."

**Response:** "Today, `areArchsCompatible()` in `OffloadBinary.cpp` already hard-codes xnack and sramecc parsing — it is one of the few places the runtime has vendor-specific logic. Our `requires_features` key extends this. You would encode `requires_features=xnack+` or `requires_features=sramecc+` in the OffloadBinary string table, and the extended `isMetadataCompatible()` checks those tokens against the device's reported capabilities. This is cleaner than the current approach of parsing target-ID suffixes from the arch string, because it separates the feature requirement from the architecture identifier. The AMDGPU writer patch — step 3 in our implementation plan, about 60 lines — would emit these keys from the target-ID that the backend already knows."

### Turn 3 — Practical Concern

**AMD Engineer:** "Does this actually work on ROCm? Have you tested on gfx90a or gfx1100?"

**Response:** "Honest answer: no. The prototype has been validated on NVIDIA GTX 1650 (sm_75) and CPU fallback. The AMD code path in the prototype exists — `kdl_discover_hip()` at lines 699-749 of `kdl.c` — but it was tested via mocked HIP entry points only. We did not have access to ROCm hardware. The poster states this explicitly. The AMDGPU backend integration for the metadata writer is designed based on the existing `areArchsCompatible()` logic and the AMD Code Object V5 metadata format, which we have studied. But hardware validation on CDNA or RDNA devices is a gap we acknowledge. If you have a system we could test on, that would be extremely valuable."

---

## Persona 6: Intel oneAPI Engineer

**Background:** Works on the Level Zero runtime, XeVM target, SPIR-V compilation path. Interested in whether this proposal includes Intel's stack or is NVIDIA/AMD-only.

### Turn 1 — Opening

**Intel Engineer:** "I see NVVM and ROCDL targets in your examples. Where does XeVM fit? What about Level Zero and SPIR-V?"

**Response:** "XeVM is a first-class target in this proposal. The tri-vendor support is actually one of the motivating factors — since PR #148286 merged in August 2025, MLIR has `#nvvm.target`, `#rocdl.target`, and `#xevm.target` in a single `gpu.binary`. But `#gpu.select_object` can only pick one at compile time. `#gpu.runtime_select` is specifically designed for the case where you compile all three and pick at runtime. The vendor detection stub probes `cuInit` for NVIDIA, `hipInit` for AMD, and `zeInit` for Intel — via `dlopen`, so none of the vendor SDKs need to be linked at compile time. For SPIR-V specifically, `requires_features` can carry tokens like `spirv_1_6` or `subgroup_16` to express SPIR-V capability requirements."

### Turn 2 — Level Zero Integration

**Intel Engineer:** "Level Zero has its own module loading path — `zeModuleCreate` takes SPIR-V or native binary. How does your dispatch table handle that?"

**Response:** "The dispatch table is format-agnostic — each entry is `{vendor_id, blob_ptr, size}`. For Intel, the blob is SPIR-V or a native Gen binary, and the runtime helper calls `zeModuleCreate` instead of `cuModuleLoadData` or `hipModuleLoadData`. The `launchKernel()` emission in `#gpu.runtime_select` delegates to the same `mgpuModuleGetFunction` + `mgpuLaunchKernel` wrappers that the existing GPU runtime support library uses. The vendor-specific loading is behind those wrappers. The one caveat: we have not measured the Level Zero dispatch path. Our flame graph is CUDA-only — GTX 1650, `cuModuleLoadData` at 10.1 microseconds warm, `cuLaunchKernel` at 1.6 microseconds. The `zeModuleCreate` path likely has different latency characteristics, especially for SPIR-V JIT. That is a measurement we would need Intel's help with."

### Turn 3 — SPIR-V Portability

**Intel Engineer:** "If I have a SPIR-V binary that can run on both Intel and AMD via their Vulkan drivers, does your selection handle that?"

**Response:** "Today, no — the vendor detection is based on `zeInit` versus `hipInit`, which are distinct entry points. A single SPIR-V binary that is portable across Vulkan implementations would need a different selection path, likely probing `vkCreateInstance` and then querying device capabilities. That is a valid extension but beyond our initial scope. The initial design handles the common case: distinct binaries compiled for distinct vendor targets, one per `#gpu.object` entry. The SPIR-V-portability case — one binary, multiple possible runtimes — is architecturally different and would be a good topic for the `requires_features` vocabulary to evolve toward. The mechanism supports it; the initial vocabulary does not cover it yet."

---

## Persona 7: Skeptical Grad Student

**Background:** Smart, well-read, doing a PhD in compilers or HPC. Has read enough systems papers to spot hand-waving. Will push on novelty and intellectual contribution.

### Turn 1 — Opening

**Grad Student:** "So this is basically a dispatch table with some metadata keys. What is actually novel here? Every GPU runtime has done variant selection since CUDA fatbin in 2008."

**Response:** "Fair challenge. The novelty is not the concept of runtime dispatch — you are right that CUDA fatbin has done this for NVIDIA-only binaries since 2008. The novelty is three specific things that do not exist today in LLVM. First: there is no standard metadata vocabulary for OffloadBinary beyond `triple` and `arch` — the string table has been extensible for four years and nobody has defined keys. We propose five. Second: nobody has published a per-layer latency decomposition of the LLVM GPU dispatch stack — we measured it. Hot-path dispatch is 4.26 microseconds on GTX 1650. Module load dominates at 10.1 microseconds warm. Symbol lookup is 57 nanoseconds — negligible. Those numbers did not exist before. Third: there is no MLIR-native mechanism for cross-vendor runtime selection. `#gpu.select_object` is compile-time-only. IREE has it at the HAL level but that is a full runtime framework, not an MLIR attribute."

### Turn 2 — Pushing Harder

**Grad Student:** "But the measurement contribution is just instrumenting CUDA driver calls. And the metadata keys are just five strings in a table. Where is the intellectual depth?"

**Response:** "The depth is in what the measurements reveal about where optimization matters and where it does not. Before our numbers, a reasonable hypothesis was that OffloadBinary parsing and variant selection are significant overheads worth optimizing. Our data shows the opposite: module load dominates at 10 microseconds, kernel launch is 1.6 microseconds, and the selection logic itself — the dispatch table scan — is 2 nanoseconds. The interesting finding is that selection is irrelevant to hot-path performance. What matters is module *caching* — the one-time 46 microsecond cold load versus 0 nanosecond incremental cost on cached handles. That informs the design: `#gpu.runtime_select` caches the module pointer globally so `launchKernel()` is identical to `SelectObjectAttr` after the first call. The measurement drove the design decision, not the other way around."

### Turn 3 — The Comparison

**Grad Student:** "How does this compare to CPU Function Multi-Versioning? That seems like the direct analogue — `target_clones` already emits IFunc resolvers at compile time."

**Response:** "CPU FMV is the direct inspiration, and we cite it explicitly. Three key differences. First, the resolution mechanism: IFunc resolvers use CPUID, which is a nanosecond instruction. GPU vendor detection uses `dlopen`-probed library calls (`cuInit`, `hipInit`, `zeInit`), which are microsecond-scale operations — you cannot call them in a hot loop. That is why `#gpu.runtime_select` resolves once at module-load time via `global_ctors`, not per-call like IFunc. Second, the cost profile: IFunc resolution is nanoseconds; GPU module loading is 10-46 microseconds. The one-time cost matters for GPU in a way it never does for CPU. Third, the scope: CPU FMV selects among variants of the same ISA (x86 with different feature sets). GPU runtime selection is cross-ISA, cross-vendor — CUDA PTX versus HSACO versus SPIR-V. The dispatch table must handle fundamentally different binary formats, not just different feature levels of the same architecture."

---

## Persona 8: ML Framework Developer (PyTorch/JAX)

**Background:** Works on PyTorch or JAX's compiler backend. Has a custom dispatch mechanism. Skeptical about adding another layer of dispatch.

### Turn 1 — Opening

**ML Dev:** "PyTorch already has its own dispatch mechanism — torch.compile, Triton kernels, TorchInductor. Why would I care about another dispatch layer in MLIR?"

**Response:** "You would care if you want to ship a single binary that runs on both NVIDIA and AMD without maintaining separate codepaths. Today, vLLM — which I have contributed to — maintains separate CUDA and HIP kernel implementations. torch-mlir is working on compiling PyTorch models through MLIR's GPU dialect to produce `gpu.binary` with multiple targets. When torch-mlir produces a binary with an NVIDIA cubin and an AMD HSACO, someone has to pick which one to load. Today that is the framework's job — torch-mlir or ONNX-RT reimplements selection logic. Our proposal moves that selection into the LLVM layer so every framework gets it for free. You do not replace your Triton dispatch — you let the LLVM layer handle the 'which vendor's GPU is present' question that is currently framework-specific."

### Turn 2 — Performance Concern

**ML Dev:** "What is the overhead? If I am dispatching thousands of kernels per second in an LLM serving pipeline, even microseconds matter."

**Response:** "The steady-state overhead is zero. Literally zero. After the one-time vendor detection and module load, the cached module pointer is stored in a global. Every subsequent `launchKernel()` call loads from that global and calls `cuModuleGetFunction` + `cuLaunchKernel` — the exact same code path as `#gpu.select_object`, which is what you use today if you go through MLIR. Our benchmark data: hot-path dispatch is 4.26 microseconds total (launch submission 1.6 microseconds + GPU synchronization 2.5 microseconds on GTX 1650). The one-time cold cost is 46 microseconds for the full module load path. For an LLM serving pipeline dispatching a 10 millisecond attention kernel, that one-time 46 microsecond cost is 0.46% of a single kernel and amortized to zero across the thousands of subsequent calls. The dispatch table scan itself is 2 nanoseconds."

### Turn 3 — Integration Path

**ML Dev:** "Concretely, what would change in the torch-mlir pipeline to use this?"

**Response:** "One attribute change. Today, torch-mlir's GPU lowering attaches `#gpu.select_object<0>` to `gpu.binary`, which hardcodes 'use the first object.' To use runtime selection, you replace that with `#gpu.runtime_select<strategy = \"rank_by_priority\", fallback = \"cpu\">`. The rest of the pipeline — `gpu-module-to-binary`, target attachment, kernel launch lowering — is unchanged. The `--gpu-mark-runtime-select` pass can do this rewrite automatically when it detects multiple `#gpu.object` entries in a `gpu.binary`. No changes to the torch-mlir frontend, no changes to the Triton kernel compilation, no changes to the serving runtime. The attribute swap happens at the MLIR-to-LLVM-IR translation boundary."

---

## Persona 9: HPC Center Operator

**Background:** Runs a heterogeneous cluster with NVIDIA A100s, AMD MI250Xs, and possibly Intel GPUs. Maintains dozens of build configurations. Cares about operational simplicity.

### Turn 1 — Opening

**Operator:** "We maintain 80 build configurations for CMS experiment software across our GPU cluster — different architectures, different driver versions, different feature combinations. Can this actually reduce that?"

**Response:** "That is exactly the use case that motivates this work. We cite HEP-CCE — the CERN CMS experiment — which maintains roughly 80 build configurations to target heterogeneous GPU clusters with A100, V100, MI250X, and CPU fallback. Each configuration is a separate build because the compiler commits to a single GPU target at compile time. With a fat binary carrying multiple `#gpu.object` entries — one per target architecture — and runtime selection via `#gpu.runtime_select`, you build once with all targets and the runtime picks the right binary for each node. 80 configurations collapse to one fat build. The metadata keys ensure the right image is loaded: `min_sm=80` for A100, `min_gfx=gfx90a:cdna2` for MI250X, CPU fallback if neither GPU is present."

### Turn 2 — Operational Reality

**Operator:** "What about driver version mismatches? A cubin compiled for CUDA 12.x might not work on a node running CUDA 11.x drivers."

**Response:** "That is a real concern and the `requires_features` key is designed for it. You can encode `requires_features=cuda_12` to express a minimum driver capability. The extended `isMetadataCompatible()` checks that token against the device's reported capabilities and rejects the image if the requirement is not met. The runtime falls through to the next compatible image — which might be a more conservative cubin compiled for CUDA 11.x, or a CPU fallback. The fat binary carries all the variants; the metadata ensures the runtime only loads what the node can actually execute. This does not solve every driver compatibility issue — the capability token vocabulary needs to be defined carefully — but it converts a build-time problem (maintain N builds) into a metadata-curation problem (annotate each image correctly)."

### Turn 3 — Deployment Concern

**Operator:** "How much bigger are these fat binaries? Storage is not free on 10,000 nodes."

**Response:** "A fat binary with three targets is approximately three times the single-target size. Our test cubins: sm_75 is 4,328 bytes, sm_86 is 4,712 bytes, sm_89 is 4,712 bytes. A three-variant fat binary is about 14 KB. For real ML kernels the ratio holds — if a single cubin is 500 KB, a three-target fat binary is about 1.5 MB. The metadata overhead is negligible: five string keys per image add maybe 200 bytes. The tradeoff is 3x binary size versus 80x build configurations. For a cluster operator, that is a clear win — you store one binary that is 3x larger instead of maintaining 80 separate builds, each with its own CI pipeline, testing matrix, and deployment path. The one-time module load cost of the larger binary is 10 microseconds for a 4 KB cubin — it scales roughly linearly with binary size but is still sub-millisecond for any realistic kernel."

---

## Persona 10: First-Time Conference Attendee

**Background:** Masters student or early-career developer attending EuroLLVM for the first time. Interested but unfamiliar with MLIR, OffloadBinary, or the offload stack. Needs the 60-second version.

### Turn 1 — Opening

**Attendee:** "Hi, this looks interesting but I am new to MLIR. Can you explain what this poster is about in simple terms?"

**Response:** "Sure. When you write a program that runs on a GPU, the compiler turns your code into a binary that only works on one specific GPU — say, an NVIDIA card. If you also want it to run on an AMD card, you compile a second binary. Today, you have to pick which binary to use before you run the program. Our work lets you ship both binaries in a single file — a fat binary — and the program automatically picks the right one when it starts, based on which GPU is actually plugged in. Think of it like a universal charger that checks the plug shape and switches to the right voltage. MLIR is the compiler framework we use to build this — it is part of LLVM, which is the open-source compiler infrastructure behind Clang, Rust, Swift, and many others."

### Turn 2 — Follow-Up

**Attendee:** "So it is like a USB-C adapter for GPUs? How does it know which GPU is there?"

**Response:** "Good analogy. When the program starts, it does a quick check — it tries to load NVIDIA's GPU library, AMD's GPU library, and Intel's GPU library, one by one. Whichever one responds, that tells the program which vendor's GPU is installed. Then it looks at the fat binary, finds the matching GPU binary, and loads it. The whole detection takes about 46 microseconds — that is 0.046 milliseconds, essentially instant. After that first check, every subsequent GPU operation uses the cached result and runs at full speed with zero overhead. We measured this: the actual GPU kernel launch takes about 4 microseconds, and our selection logic adds 2 nanoseconds on top of that — a thousand times smaller than the launch itself."

### Turn 3 — Why It Matters

**Attendee:** "Why does this matter? Can't you just compile for both GPUs and pick the right binary yourself?"

**Response:** "You can, and that is what everyone does today — but it does not scale. CERN's CMS experiment maintains 80 different build configurations because they have clusters with different GPU models, different driver versions, and different features. Every time they add a new GPU model, they add more builds. Cloud providers like AWS and Google have similar problems — when you rent a GPU instance, you might not know which specific GPU model you will get until the program starts. Our approach reduces 80 builds to one. The compiler produces a single fat binary with all the variants, and the runtime picks the right one. That is less infrastructure to maintain, fewer things to break, and faster deployment. The poster shows the numbers proving the overhead is negligible, and proposes the standard format for how GPU binaries should describe their requirements."

---

## Quick Reference: Data Points for All Conversations

Use these numbers with confidence — all measured on GTX 1650 (sm_75), CUDA 13.1.

| Data Point | Value | Source |
|------------|-------|--------|
| Hot-path dispatch (launch + sync) | 4.26 us | layer-benchmark-results.md |
| cuLaunchKernel submit (CPU-side) | 1.57 us median | layer-benchmark-results.md |
| cuStreamSynchronize (GPU RTT) | 2.48 us median | layer-benchmark-results.md |
| cuModuleLoadData warm | 10.1 us median | layer-benchmark-results.md |
| cuModuleLoadData cold (exec-child) | 42.7 us median | layer-benchmark-results.md |
| cuModuleGetFunction (symbol lookup) | 60 ns median | layer-benchmark-results.md |
| cuDeviceGet (driver shim floor) | 50 ns median | layer-benchmark-results.md |
| Module load / launch overhead ratio | 6.0x | layer-benchmark-results.md |
| kdl_select cold | 46.2 us median | benchmark-results.md Run 3 |
| kdl_select cached | 44.9 us median | benchmark-results.md Run 3 |
| kdl_load_bundle | 4.9 us median | benchmark-results.md Run 3 |
| cuda_direct_launch | 841 ns median | benchmark-results.md Run 3 |
| Pure dispatch table scan | ~2 ns | runtime_select_poc |
| Pinned cuLaunchKernel (3-run aggregate) | 1,600 ns median | pinned-benchmark-results.md |
| Pinned cuStreamSync (3-run aggregate) | 2,465 ns median | pinned-benchmark-results.md |
| Test cubin sizes | 4.3-4.7 KB | real-offloadbinary-results.md |
| OffloadBinary standard keys (today) | 2 (triple, arch) | proposal-v2.md |
| Proposed new keys | 5 | proposal-v2.md |
| `isMetadataCompatible()` extension | ~40 LOC | proposal-v2.md |
| Header constants patch | ~20 LOC | proposal-v2.md |
| Full `RuntimeSelectAttr.cpp` estimate | ~780 LOC | proposal-v2.md |
| TaxBreak H100 null-kernel baseline | 4.71 us avg | arXiv:2603.12465 Table III |

---

## Conversation Flow Cheat Sheet

**If they ask about liboffload competition:** "Complement, not compete. Mechanism vs. policy. The layers compose."

**If they ask about IREE:** "Different layer. IREE is full-stack runtime. We are MLIR attribute + OffloadBinary metadata. Non-IREE pipelines need this."

**If they ask about novelty:** "Three firsts — metadata vocabulary, dispatch flame graph, MLIR-native cross-vendor selection. Concept is not new. Specific LLVM implementation is."

**If they ask about AMD/Intel testing:** "NVIDIA + CPU validated. AMD mocked. Intel designed but unmeasured. Honest about gaps."

**If they ask about overhead:** "Zero steady-state. 46 us one-time. 2 ns dispatch table scan. Dominated by GPU hardware, not our logic."

**If they push on 'just a dispatch table':** "The table is simple. The value is the standard metadata that feeds it, and the first numbers showing where time actually goes in the LLVM GPU stack."

**If they ask about prototype vs. upstream:** "Prototype uses custom MTB format, not OffloadBinary. Validates runtime mechanics. Upstream contribution translates these concepts to OffloadBinary + MLIR. The mapping is conceptual, not structural — we are honest about that."
