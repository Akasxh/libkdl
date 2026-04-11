# Wave 01: MLIR GPU Dialect Dispatch
Search query: "MLIR gpu dialect kernel launch dispatch heterogeneous target selection"
Sources found: 10
Date: 2026-04-06

## Sources

### 1. MLIR GPU Dialect Official Documentation
- URL: https://mlir.llvm.org/docs/Dialects/GPU/
- Type: docs
- Date: Continuously updated (current as of 2025-2026)
- Relevance: 10/10
- Novelty: 4/10
- Summary: Canonical reference for the `gpu` dialect. Defines the two-stage compilation model: (1) GPU module serialization via Target attributes + `gpu-module-to-binary` pass, (2) offloading operations translation via Offloading attributes at LLVM translation time. The `gpu.launch_func` op is the central dispatch primitive — it references a `gpu.binary` and a kernel function within it. By default, `#gpu.select_object` embeds a single object as a global string; when a module has multiple target attributes, the binary contains one `gpu.object` per target.
- Key detail: There is no runtime target selection mechanism in the GPU dialect itself. The `#gpu.select_object` attribute picks one object at *compile time* (by index). True heterogeneous runtime dispatch — selecting NVVM vs ROCDL vs XeVM based on detected hardware at kernel launch — is absent from mainline MLIR. The offloading attribute invokes the kernel launch procedure statically resolved during LLVM translation.

---

### 2. D154149: [mlir][gpu] Add the `gpu-module-to-binary` pass (Phabricator)
- URL: https://reviews.llvm.org/D154149
- Type: PR
- Date: 2023 (landed in LLVM 17 era)
- Relevance: 9/10
- Novelty: 8/10
- Summary: Foundational patch introducing the `gpu-module-to-binary` pass, which scans nested GPU modules and calls `serializeToObject()` on each Target attribute. Produces a `gpu.binary` with one `gpu.object` per attached target. This is the multi-target binary packaging mechanism in MLIR — the direct analogue to a CUDA fatbinary.
- Key detail: The extensible `GPUTargetAttrInterface` with `serializeToObject()` is the hook point for all vendor backends (NVVM, ROCDL, SPIR-V, XeVM). Adding a new target requires only implementing this interface — the generic pass logic is untouched. However, the downstream dispatch step (picking the right object at runtime) has no corresponding interface; it is left entirely to the Offloading attribute's LLVM translation.

---

### 3. PR #119440: [mlir][gpu] Adding ELF section option to gpu-module-to-binary pass (GitHub)
- URL: https://github.com/llvm/llvm-project/pull/119440
- Type: PR
- Date: December 2024
- Relevance: 7/10
- Novelty: 6/10
- Summary: Extends the `gpu-module-to-binary` pass with an ELF section option, allowing GPU binary objects to be embedded into named ELF sections of the host binary. Enables fat-binary-style deployment where the host ELF contains GPU code objects for multiple targets, discoverable at load time by section name.
- Key detail: This is critical infrastructure for a future runtime dispatch layer: if GPU objects live in ELF sections with target-annotated names, a runtime linker (e.g., libkdl) could iterate ELF sections, query device capabilities, and select the correct kernel object without the MLIR layer knowing the final target. The patch closes the last gap between MLIR binary packaging and ELF-based deployment models.

---

### 4. PR #66220: [mlir][gpu][NVPTX] Enable NVIDIA GPU JIT compilation path (GitHub)
- URL: https://github.com/llvm/llvm-project/pull/66220
- Type: PR
- Date: 2023 (by fabianmcg)
- Relevance: 8/10
- Novelty: 7/10
- Summary: Enables a JIT compilation path for NVIDIA GPUs within the MLIR GPU pipeline, allowing PTX to be compiled to CUBIN at runtime by the CUDA driver rather than requiring pre-built cubins. This shifts part of target specialization from AOT compile time to runtime, giving the CUDA driver the opportunity to select the optimal compilation for the detected GPU architecture.
- Key detail: JIT via CUDA driver is the *only* current mechanism in MLIR where target selection happens at runtime — not through MLIR IR but by delegating to the driver's own PTX compiler. This is not heterogeneous vendor selection (you still need to know it's CUDA at compile time), but it is runtime architecture selection within the NVIDIA ecosystem. An analogous mechanism for ROCm does not appear to exist in upstream MLIR.

---

### 5. RFC: Cleaning the GPU Dialect (LLVM Discourse)
- URL: https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- Type: RFC
- Date: September 2025 (ongoing multi-page discussion)
- Relevance: 8/10
- Novelty: 9/10
- Summary: Fabian Mora's RFC to restructure the GPU dialect by removing operations that do not semantically belong there and moving them to appropriate dialects (or creating new ones). The proposal identifies that the GPU dialect has accumulated disparate ops — some device-specific, some host-side orchestration, some compilation-pipeline artifacts — weakening its semantic contract.
- Key detail: The restructuring is directly relevant to heterogeneous dispatch: if dialect semantics are tightened, it becomes clearer which ops represent abstract kernel dispatch (vendor-neutral) vs. which encode vendor-specific lowering details. A cleaner boundary would enable a true vendor-neutral `gpu.dispatch` abstraction sitting above NVVM/ROCDL/XeVM backends. This RFC is the architectural prerequisite for a heterogeneous dispatch layer in MLIR.

---

### 6. RFC: An MLIR Dialect for Distributed Heterogeneous Computing (LLVM Discourse / PLDI 2025 SRC)
- URL: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- Type: RFC
- Date: June 2025 (PLDI 2025 Student Research Competition)
- Relevance: 8/10
- Novelty: 9/10
- Summary: Proposes a new MLIR dialect from IIT Madras PACE Lab that introduces a `schedule` op grouping `task` ops, each annotated with a target (e.g., `cpu`, `gpu`). Supports fine- and coarse-grained parallelism, explicit orchestration and static analysis, and lowers to LLVM IR or MPI dialect. Submitted at PLDI 2025.
- Key detail: This is the most explicit proposal in the MLIR ecosystem for *task-level target annotation* — specifying which hardware runs which computation at the IR level, above the GPU dialect. The `schedule`/`task` model is conceptually the same as what libkdl aims to provide at the runtime linker level. A combined approach (MLIR IR encodes dispatch intent; libkdl resolves binary selection at runtime) would be complementary.

---

### 7. Intel XeVM Dialect Upstreamed to LLVM (Phoronix / LLVM Discourse RFC)
- URL: https://www.phoronix.com/news/Intel-XeVM-MLIR-In-LLVM
- Type: news/PR
- Date: August 2025
- Relevance: 8/10
- Novelty: 9/10
- Summary: Intel upstreamed the XeVM dialect into mainline LLVM on 19 August 2025. XeVM extends the LLVM dialect for Intel Xe GPU hardware, providing a `#xevm.target` attribute, 2D block loads/stores, prefetch, and MMA operations. Includes SYCL runtime integration tests and serialization support, making it a first-class GPU target in the MLIR multi-target compilation model.
- Key detail: With XeVM landing, MLIR now has three production GPU target attribute implementations: `#nvvm.target` (NVIDIA), `#rocdl.target` (AMD), and `#xevm.target` (Intel). A single `gpu.module` can carry all three, and `gpu-module-to-binary` will produce a `gpu.binary` with three `gpu.object` entries. The *dispatch* problem — picking the right one at runtime based on detected vendor — becomes an urgent engineering question now that tri-vendor support is available at the IR level.

---

### 8. How to Generate AMDGPU Code from MLIR — Missing Pipeline Parity (LLVM Discourse)
- URL: https://discourse.llvm.org/t/how-to-generate-amdgpu-code-from-mlir-is-there-a-pipeline-similar-to-gpu-lower-to-nvvm-pipeline/88627
- Type: RFC/discussion
- Date: 2025
- Relevance: 7/10
- Novelty: 7/10
- Summary: Community thread asking whether an AMD equivalent to `gpu-lower-to-nvvm-pipeline` exists for generating AMDGPU/HSACO binaries from MLIR. Reveals a significant parity gap: while ROCDL dialect exists and `#rocdl.target` is supported in `gpu-module-to-binary`, there is no single documented end-to-end pipeline for AMD comparable to NVIDIA's. Users are directed to use IREE or vendor-specific stacks.
- Key detail: The absence of a `gpu-lower-to-rocdl-pipeline` (despite ROCDL dialect completeness) means AMD targets in upstream MLIR require more manual pipeline construction. This asymmetry makes heterogeneous NVIDIA+AMD dispatch harder in practice than the IR model implies — the MLIR tooling is NVIDIA-first, AMD-second. This is a concrete gap that libkdl or a similar runtime layer needs to bridge.

---

### 9. D149559: GPU Serialization Pipeline for Clang Offloading Annotations (Phabricator)
- URL: https://reviews.llvm.org/D149559?id=518365
- Type: PR
- Date: 2023
- Relevance: 7/10
- Novelty: 6/10
- Summary: Adds a GPU serialization pipeline that annotates GPU dialect ops with clang-compatible offloading metadata, allowing MLIR-generated GPU code to participate in clang's heterogeneous compilation flow, including fat-binary creation and CUDA/HIP driver runtime target selection.
- Key detail: By delegating final binary selection to the CUDA/HIP driver through clang's offloading model, this patch implicitly inherits the driver's runtime architecture selection (driver reads the fat-binary and picks the best cubin for detected hardware). This is the pragmatic path to runtime dispatch today — not MLIR-native, but driver-delegated. A libkdl-style approach would replicate this selection logic in a vendor-neutral form.

---

### 10. Stephen Diehl: GPU Compilation with MLIR (Blog / Tutorial)
- URL: https://www.stephendiehl.com/posts/mlir_gpu/
- Type: blog
- Date: 2023-2024
- Relevance: 6/10
- Novelty: 3/10
- Summary: Detailed walkthrough of the MLIR GPU compilation pipeline from high-level tensor ops through bufferization, loop mapping, GPU dialect, NVVM/ROCDL lowering, to PTX/HSACO emission. Good pedagogical reference for the complete static pipeline.
- Key detail: Explicitly confirms the pipeline is target-determined at compile time — you choose NVVM or ROCDL at pipeline construction, never at runtime. This is the clearest statement of the gap that heterogeneous dispatch tools like libkdl must fill.

---

## Angle Assessment

**Coverage:** The MLIR GPU dialect dispatch angle is well-explored for the *static* pipeline (target attributes, serialization, gpu-module-to-binary, LLVM translation). The critical *dynamic* dispatch layer (runtime vendor/architecture selection) is absent from mainline MLIR and only partially addressed through driver delegation (clang offloading, CUDA JIT PTX). Coverage of this gap across sources is consistent and confirming.

**Surprise findings:**
- Intel XeVM landed in LLVM mainline in August 2025, making MLIR a genuine tri-vendor GPU target platform (NVVM + ROCDL + XeVM). This is more recent than most secondary sources acknowledge and raises the urgency of the runtime dispatch problem — having three `gpu.object` entries in a `gpu.binary` is already possible, but no MLIR-native mechanism exists to select among them at runtime.
- The "RFC: Cleaning the GPU dialect" (September 2025) from Fabian Mora is ongoing and could architecturally enable a cleaner heterogeneous dispatch abstraction if it results in a tighter semantic core for the gpu dialect.
- The AMDGPU pipeline parity gap is severe in practice: despite `#rocdl.target` existing, there is no `gpu-lower-to-rocdl-pipeline` equivalent — users must use IREE or construct pipelines manually.
- The `#gpu.select_object` attribute selects by *index at compile time*, not by device capability at runtime. This means even a tri-vendor `gpu.binary` requires external (non-MLIR) logic to dispatch correctly at runtime.

**Gaps:**
- No MLIR-native op for runtime GPU capability query (detecting CUDA vs ROCm vs Level Zero / Xe at runtime within MLIR IR).
- No standardized runtime dispatch interface: the `GPUTargetAttrInterface` covers compile-time serialization but has no counterpart `GPURuntimeDispatchInterface` for selecting among pre-compiled objects at kernel launch time.
- AMD pipeline parity: `gpu-lower-to-rocdl-pipeline` does not exist in upstream; users must fall back to IREE or manual pass construction.
- No MLIR-level equivalent of cuFuncGetAttribute / hipModuleGetFunction for dynamic kernel variant selection (multi-versioned kernels by architecture within one binary).
- The ELF section embedding (PR #119440) provides the storage mechanism but no corresponding runtime lookup standard.

**Suggested follow-up angles:**
1. **GPUTargetAttrInterface extensibility in practice** — How hard is it to register a new vendor target (e.g., a hypothetical libkdl soft-target)? What does implementing `serializeToObject()` require?
2. **XeVM target attribute dispatch** — How does `#xevm.target` interact with the SYCL runtime for kernel launch? Is there a Level Zero dispatch path?
3. **IREE's HAL as the reference runtime dispatch implementation** — IREE has solved exactly this problem above MLIR; map its device selection logic to what MLIR's GPU dialect lacks.
4. **RFC: GPU dialect capability query ops** — Does any RFC exist (or should one be filed) for `gpu.get_device_vendor` / `gpu.get_device_arch` runtime introspection ops that would feed a dispatch decision?
5. **Fat-binary ELF section + runtime loader** — The combination of PR #119440 (ELF sections) + a runtime capable of iterating sections and matching device capability is precisely the libkdl architecture. Map this to a concrete MLIR integration path.
