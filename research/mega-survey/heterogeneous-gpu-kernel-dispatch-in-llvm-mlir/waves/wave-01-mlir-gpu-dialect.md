# Wave 01: MLIR GPU Dialect Runtime Dispatch
Search query: MLIR gpu dialect runtime target selection dispatch mechanism
Sources found: 10
Date: 2026-04-06

## Sources

### 1. MLIR GPU Dialect Official Documentation
- URL: https://mlir.llvm.org/docs/Dialects/GPU/
- Type: docs
- Date: Continuously updated (last confirmed 2025-2026)
- Relevance: 10/10
- Novelty: 4/10
- Summary: The canonical reference for the `gpu` dialect. Defines the two-stage compilation model: (1) GPU module serialization via Target attributes and the `gpu-module-to-binary` pass, (2) offloading operations translation via Offloading attributes. The dialect targets CUDA/OpenCL-like programming models and abstracts away device/driver-specific kernel launch details.
- Key detail: The `gpu-module-to-binary` pass produces a `gpu.binary` containing a `gpu.object` for *every* target attribute attached to the module — this is the core multi-target binary mechanism. The binary stores objects in formats: offloading, assembly, binary, or fatbinary. Runtime dispatch happens at LLVM translation time when the Offloading attribute searches the binary and invokes the kernel launch procedure.

---

### 2. D154149: [mlir][gpu] Add the `gpu-module-to-binary` pass (Phabricator Review)
- URL: https://reviews.llvm.org/D154149
- Type: PR
- Date: 2023 (landed in LLVM 17-era)
- Relevance: 9/10
- Novelty: 8/10
- Summary: The foundational patch introducing the `gpu-module-to-binary` pass to MLIR. Defines how GPU modules with multiple target attributes get serialized into a single `gpu.binary` operation containing per-target objects. This is the architectural pivot point enabling multi-vendor binary packaging in MLIR.
- Key detail: The pass calls `serializeToObject()` on each Target attribute interface implementation. This extensible interface is the hook point for adding new vendor targets (NVVM, ROCDL, SPIR-V) without modifying core pass logic.

---

### 3. D154104: [mlir][gpu] Add GPU Target Attribute Interface (Phabricator Review)
- URL: https://reviews.llvm.org/D154104
- Type: PR
- Date: 2023 (companion to D154149)
- Relevance: 9/10
- Novelty: 8/10
- Summary: Defines the `GPUTargetAttrInterface` — the extensibility mechanism allowing any dialect to register itself as a GPU compilation target. Implementations must provide `serializeToObject()` to produce binary blobs. Pairs with D154149 to form the complete target-agnostic serialization framework.
- Key detail: The interface decouples target-specific compilation (PTX→CUBIN, ROCDL→HSACO, MLIR→SPIR-V) from the generic module-to-binary pass, enabling pluggable vendor backends at the IR level.

---

### 4. RFC: Extending MLIR GPU Device Codegen Pipeline (LLVM Discourse)
- URL: https://discourse.llvm.org/t/rfc-extending-mlir-gpu-device-codegen-pipeline/70199
- Type: RFC
- Date: April 2023 (multi-page discussion through 2024)
- Relevance: 9/10
- Novelty: 7/10
- Summary: Long-running RFC discussing limitations of the existing MLIR GPU pipeline — specifically inability to link against bytecode libraries (e.g., libdevice), lack of device-link support, and requirement to build MLIR on-target. Proposes generating clang-compatible offload LLVM IR and delegating final binary generation to clang, decoupling MLIR from the final link step.
- Key detail: The core issue identified: MLIR's GPU pipeline currently assumes static, compile-time target knowledge — there is no mechanism for runtime target selection. The RFC's proposed clang-delegation approach partially addresses this by pushing target resolution later in the toolchain.

---

### 5. PR #119440: [mlir][gpu] Adding ELF section option to gpu-module-to-binary pass (GitHub)
- URL: https://github.com/llvm/llvm-project/pull/119440
- Type: PR
- Date: December 2024
- Relevance: 7/10
- Novelty: 6/10
- Summary: Extends the `gpu-module-to-binary` pass with an ELF section option, allowing GPU binary objects to be embedded into specific ELF sections of the host binary. Enables more flexible deployment of GPU kernels where the binary loader can locate objects by section name rather than requiring runtime file I/O.
- Key detail: This is infrastructure for fat-binary embedding — a prerequisite for true runtime dispatch where a host binary contains objects for multiple targets and selects at load time based on detected hardware.

---

### 6. D149559: [mlir][gpu] GPU Serialization Pipeline for Clang Offloading Annotations (Phabricator)
- URL: https://reviews.llvm.org/D149559?id=518365
- Type: PR
- Date: 2023
- Relevance: 8/10
- Novelty: 7/10
- Summary: Adds a GPU serialization pipeline that annotates GPU dialect ops with clang-compatible offloading metadata. Allows MLIR-generated GPU code to participate in clang's heterogeneous compilation flow (which includes fat-binary creation and runtime target selection via the CUDA/HIP driver's device selection APIs).
- Key detail: By bridging to clang's offloading model, this patch implicitly gains access to CUDA's cubin selection mechanism — where the driver selects the correct cubin from a fat-binary at runtime based on detected GPU architecture. This is the closest MLIR currently gets to *runtime* target selection.

---

### 7. RFC: An MLIR Dialect for Distributed Heterogeneous Computing (LLVM Discourse / PLDI 2025 SRC)
- URL: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- Type: RFC
- Date: June 2025 (PLDI 2025 Student Research Competition entry)
- Relevance: 8/10
- Novelty: 9/10
- Summary: Proposes a new MLIR dialect for distributed heterogeneous systems with a `schedule` operation that groups `task` operations each annotated with a target (e.g., `cpu`, `gpu`). Supports both fine- and coarse-grained parallelism, enables explicit orchestration and static analysis, and lowers to existing MLIR dialects (LLVM IR, MPI dialect). From IIT Madras PACE Lab.
- Key detail: Introduces *target annotation at the task operation level* — a higher abstraction than the gpu dialect's module-level target attributes. This is the direction toward first-class heterogeneous dispatch in MLIR IR, not just in compilation pipelines.

---

### 8. Stephen Diehl: GPU Compilation with MLIR (Blog / Tutorial)
- URL: https://www.stephendiehl.com/posts/mlir_gpu/
- Type: blog
- Date: 2023-2024
- Relevance: 7/10
- Novelty: 3/10
- Summary: Detailed walkthrough of the MLIR GPU compilation pipeline — from high-level tensor ops through bufferization, loop mapping, GPU dialect, NVVM/ROCDL lowering, to PTX/HSACO emission. Covers the gpu-lower-to-nvvm pipeline and the progressive lowering strategy. Good technical reference for understanding the pipeline stages.
- Key detail: Confirms the pipeline is *statically* target-determined at compile time: you pick NVVM or ROCDL paths at pipeline construction time, not at runtime. There is no runtime fallback or dynamic vendor selection in the baseline MLIR GPU pipeline.

---

### 9. AMD MLIR-to-SPIR-V Generic Pass (Phoronix, LLVM 19)
- URL: https://www.phoronix.com/news/LLVM-19-MLIR-To-SPIR-V
- Type: blog/news
- Date: 2024 (LLVM 19 release cycle)
- Relevance: 7/10
- Novelty: 6/10
- Summary: An AMD AI Compiler engineer contributed a generic MLIR-to-SPIR-V lowering pass in LLVM 19, intended to improve coverage of upstream SPIR-V compilation and enable writing simple kernels targeting Vulkan/OpenCL/OpenGL without vendor-specific IR. Addresses a gap where ROCm/HIP kernels could not easily target the SPIR-V path.
- Key detail: The pass being "generic" (not AMD-specific) is architecturally significant — it means MLIR now has a vendor-neutral SPIR-V lowering path that any target can use, reducing the N×M backend problem. This is infrastructure for portable kernel dispatch.

---

### 10. MLIR omp.target for GPU Offloading (LLVM Discourse Thread)
- URL: https://discourse.llvm.org/t/mlir-omp-target-for-gpu-offloading/72579
- Type: RFC/discussion
- Date: 2023-2024
- Relevance: 7/10
- Novelty: 5/10
- Summary: Discussion of using the MLIR OpenMP dialect's `omp.target` construct for GPU offloading, as an alternative to the native GPU dialect path. The OpenMP model provides source-level target-agnostic specification (the `target` directive selects the device), which is then lowered through either GPU dialect ops or direct LLVM offloading IR. Flang uses this path.
- Key detail: `omp.target` provides *logical* heterogeneous dispatch (target is specified at the source level) but physical dispatch still resolves statically during compilation. The OpenMP runtime does support device enumeration and selection at runtime (via `omp_get_num_devices()`), giving this path a runtime dispatch capability the pure GPU dialect lacks.

---

## Angle Assessment

**Coverage:** This angle is well-explored in terms of the static compilation pipeline (passes, target attributes, serialization). The gap — runtime target *selection* after the binary is built — is acknowledged in RFCs but not yet architecturally solved in mainline MLIR.

**Surprise findings:**
- The `gpu-module-to-binary` pass already supports multi-target binary objects (one `gpu.object` per target) inside a single `gpu.binary` — this is exactly the fat-binary model needed for runtime dispatch. The missing piece is a runtime lookup mechanism that selects the correct object at kernel launch time based on detected hardware.
- The PLDI 2025 dialect RFC (IIT Madras) shows the community *is* thinking about task-level target annotation, which is a precursor to runtime dispatch.
- `omp.target` via OpenMP runtime already has `omp_get_num_devices()` — a weak form of runtime target selection that MLIR GPU dialect does not expose natively.

**Gaps:**
- No MLIR-native mechanism for *runtime* GPU vendor/architecture selection — the Offloading attribute model is still compile-time-bound.
- No standardized "GPU capability query" op in the GPU dialect (detect CUDA vs ROCm vs SPIR-V at runtime).
- The fat-binary infrastructure (ELF sections, gpu.object per target) exists but the runtime dispatch table linking binary selection to device detection is absent.
- MLIR GPU dialect has no equivalent of CUDA's `cuFuncGetAttribute` or HIP's `hipModuleGetFunction` for dynamic kernel variant selection.

**Suggested follow-up angles:**
1. **IREE's GPU dispatch model** — IREE implements its own runtime target selection on top of MLIR, worth examining as a production example of what MLIR itself lacks.
2. **Fat-binary format internals** — How CUDA/HIP fatbinaries encode multi-arch objects and how the driver selects at load time; compare to MLIR's `gpu.binary`/`gpu.object` model.
3. **OpenXLA/PJRT device selection** — How JAX's PJRT plugin model handles multi-vendor dispatch at the framework level above MLIR.
4. **GPU dialect capability query ops** — Whether any RFC exists (or should exist) for `gpu.get_device_target` or similar runtime introspection ops.
5. **Triton's runtime dispatch** — Triton generates multiple kernel variants (for different tile sizes / architectures) and selects at Python launch time; compare to MLIR's compile-time model.
