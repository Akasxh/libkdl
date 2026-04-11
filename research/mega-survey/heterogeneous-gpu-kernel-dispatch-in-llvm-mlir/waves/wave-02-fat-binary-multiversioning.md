# Wave 02: Fat Binary Multi-Versioning
Search query: "fat binary multi-versioned kernel GPU CUDA fatbin clang-offload-bundler"
Sources found: 9
Date: 2026-04-06

## Sources

### 1. CUDA C++ Programming Guide — NVCC Compilation and Fatbin
- URL: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html
- Type: docs
- Date: 2026 (CUDA 13.2)
- Relevance: 10/10
- Novelty: 5/10
- Summary: Authoritative NVIDIA reference on how nvcc compiles CUDA programs into fatbin containers. Describes how a single fatbin embeds multiple PTX (virtual ISA) and cubin (hardware ISA) targets, with the CUDA runtime selecting the best match for the executing GPU's compute capability. JIT compilation from PTX serves as the fallback when no precompiled cubin matches.
- Key detail: The two-tier system — cubin for performance (binary-compatible only within major.minor range) and PTX for forward compatibility (JIT-compilable across major versions) — is the canonical fat binary runtime selection model. Compute capability `sm_XY` cubin runs on CC X.Y or higher within the same major; `compute_XY` PTX JIT-compiles for any CC >= X.Y across major boundaries.

---

### 2. Understanding PTX — NVIDIA Technical Blog
- URL: https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/
- Type: blog
- Date: 2025
- Relevance: 9/10
- Novelty: 6/10
- Summary: Explains PTX as a virtual machine ISA that decouples high-level CUDA code from GPU architecture specifics. A 2018 Turing-targeted binary runs on 2025 Blackwell via JIT because PTX is architecture-agnostic. A fatbin embedding `compute_70` PTX alongside `sm_70`/`sm_80`/`sm_86` cubins provides runtime-optimal selection: exact cubin match wins, PTX JIT is the safety net.
- Key detail: PTX is the forward-compatibility mechanism in CUDA fat binaries. The driver JIT-compiles PTX for architectures that did not exist at compile time — this is not a workaround, it is a first-class design goal.

---

### 3. Clang Offload Bundler — Clang Documentation (23.0)
- URL: https://clang.llvm.org/docs/ClangOffloadBundler.html
- Type: docs
- Date: 2026 (Clang 23 trunk)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Describes the clang-offload-bundler tool that combines host and multiple device code objects (one per GPU target) into a single bundled file for CUDA, HIP, and OpenMP offloading. Bundle entry IDs encode the offload kind, target triple, and optional target-ID (processor + feature flags). The bundler is the older model; the newer clang-offload-packager replaces it in Clang's new offload pipeline.
- Key detail: Bundle entry ID format: `<offload-kind>-<target-triple>[-<target-id>]`. AMD GPUs support rich target IDs (e.g., `gfx908:xnack+:sramecc-`); NVIDIA NVPTX targets use only the target triple with no target ID, relying instead on fatbin-level cubin/PTX layering for versioning.

---

### 4. Offloading Design & Internals — Clang Documentation (23.0)
- URL: https://clang.llvm.org/docs/OffloadingDesign.html
- Type: docs
- Date: 2026 (Clang 23 trunk)
- Relevance: 10/10
- Novelty: 7/10
- Summary: Describes Clang's new offloading pipeline: device code is embedded into host objects in the `.llvm.offloading` section using a magic-byte-prefixed binary format (`0x10FF10AD`). The clang-linker-wrapper scans input objects, extracts embedded device binaries, links them per target, and emits runtime registration code. This replaces the older clang-offload-bundler-centric model.
- Key detail: The `__tgt_bin_desc` structure holds an array of `__tgt_device_image` pointers, registered at startup via `__tgt_register_lib()`. Each device image corresponds to one target architecture. Runtime target selection happens inside the OpenMP/CUDA/HIP runtime, not in Clang tooling — the toolchain's job ends at packaging all images into the host binary.

---

### 5. D125165: Introduce clang-offload-packager — LLVM Phabricator
- URL: https://reviews.llvm.org/D125165
- Type: PR
- Date: 2022 (landed in LLVM 15)
- Relevance: 9/10
- Novelty: 8/10
- Summary: Introduces clang-offload-packager as a replacement/complement to clang-offload-bundler for the new LLVM offload model. Unlike the bundler, the packager creates a standalone binary compatible with standard LLVM tooling (can be piped into `opt`), then embeds it via `-fembed-offload-object`. Mandatory metadata keys are `triple` and `arch`; optional `kind` field selects offloading framework.
- Key detail: The packager's binary format uses magic bytes `0x10FF10AD` to allow the linker to locate all embedded offloading sections even after they are merged during relocatable linking. This is the mechanism that makes multi-target fat binary embedding robust in the new model — each device image is independently discoverable in the final ELF.

---

### 6. Clang Offload Packager — Clang 20.1.0 Documentation
- URL: https://releases.llvm.org/20.1.0/tools/clang/docs/ClangOffloadPackager.html
- Type: docs
- Date: 2025
- Relevance: 8/10
- Novelty: 5/10
- Summary: Official stable documentation for the clang-offload-packager tool as shipped in LLVM 20. Accepts multiple `--image` arguments (file, triple, arch, kind) to bundle heterogeneous device binaries into a single output. The packager supersedes the bundler for new-model LLVM offloading and is the tool Clang now invokes internally during fat binary construction.
- Key detail: The packager embeds all images into a string-map-based binary format tagged with per-image metadata, decoupling device packaging from host compilation entirely. This enables cleaner multi-arch compilation pipelines where different GPU targets can be compiled in parallel and packaged post-hoc.

---

### 7. User Guide for AMDGPU Backend — LLVM Documentation (23.0)
- URL: https://llvm.org/docs/AMDGPUUsage.html
- Type: docs
- Date: 2026 (trunk)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Defines AMD code object versions V2–V6. V3+ uses ELF with MsgPack-encoded metadata; the processor is encoded in the `EF_AMDGPU_MACH` field of `e_flags`. The HSA runtime matches a code object to a device by comparing the target ID (processor + XNACK/SRAMECC feature flags) embedded in ELF note records against the executing GPU's capabilities. Multiple code objects for different GPU targets can be bundled via clang-offload-bundler.
- Key detail: Code object V4 and V5 differ in XNACK default semantics — V4+ allows either `xnack+` or `xnack-` without assuming a default, requiring explicit matching at runtime. This feature-flag-level versioning is AMD's equivalent of CUDA's compute capability versioning, and is finer-grained than NVIDIA's approach because it gates functionality (demand paging) rather than just ISA compatibility.

---

### 8. HetGPU: The Pursuit of Making Binary Compatibility Towards GPUs
- URL: https://arxiv.org/html/2506.15993v1
- Type: paper
- Date: 2025
- Relevance: 9/10
- Novelty: 10/10
- Summary: Proposes hetIR, an architecture-agnostic GPU virtual ISA, and a compiler+runtime system that JIT-translates it to NVIDIA PTX, AMD SPIR-V/GCN, Intel Xe, or Tenstorrent Metalium at load time. Unlike traditional fat binaries, there is no per-vendor binary stored at compile time — a single hetIR binary dispatches to any GPU via runtime translation. Achieves <10% overhead for compute-bound kernels; also enables live GPU kernel migration.
- Key detail: HetGPU's runtime dispatch is: `hetIR → {PTX→SASS | SPIR-V→GCN | SPIR-V→Xe | Metalium ASM}` with per-platform JIT caches. First execution costs 50–200ms; subsequent launches use cached binaries. This is the most ambitious "single fat binary" approach found — it extends across vendor boundaries that traditional fatbins cannot cross.

---

### 9. ROCm Code Object and Compiler Reference — ROCmCC
- URL: https://rocm.docs.amd.com/en/docs-6.0.2/reference/rocmcc.html
- Type: docs
- Date: 2024 (ROCm 6.0)
- Relevance: 8/10
- Novelty: 5/10
- Summary: Documents how amdclang++ generates fat binaries containing multiple AMDGPU code objects targeting different ISAs (gfx906, gfx908, gfx90a, gfx1100, etc.) in a single .hsaco or host-embedded binary. The ROCm runtime selects the matching image at kernel launch time based on device ISA. Static libraries embed code objects as fat binaries using `--emit-static-lib`.
- Key detail: ROCm fat binaries can contain images differing along three axes simultaneously: (1) different GPU architectures (gfx906 vs gfx1100), (2) same architecture with different feature flags (gfx908:xnack+ vs gfx908:xnack-), (3) same architecture and features but different optimizations. This three-dimensional versioning space has no direct CUDA equivalent — CUDA only versions along the compute capability axis.

---

## Angle Assessment

- Coverage: Well-explored for CUDA (NVIDIA docs are thorough) and reasonably covered for AMD (ROCm docs + LLVM AMDGPU backend docs). The LLVM toolchain side (bundler vs packager transition) is well-documented in Clang docs and Phabricator. HetGPU is an outlier that extends the concept beyond the traditional fat binary model.

- Surprise findings: HetGPU (arxiv 2506.15993, 2025) is a significant surprise — it proposes replacing fat binaries entirely with a virtual GPU ISA (hetIR) and runtime JIT translation. This is conceptually similar to what libkdl aims for but goes further by enabling live kernel migration across GPU vendors. The `0x10FF10AD` magic-byte format in clang-offload-packager is an under-documented but critical detail: it makes every embedded device image independently discoverable in a linked ELF, which is the actual mechanism enabling multi-target fat binary robustness in modern LLVM.

- Gaps: (1) No found documentation on how CUDA's fatbin selection algorithm is implemented inside `libcuda.so` / the driver — only the behavior is documented, not the implementation. (2) No analysis of the performance cost of the bundler's target matching at load time for large fat binaries with many embedded targets. (3) Intel oneAPI/Level Zero fat binary format is not covered — the clang-offload-bundler docs note Intel targets but do not detail the target ID scheme. (4) SPIR-V as a fat binary substrate (e.g., for OpenCL/Vulkan multi-GPU targeting) is absent from these sources.

- Suggested follow-up angles:
  - `intel-oneapi-sycl-fat-binary`: How Intel's JIT compilation from SPIR-V at runtime maps to the fat binary concept; comparison with NVIDIA PTX JIT
  - `libcuda-fatbin-selection-internals`: Reverse-engineered or documented internals of how the CUDA driver selects cubin vs PTX from a fatbin (cuobjdump, binary inspection)
  - `amdgpu-xnack-runtime-matching`: Deep-dive into AMD HSA runtime code object selection with feature flags — implications for dispatch in heterogeneous XNACK-capable/incapable cluster environments
  - `hetir-virtual-gpu-isa`: Follow-up on HetGPU paper — IR design decisions, comparison with SPIR-V as a common substrate, performance on real ML workloads
  - `fatbin-size-overhead`: Fat binary size explosion with many GPU targets — profiling and mitigation strategies (PTX-only builds, lazy JIT, compressed fatbins)
