# Wave 05: LLVM Discourse Heterogeneous Offloading RFCs
Search query: site:discourse.llvm.org heterogeneous offloading multi-target GPU runtime RFC
Sources found: 14
Date: 2026-04-06

## Sources

### 1. [RFC] Introducing `llvm-project/offload` — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302
- Type: RFC
- Date: October 2023
- Relevance: 10/10
- Novelty: 9/10 (to this wave; documented in wave-02 but critical anchor)
- Summary: Foundational RFC by Johannes Doerfert (LLNL) proposing the libomptarget-to-liboffload migration. The explicit framing: every hardware vendor builds their own LLVM offloading runtime downstream — this RFC proposes to upstream that work into a shared foundation serving CUDA, HIP, SYCL, OpenMP, AI accelerators, FPGAs, and remote targets. The community response was "basically unanimous support" from AMD, NVIDIA, Intel, and LLNL. The RFC draws the distinction between OpenMP-semantic dispatch (libomptarget, requires registered target regions) and raw binary dispatch (liboffload, takes arbitrary blobs by name). This two-layer split maps directly onto a KDL architecture.
- Key detail: The RFC's stated problem is identical to libkdl's motivation verbatim: "right now each vendor is basically creating their own LLVM offloading run-time among a lot of other duplicated — and often downstream only — code." libkdl's poster should cite this as community validation of the problem statement.

### 2. [RFC] `llvm-project/offload` Roadmap — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-llvm-project-offload-roadmap/75611
- Type: RFC
- Date: November/December 2023
- Relevance: 9/10
- Novelty: 8/10
- Summary: Follow-up RFC detailing the concrete roadmap. Proposes a new stable C API (`liboffload`, `ol`-prefixed) with OpenMP runtime preserved as a functional wrapper. Roadmap phases: (1) move libomptarget to `offload/`, (2) introduce offload-tblgen for TableGen-based API generation, (3) implement a minimal device/program/kernel/memory API, (4) enable SYCL and Unified Runtime interop via the same plugin layer. Post #15 (jdoerfert) explicitly acknowledges that the new API is intentionally lower-level than any language model — it is a mechanism layer, not a policy layer.
- Key detail: The roadmap explicitly does not include a multi-version kernel selection policy. The API gives raw binary-blob-to-kernel-name dispatch only. A KDL-style policy layer (select kernel X for architecture Y from bundle Z) is out of scope for liboffload and is therefore a genuine gap. This is the strongest structural argument for libkdl as a complementary layer.

### 3. [Offload] New Subproject + Pending Move of libomptarget — LLVM Discourse
- URL: https://discourse.llvm.org/t/offload-new-subproject-pending-move-of-libomptarget/78185
- Type: RFC/announcement
- Date: April 2024
- Relevance: 8/10
- Novelty: 7/10
- Summary: Formal announcement that `offload/` was created as a new top-level LLVM sub-project directory, and that migration of libomptarget source from `openmp/libomptarget/` to `offload/` was imminent. Directory structure confirmed: `offload/libomptarget/`, `offload/liboffload/`, `offload/plugins-nextgen/`, `offload/DeviceRTL/`. Migration completed mid-2024. Any tool built on liboffload must target `offload/` paths.
- Key detail: Confirms the migration is real, complete, and the infrastructure is now settled enough for external consumers to build on.

### 4. Showcasing LLVM/Offload: CUDA Kernel on AMD GPU — LLVM Discourse
- URL: https://discourse.llvm.org/t/showcasing-llvm-offload/75722
- Type: demonstration thread
- Date: December 2023
- Relevance: 9/10
- Novelty: 9/10
- Summary: Community demonstration (December 2023) showing that the new liboffload plugin architecture allows a CUDA kernel to execute on an AMD GPU — with automatic fallback to the host CPU if no suitable GPU is found. The key insight is that the NextGen plugin infrastructure (`GenericPluginTy` + vendor subclasses) is target-agnostic at the API level: the binary format (ELF + `.llvm.offloading` section) is passed through to whichever plugin can handle it. The runtime selects the plugin based on device availability, not binary content.
- Key detail: This is the closest existing LLVM-native behavior to what libkdl proposes — but it is opaque, policy-free, and first-valid (not best-match). The demonstration shows that the plumbing for heterogeneous execution exists; it does not implement any capability-aware selection. libkdl's contribution is adding the selection policy on top of this mechanism.

### 5. [RFC] Use the New Offloading Driver for CUDA and HIP by Default — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-use-the-new-offloding-driver-for-cuda-and-hip-compilation-by-default/77468
- Type: RFC
- Date: March 2024
- Relevance: 8/10
- Novelty: 8/10
- Summary: RFC proposing that `--offload-new-driver` be enabled by default for all current offloading languages (CUDA, HIP, OpenMP). The new unified driver creates fat objects embedding device images in `.llvm.offloading` ELF sections, eliminating the older multi-pass compilation approach. Accepted and implemented by PR #84420 (merged August 2024). This change makes the `.llvm.offloading` binary format the standard packaging format for all vendor GPU targets in LLVM.
- Key detail: After this change (LLVM 19+), all CUDA and HIP compilations produce fat objects in the unified format. A KDL that can read the `.llvm.offloading` section format now has direct access to device binaries produced by the standard LLVM toolchain — no custom packaging required. This dramatically lowers the barrier to libkdl interoperating with existing LLVM-compiled GPU code.

### 6. RFC: SPIR-V IR as a Vendor-Agnostic GPU Representation — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- Type: RFC
- Date: March 2025
- Relevance: 9/10
- Novelty: 10/10
- Summary: RFC (March 2025) proposing SPIR-V as the single portable GPU IR across NVIDIA (nvptx), AMD (amdgcn), and Intel GPUs. The core argument: LLVM has already had success emitting bitcode that ignores incidental GPU architecture differences, leaving the backend to sort out architecture-specific details. The RFC proposes extending this to SPIR-V — emit a single SPIR-V module, then lower to nvptx/amdgcn/GenericELF64 at either AOT compile time or JIT time. Tags MLIR explicitly, noting they "might like to work with a single GPU target that converts to amdgcn/nvptx somewhere downstream." Motivated PR #131190 introducing `llvm.gpu` intrinsics as abstract cross-vendor thread/warp operations.
- Key detail: This RFC directly addresses the "single portable kernel image" question that libkdl also faces. If SPIR-V-as-portable-IR lands, a KDL could store one SPIR-V module per kernel and JIT-compile to the native target at dispatch time — collapsing the multi-version bundle from N binaries to 1. However, the RFC explicitly notes this is "not yet achieved" for all GPU differences (e.g., warp size, memory model extensions). SPIR-V-as-IR is a complement to, not replacement of, multi-versioned native binaries.

### 7. Proposing `llvm.gpu` Intrinsics — LLVM Discourse
- URL: https://discourse.llvm.org/t/proposing-llvm-gpu-intrinsics/75374
- Type: proposal/discussion
- Date: December 2023
- Relevance: 8/10
- Novelty: 9/10
- Summary: Joseph Huber and contributors propose adding `llvm.gpu.*` intrinsics — architecture-neutral LLVM IR operations for common GPU primitives (thread ID, warp shuffle, ballot, etc.). Rationale: these operations have at least one amdgpu and one nvptx intrinsic that are functionally identical but syntactically different, creating unnecessary target fragmentation. Abstract intrinsics would allow shared optimizations, shared libc GPU library code, and SPIR-V emission without target commitment. PR #131190 implements `__builtin_gpu_*` clang builtins + corresponding `llvm.gpu.*` intrinsics, explicitly enabling "postponing choosing a target architecture for libc until JIT time."
- Key detail: This is the IR-level infrastructure that makes SPIR-V-as-portable-IR feasible. The `llvm.gpu` intrinsics decouple kernel semantics from vendor ISA at the LLVM IR layer — exactly the portability primitive a KDL needs to support single-binary dispatch where the binary is LLVM IR + `llvm.gpu` intrinsics rather than pre-compiled vendor-native code.

### 8. [RFC] SYCL Runtime Upstreaming — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-sycl-runtime-upstreaming/74479
- Type: RFC
- Date: October 2023
- Relevance: 8/10
- Novelty: 8/10
- Summary: Intel RFC (October 2023) to upstream the SYCL runtime (`libsycl`) as a top-level LLVM runtime project. Accepted and merged: `libsycl` is now in the LLVM monorepo as a SYCL 2020 API implementation. The SYCL runtime uses the Unified Runtime (UR) API internally as its device abstraction layer, with UR providing backends for CUDA (via NVIDIA), HIP (via AMD), Level Zero (via Intel), and OpenCL. Follow-up RFC (#80323, July 2024) addresses open questions about UR's relationship with liboffload's plugin architecture. The critical observation: both liboffload and the SYCL Unified Runtime solve the same problem (vendor-neutral device/kernel/queue abstractions) with different APIs. No unification effort is currently proposed — they coexist as competing standards within the same monorepo.
- Key detail: The coexistence of liboffload and Unified Runtime (UR) within LLVM means the "canonical" GPU dispatch API is still contested as of 2025-2026. A KDL positioned above both (or abstracting between them) has a stronger architectural story than one tied to either. The libkdl poster should note this fragmentation as ongoing motivation.

### 9. [RFC] LLVM Policy for Top-Level Directories and Language Runtimes — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-llvm-policy-for-top-level-directories-and-language-runtimes/86143
- Type: RFC/governance
- Date: May 2025
- Relevance: 7/10
- Novelty: 8/10
- Summary: Governance RFC (May 2025, initiated by Joseph Huber) addressing "the future where `offload/` begins to contain other languages' runtimes." The question: should SYCL runtime, OpenACC runtime, future language runtimes all live under `offload/`? Or separate top-level directories? The RFC signals that liboffload is being actively positioned as the common substrate for all GPU language runtimes in the LLVM monorepo, not just OpenMP. Community discussion is ongoing and unresolved as of April 2026.
- Key detail: This RFC reveals that the LLVM community is actively debating whether `offload/` becomes the universal GPU runtime home. If resolved in favor of consolidation, liboffload's API surface would become the standard interface for all compiled GPU code in the LLVM ecosystem — making it an even stronger anchor point for a libkdl policy layer.

### 10. [RFC] An MLIR Dialect for Distributed Heterogeneous Computing — LLVM Discourse
- URL: https://discourse.llvm.org/t/rfc-an-mlir-dialect-for-distributed-heterogeneous-computing/86960
- Type: RFC
- Date: June 2025
- Relevance: 7/10
- Novelty: 9/10
- Summary: RFC (June 2025, Robert K Samuel, IIT Madras PACE Lab; also presented at PLDI 2025 SRC) proposing a new MLIR dialect for distributed heterogeneous systems. The dialect introduces a `schedule` operation that groups `task` operations, each annotated with a target attribute (e.g., `cpu`, `gpu`, `fpga`). Enables explicit orchestration, static analysis of task placement, and lowering to MPI Dialect + existing GPU/CPU dialects. A unified intermediate representation for diverse hardware and programming models, supporting both fine- and coarse-grained parallelism with automated task scheduling and optimization.
- Key detail: This RFC proposes exactly the kind of MLIR-native multi-target dispatch representation that previous survey waves identified as a gap. The `schedule` + `task` + `target` pattern is a compile-time analog to what libkdl does at runtime. The RFC does not address runtime selection (it assumes target is known at compile time), which is where libkdl's contribution remains differentiated. If upstreamed, this dialect would be a natural compile-time counterpart to libkdl's runtime dispatch.

### 11. [RFC] Cleaning the GPU Dialect — LLVM Discourse (MLIR)
- URL: https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170
- Type: RFC
- Date: September 2025
- Relevance: 7/10
- Novelty: 8/10
- Summary: RFC (September 2025) to clean the MLIR `gpu` dialect, removing operations that "don't really belong to the dialect" and strengthening its semantics. The dialect has accumulated operations from multiple contributors with overlapping concerns — the RFC aims to establish a cleaner boundary between: (1) target-independent GPU programming model operations (`gpu.launch`, `gpu.func`, barrier ops), (2) target-specific binary management (`gpu.module`, `gpu.binary`), and (3) runtime interaction ops that should move to a separate dialect or lowering pass. Three pages of discussion indicate active community engagement.
- Key detail: The `gpu.binary` operation — which embeds compiled GPU binary objects (PTX, HSACO, SPIR-V) into MLIR modules — is directly relevant to libkdl. The cleanup RFC is clarifying what `gpu.binary` should and should not do: it should represent a compiled binary artifact, not a dispatch policy. This creates a clean boundary where `gpu.binary` = kernel container and a future `gpu.dispatch_select` op (or equivalent) = runtime selection policy. This gap is precisely what libkdl fills.

### 12. GPU/Offloading Workshop 2025 — LLVM Developers' Meeting
- URL: https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832
- Type: workshop slides/announcement
- Date: November 2025 (slides posted after Oct 27 workshop)
- Relevance: 9/10
- Novelty: 9/10
- Summary: Half-day workshop at 2025 LLVM Developers' Meeting (October 27, Santa Clara). Theme: "LLVM/Offload — Where are we, where are we going?" Workshop slides posted include: "LLVM Offloading — Where are We Going?" (state-of-the-art + roadmap), "SYCL status update" (libsycl + UR integration), "Not-Compiler Runtime Library GPUs" (a 633KB slide deck on using LLVM's GPU infrastructure for non-compiler use cases). Joseph Huber's accompanying technical talk slides are at `llvm.org/devmtg/2025-10/slides/technical_talks/huber.pdf`.
- Key detail: The talk titled "Not-Compiler Runtime Library GPUs" is particularly relevant — it explicitly addresses using LLVM's GPU runtime infrastructure from user-space applications, not just from compilers. This is the use case libkdl implements. The workshop's existence and the theme "where are we going?" signals that the community has not yet converged on a complete solution for runtime GPU dispatch, leaving the problem space open for libkdl's contribution.

### 13. LLVM/Offload Workshop 2024 — Preliminary Agenda
- URL: https://discourse.llvm.org/t/announcing-the-preliminary-program-agenda-of-llvm-offload-workshop-llvm-developers-meeting-2024/82535
- Type: workshop agenda
- Date: October 2024
- Relevance: 8/10
- Novelty: 7/10
- Summary: LLVM/Offload Workshop at the 2024 LLVM Developers' Meeting (October 22, Santa Clara). Agenda included: "OMPT Device Support in LLVM" (Dhruva Chakrabarti, AMD) — tool-level observability of GPU kernel dispatch; "Xbc: An Extensible Compiler for Heterogeneous Computing" (Fabian Mora, University of Delaware) — MLIR-based extensible compiler targeting X86, NVIDIA, AMD, and quantum systems; "Rust GPU Offload" (Manuel Drehwald) — safe Rust GPU offloading. The workshop is annual and growing — 2023 was a pre-workshop, 2024 was a half-day, 2025 was a full half-day with CFP.
- Key detail: The "Xbc" paper (published concurrently, Zenodo 2024) reports comparable or better performance vs vendor compilers on NVIDIA A100 and AMD MI250x. xbc demonstrates that MLIR-based extensible compilers for heterogeneous targets are production-feasible. Its model — MLIR dialect + target-specific lowering passes — is architecturally analogous to what the MLIR-native dispatch direction proposes.

### 14. Enhance SYCL Offloading Support Talk — LLVM DevMtg 2024
- URL: https://llvm.org/devmtg/2024-10/slides/techtalk/Narayanaswamy-EnhanceSYCL-offloading-support.pdf
- Type: talk slides
- Date: October 2024
- Relevance: 8/10
- Novelty: 8/10
- Summary: Technical talk on migrating SYCL's compilation flow to the new unified offloading driver model. Key changes: adding a `clang-sycl-linker` tool that runs SYCL finalization steps (device code splitting, post-link processing) inside the unified link pipeline; modifying `clang-linker-wrapper` to invoke clang-sycl-linker for SYCL cases; full AOT compilation support for Intel, AMD, and NVIDIA GPUs via the same fat-object flow. The talk explicitly describes "device code splitting" as a finalization step — individual SYCL kernels can be split into per-kernel binary objects within the fat object container.
- Key detail: The SYCL per-kernel splitting model maps directly onto libkdl's kernel registry. After splitting, each SYCL kernel is an individually addressable binary artifact — exactly the granularity at which libkdl operates (`kdl_load_kernel(bundle, "kernel_name")`). SYCL's new flow confirms that the LLVM ecosystem is moving toward finer-grained kernel binary management, validating libkdl's design.

---

## Angle Assessment

**What is the state of LLVM community consensus on heterogeneous GPU dispatch?**

The discourse threads reveal a community that has clearly identified the fragmentation problem and is actively building infrastructure, but has not yet converged on a complete solution. The key observations:

1. **Problem consensus is unanimous**: Every RFC from 2023-2025 opens with the same framing — vendor-specific runtimes duplicating code, no shared dispatch infrastructure. This directly validates libkdl's motivation without requiring citation of external literature.

2. **Mechanism exists, policy does not**: liboffload provides `olCreateProgram(binary_blob)` + `olCreateKernel(program, "name")` + `olEnqueueKernelLaunch()`. This is a complete dispatch mechanism. What does not exist: selection among multiple binary blobs based on runtime device capability. This is the multi-version selection gap — libkdl's primary contribution.

3. **Binary format is converging**: The new unified driver (default in LLVM 19+) produces `.llvm.offloading` section fat objects for all GPU targets. This is now the standard packaging format. libkdl should document alignment with this format.

4. **SPIR-V-as-portable-IR is actively proposed but unproven**: The March 2025 RFC proposes SPIR-V + `llvm.gpu` intrinsics as a path to single-binary dispatch. This is speculative for performance-sensitive code — SPIR-V → nvptx translation has latency and quality issues (documented in wave-01-spirv-portable-ir.md). Multi-versioned native binaries remain the pragmatic approach for production dispatch.

5. **Two competing runtime standards coexist**: liboffload and oneAPI Unified Runtime both provide vendor-neutral device/kernel APIs. They are not unified. A KDL positioned above both is architecturally stronger than one tied to either.

6. **No runtime dispatch policy anywhere in LLVM**: Despite the MLIR Distributed Heterogeneous Computing RFC (June 2025) and the GPU dialect cleanup (September 2025), no existing RFC proposes runtime selection among multiple pre-compiled kernel variants. The gap is real and confirmed by community absence.

**Strategic framing for the poster:**

The Discourse record is the strongest possible confirmation that libkdl addresses a recognized, unresolved problem in the LLVM ecosystem. The poster should present libkdl as:
- Built on liboffload's mechanism layer (not competing with it)
- Filling the policy gap that liboffload explicitly excludes from its roadmap
- Aligned with the converging `.llvm.offloading` fat-object packaging standard
- Complementary to the SPIR-V portable IR effort (serving the near-term before SPIR-V matures)
- The user-space runtime analog to what the "Not-Compiler Runtime Library GPUs" talk describes as needed

**New gaps identified in this wave:**

1. The "Not-Compiler Runtime Library GPUs" slide deck (GPU workshop 2025) — not yet analyzed, may contain direct references to user-space dispatch needs
2. The governance RFC (#86143, May 2025) on `offload/` as universal home — if resolved, affects libkdl's positioning relative to liboffload
3. The MLIR Distributed Heterogeneous Computing dialect (RFC #86960) — if upstreamed, could be a compile-time partner to libkdl's runtime dispatch
4. xbc (LLVM DevMtg 2024) — extensible heterogeneous compiler achieving 5x over Clang OpenMP on AMD MI250x; performance data may be useful for benchmarking context

**Cross-references to prior waves:**
- liboffload API internals: wave-02-llvm-offloading.md (sources 2, 3, 4, 7)
- SPIR-V portability limitations: wave-01-spirv-portable-ir.md
- MLIR gpu dialect gaps: wave-01-mlir-gpu-dialect.md
- AdaptiveCpp SSCP (SPIR-V JIT): wave-03-adaptivecpp.md
- Multi-versioned kernel formats: wave-03-multi-versioned-kernels.md
