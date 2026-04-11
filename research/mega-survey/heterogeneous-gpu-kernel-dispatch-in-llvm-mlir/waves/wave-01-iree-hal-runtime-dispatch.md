# Wave 01: IREE HAL Runtime Dispatch
Search query: "IREE HAL runtime dispatch multi-backend GPU CPU target selection"
Sources found: 10
Date: 2026-04-06

NOTE: A companion file `wave-01-iree-hal.md` covers the HAL architectural overview from a
different search angle. This file focuses on runtime dispatch mechanics, unified target
descriptions, and the GPU backend unification epic — areas not covered by that earlier wave.

---

## Sources

### 1. IREE Deployment Configurations — Official Docs
- URL: https://iree.dev/guides/deployment-configurations/
- Type: docs
- Date: 2024 (continuously updated)
- Relevance: 9/10
- Novelty: 5/10
- Summary: Canonical overview of all supported IREE runtime backends: `cuda`, `hip`, `local-sync`,
  `local-task`, `vulkan`. Each backend maps to a distinct HAL driver registered at process start.
  Target selection at compilation time uses `--iree-hal-target-backends=<name>` flags; at runtime
  the matching HAL driver must be present. No cross-backend fallback is performed automatically.
- Key detail: CPU backends split into `local-sync` (blocking, single-threaded, minimal overhead)
  and `local-task` (multi-threaded via task system). This bifurcation matters for dispatch overhead
  measurement: `local-sync` gives a near-zero baseline whereas `local-task` adds queue/scheduling
  latency analogous to a GPU driver.

### 2. IREE HAL Dialect Reference
- URL: https://iree.dev/reference/mlir-dialects/HAL/
- Type: docs
- Date: 2024 (continuously updated)
- Relevance: 10/10
- Novelty: 8/10
- Summary: Specifies the `hal.executable.variant` and `hal.executable.export` ops. An executable
  may carry multiple variants for different target architectures. The optional `condition` region on
  `hal.executable.export` is a boolean predicate evaluated against the runtime `!hal.device`; when
  it returns false the system falls back to the next export in a user-defined chain. This is the
  primary multi-architecture dispatch mechanism.
- Key detail: Variant selection is evaluated **at module load time**, not per-dispatch. The
  condition region can only inspect static device properties (ISA capabilities, Vulkan extension
  availability) — it cannot respond to runtime load, thermal state, or per-kernel profiling data.
  This is IREE's fundamental static-dispatch limitation.

### 3. IREE Design Roadmap — Heterogeneous Device Placement
- URL: https://iree.dev/developers/design-docs/design-roadmap/
- Type: docs
- Date: 2024
- Relevance: 10/10
- Novelty: 9/10
- Summary: The roadmap explicitly describes planned heterogeneous dispatch: operations, dispatches,
  and streams will carry device-category attributes resolved by constraint solvers. Three solver
  strategies are described — generic heuristics ("big GEMMs go on the accelerator"),
  profile-guided benchmark databases, and ML-learned traits. Constraint solving operates in the
  `flow` dialect before lowering to HAL, respecting per-device limits on in-flight memory and
  scheduling depth.
- Key detail: As of April 2026 this remains a roadmap item, not a shipping feature. The production
  gap between the described constraint-based multi-device dispatch and the current load-time variant
  selection is the exact research contribution space occupied by libkdl.

### 4. [Epic] Rework GPU Compiler Backends — GitHub Issue #16341
- URL: https://github.com/iree-org/iree/issues/16341
- Type: issue (epic tracker)
- Date: 2023–2024
- Relevance: 9/10
- Novelty: 8/10
- Summary: Tracks unification of IREE's two GPU codegen paths: LLVMGPU (CUDA/ROCm) and SPIR-V
  (Vulkan/Metal/WebGPU). Goal is a single `#iree_gpu.target` attribute shared across both paths,
  enabling multi-target codegen (compiling one kernel for both CUDA and Vulkan in one pass). Also
  separates CUDA and ROCm into distinct NVVM and ROCDL subdirectories.
- Key detail: SPIR-V was historically over-coupled with Vulkan; the rework decouples SPIR-V target
  management from Vulkan API specifics. Multi-target codegen requires SPIR-V default target set to
  null and separate null defaults for HIP and CUDA — signaling that the target is determined at
  runtime rather than hard-coded at compile time. This is the compiler-side precondition for
  runtime dispatch flexibility.

### 5. Rework HAL Device Querying — Uniform Device Identifier (Issue #9343)
- URL: https://github.com/iree-org/iree/issues/9343
- Type: issue (CLOSED, completed May 2023)
- Date: 2022–2023
- Relevance: 8/10
- Novelty: 7/10
- Summary: Implemented a URI-based device identifier scheme so all IREE backends share one
  selection syntax. Format: `scheme://path?query#fragment` maps drivers to URI schemes and physical
  devices to URI paths. Example: `--device=cuda://GPU-abcd0` or `--device=vulkan://uuid/<uuid>`.
  The push-style design means modules declare a preference list; the runtime does not query
  available devices unless explicitly requested (avoiding expensive subsystem bring-up on startup).
- Key detail: Design principle "cheaper to try and fail than to query and conditionally create"
  directly minimizes dispatch infrastructure overhead. Multiple `--device=` flags compose a
  heterogeneous execution pool: `--device=local-task --device=vulkan` enables CPU+GPU co-execution
  with the module selecting from the provided list.

### 6. Support for MLIR-Based Microkernels on GPU Backends — GitHub Issue #17788
- URL: https://github.com/iree-org/iree/issues/17788
- Type: issue (open)
- Date: 2023–2024
- Relevance: 8/10
- Novelty: 9/10
- Summary: Proposes replacing LLVM-bitcode microkernels on GPU with MLIR-based microkernels linked
  before the LLVM lowering step. GPU microkernels face a constraint not present on CPU: workgroup
  size and subgroup size must be consistent between the microkernel and its calling context, so
  these parameters must be annotated on the microkernel function and validated at kernel config
  time. Current GPU microkernels exist only as LLVM bitcode for CUDA/ROCm.
- Key detail: The MLIR-level linking approach (link before lowering to LLVM/SPIR-V) is strictly
  more flexible than bitcode-level linking — it enables target-agnostic microkernel authoring in
  MLIR that specializes to CUDA PTX, SPIR-V, or ROCDL during compilation. This is directly
  relevant to libkdl's "kernel binary" abstraction: MLIR microkernels are the IREE equivalent of
  libkdl's multi-target dispatch units.

### 7. The Long Tail of AI: SPIR-V in IREE and MLIR (Vulkanised 2025)
- URL: https://vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf
- Type: conference talk (Vulkanised, Feb 2025)
- Date: 2025-02
- Relevance: 9/10
- Novelty: 10/10
- Summary: AMD engineer Jakub Kuderski (principal contributor to IREE GPU codegen) presented
  IREE+MLIR as AMD's mechanism for a unified AI software stack spanning Radeon GPUs, Instinct
  accelerators (MI300X), Ryzen AI NPUs, and CPU. SPIR-V is positioned as the portable exchange IR
  across all of these. The talk explicitly calls out work-in-progress heterogeneous executables
  carrying GPU and NPU kernels in the same program binary, plus "one-click" whole-model PGO tuning.
- Key detail: Ongoing work on heterogeneous GPU+NPU kernel dispatch within a single IREE module
  binary is confirmed active at AMD as of Feb 2025. This is the most direct public statement that
  IREE is actively pursuing the problem libkdl addresses — making this a critical competitive
  reference for the Dublin poster.

### 8. IREE nod-ai Kernel Benchmark Repository
- URL: https://github.com/nod-ai/iree-kernel-benchmark
- Type: code/tool
- Date: 2024–2025
- Relevance: 7/10
- Novelty: 6/10
- Summary: Framework for benchmarking individual kernel performance through IREE's dispatch
  infrastructure. Supports both local IREE builds and pip-installed packages. Used by nod-ai
  (AMD's IREE productization team) to track roofline efficiency of GEMM and attention kernels
  across RDNA/CDNA hardware. Measures dispatch-level throughput including HAL overhead.
- Key detail: The benchmark infrastructure uses `-iree-flow-export-benchmark-funcs` to isolate
  individual dispatch functions, giving per-dispatch latency and FLOPS data. No published
  microsecond-level HAL dispatch overhead numbers were found, only end-to-end kernel throughput.

### 9. IREE CUDA HAL Driver Design Doc
- URL: https://iree.dev/developers/design-docs/cuda-hal-driver/
- Type: docs
- Date: 2023–2024
- Relevance: 8/10
- Novelty: 5/10
- Summary: CUDA HAL driver uses CUDA Driver API (not Runtime API) for minimal dependency surface.
  PTX stored in FlatBuffer alongside entry-point metadata; JIT-compiled by CUDA driver at first
  load and cached for process lifetime. Two command buffer backends: CUDA Graphs (default, lower
  per-kernel overhead via graph replay) and CUDA Streams (direct issue, higher per-dispatch cost).
- Key detail: CUgraph-backed command buffers record once and replay many times — the HAL's primary
  mechanism for amortizing dispatch overhead. This is the same pattern libkdl's persistent kernel
  registry exploits. The process-lifetime cache is a known limitation: no cross-process or
  persistent-to-disk kernel cache, forcing re-JIT on every process start.

### 10. Single-Node ML Runtime Foundation (Lei.Chat blog)
- URL: https://www.lei.chat/posts/single-node-ml-runtime-foundation/
- Type: blog
- Date: 2023–2024
- Relevance: 8/10
- Novelty: 7/10
- Summary: Architectural analysis of IREE as a single-node ML runtime. Key insight: IREE bundles
  kernels for multiple vendors in a fat FlatBuffer binary; the runtime probes hardware capabilities
  and selects matching kernels. The host VM and device HAL are both optional/pluggable — IREE can
  run as a library embedded inside PyTorch or TF with the host scheduler replaced. Emphasizes that
  the same `flow` graph partitioning scheme is used across all backends, ensuring exchangeable
  kernels target the same source subgraph.
- Key detail: "Same graph partitioning and dispatch region formation scheme to make sure kernels
  targeting the same source subgraph are exchangeable" — this is the architectural invariant that
  makes fat-binary multi-backend dispatch correct. libkdl must enforce an analogous invariant:
  kernels registered for the same dispatch slot must compute identical semantics.

---

## Angle Assessment

### Coverage
This angle is **thoroughly documented** at the architectural and design level. IREE's public
documentation, GitHub issues, and conference talks together give a clear picture of:
- Current state: load-time variant selection via boolean condition ops
- Near-term work: unified `#iree_gpu.target`, multi-target codegen, URI device selection (shipped)
- Future roadmap: constraint-based heterogeneous dispatch in the `flow` dialect (unshipped)
- Production usage: AMD nod-ai using IREE for Instinct+Radeon+NPU unified dispatch

What is missing: empirical dispatch overhead numbers (microsecond-level HAL vs. direct API cost),
and a publicly available implementation of the constraint-based multi-device scheduler.

### Surprise Findings
1. **AMD is already building GPU+NPU heterogeneous dispatch inside IREE** (Vulkanised 2025 talk by
   Kuderski). This is a direct competitive parallel to libkdl's contribution — the Dublin poster
   must differentiate explicitly from this in-progress IREE work.
2. **URI-based device selection (issue #9343) shipped in 2023** and is production-ready. IREE
   already solves the "which backend to use" question at process invocation time. libkdl operates
   at a finer granularity: per-kernel dispatch, not per-process device registration.
3. **`local-sync` vs. `local-task` CPU split** reveals that IREE's own team recognized dispatch
   infrastructure overhead as significant enough to warrant a dedicated zero-overhead CPU path.
4. **MLIR microkernel linking** (issue #17788) proposes a target-agnostic kernel authoring model
   nearly identical to what libkdl's `.kdl` format attempts — this is prior art that should be
   cited and differentiated from.

### Gaps
1. **No published HAL dispatch latency numbers** — the docs discuss CUDA Graph amortization but
   give no concrete overhead figures. The iree-kernel-benchmark repo measures throughput only.
2. **No runtime-adaptive selection** — IREE's variant selection cannot respond to current GPU
   utilization, thermal throttling, or per-kernel execution history. This is libkdl's primary
   differentiation.
3. **No CPU+GPU co-dispatch within a single kernel** — device affinity is assigned at the
   executable/command-buffer level, not at the subgroup/warp level.
4. **No cross-process kernel binary caching** — HAL JIT results are per-process only. libkdl's
   persistent kernel registry is a direct improvement over this.
5. **IREE Stream dialect** (the layer above HAL that does async scheduling and partitioning) is
   under-researched in this wave — it is likely where future multi-device work will first appear.

### Suggested Follow-Up Angles
1. **IREE Stream dialect** — `stream.cmd.dispatch`, `stream.resource`, partition assignment ops;
   this is where the constraint solver described in the roadmap would actually live.
2. **IREE PGO / tuning infrastructure** — the Vulkanised talk mentions "one-click whole-model PGO
   tuning"; understanding this pipeline is relevant to libkdl's performance-adaptive dispatch.
3. **IREE PJRT plugin** (RFC: merge PJRT into iree repo) — how IREE surfaces its runtime to JAX
   and XLA, which directly addresses the "connect to PyTorch/TF ecosystem" poster requirement.
4. **IREE + ROCm HIP HAL driver** (iree.dev/developers/design-docs/hip-hal-driver/) — the HIP
   driver is the productionized AMD path; its design diffs from CUDA HAL reveal backend-specific
   dispatch overhead trade-offs.
5. **Measured HAL overhead** — run `iree-benchmark-module` on a GEMM kernel across
   `local-sync`, `local-task`, `cuda`, and `vulkan` backends on the GTX 1650 to get concrete
   dispatch latency numbers for the poster.
