# Wave 01: XLA PjRt Plugin Device Abstraction
Search query: XLA PjRt plugin API device abstraction multi-backend dispatch JAX
Sources found: 10
Date: 2026-04-06

## Sources

### 1. PJRT - Uniform Device API (OpenXLA Official Docs)
- URL: https://openxla.org/xla/pjrt
- Type: docs
- Date: 2024–2025 (actively maintained)
- Relevance: 10/10
- Novelty: 5/10
- Summary: The canonical reference for PjRt's architecture and vision. Describes PjRt as "Pretty much just a Runtime" — a stable, toolchain-independent interface where frameworks (JAX, TF, PyTorch) call PjRt, which dispatches to device-specific plugin implementations opaque to the framework. Each hardware vendor ships a PjRt plugin that the framework dynamically loads at runtime.
- Key detail: Long-term vision explicitly states: "(1) frameworks call PJRT, device implementations are opaque to frameworks; (2) each device implements PJRT APIs as a plugin opaque to frameworks." This is effectively ld.so-style late binding for ML compute — directly analogous to libkdl's design goal.

### 2. PJRT C++ Device API Overview (OpenXLA)
- URL: https://openxla.org/xla/pjrt/cpp_api_overview
- Type: docs
- Date: 2024–2025
- Relevance: 9/10
- Novelty: 6/10
- Summary: Documents the C++ layer of PjRt: Client owns devices and memory spaces for a plugin; Device describes a single hardware unit with a DeviceDescription (unique kind hash identifying GPU/CPU/xPU) plus local and global grid location. The client-device model mirrors hardware abstraction layers in OS kernels and is directly comparable to KDL's device registry.
- Key detail: Device kind is identified by a unique hash — this is PjRt's analog to KDL's capability fingerprinting. The `Client` abstraction manages all framework-to-plugin communication and owns buffer allocation lifecycle.

### 3. PJRT plugin integration guide (OpenXLA)
- URL: https://openxla.org/xla/pjrt/pjrt_integration
- Type: docs
- Date: 2024–2025
- Relevance: 9/10
- Novelty: 6/10
- Summary: Explains the two implementation paths for a PjRt plugin: (A) implement the C API directly (stable ABI), or (B) implement C++ API and use the C→C++ wrapper when building against the XLA source tree. Also describes the Python package structure: a shared library exposing the C API + a Python `DevicePlugin` interface for framework discovery.
- Key detail: The plugin is discovered at runtime — frameworks scan for PjRt plugins (Python packages), call `PJRT_Plugin_Initialize` once for one-time setup, then use `PJRT_Client_Create` to instantiate device access. This is a dynamic linking pattern that directly motivates libkdl.

### 4. RFC: OpenXLA PJRT Plugin (20230123)
- URL: https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md
- Type: RFC
- Date: 2023-01-23
- Relevance: 9/10
- Novelty: 7/10
- Summary: The design RFC that defined PjRt's plugin mechanism. Specifies that the API is backward compatible so plugins need not change often, with version-checking for features. Dynamic plugin loading was described as "brand new as of 12/21/2022" with "sharp edges still" — indicating this is a recent, evolving capability. Addresses whether multiple frameworks can share a device client (hardware-dependent, not mandated by the API).
- Key detail: As of RFC date, dynamic plugin loading was experimental. The RFC's explicit "sharp edges still" admission reveals the immaturity of the runtime dispatch infrastructure at the time, creating a gap that a dedicated KDL-style runtime could fill more robustly.

### 5. pjrt_c_api.h — C API Header with ABI versioning (openxla/xla)
- URL: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
- Type: commit / source file
- Date: actively updated (2023–2025)
- Relevance: 8/10
- Novelty: 7/10
- Summary: The stable ABI contract between frameworks and plugins. Uses `PJRT_STRUCT_SIZE` and `PJRT_DEFINE_STRUCT_TRAITS` macros for size-aware struct evolution (forward/backward compatible). API uses major+minor versioning: major incremented for incompatible changes (deleting methods, changing argument types, rearranging fields), minor for additive changes. Extension structs allow opt-in capabilities without breaking the core ABI.
- Key detail: The `PJRT_Extension_Base` pattern (each struct carries `struct_size` for compatibility checks) is a well-known ABI evolution technique analogous to COM/D-Bus. The 6-week minor-version forward-compatibility window is a concrete SLA — any KDL design claiming ABI stability must provide a comparable guarantee.

### 6. PJRT Plugin to Accelerate Machine Learning (Google Open Source Blog, March 2024)
- URL: https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html
- Type: blog
- Date: 2024-03
- Relevance: 8/10
- Novelty: 6/10
- Summary: Announces the production maturity of the PjRt plugin ecosystem, listing Apple silicon (Metal), Google Cloud TPU, NVIDIA GPU, and Intel Max GPU as shipping plugins. Confirms PjRt is the sole interface for JAX and the primary interface for TensorFlow — making this the de-facto standard for ML-framework-to-hardware dispatch in the XLA world.
- Key detail: Apple's Metal plugin (contributors: Chalana Bezawada, Daniel Doctor, Kulin Seth, Shuhan Ding) was implemented by Apple engineers, not Google — this is direct evidence that non-Google hardware vendors can and do build PjRt plugins, validating the plugin model's openness and relevance to a vendor-agnostic KDL approach.

### 7. Accelerate JAX models on Intel GPUs via PJRT (Google Open Source Blog, June 2023)
- URL: https://opensource.googleblog.com/2023/06/accelerate-jax-models-on-intel-gpus-via-pjrt.html
- Type: blog
- Date: 2023-06
- Relevance: 8/10
- Novelty: 6/10
- Summary: Describes Intel's PjRt plugin for Intel Max GPU. The plugin compiles StableHLO using XLA's compilation infrastructure, adds Intel-specific MLIR passes, targets SPIR-V IR via LLVM, and dispatches using the SYCL runtime + oneAPI libraries (oneDNN, oneMKL). This is a concrete end-to-end example: StableHLO → MLIR passes → SPIR-V → SYCL dispatch.
- Key detail: LLVM + SPIR-V code-gen is the IR path chosen by Intel — not vendor-specific IR. This directly connects PjRt to LLVM/MLIR pipelines and shows how PjRt plugins can use MLIR internally while presenting a unified C API externally.

### 8. Accelerated JAX on Mac — Apple Metal Developer Page
- URL: https://developer.apple.com/metal/jax/
- Type: docs
- Date: 2024
- Relevance: 7/10
- Novelty: 6/10
- Summary: Apple's official guide for JAX on Metal via PjRt. The Metal plugin translates StableHLO to MPSGraph executables and Metal runtime API calls for GPU dispatch. Demonstrates the full stack: JAX → StableHLO (via OpenXLA compiler) → MPSGraph/Metal kernel dispatch — with PjRt as the seam between framework and hardware.
- Key detail: The translation target is MPSGraph (not SPIR-V or PTX), meaning PjRt abstracts away fundamentally different native dispatch mechanisms across vendors — exactly the problem KDL solves at the kernel binary level.

### 9. PJRT: Simplifying ML Hardware and Framework Integration (Google Open Source Blog, May 2023)
- URL: https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html
- Type: blog
- Date: 2023-05
- Relevance: 7/10
- Novelty: 5/10
- Summary: The original public announcement of PjRt as a unifying runtime interface. Frames the motivation: previously each framework-hardware pair required a custom integration; PjRt provides a single stable API so N frameworks × M hardware vendors collapses to N+M integrations. This is the "N×M → N+M" argument central to KDL's positioning.
- Key detail: Explicitly states "PjRt is not tied to a specific compiler and runtime" — toolchain independence is a first-class design goal, the same property KDL targets for binary kernel dispatch.

### 10. Intel Extension for OpenXLA — GitHub (intel/intel-extension-for-openxla)
- URL: https://github.com/intel/intel-extension-for-openxla
- Type: PR / commit / repo
- Date: 2023–2025 (active)
- Relevance: 7/10
- Novelty: 6/10
- Summary: Open-source reference implementation of a non-Google PjRt plugin. Compiles StableHLO → SPIR-V via LLVM, uses SYCL runtime for device dispatch, integrates oneDNN/oneMKL for accelerated kernels. Provides JAX, TensorFlow, and PyTorch/XLA support from a single plugin binary. Also includes an IREE-based PJRT plugin path (stellaraccident/iree-pjrt), showing PjRt can wrap alternative compilers.
- Key detail: The `stellaraccident/iree-pjrt` repository shows PjRt being used as a thin wrapper around IREE — meaning PjRt is composable with other MLIR-based runtimes, not a monolithic stack. This "PjRt as a dispatch facade" pattern is architecturally aligned with KDL's separation of dispatch from compilation.

---

## Angle Assessment

- **Coverage:** This angle is well-explored in public documentation. PjRt has official docs, an RFC, a public C header, multiple blog posts, and multiple open-source third-party plugin implementations. The core architecture is stable and well-documented. What is less documented: (1) the failure modes of dynamic plugin loading under ABI mismatch, (2) cross-plugin dispatch (using multiple PjRt plugins simultaneously for heterogeneous hardware), and (3) performance characteristics of the plugin dispatch overhead itself.

- **Surprise findings:**
  - `stellaraccident/iree-pjrt` exists — PjRt wrapping IREE as a backend. This is a direct intersection with the IREE wave and shows PjRt is not the end of the stack but a seam within a larger MLIR pipeline.
  - The Apple Metal plugin dispatches to MPSGraph, not a portable IR. This means PjRt-level abstraction does NOT imply portable binary kernels — PjRt abstracts the dispatch protocol but not the kernel binary format. This is KDL's unique contribution: portable kernel binaries with dynamic selection, which PjRt does not address.
  - The RFC's "sharp edges still" comment on dynamic loading (as of late 2022) reveals that runtime plugin loading was an afterthought in PjRt's design, not a first-class mechanism. KDL could be positioned as solving this more robustly.

- **Gaps:**
  - No AMD ROCm PjRt plugin found in public sources — AMD GPU support in JAX appears to go through the XLA HLO path directly, not a plugin. This is a meaningful gap in the ecosystem that KDL could address.
  - No documentation on simultaneous multi-backend dispatch (e.g., split a workload between NVIDIA and Intel GPU using two PjRt plugins in the same process). PjRt's multi-client model is underspecified for heterogeneous dispatch.
  - No published latency benchmarks for PjRt plugin dispatch overhead (dlopen + C API call chain) vs. direct CUDA/HIP dispatch. This is a critical gap for a poster claiming KDL is lower-overhead.
  - PjRt's ABI stability is not yet fully delivered ("will start supporting ABI compatibility soon" as of 2024) — a real engineering risk for production plugin authors.

- **Suggested follow-up angles:**
  1. **IREE as a PjRt backend** — `stellaraccident/iree-pjrt` merges IREE's MLIR pipeline with PjRt's dispatch interface; directly relevant to LLVM/MLIR track.
  2. **StableHLO ↔ MLIR dialect evolution** — how StableHLO versioning interacts with PjRt plugin ABI versioning; a two-layer stability problem.
  3. **Multi-plugin simultaneous dispatch in JAX** — is there a mechanism to route different ops to different PjRt plugins? If not, this is a hard limitation KDL solves at the binary kernel level.
  4. **PjRt dispatch overhead measurement** — latency from JAX jit call through PjRt C API to first GPU instruction; compare to KDL's measured dispatch latency.
  5. **AMD ROCm + PjRt gap** — why is there no public ROCm PjRt plugin? XLA's HLO-to-ROCm path, IREE's ROCm support, and KDL's HIP dispatch all attempt to fill this gap differently.
