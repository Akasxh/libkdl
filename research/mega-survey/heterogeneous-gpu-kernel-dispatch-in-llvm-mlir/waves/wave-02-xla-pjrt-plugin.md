# Wave 02: XLA PJRT Plugin Interface
Search query: "XLA PJRT plugin interface device dispatch JAX multi-GPU heterogeneous"
Sources found: 10
Date: 2026-04-06

## Sources

### 1. PJRT Plugin to Accelerate Machine Learning (Google Open Source Blog, March 2024)
- URL: https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html
- Type: blog
- Date: 2024-03-01
- Relevance: 9/10
- Novelty: 6/10
- Summary: Comprehensive 2024 status update on the PJRT plugin ecosystem. Describes the full workflow: JAX programs lower to StableHLO, which is passed to PJRT for compilation and execution on a target device. Covers multi-node coordination via key-value store callbacks that CUDA/NCCL plugins use to establish cluster-wide device topology.
- Key detail: Multi-node PJRT uses framework-provided key-value store callbacks at plugin init time to build a global device topology map and generate NCCL communicator IDs — the plugin itself drives cross-node discovery, not the framework.

### 2. PJRT Plugin Integration | OpenXLA Docs
- URL: https://openxla.org/xla/pjrt/pjrt_integration
- Type: docs
- Date: 2025-04-30 (last updated)
- Relevance: 10/10
- Novelty: 5/10
- Summary: Official authoritative guide for implementing a PJRT plugin. Defines the two implementation paths: (a) direct C API via `GetPjRtApi()` returning a `PJRT_Api*` struct of function pointers, or (b) C++ API with a C→C++ shim wrapper. Covers plugin discovery via `PJRT_PLUGIN_LIBRARY_PATH` env var and Python module naming convention.
- Key detail: `GetPjRtApi()` is the single ABI entry point — it returns a struct of function pointers covering client creation, device enumeration, buffer management, executable compilation, and async execution. The entire device runtime is behind this one C-ABI boundary.

### 3. PJRT - Uniform Device API | OpenXLA Project
- URL: https://openxla.org/xla/pjrt
- Type: docs
- Date: 2025-04-25 (last updated)
- Relevance: 9/10
- Novelty: 4/10
- Summary: Top-level conceptual overview of PJRT as a hardware- and framework-agnostic runtime interface. Positions PJRT as the single interface for JAX, primary interface for TensorFlow, and supported (experimental) interface for PyTorch/XLA. Frameworks call PJRT; device implementations are opaque plugins.
- Key detail: PJRT is the *only* interface JAX uses for device execution — there is no fallback path. This makes PJRT the chokepoint for any heterogeneous kernel dispatch in the JAX ecosystem.

### 4. PJRT C++ Device API Overview | OpenXLA Project
- URL: https://openxla.org/xla/pjrt/cpp_api_overview
- Type: docs
- Date: 2025
- Relevance: 8/10
- Novelty: 6/10
- Summary: Detailed reference for the C++ layer of PJRT. Describes the core abstract classes: `PjRtClient`, `PjRtDevice`, `PjRtBuffer`, `PjRtLoadedExecutable`, and `PjRtExecuteOptions`. The C API is a stable ABI projection of these C++ abstractions, with a versioned wrapper translating between the two layers.
- Key detail: `PjRtDevice` carries device ordinal, device kind (string), and memory spaces — the heterogeneous dispatch decision in JAX (`device_put`, `jit(device=...)`) resolves through `PjRtClient::devices()` which returns a flat list across all device kinds registered with the client.

### 5. OpenXLA PJRT Plugin RFC (openxla/community)
- URL: https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md
- Type: RFC
- Date: 2023-01-23
- Relevance: 9/10
- Novelty: 8/10
- Summary: Original design RFC that established the PJRT plugin mechanism. Defines the plugin discovery contract, the Python `xb.register_plugin()` call, and the rationale for a stable C ABI boundary separating framework from device runtime. Establishes the principle that plugins can evolve independently of jaxlib within a compatibility window.
- Key detail: The RFC explicitly sets a 6-week forwards compatibility window for minor version bumps — a framework newer than a plugin by up to 6 weeks of minor versions must still work. Full ABI stability (major version lock) was deferred and noted as future work.

### 6. PJRT C API — pjrt_c_api.h (openxla/xla main)
- URL: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
- Type: commit/source
- Date: ongoing (mainline)
- Relevance: 10/10
- Novelty: 7/10
- Summary: The canonical C header defining the PJRT ABI. Every struct carries a `struct_size` field for forwards/backwards compatibility checks. Recent additions include `PJRT_Triton_Extension`, `PJRT_ExecuteContext`, new float types (F4E2M1FN, F8E8M0FNU for FP4/FP8), and async host-to-device transfer manager APIs.
- Key detail: The extension mechanism (`PJRT_Extension_Base` linked list off `PJRT_Api`) allows plugins to expose optional capabilities (e.g., Triton-native kernel injection) without breaking the core ABI — critical for heterogeneous dispatch where different backends expose different capabilities.

### 7. PJRT C API Changelog (openxla/xla)
- URL: https://github.com/openxla/xla/blob/main/xla/pjrt/c/CHANGELOG.md
- Type: commit/docs
- Date: ongoing (last checked 2025-2026)
- Relevance: 8/10
- Novelty: 7/10
- Summary: Version-by-version record of PJRT API additions and deprecations. Recent entries show DMA mapping APIs (`PJRT_Client_DmaMap`/`DmaUnmap`), Triton extension, and FP4/FP8 dtype support. Deprecations are tracked with migration paths to new preferred fields.
- Key detail: `PJRT_Client_CreateViewOfDeviceBuffer_Args` added a `memory` field (type `PJRT_Memory*`) deprecating the old `device` field — indicating the model is moving toward a finer-grained memory-space abstraction rather than device-only dispatch, which has implications for NUMA-aware and heterogeneous memory dispatch.

### 8. Intel Extension for OpenXLA — SYCL PJRT Plugin
- URL: https://github.com/intel/intel-extension-for-openxla
- Type: docs/source
- Date: 2024 (ongoing releases)
- Relevance: 8/10
- Novelty: 8/10
- Summary: Production PJRT plugin for Intel GPUs (Data Center Max, Arc B-Series) using SYCL/oneAPI as the device runtime. Compiles StableHLO through XLA with Intel-specific passes, then dispatches to SYCL runtime. Uses oneDNN v3.2+ for matrix kernels on XMX (Matrix Extensions) hardware.
- Key detail: Intel's plugin routes XLA-compiled kernels through SYCL instead of CUDA, validating that the PJRT C ABI is genuinely portable across divergent GPU software stacks — the same JAX program runs on NVIDIA (CUDA plugin) or Intel (SYCL plugin) with only a plugin swap at init time.

### 9. PyTorch/XLA Custom Hardware Plugins Docs
- URL: https://docs.pytorch.org/xla/master/contribute/plugins.html
- Type: docs
- Date: 2024-2025
- Relevance: 7/10
- Novelty: 5/10
- Summary: Guide for implementing PJRT plugins in the PyTorch/XLA context. Requires implementing `PjRtClient` with an embedded XLA compiler and runtime. PJRT is the default runtime for TPU and CPU; GPU support is experimental. Tenstorrent has a fork (`tenstorrent/pytorch-xla`) demonstrating third-party hardware integration.
- Key detail: PyTorch/XLA GPU via PJRT achieved TorchBench pass rate within 5% of TorchInductor as of 2024, with XLA's global cost model driving GPU operator fusion decisions — showing PJRT is viable for production PyTorch workloads, not just JAX.

### 10. JAX Heterogeneous CPU/GPU Device Mesh Discussion
- URL: https://github.com/jax-ml/jax/discussions/18372
- Type: PR/discussion
- Date: 2024
- Relevance: 8/10
- Novelty: 9/10
- Summary: Community discussion on whether JAX supports `jax.sharding.Mesh` constructed from mixed device types (CPU + GPU in the same mesh). Users attempting `jax.devices() + jax.devices("cpu")` encounter device identification errors — current PJRT does not expose a unified heterogeneous mesh.
- Key detail: JAX's `device_put` and `jit(device=...)` support routing to *any single device* from any registered PJRT plugin, but constructing a sharding `Mesh` across *different device kinds* (true heterogeneous dispatch) is not yet natively supported as of 2024-2025. This is a concrete gap in current PJRT.

## Angle Assessment

- Coverage: Well-explored for homogeneous multi-device (multi-GPU same kind) dispatch. The plugin registration mechanism, C ABI design, and known production plugins (CUDA, Intel SYCL, Apple Metal, TPU) are thoroughly documented. Less coverage exists for dynamic/runtime plugin selection and true cross-device-kind dispatch within a single computation graph.

- Surprise findings: Two notable surprises:
  1. The `PJRT_Memory*` migration (away from `PJRT_Device*`) in buffer APIs suggests PJRT is evolving toward a memory-space-first dispatch model rather than device-first — closer to what libkdl targets.
  2. The heterogeneous mesh gap (Discussion #18372) is a confirmed missing capability in JAX/PJRT, meaning the libkdl use-case (same computation, different device targets at dispatch time) is explicitly *not* solved by PJRT today.

- Gaps: No sources found on:
  - Runtime plugin switching (loading multiple PJRT plugins simultaneously and dispatching to best-fit device at kernel granularity)
  - PJRT interaction with MLIR GPU dialect or IREE HAL (the "compiler side" of heterogeneous dispatch)
  - Cost model / profiling APIs within PJRT for dispatch decisions
  - Lazy/JIT plugin loading vs. eager registration

- Suggested follow-up angles:
  1. `PJRT_Memory` space abstraction and its relationship to NUMA/heterogeneous memory dispatch — search "PJRT memory spaces device buffer views 2024"
  2. Simultaneous multi-plugin JAX sessions — "JAX multiple PJRT plugins same process concurrent dispatch"
  3. PJRT Triton extension in `pjrt_c_api.h` — how Triton kernels bypass XLA compilation and inject directly into PJRT execution
  4. StableHLO → PJRT pipeline compared to MLIR GPU dialect → IREE HAL — structural parallels and divergences
  5. Apple Metal PJRT plugin as case study for non-CUDA heterogeneous dispatch
