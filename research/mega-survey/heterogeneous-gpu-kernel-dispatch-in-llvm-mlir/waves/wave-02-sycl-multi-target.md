# Wave 02: SYCL Multi-Target Compilation
Search query: "SYCL multi-target compilation ahead-of-time JIT SPIR-V CUDA HIP"
Sources found: 9
Date: 2026-04-06

## Sources

### 1. Experiences Building an MLIR-Based SYCL Compiler (CGO 2024)
- URL: https://dl.acm.org/doi/10.1109/CGO57630.2024.10444866
- ArXiv: https://arxiv.org/abs/2312.13170
- Type: paper
- Date: March 2024 (CGO 2024), preprint December 2023
- Relevance: 9/10
- Novelty: 9/10
- Summary: Codeplay/Intel collaboration presenting an MLIR dialect for SYCL that models host and device code simultaneously, enabling cross-boundary optimizations lost in LLVM-only pipelines. By nesting the device code module inside the host code module in MLIR IR, the compiler can reason about both sides together, yielding speedups of up to 4.3x on SYCL benchmark applications versus two LLVM-based implementations. This is the direct CGO 2024 paper cited as a priority source for this angle.
- Key detail: MLIR's IR nesting allows device code to be treated as a sub-module of its host invocation context; cross-host/device analysis unlocks optimizations impossible when the device module is compiled in isolation. Kernel dispatch parameters visible from the host call site can flow into device-side optimization.

### 2. AdaptiveCpp SSCP — Single-Source Single Compiler Pass Documentation
- URL: https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/
- Type: docs
- Date: 2023-2024 (continuously maintained)
- Relevance: 10/10
- Novelty: 8/10
- Summary: Defines the SSCP compilation model where a single compiler pass emits backend-independent LLVM IR for both host and device; the IR is embedded in the host binary and lowered to PTX, SPIR-V, or AMDGCN at runtime via AdaptiveCpp's llvm-to-backend infrastructure. Stage 1 (compile time) produces device-agnostic IR; Stage 2 (runtime) performs target-specific lowering. The SSCP model compile overhead is only ~15% above host-only compilation, while supporting 38+ AMD GPUs, all NVIDIA GPUs, and all Intel GPUs from a single binary.
- Key detail: SSCP is the only known implementation of the SYCL 2020 spec's single-pass model. The llvm-to-backend layer is a pluggable IR lowering infrastructure — conceptually identical to what a kernel dynamic linker would need at runtime dispatch time.

### 3. Adaptivity in AdaptiveCpp: Optimizing Performance by Leveraging Runtime Information During JIT-Compilation (IWOCL 2025)
- URL: https://dl.acm.org/doi/10.1145/3731125.3731127
- Slides: https://www.iwocl.org/wp-content/uploads/iwocl-2025-aksel-alpay-adaptivity-in-adaptivecpp.pdf
- Type: paper
- Date: April 2025 (IWOCL/SYCLcon 2025)
- Relevance: 10/10
- Novelty: 10/10
- Summary: Presents a runtime adaptivity framework in AdaptiveCpp that specializes JIT-compiled kernels based on runtime information: work-group sizes, pointer alignment of kernel arguments, and argument values themselves. On NVIDIA hardware it outperforms CUDA by 30% geometric mean; on AMD and Intel it outperforms HIP and oneAPI by 44% and 23% respectively. Extreme cases show >5x gains over non-specialized code.
- Key detail: Introduces persistent on-disk JIT cache to amortize re-compilation cost. Work-group size specialization and argument-value specialization are the highest-impact techniques on AMD/Intel. This is the most direct evidence that runtime dispatch with JIT specialization beats static AOT across all major GPU vendors.

### 4. AdaptiveCpp Compilation Model Documentation
- URL: https://adaptivecpp.github.io/AdaptiveCpp/compilation/
- GitHub: https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/compilation.md
- Type: docs
- Date: 2024-2025 (continuously maintained)
- Relevance: 9/10
- Novelty: 7/10
- Summary: Describes all AdaptiveCpp compilation flows including SMCP (legacy, per-backend separate pass) and SSCP (new unified pass), together with the generic LLVM IR embedding and runtime lowering pipeline. Covers how a single binary selects the correct backend path and dispatches to OpenCL, CUDA, HIP, or Level Zero at runtime based on discovered hardware.
- Key detail: The `acpp-targets` compiler flag accepts comma-separated backend specifiers (e.g., `cuda:sm_80,hip:gfx906,generic`). The `generic` target enables SSCP mode where hardware is selected at runtime, not compile time — a direct analog to fat binary + runtime dispatch.

### 5. oneAPI DPC++ Compiler and Runtime Architecture Design
- URL: https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html
- GitHub: https://github.com/intel/llvm/blob/sycl/sycl/doc/design/CompilerAndRuntimeDesign.md
- Type: docs
- Date: 2024-2025 (living document)
- Relevance: 9/10
- Novelty: 6/10
- Summary: Canonical reference for DPC++ fat binary construction: Clang driver invokes one device compiler per target triple (spir64, spir64_x86_64, spir64_gen, nvptx64-nvidia-cuda, amdgcn-amd-amdhsa), generates per-target SPIR-V or native binary, wraps with clang-offload-wrapper, and links all device images into a single fat executable. At runtime, the SYCL RT selects the embedded image matching the discovered device and JIT-finalizes if needed (SPIR-V -> native).
- Key detail: The `-fsycl-targets` flag accepts comma-separated targets enabling a single binary with AOT images for multiple architectures plus a SPIR-V fallback. This is DPC++'s fat binary mechanism — each embedded image is essentially a versioned kernel for a specific microarchitecture.

### 6. Intel oneAPI Ahead-of-Time Compilation for GPU (Programming Guide 2024)
- URL: https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-1/ahead-of-time-compilation-for-gpu.html
- Type: docs
- Date: 2024
- Relevance: 7/10
- Novelty: 5/10
- Summary: Step-by-step guide for generating AOT binaries for Intel GPU with `-fsycl -fsycl-targets=spir64_gen -Xs "-device <pci-id>"`. Explains that AOT avoids JIT overhead at deployment, how to mix AOT and JIT targets in the same binary, and how the runtime selects the correct embedded image. Also covers CPU AOT (`spir64_x86_64`) in a companion page.
- Key detail: AOT and JIT targets can coexist in a single fat binary: `-fsycl-targets=spir64_gen,spir64` embeds a pre-compiled Intel GPU image plus a generic SPIR-V fallback for any other OpenCL-capable device. This is Intel's production mechanism for heterogeneous deployment.

### 7. A Survey of Recent Developments in SYCL Compiler Implementations (arXiv 2602.21113)
- URL: https://arxiv.org/abs/2602.21113
- Type: paper
- Date: February 2026
- Relevance: 8/10
- Novelty: 7/10
- Summary: Comprehensive survey covering the transition from SMCP to SSCP compilation models across OpenSYCL/AdaptiveCpp, DPC++, and ComputeCpp, with a section on the MLIR-based SYCL compiler from CGO 2024. Analyzes performance tradeoffs, portability claims, and the maturity of each approach. Synthesizes 2023-2025 literature into a single reference.
- Key detail: Confirms that as of 2026, AdaptiveCpp SSCP is the only production SYCL compiler implementing the SYCL 2020 single-pass spec; DPC++ remains SMCP with SPIR-V as the portability layer; the MLIR approach (CGO 2024) is still research-stage but shows the most optimization potential.

### 8. SYCL Runtime Compilation via kernel_compiler Extension (DPC++ / Codeplay, 2025)
- URL: https://codeplay.com/portal/blogs/2025/07/08/sycl-runtime-compilation
- Intel how-to: https://www.intel.com/content/www/us/en/developer/articles/technical/oneapi-compiler-kernel-compiler-extension.html
- Type: blog + docs
- Date: July 2025
- Relevance: 8/10
- Novelty: 8/10
- Summary: Documents the `sycl_ext_oneapi_kernel_compiler` extension shipping in DPC++ and oneAPI 2025.2, which lets applications compile SYCL source strings at runtime and dispatch the resulting kernels immediately. Enables C++ metaprogramming-driven kernel specialization at the application layer without rebuilding the binary. Currently targets Intel GPUs (Level Zero / OpenCL) and CPU (OpenCL).
- Key detail: This extension closes the gap between static fat binary dispatch and fully dynamic kernel generation. An application can emit a specialized kernel string at runtime, compile it via the extension, and dispatch it — a user-space kernel dynamic linker pattern built on top of SYCL's existing queue/event model.

### 9. AdaptiveCpp Stdpar: C++ Standard Parallelism Integrated Into a SYCL Compiler (IWOCL 2024)
- URL: https://dl.acm.org/doi/fullHtml/10.1145/3648115.3648117
- Type: paper
- Date: April 2024 (IWOCL 2024)
- Relevance: 7/10
- Novelty: 7/10
- Summary: Demonstrates how AdaptiveCpp extends its SSCP pipeline to support C++ standard parallelism (`std::for_each`, `std::transform_reduce`, etc.) as GPU offload without any SYCL syntax, targeting NVIDIA, AMD, and Intel GPUs from a single binary. Uses the same LLVM IR embedding and runtime lowering path as SYCL kernels, validating that the SSCP architecture generalizes beyond SYCL to other programming models.
- Key detail: First stdpar implementation to demonstrate performance portability across all three major GPU vendors from a single compilation. Validates SSCP as a general-purpose runtime dispatch foundation, not SYCL-specific.

## Angle Assessment

- Coverage: Well-explored for the SYCL ecosystem. DPC++ fat binary mechanics, AdaptiveCpp SSCP, and MLIR-based SYCL compilation are all well-documented with production implementations and 2024-2025 papers. The CGO 2024 paper (priority source) is found and fully characterized.

- Surprise findings:
  1. AdaptiveCpp IWOCL 2025 paper shows SYCL with runtime JIT specialization beating CUDA by 30% geometric mean — a strong counter-narrative to "CUDA is always fastest."
  2. The `kernel_compiler` extension in oneAPI 2025.2 enables user-space kernel compilation at runtime within a SYCL program — this is essentially a SYCL-layer kernel dynamic linker (precisely the libkdl use case from user space).
  3. AdaptiveCpp's SSCP compile overhead is only 15% above host-only compilation — far lower than the multi-pass SMCP model, making the single-binary multi-target approach practically viable.

- Gaps:
  1. No found paper specifically benchmarks cross-vendor fat binary dispatch latency (selecting Intel image vs CUDA image vs SPIR-V fallback at process startup) — the libkdl proposal fills this gap.
  2. Limited coverage of how DPC++ fat binary device image selection interacts with NUMA/multi-socket heterogeneous nodes (multiple GPUs of different vendors in the same machine).
  3. FPGA AOT path in DPC++ is documented separately and not well-integrated into the multi-target narrative.
  4. No found work on dynamic loading of pre-compiled SYCL device images from disk at runtime (dlopen-style), which is the core libkdl contribution.

- Suggested follow-up angles:
  1. `sycl-kernel-compiler-extension` — deep dive on `sycl_ext_oneapi_kernel_compiler` and its interaction with Level Zero/OpenCL JIT pipeline; benchmark compilation latency.
  2. `adaptivecpp-llvm-to-backend` — examine AdaptiveCpp's llvm-to-backend infrastructure source code as a reference implementation of the IR-to-native lowering step that libkdl needs.
  3. `fat-binary-device-image-selection-latency` — measure time from process start to first kernel dispatch across DPC++ fat binary, AdaptiveCpp SSCP, and libkdl to quantify dispatch overhead differences.
  4. `sycl-spirv-opencl-level-zero-dispatch-path` — trace how DPC++ selects between Level Zero and OpenCL backends at runtime and the cost of SPIR-V online compilation in each path.
  5. `mlir-sycl-cgo2024-artifact` — examine the Zenodo software artifact (https://zenodo.org/records/10410758) to understand what optimizations the MLIR-based SYCL compiler implements that DPC++ cannot.
