# TMVA-SOFIE GPU Support: GSoC 2025 and ACAT 2025 Status

*Research compiled 2026-04-06 for "libkdl: Kernel Dynamic Linker" poster, LLVM Dublin 2026.*

**Relevance to libkdl:** 7/10 — SOFIE is the primary HEP-domain ML inference engine. Its approach (compile-time code generation with static backend selection) is a concrete example of what libkdl would replace or augment with runtime dispatch. The 2025 ALPAKA integration is directly in the same problem space.

---

## 1. What SOFIE Is

SOFIE (System for Optimized Fast Inference code Emit) is the ML inference engine embedded in ROOT's TMVA toolkit. It is the de facto standard for deploying trained neural networks inside HEP analysis and trigger software.

**Core pipeline:**
```
ONNX / Keras / PyTorch model
  -> RModelParser_{ONNX,Keras,PyTorch}
  -> SOFIE Internal Representation (RModel)
  -> Code Generator (per-operator ROperator_<Name> emitters)
  -> .hxx header file + optional .dat weights file
  -> Compile with BLAS / cuBLAS / rocBLAS -> inference binary
```

The generated code is self-contained C++ with no runtime dependency on ML frameworks. This is the key property for HEP: physicists `#include "MyModel.hxx"` in analysis code and get inference with zero framework overhead.

**Source:** ROOT SOFIE README, https://github.com/root-project/root/blob/master/tmva/sofie/README.md

---

## 2. ONNX Operator Coverage

As of 2025, SOFIE supports 80+ ONNX operators:
- Dense: MatMul, Gemm, Add, BatchNormalization, LayerNorm
- Convolutions: Conv (1D/2D/3D), ConvTranspose
- Recurrent: LSTM, GRU, RNN
- Activations: Relu, Sigmoid, Tanh, LeakyRelu, ELU, Selu, Softmax, Gelu
- Pooling: MaxPool, AveragePool, GlobalAveragePool
- Shape: Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Gather, Concat, Slice
- Attention: MultiHeadAttention (transformer blocks)
- Supported data types: float32, float64, int32, int64, bool

CPU coverage is mature. GPU coverage via ALPAKA targets BLAS-heavy operators first (MatMul/Gemm via cuBLAS/rocBLAS). Full convolution and attention coverage on GPU backends likely lags.

**Source:** SOFIE README; ACAT 2025 abstract (Lupi, Sengupta, Moneta).

---

## 3. GPU Support: Architecture and Current Status

### 3.1 Timeline

| Date | Milestone |
|------|-----------|
| 2021 | SOFIE CPU-only launch, ONNX parser, first CHEP paper |
| 2022 | GNN support, recurrent ops, ROOT Workshop presentation |
| 2023 | SYCL backend (Panagou, Moneta, Sengupta) - Intel GPU inference |
| 2024 | ACAT 2024: SYCL GPU inference, 258x speedup claimed on Intel GPU for large conv models |
| 2025 (GSoC) | "TMVA SOFIE - GPU Support for Machine Learning Inference" (350 hours, medium difficulty) |
| 2025 (Sep) | ACAT 2025: ALPAKA + SYCL backends, cuBLAS + rocBLAS, graph optimizations (Lupi, Sengupta, Moneta) |

### 3.2 Current GPU Architecture (post-ACAT 2025)

Per ACAT 2025 presentation (Enrico Lupi, Sanjiban Sengupta, Lorenzo Moneta - CERN/INFN Padova):

**SYCL path (Intel GPUs, also NVIDIA/AMD via Codeplay):**
- Code generator emits SYCL kernel calls within generated .hxx
- Matrix operations dispatch to oneMKL (Intel MKL SYCL interface)
- First production GPU path; validated on Intel Data Center GPU Max

**ALPAKA path (NVIDIA via cuBLAS, AMD via rocBLAS):**
- Code generator emits ALPAKA-wrapped kernel calls
- MatMul/Gemm operators dispatch to cuBLAS (NVIDIA) or rocBLAS (AMD) through ALPAKA backends
- Backend selection at build time via CMake/compile flag - not runtime
- Validated on: ParticleNet, ATLAS GNNs, Smart Pixels models

### 3.3 GSoC 2025 Project Scope

The official HSF GSoC 2025 proposal "TMVA SOFIE - GPU Support for Machine Learning Inference":
- **Mentors:** Lorenzo Moneta (primary), Sanjiban Sengupta (secondary) - both CERN
- **Duration:** 350 hours (large project)
- **Difficulty:** Medium
- **Goal:** Extend SOFIE with GPU inference via CUDA, ROCm, and ALPAKA stacks
- **Reference implementation:** SYCL backend from 2023 serves as design template
- **Deliverables:** GPU stack evaluation, implementation, optional performance benchmarking

The GSoC framing ("explore different GPU stacks") indicates the ALPAKA GPU path is maturing rather than production-hardened as of the proposal date.

**Source:** https://hepsoftwarefoundation.org/gsoc/2025/proposal_TMVA-SOFIE-GPU.html

---

## 4. Code Generation: How ONNX Maps to GPU Calls

### The Generation Pipeline

For each ONNX operator, SOFIE has a corresponding `ROperator_<Name>` class that emits C++ code. For GPU backends, code generation is extended to emit GPU API calls:

**CPU (BLAS) path - generated code example:**
```cpp
TMVA::Experimental::SOFIE::BLAS::sgemm_(
    &transA, &transB, &M, &N, &K,
    &alpha, fData_input, &K, fData_weights, &N,
    &beta, fData_output, &N);
```

**ALPAKA/cuBLAS path (post-2025) - generated code:**
```cpp
// ALPAKA executor wraps cuBLAS dispatch:
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha, d_input, K, d_weights, N,
    &beta, d_output, N);
```

The operator dispatch (CPU vs GPU) is resolved at code generation time, not at inference runtime. SOFIE generates a different .hxx file per requested backend. No runtime branching in generated code.

### Memory Management

The ACAT 2025 paper describes "Structure-of-Arrays-based memory allocation with reuse":
- Operator intermediate tensors reuse memory buffers when lifetimes do not overlap
- GPU memory explicitly managed: host buffers with cudaMalloc/hipMalloc equivalent calls
- H->D transfer at inference start, D->H at output - explicit in generated code
- No automatic unified memory

### Graph Optimizations (ACAT 2025)

- **Operator fusion:** Adjacent element-wise operators merged into single kernel
- **Memory reuse:** Dead tensor buffers reclaimed for later operators
- **Kernel-level optimizations:** Loop unrolling, vectorization hints
- **SoA memory layout:** Better cache behavior for batch inference

---

## 5. Performance Data

### SYCL GPU Backend (Intel GPU, ACAT 2024)
- Up to **258x speedup** over plain C++ on Intel Data Center GPU Max 1100 for large conv models
- Note: 258x is best-case for large models; smaller models have lower GPU speedup due to H->D transfer overhead

### CPU SOFIE vs ONNX Runtime vs LWTNN (CHEP 2024)
- Benchmark: 5-layer FC network (200 units/layer) x 5M events, single-threaded
- SOFIE competitive with ONNX Runtime CPU; significantly faster than LWTNN
- No published ALPAKA-GPU vs TensorRT or ONNX Runtime GPU head-to-head

### Models Validated on GPU Backends
- **ParticleNet** (jet tagging GNN)
- **ATLAS GNNs** (track-to-vertex association)
- **Smart Pixels models** (compressed ML for Level-1 trigger)
- All validated for physics correctness (output distributions match CPU reference)

---

## 6. Limitations Relevant to libkdl

### Static Backend Selection
SOFIE's GPU path is compile-time-selected. The generated .hxx contains calls to either cuBLAS or rocBLAS - not both. To switch between NVIDIA and AMD, you regenerate the header with a different build configuration. No runtime dispatch to available hardware.

### No Cross-Model Optimization
Operator fusion is implemented within the generator but there is no intermediate representation enabling cross-operator optimization passes. The generator is template-driven C++ emission, not compiler IR manipulation. Compare MLIR's linalg -> gpu lowering with full optimization pass pipeline.

### ALPAKA Coverage Gap
The cuBLAS/rocBLAS integration targets GEMM-heavy operators first. Convolution, activation, and attention operators may still fall back to CPU implementations in the ALPAKA path as of 2025.

### No JIT or Runtime Specialization
SOFIE is purely ahead-of-time. Generated code cannot adapt to specific GPU model at runtime (choosing tile sizes for L4 vs A100). Hardware-specific tuning must be baked into the code generator at development time.

---

## 7. Relevance to libkdl

### The Gap libkdl Fills

SOFIE+ALPAKA demonstrates the HEP community's approach to heterogeneous ML inference: AOT code generation, static backend selection, explicit memory management. This works when target hardware is known at build time and per-platform compilation is acceptable.

libkdl addresses the complementary case:
- Hardware unknown until deployment (WLCG grid nodes, cloud burst compute, HL-LHC trigger farms)
- Single artifact must run on NVIDIA, AMD, or CPU without recompilation
- Runtime hardware fingerprinting to select optimal pre-compiled variant

### Argument for Poster

> "SOFIE+ALPAKA is the HEP state-of-the-art for ML inference portability: compile-time code generation targeting cuBLAS or rocBLAS, validated on ParticleNet and ATLAS GNNs, integrated into the HL-LHC trigger strategy. libkdl complements this by providing the runtime dispatch layer that SOFIE's AOT approach cannot: a single kernel artifact that discovers available hardware at load time and selects the optimal implementation variant without recompilation. The two approaches are complementary, not competing."

---

## 8. Key References

1. Lupi, E., Sengupta, S., Moneta, L. "TMVA SOFIE: Enhancements in ML Inference through graph optimizations and heterogeneous architectures." ACAT 2025. https://indico.cern.ch/event/1488410/contributions/6561436/
2. HSF GSoC 2025. "TMVA SOFIE - GPU Support for Machine Learning Inference." https://hepsoftwarefoundation.org/gsoc/2025/proposal_TMVA-SOFIE-GPU.html
3. Panagou, I.-M., Moneta, L., Sengupta, S. "Inference of ML models on Intel GPUs with SYCL and Intel OneAPI using SOFIE." Zenodo (2023). https://zenodo.org/records/8385777
4. Moneta, L. et al. "Accelerating ML Inference on GPUs with SYCL using SOFIE." ACAT 2024. https://indico.cern.ch/event/1330797/contributions/5796633/
5. Moneta, L. et al. "C++ Code Generation for Fast Inference of Deep Learning Models in ROOT/TMVA." CHEP 2021. https://www.epj-conferences.org/articles/epjconf/ref/2021/05/epjconf_chep2021_03040/epjconf_chep2021_03040.html
6. ROOT Project. SOFIE README. https://github.com/root-project/root/blob/master/tmva/sofie/README.md
7. HSF GSoC 2023. "ROOT - TMVA SOFIE Developments." https://hepsoftwarefoundation.org/gsoc/2023/proposal_ROOT-TMVA-SOFIE.html
