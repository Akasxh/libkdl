# Direction 04: Empirical Dispatch Overhead Comparison Across Runtimes

**Composite Score: 8.25/10**
**Rank: 4 of 8**

---

## Title

**Measuring the Dispatch Tax: Head-to-Head Kernel Launch Latency Across CUDA, HIP, Level Zero, liboffload, and libkdl**

## One-Sentence Description

Fill the measurement gap identified across all six survey waves by providing the first published head-to-head dispatch latency comparison across vendor driver APIs, LLVM's liboffload abstraction, and libkdl's multi-version dispatch layer on modern hardware.

---

## Evidence

| Source | Wave File | Key Contribution |
|--------|-----------|-----------------|
| TaxBreak (arXiv:2603.12465) | wave-03-dispatch-overhead | Hardware floor: 4.5-5 us on H100/H200; MoE: 9,305 launches/token |
| Dynamic Kernel Substitution (arXiv:2601.00227) | wave-03-dispatch-overhead | O(1) table lookup: 1-2 us, <0.8% e2e |
| FlashInfer-Bench | wave-03-dynamic-kernel-substitution | Application-level hash table dispatch: 1-2 us |
| gpu_ext (arXiv:2512.12615) | wave-03-dynamic-kernel-substitution | eBPF trampolines: 3-14% overhead (NVIDIA) |
| NVBit | wave-03-dynamic-kernel-substitution | Full SASS instrumentation: 85-93% overhead |
| ROCR InterceptQueue | wave-03-dynamic-kernel-substitution | HSA packet-level substitution; overhead not published |
| HIP dispatch gap (Kokkos #3670) | wave-02-rocm-hip | 4-14x slower than CUDA (25-70 us vs 3-8 us) — stale data from 2020/ROCm 3.8 |
| ROCm 7.2 dispatch fixes | wave-06-rocm-code-objects | Dynamic code object loading bugs fixed; MultiKernelDispatch profiling added |
| Level Zero dispatch latency | wave-06-level-zero-runtime | No published microsecond measurements; "significantly reduced" claim unquantified |
| AQL userspace dispatch | wave-06-rocm-code-objects | Fully userspace: MMIO doorbell, no syscall after queue creation |
| Immediate Command Lists (L0) | wave-06-level-zero-runtime | L0 v2 adapter moves to immediate-only; eliminates explicit submit step |
| liboffload ol* API overhead | wave-04-liboffload-multiversion, wave-06-llvm-offload-new-driver | No published overhead vs direct driver API |

### Identified Measurement Gaps

| Gap | Last Published Data | Hardware | Need |
|-----|-------------------|----------|------|
| ROCm dispatch floor | Kokkos #3670, 2020, ROCm 3.8 | MI100 era | Modern MI300X / ROCm 7.x measurement |
| Level Zero vs CUDA/HIP | None published | Any Intel dGPU | First head-to-head |
| liboffload indirection cost | None published | Any | olCreateProgram + olGetSymbol + olLaunchKernel vs direct |
| ROCR InterceptQueue overhead | "estimated <1 us" | Any AMD | Actual measurement |
| libkdl dispatch overhead | Prototype exists, no published benchmark | GTX 1650 | Publication-quality measurement |

---

## Novelty Argument

Despite extensive individual measurements (TaxBreak covers CUDA on H100, Kokkos covers HIP on MI100 circa 2020, chipStar covers SPIR-V JIT latency), no published work provides a head-to-head comparison of dispatch latency across:

1. Direct vendor driver APIs (cuLaunchKernel, hipModuleLaunchKernel, zeCommandListAppendLaunchKernel)
2. LLVM abstraction layer (liboffload ol* API)
3. Multi-version dispatch layer (libkdl)

The Level Zero dispatch latency gap is particularly notable: Intel claims "significantly reduced overhead" with the L0 v2 adapter but provides no microsecond numbers.

---

## Feasibility Plan

### Available Hardware
- NVIDIA GTX 1650 (CUDA 12.x)
- CPU (OpenMP/LLVM offload host plugin)
- AMD: would need borrowed hardware (MI100/MI250X) or cloud access

### Benchmark Design
```c
// Warm-up: 1000 dispatches (discard)
// Measurement: 10000 dispatches, record per-dispatch latency
// Kernel: empty kernel (measures pure dispatch overhead) + SGEMM 1024x1024

// Layer 1: Direct driver API
cuModuleLoadData(&mod, cubin_blob);
cuModuleGetFunction(&func, mod, "kernel");
timer_start();
for (i = 0; i < N; i++) cuLaunchKernel(func, ...);
cudaDeviceSynchronize();
timer_stop();

// Layer 2: liboffload
olCreateProgram(device, cubin_blob, size, &prog);
olGetSymbol(prog, "kernel", &sym);
timer_start();
for (i = 0; i < N; i++) olLaunchKernel(sym, ...);
timer_stop();

// Layer 3: libkdl
kdl_bundle_t *b = kdl_load_bundle("kernel.kdl");
kdl_kernel_t *k = kdl_select_kernel(b, "kernel");
timer_start();
for (i = 0; i < N; i++) kdl_dispatch(k, ...);
timer_stop();
```

### Deliverables
- Microsecond-resolution dispatch latency table (5+ layers)
- Per-dispatch overhead histogram (characterize tail latency)
- Dispatch-to-compute ratio for SGEMM at various sizes (where does dispatch overhead become negligible?)

---

## Poster Potential

**Yes — fills a poster panel as the quantitative anchor.**

- Bar chart: dispatch latency across layers (direct, liboffload, libkdl)
- Overhead-to-compute ratio chart at various kernel sizes
- The three measurement gaps filled (Level Zero, liboffload, InterceptQueue) are citable contributions

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **7/10** | Individual vendor measurements exist; head-to-head cross-layer comparison on modern hardware is novel. |
| **Feasibility** | **9/10** | GTX 1650 available; CUDA benchmarks straightforward. AMD/Intel require hardware access. |
| **Evidence** | **9/10** | Multiple waves identify explicit measurement gaps. TaxBreak/FlashInfer provide the methodology template. |
| **Impact** | **8/10** | Quantitative claims are more compelling than architecture diagrams. Fills gaps the community has identified. |
| **Composite** | **8.25/10** | |

---

## Limitations

- [LIMITATION] GTX 1650 is consumer-grade; datacenter (A100/H100/MI300X) numbers would be more impactful but require cloud access
- [LIMITATION] Level Zero measurement requires Intel dGPU hardware not currently available
- [LIMITATION] AMD measurement requires MI-series GPU or cloud access
