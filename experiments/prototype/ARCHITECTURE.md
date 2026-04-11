# mlir-hetero-dispatch: Architecture Document

**System:** Lightweight vendor-agnostic runtime dispatch for MLIR-compiled GPU kernels
**Author:** Akash | **Date:** 2026-04-02
**Target:** LLVM Developers' Meeting, Dublin 2026 (poster)

---

## 1. System Overview

mlir-hetero-dispatch bridges the gap between MLIR's existing multi-target compilation
infrastructure (`gpu-module-to-binary`) and actual runtime hardware dispatch. It consists
of two components:

1. **Build-time pipeline:** MLIR passes that compile a single kernel to multiple targets
   and emit a routing table alongside the bundled binary.
2. **libkdl (Kernel Dynamic Linker):** A ~500 LOC C library that discovers devices,
   matches capability contracts, ranks targets via a roofline cost model, and dispatches
   the best kernel variant.

The name "Kernel Dynamic Linker" is an analogy to `ld.so`: just as the dynamic linker
resolves shared library symbols at runtime based on the system's library search path,
libkdl resolves kernel symbols at runtime based on available GPU hardware.

---

## 2. System Diagram

```
BUILD TIME                                          RUNTIME
==========                                          =======

 linalg.matmul (or any linalg op)
       |
       v
 one-shot-bufferize
       |
       v
 convert-linalg-to-loops
       |
       v
 convert-affine-for-to-gpu
       |
       v
 gpu-kernel-outlining
       |
       v
 gpu.module @kernels {                       +---------------------------+
   gpu.func @matmul(...) kernel              |  1. kdl_init()            |
 }                                           |     |                    |
       |                                     |     v                    |
       +--- nvvm-attach-target               |  kdl_discover_devices()  |
       |      chip=sm_80                     |  -> [A100, MI300X, CPU]  |
       |                                     |     |                    |
       +--- rocdl-attach-target              |     v                    |
       |      chip=gfx942                    |  2. kdl_load_bundle()    |
       |                                     |     parse MTB header     |
       +--- (cpu target via separate         |     extract routing tbl  |
       |      lowering path)                 |     |                    |
       |                                     |     v                    |
       v                                     |  3. kdl_select_kernel()  |
 gpu-module-to-binary                        |     for each variant:    |
       |                                     |       match contract     |
       v                                     |       compute cost       |
 gpu.binary @kernels [                       |     sort by cost         |
   #gpu.object<#nvvm, cubin>,                |     -> best = sm_80      |
   #gpu.object<#rocdl, hsaco>                |     |                    |
 ]                                           |     v                    |
       |                                     |  4. kdl_launch()         |
       v                                     |     mgpuModuleLoad()     |
 kdl-routing-table-gen (custom pass)         |     mgpuLaunchKernel()   |
       |                                     |     |                    |
       v                                     |     v                    |
 +==================================+        |  5. kdl_finalize()       |
 | Multi-Target Bundle (MTB file)   |        |     unload modules       |
 |  +----------------------------+  |        +---------------------------+
 |  | Header (magic, version)    |  |
 |  +----------------------------+  |
 |  | Routing Table              |  |
 |  |  kernel -> [variants]      |  |
 |  |  variant -> contract+off   |  |
 |  +----------------------------+  |
 |  | Binary Blobs               |  |
 |  |  [cubin] [hsaco] [x86 .o]  |  |
 |  +----------------------------+  |
 +==================================+
```

---

## 3. Build-Time Pipeline

### 3.1 Exact MLIR Pass Sequence

**Phase 0: High-level to GPU (user runs before our pipeline)**
```
one-shot-bufferize
convert-linalg-to-affine-loops     (or convert-linalg-to-loops)
convert-affine-for-to-gpu          (or convert-parallel-loops-to-gpu)
```

**Phase 1: GPU outlining and multi-target attachment**
```
gpu-kernel-outlining
nvvm-attach-target="chip=sm_80 features=+ptx80 opt-level=3"
nvvm-attach-target="chip=sm_90 features=+ptx80 opt-level=3"
rocdl-attach-target="chip=gfx90a"
rocdl-attach-target="chip=gfx942"
```

**Phase 2: Serialization to multi-target binary**
```
gpu-module-to-binary
```

This produces a `gpu.binary` containing one `gpu.object` per target. MLIR internally
clones the `gpu.module` for each target and runs the appropriate lowering pipeline
(convert-gpu-to-nvvm or convert-gpu-to-rocdl) before serialization.

**Phase 3: CPU fallback (separate pipeline, run on a clone)**
```
convert-linalg-to-loops
convert-scf-to-cf
convert-func-to-llvm
convert-arith-to-llvm
convert-math-to-llvm
expand-strided-metadata
finalize-memref-to-llvm
reconcile-unrealized-casts
mlir-translate --mlir-to-llvmir
llc -march=x86-64 -mcpu=native -filetype=obj
```

**Phase 4: Bundle generation (custom tool: `kdl-bundle`)**
```
kdl-bundle \
  --kernel=matmul \
  --nvptx=matmul.sm_80.cubin  --contract-nvptx='cuda>=11.0,sm>=80,smem>=48K' \
  --nvptx=matmul.sm_90.cubin  --contract-nvptx='cuda>=12.0,sm>=90,smem>=228K' \
  --amdgcn=matmul.gfx90a.hsaco --contract-amdgcn='hip>=5.0,gfx>=90a' \
  --amdgcn=matmul.gfx942.hsaco --contract-amdgcn='hip>=6.0,gfx>=942' \
  --x86=matmul.x86.o          --contract-x86='avx2' \
  -o matmul.mtb
```

---

## 4. Multi-Target Bundle Format (MTB)

### 4.1 Binary Layout

```
Offset  Size       Field
------  ---------  -----------------------------------------------
0       8          Magic: "KDL_MTB\0"
8       4          Version: uint32_t (1)
12      4          num_kernels: uint32_t
16      4          num_variants: uint32_t (total across all kernels)
20      4          string_table_offset: uint32_t
24      4          binary_data_offset: uint32_t
28      4          reserved: uint32_t

--- Kernel Table (starts at offset 32) ---
For each kernel (12 bytes each):
  +0    4          name_offset: uint32_t (into string table)
  +4    4          first_variant_idx: uint32_t
  +8    4          num_variants: uint32_t

--- Variant Table (starts after kernel table) ---
For each variant (40 bytes each):
  +0    4          target_kind: uint32_t (0=NVPTX, 1=AMDGCN, 2=SPIRV, 3=X86_64)
  +4    4          target_chip_offset: uint32_t (into string table, e.g. "sm_80")
  +8    4          contract_offset: uint32_t (into string table, JSON)
  +12   4          priority: uint32_t (lower = preferred, for tie-breaking)
  +16   8          binary_offset: uint64_t (into binary data section)
  +24   8          binary_size: uint64_t
  +32   4          entry_point_offset: uint32_t (into string table)
  +36   4          reserved: uint32_t

--- String Table ---
NUL-terminated UTF-8 strings, concatenated

--- Binary Data ---
Concatenated device binaries (cubin, hsaco, SPIR-V, x86 ELF)
```

### 4.2 Capability Contract Format (JSON in string table)

```json
{
  "target": "nvptx",
  "min_driver": "cuda>=11.0",
  "min_arch": "sm_80",
  "min_shared_mem_kb": 48,
  "requires_tensor_cores": false,
  "min_vram_mb": 0,
  "compute_profile": {
    "flops": 2e9,
    "bytes_read": 8e6,
    "bytes_written": 4e6,
    "arithmetic_intensity": 250.0
  }
}
```

---

## 5. libkdl Runtime Library (~500 LOC C)

### 5.1 Public API

```c
/* kdl.h — Kernel Dynamic Linker public API */

typedef struct kdl_context*    kdl_ctx;
typedef struct kdl_bundle*     kdl_bundle_t;
typedef struct kdl_kernel*     kdl_kernel_t;

typedef enum {
    KDL_SUCCESS = 0,
    KDL_ERROR_NO_DEVICES,
    KDL_ERROR_NO_MATCHING_VARIANT,
    KDL_ERROR_LOAD_FAILED,
    KDL_ERROR_LAUNCH_FAILED,
    KDL_ERROR_INVALID_BUNDLE,
} kdl_status;

typedef struct {
    uint32_t    vendor;          /* 0=NVIDIA, 1=AMD, 2=INTEL, 3=CPU */
    char        name[256];
    char        arch[64];        /* "sm_80", "gfx90a", "x86-64-v3" */
    uint64_t    vram_bytes;
    uint32_t    compute_units;   /* SMs / CUs / cores */
    uint32_t    max_shared_mem;  /* bytes per block */
    uint32_t    warp_size;       /* 32/64/1 */
    double      peak_tflops_f32;
    double      peak_bw_gbps;
    uint32_t    driver_version;
    int         device_index;
} kdl_device_info;

/* Lifecycle */
kdl_status kdl_init(kdl_ctx* out_ctx);
void       kdl_finalize(kdl_ctx ctx);

/* Device Discovery */
int        kdl_get_device_count(kdl_ctx ctx);
kdl_status kdl_get_device_info(kdl_ctx ctx, int index, kdl_device_info* out);

/* Bundle Loading */
kdl_status kdl_load_bundle(kdl_ctx ctx, const char* path, kdl_bundle_t* out);
void       kdl_free_bundle(kdl_bundle_t bundle);

/* Kernel Selection (device_index=-1 for auto-select) */
kdl_status kdl_select_kernel(kdl_ctx ctx, kdl_bundle_t bundle,
                              const char* kernel_name,
                              int device_index, kdl_kernel_t* out);

/* Kernel Launch */
kdl_status kdl_launch(kdl_kernel_t kernel,
                       uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                       uint32_t block_x, uint32_t block_y, uint32_t block_z,
                       uint32_t shared_mem_bytes, void** args);
kdl_status kdl_sync(kdl_kernel_t kernel);

/* Memory Management */
kdl_status kdl_malloc(kdl_kernel_t kernel, size_t bytes, void** out_ptr);
kdl_status kdl_free_mem(kdl_kernel_t kernel, void* ptr);
kdl_status kdl_memcpy_h2d(kdl_kernel_t kernel, void* dst, const void* src, size_t bytes);
kdl_status kdl_memcpy_d2h(kdl_kernel_t kernel, void* dst, const void* src, size_t bytes);
```

### 5.2 Device Discovery (dlopen-based, no link-time deps)

```c
kdl_status kdl_discover_devices(kdl_ctx ctx) {
    ctx->num_devices = 0;

    /* NVIDIA via libcuda.so.1 */
    void* libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (libcuda) {
        /* cuInit, cuDeviceGetCount, cuDeviceGet*
         * Populate: vendor=NVIDIA, arch="sm_XX", compute_units, warp_size=32 */
    }

    /* AMD via libamdhip64.so */
    void* libhip = dlopen("libamdhip64.so", RTLD_LAZY);
    if (libhip) {
        /* hipInit, hipGetDeviceCount, hipGetDeviceProperties
         * Populate: vendor=AMD, arch=gcnArchName, warp_size=64 (CDNA) */
    }

    /* CPU: always available */
    {
        /* sysconf(_SC_NPROCESSORS_ONLN), /proc/cpuinfo
         * arch="x86-64-v3" or "x86-64-v4", warp_size=1 */
    }

    return (ctx->num_devices > 0) ? KDL_SUCCESS : KDL_ERROR_NO_DEVICES;
}
```

### 5.3 Contract Matching

```c
static int kdl_contract_matches(const kdl_contract* c, const kdl_device_info* d) {
    if (c->target != vendor_to_target[d->vendor]) return 0;
    if (kdl_parse_arch(d->arch) < c->min_arch_numeric) return 0;
    if (d->max_shared_mem < c->min_shared_mem_kb * 1024) return 0;
    if (d->driver_version < c->min_driver_version) return 0;
    return 1;
}
```

### 5.4 Cost Model (Roofline-Based)

```c
static double kdl_estimate_cost(const kdl_contract* c, const kdl_device_info* d) {
    if (!c->has_compute_profile) return 1e9; /* fallback to priority */
    double peak_compute = d->peak_tflops_f32 * 1e12;
    double peak_bw = d->peak_bw_gbps * 1e9;
    double achieved = fmin(peak_compute, peak_bw * c->arithmetic_intensity);
    double time_s = c->flops / achieved;
    time_s += (d->vendor == KDL_VENDOR_CPU) ? 1e-6 : 20e-6; /* launch overhead */
    return time_s;
}
```

### 5.5 Kernel Selection (Complete Flow)

```
1. Check cache (hash of kernel_name + device_index) -> hit? return cached
2. Find kernel in routing table
3. For each (device x variant): match contract, estimate cost
4. Pick (device, variant) with lowest cost
5. Load binary: cuModuleLoadData / hipModuleLoadData / dlopen
6. Resolve entry point: cuModuleGetFunction / hipModuleGetFunction / dlsym
7. Create stream
8. Cache result
9. Return kernel handle
```

---

## 6. Overhead Analysis

```
Selection (cache miss):  ~200us  -- amortized to zero on subsequent calls
Dispatch indirection:    <10ns   -- function pointer through ctx->backends[]
CUDA kernel launch:      ~20us   -- irreducible floor
Typical ML kernel:       1-10ms  -- actual computation

Dispatch overhead relative to kernel launch: <0.05%
Dispatch overhead relative to ML kernel:     <0.001%
```

---

## 7. Integration Architecture Sketches

### 7.1 torch.compile Integration

```python
@torch._dynamo.register_backend
def kdl(gm: torch.fx.GraphModule, example_inputs):
    mlir_module = torch_mlir.compile(gm, example_inputs, output_type="linalg")
    bundle_path = kdl_compile_pipeline(mlir_module)  # multi-target MLIR -> MTB
    ctx = _kdl.kdl_init()
    bundle = _kdl.kdl_load_bundle(ctx, bundle_path)
    def dispatch(*args):
        kernel = _kdl.kdl_select_kernel(ctx, bundle, "forward", -1)
        _kdl.kdl_launch(kernel, ...)
        return outputs
    return dispatch
```

### 7.2 ONNX Runtime Execution Provider

```cpp
class KdlExecutionProvider : public IExecutionProvider {
    GetCapability() -> claim supported ops
    Compile()       -> onnx subgraph -> torch-mlir -> MLIR -> MTB
    Compute()       -> kdl_select_kernel() + kdl_launch()
};
```

### 7.3 Relationship to IREE

IREE and mlir-hetero-dispatch are **complementary**:
- IREE: best for end-to-end model deployment with full control (100K+ LOC)
- kdl: best for adding multi-target dispatch to existing MLIR pipelines (~500 LOC)

---

## 8. Novelty Differentiation

| Dimension | IREE | SYCL/AdaptiveCpp | ALPAKA | Proteus | **libkdl** |
|-----------|------|-------------------|--------|---------|------------|
| Multi-vendor | Yes | Yes | Yes (compile) | No | **Yes** |
| Runtime select | Partial | Yes | No | Yes | **Yes (cost)** |
| Standalone | No | No | Yes | No | **Yes** |
| MLIR-native | Deep | None | None | LLVM IR | **Upstream** |
| Prog model req | IREE API | SYCL C++ | ALPAKA C++ | C/C++ | **None** |
| Cost-model | No | No | N/A | No | **Roofline** |
| LOC | 100K+ | Heavy | Header | Medium | **~500** |

---

## 9. Implementation Plan (2-Day Sprint)

### Day 1 (~8 hours)
1. Write `matmul_kernel.mlir` in linalg dialect (1h)
2. Create `build.sh`: linalg -> gpu -> multi-target binary extraction (2h)
3. Write `kdl_bundle.py` tool to create MTB files (2h)
4. Implement `kdl_init` + `kdl_discover_devices` (CUDA path) (2h)
5. Test: create valid MTB from MLIR-compiled objects (1h)

### Day 2 (~8 hours)
1. Implement `kdl_load_bundle` + `kdl_select_kernel` (3h)
2. Implement `kdl_launch` + `kdl_sync` (CUDA dispatch via driver API) (1h)
3. Add CPU fallback backend (1h)
4. Benchmark: dispatch overhead + GEMM performance (2h)
5. Generate plots + poster assets (1h)

### Minimum Viable Demo
1. One MLIR kernel (matmul) compiled to sm_80 cubin + x86 object
2. `kdl-bundle` producing an MTB file
3. `libkdl` with CUDA + CPU backends, contract matching
4. One benchmark: kdl dispatch overhead vs native cuLaunchKernel
5. Show dispatch adds <0.05% overhead

---

## 10. File Structure

```
experiments/prototype/
  ARCHITECTURE.md          # this document
  src/
    kdl.h                  # public API header
    kdl.c                  # main implementation (~500 LOC)
  tools/
    kdl_bundle.py          # MTB bundle creator
    build_kernel.sh        # MLIR compilation pipeline
  kernels/
    matmul.mlir            # demo kernel
  benchmarks/
    bench_dispatch.c       # dispatch overhead measurement
    plot_results.py        # generate figures
  results/
    *.csv, *.png
```

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| gpu-module-to-binary fails for ROCDL | Use NVPTX + CPU only for demo |
| No GPU on test machine | CPU-only demo; mock CUDA for architecture validation |
| Cost model gives wrong ranking | Validate empirically; acknowledge as limitation |
| MTB format too rigid | Label as prototype; propose llvm-offload-binary for production |
| Dispatch overhead higher than expected | Pre-validated: <10ns function pointer vs 20us launch |
