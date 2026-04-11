# GPU Runtime Device Discovery APIs — Research Notes
**For:** LLVM Dublin 2026 Poster — Heterogeneous GPU Kernel Dispatch
**Date:** 2026-04-02
**Status:** Complete first pass

---

## Overview

This document exhaustively catalogs every major API surface for GPU hardware discovery at runtime. The goal: determine whether a unified, vendor-agnostic device description can be built on top of all these APIs to enable runtime dispatch decisions without recompilation.

---

## 1. CUDA — cudaGetDeviceProperties / cudaDeviceGetAttribute

### Discovery Flow

```
cudaGetDeviceCount(&n)                  // enumerate devices
cudaGetDeviceProperties(&prop, dev_i)   // bulk query, one struct per device
cudaDeviceGetAttribute(&val, attr, dev) // single attribute, cheaper
cudaDriverGetVersion(&ver)              // driver version integer (e.g. 12040)
cudaRuntimeGetVersion(&ver)             // runtime version integer
```

### cudaDeviceProp — Complete Field Inventory

**Identification**
| Field | Type | Notes |
|---|---|---|
| `name` | `char[256]` | Human-readable device name |
| `uuid` | `cudaUUID_t` | 16-byte globally unique ID (stable across reboots) |
| `luid` | `char[8]` | Locally unique ID (Windows + TCC only) |
| `luidDeviceNodeMask` | `uint` | Windows LUID node mask |
| `pciBusID` | `int` | PCI bus number |
| `pciDeviceID` | `int` | PCI device slot |
| `pciDomainID` | `int` | PCI domain |
| `gpuPciDeviceID` | `uint` | Combined 16-bit PCI vendor+device ID |
| `gpuPciSubsystemID` | `uint` | PCI subsystem vendor+device ID |

**Compute Capability & Architecture**
| Field | Type | Notes |
|---|---|---|
| `major` | `int` | Compute capability major (e.g., 8 for Ampere) |
| `minor` | `int` | Compute capability minor (e.g., 0 for A100) |
| `multiProcessorCount` | `int` | Number of SMs (streaming multiprocessors) |
| `warpSize` | `int` | Warp width in threads (always 32 on current NVIDIA) |
| `maxThreadsPerBlock` | `int` | Max threads per block (1024 on modern GPUs) |
| `maxThreadsDim[3]` | `int[3]` | Max block dimensions (x, y, z) |
| `maxGridSize[3]` | `int[3]` | Max grid dimensions |
| `maxBlocksPerMultiProcessor` | `int` | Max resident blocks per SM |
| `maxThreadsPerMultiProcessor` | `int` | Max resident threads per SM |
| `cooperativeLaunch` | `int` | Grid-sync cooperative kernels supported |
| `computePreemptionSupported` | `int` | Compute preemption support |
| `concurrentKernels` | `int` | Multiple kernels can overlap |

**Memory**
| Field | Type | Notes |
|---|---|---|
| `totalGlobalMem` | `size_t` | Total VRAM in bytes |
| `sharedMemPerBlock` | `size_t` | Shared memory per block (default) |
| `sharedMemPerBlockOptin` | `size_t` | Max shared memory per block with opt-in |
| `sharedMemPerMultiprocessor` | `size_t` | Total shared memory per SM |
| `reservedSharedMemPerBlock` | `size_t` | Driver-reserved shared memory per block |
| `totalConstMem` | `size_t` | Constant memory (64 KB typically) |
| `regsPerBlock` | `int` | 32-bit registers per block |
| `regsPerMultiprocessor` | `int` | 32-bit registers per SM |
| `l2CacheSize` | `int` | L2 cache in bytes |
| `persistingL2CacheMaxSize` | `int` | Max L2 persisting capacity |
| `memoryBusWidth` | `int` | Global memory bus width in bits |
| `deviceNumaId` | `int` | NUMA node of GPU memory (-1 if unsupported) |
| `hostNumaId` | `int` | Nearest host NUMA node |

**Advanced / Policy**
| Field | Type | Notes |
|---|---|---|
| `ECCEnabled` | `int` | ECC enabled flag |
| `integrated` | `int` | iGPU (1) vs discrete (0) |
| `unifiedAddressing` | `int` | Unified virtual address space |
| `managedMemory` | `int` | cudaMallocManaged support |
| `asyncEngineCount` | `int` | Number of copy engines |
| `memoryPoolsSupported` | `int` | cudaMallocAsync support |
| `gpuDirectRDMASupported` | `int` | GPUDirect RDMA support |
| `ipcEventSupported` | `int` | IPC event support |
| `clusterLaunch` | `int` | Thread block cluster (Hopper+) |
| `isMultiGpuBoard` | `int` | Multi-GPU board flag |
| `mpsEnabled` | `int` | MPS sharing active |

**Note:** Structure has ~80+ fields total plus `reserved[56]` for future expansion.

### cudaDeviceGetAttribute — Key Queryable Attributes

`cudaDeviceGetAttribute` returns a single `int` for a given `cudaDeviceAttr` enum. All fields in `cudaDeviceProp` are queryable this way, plus:
- `cudaDevAttrGPUDirectRDMAWritesOrdering` — RDMA write ordering scope
- `cudaDevAttrGPUDirectRDMAFlushWritesOptions` — flush write API support

**Latency characteristics:**
- First CUDA API call per process: 1–2 seconds (driver context initialization; unavoidable cold start)
- Subsequent `cudaGetDeviceProperties` calls: ~10 µs per call (host-side struct query, cached by driver)
- `cudaDeviceGetAttribute` (single field): ~1–5 µs (lighter weight)
- All property queries are **pure CPU operations** — no GPU round-trip after initial init
- Best practice: query once at startup, cache in application-level structs

### Version Queries
```c
int driverVer, runtimeVer;
cudaDriverGetVersion(&driverVer);    // e.g., 12040 = CUDA 12.4
cudaRuntimeGetVersion(&runtimeVer);  // runtime version
```

---

## 2. HIP — hipGetDeviceProperties / hipDeviceGetAttribute

HIP intentionally mirrors the CUDA API surface for portability. The structs and enums are nearly 1:1 with CUDA equivalents, with AMD-specific extensions.

### Discovery Flow

```
hipGetDeviceCount(&n)
hipGetDeviceProperties(&prop, dev_i)
hipDeviceGetAttribute(&val, attr, dev)
hipDriverGetVersion(&ver)
hipRuntimeGetVersion(&ver)
```

### hipDeviceProp_t — CUDA-Compatible Fields

Identical semantics to `cudaDeviceProp`:
- `name[256]`, `uuid`, `totalGlobalMem`, `sharedMemPerBlock`, `regsPerBlock`
- `warpSize`, `maxThreadsPerBlock`, `maxThreadsDim[3]`, `maxGridSize[3]`
- `clockRate`, `multiProcessorCount`, `memoryClockRate`, `memoryBusWidth`
- `l2CacheSize`, `ECCEnabled`, `concurrentKernels`, `cooperativeLaunch`

### hipDeviceProp_t — HIP/AMD-Specific Extensions

These fields have **no CUDA equivalent** and enable AMD-specific dispatch decisions:

| Field | Type | Notes |
|---|---|---|
| `gcnArchName` | `char[256]` | Architecture string: `gfx906`, `gfx90a`, `gfx1100` etc. |
| `arch` | `hipDeviceArch_t` | Bitfield: hasGlobalInt32Atomics, hasDoubles, hasDynamicParallelism, etc. |
| `asicRevision` | `int` | Silicon revision within the GCN/CDNA/RDNA family |
| `clockInstructionRate` | `int` | Device-side instruction clock in kHz (HIP only) |
| `maxSharedMemoryPerMultiProcessor` | `size_t` | Max shared memory per CU (HIP label) |
| `hdpMemFlushCntl` | `uint*` | Address of HDP_MEM_COHERENCY_FLUSH_CNTL register |
| `hdpRegFlushCntl` | `uint*` | Address of HDP_REG_COHERENCY_FLUSH_CNTL register |
| `isLargeBar` | `int` | Large BAR PCI device (visible host mapping) |
| `cooperativeMultiDeviceLaunch` | `int` | Multi-device cooperative kernel support |
| `cooperativeMultiDeviceUnmatchedFunc` | `int` | Unmatched function multi-device cooperative |
| `gpuDirectRDMASupported` | `int` | RDMA support |
| `memoryPoolsSupported` | `int` | hipMallocAsync support |
| `ipcEventSupported` | `int` | IPC events |
| `clusterLaunch` | `int` | Thread block clusters (MI300+ era) |

**Reserved fields:** `reserved[63]` + `hipReserved[32]` for future growth.

### CUDA vs HIP API Surface Comparison

| Capability | CUDA | HIP |
|---|---|---|
| Device enumeration | `cudaGetDeviceCount` | `hipGetDeviceCount` |
| Bulk properties | `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| Single attribute | `cudaDeviceGetAttribute` | `hipDeviceGetAttribute` |
| Architecture string | `major.minor` compute cap | `gcnArchName` (gfxXYZ) |
| Warp size | always 32 | 32 (NVIDIA backend) or 64 (AMD GCN/CDNA) |
| Driver version | `cudaDriverGetVersion` | `hipDriverGetVersion` |
| Sub-device topo | none (NVLink via NVML) | none (XGMI via RSMI) |

**Key difference:** CUDA uses semantic compute capability versioning (8.0, 9.0) that maps to microarchitectures. HIP uses explicit ISA target strings (`gfx906`=MI50, `gfx90a`=MI200, `gfx1100`=RX 7900). For dispatch logic, `gcnArchName` is more directly actionable than compute capability decoding.

### Latency
Same profile as CUDA: cold start 1–2s, subsequent calls ~10 µs, `hipDeviceGetAttribute` ~1–5 µs.

---

## 3. Vulkan — vkEnumeratePhysicalDevices / vkGetPhysicalDeviceProperties

Vulkan is the only cross-vendor GPU API with production support for NVIDIA, AMD, Intel, and Apple Silicon simultaneously.

### Discovery Flow

```c
// Step 1: Create instance (no GPU contact yet)
vkCreateInstance(&instanceCI, null, &instance)

// Step 2: Enumerate physical devices (two-call pattern)
uint32_t count = 0;
vkEnumeratePhysicalDevices(instance, &count, NULL);
VkPhysicalDevice devs[count];
vkEnumeratePhysicalDevices(instance, &count, devs);

// Step 3: Query properties for each device
VkPhysicalDeviceProperties props;
vkGetPhysicalDeviceProperties(dev, &props);         // Core 1.0
vkGetPhysicalDeviceProperties2(dev, &props2);       // 1.1+, with pNext chain

// Step 4: Query features and memory
vkGetPhysicalDeviceFeatures(dev, &features);
vkGetPhysicalDeviceMemoryProperties(dev, &memProps);
```

### VkPhysicalDeviceProperties Fields

| Field | Type | Notes |
|---|---|---|
| `apiVersion` | `uint32_t` | Vulkan API version supported |
| `driverVersion` | `uint32_t` | Vendor-specific driver version |
| `vendorID` | `uint32_t` | PCI vendor ID (0x10DE=NVIDIA, 0x1002=AMD, 0x8086=Intel) |
| `deviceID` | `uint32_t` | PCI device ID |
| `deviceType` | `VkPhysicalDeviceType` | See enum below |
| `deviceName` | `char[256]` | Human-readable name |
| `pipelineCacheUUID` | `uint8_t[16]` | UUID for pipeline cache compatibility |
| `limits` | `VkPhysicalDeviceLimits` | Capability limits struct |
| `sparseProperties` | `VkPhysicalDeviceSparseProperties` | Sparse memory properties |

### VkPhysicalDeviceType Enum

```c
VK_PHYSICAL_DEVICE_TYPE_OTHER            // unclassified
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU  // iGPU (AMD APU, Intel UHD)
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU    // dedicated GPU (NVIDIA, AMD dGPU)
VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU     // VM/guest GPU
VK_PHYSICAL_DEVICE_TYPE_CPU             // software/CPU implementation (e.g. SwiftShader)
```

### VkPhysicalDeviceLimits — Key Compute Fields

Selected from the ~100+ fields in this struct:

| Field | Notes |
|---|---|
| `maxComputeSharedMemorySize` | Max workgroup shared memory in bytes |
| `maxComputeWorkGroupCount[3]` | Max dispatch counts per dimension |
| `maxComputeWorkGroupInvocations` | Max total invocations per workgroup |
| `maxComputeWorkGroupSize[3]` | Max workgroup dimensions |
| `maxMemoryAllocationCount` | Max concurrent memory allocations |
| `maxStorageBufferRange` | Max storage buffer binding size |
| `maxUniformBufferRange` | Max uniform buffer binding size |
| `timestampPeriod` | Nanoseconds per timestamp tick |

### Extended Properties via pNext Chain

`vkGetPhysicalDeviceProperties2` with chained structures exposes:
- `VkPhysicalDeviceVulkan11Properties` — subgroup size, multiview, protected memory
- `VkPhysicalDeviceVulkan12Properties` — driver ID, shader float controls, descriptor indexing
- `VkPhysicalDeviceVulkan13Properties` — inline uniform blocks, maintenance features
- `VkPhysicalDeviceDriverProperties` — driverID enum, driver name string, conformance version
- `VkPhysicalDeviceSubgroupProperties` — subgroupSize, supported stages, supported operations
- Cooperative matrix, ray tracing, mesh shader capability structs (extensions)

### VkPhysicalDeviceMemoryProperties

Reports `memoryHeaps[]` (total size + flags: DEVICE_LOCAL, HOST_VISIBLE) and `memoryTypes[]`. This gives actual VRAM size and whether unified memory exists.

### What Vulkan Does NOT Report

- Clock speeds (no API for current/max core or memory clock)
- Number of compute units/SMs (not exposed at the Vulkan level)
- Vendor-specific microarchitecture details

For those, Vulkan must be combined with vendor-specific extensions or sysfs.

### Latency

- `vkCreateInstance`: ~5–20 ms (one-time)
- `vkEnumeratePhysicalDevices`: < 1 ms (cached by driver)
- `vkGetPhysicalDeviceProperties`: < 100 µs (reads cached driver data)
- All property queries are lock-free and safe to call from multiple threads

---

## 4. SYCL — sycl::device::get_info

SYCL provides the highest-level device discovery model, modeling the OpenCL platform hierarchy (platform → device) but with a modern C++17 API.

### Discovery Flow

```cpp
// Enumerate all platforms
auto platforms = sycl::platform::get_platforms();
for (auto& plat : platforms) {
    auto devices = plat.get_devices();
    for (auto& dev : devices) {
        auto name = dev.get_info<sycl::info::device::name>();
        auto cu   = dev.get_info<sycl::info::device::max_compute_units>();
        // ...
    }
}

// Or use device selectors
sycl::queue q{sycl::gpu_selector_v};  // SYCL 2020
```

### sycl::info::device Query Descriptors (SYCL 2020 Complete List)

**Identification**
| Descriptor | Return Type | Notes |
|---|---|---|
| `info::device::device_type` | `info::device_type` | cpu/gpu/accelerator/custom |
| `info::device::vendor_id` | `uint32_t` | Numeric vendor ID |
| `info::device::name` | `std::string` | Device name |
| `info::device::vendor` | `std::string` | Vendor name string |
| `info::device::version` | `std::string` | Backend-defined version |
| `info::device::driver_version` | `std::string` | Driver version string |
| `info::device::aspects` | `std::vector<sycl::aspect>` | Feature capabilities |

**Compute**
| Descriptor | Return Type | Notes |
|---|---|---|
| `info::device::max_compute_units` | `uint32_t` | Compute units (CUs/SMs) |
| `info::device::max_work_item_dimensions` | `uint32_t` | Max ND-range dimensions |
| `info::device::max_work_item_sizes<1/2/3>` | `sycl::range` | Max work-items per dimension |
| `info::device::max_work_group_size` | `size_t` | Max work-group total size |
| `info::device::max_num_sub_groups` | `uint32_t` | Max sub-groups per work-group |
| `info::device::sub_group_sizes` | `std::vector<size_t>` | Supported sub-group widths |
| `info::device::max_clock_frequency` | `uint32_t` | Clock in MHz |

**Memory**
| Descriptor | Return Type | Notes |
|---|---|---|
| `info::device::global_mem_size` | `uint64_t` | Total global memory bytes |
| `info::device::local_mem_size` | `uint64_t` | Local/shared memory bytes |
| `info::device::local_mem_type` | `info::local_mem_type` | LOCAL or GLOBAL |
| `info::device::global_mem_cache_size` | `uint64_t` | L1/L2 cache size |
| `info::device::global_mem_cache_line_size` | `uint32_t` | Cache line bytes |
| `info::device::global_mem_cache_type` | `info::global_mem_cache_type` | None/Read/ReadWrite |
| `info::device::max_mem_alloc_size` | `uint64_t` | Max single allocation |
| `info::device::mem_base_addr_align` | `uint32_t` | Alignment requirement in bits |

**Floating-Point Config**
| Descriptor | Return Type | Notes |
|---|---|---|
| `info::device::half_fp_config` | `vector<info::fp_config>` | FP16 capabilities |
| `info::device::single_fp_config` | `vector<info::fp_config>` | FP32 capabilities |
| `info::device::double_fp_config` | `vector<info::fp_config>` | FP64 capabilities |

**Vector Widths**
| Descriptor | Notes |
|---|---|
| `info::device::preferred_vector_width_{char,short,int,long,float,double,half}` | Preferred SIMD width for each type |
| `info::device::native_vector_width_{char,short,int,long,float,double,half}` | Native ISA SIMD width |

**Device Aspects (Feature Flags — SYCL 2020)**
```cpp
aspect::cpu, aspect::gpu, aspect::accelerator
aspect::fp16, aspect::fp64, aspect::atomic64
aspect::usm_device_allocations, aspect::usm_host_allocations
aspect::usm_shared_allocations, aspect::usm_system_allocations
aspect::online_compiler, aspect::online_linker
aspect::queue_profiling
aspect::image
```

**Synchronization & Partitioning**
| Descriptor | Notes |
|---|---|
| `info::device::atomic_memory_order_capabilities` | Memory orderings supported |
| `info::device::partition_max_sub_devices` | Sub-device count |
| `info::device::partition_properties` | Partition strategies available |
| `info::device::profiling_timer_resolution` | Timer precision in nanoseconds |

### Platform Queries

```cpp
auto plat_name    = plat.get_info<sycl::info::platform::name>();
auto plat_vendor  = plat.get_info<sycl::info::platform::vendor>();
auto plat_version = plat.get_info<sycl::info::platform::version>();
```

### SYCL Implementation Notes

- DPC++ (Intel): backed by Level Zero (Intel GPU), CUDA (NVIDIA), HIP (AMD), OpenCL
- AdaptiveCpp (formerly hipSYCL): CUDA, HIP, Level Zero backends
- `max_compute_units` semantics vary by backend: CUs on AMD, SMs on NVIDIA, EUs on Intel
- No direct way to get `warpSize`/subgroup size without querying `sub_group_sizes` vector
- `driver_version` returns a vendor-specific string (not a stable numeric version)

### Latency
- `get_platforms()` + `get_devices()`: ~1–5 ms (full platform enumeration, backend-dependent)
- Individual `get_info<>()` calls: ~1–50 µs (cached from underlying API)
- More overhead than CUDA/HIP due to abstraction layers and backend dispatch

---

## 5. OpenCL — clGetDeviceInfo / clGetPlatformInfo

OpenCL provides the most comprehensive standardized device query table, having been designed explicitly for heterogeneous hardware discovery across CPU, GPU, FPGA, and DSP.

### Discovery Flow

```c
// Enumerate platforms
cl_uint nPlat;
clGetPlatformIDs(0, NULL, &nPlat);
cl_platform_id platforms[nPlat];
clGetPlatformIDs(nPlat, platforms, NULL);

// For each platform, enumerate devices
cl_uint nDev;
clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &nDev);
cl_device_id devs[nDev];
clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nDev, devs, NULL);

// Query device properties
cl_uint cu;
clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
```

### clGetPlatformInfo Parameters

| Param Name | Return Type | Notes |
|---|---|---|
| `CL_PLATFORM_NAME` | `char[]` | Platform name (e.g., "AMD Accelerated Parallel Processing") |
| `CL_PLATFORM_VENDOR` | `char[]` | Vendor name |
| `CL_PLATFORM_VERSION` | `char[]` | OpenCL version string |
| `CL_PLATFORM_PROFILE` | `char[]` | "FULL_PROFILE" or "EMBEDDED_PROFILE" |
| `CL_PLATFORM_EXTENSIONS` | `char[]` | Space-separated extension list |

### clGetDeviceInfo Parameters — Complete Table (OpenCL 3.0)

**Type / Identification**
| Param | Type | Notes |
|---|---|---|
| `CL_DEVICE_TYPE` | `cl_device_type` | CL_DEVICE_TYPE_GPU/CPU/ACCELERATOR/ALL |
| `CL_DEVICE_VENDOR_ID` | `cl_uint` | Numeric vendor ID |
| `CL_DEVICE_NAME` | `char[]` | Device name |
| `CL_DEVICE_VENDOR` | `char[]` | Vendor string |
| `CL_DRIVER_VERSION` | `char[]` | Driver version string |
| `CL_DEVICE_VERSION` | `char[]` | OpenCL version string |
| `CL_DEVICE_PROFILE` | `char[]` | FULL_PROFILE or EMBEDDED_PROFILE |
| `CL_DEVICE_EXTENSIONS` | `char[]` | Extension list |

**Compute**
| Param | Type | Notes |
|---|---|---|
| `CL_DEVICE_MAX_COMPUTE_UNITS` | `cl_uint` | CUs/SMs/EUs |
| `CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS` | `cl_uint` | Max ND-range dims |
| `CL_DEVICE_MAX_WORK_ITEM_SIZES` | `size_t[]` | Max work-items per dim |
| `CL_DEVICE_MAX_WORK_GROUP_SIZE` | `size_t` | Max work-group size |
| `CL_DEVICE_MAX_CLOCK_FREQUENCY` | `cl_uint` | Max clock in MHz |
| `CL_DEVICE_ADDRESS_BITS` | `cl_uint` | 32 or 64 |
| `CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` | `size_t` | Optimal WG size multiple |

**Memory**
| Param | Type | Notes |
|---|---|---|
| `CL_DEVICE_GLOBAL_MEM_SIZE` | `cl_ulong` | Total global memory bytes |
| `CL_DEVICE_LOCAL_MEM_SIZE` | `cl_ulong` | Shared/local memory bytes |
| `CL_DEVICE_LOCAL_MEM_TYPE` | `cl_device_local_mem_type` | CL_LOCAL or CL_GLOBAL |
| `CL_DEVICE_GLOBAL_MEM_CACHE_SIZE` | `cl_ulong` | Cache size bytes |
| `CL_DEVICE_GLOBAL_MEM_CACHE_TYPE` | `cl_device_mem_cache_type` | NONE/READ_ONLY/READ_WRITE |
| `CL_DEVICE_GLOBAL_MEM_CACHE_LINE_SIZE` | `cl_uint` | Cache line size bytes |
| `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | `cl_ulong` | Max single allocation |
| `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` | `cl_ulong` | Constant buffer limit |
| `CL_DEVICE_MEM_BASE_ADDR_ALIGN` | `cl_uint` | Alignment in bits |

**Float Capabilities**
| Param | Notes |
|---|---|
| `CL_DEVICE_HALF_FP_CONFIG` | FP16 ops supported (bitfield) |
| `CL_DEVICE_SINGLE_FP_CONFIG` | FP32 ops supported |
| `CL_DEVICE_DOUBLE_FP_CONFIG` | FP64 ops supported |

**Vector Widths**
| Param | Notes |
|---|---|
| `CL_DEVICE_PREFERRED_VECTOR_WIDTH_{CHAR,SHORT,INT,LONG,FLOAT,DOUBLE,HALF}` | Preferred SIMD widths |
| `CL_DEVICE_NATIVE_VECTOR_WIDTH_{CHAR,SHORT,INT,LONG,FLOAT,DOUBLE,HALF}` | Native ISA widths |

**Capabilities**
| Param | Type | Notes |
|---|---|---|
| `CL_DEVICE_IMAGE_SUPPORT` | `cl_bool` | Image read/write supported |
| `CL_DEVICE_ERROR_CORRECTION_SUPPORT` | `cl_bool` | ECC memory |
| `CL_DEVICE_ENDIAN_LITTLE` | `cl_bool` | Endianness |
| `CL_DEVICE_AVAILABLE` | `cl_bool` | Device is available |
| `CL_DEVICE_COMPILER_AVAILABLE` | `cl_bool` | Online compiler present |
| `CL_DEVICE_LINKER_AVAILABLE` | `cl_bool` | Separate compilation supported |
| `CL_DEVICE_BUILT_IN_KERNELS` | `char[]` | Built-in kernel list |
| `CL_DEVICE_PARTITION_PROPERTIES` | `cl_device_partition_property[]` | Partition modes |
| `CL_DEVICE_PROFILING_TIMER_RESOLUTION` | `size_t` | Timer resolution in ns |

### OpenCL Latency

- Platform enumeration (`clGetPlatformIDs`): ~1–5 ms
- Device enumeration per platform: ~100–500 µs
- Individual `clGetDeviceInfo` calls: ~1–10 µs (data already cached in driver)
- Cold path (first call after library load): can be 100+ ms on some implementations
- OpenCL kernel dispatch overhead: ~130–150 µs on iGPU, higher on discrete GPU (vs. ~5 µs for CUDA null kernel)

---

## 6. Linux sysfs — /sys/class/drm and Kernel Interfaces

Sysfs is the lowest-level, vendor-agnostic, zero-dependency mechanism for GPU discovery. It requires no GPU-specific runtime libraries.

### Primary Discovery Path

```
/sys/class/drm/card{N}/          # One directory per GPU
/sys/class/drm/card{N}/device/   # PCI device symlink + properties
```

### Properties Available via sysfs

**PCI Identity (Universal)**
```
/sys/class/drm/card0/device/vendor        # hex: 0x10de (NVIDIA), 0x1002 (AMD), 0x8086 (Intel)
/sys/class/drm/card0/device/device        # PCI device ID (hex)
/sys/class/drm/card0/device/subsystem_vendor
/sys/class/drm/card0/device/subsystem_device
/sys/class/drm/card0/device/class         # PCI class: 0x030200 = 3D Controller
```

**AMDGPU-specific (readable without ROCm)**
```
/sys/class/drm/card0/device/gpu_metrics         # Binary struct: temps, clocks, power, activity
/sys/class/drm/card0/device/mem_info_vram_total # VRAM total in bytes
/sys/class/drm/card0/device/mem_info_vram_used  # VRAM in use
/sys/class/drm/card0/device/pp_dpm_sclk         # GPU shader clock states (MHz)
/sys/class/drm/card0/device/pp_dpm_mclk         # Memory clock states (MHz)
/sys/class/drm/card0/device/pp_od_clk_voltage   # OD clock/voltage table (since Linux 4.17)
/sys/class/drm/card0/device/pp_power_profile_mode  # Power profile modes
/sys/class/drm/card0/device/unique_id           # 16-char GPU UUID
```

**Intel GPU (i915 driver)**
```
/sys/class/drm/card0/gt/gt0/rps_RP0_freq_mhz   # Max GPU frequency
/sys/class/drm/card0/gt/gt0/rps_max_freq_mhz   # Software max frequency
/sys/class/drm/card0/gt/gt0/rps_cur_freq_mhz   # Current frequency
```

**NVIDIA (nouveau driver — proprietary NVIDIA driver does NOT expose sysfs)**
- PCI identity visible via `/sys/bus/pci/devices/`
- NVML required for any meaningful properties under proprietary driver

### lspci — PCI-Level Discovery (No Driver Required)

```bash
lspci -nn -d ::03xx           # List all display controllers (class 0x03xx)
lspci -vvv -s <BDF>           # Full verbose info for a specific device
lspci -nn | grep -i 'vga\|3d\|2d'  # Filter GPUs
```

Output includes:
- BDF (Bus:Device.Function) address
- Vendor and device IDs with names (from PCI ID database)
- Subsystem vendor/device
- PCIe link speed and width (e.g., "LnkSta: Speed 16GT/s, Width x16")
- BAR addresses and sizes
- Interrupt routing

**PCI class codes for GPUs:**
- `0x030000` — VGA compatible controller
- `0x030200` — 3D controller (compute-only GPUs, e.g., A100/H100 in HGX)
- `0x038000` — Display controller

### hwloc — Portable Hardware Locality (Topology API)

hwloc provides a C API that integrates multiple GPU backends into a unified topology tree with NUMA awareness.

**GPU Backends Supported:**
| Backend | Object Created | Source |
|---|---|---|
| CUDA (via NVML) | `cuda0`, `nvml0` | NVIDIA NVML library |
| AMD ROCm | `rsmi0` | ROCm SMI library |
| Intel GPU | `ze0`, `ze0.0` (sub-devices) | Level Zero |
| OpenCL | `ocl0d0` | OpenCL ICD |

**Topology Information Exposed:**
- GPU ↔ CPU NUMA distance (PCIe locality)
- NVLinkBandwidth distance matrix between NVIDIA GPUs (via NVML backend)
- XGMIBandwidth distance matrix between AMD GPUs (via RSMI backend)
- XeLinkBandwidth between Intel Xe sub-devices (via Level Zero)
- PCIe link bandwidth and topology path

**Key hwloc API for GPU affinity:**
```c
hwloc_topology_t topo;
hwloc_topology_init(&topo);
hwloc_topology_set_flags(topo, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
hwloc_topology_load(topo);

// Get GPU OS object
hwloc_obj_t gpu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_OS_DEVICE, idx);

// Get CPU affinity set for that GPU (PCIe-local CPUs)
hwloc_bitmap_t cpuset = gpu->cpuset;
```

### ROCm SMI — AMD GPU Management Interface

For AMD GPUs, `rocm-smi` / `librocm_smi64` provides:
- Device discovery via `/sys/class/drm` scan + KFD topology
- Topology: GPU-to-GPU link weights, hop counts, link type (XGMI/PCIe)
- BDF-based stable device ordering

---

## 7. Metal — MTLDevice (Apple Silicon Context)

Metal is Apple-only. Relevant because Apple Silicon (M1/M2/M3/M4) unifies CPU and GPU memory, making device discovery semantics meaningfully different from discrete GPU platforms.

### Discovery Flow

```swift
// Single preferred GPU
let device = MTLCreateSystemDefaultDevice()!

// All GPUs (macOS, multi-GPU Macs)
let devices = MTLCopyAllDevices()

// Explicit GPU selection
let device = MTLCopyAllDevices().filter { !$0.isLowPower }.first!
```

### MTLDevice Key Properties

| Property | Type | Notes |
|---|---|---|
| `name` | `String` | Device name (e.g., "Apple M3 Max") |
| `registryID` | `UInt64` | Stable unique device identifier |
| `hasUnifiedMemory` | `Bool` | Always `true` on Apple Silicon; CPU+GPU share memory |
| `recommendedMaxWorkingSetSize` | `UInt64` | ~75% of physical RAM (e.g., ~96 GB on 128 GB M4 Max) |
| `maxTransferRate` | `UInt64` | Memory bandwidth in bytes/sec |
| `maxThreadsPerThreadgroup` | `MTLSize` | Max threads per compute threadgroup |
| `isLowPower` | `Bool` | Integrated/efficiency GPU (true on macOS iGPU) |
| `isRemovable` | `Bool` | eGPU (external GPU) |
| `isHeadless` | `Bool` | No display output |
| `currentAllocatedSize` | `UInt64` | Current GPU memory in use |
| `gpuFamily` | queried via `supportsFamily()` | Apple GPU family (Apple7, Apple8, etc.) |
| `supportsBCTextureCompression` | `Bool` | BC texture support |

**Feature detection:**
```swift
device.supportsFamily(.apple7)           // M1-era feature set
device.supportsFamily(.apple8)           // M2-era feature set
device.supportsFeatureSet(.macOS_GPUFamily2_v1) // Older API
device.supportsVertexAmplificationCount(2)
```

### Apple Silicon Dispatch Implications

- `hasUnifiedMemory = true` means zero-copy between CPU and GPU — no explicit transfer needed
- `recommendedMaxWorkingSetSize` is the practical "VRAM budget" (75% of unified memory pool)
- No explicit compute unit count exposed — `maxThreadsPerThreadgroup` + empirical profiling required
- Metal 4 (WWDC 2025) added `MTL4CommandAllocator` for direct command buffer memory control

### Latency
- `MTLCreateSystemDefaultDevice()`: ~1–5 ms (first call initializes Metal runtime)
- Property access after init: < 1 µs (all in-memory struct reads)

---

## 8. Level Zero — Intel's Low-Level GPU API

Level Zero (L0) is Intel's direct-to-hardware API for GPU control, used as the backend for SYCL/DPC++ on Intel GPU and for XPU management.

### Discovery Flow

```c
// Initialize (v1.10+ preferred)
ze_init_driver_type_desc_t desc = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC};
desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
uint32_t nDrivers = 0;
zeInitDrivers(&nDrivers, NULL, &desc);
ze_driver_handle_t drivers[nDrivers];
zeInitDrivers(&nDrivers, drivers, &desc);

// Enumerate devices per driver
uint32_t nDev = 0;
zeDeviceGet(driver, &nDev, NULL);
ze_device_handle_t devs[nDev];
zeDeviceGet(driver, &nDev, devs);

// Query properties
ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
zeDeviceGetProperties(dev, &props);

ze_device_compute_properties_t compute = {ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
zeDeviceGetComputeProperties(dev, &compute);

ze_device_memory_properties_t mem = {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES};
uint32_t nMem = 1;
zeDeviceGetMemoryProperties(dev, &nMem, &mem);
```

### ze_device_properties_t Fields

| Field | Notes |
|---|---|
| `type` | `ZE_DEVICE_TYPE_GPU`, `_CPU`, `_FPGA`, `_MCA`, `_VPU` |
| `vendorId` | PCI vendor ID |
| `deviceId` | PCI device ID |
| `name` | Device name string |
| `uuid` | `ze_device_uuid_t` — 16-byte UUID |
| `flags` | Bitfield: sub-device flag, ECC, etc. |
| `subdeviceId` | Sub-device index (tiles on Ponte Vecchio / Xe HPC) |
| `coreClockRate` | Core clock rate in MHz |
| `maxMemAllocSize` | Max single memory allocation bytes |
| `maxHardwareContexts` | Max command queues (hardware execution contexts) |
| `maxCommandQueuePriority` | Priority levels for queues |
| `numThreadsPerEU` | SIMT threads per Execution Unit |
| `physicalEUSimdWidth` | Physical SIMD width of each EU |
| `numEUsPerSubslice` | EUs per subslice |
| `numSubslicesPerSlice` | Subslices per slice |
| `numSlices` | Slices in device (top-level partition) |
| `timerResolution` | Timestamp resolution (cycles/sec since L0 v1.1) |
| `timestampValidBits` | Valid bits in kernel timestamps |
| `kernelTimestampValidBits` | Valid bits in kernel timestamps |

**Intel-specific topology:** `numSlices × numSubslicesPerSlice × numEUsPerSubslice × numThreadsPerEU` gives the full EU thread count.

### ze_device_compute_properties_t Fields

| Field | Notes |
|---|---|
| `maxTotalGroupSize` | Max total threads per work-group |
| `maxGroupSizeX/Y/Z` | Max per-dimension work-group size |
| `maxGroupCountX/Y/Z` | Max dispatch counts |
| `maxSharedLocalMemory` | Max shared local memory per group (bytes) |
| `numSubGroupSizes` | Number of supported sub-group sizes |
| `subGroupSizes[]` | Supported sub-group sizes array |

### ze_device_memory_properties_t Fields

| Field | Notes |
|---|---|
| `flags` | TBD |
| `maxClockRate` | Memory clock rate in MHz |
| `maxBusWidth` | Memory bus width in bits |
| `totalSize` | Total memory size in bytes |
| `name` | Memory type name |

### Sub-Device Discovery (Tiles / HBM stacks)

```c
uint32_t nSub = 0;
zeDeviceGetSubDevices(dev, &nSub, NULL);
ze_device_handle_t subdevs[nSub];
zeDeviceGetSubDevices(dev, &nSub, subdevs);
// Each sub-device has its own zeDeviceGetProperties
```

This enables tile-level dispatch on Ponte Vecchio (PVC) and future Intel GPUs with heterogeneous tiles.

### Latency

- `zeInitDrivers`: ~5–20 ms (one-time)
- `zeDeviceGetProperties`: < 100 µs (cached, lock-free per spec)
- `zeDeviceGetComputeProperties`: < 100 µs
- The spec explicitly states device objects are read-only global constructs — repeated `zeDeviceGet` returns identical handles, enabling safe global caching

---

## 9. Cross-API Latency Summary

| API | Cold Start | Repeated Property Query | Notes |
|---|---|---|---|
| CUDA | 1–2 s | ~10 µs (bulk), ~1–5 µs (single attr) | Driver context init dominates cold start |
| HIP | 1–2 s | ~10 µs | Same pattern as CUDA |
| Vulkan | 5–20 ms | < 100 µs | Instance creation is the cost; queries are cheap |
| SYCL | 1–5 ms | ~1–50 µs | Platform enumeration cost backend-dependent |
| OpenCL | 100+ ms (cold) | ~1–10 µs | ICD loader adds overhead; cold path expensive |
| Level Zero | 5–20 ms | < 100 µs | Lock-free; designed for repeated calls |
| sysfs | < 1 ms | < 100 µs | Pure filesystem reads; no library overhead |
| lspci | ~100 ms | N/A (CLI tool) | Reads `/sys/bus/pci/devices/` + PCI ID database |
| Metal | 1–5 ms | < 1 µs | macOS-only; properties are struct reads |

**Key insight for dispatch design:** All APIs are cheap for repeated calls after cold start. The appropriate strategy is:
1. Run device discovery once at application startup (or library initialization)
2. Cache results in a unified descriptor struct
3. Use cached data for all dispatch decisions at zero runtime cost

---

## 10. Unified Device Discovery Interface Design

### Motivation

Every API above returns overlapping but non-identical properties. A unified discovery layer would:
- Abstract away which backend is available
- Produce a canonical device descriptor usable for dispatch scoring
- Degrade gracefully when a specific API is unavailable (sysfs fallback)

### Proposed Canonical Device Descriptor

```c
typedef enum {
    GPU_VENDOR_NVIDIA = 0x10DE,
    GPU_VENDOR_AMD    = 0x1002,
    GPU_VENDOR_INTEL  = 0x8086,
    GPU_VENDOR_APPLE  = 0x106B,
    GPU_VENDOR_OTHER  = 0x0000,
} gpu_vendor_t;

typedef enum {
    GPU_ARCH_NVIDIA_AMPERE,     // SM 8.x
    GPU_ARCH_NVIDIA_HOPPER,     // SM 9.x
    GPU_ARCH_AMD_CDNA2,         // gfx90a (MI200)
    GPU_ARCH_AMD_CDNA3,         // gfx940/941 (MI300)
    GPU_ARCH_AMD_RDNA3,         // gfx1100 (RX 7900)
    GPU_ARCH_INTEL_XE_HPG,      // Arc Alchemist
    GPU_ARCH_INTEL_XE_HPC,      // Ponte Vecchio
    GPU_ARCH_APPLE_M_SERIES,
    GPU_ARCH_UNKNOWN,
} gpu_arch_t;

typedef enum {
    BACKEND_CUDA    = 1 << 0,
    BACKEND_HIP     = 1 << 1,
    BACKEND_VULKAN  = 1 << 2,
    BACKEND_OPENCL  = 1 << 3,
    BACKEND_LEVEL0  = 1 << 4,
    BACKEND_SYCL    = 1 << 5,
    BACKEND_METAL   = 1 << 6,
    BACKEND_SYSFS   = 1 << 7,   // always available on Linux
} backend_mask_t;

typedef struct {
    /* --- Identification --- */
    char            name[256];
    uint8_t         uuid[16];         // from CUDA/HIP/L0/Vulkan
    gpu_vendor_t    vendor;           // PCI vendor ID
    uint32_t        pci_device_id;    // PCI device ID
    uint32_t        pci_bus;
    uint32_t        pci_device;
    uint32_t        pci_domain;
    gpu_arch_t      arch;             // decoded from SM version / gcnArchName / EU topology
    char            arch_string[64];  // raw: "gfx90a", "sm_80", "xe-hpc", "apple-m3"

    /* --- Compute Capabilities --- */
    uint32_t        compute_units;    // SMs / CUs / EUs (normalized name)
    uint32_t        max_threads_per_cu;
    uint32_t        max_threads_per_block;
    uint32_t        warp_size;        // 32 (NVIDIA/Intel), 64 (AMD GCN/CDNA)
    uint32_t        max_clock_mhz;    // core clock
    uint32_t        subgroup_sizes[8]; // supported sub-group/warp sizes
    uint32_t        subgroup_count;

    /* --- Memory --- */
    uint64_t        global_mem_bytes;
    uint64_t        shared_mem_per_block_bytes;
    uint64_t        l2_cache_bytes;
    uint32_t        mem_bus_width_bits;
    uint32_t        mem_clock_mhz;
    bool            has_unified_memory;  // Apple Silicon, AMD APU

    /* --- Feature Flags --- */
    bool            has_fp16;
    bool            has_fp64;
    bool            has_bf16;
    bool            has_int8;
    bool            has_tensor_cores;   // NVIDIA only via SM major >= 7
    bool            has_ecc;
    bool            has_rdma;
    bool            is_integrated;
    bool            is_discrete;

    /* --- NUMA / Topology --- */
    int             numa_node;          // -1 if unknown
    int             host_numa_node;

    /* --- Available Backends --- */
    backend_mask_t  available_backends;
    int             cuda_device_index;  // -1 if CUDA not available
    int             hip_device_index;
    int             vulkan_device_index;
    int             opencl_device_index;
    int             level_zero_device_index;

} gpu_device_descriptor_t;
```

### Discovery Priority Chain

```
1. Try CUDA (if NVIDIA GPU detected via sysfs vendor ID)
   → Provides: SM, memory, all cudaDeviceProp fields
   → Falls back if: no NVIDIA driver, CUDA runtime unavailable

2. Try HIP (if AMD GPU detected via sysfs vendor ID)
   → Provides: gcnArchName, all hipDeviceProp_t fields
   → Falls back if: no ROCm installation

3. Try Level Zero (if Intel GPU detected or for sub-device topology)
   → Provides: EU topology, tile sub-devices, XeLink bandwidth
   → Falls back if: no oneAPI installation

4. Try Vulkan (cross-vendor fallback for basic properties)
   → Provides: device type, vendor, memory heaps, limits
   → Available on: NVIDIA, AMD, Intel, Apple Silicon simultaneously
   → Best for: first-pass device enumeration before vendor API init

5. Try OpenCL (legacy fallback)
   → Provides: compute units, memory, clock, extensions
   → Broadest hardware support (FPGA, DSP, CPU as GPU)

6. sysfs fallback (Linux, no GPU driver required)
   → Provides: PCI vendor/device ID, VRAM (AMD), clock states (AMD)
   → Always available on Linux regardless of installed runtimes
```

### Discovery Function Sketch

```c
// Discover all GPU devices on the system.
// Returns: number of devices found.
// Fills: descriptors[] with canonical device info.
// Strategy: vendor-specific API first, Vulkan for cross-vendor, sysfs as floor.
int gpu_discover_devices(gpu_device_descriptor_t* descriptors, int max_count);

// Score a device for a given workload class.
// Returns a float in [0,1] where 1.0 = ideal device for this workload.
float gpu_score_device(const gpu_device_descriptor_t* dev,
                       gpu_workload_class_t workload);

// Select best device for a workload class.
int gpu_select_device(const gpu_device_descriptor_t* devs, int n_devs,
                      gpu_workload_class_t workload);
```

---

## 11. Cross-API Comparison Table

| Property | CUDA | HIP | Vulkan | SYCL | OpenCL | L0 | sysfs | Metal |
|---|---|---|---|---|---|---|---|---|
| Vendor ID | via PCI | via PCI | VendorID | VendorID | VendorID | VendorID | /device/vendor | N/A |
| Device name | name[256] | name[256] | deviceName | name | CL_DEVICE_NAME | name | N/A | name |
| Compute units | multiProcessorCount | multiProcessorCount | not exposed | max_compute_units | CL_DEVICE_MAX_COMPUTE_UNITS | numSlices×... | not exposed | not exposed |
| Warp/wave size | warpSize (32) | warpSize (32/64) | subgroup (ext) | sub_group_sizes | not standard | subGroupSizes | not exposed | not exposed |
| Core clock | clockRate | clockRate | not exposed | max_clock_frequency | CL_DEVICE_MAX_CLOCK_FREQUENCY | coreClockRate | pp_dpm_sclk | not exposed |
| VRAM total | totalGlobalMem | totalGlobalMem | memoryHeaps | global_mem_size | CL_DEVICE_GLOBAL_MEM_SIZE | totalSize | mem_info_vram_total | recommendedMaxWorkingSetSize |
| Shared mem | sharedMemPerBlock | sharedMemPerBlock | maxComputeSharedMemorySize | local_mem_size | CL_DEVICE_LOCAL_MEM_SIZE | maxSharedLocalMemory | not exposed | N/A |
| L2 cache | l2CacheSize | l2CacheSize | not exposed | not exposed | not standard | not exposed | not exposed | not exposed |
| Arch string | SM major.minor | gcnArchName | not exposed | not exposed | not exposed | numSlices/EUs | not exposed | GPU family |
| Sub-devices | NVLink (NVML) | not in HIP | not standard | partition | not standard | zeDeviceGetSubDevices | N/A | not exposed |
| UUID | uuid | uuid | pipelineCacheUUID | not standard | not standard | uuid | unique_id (AMD) | registryID |
| Unified memory | integrated flag | integrated flag | integrated type | not direct | not direct | not direct | not exposed | hasUnifiedMemory |
| FP16 support | SM >= 6.0 | arch.hasHalfIntrinsics | features.shaderFloat16 | aspect::fp16 | CL_DEVICE_HALF_FP_CONFIG | via compute | not exposed | supportsFamily |
| ECC | ECCEnabled | ECCEnabled | not exposed | error_correction_support | error_correction_support | flags | not exposed | not exposed |
| NUMA node | deviceNumaId | not in hipDeviceProp | not exposed | not exposed | not exposed | not exposed | PCIe topology | not exposed |

---

## 12. Key Findings for the Poster

1. **No single API covers everything.** CUDA gives the richest NVIDIA-specific data. HIP gives AMD architecture strings. L0 gives Intel EU topology. Vulkan is the only cross-vendor API that enumerates all three simultaneously without vendor-specific setup.

2. **Vulkan is the best foundation for a unified enumerator.** It runs on all GPU vendors (including Apple via MoltenVK), requires only instance creation, and can enumerate all physical devices before any compute context is created. Compute unit counts and clock speeds require vendor-specific APIs as a second pass.

3. **Warp/wavefront size is the most dispatch-critical property.** NVIDIA is always 32; AMD GCN/CDNA is 64; AMD RDNA is 32; Intel is variable (8, 16, 32). This single property determines vectorization granularity for ML kernels and is exposed differently across all APIs.

4. **Cold-start latency is the real bottleneck, not query latency.** CUDA/HIP initialization is 1–2 seconds. A unified dispatcher should run discovery at process startup and cache all descriptors. Subsequent dispatch decisions are then purely in-memory operations (<1 µs).

5. **sysfs is always available on Linux** and can determine GPU vendor from PCI IDs before loading any runtime. This enables a zero-overhead "which backends should I try?" pre-filter that avoids attempting to load CUDA on an AMD system.

6. **Architecture strings vs. compute capability:** CUDA uses `major.minor` (semantic versioning tied to microarchitectures). HIP uses ISA target strings (`gfx90a`). This matters for dispatch: a dispatcher selecting between a kernel compiled for SM 8.0 vs SM 9.0 needs compute capability; a dispatcher choosing between GCN and CDNA microarchitectures needs `gcnArchName`.

7. **IRIS (ORNL) is the closest existing system** to a unified dispatcher: it wraps CUDA, HIP, Level Zero, OpenCL, and OpenMP under a task-based scheduling API. Published in IEEE TPDS 2024. Key gap: it doesn't expose a normalized device descriptor for user-space dispatch scoring.

8. **hwloc is the right tool for NUMA/PCIe topology.** It wraps NVML, RSMI, Level Zero, and OpenCL backends into a single topology tree with distance matrices (NVLink bandwidth, XGMI bandwidth, XeLink bandwidth). A unified dispatcher should use hwloc for topology-aware kernel placement.

---

## References

- CUDA Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
- HIP Device Management: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html
- Vulkan Spec — Devices and Queues: https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html
- SYCL Reference — Device: https://github.khronos.org/SYCL_Reference/iface/device.html
- OpenCL 3.0 clGetDeviceInfo: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html
- Level Zero Spec: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html
- Metal MTLDevice: https://developer.apple.com/documentation/metal/mtldevice
- hwloc: https://www.open-mpi.org/projects/hwloc/
- IRIS Framework (IEEE TPDS 2024): https://dl.acm.org/doi/10.1109/TPDS.2024.3429010
- ROCm SMI Overview: https://rocm.blogs.amd.com/software-tools-optimization/amd-smi-overview/README.html
- AMD SMI Topology: https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html
- Intel Device Discovery with SYCL: https://www.intel.com/content/www/us/en/developer/articles/technical/device-discovery-with-sycl.html
