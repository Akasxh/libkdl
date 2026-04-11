# Fat Binary and Multi-Versioned Kernel Techniques — Research Notes

**Project:** LLVM Dublin 2026 Poster — Heterogeneous GPU Kernel Dispatch
**Author:** Research sweep, 2026-04-02
**Scope:** CUDA fat binaries, HIP fat binaries, LLVM offload format, cross-vendor multi-target packaging, runtime dispatch mechanics

---

## 1. CUDA Fat Binaries: PTX + SASS for Multiple SM Versions

### Overview

CUDA "fat binaries" (fatbinaries or fatbin) are containers that bundle multiple translations of the same GPU source code — both PTX (portable virtual ISA) and SASS (Streaming ASSembly, the native binary per SM version) — into a single deployable artifact embedded inside the host ELF.

The CUDA driver selects the most appropriate translation at kernel launch time. If a native SASS cubin for the target GPU is present, it is used directly. If only PTX is available, the driver JIT-compiles PTX to SASS.

### Internal Format: `fatBinaryHeader`

The binary format is nominally undocumented by NVIDIA but has been reverse-engineered and is confirmed via CUDA SDK headers (`fatbinary.h`):

```c
struct fatBinaryHeader {           // total: 16 bytes, 8-byte aligned
    unsigned int           magic;       // offset  0, 4 bytes
    unsigned short         version;     // offset  4, 2 bytes
    unsigned short         headerSize;  // offset  6, 2 bytes (must be multiple of 8)
    unsigned long long int fatSize;     // offset  8, 8 bytes (size excl. header, multiple of 8)
};
```

Magic constants:
- `FATBIN_MAGIC        = 0xBA55ED50U`  — current format
- `OLD_STYLE_FATBIN_MAGIC = 0x1EE55A01U` — legacy
- `FATBIN_VERSION      = 0x0001U`

Code section kinds embedded after the header:

```c
typedef enum {
    FATBIN_KIND_PTX      = 0x0001,  // PTX text/bitcode
    FATBIN_KIND_ELF      = 0x0002,  // cubin (ELF-formatted SASS binary)
    FATBIN_KIND_OLDCUBIN = 0x0004,  // legacy cubin, no longer generated
    FATBIN_KIND_IR       = 0x0008,  // NVVM LTO-IR
} fatBinaryCodeKind;
```

Each embedded section specifies a target SM (e.g., `sm_80`) and its kind. Sections are concatenated; `fatSize` gives total length.

### ELF Embedding

Inside the host ELF, two sections carry the fat binary:

| ELF Section          | Platform | Purpose                                      |
|----------------------|----------|----------------------------------------------|
| `.nv_fatbin`         | Linux    | Raw fatbinary blob data                       |
| `.nvFatBinSegment`   | Linux    | Wrapper control struct (`__fatBinC_Wrapper_t`) |
| `__nv_fatbin`        | macOS    | Raw fatbinary blob (Mach-O naming)            |
| `__fatbin`           | macOS    | Wrapper control struct (Mach-O)               |

The wrapper structure that lives in `.nvFatBinSegment`:

```c
typedef struct {
    int                        magic;    // 0x466243B1 (FATBINC_MAGIC)
    int                        version;  // 1 = offline file, 2 = prelinked fatbin
    const unsigned long long*  data;     // pointer into .nv_fatbin
    void*                      filename_or_fatbins;
} __fatBinC_Wrapper_t;
```

`FATBINC_MAGIC = 0x466243B1` identifies the wrapper (distinct from `FATBIN_MAGIC` which identifies the inner blob).

### nvFatbin API (CUDA 12.4+)

NVIDIA introduced the `nvFatbin` runtime API (CUDA Toolkit 12.4) to construct fat binaries programmatically at runtime. Input types: cubin, PTX, LTO-IR, TileIR, relocatable PTX. Output is compatible with `cuModuleLoadData`. Supports compression modes (default / size / speed / balance / none).

---

## 2. nvcc's `-gencode` Flag: Multi-Architecture Compilation

### Two-Stage Compilation Model

`nvcc` uses a two-stage device compilation model:

1. **Stage 1 (PTX generation):** Source → PTX. Controlled by `-arch=compute_XY` (virtual architecture).
2. **Stage 2 (SASS generation):** PTX → CUBIN. Controlled by `-code=sm_XY` (real architecture).

The `-gencode` flag is a combined shorthand:

```
-gencode=arch=compute_XY,code=[sm_XY | compute_XY]
```

Multiple `-gencode` invocations are concatenated into the fat binary. Each invocation adds:
- One SASS cubin section if `code=sm_*` is specified
- One PTX section if `code=compute_*` is specified (for JIT forward compatibility)

### Compatibility Constraint

The virtual architecture (`arch=compute_X`) must be ≤ the real architecture SM number (`code=sm_Y`). `arch=compute_103,code=sm_100` is a fatal error.

### Architecture Mapping (Current as of 2026)

| Architecture       | SM     | Hardware examples              | gencode flag                                  |
|--------------------|--------|-------------------------------|-----------------------------------------------|
| Volta              | sm_70  | V100, Titan V                  | `-gencode=arch=compute_70,code=sm_70`         |
| Ampere (A100)      | sm_80  | A100, DGX-A100                 | `-gencode=arch=compute_80,code=sm_80`         |
| Ampere (RTX 30)    | sm_86  | RTX 3080/3090, A5000           | `-gencode=arch=compute_86,code=sm_86`         |
| Ada Lovelace       | sm_89  | RTX 4090, L40, L4              | `-gencode=arch=compute_89,code=sm_89`         |
| Hopper             | sm_90  | H100, H200                     | `-gencode=arch=compute_90,code=sm_90`         |
| Hopper (extended)  | sm_90a | H100 (with wgmma/tma ops)     | `-gencode=arch=compute_90a,code=sm_90a`       |
| Blackwell          | sm_100 | B100, B200                     | `-gencode=arch=compute_100,code=sm_100`       |
| Blackwell (RTX 50) | sm_120 | RTX 5090 etc.                  | `-gencode=arch=compute_120,code=sm_120`       |

### Typical Multi-SM Production Pattern

```bash
nvcc kernel.cu \
  -gencode=arch=compute_80,code=sm_80 \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_89,code=sm_89 \
  -gencode=arch=compute_90,code=sm_90 \
  -gencode=arch=compute_90,code=compute_90   # PTX for JIT on future GPUs
```

This produces a fat binary with 4 cubins + 1 PTX blob.

---

## 3. HIP Fat Binaries: AMD's Multi-Architecture Bundling

### Overview

HIP on AMD GPUs uses a parallel mechanism to CUDA fat binaries. The Clang compiler (with ROCm's LLVM fork) bundles multiple GFX architecture device images using `clang-offload-bundler`, embedding the result as `__hip_fatbin` in the `.hip_fatbin` ELF section.

### Compilation Invocation

Multi-architecture HIP compilation:

```bash
clang++ -x hip kernel.hip \
  --offload-arch=gfx906 \
  --offload-arch=gfx908 \
  --offload-arch=gfx90a \
  --offload-arch=gfx1100
```

The `--offload-arch` flag can be repeated. `--offload-arch=native` auto-detects locally available GPUs (via the `amdgpu-arch` tool).

### Compilation Modes

- **Non-RDC mode (`-fno-gpu-rdc`, default):** Each translation unit produces its own fully linked fat binary. No cross-TU device linking. Lower link overhead.
- **RDC mode (`-fgpu-rdc`):** Relocatable device compilation — device code from multiple TUs is linked together at executable link time via `clang-linker-wrapper`.

### Fat Binary Registration

Clang auto-generates a `__hip_module_ctor` function placed in `@llvm.global_ctors`:

```
__hip_module_ctor:
  - calls __hipRegisterFatBinary(wrapper_ptr)  → returns handle
  - calls __hip_register_globals(handle)       → registers kernels + device globals
  - registers __hip_module_dtor via atexit()
```

At kernel launch, the HIP runtime uses the registration handle to identify which device image to load for the current GPU.

### ELF Section

The fat binary blob is stored in the `.hip_fatbin` ELF section with the global symbol `__hip_fatbin` pointing to it. The bundle entry ID format (from `clang-offload-bundler`) encodes target identity.

---

## 4. LLVM's `clang-offload-bundler`: Format Specification

### Purpose

`clang-offload-bundler` combines code objects for host and multiple offload targets into a single file. It is used by both HIP and OpenMP offloading toolchains. It is distinct from the newer `llvm-offload-binary` format used by the LLVM new offloading design.

### Bundle Entry ID Syntax (BNF)

```
<bundle-entry-id>  ::= <offload-kind> "-" <target-triple> [ "-" <target-id> ]
<offload-kind>     ::= "host" | "hip" | "hipv4" | "openmp"
<target-triple>    ::= <arch>-<vendor>-<sys>-<env>  (must be 4 fields)
<target-id>        ::= <processor> [ (":" <feature> ("+" | "-"))* ]
```

Example IDs:
```
host-x86_64-unknown-linux-gnu
hip-amdgcn-amd-amdhsa-gfx906:sramecc+:xnack-
hipv4-amdgcn-amd-amdhsa-gfx90a
openmp-amdgcn-amdhsa-gfx908:sramecc-:xnack+
```

Features (e.g., `sramecc`, `xnack`) can be `+` (on), `-` (off), or absent (any/default).

### Binary Bundle Format

For binary file types (`.o`, `.bc`, `.a`):

| Field             | Size       | Content                                      |
|-------------------|------------|----------------------------------------------|
| Magic string      | 24 bytes   | `__CLANG_OFFLOAD_BUNDLE__`                   |
| Entry count       | 8 bytes    | Number of bundled code objects (uint64)      |
| Per-entry header  | variable   | [offset:8][size:8][id_len:8][id:id_len]      |
| Code objects      | variable   | Actual compiled binaries (PTX, ELF, etc.)    |

### Text Bundle Format

For text files (`.ll`, `.s`, `.ii`):

```
; __CLANG_OFFLOAD_BUNDLE__START__ hip-amdgcn-amd-amdhsa-gfx906
... device code ...
; __CLANG_OFFLOAD_BUNDLE__END__ hip-amdgcn-amd-amdhsa-gfx906
```

Comment character varies by file type (`;` for LLVM IR, `#` for assembly).

### Compression: CCOB Format

Compressed bundles begin with a 4-byte `CCOB` magic prefix (Compressed Clang Offload Bundle), followed by:

| Field             | v2 size  | v3 size  | Content                       |
|-------------------|----------|----------|-------------------------------|
| Magic             | 4 bytes  | 4 bytes  | `CCOB`                        |
| Version           | 2 bytes  | 2 bytes  | 2 or 3                        |
| Method            | 2 bytes  | 2 bytes  | compression algorithm enum    |
| Total size        | 4 bytes  | 8 bytes  | file size                     |
| Uncompressed size | 4 bytes  | 8 bytes  | original data size            |
| Hash              | 8 bytes  | 8 bytes  | 64-bit truncated MD5          |
| Compressed data   | variable | variable | zlib or zstd compressed content |

### Supported File Types

| Format     | Type code | Binary/Text |
|------------|-----------|-------------|
| LLVM IR    | ll        | Text        |
| Bitcode    | bc        | Binary      |
| Assembler  | s         | Text        |
| Object     | o         | Binary      |
| Archive    | a         | Binary      |
| CUDA/HIP   | cui       | Text        |

---

## 5. Cross-Vendor GPU Fat Binaries: NVPTX + AMDGCN + SPIR-V + x86

### Current State: No Universal Standard

There is no single standardized "universal GPU fat binary" format that natively bundles NVPTX, AMDGCN, SPIR-V, and x86 host code in a single container with a common runtime selector. Each vendor ecosystem uses its own format:

| Ecosystem       | Container Format           | Runtime Selector                  |
|-----------------|----------------------------|-----------------------------------|
| NVIDIA CUDA     | CUDA fatbin (`.nv_fatbin`) | CUDA driver (cuModuleLoad*)       |
| AMD HIP         | clang-offload-bundle       | HIP runtime (__hipRegisterFatBinary) |
| Intel SYCL/DPC++ | SYCL fat binary (clang-offload-wrapper) | SYCL runtime PI layer |
| OpenMP offload  | LLVM .llvm.offloading      | libomptarget                      |

### Intel DPC++ / SYCL: Closest to Multi-Vendor

Intel's oneAPI DPC++ compiler (`-fsycl`) is the closest existing approach to a cross-vendor fat binary. With `-fsycl-targets=triple1,triple2`, a single compilation produces a host binary containing device images for multiple targets simultaneously:

```bash
clang++ -fsycl \
  -fsycl-targets=spir64,nvptx64-nvidia-cuda,amdgcn-amd-amdhsa \
  kernel.cpp
```

Supported target triples:
- `spir64` — generic SPIR-V (JIT on Intel GPU via OpenCL)
- `spir64_gen` — AOT Intel GPU (via `ocloc`)
- `spir64_x86_64` — Intel CPU SPIR-V
- `nvptx64-nvidia-cuda` — NVIDIA GPU (via NVPTX backend + ptxas)
- `amdgcn-amd-amdhsa` — AMD GPU (via AMDGPU backend)

The compiler runs a separate device compilation pipeline per target, links each via `llvm-link`, translates to target IR (SPIR-V or PTX/cubin), then wraps all images into the host binary via `clang-offload-wrapper`. The resulting executable contains all device images and registers them all at startup.

**Limitation:** This requires building with the Intel LLVM fork. Upstream LLVM does not yet support this flow for all combinations natively.

### HetGPU (2025 Research): True Cross-Vendor Portable Binary

A 2025 research system (HetGPU, arxiv 2506.15993) introduces a genuinely cross-vendor approach:

- Defines `hetIR`: a target-independent GPU virtual ISA (SPMD model, explicit barriers, abstract memory ops: `LD_GLOBAL`, `ST_GLOBAL`, `LD_SHARED`)
- Compiles once to hetIR; JIT-translates at launch time to:
  - PTX (NVIDIA) via NVVM
  - SPIR-V (AMD, Intel) via SPIRV-Tools
  - Metalium/RISC-V (Tenstorrent)
- Single binary, runtime device detection, JIT caching after first run
- Overhead: `<8%` on NVIDIA, `<6%` on AMD vs. native; JIT latency 50–200ms per kernel (amortized)
- **Key distinction from fat binaries:** Stores one abstract IR instead of N pre-compiled machine code blobs. Smaller binaries, broader portability, but requires JIT infrastructure.

### LLVM RFC: SPIR-V as Vendor-Agnostic GPU IR

An active LLVM RFC (2024) proposes using SPIR-V as a single GPU target that downstream-lowers to `amdgcn`/`nvptx`. AMD has separately implemented "vendor-flavored SPIR-V" (AMDGCN-flavored SPIR-V) that the LLVM AMDGPU backend can consume. This would enable a single SPIR-V blob as the portable representation, with vendor backends handling final lowering — a middle ground between fat binary (N blobs) and JIT from abstract IR.

---

## 6. LLVM's Support for Multi-Target Object Files

### The New Offloading Design (LLVM 14+)

LLVM's offloading infrastructure (unified across CUDA, HIP, OpenMP since ~LLVM 14) is built around the `.llvm.offloading` ELF section and the `llvm-offload-binary` format. This replaces the older `clang-offload-bundler`-based pipeline for CUDA/HIP in RDC mode and for all OpenMP offloading.

### Key Tools

| Tool                    | Role                                                                 |
|-------------------------|----------------------------------------------------------------------|
| `llvm-offload-binary`   | Creates binary format with magic `0x10FF10AD`, embeds device images  |
| `clang-offload-wrapper` | Wraps device binaries into host LLVM IR (bitcode) with registration code |
| `clang-linker-wrapper`  | Intercepts the host link step, extracts `.llvm.offloading`, links device code, re-embeds |
| `clang-offload-bundler` | Older tool, still used for HIP non-RDC and OpenMP archive bundling   |

### Object File Lifecycle

1. **Compile:** `clang -c -fopenmp-targets=...` or `clang -x cuda`
   - Produces fat object: host `.o` with embedded `.llvm.offloading` section
2. **Relocatable link:** `clang -r` merges `.llvm.offloading` sections (no device linking yet)
3. **Final link:** `clang-linker-wrapper` intercepts:
   - Scans all input `.o` for `.llvm.offloading`
   - Extracts device binaries
   - Routes each to appropriate device linker (`lld`, `nvptx-ld`, etc.)
   - Wraps linked device ELFs via `clang-offload-wrapper` into registration bitcode
   - Merges registration bitcode into final host link

The `.llvm.offloading` section is marked `SHF_EXCLUDE` — stripped from the final executable after extraction, keeping host binary clean.

---

## 7. ELF Sections for GPU Code: `.nv_fatbin` and `.hip_fatbin`

### NVIDIA: `.nv_fatbin` + `.nvFatBinSegment`

| ELF Section          | Contains                                                     |
|----------------------|--------------------------------------------------------------|
| `.nv_fatbin`         | Raw CUDA fat binary blob (starts with `FATBIN_MAGIC 0xBA55ED50`) |
| `.nvFatBinSegment`   | `__fatBinC_Wrapper_t` struct: magic `0x466243B1`, version, ptr into `.nv_fatbin` |

The fat binary blob is a sequence of entries. Each entry has a compressed/uncompressed header indicating SM version and code kind (PTX/ELF). The `headerSize` and `fatSize` fields allow parsing without knowing ahead-of-time how many entries exist.

On macOS/Mach-O, these live in `__NV_CUDA` segment as `__nv_fatbin` (data) and `__fatbin` (wrapper), subject to Mach-O's 15-character section name limit.

### AMD: `.hip_fatbin`

| ELF Section    | Contains                                                         |
|----------------|------------------------------------------------------------------|
| `.hip_fatbin`  | clang-offload-bundler binary bundle, global symbol `__hip_fatbin` |

The bundle in `.hip_fatbin` uses the `__CLANG_OFFLOAD_BUNDLE__` magic string (24 bytes) format. Each entry's bundle ID encodes both the GFX processor and feature flags (`gfx906:sramecc+:xnack-`), allowing the runtime to do exact-match or feature-compatible selection.

### LLVM Unified: `.llvm.offloading`

| ELF Section       | Contains                                                    |
|-------------------|-------------------------------------------------------------|
| `.llvm.offloading` | `llvm-offload-binary` format blobs, one per device target |

Each blob starts with `0x10FF10AD`. The section is marked `SHF_EXCLUDE` and is removed by `clang-linker-wrapper` after device extraction. It is never present in the final deployed executable.

---

## 8. How the CUDA Runtime Selects the Right Binary at Kernel Launch

### Registration Phase (at process startup)

Automatically generated host code calls `__cudaRegisterFatBinary` for each translation unit. The function:

1. Reads the `__fatBinC_Wrapper_t` struct from `.nvFatBinSegment`
2. Validates `magic == 0x466243B1`
3. Passes the wrapper's `data` pointer (pointing into `.nv_fatbin`) to the CUDA driver
4. The driver registers the fat binary and returns a handle

Subsequently, `__cudaRegisterFunction` is called per kernel, mapping a host-side function pointer to a kernel name string. The mapping goes: `host function ptr → kernel name → fat binary handle`.

### Selection Phase (at kernel launch, `cuLaunchKernel`)

When a kernel is launched, the CUDA driver:

1. Identifies the target GPU's SM version (queried via device attributes)
2. Searches the registered fat binary for a cubin entry matching that SM version
3. **If a native cubin is found:** Loads it directly (no JIT, minimal latency)
4. **If no exact cubin but PTX is present:** Invokes the PTX JIT compiler
   - Compiled SASS is cached on disk (controlled by `CUDA_CACHE_PATH`, default 256 MiB)
   - Cached SASS is reused on subsequent launches, until driver upgrade invalidates cache
5. **If neither is found:** Launch fails with `CUDA_ERROR_NO_BINARY_FOR_GPU`

### JIT Cache Control Environment Variables

| Variable                | Default        | Effect                                            |
|-------------------------|----------------|---------------------------------------------------|
| `CUDA_CACHE_DISABLE`    | 0              | Set to 1 to disable on-disk JIT cache             |
| `CUDA_CACHE_MAXSIZE`    | 256 MiB        | Max cache size (up to 4 GiB)                      |
| `CUDA_CACHE_PATH`       | OS default     | Override cache directory location                 |
| `CUDA_FORCE_PTX_JIT`    | 0              | Set to 1 to force JIT even when cubin is present  |

### Forward Compatibility via PTX

PTX is versioned and forward-compatible: PTX generated for `compute_80` will JIT correctly on `sm_90`. This is why production deployments typically include a `code=compute_XY` PTX entry as the highest-version `-gencode`, ensuring the binary runs on future GPUs without recompilation.

---

## 9. LLVM Offload Binary Format: `llvm-offload-binary`, `clang-linker-wrapper`

### `llvm-offload-binary` Binary Format

The format is used to embed device images into the `.llvm.offloading` ELF section.

**Header** (40 bytes total):

| Field         | Type      | Offset | Size | Content                          |
|---------------|-----------|--------|------|----------------------------------|
| magic         | uint8_t   | 0      | 4    | `0x10FF10AD`                     |
| version       | uint32_t  | 4      | 4    | Currently version 1              |
| size          | uint64_t  | 8      | 8    | Total binary size in bytes       |
| entry_offset  | uint64_t  | 16     | 8    | Absolute offset to entry array   |
| entry_size    | uint64_t  | 24     | 8    | Size of entry array section      |

**Per-Entry Structure:**

| Field         | Type      | Size | Content                              |
|---------------|-----------|------|--------------------------------------|
| image_kind    | uint16_t  | 2    | Binary format (object/bitcode/cubin/fatbin/PTX) |
| offload_kind  | uint16_t  | 2    | Producer (OpenMP/CUDA/HIP/SYCL)      |
| flags         | uint32_t  | 4    | Generic flags                        |
| string_offset | uint64_t  | 8    | Offset into metadata string table    |
| num_strings   | uint64_t  | 8    | Count of key-value metadata pairs    |
| image_offset  | uint64_t  | 8    | Offset of device image data          |
| image_size    | uint64_t  | 8    | Size of device image data            |

**Metadata String Table:**
Key-value pairs (null-terminated strings) encode target triple, architecture (e.g., `sm_90`), and other properties.

**Multiple Targets:**
Multiple entries are concatenated after the header. Each entry is self-describing; tools can locate all entries from any position using offset fields.

### `clang-offload-wrapper`

Inputs: device binary files (ELF, cubin, PTX, SPIR-V, etc.)
Output: LLVM bitcode file (`.bc`) for the host

The generated bitcode contains:
- A global symbol (offload descriptor) pointing to all device binaries and their metadata
- Registration functions placed in `@llvm.global_ctors` (called at process init) and `@llvm.global_dtors`
- Registration invokes the appropriate runtime: `__tgt_register_lib` for OpenMP, `__cudaRegisterFatBinary` for CUDA, `__hipRegisterFatBinary` for HIP

OpenMP's runtime structures:
```c
struct __tgt_offload_entry { void* addr; char* name; size_t size; int32_t flags; int32_t reserved; };
struct __tgt_device_image   { void* ImageStart; void* ImageEnd; __tgt_offload_entry* EntriesBegin; __tgt_offload_entry* EntriesEnd; };
struct __tgt_bin_desc        { int32_t NumDeviceImages; __tgt_device_image* DeviceImages; ... };
```

### `clang-linker-wrapper`

Invoked as a wrapper around the system linker (`ld`, `lld`). Pipeline:

1. Parse all linker inputs for `.llvm.offloading` sections
2. Extract device binary blobs (identified by `0x10FF10AD` magic)
3. Group extracted binaries by target triple + architecture
4. Invoke device linker per group (e.g., `lld --target=nvptx64` or `lld --target=amdgcn-amd-amdhsa`)
5. For LTO targets (AMDGPU lacks relocatable ELF): link bitcode via LTO pass
6. Wrap linked device images via `clang-offload-wrapper` → registration bitcode
7. Compile registration bitcode → host `.o`
8. Merge that `.o` into the final host link invocation
9. Strip `.llvm.offloading` sections from output (`SHF_EXCLUDE` handles this)

Multi-target compatibility rule: device images are linked together only when `(target_triple, arch)` match. Exception: `arch=generic` links with any image sharing the same triple.

**Key flags:**
- `--device-linker=<triple>=<value>` — pass extra args to a specific target's device linker
- `--override-image=<kind=file>` — inject a custom pre-built device binary
- `--relocatable` — run device linking eagerly during `-r` (relocatable link) step
- `--opt-level=<O0..O3>` — device LTO optimization level

---

## Summary: Comparative Analysis

| Dimension                     | CUDA fatbin          | HIP fatbin            | LLVM .llvm.offloading  | DPC++ SYCL fat binary  | hetGPU (research)     |
|-------------------------------|----------------------|-----------------------|------------------------|------------------------|-----------------------|
| Container magic               | `0xBA55ED50`         | `__CLANG_OFFLOAD_BUNDLE__` | `0x10FF10AD`       | clang-offload-wrapper  | hetIR text/binary     |
| Multi-SM per vendor           | Yes (n cubins + PTX) | Yes (n gfx images)    | Yes (n entries)        | Yes                    | No (1 abstract image) |
| Cross-vendor in one binary    | No                   | No                    | No (one runtime model) | Yes (NVPTX+AMDGCN+SPIR-V) | Yes (4 vendors)   |
| Runtime JIT                   | PTX fallback         | No (SASS only)        | No (pre-linked ELF)    | SPIR-V→native JIT      | Always (hetIR→ISA)    |
| ELF section                   | `.nv_fatbin`         | `.hip_fatbin`         | `.llvm.offloading`     | `.llvm.offloading`     | N/A (loaded separately) |
| Standardized in upstream LLVM | Partially (LLVM can emit) | Yes (clang)      | Yes                    | Intel fork only        | Research prototype     |
| Compressed format             | Proprietary          | CCOB (zlib/zstd)      | N/A                    | N/A                    | N/A                   |

---

## Implications for the Poster: Vendor-Agnostic Runtime Dispatch

The fat binary landscape reveals a clear gap:

1. **Within-vendor fat binaries are mature** (CUDA sm_70→sm_120, HIP gfx906→gfx1100) and well-understood.
2. **Cross-vendor fat binaries have no standard container.** DPC++ achieves it by piggy-backing on the LLVM offload wrapper with multiple targets, but requires Intel's fork and adds SYCL semantics overhead.
3. **The `.llvm.offloading` section + `llvm-offload-binary` format is the most promising upstream hook** for a custom cross-vendor fat binary: it is format-agnostic (any image kind), has flexible string-map metadata, and the `clang-linker-wrapper` already supports multi-target extraction.
4. **A lightweight dispatch layer could extend `llvm-offload-binary`** to bundle an NVPTX cubin, an AMDGCN ELF, a SPIR-V blob, and x86 host code with a unified metadata header — then implement a thin runtime that replaces `__tgt_register_lib` with device-querying logic to select the right image.
5. **HetGPU's hetIR approach** (2025) represents the extreme end: single abstract IR with runtime JIT to any backend, at the cost of 50–200ms per-kernel JIT latency and no access to vendor-specific tensor cores.

The sweet spot for the poster contribution: a **prototype cross-vendor fat binary container** built on the `llvm-offload-binary` format (magic `0x10FF10AD`), extending its metadata scheme to carry vendor discriminators, paired with a thin runtime dispatcher that queries `cudaGetDeviceProperties` / `hipGetDeviceProperties` / `clGetPlatformIDs` at startup and loads the correct image — avoiding both the full DPC++/SYCL abstraction overhead and the hetGPU JIT latency.

---

## References and Sources

- NVIDIA CUDA Compiler Driver documentation: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- nvFatbin API documentation (CUDA 13.2): https://docs.nvidia.com/cuda/nvfatbin/index.html
- CUDA Binary Utilities (cuobjdump, nvdisasm): https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- CUDA fat binary JIT caching (NVIDIA Technical Blog): https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
- nvcc -gencode flag guide: https://kaixih.github.io/nvcc-options/ | https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
- Clang HIP Support documentation: https://clang.llvm.org/docs/HIPSupport.html
- Clang Offload Bundler specification: https://clang.llvm.org/docs/ClangOffloadBundler.html
- LLVM Offload Binary format: https://llvm.org/docs/CommandGuide/llvm-offload-binary.html
- LLVM Offloading Design and Internals: https://clang.llvm.org/docs/OffloadingDesign.html
- Clang Linker Wrapper: https://clang.llvm.org/docs/ClangLinkerWrapper.html
- Clang Offload Wrapper (ROCm): https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/clang/html/ClangOffloadWrapper.html
- Intel DPC++ Compiler and Runtime Architecture: https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html
- LLVM RFC: SPIR-V as vendor-agnostic GPU IR: https://discourse.llvm.org/t/rfc-spirv-ir-as-a-vendor-agnostic-gpu-representation/85115
- HetGPU paper (2025): https://arxiv.org/html/2506.15993v1
- fatbinary.h header (reverse-engineered): https://github.com/chengenbao/cuda_headers/blob/master/fatbinary.h
- fatbinary_section.h header: https://github.com/chengenbao/cuda_headers/blob/master/fatbinary_section.h
- CUDA fat binary deep dive (LLVM Phab D120932): https://reviews.llvm.org/D120932
- AMD AMDGPU Backend User Guide: https://llvm.org/docs/AMDGPUUsage.html
