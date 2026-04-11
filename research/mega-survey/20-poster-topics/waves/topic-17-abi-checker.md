# Topic 17: GPU Kernel ABI Compatibility Checker

**Survey:** 20 LLVM/MLIR Poster Topics
**Topic ID:** 17
**Config name:** gpu-kernel-abi-compat-checker
**Title:** llvm-offload-abi-check: Verifying Substitutability of Fat Binary Kernel Variants
**Persona:** compiler safety / binary verification engineer
**Date:** 2026-04-07

---

## Gap

A CUDA fat binary carries an `sm_75` cubin and an `sm_89` cubin for the same kernel.
A HIP fat binary carries `gfx906` and `gfx1100` images.
A libkdl `.kdl` bundle carries CUBIN, HSACO, and CPU `.o` variants of `sgemm`.

The LLVM toolchain and all GPU runtimes assume these variants are **semantically
substitutable** — that dispatching the `sm_89` cubin when the host sends 48 bytes of
kernel arguments is equivalent to dispatching `sm_75`. Nothing verifies this assumption.

The failure modes when it is false are severe and silent:

| Failure scenario | Symptom | Detection point |
|-----------------|---------|----------------|
| `sm_89` variant has one more pointer argument than `sm_75` | Wrong memory written | Wrong answer in output (no crash) |
| AMD variant packs a `float3` differently (alignment 16 vs 4) | Partial argument corruption | Silent NaN or garbage value |
| `sm_90a` variant restructured shared memory layout, 96 KB required | Launch failure or silent misfire on hardware with 48 KB | Sporadic test failure |
| CPU fallback passes `float *` where GPU variant expects `int *` | Undefined behaviour on CPU | Crash or wrong result |
| Version mismatch: v2 bundle updated `sm_89` cubin but not `sm_75` | Divergent correctness | Output matches on sm_89, fails on sm_75 only |

None of these errors is caught by the LLVM toolchain today.
None is caught by any vendor runtime.
The CUDA driver performs architecture compatibility checking (SM version >= required SM)
but performs **no argument-type or argument-count checking** when selecting among
cubins in a fat binary.
AMD's HSA runtime performs GFX target-ID + xnack/sramecc flag matching but no
kernarg layout checking.

**The core issue:** Every GPU fat binary encodes enough metadata to answer "is this
kernel's argument signature the same across variants?" — but no tool reads that
metadata for a cross-variant consistency check.

### What the metadata actually contains today

**CUDA cubin `.nv.info` section (EIATTR attributes, per kernel):**

```
EIATTR_REGCOUNT        (0x1F)  registers per thread
EIATTR_MAX_THREADS     (0x05)  max threads per block
EIATTR_REQNTID         (0x10)  required block dims (reqntid pragma)
EIATTR_MIN_STACK_SIZE  (0x12)  minimum stack frame bytes
EIATTR_SHMEM_PARAM_SIZE        static shared memory bytes
EIATTR_CBANK_PARAM_SIZE        constant bank param size (proxy for total kernarg size)
EIATTR_PARAM_SIZE      (0x17)  total parameter buffer size in bytes
```

The `EIATTR_PARAM_SIZE` attribute encodes the total size of the kernel's parameter
buffer — i.e., the total byte width of all kernel arguments after alignment padding.
This is the direct ABI fingerprint: two cubins with different `EIATTR_PARAM_SIZE`
values **cannot be substitutes** for the same kernel call site.

`cuobjdump --dump-elf-raw` exposes all EIATTR note records from `.nv.info.*` sections.
There is no LLVM tool that reads these.

**AMD HSACO kernel descriptor (64-byte struct at symbol value, Code Object V5):**

```c
struct amdgpu_kernel_descriptor {
    uint32_t group_segment_fixed_size;  // LDS bytes (ABI-affecting: alters launch)
    uint32_t private_segment_fixed_size; // scratch bytes per lane
    uint32_t kernarg_size;               // total kernel argument buffer size
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t  reserved2[6];
};
```

`kernarg_size` is the direct AMD equivalent of CUDA's `EIATTR_PARAM_SIZE`.
Two HSACO images with different `kernarg_size` values cannot be substitutes.
`llvm-nm` and `llvm-readelf` do not parse kernel descriptors — they treat the
kernel symbol's memory as opaque executable code.

**OffloadBinary StringMap (LLVM 20+ standard container):**

The standard OffloadBinary string table carries `triple` and `arch` per image.
Topic 07 proposes adding `kernarg_size`, `shared_mem_bytes`, and `requires_features`
as standard keys. Until that RFC lands, none of these values appear in the string table.
A cross-variant ABI checker must therefore parse the embedded binary payloads directly.

**SPIR-V / PTX:**

- PTX: `OpParam` declarations after each `.entry` directive list the parameter
  list with explicit type and size. Two `.entry` declarations of the same name
  with different parameter lists cannot be substitutes.
- SPIR-V: `OpFunctionParameter` instructions within an `OpEntryPoint` function
  encode per-argument type references. Argument count and scalar-type widths are
  directly verifiable.

---

## Proposal

**`llvm-offload-abi-check`**: a new LLVM tool that reads a multi-image OffloadBinary
container (`.o` file with `.llvm.offloading` section, or standalone `.kdl` bundle)
and checks whether all images carrying the same kernel name have **ABI-compatible
argument signatures**.

### Core operation

```
$ llvm-offload-abi-check gemm.o

Checking kernel: sgemm_nt_128x128
  image 0: nvptx64-nvidia-cuda / sm_75  (CUBIN)
    kernarg_size = 48 bytes
    arg[0]: ptr, 8 bytes, offset  0
    arg[1]: ptr, 8 bytes, offset  8
    arg[2]: ptr, 8 bytes, offset 16
    arg[3]: i32, 4 bytes, offset 24
    arg[4]: i32, 4 bytes, offset 28
    arg[5]: f32, 4 bytes, offset 32
    arg[6]: f32, 4 bytes, offset 36
    [padding: 8 bytes to align total to 48]
  image 1: nvptx64-nvidia-cuda / sm_89  (CUBIN)
    kernarg_size = 48 bytes
    [MATCH]
  image 2: amdgcn-amd-amdhsa / gfx1100  (HSACO)
    kernarg_size = 56 bytes
    [MISMATCH] kernarg_size 56 != reference 48

ERRORS:
  sgemm_nt_128x128: ABI mismatch between image 0 (sm_75) and image 2 (gfx1100)
  kernarg_size differs: 48 vs 56. Cross-vendor substitution will corrupt arguments.

$ echo $?  # exits 1 on ABI violation
1
```

The tool exits zero if all same-named kernels across all images share the same
`kernarg_size`. It exits non-zero on any mismatch, suitable for CI integration.

### What constitutes GPU kernel ABI

The GPU kernel ABI is the calling convention between the host-side dispatch site
and the device-side kernel entry point. It has five dimensions:

| ABI dimension | CUDA encoding | AMD encoding | SPIR-V encoding |
|---------------|--------------|--------------|-----------------|
| Total argument buffer size | `EIATTR_PARAM_SIZE` in `.nv.info` | `kernarg_size` in kernel descriptor | Sum of `OpFunctionParameter` type widths |
| Per-argument type | `.nv.info` attribute records (partially) | `NT_AMDGPU_METADATA` msgpack kernel args array | `OpTypePointer` / `OpTypeInt` / `OpTypeFloat` on each parameter |
| Per-argument alignment | Implicit from type (4/8-byte natural) | Explicit in msgpack `.offset` field per arg | Implicit from SPIR-V type rules |
| Shared memory layout | `EIATTR_SHMEM_PARAM_SIZE` | `group_segment_fixed_size` | N/A (dynamic via `OpVariable addrspace(3)`) |
| Entry point name | ELF symbol name (`.text` section) | ELF symbol name (`STT_AMDGPU_HSA_KERNEL`) | `OpEntryPoint` name literal |

Two kernel variants are ABI-substitutable if and only if they agree on all five
dimensions for every kernel with the same entry-point name.

**Critical case — sm_89 vs sm_90a substitutability:**
`sm_90a` kernels using WGMMA (Hopper Warp Group Matrix Multiply-Accumulate) or TMA
(Tensor Memory Accelerator) have fundamentally different argument structures from
their `sm_89` counterparts — descriptor pointers replace raw pointers, and shared
memory is allocated differently. The `EIATTR_PARAM_SIZE` values will differ. This
is exactly what `llvm-offload-abi-check` would catch before deployment.

### Extended mode: argument-level detail

```
$ llvm-offload-abi-check --args gemm.o
```

In extended mode the tool parses per-argument type information from:
- AMD: the `NT_AMDGPU_METADATA` MsgPack note record, which contains a `.args` array
  with per-argument `.size`, `.offset`, `.value_kind` (global_buffer / by_value /
  hidden_global_offset_x / ...) and `.type_name` strings in Code Object V5.
- CUDA: `.nv.info` EIATTR records (argument-level type data is limited in CUDA;
  the kernel parameter list is more reliably recovered from PTX `.param` directives).
- SPIR-V: `OpFunctionParameter` type chain via `OpTypePointer` / scalar types.

The extended mode reports per-argument mismatches:

```
  arg[2]: MISMATCH
    sm_75:   ptr (global), 8 bytes, offset 16
    gfx1100: ptr (global), 8 bytes, offset 24   [padding difference]
```

### ABI equivalence definition (formal)

Two variants V1 and V2 of kernel `K` are **ABI-compatible** iff:

1. `kernarg_size(V1) == kernarg_size(V2)` — total argument buffer identical
2. For each argument index `i`: `size(arg_i, V1) == size(arg_i, V2)` and
   `offset(arg_i, V1) == offset(arg_i, V2)` — individual argument positions match
3. `count(args, V1) == count(args, V2)` — argument count is the same
4. For pointer arguments: address space class is compatible
   (global/constant/shared classify differently across vendors; the check should
   flag cross-space mismatches as warnings, not hard errors)

Condition (1) is a necessary but insufficient condition for (2)+(3). A quick
check mode tests only (1); the `--args` mode tests all four.

### Connection to libkdl / OffloadBinary

libkdl (`experiments/prototype/src/kdl.c`) stores a `contract_offset` per variant
pointing into a JSON string table. The contract encodes `num_args`, `arg0_size`,
`arg0_kind`, `arg0_name`, etc. (see `kdl_kernel_get_arg_info`, lines 2758–2790).

This contract is hand-authored when assembling a `.kdl` bundle. `llvm-offload-abi-check`
would automate the verification that the hand-authored contract is consistent with
the embedded binary metadata — and flag cases where a bundle was updated for one
target but not another.

The tool also serves as a CI gate for libkdl bundle publishing:

```makefile
# In libkdl's Makefile publish target:
llvm-offload-abi-check gemm.kdl || (echo "ABI mismatch — bundle is broken" && exit 1)
```

---

## Evidence

### Gap evidence: no existing tool performs cross-variant ABI checking

| Tool | What it checks | Cross-variant? | ABI signature? |
|------|---------------|----------------|----------------|
| `cuobjdump --list-elf` | Lists cubins by arch | No | No |
| `cuobjdump -res-usage` | Per-kernel resource counts (regs, shmem) | No | No |
| `llvm-readelf -s file.hsaco` | ELF symbol table | No | No |
| `llvm-offload-binary` | Extracts images to disk | No | No |
| `llvm-nm file.cubin` | Symbol names (type `?` for kernel symbols) | No | No |
| AMD `rocm_agent_enumerator` | Device enumeration | N/A | No |
| LLVM `areTargetsCompatible()` | xnack/sramecc flag string match | Same-vendor only | No |

Zero tools perform cross-variant ABI signature comparison across images in a fat binary.
The LLVM GitHub issue tracker (searched April 2026) returns no existing issue requesting
this functionality. This is a clean, unaddressed gap.

### Evidence that the problem causes real failures

**Case 1: PyTorch CUDA kernel ABI mismatch with sm_90a**

When NVIDIA introduced sm_90a with architecture-specific features (WGMMA, TMA), kernels
compiled for sm_90a have different argument structures: TMA descriptors are passed as
64-byte opaque values rather than raw pointers. PyTorch's Triton-based Flash Attention
had bugs where the sm_90 and sm_90a variants had incompatible argument passing, causing
silent wrong results on H100 (Hopper) hardware. The bugs were caught by correctness
tests, not by any binary compatibility tool. (Source: wave-06-kernel-binary-abi.md §3.)

**Case 2: AMD HSACO xnack mode interaction with kernarg layout**

Code Object V2/V3 HSACO binaries with and without `xnack+` feature flags can have
different kernarg layouts in kernels that use demand-paging features. The AMD ROCm
runtime's `areTargetsCompatible()` checks the xnack flag string but does not compare
kernarg layouts. (Source: wave-06-kernel-binary-abi.md §6.)

**Case 3: CPU fallback alignment mismatch in libkdl bundles**

libkdl's CPU backend uses `dlopen` + `dlsym` on a native `.o`. The CPU calling
convention for structs (e.g., `float3` with alignment 16 on x86 ABI) differs from
the GPU kernarg convention where `float3` is packed as 12 bytes. A bundle where
the CPU variant has different argument packing than the GPU variants would silently
corrupt the third argument. libkdl's `kdl_arg_info` struct (kdl.h line 556–560)
tracks `size_bytes`, `offset`, and `kind` — but only if the contract JSON was
correctly hand-authored.

### LLVM infrastructure that already provides the raw data

The tool requires no new data sources — it assembles existing parsers:

| Data needed | LLVM API | Upstream status |
|------------|----------|-----------------|
| OffloadBinary image iteration | `llvm::object::OffloadBinary::create()` | Stable since LLVM 15 |
| CUBIN ELF + `.nv.info` sections | `llvm::object::ELFObjectFile` + section iteration | Stable |
| HSACO kernel descriptor | `llvm::object::ELFObjectFile` + symbol value offset | Stable; descriptor struct in `llvm/lib/Target/AMDGPU/Utils/` |
| AMDGPU metadata msgpack | `llvm::AMDGPU::HSAMD::Kernel::Arg` msgpack schema | Upstream in `llvm/lib/Support/AMDGPUMetadata.cpp` |
| SPIR-V `OpEntryPoint` + `OpFunctionParameter` | Raw word-stream scan, no external dep | No dep needed |
| PTX `.param` directive scan | `llvm::MemoryBuffer` + text scan | No dep needed |

### llvm-ifs as structural precedent

LLVM ships `llvm-ifs` (Interface Stub Files) — a tool that generates `.ifs` files
describing the public ABI surface (symbols, types, architecture) of a shared library,
and a `--verify` mode that checks a binary's exported symbols against a reference
`.ifs` stub.

`llvm-ifs --verify` is the CPU-side ABI compatibility checker.
`llvm-offload-abi-check` is the GPU kernel-side ABI compatibility checker.

The precedent demonstrates that the LLVM community accepts binary-level ABI verification
tools as first-class toolchain contributions. The libc++ ABI test suite (`libcxx/test/abi/`)
additionally provides a regression-test pattern for ABI stability: a reference manifest of
type layouts checked against each release. `llvm-offload-abi-check` is the same concept
applied to GPU kernel argument signatures across fat binary variants.

---

## Feasibility

### Technical feasibility: high

All parsing components are available in the LLVM monorepo or require only a
lightweight custom parser:

| Component | LOC | Risk |
|-----------|-----|------|
| OffloadBinary image iterator | ~50 | Zero — API is stable |
| CUBIN ELF parser (`.nv.info` section + EIATTR records) | ~150 | Low — semi-documented; `cuobjdump` confirms the schema |
| HSACO ELF parser (kernel descriptor at symbol value offset) | ~100 | Low — struct is documented in LLVM AMDGPU user guide |
| AMDGPU metadata msgpack (`.args` array) | ~120 | Low — `llvm/lib/Support/AMDGPUMetadata.cpp` already parses this |
| SPIR-V `OpFunctionParameter` scan | ~80 | Zero — 5-word instruction format, no library needed |
| PTX `.param` text scanner | ~60 | Zero — simple regex scan |
| ABI equivalence comparator | ~100 | Low — field-by-field struct comparison |
| Output formatter + exit code | ~80 | Zero |
| **Total** | **~740** | All deps in LLVM monorepo |

### Prototype effort: 3–4 days

A minimal prototype covering CUBIN (`kernarg_size` from `.nv.info`) and HSACO
(`kernarg_size` from kernel descriptor) is achievable in two days. Adding SPIR-V
and the `--args` extended mode adds two more days.

The hardest piece is the CUBIN `.nv.info` parser. The format uses a note-record
scheme with attribute codes (EIATTR_*) defined in NVIDIA's `ptxas` documentation
and confirmed by `cuobjdump` output. The key attribute `EIATTR_PARAM_SIZE` (0x17)
appears in every cubin with a kernel entry. The parser is ~80 lines: iterate
`.nv.info.*` sections, walk note records, match the attribute code for the target
kernel symbol.

### Demo for poster

Run `llvm-offload-abi-check` on a libkdl `.kdl` bundle that intentionally has an
ABI mismatch between the CUDA and CPU variants:

```
$ llvm-offload-abi-check --args sgemm_mismatch.kdl
Checking kernel: sgemm_nt_128x128
  image 0: nvptx64-nvidia-cuda / sm_89  kernarg_size=48  [reference]
  image 1: native-cpu / x86_64          kernarg_size=52  [MISMATCH: +4 bytes, alignment difference]
ERRORS: 1 ABI violation found.
$ echo $?
1
```

Then run on the correctly assembled bundle to show the tool passing:

```
$ llvm-offload-abi-check sgemm_correct.kdl
Checking kernel: sgemm_nt_128x128: 3 images — [MATCH]
Checking kernel: sgemm_tt_128x128: 3 images — [MATCH]
No ABI violations found.
$ echo $?
0
```

Both demos run on the GTX 1650 + CPU machine used for libkdl development.

### Risk table

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUBIN `.nv.info` format undocumented | Medium | `EIATTR_PARAM_SIZE` is confirmed by `cuobjdump --dump-elf-raw` on any cubin; structure is recoverable empirically even without formal spec |
| CUDA 13+ changes `.nv.info` schema | Low | The tool targets current CUDA 12/13; version-check the ELF and warn |
| AMD msgpack arg metadata absent in old code objects (V2/V3) | Medium | Fall back to `kernarg_size` from the kernel descriptor — always present in V2+ |
| Tool classified as "too small for a poster" | Medium | Frame as "enabling infrastructure for fat binary safety" — pair with libkdl poster to show the CI workflow |
| Overlap with topic-08 (llvm-offload-nm) | Low | llvm-offload-nm lists symbols and resource counts; this tool specifically checks ABI *cross-variant consistency*. Distinct focus. |

---

## Upstream Path

### Stage 1 — Prototype tool (for Dublin poster)

```
llvm/tools/llvm-offload-abi-check/
  llvm-offload-abi-check.cpp      (~740 LOC)
  CMakeLists.txt
llvm/docs/CommandGuide/llvm-offload-abi-check.rst
```

Pattern: follow `llvm/tools/llvm-nm/` structure. The tool is self-contained —
no new libraries, no new IR passes, no new dialect changes.

### Stage 2 — Upstream RFC + PR

RFC subject: `[RFC] llvm-offload-abi-check: ABI compatibility verification for GPU fat binaries`

CC: Joseph Huber (offload maintainer, AMD), Johannes Doerfert (LLNL), AMDGPU
backend owners, NVPTX backend owners.

Framing for the RFC:
1. Fat binaries implicitly assert that all images are substitutable — the runtime
   picks one, calls it with the same argument buffer. No tool checks this.
2. `llvm-ifs --verify` already validates CPU shared library ABI; this is the GPU
   equivalent, using binary-level metadata rather than IR-level stubs.
3. Enables CI pipelines to catch ABI breaks introduced by one-sided kernel updates
   (e.g., updating the sm_89 cubin but forgetting to update gfx1100).
4. Implementation is additive and read-only — no format changes, no runtime changes.

### Patch sequence

| Step | Patch | Reviewer group |
|------|-------|----------------|
| 1 | Tool skeleton + OffloadBinary iterator | llvm/Object owners |
| 2 | CUBIN `.nv.info` EIATTR parser | NVPTX backend owners |
| 3 | HSACO kernel descriptor + msgpack args parser | AMDGPU backend owners |
| 4 | SPIR-V `OpFunctionParameter` scanner | SPIR-V / MLIR owners |
| 5 | PTX `.param` text scanner | NVPTX backend owners |
| 6 | ABI comparator + exit code + docs | llvm/tools owners |

Steps 2–5 are independent and can be reviewed in parallel after step 1 merges.

### Long-term integration opportunity

Once the OffloadBinary StringMap is extended with standard capability keys (Topic 07),
`llvm-offload-abi-check` can shift from parsing embedded binary payloads to reading
the string table directly. The binary-payload parsing path then serves as a fallback
for bundles built without the standard keys — making the tool backward-compatible with
all existing fat binaries.

Additionally, `areTargetsCompatible()` in `llvm/lib/Object/OffloadBinary.cpp` could
call `llvm-offload-abi-check`'s core comparison logic as a library, surfacing ABI
warnings at link time when `clang-offload-packager` assembles a bundle.

---

## Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Novelty** | 8/10 | No existing tool does cross-variant GPU kernel ABI checking. The gap is real and unaddressed. The concept (binary-level ABI verification) is well-understood for CPUs via `llvm-ifs`, making the GPU extension immediately legible. |
| **Upstream impact** | 7/10 | Enables CI-level fat binary safety checking for every project that ships multi-variant GPU kernels (CUTLASS, cuBLAS, Triton AOT cache, libkdl). Tooling contribution, not a language or dialect change — bounded scope but broad reach. |
| **Prototype alignment** | 9/10 | libkdl's `kdl_kernel_get_arg_info` (kdl.c:2758–2790) and `kdl_arg_info` struct (kdl.h:556–560) already implement the cross-vendor argument model. The prototype directly demonstrates the data this tool would check, making the poster story self-contained. |
| **Implementation risk** | 3/10 | All parsing dependencies are upstream and stable. `.nv.info` format is semi-documented but empirically verifiable. No ABI-breaking changes to any format. |
| **Reviewer friction** | 3/10 | Read-only inspection tool — no format changes, no runtime changes, no new pass infrastructure. The `llvm-ifs` precedent makes the contribution category immediately recognizable. |
| **Poster fit** | 7/10 | Clear before/after story: "fat binary assumes variants are equivalent — here is the first tool that checks." Table of ABI dimensions across CUDA/AMD/SPIR-V makes a strong visual. Exit-code CI integration is immediately practical. Pairs naturally with libkdl (the dispatcher) to complete the "build, check, dispatch" workflow. |
| **Composite** | 7.8/10 | Strong supporting-contribution poster. Best presented alongside Topic 05 (offload-ld / libkdl) or Topic 07 (OffloadBinary metadata keys) to show a unified fat binary toolchain story. |

---

## Pitch

Every GPU fat binary is an implicit contract: the `sm_75` cubin and the `sm_89` cubin
implement the same kernel, accept the same argument buffer layout, and produce the same
results on their respective hardware. The CUDA driver, HIP runtime, and LLVM liboffload
all enforce the architecture half of this contract — they pick the right cubin for the
right GPU. No tool enforces the ABI half. If the `sm_89` cubin was compiled after a
kernel refactor that changed argument order, and the `sm_75` cubin was not rebuilt, the
runtime will silently dispatch the stale cubin with a mismatched argument buffer. The
result is wrong answers, not a crash or error.

Every CUDA cubin encodes `EIATTR_PARAM_SIZE` in its `.nv.info` section — the total byte
width of the kernel argument buffer. Every AMD HSACO encodes `kernarg_size` in a 64-byte
kernel descriptor at the kernel's ELF symbol address. Every SPIR-V module lists
`OpFunctionParameter` types for each entry point. The data exists. Nothing reads it for
consistency.

We propose `llvm-offload-abi-check`: a ~740-line LLVM tool that iterates all images in
an OffloadBinary container, finds same-named kernels across CUBIN, HSACO, SPIR-V, and
PTX images, compares their `kernarg_size` values (and optionally their per-argument
offsets and types from AMD msgpack metadata), and exits non-zero on any mismatch. It is
the GPU equivalent of `llvm-ifs --verify`: a CI-integrable binary-level ABI regression
check. The tool catches precisely the class of bug that the sm_90a transition introduced
in practice — architecture-specific argument restructuring that breaks substitutability.
A working prototype runs against libkdl `.kdl` bundles on GTX 1650 + CPU at the poster.

---

## Related Work / Prior Art

- `llvm-ifs` (`llvm/tools/llvm-ifs/`) — CPU shared library ABI checker via Interface Stub
  Files; the structural precedent for this tool
- libc++ ABI test suite (`libcxx/test/abi/`) — type-layout regression manifest pattern
- `cuobjdump -res-usage` — NVIDIA-proprietary per-kernel resource inspector (no cross-variant
  comparison, no cross-vendor support)
- `llvm::object::OffloadBinary` — container API (D122069, stable since LLVM 15)
- `areTargetsCompatible()` — `llvm/lib/Object/OffloadBinary.cpp`; checks arch string
  compatibility (xnack/sramecc) but not kernarg layout
- AMDGPU Code Object V5 kernel descriptor spec — `llvm/docs/AMDGPUUsage.rst`
- CUDA Binary Utilities: EIATTR attributes in `.nv.info` — NVIDIA docs v13.2
- Topic 07 (this survey) — OffloadBinary standard capability metadata keys: if merged,
  `llvm-offload-abi-check` can use string-table keys instead of binary-payload parsing
- Topic 08 (this survey) — `llvm-offload-nm`: lists kernel symbols and resource counts;
  orthogonal to this tool's cross-variant ABI consistency focus
- libkdl `kdl_kernel_get_arg_info` — prototype cross-vendor argument metadata accessor
  (kdl.c:2758–2790, kdl.h:556–560); the prototype-level realization of this proposal
- wave-06-kernel-binary-abi.md — full binary compatibility research notes for this project

---

*Generated: 2026-04-07 | Research basis: CUDA Binary Utilities 13.2 (EIATTR attributes),
LLVM AMDGPUUsage.rst (Code Object V5 kernel descriptor), OffloadBinary.h/cpp (LLVM 20),
llvm-ifs tool precedent, libkdl kdl.h:556–560 / kdl.c:2758–2790, wave-06-kernel-binary-abi.md,
topic-07-offloadbinary-metadata.md, topic-08-offload-nm.md*
