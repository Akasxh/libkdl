# Topic 08: llvm-offload-nm — Symbol Inspection Tool for GPU Fat Binaries

**Survey:** 20 LLVM/MLIR Poster Topics
**Topic ID:** 08
**Config name:** offload-nm-kernel-symbol-inspection
**Title:** llvm-offload-nm: What's Inside Your GPU Fat Binary?
**Persona:** toolchain completionist / binary inspector
**Date:** 2026-04-07

---

## Gap

LLVM ships `llvm-nm`, `llvm-readelf`, and `llvm-objdump` for CPU ELF inspection.
None of them answer the question a GPU developer actually needs:
"which kernel functions are compiled into this fat binary, for which targets,
and what are their resource requirements?"

The specific breakdown by layer:

### Layer 1: OffloadBinary container — `llvm-offload-binary` cannot inspect

`llvm-offload-binary` (the only official LLVM tool for the `.llvm.offloading`
section format) has exactly two modes: **bundle** (pack device objects into a
container) and **extract** (unpack them). It has no `-dump`, no `-list`, no
`-print-symbols` flag. Its stdout is empty during normal operation; it only
writes files to disk. Source confirmation: `llvm-offload-binary.cpp` produces
no human-readable output and has no inspection code path.

The OffloadBinary C++ API (`llvm/include/llvm/Object/OffloadBinary.h`) exposes
per-image metadata — `getImageKind()`, `getOffloadKind()`, `getTriple()`,
`getArch()`, `strings()` — but these are **container-level** attributes (which
target, which offload model, which architecture string). The API explicitly
cannot extract kernel names or symbol information from the embedded device
binary payloads; that requires a separate per-format parser.

What you can get today from the LLVM toolchain to inspect a fat binary:

```
$ llvm-readelf -WS object.o   # shows .llvm.offloading section exists, byte size only
$ llvm-offload-binary object.o  # extracts payloads to disk — then what?
$ llvm-nm extracted.cubin        # ERROR: not an ELF (CUBIN uses ELF but llvm-nm
                                  # does not understand CUBIN-specific section layout)
```

There is no single command to answer: "what kernels are in `gemm.o`, which SM
versions are they compiled for, how many registers do they use?"

### Layer 2: Per-format kernel symbol inspection requires vendor tools

**CUBIN (NVIDIA):** `cuobjdump --list-elf fatbin.o` lists embedded ELF cubin
files by architecture (`sm_89`, `sm_90a`). `cuobjdump -res-usage` gives
per-kernel resource counts: registers, shared memory bytes, stack frame size,
constant memory. `nvdisasm` provides disassembly. These are NVIDIA-proprietary
tools, unavailable on non-CUDA systems and absent from the LLVM toolchain.

**HSACO (AMD):** HSACO is a valid ELF — `llvm-nm` can parse its symbol table.
Kernel symbols use symbol type `STT_AMDGPU_HSA_KERNEL` (value 10, AMDGPU-
specific extension to the ELF ST_TYPE field). `llvm-nm` prints these as type
`?` or silently omits them because it does not recognize the type code.
`llvm-readelf -s` will list them without the type annotation. The kernel
descriptor (a 64-byte struct preceding each kernel in `.text`) contains
`granulated_workitem_vgpr_count`, `granulated_wavefront_sgpr_count`,
`group_segment_fixed_size` (LDS bytes), `private_segment_fixed_size`
(scratch bytes), and `kernarg_size` — none of which `llvm-nm` reads.
Actual resource counts require parsing the kernel descriptor at the symbol's
value offset.

**SPIR-V:** `OpEntryPoint` instructions in the SPIR-V word stream enumerate
kernel entry points with `ExecutionModel=Kernel`, entry point function ID,
and a name string. No LLVM tool iterates these. `spirv-dis` (SPIRV-Tools)
disassembles the full module; there is no `spirv-nm`-equivalent that prints
a compact symbol table.

**Multi-vendor bundle:** A real-world LLVM 20 fat object embeds CUBIN +
HSACO + SPIR-V images in a single `.llvm.offloading` section. No tool
simultaneously lists all kernel entry points across all embedded images.

### Layer 3: Confirmed gap — no LLVM GitHub issue requests this tool

A search of the LLVM issue tracker (April 2026) returns zero issues requesting
`llvm-offload-nm`, kernel symbol listing for fat binaries, or an inspection
mode for OffloadBinary containers. The gap has not been named or filed.

---

## Proposal

**`llvm-offload-nm`**: a new LLVM tool that reads an object file or standalone
OffloadBinary container and prints a symbol table across all embedded GPU images,
analogous to `llvm-nm` for CPU ELFs.

### Target output format

```
$ llvm-offload-nm gemm.o

gemm.o [.llvm.offloading]:
  image 0: nvptx64-nvidia-cuda / sm_89  (CUBIN)
    0000000000000000  K  sgemm_nt_128x128
    0000000000000040  K  sgemm_tt_128x128
    0000000000000080  K  sgemm_nt_64x64
  image 1: nvptx64-nvidia-cuda / sm_90a  (CUBIN)
    0000000000000000  K  sgemm_nt_128x128
    0000000000000040  K  sgemm_tt_128x128
  image 2: amdgcn-amd-amdhsa / gfx1100  (HSACO)
    0000000000000000  K  sgemm_nt_128x128
    0000000000000040  K  sgemm_tt_128x128
  image 3: spirv64-intel-unknown / generic  (SPIRV)
    [entry]  sgemm_nt_128x128
    [entry]  sgemm_tt_128x128
```

Symbol type column: `K` = GPU kernel (analogous to `T` for text symbol in nm).

### Extended resource output (`-r` / `--resources`)

```
$ llvm-offload-nm -r gemm.o

  image 0: nvptx64-nvidia-cuda / sm_89  (CUBIN)
    sgemm_nt_128x128   vgpr=128  sgpr=32  shmem=16384  stack=0  kernargs=48
    sgemm_tt_128x128   vgpr=96   sgpr=28  shmem=8192   stack=0  kernargs=48
  image 2: amdgcn-amd-amdhsa / gfx1100  (HSACO)
    sgemm_nt_128x128   vgpr=128  sgpr=32  lds=16384    scratch=0  kernargs=48
    sgemm_tt_128x128   vgpr=96   sgpr=28  lds=8192     scratch=0  kernargs=48
```

### Architecture

The tool is a thin compositor of three already-available parsing paths:

1. **OffloadBinary container layer** — `llvm::object::OffloadBinary::create()`
   iterates images, yields per-image `(ImageKind, OffloadKind, triple, arch,
   StringMap, payload_bytes)` tuples. Already implemented in LLVM.

2. **Per-image symbol extraction**:
   - **CUBIN**: The CUBIN payload is a valid ELF. Use `llvm::object::ELFObjectFile`
     to iterate the symbol table. Kernel symbols are in `.text` with `STT_FUNC`
     type; resource data is in `.nv.info.*` sections (NVPTX-specific note sections
     carrying per-kernel register and shared memory counts). Parsing `.nv.info`
     is the only NVIDIA-specific piece.
   - **HSACO**: Valid ELF. Iterate symbol table; filter `st_type ==
     STT_AMDGPU_HSA_KERNEL` (value 10). Each kernel symbol's `st_value` points
     to the 64-byte kernel descriptor in `.text`; parse the descriptor struct to
     extract VGPR count, SGPR count, `group_segment_fixed_size` (LDS),
     `private_segment_fixed_size` (scratch), and `kernarg_size`. These fields
     are documented in the LLVM AMDGPU backend user guide under "Code Object V5+
     Kernel Descriptor."
   - **SPIR-V**: Parse the SPIR-V binary word stream. Each `OpEntryPoint`
     instruction has opcode 15, followed by execution model (4 = Kernel),
     result-id, name string, and interface variable IDs. Enumerate all
     `OpEntryPoint` instructions where execution model == Kernel. The
     SPIRV-Tools `libspirv.a` or the LLVM SPIR-V translator can do this
     without a full disassembly pass.
   - **PTX** (text): Scan for `.entry` assembler directives. Each `.entry`
     declares a kernel with its parameter list and optional `.maxntid`,
     `.reqntid`, `.minnctapersm` pragma metadata.

3. **Output formatter**: Print in `nm`-style columns (address, type char,
   name) or in the extended resource table format.

### Implementation estimate

| Component | LOC | Depends on |
|-----------|-----|-----------|
| OffloadBinary container iterator | ~80 | `llvm::object::OffloadBinary` (already in LLVM) |
| CUBIN ELF symbol extractor | ~120 | `llvm::object::ELFObjectFile` + `.nv.info` parser |
| HSACO ELF symbol + descriptor parser | ~150 | `llvm::object::ELFObjectFile` + AMDGPU kernel descriptor struct |
| SPIR-V `OpEntryPoint` scanner | ~100 | Raw word-stream parser (no external dependency) |
| PTX `.entry` text scanner | ~60 | `llvm::MemoryBuffer` + regex or manual scan |
| Output formatter | ~120 | None |
| **Total** | **~630** | All deps already in LLVM monorepo |

This is a single-file tool comparable in scope to `llvm-readobj` format plugins.
No new library dependencies. No new binary formats. The hard technical work
(OffloadBinary parsing, AMDGPU ELF support) is already in the LLVM codebase.

### Connection to libkdl

`llvm-offload-nm` is the inspection complement to libkdl's dispatch mechanism.
libkdl reads capability contracts from OffloadBinary StringMap entries; if those
entries are absent (kernels compiled without explicit contracts), `llvm-offload-nm
-r` provides the resource data needed to construct them manually. The tool also
validates that a libkdl bundle contains the expected variants:

```
$ llvm-offload-nm gemm.kdl | grep sgemm_nt_128x128
# verify: sm_89, sm_90a, gfx1100, spirv all present before shipping
```

---

## Evidence

### Gap evidence — what the LLVM toolchain cannot do today

| Task | Current workaround | Gap severity |
|------|-------------------|--------------|
| List all kernel names in a fat `.o` | Extract with `llvm-offload-binary`, then run vendor tools on each image | Requires CUDA toolkit + ROCm on the inspection machine |
| Check which SM variants are compiled | `llvm-offload-binary` extraction + filename inspection | No resource data |
| Get register count for a CUDA kernel | `cuobjdump -res-usage` (NVIDIA-only, not in LLVM) | Blocked on non-CUDA system |
| Get VGPR/LDS for an AMD kernel | `llvm-readelf -s` on extracted HSACO + manual descriptor parse | No tooling, manual binary arithmetic |
| List SPIR-V kernel entry points | `spirv-dis` (SPIRV-Tools, not in LLVM) + grep `.entry` | Separate tool chain |
| CI check: "does this fat binary have all expected variants?" | No automated tool | Must write custom scripts |

### LLVM tool capabilities (confirmed)

- `llvm-offload-binary`: bundle / extract only. No inspection output. (Source:
  `llvm-offload-binary.cpp` — no stdout in non-error paths, no `-list` flag.)
- `llvm-readelf -WS object.o`: shows `.llvm.offloading` section exists and its
  byte size. Does not parse the OffloadBinary magic-byte format inside.
- `llvm-nm file.hsaco`: parses ELF symbol table but does not recognize
  `STT_AMDGPU_HSA_KERNEL` (value 10) — prints type `?` for kernel symbols.
  Does not parse kernel descriptors for resource counts.
- `llvm-nm file.cubin`: parses ELF but does not parse `.nv.info.*` sections
  that carry CUDA resource counts.

### Vendor tool capabilities (outside LLVM)

| Tool | Platform | Kernel names | Register count | Shared mem | Cross-vendor? |
|------|----------|-------------|---------------|------------|--------------|
| `cuobjdump --list-elf` | NVIDIA (CUDA toolkit required) | Yes | No | No | No |
| `cuobjdump -res-usage` | NVIDIA | Yes | Yes | Yes | No |
| `nvdisasm` | NVIDIA | Yes (headers) | Via disassembly | Via disassembly | No |
| `llvm-readelf -s file.hsaco` | AMD (any system) | Yes (mangled) | No | No | No |
| `spirv-dis` | Any (SPIRV-Tools) | Yes (full disasm) | N/A | N/A | No |

No tool spans all three GPU formats from a single invocation on an OffloadBinary
container. `llvm-offload-nm` fills this exact slot.

### Prior discussion

A search of LLVM Discourse (April 2026) and the LLVM GitHub issue tracker finds
**zero existing requests** for `llvm-offload-nm` or equivalent functionality.
The tool has not been proposed, named, or filed as a feature request anywhere in
the public LLVM infrastructure. This is a clean gap with no competing implementation
in flight.

The closest adjacent work:

- **PR #153504** (merged Aug 2025): introduces `llvm-offload-wrapper` — a tool for
  wrapping device binaries for embedding, not for inspection.
- **Issue #75356** (Nov 2023, still open): requests name-based kernel *loading* at
  runtime. Orthogonal to static binary inspection.
- **KernelInfo pass** (PR #102944, merged Jan 2025): an IR-level analysis pass that
  emits optimization remarks about kernel structure during compilation. Operates on
  LLVM IR, not on compiled fat binaries. Complementary but not the same layer.

---

## Feasibility

### Technical feasibility: high

All parsing components rely on LLVM APIs that are already upstream and stable:

- `llvm::object::OffloadBinary` — `OffloadBinary.h` / `OffloadBinary.cpp`, in LLVM
  since LLVM 15 (D125165), stable in LLVM 20+.
- `llvm::object::ELFObjectFile` — core LLVM Object library, used by `llvm-nm`,
  `llvm-readelf`, `llvm-objdump`. Fully stable.
- AMDGPU kernel descriptor struct — defined in
  `llvm/lib/Target/AMDGPU/Utils/AMDKernelCodeTUtils.h` and documented in
  LLVM AMDGPUUsage.html as Code Object V5 Kernel Descriptor. Available upstream.
- NVPTX `.nv.info` section format — semi-documented; used internally by the NVPTX
  backend. Parsing is ~60 lines of section iteration and note record reading.
- SPIR-V `OpEntryPoint` scan — SPIR-V binary is a word-aligned stream; scanning
  for opcode 15 (OpEntryPoint) with execution model 6 (Kernel) is a
  straightforward single-pass scan requiring no external dependency.

### Prototype feasibility: 2–3 days

A basic prototype covering HSACO + CUBIN kernel name listing (no resource counts)
is achievable in one day using `llvm::object::ELFObjectFile` + `OffloadBinary::create()`.
Adding SPIR-V `OpEntryPoint` scanning adds half a day. The `-r` resource mode
for HSACO (kernel descriptor parsing) adds one more day.

A working prototype demonstrating the concept on `experiments/prototype/src/` output
objects from libkdl is achievable before the Dublin deadline.

### Risk table

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `.nv.info.*` section format undocumented | Medium | Resource data is bonus; kernel names alone are novel |
| CUBIN ELF format changes between CUDA versions | Low | Use LLVM's `ELFObjectFile` which handles both 32/64-bit ELF |
| SPIR-V module without debug names has opaque function IDs | Low | `OpName` decoration provides names; `OpEntryPoint` has literal name field directly |
| Tool is "too simple" to be a poster topic on its own | Medium | Position as tooling contribution enabling libkdl validation workflow |

---

## Upstream Path

### Stage 1 — Standalone prototype (poster demo)

Build `llvm-offload-nm` as a new tool under `llvm/tools/llvm-offload-nm/`.
Follow the same structure as `llvm/tools/llvm-nm/`: main `.cpp` file (~600 LOC),
entry in `llvm/tools/CMakeLists.txt`, documentation stub in
`llvm/docs/CommandGuide/llvm-offload-nm.rst`.

Target: demonstrate at Dublin that running one command on a libkdl `.kdl` bundle
prints all kernel names, their target architectures, and their resource budgets.

### Stage 2 — Upstream RFC

Post to LLVM Discourse (toolchain/offload category) with subject:
"RFC: llvm-offload-nm — symbol inspection for GPU fat binaries."

CC: Joseph Huber (huber@, AMD/LLVM offload maintainer), Johannes Doerfert (LLNL),
offload working group (biweekly meetings established Jan 2024).

Justification to the community:
1. Closes the "black box" problem for OffloadBinary containers — CI pipelines
   can validate fat binary contents without vendor tools.
2. Enables debugging of multi-target compilation failures: "was the gfx1100
   image actually included?"
3. Natural complement to `clang-offload-packager` (build) and
   `llvm-offload-binary` (extract) — completes the inspect/build/extract trilogy.
4. The resource output (`-r`) provides data for libkdl capability contract
   generation without requiring a live GPU.

### Stage 3 — Integration with `llvm-offload-binary`

Long-term: merge `llvm-offload-nm` inspection mode into `llvm-offload-binary`
as a `--dump-symbols` flag, consolidating the two tools. Precedent:
`llvm-objdump` subsumes many separate `objdump`-style capabilities.

---

## Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | **7/10** | The gap is real and unaddressed. The concept is not technically surprising — it is a straightforward composition of existing LLVM APIs. Novelty is in noticing the gap and filling it, not in the algorithm. |
| **Feasibility** | **10/10** | All dependencies exist upstream. No new binary formats. Estimated ~630 LOC for a complete implementation. A working prototype is achievable in 2–3 days. |
| **Evidence** | **8/10** | Zero existing tools span all three GPU formats from a single OffloadBinary container. LLVM issue tracker confirms no prior request. Vendor tool gap table is concrete and verifiable. |
| **Impact** | **7/10** | Every team shipping multi-vendor GPU kernels needs this. CI pipelines validating fat binary contents today require either vendor-specific tools or custom scripts. Impact is broad but the tool is simple utility infrastructure, not a research breakthrough. |
| **Community fit** | **9/10** | Dublin 2026 LLVM audience builds and ships fat binaries. The "where is my kernel?" question is immediate and practical. The LLVM offload working group (Joseph Huber's team) would accept this contribution given the biweekly RFC pathway established Jan 2024. |
| **As poster topic** | **5/10** | Strong as a *contribution* item but thin as a standalone poster. Best positioned as a section within a broader libkdl poster ("the inspection tool for our bundle format") rather than the headline contribution. |
| **Composite** | **7.7/10** | |

---

## Pitch

`llvm-nm` tells you what is in a `.o` file. `llvm-readelf` tells you what
sections it has. Neither tells you what GPU kernels are compiled into the
`.llvm.offloading` section that LLVM 20 now embeds by default in every CUDA, HIP,
and OpenMP compilation. `llvm-offload-binary` can unpack the images to disk — but
then you need `cuobjdump` (NVIDIA-only), `llvm-readelf` with manual AMDGPU
descriptor arithmetic, and `spirv-dis` (a separate tool chain) to get actual
kernel names and register counts. We propose `llvm-offload-nm`, a ~630-line LLVM
tool that reads an OffloadBinary container and prints a unified symbol table:
kernel names, target architectures, and resource budgets (VGPRs, LDS bytes, stack
frame, kernarg size) across CUBIN, HSACO, SPIR-V, and PTX images simultaneously.
All parsing relies on LLVM APIs already upstream (`OffloadBinary::create()`,
`ELFObjectFile`, AMDGPU kernel descriptor structs). The tool completes the
inspect/build/extract trilogy for the `llvm-offload-*` family and enables libkdl
users to validate their multi-vendor kernel bundles without vendor tools. A working
prototype on libkdl `.kdl` bundles is demonstrated at the poster.

---

## Key References

- [`llvm/include/llvm/Object/OffloadBinary.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/OffloadBinary.h) — container API
- [`llvm/lib/Object/OffloadBinary.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Object/OffloadBinary.cpp) — `create()` parser
- [`llvm/tools/llvm-offload-binary/llvm-offload-binary.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/tools/llvm-offload-binary/llvm-offload-binary.cpp) — existing tool (bundle/extract only, no inspection)
- [LLVM AMDGPUUsage.html — Code Object V5 Kernel Descriptor](https://llvm.org/docs/AMDGPUUsage.html)
- [LLVM CommandGuide: llvm-offload-binary](https://llvm.org/docs/CommandGuide/llvm-offload-binary.html)
- [CUDA Binary Utilities — cuobjdump, nvdisasm (NVIDIA docs)](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [SPIR-V Specification — OpEntryPoint (Khronos)](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [Clang Offloading Design — `.llvm.offloading` section, new driver](https://clang.llvm.org/docs/OffloadingDesign.html)
- [LLVM PR #153504 — llvm-offload-wrapper (Aug 2025)](https://github.com/llvm/llvm-project/pull/153504) — adjacent tool, not inspection
- [LLVM Issue #75356 — Name-based kernel loading (Nov 2023, open)](https://github.com/llvm/llvm-project/issues/75356) — runtime side of same gap
- [KernelInfo pass — PR #102944 (merged Jan 2025)](https://github.com/llvm/llvm-project/pull/102944) — compile-time complement, different layer
- [RFC: Introducing llvm-project/offload — LLVM Discourse #74302](https://discourse.llvm.org/t/rfc-introducing-llvm-project-offload/74302)
- [GPU/Offloading Workshop 2025 Slides — LLVM Discourse](https://discourse.llvm.org/t/gpu-offloading-workshop-2025-slides/88832)
