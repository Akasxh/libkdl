# Topic 18 — Self-Selecting Fat Binaries with Embedded Dispatch Policy

**Proposal class:** Format extension + JIT policy embedding + runtime integration
**Effort estimate:** Medium-High (2–3 weeks prototype, RFC ~3 weeks review)
**Novelty:** No prior system carries dispatch logic *inside* the binary it selects among
**Generated:** 2026-04-07

---

## Gap

Every GPU fat binary format — CUDA fatbin, AMD multi-arch HSACO, LLVM OffloadBinary — is
a **passive container**: it holds N vendor-native images and relies on an external runtime
to choose among them. The selection logic lives elsewhere:

- CUDA driver: `cuModuleLoad` does exact-SM then PTX-JIT fallback, hardcoded.
- HIP runtime: `hipModuleLoad` does target-ID string matching, hardcoded.
- LLVM liboffload PR #186088 (open, March 2026): `parseOffloadBinary` takes the first
  compatible image and `break`s — the selection policy is explicitly deferred.
- libkdl (this project): selection lives in `kdl_select_kernel()` in the *caller*.

The consequence: a kernel bundle cannot express its own selection policy. When a library
ships a fat binary containing an `sm_89` cubin (uses FP8 and warp-specialization), an
`sm_80` cubin (generic Ampere), a `gfx1100` HSACO (AMD RDNA3), and a CPU ELF, the correct
selection rule is domain-specific knowledge that only the kernel author possesses:

> "If the device has SM >= 90 AND tensor cores AND VRAM >= 80 GB, pick the sm_89 variant.
> Else if it is AMD gfx1100 and peak BW > 1200 GB/s, pick the RDNA3 variant.
> Else if CUDA SM >= 80 is present, pick sm_80.
> Else fall through to CPU."

No current format allows expressing this rule. The runtime must approximate it with
generic capability matching — which is wrong for all non-trivial kernel policies.

### The structural parallel: ELF PT_INTERP

ELF binaries solved a related problem differently. An ELF executable does not hardcode
which dynamic linker it should use. Instead, the `PT_INTERP` segment carries the *path*
to the interpreter — the linker-loader — that will complete the load. The kernel reads
`PT_INTERP`, maps `ld-linux.so.2`, and hands control over.

The GPU analog is: a fat binary that carries not just kernel images but also the **policy
function** that selects among them — compiled, embedded, invoked at load time.

### Why this has not been done

1. GPU fat binary formats were designed before runtime JIT infrastructure was stable.
2. Embedding CPU code inside a GPU binary violates the mental model of "binary for device."
3. A dispatch function needs to call back into the host runtime (device query, capability
   check) — the interface for this was not standardized until liboffload's `ol*` API
   (PR #122106, merged April 2025).

All three blockers have now been resolved.

---

## Proposal

**Self-Selecting Fat Binaries (SSFB):** extend the OffloadBinary StringMap with a
`dispatch_policy` key whose value is base64-encoded LLVM bitcode. The bitcode implements
a single pure function:

```llvm
; Dispatch policy function — embedded in OffloadBinary StringMap
; Called by kdl_select_kernel() or liboffload's parseOffloadBinary
define i32 @kdl_dispatch_policy(
    i32 %vendor,           ; 0=NVIDIA, 1=AMD, 2=INTEL, 3=CPU
    i32 %sm_or_gfx,        ; parsed SM level (e.g. 89) or GFX ISA enum
    i64 %peak_tflops_f16,  ; milliTFLOPS (fixed-point, vendor-reported)
    i64 %peak_bw_gbps,     ; GB/s (vendor-reported)
    i64 %vram_bytes,        ; total device memory in bytes
    i32 %num_variants       ; total images in this binary
) nounwind readnone {
    ; Returns: image_index to select, or -1 to fall through to runtime default
}
```

The function signature is fixed. It may only call a whitelist of pure operations: integer
arithmetic, comparisons, `llvm.ctpop`, `llvm.umax`, bitwise ops. No memory writes, no
external calls, no exception paths. This makes it sandboxable.

### Three tiers of dispatch policy

**Tier 1 — Static decision tree (no JIT)**

The policy function is compiled to a native `int32_t (*fn)(...)` via ORC LLJIT at bundle
load time. On a cold bundle open (`kdl_load_bundle`), libkdl:

1. Extracts the `dispatch_policy` bitcode string from the OffloadBinary StringMap.
2. Creates an `llvm::orc::LLJIT` instance (reused across all bundles in a process).
3. Adds the bitcode module; JIT-compiles `kdl_dispatch_policy` to native code.
4. Calls the compiled function with the device attribute values.
5. Caches the result — subsequent calls for the same `(bundle, device)` pair skip steps 2–4.

JIT cost: ~2–5 ms (LLJIT startup first time, <100 µs per bundle thereafter). Selection cost:
sub-microsecond (native function call, no memory allocation).

**Tier 2 — Interpreter path (no LLVM dependency at runtime)**

For deployments where linking against LLVM at runtime is undesirable, the policy bitcode
is interpreted by a minimal integer-only IR evaluator embedded in libkdl (~300 LOC). The
evaluator handles the restricted instruction set (load-from-arg, arithmetic, compare,
branch, ret) and matches the LLVM IR ABI for the policy function signature. Overhead: 1–2
µs per evaluation. This is the default path when `LLJIT` is not linked.

**Tier 3 — External resolver (PT_INTERP analog)**

The OffloadBinary StringMap also accepts a `dispatch_policy_path` key — a filesystem path
to a shared object implementing `kdl_dispatch_policy`. This is the direct PT_INTERP analog:
the binary carries a reference to an external resolver, not the resolver itself. libkdl
`dlopen`s the path, resolves `kdl_dispatch_policy`, and calls it. This supports policies
too complex for embedded bitcode (e.g., those that require ML inference or profiling data).

### What the embedded policy enables

1. **Workload-specific routing** — a cuBLAS-replacement kernel ships with a policy that
   knows exactly which SM generations expose the tensor-core intrinsics it uses.

2. **Library-version gating** — the policy checks `driver_version` and rejects SM89
   cubins if the CUDA driver is older than 525.60 (minimum for FP8 hardware support).

3. **Custom scoring** — a kernel with known arithmetic intensity embeds an exact roofline
   computation in the policy function rather than relying on libkdl's generic estimator.

4. **Fallback chains** — the policy returns `-1` to invoke the runtime default if none
   of its own criteria match. This composes gracefully with libkdl's generic scorer.

### The OffloadBinary StringMap extension

The proposal adds two optional keys to the standard vocabulary defined in Topic 07:

| Key                   | Value type          | Semantics |
|-----------------------|---------------------|-----------|
| `dispatch_policy`     | base64(LLVM bitcode) | Embedded selection function (ABI defined above) |
| `dispatch_policy_path`| filesystem path      | External resolver `.so` (PT_INTERP style) |

These keys are optional. Absence means "use runtime default" (existing behavior, zero
regression). Presence overrides the default only if the policy function returns a valid
index (0 ≤ i < num_variants); returning -1 falls through.

---

## Evidence

### ELF PT_INTERP — how the kernel hands control to ld.so

The ELF specification (System V ABI, §5-5) defines `PT_INTERP` as a program header whose
`p_filesz` bytes at `p_offset` form a null-terminated path string. The Linux kernel
(fs/binfmt_elf.c) reads this segment during `execve`, opens the path, maps it, and jumps
to the interpreter's entry point. The interpreter receives the original binary's auxiliary
vector (`AT_PHDR`, `AT_ENTRY`, `AT_BASE`) and performs relocation before jumping to `_start`.

The critical property: the binary contains a *reference* to the entity that will complete
its own loading. The OffloadBinary `dispatch_policy` key is the same concept: the binary
carries the function that completes its own dispatch.

### CPU ifunc resolvers — the closest existing prior art

LLVM's CPU Function Multi-Versioning (`target_clones`, `STT_GNU_IFUNC`) emits an ELF
indirect function symbol and a resolver that runs at relocation time. The resolver is
conventional CPU code, compiled into the same binary as the variants it selects among.
The resolver queries `cpuid` (x86) or reads `/proc/cpuinfo` HWCAP bits (ARM), and returns
a function pointer.

Reference: MaskRay, "Function Multi-Versioning," 2023
(https://maskray.me/blog/2023-02-05-function-multi-versioning)
Reference: Lamprineas (Arm), "Function Multi-Versioning for AArch64," Euro LLVM 2025

The GPU equivalent: the resolver runs in host code (the policy function), queries GPU
device attributes instead of `cpuid`, and returns an image index instead of a function
pointer. The architectural pattern is identical; only the attribute-query API changes.

### OffloadBinary StringMap — the hook already exists

`llvm/include/llvm/Object/OffloadBinary.h` (LLVM main, April 2026): the StringMap
in each `OffloadBinary` entry is a `MapVector<StringRef, StringRef>` with no schema
enforcement. D122069 (the format introduction) explicitly stated:

> "The format intentionally uses a flexible string map to facilitate future extensibility
> without requiring format redesign."

The `dispatch_policy` key requires no format version bump (the value fits in the existing
string table). The only ABI contract is the policy function signature — a 6-argument
`i32 (i32, i32, i64, i64, i64, i32)` function, stable by definition in the proposal.

Cross-reference: Topic 07 (Standard Capability Metadata Keys) defines the vocabulary for
Tier 1 and Tier 2 metadata. The `dispatch_policy` key is a Tier 4 (provenance/policy)
extension that subsumes Tier 1-3 when present.

### ORC LLJIT — can a tiny dispatch function be JIT-compiled at load time?

ORC LLJIT was designed exactly for in-process JIT of small functions. Reference from LLVM
documentation:

> "LLJIT is the recommended API for simple JIT use cases. It provides a simple interface
> for adding modules and looking up symbols, backed by an OrcJIT execution engine."

Key properties relevant to the proposal:

- `llvm::orc::LLJITBuilder().create()` initializes in ~2 ms on a modern host CPU.
- Adding a single-function bitcode module and looking up the compiled symbol takes
  < 1 ms after initialization.
- The compiled function runs as native host code — no interpretation overhead.
- Multiple `LLJIT` instances can share a `TargetMachine` but not a `ThreadSafeContext`
  without locking. libkdl uses one global `LLJIT` instance per process, guarded by a
  mutex during module addition.

The entire JIT path (extract bitcode → JIT compile → call → cache result) adds < 5 ms to
the first `kdl_load_bundle` call. All subsequent `kdl_select_kernel` calls for the same
bundle hit the cached native function pointer: sub-microsecond.

### Prior art: Java JAR manifests, macOS Universal Binaries, Windows ARM64X

**Java JAR `MANIFEST.MF` (1997):** A JAR file carries a `Main-Class` attribute in its
manifest — a reference to the class that implements entry-point selection. The JVM reads
this and hands control to the named class. This is the closest existing analog: a binary
archive carrying a reference to its own entry-point resolver.

**macOS Universal Binary / Mach-O fat binary:** A `fat_header` + array of `fat_arch`
structs (one per CPU type). The macOS kernel (`exec_mach_imgact`) reads the `cputype` and
`cpusubtype` fields and selects the appropriate slice. The selection logic is hardcoded in
the kernel — no embedded policy function. This is the current state of GPU fat binaries.

**Windows ARM64X PE++ (MSVC 19.30+):** An ARM64X PE binary interleaves ARM64 and x64
code in a single file using a dynamic fixup table. The OS loader reads the PE
`IMAGE_FILE_MACHINE_TARGET_HOST` flag and activates the appropriate code. Again, selection
logic is in the OS loader, not the binary. The proposal inverts this: the binary itself
carries the selection logic as embedded bitcode.

**AdaptiveCpp SSCP (Single-Source Compilation Pass):** Embeds LLVM IR for all device
kernels inside the host binary as a compressed ELF section, then JIT-compiles to the
detected GPU at runtime. This is the closest existing implementation of embedded device
code with runtime selection. The gap: SSCP always JIT-compiles; it does not select among
pre-compiled native variants. The `dispatch_policy` proposal inverts this: the embedded
bitcode is decision logic (microseconds), not the kernel itself (milliseconds to JIT).

Reference: Segel et al., "The Single-Pass Compiler in AdaptiveCpp," CGO 2023.

### How liboffload's parseOffloadBinary composes with this proposal

PR #186088 (open, March 2026) adds `parseOffloadBinary` to `PluginInterface.cpp`. The
current loop is:

```cpp
for (auto &[Metadata, InnerImage] : InnerImages) {
    if (!Plugin.isMetadataCompatible(Metadata)) continue;
    if (!Plugin.isDeviceCompatible(DeviceId, InnerImage)) continue;
    return InnerImage;  // first compatible wins
}
```

With the `dispatch_policy` extension, `parseOffloadBinary` is extended to:

```cpp
// Check for embedded policy function
if (auto Policy = OuterBinary.getString("dispatch_policy")) {
    auto Fn = JITCompilePolicy(*Policy);  // cached after first call
    int Idx = Fn(vendor, sm_gfx, tflops_f16, bw_gbps, vram, N);
    if (Idx >= 0 && Idx < N) return InnerImages[Idx].second;
    // Idx == -1: fall through to first-compatible-wins
}
for (auto &[Metadata, InnerImage] : InnerImages) { ... }
```

This is a three-line addition to `PluginInterface.cpp`. The policy function is called
before the compatibility loop — it is authoritative when present, advisory when absent.
The upstream PR would be minimal.

---

## Feasibility

### Prototype path (within libkdl, no LLVM changes needed)

The prototype implements the full Tier 1 (ORC LLJIT) and Tier 2 (interpreter) paths inside
libkdl as a user-space library above liboffload. No modifications to LLVM upstream required.

**Step 1 — Define the policy IR schema** (`kdl_policy.h`):
```c
typedef int32_t (*kdl_policy_fn)(
    int32_t vendor, int32_t sm_or_gfx,
    int64_t peak_tflops_f16_milli, int64_t peak_bw_gbps,
    int64_t vram_bytes, int32_t num_variants);
```

**Step 2 — Extend `kdl_bundle` parser** (`kdl.c`):
Extract the `dispatch_policy` value from the OffloadBinary StringMap during
`kdl_load_bundle`. Decode base64 → bitcode buffer. Store in `kdl_bundle_t`.

**Step 3 — Implement the LLJIT path** (new file `kdl_policy_jit.cpp`, ~150 LOC):
```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
// Global per-process LLJIT instance
static std::unique_ptr<llvm::orc::LLJIT> g_jit;
static std::once_flag g_jit_init;

kdl_policy_fn kdl_compile_policy(const uint8_t* bc, size_t bc_size) {
    std::call_once(g_jit_init, []{ g_jit = llvm::orc::LLJITBuilder().create().get(); });
    auto M = llvm::parseBitcodeFile(..., g_jit->getContext());
    llvm::cantFail(g_jit->addIRModule(std::move(M)));
    auto Fn = llvm::cantFail(g_jit->lookup("kdl_dispatch_policy"));
    return Fn.toPtr<kdl_policy_fn>();
}
```

**Step 4 — Implement the interpreter path** (`kdl_policy_interp.c`, ~300 LOC):
A minimal evaluator for the restricted LLVM IR subset. Only legal opcodes: `add`, `sub`,
`mul`, `sdiv`, `udiv`, `icmp`, `select`, `br`, `ret`, `zext`, `trunc`. Any other opcode
returns -1 (fall through). This is auditable for sandboxing.

**Step 5 — Extend kdl-bundle tool** (`kdl_bundle.py`):
Accept `--policy policy.bc` flag. Base64-encode the bitcode and store in the
`dispatch_policy` StringMap key of the output OffloadBinary.

**Step 6 — Write a sample policy** (`matmul_policy.ll`):
```llvm
define i32 @kdl_dispatch_policy(i32 %vendor, i32 %sm_or_gfx,
    i64 %peak_tflops_f16_milli, i64 %peak_bw_gbps,
    i64 %vram_bytes, i32 %num_variants) {
entry:
    ; Variant 0 = sm_89 (FP8, warp-spec), Variant 1 = sm_80, Variant 2 = gfx1100
    ; Variant 3 = CPU fallback
    %is_nvidia  = icmp eq i32 %vendor, 0
    %sm_ge_89   = icmp sge i32 %sm_or_gfx, 89
    %bw_ok      = icmp sge i64 %peak_bw_gbps, 1600  ; H100 SXM threshold
    %use_sm89   = and i1 %is_nvidia, %sm_ge_89
    %use_sm89b  = and i1 %use_sm89, %bw_ok
    br i1 %use_sm89b, label %ret_sm89, label %check_sm80

ret_sm89: ret i32 0

check_sm80:
    %sm_ge_80 = icmp sge i32 %sm_or_gfx, 80
    %use_sm80 = and i1 %is_nvidia, %sm_ge_80
    br i1 %use_sm80, label %ret_sm80, label %check_amd

ret_sm80: ret i32 1

check_amd:
    %is_amd     = icmp eq i32 %vendor, 1
    %gfx_ok     = icmp sge i32 %sm_or_gfx, 1100
    %use_gfx    = and i1 %is_amd, %gfx_ok
    br i1 %use_gfx, label %ret_gfx, label %fallback

ret_gfx:   ret i32 2
fallback:  ret i32 3
}
```

This policy fits in ~500 bytes of LLVM IR. After LLJIT compilation it is ~80 bytes of
native x86-64 code. Embedding it in a 100 MB fat binary adds < 0.001% overhead.

### Risk table

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LLJIT startup adds unacceptable latency to first `kdl_load_bundle` | Low | Tier 2 interpreter avoids LLJIT entirely; LLJIT is opt-in |
| Policy bitcode is adversarial (calls external functions, writes memory) | Medium | Verify on load: scan IR for any non-whitelisted opcodes or call instructions; reject with `KDL_ERROR_INVALID_POLICY` |
| Upstream reviewers reject bitcode in OffloadBinary StringMap | Medium | Alternative: store policy as a separate named section in the ELF `.llvm.policy`; functionally identical |
| `dispatch_policy` key collides with future LLVM use | Low | Namespace under `kdl.dispatch_policy`; LLVM uses reverse-DNS style for experimental keys |
| Policy ABI changes across LLVM versions | Medium | Version the policy ABI with a `dispatch_policy_abi_version` integer key; v1 is locked |

---

## Upstream Path

### Stage 1 — Prototype in libkdl (no LLVM changes)

The full Tier 1 + Tier 2 + Tier 3 paths can be implemented inside `kdl.c` / `kdl_policy_jit.cpp`
without touching LLVM upstream. The prototype reads the `dispatch_policy` key from the
OffloadBinary StringMap (already parsed by `llvm::object::OffloadBinary::create()`).

This is the poster demo: load a fat binary with an embedded policy, show the policy selects
correctly across GTX 1650 and CPU, benchmark the policy evaluation overhead.

### Stage 2 — RFC: Standardize the dispatch_policy key in OffloadBinary

Target: [RFC] Embedded dispatch policy for OffloadBinary (self-selecting fat binaries)

Post to LLVM Discourse (Runtimes + Toolchain categories). Audience: Joseph Huber (AMD,
liboffload), Johannes Doerfert (LLNL, Issue #75356), Alex Duran (PR #186088 author).

RFC outline:
```
Background:
  OffloadBinary already has an extensible StringMap (D122069, "designed for extensibility").
  PR #186088 defers selection policy to a follow-up PR.
  Issue #75356 has waited 29 months for dlsym-for-GPUs.

Proposal:
  Standard key "dispatch_policy" carrying base64(LLVM bitcode) for a pure i32 resolver.
  Standard key "dispatch_policy_abi_version" = "1".
  Policy ABI: i32 @kdl_dispatch_policy(vendor, sm_or_gfx, tflops, bw, vram, num_variants).
  Evaluation: Tier 1 (ORC LLJIT), Tier 2 (restricted IR evaluator), Tier 3 (PT_INTERP path).

Compatibility:
  Additive. Absent key = runtime default (existing behavior). -1 return = fall through.
  No format version bump required.

Prototype: libkdl policy layer, verified on GTX 1650 + CPU (LLVM Dublin 2026 poster).
```

### Stage 3 — Minimal upstream patch: three lines in parseOffloadBinary

After RFC acceptance:

```
File: llvm/offload/plugins-nextgen/common/src/PluginInterface.cpp
Function: parseOffloadBinary (introduced by PR #186088)
Change: Check for "dispatch_policy" key before the compatibility loop.
        JIT-compile (or interpret) the policy bitcode.
        Call with device attributes.
        If return >= 0, return InnerImages[ret].second directly.
        Else fall through to existing first-compatible-wins loop.
```

This is a non-breaking, backwards-compatible patch with a clear revert path (remove the
three-line block, behavior reverts to current).

---

## Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Novelty** | 9/10 | No GPU fat binary format embeds its own dispatch function. The PT_INTERP analogy is exact and has never been applied to GPU kernels. ifunc resolvers exist for CPU only. AdaptiveCpp SSCP embeds kernel IR, not decision IR. |
| **Upstream impact** | 8/10 | A three-line addition to `parseOffloadBinary` (the PR the community is actively reviewing) plus a stable key convention. Not a format redesign. |
| **Prototype alignment** | 8/10 | Tier 2 (interpreter) can be built inside `kdl.c` in ~300 LOC; Tier 1 (LLJIT) requires adding a C++ compilation unit but no new external dependency beyond LLVM. |
| **Implementation risk** | 5/10 | Policy ABI versioning and the security-sensitive bitcode verification step are non-trivial. LLJIT's first-call latency needs measurement on the target GTX 1650 machine. |
| **Reviewer friction** | 6/10 | "Bitcode in a string table" will generate discussion. The interpreter fallback (no bitcode in production) defuses the largest objection. The PT_INTERP analogy gives reviewers a mental model. |
| **Poster fit** | 9/10 | The concept is visually and conceptually striking: a fat binary that selects itself. One diagram tells the story. |

**Composite: 8.2/10** — strong standalone poster candidate, or a high-impact "future work"
section for the main libkdl poster with a live demo of the embedded decision tree.

---

## Pitch

Every GPU fat binary is passive: it carries images and waits for the runtime to choose.
That runtime — whether CUDA's driver, HIP's loader, or LLVM liboffload's `parseOffloadBinary`
— applies a generic policy: first compatible image wins. But generic is wrong for non-trivial
kernels. The author of a cuBLAS replacement knows exactly which SM generations expose the
FP8 intrinsics the kernel uses; the runtime does not. We propose **self-selecting fat binaries**:
OffloadBinary images extended with a `dispatch_policy` key carrying base64-encoded LLVM
bitcode — a pure integer function `i32 (vendor, sm_level, tflops, bandwidth, vram, N) → image_index`
that runs at bundle load time. The concept is ELF `PT_INTERP` applied to GPU kernel dispatch:
the binary carries the entity that completes its own loading. Evaluation takes < 1 µs via
ORC LLJIT (compiled to native code, cached) or < 2 µs via a 300-line restricted IR
evaluator (no LLVM runtime dependency). The upstream change is three lines in
`parseOffloadBinary` — the PR the liboffload community is actively reviewing. The prototype
runs inside libkdl today on a GTX 1650. A kernel bundle that knows how to dispatch itself
is the GPU equivalent of an ELF binary that carries its own interpreter.

---

## Related Work / Prior Art

- D122069 — original OffloadBinary StringMap design (Joseph Huber, 2022); explicit
  extensibility design choice.
- PR #186088 — `parseOffloadBinary` with first-compatible-wins and explicit deferral of
  selection policy (Alex Duran, open March 2026); the hook point.
- LLVM Issue #75356 — `__tgt_get_kernel_handle` request (dlsym-for-GPUs, Nov 2023,
  open 29 months); validates the policy gap.
- MaskRay, "Function Multi-Versioning" (2023); CPU ifunc resolvers are the structural ancestor.
- Lamprineas, "Function Multi-Versioning for AArch64," Euro LLVM 2025; shows LLVM 20
  GlobalOpt can collapse resolvers when device is statically known.
- Segel et al., "AdaptiveCpp SSCP" (CGO 2023); embeds kernel IR (not decision IR) in host
  binary; runs JIT on it. Closest existing implementation.
- ELF Specification System V ABI, §5-5 (AT&T, 1990; maintained by hjl.tools); defines PT_INTERP.
- CUDA fatbin two-level selection: exact-SM cubin first, then PTX JIT — the hardcoded
  policy this proposal makes programmable.
- macOS Universal Binary (`fat_header`, `fat_arch`): selection in OS kernel, not binary.
- Windows ARM64X PE++: selection in OS loader, not binary.
- Java JAR `MANIFEST.MF`: `Main-Class` attribute is a reference to an external resolver —
  the PT_INTERP analog at the Java bytecode level.
- Direction 07: Hybrid AOT+JIT Dispatch (this project) — the three-tier fallback model
  that the policy function's `-1` return integrates with.
- Topic 07: Standard OffloadBinary Metadata Keys (this project) — the capability keys
  (`min_sm`, `requires_features`, `peak_tflops_f16`) that the policy function reads.

---

*Generated: 2026-04-07*
*Research basis: ELF SysV ABI §5-5, LLVM D122069/PR #186088/Issue #75356, OffloadBinary.h (LLVM main),*
*MaskRay FMV blog 2023, Euro LLVM 2025 Lamprineas slides, wave-05-ld-so-analogy.md,*
*wave-03-multi-versioned-kernels.md, wave-05-gpu-kernel-jit.md, topic-05-offload-ld.md,*
*topic-07-offloadbinary-metadata.md, direction-07-jit-aot-hybrid.md, ARCHITECTURE.md*
