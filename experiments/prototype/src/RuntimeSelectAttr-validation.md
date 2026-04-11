# RuntimeSelectAttr.cpp.sketch — Validation Report

**File:** `experiments/prototype/src/RuntimeSelectAttr.cpp.sketch` (889 lines)
**Reviewer:** Automated API audit against MLIR trunk (April 2026)
**Status:** Design sketch, explicitly marked NOT compilable

---

## 1. Summary of What the Sketch Does

The sketch describes a new MLIR attribute `#gpu.runtime_select` that implements
`OffloadingLLVMTranslationAttrInterface` — the same interface that the upstream
`#gpu.select_object` (`SelectObjectAttr`) uses. Where `SelectObjectAttr` commits
to a single GPU binary at LLVM-IR translation time, `RuntimeSelectAttr` embeds
**all** binary variants and generates a startup constructor that probes vendor
runtimes (CUDA, HIP, Level Zero) at process load time and selects the correct
one.

The design is divided into five phases in `embedBinary()`:

1. Embed each `#gpu.object` blob as a separate `internal constant` LLVM global.
2. Emit a dispatch table (`[N x %RuntimeSelectEntry]`) mapping vendor IDs to blobs.
3. Emit two mutable globals: `{sym}_selected_idx` (i32, -1 sentinel) and
   `{sym}_module_ptr` (ptr, null).
4. Emit a constructor (`@{sym}_ctor`, priority 123) that calls
   `__gpu_runtime_select_detect_vendor()` + `__gpu_runtime_select_rank()`,
   then calls `mgpuModuleLoad()` and stores the handle.
5. Emit a symmetric destructor (`@{sym}_dtor`, priority 123) that null-guards
   and calls `mgpuModuleUnload()`.

`launchKernel()` is intentionally identical to `SelectObjectAttr`: a single
load of `{sym}_module_ptr` followed by `mgpuModuleGetFunction` + `mgpuLaunchKernel`.
This is the central design claim (zero per-launch overhead).

The two runtime C helpers (`__gpu_runtime_select_detect_vendor` and
`__gpu_runtime_select_rank`) are defined as plain `extern "C"` functions
intended for a new `GPURuntimeSelectWrappers.cpp` file.

---

## 2. API Correctness Assessment

### 2.1 OffloadingLLVMTranslationAttrInterface Method Signatures — WRONG

**Sketch signatures (lines ~163, ~524):**

```cpp
LogicalResult RuntimeSelectAttr::embedBinary(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation)

LogicalResult RuntimeSelectAttr::launchKernel(
    Operation *launchFuncOp, Operation *binaryOp,
    llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation)
```

**Actual interface (`CompilationAttrInterfaces.td`, current trunk):**

The interface-generated method signatures include an additional leading
`Attribute` parameter representing the concrete attribute instance, consistent
with how TableGen generates interface dispatch stubs. The actual
`SelectObjectAttr.cpp` implementation uses this signature:

```cpp
LogicalResult embedBinary(Attribute attribute, Operation *operation,
                          llvm::IRBuilderBase &builder,
                          LLVM::ModuleTranslation &moduleTranslation) const;

LogicalResult launchKernel(Attribute attribute,
                           Operation *launchFuncOperation,
                           Operation *binaryOperation,
                           llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) const;
```

**Delta:**
- Both methods are missing the leading `Attribute attribute` parameter.
- Both methods are missing the `const` qualifier.
- Parameter naming diverges (`operation` vs `op`, `launchFuncOperation` vs
  `launchFuncOp`) — this does not affect compilation but should match style.

**Fix required:** Add `Attribute attribute` as the first parameter to both
methods and add `const`. The `attribute` can be cast to `RuntimeSelectAttr`
to access `getStrategy()` and `getFallback()` instead of calling them directly
on `this` (which only works if the implementation class is the attribute class
itself — in the TableGen-generated pattern, the interface methods receive the
attribute by value).

**Note on interface vs. impl class:** The sketch treats `RuntimeSelectAttr` as
if it directly contains `getStrategy()` / `getFallback()` methods and calls
them via `this`. This is consistent with how `DeclareAttrInterfaceMethods` +
`GPU_Attr<>` generates an attribute class with those accessors. The only
adjustment is adding the `Attribute attribute` dispatch parameter.

---

### 2.2 `mgpuModuleLoad` Signature — CORRECT but INCOMPLETE

**Sketch declaration (line ~377):**

```cpp
auto *mgpuModuleLoadTy = llvm::FunctionType::get(
    ptrTy, {ptrTy, i64Ty}, false);
auto mgpuModuleLoadCallee = module->getOrInsertFunction(
    "mgpuModuleLoad", mgpuModuleLoadTy);
```

**Actual signature in `CudaRuntimeWrappers.cpp` / `RocmRuntimeWrappers.cpp`:**

```c
CUmodule mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/)
```

This matches — `(ptr, i64) -> ptr`. The sketch correctly uses the two-parameter
form.

**Gap:** The sketch does not handle the `mgpuModuleLoadJIT` path. As of the
Apr 3 2026 commit, when the `#gpu.object` format is `Assembly` (PTX or LLVM IR
for JIT), `SelectObjectAttr` calls `mgpuModuleLoadJIT(ptr, i32 optLevel, i64
assemblySize)` instead. The sketch hard-codes the `mgpuModuleLoad` path for all
objects, which means it silently fails (or calls the wrong function) for any
`#gpu.object` with `format = Assembly`. For a Dublin demo with pre-compiled
cubins, this is fine. For completeness in a real implementation, a format check
is needed.

---

### 2.3 `mgpuLaunchKernel` Signature — MISSING PARAMETER

**Sketch declaration (lines ~595-599):**

```cpp
auto *mgpuLaunchKernelFnTy = llvm::FunctionType::get(
    llvm::Type::getVoidTy(ctx),
    {ptrTy, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty,
     llvm::Type::getInt32Ty(ctx), ptrTy, ptrTy, ptrTy},
    false);
```

**11 parameters declared.**

**Actual signature in upstream `SelectObjectAttr.cpp`:**

```cpp
FunctionType::get(voidTy,
    ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy,
        intPtrTy, intPtrTy, intPtrTy, i32Ty,
        ptrTy, ptrTy, ptrTy, i64Ty}),
    false)
```

**12 parameters** — there is a trailing `i64Ty` (paramsCount) that the sketch
omits.

Additionally, the sketch uses `i64Ty` for grid/block dimensions but upstream
uses `intPtrTy` (pointer-sized integer, `llvm::Type::getIntNTy(ctx,
DL.getPointerSizeInBits())`). On 64-bit platforms these are the same, but the
sketch should match the upstream type to avoid potential issues on 32-bit or
non-standard targets.

**Fix required:**
- Add `i64Ty` (paramsCount) as a 12th parameter to `mgpuLaunchKernelFnTy`.
- Add the corresponding `llvm::ConstantInt::get(i64Ty, numKernelParams)` to
  the `CreateCall` arguments.
- Change grid/block dimension types from `i64Ty` to `intPtrTy` for strict
  upstream compatibility.

---

### 2.4 `mgpuModuleGetFunction` Call — CORRECT PATTERN, MINOR STYLE ISSUE

**Sketch pattern (lines ~560-575):**

```cpp
auto *mgpuModuleGetFunctionFn = module->getOrInsertFunction(
    "mgpuModuleGetFunction",
    llvm::FunctionType::get(ptrTy, {ptrTy, ptrTy}, false)).getCallee();

llvm::Value *func = builder.CreateCall(
    llvm::FunctionType::get(ptrTy, {ptrTy, ptrTy}, false),
    mgpuModuleGetFunctionFn, ...);
```

**Actual runtime signature:**

```c
CUfunction mgpuModuleGetFunction(CUmodule module, const char *name)
```

Two pointer parameters returning a pointer — matches the `{ptrTy, ptrTy} -> ptrTy`
declaration. Correct.

**Style note:** The sketch calls `.getCallee()` and then passes the raw function
type again to `CreateCall`. Upstream uses `FunctionCallee` directly in
`CreateCall` without extracting the callee pointer separately. Both compile; the
sketch pattern is slightly verbose.

---

### 2.5 `gpu::ObjectAttr` Accessor — CORRECT

**Sketch usage (line ~184):**

```cpp
auto obj = cast<gpu::ObjectAttr>(objAttr);
StringRef blob = obj.getObject().getValue();
...
vendorIds.push_back(getVendorIdFromTarget(obj.getTarget()));
```

`GPU_ObjectAttr` has parameters: `target` (Attribute), `format`
(CompilationTarget enum), `object` (StringAttr), `properties` (optional
DictionaryAttr), `kernels` (optional KernelTableAttr). `getObject()` returns
`StringAttr` and `getTarget()` returns `Attribute`. Both accessors are correct.

---

### 2.6 `gpu::BinaryOp` Accessor — MINOR ISSUE

**Sketch usage:**

```cpp
StringRef symName = binaryOp.getSymName();
ArrayAttr objects = binaryOp.getObjectsAttr();
```

`gpu::BinaryOp` is a `Symbol` and exposes `getSymName()` returning `StringAttr`
(not `StringRef` directly). The sketch implicitly converts — `StringAttr` has an
implicit `operator StringRef` so this compiles, but upstream style uses
`.getValue()` explicitly. `getObjectsAttr()` returns `ArrayAttr` — correct.

**Note:** `SelectObjectAttr.cpp` uses `op.getName()` (returning `StringAttr`)
not `getSymName()`. Both resolve to the same string because `BinaryOp` is a
`SymbolOpInterface`; `getName()` comes from the symbol interface. Either works
but `getName()` matches upstream style.

---

### 2.7 `appendToGlobalCtors` / `appendToGlobalDtors` — CORRECT

**Sketch usage:**

```cpp
appendToGlobalCtors(*module, ctorFn, /*Priority=*/123);
appendToGlobalDtors(*module, dtorFn, /*Priority=*/123);
```

These are `llvm::appendToGlobalCtors` / `llvm::appendToGlobalDtors` from
`llvm/Transforms/Utils/ModuleUtils.h`. Signatures:

```cpp
void appendToGlobalCtors(Module &M, Function *F, int Priority,
                         Constant *Data = nullptr);
void appendToGlobalDtors(Module &M, Function *F, int Priority,
                         Constant *Data = nullptr);
```

The sketch passes `*module` (by ref) and priority 123 — this matches. Priority
123 also matches `SelectObjectAttr`'s actual priority (confirmed in commit
history: "load/unload GPU modules in global ctors/dtors", Apr 2025).

---

### 2.8 `llvm::PointerType::getUnqual(ctx)` — CORRECT for LLVM 17+

```cpp
auto *ptrTy = llvm::PointerType::getUnqual(ctx);
```

LLVM 17 introduced opaque pointer types and deprecated typed pointers.
`getUnqual(LLVMContext &)` is the correct 2025-2026 idiom. Correct.

---

### 2.9 `llvm::ConstantDataArray::getString` for Binary Data — POTENTIAL BUG

**Sketch usage (line ~196):**

```cpp
auto *data = llvm::ConstantDataArray::getString(ctx, blob, /*AddNull=*/false);
```

`ConstantDataArray::getString` is designed for **null-terminated C strings**
and stores data as `i8` elements with character encoding assumptions. For binary
blobs (cubin, HSACO) that may contain arbitrary bytes including embedded nulls,
the correct call is:

```cpp
llvm::ConstantDataArray::getRaw(blob, blob.size(), i8Ty)
```

or equivalently in current LLVM:

```cpp
llvm::ConstantDataArray::get(ctx,
    llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(blob.data()), blob.size()))
```

`SelectObjectAttr.cpp` uses `ConstantDataArray::getRaw` (or constructs via
`ConstantDataArray::get` with `ArrayRef<uint8_t>`). Using `getString` on a
cubin will silently truncate at any embedded null byte. **This is a correctness
bug** — cubin files routinely contain null bytes in headers and padding.

**Fix required:** Replace `ConstantDataArray::getString(ctx, blob, false)` with
`ConstantDataArray::getRaw(blob, blob.size(), i8Ty)`.

---

### 2.10 TableGen Attribute Definition — CORRECT STRUCTURE, ONE TRAIT NOTE

**Sketch TableGen (lines ~64-78):**

```tablegen
def GPU_RuntimeSelectAttr : GPU_Attr<"RuntimeSelect", "runtime_select",
    [OffloadingTranslationAttrTrait,
     DeclareAttrInterfaceMethods<OffloadingLLVMTranslationAttrInterface>]> {
```

This correctly mirrors `GPU_SelectObjectAttr`'s trait list. `CompilationAttrs.td`
shows `SelectObjectAttr` uses only `OffloadingTranslationAttrTrait` (not
`DeclareAttrInterfaceMethods<>` separately) — the `OffloadingTranslationAttrTrait`
is a `NativeTrait` that already implies interface attachment via the `.td`
infrastructure. However, adding `DeclareAttrInterfaceMethods<>` explicitly is
valid and forces concrete method declarations. Either form compiles.

**`DefaultValuedParameter` syntax is correct.** The `$_builder.getStringAttr()`
idiom is the standard way to specify default values for `StringAttr` parameters
in TableGen.

---

### 2.11 `getVendorIdFromTarget` — CORRECT with One Gap

```cpp
StringRef ns = targetAttr.getDialect().getNamespace();
if (ns == "nvvm")  return 1;
if (ns == "rocdl") return 2;
if (ns == "xevm")  return 3;
if (ns == "spirv") return 3;
```

The Intel XeVM target is registered as `xevm` in the `xevm` dialect (PR
#148286, Aug 2025). The SPIRV target environment attribute comes from the
`spirv` dialect. Both mapping to vendor ID 3 (Level Zero) is architecturally
correct since SPIR-V binaries are loaded via the Level Zero API on Intel
hardware. The `nvvm` and `rocdl` namespace strings match the upstream NVVM and
ROCDL dialects. No issue here.

**Gap:** There is no entry for `#gpu.object<#spirv.target_env<...>, ...>` vs
`#gpu.object<#xevm.target<...>, ...>` — both map to vendor 3, so both work.
Future: if SPIR-V targeting non-Intel (e.g., OpenCL CPU devices) is added, the
vendor ID mapping would need refinement.

---

### 2.12 Destructor `mgpuModuleUnload` Call Pattern — STYLE ISSUE

**Sketch pattern (lines ~498-507):**

```cpp
auto *mgpuModuleUnloadFn = module->getOrInsertFunction(
    "mgpuModuleUnload",
    llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {ptrTy}, false))
    .getCallee();
builder.CreateCall(
    llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {ptrTy}, false),
    mgpuModuleUnloadFn, {loadedMod});
```

This redundantly specifies the function type twice — once to get the callee,
once in `CreateCall`. Upstream prefers retaining the `FunctionCallee` object
and passing it directly to `CreateCall`. Not a correctness issue, but would
generate a compiler warning about discarding the `FunctionCallee` return value.

---

## 3. Structural Completeness

### Covered well:
- Binary blob embedding for all objects (Section 2, Phase 1). Complete.
- Dispatch table emission (Phase 2). Complete and well-reasoned.
- Mutable state globals with -1 sentinel (Phase 3). Complete.
- Constructor control flow with three basic blocks (entry/load/fallback). Complete.
- CPU fallback path (null module_ptr) and error path (puts + abort). Complete.
- Destructor with null guard. Complete.
- launchKernel as a thin load-and-launch (Section 3). Complete.
- Vendor detection helper with dlopen + init probe (Section 4). Complete.
- Ranking helper with three strategies (Section 5). Strategies 1 and 2 are
  placeholder stubs, clearly documented as such.
- Helper utilities (`getVendorIdFromTarget`, `encodeStrategy`). Complete.

### Not covered / acknowledged gaps in the sketch:
- **Parameter extraction in launchKernel:** The sketch explicitly marks the
  grid/block/stream/params extraction as elided (`[... elided for poster
  clarity ...]`). This is the most substantial omitted piece — roughly 30-40
  lines of `moduleTranslation.lookupValue()` calls. Not a design flaw; the
  sketch correctly defers to `SelectObjectAttr.cpp` lines ~180-220 as the
  pattern to follow.
- **JIT format handling:** `mgpuModuleLoadJIT` path not implemented.
  Acknowledged in spirit but not called out explicitly.
- **Thread-safety of global_ctors:** Acknowledged as a known issue
  (redundant dlopen across TUs; `call_once` guard not yet designed).
- **Multiple-TU symName collision:** Acknowledged as a design issue.
- **rank_by_priority and rank_by_capability strategies:** Both are stubs.
  The sketch honestly labels them incomplete.
- **`paramsCount` in mgpuLaunchKernel:** Not present (the primary compilability
  bug in launchKernel).

---

## 4. What Would Need to Change to Compile Against MLIR Trunk

In order of severity:

### 4.1 Critical — Would Prevent Compilation

**a. Interface method signatures (both `embedBinary` and `launchKernel`):**

Add leading `Attribute attribute` parameter and trailing `const` to match
the generated interface dispatch:

```cpp
// embedBinary
LogicalResult embedBinary(Attribute attribute, Operation *op,
                          llvm::IRBuilderBase &builder,
                          LLVM::ModuleTranslation &moduleTranslation) const {
  auto self = llvm::cast<RuntimeSelectAttr>(attribute);
  // use self.getStrategy(), self.getFallback() instead of this->...
  ...
}

// launchKernel
LogicalResult launchKernel(Attribute attribute,
                           Operation *launchFuncOp, Operation *binaryOp,
                           llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) const {
  ...
}
```

**b. `mgpuLaunchKernel` missing 12th parameter (paramsCount i64):**

```cpp
auto *mgpuLaunchKernelFnTy = llvm::FunctionType::get(
    llvm::Type::getVoidTy(ctx),
    {ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
     i32Ty, ptrTy, ptrTy, ptrTy, i64Ty},   // +i64Ty at end
    false);
// ...and add paramsCount to the CreateCall args
```

### 4.2 Correctness Bug — Compiles But Silently Wrong

**c. `ConstantDataArray::getString` → `getRaw` for binary blobs:**

```cpp
// WRONG (truncates at null bytes in cubin/hsaco):
auto *data = llvm::ConstantDataArray::getString(ctx, blob, false);

// CORRECT:
auto *data = llvm::ConstantDataArray::getRaw(blob, blob.size(), i8Ty);
// or:
auto *data = llvm::ConstantDataArray::get(
    ctx, llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t*>(blob.data()), blob.size()));
```

### 4.3 Minor — Style / Warnings, Not Errors

**d. Grid/block dimension types:** Change `i64Ty` to `intPtrTy` for
`mgpuLaunchKernel` dimensions (no functional issue on 64-bit; upstream
consistency).

**e. Duplicate FunctionType in destructor:** Retain `FunctionCallee` and pass
directly to `CreateCall`.

**f. `binaryOp.getSymName()` → `binaryOp.getName()`:** Matches upstream style
(`SelectObjectAttr` uses `getName()`).

### 4.4 Missing Includes

The following headers are not listed in the sketch but are required:
- `mlir/Dialect/GPU/IR/GPUDialect.h` (for `gpu::BinaryOp`, `gpu::ObjectAttr`,
  `gpu::LaunchFuncOp`)
- `mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h` (interface decl)
- `llvm/Transforms/Utils/ModuleUtils.h` (for `appendToGlobalCtors/Dtors`)
- `llvm/IR/Constants.h`, `llvm/IR/GlobalVariable.h`, `llvm/IR/Function.h`
- `dlfcn.h` in the runtime wrappers file (for `dlopen`/`dlsym`/`dlclose`)

### 4.5 Registration Boilerplate

A real implementation also needs:
- `registerRuntimeSelectAttrInterface()` or equivalent registration in
  `GPUToLLVMIRTranslation.cpp`
- CMakeLists.txt entry for the new source file
- `add_mlir_library` dependency on `MLIRGPUToLLVMIRTranslation`

---

## 5. Risk Assessment for Dublin

### Technical Accuracy — HIGH CONFIDENCE

The core architecture is correct and the design intent is coherent. The sketch
accurately represents:
- How `OffloadingLLVMTranslationAttrInterface` is used in practice.
- The LLVM IR that would be emitted (validated by `runtime_select_example.ll`).
- The mgpu* runtime wrapper ABI.
- The priority-123 constructor pattern from upstream.
- The zero-hot-path-overhead claim (launchKernel is structurally identical to
  SelectObjectAttr).

### API Bugs That Reviewers Might Catch

There are **three issues** that an MLIR compiler engineer would notice
immediately:

1. **Missing `Attribute attribute` parameter** — any reviewer who has written
   or reviewed `OffloadingLLVMTranslationAttrInterface` implementations will
   spot this. Moderate risk.

2. **mgpuLaunchKernel missing paramsCount** — caught by anyone comparing to
   `SelectObjectAttr.cpp`. Low-to-moderate risk (it's easy to miss a trailing
   unused parameter).

3. **`ConstantDataArray::getString` for binary data** — a subtle but real bug.
   High credibility risk if a reviewer knows LLVM IR well, because this would
   produce silently broken code for any cubin with embedded nulls.

### Claims That Are Fully Defensible

- "Zero per-launch overhead" — proven by `launchKernel` structure and the
  `.ll` proof file.
- "Extends SelectObjectAttr without replacing it" — correct; two separate
  `OffloadingLLVMTranslationAttrInterface` implementations can coexist.
- "~400 new LOC in one file" — the sketch itself is 889 lines but is heavily
  commented; the actual implementation would be ~400 production LOC.
- Prototype validation numbers (kdl.c on GTX 1650, ~12 µs detection, ~46 µs
  module load, ~0 ns per-launch) — grounded in the prototype.
- RFC #88170 alignment — the offloadingHandler slot is exactly where this
  attribute belongs.
- XeVM tri-vendor claim — Intel XeVM PR #148286 is real and merged Aug 2025.

### Overall Dublin Risk: LOW-MEDIUM

The sketch is architecturally sound. The three bugs above should be fixed in
the comments or an errata section before the poster is printed. The
`ConstantDataArray::getString` bug is the only one that would produce
incorrect output in a real binary; the interface signature mismatch and missing
paramsCount are compilation errors that would be caught immediately when
attempting to build.

**Recommendation:** Add a one-line "API delta from SelectObjectAttr" note to
the poster acknowledging the four changes required for compilability. This
demonstrates awareness of the codebase and turns a potential attack vector into
a show of rigor.

---

## 6. Cross-Reference: SelectObjectAttr Pattern Alignment

| Aspect | SelectObjectAttr (upstream) | RuntimeSelectAttr (sketch) | Match? |
|--------|----------------------------|---------------------------|--------|
| Interface | OffloadingLLVMTranslationAttrInterface | Same | Yes |
| Trait | OffloadingTranslationAttrTrait | Same | Yes |
| embedBinary first param | `Attribute attribute` | Missing | **No** |
| launchKernel first param | `Attribute attribute` | Missing | **No** |
| const method qualifier | Yes | Missing | **No** |
| ctor/dtor priority | 123 | 123 | Yes |
| appendToGlobalCtors/Dtors | Yes, from ModuleUtils.h | Yes | Yes |
| mgpuModuleLoad signature | `(ptr, i64)` | `(ptr, i64)` | Yes |
| mgpuModuleLoadJIT path | Yes (for Assembly format) | Not handled | Partial |
| mgpuLaunchKernel params | 12 (trailing i64 paramsCount) | 11 (missing) | **No** |
| Blob embedding method | ConstantDataArray::getRaw | ::getString (bug) | **No** |
| Opaque pointer type | getUnqual(ctx) | getUnqual(ctx) | Yes |
| Grid/block dim type | intPtrTy | i64Ty | Minor delta |

---

*Report generated: 2026-04-10*
*MLIR trunk reference: April 2026 (post-XeVM merge, post-Apr-3 assemblySize commit)*
