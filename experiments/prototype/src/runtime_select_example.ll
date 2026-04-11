;; ===----------------------------------------------------------------------===
;; runtime_select_example.ll -- Hand-written LLVM IR design proof
;;
;; This file demonstrates the EXACT LLVM IR that RuntimeSelectAttr::embedBinary()
;; would emit for a gpu.binary containing two #gpu.object variants:
;;   - Variant 0: NVIDIA cubin  (vendor_id = 1)
;;   - Variant 1: CPU fallback  (vendor_id = 0)
;;
;; Input MLIR (conceptual):
;;   gpu.binary @kernels <#gpu.runtime_select<
;;       strategy = "first_compatible", fallback = "error">> [
;;     #gpu.object<#nvvm.target<chip = "sm_75">, bin = "NVBLOB00">,
;;     #gpu.object</*cpu target*/,                bin = "CPUBLOB0">
;;   ]
;;
;; Cross-reference:
;;   RuntimeSelectAttr.cpp.sketch  -- C++ that generates this IR
;;   SelectObjectAttr.cpp          -- upstream single-binary baseline
;;   kdl.c                         -- prototype validating these mechanics
;;
;; Verify well-formedness:
;;   llvm-as runtime_select_example.ll -o runtime_select_example.bc
;;   opt -verify runtime_select_example.ll -o /dev/null
;;
;; ===----------------------------------------------------------------------===


;; =========================================================================
;; Phase 1: Binary Blob Globals
;; =========================================================================
;;
;; embedBinary() Phase 1 (sketch lines 171-204):
;;   For each #gpu.object in the gpu.binary op, emit a separate
;;   [N x i8] constant holding the raw binary data (cubin, hsaco, etc.).
;;
;; WHY separate globals (not one concatenated blob):
;;   Each blob is a self-contained vendor binary. cuModuleLoadData and
;;   hipModuleLoadData expect a pointer to the START of a complete image.
;;   Separate globals let LLVM handle alignment; the dispatch table
;;   provides the indirection.
;;
;; WHY internal linkage:
;;   Same as SelectObjectAttr. Module-local constants consumed only by
;;   the constructor we emit. External visibility would pollute the
;;   symbol table and inhibit dead-stripping.
;;
;; NAMING: {symName}_blob_{idx}
;;   Suffix is the positional index, not the target name, because
;;   multiple objects can share a target (e.g., two NVVM objects for
;;   sm_75 and sm_90).
;;
;; In a real compilation, these would be thousands of bytes of cubin/hsaco.
;; Here we use 8-byte placeholders to keep the example readable.

@kernels_blob_0 = internal constant [8 x i8] c"NVBLOB00", align 1
@kernels_blob_1 = internal constant [8 x i8] c"CPUBLOB0", align 1


;; =========================================================================
;; Phase 2: Dispatch Table
;; =========================================================================
;;
;; embedBinary() Phase 2 (sketch lines 226-270):
;;   Emit a constant global array of RuntimeSelectEntry structs.
;;   Each entry maps a vendor_id to a blob pointer and its byte size.
;;
;; STRUCT LAYOUT: { i32 vendor_id, ptr blob_ptr, i64 blob_size }
;;   - vendor_id:  which runtime can load this blob
;;                 (0=NONE, 1=NVIDIA, 2=AMD, 3=INTEL)
;;   - blob_ptr:   pointer to the @kernels_blob_N constant
;;   - blob_size:  byte length needed by cuModuleLoadData et al.
;;
;; WHY a table (not if-else in the constructor):
;;   The table is data, not code. It composes with the ranking helper
;;   (__gpu_runtime_select_rank) which iterates it generically.
;;   Adding new strategies requires changing the ranking helper, not
;;   the table layout. Mirrors kdl.c's mtb_variant_entry table
;;   (kdl.c:96-106).
;;
;; Future extension: add i32 priority field when OffloadBinary metadata
;; vocabulary (T07) lands variant_priority.

%RuntimeSelectEntry = type { i32, ptr, i64 }

@kernels_dispatch_table = internal constant [2 x %RuntimeSelectEntry] [
  %RuntimeSelectEntry { i32 1, ptr @kernels_blob_0, i64 8 },
  %RuntimeSelectEntry { i32 0, ptr @kernels_blob_1, i64 8 }
], align 8


;; =========================================================================
;; Phase 3: Mutable State Globals
;; =========================================================================
;;
;; embedBinary() Phase 3 (sketch lines 280-310):
;;   Two mutable globals track the result of vendor detection.
;;
;; @kernels_selected_idx:
;;   Which dispatch table entry was chosen. -1 = not yet resolved.
;;   Used for diagnostics (e.g., dispatch flame graph reads this).
;;
;; @kernels_module_ptr:
;;   The GPU module handle, populated by the constructor.
;;   This is what launchKernel() reads -- the SAME global that
;;   SelectObjectAttr would produce. After the constructor runs,
;;   downstream code cannot distinguish runtime_select from select_object.
;;
;; WHY NOT thread-local:
;;   GPU module handles are process-global resources in both CUDA and HIP.
;;   A cuModule loaded in one thread is usable from any thread.
;;   Thread-local storage would cause redundant module loads and waste
;;   GPU memory. Same reasoning as SelectObjectAttr.

@kernels_selected_idx = internal global i32 -1, align 4
@kernels_module_ptr   = internal global ptr null, align 8


;; =========================================================================
;; Runtime Helper Declarations
;; =========================================================================
;;
;; These functions are defined in GPURuntimeSelectWrappers.c (sketch
;; Sections 4 and 5). They are linked at execution time via sharedLibPaths,
;; the same mechanism that links CudaRuntimeWrappers.cpp.
;;
;; __gpu_runtime_select_detect_vendor():
;;   dlopen-probes libcuda.so.1, libamdhip64.so, libze_loader.so.1 in order.
;;   Calls cuInit/hipInit/zeInit to verify the driver is functional.
;;   Returns vendor_id enum (0=NONE, 1=NVIDIA, 2=AMD, 3=INTEL).
;;
;; __gpu_runtime_select_rank():
;;   Linear scan of the dispatch table, returns the index of the best
;;   matching entry for the detected vendor and selected strategy.
;;   Returns -1 if no compatible entry is found.

declare i32 @__gpu_runtime_select_detect_vendor()
declare i32 @__gpu_runtime_select_rank(ptr, i32, i32, i32)

;; mgpuModuleLoad / mgpuModuleUnload:
;;   Already declared if SelectObjectAttr is present in the same module.
;;   These are the vendor-agnostic MLIR GPU runtime wrappers.
;;   mgpuModuleLoad internally calls cuModuleLoadData (NVIDIA) or
;;   hipModuleLoadData (AMD) depending on which runtime is linked.

declare ptr @mgpuModuleLoad(ptr, i64)
declare void @mgpuModuleUnload(ptr)

;; mgpuModuleGetFunction / mgpuLaunchKernel:
;;   Standard MLIR GPU runtime wrappers used by the launch function.
;;   These are identical to what SelectObjectAttr uses.

declare ptr @mgpuModuleGetFunction(ptr, ptr)
declare void @mgpuLaunchKernel(ptr, i64, i64, i64, i64, i64, i64, i32, ptr, ptr, ptr)

;; C library functions used in the error fallback path.

declare i32 @puts(ptr)
declare void @abort()


;; =========================================================================
;; Phase 4: Constructor Function (Vendor Detect + Module Load)
;; =========================================================================
;;
;; embedBinary() Phase 4 (sketch lines 320-430):
;;   This function runs at global_ctors time (before main).
;;   It detects the GPU vendor, selects the best blob from the dispatch
;;   table, loads it via mgpuModuleLoad, and stores the handle.
;;
;; REGISTERED AT PRIORITY 123 because:
;;   - SelectObjectAttr ctors use priority 0 (highest)
;;   - Sanitizer/profiler inits typically < 100
;;   - Vendor detection via dlopen is heavier, so run after simpler inits
;;   - But before application constructors (priority 65535)
;;
;; WHY a constructor (not lazy init at first kernel launch):
;;   SelectObjectAttr already loads at global_ctors time. Matching this
;;   pattern means the module is ready before main() -- no first-launch
;;   latency spike. The one-time cost (~46 us for CUDA cuModuleLoadData)
;;   is amortized over the process lifetime.
;;
;; CONTROL FLOW:
;;   entry:
;;     detect vendor -> rank entries -> branch on success/failure
;;   load:
;;     GEP into dispatch table -> load blob ptr + size -> mgpuModuleLoad
;;   fallback:
;;     (strategy="error"): print diagnostic, abort
;;     (strategy="cpu"):   leave module_ptr as null, return

define internal void @kernels_ctor() {
entry:
  ;; Step 1: Detect which vendor runtime is available on this machine.
  ;; __gpu_runtime_select_detect_vendor() dlopen-probes CUDA/HIP/L0.
  ;; Returns: 0=NONE, 1=NVIDIA, 2=AMD, 3=INTEL
  ;; Sketch line 388: builder.CreateCall(detectVendorCallee, {}, "vendor")
  %vendor = call i32 @__gpu_runtime_select_detect_vendor()

  ;; Step 2: Rank dispatch table entries for this vendor.
  ;; Arguments: table_ptr, num_entries, detected_vendor, strategy
  ;; Strategy 0 = first_compatible (default).
  ;; Returns index into dispatch table, or -1 if no match.
  ;; Sketch lines 396-401: builder.CreateCall(rankCallee, {...}, "selected_idx")
  %selected_idx = call i32 @__gpu_runtime_select_rank(
      ptr @kernels_dispatch_table, i32 2, i32 %vendor, i32 0)

  ;; Step 3: Branch on whether selection succeeded.
  ;; If selected_idx < 0, no compatible entry was found.
  ;; Sketch lines 403-405: builder.CreateICmpSLT + CreateCondBr
  %no_match = icmp slt i32 %selected_idx, 0
  br i1 %no_match, label %fallback, label %load

fallback:
  ;; FALLBACK="error" path (sketch lines 412-423):
  ;; Print a diagnostic message and abort.
  ;; A "cpu" fallback would instead just return, leaving module_ptr null.
  %msg = getelementptr [59 x i8], ptr @.str.err, i64 0, i64 0
  call i32 @puts(ptr %msg)
  call void @abort()
  unreachable

load:
  ;; Store which variant was selected (for diagnostics / flame graph).
  ;; Sketch line 427: builder.CreateStore(idx, selectedIdx)
  store i32 %selected_idx, ptr @kernels_selected_idx, align 4

  ;; GEP into the dispatch table to get the selected entry.
  ;; The table is [2 x %RuntimeSelectEntry]. We index with [0, %selected_idx].
  ;; Sketch line 430-431: builder.CreateInBoundsGEP(tableTy, tableGlobal, ...)
  %entry_ptr = getelementptr inbounds [2 x %RuntimeSelectEntry],
      ptr @kernels_dispatch_table, i64 0, i32 %selected_idx

  ;; Load blob pointer (field 1 of RuntimeSelectEntry).
  ;; Sketch line 434: builder.CreateStructGEP(entryTy, entryPtr, 1)
  %blob_field_ptr = getelementptr inbounds %RuntimeSelectEntry,
      ptr %entry_ptr, i32 0, i32 1
  %blob = load ptr, ptr %blob_field_ptr, align 8

  ;; Load blob size (field 2 of RuntimeSelectEntry).
  ;; Sketch line 436: builder.CreateStructGEP(entryTy, entryPtr, 2)
  %size_field_ptr = getelementptr inbounds %RuntimeSelectEntry,
      ptr %entry_ptr, i32 0, i32 2
  %size = load i64, ptr %size_field_ptr, align 8

  ;; Call mgpuModuleLoad -- the SAME function SelectObjectAttr calls.
  ;; Internally dispatches to cuModuleLoadData or hipModuleLoadData
  ;; based on which runtime wrapper library is linked.
  ;; Sketch lines 443-444: builder.CreateCall(mgpuModuleLoadCallee, ...)
  %gpu_module = call ptr @mgpuModuleLoad(ptr %blob, i64 %size)

  ;; Store the module handle -- this is what launchKernel reads.
  ;; After this store, @kernels_module_ptr is indistinguishable from
  ;; what SelectObjectAttr would have produced.
  ;; Sketch line 447: builder.CreateStore(mod, modulePtr)
  store ptr %gpu_module, ptr @kernels_module_ptr, align 8
  ret void
}

;; Error message for the fallback path.
@.str.err = private unnamed_addr constant [59 x i8] c"RuntimeSelectAttr: no compatible GPU runtime for 'kernels'\00", align 1


;; =========================================================================
;; Phase 5: Destructor Function (Symmetric Cleanup)
;; =========================================================================
;;
;; embedBinary() Phase 5 (sketch lines 460-500+):
;;   Symmetric cleanup at process exit. Unloads the GPU module if one
;;   was loaded. Skips if module_ptr is null (CPU fallback path).
;;   Registered at matching priority 123 so ctor/dtor are paired.
;;   Identical pattern to SelectObjectAttr's destructor.

define internal void @kernels_dtor() {
entry:
  ;; Load the module handle stored by the constructor.
  ;; Sketch line 477: builder.CreateLoad(ptrTy, modulePtr, "mod")
  %mod = load ptr, ptr @kernels_module_ptr, align 8

  ;; Guard: if module_ptr is null (CPU fallback or no GPU), skip unload.
  ;; Sketch line 480: builder.CreateICmpEQ(loadedMod, null)
  %is_null = icmp eq ptr %mod, null
  br i1 %is_null, label %done, label %unload

unload:
  ;; Sketch lines 501-506: builder.CreateCall(mgpuModuleUnload, ...)
  call void @mgpuModuleUnload(ptr %mod)
  br label %done

done:
  ret void
}


;; =========================================================================
;; Global Constructors / Destructors Registration
;; =========================================================================
;;
;; Sketch line 450: appendToGlobalCtors(*module, ctorFn, /*Priority=*/123)
;; Sketch line 515: appendToGlobalDtors(*module, dtorFn, /*Priority=*/123)
;;
;; Priority 123 ordering:
;;   0       = SelectObjectAttr ctors (static single-binary loads first)
;;   <100    = sanitizer/profiler initializers
;;   123     = RuntimeSelectAttr ctor (dlopen-probe is heavier)
;;   65535   = application-level constructors (default)

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 123, ptr @kernels_ctor, ptr null }
]

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 123, ptr @kernels_dtor, ptr null }
]


;; =========================================================================
;; Launch Function -- IDENTICAL to SelectObjectAttr
;; =========================================================================
;;
;; launchKernel() (sketch Section 3, lines 522-610):
;;   THIS IS THE ENTIRE POINT: zero hot-path overhead.
;;
;;   SelectObjectAttr::launchKernel loads from @{sym}_module_ptr and calls
;;   mgpuModuleGetFunction + mgpuLaunchKernel.
;;
;;   RuntimeSelectAttr::launchKernel does EXACTLY the same thing.
;;   The dispatch table, vendor detection, and binary selection all
;;   happened in the constructor (Phase 4). By the time any kernel
;;   launches, @kernels_module_ptr holds a valid, vendor-appropriate
;;   module handle.
;;
;;   Compare this function side-by-side with SelectObjectAttr's output:
;;   the ONLY difference in the entire module is phases 1-5 above.
;;   This function is byte-for-byte identical.
;;
;; Kernel name string constant (from gpu.launch_func op's kernel attribute).

@.str.kernel_name = private unnamed_addr constant [11 x i8] c"matmul_f32\00", align 1

;; The launch function would be emitted inline at each gpu.launch_func
;; call site. Here we show it as a standalone function for clarity.
;; Arguments are placeholders; in practice they come from the
;; gpu.launch_func op's grid/block dimensions.

define void @launch_kernel(ptr %stream) {
entry:
  ;; Load the module handle (set at global_ctors time by @kernels_ctor).
  ;; This is a SINGLE load instruction. No dispatch table lookup.
  ;; No vendor check. Just a load from the global.
  ;; Sketch line 568: builder.CreateLoad(ptrTy, modulePtr, "gpu_module")
  %gpu_module = load ptr, ptr @kernels_module_ptr, align 8

  ;; Get the kernel function handle from the loaded module.
  ;; mgpuModuleGetFunction(module, "matmul_f32") -> function handle.
  ;; Sketch lines 579-584: builder.CreateCall(mgpuModuleGetFunctionFn, ...)
  %kernel_func = call ptr @mgpuModuleGetFunction(
      ptr %gpu_module, ptr @.str.kernel_name)

  ;; Launch the kernel with placeholder dimensions.
  ;; gridX=1, gridY=1, gridZ=1, blockX=256, blockY=1, blockZ=1
  ;; sharedMem=0, stream=%stream, params=null, extra=null
  ;;
  ;; In a real compilation, grid/block dims come from the gpu.launch_func
  ;; op via moduleTranslation.lookupValue(). The call is byte-for-byte
  ;; identical to what SelectObjectAttr emits.
  ;; Sketch lines 607-612: builder.CreateCall(mgpuLaunchKernelFnTy, ...)
  call void @mgpuLaunchKernel(
      ptr %kernel_func,
      i64 1, i64 1, i64 1,       ; gridX, gridY, gridZ
      i64 256, i64 1, i64 1,     ; blockX, blockY, blockZ
      i32 0,                      ; sharedMemBytes
      ptr %stream,                ; CUDA/HIP stream
      ptr null,                   ; kernel params (placeholder)
      ptr null)                   ; extra (always null)

  ret void
}


;; =========================================================================
;; Summary: What this IR proves
;; =========================================================================
;;
;; 1. COMPLETENESS: Every phase of embedBinary() (blob globals, dispatch
;;    table, state globals, constructor, destructor) maps to concrete,
;;    well-formed LLVM IR. No hand-waving.
;;
;; 2. ZERO HOT-PATH OVERHEAD: @launch_kernel reads from @kernels_module_ptr
;;    and calls mgpuModuleGetFunction + mgpuLaunchKernel. This is identical
;;    to SelectObjectAttr. The per-launch path has zero additional
;;    instructions from runtime_select.
;;
;; 3. CLEAN SEPARATION: All dispatch logic (vendor detection, table ranking,
;;    module loading) is confined to @kernels_ctor, which runs once at
;;    process startup. The launch path is oblivious to it.
;;
;; 4. COMPOSABILITY: The constructor calls well-defined C functions
;;    (__gpu_runtime_select_detect_vendor, __gpu_runtime_select_rank,
;;    mgpuModuleLoad). Each can be tested independently. New strategies
;;    require changing only the rank function, not the IR emission.
;;
;; 5. SYMMETRY WITH UPSTREAM: The destructor mirrors SelectObjectAttr
;;    exactly. The global_ctors/global_dtors registration pattern is
;;    identical. The mgpu* wrapper functions are shared.
;;
;; 6. EXTENSIBILITY: Adding a third variant (e.g., AMD hsaco) requires
;;    only: one more @kernels_blob_N global, one more dispatch table
;;    entry, and incrementing the num_entries argument to rank. The
;;    constructor logic is unchanged.
;;
;; ===----------------------------------------------------------------------===
