// matmul.mlir — Matrix multiplication kernel in MLIR linalg dialect
//
// This file is the entry point for the mlir-hetero-dispatch build pipeline.
// It defines a rank-2 matmul (C = A * B) using linalg.matmul, which is the
// canonical MLIR representation before any target-specific lowering.
//
// Compilation pipeline (see ../tools/build_kernel.sh):
//   1. one-shot-bufferize          -> convert tensors to memrefs
//   2. convert-linalg-to-affine-loops -> loop nest
//   3. convert-parallel-loops-to-gpu  -> gpu.launch block
//   4. gpu-kernel-outlining        -> gpu.module / gpu.func kernel
//   5. nvvm-attach-target / rocdl-attach-target
//   6. gpu-module-to-binary        -> gpu.binary with embedded cubins/hsacos
//   7. (separate CPU path) llc    -> x86 ELF object
//   8. kdl_bundle.py               -> .mtb bundle

module @matmul_module {

  // ── Tensor-level matmul (input to the bufferization pass) ────────────────
  //
  // Signature:
  //   %A : tensor<M x K x f32>   (left operand)
  //   %B : tensor<K x N x f32>   (right operand)
  //   %C : tensor<M x N x f32>   (accumulator, overwritten with A*B + C)
  //
  // Dynamic shapes let the same kernel handle arbitrary tile sizes at runtime.
  func.func @matmul(
      %A : tensor<?x?xf32>,
      %B : tensor<?x?xf32>,
      %C : tensor<?x?xf32>) -> tensor<?x?xf32>
  {
    // linalg.matmul implements C += A * B over the standard matmul indexing map:
    //   C[i, j] += A[i, k] * B[k, j]
    %result = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                            outs(%C        : tensor<?x?xf32>)
             -> tensor<?x?xf32>
    func.return %result : tensor<?x?xf32>
  }

  // ── Static-shape variant (useful for fixed-size tile benchmarks) ─────────
  //
  // 128 x 128 tile — matches a single thread-block tile on sm_80 with 48 KB
  // shared memory at f32 precision (128*128*4 = 65536 bytes, fits in two
  // operand tiles using double-buffering at 32 KB each).
  func.func @matmul_128x128x128(
      %A : tensor<128x128xf32>,
      %B : tensor<128x128xf32>,
      %C : tensor<128x128xf32>) -> tensor<128x128xf32>
  {
    %result = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%C        : tensor<128x128xf32>)
             -> tensor<128x128xf32>
    func.return %result : tensor<128x128xf32>
  }

  // ── Memref variant (post-bufferization reference for verification) ────────
  //
  // build_kernel.sh produces this form automatically via one-shot-bufferize.
  // Keeping it here lets us verify the bufferization result with mlir-opt
  // without re-running the pass:
  //
  //   mlir-opt --verify-each matmul.mlir
  //
  func.func @matmul_memref(
      %A  : memref<?x?xf32>,
      %B  : memref<?x?xf32>,
      %C  : memref<?x?xf32>)
  {
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                  outs(%C       : memref<?x?xf32>)
    func.return
  }

} // module @matmul_module
