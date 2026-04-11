#!/bin/bash
# generate_null_cubin.sh — Create null kernel CUBIN for dispatch benchmarks
# Usage: ./generate_null_cubin.sh [output_dir]
#
# Generates null_sm75.cubin (and optionally PTX) for use with:
#   - bench_layers (as embedded or file input)
#   - runtime_select_poc (as directory or OffloadBinary input)
#   - bench_ptx_vs_cubin (CUBIN vs PTX comparison)

set -euo pipefail

OUTDIR="${1:-/tmp}"

cat > "${OUTDIR}/null_kernel.cu" << 'CUDA'
extern "C" __global__ void null_kernel() {}
extern "C" __global__ void add_kernel(float* a, const float* b, const float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + c[i];
}
CUDA

echo "Compiling null kernel to CUBIN (sm_75)..."
nvcc -arch=sm_75 -cubin -o "${OUTDIR}/null_sm75.cubin" "${OUTDIR}/null_kernel.cu"
echo "  → ${OUTDIR}/null_sm75.cubin ($(wc -c < "${OUTDIR}/null_sm75.cubin") bytes)"

echo "Compiling null kernel to PTX (sm_75)..."
nvcc -arch=sm_75 -ptx -o "${OUTDIR}/null_sm75.ptx" "${OUTDIR}/null_kernel.cu"
echo "  → ${OUTDIR}/null_sm75.ptx ($(wc -c < "${OUTDIR}/null_sm75.ptx") bytes)"

echo "Done. Use with:"
echo "  ./bench_layers                              # uses embedded PTX"
echo "  ./runtime_select_poc ${OUTDIR}/             # loads cubins from directory"
echo "  ./bench_ptx_vs_cubin ${OUTDIR}/null_sm75.cubin ${OUTDIR}/null_sm75.ptx"
