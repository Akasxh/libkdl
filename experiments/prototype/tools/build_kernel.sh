#!/usr/bin/env bash
# build_kernel.sh — MLIR compilation pipeline for mlir-hetero-dispatch
#
# Usage:
#   ./build_kernel.sh [INPUT.mlir]
#   ./build_kernel.sh matmul.mlir        # produces matmul.sm_80.cubin, matmul.x86.o, matmul.mtb
#
# Requires:
#   mlir-opt        (from an LLVM/MLIR build with GPU support)
#   mlir-translate  (same build)
#   llc             (same build, with x86 and NVPTX backends)
#   python3         (for kdl_bundle.py)
#
# Environment variables:
#   LLVM_BUILD      Path to LLVM build directory (default: ~/llvm-build)
#   MLIR_OPT        Path to mlir-opt binary     (default: $LLVM_BUILD/bin/mlir-opt)
#   MLIR_TRANSLATE  Path to mlir-translate       (default: $LLVM_BUILD/bin/mlir-translate)
#   LLC             Path to llc                  (default: $LLVM_BUILD/bin/llc)
#   NVPTX_ARCH      CUDA compute arch            (default: sm_80)
#   OPT_LEVEL       Optimization level           (default: 3)

set -euo pipefail

# ── Tool resolution ──────────────────────────────────────────────────────────

LLVM_BUILD="${LLVM_BUILD:-${HOME}/llvm-build}"
MLIR_OPT="${MLIR_OPT:-${LLVM_BUILD}/bin/mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-${LLVM_BUILD}/bin/mlir-translate}"
LLC="${LLC:-${LLVM_BUILD}/bin/llc}"
NVPTX_ARCH="${NVPTX_ARCH:-sm_80}"
OPT_LEVEL="${OPT_LEVEL:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDL_BUNDLE="${SCRIPT_DIR}/kdl_bundle.py"

check_tool() {
    local name="$1"
    local path="$2"
    if ! command -v "$path" &>/dev/null && [ ! -x "$path" ]; then
        echo "[ERROR] $name not found at: $path" >&2
        echo "        Set ${name//-/_} env var or LLVM_BUILD to your LLVM build directory." >&2
        echo "        Build LLVM with: cmake -DLLVM_ENABLE_PROJECTS='mlir' \\" >&2
        echo "                                -DLLVM_TARGETS_TO_BUILD='X86;NVPTX;AMDGPU' ..." >&2
        return 1
    fi
}

missing_tools=0
check_tool "mlir-opt"       "$MLIR_OPT"       || missing_tools=1
check_tool "mlir-translate" "$MLIR_TRANSLATE"  || missing_tools=1
check_tool "llc"            "$LLC"             || missing_tools=1

if [ ! -f "$KDL_BUNDLE" ]; then
    echo "[ERROR] kdl_bundle.py not found at: $KDL_BUNDLE" >&2
    missing_tools=1
fi

if [ "$missing_tools" -ne 0 ]; then
    echo "" >&2
    echo "At least one required tool is missing. Aborting." >&2
    exit 1
fi

# ── Input handling ───────────────────────────────────────────────────────────

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 INPUT.mlir" >&2
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "[ERROR] Input file not found: $INPUT" >&2
    exit 1
fi

# Derive base name (e.g. "matmul" from "matmul.mlir" or "../kernels/matmul.mlir")
BASE="$(basename "${INPUT%.mlir}")"
WORKDIR="$(pwd)"

echo "=== mlir-hetero-dispatch build pipeline ==="
echo "    input:     $INPUT"
echo "    base:      $BASE"
echo "    nvptx arch: $NVPTX_ARCH"
echo "    opt level:  $OPT_LEVEL"
echo ""

# ── Phase 0: High-level linalg → GPU loops ───────────────────────────────────
#
# Bufferize tensors to memrefs, lower linalg to affine loops, then convert the
# parallel loop structure to gpu.launch blocks.  We use convert-linalg-to-loops
# rather than convert-linalg-to-affine-loops because the GPU backend expects
# scf.parallel ops that convert-parallel-loops-to-gpu can consume.

echo "[1/6] Phase 0 — linalg -> GPU launch"
"$MLIR_OPT" "$INPUT" \
    --one-shot-bufferize="allow-return-allocs-from-loops allow-unknown-ops" \
    --convert-linalg-to-parallel-loops \
    --convert-parallel-loops-to-gpu \
    -o "${BASE}.gpu.mlir"
echo "      -> ${BASE}.gpu.mlir"

# ── Phase 1: GPU kernel outlining + multi-target annotation ─────────────────
#
# gpu-kernel-outlining extracts each gpu.launch body into a gpu.module /
# gpu.func @kernel construct.  We then attach NVPTX and ROCm targets; each
# attach-target pass annotates the gpu.module with compilation metadata but
# does NOT yet compile — that happens in Phase 2.

echo "[2/6] Phase 1 — gpu-kernel-outlining + target attachment"
"$MLIR_OPT" "${BASE}.gpu.mlir" \
    --gpu-kernel-outlining \
    --nvvm-attach-target="chip=${NVPTX_ARCH} features=+ptx80 opt-level=${OPT_LEVEL}" \
    --rocdl-attach-target="chip=gfx942" \
    -o "${BASE}.outlined.mlir"
echo "      -> ${BASE}.outlined.mlir"

# ── Phase 2: Multi-target binary serialization ───────────────────────────────
#
# gpu-module-to-binary clones the gpu.module for each attached target, runs
# the appropriate target-specific lowering pipeline (convert-gpu-to-nvvm or
# convert-gpu-to-rocdl), serializes each clone to a device binary (cubin /
# hsaco), and embeds the results as a gpu.binary attribute.
#
# This step requires:
#   - CUDA toolkit (nvcc or ptxas) in PATH for the NVPTX target
#   - ROCm toolkit (clang with AMDGPU support) for the ROCm target
# If either toolkit is absent, mlir-opt will print a warning and skip that
# target rather than failing the whole pipeline.

echo "[3/6] Phase 2 — gpu-module-to-binary (multi-target)"
"$MLIR_OPT" "${BASE}.outlined.mlir" \
    --gpu-module-to-binary \
    -o "${BASE}.binary.mlir"
echo "      -> ${BASE}.binary.mlir"

# ── Phase 2b: Extract device binaries ────────────────────────────────────────
#
# gpu-module-to-binary embeds binaries as MLIR attributes.  We use
# mlir-translate to extract each one:
#   - --mlir-to-cubin  -> extracts the first NVPTX cubin
#   - --mlir-to-hsaco  -> extracts the first AMDGCN hsaco
# (mlir-translate flags are build-dependent; adjust if your MLIR version
#  exposes them under different names.)

echo "[4/6] Extracting device binaries"
CUBIN="${BASE}.${NVPTX_ARCH}.cubin"
HSACO="${BASE}.gfx942.hsaco"

if "$MLIR_TRANSLATE" --mlir-to-cubin "${BASE}.binary.mlir" -o "$CUBIN" 2>/dev/null; then
    echo "      -> $CUBIN  ($(wc -c < "$CUBIN") bytes)"
else
    echo "      [SKIP] NVPTX cubin extraction failed (no CUDA toolkit?)"
    CUBIN=""
fi

if "$MLIR_TRANSLATE" --mlir-to-hsaco "${BASE}.binary.mlir" -o "$HSACO" 2>/dev/null; then
    echo "      -> $HSACO  ($(wc -c < "$HSACO") bytes)"
else
    echo "      [SKIP] AMDGCN hsaco extraction failed (no ROCm toolkit?)"
    HSACO=""
fi

# ── Phase 3: CPU fallback via native LLVM IR → x86 ELF object ───────────────
#
# We start from the original input (not the GPU-outlined version) and lower
# fully to LLVM IR via the CPU path, then assemble to a native ELF object.
# The entry symbol will be <BASE>_matmul_kernel (or as reported by mlir-opt).

echo "[5/6] Phase 3 — CPU fallback: linalg -> LLVM IR -> x86 ELF"
X86_OBJ="${BASE}.x86.o"

"$MLIR_OPT" "$INPUT" \
    --one-shot-bufferize="allow-return-allocs-from-loops allow-unknown-ops" \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --convert-math-to-llvm \
    --expand-strided-metadata \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "${BASE}.cpu.llvm.mlir"

"$MLIR_TRANSLATE" --mlir-to-llvmir "${BASE}.cpu.llvm.mlir" \
    | "$LLC" -march=x86-64 -mcpu=native -filetype=obj -o "$X86_OBJ"

echo "      -> $X86_OBJ  ($(wc -c < "$X86_OBJ") bytes)"

# ── Phase 4: Bundle generation ───────────────────────────────────────────────
#
# kdl_bundle.py packs all available device binaries into a single MTB file.
# The capability contracts (JSON) encode the minimum hardware requirements
# that libkdl will check at runtime via kdl_contract_matches().

echo "[6/6] Phase 4 — generating MTB bundle"
MTB="${BASE}.mtb"

BUNDLE_ARGS=(
    --kernel "${BASE}"
    -o "$MTB"
)

if [ -n "$CUBIN" ]; then
    NVPTX_CONTRACT="$(printf '{"target":"nvptx","min_arch":"%s","min_shared_mem_kb":48}' "$NVPTX_ARCH")"
    BUNDLE_ARGS+=(--variant "nvptx:${NVPTX_ARCH}:${CUBIN}:${NVPTX_CONTRACT}:${BASE}_kernel")
fi

if [ -n "$HSACO" ]; then
    AMDGCN_CONTRACT='{"target":"amdgcn","min_arch":"gfx942"}'
    BUNDLE_ARGS+=(--variant "amdgcn:gfx942:${HSACO}:${AMDGCN_CONTRACT}:${BASE}_kernel")
fi

# x86 fallback is always included
X86_CONTRACT='{"target":"x86","min_arch":"x86-64-v3"}'
BUNDLE_ARGS+=(--variant "x86:x86-64-v3:${X86_OBJ}:${X86_CONTRACT}:${BASE}_kernel")

python3 "$KDL_BUNDLE" "${BUNDLE_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "    MTB bundle: ${WORKDIR}/${MTB}"
echo ""
echo "Load with libkdl at runtime:"
echo "    kdl_ctx ctx;"
echo "    kdl_init(&ctx);"
echo "    kdl_bundle_t b;"
echo "    kdl_load_bundle(ctx, \"${MTB}\", &b);"
echo "    kdl_kernel_t k;"
echo "    kdl_select_kernel(ctx, b, \"${BASE}\", -1, &k);"
echo "    kdl_launch(k, grid_x, grid_y, 1, block_x, block_y, 1, 0, args);"
