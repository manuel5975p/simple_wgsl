#!/bin/bash
#
# run_preload_test.sh - Full E2E test using LD_PRELOAD
#
# Compiles .cu kernels with nvcc → PTX, builds a host test program linked
# against the real CUDA stub library, then runs it with LD_PRELOAD pointing
# to our libcuvk_runtime.so. This proves the Vulkan backend is a drop-in
# replacement for libcuda.so.
#
# Usage:
#   ./run_preload_test.sh                  # auto-detect paths
#   ./run_preload_test.sh /path/to/build   # specify build directory
#

set -euo pipefail

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${1:-$SRC_DIR/build}"

NVCC="${NVCC:-/opt/cuda/bin/nvcc}"
CUDA_INCLUDE="${CUDA_INCLUDE:-/opt/cuda/include}"
CUDA_STUBS="${CUDA_STUBS:-/opt/cuda/lib64/stubs}"
LIBCUVK="$BUILD_DIR/cuvk_runtime/libcuvk_runtime.so"

KERNEL_DIR="$SCRIPT_DIR/kernels"
WORK_DIR="$(mktemp -d /tmp/cuvk_preload_XXXXXX)"
trap "rm -rf $WORK_DIR" EXIT

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}=== CUDA-on-Vulkan LD_PRELOAD E2E Test ===${RESET}"
echo ""

# ---- Sanity checks ----
if [ ! -x "$NVCC" ]; then
    echo "nvcc not found at $NVCC" >&2; exit 1
fi
if [ ! -f "$LIBCUVK" ]; then
    echo "libcuvk_runtime.so not found at $LIBCUVK (build first)" >&2; exit 1
fi
if [ ! -d "$CUDA_STUBS" ]; then
    echo "CUDA stubs not found at $CUDA_STUBS" >&2; exit 1
fi

echo "nvcc:     $NVCC"
echo "stubs:    $CUDA_STUBS"
echo "runtime:  $LIBCUVK"
echo ""

# ---- Step 1: Compile kernels to PTX ----
echo -e "${BOLD}[1/3] Compiling .cu → .ptx${RESET}"

GPU_ARCH="${GPU_ARCH:-sm_75}"
PTX_FILES=""

for cu in "$KERNEL_DIR"/*.cu; do
    name="$(basename "$cu" .cu)"
    ptx="$WORK_DIR/${name}.ptx"
    echo "  $cu → $ptx"
    "$NVCC" --ptx -arch="$GPU_ARCH" -o "$ptx" "$cu"
    PTX_FILES="$PTX_FILES $ptx"
done
echo ""

# ---- Step 2: Build host test (linked against CUDA stub) ----
echo -e "${BOLD}[2/3] Building host test program${RESET}"

HOST_SRC="$SCRIPT_DIR/preload_test.c"
HOST_BIN="$WORK_DIR/preload_test"

echo "  gcc $HOST_SRC → $HOST_BIN"
gcc -O2 -o "$HOST_BIN" "$HOST_SRC" \
    -I"$CUDA_INCLUDE" \
    -L"$CUDA_STUBS" -lcuda -lm \
    -Wl,-rpath,"$CUDA_STUBS"

# Verify it's linked against libcuda.so (from stubs)
echo "  Linked libraries:"
ldd "$HOST_BIN" 2>&1 | grep -E "cuda|vulkan" | sed 's/^/    /'
echo ""

# ---- Step 3: Run with LD_PRELOAD ----
echo -e "${BOLD}[3/3] Running with LD_PRELOAD${RESET}"
echo "  LD_PRELOAD=$LIBCUVK"
echo "  $HOST_BIN$PTX_FILES"
echo ""

if LD_PRELOAD="$LIBCUVK" "$HOST_BIN" $PTX_FILES; then
    echo ""
    echo -e "${GREEN}${BOLD}ALL TESTS PASSED${RESET}"
    exit 0
else
    echo ""
    echo -e "${RED}${BOLD}SOME TESTS FAILED${RESET}"
    exit 1
fi
