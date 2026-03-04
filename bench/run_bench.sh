#!/bin/bash
# run_bench.sh - Compile and run cuvk vs real CUDA benchmarks
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "Compiling benchmark..."
gcc -O2 -Wall -o "$SCRIPT_DIR/bench_cuvk" "$SCRIPT_DIR/bench_cuvk.c" -ldl -lm

REAL_CUDA="/usr/lib/libcuda.so.1"
CUVK_CUDA="$BUILD_DIR/cuvk_runtime/libcuda.so.1"

if [ ! -f "$CUVK_CUDA" ]; then
    echo "ERROR: cuvk library not found at $CUVK_CUDA"
    echo "Build the project first: cd build && cmake .. && ninja"
    exit 1
fi

if [ ! -f "$REAL_CUDA" ]; then
    echo "ERROR: Real CUDA library not found at $REAL_CUDA"
    exit 1
fi

echo ""
echo "================================================================"
echo "  Running benchmark with REAL CUDA"
echo "================================================================"
"$SCRIPT_DIR/bench_cuvk" "$REAL_CUDA" "Real CUDA"

echo ""
echo ""
echo "================================================================"
echo "  Running benchmark with cuvk (CUDA-on-Vulkan)"
echo "================================================================"
"$SCRIPT_DIR/bench_cuvk" "$CUVK_CUDA" "cuvk (Vulkan)"
