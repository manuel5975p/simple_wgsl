#!/bin/bash
# Run both benchmarks and save results for comparison
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

gcc -O2 -Wall -o bench_cuvk bench_cuvk.c -ldl -lm

echo "Running Real CUDA benchmark..."
./bench_cuvk /usr/lib/libcuda.so.1 "Real CUDA" > results_real.txt 2>&1

echo "Running cuvk benchmark..."
./bench_cuvk ../build/cuvk_runtime/libcuda.so.1 "cuvk (Vulkan)" > results_cuvk.txt 2>&1

echo ""
echo "Results saved to results_real.txt and results_cuvk.txt"
echo ""
echo "Use: diff -y results_real.txt results_cuvk.txt"
