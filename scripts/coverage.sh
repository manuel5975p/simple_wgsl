#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-coverage"
REPORT="$ROOT/coverage-report"

# Configure with clang + coverage
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DWGSL_COVERAGE=ON \
    -DWGSL_BUILD_TESTS=ON

# Build
cmake --build "$BUILD"

# Run tests, collecting profile data
LLVM_PROFILE_FILE="$BUILD/default.profraw" \
    ctest --test-dir "$BUILD" --output-on-failure

# Merge raw profiles
llvm-profdata merge -sparse "$BUILD/default.profraw" -o "$BUILD/coverage.profdata"

BIN="$BUILD/tests/wgsl_tests"
PROF="$BUILD/coverage.profdata"
FILTER=(-ignore-filename-regex='_deps|googletest|/usr/|tests/|stb_')

# Generate HTML report with source-level detail
llvm-cov show \
    -instr-profile="$PROF" \
    -format=html \
    -output-dir="$REPORT" \
    -show-line-counts-or-regions \
    -show-branches=count \
    -show-expansions \
    -tab-size=4 \
    -coverage-watermark=80,40 \
    -Xdemangler c++filt \
    -j=0 \
    "${FILTER[@]}" \
    "$BIN"

# Print summary to terminal
llvm-cov report \
    -instr-profile="$PROF" \
    -show-region-summary \
    -show-branch-summary \
    "${FILTER[@]}" \
    "$BIN"

echo ""
echo "HTML report: file://$REPORT/index.html"
