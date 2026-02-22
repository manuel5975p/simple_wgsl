#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-coverage"
REPORT="$ROOT/coverage-report"
ANNOTATED="$ROOT/coverage-annotated"

# Parse flags
FORMAT=html
for arg in "$@"; do
    case "$arg" in
        --annotated|--text) FORMAT=text ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Configure with clang + coverage
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-DNDEBUG" \
    -DCMAKE_CXX_FLAGS="-DNDEBUG" \
    -DWGSL_COVERAGE=ON \
    -DWGSL_BUILD_TESTS=ON

# Build
cmake --build "$BUILD"

# Run tests, collecting profile data.
# Allow test failures (expression tests may fail) – coverage is still collected.
LLVM_PROFILE_FILE="$BUILD/default.profraw" \
    ctest --test-dir "$BUILD" --output-on-failure || true

# Merge raw profiles
llvm-profdata merge -sparse "$BUILD/default.profraw" -o "$BUILD/coverage.profdata"

BIN="$BUILD/tests/wgsl_tests"
PROF="$BUILD/coverage.profdata"
FILTER=(-ignore-filename-regex='_deps|googletest|/usr/|tests/|stb_')

if [ "$FORMAT" = text ]; then
    # Generate annotated text files (one .txt per source file)
    mkdir -p "$ANNOTATED"
    llvm-cov show \
        -instr-profile="$PROF" \
        -format=text \
        -output-dir="$ANNOTATED" \
        -show-line-counts-or-regions \
        -show-branches=count \
        -show-expansions \
        -tab-size=4 \
        -use-color=false \
        -Xdemangler c++filt \
        "${FILTER[@]}" \
        "$BIN"
else
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
fi

# Print summary to terminal
llvm-cov report \
    -instr-profile="$PROF" \
    -show-region-summary \
    -show-branch-summary \
    "${FILTER[@]}" \
    "$BIN"

echo ""
if [ "$FORMAT" = text ]; then
    echo "Annotated text files: $ANNOTATED/"
else
    echo "HTML report: file://$REPORT/index.html"
fi
