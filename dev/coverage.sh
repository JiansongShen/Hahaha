#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./dev/coverage.sh [builddir] [--clean] [--cuda|--no-cuda]
#
# Policy:
# - Focus on "core" and exclude display (UI/visualization) code from coverage.
# - Use gcovr merge options to avoid double-counting template instantiations
#   across multiple translation units.
# --clean means clean the builddir before running.
# --cuda enables CUDA support for coverage testing.
# --no-cuda disables CUDA support for coverage testing.
# Default behavior: auto-detect CUDA availability and enable if possible.


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="builddir"
ENABLE_CUDA="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=true
      shift
      ;;
    --cuda)
      ENABLE_CUDA="on"
      shift
      ;;
    --no-cuda)
      ENABLE_CUDA="off"
      shift
      ;;
    *)
      BUILD_DIR="$1"
      shift
      ;;
  esac
done

cd "${ROOT_DIR}"

# Auto-detect CUDA availability if not specified
if [ "$ENABLE_CUDA" = "auto" ]; then
  if command -v nvcc &>/dev/null && [ -d "/opt/cuda" ]; then
    ENABLE_CUDA="on"
    echo "CUDA detected, enabling CUDA support for coverage testing."
  else
    ENABLE_CUDA="off"
    echo "CUDA not detected, disabling CUDA support for coverage testing."
  fi
fi

if [ "${CLEAN:-false}" = true ] && [ -d "${BUILD_DIR}" ]; then
  rm -rf "${BUILD_DIR}"
fi

# Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"

# Configure with appropriate CUDA settings
cmake_args=(
  -S .
  -B "${BUILD_DIR}"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Debug
  -DHAHAHA_DISPLAY=OFF
  -DHAHAHA_BUILD_TESTS=ON
  -DHAHAHA_BUILD_EXAMPLES=OFF
  -DHAHAHA_ENABLE_COVERAGE=ON
)

if [ "$ENABLE_CUDA" = "on" ]; then
  cmake_args+=(-DHAHAHA_USE_CUDA=ON)
  echo "Building with CUDA support enabled."
else
  cmake_args+=(-DHAHAHA_USE_CUDA=OFF)
  echo "Building without CUDA support."
fi

cmake "${cmake_args[@]}"

cmake --build "${BUILD_DIR}" --parallel 8

# Run tests based on CUDA availability
if [ "$ENABLE_CUDA" = "on" ]; then
  echo "Running tests with CUDA enabled..."
  # Run all tests including CUDA tests
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
else
  echo "Running tests without CUDA..."
  # Run tests excluding CUDA-specific tests (if any filtering is needed)
  ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

GCOVR_COMMON_ARGS=(
  -r .
  --merge-mode-functions=merge-use-line-min
  --exclude-unreachable-branches
  --exclude-noncode-lines
  --filter 'core/.'
  --exclude 'subprojects/.'
  --exclude 'examples/.'
  --exclude 'core/src/display/.'
  --exclude 'core/include/display/.'
#  -j $(nproc)
  # --html-details
  # -o cover/coverage.html
)

echo "== line coverage (core, exclude display) =="
gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric line --fail-under-line 80

echo
echo "== branch coverage (core, exclude display) =="
# For CUDA builds, we expect higher branch coverage due to additional CUDA code paths
# if [ "$ENABLE_CUDA" = "on" ]; then
  # With CUDA enabled, aim for higher branch coverage since CUDA code adds more branches
  # gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric branch --fail-under-branch 50
# else
#   # Without CUDA, use the original threshold
  gcovr "${GCOVR_COMMON_ARGS[@]}" --txt-metric branch --fail-under-branch 35
# fi
