#!/usr/bin/env sh
set -eu

# Fetch third-party dependencies into ./subprojects using Meson WrapDB (.wrap files),
# WITHOUT using meson (CMake-friendly).
#
# Why:
# - Reuse the repo-tracked wrap versions (same as Meson builds / CI policy).
# - Keep CMake builds reproducible without hardcoding versioned directories.
#
# Usage:
#   ./dev/fetch_wrap_deps.sh
#
# Notes:
# - Requires: python3 and network access (downloads sources/patches).
# - This script does NOT run any CMake build; it only populates ./subprojects/.

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  exit 1
fi

mkdir -p subprojects

if [ ! -f "subprojects/gtest.wrap" ] || [ ! -f "subprojects/imgui.wrap" ]; then
  echo "ERROR: expected wrap files not found under ./subprojects (gtest.wrap, imgui.wrap)." >&2
  exit 1
fi

python3 dev/fetch_wrap_deps.py --all

echo "Done. Subprojects downloaded."
echo "Now you can build with CMake, e.g.:"
echo "  cmake -S . -B build-cmake -G Ninja -DHAHAHA_DISPLAY=OFF"
echo "  cmake --build build-cmake"
