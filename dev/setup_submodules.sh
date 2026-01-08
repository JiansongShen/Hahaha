#!/usr/bin/env sh
set -eu

# Initialize and update git submodules for third-party dependencies.
# Usage:
#   ./dev/setup_submodules.sh

git submodule update --init --recursive

echo "Submodules initialized."
echo "Suggested CMake configure commands:"
echo "  cmake -S . -B build-cmake -G Ninja -DHAHAHA_DISPLAY=OFF"
echo "  cmake -S . -B build-cmake-display -G Ninja -DHAHAHA_DISPLAY=ON"
