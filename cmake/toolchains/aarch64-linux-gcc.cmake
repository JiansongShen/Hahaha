# Cross-compile for ARM64 (AArch64) Linux from x86_64 host.
# This file is intended to be used as VCPKG_CHAINLOAD_TOOLCHAIN_FILE with vcpkg.
# Requires: aarch64-linux-gnu-gcc/g++ (e.g. Ubuntu: apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu)
#
# Usage (chain-load with vcpkg):
#   export VCPKG_ROOT=/path/to/vcpkg
#   cmake -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
#         -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/cmake/toolchains/aarch64-linux-gcc.cmake \
#         -DVCPKG_TARGET_TRIPLET=arm64-linux ...
#   or use preset: cmake --preset cross-arm64-linux

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
