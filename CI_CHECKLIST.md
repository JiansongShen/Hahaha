# CI Checklist - CMake + vcpkg Build

## ✅ All Platforms Verified

### Build Matrix
- ✅ Linux x64 (GCC) - dynamic linking
- ✅ Linux x64 (Clang) - dynamic linking
- ✅ macOS ARM64 (Homebrew LLVM) - dynamic linking
- ✅ Windows x64 (MSVC) - **static linking** (avoids DLL PATH issues)

### Key Changes for CI Success

1. **vcpkg Setup**: Workflow now clones vcpkg if not present
2. **GTest Installation**: Installed via vcpkg before CMake configure
3. **Triplet Auto-detection**: CMake automatically selects correct triplet
4. **Headless Build**: All CI uses `-DHAHAHA_DISPLAY=OFF`

## Local CI Replication

```bash
# Clean start
rm -rf vcpkg/vcpkg_root builddir

# Setup vcpkg
git clone https://github.com/microsoft/vcpkg.git vcpkg/vcpkg_root
./vcpkg/vcpkg_root/bootstrap-vcpkg.sh  # or .bat on Windows

# Install gtest
# Linux/macOS:
./vcpkg/vcpkg_root/vcpkg install gtest
# Windows (static):
# .\vcpkg\vcpkg_root\vcpkg.exe install gtest --triplet x64-windows-static

# Build and test (headless, like CI)
cmake -S . -B builddir -G Ninja -DCMAKE_BUILD_TYPE=Debug -DHAHAHA_DISPLAY=OFF -DHAHAHA_BUILD_TESTS=ON
cmake --build builddir --parallel 2
ctest --test-dir builddir --output-on-failure
```
