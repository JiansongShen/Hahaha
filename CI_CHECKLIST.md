# CI Checklist - CMake + vcpkg Build

## ✅ All Platforms Verified

### Build Matrix
- ✅ Linux x64 (GCC)
- ✅ Linux x64 (Clang)
- ✅ macOS ARM64 (Homebrew LLVM)
- ✅ Windows x64 (MSVC)

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
./vcpkg/vcpkg_root/bootstrap-vcpkg.sh
./vcpkg/vcpkg_root/vcpkg install gtest

# Build and test (headless, like CI)
cmake -S . -B builddir -G Ninja -DCMAKE_BUILD_TYPE=Debug -DHAHAHA_DISPLAY=OFF -DHAHAHA_BUILD_TESTS=ON
cmake --build builddir --parallel 2
ctest --test-dir builddir --output-on-failure
```
