# Backend Macros and Definitions

This document records macros and definitions used in the `backend/` directory that are not explained in detail elsewhere.

---

## Build-time Feature Flags

### `HAHAHA_USE_CUDA`

**Definition**: Defined in CMake when CUDA support is enabled (`-DHAHAHA_USE_CUDA`).

**Usage**: Used throughout CUDA-related headers to conditionally compile CUDA code or provide stub implementations.

**Example**:
```cpp
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
// CUDA implementation
#else
// Stub when headers not available
#endif
#else
// Stub when CUDA not enabled
#endif
```

**Location**: Defined in `CMakeLists.txt` when `HAHAHA_USE_CUDA` option is ON.

---

## Architecture Detection Macros

These macros are **preferred to be defined by CMake** (see `CMakeLists.txt`), but `common/macros.h` provides fallback definitions if CMake detection fails. All macros use `#ifndef` guards, so CMake definitions take precedence.

### Architecture Type Macros

| Macro          | Value |  Description                      |
|----------------|-------|-----------------------------------|
| `HAHAHA_ARCH_X86_64`  | `1`   | x86-64 (AMD64) architecture       |
| `HAHAHA_ARCH_X86_32`  | `1`   | x86 (32-bit) architecture         |
| `HAHAHA_ARCH_ARM64`   | `1`   | ARM 64-bit (AArch64) architecture |
| `HAHAHA_ARCH_ARM32`   | `1`   | ARM 32-bit architecture           |
| `HAHAHA_ARCH_PPC64`   | `1`   | PowerPC 64-bit architecture       |
| `HAHAHA_ARCH_RISCV64` | `1`   | RISC-V 64-bit architecture        |
| `HAHAHA_ARCH_UNKNOWN` | `1`   | Unknown/unsupported architecture  |

### Architecture Family Macros

| Macro                | Value   | Description                    |
|----------------------|---------|--------------------------------|
| `HAHAHA_ARCH_IS_X86_FAMILY` | `1`     | Set for both x86-64 and x86-32 |

### Architecture Properties

| Macro        | Value          | Description                                                   |
|--------------|----------------|---------------------------------------------------------------|
| `HAHAHA_ARCH_BITS`  | `32` or `64`   | Architecture bitness                                          |
| `HAHAHA_ARCH_NAME`  | String literal | Human-readable architecture name (e.g. `"x86-64"`, `"arm64"`) |

**Definition Priority**:
1. **CMake** (via `target_compile_definitions` in `CMakeLists.txt`) - preferred
2. **macros.h** (fallback via preprocessor checks) - if CMake didn't define them

**Example Usage**:
```cpp
#if defined(HAHAHA_ARCH_X86_64) || defined(HAHAHA_ARCH_X86_32)
    // x86-specific code
    #if HAHAHA_ARCH_BITS == 64
        // 64-bit x86 optimizations
    #endif
#elif defined(HAHAHA_ARCH_ARM64)
    // ARM64-specific code
#endif
```

**CMake Detection**: CMake detects architecture via `CMAKE_SYSTEM_PROCESSOR` and `CMAKE_SIZEOF_VOID_P`, then defines these macros before compilation. See `CMakeLists.txt` for details.

---
