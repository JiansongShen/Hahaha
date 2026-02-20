//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//
//  x86 SIMD capability detection (compile-time and optional runtime via CPUID).
//

#ifndef HAHAHA_SIMD_CAPABILITIES_H_X86
#define HAHAHA_SIMD_CAPABILITIES_H_X86

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY

#if defined(__GNUC__)
#include <cpuid.h>
#endif

#if defined(__clang__)
// #include <cpuid.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hahaha::backend {

/**
 * @brief Compile-time SIMD capability on x86.
 * True when the corresponding compiler flag is used (e.g. -mavx, /arch:AVX).
 */
struct SimdCapabilitiesCompileTime {
    static constexpr bool kSSE2 =
#if defined(__SSE2__)                                                               \
    || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
        true;
#else
        false;
#endif
    static constexpr bool kSSE41 =
#if defined(__SSE4_1__)
        true;
#else
        false;
#endif
    static constexpr bool kAVX =
#if defined(__AVX__)
        true;
#else
        false;
#endif
    static constexpr bool kAVX2 =
#if defined(__AVX2__)
        true;
#else
        false;
#endif
    static constexpr bool kAVX512F =
#if defined(__AVX512F__)
        true;
#else
        false;
#endif
};

/**
 * @brief Runtime x86 SIMD capability (CPUID). Use when building for baseline
 * but wanting to dispatch to AVX/AVX2/AVX-512 on supported CPUs.
 */
struct SimdCapabilitiesRuntime {
    bool sse2{false};
    bool sse41{false};
    bool avx{false};
    bool avx2{false};
    bool avx512f{false};

    static SimdCapabilitiesRuntime detect() {
        SimdCapabilitiesRuntime r;
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 1);
        r.sse2 = (info[3] & (1 << 26)) != 0;
        r.sse41 = (info[2] & (1 << 19)) != 0;
        r.avx = (info[2] & (1 << 28)) != 0;
        __cpuidex(info, 7, 0);
        r.avx2 = (info[1] & (1 << 5)) != 0;
        r.avx512f = (info[1] & (1 << 16)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (__get_cpuid_count(1, 0, &a, &b, &c, &d)) {
            r.sse2 = (d & (1u << 26)) != 0;
            r.sse41 = (c & (1u << 19)) != 0;
            r.avx = (c & (1u << 28)) != 0;
        }
        if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
            r.avx2 = (b & (1u << 5)) != 0;
            r.avx512f = (b & (1u << 16)) != 0;
        }
#endif
        return r;
    }
};

} // namespace hahaha::backend

#endif // HAHAHA_ARCH_IS_X86_FAMILY

#endif // HAHAHA_SIMD_CAPABILITIES_H_X86
