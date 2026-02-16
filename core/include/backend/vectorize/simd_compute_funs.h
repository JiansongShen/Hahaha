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

#ifndef HAHAHA_SIMD_COMPUTE_FUNS_H_7F6243BB8DF1454D92CC725AE19AEDEC
#define HAHAHA_SIMD_COMPUTE_FUNS_H_7F6243BB8DF1454D92CC725AE19AEDEC

#include <cstddef>

#include "backend/vectorize/simd_vec.h"
#include "common/definitions.h"

#include "backend/vectorize/arch/simd_impl.h"

namespace hahaha::backend {
using common::f32;
using common::f64;

/**
 * @brief Contiguous elementwise add: out[i] = a[i] + b[i], vectorized when SimdVec is available.
 */
inline void simdAddContiguous(f32 const* a, f32 const* b, f32* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f32, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f32, 128>::load(a + i);
        auto vb = SimdVec<f32, 128>::load(b + i);
        (va + vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
#endif
}

/**
 * @brief Contiguous elementwise add for double.
 */
inline void simdAddContiguous(f64 const* a, f64 const* b, f64* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f64, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f64, 128>::load(a + i);
        auto vb = SimdVec<f64, 128>::load(b + i);
        (va + vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
#endif
}

/**
 * @brief Contiguous elementwise subtract: out[i] = a[i] - b[i].
 */
inline void simdSubContiguous(f32 const* a, f32 const* b, f32* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f32, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f32, 128>::load(a + i);
        auto vb = SimdVec<f32, 128>::load(b + i);
        (va - vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] - b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] - b[i];
#endif
}

/**
 * @brief Contiguous elementwise subtract for double.
 */
inline void simdSubContiguous(f64 const* a, f64 const* b, f64* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f64, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f64, 128>::load(a + i);
        auto vb = SimdVec<f64, 128>::load(b + i);
        (va - vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] - b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] - b[i];
#endif
}

/**
 * @brief Contiguous elementwise multiply: out[i] = a[i] * b[i].
 */
inline void simdMulContiguous(f32 const* a, f32 const* b, f32* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f32, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f32, 128>::load(a + i);
        auto vb = SimdVec<f32, 128>::load(b + i);
        (va * vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
#endif
}

/**
 * @brief Contiguous elementwise multiply for double.
 */
inline void simdMulContiguous(f64 const* a, f64 const* b, f64* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f64, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f64, 128>::load(a + i);
        auto vb = SimdVec<f64, 128>::load(b + i);
        (va * vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
#endif
}

/**
 * @brief Contiguous elementwise divide: out[i] = a[i] / b[i].
 */
inline void simdDivContiguous(f32 const* a, f32 const* b, f32* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f32, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f32, 128>::load(a + i);
        auto vb = SimdVec<f32, 128>::load(b + i);
        (va / vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] / b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] / b[i];
#endif
}

/**
 * @brief Contiguous elementwise divide for double.
 */
inline void simdDivContiguous(f64 const* a, f64 const* b, f64* out, std::size_t n) {
#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
    constexpr std::size_t lanes = SimdVec<f64, 128>::kLanes;
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = SimdVec<f64, 128>::load(a + i);
        auto vb = SimdVec<f64, 128>::load(b + i);
        (va / vb).store(out + i);
    }
    for (; i < n; ++i)
        out[i] = a[i] / b[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] / b[i];
#endif
}

} // namespace hahaha::backend

#endif // HAHAHA_SIMD_COMPUTE_FUNS_H_7F6243BB8DF1454D92CC725AE19AEDEC