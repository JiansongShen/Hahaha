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
//  x86 SIMD 128-bit (SSE2 baseline, optional SSE4.1 for i32 mul).
//

#ifndef HAHAHA_SIMD_VEC_IMPL_128_H_X86
#define HAHAHA_SIMD_VEC_IMPL_128_H_X86

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>

#include "backend/vectorize/simd_vec.h"
#include "common/definitions.h"
#include "utils/common/helper_structs.h"

namespace hahaha::backend {

using common::f32;
using common::f64;
using common::i32;
using common::i64;


// ---- 128-bit (SSE2 baseline) ----

template <>
struct SimdVec<f32, 128> {
    using scalar_type = f32;
    static constexpr std::size_t kBits = 128;
    static constexpr std::size_t kLanes = 4;

    __m128 v_{};

    SimdVec() = default;
    explicit SimdVec(__m128 v) : v_(v) {}

    static SimdVec load(f32 const* ptr) {
        return SimdVec(_mm_loadu_ps(ptr));
    }
    static SimdVec load_aligned(f32 const* ptr) {
        return SimdVec(_mm_load_ps(ptr));
    }
    void store(f32* ptr) const {
        _mm_storeu_ps(ptr, v_);
    }
    void store_aligned(f32* ptr) const {
        _mm_store_ps(ptr, v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm_sub_ps(_mm_setzero_ps(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm_add_ps(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm_sub_ps(v_, other.v_));
    }
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm_mul_ps(v_, other.v_));
    }
    SimdVec operator/(SimdVec const& other) const {
        return SimdVec(_mm_div_ps(v_, other.v_));
    }
    SimdVec& operator+=(SimdVec const& other) { v_ = _mm_add_ps(v_, other.v_); return *this; }
    SimdVec& operator-=(SimdVec const& other) { v_ = _mm_sub_ps(v_, other.v_); return *this; }
    SimdVec& operator*=(SimdVec const& other) { v_ = _mm_mul_ps(v_, other.v_); return *this; }
    SimdVec& operator/=(SimdVec const& other) { v_ = _mm_div_ps(v_, other.v_); return *this; }
};

template <>
struct SimdVec<f64, 128> {
    using scalar_type = f64;
    static constexpr std::size_t kBits = 128;
    static constexpr std::size_t kLanes = 2;

    __m128d v_{};

    SimdVec() = default;
    explicit SimdVec(__m128d v) : v_(v) {}

    static SimdVec load(f64 const* ptr) {
        return SimdVec(_mm_loadu_pd(ptr));
    }
    static SimdVec load_aligned(f64 const* ptr) {
        return SimdVec(_mm_load_pd(ptr));
    }
    void store(f64* ptr) const {
        _mm_storeu_pd(ptr, v_);
    }
    void store_aligned(f64* ptr) const {
        _mm_store_pd(ptr, v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm_sub_pd(_mm_setzero_pd(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm_add_pd(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm_sub_pd(v_, other.v_));
    }
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm_mul_pd(v_, other.v_));
    }
    SimdVec operator/(SimdVec const& other) const {
        return SimdVec(_mm_div_pd(v_, other.v_));
    }
    SimdVec& operator+=(SimdVec const& other) { v_ = _mm_add_pd(v_, other.v_); return *this; }
    SimdVec& operator-=(SimdVec const& other) { v_ = _mm_sub_pd(v_, other.v_); return *this; }
    SimdVec& operator*=(SimdVec const& other) { v_ = _mm_mul_pd(v_, other.v_); return *this; }
    SimdVec& operator/=(SimdVec const& other) { v_ = _mm_div_pd(v_, other.v_); return *this; }
};

template <>
struct SimdVec<i32, 128> {
    using scalar_type = i32;
    static constexpr std::size_t kBits = 128;
    static constexpr std::size_t kLanes = 4;

    __m128i v_{};

    SimdVec() = default;
    explicit SimdVec(__m128i v) : v_(v) {}

    static SimdVec load(i32 const* ptr) {
        return SimdVec(_mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)));
    }
    static SimdVec load_aligned(i32 const* ptr) {
        return SimdVec(_mm_load_si128(reinterpret_cast<__m128i const*>(ptr)));
    }
    void store(i32* ptr) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v_);
    }
    void store_aligned(i32* ptr) const {
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm_sub_epi32(_mm_setzero_si128(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm_add_epi32(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm_sub_epi32(v_, other.v_));
    }
#if defined(__SSE4_1__)
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm_mullo_epi32(v_, other.v_));
    }
    SimdVec& operator*=(SimdVec const& other) { v_ = _mm_mullo_epi32(v_, other.v_); return *this; }
#endif
    SimdVec& operator+=(SimdVec const& other) { v_ = _mm_add_epi32(v_, other.v_); return *this; }
    SimdVec& operator-=(SimdVec const& other) { v_ = _mm_sub_epi32(v_, other.v_); return *this; }
};

template <>
struct SimdVec<i64, 128> {
    using scalar_type = i64;
    static constexpr std::size_t kBits = 128;
    static constexpr std::size_t kLanes = 2;

    __m128i v_{};

    SimdVec() = default;
    explicit SimdVec(__m128i v) : v_(v) {}

    static SimdVec load(i64 const* ptr) {
        return SimdVec(_mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)));
    }
    static SimdVec load_aligned(i64 const* ptr) {
        return SimdVec(_mm_load_si128(reinterpret_cast<__m128i const*>(ptr)));
    }
    void store(i64* ptr) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v_);
    }
    void store_aligned(i64* ptr) const {
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm_sub_epi64(_mm_setzero_si128(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm_add_epi64(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm_sub_epi64(v_, other.v_));
    }
    SimdVec& operator+=(SimdVec const& other) { v_ = _mm_add_epi64(v_, other.v_); return *this; }
    SimdVec& operator-=(SimdVec const& other) { v_ = _mm_sub_epi64(v_, other.v_); return *this; }
};

} // namespace hahaha::backend

#endif // HAHAHA_ARCH_IS_X86_FAMILY

#endif // HAHAHA_SIMD_VEC_IMPL_128_H_X86
