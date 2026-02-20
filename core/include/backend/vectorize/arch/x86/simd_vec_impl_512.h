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
//  x86 SIMD 512-bit (AVX-512F).
//

#ifndef HAHAHA_SIMD_VEC_IMPL_512_H_X86
#define HAHAHA_SIMD_VEC_IMPL_512_H_X86

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY
#if defined(__AVX512F__)

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

template <> struct SimdVec<f32, 512> {
    using scalar_type = f32;
    static constexpr std::size_t kBits = 512;
    static constexpr std::size_t kLanes = 16;

    __m512 v_{};

    SimdVec() = default;
    explicit SimdVec(__m512 v) : v_(v) {
    }

    static SimdVec load(f32 const* ptr) {
        return SimdVec(_mm512_loadu_ps(ptr));
    }
    static SimdVec load_aligned(f32 const* ptr) {
        return SimdVec(_mm512_load_ps(ptr));
    }
    void store(f32* ptr) const {
        _mm512_storeu_ps(ptr, v_);
    }
    void store_aligned(f32* ptr) const {
        _mm512_store_ps(ptr, v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm512_sub_ps(_mm512_setzero_ps(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm512_add_ps(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm512_sub_ps(v_, other.v_));
    }
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm512_mul_ps(v_, other.v_));
    }
    SimdVec operator/(SimdVec const& other) const {
        return SimdVec(_mm512_div_ps(v_, other.v_));
    }
    SimdVec& operator+=(SimdVec const& other) {
        v_ = _mm512_add_ps(v_, other.v_);
        return *this;
    }
    SimdVec& operator-=(SimdVec const& other) {
        v_ = _mm512_sub_ps(v_, other.v_);
        return *this;
    }
    SimdVec& operator*=(SimdVec const& other) {
        v_ = _mm512_mul_ps(v_, other.v_);
        return *this;
    }
    SimdVec& operator/=(SimdVec const& other) {
        v_ = _mm512_div_ps(v_, other.v_);
        return *this;
    }
};

template <> struct SimdVec<f64, 512> {
    using scalar_type = f64;
    static constexpr std::size_t kBits = 512;
    static constexpr std::size_t kLanes = 8;

    __m512d v_{};

    SimdVec() = default;
    explicit SimdVec(__m512d v) : v_(v) {
    }

    static SimdVec load(f64 const* ptr) {
        return SimdVec(_mm512_loadu_pd(ptr));
    }
    static SimdVec load_aligned(f64 const* ptr) {
        return SimdVec(_mm512_load_pd(ptr));
    }
    void store(f64* ptr) const {
        _mm512_storeu_pd(ptr, v_);
    }
    void store_aligned(f64* ptr) const {
        _mm512_store_pd(ptr, v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm512_sub_pd(_mm512_setzero_pd(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm512_add_pd(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm512_sub_pd(v_, other.v_));
    }
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm512_mul_pd(v_, other.v_));
    }
    SimdVec operator/(SimdVec const& other) const {
        return SimdVec(_mm512_div_pd(v_, other.v_));
    }
    SimdVec& operator+=(SimdVec const& other) {
        v_ = _mm512_add_pd(v_, other.v_);
        return *this;
    }
    SimdVec& operator-=(SimdVec const& other) {
        v_ = _mm512_sub_pd(v_, other.v_);
        return *this;
    }
    SimdVec& operator*=(SimdVec const& other) {
        v_ = _mm512_mul_pd(v_, other.v_);
        return *this;
    }
    SimdVec& operator/=(SimdVec const& other) {
        v_ = _mm512_div_pd(v_, other.v_);
        return *this;
    }
};

// AVX-512F includes integer operations
template <> struct SimdVec<i32, 512> {
    using scalar_type = i32;
    static constexpr std::size_t kBits = 512;
    static constexpr std::size_t kLanes = 16;

    __m512i v_{};

    SimdVec() = default;
    explicit SimdVec(__m512i v) : v_(v) {
    }

    static SimdVec load(i32 const* ptr) {
        return SimdVec(_mm512_loadu_si512(reinterpret_cast<__m512i const*>(ptr)));
    }
    static SimdVec load_aligned(i32 const* ptr) {
        return SimdVec(_mm512_load_si512(reinterpret_cast<__m512i const*>(ptr)));
    }
    void store(i32* ptr) const {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v_);
    }
    void store_aligned(i32* ptr) const {
        _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm512_sub_epi32(_mm512_setzero_si512(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm512_add_epi32(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm512_sub_epi32(v_, other.v_));
    }
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm512_mullo_epi32(v_, other.v_));
    }
    SimdVec& operator+=(SimdVec const& other) {
        v_ = _mm512_add_epi32(v_, other.v_);
        return *this;
    }
    SimdVec& operator-=(SimdVec const& other) {
        v_ = _mm512_sub_epi32(v_, other.v_);
        return *this;
    }
    SimdVec& operator*=(SimdVec const& other) {
        v_ = _mm512_mullo_epi32(v_, other.v_);
        return *this;
    }
};

template <> struct SimdVec<i64, 512> {
    using scalar_type = i64;
    static constexpr std::size_t kBits = 512;
    static constexpr std::size_t kLanes = 8;

    __m512i v_{};

    SimdVec() = default;
    explicit SimdVec(__m512i v) : v_(v) {
    }

    static SimdVec load(i64 const* ptr) {
        return SimdVec(_mm512_loadu_si512(reinterpret_cast<__m512i const*>(ptr)));
    }
    static SimdVec load_aligned(i64 const* ptr) {
        return SimdVec(_mm512_load_si512(reinterpret_cast<__m512i const*>(ptr)));
    }
    void store(i64* ptr) const {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v_);
    }
    void store_aligned(i64* ptr) const {
        _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), v_);
    }

    SimdVec operator-() const {
        return SimdVec(_mm512_sub_epi64(_mm512_setzero_si512(), v_));
    }
    SimdVec operator+(SimdVec const& other) const {
        return SimdVec(_mm512_add_epi64(v_, other.v_));
    }
    SimdVec operator-(SimdVec const& other) const {
        return SimdVec(_mm512_sub_epi64(v_, other.v_));
    }
#if defined(__AVX512DQ__)
    // AVX-512DQ provides 64-bit integer multiplication
    SimdVec operator*(SimdVec const& other) const {
        return SimdVec(_mm512_mullo_epi64(v_, other.v_));
    }
    SimdVec& operator*=(SimdVec const& other) {
        v_ = _mm512_mullo_epi64(v_, other.v_);
        return *this;
    }
#endif
    SimdVec& operator+=(SimdVec const& other) {
        v_ = _mm512_add_epi64(v_, other.v_);
        return *this;
    }
    SimdVec& operator-=(SimdVec const& other) {
        v_ = _mm512_sub_epi64(v_, other.v_);
        return *this;
    }
};

} // namespace hahaha::backend

#endif // __AVX512F__
#endif // HAHAHA_ARCH_IS_X86_FAMILY

#endif // HAHAHA_SIMD_VEC_IMPL_512_H_X86
