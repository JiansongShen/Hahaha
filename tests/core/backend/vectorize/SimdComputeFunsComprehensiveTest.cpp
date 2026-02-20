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

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "backend/vectorize/simd_compute_funs.h"
#include "common/definitions.h"

using namespace hahaha::backend;
using namespace hahaha::common;

// Helper function to compare floating point arrays with tolerance
template <typename T>
void expectArrayNear(const T* expected,
                     const T* actual,
                     std::size_t n,
                     T tolerance = static_cast<T>(1e-5)) {
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(expected[i], actual[i], tolerance)
            << "Arrays differ at index " << i;
    }
}

// ============================================================================
// Float32 Tests
// ============================================================================

TEST(SimdComputeFuns, Float32_Add_Small) {
    const std::size_t n = 10;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i + 1);
        b[i] = static_cast<f32>(i + 10);
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

TEST(SimdComputeFuns, Float32_Add_Large) {
    const std::size_t n = 1000;
    std::vector<f32> a(n), b(n), result(n), expected(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<f32> dis(-100.0f, 100.0f);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a.data(), b.data(), result.data(), n);
    expectArrayNear(expected.data(), result.data(), n);
}

TEST(SimdComputeFuns, Float32_Add_NotMultipleOfVectorSize) {
    // Test with size that's not a multiple of 4 (SSE lane count)
    const std::size_t n = 13;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i * 2);
        b[i] = static_cast<f32>(i * 3);
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

TEST(SimdComputeFuns, Float32_Subtract_Basic) {
    const std::size_t n = 16;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(100 + i);
        b[i] = static_cast<f32>(i);
        expected[i] = a[i] - b[i];
    }

    simdSubContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

TEST(SimdComputeFuns, Float32_Multiply_Basic) {
    const std::size_t n = 20;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i + 1);
        b[i] = static_cast<f32>(2);
        expected[i] = a[i] * b[i];
    }

    simdMulContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

TEST(SimdComputeFuns, Float32_Divide_Basic) {
    const std::size_t n = 24;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>((i + 1) * 10);
        b[i] = static_cast<f32>(i + 1);
        expected[i] = a[i] / b[i];
    }

    simdDivContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

TEST(SimdComputeFuns, Float32_ZeroArray) {
    const std::size_t n = 16;
    f32 a[n], b[n], result[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 0.0f;
        b[i] = static_cast<f32>(i + 1);
    }

    simdAddContiguous(a, b, result, n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], b[i]);
    }

    simdMulContiguous(a, b, result, n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

TEST(SimdComputeFuns, Float32_NegativeNumbers) {
    const std::size_t n = 16;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = -static_cast<f32>(i + 1);
        b[i] = static_cast<f32>(i + 1);
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

TEST(SimdComputeFuns, Float32_MixedSigns) {
    const std::size_t n = 32;
    f32 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = (i % 2 == 0) ? static_cast<f32>(i) : -static_cast<f32>(i);
        b[i] = (i % 3 == 0) ? static_cast<f32>(i) : -static_cast<f32>(i);
        expected[i] = a[i] * b[i];
    }

    simdMulContiguous(a, b, result, n);
    expectArrayNear(expected, result, n);
}

// ============================================================================
// Float64 Tests
// ============================================================================

TEST(SimdComputeFuns, Float64_Add_Basic) {
    const std::size_t n = 10;
    f64 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f64>(i + 1) * 1.5;
        b[i] = static_cast<f64>(i + 10) * 2.5;
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a, b, result, n);
    expectArrayNear(expected, result, n, 1e-10);
}

TEST(SimdComputeFuns, Float64_Subtract_Large) {
    const std::size_t n = 500;
    std::vector<f64> a(n), b(n), result(n), expected(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<f64> dis(-1000.0, 1000.0);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
        expected[i] = a[i] - b[i];
    }

    simdSubContiguous(a.data(), b.data(), result.data(), n);
    expectArrayNear(expected.data(), result.data(), n, 1e-10);
}

TEST(SimdComputeFuns, Float64_Multiply_HighPrecision) {
    const std::size_t n = 16;
    f64 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1.0 / static_cast<f64>(i + 1);
        b[i] = static_cast<f64>(i + 1);
        expected[i] = a[i] * b[i];
    }

    simdMulContiguous(a, b, result, n);
    expectArrayNear(expected, result, n, 1e-10);
}

TEST(SimdComputeFuns, Float64_Divide_HighPrecision) {
    const std::size_t n = 20;
    f64 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f64>(i + 1);
        b[i] = static_cast<f64>(i + 2);
        expected[i] = a[i] / b[i];
    }

    simdDivContiguous(a, b, result, n);
    expectArrayNear(expected, result, n, 1e-10);
}

TEST(SimdComputeFuns, Float64_NotMultipleOfVectorSize) {
    // Test with size that's not a multiple of 2 (SSE lane count for f64)
    const std::size_t n = 11;
    f64 a[n], b[n], result[n], expected[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f64>(i * 3.7);
        b[i] = static_cast<f64>(i * 2.3);
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a, b, result, n);
    expectArrayNear(expected, result, n, 1e-10);
}

// ============================================================================
// Performance-oriented tests (larger arrays)
// ============================================================================

TEST(SimdComputeFuns, Float32_Add_Performance_1K) {
    const std::size_t n = 1024;
    std::vector<f32> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i);
        b[i] = static_cast<f32>(n - i);
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a.data(), b.data(), result.data(), n);
    expectArrayNear(expected.data(), result.data(), n);
}

TEST(SimdComputeFuns, Float32_Multiply_Performance_4K) {
    const std::size_t n = 4096;
    std::vector<f32> a(n), b(n), result(n), expected(n);

    std::mt19937 gen(123);
    std::uniform_real_distribution<f32> dis(-10.0f, 10.0f);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
        expected[i] = a[i] * b[i];
    }

    simdMulContiguous(a.data(), b.data(), result.data(), n);
    expectArrayNear(expected.data(), result.data(), n, 1e-4f);
}

TEST(SimdComputeFuns, Float64_Add_Performance_1K) {
    const std::size_t n = 1024;
    std::vector<f64> a(n), b(n), result(n), expected(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f64>(i) * 0.5;
        b[i] = static_cast<f64>(i) * 1.5;
        expected[i] = a[i] + b[i];
    }

    simdAddContiguous(a.data(), b.data(), result.data(), n);
    expectArrayNear(expected.data(), result.data(), n, 1e-10);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(SimdComputeFuns, Float32_SingleElement) {
    f32 a[] = {42.0f};
    f32 b[] = {8.0f};
    f32 result[1];

    simdAddContiguous(a, b, result, 1);
    EXPECT_FLOAT_EQ(result[0], 50.0f);

    simdMulContiguous(a, b, result, 1);
    EXPECT_FLOAT_EQ(result[0], 336.0f);
}

TEST(SimdComputeFuns, Float64_SingleElement) {
    f64 a[] = {100.5};
    f64 b[] = {10.5};
    f64 result[1];

    simdSubContiguous(a, b, result, 1);
    EXPECT_DOUBLE_EQ(result[0], 90.0);

    simdDivContiguous(a, b, result, 1);
    EXPECT_NEAR(result[0], 100.5 / 10.5, 1e-10);
}

TEST(SimdComputeFuns, Float32_VectorSizeExact) {
    // Exactly 4 elements (one SSE vector)
    f32 a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    f32 b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    f32 result[4];

    simdAddContiguous(a, b, result, 4);
    EXPECT_FLOAT_EQ(result[0], 6.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 10.0f);
    EXPECT_FLOAT_EQ(result[3], 12.0f);
}

TEST(SimdComputeFuns, Float32_MultipleVectors) {
    // Exactly 8 elements (two SSE vectors)
    f32 a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    f32 b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    f32 result[8];

    simdMulContiguous(a, b, result, 8);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] * b[i]);
    }
}

TEST(SimdComputeFuns, Float32_VerySmallNumbers) {
    const std::size_t n = 16;
    f32 a[n], b[n], result[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1e-30f;
        b[i] = 1e-30f;
    }

    simdAddContiguous(a, b, result, n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(result[i], 2e-30f, 1e-35f);
    }
}

TEST(SimdComputeFuns, Float32_VeryLargeNumbers) {
    const std::size_t n = 16;
    f32 a[n], b[n], result[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1e30f;
        b[i] = 1e30f;
    }

    simdMulContiguous(a, b, result, n);
    for (std::size_t i = 0; i < n; ++i) {
        // 1e30 * 1e30 = 1e60, which exceeds float range and becomes infinity
        EXPECT_TRUE(std::isinf(result[i]));
    }
}

TEST(SimdComputeFuns, Float32_ConsecutiveOperations) {
    const std::size_t n = 32;
    f32 a[n], b[n], c[n], temp[n], result[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i + 1);
        b[i] = static_cast<f32>(i + 2);
        c[i] = static_cast<f32>(i + 3);
    }

    // (a + b) * c
    simdAddContiguous(a, b, temp, n);
    simdMulContiguous(temp, c, result, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], (a[i] + b[i]) * c[i]);
    }
}

TEST(SimdComputeFuns, Float64_ConsecutiveOperations) {
    const std::size_t n = 20;
    f64 a[n], b[n], c[n], temp[n], result[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f64>(i + 1) * 2.5;
        b[i] = static_cast<f64>(i + 2) * 1.5;
        c[i] = static_cast<f64>(i + 3);
    }

    // (a - b) / c
    simdSubContiguous(a, b, temp, n);
    simdDivContiguous(temp, c, result, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(result[i], (a[i] - b[i]) / c[i], 1e-10);
    }
}
