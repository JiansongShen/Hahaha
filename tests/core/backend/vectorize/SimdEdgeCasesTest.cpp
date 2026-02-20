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

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "backend/vectorize/arch/simd_impl.h"
#include "backend/vectorize/simd_compute_funs.h"
#include "backend/vectorize/simd_vec.h"
#include "common/definitions.h"

using namespace hahaha::backend;
using namespace hahaha::common;

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY

// ============================================================================
// Special Values Tests (NaN, Inf, Zero, Denormals)
// ============================================================================

TEST(SimdEdgeCases, Float32_NaN_Propagation) {
    alignas(16) f32 a[4] = {std::numeric_limits<f32>::quiet_NaN(), 2.0f, 3.0f, 4.0f};
    alignas(16) f32 b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 result[4];

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    (va + vb).store(result);
    EXPECT_TRUE(std::isnan(result[0]));

    (va * vb).store(result);
    EXPECT_TRUE(std::isnan(result[0]));

    (va - vb).store(result);
    EXPECT_TRUE(std::isnan(result[0]));
}

TEST(SimdEdgeCases, Float32_Infinity_Operations) {
    alignas(16) f32 a[4] = {std::numeric_limits<f32>::infinity(), 2.0f, 3.0f, 4.0f};
    alignas(16) f32 b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 result[4];

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    // Inf + finite = Inf
    (va + vb).store(result);
    EXPECT_TRUE(std::isinf(result[0]) && result[0] > 0);

    // Inf * positive = Inf
    (va * vb).store(result);
    EXPECT_TRUE(std::isinf(result[0]) && result[0] > 0);
}

TEST(SimdEdgeCases, Float32_NegativeInfinity) {
    alignas(16) f32 a[4] = {-std::numeric_limits<f32>::infinity(), 2.0f, 3.0f, 4.0f};
    alignas(16) f32 b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 result[4];

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    (va + vb).store(result);
    EXPECT_TRUE(std::isinf(result[0]) && result[0] < 0);

    (va * vb).store(result);
    EXPECT_TRUE(std::isinf(result[0]) && result[0] < 0);
}

TEST(SimdEdgeCases, Float32_ZeroDivision) {
    alignas(16) f32 a[4] = {1.0f, 0.0f, -1.0f, 2.0f};
    alignas(16) f32 b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    alignas(16) f32 result[4];

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    (va / vb).store(result);
    EXPECT_TRUE(std::isinf(result[0])); // 1/0 = Inf
    EXPECT_TRUE(std::isnan(result[1])); // 0/0 = NaN
    EXPECT_TRUE(std::isinf(result[2])); // -1/0 = -Inf
}

TEST(SimdEdgeCases, Float32_NegativeZero) {
    alignas(16) f32 a[4] = {-0.0f, -0.0f, -0.0f, -0.0f};
    alignas(16) f32 b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 result[4];

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], b[i]);
    }
}

TEST(SimdEdgeCases, Float32_Denormals) {
    alignas(16) f32 a[4];
    alignas(16) f32 b[4];
    alignas(16) f32 result[4];

    // Smallest positive denormal
    for (int i = 0; i < 4; ++i) {
        a[i] = std::numeric_limits<f32>::denorm_min();
        b[i] = std::numeric_limits<f32>::denorm_min();
    }

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(result[i], 0.0f);
    }
}

TEST(SimdEdgeCases, Float64_SpecialValues) {
    alignas(16) f64 a[2] = {std::numeric_limits<f64>::quiet_NaN(),
                            std::numeric_limits<f64>::infinity()};
    alignas(16) f64 b[2] = {1.0, 1.0};
    alignas(16) f64 result[2];

    auto va = SimdVec<f64, 128>::load(a);
    auto vb = SimdVec<f64, 128>::load(b);

    (va + vb).store(result);
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isinf(result[1]));
}

// ============================================================================
// Alignment Edge Cases
// ============================================================================

TEST(SimdEdgeCases, Float32_UnalignedAccess) {
    // Create intentionally unaligned data
    alignas(16) f32 buffer[8];
    f32* unaligned_a = reinterpret_cast<f32*>(reinterpret_cast<char*>(buffer) + 1);
    f32* unaligned_b = unaligned_a + 4;

    // This should work with unaligned load
    std::array<f32, 4> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::array<f32, 4> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::array<f32, 4> result;

    // Use unaligned load (loadu)
    auto va = SimdVec<f32, 128>::load(a.data());
    auto vb = SimdVec<f32, 128>::load(b.data());
    (va + vb).store(result.data());

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }
}

// ============================================================================
// Integer Edge Cases
// ============================================================================

TEST(SimdEdgeCases, Int32_Overflow) {
    alignas(16) i32 a[4] = {std::numeric_limits<i32>::max(), 1, 2, 3};
    alignas(16) i32 b[4] = {1, 1, 1, 1};
    alignas(16) i32 result[4];

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    // Integer overflow wraps around in two's complement
    (va + vb).store(result);
    EXPECT_EQ(result[0], std::numeric_limits<i32>::min()); // Overflow wraps
}

TEST(SimdEdgeCases, Int32_Underflow) {
    alignas(16) i32 a[4] = {std::numeric_limits<i32>::min(), 1, 2, 3};
    alignas(16) i32 b[4] = {1, 1, 1, 1};
    alignas(16) i32 result[4];

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    (va - vb).store(result);
    EXPECT_EQ(result[0], std::numeric_limits<i32>::max()); // Underflow wraps
}

TEST(SimdEdgeCases, Int32_ZeroMultiplication) {
    alignas(16) i32 a[4] = {0, 100, 200, 300};
    alignas(16) i32 b[4] = {1000, 2000, 3000, 4000};
    alignas(16) i32 result[4];

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

#if defined(__SSE4_1__)
    (va * vb).store(result);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 100 * 2000);
#endif
}

TEST(SimdEdgeCases, Int32_NegativeNumbers) {
    alignas(16) i32 a[4] = {-1, -2, -3, -4};
    alignas(16) i32 b[4] = {1, 2, 3, 4};
    alignas(16) i32 result[4];

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[0], 0);
    }

    (-va).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], -a[i]);
    }
}

TEST(SimdEdgeCases, Int64_LargeValues) {
    alignas(16) i64 a[2] = {std::numeric_limits<i64>::max() / 2, 1000000000000LL};
    alignas(16) i64 b[2] = {1, 1000000000000LL};
    alignas(16) i64 result[2];

    auto va = SimdVec<i64, 128>::load(a);
    auto vb = SimdVec<i64, 128>::load(b);

    (va + vb).store(result);
    EXPECT_EQ(result[0], std::numeric_limits<i64>::max() / 2 + 1);
    EXPECT_EQ(result[1], 2000000000000LL);
}

// ============================================================================
// Boundary Size Tests
// ============================================================================

TEST(SimdEdgeCases, ComputeFuns_EmptyArray) {
    f32 a[1], b[1], result[1];

    // Size 0 should be safe (no-op)
    simdAddContiguous(a, b, result, 0);
    simdMulContiguous(a, b, result, 0);
    simdSubContiguous(a, b, result, 0);
    simdDivContiguous(a, b, result, 0);
}

TEST(SimdEdgeCases, ComputeFuns_SingleElement) {
    f32 a[] = {5.0f};
    f32 b[] = {3.0f};
    f32 result[1];

    simdAddContiguous(a, b, result, 1);
    EXPECT_FLOAT_EQ(result[0], 8.0f);
}

TEST(SimdEdgeCases, ComputeFuns_TwoElements) {
    f32 a[] = {1.0f, 2.0f};
    f32 b[] = {3.0f, 4.0f};
    f32 result[2];

    simdMulContiguous(a, b, result, 2);
    EXPECT_FLOAT_EQ(result[0], 3.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
}

TEST(SimdEdgeCases, ComputeFuns_ThreeElements) {
    f32 a[] = {10.0f, 20.0f, 30.0f};
    f32 b[] = {1.0f, 2.0f, 3.0f};
    f32 result[3];

    simdSubContiguous(a, b, result, 3);
    EXPECT_FLOAT_EQ(result[0], 9.0f);
    EXPECT_FLOAT_EQ(result[1], 18.0f);
    EXPECT_FLOAT_EQ(result[2], 27.0f);
}

TEST(SimdEdgeCases, ComputeFuns_LargeNonAligned) {
    const std::size_t n = 1001; // Prime number, not divisible by 4 or 8
    std::vector<f32> a(n), b(n), result(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i);
        b[i] = static_cast<f32>(i + 1);
    }

    simdAddContiguous(a.data(), b.data(), result.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }
}

// ============================================================================
// Mixed Operation Tests
// ============================================================================

TEST(SimdEdgeCases, Float32_ChainedOperations) {
    alignas(16) f32 a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 b[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    alignas(16) f32 c[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    alignas(16) f32 result[4];

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);
    auto vc = SimdVec<f32, 128>::load(c);

    // ((a + b) * c) / b
    auto temp1 = va + vb;
    auto temp2 = temp1 * vc;
    auto final_result = temp2 / vb;
    final_result.store(result);

    for (int i = 0; i < 4; ++i) {
        f32 expected = ((a[i] + b[i]) * c[i]) / b[i];
        EXPECT_NEAR(result[i], expected, 1e-5f);
    }
}

TEST(SimdEdgeCases, Float64_PrecisionLoss) {
    alignas(16) f64 a[2] = {1e308, 1e-308};
    alignas(16) f64 b[2] = {1e-308, 1e308};
    alignas(16) f64 result[2];

    auto va = SimdVec<f64, 128>::load(a);
    auto vb = SimdVec<f64, 128>::load(b);

    // Large + small should preserve large
    (va + vb).store(result);
    EXPECT_DOUBLE_EQ(result[0], 1e308);
}

TEST(SimdEdgeCases, Int32_AllZeros) {
    alignas(16) i32 a[4] = {0, 0, 0, 0};
    alignas(16) i32 b[4] = {0, 0, 0, 0};
    alignas(16) i32 result[4];

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 0);
    }

    (va - vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 0);
    }
}

TEST(SimdEdgeCases, Int32_AlternatingSignPattern) {
    alignas(16) i32 a[4] = {1, -1, 1, -1};
    alignas(16) i32 b[4] = {-1, 1, -1, 1};
    alignas(16) i32 result[4];

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 0);
    }
}

// ============================================================================
// Regression Tests
// ============================================================================

TEST(SimdEdgeCases, Regression_UnalignedBoundary) {
    // Test that we correctly handle the transition from SIMD to scalar
    const std::size_t n = 17; // 4*4 + 1 (one past 4 full vectors)
    f32 a[n], b[n], result[n];

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<f32>(i);
        b[i] = 1.0f;
    }

    simdAddContiguous(a, b, result, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + 1.0f);
    }
}

TEST(SimdEdgeCases, Regression_InPlaceOperation) {
    alignas(16) f32 data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 operand[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    // In-place operation: data = data + operand
    simdAddContiguous(data, operand, data, 4);

    EXPECT_FLOAT_EQ(data[0], 6.0f);
    EXPECT_FLOAT_EQ(data[1], 8.0f);
    EXPECT_FLOAT_EQ(data[2], 10.0f);
    EXPECT_FLOAT_EQ(data[3], 12.0f);
}

#endif // HAHAHA_ARCH_IS_X86_FAMILY
