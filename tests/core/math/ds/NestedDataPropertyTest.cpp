// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "common/definitions.h"
#include "math/ds/NestedData.h"

using hahaha::common::f32;
using hahaha::common::f64;
using hahaha::common::i16;
using hahaha::common::i32;
using hahaha::common::i64;
using hahaha::common::i8;
using hahaha::common::u16;
using hahaha::common::u32;
using hahaha::common::u64;
using hahaha::common::u8;
using hahaha::math::NestedData;

// Define all supported numeric types from definitions.h
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T>
class NestedDataPropertyTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to compare values with appropriate tolerance
    void expectNear(T expected, T actual) {
        if constexpr (std::is_same_v<T, f64>) {
            EXPECT_DOUBLE_EQ(expected, actual);
        } else if constexpr (isFloatingPoint()) {
            EXPECT_FLOAT_EQ(expected, actual);
        } else {
            EXPECT_EQ(expected, actual);
        }
    }
};

TYPED_TEST_SUITE(NestedDataPropertyTypedTest, NumericTypes);

class NestedDataPropertyTest : public ::testing::Test {};

// ============================================================================
// Property Tests - Typed Test
// ============================================================================

TYPED_TEST(NestedDataPropertyTypedTest, Getters) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    NestedData<T> nd({v1, v2});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    this->expectNear(nd.getFlatData().at(0), v1);
    ASSERT_EQ(nd.getShape().size(), 1);
}

TYPED_TEST(NestedDataPropertyTypedTest, DeepNesting) {
    using T = TestFixture::Type;
    T v1 = T(1);
    NestedData<T> nd = {{{{v1}}}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShape().size(), 4);
}

TYPED_TEST(NestedDataPropertyTypedTest, LargeList) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T v9 = T(9);
    T v10 = T(10);
    NestedData<T> nd = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10};
    ASSERT_EQ(nd.getFlatData().size(), 10);
    ASSERT_EQ(nd.getShape().size(), 1);
}

TYPED_TEST(NestedDataPropertyTypedTest, SingleElementList) {
    using T = TestFixture::Type;
    T v1 = T(1);
    NestedData<T> nd = {{v1}};
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape()[0], 1);
    ASSERT_EQ(nd.getShape()[1], 1);
}

// ============================================================================
// Boundary Value Tests - Typed Test
// ============================================================================

TYPED_TEST(NestedDataPropertyTypedTest, LargeValues) {
    using T = TestFixture::Type;
    T max_val = std::numeric_limits<T>::max();
    T half_max;
    if constexpr (TestFixture::isFloatingPoint()) {
        half_max = max_val / T(2.0);
    } else {
        half_max = max_val / T(2);
    }
    NestedData<T> nd({max_val, half_max});
    this->expectNear(nd.getFlatData().at(0), max_val);
}

TYPED_TEST(NestedDataPropertyTypedTest, SmallValues) {
    using T = TestFixture::Type;
    T lowest = std::numeric_limits<T>::lowest();
    T zero = T(0);
    NestedData<T> nd({lowest, zero});
    this->expectNear(nd.getFlatData().at(0), lowest);
}

TYPED_TEST(NestedDataPropertyTypedTest, ZeroValues) {
    using T = TestFixture::Type;
    T zero = T(0);
    NestedData<T> nd = {{zero, zero}, {zero, zero}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        this->expectNear(nd.getFlatData().at(i), zero);
    }
}

TYPED_TEST(NestedDataPropertyTypedTest, MixedZeroAndNonZero) {
    using T = TestFixture::Type;
    T zero = T(0);
    T v1 = T(1);
    T v2 = T(2);
    NestedData<T> nd = {{zero, v1}, {v2, zero}};
    this->expectNear(nd.getFlatData().at(0), zero);
    this->expectNear(nd.getFlatData().at(1), v1);
}

// ============================================================================
// Floating Point Special Values Tests - Typed Test (Floating Point Only)
// ============================================================================

using FloatingPointTypes = ::testing::Types<f32, f64>;

template <typename T>
class NestedDataPropertyFloatingPointTypedTest
    : public NestedDataPropertyTypedTest<T> {};
TYPED_TEST_SUITE(NestedDataPropertyFloatingPointTypedTest, FloatingPointTypes);

TYPED_TEST(NestedDataPropertyFloatingPointTypedTest,
           FloatingPoint_Infinity_Storage) {
    using T = TestFixture::Type;
    T inf_val = std::numeric_limits<T>::infinity();
    T normal_val;
    if constexpr (std::is_same_v<T, f64>) {
        normal_val = T(1.0);
    } else {
        normal_val = T(1.0f);
    }
    NestedData<T> nd({inf_val, normal_val});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    EXPECT_TRUE(std::isinf(nd.getFlatData().at(0)));
    this->expectNear(nd.getFlatData().at(1), normal_val);
}

TYPED_TEST(NestedDataPropertyFloatingPointTypedTest,
           FloatingPoint_NaN_Storage) {
    using T = TestFixture::Type;
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    T normal_val;
    if constexpr (std::is_same_v<T, f64>) {
        normal_val = T(1.0);
    } else {
        normal_val = T(1.0f);
    }
    NestedData<T> nd({nan_val, normal_val});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    EXPECT_TRUE(std::isnan(nd.getFlatData().at(0)));
    this->expectNear(nd.getFlatData().at(1), normal_val);
}

TYPED_TEST(NestedDataPropertyFloatingPointTypedTest, FloatingPoint_Precision) {
    using T = TestFixture::Type;
    T val1, val2, val3, val4;
    if constexpr (std::is_same_v<T, f64>) {
        val1 = T(0.1);
        val2 = T(0.2);
        val3 = T(0.3);
        val4 = T(0.4);
    } else {
        val1 = T(0.1f);
        val2 = T(0.2f);
        val3 = T(0.3f);
        val4 = T(0.4f);
    }
    NestedData<T> nd = {{val1, val2}, {val3, val4}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    EXPECT_NEAR(
        nd.getFlatData().at(0), val1, std::numeric_limits<T>::epsilon() * 10);
    EXPECT_NEAR(
        nd.getFlatData().at(1), val2, std::numeric_limits<T>::epsilon() * 10);
}

// ============================================================================
// Complex Nested Structures - Typed Test
// ============================================================================

TYPED_TEST(NestedDataPropertyTypedTest, ComplexNestedStructure_4D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T v9 = T(9);
    T v10 = T(10);
    T v11 = T(11);
    T v12 = T(12);
    T v13 = T(13);
    T v14 = T(14);
    T v15 = T(15);
    T v16 = T(16);
    NestedData<T> nd = {
        {
            {{v1, v2}, {v3, v4}},
            {{v5, v6}, {v7, v8}},
        },
        {
            {{v9, v10}, {v11, v12}},
            {{v13, v14}, {v15, v16}},
        },
    };
    ASSERT_EQ(nd.getFlatData().size(), 16);
    ASSERT_EQ(nd.getShape().size(), 4);
}

TYPED_TEST(NestedDataPropertyTypedTest, RectangularNestedStructure) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    NestedData<T> nd = {
        {v1, v2, v3},
        {v4, v5, v6},
    };
    ASSERT_EQ(nd.getFlatData().size(), 6);
    ASSERT_EQ(nd.getShape().size(), 2);
}

TYPED_TEST(NestedDataPropertyTypedTest, VeryLargeNestedStructure) {
    using T = TestFixture::Type;
    NestedData<T> nd = {
        T(1),  T(2),  T(3),  T(4),  T(5),  T(6),  T(7),  T(8),  T(9),  T(10),
        T(11), T(12), T(13), T(14), T(15), T(16), T(17), T(18), T(19), T(20),
        T(21), T(22), T(23), T(24), T(25), T(26), T(27), T(28), T(29), T(30),
    };
    ASSERT_EQ(nd.getFlatData().size(), 30);
    ASSERT_EQ(nd.getShape().size(), 1);
}
