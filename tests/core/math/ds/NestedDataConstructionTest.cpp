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

#include <gtest/gtest.h>

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
class NestedDataConstructionTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get scalar value
    static constexpr T scalarValue() {
        if constexpr (isFloatingPoint()) {
            return T(42.0);
        } else {
            return T(42);
        }
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

TYPED_TEST_SUITE(NestedDataConstructionTypedTest, NumericTypes);

class NestedDataConstructionTest : public ::testing::Test {};

// ============================================================================
// Initialization Tests - Typed Test
// ============================================================================

TYPED_TEST(NestedDataConstructionTypedTest, InitializeViaInitializerList_1D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    NestedData<T> nd(NestedData<T>{v1, v2, v3, v4, v5});
    ASSERT_EQ(nd.getFlatData().size(), 5);
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape().at(0), 5);
    this->expectNear(nd.getFlatData().at(0), v1);
}

TYPED_TEST(NestedDataConstructionTypedTest, ZeroDimension_Scalar) {
    using T = TestFixture::Type;
    T value = TestFixture::scalarValue();
    NestedData<T> nd(value);
    ASSERT_EQ(nd.getFlatData().size(), 1);
    this->expectNear(nd.getFlatData()[0], value);
    ASSERT_EQ(nd.getShape().size(), 0);
}

TYPED_TEST(NestedDataConstructionTypedTest, OneDimension_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    NestedData<T> nd({v1, v2, v3});
    ASSERT_EQ(nd.getFlatData().size(), 3);
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape()[0], 3);
}

TYPED_TEST(NestedDataConstructionTypedTest, TwoDimension_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    NestedData<T> nd = {{v1, v2}, {v3, v4}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape()[0], 2);
    ASSERT_EQ(nd.getShape()[1], 2);
}

TYPED_TEST(NestedDataConstructionTypedTest, ThreeDimension_Tensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    NestedData<T> nd = {{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}};
    ASSERT_EQ(nd.getFlatData().size(), 8);
    ASSERT_EQ(nd.getShape().size(), 3);
    ASSERT_EQ(nd.getShape()[0], 2);
    ASSERT_EQ(nd.getShape()[1], 2);
    ASSERT_EQ(nd.getShape()[2], 2);
}

TYPED_TEST(NestedDataConstructionTypedTest, InitializeWithEmptyList) {
    using T = TestFixture::Type;
    NestedData<T> nd({});
    ASSERT_EQ(nd.getFlatData().size(), 0);
    ASSERT_EQ(nd.getShape().size(), 0);
}

TYPED_TEST(NestedDataConstructionTypedTest, ScalarImplicitConversion) {
    using T = TestFixture::Type;
    T val = TestFixture::scalarValue();
    NestedData<T> nd(val);
    ASSERT_EQ(nd.getFlatData().size(), 1);
    this->expectNear(nd.getFlatData().at(0), val);
}

// NOTE: In fact, we do not recommend to build NestedData from this way
// because it has a different behavior between gcc and clang/msvc
TYPED_TEST(NestedDataConstructionTypedTest, OneDimension_SingleElement) {
    // using T = TestFixture::Type;
    // T v1 = T(1);
    // NestedData<T> nd(v1);
    // ASSERT_EQ(nd.getFlatData().size(), 1);
    // ASSERT_EQ(nd.getShape().size(), 1);
    // ASSERT_EQ(nd.getShape()[0], 1);
    // this->expectNear(nd.getFlatData()[0], v1);
    GTEST_SKIP() << "Skipped";
}

TYPED_TEST(NestedDataConstructionTypedTest, TwoDimension_SingleElement) {
    using T = TestFixture::Type;
    T v1 = T(1);
    NestedData<T> nd = {{v1}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape()[0], 1);
    ASSERT_EQ(nd.getShape()[1], 1);
    this->expectNear(nd.getFlatData()[0], v1);
}

TYPED_TEST(NestedDataConstructionTypedTest, ThreeDimension_SingleElement) {
    using T = TestFixture::Type;
    T v1 = T(1);
    NestedData<T> nd = {{{v1}}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShape().size(), 3);
    ASSERT_EQ(nd.getShape()[0], 1);
    ASSERT_EQ(nd.getShape()[1], 1);
    ASSERT_EQ(nd.getShape()[2], 1);
    this->expectNear(nd.getFlatData()[0], v1);
}
