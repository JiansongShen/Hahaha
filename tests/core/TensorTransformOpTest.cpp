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
#include "public/Tensor.h"

using hahaha::Tensor;
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

template <typename T> class TensorTransformOpTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get test value
    static constexpr T testValue(T multiplier = T(1)) {
        if constexpr (isFloatingPoint()) {
            return T(1.0) * multiplier;
        } else {
            return T(1) * multiplier;
        }
    }
};

TYPED_TEST_SUITE(TensorTransformOpTypedTest, NumericTypes);

class TensorTransformOpTest : public ::testing::Test {};

// ============================================================================
// Reshape Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorTransformOpTypedTest, Reshape_1D_To2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a1(NestedData<T>{v1, v2, v3, v4});
    auto b1 = a1.reshape({2, 2});
    EXPECT_EQ(b1.getShape().size(), 2u);
    EXPECT_EQ(b1.getShape()[0], 2u);
    EXPECT_EQ(b1.getShape()[1], 2u);
}

TYPED_TEST(TensorTransformOpTypedTest, Reshape_2D_To1D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto b2 = a2.reshape({4});
    EXPECT_EQ(b2.getShape().size(), 1u);
    EXPECT_EQ(b2.getShape()[0], 4u);
}

TYPED_TEST(TensorTransformOpTypedTest, Reshape_3D_To1D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    Tensor<T> a3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    auto b3 = a3.reshape({8});
    EXPECT_EQ(b3.getShape().size(), 1u);
    EXPECT_EQ(b3.getShape()[0], 8u);
}

TYPED_TEST(TensorTransformOpTypedTest, Reshape_3D_To2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    Tensor<T> a3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    auto b3 = a3.reshape({4, 2});
    EXPECT_EQ(b3.getShape().size(), 2u);
    EXPECT_EQ(b3.getShape()[0], 4u);
    EXPECT_EQ(b3.getShape()[1], 2u);
}

// ============================================================================
// Transpose Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorTransformOpTypedTest, Transpose_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a(NestedData<T>{{v1, v2}, {v3, v4}});
    auto c = a.transpose();
    EXPECT_EQ(c.getShape()[0], 2u);
    EXPECT_EQ(c.getShape()[1], 2u);
}
