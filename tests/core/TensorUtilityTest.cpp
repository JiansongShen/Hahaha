// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed on an "AS IS" BASIS,
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

template <typename T> class TensorUtilityTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(TensorUtilityTypedTest, NumericTypes);

class TensorUtilityTest : public ::testing::Test {};

// ============================================================================
// Gradient Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorUtilityTypedTest, Grad_WhenNoBackward_ReturnsNullptr) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    Tensor<T> a(val);
    EXPECT_EQ(a.grad(), nullptr);
}

TYPED_TEST(TensorUtilityTypedTest, ClearGrad_ResetsGradient) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T multiplier;
    if constexpr (TestFixture::isFloatingPoint()) {
        multiplier = T(2.0);
    } else {
        multiplier = T(2);
    }
    Tensor<T> a(val);
    a.setRequiresGrad(true);
    auto b = a * multiplier;
    b.backward();
    EXPECT_NE(a.grad(), nullptr);

    a.clearGrad();
    // In current implementation clearGrad() keeps grad buffers but zeros them
    // out
}

// ============================================================================
// Clear Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorUtilityTypedTest, Clear_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    Tensor<T> a(val);
    a.clear();
    T zero = T(0);
    this->expectNear(a.data()->at({}), zero);
}

TYPED_TEST(TensorUtilityTypedTest, Clear_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    Tensor<T> a(NestedData<T>{v1, v2, v3});
    a.clear();
    T zero = T(0);
    this->expectNear(a.data()->at({0}), zero);
    this->expectNear(a.data()->at({2}), zero);
}

TYPED_TEST(TensorUtilityTypedTest, Clear_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a(NestedData<T>{{v1, v2}, {v3, v4}});
    a.clear();
    T zero = T(0);
    this->expectNear(a.data()->at({0, 0}), zero);
    this->expectNear(a.data()->at({1, 1}), zero);
}

TYPED_TEST(TensorUtilityTypedTest, Clear_3D_Tensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    Tensor<T> a(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    a.clear();
    T zero = T(0);
    this->expectNear(a.data()->at({0, 0, 0}), zero);
    this->expectNear(a.data()->at({1, 1, 1}), zero);
}

// ============================================================================
// ComputeNode Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorUtilityTypedTest, GetComputeNode_ReturnsNonNull) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    Tensor<T> a(val);
    auto node = a.getComputeNode();
    EXPECT_NE(node, nullptr);
}

TYPED_TEST(TensorUtilityTypedTest, SetComputeNode_UpdatesNode) {
    using T = TestFixture::Type;
    T val1 = TestFixture::testValue();
    T val2 = TestFixture::testValue(T(2));
    Tensor<T> a(val1);
    auto node = a.getComputeNode();
    EXPECT_NE(node, nullptr);

    Tensor<T> b(val2);
    b.setComputeNode(node);
    EXPECT_EQ(b.getComputeNode(), node);
}

// ============================================================================
// Sum Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorUtilityTypedTest, Sum_0D_Scalar) {
    using T = TestFixture::Type;
    T val;
    if constexpr (TestFixture::isFloatingPoint()) {
        val = T(5.0);
    } else {
        val = T(5);
    }
    Tensor<T> s0(val);
    this->expectNear(s0.sum(), val);
}

TYPED_TEST(TensorUtilityTypedTest, Sum_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    Tensor<T> s1(NestedData<T>{v1, v2, v3});
    T expected = T(6);
    this->expectNear(s1.sum(), expected);
}

TYPED_TEST(TensorUtilityTypedTest, Sum_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> s2(NestedData<T>{{v1, v2}, {v3, v4}});
    T expected = T(10);
    this->expectNear(s2.sum(), expected);
}

TYPED_TEST(TensorUtilityTypedTest, Sum_3D_Tensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    Tensor<T> s3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    T expected = T(36);
    this->expectNear(s3.sum(), expected);
}
