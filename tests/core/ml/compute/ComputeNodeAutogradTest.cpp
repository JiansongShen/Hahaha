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

template <typename T> class ComputeNodeAutogradTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get test value
    static constexpr T testValue(T multiplier = T(1)) {
        if constexpr (isFloatingPoint()) {
            return T(2.0) * multiplier;
        } else {
            return T(2) * multiplier;
        }
    }

    // Helper to compare values with appropriate tolerance
    void expectNear(T expected, T actual, T tolerance = T(0)) {
        if constexpr (std::is_same_v<T, f64>) {
            if (tolerance == T(0)) {
                EXPECT_DOUBLE_EQ(expected, actual);
            } else {
                EXPECT_NEAR(expected, actual, tolerance);
            }
        } else if constexpr (isFloatingPoint()) {
            if (tolerance == T(0)) {
                EXPECT_FLOAT_EQ(expected, actual);
            } else {
                EXPECT_NEAR(expected, actual, tolerance);
            }
        } else {
            EXPECT_EQ(expected, actual);
        }
    }
};

TYPED_TEST_SUITE(ComputeNodeAutogradTypedTest, NumericTypes);

class ComputeNodeAutogradTest : public ::testing::Test {};

// ============================================================================
// Backward Propagation Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(ComputeNodeAutogradTypedTest, Backward_MixedRequiresGrad_0D_Scalar) {
    using T = TestFixture::Type;
    // f = a * b + c
    // a.reqGrad = true, b.reqGrad = false, c.reqGrad = true
    T val_a = TestFixture::testValue(); // 2
    T val_b;
    T val_c;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_b = T(3.0);
        val_c = T(5.0);
    } else {
        val_b = T(3);
        val_c = T(5);
    }

    Tensor<T> a(val_a);
    a.setRequiresGrad(true);
    Tensor<T> b(val_b);
    b.setRequiresGrad(false);
    Tensor<T> c(val_c);
    c.setRequiresGrad(true);

    auto f = a * b + c;
    f.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    T expected_a_grad = val_b; // df/da = b = 3
    this->expectNear(a.grad().at({}), expected_a_grad);

    EXPECT_TRUE(b.grad().isEmpty());

    ASSERT_FALSE(c.grad().isEmpty());
    T expected_c_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_c_grad = T(1.0); // df/dc = 1
    } else {
        expected_c_grad = T(1); // df/dc = 1
    }
    this->expectNear(c.grad().at({}), expected_c_grad);
}

TYPED_TEST(ComputeNodeAutogradTypedTest, Backward_MixedRequiresGrad_1D_Vector) {
    using T = TestFixture::Type;
    T val_a1;
    T val_a2;
    T val_b1;
    T val_b2;
    T val_c1;
    T val_c2;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a1 = T(2.0);
        val_a2 = T(3.0);
        val_b1 = T(4.0);
        val_b2 = T(5.0);
        val_c1 = T(6.0);
        val_c2 = T(7.0);
    } else {
        val_a1 = T(2);
        val_a2 = T(3);
        val_b1 = T(4);
        val_b2 = T(5);
        val_c1 = T(6);
        val_c2 = T(7);
    }

    Tensor<T> a(NestedData<T>{val_a1, val_a2});
    a.setRequiresGrad(true);
    Tensor<T> b(NestedData<T>{val_b1, val_b2});
    b.setRequiresGrad(false);
    Tensor<T> c(NestedData<T>{val_c1, val_c2});
    c.setRequiresGrad(true);

    auto f = a * b + c;
    f.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    this->expectNear(a.grad().at({0}), val_b1);
    this->expectNear(a.grad().at({1}), val_b2);

    EXPECT_TRUE(b.grad().isEmpty());

    ASSERT_FALSE(c.grad().isEmpty());
    T expected_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_grad = T(1.0);
    } else {
        expected_grad = T(1);
    }
    this->expectNear(c.grad().at({0}), expected_grad);
    this->expectNear(c.grad().at({1}), expected_grad);
}

TYPED_TEST(ComputeNodeAutogradTypedTest, Backward_MixedRequiresGrad_2D_Matrix) {
    using T = TestFixture::Type;
    T val_a1, val_a2, val_a3, val_a4;
    T val_b1, val_b2, val_b3, val_b4;
    T val_c1, val_c2, val_c3, val_c4;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a1 = T(2.0);
        val_a2 = T(3.0);
        val_a3 = T(4.0);
        val_a4 = T(5.0);
        val_b1 = T(6.0);
        val_b2 = T(7.0);
        val_b3 = T(8.0);
        val_b4 = T(9.0);
        val_c1 = T(10.0);
        val_c2 = T(11.0);
        val_c3 = T(12.0);
        val_c4 = T(13.0);
    } else {
        val_a1 = T(2);
        val_a2 = T(3);
        val_a3 = T(4);
        val_a4 = T(5);
        val_b1 = T(6);
        val_b2 = T(7);
        val_b3 = T(8);
        val_b4 = T(9);
        val_c1 = T(10);
        val_c2 = T(11);
        val_c3 = T(12);
        val_c4 = T(13);
    }

    Tensor<T> a(NestedData<T>{{val_a1, val_a2}, {val_a3, val_a4}});
    a.setRequiresGrad(true);
    Tensor<T> b(NestedData<T>{{val_b1, val_b2}, {val_b3, val_b4}});
    b.setRequiresGrad(false);
    Tensor<T> c(NestedData<T>{{val_c1, val_c2}, {val_c3, val_c4}});
    c.setRequiresGrad(true);

    auto f = a * b + c;
    f.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    this->expectNear(a.grad().at({0, 0}), val_b1);
    this->expectNear(a.grad().at({0, 1}), val_b2);
    this->expectNear(a.grad().at({1, 0}), val_b3);
    this->expectNear(a.grad().at({1, 1}), val_b4);

    EXPECT_TRUE(b.grad().isEmpty());

    ASSERT_FALSE(c.grad().isEmpty());
    T expected_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_grad = T(1.0);
    } else {
        expected_grad = T(1);
    }
    this->expectNear(c.grad().at({0, 0}), expected_grad);
    this->expectNear(c.grad().at({1, 1}), expected_grad);
}

TYPED_TEST(ComputeNodeAutogradTypedTest, Backward_MixedRequiresGrad_3D_Tensor) {
    using T = TestFixture::Type;
    T val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8;
    T val_b1, val_b2, val_b3, val_b4, val_b5, val_b6, val_b7, val_b8;
    T val_c1, val_c2, val_c3, val_c4, val_c5, val_c6, val_c7, val_c8;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a1 = T(2.0);
        val_a2 = T(3.0);
        val_a3 = T(4.0);
        val_a4 = T(5.0);
        val_a5 = T(6.0);
        val_a6 = T(7.0);
        val_a7 = T(8.0);
        val_a8 = T(9.0);
        val_b1 = T(1.0);
        val_b2 = T(2.0);
        val_b3 = T(3.0);
        val_b4 = T(4.0);
        val_b5 = T(5.0);
        val_b6 = T(6.0);
        val_b7 = T(7.0);
        val_b8 = T(8.0);
        val_c1 = T(10.0);
        val_c2 = T(11.0);
        val_c3 = T(12.0);
        val_c4 = T(13.0);
        val_c5 = T(14.0);
        val_c6 = T(15.0);
        val_c7 = T(16.0);
        val_c8 = T(17.0);
    } else {
        val_a1 = T(2);
        val_a2 = T(3);
        val_a3 = T(4);
        val_a4 = T(5);
        val_a5 = T(6);
        val_a6 = T(7);
        val_a7 = T(8);
        val_a8 = T(9);
        val_b1 = T(1);
        val_b2 = T(2);
        val_b3 = T(3);
        val_b4 = T(4);
        val_b5 = T(5);
        val_b6 = T(6);
        val_b7 = T(7);
        val_b8 = T(8);
        val_c1 = T(10);
        val_c2 = T(11);
        val_c3 = T(12);
        val_c4 = T(13);
        val_c5 = T(14);
        val_c6 = T(15);
        val_c7 = T(16);
        val_c8 = T(17);
    }

    Tensor<T> a(NestedData<T>{{{val_a1, val_a2}, {val_a3, val_a4}},
                              {{val_a5, val_a6}, {val_a7, val_a8}}});
    a.setRequiresGrad(true);
    Tensor<T> b(NestedData<T>{{{val_b1, val_b2}, {val_b3, val_b4}},
                              {{val_b5, val_b6}, {val_b7, val_b8}}});
    b.setRequiresGrad(false);
    Tensor<T> c(NestedData<T>{{{val_c1, val_c2}, {val_c3, val_c4}},
                              {{val_c5, val_c6}, {val_c7, val_c8}}});
    c.setRequiresGrad(true);

    auto f = a * b + c;
    f.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    this->expectNear(a.grad().at({0, 0, 0}), val_b1);
    this->expectNear(a.grad().at({1, 1, 1}), val_b8);

    EXPECT_TRUE(b.grad().isEmpty());

    ASSERT_FALSE(c.grad().isEmpty());
    T expected_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_grad = T(1.0);
    } else {
        expected_grad = T(1);
    }
    this->expectNear(c.grad().at({0, 0, 0}), expected_grad);
    this->expectNear(c.grad().at({1, 1, 1}), expected_grad);
}
