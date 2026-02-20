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
//

#include <gtest/gtest.h>

#include "../../../../core/include/public/Tensor.h"
#include "common/definitions.h"

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

template <typename T> class UnaryOpTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(UnaryOpTypedTest, NumericTypes);

class UnaryOpTest : public ::testing::Test {};

TYPED_TEST(UnaryOpTypedTest, Reshape_Backward_Full) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a(NestedData<T>{v1, v2, v3, v4});
    a.setRequiresGrad(true);
    auto b = a.reshape({2, 2});

    // Create upstream gradient
    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    Tensor<T> grad(NestedData<T>{{g1, g2}, {g3, g4}});
    b.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad.data()->clone()));

    // Trigger gradFun
    b.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a.grad().isEmpty());
    this->expectNear(a.grad().data()->at({0}), g1);
    this->expectNear(a.grad().data()->at({3}), g4);
}

TYPED_TEST(UnaryOpTypedTest, Transpose_Backward_Full) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a(NestedData<T>{{v1, v2}, {v3, v4}});
    a.setRequiresGrad(true);
    auto b = a.transpose();

    // Upstream grad
    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    Tensor<T> grad(NestedData<T>{{g1, g2}, {g3, g4}});
    b.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad.data()->clone()));

    b.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a.grad().isEmpty());
    this->expectNear(a.grad().data()->at({0, 1}), g3);
}

TYPED_TEST(UnaryOpTypedTest, Negate_Backward_Complex) {
    using T = TestFixture::Type;
    // f = -(a * b)
    T val_a, val_b;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a = T(2.0);
        val_b = T(3.0);
    } else {
        val_a = T(2);
        val_b = T(3);
    }
    Tensor<T> a(val_a);
    a.setRequiresGrad(true);
    Tensor<T> b(val_b);
    b.setRequiresGrad(true);

    auto ab = a * b;
    auto f = -ab;

    f.backward();

    // df/da = -b = -3
    // df/db = -a = -2
    T expected_a_grad, expected_b_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_a_grad = T(-3.0);
        expected_b_grad = T(-2.0);
    } else {
        expected_a_grad = T(-3);
        expected_b_grad = T(-2);
    }
    ASSERT_FALSE(a.grad().isEmpty());
    this->expectNear(a.grad().data()->at({}), expected_a_grad);
    ASSERT_FALSE(b.grad().isEmpty());
    this->expectNear(b.grad().data()->at({}), expected_b_grad);
}

// ============================================================================
// Reshape Backward Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(UnaryOpTypedTest, Reshape_Backward_1D_To2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a1(NestedData<T>{v1, v2, v3, v4});
    a1.setRequiresGrad(true);
    auto b1 = a1.reshape({2, 2});

    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    Tensor<T> grad1(NestedData<T>{{g1, g2}, {g3, g4}});
    b1.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad1.data()->clone()));

    b1.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a1.grad().isEmpty());
    this->expectNear(a1.grad().data()->at({0}), g1);
    this->expectNear(a1.grad().data()->at({3}), g4);
}

TYPED_TEST(UnaryOpTypedTest, Reshape_Backward_2D_To1D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a2(NestedData<T>{{v1, v2}, {v3, v4}});
    a2.setRequiresGrad(true);
    auto b2 = a2.reshape({4});

    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    Tensor<T> grad2(NestedData<T>{g1, g2, g3, g4});
    b2.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad2.data()->clone()));

    b2.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a2.grad().isEmpty());
    this->expectNear(a2.grad().data()->at({0, 0}), g1);
    this->expectNear(a2.grad().data()->at({1, 1}), g4);
}

TYPED_TEST(UnaryOpTypedTest, Reshape_Backward_3D_To1D) {
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
    a3.setRequiresGrad(true);
    auto b3 = a3.reshape({8});

    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    T g5 = T(50);
    T g6 = T(60);
    T g7 = T(70);
    T g8 = T(80);
    Tensor<T> grad3(NestedData<T>{g1, g2, g3, g4, g5, g6, g7, g8});
    b3.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad3.data()->clone()));

    b3.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a3.grad().isEmpty());
    this->expectNear(a3.grad().data()->at({0, 0, 0}), g1);
    this->expectNear(a3.grad().data()->at({1, 1, 1}), g8);
}

TYPED_TEST(UnaryOpTypedTest, Reshape_Backward_0D_Scalar_To1D) {
    using T = TestFixture::Type;
    T val;
    T grad_val;
    if constexpr (TestFixture::isFloatingPoint()) {
        val = T(10.0);
        grad_val = T(5.0);
    } else {
        val = T(10);
        grad_val = T(5);
    }
    Tensor<T> a0(val);
    a0.setRequiresGrad(true);
    auto b0 = a0.reshape({1});

    Tensor<T> grad0(NestedData<T>{grad_val});
    b0.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad0.data()->clone()));

    b0.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a0.grad().isEmpty());
    this->expectNear(a0.grad().data()->at({}), grad_val);
}

// ============================================================================
// Transpose Backward Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(UnaryOpTypedTest, Transpose_Backward_2D_SquareMatrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a2(NestedData<T>{{v1, v2}, {v3, v4}});
    a2.setRequiresGrad(true);
    auto b2 = a2.transpose();

    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    Tensor<T> grad2(NestedData<T>{{g1, g2}, {g3, g4}});
    b2.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad2.data()->clone()));

    b2.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a2.grad().isEmpty());
    this->expectNear(a2.grad().data()->at({0, 1}), g3);
    this->expectNear(a2.grad().data()->at({1, 0}), g2);
}

TYPED_TEST(UnaryOpTypedTest, Transpose_Backward_2D_RectangularMatrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    Tensor<T> a2b(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}});
    a2b.setRequiresGrad(true);
    auto b2b = a2b.transpose();

    T g1 = T(10);
    T g2 = T(20);
    T g3 = T(30);
    T g4 = T(40);
    T g5 = T(50);
    T g6 = T(60);
    Tensor<T> grad2b(NestedData<T>{{g1, g2}, {g3, g4}, {g5, g6}});
    b2b.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<T>>(grad2b.data()->clone()));

    b2b.getComputeNode()->getGradFun()();

    ASSERT_FALSE(a2b.grad().isEmpty());
    this->expectNear(a2b.grad().data()->at({0, 0}), g1);
    this->expectNear(a2b.grad().data()->at({0, 1}), g3);
    this->expectNear(a2b.grad().data()->at({0, 2}), g5);
}

// ============================================================================
// Negate Backward Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(UnaryOpTypedTest, Negate_Backward_0D_Scalar) {
    using T = TestFixture::Type;
    T val_a, val_b;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a = T(2.0);
        val_b = T(3.0);
    } else {
        val_a = T(2);
        val_b = T(3);
    }
    Tensor<T> a0(val_a);
    a0.setRequiresGrad(true);
    Tensor<T> b0(val_b);
    b0.setRequiresGrad(true);

    auto ab0 = a0 * b0;
    auto f0 = -ab0;
    f0.backward();

    T expected_a_grad, expected_b_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_a_grad = T(-3.0);
        expected_b_grad = T(-2.0);
    } else {
        expected_a_grad = T(-3);
        expected_b_grad = T(-2);
    }
    ASSERT_FALSE(a0.grad().isEmpty());
    this->expectNear(a0.grad().data()->at({}), expected_a_grad);
    ASSERT_FALSE(b0.grad().isEmpty());
    this->expectNear(b0.grad().data()->at({}), expected_b_grad);
}

TYPED_TEST(UnaryOpTypedTest, Negate_Backward_1D_Vector) {
    using T = TestFixture::Type;
    T val_a1, val_a2, val_b1, val_b2;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a1 = T(2.0);
        val_a2 = T(3.0);
        val_b1 = T(4.0);
        val_b2 = T(5.0);
    } else {
        val_a1 = T(2);
        val_a2 = T(3);
        val_b1 = T(4);
        val_b2 = T(5);
    }
    Tensor<T> a1(NestedData<T>{val_a1, val_a2});
    a1.setRequiresGrad(true);
    Tensor<T> b1(NestedData<T>{val_b1, val_b2});
    b1.setRequiresGrad(true);

    auto ab1 = a1 * b1;
    auto f1 = -ab1;
    f1.backward();

    T expected_a1_grad, expected_a2_grad, expected_b1_grad, expected_b2_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_a1_grad = T(-4.0);
        expected_a2_grad = T(-5.0);
        expected_b1_grad = T(-2.0);
        expected_b2_grad = T(-3.0);
    } else {
        expected_a1_grad = T(-4);
        expected_a2_grad = T(-5);
        expected_b1_grad = T(-2);
        expected_b2_grad = T(-3);
    }
    ASSERT_FALSE(a1.grad().isEmpty());
    this->expectNear(a1.grad().data()->at({0}), expected_a1_grad);
    this->expectNear(a1.grad().data()->at({1}), expected_a2_grad);
    ASSERT_FALSE(b1.grad().isEmpty());
    this->expectNear(b1.grad().data()->at({0}), expected_b1_grad);
    this->expectNear(b1.grad().data()->at({1}), expected_b2_grad);
}

TYPED_TEST(UnaryOpTypedTest, Negate_Backward_2D_Matrix) {
    using T = TestFixture::Type;
    T val_a1, val_a2, val_a3, val_a4, val_b1, val_b2, val_b3, val_b4;
    if constexpr (TestFixture::isFloatingPoint()) {
        val_a1 = T(2.0);
        val_a2 = T(3.0);
        val_a3 = T(4.0);
        val_a4 = T(5.0);
        val_b1 = T(6.0);
        val_b2 = T(7.0);
        val_b3 = T(8.0);
        val_b4 = T(9.0);
    } else {
        val_a1 = T(2);
        val_a2 = T(3);
        val_a3 = T(4);
        val_a4 = T(5);
        val_b1 = T(6);
        val_b2 = T(7);
        val_b3 = T(8);
        val_b4 = T(9);
    }
    Tensor<T> a2(NestedData<T>{{val_a1, val_a2}, {val_a3, val_a4}});
    a2.setRequiresGrad(true);
    Tensor<T> b2(NestedData<T>{{val_b1, val_b2}, {val_b3, val_b4}});
    b2.setRequiresGrad(true);

    auto ab2 = a2 * b2;
    auto f2 = -ab2;
    f2.backward();

    T expected_a00_grad, expected_a11_grad, expected_b00_grad, expected_b11_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_a00_grad = T(-6.0);
        expected_a11_grad = T(-9.0);
        expected_b00_grad = T(-2.0);
        expected_b11_grad = T(-5.0);
    } else {
        expected_a00_grad = T(-6);
        expected_a11_grad = T(-9);
        expected_b00_grad = T(-2);
        expected_b11_grad = T(-5);
    }
    ASSERT_FALSE(a2.grad().isEmpty());
    this->expectNear(a2.grad().data()->at({0, 0}), expected_a00_grad);
    this->expectNear(a2.grad().data()->at({1, 1}), expected_a11_grad);
    ASSERT_FALSE(b2.grad().isEmpty());
    this->expectNear(b2.grad().data()->at({0, 0}), expected_b00_grad);
    this->expectNear(b2.grad().data()->at({1, 1}), expected_b11_grad);
}

TYPED_TEST(UnaryOpTypedTest, Negate_Backward_3D_Tensor) {
    using T = TestFixture::Type;
    T val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8;
    T val_b1, val_b2, val_b3, val_b4, val_b5, val_b6, val_b7, val_b8;
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
    }
    Tensor<T> a3(NestedData<T>{{{val_a1, val_a2}, {val_a3, val_a4}},
                               {{val_a5, val_a6}, {val_a7, val_a8}}});
    a3.setRequiresGrad(true);
    Tensor<T> b3(NestedData<T>{{{val_b1, val_b2}, {val_b3, val_b4}},
                               {{val_b5, val_b6}, {val_b7, val_b8}}});
    b3.setRequiresGrad(true);

    auto ab3 = a3 * b3;
    auto f3 = -ab3;
    f3.backward();

    T expected_a000_grad, expected_a111_grad, expected_b000_grad, expected_b111_grad;
    if constexpr (TestFixture::isFloatingPoint()) {
        expected_a000_grad = T(-1.0);
        expected_a111_grad = T(-8.0);
        expected_b000_grad = T(-2.0);
        expected_b111_grad = T(-9.0);
    } else {
        expected_a000_grad = T(-1);
        expected_a111_grad = T(-8);
        expected_b000_grad = T(-2);
        expected_b111_grad = T(-9);
    }
    ASSERT_FALSE(a3.grad().isEmpty());
    this->expectNear(a3.grad().data()->at({0, 0, 0}), expected_a000_grad);
    this->expectNear(a3.grad().data()->at({1, 1, 1}), expected_a111_grad);
    ASSERT_FALSE(b3.grad().isEmpty());
    this->expectNear(b3.grad().data()->at({0, 0, 0}), expected_b000_grad);
    this->expectNear(b3.grad().data()->at({1, 1, 1}), expected_b111_grad);
}
