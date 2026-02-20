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
#include "math/TensorWrapper.h"
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
using hahaha::math::TensorWrapper;

// Define all supported numeric types from definitions.h
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T>
class TensorWrapperArithmeticTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get test value
    static constexpr T testValue() {
        if constexpr (isFloatingPoint()) {
            return T(5.0);
        } else {
            return T(5);
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

TYPED_TEST_SUITE(TensorWrapperArithmeticTypedTest, NumericTypes);

class TensorWrapperArithmeticTest : public ::testing::Test {};

// ============================================================================
// Addition Tests - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_TwoTensors_2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> t2(NestedData<T>{{v5, v6}, {v7, v8}});
    auto res = t1 + t2;
    this->expectNear(res.at({0, 0}), T(6));
    this->expectNear(res.at({1, 1}), T(12));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_0Dvs0D_ScalarToScalar) {
    using T = TestFixture::Type;
    T val_a = T(5);
    T val_b = T(3);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    auto r = s0a + s0b;
    this->expectNear(r.at({}), T(8));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_0Dvs1D_ScalarToVector) {
    using T = TestFixture::Type;
    T val = T(5);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> s0(val);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto r = s0 + v1_tensor;
    this->expectNear(r.at({0}), T(6));
    this->expectNear(r.at({2}), T(8));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_0Dvs2D_ScalarToMatrix) {
    using T = TestFixture::Type;
    T val = T(5);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> s0(val);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto r = s0 + m2;
    this->expectNear(r.at({0, 0}), T(6));
    this->expectNear(r.at({1, 1}), T(9));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_0Dvs3D_ScalarToTensor) {
    using T = TestFixture::Type;
    T val = T(5);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> s0(val);
    TensorWrapper<T> t3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    auto r = s0 + t3;
    this->expectNear(r.at({0, 0, 0}), T(6));
    this->expectNear(r.at({1, 1, 1}), T(13));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_1Dvs1D_VectorToVector) {
    using T = TestFixture::Type;
    T v1a = T(1);
    T v2a = T(2);
    T v3a = T(3);
    T v1b = T(4);
    T v2b = T(5);
    T v3b = T(6);
    TensorWrapper<T> v1a_tensor(NestedData<T>{v1a, v2a, v3a});
    TensorWrapper<T> v1b_tensor(NestedData<T>{v1b, v2b, v3b});
    auto r = v1a_tensor + v1b_tensor;
    this->expectNear(r.at({0}), T(5));
    this->expectNear(r.at({2}), T(9));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_2Dvs2D_MatrixToMatrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> m2c(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> m2d(NestedData<T>{{v5, v6}, {v7, v8}});
    auto r = m2c + m2d;
    this->expectNear(r.at({0, 0}), T(6));
    this->expectNear(r.at({1, 1}), T(12));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_3Dvs3D_TensorToTensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T o1 = T(1) / T(10); // 0.1 for floats, 0 for integers
    T o2 = T(2) / T(10);
    T o3 = T(3) / T(10);
    T o4 = T(4) / T(10);
    T o5 = T(5) / T(10);
    T o6 = T(6) / T(10);
    T o7 = T(7) / T(10);
    T o8 = T(8) / T(10);
    TensorWrapper<T> t3a(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> t3b(NestedData<T>{{{o1, o2}, {o3, o4}}, {{o5, o6}, {o7, o8}}});
    auto r = t3a + t3b;
    if constexpr (TestFixture::isFloatingPoint()) {
        this->expectNear(r.at({0, 0, 0}), T(1.1));
        this->expectNear(r.at({1, 1, 1}), T(8.8));
    } else {
        // For integers, small values will be 0
        this->expectNear(r.at({0, 0, 0}), v1);
        this->expectNear(r.at({1, 1, 1}), v8);
    }
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_TensorScalar) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T scalar = T(10);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    auto res = tensor + scalar;
    this->expectNear(res.at({0, 0}), T(11));
    this->expectNear(res.at({0, 1}), T(12));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Add_ScalarTensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T scalar = T(10);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    auto res = scalar + tensor;
    this->expectNear(res.at({0, 0}), T(11));
    this->expectNear(res.at({0, 1}), T(12));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest,
           Add_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{{v1, v2, v3}});
    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

// ============================================================================
// Subtraction Tests - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_TwoTensors_2D) {
    using T = TestFixture::Type;
    T v5 = T(5);
    T v6 = T(6);
    T v1 = T(1);
    T v2 = T(2);
    TensorWrapper<T> t1(NestedData<T>{{v5, v6}});
    TensorWrapper<T> t2(NestedData<T>{{v1, v2}});
    auto res = t1 - t2;
    this->expectNear(res.at({0, 0}), T(4));
    this->expectNear(res.at({0, 1}), T(4));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_0Dvs0D_ScalarToScalar) {
    using T = TestFixture::Type;
    T val_a = T(10);
    T val_b = T(3);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    auto r = s0a - s0b;
    this->expectNear(r.at({}), T(7));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_1Dvs1D_VectorToVector) {
    using T = TestFixture::Type;
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1a(NestedData<T>{v5, v6, v7});
    TensorWrapper<T> v1b(NestedData<T>{v1, v2, v3});
    auto r = v1a - v1b;
    this->expectNear(r.at({0}), T(4));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_2Dvs2D_MatrixToMatrix) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2a(NestedData<T>{{v10, v20}, {v30, v40}});
    TensorWrapper<T> m2b(NestedData<T>{{v1, v2}, {v3, v4}});
    auto r = m2a - m2b;
    this->expectNear(r.at({0, 0}), T(9));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_3Dvs3D_TensorToTensor) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v50 = T(50);
    T v60 = T(60);
    T v70 = T(70);
    T v80 = T(80);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> t3a(
        NestedData<T>{{{v10, v20}, {v30, v40}}, {{v50, v60}, {v70, v80}}});
    TensorWrapper<T> t3b(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    auto r = t3a - t3b;
    this->expectNear(r.at({0, 0, 0}), T(9));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_TensorScalar) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T scalar = T(5);
    TensorWrapper<T> tensor(NestedData<T>{{v10, v20}});
    auto res = tensor - scalar;
    this->expectNear(res.at({0, 0}), T(5));
    this->expectNear(res.at({0, 1}), T(15));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Subtract_ScalarTensor) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T scalar = T(30);
    TensorWrapper<T> tensor(NestedData<T>{{v10, v20}});
    auto res = scalar - tensor; // 30 - {10, 20} = {20, 10}
    this->expectNear(res.at({0, 0}), T(20));
    this->expectNear(res.at({0, 1}), T(10));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest,
           Subtract_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{{v1, v2, v3}});
    EXPECT_THROW(t1 - t2, std::invalid_argument);
}

// ============================================================================
// Multiplication Tests - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_TwoTensors_2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{{v3, v4}});
    auto res = t1 * t2;
    this->expectNear(res.at({0, 0}), T(3));
    this->expectNear(res.at({0, 1}), T(8));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_0Dvs0D_ScalarToScalar) {
    using T = TestFixture::Type;
    T val_a = T(5);
    T val_b = T(3);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    auto r = s0a * s0b;
    this->expectNear(r.at({}), T(15));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_1Dvs1D_VectorToVector) {
    using T = TestFixture::Type;
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    TensorWrapper<T> v1a(NestedData<T>{v2, v3, v4});
    TensorWrapper<T> v1b(NestedData<T>{v5, v6, v7});
    auto r = v1a * v1b;
    this->expectNear(r.at({0}), T(10));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_2Dvs2D_MatrixToMatrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> m2a(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> m2b(NestedData<T>{{v5, v6}, {v7, v8}});
    auto r = m2a * m2b;
    this->expectNear(r.at({0, 0}), T(5));
    this->expectNear(r.at({1, 1}), T(32));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_3Dvs3D_TensorToTensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T o1 = T(1) / T(10); // 0.1 for floats, 0 for integers
    T o2 = T(2) / T(10);
    T o3 = T(3) / T(10);
    T o4 = T(4) / T(10);
    T o5 = T(5) / T(10);
    T o6 = T(6) / T(10);
    T o7 = T(7) / T(10);
    T o8 = T(8) / T(10);
    TensorWrapper<T> t3a(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> t3b(NestedData<T>{{{o1, o2}, {o3, o4}}, {{o5, o6}, {o7, o8}}});
    auto r = t3a * t3b;
    if constexpr (TestFixture::isFloatingPoint()) {
        this->expectNear(r.at({0, 0, 0}), T(0.1));
        this->expectNear(r.at({1, 1, 1}), T(6.4));
    } else {
        // For integers, small values will be 0
        this->expectNear(r.at({0, 0, 0}), T(0));
        this->expectNear(r.at({1, 1, 1}), T(0));
    }
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_TensorScalar) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T scalar = T(5);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    auto res = tensor * scalar;
    this->expectNear(res.at({0, 0}), T(5));
    this->expectNear(res.at({0, 1}), T(10));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest, Multiply_ScalarTensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T scalar = T(5);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    auto res = scalar * tensor;
    this->expectNear(res.at({0, 0}), T(5));
    this->expectNear(res.at({0, 1}), T(10));
}

TYPED_TEST(TensorWrapperArithmeticTypedTest,
           Multiply_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{{v1, v2, v3}});
    EXPECT_THROW(t1 * t2, std::invalid_argument);
}

// ============================================================================
// Division Tests - Typed Test (Only Floating Point Types)
// ============================================================================

// Division tests only for floating point types
template <typename T> class TensorWrapperDivisionTypedTest : public ::testing::Test {
  protected:
    using Type = T;
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }
    void expectNear(T expected, T actual, T tolerance = T(0)) {
        if constexpr (std::is_same_v<T, f64>) {
            if (tolerance == T(0)) {
                EXPECT_DOUBLE_EQ(expected, actual);
            } else {
                EXPECT_NEAR(expected, actual, tolerance);
            }
        } else {
            if (tolerance == T(0)) {
                EXPECT_FLOAT_EQ(expected, actual);
            } else {
                EXPECT_NEAR(expected, actual, tolerance);
            }
        }
    }
};

using FloatingPointTypes = ::testing::Types<f32, f64>;
TYPED_TEST_SUITE(TensorWrapperDivisionTypedTest, FloatingPointTypes);

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_TwoTensors_2D) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v2 = T(2);
    T v4 = T(4);
    TensorWrapper<T> t1(NestedData<T>{{v10, v20}});
    TensorWrapper<T> t2(NestedData<T>{{v2, v4}});
    auto res = t1 / t2;
    this->expectNear(res.at({0, 0}), T(5), T(0.0001));
    this->expectNear(res.at({0, 1}), T(5), T(0.0001));
}

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_0Dvs0D_ScalarToScalar) {
    using T = TestFixture::Type;
    T val_a = T(10);
    T val_b = T(2);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    auto r = s0a / s0b;
    this->expectNear(r.at({}), T(5));
}

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_1Dvs1D_VectorToVector) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v2 = T(2);
    T v4 = T(4);
    T v5 = T(5);
    TensorWrapper<T> v1a(NestedData<T>{v10, v20, v30});
    TensorWrapper<T> v1b(NestedData<T>{v2, v4, v5});
    auto r = v1a / v1b;
    this->expectNear(r.at({0}), T(5));
    this->expectNear(r.at({1}), T(5));
    this->expectNear(r.at({2}), T(6));
}

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_2Dvs2D_MatrixToMatrix) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v2 = T(2);
    T v4 = T(4);
    T v5 = T(5);
    T v10_d = T(10);
    TensorWrapper<T> m2a(NestedData<T>{{v10, v20}, {v30, v40}});
    TensorWrapper<T> m2b(NestedData<T>{{v2, v4}, {v5, v10_d}});
    auto r = m2a / m2b;
    this->expectNear(r.at({0, 0}), T(5));
    this->expectNear(r.at({1, 1}), T(4));
}

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_3Dvs3D_TensorToTensor) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v50 = T(50);
    T v60 = T(60);
    T v70 = T(70);
    T v80 = T(80);
    T v2 = T(2);
    T v4 = T(4);
    T v5 = T(5);
    T v10_d = T(10);
    T v12 = T(12);
    T v14 = T(14);
    T v16 = T(16);
    TensorWrapper<T> t3a(
        NestedData<T>{{{v10, v20}, {v30, v40}}, {{v50, v60}, {v70, v80}}});
    TensorWrapper<T> t3b(
        NestedData<T>{{{v2, v4}, {v5, v10_d}}, {{v10_d, v12}, {v14, v16}}});
    auto r = t3a / t3b;
    this->expectNear(r.at({0, 0, 0}), T(5));
    this->expectNear(r.at({1, 1, 1}), T(5));
}

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_TensorScalar) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T scalar = T(2);
    TensorWrapper<T> tensor(NestedData<T>{{v10, v20}});
    auto res = tensor / scalar;
    this->expectNear(res.at({0, 0}), T(5), T(0.0001));
    this->expectNear(res.at({0, 1}), T(10), T(0.0001));
}

TYPED_TEST(TensorWrapperDivisionTypedTest, Divide_ScalarTensor) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T scalar = T(100);
    TensorWrapper<T> tensor(NestedData<T>{{v10, v20}});
    auto res = scalar / tensor; // 100 / {10, 20} = {10, 5}
    this->expectNear(res.at({0, 0}), T(10), T(0.0001));
    this->expectNear(res.at({0, 1}), T(5), T(0.0001));
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{{v1, v2, v3}});
    EXPECT_THROW(t1 / t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_DivisionByZero_0D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v0 = T(0);
    TensorWrapper<T> s0(v1);
    TensorWrapper<T> s0_zero(v0);
    EXPECT_THROW(s0 / s0_zero, std::runtime_error);
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_DivisionByZero_1D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v0 = T(0);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2});
    TensorWrapper<T> v1_zero(NestedData<T>{v0, v1});
    EXPECT_THROW(v1_tensor / v1_zero, std::runtime_error);
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_DivisionByZero_2D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v0 = T(0);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> m2_zero(NestedData<T>{{v0, v1}, {v2, v3}});
    EXPECT_THROW(m2 / m2_zero, std::runtime_error);
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_DivisionByZero_3D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T v0 = T(0);
    TensorWrapper<T> t3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> t3_zero(
        NestedData<T>{{{v0, v1}, {v2, v3}}, {{v4, v5}, {v6, v7}}});
    EXPECT_THROW(t3 / t3_zero, std::runtime_error);
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_DivisionByZero_Scalar_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    EXPECT_THROW(tensor / T(0), std::runtime_error);
}

TYPED_TEST(TensorWrapperDivisionTypedTest,
           Divide_DivisionByZero_ScalarDividedByTensor_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v0 = T(0);
    T v1 = T(1);
    T scalar = T(10);
    TensorWrapper<T> tensor_with_zero(NestedData<T>{{v0, v1}});
    EXPECT_THROW(scalar / tensor_with_zero, std::runtime_error);
}
