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
class TensorWrapperInPlaceTypedTest : public ::testing::Test {
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

    // Helper to get scalar value
    static constexpr T scalarValue(T val) {
        return val;
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

TYPED_TEST_SUITE(TensorWrapperInPlaceTypedTest, NumericTypes);

class TensorWrapperInPlaceTest : public ::testing::Test {};

// ============================================================================
// In-Place Addition - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceAdd_TwoTensors_2D) {
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
    t1 += t2;
    this->expectNear(t1.at({0, 0}), T(6));
    this->expectNear(t1.at({1, 1}), T(12));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceAdd_0D_Scalar) {
    using T = TestFixture::Type;
    T val_a = T(5);
    T val_b = T(3);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    s0a += s0b;
    this->expectNear(s0a.at({}), T(8));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceAdd_1D_Vector) {
    using T = TestFixture::Type;
    T v1a = T(1);
    T v2a = T(2);
    T v3a = T(3);
    T v1b = T(4);
    T v2b = T(5);
    T v3b = T(6);
    TensorWrapper<T> v1a_tensor(NestedData<T>{v1a, v2a, v3a});
    TensorWrapper<T> v1b_tensor(NestedData<T>{v1b, v2b, v3b});
    v1a_tensor += v1b_tensor;
    this->expectNear(v1a_tensor.at({0}), T(5));
    this->expectNear(v1a_tensor.at({2}), T(9));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceAdd_2D_Matrix) {
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
    m2a += m2b;
    this->expectNear(m2a.at({0, 0}), T(6));
    this->expectNear(m2a.at({1, 1}), T(12));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceAdd_3D_Tensor) {
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
    TensorWrapper<T> t3a(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> t3b(
        NestedData<T>{{{o1, o2}, {o3, o4}}, {{o5, o6}, {o7, o8}}});
    t3a += t3b;
    if constexpr (TestFixture::isFloatingPoint()) {
        this->expectNear(t3a.at({0, 0, 0}), T(1.1));
        this->expectNear(t3a.at({1, 1, 1}), T(8.8));
    } else {
        // For integers, small values will be 0
        this->expectNear(t3a.at({0, 0, 0}), v1);
        this->expectNear(t3a.at({1, 1, 1}), v8);
    }
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceAdd_Scalar) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T scalar = T(2);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    tensor += scalar;
    this->expectNear(tensor.at({0, 0}), T(3));
    this->expectNear(tensor.at({0, 1}), T(4));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest,
           InPlaceAdd_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(t1 += t2, std::invalid_argument);
}

// ============================================================================
// In-Place Subtraction - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceSubtract_TwoTensors_2D) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> t1(NestedData<T>{{v10, v20}, {v30, v40}});
    TensorWrapper<T> t2(NestedData<T>{{v1, v2}, {v3, v4}});
    t1 -= t2;
    this->expectNear(t1.at({0, 0}), T(9));
    this->expectNear(t1.at({1, 1}), T(36));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceSubtract_0D_Scalar) {
    using T = TestFixture::Type;
    T val_a = T(10);
    T val_b = T(3);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    s0a -= s0b;
    this->expectNear(s0a.at({}), T(7));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceSubtract_1D_Vector) {
    using T = TestFixture::Type;
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1a(NestedData<T>{v5, v6, v7});
    TensorWrapper<T> v1b(NestedData<T>{v1, v2, v3});
    v1a -= v1b;
    this->expectNear(v1a.at({0}), T(4));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceSubtract_2D_Matrix) {
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
    m2a -= m2b;
    this->expectNear(m2a.at({0, 0}), T(9));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceSubtract_3D_Tensor) {
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
    TensorWrapper<T> t3b(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    t3a -= t3b;
    this->expectNear(t3a.at({0, 0, 0}), T(9));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceSubtract_Scalar) {
    using T = TestFixture::Type;
    T v5 = T(5);
    T v6 = T(6);
    T scalar = T(1);
    TensorWrapper<T> tensor(NestedData<T>{{v5, v6}});
    tensor -= scalar;
    this->expectNear(tensor.at({0, 0}), T(4));
    this->expectNear(tensor.at({0, 1}), T(5));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest,
           InPlaceSubtract_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(t1 -= t2, std::invalid_argument);
}

// ============================================================================
// In-Place Multiplication - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceMultiply_TwoTensors_2D) {
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
    t1 *= t2;
    this->expectNear(t1.at({0, 0}), T(5));
    this->expectNear(t1.at({1, 1}), T(32));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceMultiply_0D_Scalar) {
    using T = TestFixture::Type;
    T val_a = T(5);
    T val_b = T(3);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    s0a *= s0b;
    this->expectNear(s0a.at({}), T(15));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceMultiply_1D_Vector) {
    using T = TestFixture::Type;
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    TensorWrapper<T> v1a(NestedData<T>{v2, v3, v4});
    TensorWrapper<T> v1b(NestedData<T>{v5, v6, v7});
    v1a *= v1b;
    this->expectNear(v1a.at({0}), T(10));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceMultiply_2D_Matrix) {
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
    m2a *= m2b;
    this->expectNear(m2a.at({0, 0}), T(5));
    this->expectNear(m2a.at({1, 1}), T(32));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceMultiply_3D_Tensor) {
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
    TensorWrapper<T> t3a(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> t3b(
        NestedData<T>{{{o1, o2}, {o3, o4}}, {{o5, o6}, {o7, o8}}});
    t3a *= t3b;
    if constexpr (TestFixture::isFloatingPoint()) {
        this->expectNear(t3a.at({0, 0, 0}), T(0.1));
        this->expectNear(t3a.at({1, 1, 1}), T(6.4));
    } else {
        // For integers, small values will be 0
        this->expectNear(t3a.at({0, 0, 0}), T(0));
        this->expectNear(t3a.at({1, 1, 1}), T(0));
    }
}

TYPED_TEST(TensorWrapperInPlaceTypedTest, InPlaceMultiply_Scalar) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T scalar = T(3);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}});
    tensor *= scalar;
    this->expectNear(tensor.at({0, 0}), T(3));
    this->expectNear(tensor.at({0, 1}), T(6));
}

TYPED_TEST(TensorWrapperInPlaceTypedTest,
           InPlaceMultiply_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(t1 *= t2, std::invalid_argument);
}

// ============================================================================
// In-Place Division - Typed Test (Only Floating Point Types)
// ============================================================================

// Division tests only for floating point types
template <typename T>
class TensorWrapperInPlaceDivideTypedTest : public ::testing::Test {
  protected:
    using Type = T;
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }
    void expectNear(T expected, T actual) {
        if constexpr (std::is_same_v<T, f64>) {
            EXPECT_DOUBLE_EQ(expected, actual);
        } else {
            EXPECT_FLOAT_EQ(expected, actual);
        }
    }
};

using FloatingPointTypes = ::testing::Types<f32, f64>;
TYPED_TEST_SUITE(TensorWrapperInPlaceDivideTypedTest, FloatingPointTypes);

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest, InPlaceDivide_TwoTensors_2D) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v2 = T(2);
    T v4 = T(4);
    T v5 = T(5);
    T v10_d = T(10);
    TensorWrapper<T> t1(NestedData<T>{{v10, v20}, {v30, v40}});
    TensorWrapper<T> t2(NestedData<T>{{v2, v4}, {v5, v10_d}});
    t1 /= t2;
    this->expectNear(t1.at({0, 0}), T(5));
    this->expectNear(t1.at({1, 1}), T(4));
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest, InPlaceDivide_0D_Scalar) {
    using T = TestFixture::Type;
    T val_a = T(10);
    T val_b = T(2);
    TensorWrapper<T> s0a(val_a);
    TensorWrapper<T> s0b(val_b);
    s0a /= s0b;
    this->expectNear(s0a.at({}), T(5));
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest, InPlaceDivide_1D_Vector) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T v30 = T(30);
    T v2 = T(2);
    T v4 = T(4);
    T v5 = T(5);
    TensorWrapper<T> v1a(NestedData<T>{v10, v20, v30});
    TensorWrapper<T> v1b(NestedData<T>{v2, v4, v5});
    v1a /= v1b;
    this->expectNear(v1a.at({0}), T(5));
    this->expectNear(v1a.at({1}), T(5));
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest, InPlaceDivide_2D_Matrix) {
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
    m2a /= m2b;
    this->expectNear(m2a.at({0, 0}), T(5));
    this->expectNear(m2a.at({1, 1}), T(4));
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest, InPlaceDivide_3D_Tensor) {
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
    t3a /= t3b;
    this->expectNear(t3a.at({0, 0, 0}), T(5));
    this->expectNear(t3a.at({1, 1, 1}), T(5));
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest, InPlaceDivide_Scalar) {
    using T = TestFixture::Type;
    T v10 = T(10);
    T v20 = T(20);
    T scalar = T(5);
    TensorWrapper<T> tensor(NestedData<T>{{v10, v20}});
    tensor /= scalar;
    this->expectNear(tensor.at({0, 0}), T(2));
    this->expectNear(tensor.at({0, 1}), T(4));
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest,
           InPlaceDivide_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(t1 /= t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest,
           InPlaceDivide_DivisionByZero_0D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    TensorWrapper<T> s0(v1);
    EXPECT_THROW(s0 /= T(0), std::runtime_error);
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest,
           InPlaceDivide_DivisionByZero_1D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2});
    EXPECT_THROW(v1_tensor /= T(0), std::runtime_error);
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest,
           InPlaceDivide_DivisionByZero_2D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(m2 /= T(0), std::runtime_error);
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest,
           InPlaceDivide_DivisionByZero_3D_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> t3(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_THROW(t3 /= T(0), std::runtime_error);
}

TYPED_TEST(TensorWrapperInPlaceDivideTypedTest,
           InPlaceDivide_DivisionByZero_Tensor_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v0 = T(0);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}});
    TensorWrapper<T> t2(NestedData<T>{{v0, v1}});
    EXPECT_THROW(t1 /= t2, std::runtime_error);
}
