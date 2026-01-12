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

#include "backend/Device.h"
#include "common/definitions.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"

using hahaha::backend::Device;
using hahaha::backend::DeviceType;
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
using hahaha::math::TensorShape;
using hahaha::math::TensorWrapper;

// Define all supported numeric types from definitions.h
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T>
class TensorWrapperUtilityTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(TensorWrapperUtilityTypedTest, NumericTypes);

class TensorWrapperUtilityTest : public ::testing::Test {};

// ============================================================================
// Unary Negation - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, UnaryNegation_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(-2);
    T v3 = T(3);
    T v4 = T(-4);
    TensorWrapper<T> t(NestedData<T>{{v1, v2}});
    auto r = -t;
    this->expectNear(r.at({0, 0}), T(-1));
    this->expectNear(r.at({0, 1}), T(2));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, UnaryNegation_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    auto r0 = -s0;
    this->expectNear(r0.at({}), T(-5));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, UnaryNegation_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(-3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto r1 = -v1_tensor;
    this->expectNear(r1.at({0}), T(-1));
    this->expectNear(r1.at({1}), T(-2));
    this->expectNear(r1.at({2}), T(3));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, UnaryNegation_2D_Matrix_AllElements) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(-2);
    T v3 = T(3);
    T v4 = T(-4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto r2 = -m2;
    this->expectNear(r2.at({0, 0}), T(-1));
    this->expectNear(r2.at({0, 1}), T(2));
    this->expectNear(r2.at({1, 0}), T(-3));
    this->expectNear(r2.at({1, 1}), T(4));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, UnaryNegation_3D_Tensor) {
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
    auto r3 = -t3;
    this->expectNear(r3.at({0, 0, 0}), T(-1));
    this->expectNear(r3.at({1, 1, 1}), T(-8));
}

// ============================================================================
// Clear - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, Clear_2D_Matrix) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> t1(TensorShape({2, 2}), val);
    t1.clear();
    T expected = T(0);
    this->expectNear(t1.at({0, 0}), expected);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clear_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    s0.clear();
    T expected = T(0);
    this->expectNear(s0.at({}), expected);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clear_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    v1_tensor.clear();
    T expected = T(0);
    this->expectNear(v1_tensor.at({0}), expected);
    this->expectNear(v1_tensor.at({2}), expected);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clear_2D_Matrix_AllElements) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    m2.clear();
    T expected = T(0);
    this->expectNear(m2.at({0, 0}), expected);
    this->expectNear(m2.at({1, 1}), expected);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clear_3D_Tensor) {
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
    t3.clear();
    T expected = T(0);
    this->expectNear(t3.at({0, 0, 0}), expected);
    this->expectNear(t3.at({1, 1, 1}), expected);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clear_OnEmptyTensor_NoThrow) {
    using T = TestFixture::Type;
    TensorWrapper<T> tensor;
    EXPECT_NO_THROW(tensor.clear());
    EXPECT_EQ(tensor.getTotalSize(), 0);
}

// ============================================================================
// Clone - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, Clone_2D_Matrix_CreatesDeepCopy) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v100 = T(100);
    TensorWrapper<T> t1(NestedData<T>{{v1, v2}, {v3, v4}});
    auto c1 = t1.clone();
    this->expectNear(c1.at({0, 0}), v1);
    t1.at({0, 0}) = v100;
    this->expectNear(c1.at({0, 0}), v1); // Copy should not change
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clone_0D_Scalar_CreatesDeepCopy) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T val_10 = T(10);
    TensorWrapper<T> s0(val);
    auto c0 = s0.clone();
    this->expectNear(c0.at({}), val);
    s0.at({}) = val_10;
    this->expectNear(c0.at({}), val); // Copy should not change
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clone_1D_Vector_CreatesDeepCopy) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v100 = T(100);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto c1 = v1_tensor.clone();
    this->expectNear(c1.at({0}), v1);
    v1_tensor.at({0}) = v100;
    this->expectNear(c1.at({0}), v1); // Copy should not change
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clone_3D_Tensor_CreatesDeepCopy) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T v100 = T(100);
    TensorWrapper<T> t3(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    auto c3 = t3.clone();
    this->expectNear(c3.at({0, 0, 0}), v1);
    t3.at({0, 0, 0}) = v100;
    this->expectNear(c3.at({0, 0, 0}), v1); // Copy should not change
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Clone_PreservesShapeAndStride) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> original(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    auto copy = original.clone();
    EXPECT_EQ(copy.getShape(), original.getShape());
    EXPECT_EQ(copy.getStride().toString(), original.getStride().toString());
    EXPECT_EQ(copy.getDevice().type, original.getDevice().type);
}

// ============================================================================
// Axpy - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T a = T(2);
    TensorWrapper<T> x(NestedData<T>{v1, v2});
    TensorWrapper<T> y(NestedData<T>{v3, v4});
    y.axpy(a, x); // y = 2*x + y = {2+3, 4+4} = {5, 8}
    this->expectNear(y.at({0}), T(5));
    this->expectNear(y.at({1}), T(8));
    this->expectNear(x.at({0}), v1); // x unchanged
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_0D_Scalar) {
    using T = TestFixture::Type;
    T x_val = T(2);
    T y_val = T(3);
    T a = T(2);
    TensorWrapper<T> x0(x_val);
    TensorWrapper<T> y0(y_val);
    y0.axpy(a, x0); // y = 2*2 + 3 = 7
    this->expectNear(y0.at({}), T(7));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T a = T(2);
    TensorWrapper<T> x2(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> y2(NestedData<T>{{v5, v6}, {v7, v8}});
    y2.axpy(a, x2);                         // y = 2*x + y
    this->expectNear(y2.at({0, 0}), T(7));  // 2*1 + 5
    this->expectNear(y2.at({1, 1}), T(16)); // 2*4 + 8
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_3D_Tensor) {
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
    T a = T(2);
    TensorWrapper<T> x3(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> y3(
        NestedData<T>{{{o1, o2}, {o3, o4}}, {{o5, o6}, {o7, o8}}});
    y3.axpy(a, x3); // y = 2*x + y
    if constexpr (TestFixture::isFloatingPoint()) {
        this->expectNear(y3.at({0, 0, 0}), T(2.1));  // 2*1 + 0.1
        this->expectNear(y3.at({1, 1, 1}), T(16.8)); // 2*8 + 0.8
    } else {
        // For integers, small values will be 0
        this->expectNear(y3.at({0, 0, 0}), T(2));  // 2*1 + 0
        this->expectNear(y3.at({1, 1, 1}), T(16)); // 2*8 + 0
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_ShapeMismatch_1Dvs1D_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> x1(NestedData<T>{v1});
    TensorWrapper<T> y1(NestedData<T>{v3, v4});
    EXPECT_THROW(y1.axpy(T(1), x1), std::invalid_argument);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_ShapeMismatch_1Dvs2D_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> x1b(NestedData<T>{v1, v2});
    TensorWrapper<T> y2(NestedData<T>{{v3, v4}, {v5, v6}});
    EXPECT_THROW(y2.axpy(T(1), x1b), std::invalid_argument);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_ShapeMismatch_2Dvs2D_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> x2(NestedData<T>{{v1, v2}});
    TensorWrapper<T> y2b(NestedData<T>{{v3, v4}, {v5, v6}});
    EXPECT_THROW(y2b.axpy(T(1), x2), std::invalid_argument);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_ShapeMismatch_2Dvs3D_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> x2b(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> y3(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_THROW(y3.axpy(T(1), x2b), std::invalid_argument);
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Axpy_ShapeMismatch_3Dvs3D_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> x3(NestedData<T>{{{v1, v2}, {v3, v4}}});
    TensorWrapper<T> y3b(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_THROW(y3b.axpy(T(1), x3), std::invalid_argument);
}

// ============================================================================
// Device Operations
// ============================================================================

TEST_F(TensorWrapperUtilityTest, ToDevice_SameDevice_NoChangeAndNoThrow) {
    TensorWrapper<float> tensor({2, 2}, 1.0f, Device(DeviceType::CPU, 0));
    EXPECT_NO_THROW(tensor.to(Device(DeviceType::CPU, 0)));
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
}

TEST_F(TensorWrapperUtilityTest,
       ToDevice_UnsupportedTransfers_ThrowsRuntimeError) {
    TensorWrapper<float> tensor({1}, 1.0f);
    // CPU to GPU (throws)
    EXPECT_THROW(tensor.to(Device(DeviceType::GPU, 0)), std::runtime_error);
}

// ============================================================================
// Move Assignment Self-Assignment
// ============================================================================

TEST_F(TensorWrapperUtilityTest, MoveAssignment_SelfAssignment) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    TensorWrapper<float>* ptr = &a;
    a = std::move(
        *ptr); // Handle self-assignment through pointer to bypass -Wself-move
    EXPECT_EQ(a.getTotalSize(), 4u);
}

// ============================================================================
// GetTotalSize with Null Data
// ============================================================================

TEST_F(TensorWrapperUtilityTest, GetTotalSize_NullData) {
    TensorWrapper<float> a;
    EXPECT_EQ(a.getTotalSize(), 0u);
}

// ============================================================================
// Scalar Tensor Operations
// ============================================================================

TEST_F(TensorWrapperUtilityTest, ScalarTensorTensorOps_TwoScalars_Work) {
    TensorWrapper<float> a(2.0f);
    TensorWrapper<float> b(3.0f);
    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;
    auto quot = b / a;
    EXPECT_FLOAT_EQ(sum.at({}), 5.0f);
    EXPECT_FLOAT_EQ(diff.at({}), -1.0f);
    EXPECT_FLOAT_EQ(prod.at({}), 6.0f);
    EXPECT_FLOAT_EQ(quot.at({}), 1.5f);
}

TEST_F(TensorWrapperUtilityTest, ScalarTensorTensorOps_AllScalarTypes_Add) {
    // Test operations with different scalar representations
    // 0D scalar
    TensorWrapper<float> s0(2.0f);
    TensorWrapper<float> s0b(3.0f);
    auto r0 = s0 + s0b;
    EXPECT_FLOAT_EQ(r0.at({}), 5.0f);

    // 1D scalar [1]
    TensorWrapper<float> s1(NestedData<float>{2.0f});
    TensorWrapper<float> s1b(NestedData<float>{3.0f});
    auto r1 = s1 + s1b;
    EXPECT_FLOAT_EQ(r1.at({0}), 5.0f);

    // 2D scalar [1, 1]
    TensorWrapper<float> s2(NestedData<float>{{2.0f}});
    TensorWrapper<float> s2b(NestedData<float>{{3.0f}});
    auto r2 = s2 + s2b;
    EXPECT_FLOAT_EQ(r2.at({0, 0}), 5.0f);

    // 3D scalar [1, 1, 1]
    TensorWrapper<float> s3(NestedData<float>{{{2.0f}}});
    TensorWrapper<float> s3b(NestedData<float>{{{3.0f}}});
    auto r3 = s3 + s3b;
    EXPECT_FLOAT_EQ(r3.at({0, 0, 0}), 5.0f);
}
TEST_F(TensorWrapperUtilityTest, ScalarTensorTensorOps_AllScalarTypes_Sub) {
    // Test operations with different scalar representations
    // 0D scalar
    TensorWrapper<float> s0(2.0f);
    TensorWrapper<float> s0b(3.0f);
    auto r0 = s0 - s0b;
    EXPECT_FLOAT_EQ(r0.at({}), -1.0f);

    // 1D scalar [1]
    TensorWrapper<float> s1(NestedData<float>{2.0f});
    TensorWrapper<float> s1b(NestedData<float>{3.0f});
    auto r1 = s1 - s1b;
    EXPECT_FLOAT_EQ(r1.at({0}), -1.0f);

    // 2D scalar [1, 1]
    TensorWrapper<float> s2(NestedData<float>{{2.0f}});
    TensorWrapper<float> s2b(NestedData<float>{{3.0f}});
    auto r2 = s2 - s2b;
    EXPECT_FLOAT_EQ(r2.at({0, 0}), -1.0f);

    // 3D scalar [1, 1, 1]
    TensorWrapper<float> s3(NestedData<float>{{{2.0f}}});
    TensorWrapper<float> s3b(NestedData<float>{{{3.0f}}});
    auto r3 = s3 - s3b;
    EXPECT_FLOAT_EQ(r3.at({0, 0, 0}), -1.0f);
}
TEST_F(TensorWrapperUtilityTest, ScalarTensorTensorOps_AllScalarTypes_Mul) {
    // Test operations with different scalar representations
    // 0D scalar
    TensorWrapper<float> s0(2.0f);
    TensorWrapper<float> s0b(3.0f);
    auto r0 = s0 * s0b;
    EXPECT_FLOAT_EQ(r0.at({}), 6.0f);

    // 1D scalar [1]
    TensorWrapper<float> s1(NestedData<float>{2.0f});
    TensorWrapper<float> s1b(NestedData<float>{3.0f});
    auto r1 = s1 * s1b;
    EXPECT_FLOAT_EQ(r1.at({0}), 6.0f);

    // 2D scalar [1, 1]
    TensorWrapper<float> s2(NestedData<float>{{2.0f}});
    TensorWrapper<float> s2b(NestedData<float>{{3.0f}});
    auto r2 = s2 * s2b;
    EXPECT_FLOAT_EQ(r2.at({0, 0}), 6.0f);

    // 3D scalar [1, 1, 1]
    TensorWrapper<float> s3(NestedData<float>{{{2.0f}}});
    TensorWrapper<float> s3b(NestedData<float>{{{3.0f}}});
    auto r3 = s3 * s3b;
    EXPECT_FLOAT_EQ(r3.at({0, 0, 0}), 6.0f);
}
TEST_F(TensorWrapperUtilityTest, ScalarTensorTensorOps_AllScalarTypes_Div) {
    // Test operations with different scalar representations
    // 0D scalar
    TensorWrapper<float> s0(2.0f);
    TensorWrapper<float> s0b(3.0f);
    auto r0 = s0 / s0b;
    EXPECT_FLOAT_EQ(r0.at({}), 2.0f / 3.0f);

    // 1D scalar [1]
    TensorWrapper<float> s1(NestedData<float>{2.0f});
    TensorWrapper<float> s1b(NestedData<float>{3.0f});
    auto r1 = s1 / s1b;
    EXPECT_FLOAT_EQ(r1.at({0}), 2.0f / 3.0f);

    // 2D scalar [1, 1]
    TensorWrapper<float> s2(NestedData<float>{{2.0f}});
    TensorWrapper<float> s2b(NestedData<float>{{3.0f}});
    auto r2 = s2 / s2b;
    EXPECT_FLOAT_EQ(r2.at({0, 0}), 2.0f / 3.0f);

    // 3D scalar [1, 1, 1]
    TensorWrapper<float> s3(NestedData<float>{{{2.0f}}});
    TensorWrapper<float> s3b(NestedData<float>{{{3.0f}}});
    auto r3 = s3 / s3b;
    EXPECT_FLOAT_EQ(r3.at({0, 0, 0}), 2.0f / 3.0f);
}
