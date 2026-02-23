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

template <typename T> class TensorScalarOpTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get test value
    static constexpr T testValue() {
        if constexpr (isFloatingPoint()) {
            return T(10.0);
        } else {
            return T(10);
        }
    }

    // Helper to get scalar value for operations
    static constexpr T scalarValue(T multiplier = T(1)) {
        if constexpr (isFloatingPoint()) {
            return T(5.0) * multiplier;
        } else {
            return T(5) * multiplier;
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

TYPED_TEST_SUITE(TensorScalarOpTypedTest, NumericTypes);

class TensorScalarOpTest : public ::testing::Test {};

// ============================================================================
// Scalar Operations Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_0D_Scalar_Add) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar = TestFixture::scalarValue();
    Tensor<T> a(val);
    auto b = a + scalar;
    auto c = scalar + a;
    T expected = val + scalar;
    this->expectNear(b.data()->at({}), expected);
    this->expectNear(c.data()->at({}), expected);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_0D_Scalar_Subtract) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar_2 = TestFixture::scalarValue(T(0.4)); // 2 for integers, 2.0 for floats
    T scalar_20 = TestFixture::testValue() * T(2);
    Tensor<T> a(val);
    auto d = a - scalar_2;
    auto e = scalar_20 - a;
    T expected_d = val - scalar_2;
    T expected_e = scalar_20 - val;
    this->expectNear(d.data()->at({}), expected_d);
    this->expectNear(e.data()->at({}), expected_e);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_0D_Scalar_Multiply) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar = T(2);
    Tensor<T> a(val);
    auto f = a * scalar;
    auto g = scalar * a;
    T expected = val * scalar;
    this->expectNear(f.data()->at({}), expected);
    this->expectNear(g.data()->at({}), expected);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_1D_SingleElement_Add) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar = TestFixture::scalarValue();
    Tensor<T> s1(NestedData<T>{val});
    auto b1 = s1 + scalar;
    T expected = val + scalar;
    this->expectNear(b1.data()->at({0}), expected);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_2D_SingleElement_Add) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar = TestFixture::scalarValue();
    Tensor<T> s2(NestedData<T>{{val}});
    auto b2 = s2 + scalar;
    T expected = val + scalar;
    this->expectNear(b2.data()->at({0, 0}), expected);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_3D_SingleElement_Add) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar = TestFixture::scalarValue();
    Tensor<T> s3(NestedData<T>{{{val}}});
    auto b3 = s3 + scalar;
    T expected = val + scalar;
    this->expectNear(b3.data()->at({0, 0, 0}), expected);
}

// ============================================================================
// Scalar Types Distinction Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(TensorScalarOpTypedTest, ScalarTypes_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::scalarValue();
    Tensor<T> s0(val);
    EXPECT_EQ(s0.getShapeVecRef().size(), 0);
    EXPECT_EQ(s0.getTotalSize(), 1);
    this->expectNear(s0.at({}), val);
    this->expectNear(s0.sum(), val);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarTypes_1D_SingleElement) {
    using T = TestFixture::Type;
    T val = TestFixture::scalarValue();
    Tensor<T> s1(NestedData<T>{val});
    EXPECT_EQ(s1.getShapeVecRef().size(), 1);
    EXPECT_EQ(s1.getShapeVecRef()[0], 1);
    EXPECT_EQ(s1.getTotalSize(), 1);
    this->expectNear(s1.at({0}), val);
    this->expectNear(s1.sum(), val);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarTypes_2D_SingleElement) {
    using T = TestFixture::Type;
    T val = TestFixture::scalarValue();
    Tensor<T> s2(NestedData<T>{{val}});
    EXPECT_EQ(s2.getShapeVecRef().size(), 2);
    EXPECT_EQ(s2.getShapeVecRef()[0], 1);
    EXPECT_EQ(s2.getShapeVecRef()[1], 1);
    EXPECT_EQ(s2.getTotalSize(), 1);
    this->expectNear(s2.at({0, 0}), val);
    this->expectNear(s2.sum(), val);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarTypes_3D_SingleElement) {
    using T = TestFixture::Type;
    T val = TestFixture::scalarValue();
    Tensor<T> s3(NestedData<T>{{{val}}});
    EXPECT_EQ(s3.getShapeVecRef().size(), 3);
    EXPECT_EQ(s3.getShapeVecRef()[0], 1);
    EXPECT_EQ(s3.getShapeVecRef()[1], 1);
    EXPECT_EQ(s3.getShapeVecRef()[2], 1);
    EXPECT_EQ(s3.getTotalSize(), 1);
    this->expectNear(s3.at({0, 0, 0}), val);
    this->expectNear(s3.sum(), val);
}

// ============================================================================
// Division Tests - All Types
// ============================================================================

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_0D_Scalar_Divide) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar_divisor;
    T scalar_dividend;

    if constexpr (TestFixture::isFloatingPoint()) {
        scalar_divisor = T(2.0);
        scalar_dividend = T(20.0);
    } else {
        scalar_divisor = T(2);
        scalar_dividend = T(20);
    }

    Tensor<T> a(val);
    auto h = a / scalar_divisor;
    auto i = scalar_dividend / a;
    T expected_h = val / scalar_divisor;
    T expected_i = scalar_dividend / val;
    this->expectNear(h.data()->at({}), expected_h);
    this->expectNear(i.data()->at({}), expected_i);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_1D_SingleElement_Divide) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar_divisor;
    if constexpr (TestFixture::isFloatingPoint()) {
        scalar_divisor = T(2.0);
    } else {
        scalar_divisor = T(2);
    }
    Tensor<T> s1(NestedData<T>{val});
    auto b1 = s1 / scalar_divisor;
    T expected = val / scalar_divisor;
    this->expectNear(b1.data()->at({0}), expected);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_2D_SingleElement_Divide) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar_divisor;
    if constexpr (TestFixture::isFloatingPoint()) {
        scalar_divisor = T(2.0);
    } else {
        scalar_divisor = T(2);
    }
    Tensor<T> s2(NestedData<T>{{val}});
    auto b2 = s2 / scalar_divisor;
    T expected = val / scalar_divisor;
    this->expectNear(b2.data()->at({0, 0}), expected);
}

TYPED_TEST(TensorScalarOpTypedTest, ScalarOps_3D_SingleElement_Divide) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T scalar_divisor;
    if constexpr (TestFixture::isFloatingPoint()) {
        scalar_divisor = T(2.0);
    } else {
        scalar_divisor = T(2);
    }
    Tensor<T> s3(NestedData<T>{{{val}}});
    auto b3 = s3 / scalar_divisor;
    T expected = val / scalar_divisor;
    this->expectNear(b3.data()->at({0, 0, 0}), expected);
}
