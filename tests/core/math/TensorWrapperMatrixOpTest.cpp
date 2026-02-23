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

template <typename T> class TensorWrapperMatrixOpTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
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

TYPED_TEST_SUITE(TensorWrapperMatrixOpTypedTest, NumericTypes);

class TensorWrapperMatrixOpTest : public ::testing::Test {};

// ============================================================================
// Matrix Multiplication - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Matmul_2x2_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> m1(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> m2(NestedData<T>{{v5, v6}, {v7, v8}});
    auto r = m1.matmul(m2);
    T expected_00 = T(19); // 1*5 + 2*7 = 5 + 14 = 19
    this->expectNear(r.at({0, 0}), expected_00);
    T expected_01 = T(22); // 1*6 + 2*8 = 6 + 16 = 22
    this->expectNear(r.at({0, 1}), expected_01);
    T expected_10 = T(43); // 3*5 + 4*7 = 15 + 28 = 43
    this->expectNear(r.at({1, 0}), expected_10);
    T expected_11 = T(50); // 3*6 + 4*8 = 18 + 32 = 50
    this->expectNear(r.at({1, 1}), expected_11);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Matmul_1x1_Matrix) {
    using T = TestFixture::Type;
    T v5 = T(5);
    T v3 = T(3);
    TensorWrapper<T> m1x1_a(NestedData<T>{{v5}});
    TensorWrapper<T> m1x1_b(NestedData<T>{{v3}});
    auto r1x1 = m1x1_a.matmul(m1x1_b);
    T expected = T(15);
    this->expectNear(r1x1.at({0, 0}), expected);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Matmul_2x3_3x2_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2x3(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}});
    TensorWrapper<T> m3x2(NestedData<T>{{v1, v2}, {v3, v4}, {v5, v6}});
    auto r2x2 = m2x3.matmul(m3x2);
    EXPECT_EQ(r2x2.getShapeVecRef().size(), 2);
    EXPECT_EQ(r2x2.getShapeVecRef()[0], 2);
    EXPECT_EQ(r2x2.getShapeVecRef()[1], 2);
    T expected_00 = T(22); // 1*1 + 2*3 + 3*5 = 1+6+15 = 22
    this->expectNear(r2x2.at({0, 0}), expected_00);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Matmul_ErrorHandling_0D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T val = TestFixture::isFloatingPoint() ? T(5.0) : T(5);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> s0(val);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(s0.matmul(m2), std::invalid_argument);
    EXPECT_THROW(m2.matmul(s0), std::invalid_argument);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Matmul_ErrorHandling_1D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(v1_tensor.matmul(m2), std::invalid_argument);
    EXPECT_THROW(m2.matmul(v1_tensor), std::invalid_argument);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Matmul_ErrorHandling_2D_IncompatibleDims_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2a(NestedData<T>{{v1, v2}, {v3, v4}});         // 2x2
    TensorWrapper<T> m2b(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    EXPECT_NO_THROW(m2a.matmul(m2b)); // (2x2) @ (2x3) -> inner dims 2 == 2

    TensorWrapper<T> m2c(NestedData<T>{{v1, v2}, {v3, v4}});           // 2x2
    TensorWrapper<T> m2d(NestedData<T>{{v1, v2}, {v3, v4}, {v5, v6}}); // 3x2
    EXPECT_THROW(m2c.matmul(m2d),
                 std::invalid_argument); // (2x2) @ (3x2) -> inner dims 2 != 3
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Matmul_ErrorHandling_3D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> t3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(t3.matmul(m2), std::invalid_argument);
    EXPECT_THROW(m2.matmul(t3), std::invalid_argument);
}

// ============================================================================
// Transpose - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Transpose_2x3_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> tensor_orig(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    auto tensor_transposed = tensor_orig.transpose();                        // 3x2
    EXPECT_EQ(tensor_transposed.getShapeVecRef()[0], 3);
    EXPECT_EQ(tensor_transposed.getShapeVecRef()[1], 2);
    this->expectNear(tensor_transposed.at({0, 0}), v1);
    this->expectNear(tensor_transposed.at({0, 1}), v4);
    this->expectNear(tensor_transposed.at({2, 1}), v6);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Transpose_1x1_Matrix) {
    using T = TestFixture::Type;
    T val = T(5);
    TensorWrapper<T> m1x1(NestedData<T>{{val}});
    auto r1x1 = m1x1.transpose();
    EXPECT_EQ(r1x1.getShapeVecRef()[0], 1);
    EXPECT_EQ(r1x1.getShapeVecRef()[1], 1);
    this->expectNear(r1x1.at({0, 0}), val);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Transpose_3x2_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m3x2(NestedData<T>{{v1, v2}, {v3, v4}, {v5, v6}});
    auto r2x3 = m3x2.transpose();
    EXPECT_EQ(r2x3.getShapeVecRef()[0], 2);
    EXPECT_EQ(r2x3.getShapeVecRef()[1], 3);
    this->expectNear(r2x3.at({0, 0}), v1);
    this->expectNear(r2x3.at({1, 2}), v6);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Transpose_ErrorHandling_0D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T val = TestFixture::isFloatingPoint() ? T(5.0) : T(5);
    TensorWrapper<T> s0(val);
    EXPECT_THROW(s0.transpose(), std::invalid_argument);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Transpose_ErrorHandling_1D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(v1_tensor.transpose(), std::invalid_argument);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest,
           Transpose_ErrorHandling_3D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> t3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_THROW(t3.transpose(), std::invalid_argument);
}

TYPED_TEST(TensorWrapperMatrixOpTypedTest, Transpose_Transpose_Identity) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> original(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}});
    auto transposed = original.transpose();
    auto double_transposed = transposed.transpose();
    EXPECT_EQ(double_transposed.getShapeVecRef(), original.getShapeVecRef());
    this->expectNear(double_transposed.at({0, 0}), original.at({0, 0}));
    this->expectNear(double_transposed.at({1, 2}), original.at({1, 2}));
}
