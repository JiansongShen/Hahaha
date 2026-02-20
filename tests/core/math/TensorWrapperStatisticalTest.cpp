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
class TensorWrapperStatisticalTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(TensorWrapperStatisticalTypedTest, NumericTypes);

class TensorWrapperStatisticalTest : public ::testing::Test {};

// ============================================================================
// Sum Operation - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> t1(NestedData<T>{v1, v2, v3});
    T expected = T(6);
    this->expectNear(t1.sum(), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    this->expectNear(s0.sum(), val);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    T expected = T(10);
    this->expectNear(m2.sum(), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_3D_Tensor) {
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
    T expected = T(36);
    this->expectNear(t3.sum(), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_EmptyTensor_ReturnsZero) {
    using T = TestFixture::Type;
    TensorWrapper<T> tensor;
    T expected = T(0);
    this->expectNear(tensor.sum(), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_WithAxes_EmptyAxes_ReturnsClone) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v100 = T(100);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}});
    auto res = tensor.sum({});
    EXPECT_EQ(res.getShape(), tensor.getShape());
    this->expectNear(res.at({0, 0}), v1);
    tensor.at({0, 0}) = v100;
    this->expectNear(res.at({0, 0}), v1); // clone, not view
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_InvalidAxis_2D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}}); // rank 2
    EXPECT_THROW(tensor.sum({2}), std::invalid_argument);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_InvalidAxis_0D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    EXPECT_THROW(s0.sum({0}), std::invalid_argument); // 0D has no axes
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_InvalidAxis_1D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(v1_tensor.sum({1}),
                 std::invalid_argument); // 1D has only axis 0
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_InvalidAxis_3D_ThrowsInvalidArgument) {
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
    EXPECT_THROW(t3.sum({3}), std::invalid_argument); // 3D has only axes 0,1,2
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_1D_Axis0_ReducesToScalar) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto sum1 = v1_tensor.sum({0});
    EXPECT_EQ(sum1.getDimensions(), 0); // Becomes scalar
    T expected = T(6);
    this->expectNear(sum1.at({}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_2D_Axis0_ReducesCorrectly) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    auto sum0 = m2.sum({0});
    EXPECT_EQ(sum0.getShape().size(), 1);
    EXPECT_EQ(sum0.getShape()[0], 3);
    T expected_0 = T(5); // 1+4
    T expected_2 = T(9); // 3+6
    this->expectNear(sum0.at({0}), expected_0);
    this->expectNear(sum0.at({2}), expected_2);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_2D_Axis1_ReducesCorrectly) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    auto sum1_2d = m2.sum({1});
    EXPECT_EQ(sum1_2d.getShape().size(), 1);
    EXPECT_EQ(sum1_2d.getShape()[0], 2);
    T expected_0 = T(6);  // 1+2+3
    T expected_1 = T(15); // 4+5+6
    this->expectNear(sum1_2d.at({0}), expected_0);
    this->expectNear(sum1_2d.at({1}), expected_1);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_3D_Axis0_ReducesCorrectly) {
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
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}}); // 2x2x2
    auto sum0_3d = t3.sum({0});
    EXPECT_EQ(sum0_3d.getShape().size(), 2);
    EXPECT_EQ(sum0_3d.getShape()[0], 2);
    EXPECT_EQ(sum0_3d.getShape()[1], 2);
    T expected_00 = T(6);  // 1+5
    T expected_11 = T(12); // 4+8
    this->expectNear(sum0_3d.at({0, 0}), expected_00);
    this->expectNear(sum0_3d.at({1, 1}), expected_11);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_3D_Axis1_ReducesCorrectly) {
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
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}}); // 2x2x2
    auto sum1_3d = t3.sum({1});
    EXPECT_EQ(sum1_3d.getShape().size(), 2);
    EXPECT_EQ(sum1_3d.getShape()[0], 2);
    EXPECT_EQ(sum1_3d.getShape()[1], 2);
    T expected_00 = T(4);  // 1+3
    T expected_11 = T(14); // 6+8
    this->expectNear(sum1_3d.at({0, 0}), expected_00);
    this->expectNear(sum1_3d.at({1, 1}), expected_11);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_3D_Axis2_ReducesCorrectly) {
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
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}}); // 2x2x2
    auto sum2_3d = t3.sum({2});
    EXPECT_EQ(sum2_3d.getShape().size(), 2);
    EXPECT_EQ(sum2_3d.getShape()[0], 2);
    EXPECT_EQ(sum2_3d.getShape()[1], 2);
    T expected_00 = T(3);  // 1+2
    T expected_11 = T(15); // 7+8
    this->expectNear(sum2_3d.at({0, 0}), expected_00);
    this->expectNear(sum2_3d.at({1, 1}), expected_11);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_3D_MultipleAxes_ReducesCorrectly) {
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
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}}); // 2x2x2
    auto sum01_3d = t3.sum({0, 1});
    EXPECT_EQ(sum01_3d.getShape().size(), 1);
    EXPECT_EQ(sum01_3d.getShape()[0], 2);
    T expected_0 = T(16); // Sum over first two axes for first element
    T expected_1 = T(20); // Sum over first two axes for second element
    this->expectNear(sum01_3d.at({0}), expected_0);
    this->expectNear(sum01_3d.at({1}), expected_1);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_3D_AllAxes_ReducesToScalar) {
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
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}}); // 2x2x2
    auto sum_all_3d = t3.sum({0, 1, 2});
    EXPECT_EQ(sum_all_3d.getDimensions(), 0); // Becomes scalar
    T expected = T(36);
    this->expectNear(sum_all_3d.at({}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_KeepDims_1D_KeepsDimension) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto sum1_keep = v1_tensor.sum({0}, true);
    // the result is a scalar
    EXPECT_EQ(sum1_keep.getShape().size(), 0);
    T expected = T(6);
    this->expectNear(sum1_keep.at({}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_KeepDims_2D_Axis0_KeepsDimension) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    auto sum0_keep = m2.sum({0}, true);
    EXPECT_EQ(sum0_keep.getShape().size(), 2);
    EXPECT_EQ(sum0_keep.getShape()[0], 1);
    EXPECT_EQ(sum0_keep.getShape()[1], 3);
    T expected = T(5); // 1+4
    this->expectNear(sum0_keep.at({0, 0}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_KeepDims_2D_Axis1_KeepsDimension) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    auto sum1_keep_2d = m2.sum({1}, true);
    EXPECT_EQ(sum1_keep_2d.getShape().size(), 2);
    EXPECT_EQ(sum1_keep_2d.getShape()[0], 2);
    EXPECT_EQ(sum1_keep_2d.getShape()[1], 1);
    T expected = T(6); // 1+2+3
    this->expectNear(sum1_keep_2d.at({0, 0}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_KeepDims_3D_Axis0_KeepsDimension) {
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
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}}); // 2x2x2
    auto sum0_keep_3d = t3.sum({0}, true);
    EXPECT_EQ(sum0_keep_3d.getShape().size(), 3);
    EXPECT_EQ(sum0_keep_3d.getShape()[0], 1);
    EXPECT_EQ(sum0_keep_3d.getShape()[1], 2);
    EXPECT_EQ(sum0_keep_3d.getShape()[2], 2);
    T expected = T(6); // 1+5
    this->expectNear(sum0_keep_3d.at({0, 0, 0}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_WithAxes_KeepDimsFalse_ReducesRank) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}}); // 2x2
    auto sum0 = tensor.sum({0}, false);                         // Should be {2}
    EXPECT_EQ(sum0.getShape().size(), 1);
    EXPECT_EQ(sum0.getShape()[0], 2);

    auto sum1 = tensor.sum({1}, false); // Should be {2}
    EXPECT_EQ(sum1.getShape().size(), 1);
    EXPECT_EQ(sum1.getShape()[0], 2);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_0D_EmptyAxes_ReturnsSameValue) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    auto r0 = s0.sum({});
    this->expectNear(r0.at({}), val);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_1D_EmptyAxes_ReturnsClone) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto r1_empty = v1_tensor.sum({});
    EXPECT_EQ(r1_empty.getShape(), v1_tensor.getShape());
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_2D_EmptyAxes_ReturnsClone) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto r2_empty = m2.sum({});
    EXPECT_EQ(r2_empty.getShape(), m2.getShape());
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_2D_AllAxes_ReducesToScalar) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto r2_all = m2.sum({0, 1});
    EXPECT_EQ(r2_all.getDimensions(), 0);
    T expected = T(10);
    this->expectNear(r2_all.at({}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_3D_EmptyAxes_ReturnsClone) {
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
    auto r3_empty = t3.sum({});
    EXPECT_EQ(r3_empty.getShape(), t3.getShape());
}

TYPED_TEST(TensorWrapperStatisticalTypedTest,
           Sum_3D_MultipleAxes_01_ReducesCorrectly) {
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
    auto r3_01 = t3.sum({0, 1});
    EXPECT_EQ(r3_01.getShape().size(), 1);
    EXPECT_EQ(r3_01.getShape()[0], 2);
    T expected = T(16);
    this->expectNear(r3_01.at({0}), expected);
}

TYPED_TEST(TensorWrapperStatisticalTypedTest, Sum_3D_AllAxes_ReducesToScalar) {
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
    auto r3_all = t3.sum({0, 1, 2});
    EXPECT_EQ(r3_all.getDimensions(), 0);
    T expected = T(36);
    this->expectNear(r3_all.at({}), expected);
}
