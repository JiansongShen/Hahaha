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
#include "math/ds/TensorShape.h"

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

template <typename T> class TensorWrapperShapeOpTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(TensorWrapperShapeOpTypedTest, NumericTypes);

class TensorWrapperShapeOpTest : public ::testing::Test {};

// ============================================================================
// Reshape - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_1D_To2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3, v4, v5, v6});
    auto r1 = v1_tensor.reshape({2, 3});
    EXPECT_EQ(r1.getShape().size(), 2);
    EXPECT_EQ(r1.getShape()[0], 2);
    EXPECT_EQ(r1.getShape()[1], 3);
    this->expectNear(r1.at({0, 0}), v1);
    this->expectNear(r1.at({1, 2}), v6);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_1D_To3D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> v2_tensor(NestedData<T>{v1, v2, v3, v4, v5, v6, v7, v8});
    auto r2 = v2_tensor.reshape({2, 2, 2});
    EXPECT_EQ(r2.getShape().size(), 3);
    EXPECT_EQ(r2.getShape()[0], 2);
    EXPECT_EQ(r2.getShape()[1], 2);
    EXPECT_EQ(r2.getShape()[2], 2);
    this->expectNear(r2.at({0, 0, 0}), v1);
    this->expectNear(r2.at({1, 1, 1}), v8);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_2D_To1D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto r3 = m2.reshape({4});
    EXPECT_EQ(r3.getShape().size(), 1);
    EXPECT_EQ(r3.getShape()[0], 4);
    this->expectNear(r3.at({0}), v1);
    this->expectNear(r3.at({3}), v4);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_2D_To2D_DifferentShape) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> m2a(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}});
    auto r4 = m2a.reshape({3, 2});
    EXPECT_EQ(r4.getShape().size(), 2);
    EXPECT_EQ(r4.getShape()[0], 3);
    EXPECT_EQ(r4.getShape()[1], 2);
    this->expectNear(r4.at({0, 0}), v1);
    this->expectNear(r4.at({2, 1}), v6);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_2D_To3D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> m2b(NestedData<T>{{v1, v2, v3, v4}, {v5, v6, v7, v8}});
    auto r5 = m2b.reshape({2, 2, 2});
    EXPECT_EQ(r5.getShape().size(), 3);
    EXPECT_EQ(r5.getShape()[0], 2);
    EXPECT_EQ(r5.getShape()[1], 2);
    EXPECT_EQ(r5.getShape()[2], 2);
    this->expectNear(r5.at({0, 0, 0}), v1);
    this->expectNear(r5.at({1, 1, 1}), v8);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_3D_To1D) {
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
    auto r6 = t3.reshape({8});
    EXPECT_EQ(r6.getShape().size(), 1);
    EXPECT_EQ(r6.getShape()[0], 8);
    this->expectNear(r6.at({0}), v1);
    this->expectNear(r6.at({7}), v8);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_3D_To2D) {
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
    auto r7 = t3.reshape({4, 2});
    EXPECT_EQ(r7.getShape().size(), 2);
    EXPECT_EQ(r7.getShape()[0], 4);
    EXPECT_EQ(r7.getShape()[1], 2);
    this->expectNear(r7.at({0, 0}), v1);
    this->expectNear(r7.at({3, 1}), v8);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_3D_To3D_DifferentShape) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T v9 = T(9);
    T v10 = T(10);
    T v11 = T(11);
    T v12 = T(12);
    T v13 = T(13);
    T v14 = T(14);
    T v15 = T(15);
    T v16 = T(16);
    TensorWrapper<T> t3a(NestedData<T>{{{v1, v2, v3, v4}, {v5, v6, v7, v8}},
                                       {{v9, v10, v11, v12}, {v13, v14, v15, v16}}});
    auto r8 = t3a.reshape({2, 4, 2});
    EXPECT_EQ(r8.getShape().size(), 3);
    EXPECT_EQ(r8.getShape()[0], 2);
    EXPECT_EQ(r8.getShape()[1], 4);
    EXPECT_EQ(r8.getShape()[2], 2);
    this->expectNear(r8.at({0, 0, 0}), v1);
    this->expectNear(r8.at({1, 3, 1}), v16);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           Reshape_SizeMismatch_0D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    EXPECT_THROW(s0.reshape({2}), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           Reshape_SizeMismatch_1D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3, v4});
    EXPECT_THROW(v1_tensor.reshape({3}),
                 std::invalid_argument); // Total size 3 != 4
    EXPECT_THROW(v1_tensor.reshape({2, 3}),
                 std::invalid_argument); // Total size 6 != 4
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           Reshape_SizeMismatch_2D_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(m2.reshape({5}), std::invalid_argument);
    EXPECT_THROW(m2.reshape({2, 3}), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           Reshape_SizeMismatch_3D_ThrowsInvalidArgument) {
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
    EXPECT_THROW(t3.reshape({7}), std::invalid_argument);
    EXPECT_THROW(t3.reshape({2, 3}), std::invalid_argument);
    EXPECT_THROW(t3.reshape({2, 2, 3}), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, Reshape_PreservesDataOrder) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    TensorWrapper<T> original(NestedData<T>{v1, v2, v3, v4, v5, v6});

    // Reshape to 2x3
    auto r1 = original.reshape({2, 3});
    this->expectNear(r1.at({0, 0}), v1);
    this->expectNear(r1.at({0, 1}), v2);
    this->expectNear(r1.at({0, 2}), v3);
    this->expectNear(r1.at({1, 0}), v4);
    this->expectNear(r1.at({1, 1}), v5);
    this->expectNear(r1.at({1, 2}), v6);

    // Reshape to 3x2
    auto r2 = original.reshape({3, 2});
    this->expectNear(r2.at({0, 0}), v1);
    this->expectNear(r2.at({0, 1}), v2);
    this->expectNear(r2.at({1, 0}), v3);
    this->expectNear(r2.at({1, 1}), v4);
    this->expectNear(r2.at({2, 0}), v5);
    this->expectNear(r2.at({2, 1}), v6);
}

// ============================================================================
// Broadcast - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_0D_To0D_SameShape) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    auto b00 = s0.broadcastTo(TensorShape({}));
    EXPECT_EQ(b00.getShape().size(), 0);
    this->expectNear(b00.at({}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_0D_To1D) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    auto b01 = s0.broadcastTo(TensorShape({3}));
    EXPECT_EQ(b01.getShape().size(), 1);
    EXPECT_EQ(b01.getShape()[0], 3);
    this->expectNear(b01.at({0}), val);
    this->expectNear(b01.at({2}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_0D_To2D) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    auto b02 = s0.broadcastTo(TensorShape({2, 3}));
    EXPECT_EQ(b02.getShape().size(), 2);
    EXPECT_EQ(b02.getShape()[0], 2);
    EXPECT_EQ(b02.getShape()[1], 3);
    this->expectNear(b02.at({0, 0}), val);
    this->expectNear(b02.at({1, 2}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_0D_To3D) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> s0(val);
    auto b03 = s0.broadcastTo(TensorShape({2, 2, 3}));
    EXPECT_EQ(b03.getShape().size(), 3);
    EXPECT_EQ(b03.getShape()[0], 2);
    EXPECT_EQ(b03.getShape()[1], 2);
    EXPECT_EQ(b03.getShape()[2], 3);
    this->expectNear(b03.at({0, 0, 0}), val);
    this->expectNear(b03.at({1, 1, 2}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_1D_To1D_SameShape) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto b11 = v1_tensor.broadcastTo(TensorShape({3}));
    EXPECT_EQ(b11.getShape().size(), 1);
    EXPECT_EQ(b11.getShape()[0], 3);
    this->expectNear(b11.at({0}), v1);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_1D_To2D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto b12 = v1_tensor.broadcastTo(TensorShape({2, 3}));
    EXPECT_EQ(b12.getShape().size(), 2);
    EXPECT_EQ(b12.getShape()[0], 2);
    EXPECT_EQ(b12.getShape()[1], 3);
    this->expectNear(b12.at({0, 0}), v1);
    this->expectNear(b12.at({1, 0}), v1); // broadcast first dim
    this->expectNear(b12.at({0, 2}), v3);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_1D_To3D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    auto b13 = v1_tensor.broadcastTo(TensorShape({2, 2, 3}));
    EXPECT_EQ(b13.getShape().size(), 3);
    EXPECT_EQ(b13.getShape()[0], 2);
    EXPECT_EQ(b13.getShape()[1], 2);
    EXPECT_EQ(b13.getShape()[2], 3);
    this->expectNear(b13.at({0, 0, 0}), v1);
    this->expectNear(b13.at({1, 1, 0}), v1); // broadcast first two dims
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_2D_To2D_SameShape) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    auto b22 = m2.broadcastTo(TensorShape({2, 2}));
    EXPECT_EQ(b22.getShape().size(), 2);
    EXPECT_EQ(b22.getShape()[0], 2);
    EXPECT_EQ(b22.getShape()[1], 2);
    EXPECT_EQ(b22.getStride().toString(), m2.getStride().toString());
    EXPECT_EQ(b22.getRawData().get(), m2.getRawData().get());
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_2D_To2D_Dim1_Broadcasting) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> m2a(NestedData<T>{{v1, v2, v3}}); // [1, 3]
    auto b22b = m2a.broadcastTo(TensorShape({2, 3}));
    EXPECT_EQ(b22b.getShape().size(), 2);
    EXPECT_EQ(b22b.getShape()[0], 2);
    EXPECT_EQ(b22b.getShape()[1], 3);
    this->expectNear(b22b.at({0, 0}), v1);
    this->expectNear(b22b.at({1, 0}), v1); // broadcast first dim
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_2D_To3D) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2b(NestedData<T>{{v1, v2}, {v3, v4}});
    auto b23 = m2b.broadcastTo(TensorShape({2, 2, 2}));
    EXPECT_EQ(b23.getShape().size(), 3);
    EXPECT_EQ(b23.getShape()[0], 2);
    EXPECT_EQ(b23.getShape()[1], 2);
    EXPECT_EQ(b23.getShape()[2], 2);
    this->expectNear(b23.at({0, 0, 0}), v1);
    this->expectNear(b23.at({1, 0, 0}), v1); // broadcast first dim
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_3D_To3D_SameShape) {
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
    auto b33 = t3.broadcastTo(TensorShape({2, 2, 2}));
    EXPECT_EQ(b33.getShape().size(), 3);
    EXPECT_EQ(b33.getShape()[0], 2);
    EXPECT_EQ(b33.getShape()[1], 2);
    EXPECT_EQ(b33.getShape()[2], 2);
    EXPECT_EQ(b33.getStride().toString(), t3.getStride().toString());
    EXPECT_EQ(b33.getRawData().get(), t3.getRawData().get());
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_ScalarTypes_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorShape target({2, 2, 2});
    TensorWrapper<T> s0(val);
    auto b0 = s0.broadcastTo(target);
    this->expectNear(b0.at({0, 0, 0}), val);
    this->expectNear(b0.at({1, 1, 1}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_ScalarTypes_1D_SingleElement) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorShape target({2, 2, 2});
    TensorWrapper<T> s1(NestedData<T>{val});
    auto b1 = s1.broadcastTo(target);
    this->expectNear(b1.at({0, 0, 0}), val);
    this->expectNear(b1.at({1, 1, 1}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_ScalarTypes_2D_SingleElement) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorShape target({2, 2, 2});
    TensorWrapper<T> s2(NestedData<T>{{val}});
    auto b2 = s2.broadcastTo(target);
    this->expectNear(b2.at({0, 0, 0}), val);
    this->expectNear(b2.at({1, 1, 1}), val);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest, BroadcastTo_ScalarTypes_3D_SingleElement) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorShape target({2, 2, 2});
    TensorWrapper<T> s3(NestedData<T>{{{val}}});
    auto b3 = s3.broadcastTo(target);
    this->expectNear(b3.at({0, 0, 0}), val);
    this->expectNear(b3.at({1, 1, 1}), val);
}

TEST_F(TensorWrapperShapeOpTest, BroadcastTo_SameShape_ReturnsViewWithSameStride) {
    TensorWrapper<int> src(NestedData<int>{{1, 2}, {3, 4}}); // shape (2,2)
    auto view = src.broadcastTo(TensorShape({2, 2}));

    EXPECT_EQ(view.getShape().size(), 2);
    EXPECT_EQ(view.getShape()[0], 2);
    EXPECT_EQ(view.getShape()[1], 2);
    EXPECT_EQ(view.getStride().toString(), src.getStride().toString());
    EXPECT_EQ(view.getRawData().get(), src.getRawData().get());
}

TEST_F(TensorWrapperShapeOpTest, BroadcastTo_PrefixDim_InsertsZeroStride) {
    TensorWrapper<int> src(NestedData<int>{1, 2, 3}); // shape (3)
    auto view = src.broadcastTo(TensorShape({2, 3})); // view shape (2,3)
    EXPECT_EQ(view.getShape().size(), 2);
    EXPECT_EQ(view.getShape()[0], 2);
    EXPECT_EQ(view.getShape()[1], 3);

    // prefix dim stride is 0, last dim stride is original 1
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 0);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1);
    EXPECT_EQ(view.getRawData().get(), src.getRawData().get());
}

TEST_F(TensorWrapperShapeOpTest, BroadcastTo_DimOneGetsZeroStride) {
    TensorWrapper<int> src(NestedData<int>{{1, 2, 3}}); // shape (1,3)
    auto view = src.broadcastTo(TensorShape({2, 3}));   // (2,3)
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 0);
    EXPECT_EQ(view.getStride().getStrideVec()[1], src.getStride().getStrideVec()[1]);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           BroadcastTo_ErrorHandling_1D_ToLowerRank_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> v1_tensor(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(v1_tensor.broadcastTo(TensorShape({})), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           BroadcastTo_ErrorHandling_1D_IncompatibleSize_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    TensorWrapper<T> v2_tensor(NestedData<T>{v1, v2});
    EXPECT_THROW(v2_tensor.broadcastTo(TensorShape({3})), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           BroadcastTo_ErrorHandling_2D_ToLowerRank_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(m2.broadcastTo(TensorShape({3})), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           BroadcastTo_ErrorHandling_2D_IncompatibleShape_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> m2b(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(m2b.broadcastTo(TensorShape({2, 3})), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           BroadcastTo_ErrorHandling_3D_ToLowerRank_ThrowsInvalidArgument) {
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
    EXPECT_THROW(t3.broadcastTo(TensorShape({2, 2})), std::invalid_argument);
}

TYPED_TEST(TensorWrapperShapeOpTypedTest,
           BroadcastTo_ErrorHandling_3D_IncompatibleShape_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> t3b(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_THROW(t3b.broadcastTo(TensorShape({2, 2, 3})), std::invalid_argument);
}

TYPED_TEST(
    TensorWrapperShapeOpTypedTest,
    BroadcastTo_ErrorHandling_2D_IncompatibleBroadcast_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    TensorWrapper<T> m2b(NestedData<T>{{v1, v2}}); // [1, 2]
    EXPECT_THROW(m2b.broadcastTo(TensorShape({2, 3})),
                 std::invalid_argument); // [1, 2] can't broadcast to [2, 3]
}
