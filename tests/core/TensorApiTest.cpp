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

#include <gtest/gtest.h>

#include "../../core/include/public/Tensor.h"
#include "backend/gpu/GPUDevice.h"

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

using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <class T> class TensorApiTest : public testing::Test {};

TYPED_TEST_SUITE(TensorApiTest, NumericTypes);

TYPED_TEST(TensorApiTest, BuildFromVector_Creates1DTensor) {
    auto t = Tensor<TypeParam>::buildFromVector({1, 2, 3});
    EXPECT_EQ(t.getShape().size(), 1);
    EXPECT_EQ(t.getShape()[0], 3);
    EXPECT_EQ(t.at({0}), TypeParam(1));
    EXPECT_EQ(t.at({2}), TypeParam(3));
}

TYPED_TEST(TensorApiTest, Data_ReturnsSharedPtr) {
    Tensor<TypeParam> t(3);
    auto data = t.data();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->getTotalSize(), TypeParam(1));
    EXPECT_EQ(data->at({}), TypeParam(3));
}

TYPED_TEST(TensorApiTest, Grad_WhenNoBackward_ReturnsNullptr) {
    Tensor<TypeParam> t(3);
    EXPECT_EQ(t.grad(), nullptr);
}

TYPED_TEST(TensorApiTest, Clear_ResetsUnderlyingData) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2}, {3, 4}});
    t.clear();
    EXPECT_EQ(t.at({0, 0}), TypeParam(0));
    EXPECT_EQ(t.at({1, 1}), TypeParam(0));
}

TYPED_TEST(TensorApiTest, To_Gpu_ThrowsRuntimeError) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2}, {3, 4}});
    // TODO: Test success or not
}

TYPED_TEST(TensorApiTest, GetTotalSize_MatchesUnderlyingWrapper) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2}, {3, 4}});
    EXPECT_EQ(t.getTotalSize(), 4);
}

TYPED_TEST(TensorApiTest, ClearGrad_NoGrad_NoThrow) {
    Tensor<TypeParam> t(1);
    EXPECT_NO_THROW(t.clearGrad());
}

TYPED_TEST(TensorApiTest, GetComputeNodeAndSetComputeNode_Works) {
    Tensor<TypeParam> a(1);
    Tensor<TypeParam> b(2);
    auto nodeA = a.getComputeNode();
    auto nodeB = b.getComputeNode();
    ASSERT_NE(nodeA, nullptr);
    ASSERT_NE(nodeB, nullptr);

    a.setComputeNode(nodeB);
    EXPECT_EQ(a.at({}), TypeParam(2));

    // Restore for cleanliness
    a.setComputeNode(nodeA);
    EXPECT_EQ(a.at({}), TypeParam(1));
}

TYPED_TEST(TensorApiTest, RequiresGrad_FlagIsStoredOnNode) {
    using T = TypeParam;
    Tensor<T> t(T(1.0));
    EXPECT_FALSE(t.getRequiresGrad());
    t.setRequiresGrad(true);
    EXPECT_TRUE(t.getRequiresGrad());
}

TYPED_TEST(TensorApiTest, Sum_ForwardsToTensorWrapper) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2}, {3, 4}});
    TypeParam expected =
        TypeParam(1) + TypeParam(2) + TypeParam(3) + TypeParam(4);
    EXPECT_EQ(t.sum(), expected);
}

// ============================================================================
// Dimension-specific tests: 0D to 3D API tests
// ============================================================================

// ============================================================================
// GetTotalSize Tests - All Dimensions
// ============================================================================

TYPED_TEST(TensorApiTest, GetTotalSize_0D_Scalar) {
    Tensor<TypeParam> s0(TypeParam(5));
    EXPECT_EQ(s0.getTotalSize(), 1);
}

TYPED_TEST(TensorApiTest, GetTotalSize_1D_Vector) {
    Tensor<TypeParam> t1D(NestedData<TypeParam>{1, 2, 3});
    EXPECT_EQ(t1D.getTotalSize(), 3);
}

TYPED_TEST(TensorApiTest, GetTotalSize_2D_Matrix) {
    Tensor<TypeParam> t2D(NestedData<TypeParam>{{1, 2}, {3, 4}});
    EXPECT_EQ(t2D.getTotalSize(), 4);
}

TYPED_TEST(TensorApiTest, GetTotalSize_3D_Tensor) {
    Tensor<TypeParam> t3D(
        NestedData<TypeParam>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(t3D.getTotalSize(), 8);
}

// ============================================================================
// Clear Tests - All Dimensions
// ============================================================================

TYPED_TEST(TensorApiTest, Clear_0D_Scalar) {
    Tensor<TypeParam> s0(TypeParam(5));
    s0.clear();
    EXPECT_EQ(s0.at({}), TypeParam(0));
}

TYPED_TEST(TensorApiTest, Clear_1D_Vector) {
    Tensor<TypeParam> t1D(NestedData<TypeParam>{1, 2, 3});
    t1D.clear();
    EXPECT_EQ(t1D.at({0}), TypeParam(0));
    EXPECT_EQ(t1D.at({2}), TypeParam(0));
}

TYPED_TEST(TensorApiTest, Clear_2D_Matrix) {
    Tensor<TypeParam> t2D(NestedData<TypeParam>{{1, 2}, {3, 4}});
    t2D.clear();
    EXPECT_EQ(t2D.at({0, 0}), TypeParam(0));
    EXPECT_EQ(t2D.at({1, 1}), TypeParam(0));
}

TYPED_TEST(TensorApiTest, Clear_3D_Tensor) {
    Tensor<TypeParam> t3D(
        NestedData<TypeParam>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    t3D.clear();
    EXPECT_EQ(t3D.at({0, 0, 0}), TypeParam(0));
    EXPECT_EQ(t3D.at({1, 1, 1}), TypeParam(0));
}

// ============================================================================
// Sum Tests - All Dimensions
// ============================================================================

TYPED_TEST(TensorApiTest, Sum_0D_Scalar) {
    Tensor<TypeParam> s0(TypeParam(5));
    EXPECT_EQ(s0.sum(), TypeParam(5));
}

TYPED_TEST(TensorApiTest, Sum_1D_Vector) {
    Tensor<TypeParam> t1D(NestedData<TypeParam>{1, 2, 3});
    TypeParam expected = TypeParam(1) + TypeParam(2) + TypeParam(3);
    EXPECT_EQ(t1D.sum(), expected);
}

TYPED_TEST(TensorApiTest, Sum_2D_Matrix) {
    Tensor<TypeParam> t2D(NestedData<TypeParam>{{1, 2}, {3, 4}});
    TypeParam expected =
        TypeParam(1) + TypeParam(2) + TypeParam(3) + TypeParam(4);
    EXPECT_EQ(t2D.sum(), expected);
}

TYPED_TEST(TensorApiTest, Sum_3D_Tensor) {
    Tensor<TypeParam> t3D(
        NestedData<TypeParam>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    TypeParam expected = TypeParam(1) + TypeParam(2) + TypeParam(3)
        + TypeParam(4) + TypeParam(5) + TypeParam(6) + TypeParam(7)
        + TypeParam(8);
    EXPECT_EQ(t3D.sum(), expected);
}
