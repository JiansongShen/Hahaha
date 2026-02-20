// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
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

// =============================================================================
// Tests for Tensor public view/inspection APIs:
//   getDimensions(), getStride(),
//   narrow(), select(), sliceDim(), slice(), broadcastTo()
//
// Strategy:
//  - Typed tests (all numeric types) verify shape / dimension contracts.
//  - Float-only fixture tests verify element values exhaustively.
//  - Error-handling tests verify the correct exception types are thrown.
// =============================================================================

#include <gtest/gtest.h>
#include <stdexcept>

#include "common/definitions.h"
#include "math/slice_setting.h"
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

using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

// =============================================================================
// Typed test fixture (all numeric types)
// =============================================================================

template <typename T> class TensorViewOpTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE(TensorViewOpTypedTest, NumericTypes);

// =============================================================================
// getDimensions() – Typed Tests
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, GetDimensions_1D) {
    Tensor<TypeParam> t(NestedData<TypeParam>{1, 2, 3});
    EXPECT_EQ(t.getDimensions(), 1u);
}

TYPED_TEST(TensorViewOpTypedTest, GetDimensions_2D) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2}, {3, 4}});
    EXPECT_EQ(t.getDimensions(), 2u);
}

TYPED_TEST(TensorViewOpTypedTest, GetDimensions_3D) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(t.getDimensions(), 3u);
}

// =============================================================================
// getStride() – Typed Tests
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, GetStride_1D_IsOne) {
    Tensor<TypeParam> t(NestedData<TypeParam>{1, 2, 3, 4});
    ASSERT_EQ(t.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(t.getStride().getStrideVec()[0], 1u);
}

TYPED_TEST(TensorViewOpTypedTest, GetStride_2D_RowMajor) {
    // shape (2,4): row stride = 4, col stride = 1
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2, 3, 4}, {5, 6, 7, 8}});
    ASSERT_EQ(t.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(t.getStride().getStrideVec()[0], 4u);
    EXPECT_EQ(t.getStride().getStrideVec()[1], 1u);
}

// =============================================================================
// narrow() – Typed Tests (shape contract)
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, Narrow_1D_Shape) {
    // [0..4], narrow(0,1,3) => shape (3,)
    Tensor<TypeParam> t(NestedData<TypeParam>{0, 1, 2, 3, 4});
    auto v = t.narrow(0, 1, 3);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 3u);
    EXPECT_EQ(v.getTotalSize(), 3u);
}

TYPED_TEST(TensorViewOpTypedTest, Narrow_2D_AlongDim0_Shape) {
    // shape (3,3), narrow(0,1,2) => shape (2,3)
    Tensor<TypeParam> t(NestedData<TypeParam>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto v = t.narrow(0, 1, 2);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 3u);
    EXPECT_EQ(v.getTotalSize(), 6u);
}

TYPED_TEST(TensorViewOpTypedTest, Narrow_2D_AlongDim1_Shape) {
    // shape (2,4), narrow(1,1,2) => shape (2,2)
    Tensor<TypeParam> t(NestedData<TypeParam>{{0, 1, 2, 3}, {4, 5, 6, 7}});
    auto v = t.narrow(1, 1, 2);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 2u);
    EXPECT_EQ(v.getTotalSize(), 4u);
}

// =============================================================================
// select() – Typed Tests (shape contract)
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, Select_2D_AlongDim0_ReducesDim) {
    // shape (3,4), select(0,1) => shape (4,)
    Tensor<TypeParam> t(
        NestedData<TypeParam>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
    auto v = t.select(0, 1);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 4u);
}

TYPED_TEST(TensorViewOpTypedTest, Select_2D_AlongDim1_ReducesDim) {
    // shape (2,3), select(1,2) => shape (2,)
    Tensor<TypeParam> t(NestedData<TypeParam>{{0, 1, 2}, {3, 4, 5}});
    auto v = t.select(1, 2);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 2u);
}

TYPED_TEST(TensorViewOpTypedTest, Select_3D_AlongDim0_Reduces3DTo2D) {
    // shape (2,2,2), select(0,1) => shape (2,2)
    Tensor<TypeParam> t(NestedData<TypeParam>{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
    auto v = t.select(0, 1);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 2u);
}

// =============================================================================
// sliceDim() – Typed Tests (shape contract)
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, SliceDim_1D_Step1_Shape) {
    // [0..4], sliceDim(0, 1, 4) => shape (3,)
    Tensor<TypeParam> t(NestedData<TypeParam>{0, 1, 2, 3, 4});
    auto v = t.sliceDim(0, 1, 4);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 3u);
}

TYPED_TEST(TensorViewOpTypedTest, SliceDim_1D_Step2_Shape) {
    // [0..4], sliceDim(0, 0, 5, 2) => ceil(5/2)=3 elements => shape (3,)
    Tensor<TypeParam> t(NestedData<TypeParam>{0, 1, 2, 3, 4});
    auto v = t.sliceDim(0, 0, 5, 2);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 3u);
}

TYPED_TEST(TensorViewOpTypedTest, SliceDim_2D_AlongDim0_Shape) {
    // shape (4,3), sliceDim(0, 0, 4, 2) => shape (2,3)
    Tensor<TypeParam> t(
        NestedData<TypeParam>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}});
    auto v = t.sliceDim(0, 0, 4, 2);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 3u);
}

TYPED_TEST(TensorViewOpTypedTest, SliceDim_DefaultArgs_FullDimension) {
    // No start/end => full slice, shape unchanged
    Tensor<TypeParam> t(NestedData<TypeParam>{0, 1, 2, 3});
    auto v = t.sliceDim(0);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 4u);
}

// =============================================================================
// slice() (SliceSetting) – Typed Tests (shape contract)
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, Slice_OneDimSetting_Shape) {
    // shape (3,3), slice dim0 [1:3:1] => shape (2,3)
    Tensor<TypeParam> t(NestedData<TypeParam>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    SliceSetting setting({{0, {1, 3, 1}}});
    auto v = t.slice(setting);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 3u);
}

TYPED_TEST(TensorViewOpTypedTest, Slice_TwoDimSetting_Shape) {
    // shape (3,4), dim0:[0:2:1], dim1:[1:3:1] => shape (2,2)
    Tensor<TypeParam> t(
        NestedData<TypeParam>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
    SliceSetting setting({{0, {0, 2, 1}}, {1, {1, 3, 1}}});
    auto v = t.slice(setting);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 2u);
}

// =============================================================================
// broadcastTo() – Typed Tests (shape contract)
// =============================================================================

TYPED_TEST(TensorViewOpTypedTest, BroadcastTo_1x3_To_2x3_Shape) {
    Tensor<TypeParam> t(NestedData<TypeParam>{{1, 2, 3}}); // shape (1,3)
    auto v = t.broadcastTo({2, 3});
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 3u);
    EXPECT_EQ(v.getTotalSize(), 6u);
}

TYPED_TEST(TensorViewOpTypedTest, BroadcastTo_1D_To_3D_Shape) {
    // shape (1,) -> (2, 3, 1)
    Tensor<TypeParam> t(NestedData<TypeParam>{5}); // shape (1,)
    auto v = t.broadcastTo({2, 3, 1});
    EXPECT_EQ(v.getDimensions(), 3u);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 3u);
    EXPECT_EQ(v.getShape()[2], 1u);
}

// =============================================================================
// Value-checked tests (float only)
// =============================================================================

class TensorViewOpTest : public ::testing::Test {};

// ── narrow() values ──────────────────────────────────────────────────────────

TEST_F(TensorViewOpTest, Narrow_1D_Values) {
    // [0,1,2,3,4], narrow(0,1,3) => [1,2,3]
    Tensor<float> t(NestedData<float>{0, 1, 2, 3, 4});
    auto v = t.narrow(0, 1, 3);
    EXPECT_FLOAT_EQ(v.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 3.0f);
}

TEST_F(TensorViewOpTest, Narrow_2D_AlongDim0_Values) {
    // [[0,1,2],[3,4,5],[6,7,8]], narrow(0,1,2) => [[3,4,5],[6,7,8]]
    Tensor<float> t(NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto v = t.narrow(0, 1, 2);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(v.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(v.at({0, 2}), 5.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(v.at({1, 2}), 8.0f);
}

TEST_F(TensorViewOpTest, Narrow_2D_AlongDim1_Values) {
    // [[0,1,2,3],[4,5,6,7]], narrow(1,1,2) => [[1,2],[5,6]]
    Tensor<float> t(NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}});
    auto v = t.narrow(1, 1, 2);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 6.0f);
}

TEST_F(TensorViewOpTest, Narrow_StartZero_IsViewOfHead) {
    // narrow starting at 0 => first N elements
    Tensor<float> t(NestedData<float>{10, 20, 30, 40, 50});
    auto v = t.narrow(0, 0, 3);
    EXPECT_FLOAT_EQ(v.at({0}), 10.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 20.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 30.0f);
}

TEST_F(TensorViewOpTest, Narrow_ViewIsWriteThrough) {
    // Writing into the narrow view should be reflected in the original tensor
    Tensor<float> t(NestedData<float>{1, 2, 3, 4, 5});
    auto v = t.narrow(0, 1, 3); // [2,3,4]
    v.at({0}) = 99.0f;
    EXPECT_FLOAT_EQ(t.at({1}), 99.0f); // write-through to base
    EXPECT_FLOAT_EQ(t.at({0}), 1.0f);  // unaffected
    EXPECT_FLOAT_EQ(t.at({4}), 5.0f);  // unaffected
}

// ── select() values ──────────────────────────────────────────────────────────

TEST_F(TensorViewOpTest, Select_2D_Dim0_Values) {
    // [[0,1,2],[3,4,5]], select(0,1) => [3,4,5]
    Tensor<float> t(NestedData<float>{{0, 1, 2}, {3, 4, 5}});
    auto v = t.select(0, 1);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_FLOAT_EQ(v.at({0}), 3.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 4.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 5.0f);
}

TEST_F(TensorViewOpTest, Select_2D_Dim1_Values) {
    // [[0,1,2],[3,4,5]], select(1,2) => [2,5]
    Tensor<float> t(NestedData<float>{{0, 1, 2}, {3, 4, 5}});
    auto v = t.select(1, 2);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_FLOAT_EQ(v.at({0}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 5.0f);
}

TEST_F(TensorViewOpTest, Select_3D_Dim0_Values) {
    // [[[0,1],[2,3]],[[4,5],[6,7]]], select(0,1) => [[4,5],[6,7]]
    Tensor<float> t(NestedData<float>{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
    auto v = t.select(0, 1);
    EXPECT_EQ(v.getDimensions(), 2u);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(v.at({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 7.0f);
}

TEST_F(TensorViewOpTest, Select_ViewIsWriteThrough) {
    Tensor<float> t(NestedData<float>{{1, 2, 3}, {4, 5, 6}});
    auto row = t.select(0, 1); // [4,5,6]
    row.at({1}) = 99.0f;
    EXPECT_FLOAT_EQ(t.at({1, 1}), 99.0f);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f); // unaffected
}

// ── sliceDim() values ─────────────────────────────────────────────────────────

TEST_F(TensorViewOpTest, SliceDim_1D_Step1_Values) {
    // [0,1,2,3,4], sliceDim(0,1,4,1) => [1,2,3]
    Tensor<float> t(NestedData<float>{0, 1, 2, 3, 4});
    auto v = t.sliceDim(0, 1, 4, 1);
    EXPECT_EQ(v.getDimensions(), 1u);
    EXPECT_EQ(v.getShape()[0], 3u);
    EXPECT_FLOAT_EQ(v.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 3.0f);
}

TEST_F(TensorViewOpTest, SliceDim_1D_Step2_Values) {
    // [0,1,2,3,4], sliceDim(0, 0, 5, 2) => [0,2,4]
    Tensor<float> t(NestedData<float>{0, 1, 2, 3, 4});
    auto v = t.sliceDim(0, 0, 5, 2);
    EXPECT_EQ(v.getShape()[0], 3u);
    EXPECT_FLOAT_EQ(v.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 4.0f);
}

TEST_F(TensorViewOpTest, SliceDim_2D_AlongDim0_Step2_Values) {
    // [[0,1],[2,3],[4,5],[6,7]], sliceDim(0,0,4,2) => rows 0,2 => [[0,1],[4,5]]
    Tensor<float> t(NestedData<float>{{0, 1}, {2, 3}, {4, 5}, {6, 7}});
    auto v = t.sliceDim(0, 0, 4, 2);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 2u);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(v.at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 5.0f);
}

TEST_F(TensorViewOpTest, SliceDim_DefaultArgs_AllElements) {
    Tensor<float> t(NestedData<float>{10, 20, 30});
    auto v = t.sliceDim(0);
    EXPECT_EQ(v.getShape()[0], 3u);
    EXPECT_FLOAT_EQ(v.at({0}), 10.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 30.0f);
}

TEST_F(TensorViewOpTest, SliceDim_StepMatchesStride) {
    // stride for dim0 should be base_stride * step
    Tensor<float> t(NestedData<float>{0, 1, 2, 3, 4, 5});
    auto v = t.sliceDim(0, 0, 6, 3); // step=3
    ASSERT_EQ(v.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(v.getStride().getStrideVec()[0], 3u);
}

// ── slice() (SliceSetting) values ────────────────────────────────────────────

TEST_F(TensorViewOpTest, Slice_OneDim_Values) {
    // [[0,1,2],[3,4,5],[6,7,8]], dim0:[1:3:1] => [[3,4,5],[6,7,8]]
    Tensor<float> t(NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    SliceSetting setting({{0, {1, 3, 1}}});
    auto v = t.slice(setting);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(v.at({0, 2}), 5.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(v.at({1, 2}), 8.0f);
}

TEST_F(TensorViewOpTest, Slice_TwoDims_Values) {
    // [[0,1,2,3],[4,5,6,7],[8,9,10,11]], dim0:[0:2:1], dim1:[1:3:1]
    // => rows 0..1, cols 1..2 => [[1,2],[5,6]]
    Tensor<float> t(NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
    SliceSetting setting({{0, {0, 2, 1}}, {1, {1, 3, 1}}});
    auto v = t.slice(setting);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 6.0f);
}

TEST_F(TensorViewOpTest, Slice_WithStep2_Values) {
    // [0,1,2,3,4,5], dim0:[0:6:2] => [0,2,4]
    Tensor<float> t(NestedData<float>{0, 1, 2, 3, 4, 5});
    SliceSetting setting({{0, {0, 6, 2}}});
    auto v = t.slice(setting);
    EXPECT_EQ(v.getShape()[0], 3u);
    EXPECT_FLOAT_EQ(v.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 4.0f);
}

TEST_F(TensorViewOpTest, Slice_UnsortedAxes_StillCorrect) {
    // Supply axes in reverse order; sortAxis() inside slice() fixes it
    Tensor<float> t(NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
    SliceSetting setting({{1, {1, 3, 1}}, {0, {0, 2, 1}}}); // dim1 before dim0
    auto v = t.slice(setting);
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 2u);
    EXPECT_FLOAT_EQ(v.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 6.0f);
}

// ── broadcastTo() values ──────────────────────────────────────────────────────

TEST_F(TensorViewOpTest, BroadcastTo_1x3_To_2x3_Values) {
    // [[1,2,3]] shape (1,3) broadcast to (2,3)
    // Both rows should read the same values (stride[0] = 0)
    Tensor<float> t(NestedData<float>{{1, 2, 3}});
    auto v = t.broadcastTo({2, 3});
    EXPECT_EQ(v.getShape()[0], 2u);
    EXPECT_EQ(v.getShape()[1], 3u);
    // Row 0 and row 1 are identical
    EXPECT_FLOAT_EQ(v.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(v.at({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(v.at({1, 1}), 2.0f);
    EXPECT_FLOAT_EQ(v.at({1, 2}), 3.0f);
}

TEST_F(TensorViewOpTest, BroadcastTo_Stride_IszeroOnBroadcastedAxis) {
    // (1,3) => (2,3): stride[0] should be 0 (broadcasted), stride[1] stays 1
    Tensor<float> t(NestedData<float>{{10, 20, 30}});
    auto v = t.broadcastTo({2, 3});
    ASSERT_EQ(v.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(v.getStride().getStrideVec()[0], 0u);
    EXPECT_EQ(v.getStride().getStrideVec()[1], 1u);
}

TEST_F(TensorViewOpTest, BroadcastTo_IdentityBroadcast_ValuesUnchanged) {
    // Broadcasting to the same shape is a no-op
    Tensor<float> t(NestedData<float>{5, 6, 7}); // shape (3,)
    auto v = t.broadcastTo({3});
    EXPECT_EQ(v.getShape()[0], 3u);
    EXPECT_FLOAT_EQ(v.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(v.at({1}), 6.0f);
    EXPECT_FLOAT_EQ(v.at({2}), 7.0f);
}

// ── Chained views ─────────────────────────────────────────────────────────────

TEST_F(TensorViewOpTest, NarrowThenSelect_ChainedValues) {
    // [[0,1,2,3],[4,5,6,7],[8,9,10,11]], narrow(0,1,2) => [[4..7],[8..11]]
    // then select(1,1) => [5,9]
    Tensor<float> t(NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
    auto narrowed = t.narrow(0, 1, 2);     // shape (2,4)
    auto selected = narrowed.select(1, 1); // shape (2,): col 1 of each row
    EXPECT_EQ(selected.getDimensions(), 1u);
    EXPECT_EQ(selected.getShape()[0], 2u);
    EXPECT_FLOAT_EQ(selected.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(selected.at({1}), 9.0f);
}

TEST_F(TensorViewOpTest, SliceDimThenNarrow_ChainedValues) {
    // [0..9], sliceDim(0,2,8,1) => [2,3,4,5,6,7]
    // then narrow(0,1,3) => [3,4,5]
    Tensor<float> t(NestedData<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto sliced = t.sliceDim(0, 2, 8, 1);   // [2..7]
    auto narrowed = sliced.narrow(0, 1, 3); // [3,4,5]
    EXPECT_EQ(narrowed.getShape()[0], 3u);
    EXPECT_FLOAT_EQ(narrowed.at({0}), 3.0f);
    EXPECT_FLOAT_EQ(narrowed.at({1}), 4.0f);
    EXPECT_FLOAT_EQ(narrowed.at({2}), 5.0f);
}

TEST_F(TensorViewOpTest, NarrowThenClone_DeepCopy) {
    // Clone must be independent from the original
    Tensor<float> t(NestedData<float>{1, 2, 3, 4, 5});
    auto v = t.narrow(0, 1, 3); // view of [2,3,4]
    auto c = v.clone();
    c.at({0}) = 99.0f;
    EXPECT_FLOAT_EQ(t.at({1}), 2.0f); // original unaffected
    EXPECT_FLOAT_EQ(v.at({0}), 2.0f); // view unaffected
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST_F(TensorViewOpTest, Narrow_DimOutOfRange_Throws) {
    Tensor<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.narrow(1, 0, 1), std::out_of_range);
}

TEST_F(TensorViewOpTest, Narrow_StartOutOfRange_Throws) {
    Tensor<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.narrow(0, 3, 1), std::out_of_range);
}

TEST_F(TensorViewOpTest, Narrow_LengthExceedsBounds_Throws) {
    Tensor<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.narrow(0, 1, 3), std::out_of_range);
}

TEST_F(TensorViewOpTest, Select_DimOutOfRange_Throws) {
    Tensor<float> t(NestedData<float>{{1, 2}, {3, 4}});
    EXPECT_THROW(t.select(2, 0), std::out_of_range);
}

TEST_F(TensorViewOpTest, Select_IndexOutOfRange_Throws) {
    Tensor<float> t(NestedData<float>{{1, 2}, {3, 4}});
    EXPECT_THROW(t.select(0, 2), std::out_of_range);
}

TEST_F(TensorViewOpTest, SliceDim_DimOutOfRange_Throws) {
    Tensor<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(1, 0, 3, 1), std::out_of_range);
}

TEST_F(TensorViewOpTest, SliceDim_StepZero_Throws) {
    Tensor<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(0, 0, 3, 0), std::invalid_argument);
}

TEST_F(TensorViewOpTest, SliceDim_StartGEEnd_Throws) {
    Tensor<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(0, 2, 1, 1), std::invalid_argument);
}

TEST_F(TensorViewOpTest, BroadcastTo_IncompatibleShape_Throws) {
    Tensor<float> t(NestedData<float>{{1, 2, 3}, {4, 5, 6}}); // shape (2,3)
    // Cannot broadcast (2,3) => (2,4): 3 != 4 and neither is 1
    EXPECT_THROW(t.broadcastTo({2, 4}), std::invalid_argument);
}
