//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//
// =============================================================================
// Comprehensive tests for TensorWrapper view operations.
//
// For every test we explicitly verify ALL observable fields:
//   - shape  (getDimensions(), getShape()[i])
//   - stride (getStride().getStrideVec()[i])
//   - device (getDevice()->getType())
//   - data sharing / offset  (getRawData().get() ptr equality, value checks)
//   - isContiguous()
//   - element values via at()
// =============================================================================

#include <gtest/gtest.h>
#include <stdexcept>

#include "backend/DeviceRegistry.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"
#include "math/slice_setting.h"

using hahaha::backend::DeviceType;
using hahaha::math::NestedData;
using hahaha::math::TensorWrapper;

// =============================================================================
// isContiguous() Tests
// =============================================================================

class IsContiguousTest : public ::testing::Test {};

TEST_F(IsContiguousTest, FreshTensor_IsContiguous) {
    TensorWrapper<float> t(NestedData<float>{{1, 2, 3}, {4, 5, 6}});
    // shape (2,3), strides [3,1], offset 0
    EXPECT_TRUE(t.isContiguous());
    EXPECT_EQ(t.getDimensions(), 2u);
    EXPECT_EQ(t.getShape()[0], 2u);
    EXPECT_EQ(t.getShape()[1], 3u);
    EXPECT_EQ(t.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(t.getStride().getStrideVec()[1], 1u);
    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CPU);
}

TEST_F(IsContiguousTest, NarrowedView_IsNotContiguous) {
    // narrow introduces a non-zero offset
    TensorWrapper<float> t(NestedData<float>{1, 2, 3, 4, 5});
    auto view = t.narrow(0, 1, 3); // elements [1,2,3] (indices 1..3)
    EXPECT_FALSE(view.isContiguous());
}

TEST_F(IsContiguousTest, SliceWithStep1_IsNotContiguous_DueToOffset) {
    TensorWrapper<float> t(NestedData<float>{0, 1, 2, 3, 4});
    auto view = t.sliceDim(0, 1, 4, 1); // [1,2,3], step=1 but offset=1
    EXPECT_FALSE(view.isContiguous());
}

TEST_F(IsContiguousTest, SliceWithStep2_IsNotContiguous) {
    TensorWrapper<float> t(NestedData<float>{0, 1, 2, 3, 4});
    auto view = t.sliceDim(0, 0, 5, 2); // every other element; stride[0] = 2
    EXPECT_FALSE(view.isContiguous());
}

TEST_F(IsContiguousTest, CloneOfView_IsContiguous) {
    TensorWrapper<float> t(NestedData<float>{0, 1, 2, 3, 4});
    auto view = t.narrow(0, 1, 3);
    EXPECT_FALSE(view.isContiguous());
    auto cloned = view.clone();
    EXPECT_TRUE(cloned.isContiguous());
}

// =============================================================================
// narrow() Tests
// =============================================================================

class NarrowTest : public ::testing::Test {};

TEST_F(NarrowTest, Narrow1D_CorrectShapeStrideAndValues) {
    // Base tensor: [0, 1, 2, 3, 4]
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});

    // narrow(dim=0, start=1, length=3) => view of [1, 2, 3]
    auto view = base.narrow(0, 1, 3);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);

    // --- Stride (unchanged from base: [1]) ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing: same underlying buffer ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous (offset != 0 => false) ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values via at() ---
    EXPECT_FLOAT_EQ(view.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 3.0f);
}

TEST_F(NarrowTest, Narrow2D_AlongDim0_CorrectFields) {
    // Base: [[0,1,2],[3,4,5],[6,7,8]] shape (3,3)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});

    // narrow(0, 1, 2) => rows [1,2] => [[3,4,5],[6,7,8]]
    auto view = base.narrow(0, 1, 2);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 3u);

    // --- Stride: row stride = 3, col stride = 1 (same as base) ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(view.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(view.at({0, 2}), 5.0f);
    EXPECT_FLOAT_EQ(view.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(view.at({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(view.at({1, 2}), 8.0f);
}

TEST_F(NarrowTest, Narrow2D_AlongDim1_CorrectFields) {
    // Base: [[0,1,2,3],[4,5,6,7]] shape (2,4)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}});

    // narrow(1, 1, 2) => cols [1,2] => [[1,2],[5,6]]
    auto view = base.narrow(1, 1, 2);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 2u);

    // --- Stride: row stride = 4 (same as base), col stride = 1 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 4u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(view.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(view.at({1, 1}), 6.0f);
}

TEST_F(NarrowTest, Narrow_StartZero_FirstElements) {
    TensorWrapper<float> base(NestedData<float>{10, 20, 30, 40, 50});
    auto view = base.narrow(0, 0, 3); // [10, 20, 30]

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);

    // --- Stride ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: offset=0 and strides match => true ---
    EXPECT_TRUE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0}), 10.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 20.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 30.0f);
}

TEST_F(NarrowTest, Narrow_ErrorHandling_DimOutOfRange) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.narrow(1, 0, 1), std::out_of_range);
}

TEST_F(NarrowTest, Narrow_ErrorHandling_StartOutOfRange) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.narrow(0, 3, 1), std::out_of_range);
}

TEST_F(NarrowTest, Narrow_ErrorHandling_LengthExceedsBounds) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.narrow(0, 1, 3), std::out_of_range);
}

// =============================================================================
// select() Tests
// =============================================================================

class SelectTest : public ::testing::Test {};

TEST_F(SelectTest, Select2D_AlongDim0_ReducesDimension) {
    // Base: [[0,1,2],[3,4,5]] shape (2,3)
    TensorWrapper<float> base(NestedData<float>{{0, 1, 2}, {3, 4, 5}});

    // select(0, 1) => row 1 => [3,4,5]  shape (3,)
    auto view = base.select(0, 1);

    // --- Shape (dimension removed) ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);

    // --- Stride: col stride = 1 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: offset = 1*3 = 3, so NOT contiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0}), 3.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 4.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 5.0f);
}

TEST_F(SelectTest, Select2D_AlongDim1_ReducesDimension) {
    // Base: [[0,1,2],[3,4,5]] shape (2,3)
    TensorWrapper<float> base(NestedData<float>{{0, 1, 2}, {3, 4, 5}});

    // select(1, 2) => col 2 => [2,5]  shape (2,)
    auto view = base.select(1, 2);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 2u);

    // --- Stride: row stride = 3 (from base) ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 3u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: offset=2, stride[0]=3 != 1 => NOT contiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 5.0f);
}

TEST_F(SelectTest, Select3D_AlongDim0_Reduces3DTo2D) {
    // Base: [[[0,1],[2,3]],[[4,5],[6,7]]] shape (2,2,2)
    TensorWrapper<float> base(
        NestedData<float>{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});

    // select(0, 1) => second "slice" => [[4,5],[6,7]] shape (2,2)
    auto view = base.select(0, 1);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 2u);

    // --- Stride: [2, 1] (row stride 2, col stride 1 from base dim 1,2) ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: offset = 1*4 = 4, NOT zero => NOT contiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(view.at({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(view.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(view.at({1, 1}), 7.0f);
}

TEST_F(SelectTest, Select_ErrorHandling_DimOutOfRange) {
    TensorWrapper<float> t(NestedData<float>{{1, 2}, {3, 4}});
    EXPECT_THROW(t.select(2, 0), std::out_of_range);
}

TEST_F(SelectTest, Select_ErrorHandling_IndexOutOfRange) {
    TensorWrapper<float> t(NestedData<float>{{1, 2}, {3, 4}});
    EXPECT_THROW(t.select(0, 2), std::out_of_range);
}

// =============================================================================
// sliceDim() Tests
// =============================================================================

class SliceDimTest : public ::testing::Test {};

TEST_F(SliceDimTest, SliceDim1D_Step1_CorrectFields) {
    // Base: [0, 1, 2, 3, 4]
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});

    // sliceDim(0, 1, 4, 1) => [1, 2, 3]
    auto view = base.sliceDim(0, 1, 4, 1);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);

    // --- Stride: step=1 => stride[0] = 1*1 = 1 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: offset=1 => NOT contiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 3.0f);
}

TEST_F(SliceDimTest, SliceDim1D_Step2_CorrectFields) {
    // Base: [0, 1, 2, 3, 4]
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});

    // sliceDim(0, 0, 5, 2) => [0, 2, 4]
    auto view = base.sliceDim(0, 0, 5, 2);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);  // ceil((5-0)/2) = 3

    // --- Stride: step=2 => stride[0] = 1*2 = 2 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 2u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: stride != 1 => NOT contiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 4.0f);
}

TEST_F(SliceDimTest, SliceDim2D_AlongDim0_Step1) {
    // Base: [[0,1,2],[3,4,5],[6,7,8]] shape (3,3)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});

    // sliceDim(0, 1, 3, 1) => rows [1,2] => [[3,4,5],[6,7,8]]
    auto view = base.sliceDim(0, 1, 3, 1);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 3u);

    // --- Stride: row stride 3*1=3, col stride 1 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(view.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(view.at({0, 2}), 5.0f);
    EXPECT_FLOAT_EQ(view.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(view.at({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(view.at({1, 2}), 8.0f);
}

TEST_F(SliceDimTest, SliceDim2D_AlongDim0_Step2) {
    // Base: [[0,1],[2,3],[4,5],[6,7]] shape (4,2)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1}, {2, 3}, {4, 5}, {6, 7}});

    // sliceDim(0, 0, 4, 2) => rows [0,2] => [[0,1],[4,5]]
    auto view = base.sliceDim(0, 0, 4, 2);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 2u);

    // --- Stride: row stride = 2*2=4, col stride = 1 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 4u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(view.at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(view.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(view.at({1, 1}), 5.0f);
}

TEST_F(SliceDimTest, SliceDim_DefaultStartEnd_GetsFullDim) {
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});
    auto view = base.sliceDim(0); // no start/end => full slice

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 5u);

    // --- Stride ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous: offset=0, stride=1 => true ---
    EXPECT_TRUE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(view.at({4}), 4.0f);
}

TEST_F(SliceDimTest, SliceDim_ErrorHandling_DimOutOfRange) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(1, 0, 3, 1), std::out_of_range);
}

TEST_F(SliceDimTest, SliceDim_ErrorHandling_StepZero) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(0, 0, 3, 0), std::invalid_argument);
}

TEST_F(SliceDimTest, SliceDim_ErrorHandling_StartGEEnd) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(0, 2, 1, 1), std::invalid_argument);
}

TEST_F(SliceDimTest, SliceDim_ErrorHandling_StartOutOfRange) {
    TensorWrapper<float> t(NestedData<float>{1, 2, 3});
    EXPECT_THROW(t.sliceDim(0, 3, 4, 1), std::out_of_range);
}

// =============================================================================
// slice() Tests (using SliceSetting)
// =============================================================================

class SliceTest : public ::testing::Test {};

TEST_F(SliceTest, Slice2D_OneDim_CorrectFields) {
    // Base: [[0,1,2],[3,4,5],[6,7,8]] shape (3,3)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});

    SliceSetting setting({{0, {1, 3, 1}}}); // dim0: [1:3:1]
    auto view = base.slice(setting);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 3u);

    // --- Stride ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(view.at({1, 2}), 8.0f);
}

TEST_F(SliceTest, Slice2D_TwoDims_CorrectFields) {
    // Base: [[0,1,2,3],[4,5,6,7],[8,9,10,11]] shape (3,4)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});

    // dim0: [0:2:1], dim1: [1:3:1]
    SliceSetting setting({{0, {0, 2, 1}}, {1, {1, 3, 1}}});
    auto view = base.slice(setting);

    // --- Shape ---
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 2u);

    // --- Stride: row = 4, col = 1 ---
    ASSERT_EQ(view.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 4u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(view.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view.isContiguous());

    // --- Values: rows 0..1, cols 1..2 ---
    EXPECT_FLOAT_EQ(view.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(view.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(view.at({1, 1}), 6.0f);
}

// =============================================================================
// Clone of View Tests
// =============================================================================

class CloneViewTest : public ::testing::Test {};

TEST_F(CloneViewTest, CloneNarrowedView_IsContiguousDeepCopy) {
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});
    auto view = base.narrow(0, 1, 3); // [1, 2, 3]
    EXPECT_FALSE(view.isContiguous());

    auto cloned = view.clone();

    // --- Shape ---
    EXPECT_EQ(cloned.getDimensions(), 1u);
    EXPECT_EQ(cloned.getShape()[0], 3u);

    // --- Stride: contiguous strides ---
    ASSERT_EQ(cloned.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(cloned.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(cloned.getDevice()->getType(), DeviceType::CPU);

    // --- Data NOT shared: deep copy ---
    EXPECT_NE(cloned.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_TRUE(cloned.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(cloned.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(cloned.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(cloned.at({2}), 3.0f);

    // --- Modifying clone does NOT affect original ---
    cloned.at({0}) = 99.0f;
    EXPECT_FLOAT_EQ(base.at({1}), 1.0f);
}

TEST_F(CloneViewTest, CloneSlicedView_2D_DeepCopy) {
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto view = base.narrow(0, 1, 2); // rows 1..2

    auto cloned = view.clone();

    // --- Shape ---
    EXPECT_EQ(cloned.getDimensions(), 2u);
    EXPECT_EQ(cloned.getShape()[0], 2u);
    EXPECT_EQ(cloned.getShape()[1], 3u);

    // --- Stride: contiguous row-major ---
    ASSERT_EQ(cloned.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(cloned.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(cloned.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(cloned.getDevice()->getType(), DeviceType::CPU);

    // --- Data NOT shared ---
    EXPECT_NE(cloned.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_TRUE(cloned.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(cloned.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(cloned.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(cloned.at({0, 2}), 5.0f);
    EXPECT_FLOAT_EQ(cloned.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(cloned.at({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(cloned.at({1, 2}), 8.0f);
}

TEST_F(CloneViewTest, CloneStepView_AllValues) {
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4, 5});
    auto view = base.sliceDim(0, 0, 6, 2); // [0, 2, 4]

    auto cloned = view.clone();

    // --- Shape ---
    EXPECT_EQ(cloned.getDimensions(), 1u);
    EXPECT_EQ(cloned.getShape()[0], 3u);

    // --- Stride: contiguous ---
    ASSERT_EQ(cloned.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(cloned.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(cloned.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(cloned.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(cloned.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(cloned.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(cloned.at({2}), 4.0f);
}

// =============================================================================
// Arithmetic Operations on Views
// =============================================================================

class ViewArithmeticTest : public ::testing::Test {};

TEST_F(ViewArithmeticTest, AddTwoViews_CorrectResultFields) {
    // a = [0,1,2,3,4], view_a = [1,2,3]
    TensorWrapper<float> a(NestedData<float>{0, 1, 2, 3, 4});
    TensorWrapper<float> b(NestedData<float>{10, 20, 30});
    auto view_a = a.narrow(0, 1, 3); // [1,2,3]

    auto result = view_a.add(b);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 1u);
    EXPECT_EQ(result.getShape()[0], 3u);

    // --- Stride: result is contiguous ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0}), 11.0f);  // 1 + 10
    EXPECT_FLOAT_EQ(result.at({1}), 22.0f);  // 2 + 20
    EXPECT_FLOAT_EQ(result.at({2}), 33.0f);  // 3 + 30

    // --- Original tensors unchanged ---
    EXPECT_FLOAT_EQ(a.at({1}), 1.0f);
    EXPECT_FLOAT_EQ(a.at({2}), 2.0f);
    EXPECT_FLOAT_EQ(b.at({0}), 10.0f);
}

TEST_F(ViewArithmeticTest, SubtractView_CorrectResultFields) {
    TensorWrapper<float> a(NestedData<float>{10, 20, 30, 40, 50});
    auto view_a = a.narrow(0, 2, 3); // [30, 40, 50]
    TensorWrapper<float> b(NestedData<float>{5, 10, 15});

    auto result = view_a.subtract(b);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 1u);
    EXPECT_EQ(result.getShape()[0], 3u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0}), 25.0f);  // 30 - 5
    EXPECT_FLOAT_EQ(result.at({1}), 30.0f);  // 40 - 10
    EXPECT_FLOAT_EQ(result.at({2}), 35.0f);  // 50 - 15
}

TEST_F(ViewArithmeticTest, MultiplyView_CorrectResultFields) {
    TensorWrapper<float> a(NestedData<float>{1, 2, 3, 4, 5});
    auto view_a = a.narrow(0, 0, 4); // [1,2,3,4] (start=0 => contiguous)
    TensorWrapper<float> b(NestedData<float>{10, 10, 10, 10});

    auto result = view_a.multiply(b);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 1u);
    EXPECT_EQ(result.getShape()[0], 4u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0}), 10.0f);
    EXPECT_FLOAT_EQ(result.at({1}), 20.0f);
    EXPECT_FLOAT_EQ(result.at({2}), 30.0f);
    EXPECT_FLOAT_EQ(result.at({3}), 40.0f);
}

TEST_F(ViewArithmeticTest, DivideView_CorrectResultFields) {
    TensorWrapper<float> a(NestedData<float>{0, 10, 20, 30, 40});
    auto view_a = a.narrow(0, 1, 4); // [10, 20, 30, 40]
    TensorWrapper<float> b(NestedData<float>{2, 4, 5, 8});

    auto result = view_a.divide(b);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 1u);
    EXPECT_EQ(result.getShape()[0], 4u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0}), 5.0f);   // 10/2
    EXPECT_FLOAT_EQ(result.at({1}), 5.0f);   // 20/4
    EXPECT_FLOAT_EQ(result.at({2}), 6.0f);   // 30/5
    EXPECT_FLOAT_EQ(result.at({3}), 5.0f);   // 40/8
}

TEST_F(ViewArithmeticTest, AddViewScalar_CorrectResultFields) {
    TensorWrapper<float> a(NestedData<float>{0, 1, 2, 3, 4});
    auto view_a = a.narrow(0, 2, 3); // [2, 3, 4]

    auto result = view_a.add(100.0f);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 1u);
    EXPECT_EQ(result.getShape()[0], 3u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0}), 102.0f);
    EXPECT_FLOAT_EQ(result.at({1}), 103.0f);
    EXPECT_FLOAT_EQ(result.at({2}), 104.0f);
}

TEST_F(ViewArithmeticTest, AddTwoStepViews_CorrectResultFields) {
    // view with step=2 on both sides
    TensorWrapper<float> a(NestedData<float>{0, 1, 2, 3, 4, 5});
    TensorWrapper<float> b(NestedData<float>{0, 10, 20, 30, 40, 50});
    auto va = a.sliceDim(0, 0, 6, 2); // [0,2,4]
    auto vb = b.sliceDim(0, 0, 6, 2); // [0,20,40]

    auto result = va.add(vb);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 1u);
    EXPECT_EQ(result.getShape()[0], 3u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(result.at({1}), 22.0f);
    EXPECT_FLOAT_EQ(result.at({2}), 44.0f);
}

TEST_F(ViewArithmeticTest, Add2DViews_CorrectResultFields) {
    // Base: [[0,1,2],[3,4,5],[6,7,8]] shape (3,3)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    TensorWrapper<float> addend(NestedData<float>{{10, 10, 10}, {20, 20, 20}});

    auto view = base.narrow(0, 1, 2); // rows 1..2
    auto result = view.add(addend);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 2u);
    EXPECT_EQ(result.getShape()[0], 2u);
    EXPECT_EQ(result.getShape()[1], 3u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(result.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(result.at({0, 0}), 13.0f);  // 3 + 10
    EXPECT_FLOAT_EQ(result.at({0, 1}), 14.0f);  // 4 + 10
    EXPECT_FLOAT_EQ(result.at({0, 2}), 15.0f);  // 5 + 10
    EXPECT_FLOAT_EQ(result.at({1, 0}), 26.0f);  // 6 + 20
    EXPECT_FLOAT_EQ(result.at({1, 1}), 27.0f);  // 7 + 20
    EXPECT_FLOAT_EQ(result.at({1, 2}), 28.0f);  // 8 + 20
}

// =============================================================================
// sum() on Views
// =============================================================================

class ViewSumTest : public ::testing::Test {};

TEST_F(ViewSumTest, SumNarrowedView_CorrectValue) {
    TensorWrapper<float> base(NestedData<float>{1, 2, 3, 4, 5});
    auto view = base.narrow(0, 1, 3); // [2, 3, 4]

    float s = view.sum();

    EXPECT_FLOAT_EQ(s, 9.0f);

    // Verify base tensor shape/stride/device unchanged
    EXPECT_EQ(base.getDimensions(), 1u);
    EXPECT_EQ(base.getShape()[0], 5u);
    EXPECT_EQ(base.getStride().getStrideVec()[0], 1u);
    EXPECT_EQ(base.getDevice()->getType(), DeviceType::CPU);
}

TEST_F(ViewSumTest, Sum2DViewAlongAxis_CorrectValue) {
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto view = base.narrow(0, 1, 2); // rows 1,2 => [[3,4,5],[6,7,8]]

    float s = view.sum();

    EXPECT_FLOAT_EQ(s, 33.0f); // 3+4+5+6+7+8

    // Verify shape/stride/device of view (should be unchanged after sum)
    EXPECT_EQ(view.getDimensions(), 2u);
    EXPECT_EQ(view.getShape()[0], 2u);
    EXPECT_EQ(view.getShape()[1], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[1], 1u);
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);
}

TEST_F(ViewSumTest, SumStepView_CorrectValue) {
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});
    auto view = base.sliceDim(0, 0, 5, 2); // [0, 2, 4]

    float s = view.sum();

    EXPECT_FLOAT_EQ(s, 6.0f);
}

// =============================================================================
// Chained Views
// =============================================================================

class ChainedViewTest : public ::testing::Test {};

TEST_F(ChainedViewTest, NarrowThenNarrow_CorrectFields) {
    // Base: [0..9]
    TensorWrapper<float> base(
        NestedData<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto view1 = base.narrow(0, 2, 6); // [2,3,4,5,6,7]
    auto view2 = view1.narrow(0, 1, 3); // [3,4,5]

    // --- Shape ---
    EXPECT_EQ(view2.getDimensions(), 1u);
    EXPECT_EQ(view2.getShape()[0], 3u);

    // --- Stride ---
    ASSERT_EQ(view2.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(view2.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(view2.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing: same buffer as original base ---
    EXPECT_EQ(view2.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(view2.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(view2.at({0}), 3.0f);
    EXPECT_FLOAT_EQ(view2.at({1}), 4.0f);
    EXPECT_FLOAT_EQ(view2.at({2}), 5.0f);
}

TEST_F(ChainedViewTest, SelectThenNarrow_CorrectFields) {
    // Base: [[0,1,2,3],[4,5,6,7],[8,9,10,11]] shape (3,4)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});

    // select row 1 => [4,5,6,7], shape (4,)
    auto row = base.select(0, 1);
    // narrow to [1,2] columns => [5,6]
    auto sub = row.narrow(0, 1, 2);

    // --- Shape ---
    EXPECT_EQ(sub.getDimensions(), 1u);
    EXPECT_EQ(sub.getShape()[0], 2u);

    // --- Stride ---
    ASSERT_EQ(sub.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(sub.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(sub.getDevice()->getType(), DeviceType::CPU);

    // --- Data sharing ---
    EXPECT_EQ(sub.getRawData().get(), base.getRawData().get());

    // --- isContiguous ---
    EXPECT_FALSE(sub.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(sub.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(sub.at({1}), 6.0f);
}

// =============================================================================
// Write-through: modifying via view modifies base
// =============================================================================

class ViewWriteThroughTest : public ::testing::Test {};

TEST_F(ViewWriteThroughTest, WriteToNarrowedView_ModifiesBase) {
    TensorWrapper<float> base(NestedData<float>{10, 20, 30, 40, 50});
    auto view = base.narrow(0, 1, 3); // [20, 30, 40]

    view.at({0}) = 200.0f;
    view.at({1}) = 300.0f;

    // Base should reflect changes (view shares data)
    EXPECT_FLOAT_EQ(base.at({1}), 200.0f);
    EXPECT_FLOAT_EQ(base.at({2}), 300.0f);
    // Unmodified base elements
    EXPECT_FLOAT_EQ(base.at({0}), 10.0f);
    EXPECT_FLOAT_EQ(base.at({3}), 40.0f);
    EXPECT_FLOAT_EQ(base.at({4}), 50.0f);
}

TEST_F(ViewWriteThroughTest, WriteToSelectedView_ModifiesBase) {
    TensorWrapper<float> base(
        NestedData<float>{{1, 2, 3}, {4, 5, 6}});
    auto row1 = base.select(0, 1); // [4, 5, 6]

    row1.at({1}) = 99.0f;

    EXPECT_FLOAT_EQ(base.at({1, 1}), 99.0f);
    // Other elements unchanged
    EXPECT_FLOAT_EQ(base.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(base.at({1, 0}), 4.0f);
}

// =============================================================================
// axpy() on View
// =============================================================================

class ViewAxpyTest : public ::testing::Test {};

TEST_F(ViewAxpyTest, Axpy_OnNarrowedView_CorrectFields) {
    // y = narrow view [3, 4, 5] from [0..5]
    TensorWrapper<float> data_y(NestedData<float>{0, 1, 2, 3, 4, 5});
    auto y_view = data_y.narrow(0, 3, 3); // [3, 4, 5]

    TensorWrapper<float> x(NestedData<float>{10, 20, 30});

    // y = alpha*x + y => y = 2*[10,20,30] + [3,4,5] = [23,44,65]
    y_view.axpy(2.0f, x);

    // View still has same shape
    EXPECT_EQ(y_view.getDimensions(), 1u);
    EXPECT_EQ(y_view.getShape()[0], 3u);
    EXPECT_EQ(y_view.getStride().getStrideVec()[0], 1u);
    EXPECT_EQ(y_view.getDevice()->getType(), DeviceType::CPU);
    EXPECT_FALSE(y_view.isContiguous());

    // Values via view's at()
    EXPECT_FLOAT_EQ(y_view.at({0}), 23.0f);
    EXPECT_FLOAT_EQ(y_view.at({1}), 44.0f);
    EXPECT_FLOAT_EQ(y_view.at({2}), 65.0f);

    // Write-through: base tensor reflects changes
    EXPECT_FLOAT_EQ(data_y.at({3}), 23.0f);
    EXPECT_FLOAT_EQ(data_y.at({4}), 44.0f);
    EXPECT_FLOAT_EQ(data_y.at({5}), 65.0f);
    // Un-modified elements
    EXPECT_FLOAT_EQ(data_y.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(data_y.at({1}), 1.0f);
    EXPECT_FLOAT_EQ(data_y.at({2}), 2.0f);
}

// =============================================================================
// clear() on View
// =============================================================================

class ViewClearTest : public ::testing::Test {};

TEST_F(ViewClearTest, Clear_NarrowedView_ZerosViewOnlyInBase) {
    TensorWrapper<float> base(NestedData<float>{1, 2, 3, 4, 5});
    auto view = base.narrow(0, 1, 3); // [2, 3, 4]

    view.clear();

    // View sees zeros
    EXPECT_FLOAT_EQ(view.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 0.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 0.0f);

    // Write-through: affected base elements are zero
    EXPECT_FLOAT_EQ(base.at({1}), 0.0f);
    EXPECT_FLOAT_EQ(base.at({2}), 0.0f);
    EXPECT_FLOAT_EQ(base.at({3}), 0.0f);

    // Adjacent base elements untouched
    EXPECT_FLOAT_EQ(base.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(base.at({4}), 5.0f);

    // Shape/stride of view unchanged
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);
}

// =============================================================================
// reshape() requires contiguous: test that non-contiguous throws
// =============================================================================

class ViewReshapeTest : public ::testing::Test {};

TEST_F(ViewReshapeTest, Reshape_NonContiguousView_ThrowsInvalidArgument) {
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});
    auto view = base.narrow(0, 1, 3); // non-contiguous
    EXPECT_FALSE(view.isContiguous());
    EXPECT_THROW(view.reshape({3}), std::invalid_argument);
}

TEST_F(ViewReshapeTest, Reshape_ClonedView_Succeeds) {
    TensorWrapper<float> base(NestedData<float>{0, 1, 2, 3, 4});
    auto view = base.narrow(0, 1, 4); // [1,2,3,4]
    auto cloned = view.clone();

    EXPECT_TRUE(cloned.isContiguous());

    auto reshaped = cloned.reshape({2, 2});

    // --- Shape ---
    EXPECT_EQ(reshaped.getDimensions(), 2u);
    EXPECT_EQ(reshaped.getShape()[0], 2u);
    EXPECT_EQ(reshaped.getShape()[1], 2u);

    // --- Stride ---
    ASSERT_EQ(reshaped.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(reshaped.getStride().getStrideVec()[0], 2u);
    EXPECT_EQ(reshaped.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(reshaped.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(reshaped.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(reshaped.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(reshaped.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(reshaped.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(reshaped.at({1, 1}), 4.0f);
}

// =============================================================================
// transpose() requires contiguous
// =============================================================================

class ViewTransposeTest : public ::testing::Test {};

TEST_F(ViewTransposeTest, Transpose_NonContiguousView_ThrowsInvalidArgument) {
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto view = base.narrow(0, 1, 2); // rows 1,2 – non-contiguous
    EXPECT_FALSE(view.isContiguous());
    EXPECT_THROW(view.transpose(), std::invalid_argument);
}

TEST_F(ViewTransposeTest, Transpose_ClonedView_Succeeds) {
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto view = base.narrow(0, 1, 2); // [[3,4,5],[6,7,8]] shape (2,3)
    auto cloned = view.clone();

    auto transposed = cloned.transpose(); // shape (3,2)

    // --- Shape ---
    EXPECT_EQ(transposed.getDimensions(), 2u);
    EXPECT_EQ(transposed.getShape()[0], 3u);
    EXPECT_EQ(transposed.getShape()[1], 2u);

    // --- Stride ---
    ASSERT_EQ(transposed.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(transposed.getStride().getStrideVec()[0], 2u);
    EXPECT_EQ(transposed.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(transposed.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(transposed.isContiguous());

    // --- Values: transposed [[3,6],[4,7],[5,8]] ---
    EXPECT_FLOAT_EQ(transposed.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(transposed.at({0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(transposed.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(transposed.at({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(transposed.at({2, 0}), 5.0f);
    EXPECT_FLOAT_EQ(transposed.at({2, 1}), 8.0f);
}

// =============================================================================
// matmul() requires contiguous
// =============================================================================

class ViewMatmulTest : public ::testing::Test {};

TEST_F(ViewMatmulTest, Matmul_NonContiguousView_ThrowsInvalidArgument) {
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto view = base.narrow(0, 1, 2); // non-contiguous
    TensorWrapper<float> b(NestedData<float>{{1, 0}, {0, 1}, {0, 0}});
    EXPECT_FALSE(view.isContiguous());
    EXPECT_THROW(view.matmul(b), std::invalid_argument);
}

TEST_F(ViewMatmulTest, Matmul_ClonedView_CorrectFields) {
    // view = [[3,4,5],[6,7,8]] after narrow, then clone => contiguous (2,3)
    TensorWrapper<float> base(
        NestedData<float>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    auto view = base.narrow(0, 1, 2).clone(); // [[3,4,5],[6,7,8]] contiguous
    TensorWrapper<float> rhs(
        NestedData<float>{{1, 0}, {0, 1}, {0, 0}}); // (3,2)

    auto result = view.matmul(rhs);

    // --- Shape ---
    EXPECT_EQ(result.getDimensions(), 2u);
    EXPECT_EQ(result.getShape()[0], 2u);
    EXPECT_EQ(result.getShape()[1], 2u);

    // --- Stride ---
    ASSERT_EQ(result.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(result.getStride().getStrideVec()[0], 2u);
    EXPECT_EQ(result.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(result.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(result.isContiguous());

    // --- Values: [[3,4,5],[6,7,8]] @ [[1,0],[0,1],[0,0]] = [[3,4],[6,7]] ---
    EXPECT_FLOAT_EQ(result.at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(result.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 7.0f);
}

// =============================================================================
// sqrtInPlace / squareInPlace on Views
// =============================================================================

class ViewInPlaceTest : public ::testing::Test {};

TEST_F(ViewInPlaceTest, SqrtInPlace_OnNarrowedView_WriteThrough) {
    TensorWrapper<float> base(NestedData<float>{0, 4, 9, 16, 25});
    auto view = base.narrow(0, 1, 3); // [4, 9, 16]

    view.sqrtInPlace();

    // View values updated
    EXPECT_FLOAT_EQ(view.at({0}), 2.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 3.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 4.0f);

    // Write-through: base reflects changes
    EXPECT_FLOAT_EQ(base.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(base.at({2}), 3.0f);
    EXPECT_FLOAT_EQ(base.at({3}), 4.0f);

    // Unmodified
    EXPECT_FLOAT_EQ(base.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(base.at({4}), 25.0f);

    // Shape/stride/device unchanged
    EXPECT_EQ(view.getDimensions(), 1u);
    EXPECT_EQ(view.getShape()[0], 3u);
    EXPECT_EQ(view.getStride().getStrideVec()[0], 1u);
    EXPECT_EQ(view.getDevice()->getType(), DeviceType::CPU);
    EXPECT_FALSE(view.isContiguous());
}

TEST_F(ViewInPlaceTest, SquareInPlace_OnStepView_WriteThrough) {
    TensorWrapper<float> base(NestedData<float>{1, 2, 3, 4, 5});
    auto view = base.sliceDim(0, 0, 5, 2); // [1, 3, 5] (indices 0,2,4)
    EXPECT_FALSE(view.isContiguous());

    view.squareInPlace();

    // View values
    EXPECT_FLOAT_EQ(view.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(view.at({1}), 9.0f);
    EXPECT_FLOAT_EQ(view.at({2}), 25.0f);

    // Write-through
    EXPECT_FLOAT_EQ(base.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(base.at({2}), 9.0f);
    EXPECT_FLOAT_EQ(base.at({4}), 25.0f);

    // Unmodified (odd indices)
    EXPECT_FLOAT_EQ(base.at({1}), 2.0f);
    EXPECT_FLOAT_EQ(base.at({3}), 4.0f);
}

// =============================================================================
// ones() / zeros() / sameShapeWithValue() on Views
// =============================================================================

class ViewHelperTensorsTest : public ::testing::Test {};

TEST_F(ViewHelperTensorsTest, Ones_FromView_CorrectShapeAndValues) {
    TensorWrapper<float> base(NestedData<float>{1, 2, 3, 4, 5});
    auto view = base.narrow(0, 2, 3); // shape (3,)

    auto ones_tensor = view.ones();

    // --- Shape: same as view ---
    EXPECT_EQ(ones_tensor.getDimensions(), 1u);
    EXPECT_EQ(ones_tensor.getShape()[0], 3u);

    // --- Stride ---
    ASSERT_EQ(ones_tensor.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(ones_tensor.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(ones_tensor.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(ones_tensor.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(ones_tensor.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(ones_tensor.at({1}), 1.0f);
    EXPECT_FLOAT_EQ(ones_tensor.at({2}), 1.0f);
}

TEST_F(ViewHelperTensorsTest, Zeros_FromView_CorrectShapeAndValues) {
    TensorWrapper<float> base(NestedData<float>{1, 2, 3, 4, 5});
    auto view = base.narrow(0, 1, 4); // shape (4,)

    auto zeros_tensor = view.zeros();

    // --- Shape ---
    EXPECT_EQ(zeros_tensor.getDimensions(), 1u);
    EXPECT_EQ(zeros_tensor.getShape()[0], 4u);

    // --- Stride ---
    ASSERT_EQ(zeros_tensor.getStride().getStrideVec().size(), 1u);
    EXPECT_EQ(zeros_tensor.getStride().getStrideVec()[0], 1u);

    // --- Device ---
    EXPECT_EQ(zeros_tensor.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(zeros_tensor.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(zeros_tensor.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(zeros_tensor.at({3}), 0.0f);
}

TEST_F(ViewHelperTensorsTest, SameShapeWithValue_FromView_CorrectFields) {
    TensorWrapper<float> base(
        NestedData<float>{{1, 2}, {3, 4}, {5, 6}});
    auto view = base.narrow(0, 1, 2); // rows 1,2 shape (2,2)

    auto filled = view.sameShapeWithValue(7.0f);

    // --- Shape ---
    EXPECT_EQ(filled.getDimensions(), 2u);
    EXPECT_EQ(filled.getShape()[0], 2u);
    EXPECT_EQ(filled.getShape()[1], 2u);

    // --- Stride ---
    ASSERT_EQ(filled.getStride().getStrideVec().size(), 2u);
    EXPECT_EQ(filled.getStride().getStrideVec()[0], 2u);
    EXPECT_EQ(filled.getStride().getStrideVec()[1], 1u);

    // --- Device ---
    EXPECT_EQ(filled.getDevice()->getType(), DeviceType::CPU);

    // --- isContiguous ---
    EXPECT_TRUE(filled.isContiguous());

    // --- Values ---
    EXPECT_FLOAT_EQ(filled.at({0, 0}), 7.0f);
    EXPECT_FLOAT_EQ(filled.at({1, 1}), 7.0f);
}
