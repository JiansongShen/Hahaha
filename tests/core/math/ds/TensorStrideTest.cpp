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

#include "math/ds/TensorStride.h"

#include <gtest/gtest.h>

#include "math/ds/TensorShape.h"

using hahaha::math::TensorShape;
using hahaha::math::TensorStride;

class TensorStrideTest : public ::testing::Test {};

TEST_F(TensorStrideTest, StrideFromVector1D) {
    std::vector<size_t> dims = {5};
    TensorStride stride(dims);
    ASSERT_EQ(stride.getStrideSize(), 1);
    ASSERT_EQ(stride[0], 1);
}

TEST_F(TensorStrideTest, StrideFromVector) {
    std::vector<size_t> dims = {2, 3, 4};
    TensorStride stride(dims);
    ASSERT_EQ(stride.getStrideSize(), 3);
    EXPECT_EQ(stride[0], 12);
    EXPECT_EQ(stride[1], 4);
    EXPECT_EQ(stride[2], 1);
}

TEST_F(TensorStrideTest, StrideFromShape) {
    TensorShape shape({2, 3});
    TensorStride stride(shape);
    EXPECT_EQ(stride[0], 3);
    EXPECT_EQ(stride[1], 1);
}

TEST_F(TensorStrideTest, ToString) {
    TensorStride stride(std::vector<size_t>{2, 3});
    EXPECT_EQ(stride.toString(), "[3, 1]");

    TensorStride empty;
    EXPECT_EQ(empty.toString(), "[]");
}

TEST_F(TensorStrideTest, Reverse) {
    TensorStride stride(std::vector<size_t>{2, 3});
    stride.reverse();
    EXPECT_EQ(stride[0], 1);
    EXPECT_EQ(stride[1], 3);
}

TEST_F(TensorStrideTest, AtAccess) {
    TensorStride stride(std::vector<size_t>{2, 3});
    EXPECT_EQ(stride.at(0), 3);
    EXPECT_THROW((void) stride.at(2), std::out_of_range);

    const TensorStride cstride(std::vector<size_t>{2, 3});
    EXPECT_EQ(cstride.at(1), 1);
    EXPECT_THROW((void) cstride.at(2), std::out_of_range);
}

TEST_F(TensorStrideTest, MutableGetStrides) {
    TensorStride stride(std::vector<size_t>{2, 3});
    stride.getStrideVec()[0] = 10;
    EXPECT_EQ(stride[0], 10);

    const auto& constStrides = stride.getStrideVec();
    EXPECT_EQ(constStrides[0], 10);
}

TEST_F(TensorStrideTest, TemplateConstructorDifferentTypes) {
    std::vector<int> dimsInt = {2, 3};
    TensorStride s1(dimsInt);
    EXPECT_EQ(s1[0], 3);

    std::vector<hahaha::common::u32> dimsU32 = {4, 5};
    TensorStride s2(dimsU32);
    EXPECT_EQ(s2[0], 5);
}

TEST_F(TensorStrideTest, StrideFromEmptyVector) {
    std::vector<size_t> dims = {};
    TensorStride stride(dims);
    EXPECT_EQ(stride.getStrideSize(), 0);
    EXPECT_EQ(stride.toString(), "[]");
}

TEST_F(TensorStrideTest, AtAccessConst) {
    const TensorStride stride(std::vector<size_t>{2, 3, 4});
    EXPECT_EQ(stride.at(0), 12);
    EXPECT_EQ(stride.at(1), 4);
    EXPECT_EQ(stride.at(2), 1);
    EXPECT_THROW((void) stride.at(3), std::out_of_range);
}

TEST_F(TensorStrideTest, NonConstAtAccess) {
    TensorStride stride(std::vector<size_t>{2, 3});
    EXPECT_EQ(stride.at(0), 3);
}

TEST_F(TensorStrideTest, ConstOperatorIndex) {
    const TensorStride stride(std::vector<size_t>{5, 6});
    EXPECT_EQ(stride[0], 6);
    EXPECT_EQ(stride[1], 1);
}

TEST_F(TensorStrideTest, NonConstOperatorIndex) {
    TensorStride stride(std::vector<size_t>{7, 8});
    EXPECT_EQ(stride[0], 8);
}

TEST_F(TensorStrideTest, GetStridesConst) {
    const TensorStride stride(std::vector<size_t>{2, 2});
    const std::vector<size_t>& s = stride.getStrideVec();
    ASSERT_EQ(s.size(), 2);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 1);
}

TEST_F(TensorStrideTest, ToStringBranches) {
    EXPECT_EQ(TensorStride().toString(), "[]");
    EXPECT_EQ(TensorStride(std::vector<size_t>{5}).toString(), "[1]");
    EXPECT_EQ(TensorStride(std::vector<size_t>{2, 3}).toString(), "[3, 1]");
}

TEST_F(TensorStrideTest, StrideFromVectorBranches) {
    TensorStride s1((std::vector<size_t>{}));
    EXPECT_EQ(s1.getStrideSize(), 0);

    TensorStride s2(std::vector<size_t>{10});
    TensorStride s3(std::vector<size_t>{2, 5});
}

// ============================================================================
// Dimension-specific tests: 0D to 3D stride
// ============================================================================

TEST_F(TensorStrideTest, ZeroDimension_EmptyStride) {
    // 0D scalar has empty stride
    std::vector<size_t> dims = {};
    TensorStride stride(dims);
    EXPECT_EQ(stride.getStrideSize(), 0);
    EXPECT_EQ(stride.toString(), "[]");
}

TEST_F(TensorStrideTest, OneDimension_Stride) {
    // 1D: [5] -> stride [1]
    std::vector<size_t> dims = {5};
    TensorStride stride(dims);
    EXPECT_EQ(stride.getStrideSize(), 1);
    EXPECT_EQ(stride[0], 1);
    EXPECT_EQ(stride.toString(), "[1]");
}

TEST_F(TensorStrideTest, TwoDimension_Stride) {
    // 2D: [2, 3] -> stride [3, 1]
    std::vector<size_t> dims = {2, 3};
    TensorStride stride(dims);
    EXPECT_EQ(stride.getStrideSize(), 2);
    EXPECT_EQ(stride[0], 3); // stride for first dimension
    EXPECT_EQ(stride[1], 1); // stride for second dimension
    EXPECT_EQ(stride.toString(), "[3, 1]");
}

TEST_F(TensorStrideTest, ThreeDimension_Stride) {
    // 3D: [2, 3, 4] -> stride [12, 4, 1]
    std::vector<size_t> dims = {2, 3, 4};
    TensorStride stride(dims);
    EXPECT_EQ(stride.getStrideSize(), 3);
    EXPECT_EQ(stride[0], 12); // stride for first dimension: 3*4 = 12
    EXPECT_EQ(stride[1], 4);  // stride for second dimension: 4
    EXPECT_EQ(stride[2], 1);  // stride for third dimension: 1
}

TEST_F(TensorStrideTest, ThreeDimension_FromShape) {
    // 3D: [2, 2, 2] -> stride [4, 2, 1]
    TensorShape shape({2, 2, 2});
    TensorStride stride(shape);
    EXPECT_EQ(stride.getStrideSize(), 3);
    EXPECT_EQ(stride[0], 4); // 2*2 = 4
    EXPECT_EQ(stride[1], 2); // 2
    EXPECT_EQ(stride[2], 1); // 1
}

TEST_F(TensorStrideTest, SingleElementShapes) {
    // Test single element shapes: [1], [1, 1], [1, 1, 1]

    // 1D: [1] -> stride [1]
    TensorStride stride1D({1});
    EXPECT_EQ(stride1D.getStrideSize(), 1);
    EXPECT_EQ(stride1D[0], 1);

    // 2D: [1, 1] -> stride [1, 1]
    TensorStride stride2D({1, 1});
    EXPECT_EQ(stride2D.getStrideSize(), 2);
    EXPECT_EQ(stride2D[0], 1);
    EXPECT_EQ(stride2D[1], 1);

    // 3D: [1, 1, 1] -> stride [1, 1, 1]
    TensorStride stride3D({1, 1, 1});
    EXPECT_EQ(stride3D.getStrideSize(), 3);
    EXPECT_EQ(stride3D[0], 1);
    EXPECT_EQ(stride3D[1], 1);
    EXPECT_EQ(stride3D[2], 1);
}

TEST_F(TensorStrideTest, Reverse_AllDimensions) {
    // Test reverse for 0D, 1D, 2D, 3D

    // 1D reverse
    TensorStride stride1D({5});
    stride1D.reverse();
    EXPECT_EQ(stride1D[0], 1);

    // 2D reverse
    TensorStride stride2D({2, 3});
    stride2D.reverse();
    EXPECT_EQ(stride2D[0], 1); // becomes [1, 3]
    EXPECT_EQ(stride2D[1], 3);

    // 3D reverse
    TensorStride stride3D({2, 3, 4});
    stride3D.reverse();
    EXPECT_EQ(stride3D[0], 1); // becomes [1, 4, 12]
    EXPECT_EQ(stride3D[1], 4);
    EXPECT_EQ(stride3D[2], 12);
}

TEST_F(TensorStrideTest, LargeDimensions) {
    // Test with larger dimensions
    std::vector<size_t> dims = {1000, 1000};
    TensorStride stride(dims);
    EXPECT_EQ(stride[0], 1000); // stride for first dimension
    EXPECT_EQ(stride[1], 1);    // stride for second dimension
}

TEST_F(TensorStrideTest, BoundaryAccess) {
    TensorStride stride(std::vector<size_t>{2, 3});

    // Valid access
    EXPECT_EQ(stride[0], 3);
    EXPECT_EQ(stride[1], 1);

    // Out of bounds access with at() method
    EXPECT_THROW((void) stride.at(2), std::out_of_range);

    const TensorStride constStride(std::vector<size_t>{2, 3});
    EXPECT_THROW((void) constStride.at(2), std::out_of_range);
}

TEST_F(TensorStrideTest, DefaultConstructor) {
    TensorStride defaultStride;
    EXPECT_EQ(defaultStride.getStrideSize(), 0);
    EXPECT_EQ(defaultStride.toString(), "[]");
}

TEST_F(TensorStrideTest, FromDifferentTypes) {
    // Test with different numeric types
    std::vector<int> intDims = {2, 3};
    TensorStride strideInt(intDims);
    EXPECT_EQ(strideInt[0], 3);
    EXPECT_EQ(strideInt[1], 1);

    std::vector<long> longDims = {3, 4};
    TensorStride strideLong(longDims);
    EXPECT_EQ(strideLong[0], 4);
    EXPECT_EQ(strideLong[1], 1);
}

TEST_F(TensorStrideTest, GetStridesNonConstAccess) {
    TensorStride stride(std::vector<size_t>{2, 3});
    auto& stridesRef = stride.getStrideVec();
    stridesRef[0] = 10;

    EXPECT_EQ(stride[0], 10);
    EXPECT_EQ(stride.getStrideVec()[0], 10);
}
