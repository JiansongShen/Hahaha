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

#include "math/ds/NestedData.h"

#include <gtest/gtest.h>

using hahaha::math::NestedData;

class NestedDataTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(NestedDataTest, InitializeViaInitializerList) {
    NestedData<int> nd({1, 2, 3, 4, 5});
    ASSERT_EQ(nd.getFlatData().size(), 5);
    ASSERT_EQ(nd.getShapeVecRef().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 5);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getFlatData().at(2), 3);
    ASSERT_EQ(nd.getFlatData().at(3), 4);
    ASSERT_EQ(nd.getFlatData().at(4), 5);
}

TEST_F(NestedDataTest, InitializeViaNestedInitializerList1) {
    NestedData<int> nd = {
        NestedData<int>({1, 2, 3}),
        NestedData<int>({4, 5, 6}),
        NestedData<int>({7, 8, 9}),
    };
    ASSERT_EQ(nd.getFlatData().size(), 9);
    ASSERT_EQ(nd.getShapeVecRef().size(), 2);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 3);
    ASSERT_EQ(nd.getShapeVecRef().at(1), 3);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getFlatData().at(2), 3);
    ASSERT_EQ(nd.getFlatData().at(3), 4);
    ASSERT_EQ(nd.getFlatData().at(4), 5);
    ASSERT_EQ(nd.getFlatData().at(5), 6);
    ASSERT_EQ(nd.getFlatData().at(6), 7);
    ASSERT_EQ(nd.getFlatData().at(7), 8);
    ASSERT_EQ(nd.getFlatData().at(8), 9);
}

TEST_F(NestedDataTest, InitializeViaNestedInitializerList2) {
    NestedData<int> nd = {
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    };
    ASSERT_EQ(nd.getFlatData().size(), 8);
    ASSERT_EQ(nd.getShapeVecRef().size(), 3);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 2);
    ASSERT_EQ(nd.getShapeVecRef().at(1), 2);
    ASSERT_EQ(nd.getShapeVecRef().at(2), 2);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getFlatData().at(2), 3);
    ASSERT_EQ(nd.getFlatData().at(3), 4);
    ASSERT_EQ(nd.getFlatData().at(4), 5);
    ASSERT_EQ(nd.getFlatData().at(5), 6);
    ASSERT_EQ(nd.getFlatData().at(6), 7);
    ASSERT_EQ(nd.getFlatData().at(7), 8);
}

TEST_F(NestedDataTest, InitializeViaNestedInitializerList3) {
    NestedData<int> nestedData({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });

    ASSERT_EQ(nestedData.getFlatData().size(), 8);
    ASSERT_EQ(nestedData.getShapeVecRef().size(), 3);
    ASSERT_EQ(nestedData.getShapeVecRef().at(0), 2);
    ASSERT_EQ(nestedData.getShapeVecRef().at(1), 2);
    ASSERT_EQ(nestedData.getShapeVecRef().at(2), 2);
    ASSERT_EQ(nestedData.getFlatData().at(0), 1);
    ASSERT_EQ(nestedData.getFlatData().at(1), 2);
    ASSERT_EQ(nestedData.getFlatData().at(2), 3);
    ASSERT_EQ(nestedData.getFlatData().at(3), 4);
    ASSERT_EQ(nestedData.getFlatData().at(4), 5);
    ASSERT_EQ(nestedData.getFlatData().at(5), 6);
    ASSERT_EQ(nestedData.getFlatData().at(6), 7);
    ASSERT_EQ(nestedData.getFlatData().at(7), 8);
}

TEST_F(NestedDataTest, InitializeWithEmptyList) {
    ASSERT_NO_THROW(NestedData<int> nd({}));
    NestedData<int> nd({});
    ASSERT_EQ(nd.getFlatData().size(), 0);
    ASSERT_EQ(nd.getShapeVecRef().size(), 0);
}

TEST_F(NestedDataTest, SingleValueConstruction) {
    NestedData<int> nd(42);
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getFlatData()[0], 42);
    ASSERT_EQ(nd.getShapeVecRef().size(), 0);
}

TEST_F(NestedDataTest, InconsistentShapesThrows) {
    EXPECT_THROW((NestedData<int>{{1, 2, 3}, {4, 5}}), std::invalid_argument);
}

TEST_F(NestedDataTest, Getters) {
    NestedData<int> nd({1, 2});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    ASSERT_EQ(nd.getFlatData().at(0), 1);
    ASSERT_EQ(nd.getFlatData().at(1), 2);
    ASSERT_EQ(nd.getShapeVecRef().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 2);
}

TEST_F(NestedDataTest, DifferentTypes) {
    NestedData<double> nd = {{1.1, 2.2}, {3.3, 4.4}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    ASSERT_DOUBLE_EQ(nd.getFlatData().at(0), 1.1);
    ASSERT_EQ(nd.getShapeVecRef().size(), 2);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 2);
    ASSERT_EQ(nd.getShapeVecRef().at(1), 2);
}

TEST_F(NestedDataTest, DeepNesting) {
    NestedData<int> nd = {{{{1}}}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().size(), 4);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 1);
    ASSERT_EQ(nd.getShapeVecRef().at(1), 1);
    ASSERT_EQ(nd.getShapeVecRef().at(2), 1);
    ASSERT_EQ(nd.getShapeVecRef().at(3), 1);
}

TEST_F(NestedDataTest, LargeList) {
    NestedData<int> nd = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ASSERT_EQ(nd.getFlatData().size(), 10);
    ASSERT_EQ(nd.getShapeVecRef().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().at(0), 10);
}

TEST_F(NestedDataTest, NestedEmptyListThrows) {
    // Current implementation: if one list is empty, its shape is empty.
    // If we have {{}, {1}}, the first has shape (), second has shape (1).
    // This should throw inconsistent shapes.
    EXPECT_THROW((NestedData<int>{{}, {1}}), std::invalid_argument);
}

TEST_F(NestedDataTest, ScalarImplicitConversion) {
    NestedData<int> nd = 5;
    EXPECT_EQ(nd.getFlatData().size(), 1);
    EXPECT_EQ(nd.getFlatData()[0], 5);
}

TEST_F(NestedDataTest, InconsistentShapesDeeper) {
    EXPECT_THROW((NestedData<int>{{{1, 2}, {3, 4}}, {{5, 6, 7}, {8, 9, 10}}}),
                 std::invalid_argument);
}

TEST_F(NestedDataTest, SingleElementList) {
    NestedData<int> nd = {{1}};
    ASSERT_EQ(nd.getShapeVecRef().size(), 2);
    ASSERT_EQ(nd.getShapeVecRef()[0], 1);
    ASSERT_EQ(nd.getShapeVecRef()[1], 1);
}

// ============================================================================
// Dimension-specific tests: 0D to 3D initialization
// ============================================================================

TEST_F(NestedDataTest, ZeroDimension_Scalar) {
    // 0D scalar: SingleValueConstruction
    NestedData<int> nd(42);
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getFlatData()[0], 42);
    ASSERT_EQ(nd.getShapeVecRef().size(), 0);
}

TEST_F(NestedDataTest, OneDimension_Vector) {
    // 1D: [1, 2, 3]
    NestedData<int> nd({1, 2, 3});
    ASSERT_EQ(nd.getFlatData().size(), 3);
    ASSERT_EQ(nd.getShapeVecRef().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef()[0], 3);
    ASSERT_EQ(nd.getFlatData()[0], 1);
    ASSERT_EQ(nd.getFlatData()[2], 3);
}

TEST_F(NestedDataTest, OneDimension_SingleElement) {
    // 1D with single element: [1] (different from scalar)
    NestedData<int> nd{1};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef()[0], 1);
    ASSERT_EQ(nd.getFlatData()[0], 1);
}

TEST_F(NestedDataTest, TwoDimension_Matrix) {
    // 2D: {{1, 2}, {3, 4}}
    NestedData<int> nd = {{1, 2}, {3, 4}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    ASSERT_EQ(nd.getShapeVecRef().size(), 2);
    ASSERT_EQ(nd.getShapeVecRef()[0], 2);
    ASSERT_EQ(nd.getShapeVecRef()[1], 2);
    ASSERT_EQ(nd.getFlatData()[0], 1);
    ASSERT_EQ(nd.getFlatData()[3], 4);
}

TEST_F(NestedDataTest, TwoDimension_SingleElement) {
    // 2D with single element: {{1}} (high-dimensional scalar)
    NestedData<int> nd = {{1}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().size(), 2);
    ASSERT_EQ(nd.getShapeVecRef()[0], 1);
    ASSERT_EQ(nd.getShapeVecRef()[1], 1);
    ASSERT_EQ(nd.getFlatData()[0], 1);
}

TEST_F(NestedDataTest, ThreeDimension_Tensor) {
    // 3D: {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}
    NestedData<int> nd = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    ASSERT_EQ(nd.getFlatData().size(), 8);
    ASSERT_EQ(nd.getShapeVecRef().size(), 3);
    ASSERT_EQ(nd.getShapeVecRef()[0], 2);
    ASSERT_EQ(nd.getShapeVecRef()[1], 2);
    ASSERT_EQ(nd.getShapeVecRef()[2], 2);
    ASSERT_EQ(nd.getFlatData()[0], 1);
    ASSERT_EQ(nd.getFlatData()[7], 8);
}

TEST_F(NestedDataTest, ThreeDimension_SingleElement) {
    // 3D with single element: {{{1}}} (3D scalar)
    NestedData<int> nd = {{{1}}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShapeVecRef().size(), 3);
    ASSERT_EQ(nd.getShapeVecRef()[0], 1);
    ASSERT_EQ(nd.getShapeVecRef()[1], 1);
    ASSERT_EQ(nd.getShapeVecRef()[2], 1);
    ASSERT_EQ(nd.getFlatData()[0], 1);
}

TEST_F(NestedDataTest, ThreeDimension_IrregularShape) {
    // 3D with different sizes in last dimension: {{{1, 2}, {3}}, {{4, 5}, {6}}}
    // This should fail validation
    EXPECT_THROW((NestedData<int>{{{{1, 2}, {3}}, {{4, 5}, {6}}}}),
                 std::invalid_argument);
}

TEST_F(NestedDataTest, EmptyNestedDataConstruction) {
    // Test construction with completely empty nested data
    NestedData<int> nd({});
    ASSERT_EQ(nd.getFlatData().size(), 0);
    ASSERT_EQ(nd.getShapeVecRef().size(), 0);
}

TEST_F(NestedDataTest, ConsistencyCheckInConstructor) {
    // Additional test for consistency checking
    EXPECT_THROW((NestedData<int>{{1, 2, 3}, {4, 5}}), std::invalid_argument);
    EXPECT_THROW((NestedData<int>{{{1, 2}}, {{3, 4, 5}}}), std::invalid_argument);
}

TEST_F(NestedDataTest, ComplexNestedStructure) {
    // Test with more complex nested structures
    NestedData<int> nd = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    ASSERT_EQ(nd.getFlatData().size(), 12);
    ASSERT_EQ(nd.getShapeVecRef().size(), 3);
    ASSERT_EQ(nd.getShapeVecRef()[0], 3); // 3 major groups
    ASSERT_EQ(nd.getShapeVecRef()[1], 2); // 2 rows in each group
    ASSERT_EQ(nd.getShapeVecRef()[2], 2); // 2 elements in each row
}

TEST_F(NestedDataTest, GettersAccessors) {
    // Test the const getter accessors
    const NestedData<int> nd({1, 2, 3, 4});
    const auto& flatData = nd.getFlatData();
    const auto& shape = nd.getShapeVecRef();

    ASSERT_EQ(flatData.size(), 4);
    ASSERT_EQ(shape.size(), 1);
    ASSERT_EQ(shape[0], 4);
}
