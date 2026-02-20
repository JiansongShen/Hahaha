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

#include "math/ds/TensorShape.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <optional>

class TensorShapeTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

using hahaha::common::u32;
using hahaha::math::TensorShape;

TEST_F(TensorShapeTest, InitWithInitializerList) {
    auto ts1 = TensorShape({1, 2, 3});
    ASSERT_EQ(ts1.getDims().size(), 3);
    ASSERT_EQ(ts1.getDims().at(0), 1);
    ASSERT_EQ(ts1.getDims().at(1), 2);
    ASSERT_EQ(ts1.getDims().at(2), 3);
}

TEST_F(TensorShapeTest, InitWithNoParam) {
    auto ts1 = TensorShape();
    ASSERT_EQ(ts1.getDims().size(), 0);
}

TEST_F(TensorShapeTest, InitWithStdVector) {
    std::vector<size_t> dims = {4, 5, 6, 7};
    auto ts1 = TensorShape(dims);
    ASSERT_EQ(ts1.getDims().size(), 4);
    ASSERT_EQ(ts1.getDims().at(0), 4);
    ASSERT_EQ(ts1.getDims().at(1), 5);
    ASSERT_EQ(ts1.getDims().at(2), 6);
    ASSERT_EQ(ts1.getDims().at(3), 7);
}

TEST_F(TensorShapeTest, InitWithEmptyStdVector_IsScalarShape) {
    std::vector<size_t> dims;
    auto ts = TensorShape(dims);
    ASSERT_EQ(ts.getDims().size(), 0);
    ASSERT_EQ(ts.getTotalSize(), 1);
    ASSERT_EQ(ts.toString(), "()");
}

TEST_F(TensorShapeTest, MoveConstructor) {
    TensorShape ts1({8, 9, 10});
    TensorShape ts2(std::move(ts1));
    ASSERT_EQ(ts1.getDims().size(), 0);
    ASSERT_EQ(ts2.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().at(0), 8);
    ASSERT_EQ(ts2.getDims().at(1), 9);
    ASSERT_EQ(ts2.getDims().at(2), 10);
}

TEST_F(TensorShapeTest, CopyConstructor) {
    TensorShape ts1({11, 12, 13});
    TensorShape ts2(ts1);
    ASSERT_EQ(ts1.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().at(0), 11);
    ASSERT_EQ(ts2.getDims().at(1), 12);
    ASSERT_EQ(ts2.getDims().at(2), 13);
}

TEST_F(TensorShapeTest, AssignWithLValue) {
    TensorShape ts1({14, 15});
    TensorShape ts2;
    ASSERT_EQ(ts2.getDims().size(), 0);
    ts2 = ts1;
    ASSERT_EQ(ts1.getDims().size(), 2);
    ASSERT_EQ(ts2.getDims().size(), 2);
    ASSERT_EQ(ts2.getDims().at(0), 14);
    ASSERT_EQ(ts2.getDims().at(1), 15);
}

TEST_F(TensorShapeTest, AssignWithRValue) {
    TensorShape ts1({16, 17, 18});
    TensorShape ts2;
    ASSERT_EQ(ts2.getDims().size(), 0);
    ts2 = std::move(ts1);
    ASSERT_EQ(ts1.getDims().size(), 0);
    ASSERT_EQ(ts2.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().at(0), 16);
    ASSERT_EQ(ts2.getDims().at(1), 17);
    ASSERT_EQ(ts2.getDims().at(2), 18);
    ts2 = TensorShape({19, 20});
    ASSERT_EQ(ts2.getDims().size(), 2);
    ASSERT_EQ(ts2.getDims().at(0), 19);
    ASSERT_EQ(ts2.getDims().at(1), 20);
}

TEST_F(TensorShapeTest, ComputeSize) {
    TensorShape ts1({2, 3, 4});
    ASSERT_EQ(ts1.getTotalSize(), 24);

    TensorShape ts2({5, 6});
    ASSERT_EQ(ts2.getTotalSize(), 30);

    TensorShape ts3;
    ASSERT_EQ(ts3.getTotalSize(), 1);

    TensorShape ts4({1024, 1024, 1024, 8});
    ASSERT_EQ(ts4.getTotalSize(), 8589934592ULL);
}

TEST_F(TensorShapeTest, DimsAccess) {
    TensorShape ts({2, 3});
    const auto& dims = ts.getDims();
    ASSERT_EQ(dims.size(), 2);
    ASSERT_EQ(dims[0], 2);
    ASSERT_EQ(dims[1], 3);

    ts.getDims()[0] = 5;
    EXPECT_EQ(ts.getDims()[0], 5);
}

TEST_F(TensorShapeTest, ToString) {
    TensorShape ts1({1, 2, 3});
    ASSERT_EQ(ts1.toString(), "(1, 2, 3)");

    TensorShape ts2({4, 5});
    ASSERT_EQ(ts2.toString(), "(4, 5)");

    TensorShape ts3;
    ASSERT_EQ(ts3.toString(), "()");
}

TEST_F(TensorShapeTest, Reverse) {
    TensorShape ts1({1, 2, 3});
    ts1.reverse();
    ASSERT_EQ(ts1.toString(), "(3, 2, 1)");

    TensorShape ts2({4, 5});
    ts2.reverse();
    ASSERT_EQ(ts2.toString(), "(5, 4)");
}

TEST_F(TensorShapeTest, Equality) {
    EXPECT_TRUE(TensorShape({1, 2}) == TensorShape({1, 2}));
    EXPECT_FALSE(TensorShape({1, 2}) == TensorShape({1, 3}));
    EXPECT_TRUE(TensorShape({1, 2}) != TensorShape({1, 3}));
    EXPECT_FALSE(TensorShape({1, 2}) != TensorShape({1, 2}));
    EXPECT_TRUE(TensorShape({1}) != TensorShape({1, 2}));
}

TEST_F(TensorShapeTest, BroadcastShape) {
    // Same shape
    auto res0 =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({2, 3}));
    ASSERT_TRUE(res0.has_value());
    EXPECT_EQ(TensorShape(*res0), TensorShape({2, 3}));

    // Prefix dims
    auto res1 = TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({3}));
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(TensorShape(*res1), TensorShape({2, 3}));

    // Dim 1 on LHS
    auto res1b =
        TensorShape::broadcastShape(TensorShape({1, 3}), TensorShape({2, 3}));
    ASSERT_TRUE(res1b.has_value());
    EXPECT_EQ(TensorShape(*res1b), TensorShape({2, 3}));

    // Dim 1 on RHS
    auto res2 =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({2, 1}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3}));

    // Incompatible
    auto res3 = TensorShape::broadcastShape(TensorShape({2}), TensorShape({3}));
    ASSERT_FALSE(res3.has_value());
}

TEST_F(TensorShapeTest, BroadcastShape_MoreComplex) {
    // (1, 2, 1) and (3, 1, 4) -> (3, 2, 4)
    auto res =
        TensorShape::broadcastShape(TensorShape({1, 2, 1}), TensorShape({3, 1, 4}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({3, 2, 4}));

    // (2, 3, 1) and (1, 3, 5) -> (2, 3, 5)
    auto res2 =
        TensorShape::broadcastShape(TensorShape({2, 3, 1}), TensorShape({1, 3, 5}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3, 5}));
}

TEST_F(TensorShapeTest, EmptyShapeOperations) {
    TensorShape ts;
    ts.reverse(); // Should not crash
    EXPECT_EQ(ts.getDims().size(), 0);
    EXPECT_EQ(ts.getTotalSize(), 1);
    EXPECT_EQ(ts.toString(), "()");
}

TEST_F(TensorShapeTest, MutableDims) {
    TensorShape ts({1, 2});
    std::vector<size_t>& dims = ts.getDims();
    dims.push_back(3);
    EXPECT_EQ(ts.getTotalSize(), 6);
    EXPECT_EQ(ts.toString(), "(1, 2, 3)");
}

TEST_F(TensorShapeTest, ConstDimsAccess) {
    const TensorShape ts({4, 5, 6});
    const std::vector<size_t>& dims = ts.getDims();
    ASSERT_EQ(dims.size(), 3);
    ASSERT_EQ(dims[0], 4);
    ASSERT_EQ(dims[1], 5);
    ASSERT_EQ(dims[2], 6);
}

TEST_F(TensorShapeTest, Inequality) {
    TensorShape ts1({1, 2});
    TensorShape ts2({1, 3});
    TensorShape ts3({1, 2, 3});
    EXPECT_TRUE(ts1 != ts2);
    EXPECT_TRUE(ts1 != ts3);
    EXPECT_FALSE(ts1 != ts1);
}

TEST_F(TensorShapeTest, BroadcastWithScalar) {
    auto res1 = TensorShape::broadcastShape(TensorShape({}), TensorShape({2, 3}));
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(TensorShape(*res1), TensorShape({2, 3}));

    auto res2 = TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, BroadcastIncompatibleRank) {
    auto res =
        TensorShape::broadcastShape(TensorShape({2, 3, 4}), TensorShape({3, 5}));
    // Last dimension 4 and 5 are incompatible
    ASSERT_FALSE(res.has_value());
}

TEST_F(TensorShapeTest, CopyAssignment) {
    TensorShape ts1({1, 2, 3});
    TensorShape ts2;
    ts2 = ts1;
    EXPECT_EQ(ts2, ts1);
}

TEST_F(TensorShapeTest, MoveAssignmentOperator) {
    TensorShape ts1({7, 8});
    TensorShape ts2;
    ts2 = std::move(ts1);
    EXPECT_EQ(ts2.getTotalSize(), 56);
}

TEST_F(TensorShapeTest, DefaultConstructorAndAssignment) {
    TensorShape ts;
    EXPECT_EQ(ts.getDims().size(), 0);
    EXPECT_EQ(ts.getTotalSize(), 1);

    TensorShape ts2({1});
    ts = ts2;
    EXPECT_EQ(ts.getDims().size(), 1);
    EXPECT_EQ(ts.getTotalSize(), 1);
}

// ============================================================================
// Dimension-specific tests: 0D to 3D broadcast
// ============================================================================

TEST_F(TensorShapeTest, Broadcast_0D_ScalarToAnyShape) {
    // 0D scalar () can broadcast to any shape
    TensorShape scalar({});

    // Scalar to 0D
    auto res0 = TensorShape::broadcastShape(scalar, TensorShape({}));
    ASSERT_TRUE(res0.has_value());
    EXPECT_EQ(TensorShape(*res0), TensorShape({}));

    // Scalar to 1D
    auto res1 = TensorShape::broadcastShape(scalar, TensorShape({3}));
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(TensorShape(*res1), TensorShape({3}));

    // Scalar to 2D
    auto res2 = TensorShape::broadcastShape(scalar, TensorShape({2, 3}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3}));

    // Scalar to 3D
    auto res3 = TensorShape::broadcastShape(scalar, TensorShape({2, 2, 2}));
    ASSERT_TRUE(res3.has_value());
    EXPECT_EQ(TensorShape(*res3), TensorShape({2, 2, 2}));
}

TEST_F(TensorShapeTest, Broadcast_1D_VectorToMatrix) {
    // 1D [3] can broadcast to 2D with matching last dim
    auto res = TensorShape::broadcastShape(TensorShape({3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));

    // 1D [2] can broadcast to 2D [2, 2] (dim=1 case)
    auto res2 = TensorShape::broadcastShape(TensorShape({1}), TensorShape({2, 2}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 2}));

    // 1D [3] cannot broadcast to [2, 4] (incompatible)
    auto res3 = TensorShape::broadcastShape(TensorShape({3}), TensorShape({2, 4}));
    ASSERT_FALSE(res3.has_value());
}

TEST_F(TensorShapeTest, Broadcast_2D_MatrixTo3D) {
    // 2D [1, 3] can not broadcast to 3D [2, 3, 4]
    auto res =
        TensorShape::broadcastShape(TensorShape({1, 3}), TensorShape({2, 3, 4}));
    ASSERT_FALSE(res.has_value());

    // 2D [3, 4] can broadcast to 3D [2, 3, 4]
    auto res2 =
        TensorShape::broadcastShape(TensorShape({3, 4}), TensorShape({2, 3, 4}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3, 4}));

    // 2D [2, 3] cannot broadcast to [2, 3, 4] (last dim mismatch: 3 != 4)
    auto res3 =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({2, 3, 4}));
    ASSERT_FALSE(res3.has_value());
}

TEST_F(TensorShapeTest, Broadcast_3D_To3D) {
    // 3D [1, 2, 3] can broadcast to [2, 2, 3]
    auto res =
        TensorShape::broadcastShape(TensorShape({1, 2, 3}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));

    // 3D [2, 1, 3] can broadcast to [2, 2, 3]
    auto res2 =
        TensorShape::broadcastShape(TensorShape({2, 1, 3}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 2, 3}));

    // 3D [2, 2, 1] can broadcast to [2, 2, 3]
    auto res3 =
        TensorShape::broadcastShape(TensorShape({2, 2, 1}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res3.has_value());
    EXPECT_EQ(TensorShape(*res3), TensorShape({2, 2, 3}));

    // 3D [2, 2, 3] cannot broadcast to [2, 2, 4] (last dim mismatch)
    auto res4 =
        TensorShape::broadcastShape(TensorShape({2, 2, 3}), TensorShape({2, 2, 4}));
    ASSERT_FALSE(res4.has_value());
}

TEST_F(TensorShapeTest, Broadcast_ScalarTypes_Distinction) {
    // Test different scalar representations

    // 0D scalar: ()
    TensorShape scalar0D({});
    EXPECT_EQ(scalar0D.getDims().size(), 0);
    EXPECT_EQ(scalar0D.getTotalSize(), 1);

    // 1D scalar: [1]
    TensorShape scalar1D({1});
    EXPECT_EQ(scalar1D.getDims().size(), 1);
    EXPECT_EQ(scalar1D.getTotalSize(), 1);

    // 2D scalar: [1, 1]
    TensorShape scalar2D({1, 1});
    EXPECT_EQ(scalar2D.getDims().size(), 2);
    EXPECT_EQ(scalar2D.getTotalSize(), 1);

    // 3D scalar: [1, 1, 1]
    TensorShape scalar3D({1, 1, 1});
    EXPECT_EQ(scalar3D.getDims().size(), 3);
    EXPECT_EQ(scalar3D.getTotalSize(), 1);

    // All can broadcast to [2, 2, 2]
    TensorShape target({2, 2, 2});

    auto res0 = TensorShape::broadcastShape(scalar0D, target);
    ASSERT_TRUE(res0.has_value());
    EXPECT_EQ(TensorShape(*res0), target);

    auto res1 = TensorShape::broadcastShape(scalar1D, target);
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(TensorShape(*res1), target);

    auto res2 = TensorShape::broadcastShape(scalar2D, target);
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), target);

    auto res3 = TensorShape::broadcastShape(scalar3D, target);
    ASSERT_TRUE(res3.has_value());
    EXPECT_EQ(TensorShape(*res3), target);
}

// ============================================================================
// Broadcast Tests - All 16 Dimension Combinations (0D, 1D, 2D, 3D)
// ============================================================================

TEST_F(TensorShapeTest, Broadcast_0Dvs0D_SameShape) {
    auto res = TensorShape::broadcastShape(TensorShape({}), TensorShape({}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({}));
}

TEST_F(TensorShapeTest, Broadcast_0Dvs1D_ScalarToVector) {
    auto res = TensorShape::broadcastShape(TensorShape({}), TensorShape({3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({3}));
}

TEST_F(TensorShapeTest, Broadcast_0Dvs2D_ScalarToMatrix) {
    auto res = TensorShape::broadcastShape(TensorShape({}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_0Dvs3D_ScalarToTensor) {
    auto res = TensorShape::broadcastShape(TensorShape({}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_1Dvs0D_VectorToScalar) {
    auto res = TensorShape::broadcastShape(TensorShape({3}), TensorShape({}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({3}));
}

TEST_F(TensorShapeTest, Broadcast_1Dvs1D_SameShape) {
    auto res = TensorShape::broadcastShape(TensorShape({3}), TensorShape({3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({3}));
}

TEST_F(TensorShapeTest, Broadcast_1Dvs1D_Incompatible) {
    auto res = TensorShape::broadcastShape(TensorShape({2}), TensorShape({3}));
    ASSERT_FALSE(res.has_value());
}

TEST_F(TensorShapeTest, Broadcast_1Dvs2D_VectorToMatrix_Compatible) {
    auto res = TensorShape::broadcastShape(TensorShape({3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_1Dvs2D_VectorToMatrix_Incompatible) {
    auto res = TensorShape::broadcastShape(TensorShape({2}), TensorShape({2, 3}));
    ASSERT_FALSE(res.has_value());
}

TEST_F(TensorShapeTest, Broadcast_1Dvs3D_VectorToTensor_Compatible) {
    auto res = TensorShape::broadcastShape(TensorShape({3}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_2Dvs0D_MatrixToScalar) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_2Dvs1D_MatrixToVector_Compatible) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_2Dvs2D_SameShape) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_2Dvs2D_DimOneBroadcasting) {
    auto res = TensorShape::broadcastShape(TensorShape({1, 3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_2Dvs3D_MatrixToTensor_Compatible) {
    auto res =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_2Dvs3D_MatrixToTensor_Incompatible) {
    auto res =
        TensorShape::broadcastShape(TensorShape({2, 2}), TensorShape({2, 2, 3}));
    ASSERT_FALSE(res.has_value());
}

TEST_F(TensorShapeTest, Broadcast_3Dvs0D_TensorToScalar) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 2, 3}), TensorShape({}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_3Dvs1D_TensorToVector_Compatible) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 2, 3}), TensorShape({3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_3Dvs2D_TensorToMatrix_Compatible) {
    auto res =
        TensorShape::broadcastShape(TensorShape({2, 2, 3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_3Dvs3D_SameShape) {
    auto res =
        TensorShape::broadcastShape(TensorShape({2, 2, 3}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, Broadcast_3Dvs3D_DimOneBroadcasting) {
    auto res =
        TensorShape::broadcastShape(TensorShape({1, 2, 3}), TensorShape({2, 2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 2, 3}));
}

TEST_F(TensorShapeTest, EdgeCasesWithLargeDimensions) {
    // Test with large dimensions
    TensorShape large({1000000, 1000000});
    EXPECT_EQ(large.getTotalSize(), 1000000000000ULL);

    // Test with very small dimensions
    TensorShape tiny({1, 1, 1});
    EXPECT_EQ(tiny.getTotalSize(), 1);
}

TEST_F(TensorShapeTest, ReverseOnEmptyShape) {
    TensorShape empty;
    empty.reverse(); // Should not crash
    EXPECT_EQ(empty.getDims().size(), 0);
}

TEST_F(TensorShapeTest, ReverseSingleDimension) {
    TensorShape single({5});
    single.reverse();
    EXPECT_EQ(single.getDims().size(), 1);
    EXPECT_EQ(single.getDims()[0], 5);
}

TEST_F(TensorShapeTest, ReverseTwoDimensions) {
    TensorShape two({3, 4});
    two.reverse();
    EXPECT_EQ(two.getDims().size(), 2);
    EXPECT_EQ(two.getDims()[0], 4);
    EXPECT_EQ(two.getDims()[1], 3);
}

TEST_F(TensorShapeTest, BroadcastWithVeryDifferentRanks) {
    // Test broadcasting with significantly different ranks
    auto res1 =
        TensorShape::broadcastShape(TensorShape({}), TensorShape({2, 3, 4, 5}));
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(TensorShape(*res1), TensorShape({2, 3, 4, 5}));

    auto res2 =
        TensorShape::broadcastShape(TensorShape({2, 3, 4, 5}), TensorShape({}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3, 4, 5}));
}

TEST_F(TensorShapeTest, InvalidBroadcastCombinations) {
    // Test various combinations that should fail
    auto res1 =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({3, 4}));
    ASSERT_FALSE(res1.has_value());

    auto res2 =
        TensorShape::broadcastShape(TensorShape({2, 3, 4}), TensorShape({2, 3, 5}));
    ASSERT_FALSE(res2.has_value());

    auto res3 = TensorShape::broadcastShape(TensorShape({2}), TensorShape({3}));
    ASSERT_FALSE(res3.has_value());
}

TEST_F(TensorShapeTest, GetDimsMutableAccess) {
    TensorShape ts({1, 2, 3});
    std::vector<size_t>& mutableDims = ts.getDims();
    mutableDims[0] = 10;
    EXPECT_EQ(ts.getDims()[0], 10);
}
