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
    auto res1 =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({3}));
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
    auto res = TensorShape::broadcastShape(TensorShape({1, 2, 1}),
                                           TensorShape({3, 1, 4}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({3, 2, 4}));

    // (2, 3, 1) and (1, 3, 5) -> (2, 3, 5)
    auto res2 = TensorShape::broadcastShape(TensorShape({2, 3, 1}),
                                            TensorShape({1, 3, 5}));
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
    auto res1 =
        TensorShape::broadcastShape(TensorShape({}), TensorShape({2, 3}));
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(TensorShape(*res1), TensorShape({2, 3}));

    auto res2 =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({}));
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(TensorShape(*res2), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, BroadcastIncompatibleRank) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 3, 4}),
                                           TensorShape({3, 5}));
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
