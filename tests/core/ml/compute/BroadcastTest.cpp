// Copyright (c) 2025 - 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include <gtest/gtest.h>

#include "../../../../core/include/public/Tensor.h"

using hahaha::Tensor;
using hahaha::math::NestedData;

class BroadcastTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

TEST_F(BroadcastTest, BroadcastTo_LeadingDims) {
    // Shape {3} -> {2, 3}
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    std::vector<size_t> targetShape = {2, 3};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    EXPECT_EQ(b.getShape(), targetShape);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(b.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(b.at({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at({1, 1}), 2.0f);
    EXPECT_FLOAT_EQ(b.at({1, 2}), 3.0f);
}

TEST_F(BroadcastTest, BroadcastTo_Scalar) {
    // Shape {1} -> {2, 2}
    Tensor<float> a(5.0f); // Scalar tensor (actually shape {1} or {})
    std::vector<size_t> targetShape = {2, 2};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    EXPECT_EQ(b.getShape(), targetShape);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(b.at({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(b.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(b.at({1, 1}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_LeadingDims) {
    // Shape {3} -> {2, 3}
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 3};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    // Sum all elements to create a scalar loss
    // L = sum(b) = sum(a_broadcasted)
    // b = [[a0, a1, a2], [a0, a1, a2]]
    // L = 2*a0 + 2*a1 + 2*a2
    // dL/da0 = 2, dL/da1 = 2, dL/da2 = 2

    // We don't have a sum reduction operation exposed as a node easily here
    // unless we use multiple ops, but let's assume backward from ones
    // which simulates dL/db = 1

    b.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape().size(), 1);
    EXPECT_EQ(a.grad().getShape()[0], 3);

    EXPECT_FLOAT_EQ(a.grad().at({0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({2}), 2.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_Scalar) {
    // Shape {1} -> {2, 2}
    Tensor<float> a(3.0f); // Size 1
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 2};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    // dL/db = 1 everywhere
    // dL/da = sum(dL/db) = 1+1+1+1 = 4
    b.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_FLOAT_EQ(a.grad().at({}), 4.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_InnerDims) {
    // Shape {3, 1} -> {3, 2}
    // This tests broadcasting on a non-leading dimension
    Tensor<float> a(NestedData<float>{{1.0f}, {2.0f}, {3.0f}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {3, 2};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    // b = [[1, 1], [2, 2], [3, 3]]
    // dL/db = 1 everywhere
    // dL/da[0] = dL/db[0,0] + dL/db[0,1] = 2

    b.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape()[0], 3);
    EXPECT_EQ(a.grad().getShape()[1], 1);

    EXPECT_FLOAT_EQ(a.grad().at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({2, 0}), 2.0f);
}

TEST_F(BroadcastTest, Broadcast_Error_Mismatch) {
    Tensor<float> a(NestedData<float>{1.0f, 2.0f}); // Shape {2}
    std::vector<size_t> targetShape = {3};          // Cannot broadcast 2 to 3

    EXPECT_THROW(
        { hahaha::ml::broadcast(a.getComputeNode(), targetShape); },
        std::runtime_error);
}

TEST_F(BroadcastTest, Broadcast_Gradient_Multiple_LeadingDims) {
    // Shape {3} -> {2, 4, 3}
    // Tests that multiple leading dimensions are correctly summed out
    // Corresponds to the fix in the second while loop (while (resIdx >= 0))
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 4, 3};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    b.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape().size(), 1);
    EXPECT_EQ(a.grad().getShape()[0], 3);

    // Each element broadcasted 2 * 4 = 8 times
    EXPECT_FLOAT_EQ(a.grad().at({0}), 8.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1}), 8.0f);
    EXPECT_FLOAT_EQ(a.grad().at({2}), 8.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_Identity_Loop_Check) {
    // Shape {2, 2} -> {2, 2}
    // Tests the fix for the infinite loop when dimensions match
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 2};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    b.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape(), targetShape);

    EXPECT_FLOAT_EQ(a.grad().at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad().at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 1}), 1.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_NonUniform_LeadingDims) {
    // Shape {3} -> {2, 3}, but make upstream gradient non-uniform via mul.
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 3};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    Tensor<float> w(NestedData<float>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto c = b * w;
    c.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape().size(), 1);
    EXPECT_EQ(a.grad().getShape()[0], 3);

    // For leading-dim broadcast, grad is summed over axis 0:
    // [[1,2,3],[4,5,6]] -> [5,7,9]
    EXPECT_FLOAT_EQ(a.grad().at({0}), 5.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1}), 7.0f);
    EXPECT_FLOAT_EQ(a.grad().at({2}), 9.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_NonUniform_InnerDims) {
    // Shape {3, 1} -> {3, 2}, make upstream gradient non-uniform via mul.
    Tensor<float> a(NestedData<float>{{1.0f}, {2.0f}, {3.0f}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {3, 2};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    Tensor<float> w(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    auto c = b * w;
    c.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape().size(), 2);
    EXPECT_EQ(a.grad().getShape()[0], 3);
    EXPECT_EQ(a.grad().getShape()[1], 1);

    // Sum over inner axis: [1+2, 3+4, 5+6] = [3, 7, 11], keepDims => (3,1)
    EXPECT_FLOAT_EQ(a.grad().at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(a.grad().at({2, 0}), 11.0f);
}

TEST_F(BroadcastTest, Broadcast_GradFun_NullGrad_DoesNothing) {
    // Directly call gradFun without running backward() so res->getGrad() is
    // null.
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 3};

    auto broadcastedNode =
        hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    ASSERT_NE(broadcastedNode->getGradFun(), nullptr);

    EXPECT_NO_THROW(broadcastedNode->getGradFun()());
    EXPECT_TRUE(a.grad().isEmpty());
}

TEST_F(BroadcastTest, Broadcast_Error_TargetNotBroadcastResult_ThrowsRuntimeError) {
    // src shape {2,3}, target shape {1,3} is NOT a valid "broadcast-to",
    // but the two shapes are broadcast-compatible (broadcast result would be
    // {2,3}).
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    std::vector<size_t> targetShape = {1, 3};

    EXPECT_THROW(
        { hahaha::ml::broadcast(a.getComputeNode(), targetShape); },
        std::runtime_error);
}

TEST_F(BroadcastTest, AutoBroadcast_Add_VectorToMatrix_ForwardAndGrad) {
    // a: (3) will be broadcast to (2,3) to match w
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    Tensor<float> w(NestedData<float>{{10.0f, 20.0f, 30.0f}, {40.0f, 50.0f, 60.0f}});
    w.setRequiresGrad(true);

    auto c = w + a; // should auto-broadcast a to (2,3)

    EXPECT_EQ(c.getShape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(c.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({0, 2}), 33.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 41.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 52.0f);
    EXPECT_FLOAT_EQ(c.at({1, 2}), 63.0f);

    // backward() uses implicit upstream grad of ones for non-scalar tensors.
    // dL/dw = 1, dL/da = sum over leading axis => [2,2,2]
    c.backward();

    ASSERT_FALSE(w.grad().isEmpty());
    EXPECT_EQ(w.grad().getShape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(w.grad().at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(w.grad().at({1, 2}), 1.0f);

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_EQ(a.grad().getShape(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(a.grad().at({0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({2}), 2.0f);
}

TEST_F(BroadcastTest, AutoBroadcast_Mul_ScalarToMatrix_Grad) {
    Tensor<float> a(2.0f);
    a.setRequiresGrad(true);
    Tensor<float> w(NestedData<float>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    auto c = w * a; // scalar should broadcast to (2,3)
    c.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    // dL/da = sum(w) because upstream grad is ones
    EXPECT_FLOAT_EQ(a.grad().at({}), 21.0f);
}

TEST_F(BroadcastTest, Broadcast_GradFun_ComplexInners) {
    // Shape {2, 1, 2} -> {2, 2, 2}
    Tensor<float> a(NestedData<float>{{{1, 2}}, {{3, 4}}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 2, 2};

    auto b = hahaha::ml::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> bt(b);

    bt.backward();

    ASSERT_FALSE(a.grad().isEmpty());
    EXPECT_FLOAT_EQ(a.grad().at({0, 0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0, 1}), 2.0f);
}

// ============================================================================
// All 16 dimension combination tests: 0D, 1D, 2D, 3D cross operations
// ============================================================================

TEST_F(BroadcastTest, Broadcast_0Dvs0D_ScalarToScalar) {
    Tensor<float> s0(5.0f);
    s0.setRequiresGrad(true);
    std::vector<size_t> target0D = {};
    auto b = hahaha::ml::broadcast(s0.getComputeNode(), target0D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 0);
    EXPECT_FLOAT_EQ(bt.at({}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_0Dvs1D_ScalarToVector) {
    Tensor<float> s0(5.0f);
    s0.setRequiresGrad(true);
    std::vector<size_t> target1D = {3};
    auto b = hahaha::ml::broadcast(s0.getComputeNode(), target1D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 1);
    EXPECT_EQ(bt.getShape()[0], 3);
    EXPECT_FLOAT_EQ(bt.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({2}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_0Dvs2D_ScalarToMatrix) {
    Tensor<float> s0(5.0f);
    s0.setRequiresGrad(true);
    std::vector<size_t> target2D = {2, 3};
    auto b = hahaha::ml::broadcast(s0.getComputeNode(), target2D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 2);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 2}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_0Dvs3D_ScalarToTensor) {
    Tensor<float> s0(5.0f);
    s0.setRequiresGrad(true);
    std::vector<size_t> target3D = {2, 2, 3};
    auto b = hahaha::ml::broadcast(s0.getComputeNode(), target3D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 3);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 2);
    EXPECT_EQ(bt.getShape()[2], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 2}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_1Dvs1D_SameShape) {
    Tensor<float> v1(NestedData<float>{1.0f, 2.0f, 3.0f});
    v1.setRequiresGrad(true);
    std::vector<size_t> target1D_same = {3};
    auto b = hahaha::ml::broadcast(v1.getComputeNode(), target1D_same);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 1);
    EXPECT_EQ(bt.getShape()[0], 3);
    EXPECT_FLOAT_EQ(bt.at({0}), 1.0f);
}

TEST_F(BroadcastTest, Broadcast_1Dvs2D_VectorToMatrix) {
    Tensor<float> v1(NestedData<float>{1.0f, 2.0f, 3.0f});
    v1.setRequiresGrad(true);
    std::vector<size_t> target1D2D = {2, 3};
    auto b = hahaha::ml::broadcast(v1.getComputeNode(), target1D2D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 2);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 0}), 1.0f); // broadcast first dim
}

TEST_F(BroadcastTest, Broadcast_1Dvs3D_VectorToTensor) {
    Tensor<float> v1(NestedData<float>{1.0f, 2.0f, 3.0f});
    v1.setRequiresGrad(true);
    std::vector<size_t> target1D3D = {2, 2, 3};
    auto b = hahaha::ml::broadcast(v1.getComputeNode(), target1D3D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 3);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 2);
    EXPECT_EQ(bt.getShape()[2], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 0}), 1.0f); // broadcast first two dims
}

TEST_F(BroadcastTest, Broadcast_1Dvs2D_Incompatible_Throws) {
    Tensor<float> v2(NestedData<float>{1.0f, 2.0f}); // shape [2]
    std::vector<size_t> target1D2D_incompat = {2,
                                               3}; // can't broadcast [2] to [2, 3]
    EXPECT_THROW(
        hahaha::ml::broadcast(v2.getComputeNode(), target1D2D_incompat),
        std::runtime_error);
}

TEST_F(BroadcastTest, Broadcast_2Dvs2D_SameShape) {
    Tensor<float> m1(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    m1.setRequiresGrad(true);
    std::vector<size_t> target2D_same = {2, 2};
    auto b = hahaha::ml::broadcast(m1.getComputeNode(), target2D_same);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 2);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 2);
    EXPECT_FLOAT_EQ(bt.at({0, 0}), 1.0f);
}

TEST_F(BroadcastTest, Broadcast_2Dvs2D_DimOneBroadcasting) {
    Tensor<float> m2(NestedData<float>{{1.0f, 2.0f, 3.0f}}); // shape [1, 3]
    std::vector<size_t> target2D_broadcast = {2, 3};
    auto b = hahaha::ml::broadcast(m2.getComputeNode(), target2D_broadcast);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 2);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 0}), 1.0f); // broadcast first dim
}

TEST_F(BroadcastTest, Broadcast_2Dvs3D_MatrixToTensor) {
    Tensor<float> m2(NestedData<float>{{1.0f, 2.0f, 3.0f}}); // shape [1, 3]
    std::vector<size_t> target2D3D = {2, 2, 3};
    auto b = hahaha::ml::broadcast(m2.getComputeNode(), target2D3D);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 3);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 2);
    EXPECT_EQ(bt.getShape()[2], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 0}), 1.0f); // broadcast first dim
}

TEST_F(BroadcastTest, Broadcast_2Dvs3D_Incompatible_Throws) {
    Tensor<float> m3(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // shape [2, 2]
    std::vector<size_t> target2D3D_incompat = {
        2, 2, 3}; // can't broadcast [2, 2] to [2, 2, 3]
    EXPECT_THROW(
        hahaha::ml::broadcast(m3.getComputeNode(), target2D3D_incompat),
        std::runtime_error);
}

TEST_F(BroadcastTest, Broadcast_3Dvs3D_SameShape) {
    Tensor<float> t1(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                       {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    t1.setRequiresGrad(true);
    std::vector<size_t> target3D_same = {2, 2, 2};
    auto b = hahaha::ml::broadcast(t1.getComputeNode(), target3D_same);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 3);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 2);
    EXPECT_EQ(bt.getShape()[2], 2);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 1.0f);
}

TEST_F(BroadcastTest, Broadcast_3Dvs3D_DimOneBroadcasting) {
    Tensor<float> t2(NestedData<float>{{{1.0f, 2.0f, 3.0f}}}); // shape [1, 1, 3]
    std::vector<size_t> target3D_broadcast = {2, 2, 3};
    auto b = hahaha::ml::broadcast(t2.getComputeNode(), target3D_broadcast);
    Tensor<float> bt(b);
    EXPECT_EQ(bt.getShape().size(), 3);
    EXPECT_EQ(bt.getShape()[0], 2);
    EXPECT_EQ(bt.getShape()[1], 2);
    EXPECT_EQ(bt.getShape()[2], 3);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 0}), 1.0f); // broadcast first two dims
}

// ============================================================================
// Scalar Types Distinction Tests - Different scalar representations
// ============================================================================

TEST_F(BroadcastTest, Broadcast_ScalarTypes_0D_Scalar) {
    std::vector<size_t> target = {2, 2, 2};
    Tensor<float> s0(5.0f);
    auto b = hahaha::ml::broadcast(s0.getComputeNode(), target);
    Tensor<float> bt(b);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 1}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_ScalarTypes_1D_SingleElement) {
    std::vector<size_t> target = {2, 2, 2};
    Tensor<float> s1(NestedData<float>{5.0f}); // shape [1]
    auto b = hahaha::ml::broadcast(s1.getComputeNode(), target);
    Tensor<float> bt(b);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 1}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_ScalarTypes_2D_SingleElement) {
    std::vector<size_t> target = {2, 2, 2};
    Tensor<float> s2(NestedData<float>{{5.0f}}); // shape [1, 1]
    auto b = hahaha::ml::broadcast(s2.getComputeNode(), target);
    Tensor<float> bt(b);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 1}), 5.0f);
}

TEST_F(BroadcastTest, Broadcast_ScalarTypes_3D_SingleElement) {
    std::vector<size_t> target = {2, 2, 2};
    Tensor<float> s3(NestedData<float>{{{5.0f}}}); // shape [1, 1, 1]
    auto b = hahaha::ml::broadcast(s3.getComputeNode(), target);
    Tensor<float> bt(b);
    EXPECT_FLOAT_EQ(bt.at({0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(bt.at({1, 1, 1}), 5.0f);
}

// ============================================================================
// Error Cases Tests - Incompatible broadcasts for all dimensions
// ============================================================================

TEST_F(BroadcastTest, Broadcast_ErrorCase_1Dvs1D_IncompatibleSizes) {
    Tensor<float> v1(NestedData<float>{1.0f, 2.0f}); // [2]
    std::vector<size_t> target = {3};
    EXPECT_THROW(hahaha::ml::broadcast(v1.getComputeNode(), target),
                 std::runtime_error);
}

TEST_F(BroadcastTest, Broadcast_ErrorCase_2Dvs2D_IncompatibleInnerDimension) {
    Tensor<float> m1(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // [2, 2]
    std::vector<size_t> target = {2, 3};
    EXPECT_THROW(hahaha::ml::broadcast(m1.getComputeNode(), target),
                 std::runtime_error);
}

TEST_F(BroadcastTest, Broadcast_ErrorCase_3Dvs3D_IncompatibleDimension) {
    Tensor<float> t1(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                       {{5.0f, 6.0f}, {7.0f, 8.0f}}}); // [2, 2, 2]
    std::vector<size_t> target = {2, 2, 3};
    EXPECT_THROW(hahaha::ml::broadcast(t1.getComputeNode(), target),
                 std::runtime_error);
}

TEST_F(BroadcastTest, Broadcast_ErrorCase_LowerRankToHigherRank_Incompatible) {
    Tensor<float> m2(NestedData<float>{{1.0f, 2.0f}}); // [1, 2]
    std::vector<size_t> target = {2, 3}; // [1, 2] can't broadcast to [2, 3]
    EXPECT_THROW(hahaha::ml::broadcast(m2.getComputeNode(), target),
                 std::runtime_error);
}
