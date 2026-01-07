// Copyright (c) 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include "Tensor.h"

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
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
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
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
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
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
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

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape().size(), 1);
    EXPECT_EQ(a.grad()->getShape()[0], 3);

    EXPECT_FLOAT_EQ(a.grad()->at({0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({2}), 2.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_Scalar) {
    // Shape {1} -> {2, 2}
    Tensor<float> a(3.0f); // Size 1
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 2};

    auto broadcastedNode =
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    // dL/db = 1 everywhere
    // dL/da = sum(dL/db) = 1+1+1+1 = 4
    b.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({}), 4.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_InnerDims) {
    // Shape {3, 1} -> {3, 2}
    // This tests broadcasting on a non-leading dimension
    Tensor<float> a(NestedData<float>{{1.0f}, {2.0f}, {3.0f}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {3, 2};

    auto broadcastedNode =
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    // b = [[1, 1], [2, 2], [3, 3]]
    // dL/db = 1 everywhere
    // dL/da[0] = dL/db[0,0] + dL/db[0,1] = 2

    b.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape()[0], 3);
    EXPECT_EQ(a.grad()->getShape()[1], 1);

    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({2, 0}), 2.0f);
}

TEST_F(BroadcastTest, Broadcast_Error_Mismatch) {
    Tensor<float> a(NestedData<float>{1.0f, 2.0f}); // Shape {2}
    std::vector<size_t> targetShape = {3};          // Cannot broadcast 2 to 3

    EXPECT_THROW(
        { hahaha::compute::broadcast(a.getComputeNode(), targetShape); },
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
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    b.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape().size(), 1);
    EXPECT_EQ(a.grad()->getShape()[0], 3);

    // Each element broadcasted 2 * 4 = 8 times
    EXPECT_FLOAT_EQ(a.grad()->at({0}), 8.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1}), 8.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({2}), 8.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_Identity_Loop_Check) {
    // Shape {2, 2} -> {2, 2}
    // Tests the fix for the infinite loop when dimensions match
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 2};

    auto broadcastedNode =
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    b.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape(), targetShape);

    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 1}), 1.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_NonUniform_LeadingDims) {
    // Shape {3} -> {2, 3}, but make upstream gradient non-uniform via mul.
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 3};

    auto broadcastedNode =
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    Tensor<float> w(NestedData<float>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto c = b * w;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape().size(), 1);
    EXPECT_EQ(a.grad()->getShape()[0], 3);

    // For leading-dim broadcast, grad is summed over axis 0:
    // [[1,2,3],[4,5,6]] -> [5,7,9]
    EXPECT_FLOAT_EQ(a.grad()->at({0}), 5.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1}), 7.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({2}), 9.0f);
}

TEST_F(BroadcastTest, Broadcast_Gradient_NonUniform_InnerDims) {
    // Shape {3, 1} -> {3, 2}, make upstream gradient non-uniform via mul.
    Tensor<float> a(NestedData<float>{{1.0f}, {2.0f}, {3.0f}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {3, 2};

    auto broadcastedNode =
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> b(broadcastedNode);

    Tensor<float> w(
        NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    auto c = b * w;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape().size(), 2);
    EXPECT_EQ(a.grad()->getShape()[0], 3);
    EXPECT_EQ(a.grad()->getShape()[1], 1);

    // Sum over inner axis: [1+2, 3+4, 5+6] = [3, 7, 11], keepDims => (3,1)
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({2, 0}), 11.0f);
}

TEST_F(BroadcastTest, Broadcast_GradFun_NullGrad_DoesNothing) {
    // Directly call gradFun without running backward() so res->getGrad() is
    // null.
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 3};

    auto broadcastedNode =
        hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    ASSERT_NE(broadcastedNode->getGradFun(), nullptr);

    EXPECT_NO_THROW(broadcastedNode->getGradFun()());
    EXPECT_EQ(a.grad(), nullptr);
}

TEST_F(BroadcastTest,
       Broadcast_Error_TargetNotBroadcastResult_ThrowsRuntimeError) {
    // src shape {2,3}, target shape {1,3} is NOT a valid "broadcast-to",
    // but the two shapes are broadcast-compatible (broadcast result would be
    // {2,3}).
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    std::vector<size_t> targetShape = {1, 3};

    EXPECT_THROW(
        { hahaha::compute::broadcast(a.getComputeNode(), targetShape); },
        std::runtime_error);
}

TEST_F(BroadcastTest, AutoBroadcast_Add_VectorToMatrix_ForwardAndGrad) {
    // a: (3) will be broadcast to (2,3) to match w
    Tensor<float> a(NestedData<float>{1.0f, 2.0f, 3.0f});
    a.setRequiresGrad(true);
    Tensor<float> w(
        NestedData<float>{{10.0f, 20.0f, 30.0f}, {40.0f, 50.0f, 60.0f}});
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

    ASSERT_NE(w.grad(), nullptr);
    EXPECT_EQ(w.grad()->getShape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(w.grad()->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(w.grad()->at({1, 2}), 1.0f);

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(a.grad()->getShape(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(a.grad()->at({0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({2}), 2.0f);
}

TEST_F(BroadcastTest, AutoBroadcast_Mul_ScalarToMatrix_Grad) {
    Tensor<float> a(2.0f);
    a.setRequiresGrad(true);
    Tensor<float> w(NestedData<float>{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    auto c = w * a; // scalar should broadcast to (2,3)
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    // dL/da = sum(w) because upstream grad is ones
    EXPECT_FLOAT_EQ(a.grad()->at({}), 21.0f);
}

TEST_F(BroadcastTest, Broadcast_GradFun_ComplexInners) {
    // Shape {2, 1, 2} -> {2, 2, 2}
    Tensor<float> a(NestedData<float>{{{1, 2}}, {{3, 4}}});
    a.setRequiresGrad(true);
    std::vector<size_t> targetShape = {2, 2, 2};

    auto b = hahaha::compute::broadcast(a.getComputeNode(), targetShape);
    Tensor<float> bt(b);

    bt.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({0, 0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad()->at({1, 0, 1}), 2.0f);
}
