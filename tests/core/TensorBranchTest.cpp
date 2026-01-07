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

#include <gtest/gtest.h>

#include "Tensor.h"

using hahaha::Tensor;
using hahaha::math::NestedData;

class TensorBranchTest : public ::testing::Test {};

TEST_F(TensorBranchTest, Grad_Null) {
    Tensor<float> a(1.0f);
    EXPECT_EQ(a.grad(), nullptr);
}

TEST_F(TensorBranchTest, ScalarOps) {
    Tensor<float> a(10.0f);
    auto b = a + 5.0f;
    auto c = 5.0f + a;
    auto d = a - 2.0f;
    auto e = 20.0f - a;
    auto f = a * 2.0f;
    auto g = 2.0f * a;
    auto h = a / 2.0f;
    auto i = 20.0f / a;

    EXPECT_FLOAT_EQ(b.data()->at({}), 15.0f);
    EXPECT_FLOAT_EQ(c.data()->at({}), 15.0f);
    EXPECT_FLOAT_EQ(d.data()->at({}), 8.0f);
    EXPECT_FLOAT_EQ(e.data()->at({}), 10.0f);
    EXPECT_FLOAT_EQ(f.data()->at({}), 20.0f);
    EXPECT_FLOAT_EQ(g.data()->at({}), 20.0f);
    EXPECT_FLOAT_EQ(h.data()->at({}), 5.0f);
    EXPECT_FLOAT_EQ(i.data()->at({}), 2.0f);
}

TEST_F(TensorBranchTest, ReshapeAndTranspose) {
    Tensor<float> a(NestedData<float>{{1, 2}, {3, 4}});
    auto b = a.reshape({4});
    EXPECT_EQ(b.getShape().size(), 1u);
    EXPECT_EQ(b.getShape()[0], 4u);

    auto c = a.transpose();
    EXPECT_EQ(c.getShape()[0], 2u);
    EXPECT_EQ(c.getShape()[1], 2u);
}

TEST_F(TensorBranchTest, ClearAndClearGrad) {
    Tensor<float> a(1.0f);
    a.setRequiresGrad(true);
    // Trigger some grad
    auto b = a * 2.0f;
    b.backward();
    EXPECT_NE(a.grad(), nullptr);

    a.clearGrad();
    // In current implementation clearGrad() keeps grad buffers but zeros them
    // out

    a.clear();
    EXPECT_FLOAT_EQ(a.data()->at({}), 0.0f);
}

TEST_F(TensorBranchTest, SetGetComputeNode) {
    Tensor<float> a(1.0f);
    auto node = a.getComputeNode();
    EXPECT_NE(node, nullptr);

    Tensor<float> b(2.0f);
    b.setComputeNode(node);
    EXPECT_EQ(b.getComputeNode(), node);
}

TEST_F(TensorBranchTest, Sum) {
    Tensor<float> a(NestedData<float>{1, 2, 3});
    EXPECT_FLOAT_EQ(a.sum(), 6.0f);
}
