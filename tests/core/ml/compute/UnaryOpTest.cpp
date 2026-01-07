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

class UnaryOpTest : public ::testing::Test {};

TEST_F(UnaryOpTest, Reshape_Backward_Full) {
    Tensor<float> a(NestedData<float>{1, 2, 3, 4});
    a.setRequiresGrad(true);
    auto b = a.reshape({2, 2});

    // Create upstream gradient
    Tensor<float> grad(NestedData<float>{{10, 20}, {30, 40}});
    b.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<float>>(
            grad.data()->clone()));

    // Trigger gradFun
    b.getComputeNode()->getGradFun()();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->data()->at({0}), 10.0f);
    EXPECT_FLOAT_EQ(a.grad()->data()->at({3}), 40.0f);
}

TEST_F(UnaryOpTest, Transpose_Backward_Full) {
    Tensor<float> a(NestedData<float>{{1, 2}, {3, 4}});
    a.setRequiresGrad(true);
    auto b = a.transpose();

    // Upstream grad
    Tensor<float> grad(NestedData<float>{{10, 20}, {30, 40}});
    b.getComputeNode()->setGrad(
        std::make_shared<hahaha::math::TensorWrapper<float>>(
            grad.data()->clone()));

    b.getComputeNode()->getGradFun()();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->data()->at({0, 1}), 30.0f);
}

TEST_F(UnaryOpTest, Negate_Backward_Complex) {
    // f = -(a * b)
    Tensor<float> a(2.0f);
    a.setRequiresGrad(true);
    Tensor<float> b(3.0f);
    b.setRequiresGrad(true);

    auto ab = a * b;
    auto f = -ab;

    f.backward();

    // df/da = -b = -3
    // df/db = -a = -2
    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->data()->at({}), -3.0f);
    ASSERT_NE(b.grad(), nullptr);
    EXPECT_FLOAT_EQ(b.grad()->data()->at({}), -2.0f);
}
