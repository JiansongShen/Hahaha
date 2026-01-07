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

class ComputeNodeBranchTest : public ::testing::Test {};

TEST_F(ComputeNodeBranchTest, DivisionByZero_Graph) {
    Tensor<float> a(10.0f);
    Tensor<float> b(0.0f);

    // This should trigger Division by zero in the compute function
    EXPECT_THROW(auto c = a / b, std::runtime_error);
    EXPECT_THROW(auto d = a / 0.0f, std::runtime_error);
    EXPECT_THROW(auto e = 10.0f / b, std::runtime_error);
}

TEST_F(ComputeNodeBranchTest, MatMul_InvalidShapes_Graph) {
    Tensor<float> a(NestedData<float>{{1, 2}, {3, 4}});
    Tensor<float> b(NestedData<float>{1, 2, 3});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST_F(ComputeNodeBranchTest, Backward_AccumulateGrad_MixedRequiresGrad) {
    // f = a * b + c
    // a.reqGrad = true, b.reqGrad = false, c.reqGrad = true
    Tensor<float> a(2.0f);
    a.setRequiresGrad(true);
    Tensor<float> b(3.0f);
    b.setRequiresGrad(false);
    Tensor<float> c(5.0f);
    c.setRequiresGrad(true);

    auto f = a * b + c;
    f.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_FLOAT_EQ(a.grad()->at({}), 3.0f); // df/da = b = 3

    EXPECT_EQ(b.grad(), nullptr);

    ASSERT_NE(c.grad(), nullptr);
    EXPECT_FLOAT_EQ(c.grad()->at({}), 1.0f); // df/dc = 1
}

TEST_F(ComputeNodeBranchTest, ComputeNode_NullData_Backward) {
    // This is hard to trigger with public API, but let's try to see if we can
    // find any other branches.
}
