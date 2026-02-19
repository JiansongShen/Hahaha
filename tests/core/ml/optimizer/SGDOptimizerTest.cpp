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

#include "ml/optimizer/SGDOptimizer.h"

#include <gtest/gtest.h>

#include "../../../../core/include/public/Tensor.h"

using namespace hahaha;
using namespace hahaha::ml;

class SGDOptimizerTest : public ::testing::Test {};

TEST_F(SGDOptimizerTest, SimpleUpdate) {
    // 1. Create a parameter tensor with initial value 10.0
    Tensor<float> w(math::NestedData<float>{10.0f});
    w.setRequiresGrad(true);

    // 2. Create optimizer with learning rate 0.1
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 0.1f);

    // 3. Manually set gradient to 2.0
    // In a real scenario, this comes from loss.backward()
    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(math::TensorShape({1}),
                                                     2.0f));

    // 4. Perform optimization step
    // Expected: w = 10.0 - (0.1 * 2.0) = 9.8
    opt.step();

    EXPECT_FLOAT_EQ(w.at({0}), 9.8f);
}

TEST_F(SGDOptimizerTest, MultipleParameters) {
    Tensor<float> w1(math::NestedData<float>{1.0f, 2.0f});
    w1.setRequiresGrad(true);
    Tensor<float> w2(math::NestedData<float>{3.0f});
    w2.setRequiresGrad(true);

    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w1.getComputeNode(), w2.getComputeNode()};
    SGDOptimizer<float> opt(params, 0.5f);

    // Set gradients: grad(w1) = [0.2, 0.4], grad(w2) = 1.0
    w1.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(
            math::NestedData<float>{0.2f, 0.4f}));
    w2.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(math::TensorShape({1}),
                                                     1.0f));

    opt.step();

    // Expected w1: [1.0 - 0.5*0.2, 2.0 - 0.5*0.4] = [0.9, 1.8]
    // Expected w2: 3.0 - 0.5*1.0 = 2.5
    EXPECT_FLOAT_EQ(w1.at({0}), 0.9f);
    EXPECT_FLOAT_EQ(w1.at({1}), 1.8f);
    EXPECT_FLOAT_EQ(w2.at({0}), 2.5f);
}

TEST_F(SGDOptimizerTest, LearningRateChange) {
    Tensor<float> w(math::NestedData<float>{1.0f});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 0.1f);

    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(math::TensorShape({1}),
                                                     1.0f));

    opt.step(); // w = 1.0 - 0.1*1.0 = 0.9
    EXPECT_FLOAT_EQ(w.at({0}), 0.9f);

    opt.setLearningRate(0.2f);
    w.clearGrad();
    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(math::TensorShape({1}),
                                                     1.0f));

    opt.step(); // w = 0.9 - 0.2*1.0 = 0.7
    EXPECT_FLOAT_EQ(w.at({0}), 0.7f);
}

TEST_F(SGDOptimizerTest, RequiresGradFalse) {
    Tensor<float> w(math::NestedData<float>{1.0f});
    w.setRequiresGrad(false);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 0.1f);

    // Even if it has a gradient, it shouldn't be updated if requiresGrad is
    // false
    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(math::TensorShape({1}),
                                                     1.0f));

    opt.step();
    EXPECT_FLOAT_EQ(w.at({0}), 1.0f);
}

TEST_F(SGDOptimizerTest, ZeroGrad) {
    Tensor<float> w(math::NestedData<float>{1.0f});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 0.1f);

    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<float>>(math::TensorShape({1}),
                                                     1.0f));

    ASSERT_FALSE(w.grad().isEmpty());
    opt.zeroGrad();

    // After zeroGrad, the grad tensor should exist but be all zeros
    auto g = w.grad();
    ASSERT_FALSE(g.isEmpty());
    EXPECT_FLOAT_EQ(g.at({0}), 0.0f);
}

TEST_F(SGDOptimizerTest, Step_WithNullGrad) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true); // but no grad accumulated

    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 0.1f);
    EXPECT_NO_THROW(opt.step());
    EXPECT_FLOAT_EQ(w.data()->at({}), 1.0f);
}
