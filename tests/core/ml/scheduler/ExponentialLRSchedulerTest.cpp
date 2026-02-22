// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// JiansongShen (jason.shen111@outlook.com) (https://github.com/jiansongshen)

#include "ml/scheduler/ExponentialLRScheduler.h"

#include <gtest/gtest.h>
#include <cmath>

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class ExponentialLRSchedulerTest : public ::testing::Test {};

TEST_F(ExponentialLRSchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    ExponentialLRScheduler<float> scheduler(opt, 1.0f, 0.95f);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getGamma(), 0.95f);
}

TEST_F(ExponentialLRSchedulerTest, Constructor_ThrowsOnInvalidGamma) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);

    EXPECT_THROW(ExponentialLRScheduler<float>(opt, 1.0f, 0.0f),
                 std::invalid_argument);
    EXPECT_THROW(ExponentialLRScheduler<float>(opt, 1.0f, 1.0f),
                 std::invalid_argument);
    EXPECT_THROW(ExponentialLRScheduler<float>(opt, 1.0f, -0.1f),
                 std::invalid_argument);
    EXPECT_THROW(ExponentialLRScheduler<float>(opt, 1.0f, 1.1f),
                 std::invalid_argument);
}

TEST_F(ExponentialLRSchedulerTest, Step_DecaysExponentially) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    ExponentialLRScheduler<float> scheduler(opt, 1.0f, 0.9f);

    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), 0.9f, 1e-5f);

    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), 0.81f, 1e-5f);

    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), 0.729f, 1e-5f);
}

TEST_F(ExponentialLRSchedulerTest, Step_MatchesFormula) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    ExponentialLRScheduler<float> scheduler(opt, 1.0f, 0.95f);

    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    float expected = 1.0f * std::pow(0.95f, 5.0f);
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
}

TEST_F(ExponentialLRSchedulerTest, Reset_RestoresBaseLearningRate) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    ExponentialLRScheduler<float> scheduler(opt, 1.0f, 0.9f);

    scheduler.step();
    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), 0.81f, 1e-5f);

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
}

TEST_F(ExponentialLRSchedulerTest, SetGamma_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    ExponentialLRScheduler<float> scheduler(opt, 1.0f, 0.9f);

    scheduler.setGamma(0.8f);
    EXPECT_FLOAT_EQ(scheduler.getGamma(), 0.8f);

    scheduler.reset();
    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), 0.8f, 1e-5f);
}
