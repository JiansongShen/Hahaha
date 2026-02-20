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

#include "ml/scheduler/PolynomialLRScheduler.h"

#include <gtest/gtest.h>
#include <cmath>

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class PolynomialLRSchedulerTest : public ::testing::Test {};

TEST_F(PolynomialLRSchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getTotalSteps(), 10u);
    EXPECT_FLOAT_EQ(scheduler.getEndLearningRate(), 0.0f);
    EXPECT_FLOAT_EQ(scheduler.getPower(), 1.0f);
}

TEST_F(PolynomialLRSchedulerTest, Step_StartsAtBaseLearningRate) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
}

TEST_F(PolynomialLRSchedulerTest, Step_DecaysToEndLearningRate) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(scheduler.getLearningRate(), 0.0f, 1e-5f);
}

TEST_F(PolynomialLRSchedulerTest, Step_MatchesPolynomialFormula) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    scheduler.step();
    float expected = (1.0f - 0.0f) * std::pow(1.0f - 1.0f / 10.0f, 1.0f) + 0.0f;
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
}

TEST_F(PolynomialLRSchedulerTest, Step_WithPower2_DecaysFaster) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 2.0f);

    scheduler.step();
    float lr1 = scheduler.getLearningRate();

    PolynomialLRScheduler<float> scheduler2(opt, 1.0f, 10, 0.0f, 1.0f);
    scheduler2.step();
    float lr2 = scheduler2.getLearningRate();

    EXPECT_LT(lr1, lr2);
}

TEST_F(PolynomialLRSchedulerTest, Reset_RestoresToBaseLearningRate) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
}

TEST_F(PolynomialLRSchedulerTest, SetTotalSteps_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    scheduler.setTotalSteps(20);
    EXPECT_EQ(scheduler.getTotalSteps(), 20u);
}

TEST_F(PolynomialLRSchedulerTest, SetEndLearningRate_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    scheduler.setEndLearningRate(0.1f);
    EXPECT_FLOAT_EQ(scheduler.getEndLearningRate(), 0.1f);
}

TEST_F(PolynomialLRSchedulerTest, SetPower_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    PolynomialLRScheduler<float> scheduler(opt, 1.0f, 10, 0.0f, 1.0f);

    scheduler.setPower(2.0f);
    EXPECT_FLOAT_EQ(scheduler.getPower(), 2.0f);
}
