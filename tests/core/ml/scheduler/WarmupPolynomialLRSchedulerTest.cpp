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

#include "ml/scheduler/WarmupPolynomialLRScheduler.h"

#include <gtest/gtest.h>
#include <cmath>

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class WarmupPolynomialLRSchedulerTest : public ::testing::Test {};

TEST_F(WarmupPolynomialLRSchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getTotalSteps(), 20u);
    EXPECT_FLOAT_EQ(scheduler.getEndLearningRate(), 0.0f);
    EXPECT_FLOAT_EQ(scheduler.getPower(), 1.0f);
    EXPECT_EQ(scheduler.getWarmupSteps(), 5u);
}

TEST_F(WarmupPolynomialLRSchedulerTest, Step_DuringWarmup_LinearIncrease) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    scheduler.step();
    float expected = 1.0f * (1.0f / 5.0f);
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);

    scheduler.step();
    expected = 1.0f * (2.0f / 5.0f);
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, Step_AtEndOfWarmup_ReachesBase) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(scheduler.getLearningRate(), 1.0f, 1e-5f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, Step_AfterWarmup_PolynomialDecay) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    for (int i = 0; i < 6; ++i) {
        scheduler.step();
    }

    float expected = (1.0f - 0.0f) * std::pow(1.0f - 6.0f / 20.0f, 1.0f) + 0.0f;
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
    EXPECT_LT(scheduler.getLearningRate(), 1.0f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, Step_AtEnd_ReachesEndLearningRate) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    for (int i = 0; i < 20; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(scheduler.getLearningRate(), 0.0f, 1e-5f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, Reset_RestoresToBaseLearningRate) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, SetTotalSteps_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    scheduler.setTotalSteps(30);
    EXPECT_EQ(scheduler.getTotalSteps(), 30u);
}

TEST_F(WarmupPolynomialLRSchedulerTest, SetEndLearningRate_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    scheduler.setEndLearningRate(0.1f);
    EXPECT_FLOAT_EQ(scheduler.getEndLearningRate(), 0.1f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, SetPower_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    scheduler.setPower(2.0f);
    EXPECT_FLOAT_EQ(scheduler.getPower(), 2.0f);
}

TEST_F(WarmupPolynomialLRSchedulerTest, SetWarmupSteps_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupPolynomialLRScheduler<float> scheduler(opt, 1.0f, 20, 0.0f, 1.0f, 5);

    scheduler.setWarmupSteps(10);
    EXPECT_EQ(scheduler.getWarmupSteps(), 10u);
}
