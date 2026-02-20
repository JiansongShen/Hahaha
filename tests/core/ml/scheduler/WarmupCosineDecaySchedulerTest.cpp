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

#include "ml/scheduler/WarmupCosineDecayScheduler.h"

#include <gtest/gtest.h>
#include <cmath>

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class WarmupCosineDecaySchedulerTest : public ::testing::Test {};

TEST_F(WarmupCosineDecaySchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 0.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.0f);
    EXPECT_FLOAT_EQ(scheduler.getMaxLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getMinLearningRate(), 0.0f);
    EXPECT_EQ(scheduler.getWarmupSteps(), 5u);
    EXPECT_EQ(scheduler.getTotalSteps(), 20u);
}

TEST_F(WarmupCosineDecaySchedulerTest, Step_DuringWarmup_Increases) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    scheduler.step();
    float expected = 1.0f * (1.0f - std::cos(M_PI * 1.0f / 5.0f)) / 2.0f;
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
    EXPECT_GT(scheduler.getLearningRate(), 0.0f);
}

TEST_F(WarmupCosineDecaySchedulerTest, Step_AtEndOfWarmup_ReachesMax) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    for (int i = 0; i < 4; ++i) {
        scheduler.step();
    }

    float expected = 1.0f * (1.0f - std::cos(M_PI * 4.0f / 5.0f)) / 2.0f;
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);

    scheduler.step();
    EXPECT_NEAR(scheduler.getLearningRate(), 1.0f, 1e-5f);
}

TEST_F(WarmupCosineDecaySchedulerTest, Step_AfterWarmup_Decays) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    scheduler.step();
    float t = static_cast<float>(scheduler.getStep() - scheduler.getWarmupSteps());
    float totalDecay =
        static_cast<float>(scheduler.getTotalSteps() - scheduler.getWarmupSteps());
    float expected =
        0.0f + (1.0f - 0.0f) * (1.0f + std::cos(M_PI * t / totalDecay)) / 2.0f;
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
    EXPECT_LT(scheduler.getLearningRate(), 1.0f);
}

TEST_F(WarmupCosineDecaySchedulerTest, Step_AtEnd_ReachesMin) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    for (int i = 0; i < 15; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(scheduler.getLearningRate(), 0.0f, 1e-5f);
}

TEST_F(WarmupCosineDecaySchedulerTest, Reset_RestoresToMin) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.0f);
}

TEST_F(WarmupCosineDecaySchedulerTest, SetWarmupSteps_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    scheduler.setWarmupSteps(10);
    EXPECT_EQ(scheduler.getWarmupSteps(), 10u);
}

TEST_F(WarmupCosineDecaySchedulerTest, SetTotalSteps_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    scheduler.setTotalSteps(30);
    EXPECT_EQ(scheduler.getTotalSteps(), 30u);
}

TEST_F(WarmupCosineDecaySchedulerTest, SetMaxLearningRate_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    scheduler.setMaxLearningRate(2.0f);
    EXPECT_FLOAT_EQ(scheduler.getMaxLearningRate(), 2.0f);
}

TEST_F(WarmupCosineDecaySchedulerTest, SetMinLearningRate_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    WarmupCosineDecayScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 20);

    scheduler.setMinLearningRate(0.1f);
    EXPECT_FLOAT_EQ(scheduler.getMinLearningRate(), 0.1f);
}
