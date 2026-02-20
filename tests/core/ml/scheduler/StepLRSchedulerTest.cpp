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

#include "ml/scheduler/StepLRScheduler.h"

#include <gtest/gtest.h>

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class StepLRSchedulerTest : public ::testing::Test {};

TEST_F(StepLRSchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 3, 0.5f);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getStepSize(), 3u);
    EXPECT_FLOAT_EQ(scheduler.getGamma(), 0.5f);
}

TEST_F(StepLRSchedulerTest, Step_NoDecayBeforeMilestone) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 3, 0.5f);

    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getStep(), 1u);

    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getStep(), 2u);
}

TEST_F(StepLRSchedulerTest, Step_DecayAtMilestone) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 3, 0.5f);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);
    EXPECT_EQ(scheduler.getStep(), 3u);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.25f);
    EXPECT_EQ(scheduler.getStep(), 6u);
}

TEST_F(StepLRSchedulerTest, Step_StepZeroNoDecay) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 3, 0.5f);

    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getStep(), 0u);
}

TEST_F(StepLRSchedulerTest, Reset_RestoresInitialState) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 3, 0.5f);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_EQ(scheduler.getStep(), 3u);
}

TEST_F(StepLRSchedulerTest, SetStepSize_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 3, 0.5f);

    scheduler.setStepSize(5);
    EXPECT_EQ(scheduler.getStepSize(), 5u);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);
}

TEST_F(StepLRSchedulerTest, SetGamma_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    StepLRScheduler<float> scheduler(opt, 1.0f, 2, 0.5f);

    scheduler.setGamma(0.1f);
    EXPECT_FLOAT_EQ(scheduler.getGamma(), 0.1f);

    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.1f);
}
