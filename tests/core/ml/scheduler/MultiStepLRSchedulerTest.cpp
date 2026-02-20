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

#include "ml/scheduler/MultiStepLRScheduler.h"

#include <gtest/gtest.h>

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class MultiStepLRSchedulerTest : public ::testing::Test {};

TEST_F(MultiStepLRSchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7, 10};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getGamma(), 0.5f);
    EXPECT_EQ(scheduler.getMilestones().size(), 3u);
}

TEST_F(MultiStepLRSchedulerTest, Constructor_SortsMilestones) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {10, 3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    const auto& sorted = scheduler.getMilestones();
    EXPECT_EQ(sorted[0], 3u);
    EXPECT_EQ(sorted[1], 7u);
    EXPECT_EQ(sorted[2], 10u);
}

TEST_F(MultiStepLRSchedulerTest, Step_NoDecayBeforeMilestone) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
}

TEST_F(MultiStepLRSchedulerTest, Step_DecayAtFirstMilestone) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);
}

TEST_F(MultiStepLRSchedulerTest, Step_DecayAtSecondMilestone) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    for (int i = 0; i < 7; ++i) {
        scheduler.step();
    }
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.25f);
}

TEST_F(MultiStepLRSchedulerTest, Step_NoDecayBetweenMilestones) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    for (int i = 0; i < 3; ++i) {
        scheduler.step();
    }
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);

    scheduler.step();
    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);
}

TEST_F(MultiStepLRSchedulerTest, Reset_RestoresInitialState) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    for (int i = 0; i < 3; ++i) {
        scheduler.step();
    }
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.5f);

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 1.0f);
}

TEST_F(MultiStepLRSchedulerTest, SetMilestones_UpdatesAndSorts) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {3, 7};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    std::vector<std::size_t> newMilestones = {5, 2, 8};
    scheduler.setMilestones(newMilestones);
    const auto& sorted = scheduler.getMilestones();
    EXPECT_EQ(sorted[0], 2u);
    EXPECT_EQ(sorted[1], 5u);
    EXPECT_EQ(sorted[2], 8u);
}

TEST_F(MultiStepLRSchedulerTest, SetGamma_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    std::vector<std::size_t> milestones = {2};
    MultiStepLRScheduler<float> scheduler(opt, 1.0f, milestones, 0.5f);

    scheduler.setGamma(0.1f);
    EXPECT_FLOAT_EQ(scheduler.getGamma(), 0.1f);

    scheduler.step();
    scheduler.step();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.1f);
}
