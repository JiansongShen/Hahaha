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

#include "ml/scheduler/CosineAnnealingWarmRestartsScheduler.h"

#include <gtest/gtest.h>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../../../../core/include/public/Tensor.h"
#include "ml/optimizer/SGDOptimizer.h"

using namespace hahaha;
using namespace hahaha::ml;

class CosineAnnealingWarmRestartsSchedulerTest : public ::testing::Test {};

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, Constructor_InitializesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    EXPECT_FLOAT_EQ(scheduler.getBaseLearningRate(), 0.0f);
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.0f);
    EXPECT_FLOAT_EQ(scheduler.getEtaMax(), 1.0f);
    EXPECT_FLOAT_EQ(scheduler.getEtaMin(), 0.0f);
    EXPECT_EQ(scheduler.getT0(), 5u);
    EXPECT_FLOAT_EQ(scheduler.getTMult(), 2.0f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, Step_BeforeT0_UsesT0Period) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    scheduler.step();
    float expected = 0.0f + 0.5f * (1.0f - 0.0f) *
                              (1.0f + std::cos(M_PI * 1.0f / 5.0f));
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, Step_AtT0_ReachesEtaMin) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    for (int i = 0; i < 4; ++i) {
        scheduler.step();
    }

    float expected =
        0.0f + 0.5f * (1.0f - 0.0f) * (1.0f + std::cos(M_PI * 4.0f / 5.0f));
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);

    scheduler.step();
    float t = 5.0f - 5.0f;
    float tN = 5.0f * 2.0f;
    expected = 0.0f + 0.5f * (1.0f - 0.0f) * (1.0f + std::cos(M_PI * t / tN));
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
    EXPECT_NEAR(scheduler.getLearningRate(), 1.0f, 1e-5f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, Step_AfterT0_UsesExtendedPeriod) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    for (int i = 0; i < 6; ++i) {
        scheduler.step();
    }

    float t = 6.0f - 5.0f;
    float tN = 5.0f * 2.0f;
    float expected = 0.0f + 0.5f * (1.0f - 0.0f) *
                              (1.0f + std::cos(M_PI * t / tN));
    EXPECT_NEAR(scheduler.getLearningRate(), expected, 1e-5f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, Reset_RestoresToEtaMin) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    for (int i = 0; i < 3; ++i) {
        scheduler.step();
    }

    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.getLearningRate(), 0.0f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, SetEtaMax_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    scheduler.setEtaMax(2.0f);
    EXPECT_FLOAT_EQ(scheduler.getEtaMax(), 2.0f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, SetEtaMin_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    scheduler.setEtaMin(0.1f);
    EXPECT_FLOAT_EQ(scheduler.getEtaMin(), 0.1f);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, SetT0_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    scheduler.setT0(10);
    EXPECT_EQ(scheduler.getT0(), 10u);
}

TEST_F(CosineAnnealingWarmRestartsSchedulerTest, SetTMult_UpdatesCorrectly) {
    Tensor<float> w(1.0f);
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<ml::ComputeNode<float>>> params = {
        w.getComputeNode()};
    SGDOptimizer<float> opt(params, 1.0f);
    CosineAnnealingWarmRestartsScheduler<float> scheduler(opt, 1.0f, 0.0f, 5, 2.0f);

    scheduler.setTMult(3.0f);
    EXPECT_FLOAT_EQ(scheduler.getTMult(), 3.0f);
}
