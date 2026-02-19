// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests for the public hahaha::AdamWOptimizer<T> wrapper defined in
// public/Optimizer.h.  The inner ml::AdamWOptimizer is tested separately in
// tests/core/ml/optimizer/AdamWOptimizerTest.cpp.

#include <gtest/gtest.h>

#include "math/ds/TensorData.h"
#include "math/ds/TensorShape.h"
#include "public/Optimizer.h"
#include "public/Tensor.h"

using namespace hahaha;
using namespace hahaha::math;
using namespace hahaha::common;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class PublicAdamWOptimizerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        w_ = Tensor<f32>(NestedData<f32>(3.0f));
        w_.setRequiresGrad(true);
    }

    void injectGrad(Tensor<f32>& t, f32 value) {
        t.getComputeNode()->accumulateGrad(
            std::make_shared<TensorWrapper<f32>>(TensorShape({}), value));
    }

    Tensor<f32> w_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(PublicAdamWOptimizerTest, ConstructionAndGetLearningRate) {
    AdamWOptimizer<f32> opt({w_}, 1e-3f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 1e-3f);
}

TEST_F(PublicAdamWOptimizerTest, SetLearningRate) {
    AdamWOptimizer<f32> opt({w_}, 1e-3f);
    opt.setLearningRate(5e-4f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 5e-4f);
}

TEST_F(PublicAdamWOptimizerTest, GetParametersCount) {
    AdamWOptimizer<f32> opt({w_}, 1e-3f);
    EXPECT_EQ(opt.getParameters().size(), 1u);
}

TEST_F(PublicAdamWOptimizerTest, AddParameter) {
    AdamWOptimizer<f32> opt({}, 1e-3f);
    opt.addParameter(w_);
    EXPECT_EQ(opt.getParameters().size(), 1u);
}

TEST_F(PublicAdamWOptimizerTest, Step_NoGrad_ParameterUnchanged) {
    f32 before = w_.at({});
    AdamWOptimizer<f32> opt({w_}, 1e-3f);
    opt.step();
    EXPECT_FLOAT_EQ(w_.at({}), before);
}

TEST_F(PublicAdamWOptimizerTest, Step_UpdatesParameter) {
    injectGrad(w_, 1.0f);
    f32 before = w_.at({});
    AdamWOptimizer<f32> opt({w_}, 1e-3f);
    opt.step();
    EXPECT_NE(w_.at({}), before);
}

TEST_F(PublicAdamWOptimizerTest, CustomHyperparameters_Accepted) {
    AdamWOptimizer<f32> opt({w_}, 1e-3f, 0.85f, 0.99f, 1e-8f, 1e-4f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 1e-3f);
    injectGrad(w_, 1.0f);
    EXPECT_NO_THROW(opt.step());
}

TEST_F(PublicAdamWOptimizerTest, ZeroGrad_SetsGradToZero) {
    injectGrad(w_, 1.0f);
    AdamWOptimizer<f32> opt({w_}, 1e-3f);
    opt.zeroGrad();
    EXPECT_FLOAT_EQ(w_.grad().at({}), 0.0f);
}
