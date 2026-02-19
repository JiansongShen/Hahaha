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

// Tests for the public hahaha::SGDOptimizer<T> wrapper defined in
// public/Optimizer.h.  The inner ml::SGDOptimizer is tested separately in
// tests/core/ml/optimizer/SGDOptimizerTest.cpp.

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
class PublicSGDOptimizerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        w_ = Tensor<f32>(NestedData<f32>(10.0f));
        w_.setRequiresGrad(true);
    }

    // Inject a known scalar gradient directly into the ComputeNode.
    void injectGrad(Tensor<f32>& t, f32 value) {
        t.getComputeNode()->accumulateGrad(
            std::make_shared<TensorWrapper<f32>>(TensorShape({}), value));
    }

    Tensor<f32> w_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(PublicSGDOptimizerTest, ConstructionAndGetLearningRate) {
    SGDOptimizer<f32> opt({w_}, 0.1f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 0.1f);
}

TEST_F(PublicSGDOptimizerTest, SetLearningRate) {
    SGDOptimizer<f32> opt({w_}, 0.1f);
    opt.setLearningRate(0.01f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 0.01f);
}

TEST_F(PublicSGDOptimizerTest, GetParametersCount) {
    SGDOptimizer<f32> opt({w_}, 0.1f);
    EXPECT_EQ(opt.getParameters().size(), 1u);
}

TEST_F(PublicSGDOptimizerTest, AddParameter) {
    SGDOptimizer<f32> opt({}, 0.1f);
    EXPECT_EQ(opt.getParameters().size(), 0u);
    opt.addParameter(w_);
    EXPECT_EQ(opt.getParameters().size(), 1u);
}

TEST_F(PublicSGDOptimizerTest, Step_NoGrad_ParameterUnchanged) {
    f32 before = w_.at({});
    SGDOptimizer<f32> opt({w_}, 0.1f);
    opt.step();
    EXPECT_FLOAT_EQ(w_.at({}), before);
}

TEST_F(PublicSGDOptimizerTest, Step_UpdatesParameter) {
    // w = 10, grad = 2, lr = 0.1 → w = 10 - 0.1*2 = 9.8
    injectGrad(w_, 2.0f);
    SGDOptimizer<f32> opt({w_}, 0.1f);
    opt.step();
    EXPECT_NEAR(w_.at({}), 9.8f, 1e-5f);
}

TEST_F(PublicSGDOptimizerTest, ZeroGrad_SetsGradToZero) {
    injectGrad(w_, 1.0f);
    ASSERT_FALSE(w_.grad().isEmpty());
    SGDOptimizer<f32> opt({w_}, 0.1f);
    opt.zeroGrad();
    // clearGrad() zeroes the values; the TensorWrapper still exists.
    EXPECT_FLOAT_EQ(w_.grad().at({}), 0.0f);
}

TEST_F(PublicSGDOptimizerTest, Step_RequiresGradFalse_ParameterUnchanged) {
    injectGrad(w_, 2.0f);
    w_.setRequiresGrad(false);
    f32 before = w_.at({});
    SGDOptimizer<f32> opt({w_}, 0.1f);
    opt.step();
    EXPECT_FLOAT_EQ(w_.at({}), before);
}
