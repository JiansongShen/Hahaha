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

// Tests for the public hahaha::SGDMOptimizer<T> wrapper defined in
// public/Optimizer.h.  The inner ml::SGDMOptimizer is tested separately in
// tests/core/ml/optimizer/SGDMOptimizerTest.cpp.

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
class PublicSGDMOptimizerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        w_ = Tensor<f32>(NestedData<f32>(5.0f));
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

TEST_F(PublicSGDMOptimizerTest, ConstructionAndGetLearningRate) {
    SGDMOptimizer<f32> opt({w_}, 0.1f, 0.9f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 0.1f);
}

TEST_F(PublicSGDMOptimizerTest, SetLearningRate) {
    SGDMOptimizer<f32> opt({w_}, 0.1f);
    opt.setLearningRate(0.05f);
    EXPECT_FLOAT_EQ(opt.getLearningRate(), 0.05f);
}

TEST_F(PublicSGDMOptimizerTest, GetParametersCount) {
    SGDMOptimizer<f32> opt({w_}, 0.1f);
    EXPECT_EQ(opt.getParameters().size(), 1u);
}

TEST_F(PublicSGDMOptimizerTest, AddParameter) {
    SGDMOptimizer<f32> opt({}, 0.1f);
    opt.addParameter(w_);
    EXPECT_EQ(opt.getParameters().size(), 1u);
}

TEST_F(PublicSGDMOptimizerTest, Step_NoGrad_ParameterUnchanged) {
    f32 before = w_.at({});
    SGDMOptimizer<f32> opt({w_}, 0.1f);
    opt.step();
    EXPECT_FLOAT_EQ(w_.at({}), before);
}

TEST_F(PublicSGDMOptimizerTest, Step_UpdatesParameter) {
    injectGrad(w_, 1.0f);
    f32 before = w_.at({});
    SGDMOptimizer<f32> opt({w_}, 0.1f, 0.9f);
    opt.step();
    // Momentum step must change the parameter value.
    EXPECT_NE(w_.at({}), before);
}

TEST_F(PublicSGDMOptimizerTest, Step_RequiresGradFalse_ParameterUnchanged) {
    injectGrad(w_, 1.0f);
    w_.setRequiresGrad(false);
    f32 before = w_.at({});
    SGDMOptimizer<f32> opt({w_}, 0.1f);
    opt.step();
    EXPECT_FLOAT_EQ(w_.at({}), before);
}
