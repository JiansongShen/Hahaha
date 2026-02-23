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

// Tests for the public hahaha::Loss<T> and hahaha::MSELoss<T> wrapper classes
// defined in public/Loss.h.  The inner ml::Loss / ml::MSELoss are tested
// separately in tests/core/ml/loss/MSELossTest.cpp.

#include <gtest/gtest.h>

#include "math/TensorWrapper.h"
#include "math/ds/TensorData.h"
#include "public/Loss.h"

using namespace hahaha;
using namespace hahaha::math;
using namespace hahaha::common;

// ===========================================================================
// hahaha::Loss<T>  (base wrapper — default implementation returns zero)
// ===========================================================================

class PublicLossBaseTest : public ::testing::Test {
  protected:
    Loss<f32> MakeLoss() {
        // Loss<T> has a protected ctor; instantiate through MSELoss to get an
        // object that exercises the base computeLoss dispatch.
        return MSELoss<f32>();
    }
};

TEST_F(PublicLossBaseTest, ComputeLossReturnsTensorWrapperType) {
    MSELoss<f32> loss;
    auto result = loss.computeLoss(TensorWrapper<f32>(NestedData<f32>(1.0f)),
                                   TensorWrapper<f32>(NestedData<f32>(1.0f)));
    // Result must be accessible as a scalar tensorWrapper.
    EXPECT_FLOAT_EQ(result.at({}), 0.0f);
}

// ===========================================================================
// hahaha::MSELoss<T>
// ===========================================================================

class PublicMSELossTest : public ::testing::Test {
  protected:
    MSELoss<f32> loss_;
};

TEST_F(PublicMSELossTest, ZeroLoss_WhenPredictionMatchesTarget) {
    auto yTrue = TensorWrapper<f32>(NestedData<f32>(3.0f));
    auto yPred = TensorWrapper<f32>(NestedData<f32>(3.0f));
    EXPECT_FLOAT_EQ(loss_.computeLoss(yTrue, yPred).at({}), 0.0f);
}

TEST_F(PublicMSELossTest, ScalarInputs_SquaredError) {
    // (3 - 1)^2 = 4
    auto yTrue = TensorWrapper<f32>(NestedData<f32>(3.0f));
    auto yPred = TensorWrapper<f32>(NestedData<f32>(1.0f));
    EXPECT_FLOAT_EQ(loss_.computeLoss(yTrue, yPred).at({}), 4.0f);
}

TEST_F(PublicMSELossTest, VectorInputs_SumOfSquaredErrors) {
    // (1-2)^2 + (2-0)^2 + (3-5)^2 = 1 + 4 + 4 = 9
    TensorWrapper<f32> yTrue(NestedData<f32>{1.0f, 2.0f, 3.0f});
    TensorWrapper<f32> yPred(NestedData<f32>{2.0f, 0.0f, 5.0f});
    EXPECT_FLOAT_EQ(loss_.computeLoss(yTrue, yPred).at({}), 9.0f);
}

TEST_F(PublicMSELossTest, MatrixInputs_SumOfSquaredErrors) {
    // (1-2)^2 + (2-1)^2 + (3-2)^2 + (4-5)^2 = 4
    TensorWrapper<f32> yTrue(NestedData<f32>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<f32> yPred(NestedData<f32>{{2.0f, 1.0f}, {2.0f, 5.0f}});
    EXPECT_FLOAT_EQ(loss_.computeLoss(yTrue, yPred).at({}), 4.0f);
}

TEST_F(PublicMSELossTest, ResultIsAlwaysNonNegative) {
    TensorWrapper<f32> yTrue(NestedData<f32>{-1.0f, 2.0f});
    TensorWrapper<f32> yPred(NestedData<f32>{3.0f, -2.0f});
    EXPECT_GE(loss_.computeLoss(yTrue, yPred).at({}), 0.0f);
}

// ===========================================================================
// hahaha::computeMSELoss free function
// ===========================================================================

TEST(PublicComputeMSELossTest, MatchesMSELossClass_ZeroError) {
    TensorWrapper<f32> y(NestedData<f32>{0.0f, 1.0f});
    EXPECT_FLOAT_EQ(computeMSELoss<f32>(y, y).at({}), 0.0f);
}

TEST(PublicComputeMSELossTest, CorrectValue) {
    // (2 - 0)^2 = 4
    TensorWrapper<f32> yTrue(NestedData<f32>(0.0f));
    TensorWrapper<f32> yPred(NestedData<f32>(2.0f));
    EXPECT_FLOAT_EQ(computeMSELoss<f32>(yTrue, yPred).at({}), 4.0f);
}

// Double-precision variant
TEST(PublicComputeMSELossTest, DoublePrecision) {
    TensorWrapper<f64> yTrue(NestedData<f64>(0.0));
    TensorWrapper<f64> yPred(NestedData<f64>(3.0));
    EXPECT_DOUBLE_EQ(computeMSELoss<f64>(yTrue, yPred).at({}), 9.0);
}
