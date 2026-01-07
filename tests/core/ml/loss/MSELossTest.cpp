// Copyright (c) 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include "ml/loss/MSELoss.h"

#include <gtest/gtest.h>

#include "ml/loss/Loss.h"

using hahaha::Tensor;
using hahaha::math::NestedData;

TEST(MSELossTest, LossBase_DefaultReturnsZeroTensor) {
    hahaha::ml::Loss<float> loss;
    auto out = loss.computeLoss(Tensor<float>(NestedData<float>(1.0f)),
                                Tensor<float>(NestedData<float>(2.0f)));
    EXPECT_FLOAT_EQ(out.at({}), 0.0f);
}

TEST(MSELossTest, ComputeMSELoss_ScalarInputs_MatchesSquaredError) {
    Tensor<float> yTrue(NestedData<float>(3.0f));
    Tensor<float> yPred(NestedData<float>(1.0f));
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // (3-1)^2 = 4
    EXPECT_FLOAT_EQ(loss.at({}), 4.0f);
}

TEST(MSELossTest, ComputeMSELoss_VectorInputs_SumsSquaredError) {
    Tensor<float> yTrue(NestedData<float>{1.0f, 2.0f, 3.0f});
    Tensor<float> yPred(NestedData<float>{2.0f, 0.0f, 5.0f});
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // (1-2)^2 + (2-0)^2 + (3-5)^2 = 1 + 4 + 4 = 9
    EXPECT_FLOAT_EQ(loss.at({}), 9.0f);
}
