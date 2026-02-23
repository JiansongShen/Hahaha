// Copyright (c) 2025 - 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

using hahaha::math::NestedData;
using hahaha::math::TensorWrapper;

TEST(MSELossTest, LossBase_DefaultReturnsZeroTensor) {
    hahaha::ml::Loss<float> loss;
    auto out = loss.computeLoss(TensorWrapper<float>(NestedData<float>(1.0f)),
                                TensorWrapper<float>(NestedData<float>(2.0f)));
    EXPECT_FLOAT_EQ(out.at({}), 0.0f);
}

TEST(MSELossTest, ComputeMSELoss_ScalarInputs_MatchesSquaredError) {
    TensorWrapper<float> yTrue(NestedData<float>(3.0f));
    TensorWrapper<float> yPred(NestedData<float>(1.0f));
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // (3-1)^2 = 4
    EXPECT_FLOAT_EQ(loss.at({}), 4.0f);
}

TEST(MSELossTest, ComputeMSELoss_VectorInputs_SumsSquaredError) {
    TensorWrapper<float> yTrue(NestedData<float>{1.0f, 2.0f, 3.0f});
    TensorWrapper<float> yPred(NestedData<float>{2.0f, 0.0f, 5.0f});
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // (1-2)^2 + (2-0)^2 + (3-5)^2 = 1 + 4 + 4 = 9
    EXPECT_FLOAT_EQ(loss.at({}), 9.0f);
}

// ============================================================================
// Dimension-specific tests: 0D to 3D loss computation
// ============================================================================

TEST(MSELossTest, ComputeMSELoss_ZeroDimension_Scalar) {
    // 0D scalar inputs
    TensorWrapper<float> yTrue(NestedData<float>(3.0f)); // Scalar tensoWrapperr
    TensorWrapper<float> yPred(NestedData<float>(1.0f));
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // (3-1)^2 = 4
    EXPECT_FLOAT_EQ(loss.at({}), 4.0f);
}

TEST(MSELossTest, ComputeMSELoss_OneDimension_Vector) {
    // 1D vector inputs (already tested above, but ensure it works)
    TensorWrapper<float> yTrue(NestedData<float>{1.0f, 2.0f, 3.0f});
    TensorWrapper<float> yPred(NestedData<float>{2.0f, 0.0f, 5.0f});
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    EXPECT_FLOAT_EQ(loss.at({}), 9.0f);
}

TEST(MSELossTest, ComputeMSELoss_TwoDimension_Matrix) {
    // 2D matrix inputs
    TensorWrapper<float> yTrue(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> yPred(NestedData<float>{{2.0f, 1.0f}, {2.0f, 5.0f}});
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // (1-2)^2 + (2-1)^2 + (3-2)^2 + (4-5)^2 = 1 + 1 + 1 + 1 = 4
    EXPECT_FLOAT_EQ(loss.at({}), 4.0f);
}

TEST(MSELossTest, ComputeMSELoss_ThreeDimension_TensorWrapper) {
    // 3D tensorWrapper inputs
    TensorWrapper<float> yTrue(NestedData<float>{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                                                 {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    TensorWrapper<float> yPred(NestedData<float>{{{2.0f, 1.0f}, {2.0f, 5.0f}},
                                                 {{4.0f, 7.0f}, {6.0f, 9.0f}}});
    auto loss = hahaha::ml::computeMSELoss(yTrue, yPred);
    // Sum of all squared errors across all 8 elements
    // (1-2)^2 + (2-1)^2 + (3-2)^2 + (4-5)^2 + (5-4)^2 + (6-7)^2 + (7-6)^2 +
    // (8-9)^2 = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 8
    EXPECT_FLOAT_EQ(loss.at({}), 8.0f);
}

TEST(MSELossTest, ComputeMSELoss_ScalarTypes_Distinction) {
    // Test different scalar representations
    // All should produce same loss value since they represent the same value

    // 0D scalar
    TensorWrapper<float> yTrue0(NestedData<float>(3.0f));
    TensorWrapper<float> yPred0(NestedData<float>(1.0f));
    auto loss0 = hahaha::ml::computeMSELoss(yTrue0, yPred0);

    // 1D scalar [1]
    TensorWrapper<float> yTrue1(NestedData<float>{3.0f});
    TensorWrapper<float> yPred1(NestedData<float>{1.0f});
    auto loss1 = hahaha::ml::computeMSELoss(yTrue1, yPred1);

    // 2D scalar [1, 1]
    TensorWrapper<float> yTrue2(NestedData<float>{{3.0f}});
    TensorWrapper<float> yPred2(NestedData<float>{{1.0f}});
    auto loss2 = hahaha::ml::computeMSELoss(yTrue2, yPred2);

    // 3D scalar [1, 1, 1]
    TensorWrapper<float> yTrue3(NestedData<float>{{{3.0f}}});
    TensorWrapper<float> yPred3(NestedData<float>{{{1.0f}}});
    auto loss3 = hahaha::ml::computeMSELoss(yTrue3, yPred3);

    // All should give same result: (3-1)^2 = 4
    EXPECT_FLOAT_EQ(loss0.at({}), 4.0f);
    EXPECT_FLOAT_EQ(loss1.at({}), 4.0f);
    EXPECT_FLOAT_EQ(loss2.at({}), 4.0f);
    EXPECT_FLOAT_EQ(loss3.at({}), 4.0f);
}
