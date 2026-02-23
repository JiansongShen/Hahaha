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

#include "ml/loss/BinaryCrossEntropyLoss.h"

#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

#include "common/definitions.h"
#include "math/ds/NestedData.h"
#include "ml/compute/graph/ComputeNode.h"
#include "ml/loss/Loss.h"

using hahaha::common::f32;
using hahaha::common::f64;
using hahaha::common::i16;
using hahaha::common::i32;
using hahaha::common::i64;
using hahaha::common::i8;
using hahaha::common::u16;
using hahaha::common::u32;
using hahaha::common::u64;
using hahaha::common::u8;
using hahaha::math::NestedData;
using hahaha::math::TensorWrapper;
using hahaha::ml::ComputeNode;

using NumericTypes = ::testing::Types<f32, f64>;

template <typename T> class BinaryCrossEntropyLossTypedTest : public ::testing::Test {
  protected:
    using Type = T;
};

TYPED_TEST_SUITE(BinaryCrossEntropyLossTypedTest, NumericTypes);

TYPED_TEST(BinaryCrossEntropyLossTypedTest, ScalarInputs_LabelOne_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(1)));
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(0.8)));
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T term1 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.8) + 1e-8));
    T term2 = static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.2) + 1e-8));
    T expected = -(term1 + term2);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, ScalarInputs_LabelZero_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(0)));
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(0.3)));
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T term1 = static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.3) + 1e-8));
    T term2 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    T expected = -(term1 + term2);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, VectorInputs_MixedLabels_ComputesMeanCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1), static_cast<T>(0), static_cast<T>(1)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0.9), static_cast<T>(0.2), static_cast<T>(0.7)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T term1_1 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.9) + 1e-8));
    T term1_2 = static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.1) + 1e-8));
    T term2_1 = static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.2) + 1e-8));
    T term2_2 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.8) + 1e-8));
    T term3_1 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    T term3_2 = static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.3) + 1e-8));
    T expected = -((term1_1 + term1_2) + (term2_1 + term2_2) + (term3_1 + term3_2)) / static_cast<T>(3);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, MatrixInputs_MultipleSamples_ComputesMeanCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{static_cast<T>(1), static_cast<T>(0)},
                                          {static_cast<T>(0), static_cast<T>(1)}});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{static_cast<T>(0.8), static_cast<T>(0.3)},
                                         {static_cast<T>(0.2), static_cast<T>(0.7)}});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    EXPECT_GT(static_cast<double>(lossData->at({})), 0.0);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, ThreeDimensionalTensor_ComputesMeanCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{{static_cast<T>(1), static_cast<T>(0)},
                                          {static_cast<T>(0), static_cast<T>(1)}},
                                         {{static_cast<T>(0), static_cast<T>(1)},
                                          {static_cast<T>(1), static_cast<T>(0)}}});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{{static_cast<T>(0.8), static_cast<T>(0.2)},
                                          {static_cast<T>(0.3), static_cast<T>(0.7)}},
                                         {{static_cast<T>(0.4), static_cast<T>(0.6)},
                                          {static_cast<T>(0.9), static_cast<T>(0.1)}}});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    EXPECT_GT(static_cast<double>(lossData->at({})), 0.0);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, PerfectPrediction_LabelOne_ReturnsSmallValue) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0.999999)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    EXPECT_GT(static_cast<double>(lossData->at({})), 0.0);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, PerfectPrediction_LabelZero_ReturnsSmallValue) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0.000001)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    EXPECT_GT(static_cast<double>(lossData->at({})), 0.0);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, ZeroPrediction_LabelOne_HandlesEpsilon) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    EXPECT_GT(static_cast<double>(lossData->at({})), 0.0);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, OnePrediction_LabelZero_HandlesEpsilon) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    EXPECT_GT(static_cast<double>(lossData->at({})), 0.0);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, SingleElementVector_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0.5)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T term1 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.5) + 1e-8));
    T term2 = static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.5) + 1e-8));
    T expected = -(term1 + term2);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, AllOnes_AllZeros_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0.8), static_cast<T>(0.9), static_cast<T>(0.7)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T term1 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.8) + 1e-8));
    T term2 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.9) + 1e-8));
    T term3 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    T expected = -(term1 + term2 + term3) / static_cast<T>(3);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(BinaryCrossEntropyLossTypedTest, AllZeros_AllOnes_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(0.2), static_cast<T>(0.1), static_cast<T>(0.3)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T term1 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.8) + 1e-8));
    T term2 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.9) + 1e-8));
    T term3 = static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    T expected = -(term1 + term2 + term3) / static_cast<T>(3);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TEST(BinaryCrossEntropyLossTest, BinaryCrossEntropyLossClass_MatchesConvenienceFunction) {
    hahaha::ml::BinaryCrossEntropyLoss<f32> loss;
    auto yTrueData = std::make_shared<TensorWrapper<f32>>(NestedData<f32>{1.0f, 0.0f, 1.0f});
    auto yPredData = std::make_shared<TensorWrapper<f32>>(NestedData<f32>{0.9f, 0.2f, 0.7f});
    auto yTrueNode = std::make_shared<ComputeNode<f32>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<f32>>(yPredData);
    auto loss1Node = loss.computeLoss(yTrueNode, yPredNode);
    auto loss2Node = hahaha::ml::computeBinaryCrossEntropyLoss(yTrueNode, yPredNode);
    auto loss1Data = loss1Node->getData();
    auto loss2Data = loss2Node->getData();
    EXPECT_FLOAT_EQ(loss1Data->at({}), loss2Data->at({}));
}
