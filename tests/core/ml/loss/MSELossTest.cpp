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

#include "ml/loss/MSELoss.h"

#include <gtest/gtest.h>

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

using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T> class MSELossTypedTest : public ::testing::Test {
  protected:
    using Type = T;
};

TYPED_TEST_SUITE(MSELossTypedTest, NumericTypes);

TYPED_TEST(MSELossTypedTest, ScalarInputs_ComputesMeanSquaredError) {
    using T = typename TestFixture::Type;
    auto yTrueData =
        std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(3)));
    auto yPredData =
        std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(1)));
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(4.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MSELossTypedTest, VectorInputs_ComputesMeanOfSquaredErrors) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{static_cast<T>(2), static_cast<T>(0), static_cast<T>(5)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(3.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MSELossTypedTest, MatrixInputs_ComputesMeanAcrossAllElements) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{{static_cast<T>(1), static_cast<T>(2)},
                      {static_cast<T>(3), static_cast<T>(4)}});
    auto yPredData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{{static_cast<T>(2), static_cast<T>(1)},
                      {static_cast<T>(2), static_cast<T>(5)}});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(1.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MSELossTypedTest, ThreeDimensionalTensor_ComputesMeanCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{{{static_cast<T>(1), static_cast<T>(2)},
                       {static_cast<T>(3), static_cast<T>(4)}},
                      {{static_cast<T>(5), static_cast<T>(6)},
                       {static_cast<T>(7), static_cast<T>(8)}}});
    auto yPredData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{{{static_cast<T>(2), static_cast<T>(1)},
                       {static_cast<T>(2), static_cast<T>(5)}},
                      {{static_cast<T>(4), static_cast<T>(7)},
                       {static_cast<T>(6), static_cast<T>(9)}}});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(1.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MSELossTypedTest, ZeroError_ReturnsZero) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MSELossTypedTest, NegativeErrors_HandlesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{static_cast<T>(1), static_cast<T>(2)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(
        NestedData<T>{static_cast<T>(3), static_cast<T>(0)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    // Errors: [1-3, 2-0] = [-2, 2]
    // Squared errors: [4, 4]
    // Mean: (4 + 4) / 2 = 4
    T expected = static_cast<T>(4.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MSELossTypedTest, SingleElementVector_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData =
        std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(5)});
    auto yPredData =
        std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(2)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(9.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})),
                static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TEST(MSELossTest, LossBase_DefaultReturnsZeroTensor) {
    hahaha::ml::Loss<f32> loss;
    auto yTrueData = std::make_shared<TensorWrapper<f32>>(NestedData<f32>(1.0f));
    auto yPredData = std::make_shared<TensorWrapper<f32>>(NestedData<f32>(2.0f));
    auto yTrueNode = std::make_shared<ComputeNode<f32>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<f32>>(yPredData);
    auto outNode = loss.computeLoss(yTrueNode, yPredNode);
    auto outData = outNode->getData();
    EXPECT_FLOAT_EQ(outData->at({}), 0.0f);
}

TEST(MSELossTest, MSELossClass_MatchesConvenienceFunction) {
    hahaha::ml::MSELoss<f32> loss;
    auto yTrueData =
        std::make_shared<TensorWrapper<f32>>(NestedData<f32>{1.0f, 2.0f, 3.0f});
    auto yPredData =
        std::make_shared<TensorWrapper<f32>>(NestedData<f32>{2.0f, 0.0f, 5.0f});
    auto yTrueNode = std::make_shared<ComputeNode<f32>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<f32>>(yPredData);
    auto loss1Node = loss.computeLoss(yTrueNode, yPredNode);
    auto loss2Node = hahaha::ml::computeMSELoss(yTrueNode, yPredNode);
    auto loss1Data = loss1Node->getData();
    auto loss2Data = loss2Node->getData();
    EXPECT_FLOAT_EQ(loss1Data->at({}), loss2Data->at({}));
}
