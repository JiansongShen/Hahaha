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

#include "ml/loss/MAELoss.h"

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

using NumericTypes = ::testing::Types<i8, i16, i32, i64, f32, f64>;

template <typename T> class MAELossTypedTest : public ::testing::Test {
  protected:
    using Type = T;
};

TYPED_TEST_SUITE(MAELossTypedTest, NumericTypes);

TYPED_TEST(MAELossTypedTest, ScalarInputs_ComputesMeanAbsoluteError) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(3)));
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>(static_cast<T>(1)));
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(2.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, VectorInputs_ComputesMeanOfAbsoluteErrors) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(2), static_cast<T>(0), static_cast<T>(5)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    // Errors: [1-2, 2-0, 3-5] = [-1, 2, -2]
    // Absolute errors: [1, 2, 2]
    // Mean: (1 + 2 + 2) / 3 = 5/3 ≈ 1.667
    T expected = static_cast<T>(5.0 / 3.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, MatrixInputs_ComputesMeanAcrossAllElements) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{static_cast<T>(1), static_cast<T>(2)},
                                          {static_cast<T>(3), static_cast<T>(4)}});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{static_cast<T>(2), static_cast<T>(1)},
                                         {static_cast<T>(2), static_cast<T>(5)}});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(1.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, ThreeDimensionalTensor_ComputesMeanCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{{static_cast<T>(1), static_cast<T>(2)},
                                          {static_cast<T>(3), static_cast<T>(4)}},
                                         {{static_cast<T>(5), static_cast<T>(6)},
                                          {static_cast<T>(7), static_cast<T>(8)}}});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{{{static_cast<T>(2), static_cast<T>(1)},
                                          {static_cast<T>(2), static_cast<T>(5)}},
                                         {{static_cast<T>(4), static_cast<T>(7)},
                                          {static_cast<T>(6), static_cast<T>(9)}}});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(1.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, ZeroError_ReturnsZero) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, NegativeErrors_UsesAbsoluteValue) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(1), static_cast<T>(2)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(3), static_cast<T>(0)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(2.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, SingleElementVector_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(5)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(2)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(3.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(MAELossTypedTest, MixedPositiveNegativeErrors_HandlesCorrectly) {
    using T = typename TestFixture::Type;
    auto yTrueData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(5), static_cast<T>(1), static_cast<T>(3)});
    auto yPredData = std::make_shared<TensorWrapper<T>>(NestedData<T>{static_cast<T>(2), static_cast<T>(4), static_cast<T>(3)});
    auto yTrueNode = std::make_shared<ComputeNode<T>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<T>>(yPredData);
    auto lossNode = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto lossData = lossNode->getData();
    T expected = static_cast<T>(2.0);
    EXPECT_NEAR(static_cast<double>(lossData->at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TEST(MAELossTest, MAELossClass_MatchesConvenienceFunction) {
    hahaha::ml::MAELoss<f32> loss;
    auto yTrueData = std::make_shared<TensorWrapper<f32>>(NestedData<f32>{1.0f, 2.0f, 3.0f});
    auto yPredData = std::make_shared<TensorWrapper<f32>>(NestedData<f32>{2.0f, 0.0f, 5.0f});
    auto yTrueNode = std::make_shared<ComputeNode<f32>>(yTrueData);
    auto yPredNode = std::make_shared<ComputeNode<f32>>(yPredData);
    auto loss1Node = loss.computeLoss(yTrueNode, yPredNode);
    auto loss2Node = hahaha::ml::computeMAELoss(yTrueNode, yPredNode);
    auto loss1Data = loss1Node->getData();
    auto loss2Data = loss2Node->getData();
    EXPECT_FLOAT_EQ(loss1Data->at({}), loss2Data->at({}));
}
