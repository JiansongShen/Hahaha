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

#include "ml/loss/CrossEntropyLoss.h"

#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

#include "common/definitions.h"
#include "ml/loss/Loss.h"
#include "math/ds/NestedData.h"

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

using NumericTypes = ::testing::Types<f32, f64>;

template <typename T> class CrossEntropyLossTypedTest : public ::testing::Test {
  protected:
    using Type = T;
};

TYPED_TEST_SUITE(CrossEntropyLossTypedTest, NumericTypes);

TYPED_TEST(CrossEntropyLossTypedTest, ScalarInputs_OneHotEncoded_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>(static_cast<T>(1)));
    TensorWrapper<T> yPred(NestedData<T>(static_cast<T>(0.8)));
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    T expected = -static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.8) + 1e-8));
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(CrossEntropyLossTypedTest, VectorInputs_OneHotEncoded_ComputesSumCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(0.9), static_cast<T>(0.05), static_cast<T>(0.05)});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    T expected = -static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.9) + 1e-8));
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(CrossEntropyLossTypedTest, MatrixInputs_MultipleSamples_ComputesSumCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{{static_cast<T>(1), static_cast<T>(0)},
                                          {static_cast<T>(0), static_cast<T>(1)}});
    TensorWrapper<T> yPred(NestedData<T>{{static_cast<T>(0.8), static_cast<T>(0.2)},
                                         {static_cast<T>(0.3), static_cast<T>(0.7)}});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    T term1 = -static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.8) + 1e-8));
    T term2 = -static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.2) + 1e-8));
    T term3 = -static_cast<T>(0) * static_cast<T>(std::log(static_cast<double>(0.3) + 1e-8));
    T term4 = -static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    T expected = term1 + term2 + term3 + term4;
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(CrossEntropyLossTypedTest, ThreeDimensionalTensor_ComputesSumCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{{{static_cast<T>(1), static_cast<T>(0)},
                                          {static_cast<T>(0), static_cast<T>(1)}},
                                         {{static_cast<T>(0), static_cast<T>(1)},
                                          {static_cast<T>(1), static_cast<T>(0)}}});
    TensorWrapper<T> yPred(NestedData<T>{{{static_cast<T>(0.8), static_cast<T>(0.2)},
                                          {static_cast<T>(0.3), static_cast<T>(0.7)}},
                                         {{static_cast<T>(0.4), static_cast<T>(0.6)},
                                          {static_cast<T>(0.9), static_cast<T>(0.1)}}});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(CrossEntropyLossTypedTest, PerfectPrediction_ReturnsSmallValue) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1), static_cast<T>(0)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(0.999999), static_cast<T>(0.000001)});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(CrossEntropyLossTypedTest, ZeroPrediction_HandlesEpsilon) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1), static_cast<T>(0)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(0), static_cast<T>(1)});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(CrossEntropyLossTypedTest, SingleElementVector_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(0.5)});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    T expected = -static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.5) + 1e-8));
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(CrossEntropyLossTypedTest, MultipleClasses_OneHotEncoded_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(0.1), static_cast<T>(0.7), static_cast<T>(0.2)});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    T expected = -static_cast<T>(1) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(CrossEntropyLossTypedTest, SoftTargets_NonOneHot_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(0.8), static_cast<T>(0.2)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(0.7), static_cast<T>(0.3)});
    auto loss = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    T term1 = -static_cast<T>(0.8) * static_cast<T>(std::log(static_cast<double>(0.7) + 1e-8));
    T term2 = -static_cast<T>(0.2) * static_cast<T>(std::log(static_cast<double>(0.3) + 1e-8));
    T expected = term1 + term2;
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TEST(CrossEntropyLossTest, CrossEntropyLossClass_MatchesConvenienceFunction) {
    hahaha::ml::CrossEntropyLoss<f32> loss;
    TensorWrapper<f32> yTrue(NestedData<f32>{1.0f, 0.0f, 0.0f});
    TensorWrapper<f32> yPred(NestedData<f32>{0.9f, 0.05f, 0.05f});
    auto loss1 = loss.computeLoss(yTrue, yPred);
    auto loss2 = hahaha::ml::computeCrossEntropyLoss(yTrue, yPred);
    EXPECT_FLOAT_EQ(loss1.at({}), loss2.at({}));
}
