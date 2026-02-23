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

#include "ml/loss/LogCoshLoss.h"

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

template <typename T> class LogCoshLossTypedTest : public ::testing::Test {
  protected:
    using Type = T;
};

TYPED_TEST_SUITE(LogCoshLossTypedTest, NumericTypes);

TYPED_TEST(LogCoshLossTypedTest, ScalarInputs_ComputesLogCoshCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>(static_cast<T>(3)));
    TensorWrapper<T> yPred(NestedData<T>(static_cast<T>(1)));
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    T error = static_cast<T>(2);
    T expected = static_cast<T>(std::log(std::cosh(static_cast<double>(error))));
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(LogCoshLossTypedTest, VectorInputs_ComputesMeanOfLogCosh) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(2), static_cast<T>(0), static_cast<T>(5)});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    T error1 = static_cast<T>(-1);
    T error2 = static_cast<T>(2);
    T error3 = static_cast<T>(-2);
    T expected = (static_cast<T>(std::log(std::cosh(static_cast<double>(error1)))) +
                  static_cast<T>(std::log(std::cosh(static_cast<double>(error2)))) +
                  static_cast<T>(std::log(std::cosh(static_cast<double>(error3))))) /
                 static_cast<T>(3);
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(LogCoshLossTypedTest, MatrixInputs_ComputesMeanAcrossAllElements) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{{static_cast<T>(1), static_cast<T>(2)},
                                         {static_cast<T>(3), static_cast<T>(4)}});
    TensorWrapper<T> yPred(NestedData<T>{{static_cast<T>(2), static_cast<T>(1)},
                                         {static_cast<T>(2), static_cast<T>(5)}});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(LogCoshLossTypedTest, ThreeDimensionalTensor_ComputesMeanCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{{{static_cast<T>(1), static_cast<T>(2)},
                                         {static_cast<T>(3), static_cast<T>(4)}},
                                        {{static_cast<T>(5), static_cast<T>(6)},
                                         {static_cast<T>(7), static_cast<T>(8)}}});
    TensorWrapper<T> yPred(NestedData<T>{{{static_cast<T>(2), static_cast<T>(1)},
                                          {static_cast<T>(2), static_cast<T>(5)}},
                                         {{static_cast<T>(4), static_cast<T>(7)},
                                          {static_cast<T>(6), static_cast<T>(9)}}});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(LogCoshLossTypedTest, ZeroError_ReturnsZero) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    T expected = static_cast<T>(0);
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-6);
}

TYPED_TEST(LogCoshLossTypedTest, NegativeErrors_HandlesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1), static_cast<T>(2)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(3), static_cast<T>(0)});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(LogCoshLossTypedTest, SingleElementVector_ComputesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(5)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(2)});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    T error = static_cast<T>(3);
    T expected = static_cast<T>(std::log(std::cosh(static_cast<double>(error))));
    EXPECT_NEAR(static_cast<double>(loss.at({})), static_cast<double>(expected),
                static_cast<double>(expected) * 1e-5);
}

TYPED_TEST(LogCoshLossTypedTest, LargeErrors_HandlesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(10)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(1)});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TYPED_TEST(LogCoshLossTypedTest, SmallErrors_HandlesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> yTrue(NestedData<T>{static_cast<T>(1.1)});
    TensorWrapper<T> yPred(NestedData<T>{static_cast<T>(1.0)});
    auto loss = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    EXPECT_GT(static_cast<double>(loss.at({})), 0.0);
}

TEST(LogCoshLossTest, LogCoshLossClass_MatchesConvenienceFunction) {
    hahaha::ml::LogCoshLoss<f32> loss;
    TensorWrapper<f32> yTrue(NestedData<f32>{1.0f, 2.0f, 3.0f});
    TensorWrapper<f32> yPred(NestedData<f32>{2.0f, 0.0f, 5.0f});
    auto loss1 = loss.computeLoss(yTrue, yPred);
    auto loss2 = hahaha::ml::computeLogCoshLoss(yTrue, yPred);
    EXPECT_FLOAT_EQ(loss1.at({}), loss2.at({}));
}
