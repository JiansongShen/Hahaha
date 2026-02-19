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

// Tests for the public hahaha::Parameters<T> wrapper defined in
// public/Parameters.h.  The inner ml::Parameters is tested separately in
// tests/core/ml/ParametersTest.cpp.

#include <gtest/gtest.h>

#include "math/ds/TensorData.h"
#include "public/Parameters.h"
#include "public/Tensor.h"

using namespace hahaha;
using namespace hahaha::math;
using namespace hahaha::common;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class PublicParametersTest : public ::testing::Test {
  protected:
    void SetUp() override {
        p1_ = Tensor<f32>(NestedData<f32>(1.0f));
        p2_ = Tensor<f32>(NestedData<f32>(2.0f));
    }

    Tensor<f32> p1_, p2_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(PublicParametersTest, DefaultConstructed_IsEmpty) {
    Parameters<f32> params;
    EXPECT_TRUE(params.getParameters().empty());
}

TEST_F(PublicParametersTest, AddSingleParameter) {
    Parameters<f32> params;
    params.addParameter(p1_);
    EXPECT_EQ(params.getParameters().size(), 1u);
}

TEST_F(PublicParametersTest, AddMultipleParameters) {
    Parameters<f32> params;
    params.addParameter(p1_);
    params.addParameter(p2_);
    EXPECT_EQ(params.getParameters().size(), 2u);
}

TEST_F(PublicParametersTest, ParameterValuesPreserved) {
    Parameters<f32> params;
    params.addParameter(p1_);
    params.addParameter(p2_);

    const auto& vec = params.getParameters();
    EXPECT_FLOAT_EQ(vec[0].at({}), 1.0f);
    EXPECT_FLOAT_EQ(vec[1].at({}), 2.0f);
}

TEST_F(PublicParametersTest, ConstGetParameters_ReturnsConstRef) {
    Parameters<f32> params;
    params.addParameter(p1_);

    const Parameters<f32>& constRef = params;
    const auto& vec = constRef.getParameters();
    EXPECT_EQ(vec.size(), 1u);
    EXPECT_FLOAT_EQ(vec[0].at({}), 1.0f);
}

TEST_F(PublicParametersTest, MutableGetParameters_AllowsModification) {
    Parameters<f32> params;
    params.addParameter(p1_);

    // Mutable access — update the value through the wrapper.
    auto& vec = params.getParameters();
    vec[0].data()->at({}) = 42.0f;

    EXPECT_FLOAT_EQ(params.getParameters()[0].at({}), 42.0f);
}
