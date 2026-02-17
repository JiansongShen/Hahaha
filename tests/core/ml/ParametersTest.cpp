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

#include "ml/Parameters.h"

#include <gtest/gtest.h>

using hahaha::Tensor;
using hahaha::math::NestedData;

TEST(ParametersTest, AddAndGetParameters_Works) {
    hahaha::ml::Parameters<float> params;
    params.addParameter(Tensor<float>(NestedData<float>(1.0f)));
    params.addParameter(Tensor<float>(NestedData<float>(2.0f)));

    const auto& constParams = params;
    const auto& vec = constParams.getParameters();
    ASSERT_EQ(vec.size(), 2u);
    EXPECT_FLOAT_EQ(vec[0].at({}), 1.0f);
    EXPECT_FLOAT_EQ(vec[1].at({}), 2.0f);
}
