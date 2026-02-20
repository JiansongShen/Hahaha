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

#include "math/TensorComputeFuns.h"

#include <cmath>
#include <gtest/gtest.h>

#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"

using namespace hahaha;
using namespace hahaha::math;

template <typename T> class TensorComputeFunsTypedTest : public ::testing::Test {
  protected:
    void expectNear(T expected, T actual, double tolerance = 1e-5) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(static_cast<double>(expected),
                        static_cast<double>(actual),
                        tolerance);
        } else {
            EXPECT_EQ(expected, actual);
        }
    }
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TensorComputeFunsTypedTest, FloatTypes);

// ============================================================================
// Sqrt Tests
// ============================================================================

TYPED_TEST(TensorComputeFunsTypedTest, Sqrt_0D) {
    using T = TypeParam;
    TensorWrapper<T> t(T(4.0));
    auto res = TensorComputeFun::sqrt(t);
    this->expectNear(T(2.0), res.at({}));
}

TYPED_TEST(TensorComputeFunsTypedTest, Sqrt_1D) {
    using T = TypeParam;
    TensorWrapper<T> t(NestedData<T>{T(1.0), T(4.0), T(9.0)});
    auto res = TensorComputeFun::sqrt(t);
    this->expectNear(T(1.0), res.at({0}));
    this->expectNear(T(2.0), res.at({1}));
    this->expectNear(T(3.0), res.at({2}));
}

TYPED_TEST(TensorComputeFunsTypedTest, Sqrt_2D) {
    using T = TypeParam;
    TensorWrapper<T> t(NestedData<T>{{T(1.0), T(4.0)}, {T(9.0), T(16.0)}});
    auto res = TensorComputeFun::sqrt(t);
    this->expectNear(T(1.0), res.at({0, 0}));
    this->expectNear(T(4.0), res.at({1, 1}));
}

TYPED_TEST(TensorComputeFunsTypedTest, Sqrt_3D) {
    using T = TypeParam;
    TensorWrapper<T> t(NestedData<T>{{{T(1.0)}}, {{T(4.0)}}});
    auto res = TensorComputeFun::sqrt(t);
    this->expectNear(T(1.0), res.at({0, 0, 0}));
    this->expectNear(T(2.0), res.at({1, 0, 0}));
}

// ============================================================================
// Square Tests
// ============================================================================

TYPED_TEST(TensorComputeFunsTypedTest, Square_0D) {
    using T = TypeParam;
    TensorWrapper<T> t(T(3.0));
    auto res = TensorComputeFun::square(t);
    this->expectNear(T(9.0), res.at({}));
}

TYPED_TEST(TensorComputeFunsTypedTest, Square_1D) {
    using T = TypeParam;
    TensorWrapper<T> t(NestedData<T>{T(1.0), T(2.0), T(3.0)});
    auto res = TensorComputeFun::square(t);
    this->expectNear(T(1.0), res.at({0}));
    this->expectNear(T(4.0), res.at({1}));
    this->expectNear(T(9.0), res.at({2}));
}

TYPED_TEST(TensorComputeFunsTypedTest, Square_2D) {
    using T = TypeParam;
    TensorWrapper<T> t(NestedData<T>{{T(1.0), T(2.0)}, {T(3.0), T(4.0)}});
    auto res = TensorComputeFun::square(t);
    this->expectNear(T(1.0), res.at({0, 0}));
    this->expectNear(T(16.0), res.at({1, 1}));
}

TYPED_TEST(TensorComputeFunsTypedTest, Square_3D) {
    using T = TypeParam;
    TensorWrapper<T> t(NestedData<T>{{{T(1.0)}}, {{T(2.0)}}});
    auto res = TensorComputeFun::square(t);
    this->expectNear(T(1.0), res.at({0, 0, 0}));
    this->expectNear(T(4.0), res.at({1, 0, 0}));
}
