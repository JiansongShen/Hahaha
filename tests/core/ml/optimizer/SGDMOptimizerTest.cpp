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

#include "ml/optimizer/SGDMOptimizer.h"

#include <gtest/gtest.h>

#include "../../../../core/include/public/Tensor.h"

using namespace hahaha;
using namespace hahaha::ml;

// Use TYPED_TEST to cover float and double
using FloatingPointTypes = ::testing::Types<float, double>;

template <typename T> class SGDMOptimizerTest : public ::testing::Test {
  protected:
    void expectNear(T expected, T actual, T tolerance = 1e-4) {
        EXPECT_NEAR(static_cast<double>(expected), static_cast<double>(actual),
                    static_cast<double>(tolerance));
    }
};

TYPED_TEST_SUITE(SGDMOptimizerTest, FloatingPointTypes);

// Branch 1: trainPrepared_ logic
TYPED_TEST(SGDMOptimizerTest, StandardUpdate_MomentumAccumulation) {
    using T = TypeParam;
    // Initial w = 1.0
    Tensor<T> w(math::NestedData<T>{T(1.0)});
    w.setRequiresGrad(true);

    // lr = 0.1, momentum = 0.9
    T lr = T(0.1);
    T mu = T(0.9);
    std::vector<Tensor<T>> params = {w};
    SGDMOptimizer<T> opt(params, lr, mu);

    // --- Step 1 ---
    // grad = 1.0
    // momentum_v = mu * v_old + (1-mu) * grad
    // v_0 = 0.9 * 0 + 0.1 * 1.0 = 0.1
    // theta_new = theta_old - lr * v_0
    // w = 1.0 - 0.1 * 0.1 = 0.99
    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));
    
    opt.step();
    this->expectNear(T(0.99), w.at({0}));

    // --- Step 2 ---
    // grad = 1.0
    // v_1 = 0.9 * 0.1 + 0.1 * 1.0 = 0.09 + 0.1 = 0.19
    // w = 0.99 - 0.1 * 0.19 = 0.99 - 0.019 = 0.971
    w.clearGrad();
    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));
    
    opt.step();
    this->expectNear(T(0.971), w.at({0}));
}

// Branch 2: requiresGrad is false
TYPED_TEST(SGDMOptimizerTest, Step_RequiresGradFalse_NoUpdate) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0)});
    w.setRequiresGrad(false);

    SGDMOptimizer<T> opt({w}, T(0.1), T(0.9));

    // Even if we somehow force a grad (though typically backward wouldn't run),
    // let's manually set one to ensure step() ignores it.
    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));

    opt.step();
    // Should remain 1.0
    EXPECT_EQ(w.at({0}), T(1.0));
}

// Branch 3: grad is empty
TYPED_TEST(SGDMOptimizerTest, Step_GradEmpty_NoUpdate) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0)});
    w.setRequiresGrad(true);

    SGDMOptimizer<T> opt({w}, T(0.1), T(0.9));

    // No gradient accumulated
    opt.step();

    // Should remain 1.0
    EXPECT_EQ(w.at({0}), T(1.0));
}

// Branch 4 & 5: addParameter and preTrainIfNeed logic
TYPED_TEST(SGDMOptimizerTest, AddParameter_BeforeAndAfterStep) {
    using T = TypeParam;
    T lr = T(0.1);
    T mu = T(0.9);

    Tensor<T> w1(math::NestedData<T>{T(1.0)});
    w1.setRequiresGrad(true);

    SGDMOptimizer<T> opt({w1}, lr, mu);

    // --- Step 1: w1 update ---
    // grad(w1) = 1.0
    // v(w1) = 0.1 * 1.0 = 0.1
    // w1 = 1.0 - 0.1 * 0.1 = 0.99
    w1.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));
    opt.step();
    this->expectNear(T(0.99), w1.at({0}));

    // --- Add w2 after step (trainPrepared_ is true) ---
    Tensor<T> w2(math::NestedData<T>{T(2.0)});
    w2.setRequiresGrad(true);
    opt.addParameter(w2);

    // --- Step 2: w1 and w2 update ---
    // grad(w1) = 1.0
    // v(w1) = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
    // w1 = 0.99 - 0.1 * 0.19 = 0.971
    
    // grad(w2) = 1.0
    // v(w2) (initially 0) = 0.9 * 0 + 0.1 * 1.0 = 0.1
    // w2 = 2.0 - 0.1 * 0.1 = 1.99

    w1.clearGrad();
    w1.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));
    
    w2.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));

    opt.step();

    this->expectNear(T(0.971), w1.at({0}));
    this->expectNear(T(1.99), w2.at({0}));
}

TYPED_TEST(SGDMOptimizerTest, ZeroGrad) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0)});
    w.setRequiresGrad(true);
    SGDMOptimizer<T> opt({w}, T(0.1));

    w.getComputeNode()->accumulateGrad(
        std::make_shared<math::TensorWrapper<T>>(math::TensorShape({1}), T(1.0)));

    ASSERT_FALSE(w.grad().isEmpty());
    opt.zeroGrad();

    auto g = w.grad();
    ASSERT_FALSE(g.isEmpty());
    EXPECT_EQ(g.at({0}), T(0.0));
}
