// Copyright (c) 2026-2026 Contributors of Hahaha
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
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include "ml/optimizer/AdamWOptimizer.h"

#include <gtest/gtest.h>

#include "public/Tensor.h"
using namespace hahaha;
using namespace hahaha::ml;

// Define floating point types for AdamW
using FloatingPointTypes = ::testing::Types<f32, f64>;

template <typename T> class AdamWOptimizerTest : public ::testing::Test {
  protected:
    // Helper to compare values with epsilon
    void expectNear(T expected, T actual, T tolerance = 1e-4) {
        EXPECT_NEAR(static_cast<double>(expected),
                    static_cast<double>(actual),
                    static_cast<double>(tolerance));
    }

    // Helper to create a gradient wrapper
    std::shared_ptr<math::TensorWrapper<T>>
    createGrad(const std::vector<size_t>& shape, T value) {
        return std::make_shared<math::TensorWrapper<T>>(math::TensorShape(shape),
                                                        value);
    }
};

TYPED_TEST_SUITE(AdamWOptimizerTest, FloatingPointTypes);

// ============================================================================
// Constructor Tests
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, Constructor_Standard) {
    using T = TypeParam;
    Tensor<T> w1(T(1.0));
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    T lr = T(0.01);

    AdamWOptimizer<T> opt(params, lr);

    EXPECT_EQ(opt.getParameters().size(), 1);
    EXPECT_EQ(opt.getLearningRate(), lr);
}

// ============================================================================
// Standard Dimension Updates (Without Weight Decay)
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, Update_0D_Scalar) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    // Set weightDecay to 0 to test pure Adam logic
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamWOptimizer<T> opt(params, T(0.1), 0.9, 0.999, 1e-8, 0.0);
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    // First step: update ≈ 0.1
    this->expectNear(T(9.9), w.at({}));
}

TYPED_TEST(AdamWOptimizerTest, Update_1D_Vector) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0), T(2.0)});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamWOptimizer<T> opt(params, T(0.1), 0.9, 0.999, 1e-8, 0.0);
    w.getComputeNode()->accumulateGrad(this->createGrad({2}, T(0.5)));
    opt.step();
    this->expectNear(T(0.9), w.at({0}));
    this->expectNear(T(1.9), w.at({1}));
}

// ============================================================================
// Weight Decay Tests
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, WeightDecay_Logic) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);

    T lr = T(0.1);
    T wd = T(0.01);
    // Grad = 0 to test pure Weight Decay effect
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamWOptimizer<T> opt(params, lr, 0.9, 0.999, 1e-8, wd);
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.0)));

    opt.step();

    // AdamW formula: theta = theta - lr * wd * theta - lr * AdamUpdate
    // Since Grad=0, AdamUpdate=0
    // theta = 10.0 - 0.1 * 0.01 * 10.0 = 10.0 - 0.01 = 9.99
    this->expectNear(T(9.99), w.at({}));
}

TYPED_TEST(AdamWOptimizerTest, WeightDecay_WithGrad) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);

    T lr = T(0.1);
    T wd = T(0.01);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamWOptimizer<T> opt(params, lr, 0.9, 0.999, 1e-8, wd);

    // Grad = 1.0 -> AdamUpdate ≈ 1.0
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();

    // theta = 10.0 - (lr * wd * 10.0) - (lr * AdamUpdate)
    // theta = 10.0 - (0.1 * 0.01 * 10.0) - (0.1 * 1.0)
    // theta = 10.0 - 0.01 - 0.1 = 9.89
    this->expectNear(T(9.89), w.at({}));
}

// ============================================================================
// Error Handling: RequiresGrad is False
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, RequiresGradFalse_NoUpdate) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(false);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamWOptimizer<T> opt(params, T(0.1), 0.9, 0.999, 1e-8, 0.01);
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({}), T(10.0));
}

// ============================================================================
// Multi-Dimensional AddParameter Tests
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, AddParameter_3D_AfterStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{{{T(1.0)}}});
    w1.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamWOptimizer<T> opt(params, T(0.1), 0.9, 0.999, 1e-8, 0.0);
    w1.getComputeNode()->accumulateGrad(this->createGrad({1, 1, 1}, T(1.0)));
    opt.step(); // turn 1

    Tensor<T> w2(math::NestedData<T>{{{T(10.0), T(20.0)}}});
    w2.setRequiresGrad(true);
    opt.addParameter(w2.getComputeNode());

    w2.getComputeNode()->accumulateGrad(this->createGrad({1, 1, 2}, T(1.0)));
    opt.step(); // turn 2

    // t=2 update ≈ 0.0744137
    T expectedUpdate = T(0.0744137);
    this->expectNear(T(10.0 - expectedUpdate), w2.at({0, 0, 0}));
    this->expectNear(T(20.0 - expectedUpdate), w2.at({0, 0, 1}));
}
