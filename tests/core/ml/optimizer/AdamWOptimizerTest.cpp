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
    void expectNear(T expected, T actual, T tolerance = 1e-4) {
        EXPECT_NEAR(static_cast<double>(expected), static_cast<double>(actual),
                    static_cast<double>(tolerance));
    }

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
    Tensor<T> w(T(1.0));
    T lr = T(0.01);
    T wd = T(0.1);

    AdamWOptimizer<T> opt({w}, lr, 0.9, 0.999, 1e-8, wd);

    EXPECT_EQ(opt.getParameters().size(), 1);
    EXPECT_EQ(opt.getLearningRate(), lr);
}

// ============================================================================
// Standard Dimension Updates (with Weight Decay)
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, Update_0D_Scalar) {
    using T = TypeParam;
    T initialValue = 10.0;
    Tensor<T> w(initialValue);
    w.setRequiresGrad(true);
    
    T lr = 0.1;
    T wd = 0.01;
    AdamWOptimizer<T> opt({w}, lr, 0.9, 0.999, 1e-8, wd);

    // Step 1: Grad = 1.0
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();

    // Theoretical calculation:
    // Adam Update part: ~0.1 (same as Adam step 1)
    // Weight Decay part: lr * wd * theta = 0.1 * 0.01 * 10.0 = 0.01
    // Total update = 0.1 + 0.01 = 0.11
    // New theta = 10.0 - 0.11 = 9.89
    this->expectNear(T(9.89), w.at({}));
}

TYPED_TEST(AdamWOptimizerTest, Update_1D_Vector) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0), T(2.0)});
    w.setRequiresGrad(true);
    AdamWOptimizer<T> opt({w}, T(0.1), 0.9, 0.999, 1e-8, T(0.01));

    w.getComputeNode()->accumulateGrad(this->createGrad({2}, T(1.0)));
    opt.step();

    // val1: 1.0 - (0.1 + 0.1*0.01*1.0) = 1.0 - 0.101 = 0.899
    // val2: 2.0 - (0.1 + 0.1*0.01*2.0) = 2.0 - 0.102 = 1.898
    this->expectNear(T(0.899), w.at({0}));
    this->expectNear(T(1.898), w.at({1}));
}

TYPED_TEST(AdamWOptimizerTest, Update_2D_Matrix) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{T(1.0)}, {T(2.0)}});
    w.setRequiresGrad(true);
    AdamWOptimizer<T> opt({w}, T(0.1), 0.9, 0.999, 1e-8, T(0.01));

    w.getComputeNode()->accumulateGrad(this->createGrad({2, 1}, T(1.0)));
    opt.step();

    this->expectNear(T(0.899), w.at({0, 0}));
    this->expectNear(T(1.898), w.at({1, 0}));
}

TYPED_TEST(AdamWOptimizerTest, Update_3D_Tensor) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1.0)}}});
    w.setRequiresGrad(true);
    AdamWOptimizer<T> opt({w}, T(0.1), 0.9, 0.999, 1e-8, T(0.01));

    w.getComputeNode()->accumulateGrad(this->createGrad({1, 1, 1}, T(1.0)));
    opt.step();

    this->expectNear(T(0.899), w.at({0, 0, 0}));
}

// ============================================================================
// Weight Decay Verification (Decoupled)
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, WeightDecay_EvenWithZeroGrad) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    
    T lr = 0.1;
    T wd = 0.5; // Large weight decay for clear effect
    AdamWOptimizer<T> opt({w}, lr, 0.9, 0.999, 1e-8, wd);

    // Grad = 0.0, Adam component should be 0
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.0)));
    opt.step();

    // theta = 10.0 - (lr * wd * 10.0) = 10.0 - (0.1 * 0.5 * 10.0) = 10.0 - 0.5 = 9.5
    EXPECT_EQ(w.at({}), T(9.5));
}

// ============================================================================
// Error Handling
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, RequiresGradFalse_NoUpdate) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(false);
    AdamWOptimizer<T> opt({w}, T(0.1), 0.9, 0.999, 1e-8, T(0.01));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({}), T(10.0));
}

TYPED_TEST(AdamWOptimizerTest, NullGrad_StillPerformsWeightDecay) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    AdamWOptimizer<T> opt({w}, T(0.1), 0.9, 0.999, 1e-8, T(0.01));
    
    // Step without backward() -> null grad
    EXPECT_NO_THROW(opt.step());
    
    // Adam component is skipped, but Weight Decay should still apply
    // theta = 10.0 - (0.1 * 0.01 * 10.0) = 9.99
    this->expectNear(T(9.99), w.at({}));
}

// ============================================================================
// Dynamic Parameters
// ============================================================================

TYPED_TEST(AdamWOptimizerTest, AddParameter_PostPreparation) {
    using T = TypeParam;
    Tensor<T> w1(T(1.0));
    w1.setRequiresGrad(true);
    AdamWOptimizer<T> opt({w1}, T(0.1), 0.9, 0.999, 1e-8, T(0.01));
    
    opt.step(); // Trigger preparation

    Tensor<T> w2(T(20.0));
    w2.setRequiresGrad(true);
    opt.addParameter(w2);

    w2.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 2 for opt, turn 1 for w2's states

    // Theoretical calculation for w2 at turn 2:
    // Adam update (t=2, g=1.0) ~ 0.0744137
    // Weight decay: 0.1 * 0.01 * 20.0 = 0.02
    // New theta = 20.0 - 0.0744137 - 0.02 = 19.9055863
    this->expectNear(T(19.9055863), w2.at({}));
}

