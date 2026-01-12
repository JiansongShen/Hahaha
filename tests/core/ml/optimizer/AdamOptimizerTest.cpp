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

#include "ml/optimizer/AdamOptimizer.h"

#include <gtest/gtest.h>

#include "public/Tensor.h"

using namespace hahaha;
using namespace hahaha::ml;

// Define floating point types for Adam
using FloatingPointTypes = ::testing::Types<f32, f64>;

template <typename T> class AdamOptimizerTest : public ::testing::Test {
  protected:
    // Helper to compare values with epsilon
    void expectNear(T expected, T actual, T tolerance = 1e-4) {
        EXPECT_NEAR(static_cast<double>(expected), static_cast<double>(actual),
                    static_cast<double>(tolerance));
    }

    // Helper to create a gradient wrapper
    std::shared_ptr<math::TensorWrapper<T>>
    createGrad(const std::vector<size_t>& shape, T value) {
        return std::make_shared<math::TensorWrapper<T>>(math::TensorShape(shape),
                                                        value);
    }
};

TYPED_TEST_SUITE(AdamOptimizerTest, FloatingPointTypes);

// ============================================================================
// Standard Dimension Updates
// ============================================================================

TYPED_TEST(AdamOptimizerTest, Update_0D_Scalar) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    this->expectNear(T(9.9), w.at({}));
}

TYPED_TEST(AdamOptimizerTest, Update_1D_Vector) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0), T(2.0), T(3.0)});
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({3}, T(0.5)));
    opt.step();
    this->expectNear(T(0.9), w.at({0}));
    this->expectNear(T(1.9), w.at({1}));
    this->expectNear(T(2.9), w.at({2}));
}

TYPED_TEST(AdamOptimizerTest, Update_2D_Matrix) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{T(1.0), T(2.0)}, {T(3.0), T(4.0)}});
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.01));
    w.getComputeNode()->accumulateGrad(this->createGrad({2, 2}, T(0.2)));
    opt.step();
    this->expectNear(T(0.99), w.at({0, 0}));
    this->expectNear(T(3.99), w.at({1, 1}));
}

TYPED_TEST(AdamOptimizerTest, Update_3D_Tensor) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1), T(1)}}, {{T(1), T(1)}}});
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({2, 1, 2}, T(1.0)));
    opt.step();
    this->expectNear(T(0.9), w.at({0, 0, 0}));
    this->expectNear(T(0.9), w.at({1, 0, 1}));
}

// ============================================================================
// Error Handling: RequiresGrad is False (No Update)
// ============================================================================

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_0D) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(false);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({}), T(10.0));
}

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_1D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0), T(2.0)});
    w.setRequiresGrad(false);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({2}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({0}), T(1.0));
    EXPECT_EQ(w.at({1}), T(2.0));
}

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_2D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{T(1.0)}, {T(2.0)}});
    w.setRequiresGrad(false);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({2, 1}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({0, 0}), T(1.0));
    EXPECT_EQ(w.at({1, 0}), T(2.0));
}

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_3D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1.0)}}});
    w.setRequiresGrad(false);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({1, 1, 1}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({0, 0, 0}), T(1.0));
}

// ============================================================================
// Error Handling: Null Gradient (Should not crash)
// ============================================================================

TYPED_TEST(AdamOptimizerTest, NullGrad_0D) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({}), T(10.0));
}

TYPED_TEST(AdamOptimizerTest, NullGrad_1D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0)});
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({0}), T(1.0));
}

TYPED_TEST(AdamOptimizerTest, NullGrad_2D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{T(1.0)}});
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({0, 0}), T(1.0));
}

TYPED_TEST(AdamOptimizerTest, NullGrad_3D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1.0)}}});
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({0, 0, 0}), T(1.0));
}

// ============================================================================
// Other Functionality
// ============================================================================

TYPED_TEST(AdamOptimizerTest, ZeroGrad_Functionality) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.zeroGrad();
    EXPECT_EQ(w.grad()->at({}), T(0.0));
}

TYPED_TEST(AdamOptimizerTest, LearningRateChange) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    AdamOptimizer<T> opt({w}, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 1 -> w ≈ 9.9
    
    opt.setLearningRate(T(0.5));
    w.clearGrad();
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 2 -> mHat=1, vHat=1 -> w = 9.9 - 0.5 = 9.4
    this->expectNear(T(9.4), w.at({}));
}
