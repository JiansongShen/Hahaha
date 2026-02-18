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
//
// Contributors:
// Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)

#include "ml/optimizer/AdadeltaOptimizer.h"

#include <gtest/gtest.h>
#include <cmath>

#include "public/Tensor.h"

using namespace hahaha;
using namespace hahaha::ml;
using namespace hahaha::common;

// Define floating point types for Adadelta
using FloatingPointTypes = ::testing::Types<f32, f64>;

template <typename T> class AdadeltaOptimizerTest : public ::testing::Test {
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

TYPED_TEST_SUITE(AdadeltaOptimizerTest, FloatingPointTypes);

// ============================================================================
// Constructor Tests
// ============================================================================

TYPED_TEST(AdadeltaOptimizerTest, Constructor_Default) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    AdadeltaOptimizer<T> opt({w});

    EXPECT_EQ(opt.getParameters().size(), 1);
    EXPECT_NEAR(static_cast<double>(opt.getDecayRate()), 0.9, 1e-6);
    EXPECT_NEAR(static_cast<double>(opt.getEpsilon()), 1e-6, 1e-9);
}

TYPED_TEST(AdadeltaOptimizerTest, Constructor_Custom) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    T decay = T(0.95);
    T eps = T(1e-5);
    AdadeltaOptimizer<T> opt({w}, decay, eps);

    EXPECT_EQ(opt.getParameters().size(), 1);
    EXPECT_NEAR(static_cast<double>(opt.getDecayRate()), static_cast<double>(decay), 1e-6);
    EXPECT_NEAR(static_cast<double>(opt.getEpsilon()), static_cast<double>(eps), 1e-9);
}

// ============================================================================
// Step Tests
// ============================================================================

TYPED_TEST(AdadeltaOptimizerTest, Step_SingleUpdate) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    w.setRequiresGrad(true);
    
    T decay = T(0.9);
    T eps = T(1e-6);
    AdadeltaOptimizer<T> opt({w}, decay, eps);
    
    // Set gradient to 0.5
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.5)));
    
    // Manual calculation:
    // g = 0.5
    // E[g^2] = 0.9 * 0 + 0.1 * 0.5^2 = 0.025
    // RMS[g] = sqrt(0.025 + 1e-6) = sqrt(0.025001)
    // RMS[dx]_prev = sqrt(0 + 1e-6) = 0.001
    // delta_x = - (RMS[dx]_prev / RMS[g]) * g
    // delta_x = - (0.001 / sqrt(0.025001)) * 0.5
    
    double g = 0.5;
    double eg2 = 0.1 * g * g;
    double rms_g = std::sqrt(eg2 + static_cast<double>(eps));
    double rms_dx_prev = std::sqrt(0.0 + static_cast<double>(eps));
    double delta_x = - (rms_dx_prev / rms_g) * g;
    double expected_w = 1.0 + delta_x;

    opt.step();
    
    this->expectNear(T(expected_w), w.at({}), T(1e-5));
}

TYPED_TEST(AdadeltaOptimizerTest, Step_MultipleUpdates) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    w.setRequiresGrad(true);
    
    T decay = T(0.9);
    T eps = T(1e-6);
    AdadeltaOptimizer<T> opt({w}, decay, eps);
    
    // Step 1
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.5)));
    
    double g1 = 0.5;
    double eg2_1 = 0.1 * g1 * g1;
    double rms_g_1 = std::sqrt(eg2_1 + static_cast<double>(eps));
    double rms_dx_prev_1 = std::sqrt(0.0 + static_cast<double>(eps));
    double delta_x_1 = - (rms_dx_prev_1 / rms_g_1) * g1;
    double w_1 = 1.0 + delta_x_1;
    double edx2_1 = 0.1 * delta_x_1 * delta_x_1;

    opt.step();
    this->expectNear(T(w_1), w.at({}), T(1e-5));

    // Step 2
    w.clearGrad();
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.2)));
    
    double g2 = 0.2;
    double eg2_2 = 0.9 * eg2_1 + 0.1 * g2 * g2;
    double rms_g_2 = std::sqrt(eg2_2 + static_cast<double>(eps));
    double rms_dx_prev_2 = std::sqrt(edx2_1 + static_cast<double>(eps));
    double delta_x_2 = - (rms_dx_prev_2 / rms_g_2) * g2;
    double w_2 = w_1 + delta_x_2;

    opt.step();
    this->expectNear(T(w_2), w.at({}), T(1e-5));
}

// ============================================================================
// Branch Coverage Tests
// ============================================================================

TYPED_TEST(AdadeltaOptimizerTest, Branch_RequiresGradFalse) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    w.setRequiresGrad(false); // Branch: !param.getRequiresGrad() -> continue
    
    AdadeltaOptimizer<T> opt({w});
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.5)));
    
    opt.step();
    
    // Should not change
    EXPECT_EQ(w.at({}), T(1.0));
}

TYPED_TEST(AdadeltaOptimizerTest, Branch_GradEmpty) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    w.setRequiresGrad(true);
    
    AdadeltaOptimizer<T> opt({w});
    // Do NOT accumulate grad -> grad is empty
    // Branch: grad.isEmpty() -> continue
    
    opt.step();
    
    // Should not change
    EXPECT_EQ(w.at({}), T(1.0));
}

TYPED_TEST(AdadeltaOptimizerTest, MultipleParameters) {
    using T = TypeParam;
    Tensor<T> w1(T(1.0));
    Tensor<T> w2(T(2.0));
    w1.setRequiresGrad(true);
    w2.setRequiresGrad(false);
    
    AdadeltaOptimizer<T> opt({w1, w2});
    
    w1.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.1)));
    w2.getComputeNode()->accumulateGrad(this->createGrad({}, T(0.1)));
    
    opt.step();
    
    // w1 should change
    EXPECT_NE(w1.at({}), T(1.0));
    // w2 should not change
    EXPECT_EQ(w2.at({}), T(2.0));
}
