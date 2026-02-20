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

TYPED_TEST_SUITE(AdamOptimizerTest, FloatingPointTypes);

// ============================================================================
// Constructor Tests
// ============================================================================

TYPED_TEST(AdamOptimizerTest, Constructor_Standard) {
    using T = TypeParam;
    Tensor<T> w1(T(1.0));
    Tensor<T> w2(T(2.0));
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode(), w2.getComputeNode()};
    T lr = T(0.01);

    AdamOptimizer<T> opt(params, lr);

    EXPECT_EQ(opt.getParameters().size(), 2);
    EXPECT_EQ(opt.getLearningRate(), lr);
}

TYPED_TEST(AdamOptimizerTest, Constructor_Copy) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt1(params, T(0.1));

    // Copy using the explicit Optimizer constructor
    AdamOptimizer<T> opt2(static_cast<const Optimizer<T>&>(opt1));

    EXPECT_EQ(opt2.getParameters().size(), 1);
    EXPECT_EQ(opt2.getLearningRate(), T(0.1));
}

TYPED_TEST(AdamOptimizerTest, Constructor_Move) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt1(params, T(0.1));

    // Move using the explicit Optimizer constructor
    AdamOptimizer<T> opt2(std::move(opt1));

    EXPECT_EQ(opt2.getParameters().size(), 1);
    EXPECT_EQ(opt2.getLearningRate(), T(0.1));
}

// ============================================================================
// Standard Dimension Updates
// ============================================================================

TYPED_TEST(AdamOptimizerTest, Update_0D_Scalar) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    this->expectNear(T(9.9), w.at({}));
}

TYPED_TEST(AdamOptimizerTest, Update_1D_Vector) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0), T(2.0), T(3.0)});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
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
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.01));
    w.getComputeNode()->accumulateGrad(this->createGrad({2, 2}, T(0.2)));
    opt.step();
    this->expectNear(T(0.99), w.at({0, 0}));
    this->expectNear(T(3.99), w.at({1, 1}));
}

TYPED_TEST(AdamOptimizerTest, Update_3D_Tensor) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1), T(1)}}, {{T(1), T(1)}}});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
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
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({}), T(10.0));
}

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_1D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0), T(2.0)});
    w.setRequiresGrad(false);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({2}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({0}), T(1.0));
    EXPECT_EQ(w.at({1}), T(2.0));
}

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_2D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{T(1.0)}, {T(2.0)}});
    w.setRequiresGrad(false);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({2, 1}, T(1.0)));
    opt.step();
    EXPECT_EQ(w.at({0, 0}), T(1.0));
    EXPECT_EQ(w.at({1, 0}), T(2.0));
}

TYPED_TEST(AdamOptimizerTest, RequiresGradFalse_3D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1.0)}}});
    w.setRequiresGrad(false);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
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
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({}), T(10.0));
}

TYPED_TEST(AdamOptimizerTest, NullGrad_1D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{T(1.0)});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({0}), T(1.0));
}

TYPED_TEST(AdamOptimizerTest, NullGrad_2D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{T(1.0)}});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({0, 0}), T(1.0));
}

TYPED_TEST(AdamOptimizerTest, NullGrad_3D) {
    using T = TypeParam;
    Tensor<T> w(math::NestedData<T>{{{T(1.0)}}});
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(w.at({0, 0, 0}), T(1.0));
}

// ============================================================================
// Other Functionality
// ============================================================================

TYPED_TEST(AdamOptimizerTest, LearningRateChange) {
    using T = TypeParam;
    Tensor<T> w(T(10.0));
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 1 -> w ≈ 9.9

    opt.setLearningRate(T(0.5));
    w.clearGrad();
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 2 -> mHat=1, vHat=1 -> w = 9.9 - 0.5 = 9.4
    this->expectNear(T(9.4), w.at({}));
}

TYPED_TEST(AdamOptimizerTest, ZeroGrad_Functionality) {
    using T = TypeParam;
    Tensor<T> w(T(1.0));
    w.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.zeroGrad();
    EXPECT_EQ(w.grad().at({}), T(0.0));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_0D_BeforeStep) {
    using T = TypeParam;
    Tensor<T> w1(T(10.0));
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    Tensor<T> w2(T(20.0));
    opt.addParameter(w2.getComputeNode());
    EXPECT_EQ(opt.getParameters().size(), 2);
    auto paramNode = opt.getParameters()[1];
    EXPECT_EQ(paramNode->getData()->at({}), T(20.0));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_1D_BeforeStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{T(1.0)});
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    Tensor<T> w2(math::NestedData<T>{T(2.0), T(3.0)});
    opt.addParameter(w2.getComputeNode());
    EXPECT_EQ(opt.getParameters().size(), 2);
    auto paramNode = opt.getParameters()[1];
    EXPECT_EQ(paramNode->getData()->at({0}), T(2.0));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_2D_BeforeStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{{T(1.0)}});
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    Tensor<T> w2(math::NestedData<T>{{T(2.0), T(3.0)}, {T(4.0), T(5.0)}});
    opt.addParameter(w2.getComputeNode());
    EXPECT_EQ(opt.getParameters().size(), 2);
    auto paramNode = opt.getParameters()[1];
    EXPECT_EQ(paramNode->getData()->at({1, 1}), T(5.0));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_3D_BeforeStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{{{T(1.0)}}});
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    Tensor<T> w2(math::NestedData<T>{{{T(1.0), T(2.0)}}, {{T(3.0), T(4.0)}}});
    opt.addParameter(w2.getComputeNode());
    EXPECT_EQ(opt.getParameters().size(), 2);
    auto paramNode = opt.getParameters()[1];
    EXPECT_EQ(paramNode->getData()->at({1, 0, 1}), T(4.0));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_0D_AfterStep) {
    using T = TypeParam;
    Tensor<T> w1(T(10.0));
    w1.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w1.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 变为 1

    Tensor<T> w2(T(20.0));
    w2.setRequiresGrad(true);
    opt.addParameter(w2.getComputeNode());

    w2.getComputeNode()->accumulateGrad(this->createGrad({}, T(1.0)));
    opt.step(); // turn 变为 2

    // t=2 时的更新量约为 0.0744137 (计算见上)
    T expectedT2 = T(20.0 - 0.0744137);
    this->expectNear(expectedT2, w2.at({}));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_1D_AfterStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{T(1.0)});
    w1.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w1.getComputeNode()->accumulateGrad(this->createGrad({1}, T(1.0)));
    opt.step();

    Tensor<T> w2(math::NestedData<T>{T(10.0), T(20.0)});
    w2.setRequiresGrad(true);
    opt.addParameter(w2.getComputeNode());

    w2.getComputeNode()->accumulateGrad(this->createGrad({2}, T(1.0)));
    opt.step();

    T val1 = T(10.0 - 0.0744137);
    T val2 = T(20.0 - 0.0744137);
    this->expectNear(val1, w2.at({0}));
    this->expectNear(val2, w2.at({1}));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_2D_AfterStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{{T(1.0)}});
    w1.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w1.getComputeNode()->accumulateGrad(this->createGrad({1, 1}, T(1.0)));
    opt.step();

    Tensor<T> w2(math::NestedData<T>{{T(10.0), T(20.0)}, {T(30.0), T(40.0)}});
    w2.setRequiresGrad(true);
    opt.addParameter(w2.getComputeNode());

    w2.getComputeNode()->accumulateGrad(this->createGrad({2, 2}, T(1.0)));
    opt.step();

    T expectedUpdate = T(0.0744137);
    this->expectNear(T(10.0 - expectedUpdate), w2.at({0, 0}));
    this->expectNear(T(40.0 - expectedUpdate), w2.at({1, 1}));
}

TYPED_TEST(AdamOptimizerTest, AddParameter_3D_AfterStep) {
    using T = TypeParam;
    Tensor<T> w1(math::NestedData<T>{{{T(1.0)}}});
    w1.setRequiresGrad(true);
    std::vector<std::shared_ptr<compute::ComputeNode<T>>> params = {
        w1.getComputeNode()};
    AdamOptimizer<T> opt(params, T(0.1));
    w1.getComputeNode()->accumulateGrad(this->createGrad({1, 1, 1}, T(1.0)));
    opt.step();

    Tensor<T> w2(math::NestedData<T>{{{T(10.0), T(20.0)}}});
    w2.setRequiresGrad(true);
    opt.addParameter(w2.getComputeNode());

    w2.getComputeNode()->accumulateGrad(this->createGrad({1, 1, 2}, T(1.0)));
    opt.step();

    T expectedUpdate = T(0.0744137);
    this->expectNear(T(10.0 - expectedUpdate), w2.at({0, 0, 0}));
    this->expectNear(T(20.0 - expectedUpdate), w2.at({0, 0, 1}));
}