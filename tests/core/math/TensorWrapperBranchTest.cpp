// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#include <gtest/gtest.h>

#include "math/TensorWrapper.h"

using namespace hahaha::math;
using namespace hahaha::backend;

class TensorWrapperBranchTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(TensorWrapperBranchTest, MoveAssignment_SelfAssignment) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    TensorWrapper<float>* ptr = &a;
    a = std::move(
        *ptr); // Handle self-assignment through pointer to bypass -Wself-move
    EXPECT_EQ(a.getTotalSize(), 4u);
}

TEST_F(TensorWrapperBranchTest, GetTotalSize_NullData) {
    TensorWrapper<float> a;
    EXPECT_EQ(a.getTotalSize(), 0u);
}

TEST_F(TensorWrapperBranchTest, ToDevice_SameDevice) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    Device d = a.getDevice();
    a.to(d); // Should return immediately
    EXPECT_EQ(a.getDevice(), d);
}

TEST_F(TensorWrapperBranchTest, ToDevice_UnsupportedTransfers) {
    TensorWrapper<float> a({1}, 1.0f);

    // CPU to GPU (throws)
    EXPECT_THROW(a.to(Device(DeviceType::GPU, 0)), std::runtime_error);

    // Mocking a GPU tensor to test GPU to CPU
    TensorWrapper<float> g({1}, 1.0f);
    // There is no easy way to create a GPU tensor without it throwing in
    // constructor because TensorData throws on GPU allocation.
}

TEST_F(TensorWrapperBranchTest, At_DimensionMismatch) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    EXPECT_THROW(a.at({1}), std::out_of_range);
    EXPECT_THROW(a.at({1, 1, 1}), std::out_of_range);

    const TensorWrapper<float>& ca = a;
    EXPECT_THROW(ca.at({1}), std::out_of_range);
}

TEST_F(TensorWrapperBranchTest, At_OutOfBounds) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    EXPECT_THROW(a.at({2, 0}), std::out_of_range);
    EXPECT_THROW(a.at({0, 2}), std::out_of_range);

    const TensorWrapper<float>& ca = a;
    EXPECT_THROW(ca.at({2, 0}), std::out_of_range);
}

TEST_F(TensorWrapperBranchTest, Reshape_SizeMismatch) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    EXPECT_THROW(a.reshape({5}), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, BinaryOps_ScalarCombinations_All) {
    TensorWrapper<float> s1(10.0f);
    TensorWrapper<float> s2(2.0f);
    TensorWrapper<float> t({2, 2}, 1.0f);

    // Test add combinations
    EXPECT_FLOAT_EQ(s1.add(t).at({0, 0}), 11.0f); // scalar + tensor
    EXPECT_FLOAT_EQ(t.add(s1).at({0, 0}), 11.0f); // tensor + scalar
    EXPECT_FLOAT_EQ(s1.add(s2).at({}), 12.0f);    // scalar + scalar

    // Test subtract combinations
    EXPECT_FLOAT_EQ(s1.subtract(t).at({0, 0}), 9.0f);  // scalar - tensor
    EXPECT_FLOAT_EQ(t.subtract(s1).at({0, 0}), -9.0f); // tensor - scalar
    EXPECT_FLOAT_EQ(s1.subtract(s2).at({}), 8.0f);     // scalar - scalar

    // Test multiply combinations
    EXPECT_FLOAT_EQ(s1.multiply(t).at({0, 0}), 10.0f); // scalar * tensor
    EXPECT_FLOAT_EQ(t.multiply(s1).at({0, 0}), 10.0f); // tensor * scalar
    EXPECT_FLOAT_EQ(s1.multiply(s2).at({}), 20.0f);    // scalar * scalar

    // Test divide combinations
    EXPECT_FLOAT_EQ(s1.divide(t).at({0, 0}), 10.0f); // scalar / tensor
    EXPECT_FLOAT_EQ(t.divide(s1).at({0, 0}), 0.1f);  // tensor / scalar
    EXPECT_FLOAT_EQ(s1.divide(s2).at({}), 5.0f);     // scalar / scalar

    EXPECT_THROW(s1.divide(TensorWrapper<float>(0.0f)), std::runtime_error);
    EXPECT_THROW(t.divide(TensorWrapper<float>(0.0f)), std::runtime_error);
}

TEST_F(TensorWrapperBranchTest, InPlaceOps_Comprehensive) {
    TensorWrapper<float> a({2, 2}, 10.0f);
    TensorWrapper<float> b({2, 2}, 2.0f);
    TensorWrapper<float> s(2.0f);

    a += b;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 12.0f);
    a -= b;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 10.0f);
    a *= b;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 20.0f);
    a /= b;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 10.0f);

    a += s;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 12.0f);
    a -= s;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 10.0f);
    a *= s;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 20.0f);
    a /= s;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 10.0f);

    // Test direct scalar operators
    a += 2.0f;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 12.0f);
    a -= 2.0f;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 10.0f);
    a *= 2.0f;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 20.0f);
    a /= 2.0f;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 10.0f);
}

TEST_F(TensorWrapperBranchTest, ShapeMismatch_BinaryOps) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    TensorWrapper<float> b({2, 3}, 1.0f);
    EXPECT_THROW(a.add(b), std::invalid_argument);
    EXPECT_THROW(a.subtract(b), std::invalid_argument);
    EXPECT_THROW(a.multiply(b), std::invalid_argument);
    EXPECT_THROW(a.divide(b), std::invalid_argument);

    EXPECT_THROW(a += b, std::invalid_argument);
    EXPECT_THROW(a -= b, std::invalid_argument);
    EXPECT_THROW(a *= b, std::invalid_argument);
    EXPECT_THROW(a /= b, std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, MatMul_Non2D) {
    TensorWrapper<float> a({2, 2, 2}, 1.0f);
    TensorWrapper<float> b({2, 2}, 1.0f);
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, MatMul_DimMismatch) {
    TensorWrapper<float> a({2, 3}, 1.0f);
    TensorWrapper<float> b({2, 2}, 1.0f);
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, Transpose_Non2D) {
    TensorWrapper<float> a({2}, 1.0f);
    EXPECT_THROW(a.transpose(), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, Sum_Comprehensive) {
    // 2x2x2 tensor
    TensorWrapper<float> a(
        NestedData<float>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});

    // 1. axes empty -> clone
    auto r1 = a.sum({}, false);
    EXPECT_EQ(r1.getShape(), (std::vector<size_t>{2, 2, 2}));
    EXPECT_FLOAT_EQ(r1.at({0, 0, 0}), 1.0f);

    // 2. all axes -> scalar
    auto r2 = a.sum({0, 1, 2}, false);
    EXPECT_EQ(r2.getDimensions(), 0u);
    EXPECT_FLOAT_EQ(r2.at({}), 36.0f);

    // 3. reduce axis 0 with keepDims
    auto r3 = a.sum({0}, true);
    EXPECT_EQ(r3.getShape(), (std::vector<size_t>{1, 2, 2}));
    EXPECT_FLOAT_EQ(r3.at({0, 0, 0}), 6.0f); // 1 + 5

    // 4. reduce axis 1
    auto r4 = a.sum({1}, false);
    EXPECT_EQ(r4.getShape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(r4.at({0, 0}), 4.0f); // 1 + 3

    // 5. error case: axis too big
    EXPECT_THROW(a.sum({3}), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, BroadcastTo_Invalid) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    EXPECT_THROW(a.broadcastTo(TensorShape({2, 3})), std::invalid_argument);

    // Rank smaller than current rank
    EXPECT_THROW(a.broadcastTo(TensorShape({2})), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, Axpy_ShapeMismatch) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    TensorWrapper<float> b({3, 3}, 1.0f);
    EXPECT_THROW(a.axpy(1.0f, b), std::invalid_argument);
}

TEST_F(TensorWrapperBranchTest, Negate_Operator) {
    TensorWrapper<float> a({2}, 5.0f);
    auto b = -a;
    EXPECT_FLOAT_EQ(b.at({0}), -5.0f);
}

TEST_F(TensorWrapperBranchTest, Clear_Operator) {
    TensorWrapper<float> a({2, 2}, 1.0f);
    a.clear();
    EXPECT_FLOAT_EQ(a.at({0, 0}), 0.0f);

    TensorWrapper<float> empty;
    empty.clear(); // Should handle null data
}

TEST_F(TensorWrapperBranchTest, Clone_Operator) {
    TensorWrapper<float> a({2}, 1.0f);
    auto b = a.clone();
    EXPECT_EQ(a.getShape(), b.getShape());
    EXPECT_FLOAT_EQ(b.at({0}), 1.0f);
}

TEST_F(TensorWrapperBranchTest, CheckSameDevice_BranchCoverage) {
    TensorWrapper<float> a({1}, 1.0f);
    TensorWrapper<float> b({1}, 2.0f);

    // Path 1: Same device
    EXPECT_NO_THROW(a.add(b));

    // Path 2: Different device
    TensorWrapper<float> c({1}, 1.0f, Device(DeviceType::SIMD, 0));
    EXPECT_THROW(a.add(c), std::invalid_argument);
}
