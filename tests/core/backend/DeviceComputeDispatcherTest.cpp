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

#include "backend/DeviceComputeDispatcher.h"

#include <gtest/gtest.h>

#include "backend/Device.h"
#include "common/Operator.h"
#include "math/TensorWrapper.h"

using hahaha::backend::Device;
using hahaha::backend::DeviceComputeDispatcher;
using hahaha::backend::DeviceType;
using hahaha::common::Operator;
using hahaha::math::NestedData;
using hahaha::math::TensorShape;
using hahaha::math::TensorWrapper;

TEST(DeviceComputeDispatcherTest, DeviceToString_CoversAllEnumValues) {
    EXPECT_EQ(Device(DeviceType::CPU, 0).toString(), "CPU:0");
    EXPECT_EQ(Device(DeviceType::GPU, 0).toString(), "GPU:0");
    EXPECT_EQ(Device(DeviceType::SIMD, 1).toString(), "SIMD:1");
}

TEST(DeviceComputeDispatcherTest, DeviceEqualityAndInequality) {
    EXPECT_TRUE(Device(DeviceType::CPU, 0) == Device(DeviceType::CPU, 0));
    EXPECT_FALSE(Device(DeviceType::CPU, 0) != Device(DeviceType::CPU, 0));

    EXPECT_FALSE(Device(DeviceType::CPU, 0) == Device(DeviceType::CPU, 1));
    EXPECT_TRUE(Device(DeviceType::CPU, 0) != Device(DeviceType::CPU, 1));

    EXPECT_FALSE(Device(DeviceType::CPU, 0) == Device(DeviceType::SIMD, 0));
    EXPECT_TRUE(Device(DeviceType::CPU, 0) != Device(DeviceType::SIMD, 0));
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_AllOps_WorkForDenseTensors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Add, a, b, res);
    EXPECT_FLOAT_EQ(res.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 44.0f);

    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Sub, b, a, res);
    EXPECT_FLOAT_EQ(res.at({0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 36.0f);

    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Mul, a, b, res);
    EXPECT_FLOAT_EQ(res.at({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 160.0f);

    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Div, b, a, res);
    EXPECT_FLOAT_EQ(res.at({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 10.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_ScalarRank0_PathWorks) {
    TensorWrapper<float> a(2.0f);
    TensorWrapper<float> b(3.0f);
    TensorWrapper<float> res(0.0f);

    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Add, a, b, res);
    EXPECT_FLOAT_EQ(res.at({}), 5.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_ShapeMismatch_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> badRes(
        TensorShape({4}), 0.0f, Device(DeviceType::CPU, 0));
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchBinary(
                     Operator::Add, a, b, badRes),
                 std::invalid_argument);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_DivisionByZero_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 0.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchBinary(
                     Operator::Div, b, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest,
     DispatchScalar_RhsScalar_AllOps_WorkAndErrors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchScalar(Operator::Add, a, 1.0f, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 5.0f);

    DeviceComputeDispatcher<float>::dispatchScalar(Operator::Sub, a, 1.0f, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 3.0f);

    DeviceComputeDispatcher<float>::dispatchScalar(Operator::Mul, a, 2.0f, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 8.0f);

    DeviceComputeDispatcher<float>::dispatchScalar(Operator::Div, a, 2.0f, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 2.0f);

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Div, a, 0.0f, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest,
     DispatchScalar_LhsScalar_AllOps_WorkAndErrors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {0.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchScalar(Operator::Add, 1.0f, a, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 5.0f);

    DeviceComputeDispatcher<float>::dispatchScalar(
        Operator::Sub, 10.0f, a, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 6.0f);

    DeviceComputeDispatcher<float>::dispatchScalar(Operator::Mul, 2.0f, a, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 8.0f);

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Div, 1.0f, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_ShapeMismatch_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> badRes(
        TensorShape({4}), 0.0f, Device(DeviceType::CPU, 0));
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Add, a, 1.0f, badRes),
                 std::invalid_argument);
}

TEST(DeviceComputeDispatcherTest, DispatchMatMul_WorksFor2D) {
    TensorWrapper<float> a(
        NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // 2x2
    TensorWrapper<float> b(
        NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}}); // 2x2
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchMatMul(a, b, res);
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    EXPECT_FLOAT_EQ(res.at({0, 0}), 19.0f);
    EXPECT_FLOAT_EQ(res.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(res.at({1, 0}), 43.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 50.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchAxpy_UpdatesInPlace) {
    TensorWrapper<float> x(NestedData<float>{1.0f, 2.0f, 3.0f});
    TensorWrapper<float> res(NestedData<float>{10.0f, 20.0f, 30.0f});

    // res = res + alpha * x, alpha = -2
    DeviceComputeDispatcher<float>::dispatchAxpy(-2.0f, x, res);
    EXPECT_FLOAT_EQ(res.at({0}), 8.0f);
    EXPECT_FLOAT_EQ(res.at({1}), 16.0f);
    EXPECT_FLOAT_EQ(res.at({2}), 24.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_UnsupportedOp_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchBinary(
                     Operator::MatMul, a, b, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_UnsupportedDevice_Throws) {
    TensorWrapper<float> a(TensorShape({2, 2}), 1.0f, Device(DeviceType::SIMD));
    TensorWrapper<float> b(TensorShape({2, 2}), 2.0f, Device(DeviceType::SIMD));
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::SIMD));

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchBinary(
                     Operator::Add, a, b, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_LhsScalar_Div_Succeeds) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchScalar(
        Operator::Div, 12.0f, a, res);
    EXPECT_FLOAT_EQ(res.at({0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(res.at({0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(res.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 3.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_UnsupportedOp_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::MatMul, a, 1.0f, res),
                 std::runtime_error);
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::MatMul, 1.0f, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchMatMul_UnsupportedDevice_Throws) {
    TensorWrapper<float> a(TensorShape({2, 2}), 1.0f, Device(DeviceType::SIMD));
    TensorWrapper<float> b(TensorShape({2, 2}), 1.0f, Device(DeviceType::SIMD));
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::SIMD));

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchMatMul(a, b, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_UnsupportedDevice_Throws) {
    TensorWrapper<float> a(TensorShape({2, 2}), 1.0f, Device(DeviceType::SIMD));
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::SIMD));

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Add, a, 1.0f, res),
                 std::runtime_error);
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Add, 1.0f, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_Broadcasting_Works) {
    // (2, 2) + (2, 1) -> (2, 2)
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f}, {20.0f}}); // shape (2, 1)
    TensorWrapper<float> b_broadcasted =
        b.broadcastTo(TensorShape({2, 2})); // strides will have 0

    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchBinary(
        Operator::Add, a, b_broadcasted, res);

    // res[0,0] = 1 + 10 = 11
    // res[0,1] = 2 + 10 = 12
    // res[1,0] = 3 + 20 = 23
    // res[1,1] = 4 + 20 = 24
    EXPECT_FLOAT_EQ(res.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(res.at({0, 1}), 12.0f);
    EXPECT_FLOAT_EQ(res.at({1, 0}), 23.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 24.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_Broadcasting_Lhs_Works) {
    // (1, 2) + (2, 2) -> (2, 2)
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}}); // shape (1, 2)
    TensorWrapper<float> a_broadcasted = a.broadcastTo(TensorShape({2, 2}));
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});

    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, Device(DeviceType::CPU, 0));

    DeviceComputeDispatcher<float>::dispatchBinary(
        Operator::Add, a_broadcasted, b, res);

    // res[0,0] = 1 + 10 = 11
    // res[0,1] = 2 + 20 = 22
    // res[1,0] = 1 + 30 = 31
    // res[1,1] = 2 + 40 = 42
    EXPECT_FLOAT_EQ(res.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(res.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(res.at({1, 0}), 31.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 42.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_Broadcasting_Deep_Works) {
    // (1, 2, 1) + (2, 1, 2) -> (2, 2, 2)
    TensorWrapper<float> a(NestedData<float>{{{1.0f}, {2.0f}}}); // 1x2x1
    TensorWrapper<float> b(
        NestedData<float>{{{10.0f, 20.0f}}, {{30.0f, 40.0f}}}); // 2x1x2

    auto ab = a.broadcastTo(TensorShape({2, 2, 2}));
    auto bb = b.broadcastTo(TensorShape({2, 2, 2}));

    TensorWrapper<float> res(TensorShape({2, 2, 2}), 0.0f);
    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Add, ab, bb, res);

    // a[0,0,0]=1, a[0,1,0]=2
    // b[0,0,0]=10, b[0,0,1]=20, b[1,0,0]=30, b[1,0,1]=40
    // res[0,0,0] = 1 + 10 = 11
    // res[0,0,1] = 1 + 20 = 21
    // res[0,1,0] = 2 + 10 = 12
    // res[0,1,1] = 2 + 20 = 22
    // res[1,0,0] = 1 + 30 = 31
    // res[1,0,1] = 1 + 40 = 41
    EXPECT_FLOAT_EQ(res.at({0, 0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1, 1}), 42.0f);
}

TEST(DeviceComputeDispatcherTest, ForEachElement_CoordinateCarry_Coverage) {
    // This is to specifically trigger the 'continue' inside the dimension loop
    // of forEachElement where it carries over to the next dimension.
    // Shape (2, 2)
    TensorWrapper<float> a(NestedData<float>{{1, 2}, {3, 4}});
    TensorWrapper<float> res(TensorShape({2, 2}), 0.0f);

    // We already have DispatchBinary tests, but let's make sure we hit the
    // inner loop multiple times
    DeviceComputeDispatcher<float>::dispatchBinary(Operator::Add, a, a, res);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 8.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_Rhs_DivZero_Throws) {
    TensorWrapper<float> a(NestedData<float>{1.0f, 2.0f});
    TensorWrapper<float> res(TensorShape({2}), 0.0f);
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Div, a, 0.0f, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_Lhs_DivZero_Throws) {
    TensorWrapper<float> a(NestedData<float>{1.0f, 0.0f});
    TensorWrapper<float> res(TensorShape({2}), 0.0f);
    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchScalar(
                     Operator::Div, 1.0f, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest,
     DispatchMatMul_InnerDimensionMismatch_NoCheckYetButRun) {
    // Current implementation doesn't check inner dimension, it just loops.
    // We should at least cover the code.
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}});   // 1x2
    TensorWrapper<float> b(NestedData<float>{{3.0f}, {4.0f}}); // 2x1
    TensorWrapper<float> res(TensorShape({1, 1}), 0.0f);
    DeviceComputeDispatcher<float>::dispatchMatMul(a, b, res);
    EXPECT_FLOAT_EQ(res.at({0, 0}), 11.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchAxpy_UnsupportedDevice_Throws) {
    TensorWrapper<float> x(TensorShape({3}), 1.0f, Device(DeviceType::SIMD));
    TensorWrapper<float> res(TensorShape({3}), 2.0f, Device(DeviceType::SIMD));

    EXPECT_THROW(DeviceComputeDispatcher<float>::dispatchAxpy(1.0f, x, res),
                 std::runtime_error);
}
