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
using hahaha::backend::DeviceType;
using hahaha::common::Operator;
using hahaha::math::NestedData;
using hahaha::math::TensorShape;
using hahaha::math::TensorWrapper;

// TEST(DeviceComputeDispatcherTest, DeviceToString_CoversAllEnumValues) {
//     EXPECT_EQ(Device(DeviceType::CPU, 0).toString(), "CPU:0");
//     EXPECT_EQ(Device(DeviceType::GPU, 0).toString(), "GPU:0");
//     EXPECT_EQ(Device(DeviceType::SIMD, 1).toString(), "SIMD:1");
// }
//

// TEST(DeviceComputeDispatcherTest, DeviceEqualityAndInequality) {
//     EXPECT_TRUE(Device(DeviceType::CPU, 0) ==
//     std::make_shared<hahaha::backend::CPUDevice>());
//     EXPECT_FALSE(Device(DeviceType::CPU, 0) !=
//     std::make_shared<hahaha::backend::CPUDevice>());
//
//     EXPECT_FALSE(Device(DeviceType::CPU, 0) == Device(DeviceType::CPU, 1));
//     EXPECT_TRUE(Device(DeviceType::CPU, 0) != Device(DeviceType::CPU, 1));
//
//     EXPECT_FALSE(Device(DeviceType::CPU, 0) == Device(DeviceType::SIMD, 0));
//     EXPECT_TRUE(Device(DeviceType::CPU, 0) != Device(DeviceType::SIMD, 0));
// }

TEST(DeviceComputeDispatcherTest, DispatchBinary_AllOps_WorkForDenseTensors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto res1 = hahaha::backend::dispatchAdd(DeviceType::CPU, a, b, res);
    EXPECT_TRUE(res1.has_value());
    EXPECT_FLOAT_EQ(res.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 44.0f);

    auto res2 = hahaha::backend::dispatchSub(DeviceType::CPU, b, a, res);
    EXPECT_TRUE(res2.has_value());
    EXPECT_FLOAT_EQ(res.at({0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 36.0f);

    auto res3 = hahaha::backend::dispatchMul(DeviceType::CPU, a, b, res);
    EXPECT_TRUE(res3.has_value());
    EXPECT_FLOAT_EQ(res.at({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 160.0f);

    auto res4 = hahaha::backend::dispatchDiv(DeviceType::CPU, b, a, res);
    EXPECT_TRUE(res4.has_value());
    EXPECT_FLOAT_EQ(res.at({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 10.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_ScalarRank0_PathWorks) {
    TensorWrapper<float> a(2.0f);
    TensorWrapper<float> b(3.0f);
    TensorWrapper<float> res(0.0f);

    auto result = hahaha::backend::dispatchAdd(DeviceType::CPU, a, b, res);
    EXPECT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(res.at({}), 5.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_ShapeMismatch_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> badRes(
        TensorShape({4}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());
    auto result = hahaha::backend::dispatchAdd(DeviceType::CPU, a, b, badRes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, hahaha::common::ErrorCode::InvalidArgument);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_DivisionByZero_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 0.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    // Note: cpuDiv still throws runtime_error currently, but we catch it or
    // expect it
    EXPECT_THROW(hahaha::backend::dispatchDiv(DeviceType::CPU, b, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_RhsScalar_AllOps_WorkAndErrors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    EXPECT_TRUE(
        hahaha::backend::dispatchAdd(DeviceType::CPU, a, 1.0f, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 5.0f);

    EXPECT_TRUE(
        hahaha::backend::dispatchSub(DeviceType::CPU, a, 1.0f, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 3.0f);

    EXPECT_TRUE(
        hahaha::backend::dispatchMul(DeviceType::CPU, a, 2.0f, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 8.0f);

    EXPECT_TRUE(
        hahaha::backend::dispatchDiv(DeviceType::CPU, a, 2.0f, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 2.0f);

    EXPECT_FALSE(
        hahaha::backend::dispatchDiv(DeviceType::CPU, a, 0.0f, res).has_value());
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_LhsScalar_AllOps_WorkAndErrors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {0.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    EXPECT_TRUE(
        hahaha::backend::dispatchAdd(DeviceType::CPU, 1.0f, a, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 5.0f);

    EXPECT_TRUE(
        hahaha::backend::dispatchSub(DeviceType::CPU, 10.0f, a, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 6.0f);

    EXPECT_TRUE(
        hahaha::backend::dispatchMul(DeviceType::CPU, 2.0f, a, res).has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 8.0f);

    EXPECT_THROW(hahaha::backend::dispatchDiv(DeviceType::CPU, 1.0f, a, res),
                 std::runtime_error);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_ShapeMismatch_Throws) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> badRes(
        TensorShape({4}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    EXPECT_FALSE(
        hahaha::backend::dispatchAdd(DeviceType::CPU, a, 1.0f, badRes).has_value());
}

TEST(DeviceComputeDispatcherTest, DispatchMatMul_WorksFor2D) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}); // 2x2
    TensorWrapper<float> b(NestedData<float>{{5.0f, 6.0f}, {7.0f, 8.0f}}); // 2x2
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto res_val = hahaha::backend::dispatchMatMul(DeviceType::CPU, a, b, res);
    EXPECT_TRUE(res_val.has_value());
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
    auto res_val = hahaha::backend::dispatchAxpy(DeviceType::CPU, -2.0f, x, res);
    EXPECT_TRUE(res_val.has_value());
    EXPECT_FLOAT_EQ(res.at({0}), 8.0f);
    EXPECT_FLOAT_EQ(res.at({1}), 16.0f);
    EXPECT_FLOAT_EQ(res.at({2}), 24.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_UnsupportedDevice_Throws) {
    TensorWrapper<float> a(
        TensorShape({2, 2}), 1.0f, std::make_shared<hahaha::backend::CPUDevice>());
    TensorWrapper<float> b(
        TensorShape({2, 2}), 2.0f, std::make_shared<hahaha::backend::CPUDevice>());
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto result = hahaha::backend::dispatchAdd(DeviceType::HIP, a, b, res);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, hahaha::common::ErrorCode::DeviceNotSupported);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_LhsScalar_Div_Succeeds) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    EXPECT_TRUE(
        hahaha::backend::dispatchDiv(DeviceType::CPU, 12.0f, a, res).has_value());
    EXPECT_FLOAT_EQ(res.at({0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(res.at({0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(res.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(res.at({1, 1}), 3.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchMatMul_UnsupportedDevice_Throws) {
    TensorWrapper<float> a(
        TensorShape({2, 2}), 1.0f, std::make_shared<hahaha::backend::CPUDevice>());
    TensorWrapper<float> b(
        TensorShape({2, 2}), 1.0f, std::make_shared<hahaha::backend::CPUDevice>());
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto res_val = hahaha::backend::dispatchMatMul(DeviceType::HIP, a, b, res);
    EXPECT_FALSE(res_val.has_value());
    EXPECT_EQ(res_val.error().code, hahaha::common::ErrorCode::DeviceNotSupported);
}

TEST(DeviceComputeDispatcherTest, DispatchScalar_UnsupportedDevice_Throws) {
    TensorWrapper<float> a(
        TensorShape({2, 2}), 1.0f, std::make_shared<hahaha::backend::CPUDevice>());
    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    EXPECT_FALSE(
        hahaha::backend::dispatchAdd(DeviceType::HIP, a, 1.0f, res).has_value());
    EXPECT_FALSE(
        hahaha::backend::dispatchAdd(DeviceType::HIP, 1.0f, a, res).has_value());
}

TEST(DeviceComputeDispatcherTest, DispatchBinary_Broadcasting_Works) {
    // (2, 2) + (2, 1) -> (2, 2)
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f}, {20.0f}}); // shape (2, 1)
    TensorWrapper<float> b_broadcasted =
        b.broadcastTo(TensorShape({2, 2})); // strides will have 0

    TensorWrapper<float> res(
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto result =
        hahaha::backend::dispatchAdd(DeviceType::CPU, a, b_broadcasted, res);
    EXPECT_TRUE(result.has_value());

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
        TensorShape({2, 2}), 0.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto result =
        hahaha::backend::dispatchAdd(DeviceType::CPU, a_broadcasted, b, res);
    EXPECT_TRUE(result.has_value());

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
    auto result = hahaha::backend::dispatchAdd(DeviceType::CPU, ab, bb, res);
    EXPECT_TRUE(result.has_value());

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
    auto result = hahaha::backend::dispatchAdd(DeviceType::CPU, a, a, res);
    EXPECT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(res.at({1, 1}), 8.0f);
}

TEST(DeviceComputeDispatcherTest,
     DispatchMatMul_InnerDimensionMismatch_NoCheckYetButRun) {
    // Current implementation doesn't check inner dimension, it just loops.
    // We should at least cover the code.
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}});   // 1x2
    TensorWrapper<float> b(NestedData<float>{{3.0f}, {4.0f}}); // 2x1
    TensorWrapper<float> res(TensorShape({1, 1}), 0.0f);
    auto res_val = hahaha::backend::dispatchMatMul(DeviceType::CPU, a, b, res);
    EXPECT_TRUE(res_val.has_value());
    EXPECT_FLOAT_EQ(res.at({0, 0}), 11.0f);
}

TEST(DeviceComputeDispatcherTest, DispatchAxpy_UnsupportedDevice_Throws) {
    TensorWrapper<float> x(
        TensorShape({3}), 1.0f, std::make_shared<hahaha::backend::CPUDevice>());
    TensorWrapper<float> res(
        TensorShape({3}), 2.0f, std::make_shared<hahaha::backend::CPUDevice>());

    auto res_val = hahaha::backend::dispatchAxpy(DeviceType::HIP, 1.0f, x, res);
    EXPECT_FALSE(res_val.has_value());
    EXPECT_EQ(res_val.error().code, hahaha::common::ErrorCode::DeviceNotSupported);
}
