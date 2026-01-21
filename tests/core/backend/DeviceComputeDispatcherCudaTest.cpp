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

#include <gtest/gtest.h>

#include "backend/DeviceComputeDispatcher.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

#include <cuda_runtime.h>

#include "backend/Device.h"
#include "backend/gpu/cuda/CudaDevice.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"

using hahaha::backend::DeviceType;
using hahaha::math::NestedData;
using hahaha::math::TensorShape;
using hahaha::math::TensorWrapper;

class DeviceComputeDispatcherCudaTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Synchronize CUDA device
        cudaDeviceSynchronize();
        // Initialize CUDA device properties
        cudaGetDeviceProperties(&prop_, 0);
        cudaDevice_ = std::make_shared<hahaha::backend::CudaDevice>(&prop_);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    cudaDeviceProp prop_{};
    std::shared_ptr<hahaha::backend::CudaDevice> cudaDevice_;
};

// Test CUDA dispatch for addition
TEST_F(DeviceComputeDispatcherCudaTest, DispatchAdd_CUDA_WorksForDenseTensors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(TensorShape({2, 2}), 0.0f);

    // Move tensors to CUDA device
    a.to(cudaDevice_);
    b.to(cudaDevice_);
    res.to(cudaDevice_);

    auto result = hahaha::backend::dispatchAdd(DeviceType::CUDA, a, b, res);
    EXPECT_TRUE(result.has_value());

    res.to(hahaha::backend::getCPUDevice());

    // Verify results (with tolerance for floating point)
    EXPECT_NEAR(res.at({0, 0}), 11.0f, 1e-5f);
    EXPECT_NEAR(res.at({0, 1}), 22.0f, 1e-5f);
    EXPECT_NEAR(res.at({1, 0}), 33.0f, 1e-5f);
    EXPECT_NEAR(res.at({1, 1}), 44.0f, 1e-5f);
}

// Test CUDA dispatch for subtraction
TEST_F(DeviceComputeDispatcherCudaTest, DispatchSub_CUDA_WorksForDenseTensors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(TensorShape({2, 2}), 0.0f);

    a.to(cudaDevice_);
    b.to(cudaDevice_);
    res.to(cudaDevice_);

    auto result = hahaha::backend::dispatchSub(DeviceType::CUDA, b, a, res);
    EXPECT_TRUE(result.has_value());

    res.to(hahaha::backend::getCPUDevice());
    EXPECT_NEAR(res.at({0, 0}), 9.0f, 1e-5f);
    EXPECT_NEAR(res.at({1, 1}), 36.0f, 1e-5f);
}

// Test CUDA dispatch for multiplication
TEST_F(DeviceComputeDispatcherCudaTest, DispatchMul_CUDA_WorksForDenseTensors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(TensorShape({2, 2}), 0.0f);

    a.to(cudaDevice_);
    b.to(cudaDevice_);
    res.to(cudaDevice_);

    auto result = hahaha::backend::dispatchMul(DeviceType::CUDA, a, b, res);
    EXPECT_TRUE(result.has_value());

    res.to(hahaha::backend::getCPUDevice());
    EXPECT_NEAR(res.at({0, 0}), 10.0f, 1e-5f);
    EXPECT_NEAR(res.at({1, 1}), 160.0f, 1e-5f);
}

// Test CUDA dispatch for division
TEST_F(DeviceComputeDispatcherCudaTest, DispatchDiv_CUDA_WorksForDenseTensors) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> res(TensorShape({2, 2}), 0.0f);

    a.to(cudaDevice_);
    b.to(cudaDevice_);
    res.to(cudaDevice_);

    auto result = hahaha::backend::dispatchDiv(DeviceType::CUDA, b, a, res);
    EXPECT_TRUE(result.has_value());

    res.to(hahaha::backend::getCPUDevice());
    EXPECT_NEAR(res.at({0, 0}), 10.0f, 1e-5f);
    EXPECT_NEAR(res.at({1, 1}), 10.0f, 1e-5f);
}

// Test CUDA dispatch with non-contiguous data (should fall back to CPU)
TEST_F(DeviceComputeDispatcherCudaTest,
       DispatchAdd_CUDA_NonContiguous_ThrowError) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f}, {20.0f}}); // shape (2, 1)
    TensorWrapper<float> b_broadcasted = b.broadcastTo(TensorShape({2, 2}));

    TensorWrapper<float> res(TensorShape({2, 2}), 0.0f);

    a.to(cudaDevice_);
    b_broadcasted.to(cudaDevice_);
    res.to(cudaDevice_);

    // Non-contiguous data should fall back to CPU
    EXPECT_THROW(auto result = hahaha::backend::dispatchAdd(
                     DeviceType::CUDA, a, b_broadcasted, res),
                 std::runtime_error);
}

// Test CUDA dispatch with large tensors
TEST_F(DeviceComputeDispatcherCudaTest, DispatchAdd_CUDA_LargeTensor) {
    const size_t size = 128;
    std::vector<float> data1(size, 1.0f);
    std::vector<float> data2(size, 2.0f);

    TensorWrapper<float> a(data1);
    TensorWrapper<float> b(data2);
    TensorWrapper<float> res(TensorShape({size}), 0.0f);

    b.to(cudaDevice_);
    a.to(cudaDevice_);
    res.to(cudaDevice_);

    auto result = hahaha::backend::dispatchAdd(DeviceType::CUDA, a, b, res);
    EXPECT_TRUE(result.has_value());

    // Check a few elements
    EXPECT_NEAR(res.at({0}), 3.0f, 1e-5f);
    EXPECT_NEAR(res.at({size / 2}), 3.0f, 1e-5f);
    EXPECT_NEAR(res.at({size - 1}), 3.0f, 1e-5f);
}

// Test CUDA dispatch with scalar tensors
TEST_F(DeviceComputeDispatcherCudaTest, DispatchAdd_CUDA_ScalarTensor) {
    TensorWrapper<float> a(2.0f);
    TensorWrapper<float> b(3.0f);
    TensorWrapper<float> res(0.0f);

    a.to(cudaDevice_);
    b.to(cudaDevice_);
    res.to(cudaDevice_);

    auto result = hahaha::backend::dispatchAdd(DeviceType::CUDA, a, b, res);
    EXPECT_TRUE(result.has_value());

    EXPECT_NEAR(res.at({}), 5.0f, 1e-5f);
}

// Test error handling for shape mismatch
TEST_F(DeviceComputeDispatcherCudaTest,
       DispatchAdd_CUDA_ShapeMismatch_ReturnsError) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    TensorWrapper<float> badRes(TensorShape({4}), 0.0f);

    a.to(cudaDevice_);
    b.to(cudaDevice_);
    badRes.to(cudaDevice_);

    auto result = hahaha::backend::dispatchAdd(DeviceType::CUDA, a, b, badRes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, hahaha::common::ErrorCode::InvalidArgument);
}

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
