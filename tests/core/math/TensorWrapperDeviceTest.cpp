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

#include "backend/Device.h"
#include "backend/cpu/CPUDevice.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaDevice.h"
#endif
#endif

using hahaha::backend::DeviceType;
using hahaha::math::NestedData;
using hahaha::math::TensorShape;
using hahaha::math::TensorWrapper;

class TensorWrapperDeviceTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        cudaDeviceSynchronize();
        cudaGetDeviceProperties(&prop_, 0);
        cudaDevice_ = std::make_shared<hahaha::backend::CudaDevice>(&prop_);
#endif
#endif
    }

    void TearDown() override {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        cudaDeviceSynchronize();
#endif
#endif
    }

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
    cudaDeviceProp prop_{};
    std::shared_ptr<hahaha::backend::CudaDevice> cudaDevice_;
#endif
#endif
};

// Test getting device from tensor
TEST_F(TensorWrapperDeviceTest, GetDevice_DefaultIsCPU) {
    TensorWrapper<float> t(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});

    auto device = t.getDevice();
    EXPECT_EQ(device->getType(), DeviceType::CPU);
    EXPECT_EQ(device->getId(), 0);
}

// Test moving tensor to CPU device (should be no-op if already on CPU)
TEST_F(TensorWrapperDeviceTest, To_CPU_AlreadyOnCPU_NoOp) {
    TensorWrapper<float> t(NestedData<float>{{1.0f, 2.0f}});
    auto originalDevice = t.getDevice();

    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    t.to(cpuDevice);

    auto newDevice = t.getDevice();
    EXPECT_EQ(*newDevice, *originalDevice);
    EXPECT_EQ(newDevice->getType(), DeviceType::CPU);
}

// Test moving tensor from CPU to CPU (same device)
TEST_F(TensorWrapperDeviceTest, To_CPU_FromCPU_NoOp) {
    TensorWrapper<float> t(NestedData<float>{{1.0f, 2.0f}});
    auto cpuDevice1 = std::make_shared<hahaha::backend::CPUDevice>();
    auto cpuDevice2 = std::make_shared<hahaha::backend::CPUDevice>();

    t.to(cpuDevice1);
    t.to(cpuDevice2); // Should be no-op

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CPU);
}

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

// Test moving tensor from CPU to CUDA
TEST_F(TensorWrapperDeviceTest, To_CUDA_FromCPU_UpdatesDevice) {
    TensorWrapper<float> t(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CPU);

    t.to(cudaDevice_);

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CUDA);
    // Data should still be accessible (currently data stays on CPU, only device
    // marker changes)
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

// Test moving tensor from CUDA back to CPU
TEST_F(TensorWrapperDeviceTest, To_CPU_FromCUDA_UpdatesDevice) {
    TensorWrapper<float> t(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});

    t.to(cudaDevice_);
    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CUDA);

    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    t.to(cpuDevice);

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CPU);
    // Data should still be accessible
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

// Test moving empty tensor to CUDA
TEST_F(TensorWrapperDeviceTest, To_CUDA_EmptyTensor) {
    TensorWrapper<float> t(TensorShape({0}));

    t.to(cudaDevice_);

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CUDA);
}

// Test moving scalar tensor to CUDA
TEST_F(TensorWrapperDeviceTest, To_CUDA_ScalarTensor) {
    TensorWrapper<float> t(42.0f);

    t.to(cudaDevice_);

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CUDA);
    EXPECT_FLOAT_EQ(t.at({}), 42.0f);
}

// Test moving large tensor to CUDA
TEST_F(TensorWrapperDeviceTest, To_CUDA_LargeTensor) {
    const size_t size = 1000;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    TensorWrapper<float> t(data);

    t.to(cudaDevice_);

    EXPECT_EQ(t.getDevice()->getType(), DeviceType::CUDA);
    // Verify data is still accessible
    EXPECT_FLOAT_EQ(t.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({size - 1}), static_cast<float>(size - 1));
}

// Test arithmetic operations on CUDA tensor
TEST_F(TensorWrapperDeviceTest, Arithmetic_CUDA_Works) {
    TensorWrapper<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    TensorWrapper<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});

    a.to(cudaDevice_);
    b.to(cudaDevice_);

    // These operations should work even if data is on GPU
    // (operations will handle device transfer internally)
    auto sum = a + b;
    EXPECT_EQ(sum.getDevice().getType(), DeviceType::CUDA);

    auto diff = b - a;
    EXPECT_EQ(diff.getDevice().getType(), DeviceType::CUDA);

    auto prod = a * b;
    EXPECT_EQ(prod.getDevice().getType(), DeviceType::CUDA);

    auto quot = b / a;
    EXPECT_EQ(quot.getDevice().getType(), DeviceType::CUDA);
}

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA

// Test that device information is preserved through copy
TEST_F(TensorWrapperDeviceTest, Copy_PreservesDevice) {
    TensorWrapper<float> t(NestedData<float>{{1.0f, 2.0f}});
    auto cpuDevice = std::make_shared<hahaha::backend::CPUDevice>();
    t.to(cpuDevice);

    TensorWrapper<float> copy = t;
    EXPECT_EQ(*copy.getDevice(), *t.getDevice());
}
