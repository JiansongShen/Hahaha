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

#include "backend/Device.h"

#include <gtest/gtest.h>

#include "backend/cpu/CPUDevice.h"
#include "backend/gpu/GPUDevice.h"

using hahaha::backend::CPUDevice;
using hahaha::backend::Device;
using hahaha::backend::DeviceType;
using hahaha::backend::GPUDevice;

TEST(DeviceTest, ToString_AllTypes) {
    EXPECT_EQ(CPUDevice().toString(), "CPU:0");
    EXPECT_EQ(GPUDevice(DeviceType::CUDA).toString(), "CUDA:0");
    EXPECT_EQ(GPUDevice(DeviceType::HIP).toString(), "HIP:0");
    EXPECT_EQ(GPUDevice(DeviceType::MPS).toString(), "MPS:0");
    EXPECT_EQ(GPUDevice(DeviceType::XLA).toString(), "XLA:0");
}

TEST(DeviceTest, Equality) {
    CPUDevice d1;
    CPUDevice d2;
    GPUDevice d3(DeviceType::CUDA);

    EXPECT_TRUE(d1 == d2);
    EXPECT_FALSE(d1 == d3);
    EXPECT_TRUE(d1 != d3);
}

TEST(DeviceTest, DifferentIds) {
    // We can't easily change ID via public API of CPUDevice/GPUDevice yet
    // without a custom constructor exposing ID. However, we can test that
    // different types are not equal.

    // For now, let's stick to testing what we can construct.
    GPUDevice d1(DeviceType::CUDA);
    GPUDevice d2(DeviceType::HIP);

    EXPECT_FALSE(d1 == d2);
    EXPECT_TRUE(d1 != d2);
}
