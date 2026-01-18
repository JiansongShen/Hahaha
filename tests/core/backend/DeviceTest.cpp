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

#include "backend/DeviceBuffer.h"
#include "backend/cpu/CPUDevice.h"
#include "backend/gpu/GPUDevice.h"

using hahaha::backend::CPUDevice;
using hahaha::backend::Device;
using hahaha::backend::DeviceBuffer;
using hahaha::backend::DeviceType;
using hahaha::backend::GPUDevice;

// Test class to allow testing of edge cases
class TestDevice : public Device {
  public:
    TestDevice(DeviceType type, std::uint8_t id) : Device(type, id) {
    }

    // Implement pure virtual methods
    DeviceBuffer allocate(size_t size) override {
        // For testing purposes, return an empty DeviceBuffer
        return DeviceBuffer();
    }

    void deallocate(DeviceBuffer buffer) override {
        // For testing purposes, do nothing
    }
};

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

// Additional tests to improve coverage
TEST(DeviceTest, DefaultConstructor) {
    CPUDevice defaultDevice;
    EXPECT_EQ(defaultDevice.getType(), DeviceType::CPU);
    EXPECT_EQ(defaultDevice.getId(), 0);
}

TEST(DeviceTest, ParameterizedConstructor) {
    TestDevice paramDevice(DeviceType::CPU, 5);
    EXPECT_EQ(paramDevice.getType(), DeviceType::CPU);
    EXPECT_EQ(paramDevice.getId(), 5);
}

TEST(DeviceTest, UnknownDeviceTypeToString) {
    // Test with an unknown device type (beyond COMPILE_TIME_MAX)
    TestDevice unknownDevice(static_cast<DeviceType>(100), 0);
    EXPECT_EQ(unknownDevice.toString(), "Unknown:0");
}

TEST(DeviceTest, DeviceWithCustomId) {
    TestDevice deviceWithId(DeviceType::CPU, 7);
    EXPECT_EQ(deviceWithId.toString(), "CPU:7");
    EXPECT_EQ(deviceWithId.getType(), DeviceType::CPU);
    EXPECT_EQ(deviceWithId.getId(), 7);
}

TEST(DeviceTest, EqualitySameDevice) {
    CPUDevice device1;
    CPUDevice device2;

    EXPECT_TRUE(device1 == device2);
    EXPECT_FALSE(device1 != device2);
}

TEST(DeviceTest, InequalityDifferentType) {
    CPUDevice device1;
    GPUDevice device2(DeviceType::CUDA);

    EXPECT_TRUE(device1 != device2);
    EXPECT_FALSE(device1 == device2);
}

TEST(DeviceTest, InequalityDifferentId) {
    TestDevice device1(DeviceType::CPU, 0);
    TestDevice device2(DeviceType::CPU, 1);

    EXPECT_TRUE(device1 != device2);
    EXPECT_FALSE(device1 == device2);
}
