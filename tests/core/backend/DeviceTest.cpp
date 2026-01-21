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

namespace {

// Concrete implementation for testing
class TestDevice : public Device {
  public:
    explicit TestDevice(DeviceType type, std::uint8_t id = 0)
        : Device(type, id) {
    }

    DeviceBuffer allocate(size_t size) override {
        return {};
    }
    void deallocate(DeviceBuffer buffer) override {
    }

    void copyMemoryFrom(std::span<std::byte> src,
                        std::span<std::byte> dst,
                        const std::shared_ptr<Device>& dstDevice) override {
    }
    void copyMemoryTo(std::span<std::byte> dst,
                      std::span<std::byte> src,
                      const std::shared_ptr<Device>& srcDevice) override {
    }
};

// Specialized class to test invalid device type
class TestDeviceWithInvalidType : public Device {
  public:
    explicit TestDeviceWithInvalidType(std::uint8_t rawType,
                                       std::uint8_t id = 0)
        : Device(static_cast<DeviceType>(0), id) {
        // Directly set the underlying type value
        *reinterpret_cast<std::uint8_t*>(&type_) = rawType;
    }

    DeviceBuffer allocate(size_t size) override {
        return DeviceBuffer();
    }
    void deallocate(DeviceBuffer buffer) override {
    }

    void copyMemoryFrom(std::span<std::byte> src,
                        std::span<std::byte> dst,
                        const std::shared_ptr<Device>& dstDevice) override {
    }

    void copyMemoryTo(std::span<std::byte> src,
                      std::span<std::byte> dst,
                      const std::shared_ptr<Device>& srcDevice) override {
    }
};

} // anonymous namespace

TEST(DeviceTest, AllDeviceTypes) {
    // Test CPU
    TestDevice cpuDevice(DeviceType::CPU, 0);
    EXPECT_EQ(cpuDevice.toString(), "CPU:0");

    // Test CUDA
    TestDevice cudaDevice(DeviceType::CUDA, 1);
    EXPECT_EQ(cudaDevice.toString(), "CUDA:1");

    // Test HIP
    TestDevice hipDevice(DeviceType::HIP, 2);
    EXPECT_EQ(hipDevice.toString(), "HIP:2");

    // Test MPS
    TestDevice mpsDevice(DeviceType::MPS, 3);
    EXPECT_EQ(mpsDevice.toString(), "MPS:3");

    // Test XLA
    TestDevice xlaDevice(DeviceType::XLA, 4);
    EXPECT_EQ(xlaDevice.toString(), "XLA:4");

    // Test invalid type (beyond valid enum values)
    TestDeviceWithInvalidType invalidDevice(100, 99);
    EXPECT_EQ(invalidDevice.toString(), "Unknown:99");
}

TEST(DeviceTest, EqualityOperators) {
    // Same device - same type and ID
    TestDevice device1(DeviceType::CPU, 0);
    TestDevice device2(DeviceType::CPU, 0);
    EXPECT_TRUE(device1 == device2);
    EXPECT_FALSE(device1 != device2);

    // Different type - same ID
    TestDevice device3(DeviceType::CUDA, 0);
    EXPECT_TRUE(device1 != device3);
    EXPECT_FALSE(device1 == device3);

    // Different ID - same type
    TestDevice device4(DeviceType::CPU, 1);
    EXPECT_TRUE(device1 != device4);
    EXPECT_FALSE(device1 == device4);

    // Different type and ID
    TestDevice device5(DeviceType::CUDA, 1);
    EXPECT_TRUE(device1 != device5);
    EXPECT_FALSE(device1 == device5);
}

TEST(DeviceTest, ConstructorAndAccessors) {
    // Default constructor
    const CPUDevice defaultDevice;
    EXPECT_EQ(defaultDevice.getType(), DeviceType::CPU);
    EXPECT_EQ(defaultDevice.getId(), 0);

    // Parameterized constructor with custom ID
    TestDevice paramDevice(DeviceType::HIP, 5);
    EXPECT_EQ(paramDevice.getType(), DeviceType::HIP);
    EXPECT_EQ(paramDevice.getId(), 5);
}

TEST(DeviceTest, Destructor) {
    // Test that destructor works properly
    CPUDevice* device = new CPUDevice();
    delete device; // Should not crash

    TestDevice* testDevice = new TestDevice(DeviceType::CUDA, 1);
    delete testDevice; // Should not crash
}

TEST(DeviceTest, StringRepresentation) {
    // Test various combinations of device types and IDs
    EXPECT_EQ(TestDevice(DeviceType::CPU, 0).toString(), "CPU:0");
    EXPECT_EQ(TestDevice(DeviceType::CPU, 1).toString(), "CPU:1");
    EXPECT_EQ(TestDevice(DeviceType::CPU, 255).toString(), "CPU:255");

    EXPECT_EQ(TestDevice(DeviceType::CUDA, 0).toString(), "CUDA:0");
    EXPECT_EQ(TestDevice(DeviceType::CUDA, 123).toString(), "CUDA:123");

    EXPECT_EQ(TestDevice(DeviceType::HIP, 0).toString(), "HIP:0");
    EXPECT_EQ(TestDevice(DeviceType::MPS, 0).toString(), "MPS:0");
    EXPECT_EQ(TestDevice(DeviceType::XLA, 0).toString(), "XLA:0");

    // Test boundary case with maximum possible ID
    EXPECT_EQ(
        TestDevice(DeviceType::CPU, std::numeric_limits<std::uint8_t>::max())
            .toString(),
        "CPU:255");
}

TEST(DeviceTest, EdgeCases) {
    // Test with zero ID (default)
    TestDevice device1(DeviceType::CPU, 0);
    EXPECT_EQ(device1.getId(), 0);

    // Test with maximum ID
    TestDevice device2(DeviceType::CUDA, 255);
    EXPECT_EQ(device2.getId(), 255);

    // Test multiple instances with same parameters
    TestDevice device3(DeviceType::CPU, 0);
    TestDevice device4(DeviceType::CPU, 0);
    EXPECT_TRUE(device3 == device4);

    // Test comparison between different device types with same ID
    TestDevice device5(DeviceType::CUDA, 0);
    EXPECT_FALSE(device1 == device5);
}
