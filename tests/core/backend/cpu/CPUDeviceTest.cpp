//  Copyright (c) 2025 - 2026 Contributors of
//  Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//
//

#include "backend/cpu/CPUDevice.h"

#include <cstddef>
#include <gtest/gtest.h>

using namespace hahaha::backend;

class CPUDeviceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up if needed
    }

    void TearDown() override {
        // Clean up if needed
    }
};

TEST_F(CPUDeviceTest, Constructor) {
    const CPUDevice device;
    EXPECT_EQ(device.getType(), DeviceType::CPU);
    EXPECT_EQ(device.getId(), 0);
}

TEST_F(CPUDeviceTest, AllocateBasic) {
    CPUDevice device;
    size_t size = 1024;

    DeviceBuffer buffer = device.allocate(size);

    EXPECT_EQ(buffer.size(), size);
    EXPECT_NE(buffer.address(), 0); // Should have a non-zero address
}

TEST_F(CPUDeviceTest, AllocateZeroSize) {
    CPUDevice device;
    size_t size = 0;

    DeviceBuffer buffer = device.allocate(size);

    EXPECT_EQ(buffer.size(), size);
    EXPECT_NE(
        buffer.address(),
        0); // Even zero-size allocation typically returns non-null pointer
}

TEST_F(CPUDeviceTest, AllocateAndDeallocate) {
    CPUDevice device;
    size_t size = 512;

    DeviceBuffer buffer = device.allocate(size);

    EXPECT_EQ(buffer.size(), size);
    EXPECT_NE(buffer.address(), 0);

    // Deallocate should not throw an exception - use move semantics since
    // DeviceBuffer is not copyable
    EXPECT_NO_THROW(device.deallocate(std::move(buffer)));
}

TEST_F(CPUDeviceTest, MultipleAllocations) {
    CPUDevice device;

    DeviceBuffer buffer1 = device.allocate(256);
    DeviceBuffer buffer2 = device.allocate(512);
    DeviceBuffer buffer3 = device.allocate(128);

    EXPECT_EQ(buffer1.size(), 256);
    EXPECT_EQ(buffer2.size(), 512);
    EXPECT_EQ(buffer3.size(), 128);

    EXPECT_NE(buffer1.address(), 0);
    EXPECT_NE(buffer2.address(), 0);
    EXPECT_NE(buffer3.address(), 0);

    // All addresses should be different
    EXPECT_NE(buffer1.address(), buffer2.address());
    EXPECT_NE(buffer1.address(), buffer3.address());
    EXPECT_NE(buffer2.address(), buffer3.address());

    // Deallocate all buffers
    EXPECT_NO_THROW(device.deallocate(std::move(buffer1)));
    EXPECT_NO_THROW(device.deallocate(std::move(buffer2)));
    EXPECT_NO_THROW(device.deallocate(std::move(buffer3)));
}

TEST_F(CPUDeviceTest, LargeAllocation) {
    CPUDevice device;
    size_t size = 1024 * 1024; // 1MB

    DeviceBuffer buffer = device.allocate(size);

    EXPECT_EQ(buffer.size(), size);
    EXPECT_NE(buffer.address(), 0);

    EXPECT_NO_THROW(device.deallocate(std::move(buffer)));
}

TEST_F(CPUDeviceTest, DeviceTypeAndId) {
    CPUDevice device;

    EXPECT_EQ(device.getType(), DeviceType::CPU);
    EXPECT_EQ(device.getId(), 0);

    // Verify the type doesn't change after allocations
    DeviceBuffer buffer = device.allocate(100);
    EXPECT_EQ(device.getType(), DeviceType::CPU);
    EXPECT_EQ(device.getId(), 0);

    EXPECT_NO_THROW(device.deallocate(std::move(buffer)));
    EXPECT_EQ(device.getType(), DeviceType::CPU);
    EXPECT_EQ(device.getId(), 0);
}

TEST_F(CPUDeviceTest, DeallocateTwice) {
    CPUDevice device;
    DeviceBuffer buffer = device.allocate(64);

    // First deallocation should be fine
    EXPECT_NO_THROW(device.deallocate(std::move(buffer)));

    // Second allocation and deallocation
    DeviceBuffer buffer2 = device.allocate(0);
    EXPECT_NO_THROW(device.deallocate(std::move(buffer2)));
}

TEST_F(CPUDeviceTest, Destructor) {
    // Creating and destroying CPUDevice should not cause issues
    {
        CPUDevice device;
        EXPECT_EQ(device.getType(), DeviceType::CPU);
        EXPECT_EQ(device.getId(), 0);

        DeviceBuffer buffer = device.allocate(32);
        EXPECT_EQ(buffer.size(), 32);
        EXPECT_NE(buffer.address(), 0);

        // Buffer gets deallocated when device goes out of scope
        device.deallocate(std::move(buffer));
    }

    // Test successful exit without crashing
    SUCCEED();
}

TEST_F(CPUDeviceTest, AllocateVerySmallSize) {
    CPUDevice device;
    size_t size = 1; // smallest possible allocation

    DeviceBuffer buffer = device.allocate(size);

    EXPECT_EQ(buffer.size(), size);
    EXPECT_NE(buffer.address(), 0);

    EXPECT_NO_THROW(device.deallocate(std::move(buffer)));
}

TEST_F(CPUDeviceTest, AllocateMediumSizes) {
    CPUDevice device;

    // Test various medium-sized allocations
    std::vector<size_t> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (size_t size : sizes) {
        DeviceBuffer buffer = device.allocate(size);

        EXPECT_EQ(buffer.size(), size);
        EXPECT_NE(buffer.address(), 0);

        EXPECT_NO_THROW(device.deallocate(std::move(buffer)));
    }
}

TEST_F(CPUDeviceTest, InheritedVirtualMethods) {
    CPUDevice device;

    // Test that virtual methods from base Device class work
    EXPECT_EQ(device.getType(), DeviceType::CPU);
    EXPECT_EQ(device.getId(), 0);

    // Test that the virtual allocate/deallocate methods work as intended
    DeviceBuffer buffer = device.allocate(100);
    EXPECT_EQ(buffer.size(), 100);
    EXPECT_NE(buffer.address(), 0);

    EXPECT_NO_THROW(device.deallocate(std::move(buffer)));
}
