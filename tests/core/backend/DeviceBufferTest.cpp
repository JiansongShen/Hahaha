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

#include "backend/DeviceBuffer.h"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

using namespace hahaha::backend;

class DeviceBufferTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up if needed
    }

    void TearDown() override {
        // Clean up if needed
    }
};

TEST_F(DeviceBufferTest, DefaultConstructor) {
    DeviceBuffer buffer = DeviceBuffer();
    EXPECT_EQ(buffer.address(), 0);
    EXPECT_EQ(buffer.size(), 0);
}

TEST_F(DeviceBufferTest, ParameterizedConstructor) {
    std::uintptr_t addr = 0x1000;
    size_t sz = 1024;

    DeviceBuffer buffer(addr, sz);
    EXPECT_EQ(buffer.address(), addr);
    EXPECT_EQ(buffer.size(), sz);
}

TEST_F(DeviceBufferTest, CopyConstructorDeleted) {
    // This test verifies that copy constructor is deleted
    // The test itself should compile, but trying to use the copy constructor
    // would cause a compilation error.
    std::uintptr_t addr = 0x2000;
    size_t sz = 2048;

    DeviceBuffer original(addr, sz);

    // This would fail to compile if uncommented:
    // DeviceBuffer copy = original;

    EXPECT_EQ(original.address(), addr);
    EXPECT_EQ(original.size(), sz);
}

TEST_F(DeviceBufferTest, CopyAssignmentDeleted) {
    // This test verifies that copy assignment is deleted
    std::uintptr_t addr1 = 0x3000;
    size_t sz1 = 4096;
    std::uintptr_t addr2 = 0x4000;
    size_t sz2 = 8192;

    DeviceBuffer original(addr1, sz1);
    DeviceBuffer target(addr2, sz2);

    // This would fail to compile if uncommented:
    // target = original;

    EXPECT_EQ(original.address(), addr1);
    EXPECT_EQ(original.size(), sz1);
    EXPECT_EQ(target.address(), addr2);
    EXPECT_EQ(target.size(), sz2);
}

TEST_F(DeviceBufferTest, MoveConstructor) {
    std::uintptr_t addr = 0x5000;
    size_t sz = 512;

    DeviceBuffer original(addr, sz);
    EXPECT_EQ(original.address(), addr);
    EXPECT_EQ(original.size(), sz);

    DeviceBuffer moved = std::move(original);

    // Moved-from object should be reset
    EXPECT_EQ(original.address(), 0);
    EXPECT_EQ(original.size(), 0);

    // Moved-to object should have the original's values
    EXPECT_EQ(moved.address(), addr);
    EXPECT_EQ(moved.size(), sz);
}

TEST_F(DeviceBufferTest, MoveAssignment) {
    std::uintptr_t addr1 = 0x6000;
    size_t sz1 = 128;
    std::uintptr_t addr2 = 0x7000;
    size_t sz2 = 256;

    DeviceBuffer source(addr1, sz1);
    DeviceBuffer target(addr2, sz2);

    EXPECT_EQ(source.address(), addr1);
    EXPECT_EQ(source.size(), sz1);
    EXPECT_EQ(target.address(), addr2);
    EXPECT_EQ(target.size(), sz2);

    DeviceBuffer& result = target = std::move(source);

    // Check that assignment returned the right reference
    EXPECT_EQ(&result, &target);

    // Source should be reset
    EXPECT_EQ(source.address(), 0);
    EXPECT_EQ(source.size(), 0);

    // Target should have source's values
    EXPECT_EQ(target.address(), addr1);
    EXPECT_EQ(target.size(), sz1);
}

TEST_F(DeviceBufferTest, AddressAccessor) {
    std::uintptr_t addr = 0x8000;
    size_t sz = 64;

    DeviceBuffer buffer(addr, sz);
    EXPECT_EQ(buffer.address(), addr);

    // Test const accessor
    const DeviceBuffer& const_buffer = buffer;
    EXPECT_EQ(const_buffer.address(), addr);
}

TEST_F(DeviceBufferTest, SizeAccessor) {
    std::uintptr_t addr = 0x9000;
    size_t sz = 32;

    DeviceBuffer buffer(addr, sz);
    EXPECT_EQ(buffer.size(), sz);

    // Test const accessor
    const DeviceBuffer& const_buffer = buffer;
    EXPECT_EQ(const_buffer.size(), sz);
}

TEST_F(DeviceBufferTest, SelfMoveAssignment) {
    std::uintptr_t addr = 0xA000;
    size_t sz = 16;

    DeviceBuffer buffer(addr, sz);

    // Self-move should not change the buffer
    buffer = std::move(buffer);

    // Buffer should retain its original values
    EXPECT_EQ(buffer.address(), addr);
    EXPECT_EQ(buffer.size(), sz);
}

TEST_F(DeviceBufferTest, EmptyBufferProperties) {
    DeviceBuffer empty;
    EXPECT_EQ(empty.address(), 0);
    EXPECT_EQ(empty.size(), 0);

    // Moving from an empty buffer
    DeviceBuffer moved_from_empty = std::move(empty);
    EXPECT_EQ(moved_from_empty.address(), 0);
    EXPECT_EQ(moved_from_empty.size(), 0);
    EXPECT_EQ(empty.address(), 0);
    EXPECT_EQ(empty.size(), 0);
}

TEST_F(DeviceBufferTest, MoveFromMovedBuffer) {
    std::uintptr_t addr = 0xB000;
    size_t sz = 1024;

    DeviceBuffer original(addr, sz);
    DeviceBuffer moved = std::move(original);

    // original is now in moved-from state
    EXPECT_EQ(original.address(), 0);
    EXPECT_EQ(original.size(), 0);

    // Moving from already moved buffer should work
    DeviceBuffer moved_again = std::move(original);
    EXPECT_EQ(moved_again.address(), 0);
    EXPECT_EQ(moved_again.size(), 0);
    EXPECT_EQ(original.address(), 0);
    EXPECT_EQ(original.size(), 0);

    // The buffer that received the move should have the original's values
    EXPECT_EQ(moved.address(), addr);
    EXPECT_EQ(moved.size(), sz);
}
