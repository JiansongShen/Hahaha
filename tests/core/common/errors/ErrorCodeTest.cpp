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

#include "common/errors/ErrorCode.h"

#include <cstddef>
#include <gtest/gtest.h>

using namespace hahaha::common;

class ErrorCodeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up if needed
    }

    void TearDown() override {
        // Clean up if needed
    }
};

TEST_F(ErrorCodeTest, ErrorCodeEnumValues) {
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::Success), 0);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::InvalidArgument), 1);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::DeviceNotSupported), 2);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::Overflow), 3);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::InternalError), 4);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::InvalidDataset), 5);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::CudaDeviceOutOfMemory), 6);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::CudaSmallObjectMemoryPoolFull), 7);
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::Count), 8);
}

TEST_F(ErrorCodeTest, ErrorMessagesArraySize) {
    // The ErrorMessages array should have the same number of entries as the
    // enum Since ErrorCode::Count represents the number of enum values, the
    // array size should match
    const std::size_t errorCodeCount = static_cast<std::size_t>(ErrorCode::Count);
    EXPECT_EQ(errorCodeCount, 8); // Updated to reflect all enum values
                                  // including CudaSmallObjectMemoryPoolFull

    // Verify that ErrorMessages array has the right number of elements
    EXPECT_STREQ(ErrorMessages[0], "Success");
    EXPECT_STREQ(ErrorMessages[1], "Invalid argument (e.g., shape mismatch)");
    EXPECT_STREQ(ErrorMessages[2], "Device unsupported");
    EXPECT_STREQ(ErrorMessages[3], "Numerical overflow");
    EXPECT_STREQ(ErrorMessages[4], "Internal error");
    EXPECT_STREQ(ErrorMessages[5], "Invalid dataset");
    EXPECT_STREQ(ErrorMessages[6], "Cuda device out of memory");
    EXPECT_STREQ(ErrorMessages[7], "CUDA small object memory pool full");
}

TEST_F(ErrorCodeTest, ErrorMessagesAccessibility) {
    // Test that all error messages can be accessed by their corresponding enum
    // values
    EXPECT_STREQ(ErrorMessages[static_cast<std::size_t>(ErrorCode::Success)],
                 "Success");
    EXPECT_STREQ(ErrorMessages[static_cast<std::size_t>(ErrorCode::InvalidArgument)],
                 "Invalid argument (e.g., shape mismatch)");
    EXPECT_STREQ(
        ErrorMessages[static_cast<std::size_t>(ErrorCode::DeviceNotSupported)],
        "Device unsupported");
    EXPECT_STREQ(ErrorMessages[static_cast<std::size_t>(ErrorCode::Overflow)],
                 "Numerical overflow");
    EXPECT_STREQ(ErrorMessages[static_cast<std::size_t>(ErrorCode::InternalError)],
                 "Internal error");
    EXPECT_STREQ(ErrorMessages[static_cast<std::size_t>(ErrorCode::InvalidDataset)],
                 "Invalid dataset");
    EXPECT_STREQ(
        ErrorMessages[static_cast<std::size_t>(ErrorCode::CudaDeviceOutOfMemory)],
        "Cuda device out of memory");
    EXPECT_STREQ(ErrorMessages[static_cast<std::size_t>(
                     ErrorCode::CudaSmallObjectMemoryPoolFull)],
                 "CUDA small object memory pool full");
}

TEST_F(ErrorCodeTest, ErrorCodeCountRepresentsTotal) {
    // The Count enum value should represent the total number of error codes
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::Count), 8);

    // This should match the actual number of error codes defined (excluding
    // Count itself)
    const std::size_t expectedCount =
        8; // Success, InvalidArgument, DeviceNotSupported, Overflow,
           // InternalError, InvalidDataset, CudaDeviceOutOfMemory,
           // CudaSmallObjectMemoryPoolFull
    EXPECT_EQ(static_cast<std::size_t>(ErrorCode::Count), expectedCount);
}

TEST_F(ErrorCodeTest, ErrorMessageArrayBoundsCheck) {
    // Ensure we don't access beyond the array bounds
    const std::size_t maxIndex = static_cast<std::size_t>(ErrorCode::Count) - 1;

    // Test that the highest valid index has a proper message
    EXPECT_STREQ(ErrorMessages[maxIndex], "CUDA small object memory pool full");

    // Test a few more to ensure array integrity
    for (std::size_t i = 0; i < static_cast<std::size_t>(ErrorCode::Count); ++i) {
        // Each index should have a valid string
        EXPECT_NE(ErrorMessages[i], nullptr);
    }
}

TEST_F(ErrorCodeTest, EnumValuesAreSequential) {
    // Check that enum values are sequential starting from 0
    EXPECT_EQ(static_cast<int>(ErrorCode::Success), 0);
    EXPECT_EQ(static_cast<int>(ErrorCode::InvalidArgument), 1);
    EXPECT_EQ(static_cast<int>(ErrorCode::DeviceNotSupported), 2);
    EXPECT_EQ(static_cast<int>(ErrorCode::Overflow), 3);
    EXPECT_EQ(static_cast<int>(ErrorCode::InternalError), 4);
    EXPECT_EQ(static_cast<int>(ErrorCode::InvalidDataset), 5);
    EXPECT_EQ(static_cast<int>(ErrorCode::CudaDeviceOutOfMemory), 6);
    EXPECT_EQ(static_cast<int>(ErrorCode::CudaSmallObjectMemoryPoolFull), 7);
    EXPECT_EQ(static_cast<int>(ErrorCode::Count), 8);
}

TEST_F(ErrorCodeTest, ErrorMessagesContent) {
    // Verify the content of each error message
    EXPECT_EQ(
        std::string(ErrorMessages[static_cast<std::size_t>(ErrorCode::Success)]),
        "Success");
    EXPECT_EQ(
        std::string(
            ErrorMessages[static_cast<std::size_t>(ErrorCode::InvalidArgument)]),
        "Invalid argument (e.g., shape mismatch)");
    EXPECT_EQ(
        std::string(
            ErrorMessages[static_cast<std::size_t>(ErrorCode::DeviceNotSupported)]),
        "Device unsupported");
    EXPECT_EQ(
        std::string(ErrorMessages[static_cast<std::size_t>(ErrorCode::Overflow)]),
        "Numerical overflow");
    EXPECT_EQ(std::string(
                  ErrorMessages[static_cast<std::size_t>(ErrorCode::InternalError)]),
              "Internal error");
    EXPECT_EQ(
        std::string(
            ErrorMessages[static_cast<std::size_t>(ErrorCode::InvalidDataset)]),
        "Invalid dataset");
    EXPECT_EQ(std::string(ErrorMessages[static_cast<std::size_t>(
                  ErrorCode::CudaDeviceOutOfMemory)]),
              "Cuda device out of memory");
    EXPECT_EQ(std::string(ErrorMessages[static_cast<std::size_t>(
                  ErrorCode::CudaSmallObjectMemoryPoolFull)]),
              "CUDA small object memory pool full");
}
