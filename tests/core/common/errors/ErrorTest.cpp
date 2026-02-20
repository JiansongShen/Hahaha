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

#include "common/errors/Error.h"

#include <gtest/gtest.h>

using namespace hahaha::common;

class ErrorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up if needed
    }

    void TearDown() override {
        // Clean up if needed
    }
};

TEST_F(ErrorTest, DefaultConstructor) {
    Error error;
    EXPECT_EQ(error.code, ErrorCode::Success);
    EXPECT_STREQ(error.message(), "Success");
    EXPECT_TRUE(error.isSuccess());
}

TEST_F(ErrorTest, ErrorCodeConstructor) {
    Error error(ErrorCode::InvalidArgument);
    EXPECT_EQ(error.code, ErrorCode::InvalidArgument);
    EXPECT_STREQ(error.message(), "Invalid argument (e.g., shape mismatch)");
    EXPECT_FALSE(error.isSuccess());
}

TEST_F(ErrorTest, isSuccessMethod) {
    Error success_error(ErrorCode::Success);
    EXPECT_TRUE(success_error.isSuccess());

    Error failure_error(ErrorCode::InvalidArgument);
    EXPECT_FALSE(failure_error.isSuccess());

    Error another_failure(ErrorCode::DeviceNotSupported);
    EXPECT_FALSE(another_failure.isSuccess());
}

TEST_F(ErrorTest, MessageMethod) {
    Error error1(ErrorCode::Success);
    EXPECT_STREQ(error1.message(), "Success");

    Error error2(ErrorCode::InvalidArgument);
    EXPECT_STREQ(error2.message(), "Invalid argument (e.g., shape mismatch)");

    Error error3(ErrorCode::DeviceNotSupported);
    EXPECT_STREQ(error3.message(), "Device unsupported");

    Error error4(ErrorCode::Overflow);
    EXPECT_STREQ(error4.message(), "Numerical overflow");

    Error error5(ErrorCode::InternalError);
    EXPECT_STREQ(error5.message(), "Internal error");

    Error error6(ErrorCode::CudaDeviceOutOfMemory);
    EXPECT_STREQ(error6.message(), "Cuda device out of memory");
}

TEST_F(ErrorTest, HelperFunctions) {
    Error invalid_arg = InvalidArgumentError();
    EXPECT_EQ(invalid_arg.code, ErrorCode::InvalidArgument);
    EXPECT_STREQ(invalid_arg.message(), "Invalid argument (e.g., shape mismatch)");
    EXPECT_FALSE(invalid_arg.isSuccess());

    Error device_not_supported = DeviceNotSupportedError();
    EXPECT_EQ(device_not_supported.code, ErrorCode::DeviceNotSupported);
    EXPECT_STREQ(device_not_supported.message(), "Device unsupported");
    EXPECT_FALSE(device_not_supported.isSuccess());

    Error cuda_out_of_memory = CudaDeviceOutOfMemoryError();
    EXPECT_EQ(cuda_out_of_memory.code, ErrorCode::CudaDeviceOutOfMemory);
    EXPECT_STREQ(cuda_out_of_memory.message(), "Cuda device out of memory");
    EXPECT_FALSE(cuda_out_of_memory.isSuccess());
}

TEST_F(ErrorTest, ErrorComparison) {
    Error error1(ErrorCode::Success);
    Error error2(ErrorCode::Success);
    Error error3(ErrorCode::InvalidArgument);

    EXPECT_EQ(error1.code, error2.code);
    EXPECT_NE(error1.code, error3.code);

    EXPECT_TRUE(error1.isSuccess());
    EXPECT_TRUE(error2.isSuccess());
    EXPECT_FALSE(error3.isSuccess());
}

TEST_F(ErrorTest, AllErrorCodesCovered) {
    Error success(ErrorCode::Success);
    Error invalidArg(ErrorCode::InvalidArgument);
    Error deviceNotSupported(ErrorCode::DeviceNotSupported);
    Error overflow(ErrorCode::Overflow);
    Error internalError(ErrorCode::InternalError);
    Error cudaOutOfMemory(ErrorCode::CudaDeviceOutOfMemory);

    EXPECT_TRUE(success.isSuccess());
    EXPECT_FALSE(invalidArg.isSuccess());
    EXPECT_FALSE(deviceNotSupported.isSuccess());
    EXPECT_FALSE(overflow.isSuccess());
    EXPECT_FALSE(internalError.isSuccess());
    EXPECT_FALSE(cudaOutOfMemory.isSuccess());

    EXPECT_STREQ(success.message(), "Success");
    EXPECT_STREQ(invalidArg.message(), "Invalid argument (e.g., shape mismatch)");
    EXPECT_STREQ(deviceNotSupported.message(), "Device unsupported");
    EXPECT_STREQ(overflow.message(), "Numerical overflow");
    EXPECT_STREQ(internalError.message(), "Internal error");
    EXPECT_STREQ(cudaOutOfMemory.message(), "Cuda device out of memory");
}
