//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include <expected>
#include <gtest/gtest.h>

#include "common/error_handler.h"

using hahaha::common::err;
using hahaha::common::Error;
using hahaha::common::ErrorCode;
using hahaha::common::ok;

class ErrorHandlerTest : public ::testing::Test {
  protected:
};

TEST_F(ErrorHandlerTest, Error_Properties) {
    Error error_success{ErrorCode::Success};
    EXPECT_TRUE(error_success.isSuccess());
    EXPECT_EQ(error_success.code, ErrorCode::Success);
    EXPECT_STREQ(error_success.message(), "Success");

    Error error_invalid{ErrorCode::InvalidArgument};
    EXPECT_FALSE(error_invalid.isSuccess());
    EXPECT_EQ(error_invalid.code, ErrorCode::InvalidArgument);
    EXPECT_STREQ(error_invalid.message(),
                 "Invalid argument (e.g., shape mismatch)");

    Error error_device{ErrorCode::DeviceNotSupported};
    EXPECT_FALSE(error_device.isSuccess());
    EXPECT_EQ(error_device.code, ErrorCode::DeviceNotSupported);
    EXPECT_STREQ(error_device.message(), "Device unsupported");
}

TEST_F(ErrorHandlerTest, Error_Helpers) {
    auto err_invalid = hahaha::common::InvalidArgumentError();
    EXPECT_EQ(err_invalid.code, ErrorCode::InvalidArgument);

    auto err_device = hahaha::common::DeviceNotSupportedError();
    EXPECT_EQ(err_device.code, ErrorCode::DeviceNotSupported);
}

std::expected<int, Error> testReturnSuccess() {
    return ok<int, Error>(42);
}

std::expected<int, Error> testReturnSuccessQuickly() {
    return 1;
}

std::expected<int, Error> testReturnFailed() {
    return err(Error{ErrorCode::InvalidArgument});
}

TEST_F(ErrorHandlerTest, OkWithValue) {
    const auto val1 = testReturnSuccess();
    ASSERT_TRUE(val1.has_value());
    ASSERT_EQ(42, val1.value());

    const auto val2 = testReturnSuccessQuickly();
    ASSERT_TRUE(val2.has_value());
    ASSERT_EQ(1, val2.value());
}

TEST_F(ErrorHandlerTest, FailedWithError) {
    const auto e = testReturnFailed();
    ASSERT_FALSE(e.has_value());
    ASSERT_EQ(e.error().code, ErrorCode::InvalidArgument);
    ASSERT_STREQ(e.error().message(),
                 "Invalid argument (e.g., shape mismatch)");
}

TEST_F(ErrorHandlerTest, FailedWithError_Manual) {
    std::expected<int, Error> e = err(Error{ErrorCode::DeviceNotSupported});
    ASSERT_FALSE(e.has_value());
    ASSERT_EQ(e.error().code, ErrorCode::DeviceNotSupported);
    ASSERT_STREQ(e.error().message(), "Device unsupported");
}
