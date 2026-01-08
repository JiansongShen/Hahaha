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
//

#include <expected>
#include <gtest/gtest.h>
#include <source_location>

#include "common/error_handler.h"

class ErrorHandlerTest : public ::testing::Test {
  protected:
};

TEST_F(ErrorHandlerTest, Error_Constructor) {
    const hahaha::common::CommonError error;

    ASSERT_EQ(error.getMessage(), "");
    ASSERT_EQ(error.getLocation(), ":0");
    ASSERT_EQ(error.getCode(), hahaha::common::ErrorCode::BasicError);
    ASSERT_STREQ(error.getErrorName(), "CommonError");
}

TEST_F(ErrorHandlerTest, Error_Constructor_With_Message) {
    const hahaha::common::CommonError error("Test error message");

    ASSERT_EQ(error.getMessage(), "Test error message");
    ASSERT_NE(error.getLocation(), "");
    ASSERT_EQ(error.getCode(), hahaha::common::ErrorCode::BasicError);
    ASSERT_STREQ(hahaha::common::CommonError::ErrorName, "CommonError");
}

TEST_F(ErrorHandlerTest, Error_Constructor_With_Location_And_Message) {
    const std::source_location loc = std::source_location::current();
    const hahaha::common::CommonError error("Test error message", loc);

    ASSERT_EQ(error.getMessage(), "Test error message");
    ASSERT_EQ(error.getLocation(),
              std::string(loc.file_name()) + ":" + std::to_string(loc.line()));
    ASSERT_EQ(error.getCode(), hahaha::common::ErrorCode::BasicError);
    ASSERT_STREQ(hahaha::common::CommonError::ErrorName, "CommonError");
}

std::expected<int, hahaha::common::CommonError> testReturnSuccess() {
    return hahaha::common::ok<int, hahaha::common::CommonError>(42);
}

std::expected<int, hahaha::common::CommonError> testReturnSuccessQuickly() {
    return 1;
}

std::expected<int, hahaha::common::CommonError> testReturnFailed() {
    return hahaha::common::err(hahaha::common::CommonError());
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
    ASSERT_EQ(e.error().getCode(), hahaha::common::ErrorCode::BasicError);
    ASSERT_STREQ(hahaha::common::CommonError::ErrorName, "CommonError");
}

TEST_F(ErrorHandlerTest, FailedWithError_WithMessage) {
    std::expected<int, hahaha::common::CommonError> e =
        hahaha::common::err(hahaha::common::CommonError("Operation failed"));
    ASSERT_FALSE(e.has_value());
    ASSERT_EQ(e.error().getCode(), hahaha::common::ErrorCode::BasicError);
    ASSERT_EQ(e.error().getMessage(), "Operation failed");
    ASSERT_STREQ(hahaha::common::CommonError::ErrorName, "CommonError");
}
