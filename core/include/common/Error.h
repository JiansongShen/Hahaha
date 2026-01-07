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

#ifndef HAHAHA_COMMON_ERROR_H
#define HAHAHA_COMMON_ERROR_H

#include <expected>
#include <string>
#include <string_view>

namespace hahaha::common {

enum class ErrorCode {
    Success = 0,
    InvalidArgument,
    OutOfRange,
    DimensionMismatch,
    DeviceMismatch,
    NotImplemented,
    RuntimeError,
    DivisionByZero
};

/**
 * @brief Represents a structured error in the Hahaha library.
 */
class Error {
  public:
    Error(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {
    }

    [[nodiscard]] ErrorCode code() const {
        return code_;
    }
    [[nodiscard]] const std::string& message() const {
        return message_;
    }

    [[nodiscard]] std::string toString() const {
        return "[" + errorCodeToString(code_) + "] " + message_;
    }

  private:
    static std::string errorCodeToString(ErrorCode code) {
        switch (code) {
        case ErrorCode::Success:
            return "Success";
        case ErrorCode::InvalidArgument:
            return "InvalidArgument";
        case ErrorCode::OutOfRange:
            return "OutOfRange";
        case ErrorCode::DimensionMismatch:
            return "DimensionMismatch";
        case ErrorCode::DeviceMismatch:
            return "DeviceMismatch";
        case ErrorCode::NotImplemented:
            return "NotImplemented";
        case ErrorCode::RuntimeError:
            return "RuntimeError";
        case ErrorCode::DivisionByZero:
            return "DivisionByZero";
        default:
            return "UnknownError";
        }
    }

    ErrorCode code_;
    std::string message_;
};

/**
 * @brief Alias for standard expected type using our custom Error.
 */
template <typename T> using Result = std::expected<T, Error>;

/**
 * @brief Helper to create an unexpected error result.
 */
inline auto makeUnexpected(ErrorCode code, std::string message) {
    return std::unexpected(Error(code, std::move(message)));
}

} // namespace hahaha::common

#endif // HAHAHA_COMMON_ERROR_H
