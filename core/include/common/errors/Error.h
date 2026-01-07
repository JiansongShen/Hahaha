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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//
//

#ifndef HAHAHA_ERROR_H_053340FD56184B63A68D6CE6E15C8175
#define HAHAHA_ERROR_H_053340FD56184B63A68D6CE6E15C8175

#include <source_location>
#include <string>

#include "common/errors/ErrorCode.h"

namespace hahaha::common {

/**
 * @brief CRTP base class for all error types.
 *
 * @tparam Derived The derived error class (CRTP pattern)
 * @tparam code The error code associated with this error type
 */
template <typename Derived, ErrorCode code> class ErrorBase {
  public:
  public:
    /**
     * @brief The error code associated with this error type.
     */
    static constexpr ErrorCode Code = code;

    /**
     * @brief Default constructor for ErrorBase.
     */
    ErrorBase() = default;

    /**
     * @brief Constructor for ErrorBase with message and optional location.
     *
     * @param message The error message string
     * @param location The source location where the error occurred (defaults to
     * current location)
     */
    explicit ErrorBase(
        std::string message,
        const std::source_location location = std::source_location::current())
        : message(std::move(message)), location(location) {
    }

    /**
     * @brief Get the error code associated with this error type.
     *
     * @return constexpr ErrorCode The error code
     */
    static constexpr ErrorCode getCode() {
        return Code;
    }

    /**
     * @brief Get the error message.
     *
     * @return std::string The error message
     */
    std::string getMessage() const {
        return message;
    }

    /**
     * @brief Get the location where the error occurred.
     *
     * @return std::string The location in format "filename:line_number"
     */
    std::string getLocation() const {
        return std::string(location.file_name()) + ":"
            + std::to_string(location.line());
    }

    /**
     * @brief Get the error name from the derived class.
     * @return The error name string
     */
    static constexpr const char* getErrorName() {
        return Derived::ErrorName;
    }

  protected:
    std::string message; ///< The error message
    std::source_location
        location; ///< The source location where the error occurred
};

} // namespace hahaha::common

#endif // HAHAHA_ERROR_H_053340FD56184B63A68D6CE6E15C8175
