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

#ifndef HAHAHA_ERROR_HANDLER_H_053340FD56184B63A68D6CE6E15C8175
#define HAHAHA_ERROR_HANDLER_H_053340FD56184B63A68D6CE6E15C8175

#include <expected>

// Include all error types
#include "common/errors/Error.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "common/errors/ErrorCode.h"

namespace hahaha::common {

/**
 * @brief Convenience function for creating an expected value (success case).
 *
 * This helper function wraps a value in `std::expected<T, E>` to indicate
 * a successful operation result.
 *
 * @tparam T The value type (result type when successful).
 * @tparam E The error type (type of error when operation fails).
 * @param val The value to wrap in an expected result.
 * @return std::expected<T, E> An expected value containing the moved value.
 *
 * @example
 * ```cpp
 * auto result = ok<int, std::string>(42);
 * if (result.has_value()) {
 *     std::cout << *result << std::endl; // Prints 42
 * }
 * ```
 */
template <typename T, typename E> auto ok(T val) {
    return std::expected<T, E>(std::move(val));
}

/**
 * @brief Convenience function for creating an unexpected value (error case).
 *
 * This helper function wraps an error value in `std::unexpected<E>` to
 * indicate a failed operation result. It can be used with `std::expected`
 * to represent error states.
 *
 * @tparam E The error type.
 * @param err The error value to wrap in an unexpected result.
 * @return std::unexpected<E> An unexpected value containing the moved error.
 *
 * @example
 * ```cpp
 * auto error = err<std::string>("Operation failed");
 * std::expected<int, std::string> result = error;
 * if (!result.has_value()) {
 *     std::cout << result.error() << std::endl; // Prints "Operation failed"
 * }
 * ```
 */
template <typename E> auto err(E err) {
    return std::unexpected(std::move(err));
}

} // namespace hahaha::common

#endif // HAHAHA_ERROR_HANDLER_H_053340FD56184B63A68D6CE6E15C8175
