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

#ifndef HAHAHA_ERROR_H_053340FD56184B63A68D6CE6E15C8175
#define HAHAHA_ERROR_H_053340FD56184B63A68D6CE6E15C8175

#include "common/errors/ErrorCode.h"

namespace hahaha::common {

/**
 * @brief Simple error structure for static error handling without dynamic
 * allocation.
 */
struct Error {
    ErrorCode code = ErrorCode::Success;

    /**
     * @brief Returns the error message associated with the current error code.
     * @return A static string representing the error.
     */
    [[nodiscard]] constexpr const char* message() const {
        return ErrorMessages[static_cast<std::size_t>(code)];
    }

    /**
     * @brief Helper to check if the error code indicates success.
     * @return True if code is Success, false otherwise.
     */
    [[nodiscard]] constexpr bool isSuccess() const {
        return code == ErrorCode::Success;
    }
};

/**
 * @brief Helper function to create an InvalidArgument error.
 */
inline constexpr Error InvalidArgumentError() {
    return Error{ErrorCode::InvalidArgument};
}

/**
 * @brief Helper function to create a DeviceNotSupported error.
 */
inline constexpr Error DeviceNotSupportedError() {
    return Error{ErrorCode::DeviceNotSupported};
}


inline constexpr Error InvalidDatasetError() {
    return Error{ErrorCode::InvalidDataset};
}

inline constexpr Error DatasetLoadFailedError() {
    return Error{ErrorCode::DatasetLoadFailed};
}
} // namespace hahaha::common

#endif // HAHAHA_ERROR_H_053340FD56184B63A68D6CE6E15C8175
