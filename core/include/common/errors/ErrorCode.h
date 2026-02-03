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

#ifndef HAHAHA_ERROR_CODE_H_053340FD56184B63A68D6CE6E15C8175
#define HAHAHA_ERROR_CODE_H_053340FD56184B63A68D6CE6E15C8175

#include <cstddef>

namespace hahaha::common {

/**
 * @brief Enumeration of error codes used throughout the project.
 */
enum class ErrorCode : std::size_t {
    Success = 0,        ///< Operation completed successfully.
    InvalidArgument,    ///< Invalid dimensions or parameters provided.
    DeviceNotSupported, ///< The requested device is not supported for this
                        ///< operation.
    Overflow,           ///< Numerical overflow occurred.
    InternalError,      ///< An internal logic error occurred.

    InvalidDataset,     ///< Invalid dataset provided.
    DatasetLoadFailed,  ///< Failed to load dataset.

    Count               ///< Number of error codes (for bounds checking).
};

/**
 * @brief Static array of error messages corresponding to ErrorCode.
 */
constexpr const char* ErrorMessages[] = {
    "Success",
    "Invalid argument (e.g., shape mismatch)",
    "Device unsupported",
    "Numerical overflow",
    "Internal error",
    "Invalid dataset",
    "Dataset load failed",
};

} // namespace hahaha::common

#endif // HAHAHA_ERROR_CODE_H_053340FD56184B63A68D6CE6E15C8175
