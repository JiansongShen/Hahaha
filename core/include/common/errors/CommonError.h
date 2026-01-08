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

#ifndef HAHAHA_COMMON_ERROR_H_053340FD56184B63A68D6CE6E15C8175
#define HAHAHA_COMMON_ERROR_H_053340FD56184B63A68D6CE6E15C8175

#include "common/errors/Error.h"

namespace hahaha::common {

/**
 * @brief Common error type using CRTP pattern.
 */
class CommonError : public ErrorBase<CommonError, ErrorCode::BasicError> {
  public:
    static constexpr auto ErrorName = "CommonError";

    // Inherit base class methods
    using ErrorBase::getCode;
    using ErrorBase::getErrorName;
    using ErrorBase::getLocation;
    using ErrorBase::getMessage;

    CommonError() = default;

    explicit CommonError(
        std::string message,
        const std::source_location location = std::source_location::current())
        : ErrorBase<CommonError, ErrorCode::BasicError>(std::move(message),
                                                        location) {
    }
};

} // namespace hahaha::common

#endif // HAHAHA_COMMON_ERROR_H_053340FD56184B63A68D6CE6E15C8175
