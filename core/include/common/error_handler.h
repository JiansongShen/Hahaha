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
// ReSharper disable once CppUnusedIncludeDirective
#include "common/errors/CommonError.h"
#include "common/errors/Error.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "common/errors/ErrorCode.h"

namespace hahaha::common {

// Convenience functions for creating expected/unexpected values
template <typename T, typename E> auto ok(T val) {
    return std::expected<T, E>(std::move(val));
}

template <typename E> auto err(E err) {
    return std::unexpected(std::move(err));
}

} // namespace hahaha::common

#endif // HAHAHA_ERROR_HANDLER_H_053340FD56184B63A68D6CE6E15C8175
