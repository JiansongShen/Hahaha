//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#ifndef HAHAHA_SIMDVEC_H_B878DE32A18743769333838EF65ACE5E
#define HAHAHA_SIMDVEC_H_B878DE32A18743769333838EF65ACE5E

#include <format>

#include "utils/common/helper_structs.h"
#include "utils/common/support_types.h"

namespace hahaha::backend {

/**
 * @brief SIMD vector type: T is element type, N is width in bits (128, 256, or 512).
 * Arch-specific specializations (e.g. x86 SSE/AVX) provide load/store and
 * arithmetic.
 */
template <typename T, std::size_t N> struct SimdVec {
    static_assert(
        utils::isLegalDataType<T>::value
        && std::format("the data type of simd vector is not legal: current is {},"
                       " but only supports: {}",
                       typeid(T).name(),
                       utils::getLegalDataTypeString())
               .data());

    static_assert(N == 128 || N == 256
                  || N == 512
                      && std::format("the size of simd vector is not supported: {}"
                                     "currently only 128, 256, 512 are supported",
                                     N)
                             .data());
};

} // namespace hahaha::backend

#endif // HAHAHA_SIMDVEC_H_B878DE32A18743769333838EF65ACE5E