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

#ifndef HAHAHA_SUPPORT_TYPES_H_5C36CB79B1EA4A78A692C4B0986B82E9
#define HAHAHA_SUPPORT_TYPES_H_5C36CB79B1EA4A78A692C4B0986B82E9
#include <string_view>


namespace hahaha::utils {
 constexpr std::string_view getLegalDataTypeString() {
    return std::string_view{
        "Supported data types:\n"
        "  Built-in:\n"
        "    - float, double\n"
        "    - short, int, long, long long\n"
        "    - unsigned int, unsigned long, unsigned long long\n"
        "\n"
        "  Aliases (hahaha::common):\n"
        "    - f32  -> float       (32-bit floating point)\n"
        "    - f64  -> double      (64-bit floating point)\n"
        "    - i8   -> int8_t      (8-bit signed)\n"
        "    - u8   -> uint8_t     (8-bit unsigned)\n"
        "    - i16  -> int16_t     (16-bit signed)\n"
        "    - u16  -> uint16_t    (16-bit unsigned)\n"
        "    - i32  -> int32_t     (32-bit signed)\n"
        "    - u32  -> uint32_t    (32-bit unsigned)\n"
        "    - i64  -> int64_t     (64-bit signed)\n"
        "    - u64  -> uint64_t    (64-bit unsigned)\n"
        "    - bool8 -> uint8_t    (8-bit boolean storage)\n"
    };
}
}

#endif //HAHAHA_SUPPORT_TYPES_H_5C36CB79B1EA4A78A692C4B0986B82E9