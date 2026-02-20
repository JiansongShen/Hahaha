// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com)
//

#ifndef HAHAHA_STRINGUTILS_H_6456D372529A4CF69DBCA69F0C736EBF
#define HAHAHA_STRINGUTILS_H_6456D372529A4CF69DBCA69F0C736EBF
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace hahaha::utils {

class StringUtils {

  public:
    /**
     * @brief Split @p s on @p delimiter.
     *
     * @param keepBlankValue  When true, empty/blank segments (e.g. consecutive
     *        delimiters "a,,b") are kept in the result as empty strings.
     *        When false, blank segments are silently dropped.
     *
     */
    static std::vector<std::string> split(const std::string& s,
                                          const char delimiter,
                                          const bool keepBlankValue = false) {
        std::vector<std::string> elems;

        size_t posBegin = 0;
        for (size_t i = 0; i < s.length(); i++) {
            if (s[i] == delimiter) {
                const auto substr = s.substr(posBegin, i - posBegin);
                if (!isBlank(substr) || keepBlankValue) {
                    elems.push_back(substr);
                }
                posBegin = i + 1; // always advance past the delimiter
            }
        }
        // Last segment (after the final delimiter, or the whole string if none)
        const auto last = s.substr(posBegin);
        if (!isBlank(last) || keepBlankValue) {
            elems.push_back(last);
        }

        return elems;
    }

    static bool isBlank(const std::string& s) {
        for (const auto c : s) {
            if (c != ' ') {
                return false;
            }
        }
        return true;
    }

    static std::string trimSideBlank(const std::string& s) {
        size_t start = 0;
        size_t end = s.length() - 1;
        while (start < end && ' ' == s[start]) {
            start++;
        }
        while (end > start && ' ' == s[end]) {
            end--;
        }
        return s.substr(start, end - start + 1);
    }

    /**
     * @brief Parse @p s into type @p T.
     * @return The parsed value, or std::nullopt if parsing failed.
     */
    template <typename T> static std::optional<T> to(const std::string& s) {
        std::istringstream iss(s);
        T value;
        if (!(iss >> value)) {
            return std::nullopt;
        }
        // Ensure no trailing non-whitespace garbage (e.g. "1.0abc" for float)
        char leftover{};
        if (iss >> leftover) {
            return std::nullopt;
        }
        return value;
    }
};

} // namespace hahaha::utils

#endif // HAHAHA_STRINGUTILS_H_6456D372529A4CF69DBCA69F0C736EBF
