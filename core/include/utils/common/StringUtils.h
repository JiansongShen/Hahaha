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
#include <string>
#include <vector>
#include <sstream>

namespace hahaha::utils {

class StringUtils {


public:
    static std::vector<std::string> split(const std::string& s,
                                          const char delimiter,
                                          const bool keepBlankValue = false) {
        std::vector<std::string> elems;

        size_t posBegin = 0;
        for (size_t i = 0; i < s.length(); i++) {
            if (s[i] == delimiter) {
                // like a ','[postBegin] ','[i];
                if (i - posBegin == 1 && keepBlankValue) {
                    elems.emplace_back("");
                    posBegin = i + 1;
                    continue;
                }

                const auto substr = s.substr(posBegin, i - posBegin);
                if (isBlank(substr) && keepBlankValue) {
                    elems.push_back(substr);
                    continue;
                }
                elems.push_back(substr);
                posBegin = i + 1;
            }
        }
        elems.push_back(s.substr(posBegin, s.length() - posBegin));

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


    template<typename T>
    static T to(const std::string& s) {
        std::istringstream iss(s);
        T value;
        iss >> value;
        return value;
    }
};

} // namespace hahaha::utils

#endif // HAHAHA_STRINGUTILS_H_6456D372529A4CF69DBCA69F0C736EBF
