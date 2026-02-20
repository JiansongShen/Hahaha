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
// jiansongshen (jason.shen111@outlook.com) (https://github.com/Jiansongshen)
//

#ifndef HAHAHA_SLICESETTING_H_8E54B01CAE494949AF7FCD4048DD572D
#define HAHAHA_SLICESETTING_H_8E54B01CAE494949AF7FCD4048DD572D

#include <algorithm>
#include <cstddef>
#include <vector>

struct SliceInnerSetting {
    using size_type = std::size_t;

    size_type start;
    size_type end;
    size_type step;
};

struct SliceSetting {
    using size_type = std::size_t;
    // map the slice setting to the axis
    using setting = std::pair<size_type, SliceInnerSetting>;

    std::vector<setting> settings;

    SliceSetting(std::initializer_list<setting> settingsInput) {
        settings = settingsInput;
    }

    // from small to big, sort the axis
    void sortAxis() {
        std::sort(settings.begin(),
                  settings.end(),
                  [](const setting& a, const setting& b) -> bool {
                      return a.first < b.first;
                  });
    }

    [[nodiscard]] bool verifyLegal(const size_type tensorDimNum) const {
        if (settings.empty())
            return true;
        return settings.back().first < tensorDimNum;
    }
};

#endif // HAHAHA_SLICESETTING_H_8E54B01CAE494949AF7FCD4048DD572D
