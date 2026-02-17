// Copyright (c) 2025-2026 Contributors of Hahaha
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

#ifndef HAHAHA_DATASET_BLANK_HANDER_H_54EDDFBDC36F4A898CB04420AB2B934E
#define HAHAHA_DATASET_BLANK_HANDER_H_54EDDFBDC36F4A898CB04420AB2B934E
#include <string>
#include <utility>
#include <vector>

#include "DatasetHandleBlankStrategy.h"

namespace hahaha::ml {
template <DatasetHandleBlankStrategy strategy>
struct DatasetBlankHandler {
    static bool handleBlankValue(std::vector<std::string> &vec) {
        std::unreachable();
    }
};

template <>
struct DatasetBlankHandler<DatasetHandleBlankStrategy::JumpOne> {
    static bool handleBlankValue(std::vector<std::string> &vec) {
        std::unreachable();
    }
};

template <>
struct DatasetBlankHandler<DatasetHandleBlankStrategy::SetNan> {
    static bool handleBlankValue(std::vector<std::string> &vec) {
        std::unreachable();
    }
};


}

#endif // HAHAHA_DATASET_BLANK_HANDER_H_54EDDFBDC36F4A898CB04420AB2B934E
