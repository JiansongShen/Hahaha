// Copyright (c) $todat.year-2026 Contributors of Hahaha
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

#ifndef HAHAHA_DATASETHANDLEBLANKSTRATEGY_H_F167E75C6E5D452A9A3414D452960B1E
#define HAHAHA_DATASETHANDLEBLANKSTRATEGY_H_F167E75C6E5D452A9A3414D452960B1E
#include <limits>

namespace hahaha::ml {
enum class DatasetHandleBlankStrategy {
    JumpOne,
    SetNan, // just for float dataset
    UseZero,
};

inline constexpr auto FloatNan = std::numeric_limits<float>::quiet_NaN();

static auto DefaultDatasetHandleBlankStrategy = DatasetHandleBlankStrategy::SetNan;

constexpr DatasetHandleBlankStrategy getDefaultDatasetHandleBlankStrategy()
{
    return DefaultDatasetHandleBlankStrategy;
}

inline void setDefaultDatasetHandleBlankStrategy(
    const DatasetHandleBlankStrategy datasetHandleBlankStrategy)
{
    DefaultDatasetHandleBlankStrategy = datasetHandleBlankStrategy;
}

}

#endif // HAHAHA_DATASETHANDLEBLANKSTRATEGY_H_F167E75C6E5D452A9A3414D452960B1E
