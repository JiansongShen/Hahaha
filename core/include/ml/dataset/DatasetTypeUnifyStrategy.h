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

#ifndef HAHAHA_DATASETLOADSTRATEGY_H_AEC8589728AA4619B1F43366886909A4
#define HAHAHA_DATASETLOADSTRATEGY_H_AEC8589728AA4619B1F43366886909A4

namespace hahaha::ml {

enum class DatasetTypeUnifyStrategy {
    AllFloat32 = 0,
    AllFloat64,
    AllFloat16, ///< not supported yet, TODO(jiansongshen): Maybe support it
                ///< latter
    UseLargestRange,
};

static auto defaultDatasetTypeUnifyStrategy =
    DatasetTypeUnifyStrategy::AllFloat32;

inline DatasetTypeUnifyStrategy getDefaultDatasetTypeUnifyStrategy() {
    return defaultDatasetTypeUnifyStrategy;
}

inline void setDefaultDatasetTypeUnifyStrategy(
    const DatasetTypeUnifyStrategy datasetTypeUnifyStrategy) {
    defaultDatasetTypeUnifyStrategy = datasetTypeUnifyStrategy;
}

} // namespace hahaha::ml

#endif // HAHAHA_DATASETLOADSTRATEGY_H_AEC8589728AA4619B1F43366886909A4
