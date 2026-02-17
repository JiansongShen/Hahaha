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

#ifndef HAHAHA_DATASETFORMAT_H_A1E176540A8E498C8D1301ECE727FEB5
#define HAHAHA_DATASETFORMAT_H_A1E176540A8E498C8D1301ECE727FEB5

#include <cstddef> // NOLINT

#include "common/definitions.h"

namespace hahaha::ml {
struct DatasetFormat {};

struct CSVDatasetFormat : DatasetFormat {
    size_t xFeatureNum;
    size_t yFeatureNum;

    using type = common::f32;

    CSVDatasetFormat() : xFeatureNum(0), yFeatureNum(0) {
    }
    CSVDatasetFormat(const size_t xFeatureNum, const size_t yFeatureNum)
        : xFeatureNum(xFeatureNum), yFeatureNum(yFeatureNum) {
    }
};

} // namespace hahaha::ml

#endif // HAHAHA_DATASETFORMAT_H_A1E176540A8E498C8D1301ECE727FEB5
