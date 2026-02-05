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

#ifndef HAHAHA_DATASET_H_4277B28506244EA690CC3B3E7CEB44C5
#define HAHAHA_DATASET_H_4277B28506244EA690CC3B3E7CEB44C5
#include <memory>
#include <string>

#include "ml/dataset/DatasetInner.h"
#include "ml/dataset/DatasetInnerLoader.h"

namespace hahaha {

template <typename T>
class Dataset {

    using Interator = std::vector<T>;
public:
    static Dataset loadFromCSV(const std::string &filePath) {
        Dataset dataset;
        ml::DatasetInnerLoader().loadFromCSVTo<T>(filePath, dataset);
        return dataset;
    }

    Tensor<T> getItem(size_t idx) {
        return datasetInner_->getItem(idx);
    }

    void shuffleDataset() {
        datasetInner_->shuffleDataset();
    }



private:
    Dataset() {}

    std::shared_ptr<ml::DatasetInner<T>> datasetInner_;
};

} // namespace hahaha

#endif // HAHAHA_DATASET_H_4277B28506244EA690CC3B3E7CEB44C5
