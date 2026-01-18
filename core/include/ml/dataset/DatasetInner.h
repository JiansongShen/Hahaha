//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef HAHAHA_DATASET_INNER_H_4B71BF801C8A477CABA1DCC84944450E
#define HAHAHA_DATASET_INNER_H_4B71BF801C8A477CABA1DCC84944450E
#include <utility>

#include "DatasetTypeUnifyStrategy.h"
#include "public/Tensor.h"

namespace hahaha::ml {
class DatasetInnerLoader;

template <typename T> class DatasetInner {

  public:
    DatasetInner(const std::vector<std::string>& features,
                 std::string datasetName,
                 const std::vector<std::string>& labels,
                 const DatasetTypeUnifyStrategy& datasetTypeUnifyStrategy,
                 Tensor<T> x,
                 Tensor<T> y)
        : datasetName_(std::move(datasetName)), features_(features),
          labels_(labels),
          typeUnifyStrategy_(getDefaultDatasetTypeUnifyStrategy()) {
    }

    [[nodiscard]] std::string getDatasetName() const {
        return datasetName_;
    }

    [[nodiscard]] std::vector<std::string> getLabels() const {
        return labels_;
    }

    [[nodiscard]] std::vector<std::string> getFeatures() const {
        return features_;
    }

    [[nodiscard]] DatasetTypeUnifyStrategy getDatasetTypeUnifyStrategy() const {
        return typeUnifyStrategy_;
    }

    Tensor<T> getX() const {
        return x;
    }

    Tensor<T> getY() const {
        return y;
    }

  private:
    std::string datasetName_;
    std::vector<std::string> features_;
    std::vector<std::string> labels_;

    DatasetTypeUnifyStrategy typeUnifyStrategy_;

    Tensor<T> x;
    Tensor<T> y;

    friend class DatasetInnerLoader;
};

} // namespace hahaha::ml

#endif // HAHAHA_DATASET_INNER_H_4B71BF801C8A477CABA1DCC84944450E
