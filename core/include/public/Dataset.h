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

#ifndef HAHAHA_DATASET_H_4277B28506244EA690CC3B3E7CEB44C5
#define HAHAHA_DATASET_H_4277B28506244EA690CC3B3E7CEB44C5

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Tensor.h"
#include "math/TensorWrapper.h"
#include "ml/dataset/DatasetInner.h"
#include "ml/dataset/DatasetInnerLoader.h"

namespace hahaha {

/**
 * @brief Public-facing dataset wrapper.
 *
 * Stores samples internally as a @ref ml::DatasetInner (which uses raw
 * @ref math::TensorWrapper storage). The public API surfaces rows as
 * @ref Tensor<T> so user code does not need to know about TensorWrapper.
 */
template <typename T>
class Dataset {
public:
    /**
     * @brief Load a dataset from a CSV file.
     * @param filePath Path to the CSV file.
     * @return Dataset containing the loaded samples.
     */
    static Dataset loadFromCSV(const std::string& filePath) {
        Dataset ds;
        ml::DatasetInnerLoader().loadFromCSVTo<T>(filePath, ds.inner_);
        return ds;
    }

    /**
     * @brief Number of samples.
     */
    [[nodiscard]] size_t size() const {
        return inner_.size();
    }

    /**
     * @brief Return sample at logical index @p idx as a Tensor view.
     *
     * The logical index is remapped through the shuffle permutation, so the
     * result reflects any prior call to shuffle().
     */
    Tensor<T> getItem(size_t idx) {
        // DatasetInner::getItem returns a TensorWrapper view.  Wrap it in a
        // Tensor so the public API stays purely in terms of Tensor<T>.
        auto view = inner_.getItem(idx);
        return Tensor<T>(
            std::make_shared<math::TensorWrapper<T>>(std::move(view)));
    }

    /** @brief Column (feature) names. */
    [[nodiscard]] std::vector<std::string> getFeatures() const {
        return inner_.getFeatures();
    }

    /** @brief Label names. */
    [[nodiscard]] std::vector<std::string> getLabels() const {
        return inner_.getLabels();
    }

    /** @brief Dataset name (set to the file path by the loader). */
    [[nodiscard]] std::string getDatasetName() const {
        return inner_.getDatasetName();
    }

    /**
     * @brief Randomly permute the iteration order (non-deterministic seed).
     */
    void shuffle() {
        inner_.shuffle();
    }

    /**
     * @brief Shuffle with an explicit seed for reproducible runs.
     */
    void shuffle(unsigned seed) {
        inner_.shuffle(seed);
    }

private:
    Dataset() = default;

    ml::DatasetInner<T> inner_;
};

} // namespace hahaha

#endif // HAHAHA_DATASET_H_4277B28506244EA690CC3B3E7CEB44C5
