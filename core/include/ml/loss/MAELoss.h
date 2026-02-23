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
//  Contributors:
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef MAELOSS_E8F2A1B9_7C3D_4F6E_9A8B_2D5C1E3F7A9B
#define MAELOSS_E8F2A1B9_7C3D_4F6E_9A8B_2D5C1E3F7A9B

// Project includes
#include <memory>

#include "Loss.h"
#include "math/TensorWrapper.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

/**
 * @brief Mean Absolute Error (MAE) loss function.
 *
 * MAE loss computes the mean of the absolute differences between predicted
 * and true values. Formula: MAE = mean(|yTrue - yPredict|)
 *
 * @tparam T The numeric type.
 */
template <typename T> class MAELoss : public Loss<T> {
  public:
    /**
     * @brief Compute the MAE loss between true and predicted values.
     * @param yTrue The true (target) values.
     * @param yPredict The predicted values.
     * @return std::shared_ptr<ComputeNode<T>> The computed MAE loss value.
     */
    std::shared_ptr<ComputeNode<T>>
    computeLoss(std::shared_ptr<ComputeNode<T>> yTrue,
                std::shared_ptr<ComputeNode<T>> yPredict) override {
        // Get data from ComputeNodes
        auto yTrueData = yTrue->getData();
        auto yPredictData = yPredict->getData();

        // Compute MAE loss: mean(|yTrue - yPredict|)
        auto diff = yTrueData->subtract(*yPredictData);
        diff.absInPlace();
        const auto totalSize = diff.getTotalSize();
        auto meanValue = diff.sum() / static_cast<T>(totalSize);

        // Create a new ComputeNode with the mean value
        auto meanData = std::make_shared<math::TensorWrapper<T>>(meanValue);
        return std::make_shared<ComputeNode<T>>(meanData);
    }
};

/**
 * @brief Convenience function to compute MAE loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) values.
 * @param yPredict The predicted values.
 * @return std::shared_ptr<ComputeNode<T>> The computed MAE loss value.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
computeMAELoss(std::shared_ptr<ComputeNode<T>> yTrue,
               std::shared_ptr<ComputeNode<T>> yPredict) {
    static MAELoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // MAELOSS_E8F2A1B9_7C3D_4F6E_9A8B_2D5C1E3F7A9B