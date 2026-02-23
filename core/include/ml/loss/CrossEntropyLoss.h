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

#ifndef CROSS_ENTROPY_LOSS_F9A3B2C1_8D4E_5F7G_6H9I_3J2K1L4M5N6O
#define CROSS_ENTROPY_LOSS_F9A3B2C1_8D4E_5F7G_6H9I_3J2K1L4M5N6O

// Standard library includes
#include <cmath>

// Project includes
#include <memory>

#include "Loss.h"
#include "math/TensorWrapper.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

/**
 * @brief Cross Entropy loss function.
 *
 * Cross Entropy loss is commonly used for classification problems.
 * Formula: CE = -sum(yTrue * log(yPredict + epsilon))
 * where epsilon is a small value to prevent log(0).
 *
 * @tparam T The numeric type.
 */
template <typename T> class CrossEntropyLoss : public Loss<T> {
  private:
    static constexpr T epsilon = static_cast<T>(1e-8);

  public:
    /**
     * @brief Compute the Cross Entropy loss between true and predicted values.
     * @param yTrue The true (target) labels (one-hot encoded).
     * @param yPredict The predicted probabilities.
     * @return std::shared_ptr<ComputeNode<T>> The computed Cross Entropy loss value.
     */
    std::shared_ptr<ComputeNode<T>>
    computeLoss(std::shared_ptr<ComputeNode<T>> yTrue,
                std::shared_ptr<ComputeNode<T>> yPredict) override {
        // Get data from ComputeNodes
        auto yTrueData = yTrue->getData();
        auto yPredictData = yPredict->getData();

        // Add epsilon to prevent log(0)
        auto safePredict = yPredictData->clone();
        *safePredict = *safePredict + epsilon;
        safePredict->logInPlace();

        // Compute cross entropy: -sum(yTrue * log(yPredict))
        auto product = yTrueData->multiply(*safePredict);
        auto sumValue = -product.sum();

        // Create a new ComputeNode with the sum value
        auto sumData = std::make_shared<math::TensorWrapper<T>>(sumValue);
        return std::make_shared<ComputeNode<T>>(sumData);
    }
};

/**
 * @brief Convenience function to compute Cross Entropy loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) labels (one-hot encoded).
 * @param yPredict The predicted probabilities.
 * @return std::shared_ptr<ComputeNode<T>> The computed Cross Entropy loss value.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
computeCrossEntropyLoss(std::shared_ptr<ComputeNode<T>> yTrue,
                        std::shared_ptr<ComputeNode<T>> yPredict) {
    static CrossEntropyLoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // CROSS_ENTROPY_LOSS_F9A3B2C1_8D4E_5F7G_6H9I_3J2K1L4M5N6O