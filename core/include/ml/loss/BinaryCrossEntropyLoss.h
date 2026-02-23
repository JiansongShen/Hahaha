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

#ifndef BINARY_CROSS_ENTROPY_LOSS_A1B2C3D4_9E8F_7G6H_5I4J_3K2L1M0N9O8P
#define BINARY_CROSS_ENTROPY_LOSS_A1B2C3D4_9E8F_7G6H_5I4J_3K2L1M0N9O8P

// Standard library includes
#include <cmath>

// Project includes
#include "Loss.h"

namespace hahaha::ml {

/**
 * @brief Binary Cross Entropy loss function.
 *
 * Binary Cross Entropy loss is used for binary classification problems.
 * Formula: BCE = -mean(yTrue * log(yPredict + epsilon) + (1 - yTrue) * log(1 - yPredict + epsilon))
 * where epsilon is a small value to prevent log(0).
 *
 * @tparam T The numeric type.
 */
template <typename T> class BinaryCrossEntropyLoss : public Loss<T> {
  private:
    static constexpr T epsilon = static_cast<T>(1e-8);

  public:
    /**
     * @brief Compute the Binary Cross Entropy loss between true and predicted values.
     * @param yTrue The true (target) labels (0 or 1).
     * @param yPredict The predicted probabilities.
     * @return TensorWrapper<T> The computed Binary Cross Entropy loss value.
     */
    TensorWrapper<T> computeLoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
        // Add epsilon to prevent log(0)
        TensorWrapper<T> safePredict = yPredict + epsilon;
        TensorWrapper<T> safeOneMinusPredict = (TensorWrapper<T>(T(1)) - yPredict) + epsilon;
        
        // Compute log terms
        safePredict.logInPlace();
        safeOneMinusPredict.logInPlace();

        // Compute binary cross entropy
        TensorWrapper<T> term1 = yTrue * safePredict;
        TensorWrapper<T> term2 =
            (TensorWrapper<T>(T(1)) - yTrue) * safeOneMinusPredict;
        TensorWrapper<T> bce = -(term1 + term2);
        const auto totalSize = bce.getTotalSize();
        return TensorWrapper<T>(bce.sum() / static_cast<T>(totalSize));
    }
};

/**
 * @brief Convenience function to compute Binary Cross Entropy loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) labels (0 or 1).
 * @param yPredict The predicted probabilities.
 * @return TensorWrapper<T> The computed Binary Cross Entropy loss value.
 */
template <typename T>
TensorWrapper<T> computeBinaryCrossEntropyLoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
    static BinaryCrossEntropyLoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // BINARYCROSSENTROPYLOSS_ABF52498_887A_4E65_8354_F4F9E536C505
