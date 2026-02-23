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

#ifndef LOG_COSH_LOSS_D2E3F4G5_6H7I_8J9K_0L1M_2N3O4P5Q6R7S
#define LOG_COSH_LOSS_D2E3F4G5_6H7I_8J9K_0L1M_2N3O4P5Q6R7S

// Standard library includes
#include <cmath>

// Project includes
#include <memory>

#include "Loss.h"
#include "math/TensorWrapper.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

/**
 * @brief Log-Cosh loss function.
 *
 * Log-Cosh loss is a smooth approximation of the Huber loss.
 * Formula: log(cosh(yTrue - yPredict))
 * where cosh(x) = (exp(x) + exp(-x)) / 2
 *
 * @tparam T The numeric type.
 */
template <typename T> class LogCoshLoss : public Loss<T> {
  public:
    /**
     * @brief Compute the Log-Cosh loss between true and predicted values.
     * @param yTrue The true (target) values.
     * @param yPredict The predicted values.
     * @return std::shared_ptr<ComputeNode<T>> The computed Log-Cosh loss value.
     */
    std::shared_ptr<ComputeNode<T>>
    computeLoss(std::shared_ptr<ComputeNode<T>> yTrue,
                std::shared_ptr<ComputeNode<T>> yPredict) override {
        // Get data from ComputeNodes
        auto yTrueData = yTrue->getData();
        auto yPredictData = yPredict->getData();

        // Compute error
        auto error = yTrueData->subtract(*yPredictData);

        // Compute cosh(error) = (exp(error) + exp(-error)) / 2
        auto expError = error.clone();
        expError.expInPlace();
        auto negError = error.clone();
        negError.negateInPlace();
        negError.expInPlace();
        auto coshError = expError.add(negError);
        coshError = coshError * T(0.5);

        // Compute log(cosh(error))
        coshError.logInPlace();
        const auto totalSize = coshError.getTotalSize();
        auto meanValue = coshError.sum() / static_cast<T>(totalSize);

        // Create a new ComputeNode with the mean value
        auto meanData = std::make_shared<math::TensorWrapper<T>>(meanValue);
        return std::make_shared<ComputeNode<T>>(meanData);
    }
};

/**
 * @brief Convenience function to compute Log-Cosh loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) values.
 * @param yPredict The predicted values.
 * @return std::shared_ptr<ComputeNode<T>> The computed Log-Cosh loss value.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
computeLogCoshLoss(std::shared_ptr<ComputeNode<T>> yTrue,
                   std::shared_ptr<ComputeNode<T>> yPredict) {
    static LogCoshLoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // LOG_COSH_LOSS_D2E3F4G5_6H7I_8J9K_0L1M_2N3O4P5Q6R7S