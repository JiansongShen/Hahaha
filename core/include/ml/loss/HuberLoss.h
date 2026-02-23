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

#ifndef HUBER_LOSS_C7D8E9F0_1A2B_3C4D_5E6F_7G8H9I0J1K2L
#define HUBER_LOSS_C7D8E9F0_1A2B_3C4D_5E6F_7G8H9I0J1K2L

// Standard library includes
#include <cmath>

// Project includes
#include "Loss.h"

namespace hahaha::ml {

/**
 * @brief Huber loss function.
 *
 * Huber loss is less sensitive to outliers than MSE. It combines MSE and MAE.
 * Formula: 
 *   if |error| <= delta: 0.5 * error^2
 *   else: delta * (|error| - 0.5 * delta)
 *
 * @tparam T The numeric type.
 */
template <typename T> class HuberLoss : public Loss<T> {
  private:
    T delta_;

  public:
    /**
     * @brief Constructor with delta parameter.
     * @param delta The threshold parameter that defines the transition point between quadratic and linear loss.
     */
    explicit HuberLoss(T delta = static_cast<T>(1.0)) : delta_(delta) {}

    /**
     * @brief Compute the Huber loss between true and predicted values.
     * @param yTrue The true (target) values.
     * @param yPredict The predicted values.
     * @return TensorWrapper<T> The computed Huber loss value.
     */
    TensorWrapper<T> computeLoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
        TensorWrapper<T> error = yTrue - yPredict;
        TensorWrapper<T> absError = error.absInPlace();
        
        // Create mask for quadratic region (|error| <= delta)
        TensorWrapper<T> quadraticMask = (absError <= delta_).template cast<T>();
        TensorWrapper<T> linearMask = TensorWrapper<T>(T(1)) - quadraticMask;
        
        // Quadratic part: 0.5 * error^2
        TensorWrapper<T> quadraticPart = T(0.5) * (error * error);
        
        // Linear part: delta * (|error| - 0.5 * delta)
        TensorWrapper<T> linearPart = delta_ * (absError - T(0.5) * delta_);
        
        // Combine both parts
        TensorWrapper<T> huberLoss = quadraticMask * quadraticPart + linearMask * linearPart;
        return TensorWrapper<T>(huberLoss.sum());
    }
};

/**
 * @brief Convenience function to compute Huber loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) values.
 * @param yPredict The predicted values.
 * @param delta The threshold parameter (default: 1.0).
 * @return TensorWrapper<T> The computed Huber loss value.
 */
template <typename T>
TensorWrapper<T> computeHuberLoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict, T delta = static_cast<T>(1.0)) {
    HuberLoss<T> loss(delta);
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // HUBER_LOSS_C7D8E9F0_1A2B_3C4D_5E6F_7G8H9I0J1K2L