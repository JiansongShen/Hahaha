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
#include "Loss.h"

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
     * @return TensorWrapper<T> The computed MAE loss value.
     */
    TensorWrapper<T> computeLoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
        // NOTE: TensorWrapper::sum() currently returns a scalar value (T), not a
        // TensorWrapper. Wrap it back into a scalar TensorWrapper.

        // The absolute value is computed in-place to avoid unnecessary copying. The
        // original error tensor is modified, but since it's a temporary object
        // created by the subtraction, it won't affect
        auto absError = yTrue - yPredict;
        absError.absInPlace();
        const auto totalSize = absError.getTotalSize();
        return TensorWrapper<T>(absError.sum() / static_cast<T>(totalSize));
    }
};

/**
 * @brief Convenience function to compute MAE loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) values.
 * @param yPredict The predicted values.
 * @return TensorWrapper<T> The computed MAE loss value.
 */
template <typename T>
TensorWrapper<T> computeMAELoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
    static MAELoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // MAELOSS_E8F2A1B9_7C3D_4F6E_9A8B_2D5C1E3F7A9B