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

#ifndef MSELOSS_CD27251A_6A1C_4A58_89BA_192C683604E8
#define MSELOSS_CD27251A_6A1C_4A58_89BA_192C683604E8

#include "Loss.h"

namespace hahaha::ml {

/**
 * @brief Mean Squared Error (MSE) loss function.
 *
 * MSE loss computes the mean of the squared differences between predicted
 * and true values. Formula: MSE = mean((yTrue - yPredict)^2)
 *
 * @tparam T The numeric type.
 */
template <typename T> class MSELoss : public Loss<T> {
  public:
    /**
     * @brief Compute the MSE loss between true and predicted values.
     * @param yTrue The true (target) values.
     * @param yPredict The predicted values.
     * @return TensorWrapper<T> The computed MSE loss value.
     */
    TensorWrapper<T> computeLoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
        // NOTE: TensorWrapper::sum() currently returns a scalar value (T), not a
        // TensorWrapper. Wrap it back into a scalar TensorWrapper.
        auto diff = yTrue - yPredict;
        diff.squareInPlace();
        const auto totalSize = diff.getTotalSize();
        return TensorWrapper<T>(diff.sum() / static_cast<T>(totalSize));
    }
};

/**
 * @brief Convenience function to compute MSE loss.
 * @tparam T The numeric type.
 * @param yTrue The true (target) values.
 * @param yPredict The predicted values.
 * @return TensorWrapper<T> The computed MSE loss value.
 */
template <typename T>
TensorWrapper<T> computeMSELoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPredict) {
    static MSELoss<T> loss;
    return loss.computeLoss(yTrue, yPredict);
}

} // namespace hahaha::ml

#endif // MSELOSS_CD27251A_6A1C_4A58_89BA_192C683604E8
