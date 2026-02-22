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

#ifndef HAHAHA_LOSS_H_B47FE5F8D49547D19923C1D379D9C24C
#define HAHAHA_LOSS_H_B47FE5F8D49547D19923C1D379D9C24C
#include "public/Tensor.h"

namespace hahaha::ml {

/**
 * @brief Base class for all loss functions.
 *
 * Loss functions measure the difference between predicted and actual values.
 * This base class provides the interface that all loss functions must
 * implement.
 *
 * @tparam T The numeric type.
 */
template <typename T> class Loss {
  public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Loss() = default;

    /**
     * @brief Compute the loss between true and predicted values.
     * @param yTrue The true (target) values.
     * @param yPredict The predicted values.
     * @return Tensor<T> The computed loss value.
     */
    virtual Tensor<T> computeLoss(Tensor<T> /*yTrue*/, Tensor<T> /*yPredict*/) {
        return Tensor<T>(T(0));
    }

    virtual Tensor<T> operator()(Tensor<T> /* yTrue */, Tensor<T> /* yPreedict */);

    virtual Tensor<T> predict(Tensor<T> x);
};

} // namespace hahaha::ml

#endif // HAHAHA_LOSS_H_B47FE5F8D49547D19923C1D379D9C24C
