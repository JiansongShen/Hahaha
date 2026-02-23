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

#ifndef HAHAHA_PUBLIC_LOSS_H_3E2F1A9B5C8D4E7F6A1B2C3D4E5F6789
#define HAHAHA_PUBLIC_LOSS_H_3E2F1A9B5C8D4E7F6A1B2C3D4E5F6789

#include <memory>

// Inner implementations (not part of the public contract).
#include "ml/loss/Loss.h"
#include "ml/loss/MSELoss.h"

namespace hahaha {

using hahaha::math::TensorWrapper;

/**
 * @brief Public handle for loss functions (pimpl).
 *
 * Holds a pointer to an inner @ref ml::Loss<T> implementation.
 * Concrete loss classes (e.g. @ref MSELoss) construct the appropriate
 * inner object and pass it to this base through the protected constructor.
 *
 * @tparam T Numeric type (float, double, …).
 */
template <typename T> class Loss {
  public:
    virtual ~Loss() = default;

    /**
     * @brief Compute the loss between true values and predictions.
     * @param yTrue  Ground-truth tensorWrapper.
     * @param yPred  Predicted tensorWrapper.
     * @return Scalar @ref TensorWrapper<T> holding the loss value.
     */
    TensorWrapper<T> computeLoss(TensorWrapper<T> yTrue,
                                 TensorWrapper<T> yPred) const {
        return impl_->computeLoss(std::move(yTrue), std::move(yPred));
    }

  protected:
    explicit Loss(std::shared_ptr<ml::Loss<T>> impl) : impl_(std::move(impl)) {
    }

  private:
    std::shared_ptr<ml::Loss<T>> impl_;
};

/**
 * @brief Mean Squared Error loss.
 *
 * Computes: loss = sum((yTrue - yPred)^2)
 *
 * @tparam T Numeric type.
 */
template <typename T> class MSELoss : public Loss<T> {
  public:
    MSELoss() : Loss<T>(std::make_shared<ml::MSELoss<T>>()) {
    }
};

/**
 * @brief Free-function convenience wrapper around @ref MSELoss.
 */
template <typename T>
TensorWrapper<T> computeMSELoss(TensorWrapper<T> yTrue, TensorWrapper<T> yPred) {
    return MSELoss<T>().computeLoss(std::move(yTrue), std::move(yPred));
}

} // namespace hahaha

#endif // HAHAHA_PUBLIC_LOSS_H_3E2F1A9B5C8D4E7F6A1B2C3D4E5F6789
