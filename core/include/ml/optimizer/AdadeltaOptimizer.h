//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  jiansongshen (jason.shen111@outlook.com ) (https://github.com/jiansongshen )
//

#ifndef HAHAHA_ADADELTAOPTIMIZER_H_525358001A394D618AF4F91698E27EB7
#define HAHAHA_ADADELTAOPTIMIZER_H_525358001A394D618AF4F91698E27EB7

#include "Optimizer.h"

namespace hahaha::ml {
template <typename T> class AdadeltaOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamOptimizer just supports float values");

  public:
    static constexpr T DefaultDecayRate = T(0.9);
    static constexpr T DefaultEpsilon = T(1e-6);

    AdadeltaOptimizer(
        std::vector<std::shared_ptr<compute::ComputeNode<T>>> parameters,
        T decayRate = DefaultDecayRate,
        T epsilon = DefaultEpsilon)
        : Optimizer<T>(std::move(parameters), 0), decayRate_(decayRate),
          epsilon_(epsilon) {
        for (const auto& param : this->getParameters()) {
            if (param) {
                gradSquareSum_.push_back(param->getData()->zeros());
                realUpdateValueSum_.push_back(param->getData()->zeros());
            }
        }
    }

    [[nodiscard]] T getDecayRate() const {
        return decayRate_;
    }

    [[nodiscard]] T getEpsilon() const {
        return epsilon_;
    }

    void step() override {
        for (size_t i = 0; i < this->getParameters().size(); ++i) {
            auto param = this->getParameters()[i];

            if (!param || !param->getRequiresGrad()) {
                continue;
            }

            auto grad = param->getGrad();
            if (!grad) {
                continue;
            }

            // Get references to accumulators
            auto& gradSquareSumData = gradSquareSum_[i];
            auto& realUpdateValueSumData = realUpdateValueSum_[i];

            // 1. E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g^2
            auto gradSquare = grad->clone();
            gradSquare.squareInPlace();

            gradSquareSumData *= getDecayRate();
            gradSquareSumData.axpy(T(1) - getDecayRate(), gradSquare);

            // 2. RMS[g]_t = sqrt(E[g^2]_t + epsilon)
            // Create temporary copy for calculation so we don't modify state
            auto rmsGrad = gradSquareSumData.clone();
            rmsGrad += getEpsilon();
            rmsGrad.sqrtInPlace();

            // 3. RMS[dx]_{t-1} = sqrt(E[dx^2]_{t-1} + epsilon)
            auto rmsX = realUpdateValueSumData.clone();
            rmsX += getEpsilon();
            rmsX.sqrtInPlace();

            // 4. delta_x = - (RMS[dx]_{t-1} / RMS[g]_t) * g
            auto deltaX = rmsX.clone();
            deltaX *= T(-1);
            deltaX /= rmsGrad;
            deltaX *= *grad;

            // 5. param = param + delta_x
            param->getData()->axpy(T(1), deltaX);

            // 6. E[dx^2]_t = rho * E[dx^2]_{t-1} + (1-rho) * delta_x^2
            realUpdateValueSumData *= getDecayRate();
            deltaX.squareInPlace();
            realUpdateValueSumData.axpy(T(1) - getDecayRate(), deltaX);
        }
    }

    void addParameter(std::shared_ptr<compute::ComputeNode<T>> param) override {
        if (param) {
            gradSquareSum_.push_back(param->getData()->zeros());
            realUpdateValueSum_.push_back(param->getData()->zeros());
        }
        Optimizer<T>::addParameter(std::move(param));
    }

  private:
    std::vector<math::TensorWrapper<T>> gradSquareSum_;
    std::vector<math::TensorWrapper<T>> realUpdateValueSum_;

    T decayRate_;
    T epsilon_;
};
} // namespace hahaha::ml

#endif // HAHAHA_ADADELTAOPTIMIZER_H_525358001A394D618AF4F91698E27EB7
