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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#ifndef HAHAHA_ADADELTAOPTIMIZER_H_525358001A394D618AF4F91698E27EB7
#define HAHAHA_ADADELTAOPTIMIZER_H_525358001A394D618AF4F91698E27EB7

#include "Optimizer.h"
#include "math/TensorWrapper.h"

namespace hahaha::ml {
template<typename T>
class AdadeltaOptimizer : public Optimizer<T>{

public:
    static constexpr T DefaultDecayRate = T(0.9);
    static constexpr T DefaultEpsilon = T(1e-6);

    AdadeltaOptimizer(std::vector<Tensor<T>> parameters,
        T decayRate = DefaultDecayRate,
        T epsilon = DefaultEpsilon) :
    Optimizer<T>(std::move(parameters), 0),
    decayRate_(decayRate),
    epsilon_(epsilon) {
        for (const auto& param : this->getParameters()) {
            gradSquareSum_.push_back(param.zeros());
            realUpdateValueSum_.push_back(param.zeros());
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
            auto& param = this->getParameters()[i];
            
            if (!param.getRequiresGrad()) {
                continue;
            }

            auto grad = param.grad();
            if (grad.isEmpty()) {
                continue;
            }

            // Get references to accumulators
            auto& gradSquareSumData = gradSquareSum_[i];
            auto& realUpdateValueSumData = realUpdateValueSum_[i];

            // 1. E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g^2
            auto gradSquare = grad.square();
            
            gradSquareSumData *= getDecayRate();
            gradSquareSumData.axpy(T(1) - getDecayRate(), gradSquare);

            // 2. RMS[g]_t = sqrt(E[g^2]_t + epsilon)
            // Create temporary copy for calculation so we don't modify state
            auto rmsGrad = gradSquareSumData + getEpsilon();
            rmsGrad.sqrtInPlace(); 

            // 3. RMS[dx]_{t-1} = sqrt(E[dx^2]_{t-1} + epsilon)
            auto rmsX = realUpdateValueSumData + getEpsilon();
            rmsX.sqrtInPlace();
            
            // 4. delta_x = - (RMS[dx]_{t-1} / RMS[g]_t) * g
            auto deltaX = rmsX; // share node, but we will modify it
            // deltaX is modified in place, but rmsX is not used afterwards, so this is safe.
            // rmsX is a temporary tensor created in this scope.
            
            deltaX *= T(-1);
            deltaX /= rmsGrad;
            deltaX *= grad;

            // 5. param = param + delta_x
            param += deltaX;

            // 6. E[dx^2]_t = rho * E[dx^2]_{t-1} + (1-rho) * delta_x^2
            realUpdateValueSumData *= getDecayRate();
            deltaX.squareInPlace();
            realUpdateValueSumData.axpy(T(1) - getDecayRate(), deltaX);
        }
    }

    void addParameter(const Tensor<T>& param) override {
        Optimizer<T>::addParameter(param);
        gradSquareSum_.push_back(param.zeros());
        realUpdateValueSum_.push_back(param.zeros());
    }

private:
    std::vector<Tensor<T>> gradSquareSum_;
    std::vector<Tensor<T>> realUpdateValueSum_;

    T decayRate_;
    T epsilon_;
};
}

#endif //HAHAHA_ADADELTAOPTIMIZER_H_525358001A394D618AF4F91698E27EB7
