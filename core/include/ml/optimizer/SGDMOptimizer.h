// Copyright (c) 2025 - 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
// JiansongShen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#ifndef HAHAHA_SGDMOPTIMIZER_H_E57DB5EDFD0E4CC4914AB64FBF0C0859
#define HAHAHA_SGDMOPTIMIZER_H_E57DB5EDFD0E4CC4914AB64FBF0C0859
#include <vector>

#include "Optimizer.h"

namespace hahaha::ml {

/**
 * @brief SGD with Momentum Optimizer.
 *
 * Update rule:
 * v(t) = mu * v(t-1) + (1 - mu) * g(t)
 * theta(t) = theta(t-1) - eta * v(t)
 */
template <typename T> class SGDMOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamOptimizer just supports float values");

  public:
    /**
     * @brief Construct a new SGDMOptimizer.
     * @param parameters List of compute nodes to optimize.
     * @param learningRate Learning rate.
     * @param momentumCoefficient Momentum factor (default 0.9).
     */
    SGDMOptimizer(std::vector<std::shared_ptr<compute::ComputeNode<T>>> parameters,
                  T learningRate,
                  T momentumCoefficient = 0.9)
        : Optimizer<T>(std::move(parameters), learningRate),
          momentumCoefficient_(momentumCoefficient) {
    }

    /**
     * @brief Adds a parameter to the optimizer.
     * @param param The compute node to be optimized.
     */
    void addParameter(std::shared_ptr<compute::ComputeNode<T>> param) override {
        if (trainPrepared_) {
            momentum_.push_back(param->getData()->zeros());
        }
        Optimizer<T>::addParameter(std::move(param));
    }

    /**
     * @brief Performs parameter updates using SGDM logic.
     */
    void step() override {
        preTrainIfNeed();
        T learningRate = this->getLearningRate();
        auto& params = this->getParameters();

        for (size_t i = 0; i < params.size(); ++i) {
            auto param = params[i];

            if (!param || !param->getRequiresGrad()) {
                continue;
            }

            auto grad = param->getGrad();
            if (!grad) {
                continue;
            }

            auto& momentum = momentum_[i];

            // Update momentum: v = mu * v + (1 - mu) * grad
            // Use in-place operations on TensorWrapper to avoid building graph
            momentum *= momentumCoefficient_;
            momentum.axpy(T(1) - momentumCoefficient_, *grad);

            // Update parameter: theta = theta - lr * momentum
            param->getData()->axpy(-learningRate, momentum);
        }
    }

  private:
    void preTrainIfNeed() {
        if (trainPrepared_) {
            return;
        }
        auto paramSize = this->getParameters().size();
        momentum_.clear();
        momentum_.reserve(paramSize);
        for (const auto& param : this->getParameters()) {
            if (param) {
                momentum_.push_back(param->getData()->zeros());
            }
        }
        trainPrepared_ = true;
    }

    std::vector<math::TensorWrapper<T>> momentum_;
    T momentumCoefficient_;
    bool trainPrepared_ = false;
};
} // namespace hahaha::ml

#endif // HAHAHA_SGDMOPTIMIZER_H_E57DB5EDFD0E4CC4914AB64FBF0C0859
