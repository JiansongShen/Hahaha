// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com ) (https://github.com/jiansongshen )

#ifndef HAHAHA_ADAMOPTIMIZER_H
#define HAHAHA_ADAMOPTIMIZER_H

#include <cmath>

#include "Optimizer.h"
#include "math/TensorComputeFuns.h"
#include "utils/log/Logger.h"

namespace hahaha::ml {

using common::f32;
using common::f64;
using common::u64;

constexpr f64 DefaultBeta1 = 0.9;
constexpr f64 DefaultBeta2 = 0.999;
constexpr f64 DefaultEpsilon = 1e-8;

struct TrainRecord {};

/**
 *
 * @tparam T /
Initialize parameters θ
Initialize m = 0, v = 0, t = 0
While not converged:
    t = t + 1
    g = compute_gradient(θ)
    m = β1 * m + (1 - β1) * g
    v = β2 * v + (1 - β2) * (g ** 2)
    m_hat = m / (1 - β1 ** t)
    v_hat = v / (1 - β2 ** t)
    θ = θ - η * m_hat / (sqrt(v_hat) + ε)
*/
template <typename T> class AdamOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamOptimizer just supports float values");

  public:
    AdamOptimizer(const std::vector<Tensor<T>>& parameters,
                  const T& learningRate)
        : Optimizer<T>(parameters, learningRate) {
    }
    explicit AdamOptimizer(const Optimizer<T>& optimizer)
        : Optimizer<T>(optimizer) {
    }

    explicit AdamOptimizer(Optimizer<T>&& optimizer) : Optimizer<T>(optimizer) {
    }

    void addParameter(const Tensor<T>& param) override {
        if (trainPrepared) {
            parametersM_.push_back(param.getComputeNode()->getData()->zeros());
            parametersV_.push_back(param.getComputeNode()->getData()->zeros());
        }
        this->parameters_.push_back(param);
    }

    void step() override {
        preTrainIfNeed();
        ++turn;
        for (size_t i = 0; i < parametersM_.size(); ++i) {
            auto param = this->parameters_[i].getComputeNode();
            auto& paramM = this->parametersM_[i];
            auto& paramV = this->parametersV_[i];

            if (!param->getRequiresGrad()) {
                continue;
            }
            auto grad = param->getGrad();
            if (grad == nullptr) {
                warn("there is a tensor requires grad but has no real grad");
                continue;
            }

            paramM = beta1 * paramM + (1 - beta1) * *grad.get();
            paramV =
                beta2 * paramV + (1 - beta2) * ((*grad.get()) * (*grad.get()));

            auto mHat = paramM / (1 - std::pow(beta1, turn));
            auto vHat = paramV / (1 - std::pow(beta2, turn));

            *param->getData() -= this->learningRate_ * mHat
                / (epsilon + math::TensorComputeFun::sqrt(vHat));
        }
    }

  private:
    /**
     * @brief before the optimize process, firstly, set the optimizer inner
     *      parameters
     */
    void preTrainIfNeed() {
        if (trainPrepared) {
            return;
        }
        auto paramSize = this->parameters_.size();
        parametersM_.clear();
        parametersV_.clear();
        parametersM_.resize(paramSize);
        parametersV_.resize(paramSize);
        turn = 0;
        for (size_t i = 0; i < paramSize; ++i) {
            parametersM_[i] =
                this->parameters_[i].getComputeNode()->getData()->zeros();
            parametersV_[i] =
                this->parameters_[i].getComputeNode()->getData()->zeros();
        }

        trainPrepared = true;
    }

    T beta1 = DefaultBeta1;
    T beta2 = DefaultBeta2;
    T epsilon = DefaultEpsilon;

    std::vector<math::TensorWrapper<T>> parametersM_{};
    std::vector<math::TensorWrapper<T>> parametersV_{};
    size_t turn = 0;

    bool trainPrepared = false;
};
} // namespace hahaha::ml

#endif // HAHAHA_ADAMOPTIMIZER_H
