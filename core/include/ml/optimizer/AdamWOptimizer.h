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
//

#ifndef HAHAHA_ML_OPTIMIZER_ADAMW_OPTIMIZER_H
#define HAHAHA_ML_OPTIMIZER_ADAMW_OPTIMIZER_H

#include <cmath>

#include "Optimizer.h"
#include "math/TensorComputeFuns.h"
#include "utils/log/Logger.h"

namespace hahaha::ml {

using common::f32;
using common::f64;
using common::u64;

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
template <typename T> class AdamWOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamOptimizer just supports float values");
    static constexpr T DefaultBeta1 = 0.9;
    static constexpr T DefaultBeta2 = 0.999;
    static constexpr T DefaultEpsilon = 1e-8;
    static constexpr T DefaultWeightDecay = 1e-6;
  public:
    AdamOptimizer(const std::vector<Tensor<T>>& parameters,
                  const T learningRate)
        : Optimizer<T>(parameters, learningRate) {
    }
    explicit AdamOptimizer(const Optimizer<T>& optimizer)
        : Optimizer<T>(optimizer) {
    }

    explicit AdamOptimizer(Optimizer<T>&& optimizer) : Optimizer<T>(optimizer) {
    }

    AdamOptimizer(const std::vector<Tensor<T>>& parameters,
                  const T& learningRate,
                  T beta1,
                  T beta2,
                  T epsilon,
                  T weightDecay)
        : Optimizer<T>(parameters, learningRate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon), weightDecay_(weightDecay){
    }

    void addParameter(const Tensor<T>& param) override {
        if (trainPrepared_) {
            parametersM_.push_back(param.getComputeNode()->getData()->zeros());
            parametersV_.push_back(param.getComputeNode()->getData()->zeros());
        }
        this->parameters_.push_back(param);
    }

    void step() override {
        preTrainIfNeed();
        ++turn_;
        beta1PowT_ *= beta1_;
        beta2PowT_ *= beta2_;
        T mCorr = 1.0 / (1.0 - beta1PowT_);
        T vCorr = 1.0 / (1.0 - beta2PowT_);
        T invSqrtVCorr = 1.0 / std::sqrt(vCorr);
        T lr = this->learningRate_;

        for (size_t i = 0; i < parametersM_.size(); ++i) {
            auto paramNode = this->parameters_[i].getComputeNode();
            auto& paramM = this->parametersM_[i];
            auto& paramV = this->parametersV_[i];

            if (!paramNode->getRequiresGrad()) continue;
            auto grad = paramNode->getGrad();
            if (grad == nullptr) continue;

            paramM *= beta1_;
            paramM.axpy(1.0 - beta1_, *grad);

            paramV *= beta2_;
            paramV.axpy(1.0 - beta2_, math::TensorComputeFun::square(*grad));

            auto denom = math::TensorComputeFun::sqrt(paramV);
            denom *= invSqrtVCorr;
            denom += epsilon_;

            auto update = (paramM * mCorr);
            update /= denom;

            // theta = theta - lr * weight_decay * theta
            paramNode->getData()->axpy(-lr * weightDecay_, *paramNode->getData());

            paramNode->getData()->axpy(-lr, update);
        }

    }

  private:
    /**
     * @brief before the optimize process, firstly, set the optimizer inner
     *      parameters
     */
    void preTrainIfNeed() {
        if (trainPrepared_) {
            return;
        }
        auto paramSize = this->parameters_.size();
        parametersM_.clear();
        parametersV_.clear();
        parametersM_.resize(paramSize);
        parametersV_.resize(paramSize);
        turn_ = 0;
        for (size_t i = 0; i < paramSize; ++i) {
            parametersM_[i] =
                this->parameters_[i].getComputeNode()->getData()->zeros();
            parametersV_[i] =
                this->parameters_[i].getComputeNode()->getData()->zeros();
        }

        trainPrepared_ = true;
    }

    T beta1_ = DefaultBeta1;
    T beta2_ = DefaultBeta2;
    T epsilon_ = DefaultEpsilon;

    T beta1PowT_ = 1;
    T beta2PowT_ = 1;

    T weightDecay_ = DefaultWeightDecay;

    std::vector<math::TensorWrapper<T>> parametersM_{};
    std::vector<math::TensorWrapper<T>> parametersV_{};
    size_t turn_ = 0;

    bool trainPrepared_ = false;
};
} // namespace hahaha::ml
#endif // HAHAHA_ML_OPTIMIZER_ADAMW_OPTIMIZER_H
