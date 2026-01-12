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
 * @brief AdamW Optimizer implementation.
 *
 * AdamW (Adam with decoupled weight decay) decouples the weight decay from the
 * gradient update, which yields better generalization performance than standard
 * Adam with L2 regularization.
 *
 * The update rule is defined as:
 *
 *     t = t + 1
 *     g_t = grad(theta_{t-1})
 *     m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *     v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *     m_hat = m_t / (1 - beta1^t)
 *     v_hat = v_t / (1 - beta2^t)
 *     theta_t = theta_{t-1} - eta * (alpha * m_hat / (sqrt(v_hat) + epsilon) + lambda * theta_{t-1})
 *
 * Where:
 * - eta: learning rate
 * - beta1, beta2: coefficients for computing running averages of gradient and its square
 * - epsilon: term added to the denominator to improve numerical stability
 * - lambda: weight decay coefficient
 *
 * @tparam T The numeric type (must be float or double).
 */
template <typename T> class AdamWOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamWOptimizer just supports float values");
    static constexpr T DefaultBeta1 = 0.9;
    static constexpr T DefaultBeta2 = 0.999;
    static constexpr T DefaultEpsilon = 1e-8;
    static constexpr T DefaultWeightDecay = 1e-6;
  public:
    AdamWOptimizer(const std::vector<Tensor<T>>& parameters,
                  const T learningRate)
        : Optimizer<T>(parameters, learningRate) {
    }
    explicit AdamWOptimizer(const Optimizer<T>& optimizer)
        : Optimizer<T>(optimizer) {
    }

    explicit AdamWOptimizer(Optimizer<T>&& optimizer) : Optimizer<T>(optimizer) {
    }

    AdamWOptimizer(const std::vector<Tensor<T>>& parameters,
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
        
        // Bias correction terms
        T biasCorrection1 = 1.0 / (1.0 - beta1PowT_);
        T biasCorrection2 = 1.0 / (1.0 - beta2PowT_);
        T biasCorrection2Sqrt = std::sqrt(biasCorrection2);
        T lr = this->learningRate_;

        for (size_t i = 0; i < parametersM_.size(); ++i) {
            auto paramNode = this->parameters_[i].getComputeNode();
            auto& paramM = this->parametersM_[i];
            auto& paramV = this->parametersV_[i];

            if (!paramNode->getRequiresGrad()) continue;
            auto grad = paramNode->getGrad();
            if (grad == nullptr) continue;

            // Update biased first moment estimate: m = β1 * m + (1 - β1) * g
            paramM *= beta1_;
            paramM.axpy(1.0 - beta1_, *grad);

            // Update biased second moment estimate: v = β2 * v + (1 - β2) * g^2
            paramV *= beta2_;
            paramV.axpy(1.0 - beta2_, math::TensorComputeFun::square(*grad));

            // Compute denominator: sqrt(v_hat) + ε = sqrt(v / (1 - β2^t)) + ε
            auto denom = math::TensorComputeFun::sqrt(paramV);
            denom *= biasCorrection2Sqrt;
            denom += epsilon_;

            // Compute update: m_hat / denom = (m / (1 - β1^t)) / denom
            auto update = (paramM * biasCorrection1);
            update /= denom;

            // Apply weight decay: θ = θ * (1 - lr * λ)
            *paramNode->getData() *= (1 - lr * weightDecay_);

            // Apply update: θ = θ - lr * update
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
