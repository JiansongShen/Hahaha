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

/**
 * @brief Adam Optimizer implementation.
 *
 * Adam (Adaptive Moment Estimation) is an algorithm for first-order gradient-based
 * optimization of stochastic objective functions, based on adaptive estimates of
 * lower-order moments.
 *
 * The update rule is defined as:
 *
 *     t = t + 1
 *     g_t = grad(theta_{t-1})
 *     m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *     v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *     m_hat = m_t / (1 - beta1^t)
 *     v_hat = v_t / (1 - beta2^t)
 *     theta_t = theta_{t-1} - eta * m_hat / (sqrt(v_hat) + epsilon)
 *
 * Where:
 * - eta: learning rate
 * - beta1, beta2: coefficients for computing running averages of gradient and its square
 * - epsilon: term added to the denominator to improve numerical stability
 *
 * @tparam T The numeric type (must be float or double).
 */
template <typename T> class AdamOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamOptimizer just supports float values");
    static constexpr T DefaultBeta1 = 0.9;
    static constexpr T DefaultBeta2 = 0.999;
    static constexpr T DefaultEpsilon = 1e-8;
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
                  T epsilon)
        : Optimizer<T>(parameters, learningRate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon) {
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

            (paramM *= beta1_).axpy((1 - beta1_), *grad);
            (paramV *= beta2_)
                .axpy((1 - beta2_), math::TensorComputeFun::square(*grad));

            auto mHat = paramM / (1 - beta1PowT_);
            auto vHat = paramV / (1 - beta2PowT_);

            param->getData()->axpy(
                -this->learningRate_,
                (mHat /= (epsilon_ + math::TensorComputeFun::sqrt(vHat))));
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

    std::vector<math::TensorWrapper<T>> parametersM_{};
    std::vector<math::TensorWrapper<T>> parametersV_{};
    size_t turn_ = 0;

    bool trainPrepared_ = false;
};
} // namespace hahaha::ml

#endif // HAHAHA_ADAMOPTIMIZER_H
