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
    /**
     * @brief Construct a new Adam Optimizer with default hyperparameters.
     * @param parameters A vector of compute nodes to be optimized.
     * @param learningRate The step size used for each iteration.
     */
    AdamOptimizer(
        const std::vector<std::shared_ptr<compute::ComputeNode<T>>>& parameters,
        const T learningRate)
        : Optimizer<T>(parameters, learningRate) {
    }

    /**
     * @brief Copy constructor from a base Optimizer.
     * @param optimizer The optimizer instance to copy from.
     */
    explicit AdamOptimizer(const Optimizer<T>& optimizer)
        : Optimizer<T>(optimizer) {
    }

    /**
     * @brief Move constructor from a base Optimizer.
     * @param optimizer The optimizer instance to move from.
     */
    explicit AdamOptimizer(Optimizer<T>&& optimizer) : Optimizer<T>(optimizer) {
    }

    /**
     * @brief Construct a new Adam Optimizer with custom hyperparameters.
     * @param parameters A vector of compute nodes to be optimized.
     * @param learningRate The step size used for each iteration.
     * @param beta1 Exponential decay rate for the first moment estimates.
     * @param beta2 Exponential decay rate for the second-moment estimates.
     * @param epsilon A small constant for numerical stability.
     */
    AdamOptimizer(
        const std::vector<std::shared_ptr<compute::ComputeNode<T>>>& parameters,
        const T& learningRate,
        T beta1,
        T beta2,
        T epsilon)
        : Optimizer<T>(parameters, learningRate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon) {
    }

    /**
     * @brief Adds a parameter to the optimizer's watch list.
     * @details If the optimizer has already been prepared/initialized, this also
     * initializes the moment vectors (M and V) for the new parameter.
     * @param param The compute node parameter to add.
     */
    void addParameter(std::shared_ptr<compute::ComputeNode<T>> param) override {
        if (trainPrepared_ && param) {
            parametersM_.push_back(param->getData()->zeros());
            parametersV_.push_back(param->getData()->zeros());
        }
        Optimizer<T>::addParameter(std::move(param));
    }

    /**
     * @brief Performs a single optimization step (parameter update).
     * @details This method calculates the bias-corrected first and second moment 
     * estimates and updates the data of each parameter that requires gradients.
     */
    void step() override {
        preTrainIfNeed();
        ++turn_;
        beta1PowT_ *= beta1_;
        beta2PowT_ *= beta2_;
        for (size_t i = 0; i < parametersM_.size(); ++i) {
            auto param = this->parameters_[i];
            auto& paramM = this->parametersM_[i];
            auto& paramV = this->parametersV_[i];

            if (!param || !param->getRequiresGrad()) {
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
     * @brief Initializes internal state (moment buffers) before the first training step.
     * @details Allocates and zeros out the `parametersM_` and `parametersV_` vectors 
     * based on the current parameters registered in the optimizer.
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
            if (this->parameters_[i]) {
                parametersM_[i] = this->parameters_[i]->getData()->zeros();
                parametersV_[i] = this->parameters_[i]->getData()->zeros();
            }
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
