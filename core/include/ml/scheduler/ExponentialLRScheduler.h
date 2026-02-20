// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// JiansongShen (jason.shen111@outlook.com) (https://github.com/jiansongshen)

#ifndef HAHAHA_ML_SCHEDULER_EXPONENTIAL_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_EXPONENTIAL_LR_SCHEDULER_H

#include <cmath>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Interface for exponential decay learning-rate schedules.
 *
 * Typical policy:
 *   lr_k = base_lr * gamma^{step}
 *
 * where @c gamma is a constant in (0, 1).
 */
template <typename T> class ExponentialLRScheduler : public OptimizerScheduler<T> {
  public:
    ExponentialLRScheduler(Optimizer<T>& optimizer, T baseLr, T gamma)
        : OptimizerScheduler<T>(optimizer, baseLr), gamma_(gamma) {
        if (gamma <= 0 || gamma >= 1) {
            throw std::invalid_argument("gamma must be in (0, 1)");
        }
        this->setOptimizerLearningRate(baseLr);
    }

    ~ExponentialLRScheduler() override = default;

    void step() override {
        this->increaseStep();
        this->setOptimizerLearningRate(
            this->getBaseLearningRate()
            * static_cast<T>(std::pow(getGamma(), this->getStep())));
    }

    void reset() override {
        // Note: gamma_ is not reset as it's a configuration parameter
        // Only reset the learning rate to base
        this->setOptimizerLearningRate(this->getBaseLearningRate());
    }

    /** @brief Get the exponential decay factor @f$\gamma@f$. */
    [[nodiscard]] virtual T getGamma() const {
        return gamma_;
    }

    /** @brief Set the exponential decay factor @f$\gamma@f$. */
    virtual void setGamma(T gamma) {
        gamma_ = gamma;
    }

  private:
    T gamma_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_EXPONENTIAL_LR_SCHEDULER_H
