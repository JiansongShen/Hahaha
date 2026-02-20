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

#ifndef HAHAHA_ML_SCHEDULER_STEP_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_STEP_LR_SCHEDULER_H

#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Interface for classic step-decay learning-rate schedules.
 *
 * This family covers strategies like StepLR where the learning rate is
 * multiplied by a factor @f$\gamma@f$ every @c stepSize steps/epochs.
 *
 * Typical policy:
 *   lr_k = base_lr * gamma^{floor(step / step_size)}
 *
 * @tparam T Numeric type (float, double, ...).
 */
template <typename T> class StepLRScheduler : public OptimizerScheduler<T> {
  public:
    StepLRScheduler(Optimizer<T>& optimizer,
                    T baseLr,
                    std::size_t stepSize,
                    T gamma = T(0.1))
        : OptimizerScheduler<T>(optimizer, baseLr), stepSize_(stepSize),
          gamma_(gamma), gammaPow_(T(1.0)) {
        // Initialize learning rate to base learning rate
        this->setOptimizerLearningRate(baseLr);
    }

    ~StepLRScheduler() override = default;

    /** @brief Advance one scheduling step (batch or epoch). */
    void step() override {
        this->increaseStep();
        // Decay learning rate when we hit a milestone step
        if (this->getStep() % stepSize_ == 0 && this->getStep() > 0) {
            gammaPow_ *= gamma_;
        }
        this->setOptimizerLearningRate(this->getBaseLearningRate() * gammaPow_);
    }

    /** @brief Reset internal counters and restore initial state. */
    void reset() override {
        gammaPow_ = T(1.0);
        this->setOptimizerLearningRate(this->getBaseLearningRate());
    }

    /** @brief Get the step interval between decays. */
    [[nodiscard]] virtual std::size_t getStepSize() const {
        return stepSize_;
    }

    /** @brief Set the step interval between decays. */
    virtual void setStepSize(std::size_t stepSize) {
        stepSize_ = stepSize;
    }

    /** @brief Get the multiplicative decay factor @f$\gamma@f$. */
    [[nodiscard]] virtual T getGamma() const {
        return gamma_;
    }

    /** @brief Set the multiplicative decay factor @f$\gamma@f$. */
    virtual void setGamma(T gamma) {
        gamma_ = gamma;
    }

  private:
    std::size_t stepSize_;
    T gamma_;
    T gammaPow_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_STEP_LR_SCHEDULER_H
