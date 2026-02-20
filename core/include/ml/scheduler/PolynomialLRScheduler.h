// Copyright (c) 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#ifndef HAHAHA_ML_SCHEDULER_POLYNOMIAL_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_POLYNOMIAL_LR_SCHEDULER_H

#include <cmath>
#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Interface for polynomial or linear LR decay schedules.
 *
 * Often combined with a separate warmup phase:
 *   lr_t = (base_lr - end_lr) * (1 - t / T)^power + end_lr
 */
template <typename T> class PolynomialLRScheduler : public OptimizerScheduler<T> {
  public:
    PolynomialLRScheduler(Optimizer<T>& optimizer,
                          T baseLr,
                          std::size_t totalSteps,
                          T endLearningRate,
                          T power)
        : OptimizerScheduler<T>(optimizer, baseLr), totalSteps_(totalSteps),
          endLearningRate_(endLearningRate), power_(power) {
        this->setOptimizerLearningRate(baseLr);
    }

    ~PolynomialLRScheduler() override = default;

    void step() override {
        this->increaseStep();
        this->setOptimizerLearningRate(
            (this->getBaseLearningRate() - getEndLearningRate())
                * static_cast<T>(
                    std::pow(1 - this->getStep() / getTotalSteps(), getPower()))
            + getEndLearningRate());
    }

    void reset() override {
        // Note: Configuration parameters are not reset
        // Only reset the learning rate to base
        this->setOptimizerLearningRate(this->getBaseLearningRate());
    }

    /** @brief Total number of decay steps. */
    [[nodiscard]] virtual std::size_t getTotalSteps() const {
        return totalSteps_;
    }
    virtual void setTotalSteps(std::size_t totalSteps) {
        totalSteps_ = totalSteps;
    }

    /** @brief Final learning rate at the end of decay. */
    [[nodiscard]] virtual T getEndLearningRate() const {
        return endLearningRate_;
    }
    virtual void setEndLearningRate(T lr) {
        endLearningRate_ = lr;
    }

    /** @brief Polynomial power (1.0 for linear decay). */
    [[nodiscard]] virtual T getPower() const {
        return power_;
    }
    virtual void setPower(T power) {
        power_ = power;
    }

  private:
    size_t totalSteps_;
    T endLearningRate_;
    T power_;
};
} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_POLYNOMIAL_LR_SCHEDULER_H
