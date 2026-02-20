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

#ifndef HAHAHA_ML_SCHEDULER_WARMUP_POLYNOMIAL_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_WARMUP_POLYNOMIAL_LR_SCHEDULER_H

#include <cmath>
#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Interface for LR schedules with linear warmup + polynomial decay.
 *
 * Typical policy:
 *   lr_t = (base_lr - end_lr) * (1 - t / T)^power + end_lr for t >= warmup_steps
 *   lr_t = base_lr * t / warmup_steps for t < warmup_steps
 * where t is the current step, T is the total number of steps,
 * end_lr is the final learning rate, and power is the power of the polynomial.
 * The learning rate is linearly increased from 0 to base_lr during the warmup phase,
 * and then decayed using the polynomial schedule.
 */
template <typename T>
class WarmupPolynomialLRScheduler : public OptimizerScheduler<T> {
  public:
    WarmupPolynomialLRScheduler(Optimizer<T>& optimizer,
                                T baseLr,
                                std::size_t totalSteps,
                                T endLearningRate,
                                T power,
                                std::size_t warmupSteps)
        : OptimizerScheduler<T>(optimizer, baseLr), totalSteps_(totalSteps),
          endLearningRate_(endLearningRate), power_(power),
          warmupSteps_(warmupSteps) {
        this->setOptimizerLearningRate(baseLr);
    }

    ~WarmupPolynomialLRScheduler() override = default;

    void step() override {
        this->increaseStep();
        if (this->getStep() < getWarmupSteps()) {
            this->setOptimizerLearningRate(
                this->getBaseLearningRate() * (static_cast<T>(this->getStep()) / static_cast<T>(getWarmupSteps())));
        } else {
            this->setOptimizerLearningRate(
                (this->getBaseLearningRate() - getEndLearningRate())
                    * static_cast<T>(
                        std::pow(1 - this->getStep() / getTotalSteps(), getPower()))
                + getEndLearningRate());
        }
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

    /** @brief Number of warmup steps. */
    [[nodiscard]] virtual std::size_t getWarmupSteps() const {
        return warmupSteps_;
    }
    virtual void setWarmupSteps(std::size_t warmupSteps) {
        warmupSteps_ = warmupSteps;
    }

  private:
    std::size_t totalSteps_;
    T endLearningRate_;
    T power_;
    std::size_t warmupSteps_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_WARMUP_POLYNOMIAL_LR_SCHEDULER_H
