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

#ifndef HAHAHA_ML_SCHEDULER_WARMUP_COSINE_DECAY_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_WARMUP_COSINE_DECAY_SCHEDULER_H

#include <cmath>
#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {
/**
 * @brief Interface for warmup + cosine decay schedules.
 *
 * formula:
 *   lr_t = lr_max * (1 - cos(pi * t / T_max)) / 2 for t < T_warmup
 *   lr_t = lr_max * (1 + cos(pi * (t - T_warmup) / (T_total - T_warmup))) / 2 for t
 * >= T_warmup where t is the current step, T_warmup is the number of warmup steps,
 * T_total is the total number of steps, lr_max is the maximum learning rate, and
 * lr_min is the minimum learning rate.
 *
 * Common in Transformer/LLM training: a linear warmup phase, followed by
 * cosine annealing towards a small learning rate.
 */
template <typename T>
class WarmupCosineDecayScheduler : public OptimizerScheduler<T> {
  public:
    WarmupCosineDecayScheduler(Optimizer<T>& optimizer,
                               T maxLearningRate,
                               T minLearningRate,
                               std::size_t warmupSteps,
                               std::size_t totalSteps)
        : OptimizerScheduler<T>(optimizer, minLearningRate),
          maxLearningRate_(maxLearningRate), minLearningRate_(minLearningRate),
          warmupSteps_(warmupSteps), totalSteps_(totalSteps) {
        this->setOptimizerLearningRate(minLearningRate);
    }

    void step() override {
        this->increaseStep();
        if (this->getStep() < getWarmupSteps()) {
            this->setOptimizerLearningRate(
                getMaxLearningRate()
                * (1
                   - std::cos(static_cast<T>(M_PI) * static_cast<T>(this->getStep())
                              / static_cast<T>(getWarmupSteps()))));
        } else {
            this->setOptimizerLearningRate(
                getMaxLearningRate()
                * (1
                   + std::cos(
                       static_cast<T>(M_PI)
                       * static_cast<T>(this->getStep() - getWarmupSteps())
                       / static_cast<T>(getTotalSteps() - getWarmupSteps()))));
        }
    }
    void reset() override {
        // Note: Configuration parameters are not reset
        // Only reset the learning rate to initial state (minLearningRate)
        this->setOptimizerLearningRate(minLearningRate_);
    }

    /** @brief Number of warmup steps before cosine decay starts. */
    [[nodiscard]] virtual std::size_t getWarmupSteps() const {
        return warmupSteps_;
    }
    virtual void setWarmupSteps(std::size_t warmupSteps) {
        warmupSteps_ = warmupSteps;
    }

    /** @brief Total number of steps including warmup + decay. */
    [[nodiscard]] virtual std::size_t getTotalSteps() const {
        return totalSteps_;
    }
    virtual void setTotalSteps(std::size_t totalSteps) {
        totalSteps_ = totalSteps;
    }

    /** @brief Maximum learning rate reached after warmup. */
    [[nodiscard]] virtual T getMaxLearningRate() const {
        return maxLearningRate_;
    }
    virtual void setMaxLearningRate(T lr) {
        maxLearningRate_ = lr;
    }

    /** @brief Final learning rate after decay. */
    [[nodiscard]] virtual T getMinLearningRate() const {
        return minLearningRate_;
    }
    virtual void setMinLearningRate(T lr) {
        minLearningRate_ = lr;
    }

  private:
    T maxLearningRate_;
    T minLearningRate_;
    std::size_t warmupSteps_;
    std::size_t totalSteps_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_WARMUP_COSINE_DECAY_SCHEDULER_H
