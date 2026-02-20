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

#ifndef HAHAHA_ML_SCHEDULER_CYCLIC_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_CYCLIC_LR_SCHEDULER_H

#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Interface for cyclic / triangular learning-rate schedules.
 *
 * LR oscillates between a lower and upper bound, e.g. in a triangular
 * fashion, often used to quickly explore good LR ranges.
 */
template <typename T> class CyclicLRScheduler : public OptimizerScheduler<T> {
  public:
    ~CyclicLRScheduler() override = default;

    void step() override = 0;
    void reset() override = 0;

    [[nodiscard]] virtual T getBaseLearningRateMin() const = 0;
    virtual void setBaseLearningRateMin(T lr) = 0;

    [[nodiscard]] virtual T getBaseLearningRateMax() const = 0;
    virtual void setBaseLearningRateMax(T lr) = 0;

    /** @brief Number of steps in the increasing half-cycle. */
    [[nodiscard]] virtual std::size_t getStepSizeUp() const = 0;
    virtual void setStepSizeUp(std::size_t stepSizeUp) = 0;

    /** @brief Number of steps in the decreasing half-cycle. */
    [[nodiscard]] virtual std::size_t getStepSizeDown() const = 0;
    virtual void setStepSizeDown(std::size_t stepSizeDown) = 0;
};

/**
 * @brief Interface for OneCycle learning-rate schedules.
 *
 * Follows Leslie Smith's OneCycle policy:
 *  - warm up LR to a maximum value,
 *  - then decrease below the initial LR in a single cycle.
 */
template <typename T> class OneCycleLRScheduler : public OptimizerScheduler<T> {
  public:
    ~OneCycleLRScheduler() override = default;

    void step() override = 0;
    void reset() override = 0;

    [[nodiscard]] virtual std::size_t getTotalSteps() const = 0;
    virtual void setTotalSteps(std::size_t totalSteps) = 0;

    [[nodiscard]] virtual T getMaxLearningRate() const = 0;
    virtual void setMaxLearningRate(T lr) = 0;

    [[nodiscard]] virtual T getMinLearningRate() const = 0;
    virtual void setMinLearningRate(T lr) = 0;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_CYCLIC_LR_SCHEDULER_H
