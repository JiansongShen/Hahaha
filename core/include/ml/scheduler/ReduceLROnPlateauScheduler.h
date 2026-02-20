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

#ifndef HAHAHA_ML_SCHEDULER_REDUCE_LR_ON_PLATEAU_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_REDUCE_LR_ON_PLATEAU_SCHEDULER_H

#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Mode for ReduceLROnPlateau-style schedulers.
 *
 * - Min:  reduce LR when the monitored metric has stopped decreasing.
 * - Max:  reduce LR when the monitored metric has stopped increasing.
 */
enum class PlateauMode { Min, Max };

/**
 * @brief Interface for performance-based LR schedulers (ReduceLROnPlateau).
 *
 * Implementations monitor a validation metric and reduce the learning rate
 * when no improvement has been observed for a given patience window.
 */
template <typename T>
class ReduceLROnPlateauScheduler : public OptimizerScheduler<T> {
  public:
    ~ReduceLROnPlateauScheduler() override = default;

    /** @brief Advance one scheduling step using the latest monitored metric. */
    virtual void step(T metric) = 0;

    /** @brief Convenience: step() without metric should be implemented or rejected.
     */
    void step() override = 0;

    /** @brief Reset internal state (best metric, cooldown, etc.). */
    void reset() override = 0;

    /** @brief Get the current operating mode (Min or Max). */
    [[nodiscard]] virtual PlateauMode getMode() const = 0;

    /** @brief Set the operating mode (Min or Max). */
    virtual void setMode(PlateauMode mode) = 0;

    /** @brief Get the multiplicative LR reduction factor. */
    [[nodiscard]] virtual T getFactor() const = 0;

    /** @brief Set the multiplicative LR reduction factor. */
    virtual void setFactor(T factor) = 0;

    /** @brief Get the patience (number of bad epochs before reducing LR). */
    [[nodiscard]] virtual std::size_t getPatience() const = 0;

    /** @brief Set the patience (number of bad epochs before reducing LR). */
    virtual void setPatience(std::size_t patience) = 0;

    /** @brief Get the cooldown period after a reduction. */
    [[nodiscard]] virtual std::size_t getCooldown() const = 0;

    /** @brief Set the cooldown period after a reduction. */
    virtual void setCooldown(std::size_t cooldown) = 0;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_REDUCE_LR_ON_PLATEAU_SCHEDULER_H
