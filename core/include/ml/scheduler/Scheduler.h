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

#ifndef HAHAHA_ML_SCHEDULER_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_SCHEDULER_H

namespace hahaha::ml {

/**
 * @brief Base interface for learning-rate schedulers.
 *
 * A scheduler controls how an optimizer's effective learning rate changes
 * over training steps or epochs. Concrete implementations should override
 * @ref step() and optionally expose configuration and state query methods.
 *
 * Typical usages:
 * - Call step() once per optimization step (per batch) or once per epoch.
 * - Optionally query the current learning rate with getLearningRate().
 *
 * @tparam T Numeric type (float, double, ...).
 */
template <typename T> class Scheduler {
  public:
    virtual ~Scheduler() = default;

    /**
     * @brief Advance the scheduler by one step.
     *
     * Implementations typically:
     * - Update internal counters (e.g., current step or epoch).
     * - Compute a new learning rate based on their policy.
     * - Apply the new learning rate to the attached optimizer.
     */
    virtual void step() = 0;

    /**
     * @brief Reset internal state to its initial configuration.
     *
     * This is useful when restarting training from scratch or when
     * reusing a scheduler for a new optimizer or dataset.
     */
    virtual void reset() = 0;

    /**
     * @brief Get the underlying base learning rate (before scheduling).
     */
    [[nodiscard]] virtual T getBaseLearningRate() const = 0;

    /**
     * @brief Get the current learning rate applied to the optimizer.
     */
    [[nodiscard]] virtual T getLearningRate() const = 0;
};
} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_SCHEDULER_H
