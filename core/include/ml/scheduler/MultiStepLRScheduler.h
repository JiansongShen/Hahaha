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

#ifndef HAHAHA_ML_SCHEDULER_MULTI_STEP_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_MULTI_STEP_LR_SCHEDULER_H

#include <algorithm>
#include <cstddef>
#include <vector>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {
/**
 * @brief Interface for multi-step learning-rate schedules.
 *
 * This family covers strategies like MultiStepLR where the learning rate
 * is decayed by a factor gamma at a predefined set of milestone steps.
 *
 * Typical policy:
 *   lr_k = base_lr * gamma^{m}
 * where m is the number of milestones that have been passed.
 */
template <typename T> class MultiStepLRScheduler : public OptimizerScheduler<T> {
  public:
    MultiStepLRScheduler(Optimizer<T>& optimizer,
                         T baseLr,
                         const std::vector<std::size_t>& milestones,
                         T gamma = T(0.1))
        : OptimizerScheduler<T>(optimizer, baseLr), milestones_(milestones),
          gamma_(gamma), gammaPow_(T(1.0)) {
        // Initialize learning rate to base learning rate
        this->setOptimizerLearningRate(baseLr);
        std::sort(milestones_.begin(), milestones_.end());
    }
    ~MultiStepLRScheduler() override = default;

    void step() override {
        this->increaseStep();
        if (std::binary_search(
                milestones_.begin(), milestones_.end(), this->getStep())) {
            gammaPow_ *= gamma_;
        }
        this->setOptimizerLearningRate(this->getBaseLearningRate() * gammaPow_);
    }
    void reset() override {
        gammaPow_ = T(1.0);
        this->setOptimizerLearningRate(this->getBaseLearningRate());
    }

    /** @brief Get the list of milestone steps. */
    [[nodiscard]] virtual const std::vector<std::size_t>& getMilestones() const {
        return milestones_;
    }

    /** @brief Replace the list of milestone steps. */
    virtual void setMilestones(const std::vector<std::size_t>& milestones) {
        milestones_ = milestones;
        std::sort(this->milestones_.begin(), this->milestones_.end());
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
    std::vector<std::size_t> milestones_;
    T gamma_;
    T gammaPow_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_MULTI_STEP_LR_SCHEDULER_H
