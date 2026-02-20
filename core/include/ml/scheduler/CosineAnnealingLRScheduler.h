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

#ifndef HAHAHA_ML_SCHEDULER_COSINE_LR_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_COSINE_LR_SCHEDULER_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {

/**
 * @brief Interface for cosine annealing LR schedules.
 *
 * Typical policy:
 *   lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max))
 */
template <typename T>
class CosineAnnealingLRScheduler : public OptimizerScheduler<T> {
  public:
    CosineAnnealingLRScheduler(Optimizer<T>& optimizer,
                               T etaMax,
                               T etaMin,
                               std::size_t tMax)
        : OptimizerScheduler<T>(optimizer, etaMax), etaMax_(etaMax), etaMin_(etaMin),
          tMax_(tMax) {
        // Start at maximum learning rate (etaMax)
        this->setOptimizerLearningRate(etaMax);
    }

    ~CosineAnnealingLRScheduler() override = default;

    void step() override {
        this->increaseStep();
        this->setOptimizerLearningRate(
            getEtaMin()
            + 0.5 * static_cast<T>(getEtaMax() - getEtaMin())
                * (1
                   + std::cos(static_cast<T>(M_PI) * static_cast<T>(this->getStep())
                              / static_cast<T>(getTMax()))));
    }
    void reset() override {
        // Note: Configuration parameters (etaMax_, etaMin_, tMax_) are not reset
        // Only reset the learning rate to initial state (etaMax)
        this->setOptimizerLearningRate(etaMax_);
    }

    [[nodiscard]] virtual T getEtaMax() const {
        return etaMax_;
    }
    virtual void setEtaMax(T etaMax) {
        etaMax_ = etaMax;
    }

    [[nodiscard]] virtual T getEtaMin() const {
        return etaMin_;
    }
    virtual void setEtaMin(T etaMin) {
        etaMin_ = etaMin;
    }

    /** @brief Get total steps in one cosine cycle. */
    [[nodiscard]] virtual std::size_t getTMax() const {
        return tMax_;
    }
    virtual void setTMax(std::size_t tMax) {
        tMax_ = tMax;
    }

  private:
    T etaMax_;
    T etaMin_;
    std::size_t tMax_;
};
} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_COSINE_LR_SCHEDULER_H
