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

#ifndef HAHAHA_ML_SCHEDULER_COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER_H

#include <cmath>
#include <cstddef>

#include "ml/scheduler/OptimizerScheduler.h"

namespace hahaha::ml {
/**
 * @brief Interface for cosine annealing with warm restarts (SGDR-style).
 *
 * formula:
 *   lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_0)) for t < T_0
 *   lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_n)) for t >=
 * T_0 where t is the current step, T_0 is the initial period (in steps) before the
 * first restart, T_n is the period (in steps) after the n-th restart, tMult is the
 * multiplicative factor to increase the period after each restart, eta_max is the
 * maximum learning rate, and eta_min is the minimum learning rate.
 */
template <typename T>
class CosineAnnealingWarmRestartsScheduler : public OptimizerScheduler<T> {
  public:
    CosineAnnealingWarmRestartsScheduler(Optimizer<T>& optimizer,
                                         T etaMax,
                                         T etaMin,
                                         std::size_t t0,
                                         T tMult)
        : OptimizerScheduler<T>(optimizer, etaMin), etaMax_(etaMax), etaMin_(etaMin),
          t0_(t0), tMult_(tMult) {
        this->setOptimizerLearningRate(etaMin);
    }

    ~CosineAnnealingWarmRestartsScheduler() override = default;

    void step() override {
        this->increaseStep();
        if (this->getStep() < getT0()) {
            this->setOptimizerLearningRate(
                getEtaMin()
                + 0.5 * static_cast<T>(getEtaMax() - getEtaMin())
                    * (1
                       + std::cos(static_cast<T>(M_PI)
                                  * static_cast<T>(this->getStep())
                                  / static_cast<T>(getT0()))));
        } else {
            std::size_t t = this->getStep() - getT0();
            std::size_t tN = (getT0()) * static_cast<std::size_t>(getTMult());
            this->setOptimizerLearningRate(
                getEtaMin()
                + 0.5 * static_cast<T>(getEtaMax() - getEtaMin())
                    * (1
                       + std::cos(static_cast<T>(M_PI) * static_cast<T>(t)
                                  / static_cast<T>(tN))));
        }
    }
    void reset() override {
        // Note: Configuration parameters are not reset
        // Only reset the learning rate to initial state (etaMin)
        this->setOptimizerLearningRate(etaMin_);
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

    /** @brief Initial period (in steps) before the first restart. */
    [[nodiscard]] virtual std::size_t getT0() const {
        return t0_;
    }
    virtual void setT0(std::size_t t0) {
        t0_ = t0;
    }

    /** @brief Multiplicative factor to increase the period after each restart. */
    [[nodiscard]] virtual T getTMult() const {
        return tMult_;
    }
    virtual void setTMult(T tMult) {
        tMult_ = tMult;
    }

  private:
    T etaMax_;
    T etaMin_;
    std::size_t t0_;
    T tMult_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER_H
