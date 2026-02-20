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

#ifndef HAHAHA_ML_SCHEDULER_OPTIMIZER_SCHEDULER_H
#define HAHAHA_ML_SCHEDULER_OPTIMIZER_SCHEDULER_H

#include "ml/optimizer/Optimizer.h"
#include "ml/scheduler/Scheduler.h"

namespace hahaha::ml {
/**
 * @brief Base class for schedulers that operate on a specific optimizer.
 *
 * This helper binds a scheduler to an optimizer instance and provides
 * protected accessors for derived classes to modify the optimizer.
 */
template <typename T> class OptimizerScheduler : public Scheduler<T> {
  public:
    explicit OptimizerScheduler(Optimizer<T>& optimizer, T baseLr)
        : optimizer_(optimizer), baseLr_(baseLr), step_(0) {
    }

    [[nodiscard]] T getBaseLearningRate() const override {
        return baseLr_;
    }

    [[nodiscard]] T getLearningRate() const override {
        return optimizer_.getLearningRate();
    }

    [[nodiscard]] virtual size_t getStep() const {
        return step_;
    }

    virtual void increaseStep() {
        ++step_;
    }

  protected:
    /** @brief Access the bound optimizer. */
    [[nodiscard]] Optimizer<T>& getOptimizer() {
        return optimizer_;
    }

    /** @brief Set a new learning rate on the bound optimizer. */
    void setOptimizerLearningRate(T lr) {
        optimizer_.setLearningRate(lr);
    }

  private:
    Optimizer<T>& optimizer_;
    T baseLr_;
    size_t step_;
};

} // namespace hahaha::ml

#endif // HAHAHA_ML_SCHEDULER_OPTIMIZER_SCHEDULER_H
