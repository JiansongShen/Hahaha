// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com)
//

#ifndef HAHAHA_PUBLIC_OPTIMIZER_H_7A8B9C0D1E2F3A4B5C6D7E8F9A0B1C2D
#define HAHAHA_PUBLIC_OPTIMIZER_H_7A8B9C0D1E2F3A4B5C6D7E8F9A0B1C2D

#include <memory>
#include <utility>
#include <vector>

// Public Tensor type — the only type users touch.
#include "Tensor.h"

// Inner optimizer implementations.
#include "ml/optimizer/AdadeltaOptimizer.h"
#include "ml/optimizer/AdamOptimizer.h"
#include "ml/optimizer/AdamWOptimizer.h"
#include "ml/optimizer/Optimizer.h"
#include "ml/optimizer/SGDMOptimizer.h"
#include "ml/optimizer/SGDOptimizer.h"

namespace hahaha {

/**
 * @brief Public handle for all optimizers (pimpl).
 *
 * Delegates all operations to an inner @ref ml::Optimizer<T> held behind a
 * @c shared_ptr.  Concrete subclasses (SGDOptimizer, AdamOptimizer, …)
 * construct the appropriate inner object and pass it to this base.
 *
 * @tparam T Numeric type.
 */
template <typename T>
class Optimizer {
  public:
    virtual ~Optimizer() = default;

    /** @brief Execute one parameter-update step. */
    void step() { impl_->step(); }

    /** @brief Zero (clear) the gradients of all tracked parameters. */
    void zeroGrad() { impl_->zeroGrad(); }

    /** @brief Replace the current learning rate. */
    void setLearningRate(T lr) { impl_->setLearningRate(lr); }

    /** @brief Return the current learning rate. */
    [[nodiscard]] T getLearningRate() const { return impl_->getLearningRate(); }

    /** @brief Register an additional parameter for optimization. */
    void addParameter(const Tensor<T>& param) { impl_->addParameter(param); }

    /** @brief Mutable access to the tracked parameter list. */
    std::vector<Tensor<T>>& getParameters() { return impl_->getParameters(); }

  protected:
    explicit Optimizer(std::shared_ptr<ml::Optimizer<T>> impl)
        : impl_(std::move(impl)) {}

  private:
    std::shared_ptr<ml::Optimizer<T>> impl_;
};

// ---------------------------------------------------------------------------
// Stochastic Gradient Descent
// ---------------------------------------------------------------------------

/** @brief Public SGD optimizer handle. */
template <typename T>
class SGDOptimizer : public Optimizer<T> {
  public:
    SGDOptimizer(std::vector<Tensor<T>> params, T lr)
        : Optimizer<T>(
              std::make_shared<ml::SGDOptimizer<T>>(std::move(params), lr)) {}
};

// ---------------------------------------------------------------------------
// SGD with Momentum
// ---------------------------------------------------------------------------

/** @brief Public SGD-with-Momentum optimizer handle. */
template <typename T>
class SGDMOptimizer : public Optimizer<T> {
  public:
    SGDMOptimizer(std::vector<Tensor<T>> params,
                  T lr,
                  T momentum = T(0.9))
        : Optimizer<T>(std::make_shared<ml::SGDMOptimizer<T>>(
              std::move(params), lr, momentum)) {}
};

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/** @brief Public Adam optimizer handle (float / double only). */
template <typename T>
class AdamOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamOptimizer requires a floating-point type");

  public:
    explicit AdamOptimizer(const std::vector<Tensor<T>>& params, T lr)
        : Optimizer<T>(
              std::make_shared<ml::AdamOptimizer<T>>(params, lr)) {}

    AdamOptimizer(const std::vector<Tensor<T>>& params,
                  T lr,
                  T beta1,
                  T beta2,
                  T eps)
        : Optimizer<T>(std::make_shared<ml::AdamOptimizer<T>>(
              params, lr, beta1, beta2, eps)) {}
};

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

/** @brief Public AdamW optimizer handle (float / double only). */
template <typename T>
class AdamWOptimizer : public Optimizer<T> {
    static_assert(utils::isLegalFloatType<T>::value,
                  "AdamWOptimizer requires a floating-point type");

  public:
    explicit AdamWOptimizer(const std::vector<Tensor<T>>& params, T lr)
        : Optimizer<T>(
              std::make_shared<ml::AdamWOptimizer<T>>(params, lr)) {}

    AdamWOptimizer(const std::vector<Tensor<T>>& params,
                   T lr,
                   T beta1,
                   T beta2,
                   T eps,
                   T weightDecay)
        : Optimizer<T>(std::make_shared<ml::AdamWOptimizer<T>>(
              params, lr, beta1, beta2, eps, weightDecay)) {}
};

// ---------------------------------------------------------------------------
// Adadelta  (no fixed learning rate — stores typed pointer for extra methods)
// ---------------------------------------------------------------------------

/** @brief Public Adadelta optimizer handle. */
template <typename T>
class AdadeltaOptimizer : public Optimizer<T> {
    // Private delegating ctor: creates base + stores typed pointer once.
    explicit AdadeltaOptimizer(
        std::shared_ptr<ml::AdadeltaOptimizer<T>> impl)
        : Optimizer<T>(impl), adaImpl_(impl) {}

  public:
    explicit AdadeltaOptimizer(std::vector<Tensor<T>> params,
                               T decayRate = T(0.9),
                               T epsilon   = T(1e-6))
        : AdadeltaOptimizer(std::make_shared<ml::AdadeltaOptimizer<T>>(
              std::move(params), decayRate, epsilon)) {}

    /** @brief Decay rate ρ used to accumulate E[g²]. */
    [[nodiscard]] T getDecayRate() const { return adaImpl_->getDecayRate(); }

    /** @brief Numerical stability constant ε. */
    [[nodiscard]] T getEpsilon() const { return adaImpl_->getEpsilon(); }

  private:
    std::shared_ptr<ml::AdadeltaOptimizer<T>> adaImpl_;
};

} // namespace hahaha

#endif // HAHAHA_PUBLIC_OPTIMIZER_H_7A8B9C0D1E2F3A4B5C6D7E8F9A0B1C2D
