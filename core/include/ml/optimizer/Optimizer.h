// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
// JiansongShen (jason.shen111@outlook.com) (https://github.com/jiansongshen)

#ifndef OPTIMIZER_B8596AC9_CA72_40A7_A99C_9A73CB770A4B
#define OPTIMIZER_B8596AC9_CA72_40A7_A99C_9A73CB770A4B

#include <memory>
#include <vector>

#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

/**
 * @brief Base class for all optimizers.
 *
 * An optimizer updates the parameters of a model based on the computed
 * gradients. It manages a list of parameters and provides a step() method
 * for performing the update logic.
 *
 * @tparam T Numeric data type (float, double, etc.).
 */
template <typename T> class Optimizer {
  public:
    /**
     * @brief Construct a new Optimizer.
     *
     * @param parameters List of compute nodes to optimize.
     * @param learningRate Learning rate for the updates.
     */
    Optimizer(std::vector<std::shared_ptr<ml::ComputeNode<T>>> parameters,
              T learningRate)
        : parameters_(std::move(parameters)), learningRate_(learningRate) {
    }

    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = default;
    Optimizer& operator=(const Optimizer&) = default;
    Optimizer(Optimizer&&) noexcept = default;
    Optimizer& operator=(Optimizer&&) noexcept = default;

    /**
     * @brief Performs a single optimization step (parameter update).
     *
     * This method must be implemented by concrete optimizer subclasses.
     * It iterates through the parameters and updates their values using
     * their gradients.
     */
    virtual void step() {};

    /**
     * @brief Resets the gradients of all optimized parameters to zero.
     *
     * This should be called before the backward pass of each training step.
     */
    virtual void zeroGrad() {
        for (auto& param : parameters_) {
            if (param) {
                param->clearGrad();
            }
        }
    }

    /**
     * @brief Sets a new learning rate.
     * @param learningRate The new learning rate value.
     */
    void setLearningRate(T learningRate) {
        learningRate_ = learningRate;
    }

    /**
     * @brief Gets the current learning rate.
     * @return T The current learning rate.
     */
    [[nodiscard]] T getLearningRate() const {
        return learningRate_;
    }

    /**
     * @brief Adds a parameter to the optimizer's tracking list.
     * @param param The compute node to be optimized.
     */
    virtual void addParameter(std::shared_ptr<ml::ComputeNode<T>> param) {
        parameters_.push_back(std::move(param));
    }

    /**
     * @brief Add a high-level `Tensor` to the optimizer.
     */
    virtual void addParameter(const hahaha::Tensor<T>& tensor) {
        addParameter(tensor.getComputeNode());
    }

    /**
     * @brief Gets the list of compute nodes (for subclasses).
     * @return std::vector<std::shared_ptr<compute::ComputeNode<T>>>& Reference to
     * compute nodes.
     */
    std::vector<std::shared_ptr<ml::ComputeNode<T>>>& getParameters() {
        return parameters_;
    }

  protected:
    std::vector<std::shared_ptr<ml::ComputeNode<T>>>
        parameters_; /**< List of compute nodes to optimize. */
    T learningRate_; /**< Learning rate. */
};

} // namespace hahaha::ml

#endif // OPTIMIZER_B8596AC9_CA72_40A7_A99C_9A73CB770A4B
