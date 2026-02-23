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
//  Contributors:
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#ifndef HAHAHA_SIGMOID_H_E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T
#define HAHAHA_SIGMOID_H_E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T

#include <memory>

#include "Activation.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

/**
 * @brief Sigmoid activation function.
 *
 * Sigmoid is a smooth, S-shaped activation function that maps any real
 * number to a value between 0 and 1. It is commonly used in binary
 * classification problems.
 *
 * Formula:
 *   f(x) = 1 / (1 + exp(-x))
 *
 * Alternative form:
 *   f(x) = exp(x) / (exp(x) + 1)
 *
 * Gradient:
 *   f'(x) = f(x) * (1 - f(x))
 *
 * Properties:
 * - Output range: (0, 1)
 * - Smooth and differentiable everywhere
 * - Can suffer from vanishing gradient problem
 *
 * @tparam T The numeric type.
 */
template <typename T> class Sigmoid : public Activation<T> {
  public:
    /**
     * @brief Apply sigmoid activation to the input tensor.
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node with sigmoid
     * applied.
     */
    std::shared_ptr<ComputeNode<T>>
    forward(std::shared_ptr<ComputeNode<T>> input) override;

    /**
     * @brief Compute the gradient of sigmoid.
     * @param input The input compute node.
     * @param gradOutput The gradient from the next layer.
     * @return std::shared_ptr<ComputeNode<T>> The gradient with respect to the
     * input.
     */
    std::shared_ptr<ComputeNode<T>>
    backward(std::shared_ptr<ComputeNode<T>> input,
             std::shared_ptr<ComputeNode<T>> gradOutput) override;
};

/**
 * @brief Convenience function to apply sigmoid activation.
 * @tparam T The numeric type.
 * @param input The input compute node.
 * @return std::shared_ptr<ComputeNode<T>> The output compute node with sigmoid
 * applied.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> sigmoid(std::shared_ptr<ComputeNode<T>> input);

} // namespace hahaha::ml

#endif // HAHAHA_SIGMOID_H_E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T
