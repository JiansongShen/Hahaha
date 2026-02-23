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

#ifndef HAHAHA_GELU_H_D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S
#define HAHAHA_GELU_H_D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S

#include "Activation.h"
#include "ml/compute/graph/ComputeNode.h"
#include <memory>

namespace hahaha::ml {

/**
 * @brief Gaussian Error Linear Unit (GELU) activation function.
 *
 * GELU is a smooth, non-monotonic activation function that has been shown
 * to work well in transformer models and other modern architectures.
 *
 * Formula:
 *   f(x) = x * Phi(x)
 *
 * Where Phi(x) is the cumulative distribution function of the standard
 * normal distribution:
 *   Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
 *
 * Approximate formula (commonly used):
 *   f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 *
 * Gradient:
 *   f'(x) = Phi(x) + x * phi(x)
 *
 * Where phi(x) is the probability density function of the standard normal.
 *
 * @tparam T The numeric type.
 */
template <typename T> class GELU : public Activation<T> {
  public:
    /**
     * @brief Apply GELU activation to the input tensor.
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node with GELU applied.
     */
    std::shared_ptr<ComputeNode<T>> forward(std::shared_ptr<ComputeNode<T>> input) override;

    /**
     * @brief Compute the gradient of GELU.
     * @param input The input compute node.
     * @param gradOutput The gradient from the next layer.
     * @return std::shared_ptr<ComputeNode<T>> The gradient with respect to the input.
     */
    std::shared_ptr<ComputeNode<T>> backward(std::shared_ptr<ComputeNode<T>> input,
                                              std::shared_ptr<ComputeNode<T>> gradOutput) override;
};

/**
 * @brief Convenience function to apply GELU activation.
 * @tparam T The numeric type.
 * @param input The input compute node.
 * @return std::shared_ptr<ComputeNode<T>> The output compute node with GELU applied.
 */
template <typename T> std::shared_ptr<ComputeNode<T>> gelu(std::shared_ptr<ComputeNode<T>> input);

} // namespace hahaha::ml

#endif // HAHAHA_GELU_H_D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S
