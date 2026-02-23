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

#ifndef HAHAHA_SOFTMAX_H_H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V2W
#define HAHAHA_SOFTMAX_H_H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V2W

#include <memory>

#include "Activation.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

/**
 * @brief Softmax activation function.
 *
 * Softmax is commonly used in the output layer of neural networks for
 * multi-class classification. It converts a vector of real numbers into
 * a probability distribution over multiple classes.
 *
 * Formula:
 *   f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 *
 * Where the sum is taken over all elements in the specified dimension.
 *
 * For numerical stability, the formula is often computed as:
 *   f(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Gradient:
 *   ∂f(x_i)/∂x_j = { f(x_i) * (1 - f(x_i)), if i = j
 *                  { -f(x_i) * f(x_j),       if i != j
 *
 * Properties:
 * - Output sums to 1 (probability distribution)
 * - All outputs are positive
 * - Commonly applied along a specific dimension (e.g., last dimension)
 *
 * @tparam T The numeric type.
 */
template <typename T> class Softmax : public Activation<T> {
  private:
    int dim_; ///< The dimension along which to apply softmax.

  public:
    /**
     * @brief Constructor with dimension parameter.
     * @param dim The dimension along which to apply softmax (default: -1, last dimension).
     */
    explicit Softmax(int dim = -1);

    /**
     * @brief Apply softmax activation to the input tensor.
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node with softmax
     * applied.
     */
    std::shared_ptr<ComputeNode<T>>
    forward(std::shared_ptr<ComputeNode<T>> input) override;

    /**
     * @brief Compute the gradient of softmax.
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
 * @brief Convenience function to apply softmax activation.
 * @tparam T The numeric type.
 * @param input The input compute node.
 * @param dim The dimension along which to apply softmax (default: -1, last
 * dimension).
 * @return std::shared_ptr<ComputeNode<T>> The output compute node with softmax
 * applied.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> softmax(std::shared_ptr<ComputeNode<T>> input,
                                        int dim = -1);

} // namespace hahaha::ml

#endif // HAHAHA_SOFTMAX_H_H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V2W
