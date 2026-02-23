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

#ifndef HAHAHA_ELU_H_C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R
#define HAHAHA_ELU_H_C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R

#include "Activation.h"
#include "ml/compute/graph/ComputeNode.h"
#include <memory>

namespace hahaha::ml {

/**
 * @brief Exponential Linear Unit (ELU) activation function.
 *
 * ELU is a smooth activation function that can produce negative outputs,
 * which helps push the mean activation closer to zero and speeds up learning.
 *
 * Formula:
 *   f(x) = { x,              if x > 0
 *          { alpha * (exp(x) - 1), if x <= 0
 *
 * Where alpha is a hyperparameter (typically 1.0).
 *
 * Gradient:
 *   f'(x) = { 1,              if x > 0
 *           { alpha * exp(x), if x <= 0
 *
 * @tparam T The numeric type.
 */
template <typename T> class ELU : public Activation<T> {
  private:
    static constexpr T DefaultAlpha = static_cast<T>(1.0);
    T alpha_;

  public:
    /**
     * @brief Constructor with alpha parameter.
     * @param alpha The hyperparameter controlling the negative slope (default: 1.0).
     */
    explicit ELU(T alpha = DefaultAlpha);

    /**
     * @brief Apply ELU activation to the input tensor.
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node with ELU applied.
     */
    std::shared_ptr<ComputeNode<T>> forward(std::shared_ptr<ComputeNode<T>> input) override;

    /**
     * @brief Compute the gradient of ELU.
     * @param input The input compute node.
     * @param gradOutput The gradient from the next layer.
     * @return std::shared_ptr<ComputeNode<T>> The gradient with respect to the input.
     */
    std::shared_ptr<ComputeNode<T>> backward(std::shared_ptr<ComputeNode<T>> input,
                                              std::shared_ptr<ComputeNode<T>> gradOutput) override;
};

/**
 * @brief Convenience function to apply ELU activation.
 * @tparam T The numeric type.
 * @param input The input compute node.
 * @param alpha The hyperparameter controlling the negative slope (default: 1.0).
 * @return std::shared_ptr<ComputeNode<T>> The output compute node with ELU applied.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> elu(std::shared_ptr<ComputeNode<T>> input, T alpha = static_cast<T>(1.0));

} // namespace hahaha::ml

#endif // HAHAHA_ELU_H_C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R
