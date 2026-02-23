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

#ifndef HAHAHA_SWISH_H_G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V
#define HAHAHA_SWISH_H_G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V

#include "Activation.h"

namespace hahaha::ml {

/**
 * @brief Swish (also known as SiLU - Sigmoid Linear Unit) activation function.
 *
 * Swish is a smooth, non-monotonic activation function that has been shown
 * to outperform ReLU in many deep learning tasks. It is self-gated, meaning
 * the gate value depends on the input itself.
 *
 * Formula:
 *   f(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Alternative form:
 *   f(x) = x * (1 / (1 + exp(-x)))
 *
 * Gradient:
 *   f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 *         = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 *
 * Properties:
 * - Smooth and differentiable everywhere
 * - Non-monotonic (can be negative for negative inputs)
 * - Bounded below but unbounded above
 * - Self-gated activation
 *
 * @tparam T The numeric type.
 */
template <typename T> class Swish : public Activation<T> {
  public:
    /**
     * @brief Apply Swish activation to the input tensor.
     * @param input The input tensor.
     * @return TensorWrapper<T> The output tensor with Swish applied.
     */
    TensorWrapper<T> forward(TensorWrapper<T> input) override;

    /**
     * @brief Compute the gradient of Swish.
     * @param input The input tensor.
     * @param gradOutput The gradient from the next layer.
     * @return TensorWrapper<T> The gradient with respect to the input.
     */
    TensorWrapper<T> backward(TensorWrapper<T> input,
                              TensorWrapper<T> gradOutput) override;
};

/**
 * @brief Convenience function to apply Swish activation.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @return TensorWrapper<T> The output tensor with Swish applied.
 */
template <typename T> TensorWrapper<T> swish(TensorWrapper<T> input);

/**
 * @brief SiLU (Sigmoid Linear Unit) - alias for Swish.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @return TensorWrapper<T> The output tensor with SiLU applied.
 */
template <typename T> TensorWrapper<T> silu(TensorWrapper<T> input);

} // namespace hahaha::ml

#endif // HAHAHA_SWISH_H_G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U1V
