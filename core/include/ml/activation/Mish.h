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

#ifndef HAHAHA_MISH_H_L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z6A
#define HAHAHA_MISH_H_L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z6A

#include "Activation.h"

namespace hahaha::ml {

/**
 * @brief Mish activation function.
 *
 * Mish is a smooth, non-monotonic activation function that has been shown
 * to work well in various deep learning tasks. It is self-regularized and
 * helps with gradient flow.
 *
 * Formula:
 *   f(x) = x * tanh(softplus(x))
 *        = x * tanh(log(1 + exp(x)))
 *
 * Where softplus(x) = log(1 + exp(x)).
 *
 * Gradient:
 *   f'(x) = sech^2(softplus(x)) * x * sigmoid(x) + f(x) / x
 *
 * Properties:
 * - Smooth and differentiable everywhere
 * - Non-monotonic (can be negative for negative inputs)
 * - Self-regularized
 * - Bounded below but unbounded above
 * - Generally outperforms ReLU and Swish in many tasks
 *
 * @tparam T The numeric type.
 */
template <typename T> class Mish : public Activation<T> {
  public:
    /**
     * @brief Apply Mish activation to the input tensor.
     * @param input The input tensor.
     * @return TensorWrapper<T> The output tensor with Mish applied.
     */
    TensorWrapper<T> forward(TensorWrapper<T> input) override;

    /**
     * @brief Compute the gradient of Mish.
     * @param input The input tensor.
     * @param gradOutput The gradient from the next layer.
     * @return TensorWrapper<T> The gradient with respect to the input.
     */
    TensorWrapper<T> backward(TensorWrapper<T> input,
                              TensorWrapper<T> gradOutput) override;
};

/**
 * @brief Convenience function to apply Mish activation.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @return TensorWrapper<T> The output tensor with Mish applied.
 */
template <typename T> TensorWrapper<T> mish(TensorWrapper<T> input);

} // namespace hahaha::ml

#endif // HAHAHA_MISH_H_L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z6A
