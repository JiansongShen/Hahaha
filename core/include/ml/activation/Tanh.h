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

#ifndef HAHAHA_TANH_H_F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U
#define HAHAHA_TANH_H_F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U

#include "Activation.h"

namespace hahaha::ml {

/**
 * @brief Hyperbolic Tangent (Tanh) activation function.
 *
 * Tanh is a smooth, S-shaped activation function that maps any real number
 * to a value between -1 and 1. It is zero-centered, which can help with
 * training compared to sigmoid.
 *
 * Formula:
 *   f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * Alternative form:
 *   f(x) = (exp(2x) - 1) / (exp(2x) + 1)
 *
 * Relationship to sigmoid:
 *   tanh(x) = 2 * sigmoid(2x) - 1
 *
 * Gradient:
 *   f'(x) = 1 - tanh^2(x) = sech^2(x)
 *
 * Properties:
 * - Output range: (-1, 1)
 * - Zero-centered
 * - Smooth and differentiable everywhere
 * - Can still suffer from vanishing gradient problem
 *
 * @tparam T The numeric type.
 */
template <typename T> class Tanh : public Activation<T> {
  public:
    /**
     * @brief Apply tanh activation to the input tensor.
     * @param input The input tensor.
     * @return TensorWrapper<T> The output tensor with tanh applied.
     */
    TensorWrapper<T> forward(TensorWrapper<T> input) override;

    /**
     * @brief Compute the gradient of tanh.
     * @param input The input tensor.
     * @param gradOutput The gradient from the next layer.
     * @return TensorWrapper<T> The gradient with respect to the input.
     */
    TensorWrapper<T> backward(TensorWrapper<T> input,
                               TensorWrapper<T> gradOutput) override;
};

/**
 * @brief Convenience function to apply tanh activation.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @return TensorWrapper<T> The output tensor with tanh applied.
 */
template <typename T> TensorWrapper<T> tanh(TensorWrapper<T> input);

} // namespace hahaha::ml

#endif // HAHAHA_TANH_H_F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U
