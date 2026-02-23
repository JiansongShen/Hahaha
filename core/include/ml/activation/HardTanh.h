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

#ifndef HAHAHA_HARD_TANH_H_K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z
#define HAHAHA_HARD_TANH_H_K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z

#include "Activation.h"

namespace hahaha::ml {

/**
 * @brief Hard Tanh activation function.
 *
 * Hard Tanh is a piecewise linear approximation of the tanh function.
 * It is faster to compute and can be more efficient in some contexts.
 *
 * Formula:
 *   f(x) = { -1, if x <= -1
 *          { 1,  if x >= 1
 *          { x,  if -1 < x < 1
 *
 * Alternative form:
 *   f(x) = clip(x, -1, 1)
 *
 * Gradient:
 *   f'(x) = { 0, if x <= -1 or x >= 1
 *           { 1, if -1 < x < 1
 *
 * Properties:
 * - Piecewise linear approximation of tanh
 * - Faster to compute than tanh
 * - Output range: [-1, 1]
 * - Non-smooth at x = -1 and x = 1
 *
 * @tparam T The numeric type.
 */
template <typename T> class HardTanh : public Activation<T> {
  public:
    /**
     * @brief Apply hard tanh activation to the input tensor.
     * @param input The input tensor.
     * @return TensorWrapper<T> The output tensor with hard tanh applied.
     */
    TensorWrapper<T> forward(TensorWrapper<T> input) override;

    /**
     * @brief Compute the gradient of hard tanh.
     * @param input The input tensor.
     * @param gradOutput The gradient from the next layer.
     * @return TensorWrapper<T> The gradient with respect to the input.
     */
    TensorWrapper<T> backward(TensorWrapper<T> input,
                               TensorWrapper<T> gradOutput) override;
};

/**
 * @brief Convenience function to apply hard tanh activation.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @return TensorWrapper<T> The output tensor with hard tanh applied.
 */
template <typename T> TensorWrapper<T> hardTanh(TensorWrapper<T> input);

} // namespace hahaha::ml

#endif // HAHAHA_HARD_TANH_H_K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z
