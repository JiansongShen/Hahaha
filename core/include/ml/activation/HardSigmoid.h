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

#ifndef HAHAHA_HARD_SIGMOID_H_J0K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y
#define HAHAHA_HARD_SIGMOID_H_J0K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y

#include "Activation.h"

namespace hahaha::ml {

/**
 * @brief Hard Sigmoid activation function.
 *
 * Hard Sigmoid is a piecewise linear approximation of the sigmoid function.
 * It is faster to compute and can be more efficient in some contexts while
 * maintaining similar properties to sigmoid.
 *
 * Formula:
 *   f(x) = { 0,              if x <= -3
 *          { 1,              if x >= 3
 *          { (x / 6) + 0.5, if -3 < x < 3
 *
 * Alternative form:
 *   f(x) = clip((x / 6) + 0.5, 0, 1)
 *
 * Gradient:
 *   f'(x) = { 0,    if x <= -3 or x >= 3
 *           { 1/6,  if -3 < x < 3
 *
 * Properties:
 * - Piecewise linear approximation of sigmoid
 * - Faster to compute than sigmoid
 * - Output range: [0, 1]
 * - Non-smooth at x = -3 and x = 3
 *
 * @tparam T The numeric type.
 */
template <typename T> class HardSigmoid : public Activation<T> {
  public:
    /**
     * @brief Apply hard sigmoid activation to the input tensor.
     * @param input The input tensor.
     * @return TensorWrapper<T> The output tensor with hard sigmoid applied.
     */
    TensorWrapper<T> forward(TensorWrapper<T> input) override;

    /**
     * @brief Compute the gradient of hard sigmoid.
     * @param input The input tensor.
     * @param gradOutput The gradient from the next layer.
     * @return TensorWrapper<T> The gradient with respect to the input.
     */
    TensorWrapper<T> backward(TensorWrapper<T> input,
                                TensorWrapper<T> gradOutput) override;
};

/**
 * @brief Convenience function to apply hard sigmoid activation.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @return TensorWrapper<T> The output tensor with hard sigmoid applied.
 */
template <typename T> TensorWrapper<T> hardSigmoid(TensorWrapper<T> input);

} // namespace hahaha::ml

#endif // HAHAHA_HARD_SIGMOID_H_J0K1L2M3N4O5P6Q7R8S9T0U1V2W3X4Y
