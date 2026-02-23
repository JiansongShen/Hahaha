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

#ifndef HAHAHA_LEAKY_RELU_H_B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q
#define HAHAHA_LEAKY_RELU_H_B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q

#include "Activation.h"

namespace hahaha::ml {

/**
 * @brief Leaky Rectified Linear Unit (LeakyReLU) activation function.
 *
 * LeakyReLU is a variant of ReLU that allows a small, non-zero gradient
 * when the input is negative. This helps prevent the "dying ReLU" problem.
 *
 * Formula:
 *   f(x) = { x,      if x > 0
 *          { alpha * x, if x <= 0
 *
 * Where alpha is a small positive constant (typically 0.01).
 *
 * Gradient:
 *   f'(x) = { 1,      if x > 0
 *           { alpha,  if x <= 0
 *
 * @tparam T The numeric type.
 */
template <typename T> class LeakyReLU : public Activation<T> {
  private:
    static constexpr T DefaultAlpha = static_cast<T>(0.01);
    T alpha_;

  public:
    /**
     * @brief Constructor with alpha parameter.
     * @param alpha The negative slope coefficient (default: 0.01).
     */
    explicit LeakyReLU(T alpha = DefaultAlpha);

    /**
     * @brief Apply LeakyReLU activation to the input tensor.
     * @param input The input tensor.
     * @return TensorWrapper<T> The output tensor with LeakyReLU applied.
     */
    TensorWrapper<T> forward(TensorWrapper<T> input) override;

    /**
     * @brief Compute the gradient of LeakyReLU.
     * @param input The input tensor.
     * @param gradOutput The gradient from the next layer.
     * @return TensorWrapper<T> The gradient with respect to the input.
     */
    TensorWrapper<T> backward(TensorWrapper<T> input,
                               TensorWrapper<T> gradOutput) override;
};

/**
 * @brief Convenience function to apply LeakyReLU activation.
 * @tparam T The numeric type.
 * @param input The input tensor.
 * @param alpha The negative slope coefficient (default: 0.01).
 * @return TensorWrapper<T> The output tensor with LeakyReLU applied.
 */
template <typename T>
TensorWrapper<T> leakyRelu(TensorWrapper<T> input,
                           T alpha = static_cast<T>(0.01));

} // namespace hahaha::ml

#endif // HAHAHA_LEAKY_RELU_H_B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q
