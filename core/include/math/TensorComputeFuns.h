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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#ifndef HAHAHA_MATH_TENSOR_COMPUTE_FUN_H
#define HAHAHA_MATH_TENSOR_COMPUTE_FUN_H

#include <cmath>

#include "TensorWrapper.h"

namespace hahaha::math {


    // TODO: Support cuda
/**
 * @brief Collection of element-wise mathematical functions for TensorWrapper.
 *
 * This class provides static methods to perform common mathematical operations
 * on tensors, supporting both out-of-place (returning new tensor) and
 * in-place modifications.
 */
class TensorComputeFun {
  public:
    /**
     * @brief Computes the element-wise square root of the input tensor.
     *
     * Formula:
     *     y[i] = sqrt(x[i])
     *
     * @tparam T The numeric type of the tensor.
     * @param input The input tensor x.
     * @return TensorWrapper<T> A new tensor containing the square root values.
     */
    template <typename T>
    static TensorWrapper<T> sqrt(TensorWrapper<T>& input) {
        TensorWrapper<T> res;
        res = input.clone();
        for (size_t i = 0; i < res.getTotalSize(); ++i) {
            res.getRawData()[i] = std::sqrt(input.getRawData()[i]);
        }
        return res;
    }

    /**
     * @brief Computes the element-wise square root in-place.
     *
     * Formula:
     *     x[i] = sqrt(x[i])
     *
     * @tparam T The numeric type of the tensor.
     * @param input The input tensor to modify x.
     * @return TensorWrapper<T> Reference to the modified input tensor.
     */
    template <typename T>
        static TensorWrapper<T> sqrtInPlace(TensorWrapper<T>& input) {
        for (size_t i = 0; i < input.getTotalSize(); ++i) {
            input.getRawData()[i] = std::sqrt(input.getRawData()[i]);
        }
        return input;
    }

    /**
     * @brief Computes the element-wise square of the input tensor.
     *
     * Formula:
     *     y[i] = x[i]^2
     *
     * @tparam T The numeric type of the tensor.
     * @param input The input tensor x.
     * @return TensorWrapper<T> A new tensor containing the squared values.
     */
    template <typename T>
    static TensorWrapper<T> square(TensorWrapper<T>& input) {
        return input * input;
    }

    /**
     * @brief Computes the element-wise square in-place.
     *
     * Formula:
     *     x[i] = x[i]^2
     *
     * @tparam T The numeric type of the tensor.
     * @param input The input tensor to modify x.
     * @return TensorWrapper<T> Reference to the modified input tensor.
     */
    template <typename T>
    static TensorWrapper<T> squareInPlace(TensorWrapper<T>& input) {
        input *= input;
        return input;
    }
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_COMPUTE_FUN_H
