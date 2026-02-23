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

#ifndef HAHAHA_ACTIVATION_H_8A9B0C1D2E3F4G5H6I7J8K9L0M1N2O
#define HAHAHA_ACTIVATION_H_8A9B0C1D2E3F4G5H6I7J8K9L0M1N2O

#include "ml/compute/graph/ComputeNode.h"
#include <memory>

namespace hahaha::ml {

/**
 * @brief Base class for all activation functions.
 *
 * Activation functions introduce non-linearity into neural networks,
 * allowing them to learn complex patterns. This base class provides the
 * interface that all activation functions must implement.
 *
 * @tparam T The numeric type.
 */
template <typename T> class Activation {
  public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Activation() = default;

    /**
     * @brief Apply the activation function to the input tensor.
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node after applying activation.
     */
    virtual std::shared_ptr<ComputeNode<T>> forward(std::shared_ptr<ComputeNode<T>> input) = 0;

    /**
     * @brief Compute the gradient of the activation function.
     * @param input The input compute node.
     * @param gradOutput The gradient from the next layer.
     * @return std::shared_ptr<ComputeNode<T>> The gradient with respect to the input.
     */
    virtual std::shared_ptr<ComputeNode<T>> backward(std::shared_ptr<ComputeNode<T>> input,
                                                      std::shared_ptr<ComputeNode<T>> gradOutput) = 0;

    /**
     * @brief Apply activation function (operator overload).
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node after applying activation.
     */
    std::shared_ptr<ComputeNode<T>> operator()(std::shared_ptr<ComputeNode<T>> input) {
        return forward(input);
    }
};

} // namespace hahaha::ml

#endif // HAHAHA_ACTIVATION_H_8A9B0C1D2E3F4G5H6I7J8K9L0M1N2O
