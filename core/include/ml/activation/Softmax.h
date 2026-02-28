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

#ifndef SOFTMAX_AC88256D_D448_410A_8562_EB1A9409C866
#define SOFTMAX_AC88256D_D448_410A_8562_EB1A9409C866

#include <memory>

#include "Activation.h"
#include "math/TensorWrapper.h"
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
    explicit Softmax(int dim = -1) : dim_(dim) {
    }

    /**
     * @brief Apply softmax activation to the input tensor.
     * @param input The input compute node.
     * @return std::shared_ptr<ComputeNode<T>> The output compute node with softmax
     * applied.
     */
    std::shared_ptr<ComputeNode<T>>
    forward(std::shared_ptr<ComputeNode<T>> input) override {
        // basic definition of softmax
        auto tensorPtr = input->getData();
        auto shapeVec = tensorPtr->getShapeVecRef();
        int dim = dim_;
        if (dim < 0) {
            dim += shapeVec.size();
            if (dim < 0 || dim >= static_cast<int>(shapeVec.size())) {
                throw std::invalid_argument("Dimension out of range");
            }
        }

        // find max value along the specified dimension for numerical stability
        auto maxVal = tensorPtr->max(dim, true);
        auto maxValBroadcasted = maxVal.broadcastTo(math::TensorShape(shapeVec));

        // sub the max value and exp
        auto tmp = tensorPtr->subtract(maxValBroadcasted);
        auto expVal = tmp.clone();
        expVal.expInPlace();

        auto sumExp = expVal.sum({static_cast<size_t>(dim)}, true);
        auto sumExpBroadcasted =
            sumExp.broadcastTo(math::TensorShape(expVal.getShapeVecRef()));

        // compute softmax
        auto outputTensor = expVal.divide(sumExpBroadcasted);

        auto outputNode = std::make_shared<ComputeNode<T>>(
            std::make_shared<math::TensorWrapper<T>>(outputTensor),
            common::Operator::Softmax);
        outputNode->addParent(input);

        if (input->getRequiresGrad()) {
            outputNode->setRequiresGrad(true);
            std::weak_ptr<ComputeNode<T>> weakOutput = outputNode;
            std::weak_ptr<ComputeNode<T>> weakInput = input;

            outputNode->setGradFun([weakInput, weakOutput, dim]() {
                auto outputNode = weakOutput.lock();
                auto input = weakInput.lock();
                if (!outputNode || !input)
                    return;

                auto gradOutput = outputNode->getGrad();
                if (!gradOutput)
                    return;

                auto outputTensorPtr = outputNode->getData(); // y

                // y * gradOutput
                auto y_grad = outputTensorPtr->multiply(*gradOutput);

                // sum(y * gradOutput)
                auto sum_y_grad = y_grad.sum({static_cast<size_t>(dim)}, true);

                // broadcast
                auto sum_y_grad_b = sum_y_grad.broadcastTo(
                    math::TensorShape(outputTensorPtr->getShapeVecRef()));

                // gradOutput - sum
                auto tmp_grad = gradOutput->subtract(sum_y_grad_b);

                // y * (gradOutput - sum)
                auto gradInput = outputTensorPtr->multiply(tmp_grad);

                input->accumulateGrad(
                    std::make_shared<math::TensorWrapper<T>>(gradInput));
            });
        }

        return outputNode;
    }

    /**
     * @brief Compute the gradient of softmax.
     * @param input The input compute node.
     * @param gradOutput The gradient from the next layer.
     * @return std::shared_ptr<ComputeNode<T>> The gradient with respect to the
     * input.
     */
    std::shared_ptr<ComputeNode<T>>
    backward(std::shared_ptr<ComputeNode<T>> input,
             std::shared_ptr<ComputeNode<T>> gradOutput) override {
        // Recompute softmax output for gradient calculation
        auto outputNode = forward(input);
        auto outputTensorPtr = outputNode->getData();

        auto tensorPtr = input->getData();
        auto shapeVec = tensorPtr->getShapeVecRef();
        int dim = dim_;
        if (dim < 0) {
            dim += shapeVec.size();
        }

        auto gradOutputTensor = gradOutput->getData();

        // y * gradOutput
        auto y_grad = outputTensorPtr->multiply(*gradOutputTensor);

        // sum(y * gradOutput)
        auto sum_y_grad = y_grad.sum({static_cast<size_t>(dim)}, true);

        // broadcast
        auto sum_y_grad_b = sum_y_grad.broadcastTo(
            math::TensorShape(outputTensorPtr->getShapeVecRef()));

        // gradOutput - sum
        auto tmp_grad = gradOutputTensor->subtract(sum_y_grad_b);

        // y * (gradOutput - sum)
        auto gradInput = outputTensorPtr->multiply(tmp_grad);

        return std::make_shared<ComputeNode<T>>(
            std::make_shared<math::TensorWrapper<T>>(gradInput));
    }
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
                                        int dim = -1) {
    return Softmax<T>(dim).forward(input);
}

} // namespace hahaha::ml

#endif // SOFTMAX_AC88256D_D448_410A_8562_EB1A9409C866
