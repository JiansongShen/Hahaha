// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_REDUCTION_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_REDUCTION_H

#include <memory>
#include <vector>
#include <numeric>

#include "ComputeFunCommon.h"
#include "ComputeFunDiv.h"
#include "math/TensorWrapper.h"

namespace hahaha::ml {

/**
 * @brief Summation operation along specified axes.
 * @tparam T The numeric type.
 * @param parent The parent compute node.
 * @param axes The axes to sum over.
 * @param keepDims Whether to keep the reduced dimensions (as size 1).
 * @return std::shared_ptr<ComputeNode<T>> A new compute node representing the sum operation.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> sum(const std::shared_ptr<ComputeNode<T>>& parent,
                                    const std::vector<size_t>& axes,
                                    bool keepDims = false) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        parent->getData()->sum(axes, keepDims));

    auto resNode = std::make_shared<ComputeNode<T>>(
        resData, common::Operator::Sum, nullptr);
    resNode->addParent(parent);
    resNode->setRequiresGrad(parent->getRequiresGrad());

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakParent = parent;
    auto inputShape = parent->getData()->getShapeVecRef();

    resNode->setGradFun([weakParent, weakRes, axes, keepDims, inputShape]() {
        auto res = weakRes.lock();
        auto parent = weakParent.lock();
        if (res && parent && parent->getRequiresGrad()) {
            auto gradOutput = res->getGrad();
            
            std::shared_ptr<math::TensorWrapper<T>> gradOutputReshaped;
            if (!keepDims) {
                // We need to restore the reduced dimensions as 1s
                std::vector<size_t> expandedShape = inputShape;
                for (auto axis : axes) {
                    if (axis < expandedShape.size()) {
                        expandedShape[axis] = 1;
                    }
                }
                
                gradOutputReshaped = std::make_shared<math::TensorWrapper<T>>(
                    gradOutput->reshape(expandedShape));
            } else {
                gradOutputReshaped = gradOutput;
            }
            
            // Broadcast to input shape
            auto gradInput = std::make_shared<math::TensorWrapper<T>>(
                gradOutputReshaped->broadcastTo(math::TensorShape(inputShape)));
                
            parent->accumulateGrad(gradInput);
        }
    });

    return resNode;
}

/**
 * @brief Mean operation along specified axes.
 * @tparam T The numeric type.
 * @param parent The parent compute node.
 * @param axes The axes to reduce over.
 * @param keepDims Whether to keep the reduced dimensions (as size 1).
 * @return std::shared_ptr<ComputeNode<T>> A new compute node representing the mean operation.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> mean(const std::shared_ptr<ComputeNode<T>>& parent,
                                     const std::vector<size_t>& axes,
                                     bool keepDims = false) {
    auto inputShape = parent->getData()->getShapeVecRef();
    size_t N = 1;
    for (auto axis : axes) {
        if (axis < inputShape.size()) {
            N *= inputShape[axis];
        }
    }
    
    auto sumNode = sum(parent, axes, keepDims);
    
    // Use div(node, scalar)
    return div(sumNode, static_cast<T>(N));
}

} // namespace hahaha::ml

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_REDUCTION_H
