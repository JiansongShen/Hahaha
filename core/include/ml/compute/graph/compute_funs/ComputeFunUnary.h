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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_UNARY_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_UNARY_H

#include <memory>

#include "ComputeFunCommon.h"
#include "math/TensorWrapper.h"

namespace hahaha::compute {

// --- Unary Operations ---

/**
 * @brief Reshape operation for computational graph nodes.
 * @tparam T The numeric type.
 * @param parent The parent compute node to reshape.
 * @param newShape The new shape for the tensor.
 * @return std::shared_ptr<ComputeNode<T>> A new compute node representing the
 * reshape operation.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
reshape(const std::shared_ptr<ComputeNode<T>>& parent,
        const std::vector<size_t>& newShape) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        parent->getData()->reshape(newShape));

    std::shared_ptr<ComputeNode<T>> resNode =
        ComputeNode<T>::createUnary(parent, resData, common::Operator::Reshape);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakParent = parent;
    auto originalShape = parent->getData()->getShape();

    resNode->setGradFun([weakParent, weakRes, originalShape]() {
        auto res = weakRes.lock();
        auto p = weakParent.lock();
        if (res && p) {
            auto gradPtr = res->getGrad();
            if (p->getRequiresGrad()) {
                auto reshapedGrad = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->reshape(originalShape));
                p->accumulateGrad(reshapedGrad);
                // p->backward();
            }
        }
    });
    return resNode;
}

/**
 * @brief Transpose operation for computational graph nodes.
 * @tparam T The numeric type.
 * @param parent The parent compute node to transpose.
 * @return std::shared_ptr<ComputeNode<T>> A new compute node representing the
 * transpose operation.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
transpose(const std::shared_ptr<ComputeNode<T>>& parent) {
    auto resData =
        std::make_shared<math::TensorWrapper<T>>(parent->getData()->transpose());

    std::shared_ptr<ComputeNode<T>> resNode =
        ComputeNode<T>::createUnary(parent, resData, common::Operator::Transpose);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakParent = parent;

    resNode->setGradFun([weakParent, weakRes]() {
        auto res = weakRes.lock();
        auto parent = weakParent.lock();
        if (res && parent) {
            auto gradPtr = res->getGrad();
            if (parent->getRequiresGrad()) {
                auto transposedGrad =
                    std::make_shared<math::TensorWrapper<T>>(gradPtr->transpose());
                parent->accumulateGrad(transposedGrad);
                // parent->backward();
            }
        }
    });
    return resNode;
}

/**
 * @brief Unary negation operation for computational graph nodes.
 * @tparam T The numeric type.
 * @param parent The parent compute node to negate.
 * @return std::shared_ptr<ComputeNode<T>> A new compute node representing the
 * negation operation.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> neg(const std::shared_ptr<ComputeNode<T>>& parent) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(-*parent->getData());
    auto resNode =
        ComputeNode<T>::createUnary(parent, resData, common::Operator::Neg);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakParent = parent;

    resNode->setGradFun([weakParent, weakRes]() {
        auto res = weakRes.lock();
        auto parent = weakParent.lock();
        auto gradPtr = res->getGrad();
        if (parent->getRequiresGrad()) {
            auto negatedGrad = std::make_shared<math::TensorWrapper<T>>(-*gradPtr);
            parent->accumulateGrad(negatedGrad);
        }
    });
    return resNode;
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_UNARY_H
