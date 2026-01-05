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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_DIV_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_DIV_H

#include "ComputeFunBroadcast.h"
#include "ComputeFunCommon.h"

namespace hahaha::compute {

// --- Division ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
div(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto [bLhs, bRhs] = broadcastNodes(lhs, rhs);

    auto resData = std::make_shared<math::TensorWrapper<T>>(
        bLhs->getData()->divide(*bRhs->getData()));

    broadcastNodeStrideRebuild(bLhs, resData);

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        bLhs, bRhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = bLhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = bRhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            auto rhsData = rhs->getData();
            if (lhs->getRequiresGrad()) {
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->divide(*rhsData));
                lhs->accumulateGrad(gradLhs);
            }
            if (rhs->getRequiresGrad()) {
                auto lhsData = lhs->getData();
                auto rhsDataSquare = rhsData->multiply(*rhsData);
                auto negLhsData = -(*lhsData);
                auto localGrad = negLhsData.divide(rhsDataSquare);
                broadcastNodeStrideRebuild(localGrad);
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(localGrad));
                rhs->accumulateGrad(gradRhs);
            }
        }
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> div(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    return div(lhs, createScalarNode(rhsScalar, lhs));
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
div(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return div(createScalarNode(lhsScalar, rhs), rhs);
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_DIV_H
