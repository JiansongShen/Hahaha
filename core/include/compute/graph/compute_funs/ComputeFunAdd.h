// Copyright (c) ${original.year} - 2026 Contributors of
// Hahaha("https://github.com/Napbad/Hahaha")
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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_ADD_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_ADD_H

#include "ComputeFunBroadcast.h"
#include "ComputeFunCommon.h"

namespace hahaha::compute {

// --- Addition ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
add(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto [bLhs, bRhs] = broadcastNodes(lhs, rhs);

    auto resData = std::make_shared<math::TensorWrapper<T>>(
        bLhs->getData()->add(*bRhs->getData()));
    broadcastNodeStrideRebuild(bLhs, resData);

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        bLhs, bRhs, resData, common::Operator::Add, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = bLhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = bRhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                lhs->accumulateGrad(gradPtr);
            }
            if (rhs->getRequiresGrad()) {
                rhs->accumulateGrad(gradPtr);
            }
        }
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> add(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    return add(lhs, createScalarNode(rhsScalar, lhs));
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
add(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return add(rhs, lhsScalar); // Commutative
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_ADD_H
