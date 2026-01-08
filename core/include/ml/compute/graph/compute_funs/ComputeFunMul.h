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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_MUL_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_MUL_H

#include "ComputeFunBroadcast.h"
#include "ComputeFunCommon.h"

namespace hahaha::compute {

// --- Multiplication ---

/**
 * @brief Elementwise multiplication with automatic broadcasting.
 *
 * Forward (plain text):
 * - z = x * y
 * - If shapes differ but are broadcast-compatible, x/y are first broadcast to
 *   a common shape using `broadcastNodes(lhs, rhs)`.
 *
 * Backward (plain text):
 * - dz/dx = y, dz/dy = x
 * - dL/dx += dL/dz * y
 * - dL/dy += dL/dz * x
 *
 * Notes:
 * - If broadcasting happened, the `Broadcast` nodes handle gradient reduction
 *   (summing along broadcasted axes).
 *
 * @tparam T Numeric type.
 * @param lhs Left operand node.
 * @param rhs Right operand node.
 * @return Result node representing z = lhs * rhs.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
mul(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto [bLhs, bRhs] = broadcastNodes(lhs, rhs);

    auto resData = std::make_shared<math::TensorWrapper<T>>(
        bLhs->getData()->multiply(*bRhs->getData()));
    broadcastNodeStrideRebuild(bLhs, resData);

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        bLhs, bRhs, resData, common::Operator::Mul, nullptr);

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
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(*rhs->getData()));
                lhs->accumulateGrad(gradLhs);
            }
            if (rhs->getRequiresGrad()) {
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(*lhs->getData()));
                rhs->accumulateGrad(gradRhs);
            }
        }
    });

    return resNode;
}

/**
 * @brief Multiply a tensor node by a scalar (lhs * scalar).
 *
 * Plain text:
 * - Convert scalar to a scalar ComputeNode on the same device, then call
 *   the tensor-tensor `mul`.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>> mul(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    return mul(lhs, createScalarNode(rhsScalar, lhs));
}

/**
 * @brief Multiply a scalar with a tensor node (scalar * rhs).
 *
 * Plain text:
 * - Multiplication is commutative, so this forwards to (rhs * scalar).
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
mul(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return mul(rhs, lhsScalar); // Commutative
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_MUL_H
