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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_MAT_MUL_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_MAT_MUL_H

#include "math/TensorWrapper.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::compute {

// --- Matrix Multiplication ---

/**
 * @brief Matrix multiplication (2D) node: Z = X @ Y.
 *
 * Forward (plain text):
 * - Z[i,j] = sum_k X[i,k] * Y[k,j]
 *
 * Backward (plain text):
 * - dL/dX = dL/dZ @ transpose(Y)
 * - dL/dY = transpose(X) @ dL/dZ
 *
 * Notes:
 * - MatMul does not use broadcasting; shapes must be valid matrix dims.
 * - transpose() and matmul() are implemented on TensorWrapper.
 *
 * @tparam T Numeric type.
 * @param lhs Left matrix node (X).
 * @param rhs Right matrix node (Y).
 * @return Result node representing Z = X @ Y.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
matmul(const std::shared_ptr<ComputeNode<T>>& lhs,
       const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->matmul(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::MatMul, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto rhsT = rhs->getData()->transpose();
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->matmul(rhsT));
                lhs->accumulateGrad(gradLhs);
                // lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto lhsT = lhs->getData()->transpose();
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    lhsT.matmul(*gradPtr));
                rhs->accumulateGrad(gradRhs);
                // rhs->backward();
            }
        }
    });

    return resNode;
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_MAT_MUL_H
