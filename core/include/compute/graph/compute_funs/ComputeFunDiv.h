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

#include "ComputeFunCommon.h"

namespace hahaha::compute {

// --- Division ---

template <typename T>
std::shared_ptr<ComputeNode<T>>
div(const std::shared_ptr<ComputeNode<T>>& lhs,
    const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->divide(*rhs->getData()));

    std::shared_ptr<ComputeNode<T>> resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

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
                if (lhs->getData()->getTotalSize() == 1
                    && res->getData()->getTotalSize() > 1) {
                    auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                        math::TensorShape({}),
                        gradLhs->sum(),
                        lhs->getData()->getDevice());
                    lhs->accumulateGrad(scalarGrad);
                } else {
                    lhs->accumulateGrad(gradLhs);
                }
                // lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                auto lhsData = lhs->getData();
                auto rhsDataSquare = rhsData->multiply(*rhsData);
                auto negLhsData = -(*lhsData);
                auto localGrad = negLhsData.divide(rhsDataSquare);
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(localGrad));
                if (rhs->getData()->getTotalSize() == 1
                    && res->getData()->getTotalSize() > 1) {
                    auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                        math::TensorShape({}),
                        gradRhs->sum(),
                        rhs->getData()->getDevice());
                    rhs->accumulateGrad(scalarGrad);
                } else {
                    rhs->accumulateGrad(gradRhs);
                }
                // rhs->backward();
            }
        }
    });

    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>> div(const std::shared_ptr<ComputeNode<T>>& lhs,
                                    const T& rhsScalar) {
    auto rhs = createScalarNode(rhsScalar, lhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        lhs->getData()->divide(rhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes, rhsScalar]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            if (lhs->getRequiresGrad()) {
                auto gradLhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->divide(rhsScalar));
                lhs->accumulateGrad(gradLhs);
                //lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // d(x/s)/ds = -x/s^2
                auto rhsSquareData = rhsScalar * rhsScalar;
                auto negLhsData = -(*(lhs->getData()));
                auto localGrad = negLhsData.divide(rhsSquareData);
                auto scalarGradVal = gradPtr->multiply(localGrad).sum();
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    scalarGradVal,
                    rhs->getData()->getDevice());
                rhs->accumulateGrad(scalarGrad);
                //rhs->backward();
            }
        }
    });
    return resNode;
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
div(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    auto lhs = createScalarNode(lhsScalar, rhs);
    auto resData = std::make_shared<math::TensorWrapper<T>>(
        rhs->getData()->divideInto(lhsScalar));

    auto resNode = std::make_shared<ComputeNode<T>>(
        lhs, rhs, resData, common::Operator::Div, nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakLhs = lhs;
    std::weak_ptr<ComputeNode<T>> weakRhs = rhs;

    resNode->setGradFun([weakLhs, weakRhs, weakRes, lhsScalar]() {
        auto res = weakRes.lock();
        auto lhs = weakLhs.lock();
        auto rhs = weakRhs.lock();
        if (res && lhs && rhs) {
            auto gradPtr = res->getGrad();
            auto rhsData = rhs->getData();
            if (lhs->getRequiresGrad()) {
                // d(s/x)/ds = 1/x
                auto localGrad = std::make_shared<math::TensorWrapper<T>>(
                                      math::TensorShape(rhsData->getShape()),
                                      T(1),
                                      rhsData->getDevice())
                                      ->divide(*rhsData);
                auto scalarGradVal = gradPtr->multiply(localGrad).sum();
                auto scalarGrad = std::make_shared<math::TensorWrapper<T>>(
                    math::TensorShape({}),
                    scalarGradVal,
                    lhs->getData()->getDevice());
                lhs->accumulateGrad(scalarGrad);
                //lhs->backward();
            }
            if (rhs->getRequiresGrad()) {
                // d(s/x)/dx = -s/x^2
                auto rhsSquareData = rhsData->multiply(*rhsData);
                auto negLhsData = -lhsScalar;
                auto localGrad = std::make_shared<math::TensorWrapper<T>>(
                                      math::TensorShape(rhsData->getShape()),
                                      negLhsData,
                                      rhsData->getDevice())
                                      ->divide(rhsSquareData);
                auto gradRhs = std::make_shared<math::TensorWrapper<T>>(
                    gradPtr->multiply(localGrad));
                rhs->accumulateGrad(gradRhs);
                //rhs->backward();
            }
        }
    });
    return resNode;
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_DIV_H

