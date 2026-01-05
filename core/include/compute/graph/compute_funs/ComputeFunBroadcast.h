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

#ifndef HAHAHA_COMPUTEFUNBROADCAST_H_CA2A2F550C26484097AE9BAB15204C65
#define HAHAHA_COMPUTEFUNBROADCAST_H_CA2A2F550C26484097AE9BAB15204C65

#include "ComputeFunCommon.h"

namespace hahaha::compute {

using math::TensorShape;
template <typename T>
void checkTensorCanBroadcastTo(
    const std::shared_ptr<ComputeNode<T>>& sourceNode,
    const std::vector<size_t>& targetTensorShape);

template <typename T>
std::shared_ptr<ComputeNode<T>>
broadcast(const std::shared_ptr<ComputeNode<T>>& sourceNode,
          const std::vector<size_t>& targetTensorShape) {

    checkTensorCanBroadcastTo(sourceNode, targetTensorShape);

    auto resNode = ComputeNode<T>::createUnary(
        sourceNode,
        std::make_shared<math::TensorWrapper<T>>(
            sourceNode->getData()->broadcastTo(TensorShape(targetTensorShape))),
        common::Operator::Broadcast,
        nullptr);

    std::weak_ptr<ComputeNode<T>> weakRes = resNode;
    std::weak_ptr<ComputeNode<T>> weakSrc = sourceNode;

    resNode->setGradFun([weakSrc, weakRes] {
        auto res = weakRes.lock();
        auto src = weakSrc.lock();
        if (res && src) {
            std::shared_ptr<math::TensorWrapper<T>> resData = res->getData();
            std::shared_ptr<math::TensorWrapper<T>> srcData = src->getData();
            // detect shape different
            std::vector<size_t> sumAxes;
            const std::vector<size_t> srcShape = srcData->getShape();
            const std::vector<size_t> resShape = resData->getShape();
            bool keepDims = resShape.size() == srcShape.size();

            // srcIdx must smaller than resIdx
            long srcIdx = static_cast<long>(srcShape.size() - 1);
            long resIdx = static_cast<long>(resShape.size() - 1);

            while (srcIdx >= 0 && resIdx >= 0) {
                if (srcShape[srcIdx] != resShape[resIdx]) {
                    sumAxes.push_back(resIdx);
                }
                --srcIdx;
                --resIdx;
            }

            while (resIdx >= 0) {
                sumAxes.push_back(resIdx);
                --resIdx;
            }

            auto grad = res->getGrad();
            if (!grad) {
                return;
            }
            auto srcGrad = grad->sum(sumAxes, keepDims);
            src->accumulateGrad(std::make_shared<math::TensorWrapper<T>>(std::move(srcGrad)));
        }
    });

    return resNode;
}

template <typename T>
void checkTensorCanBroadcastTo(
    const std::shared_ptr<ComputeNode<T>>& sourceNode,
    const std::vector<size_t>& targetTensorShape) {

    const auto srcShape = TensorShape(sourceNode->getData()->getShape());
    const TensorShape targetShape(targetTensorShape);
    const auto broadcastShape =
        TensorShape::broadcastShape(srcShape, targetShape);
    if (!broadcastShape.has_value()
        || broadcastShape.value() != targetTensorShape) {
        std::string broadcastShapeStr = broadcastShape.has_value()
                                            ? TensorShape(broadcastShape.value()).toString()
                                            : "None";
        throw std::runtime_error(
            "broadcast shape mismatch: source tensor shape is "
            + srcShape.toString() + ", \n\tbut target tensor shape is "
            + targetShape.toString() + "\tdo you want to broadcast to :"
            + broadcastShapeStr);
    }
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
broadcast(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return broadcast(rhs, lhsScalar); // Commutative
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTEFUNBROADCAST_H_CA2A2F550C26484097AE9BAB15204C65
