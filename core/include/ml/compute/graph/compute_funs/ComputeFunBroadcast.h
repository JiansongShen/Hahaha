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

#include "math/TensorWrapper.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::ml {

using math::TensorShape;

/**
 * @brief Validate that `sourceNode` can be broadcast to `targetTensorShape`.
 *
 * Broadcasting rule (plain text):
 * Compare shapes from the last dimension to the first. For each dimension:
 * - If equal: ok
 * - If one side is 1(or has no dimension (like {1, 2} and {(no dimension) 2})):
 * ok (that side can be expanded)
 * - Otherwise: incompatible
 *
 * The broadcast result shape is the per-dimension max after applying the rule.
 * This helper additionally enforces that the broadcast result equals
 * `targetTensorShape` (i.e. we are broadcasting *to* the requested shape).
 *
 * @tparam T Numeric type.
 * @param sourceNode Source node.
 * @param targetTensorShape Target shape (vector form).
 * @throws std::runtime_error if broadcasting is not possible or target is not
 *         the broadcast result.
 */
template <typename T>
void checkTensorCanBroadcastTo(const std::shared_ptr<ComputeNode<T>>& sourceNode,
                               const std::vector<size_t>& targetTensorShape);

/**
 * @brief Broadcast a node to a target shape (view semantics).
 *
 * Forward (plain text):
 * - The output is a view of the same underlying storage.
 * - Broadcasted dimensions are represented by stride = 0, so repeated reads
 *   map to the same source element.
 *
 * Backward (plain text):
 * - Let y = broadcast(x) with y.shape = target.
 * - Upstream gradient is dL/dy.
 * - Gradient for x is a reduction over broadcasted axes:
 *   dL/dx = sum(dL/dy, axes=broadcasted_axes, keepDims=rank_equal)
 *
 * Example:
 * - x.shape = (3), y.shape = (2,3)
 * - broadcasted_axes = {0}
 * - dL/dx[j] = dL/dy[0,j] + dL/dy[1,j]
 *
 * @tparam T Numeric type.
 * @param sourceNode Source node.
 * @param targetTensorShape Target shape.
 * @return New compute node representing broadcast view.
 */
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
            src->accumulateGrad(
                std::make_shared<math::TensorWrapper<T>>(std::move(srcGrad)));
        }
    });

    return resNode;
}

template <typename T>
void checkTensorCanBroadcastTo(const std::shared_ptr<ComputeNode<T>>& sourceNode,
                               const std::vector<size_t>& targetTensorShape) {

    const auto srcShape = TensorShape(sourceNode->getData()->getShape());
    const TensorShape targetShape(targetTensorShape);
    const auto broadcastShape = TensorShape::broadcastShape(srcShape, targetShape);
    if (!broadcastShape.has_value() || broadcastShape.value() != targetTensorShape) {
        std::string broadcastShapeStr = broadcastShape.has_value()
            ? TensorShape(broadcastShape.value()).toString()
            : "None";
        throw std::runtime_error(
            "broadcast shape mismatch: source tensor shape is " + srcShape.toString()
            + ", \n\tbut target tensor shape is " + targetShape.toString()
            + "\tdo you want to broadcast to :" + broadcastShapeStr);
    }
}

/**
 * @brief Broadcast two nodes to a common compatible shape if needed.
 *
 * Plain text:
 * - target = broadcast_shape(lhs.shape, rhs.shape)
 * - return (broadcast(lhs, target) if needed, broadcast(rhs, target) if needed)
 *
 * This function is typically used as the first step of any elementwise binary
 * operator to support shape mismatch and scalar/tensor mixing.
 *
 * @tparam T Numeric type.
 * @param lhs LHS node.
 * @param rhs RHS node.
 * @return Pair of nodes with identical shapes.
 * @throws std::invalid_argument if shapes are not broadcast-compatible.
 */
template <typename T>
std::pair<std::shared_ptr<ComputeNode<T>>, std::shared_ptr<ComputeNode<T>>>
broadcastNodes(const std::shared_ptr<ComputeNode<T>>& lhs,
               const std::shared_ptr<ComputeNode<T>>& rhs) {
    const auto lhsShape = TensorShape(lhs->getData()->getShape());
    const auto rhsShape = TensorShape(rhs->getData()->getShape());

    if (lhsShape == rhsShape) {
        return {lhs, rhs};
    }

    const auto commonShape = TensorShape::broadcastShape(lhsShape, rhsShape);
    if (!commonShape.has_value()) {
        throw std::invalid_argument("Shapes are not broadcast-compatible: "
                                    + lhsShape.toString() + " and "
                                    + rhsShape.toString());
    }

    auto targetShape = commonShape.value();
    auto newLhs = lhs;
    auto newRhs = rhs;

    if (lhsShape.getDims() != targetShape) {
        newLhs = broadcast(lhs, targetShape);
    }
    if (rhsShape.getDims() != targetShape) {
        newRhs = broadcast(rhs, targetShape);
    }

    return {newLhs, newRhs};
}

template <typename T>
std::shared_ptr<ComputeNode<T>>
broadcast(const T& lhsScalar, const std::shared_ptr<ComputeNode<T>>& rhs) {
    return broadcast(rhs, lhsScalar); // Commutative
}

template <typename T>
void broadcastNodeStrideRebuild(std::shared_ptr<ComputeNode<T>> bLhs,
                                std::shared_ptr<math::TensorWrapper<T>> resData) {
    if (bLhs->getOperatorType() == common::Operator::Broadcast) {
        resData->setStride(math::TensorStride(resData->getShape()));
    }
}

template <typename T>
void broadcastNodeStrideRebuild(math::TensorWrapper<T>& resData) {
    resData.setStride(math::TensorStride(resData.getShape()));
}
} // namespace hahaha::compute

#endif // HAHAHA_COMPUTEFUNBROADCAST_H_CA2A2F550C26484097AE9BAB15204C65
