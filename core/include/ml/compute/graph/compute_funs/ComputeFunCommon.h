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

#ifndef HAHAHA_COMPUTE_COMPUTE_FUN_COMMON_H
#define HAHAHA_COMPUTE_COMPUTE_FUN_COMMON_H

#include <memory>

#include "math/TensorWrapper.h"
#include "math/ds/TensorData.h"
#include "ml/compute/graph/ComputeNode.h"

namespace hahaha::compute {

/**
 * @brief Helper to create a constant scalar node on the same device as a
 * reference node.
 *
 * Plain-text semantics:
 * - Creates a leaf node holding a scalar tensor (totalSize == 1) on the same
 *   device as `refNode`.
 * - This allows binary ops to treat scalar inputs uniformly as ComputeNodes.
 *
 * Note:
 * - The scalar is represented as a 0-dim shape (TensorShape({})) in this code
 *   base.
 *
 * @tparam T Numeric type.
 * @param value Scalar value.
 * @param refNode Reference node whose device is used.
 * @return Leaf ComputeNode containing the scalar tensor.
 */
template <typename T>
std::shared_ptr<ComputeNode<T>>
createScalarNode(const T& value, const std::shared_ptr<ComputeNode<T>>& refNode) {
    auto scalarWrapper = std::make_shared<math::TensorWrapper<T>>(
        math::TensorShape({}), value, refNode->getData()->getDevice());
    return std::make_shared<ComputeNode<T>>(scalarWrapper);
}

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_FUN_COMMON_H
