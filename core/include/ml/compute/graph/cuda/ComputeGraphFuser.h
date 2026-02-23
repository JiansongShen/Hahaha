// Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// JiansongShen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#ifndef COMPUTEGRAPHFUSER_D50FBF95_3F53_4F0B_B8F5_2347CDA5F576
#define COMPUTEGRAPHFUSER_D50FBF95_3F53_4F0B_B8F5_2347CDA5F576

#include <memory>
#include <vector>

#include "ml/compute/graph/ComputeNode.h"
#include "ml/compute/graph/cuda/CudaComputeNode.h"

namespace hahaha::ml {

/**
 * @brief ComputeGraphFuser is responsible for fusing multiple compute nodes into a single
 * GPU kernel.
 *
 * The ComputeGraphFuser analyzes the compute graph and identifies opportunities to fuse
 * multiple operations together. This can reduce memory bandwidth usage and improve
 * performance by minimizing intermediate data transfers between GPU kernels.
 */
class ComputeGraphFuser {

  public:
        ComputeGraphFuser() = default;
    
        /**
        * @brief Fuse a list of compute nodes into a single GPU kernel.
        * @tparam T The numeric type used in the compute nodes.
        * @param nodes The list of compute nodes to fuse.
        * @return A new compute node representing the fused operations.
        */
        template <typename T>
        std::vector<CudaComputeNode> fuse(const std::vector<std::shared_ptr<ComputeNode<T>>>& nodes) {
            std::vector<CudaComputeNode> fusedNodes;
            return fusedNodes;
        }

};
} // namespace hahaha::ml

#endif // COMPUTEGRAPHFUSER_D50FBF95_3F53_4F0B_B8F5_2347CDA5F576
