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

#ifndef COMPUTEGRAPHEXECUTOR_F9BE967D_3ED6_4366_AC61_18D55A3DCAEF
#define COMPUTEGRAPHEXECUTOR_F9BE967D_3ED6_4366_AC61_18D55A3DCAEF

#include <memory>
#include <vector>

#include "common/errors/Error.h"
#include "ml/compute/graph/cuda/CudaComputeNode.h"
#include "ml/compute/graph/ComputeNode.h"
#include "ml/compute/graph/cuda/ComputeGraphFuser.h"

namespace hahaha::ml {
class ComputeGraphExecutor {
public:

    /**
    this node is the target node to compute, like loss or something else,
    this method will execute all the nodes that this node depends on, and then execute this node.
    the result will be stored in this node.
    */
    template <typename T>
    void
    execute(std::shared_ptr<ComputeNode<T>> node) {
        // 1. Topologically sort the graph to get execution order
        TopoSort<T> topoSort;
        std::vector<std::shared_ptr<ComputeNode<T>>> nodeExecuteList =
            topoSort.toTopoList(node);

        ComputeGraphFuser fuser;
        std::vector<CudaComputeNode> fusedGpuNodes = fuser.fuse(nodeExecuteList);

        common::Error error = executeOnGpu(fusedGpuNodes);
    }

private:
    static common::Error
    executeOnGpu(const std::vector<CudaComputeNode>& node) {
        return common::Error::Success();
    }
};
} // namespace hahaha::ml

#endif // COMPUTEGRAPHEXECUTOR_F9BE967D_3ED6_4366_AC61_18D55A3DCAEF
