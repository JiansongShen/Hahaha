// Copyright (c) $todat.year-2026 Contributors of Hahaha
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
// jiansongshen (https://github.com/jiansongshen) (jason.shen111@outlook.com)
//

#ifndef COMPUTEGRAPHUTIL_D5D0C6D3_CB84_44D7_84BC_E873CAF1DE5F
#define COMPUTEGRAPHUTIL_D5D0C6D3_CB84_44D7_84BC_E873CAF1DE5F
#include <memory>

#include "ComputeNode.h"
#include "ml/compute/graph/TopoSort.h"
#include "ml/compute/graph/cuda/ComputeGraphExecutor.h"
namespace hahaha::ml {
class ComputeGraphUtils {
public:
    template<typename T>
    static std::shared_ptr<ComputeNode<T>> compute(std::shared_ptr<ComputeNode<T>> node) {
        TopoSort<T> topoSort;
        auto nodeMap = topoSort.toTopoList(node);

        if (node->getData()->getDevice()->getType() == backend::DeviceType::CUDA) {
            ComputeGraphExecutor executor;
            executor.execute(node);
        }
    }
};
}

#endif // COMPUTEGRAPHUTIL_D5D0C6D3_CB84_44D7_84BC_E873CAF1DE5F
