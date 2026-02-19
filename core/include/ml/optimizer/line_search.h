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
#ifndef LINE_SEARCH_CEE4A3B3_DA15_4257_9AC3_AADA33509FCA
#define LINE_SEARCH_CEE4A3B3_DA15_4257_9AC3_AADA33509FCA

#include <functional>

#include "math/TensorWrapper.h"

namespace hahaha::ml {

/**
 * @brief Perform line search to find the step size for the given direction. Using
 * wolfe condition.
 */
template <typename T>
T line_search(const math::TensorWrapper<T>& x,
              const math::TensorWrapper<T>& g,
              const math::TensorWrapper<T>& p,
              const std::vector<math::TensorWrapper<T>*>& paramWrappers,
              const std::vector<size_t>& paramSizes,
              T initialStepSize,
              T lineSearchTolerance,
              int maxLineSearchIterations,
              std::function<T()> loss_function) {
    T alpha = initialStepSize;
    long currIterator{0};

    while (currIterator < maxLineSearchIterations) {
        // Evaluate the loss function at x + alpha * p
        math::TensorWrapper<T> x_plus_alpha_p = x + alpha * p;
        // T loss = loss_function(x_plus_alpha_p, paramWrappers, paramSizes);

        // Check the Wolfe conditions
    }

    return alpha;
}

} // namespace hahaha::ml

#endif // LINE_SEARCH_CEE4A3B3_DA15_4257_9AC3_AADA33509FCA
