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

#ifndef INDEXCOLLECTOR_E6604779_0FB6_450C_A41B_EE2850AEEF72
#define INDEXCOLLECTOR_E6604779_0FB6_450C_A41B_EE2850AEEF72

#include <vector>

namespace hahaha::math {

template <typename T> class IndexCollector {
  public:
    IndexCollector() = default;
    IndexCollector operator,(T index) {
        indices.push_back(index);
        return *this;
    }

    std::vector<T>& getIndices() {
        return indices;
    }

  private:
    std::vector<T> indices;
};

} // namespace hahaha::math

#endif // INDEXCOLLECTOR_E6604779_0FB6_450C_A41B_EE2850AEEF72
