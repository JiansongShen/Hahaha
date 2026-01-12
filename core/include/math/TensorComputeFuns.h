//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef HAHAHA_MATH_TENSOR_COMPUTE_FUN_H
#define HAHAHA_MATH_TENSOR_COMPUTE_FUN_H

#include <cmath>

#include "TensorWrapper.h"

namespace hahaha::math {
class TensorComputeFun {
  public:
    template <typename T>
    static TensorWrapper<T> sqrt(TensorWrapper<T>& input) {
        TensorWrapper<T> res;
        res = input.clone();
        for (size_t i = 0; i < res.getTotalSize(); ++i) {
            res.getRawData()[i] = std::sqrt(input.getRawData()[i]);
        }
        return res;
    }

    template <typename T>
    static TensorWrapper<T> square(TensorWrapper<T>& input) {
        return input * input;
    }
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_COMPUTE_FUN_H
