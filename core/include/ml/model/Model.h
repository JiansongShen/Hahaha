// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
//  Contributors:
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef HAHAHA_MODEL_H_CC9C6DBC3C23404399CA13FD2F70E408
#define HAHAHA_MODEL_H_CC9C6DBC3C23404399CA13FD2F70E408
#include "ml/Parameters.h"
#include "ml/optimizer/SGDOptimizer.h"

namespace hahaha::ml {

template <typename T> class Model {

    static constexpr int DefaultMaxTrainIterations = 1000000;
    static constexpr T DefaultLearningRate = 0.001;

  public:
    Model() {
        optimizer_ = SGDOptimizer<T>({}, DefaultLearningRate);
    }
    virtual ~Model() = default;
    virtual Parameters<T> getParameters() = 0;
    virtual void train() = 0;
    virtual math::TensorWrapper<T> predict(math::TensorWrapper<T> x) = 0;
    virtual void setOptimizer(Optimizer<T> optimizer) {
        optimizer_ = optimizer;
    }

    virtual Optimizer<T> getOptimizer() {
        return optimizer_;
    }

protected:
  math::TensorWrapper<T> loss_ = math::TensorWrapper<T>();
  Optimizer<T> optimizer_ = SGDOptimizer<T>({}, DefaultLearningRate);
};
} // namespace hahaha::ml

#endif // HAHAHA_MODEL_H_CC9C6DBC3C23404399CA13FD2F70E408
