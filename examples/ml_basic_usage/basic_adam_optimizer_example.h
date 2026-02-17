// Copyright (c) 2025-2026 Contributors of Hahaha
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
// jiansongshen (jason.shen111@outlook.com)
//
#pragma once

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "common/definitions.h"
#include "ml/optimizer/AdamOptimizer.h"
#include "ml/optimizer/SGDOptimizer.h"
#include "public/Tensor.h"

using hahaha::Tensor;
using namespace hahaha::common;

inline void basic_adam_optimizer_example() {

    constexpr int SmallestX = 0;
    constexpr int BiggestX = 100;

    std::mt19937 engine;

    std::random_device randomDevice;
    std::seed_seq seed{
        randomDevice(),
        static_cast<unsigned int>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    engine.seed(seed);

    std::uniform_real_distribution<float> dist(SmallestX, BiggestX);

    constexpr int DataSize = 100;
    constexpr int TrainLoop = 100;

    std::vector<f32> x;
    std::vector<f32> y;
    x.resize(DataSize);
    y.resize(DataSize);

    // y = 2 * x
    for (int i = 0; i < DataSize; ++i) {
        const auto val = dist(engine);
        x[i] = val / (BiggestX - SmallestX);
        y[i] = x[i] * 2;
    }

    Tensor<f32> const xTensor1 = Tensor<f32>::buildFromVector(x);
    Tensor<f32> const yTensor1 = Tensor<f32>::buildFromVector(y);
    Tensor<f32> const xTensorM = xTensor1.reshape({100, 1});
    Tensor<f32> const yTensorM = yTensor1.reshape({100, 1});
    Tensor<f32> w(1);
    w.setRequiresGrad(true);
    // y = w * x
    Tensor<f32> loss(1);
    loss.setRequiresGrad(true);

    auto optimizer = hahaha::ml::AdamOptimizer<f32>({w}, 0.1);

    for (int i = 0; i < TrainLoop; ++i) {
        optimizer.zeroGrad();
        auto tmp = yTensorM - w * xTensorM;

        loss = (yTensorM - w * xTensorM)
                   .transpose()
                   .matmul(yTensorM - w * xTensorM);

        loss.backward();
        optimizer.step();

        std::cout << "epoch: " << 1 + i << std::endl;
        std::cout << "  w: " << w.at({}) << std::endl;
        std::cout << "  loss: " << loss.data()->at({0, 0}) << std::endl;
    }

}