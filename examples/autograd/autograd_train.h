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
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
#pragma once

#include <chrono>
#include <random>
#include <vector>

#include "Tensor.h"
#include "common/definitions.h"
#include "ml/optimizer/SGDOptimizer.h"
#include "utils/log/Logger.h"

using hahaha::Tensor;
using namespace hahaha::common;

inline void nn_train_example() {
    std::mt19937 engine;

    std::random_device randomDevice;
    std::seed_seq seed{
        randomDevice(),
        static_cast<unsigned int>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    engine.seed(seed);

    std::uniform_real_distribution<float> dist(0, 1);

    constexpr int DataSize = 100;
    constexpr int TrainLoop = 100;

    std::vector<f32> x;
    std::vector<f32> y;
    x.resize(DataSize);
    y.resize(DataSize);

    // y = 2 * x
    for (int i = 0; i < DataSize; ++i) {
        const auto val = dist(engine);
        x[i] = val;
        y[i] = val * 2;
    }

    info("x: " + std::to_string(x[0]) + " y: " + std::to_string(y[0]));

    const Tensor<f32> xTensor = Tensor<f32>::buildFromVector(x);
    const Tensor<f32> yTensor = Tensor<f32>::buildFromVector(y);
    Tensor<f32> bias(static_cast<f32>(0));
    Tensor<f32> w(1);
    w.setRequiresGrad(true);
    bias.setRequiresGrad(true);
    // y = w * x
    Tensor<f32> loss(1);
    loss.setRequiresGrad(true);
    hahaha::ml::SGDOptimizer<f32> optimizer({w, bias}, 0.005);

    for (int i = 0; i < TrainLoop; ++i) {
        optimizer.zeroGrad();
        loss.clear();

        info("===================epoch " + std::to_string(i)
             + "===================");
        info("w: " + std::to_string(w.at({})));
        info("bias: " + std::to_string(bias.at({})));

        Tensor<f32> yPredict = w * xTensor + bias;

        loss = (yTensor - yPredict)
                   .reshape({1, 100})
                   .matmul((yTensor - yPredict).reshape({100, 1}));

        loss.backward();

        info("grad of    w: " + std::to_string(w.grad()->at({})));
        info("grad of bias: " + std::to_string(bias.grad()->at({})));
        info("grad of loss: " + std::to_string(loss.grad()->at({0, 0})));

        optimizer.step();

        info("loss: " + std::to_string(loss.at({0, 0})));
        info("");
    }
}
