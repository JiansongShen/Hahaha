
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

#ifndef HAHAHA_LINEARREGRESSION_H_4305E4B0E7784CD1969E389923F7743D
#define HAHAHA_LINEARREGRESSION_H_4305E4B0E7784CD1969E389923F7743D
#include "Model.h"
#include "ml/Parameters.h"
#include "ml/loss/MSELoss.h"
#include "ml/optimizer/SGDOptimizer.h"

namespace hahaha::ml {

/**
 * @brief Linear regression model.
 *
 * This model implements linear regression: y = x * weight + bias
 *
 * @tparam T The numeric type.
 */
template <typename T> class LinearRegression : public Model<T> {
  public:
    /**
     * @brief Construct a LinearRegression model with zero-initialized weights
     * and bias.
     */
    LinearRegression() : weight_(T(0)), bias_(T(0)) {
    }

    /**
     * @brief Get all trainable parameters of the model.
     * @return Parameters<T> Container with weight and bias parameters.
     */
    Parameters<T> getParameters() override {
        Parameters<T> parameters;

        parameters.addParameter(weight_);
        parameters.addParameter(bias_);
        return parameters;
    }

    /**
     * @brief Set the weight parameters.
     * @param weights Vector of weight values.
     */
    void setWeights(std::vector<T> weights) {
        weight_ = Tensor<T>::buildFromVector(weights);
    }

    /**
     * @brief Set the bias parameters.
     * @param bias Vector of bias values.
     */
    void setBias(std::vector<T> bias) {
        bias_ = Tensor<T>::buildFromVector(bias);
    }

    /**
     * @brief Train the model on the given data.
     * @param x Input features. Shape: (num_samples, num_features).
     * @param y Target values. Shape: (num_samples, num_outputs).
     */
    void train(Tensor<T> x, Tensor<T> y) override {
        // x shape is s * n1 (Size of samples and features Number)
        // y shape is s * n2 (Size of samples and output Number)

        // TODO(napbad): Implement Linear Regression
        // // reshape to a matrix to support common situations
        // auto xShape = x.getShape();
        // if (xShape.size() != 2) {
        //     x = x.reshape({xShape[0], 1}); // n rows and 1 column
        // }

        // auto shape = x.getShape();
        // auto yPredict = x.matmul(weight_) + bias_;
        // auto mseLoss = computeMSELoss(y, yPredict);

        // SGDOptimizer<T> sgdOptimizer({}, T(0.00001));
        // sgdOptimizer.addParameter(weight_);
        // sgdOptimizer.addParameter(bias_);

        // sgdOptimizer.zeroGrad();
        // mseLoss.backward();
        // sgdOptimizer.step();
    }

  private:
    Tensor<T> weight_; /**< Weight matrix. Shape: (num_input_features,
                          num_outputs). */
    Tensor<T> bias_;   /**< Bias vector. Shape: (num_outputs,). */
};

} // namespace hahaha::ml

#endif // HAHAHA_LINEARREGRESSION_H_4305E4B0E7784CD1969E389923F7743D
