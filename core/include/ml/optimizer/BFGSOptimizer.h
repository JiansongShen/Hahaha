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

#ifndef HAHAHA_BFGSOPTIMIZER_H_8A9B0C1D2E3F4A5B6C7D8E9F0A1B2C3D
#define HAHAHA_BFGSOPTIMIZER_H_8A9B0C1D2E3F4A5B6C7D8E9F0A1B2C3D

#include <cmath>
#include <vector>

#include "math/TensorWrapper.h"
#include "math/TensorWrapperUtil.h"
#include "ml/optimizer/Optimizer.h"

namespace hahaha::ml {

/**
 * @brief BFGS (Broyden-Fletcher-Goldfarb-Shanno) Optimizer
 *
 * BFGS is a quasi-Newton optimization algorithm that approximates the inverse
 * Hessian matrix to achieve faster convergence than first-order methods like SGD.
 *
 * Algorithm Overview:
 * ===================
 * 1. BFGS maintains an approximation H_k of the inverse Hessian matrix
 * 2. Search direction: p_k = -H_k * g_k  (where g_k is the gradient)
 * 3. Line search: find step size alpha_k that satisfies Wolfe conditions
 * 4. Update: x_{k+1} = x_k + alpha_k * p_k
 * 5. Update H_k using the BFGS formula:
 *
 *
 *    H_{k+1} = H_k + (1 + (y_k^T * H_k * y_k) / (s_k^T * y_k)) * (s_k * s_k^T) /
 * (s_k^T * y_k)
 *              - (H_k * y_k * s_k^T + s_k * y_k^T * H_k) / (s_k^T * y_k)
 *
 *    where:
 *
 *      s_k = x_{k+1} - x_k  (parameter change)
 *      y_k = g_{k+1} - g_k  (gradient change)
 *
 * Key Concepts:
 * =============
 * - Quasi-Newton: Approximates the Hessian without computing it explicitly
 * - Inverse Hessian: H_k approximates the inverse of the Hessian matrix, which
 * allows for efficient search direction
 * - Line Search: Finds optimal step size along the search direction
 * - Memory: Stores H_k (n×n matrix for n parameters), so memory is O(n²)
 *
 * Limitations of this implementation:
 * ===================================
 * - Simple backtracking line search (not full Wolfe conditions)
 * - Memory usage grows quadratically with parameter count
 * - For very large models, consider L-BFGS (Limited-memory BFGS) instead
 *
 * @tparam T Numeric data type (float, double, etc.). Must be floating-point.
 */
template <typename T> class BFGSOptimizer : public Optimizer<T> {
    static_assert(std::is_floating_point_v<T>,
                  "BFGSOptimizer requires floating-point types");
    static constexpr T DefaultInitialStepSize = T(1.0);
    static constexpr T DefaultLineSearchTolerance = T(1e-4);
    static constexpr size_t DefaultMaxLineSearchIterations = 20;
    static constexpr T DefaultMinimumStepSize = T(1e-6);

  public:
    /**
     * @brief Construct a new BFGS Optimizer.
     *
     * @param parameters List of tensors to optimize (should have requiresGrad =
     * true).
     * @param initialStepSize Initial step size for line search (default: 1.0).
     *                        This is the starting alpha in line search.
     * @param lineSearchTolerance Tolerance for line search convergence (default:
     * 1e-4).
     * @param maxLineSearchIterations Maximum iterations for line search (default:
     * 20).
     */
    BFGSOptimizer(std::vector<Tensor<T>> parameters,
                  T initialStepSize = DefaultInitialStepSize,
                  T lineSearchTolerance = DefaultLineSearchTolerance,
                  size_t maxLineSearchIterations = DefaultMaxLineSearchIterations)
        : Optimizer<T>(std::move(parameters), T(0.0)), // BFGS doesn't use fixed LR
          initialStepSize_(initialStepSize),
          lineSearchTolerance_(lineSearchTolerance),
          maxLineSearchIterations_(maxLineSearchIterations), iteration_(0),
          initialized_(false), n_(0) {
        // Initialize state on first step
    }

    /**
     * @brief Performs a single BFGS optimization step.
     *
     * This method:
     * 1. Flattens all parameters and gradients into vectors
     * 2. Computes search direction using current inverse Hessian approximation
     * 3. Performs line search to find optimal step size
     * 4. Updates parameters
     * 5. Updates inverse Hessian approximation using BFGS formula
     */
    void step() override {
        // Step 1: Collect all parameters and gradients that require grad
        std::vector<math::TensorWrapper<T>> paramTensors;
        std::vector<math::TensorWrapper<T>> gradTensors;
        std::vector<size_t> paramSizes; // Store original sizes for reshaping

        getNeededParamAndGrad(paramTensors, gradTensors, paramSizes);
        if (paramTensors.empty()) {
            return; // No parameters to optimize
        }

        // Step 2: Flatten parameters and gradients into single vectors
        // BFGS works on vectors, so we concatenate all parameters using
        // math::concatenate
        math::TensorWrapper<T> xFlat = math::concatenate(paramTensors);
        math::TensorWrapper<T> gFlat = math::concatenate(gradTensors);
        size_t totalSize = xFlat.getTotalSize();

        // Step 3: Initialize inverse Hessian approximation H_0 = I (identity matrix)
        // This happens on the first iteration
        if (!initialized_) {
            initializeInverseHessian(totalSize);
            initialized_ = true;
        }

        // Step 4: Update inverse Hessian using BFGS formula from previous iteration
        // On iteration k, we update H_k using s_{k-1} and y_{k-1} from iteration k-1
        if (iteration_ > 0 && sPrev_.getTotalSize() > 0) {
            // Compute y = g_current - g_previous (gradient change)
            math::TensorWrapper<T> y = gFlat.clone();
            y -= gPrev_;

            // Update H using BFGS formula: H_k = BFGS_UPDATE(H_{k-1}, s_{k-1},
            // y_{k-1})
            updateInverseHessian(sPrev_, y);
        }

        // Step 5: Compute search direction p_k = -H_k * g_k
        // This is the direction we'll move in parameter space
        math::TensorWrapper<T> searchDir = computeSearchDirection(gFlat);

        // Step 6: Perform line search to find optimal step size alpha_k
        // We use backtracking line search: start with initialStepSize_ and reduce
        // until we find a step that decreases the function value sufficiently
        T alpha =
            performLineSearch(xFlat, gFlat, searchDir, paramTensors, paramSizes);

        // Step 7: Store current x and g for next iteration's BFGS update
        math::TensorWrapper<T> xOld = xFlat.clone();
        math::TensorWrapper<T> gOld = gFlat.clone();

        // Step 8: Update parameters: x_{k+1} = x_k + alpha * p_k
        // Update flattened parameters
        xFlat.axpy(alpha, searchDir);

        // Step 9: Reshape and update actual parameter tensors
        // Distribute the updated flattened parameters back to individual tensors
        math::distributeTensors(xFlat, paramTensors, paramSizes);

        // Step 10: Store s_k = x_{k+1} - x_k for next iteration's BFGS update
        // Note: The new gradient g_{k+1} will be computed by the user calling
        // backward() again before the next step(). We'll compute y_{k+1} = g_{k+1} -
        // g_k at the start of the next step() call.
        math::TensorWrapper<T> s = xFlat.clone();
        s.axpy(T(-1), xOld); // s = x_new - x_old

        // Store for next iteration
        sPrev_ = std::move(s);
        gPrev_ = std::move(gOld);
        xPrev_ = std::move(xOld);

        iteration_++;
    }

    void getNeededParamAndGrad(std::vector<math::TensorWrapper<T>>& paramTensors,
                               std::vector<math::TensorWrapper<T>>& gradTensors,
                               std::vector<size_t>& paramSizes) {

        for (auto& param : this->getParameters()) {
            if (!param.getRequiresGrad()) {
                continue;
            }

            auto grad = param.grad();
            if (grad.isEmpty()) {
                continue;
            }

            paramTensors.push_back(param.data().get());
            gradTensors.push_back(grad.data().get());
            paramSizes.push_back(param.data()->getTotalSize());
        }
    }

  private:
    /**
     * @brief Initialize the inverse Hessian approximation H_0 = I (identity matrix).
     *
     * For the first iteration, we start with H_0 = I, which means the first
     * search direction is just the negative gradient (steepest descent).
     *
     * @param n Total number of parameters (size of the flattened vector).
     */
    void initializeInverseHessian(size_t n) {
        // H is stored as an n×n matrix using TensorWrapper
        H_ = math::TensorWrapper<T>(math::TensorShape({n, n}), T(0));
        n_ = n;

        // Set diagonal to 1.0 (identity matrix)
        T* HData = H_.getRawData();
        for (size_t i = 0; i < n; ++i) {
            HData[i * n + i] = T(1.0);
        }
    }

    /**
     * @brief Compute search direction p = -H * g.
     *
     * The search direction is computed by multiplying the inverse Hessian
     * approximation H with the negative gradient. This gives us a direction
     * that accounts for the curvature of the loss landscape.
     *
     * @param grad Flattened gradient vector.
     * @return Search direction vector p = -H * g.
     */
    math::TensorWrapper<T>
    computeSearchDirection(const math::TensorWrapper<T>& grad) {
        // p = -H * g
        math::TensorWrapper<T> p = -H_ * grad;
        return p;
    }

    /**
     * @brief Perform backtracking line search to find step size alpha.
     *
     * Backtracking line search:
     * 1. Start with alpha = initialStepSize_
     * 2. Check if f(x + alpha * p) < f(x) + c1 * alpha * g^T * p
     * 3. If not, reduce alpha by a factor (typically 0.5)
     * 4. Repeat until condition is satisfied or max iterations reached
     *
     * For this educational implementation, we use a simplified version that
     * just checks if the gradient dot product with search direction is negative
     * (ensuring we're moving in a descent direction).
     *
     * @param x Current parameter vector.
     * @param g Current gradient vector.
     * @param p Search direction.
     * @param paramWrappers Original parameter wrappers (for shape info).
     * @param paramSizes Sizes of each parameter.
     * @return Step size alpha.
     */
    T performLineSearch(const math::TensorWrapper<T>& x,
                        const math::TensorWrapper<T>& g,
                        const math::TensorWrapper<T>& p,
                        const std::vector<math::TensorWrapper<T>*>& paramWrappers,
                        const std::vector<size_t>& paramSizes) {
        // Simplified line search: use initial step size
        // In a full implementation, you would:
        // 1. Evaluate the loss function at x + alpha * p
        // 2. Check Armijo condition: f(x + alpha*p) <= f(x) + c1*alpha*g^T*p
        // 3. Backtrack if condition not met


        // For now, we'll use a simple adaptive step size based on gradient magnitude
        T gradNorm = T(0);

        gradNorm = g.square().sum();
        gradNorm = std::sqrt(gradNorm);

        // Adaptive step size: smaller when gradient is large
        T alpha = initialStepSize_ / (T(1.0) + T(0.1) * gradNorm);
        return std::max(alpha, DefaultMinimumStepSize); // Ensure minimum step size
    }

    /**
     * @brief Update inverse Hessian approximation using BFGS formula.
     *
     * BFGS update formula:
     * H_{k+1} = H_k + (1 + (y^T * H * y) / (s^T * y)) * (s * s^T) / (s^T * y)
     *              - (H * y * s^T + s * y^T * H) / (s^T * y)
     *
     * where:
     *   s = x_{k+1} - x_k  (parameter change)
     *   y = g_{k+1} - g_k  (gradient change)
     *
     * This update ensures that H_{k+1} satisfies the secant condition:
     *   H_{k+1} * y = s
     *
     * @param s Parameter change vector (x_new - x_old).
     * @param y Gradient change vector (g_new - g_old).
     */
    void updateInverseHessian(const math::TensorWrapper<T>& s,
                              const math::TensorWrapper<T>& y) {
        // Compute s^T * y (denominator in BFGS formula)
        T sTy = T(0);
        sTy = (s.transpose() * y).sum();

        // Skip update if s^T * y is too small (numerical stability)
        if (std::abs(sTy) < T(1e-10)) {
            return;
        }

        // Compute H * y (needed for update)
        math::TensorWrapper<T> Hy = H_ * y;

        // Compute y^T * H * y
        T yTHy = (y.transpose() * Hy).sum();

        // Compute BFGS update
        // H_new = H_old + term1 - term2
        // where:
        //   term1 = (1 + yTHy/sTy) * (s * s^T) / sTy
        //   term2 = (H*y*s^T + s*y^T*H) / sTy
        T factor = T(1.0) + yTHy / sTy;

        // Update H in-place
        H_ += (factor * (s * s.transpose()))
            - ((Hy * s.transpose()) + (s * Hy.transpose())) / sTy;
    }

    T initialStepSize_;              ///< Initial step size for line search
    T lineSearchTolerance_;          ///< Tolerance for line search convergence
    size_t maxLineSearchIterations_; ///< Maximum line search iterations
    size_t iteration_;               ///< Current iteration number
    bool initialized_;               ///< Whether H has been initialized

    // Inverse Hessian approximation (stored as n×n matrix using TensorWrapper)
    math::TensorWrapper<T> H_;
    size_t n_; ///< Dimension of H (total number of flattened parameters)

    // Stored from previous iteration for BFGS update
    math::TensorWrapper<T> sPrev_; ///< Previous parameter change (s_k)
    math::TensorWrapper<T> gPrev_; ///< Previous gradient (g_k)
    math::TensorWrapper<T> xPrev_; ///< Previous parameters (x_k)
};

} // namespace hahaha::ml

#endif // HAHAHA_BFGSOPTIMIZER_H_8A9B0C1D2E3F4A5B6C7D8E9F0A1B2C3D
