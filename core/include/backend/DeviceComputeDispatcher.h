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
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
#define HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "backend/Device.h"
#include "common/Operator.h"

namespace hahaha::math {
template <typename T> class TensorWrapper;
} // namespace hahaha::math

namespace hahaha::backend {

/**
 * @brief Top-level dispatcher for device-specific computations.
 *
 * This class serves as a central point to route tensor operations to their
 * respective hardware-optimized implementations (CPU, SIMD, GPU, etc.).
 * It decouples math logic in TensorWrapper from device-specific kernels.
 */
template <typename T> class DeviceComputeDispatcher {
  public:
    /**
     * @brief Stride-aware elementwise iterator.
     *
     * This supports broadcast views produced by TensorWrapper::broadcastTo()
     * (i.e. stride can contain 0), without requiring materialization.
     *
     * The traversal is multi-dimensional but writes output as a dense
     * row-major buffer.
     *
     * Plain-text formulas:
     * - Linear offset update (row-major walk):
     *   offset_next = offset + stride[dim]
     * - Broadcasted dimension is represented by stride[dim] = 0, so:
     *   offset_next = offset (re-uses the same value)
     * - Total elements:
     *   total = product(shape[d]) for d in [0..rank-1]
     *
     * @tparam Fn Callable with signature `T fn(T lhs, T rhs)`.
     * @param shape Output shape (must match elementwise broadcast result).
     * @param lhsStride LHS strides aligned with `shape` rank.
     * @param rhsStride RHS strides aligned with `shape` rank.
     * @param lhsPtr Pointer to LHS base storage (may be shared/broadcast view).
     * @param rhsPtr Pointer to RHS base storage (may be shared/broadcast view).
     * @param outPtr Pointer to dense output buffer.
     * @param fn Elementwise function.
     */
    template <typename Fn>
    static void forEachElement(const std::vector<size_t>& shape,
                               const std::vector<size_t>& lhsStride,
                               const std::vector<size_t>& rhsStride,
                               T* lhsPtr,
                               T* rhsPtr,
                               T* outPtr,
                               Fn&& fn) {
        const size_t rank = shape.size();
        if (rank == 0) {
            // scalar
            outPtr[0] = fn(lhsPtr[0], rhsPtr[0]);
            return;
        }

        std::vector<size_t> coord(rank, 0);
        size_t lhsOff = 0;
        size_t rhsOff = 0;

        // output is always written densely (outPtr[idx])
        const size_t total = [&]() {
            size_t prod = 1;
            for (size_t d : shape)
                prod *= d;
            return prod;
        }();

        for (size_t outIdx = 0; outIdx < total; ++outIdx) {
            outPtr[outIdx] = fn(lhsPtr[lhsOff], rhsPtr[rhsOff]);

            // advance coordinate + offsets
            for (long dim = static_cast<long>(rank) - 1; dim >= 0; --dim) {
                coord[static_cast<size_t>(dim)]++;
                if (coord[static_cast<size_t>(dim)]
                    == shape[static_cast<size_t>(dim)]) {
                    coord[static_cast<size_t>(dim)] = 0;
                    lhsOff -= lhsStride[static_cast<size_t>(dim)]
                        * (shape[static_cast<size_t>(dim)] - 1);
                    rhsOff -= rhsStride[static_cast<size_t>(dim)]
                        * (shape[static_cast<size_t>(dim)] - 1);
                    continue;
                }
                lhsOff += lhsStride[static_cast<size_t>(dim)];
                rhsOff += rhsStride[static_cast<size_t>(dim)];
                break;
            }
        }
    }

    static void dispatchBinary(common::Operator op,
                               const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            /**
             * Plain-text formulas (elementwise):
             * - Add: res[i] = lhs[i] + rhs[i]
             * - Sub: res[i] = lhs[i] - rhs[i]
             * - Mul: res[i] = lhs[i] * rhs[i]
             * - Div: res[i] = lhs[i] / rhs[i]   (rhs[i] != 0)
             *
             * Note: lhs/rhs may be broadcast views, so indexing uses strides.
             */
            const auto& shape = lhs.getShape();
            if (shape != rhs.getShape() || shape != res.getShape()) {
                throw std::invalid_argument("dispatchBinary: shape mismatch");
            }

            const auto& lStride = lhs.getStride().getStrides();
            const auto& rStride = rhs.getStride().getStrides();
            auto* lPtr = lhs.data_.getData().get();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            switch (op) {
            case common::Operator::Add:
                forEachElement(
                    shape, lStride, rStride, lPtr, rPtr, resPtr, [](T a, T b) {
                        return a + b;
                    });
                break;
            case common::Operator::Sub:
                forEachElement(
                    shape, lStride, rStride, lPtr, rPtr, resPtr, [](T a, T b) {
                        return a - b;
                    });
                break;
            case common::Operator::Mul:
                forEachElement(
                    shape, lStride, rStride, lPtr, rPtr, resPtr, [](T a, T b) {
                        return a * b;
                    });
                break;
            case common::Operator::Div:
                forEachElement(
                    shape, lStride, rStride, lPtr, rPtr, resPtr, [](T a, T b) {
                        if (b == T(0)) {
                            throw std::runtime_error("Division by zero");
                        }
                        return a / b;
                    });
                break;
            default:
                throw std::runtime_error("Unsupported binary op");
            }
        } else if (device.type == backend::DeviceType::GPU) {
            throw std::runtime_error("GPU dispatch not yet implemented");
        } else {
            throw std::runtime_error("Unsupported device type for dispatch");
        }
    }

    static void dispatchScalar(common::Operator op,
                               const math::TensorWrapper<T>& lhs,
                               T rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            /**
             * Plain-text formulas (scalar on RHS):
             * - Add: res[i] = lhs[i] + rhs
             * - Sub: res[i] = lhs[i] - rhs
             * - Mul: res[i] = lhs[i] * rhs
             * - Div: res[i] = lhs[i] / rhs   (rhs != 0)
             */
            const auto& shape = lhs.getShape();
            if (shape != res.getShape()) {
                throw std::invalid_argument("dispatchScalar: shape mismatch");
            }

            const auto& lStride = lhs.getStride().getStrides();
            auto* lPtr = lhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            switch (op) {
            case common::Operator::Add:
                forEachElement(shape,
                               lStride,
                               lStride,
                               lPtr,
                               lPtr,
                               resPtr,
                               [rhs](T a, T /*unused*/) { return a + rhs; });
                break;
            case common::Operator::Sub:
                forEachElement(shape,
                               lStride,
                               lStride,
                               lPtr,
                               lPtr,
                               resPtr,
                               [rhs](T a, T /*unused*/) { return a - rhs; });
                break;
            case common::Operator::Mul:
                forEachElement(shape,
                               lStride,
                               lStride,
                               lPtr,
                               lPtr,
                               resPtr,
                               [rhs](T a, T /*unused*/) { return a * rhs; });
                break;
            case common::Operator::Div:
                if (rhs == T(0))
                    throw std::runtime_error("Division by zero");
                forEachElement(shape,
                               lStride,
                               lStride,
                               lPtr,
                               lPtr,
                               resPtr,
                               [rhs](T a, T /*unused*/) { return a / rhs; });
                break;
            default:
                throw std::runtime_error("Unsupported scalar op");
            }
        } else {
            throw std::runtime_error(
                "Scalar dispatch not yet implemented for this device");
        }
    }

    static void dispatchScalar(common::Operator op,
                               T lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = rhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            /**
             * Plain-text formulas (scalar on LHS):
             * - Add: res[i] = lhs + rhs[i]
             * - Sub: res[i] = lhs - rhs[i]
             * - Mul: res[i] = lhs * rhs[i]
             * - Div: res[i] = lhs / rhs[i]   (rhs[i] != 0)
             */
            const auto& shape = rhs.getShape();
            if (shape != res.getShape()) {
                throw std::invalid_argument("dispatchScalar: shape mismatch");
            }

            const auto& rStride = rhs.getStride().getStrides();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            switch (op) {
            case common::Operator::Add:
                forEachElement(shape,
                               rStride,
                               rStride,
                               rPtr,
                               rPtr,
                               resPtr,
                               [lhs](T a, T /*unused*/) { return lhs + a; });
                break;
            case common::Operator::Sub:
                forEachElement(shape,
                               rStride,
                               rStride,
                               rPtr,
                               rPtr,
                               resPtr,
                               [lhs](T a, T /*unused*/) { return lhs - a; });
                break;
            case common::Operator::Mul:
                forEachElement(shape,
                               rStride,
                               rStride,
                               rPtr,
                               rPtr,
                               resPtr,
                               [lhs](T a, T /*unused*/) { return lhs * a; });
                break;
            case common::Operator::Div:
                forEachElement(shape,
                               rStride,
                               rStride,
                               rPtr,
                               rPtr,
                               resPtr,
                               [lhs](T a, T /*unused*/) {
                                   if (a == T(0)) {
                                       throw std::runtime_error(
                                           "Division by zero");
                                   }
                                   return lhs / a;
                               });
                break;
            default:
                throw std::runtime_error("Unsupported scalar op");
            }
        } else {
            throw std::runtime_error(
                "Scalar dispatch not yet implemented for this device");
        }
    }

    static void dispatchMatMul(const math::TensorWrapper<T>& lhs,
                               const math::TensorWrapper<T>& rhs,
                               math::TensorWrapper<T>& res) {
        auto device = lhs.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            const auto& lhsDims = lhs.getShape();
            const auto& rhsDims = rhs.getShape();

            size_t rows = lhsDims[0];
            size_t cols = rhsDims[1];
            size_t inner = lhsDims[1];

            auto* lPtr = lhs.data_.getData().get();
            auto* rPtr = rhs.data_.getData().get();
            auto* resPtr = res.data_.getData().get();

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    T sum = T(0);
                    for (size_t k = 0; k < inner; ++k) {
                        sum += lPtr[i * inner + k] * rPtr[k * cols + j];
                    }
                    resPtr[i * cols + j] = sum;
                }
            }
        } else {
            throw std::runtime_error("MatMul dispatch not yet implemented");
        }
    }

    /**
     * @brief Performs res = res + alpha * x in-place.
     * @param alpha Scaling factor.
     * @param x_tensor Input tensor.
     * @param res_tensor Result tensor (updated in-place).
     */
    static void dispatchAxpy(T alpha,
                             const math::TensorWrapper<T>& x_tensor,
                             math::TensorWrapper<T>& res_tensor) {
        auto device = res_tensor.getDevice();
        if (device.type == backend::DeviceType::CPU) {
            size_t size = res_tensor.getTotalSize();
            auto* xPtr = x_tensor.data_.getData().get();
            auto* resPtr = res_tensor.data_.getData().get();

            for (size_t i = 0; i < size; ++i) {
                resPtr[i] += alpha * xPtr[i];
            }
        } else {
            throw std::runtime_error("Axpy dispatch not yet implemented");
        }
    }
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
