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

#include <expected>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/Device.h"
#include "common/error_handler.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaDevice.h"
#include "backend/gpu/cuda/CudaMemory.h"
#include "backend/gpu/cuda/cuda_compute_fun.h"
#endif
#endif

namespace hahaha::math {
template <typename T> class TensorWrapper;
} // namespace hahaha::math

namespace hahaha::backend {

using common::DeviceNotSupportedError;
using common::err;
using common::Error;
using common::ErrorCode;
using common::InvalidArgumentError;

/**
 * @brief Helper function to check if tensor data is contiguous.
 * @param shape Tensor shape
 * @param stride Tensor stride
 * @return true if data is contiguous, false otherwise
 */
inline bool isContiguous(const std::vector<size_t>& shape,
                         const std::vector<size_t>& stride) {
    if (shape.empty() || stride.empty()) {
        return true;
    }
    size_t expectedStride = 1;
    for (long i = static_cast<long>(shape.size()) - 1; i >= 0; --i) {
        if (stride[static_cast<size_t>(i)] != expectedStride) {
            return false;
        }
        expectedStride *= shape[static_cast<size_t>(i)];
    }
    return true;
}

/**
 * @brief Stride-aware elementwise iterator.
 */
template <typename T, typename Fn>
void forEachElement(const std::vector<size_t>& shape,
                    const std::vector<size_t>& lhsStride,
                    const std::vector<size_t>& rhsStride,
                    std::span<const T> lhsBuf,
                    std::span<const T> rhsBuf,
                    std::span<T> outBuf,
                    Fn&& fn) {
    const size_t rank = shape.size();
    if (rank == 0) {
        // scalar
        outBuf[0] = fn(lhsBuf[0], rhsBuf[0]);
        return;
    }

    std::vector<size_t> coord(rank, 0);
    size_t lhsOff = 0;
    size_t rhsOff = 0;

    const size_t total = [&] {
        size_t prod = 1;
        for (const size_t d : shape)
            prod *= d;
        return prod;
    }();

    for (size_t outIdx = 0; outIdx < total; ++outIdx) {
        outBuf[outIdx] = fn(lhsBuf[lhsOff], rhsBuf[rhsOff]);

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

// --- CPU Kernels ---

template <typename T>
void cpu_add(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<const T> a,
             std::span<const T> b,
             std::span<T> out) {
    forEachElement<T>(
        shape, lStride, rStride, a, b, out, [](T x, T y) { return x + y; });
}

template <typename T>
void cpu_sub(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<const T> a,
             std::span<const T> b,
             std::span<T> out) {
    forEachElement<T>(
        shape, lStride, rStride, a, b, out, [](T x, T y) { return x - y; });
}

template <typename T>
void cpu_mul(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<const T> a,
             std::span<const T> b,
             std::span<T> out) {
    forEachElement<T>(
        shape, lStride, rStride, a, b, out, [](T x, T y) { return x * y; });
}

template <typename T>
void cpu_div(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<const T> a,
             std::span<const T> b,
             std::span<T> out) {
    forEachElement<T>(shape, lStride, rStride, a, b, out, [](T x, T y) {
        if (y == T(0))
            throw std::runtime_error("Division by zero");
        return x / y;
    });
}

// --- CUDA Kernels ---

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

// Forward declarations for CUDA functions
template <typename T>
void cuda_add(std::span<const T> a, std::span<const T> b, std::span<T> out);

template <typename T>
void cuda_sub(std::span<const T> a, std::span<const T> b, std::span<T> out);

template <typename T>
void cuda_mul(std::span<const T> a, std::span<const T> b, std::span<T> out);

template <typename T>
void cuda_div(std::span<const T> a, std::span<const T> b, std::span<T> out);

/**
 * @brief CUDA implementation for element-wise addition (float specialization).
 */
template <>
inline void cuda_add<float>(std::span<const float> a,
                            std::span<const float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    // Allocate GPU memory
    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);
    // Launch kernel
    const unsigned int blockSize = 256;
    cudaError_t err = cudaComputeAdd((cf32*) (a.data()),
                                     (cf32*) (b.data()),
                                     (cf32*) (out.data()),
                                     size,
                                     blockSize);
    if (err != cudaSuccess) {

        throw std::runtime_error("CUDA kernel launch failed");
    }

    // Synchronize and copy result back
    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation for element-wise subtraction (float
 * specialization).
 */
template <>
inline void cuda_sub<float>(std::span<const float> a,
                            std::span<const float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);
    DeviceBuffer bufA = cudaMem.allocate(bytes);
    DeviceBuffer bufB = cudaMem.allocate(bytes);
    DeviceBuffer bufOut = cudaMem.allocate(bytes);

    cudaMem.copyHostToDevice(
        bufA,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(a.data()),
                                   bytes));
    cudaMem.copyHostToDevice(
        bufB,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(b.data()),
                                   bytes));

    const unsigned int blockSize = 256;
    cudaError_t err =
        cudaComputeSubtract(reinterpret_cast<cf32*>(bufA.address()),
                            reinterpret_cast<cf32*>(bufB.address()),
                            reinterpret_cast<cf32*>(bufOut.address()),
                            size,
                            blockSize);
    if (err != cudaSuccess) {
        cudaMem.free(bufA);
        cudaMem.free(bufB);
        cudaMem.free(bufOut);
        throw std::runtime_error("CUDA kernel launch failed");
    }

    cudaDeviceSynchronize();
    cudaMem.copyDeviceToHost(
        std::span<std::byte>(reinterpret_cast<std::byte*>(out.data()), bytes),
        bufOut);

    cudaMem.free(bufA);
    cudaMem.free(bufB);
    cudaMem.free(bufOut);
}

/**
 * @brief CUDA implementation for element-wise multiplication (float
 * specialization).
 */
template <>
inline void cuda_mul<float>(std::span<const float> a,
                            std::span<const float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);
    DeviceBuffer bufA = cudaMem.allocate(bytes);
    DeviceBuffer bufB = cudaMem.allocate(bytes);
    DeviceBuffer bufOut = cudaMem.allocate(bytes);

    cudaMem.copyHostToDevice(
        bufA,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(a.data()),
                                   bytes));
    cudaMem.copyHostToDevice(
        bufB,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(b.data()),
                                   bytes));

    const unsigned int blockSize = 256;
    cudaError_t err =
        cudaComputeMultiply(reinterpret_cast<cf32*>(bufA.address()),
                            reinterpret_cast<cf32*>(bufB.address()),
                            reinterpret_cast<cf32*>(bufOut.address()),
                            size,
                            blockSize);
    if (err != cudaSuccess) {
        cudaMem.free(bufA);
        cudaMem.free(bufB);
        cudaMem.free(bufOut);
        throw std::runtime_error("CUDA kernel launch failed");
    }

    cudaDeviceSynchronize();
    cudaMem.copyDeviceToHost(
        std::span<std::byte>(reinterpret_cast<std::byte*>(out.data()), bytes),
        bufOut);

    cudaMem.free(bufA);
    cudaMem.free(bufB);
    cudaMem.free(bufOut);
}

/**
 * @brief CUDA implementation for element-wise division (float specialization).
 */
template <>
inline void cuda_div<float>(std::span<const float> a,
                            std::span<const float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);
    DeviceBuffer bufA = cudaMem.allocate(bytes);
    DeviceBuffer bufB = cudaMem.allocate(bytes);
    DeviceBuffer bufOut = cudaMem.allocate(bytes);

    cudaMem.copyHostToDevice(
        bufA,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(a.data()),
                                   bytes));
    cudaMem.copyHostToDevice(
        bufB,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(b.data()),
                                   bytes));

    const unsigned int blockSize = 256;
    cudaError_t err =
        cudaComputeDivide(reinterpret_cast<cf32*>(bufA.address()),
                          reinterpret_cast<cf32*>(bufB.address()),
                          reinterpret_cast<cf32*>(bufOut.address()),
                          size,
                          blockSize);
    if (err != cudaSuccess) {
        cudaMem.free(bufA);
        cudaMem.free(bufB);
        cudaMem.free(bufOut);
        throw std::runtime_error("CUDA kernel launch failed");
    }

    cudaDeviceSynchronize();
    cudaMem.copyDeviceToHost(
        std::span<std::byte>(reinterpret_cast<std::byte*>(out.data()), bytes),
        bufOut);

    cudaMem.free(bufA);
    cudaMem.free(bufB);
    cudaMem.free(bufOut);
}

// Stubs for non-float types
template <typename T>
void cuda_add(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

template <typename T>
void cuda_sub(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

template <typename T>
void cuda_mul(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

template <typename T>
void cuda_div(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

#else // !__has_include(<driver_types.h>)

// Stubs when CUDA headers not available
template <typename T>
void cuda_sub(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}
template <typename T>
void cuda_add(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}
template <typename T>
void cuda_mul(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}
template <typename T>
void cuda_div(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}

#endif // __has_include(<driver_types.h>)
#else  // !HAHAHA_USE_CUDA

// Stubs when CUDA not enabled
template <typename T>
void cuda_sub(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}
template <typename T>
void cuda_add(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}
template <typename T>
void cuda_mul(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}
template <typename T>
void cuda_div(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
}

#endif // HAHAHA_USE_CUDA

// --- Specific Dispatch Functions ---

// --- Add ---

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchAdd(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    const auto& shape = lhs.getShape();
    if (shape != rhs.getShape() || shape != res.getShape()) {
        return std::unexpected(InvalidArgumentError());
    }
    const auto& lStride = lhs.getStride().getStrides();
    const auto& rStride = rhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());

    if constexpr (dev == DeviceType::CPU) {
        cpu_add(shape, lStride, rStride, lBuf, rBuf, resBuf);
        return {};
    } else if constexpr (dev == DeviceType::CUDA) {
        // CUDA kernels require contiguous data
        // Check if data is contiguous, if not, fall back to CPU
        const bool lContiguous = isContiguous(shape, lStride);
        const bool rContiguous = isContiguous(shape, rStride);
        const bool resContiguous =
            isContiguous(shape, res.getStride().getStrides());

        if (lContiguous && rContiguous && resContiguous) {
            // Direct CUDA computation
            cuda_add(lBuf, rBuf, resBuf);
        } else {
            // Fall back to CPU for non-contiguous data
            cpu_add(shape, lStride, rStride, lBuf, rBuf, resBuf);
        }
        return {};
    }
    return std::unexpected(InvalidArgumentError());
}

template <typename T>
std::expected<void, Error> dispatchAdd(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    switch (dev) {
    case DeviceType::CPU:
        return dispatchAdd<T, DeviceType::CPU>(lhs, rhs, res);
    case DeviceType::CUDA:
        return dispatchAdd<T, DeviceType::CUDA>(lhs, rhs, res);
    default:
        return std::unexpected(DeviceNotSupportedError());
    }
}

template <typename T>
std::expected<void, Error> dispatchAdd(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       T rhs,
                                       math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());
    const auto& shape = lhs.getShape();
    if (shape != res.getShape())
        return std::unexpected(InvalidArgumentError());
    const auto& lStride = lhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, lStride, lStride, lBuf, lBuf, resBuf, [rhs](T a, T /*unused*/) {
            return a + rhs;
        });
    return {};
}

template <typename T>
std::expected<void, Error> dispatchAdd(const DeviceType dev,
                                       T lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    return dispatchAdd(dev, rhs, lhs, res);
}

// --- Sub ---

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchSub(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    const auto& shape = lhs.getShape();
    if (shape != rhs.getShape() || shape != res.getShape()) {
        return std::unexpected(InvalidArgumentError());
    }
    const auto& lStride = lhs.getStride().getStrides();
    const auto& rStride = rhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());

    if constexpr (dev == DeviceType::CPU) {
        cpu_sub(shape, lStride, rStride, lBuf, rBuf, resBuf);
        return {};
    } else if constexpr (dev == DeviceType::CUDA) {
        cuda_sub(lBuf, rBuf, resBuf);
        return {};
    }
    return std::unexpected(InvalidArgumentError());
}

template <typename T>
std::expected<void, Error> dispatchSub(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    switch (dev) {
    case DeviceType::CPU:
        return dispatchSub<T, DeviceType::CPU>(lhs, rhs, res);
    case DeviceType::CUDA:
        return dispatchSub<T, DeviceType::CUDA>(lhs, rhs, res);
    default:
        return std::unexpected(DeviceNotSupportedError());
    }
}

template <typename T>
std::expected<void, Error> dispatchSub(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       T rhs,
                                       math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());
    const auto& shape = lhs.getShape();
    if (shape != res.getShape())
        return std::unexpected(InvalidArgumentError());
    const auto& lStride = lhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, lStride, lStride, lBuf, lBuf, resBuf, [rhs](T a, T /*unused*/) {
            return a - rhs;
        });
    return {};
}

template <typename T>
std::expected<void, Error> dispatchSub(const DeviceType dev,
                                       T lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());
    const auto& shape = rhs.getShape();
    if (shape != res.getShape())
        return std::unexpected(InvalidArgumentError());
    const auto& rStride = rhs.getStride().getStrides();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, rStride, rStride, rBuf, rBuf, resBuf, [lhs](T a, T /*unused*/) {
            return lhs - a;
        });
    return {};
}

// --- Mul ---

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchMul(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    const auto& shape = lhs.getShape();
    if (shape != rhs.getShape() || shape != res.getShape()) {
        return std::unexpected(InvalidArgumentError());
    }
    const auto& lStride = lhs.getStride().getStrides();
    const auto& rStride = rhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());

    if constexpr (dev == DeviceType::CPU) {
        cpu_mul(shape, lStride, rStride, lBuf, rBuf, resBuf);
        return {};
    } else if constexpr (dev == DeviceType::CUDA) {
        cuda_mul(lBuf, rBuf, resBuf);
        return {};
    }
    return std::unexpected(InvalidArgumentError());
}

template <typename T>
std::expected<void, Error> dispatchMul(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    switch (dev) {
    case DeviceType::CPU:
        return dispatchMul<T, DeviceType::CPU>(lhs, rhs, res);
    case DeviceType::CUDA:
        return dispatchMul<T, DeviceType::CUDA>(lhs, rhs, res);
    default:
        return std::unexpected(DeviceNotSupportedError());
    }
}

template <typename T>
std::expected<void, Error> dispatchMul(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       T rhs,
                                       math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());
    const auto& shape = lhs.getShape();
    if (shape != res.getShape())
        return std::unexpected(InvalidArgumentError());
    const auto& lStride = lhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, lStride, lStride, lBuf, lBuf, resBuf, [rhs](T a, T /*unused*/) {
            return a * rhs;
        });
    return {};
}

template <typename T>
std::expected<void, Error> dispatchMul(const DeviceType dev,
                                       T lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    return dispatchMul(dev, rhs, lhs, res);
}

// --- Div ---

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchDiv(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    const auto& shape = lhs.getShape();
    if (shape != rhs.getShape() || shape != res.getShape()) {
        return std::unexpected(InvalidArgumentError());
    }
    const auto& lStride = lhs.getStride().getStrides();
    const auto& rStride = rhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());

    if constexpr (dev == DeviceType::CPU) {
        cpu_div(shape, lStride, rStride, lBuf, rBuf, resBuf);
        return {};
    } else if constexpr (dev == DeviceType::CUDA) {
        cuda_div(lBuf, rBuf, resBuf);
        return {};
    }
    return std::unexpected(InvalidArgumentError());
}

template <typename T>
std::expected<void, Error> dispatchDiv(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    switch (dev) {
    case DeviceType::CPU:
        return dispatchDiv<T, DeviceType::CPU>(lhs, rhs, res);
    case DeviceType::CUDA:
        return dispatchDiv<T, DeviceType::CUDA>(lhs, rhs, res);
    default:
        return std::unexpected(DeviceNotSupportedError());
    }
}

template <typename T>
std::expected<void, Error> dispatchDiv(const DeviceType dev,
                                       const math::TensorWrapper<T>& lhs,
                                       T rhs,
                                       math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());
    if (rhs == T(0))
        return std::unexpected(Error{ErrorCode::InvalidArgument});
    const auto& shape = lhs.getShape();
    if (shape != res.getShape())
        return std::unexpected(InvalidArgumentError());
    const auto& lStride = lhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, lStride, lStride, lBuf, lBuf, resBuf, [rhs](T a, T /*unused*/) {
            return a / rhs;
        });
    return {};
}

template <typename T>
std::expected<void, Error> dispatchDiv(const DeviceType dev,
                                       T lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());
    const auto& shape = rhs.getShape();
    if (shape != res.getShape())
        return std::unexpected(InvalidArgumentError());
    const auto& rStride = rhs.getStride().getStrides();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, rStride, rStride, rBuf, rBuf, resBuf, [lhs](T a, T /*unused*/) {
            if (a == T(0))
                throw std::runtime_error("Division by zero");
            return lhs / a;
        });
    return {};
}

// --- MatMul ---

template <typename T>
std::expected<void, Error> dispatchMatMul(const DeviceType dev,
                                          const math::TensorWrapper<T>& lhs,
                                          const math::TensorWrapper<T>& rhs,
                                          math::TensorWrapper<T>& res) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());

    const auto& lhsDims = lhs.getShape();
    const auto& rhsDims = rhs.getShape();
    const size_t rows = lhsDims[0];
    const size_t cols = rhsDims[1];
    const size_t inner = lhsDims[1];
    auto lRaw = lhs.getRawData();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<const T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<const T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < inner; ++k) {
                sum += lBuf[i * inner + k] * rBuf[k * cols + j];
            }
            resBuf[i * cols + j] = sum;
        }
    }
    return {};
}

// --- Axpy ---

template <typename T>
std::expected<void, Error> dispatchAxpy(const DeviceType dev,
                                        T alpha,
                                        const math::TensorWrapper<T>& x_tensor,
                                        math::TensorWrapper<T>& res_tensor) {
    if (dev != DeviceType::CPU)
        return std::unexpected(DeviceNotSupportedError());

    const size_t size = res_tensor.getTotalSize();
    auto xRaw = x_tensor.getRawData();
    auto resRaw = res_tensor.getRawData();
    std::span<const T> xBuf(xRaw.get(), x_tensor.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res_tensor.getTotalSize());
    for (size_t i = 0; i < size; ++i) {
        resBuf[i] += alpha * xBuf[i];
    }
    return {};
}

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
