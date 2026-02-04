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
#include "backend/cpu/CpuElementwiseKernels.h"
#include "backend/gpu/cuda/CudaElementwiseKernels.h"
#include "common/error_handler.h"

namespace hahaha::math {
template <typename T> class TensorWrapper;
} // namespace hahaha::math

namespace hahaha::backend {

using common::DeviceNotSupportedError;
using common::err;
using common::Error;
using common::ErrorCode;
using common::InvalidArgumentError;

// --- Helpers for element-wise dispatch (reduce duplication for Add/Sub/Mul/Div) ---

template <typename T, typename CpuKernel>
std::expected<void, Error> dispatchElementwiseCpu(
    const math::TensorWrapper<T>& lhs,
    const math::TensorWrapper<T>& rhs,
    math::TensorWrapper<T>& res,
    CpuKernel&& kernel) {
    const auto& shape = lhs.getShape();
    if (shape != rhs.getShape() || shape != res.getShape()) {
        return std::unexpected(InvalidArgumentError());
    }
    const auto& lStride = lhs.getStride().getStrides();
    const auto& rStride = rhs.getStride().getStrides();
    auto lRaw = lhs.getRawData();
    auto rRaw = rhs.getRawData();
    auto resRaw = res.getRawData();
    std::span<T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    kernel(shape, lStride, rStride, lBuf, rBuf, resBuf);
    return {};
}

template <typename T, typename CudaKernel>
std::expected<void, Error> dispatchElementwiseCuda(
    const math::TensorWrapper<T>& lhs,
    const math::TensorWrapper<T>& rhs,
    math::TensorWrapper<T>& res,
    CudaKernel&& kernel) {
    const auto& shape = lhs.getShape();
    if (shape != rhs.getShape() || shape != res.getShape()) {
        return std::unexpected(InvalidArgumentError());
    }
    const auto& lStride = lhs.getStride().getStrides();
    const auto& rStride = rhs.getStride().getStrides();
    const auto& resStride = res.getStride().getStrides();
    if (!isContiguous(shape, lStride) || !isContiguous(shape, rStride)
        || !isContiguous(shape, resStride)) {
        return std::unexpected(InvalidArgumentError());
    }
    auto lRaw = lhs.getRawGpuPtr();
    auto rRaw = rhs.getRawGpuPtr();
    auto resRaw = res.getRawGpuPtr();
    std::span<T> lBuf(reinterpret_cast<T*>(lRaw), lhs.getTotalSize());
    std::span<T> rBuf(reinterpret_cast<T*>(rRaw), rhs.getTotalSize());
    std::span<T> resBuf(reinterpret_cast<T*>(resRaw), res.getTotalSize());
    kernel(lBuf, rBuf, resBuf);
    return {};
}

// --- Specific Dispatch Functions ---
// Implemented via class template partial specialization (MSVC-compatible;
// function template partial specialization is not allowed in C++).

// --- Add ---

template <typename T, DeviceType dev>
struct DispatchAddImpl {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        (void)lhs;
        (void)rhs;
        (void)res;
        return std::unexpected(DeviceNotSupportedError());
    }
};

template <typename T>
struct DispatchAddImpl<T, DeviceType::CPU> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCpu(lhs, rhs, res,
            [](const std::vector<size_t>& s,
               const std::vector<size_t>& ls,
               const std::vector<size_t>& rs,
               std::span<T> lb,
               std::span<T> rb,
               std::span<T> ob) { cpu_add(s, ls, rs, lb, rb, ob); });
    }
};

template <typename T>
struct DispatchAddImpl<T, DeviceType::CUDA> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCuda(lhs, rhs, res,
            [](std::span<T> a, std::span<T> b, std::span<T> out) {
                cuda_add(a, b, out);
            });
    }
};

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchAdd(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    return DispatchAddImpl<T, dev>::call(lhs, rhs, res);
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
    std::span<T> lBuf(lRaw.get(), lhs.getTotalSize());
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

template <typename T, DeviceType>
struct DispatchSubImpl {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        (void)lhs;
        (void)rhs;
        (void)res;
        return std::unexpected(DeviceNotSupportedError());
    }
};

template <typename T>
struct DispatchSubImpl<T, DeviceType::CPU> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCpu(lhs, rhs, res,
            [](const std::vector<size_t>& s,
               const std::vector<size_t>& ls,
               const std::vector<size_t>& rs,
               std::span<T> lb,
               std::span<T> rb,
               std::span<T> ob) { cpu_sub(s, ls, rs, lb, rb, ob); });
    }
};

template <typename T>
struct DispatchSubImpl<T, DeviceType::CUDA> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCuda(lhs, rhs, res,
            [](std::span<T> a, std::span<T> b, std::span<T> out) {
                cuda_sub(a, b, out);
            });
    }
};

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchSub(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    return DispatchSubImpl<T, dev>::call(lhs, rhs, res);
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
    std::span<T> lBuf(lRaw.get(), lhs.getTotalSize());
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
    std::span<T> rBuf(rRaw.get(), rhs.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res.getTotalSize());
    forEachElement(
        shape, rStride, rStride, rBuf, rBuf, resBuf, [lhs](T a, T /*unused*/) {
            return lhs - a;
        });
    return {};
}

// --- Mul ---

template <typename T, DeviceType dev>
struct DispatchMulImpl {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        (void)lhs;
        (void)rhs;
        (void)res;
        return std::unexpected(DeviceNotSupportedError());
    }
};

template <typename T>
struct DispatchMulImpl<T, DeviceType::CPU> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCpu(lhs, rhs, res,
            [](const std::vector<size_t>& s,
               const std::vector<size_t>& ls,
               const std::vector<size_t>& rs,
               std::span<T> lb,
               std::span<T> rb,
               std::span<T> ob) { cpu_mul(s, ls, rs, lb, rb, ob); });
    }
};

template <typename T>
struct DispatchMulImpl<T, DeviceType::CUDA> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCuda(lhs, rhs, res,
            [](std::span<T> a, std::span<T> b, std::span<T> out) {
                cuda_mul(a, b, out);
            });
    }
};

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchMul(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    return DispatchMulImpl<T, dev>::call(lhs, rhs, res);
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
    std::span<T> lBuf(lRaw.get(), lhs.getTotalSize());
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

template <typename T, DeviceType dev>
struct DispatchDivImpl {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        (void)lhs;
        (void)rhs;
        (void)res;
        return std::unexpected(DeviceNotSupportedError());
    }
};

template <typename T>
struct DispatchDivImpl<T, DeviceType::CPU> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCpu(lhs, rhs, res,
            [](const std::vector<size_t>& s,
               const std::vector<size_t>& ls,
               const std::vector<size_t>& rs,
               std::span<T> lb,
               std::span<T> rb,
               std::span<T> ob) { cpu_div(s, ls, rs, lb, rb, ob); });
    }
};

template <typename T>
struct DispatchDivImpl<T, DeviceType::CUDA> {
    static std::expected<void, Error> call(const math::TensorWrapper<T>& lhs,
                                           const math::TensorWrapper<T>& rhs,
                                           math::TensorWrapper<T>& res) {
        return dispatchElementwiseCuda(lhs, rhs, res,
            [](std::span<T> a, std::span<T> b, std::span<T> out) {
                cuda_div(a, b, out);
            });
    }
};

template <typename T, const DeviceType dev>
std::expected<void, Error> dispatchDiv(const math::TensorWrapper<T>& lhs,
                                       const math::TensorWrapper<T>& rhs,
                                       math::TensorWrapper<T>& res) {
    return DispatchDivImpl<T, dev>::call(lhs, rhs, res);
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
    std::span<T> lBuf(lRaw.get(), lhs.getTotalSize());
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
    std::span<T> rBuf(rRaw.get(), rhs.getTotalSize());
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
    std::span<T> lBuf(lRaw.get(), lhs.getTotalSize());
    std::span<T> rBuf(rRaw.get(), rhs.getTotalSize());
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
    std::span<T> xBuf(xRaw.get(), x_tensor.getTotalSize());
    std::span<T> resBuf(resRaw.get(), res_tensor.getTotalSize());
    for (size_t i = 0; i < size; ++i) {
        resBuf[i] += alpha * xBuf[i];
    }
    return {};
}

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_COMPUTE_DISPATCHER_H
