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

#ifndef HAHAHA_BACKEND_CPU_CPU_ELEMENTWISE_KERNELS_H
#define HAHAHA_BACKEND_CPU_CPU_ELEMENTWISE_KERNELS_H

#include <span>
#include <stdexcept>
#include <vector>

namespace hahaha::backend {

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
                    std::span<T> lhsBuf,
                    std::span<T> rhsBuf,
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

// --- CPU element-wise kernels ---

template <typename T>
void cpuAdd(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<T> a,
             std::span<T> b,
             std::span<T> out) {
    forEachElement<T>(
        shape, lStride, rStride, a, b, out, [](T x, T y) { return x + y; });
}

template <typename T>
void cpuSub(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<T> a,
             std::span<T> b,
             std::span<T> out) {
    forEachElement<T>(
        shape, lStride, rStride, a, b, out, [](T x, T y) { return x - y; });
}

template <typename T>
void cpuMul(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<T> a,
             std::span<T> b,
             std::span<T> out) {
    forEachElement<T>(
        shape, lStride, rStride, a, b, out, [](T x, T y) { return x * y; });
}

template <typename T>
void cpuDiv(const std::vector<size_t>& shape,
             const std::vector<size_t>& lStride,
             const std::vector<size_t>& rStride,
             std::span<T> a,
             std::span<T> b,
             std::span<T> out) {
    forEachElement<T>(shape, lStride, rStride, a, b, out, [](T x, T y) {
        if (y == T(0))
            throw std::runtime_error("Division by zero");
        return x / y;
    });
}

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_CPU_CPU_ELEMENTWISE_KERNELS_H
