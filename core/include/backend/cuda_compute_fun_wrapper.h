//  Copyright (c) 2025 - 2026 Contributors of
//  Hahaha(https://github.com/Napbad/Hahaha)
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
//  Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//
//

#ifndef HAHAHA_CUDA_COMPUTE_FUN_WRAPPER_H_82180F40AB0A4AE08D027C34423B9521
#define HAHAHA_CUDA_COMPUTE_FUN_WRAPPER_H_82180F40AB0A4AE08D027C34423B9521

#ifdef HAHAHA_USE_CUDA
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime_api.h>
#endif
#endif

#include <span>
#include <utility>

#include "gpu/cuda/CudaMemory.h"
#include "gpu/cuda/cuda_compute_fun.h"

namespace hahaha::backend {

// Forward declarations for CUDA functions
template <typename T>
void cuda_add(std::span<const T> /*a*/, std::span<const T> /*b*/, std::span<T> /*out*/) {
    std::unreachable();
}

template <typename T>
void cuda_sub(std::span<const T> /*a*/, std::span<const T> /*b*/, std::span<T> /*out*/) {
    std::unreachable();
}

template <typename T>
void cuda_mul(std::span<const T> /*a*/, std::span<const T> /*b*/, std::span<T> /*out*/) {
    std::unreachable();
}

template <typename T>
void cuda_div(std::span<const T> /*a*/, std::span<const T> /*b*/, std::span<T> /*out*/) {
    std::unreachable();
}

#ifdef HAHAHA_USE_CUDA
/**
 * @brief CUDA implementation for element-wise addition (float specialization).
 */
template <>
inline void cuda_add<float>(const std::span<const float> a,
                            const std::span<const float> b,
                            const std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    // Allocate GPU memory
    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);

    // Launch kernel
    const unsigned int blockSize = 256;
    const cudaError_t err =
        cudaComputeAdd(a.data(), b.data(), out.data(), size, blockSize);
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
inline void cuda_sub<float>(const std::span<const float> a,
                            const std::span<const float> b,
                            const std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);

    // Launch kernel
    const unsigned int blockSize = 256;
    const cudaError_t err =
        cudaComputeSubtract(a.data(), b.data(), out.data(), size, blockSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed");
    }
    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation for element-wise multiplication (float
 * specialization).
 */
template <>
inline void cuda_mul<float>(const std::span<const float> a,
                            const std::span<const float> b,
                            const std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);
    DeviceBuffer bufA = cudaMem.allocate(bytes);
    DeviceBuffer bufB = cudaMem.allocate(bytes);
    DeviceBuffer bufOut = cudaMem.allocate(bytes);

    cudaMem.copyHostToDevice(bufA, std::span<char>((char*) (a.data()), bytes));
    cudaMem.copyHostToDevice(bufB, std::span<char>((char*) (b.data()), bytes));

    const unsigned int blockSize = 256;
    const cudaError_t err =
        cudaComputeMultiply((a.data()), b.data(), out.data(), size, blockSize);
    if (err != cudaSuccess) {
        cudaMem.free(bufA);
        cudaMem.free(bufB);
        cudaMem.free(bufOut);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation for element-wise division (float specialization).
 */
template <>
inline void cuda_div<float>(const std::span<const float> a,
                            const std::span<const float> b,
                            const std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }
    CudaMemory cudaMem;
    const size_t bytes = size * sizeof(float);
    DeviceBuffer bufA = cudaMem.allocate(bytes);
    DeviceBuffer bufB = cudaMem.allocate(bytes);
    DeviceBuffer bufOut = cudaMem.allocate(bytes);

    cudaMem.copyHostToDevice(bufA, std::span<char>((char*) (a.data()), bytes));
    cudaMem.copyHostToDevice(bufB, std::span<char>((char*) (b.data()), bytes));

    const unsigned int blockSize = 256;
    const cudaError_t err =
        cudaComputeDivide(a.data(), b.data(), out.data(), size, blockSize);
    if (err != cudaSuccess) {
        cudaMem.free(bufA);
        cudaMem.free(bufB);
        cudaMem.free(bufOut);
        throw std::runtime_error("CUDA kernel launch failed");
    }
    cudaDeviceSynchronize();
}

// // Stubs for non-float types
// template <typename T>
// void cuda_add(std::span<const T> /* a */,
//               std::span<const T> /* b*/,
//               std::span<T> /*out*/) {
//     throw std::runtime_error("CUDA operations only support float type");
// }
//
// template <typename T>
// void cuda_sub(std::span<const T> /* a */,
//               std::span<const T> /* b*/,
//               std::span<T> /*out*/) {
//     throw std::runtime_error("CUDA operations only support float type");
// }
//
// template <typename T>
// void cuda_mul(std::span<const T> /* a */,
//               std::span<const T> /* b*/,
//               std::span<T> /*out*/) {
//     throw std::runtime_error("CUDA operations only support float type");
// }
//
// template <typename T>
// void cuda_div(std::span<const T> /* a */,
//               std::span<const T> /* b*/,
//               std::span<T> /*out*/) {
//     throw std::runtime_error("CUDA operations only support float type");
// }

#else // !HAHAHA_USE_CUDA

// Stubs when CUDA not enabled
template <typename T>
void cuda_sub(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    std::unreachable();
}
template <typename T>
void cuda_add(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    std::unreachable();
}
template <typename T>
void cuda_mul(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    std::unreachable();
}
template <typename T>
void cuda_div(std::span<const T> /* a */,
              std::span<const T> /* b*/,
              std::span<T> /*out*/) {
    std::unreachable();
}

#endif // HAHAHA_USE_CUDA
} // namespace hahaha::backend
#endif // HAHAHA_CUDA_COMPUTE_FUN_WRAPPER_H_82180F40AB0A4AE08D027C34423B9521
