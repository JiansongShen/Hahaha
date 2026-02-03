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

#ifndef HAHAHA_BACKEND_GPU_CUDA_CUDA_ELEMENTWISE_KERNELS_H
#define HAHAHA_BACKEND_GPU_CUDA_CUDA_ELEMENTWISE_KERNELS_H

#include <span>
#include <stdexcept>

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaMemory.h"
#include "backend/gpu/cuda/cuda_compute_fun.h"
#endif
#endif

namespace hahaha::backend {

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

// Forward declarations for CUDA element-wise (span-based) kernels
template <typename T>
void cuda_add(std::span<T> a, std::span<T> b, std::span<T> out);

template <typename T>
void cuda_sub(std::span<T> a, std::span<T> b, std::span<T> out);

template <typename T>
void cuda_mul(std::span<T> a, std::span<T> b, std::span<T> out);

template <typename T>
void cuda_div(std::span<T> a, std::span<T> b, std::span<T> out);

/**
 * @brief CUDA implementation for element-wise addition (float specialization).
 */
template <>
inline void cuda_add<float>(std::span<float> a,
                            std::span<float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    const unsigned int blockSize = 256;
    const cudaError_t err =
        cudaComputeAdd(a.data(),
                       b.data(),
                       out.data(),
                       size,
                       blockSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation for element-wise subtraction (float
 * specialization).
 */
template <>
inline void cuda_sub<float>(std::span<float> a,
                            std::span<float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const unsigned int blockSize = 256;
    cudaError_t err =
        cudaComputeSubtract(reinterpret_cast<cf32*>(a.data()),
                            reinterpret_cast<cf32*>(b.data()),
                            reinterpret_cast<cf32*>(out.data()),
                            size,
                            blockSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation for element-wise multiplication (float
 * specialization).
 */
template <>
inline void cuda_mul<float>(std::span<float> a,
                            std::span<float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const unsigned int blockSize = 256;
    cudaError_t err =
        cudaComputeMultiply(reinterpret_cast<cf32*>(a.data()),
                            reinterpret_cast<cf32*>(b.data()),
                            reinterpret_cast<cf32*>(out.data()),
                            size,
                            blockSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

/**
 * @brief CUDA implementation for element-wise division (float specialization).
 */
template <>
inline void cuda_div<float>(std::span<float> a,
                            std::span<float> b,
                            std::span<float> out) {
    const size_t size = a.size();
    if (size == 0) {
        return;
    }

    CudaMemory cudaMem;
    const unsigned int blockSize = 256;
    cudaError_t err =
        cudaComputeDivide(reinterpret_cast<cf32*>(a.data()),
                          reinterpret_cast<cf32*>(b.data()),
                          reinterpret_cast<cf32*>(out.data()),
                          size,
                          blockSize);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

// Stubs for non-float types
template <typename T>
void cuda_add(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

template <typename T>
void cuda_sub(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

template <typename T>
void cuda_mul(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

template <typename T>
void cuda_div(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {
    throw std::runtime_error("CUDA operations only support float type");
}

#else // !__has_include(<driver_types.h>)

// Stubs when CUDA headers not available
template <typename T>
void cuda_add(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}
template <typename T>
void cuda_sub(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}
template <typename T>
void cuda_mul(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}
template <typename T>
void cuda_div(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}

#endif // __has_include(<driver_types.h>)
#else  // !HAHAHA_USE_CUDA

// Stubs when CUDA not enabled
template <typename T>
void cuda_add(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}
template <typename T>
void cuda_sub(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}
template <typename T>
void cuda_mul(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}
template <typename T>
void cuda_div(std::span<T> /* a */,
              std::span<T> /* b*/,
              std::span<T> /*out*/) {}

#endif // HAHAHA_USE_CUDA

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_GPU_CUDA_CUDA_ELEMENTWISE_KERNELS_H
