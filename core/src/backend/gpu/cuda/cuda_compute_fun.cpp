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

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

#include "backend/gpu/cuda/cuda_compute_fun.h"

#include <cuda_runtime.h>
#include <driver_types.h>

// Forward declarations of CUDA kernels
extern __global__ void
compute_add(cf32* input1, cf32* input2, cf32* output, size_t size);
extern __global__ void
compute_subtract(cf32* input1, cf32* input2, cf32* output, size_t size);
extern __global__ void
compute_multiply(cf32* input1, cf32* input2, cf32* output, size_t size);
extern __global__ void
compute_divide(cf32* input1, cf32* input2, cf32* output, size_t size);

namespace hahaha::backend {

cudaError_t cudaComputeAdd(cf32* input1,
                           cf32* input2,
                           cf32* output,
                           size_t size,
                           unsigned int blockSize) {
    if (input1 == nullptr || input2 == nullptr || output == nullptr
        || size == 0) {
        return cudaErrorInvalidValue;
    }

    // Calculate grid size
    unsigned int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    compute_add<<<gridSize, blockSize>>>(input1, input2, output, size);

    // Check for kernel launch errors
    return cudaGetLastError();
}

cudaError_t cudaComputeSubtract(cf32* input1,
                                cf32* input2,
                                cf32* output,
                                size_t size,
                                unsigned int blockSize) {
    if (input1 == nullptr || input2 == nullptr || output == nullptr
        || size == 0) {
        return cudaErrorInvalidValue;
    }

    // Calculate grid size
    unsigned int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    compute_subtract<<<gridSize, blockSize>>>(input1, input2, output, size);

    // Check for kernel launch errors
    return cudaGetLastError();
}

cudaError_t cudaComputeMultiply(cf32* input1,
                                cf32* input2,
                                cf32* output,
                                size_t size,
                                unsigned int blockSize) {
    if (input1 == nullptr || input2 == nullptr || output == nullptr
        || size == 0) {
        return cudaErrorInvalidValue;
    }

    // Calculate grid size
    unsigned int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    compute_multiply<<<gridSize, blockSize>>>(input1, input2, output, size);

    // Check for kernel launch errors
    return cudaGetLastError();
}

cudaError_t cudaComputeDivide(cf32* input1,
                              cf32* input2,
                              cf32* output,
                              size_t size,
                              unsigned int blockSize) {
    if (input1 == nullptr || input2 == nullptr || output == nullptr
        || size == 0) {
        return cudaErrorInvalidValue;
    }

    // Calculate grid size
    unsigned int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    compute_divide<<<gridSize, blockSize>>>(input1, input2, output, size);

    // Check for kernel launch errors
    return cudaGetLastError();
}

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
