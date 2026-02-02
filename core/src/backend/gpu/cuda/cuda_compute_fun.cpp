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

#include "common/definitions.h"

using cf32 = hahaha::common::f32;
/**
 * @brief External C interface functions defined in CUDA files.
 */
extern "C" {
void cuda_compute_add_c_interface(float* input1,
                                  float* input2,
                                  float* output,
                                  size_t size,
                                  unsigned int blockSize);
void cuda_compute_subtract_c_interface(float* input1,
                                       float* input2,
                                       float* output,
                                       size_t size,
                                       unsigned int blockSize);
void cuda_compute_multiply_c_interface(float* input1,
                                       float* input2,
                                       float* output,
                                       size_t size,
                                       unsigned int blockSize);
void cuda_compute_divide_c_interface(float* input1,
                                     float* input2,
                                     float* output,
                                     size_t size,
                                     unsigned int blockSize);
}

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

    // Call C interface
    cuda_compute_add_c_interface((input1),
                                 (input2),
                                 (output),
                                 size,
                                 blockSize);

    // Check kernel launch error
    return cudaSuccess;
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

    // Call C interface
    cuda_compute_subtract_c_interface((input1),
                                      (input2),
                                      (output),
                                      size,
                                      blockSize);

    // Check kernel launch error
    return cudaSuccess;
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

    // Call C interface
    cuda_compute_multiply_c_interface((input1),
                                      (input2),
                                      (output),
                                      size,
                                      blockSize);

    // Check kernel launch error
    return cudaSuccess;
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

    // Call C interface
    cuda_compute_divide_c_interface(input1,
                                    input2,
                                    output,
                                    size,
                                    blockSize);

    // Check kernel launch error
    return cudaSuccess;
}

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
