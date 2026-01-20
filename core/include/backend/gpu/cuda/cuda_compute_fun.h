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

#ifndef HAHAHA_GPU_COMPUTE_FUN_H_92EE2F9B5A5C4785A88CABD334003EA8
#define HAHAHA_GPU_COMPUTE_FUN_H_92EE2F9B5A5C4785A88CABD334003EA8

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cstddef>
#include <driver_types.h>

#include "common/definitions.h"

namespace hahaha::backend {

using cf32 = hahaha::common::f32;

/**
 * @brief Perform element-wise addition on GPU: output[i] = input1[i] +
 * input2[i]
 * @param input1 First input array (device pointer)
 * @param input2 Second input array (device pointer)
 * @param output Output array (device pointer)
 * @param size Number of elements
 * @param blockSize Number of threads per block (default: 256)
 * @return CUDA error code
 */
cudaError_t cudaComputeAdd(cf32* input1,
                           cf32* input2,
                           cf32* output,
                           size_t size,
                           unsigned int blockSize = 256);

/**
 * @brief Perform element-wise subtraction on GPU: output[i] = input1[i] -
 * input2[i]
 * @param input1 First input array (device pointer)
 * @param input2 Second input array (device pointer)
 * @param output Output array (device pointer)
 * @param size Number of elements
 * @param blockSize Number of threads per block (default: 256)
 * @return CUDA error code
 */
cudaError_t cudaComputeSubtract(cf32* input1,
                                cf32* input2,
                                cf32* output,
                                size_t size,
                                unsigned int blockSize = 256);

/**
 * @brief Perform element-wise multiplication on GPU: output[i] = input1[i] *
 * input2[i]
 * @param input1 First input array (device pointer)
 * @param input2 Second input array (device pointer)
 * @param output Output array (device pointer)
 * @param size Number of elements
 * @param blockSize Number of threads per block (default: 256)
 * @return CUDA error code
 */
cudaError_t cudaComputeMultiply(cf32* input1,
                                cf32* input2,
                                cf32* output,
                                size_t size,
                                unsigned int blockSize = 256);

/**
 * @brief Perform element-wise division on GPU: output[i] = input1[i] /
 * input2[i]
 * @param input1 First input array (device pointer, dividend)
 * @param input2 Second input array (device pointer, divisor)
 * @param output Output array (device pointer)
 * @param size Number of elements
 * @param blockSize Number of threads per block (default: 256)
 * @return CUDA error code
 */
cudaError_t cudaComputeDivide(cf32* input1,
                              cf32* input2,
                              cf32* output,
                              size_t size,
                              unsigned int blockSize = 256);

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA

#endif // HAHAHA_GPU_COMPUTE_FUN_H_92EE2F9B5A5C4785A88CABD334003EA8
