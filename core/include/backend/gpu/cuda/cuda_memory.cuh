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

#ifndef HAHAHA_CUDA_MEMORY_CUH_9F482603BCCC4C15924721026F992DB7
#define HAHAHA_CUDA_MEMORY_CUH_9F482603BCCC4C15924721026F992DB7

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <driver_types.h>

namespace hahaha::backend {

/**
 * @brief Allocate memory on CUDA device.
 * @param size Size in bytes.
 * @param ptr Output pointer to allocated memory.
 * @return CUDA error code.
 */
cudaError_t cudaMemoryAllocate(size_t size, void** ptr);

/**
 * @brief Free memory on CUDA device.
 * @param ptr Pointer to memory to free.
 * @return CUDA error code.
 */
cudaError_t cudaMemoryFree(void* ptr);

/**
 * @brief Set memory to a value on CUDA device.
 * @param ptr Pointer to memory.
 * @param value Value to set.
 * @param count Number of bytes.
 * @return CUDA error code.
 */
cudaError_t cudaMemorySet(void* ptr, int value, size_t count);

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA

#endif // HAHAHA_CUDA_MEMORY_CUH_9F482603BCCC4C15924721026F992DB7
