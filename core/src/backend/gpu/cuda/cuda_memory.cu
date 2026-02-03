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
#if __has_include(<driver_types.h>) && __has_include(<cuda_runtime_api.h>)

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "backend/gpu/cuda/cuda_memory.cuh"

namespace hahaha::backend {

cudaError_t cudaMemoryAllocate(size_t size, void** ptr) {
    if (ptr == nullptr || size == 0) {
        return cudaErrorInvalidValue;
    }
    return cudaMalloc(ptr, size);
}

cudaError_t cudaMemoryFree(void* ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
    return cudaFree(ptr);
}

cudaError_t cudaMemoryCopy(void* dst,
                           const void* src,
                           size_t count,
                           cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (count == 0) {
        return cudaSuccess;
    }
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemorySet(void* ptr, int value, size_t count) {
    if (ptr == nullptr || count == 0) {
        return cudaSuccess;
    }
    return cudaMemset(ptr, value, count);
}

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>) && __has_include(<cuda_runtime_api.h>)
#endif // HAHAHA_USE_CUDA
