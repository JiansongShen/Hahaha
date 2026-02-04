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

#include "backend/gpu/cuda/CudaMemory.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <span>
#include <stdexcept>

#include "backend/gpu/cuda/cuda_memory.cuh"
#include "utils/log/Logger.h"

namespace hahaha::backend {

void CudaMemory::free(DeviceBuffer& deviceBuffer) {
    if (deviceBuffer.address() == 0) {
        return;
    }

    void* ptr = reinterpret_cast<void*>(deviceBuffer.address());
    memoryPool_.free(ptr);
    deviceBuffer.reset();
}

DeviceBuffer CudaMemory::allocate(size_t size) {
    if (size == 0) {
        return {};
    }

    if (size < smallMemoryBlockThreshold_) {
        return allocateSmall(size);
    }

    return allocateBig(size);
}

void CudaMemory::copyDeviceToHost(std::span<std::byte> dst,
                                  const DeviceBuffer& src) {
    if (dst.size() != src.size()) {
        throw std::runtime_error("Device to host copy size mismatch");
    }

    const cudaError_t err =
        cudaMemoryCopy(dst.data(),
                       reinterpret_cast<const void*>(src.address()),
                       src.size(),
                       cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA device to host copy failed: ")
                                 + cudaGetErrorString(err));
    }
}

void CudaMemory::memset(DeviceBuffer& dst, int value, size_t count) {
    if (count == 0 || dst.address() == 0) {
        return;
    }

    const cudaError_t err =
        cudaMemorySet(reinterpret_cast<void*>(dst.address()), value, count);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA memset failed: ")
                                 + cudaGetErrorString(err));
    }
}

DeviceBuffer CudaMemory::allocateSmall(size_t size) {
    if (auto expected = memoryPool_.allocateSmall(size); expected.has_value()) {
        return DeviceBuffer{reinterpret_cast<std::uintptr_t>(expected.value()),
                            size};
    }

    warn(std::format("CUDA has not enough memory, need {}", size));
    return {};
}

DeviceBuffer CudaMemory::allocateBig(size_t size) {
    if (auto res = memoryPool_.allocateBig(size); res.has_value()) {
        return DeviceBuffer{reinterpret_cast<std::uintptr_t>(res.value()),
                            size};
    }

    warn(std::format("CUDA has not enough memory, need {}", size));
    return {};
}

void CudaMemory::copyHostToDevice(DeviceBuffer& dst,
                                  std::span<const std::byte> src) {
    if (dst.size() != src.size()) {
        throw std::runtime_error("Host to device copy size mismatch");
    }

    const cudaError_t err =
        cudaMemoryCopy(reinterpret_cast<void*>(dst.address()),
                       src.data(),
                       src.size(),
                       cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA host to device copy failed: ")
                                 + cudaGetErrorString(err));
    }
}

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>) && __has_include(<cuda_runtime_api.h>)
#endif // HAHAHA_USE_CUDA
