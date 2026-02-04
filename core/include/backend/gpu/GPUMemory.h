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

#ifndef HAHAHA_BACKEND_GPU_GPU_MEMORY_H
#define HAHAHA_BACKEND_GPU_GPU_MEMORY_H

#include <cstddef>
#include <span>
#include <utility>

#include "backend/DeviceBuffer.h"
#include "backend/Memory.h"

namespace hahaha::backend {

/**
 * @brief Utilities for GPU memory allocation and data transfer.
 */
class GPUMemory : public Memory {
  public:
    /**
     * @brief Allocates a block of memory on the device.
     * @param size The size of the memory block in bytes.
     * @return A DeviceBuffer object representing the allocated memory.
     */
    [[nodiscard]] DeviceBuffer allocate(size_t size) override {
        std::unreachable();
    }

    void free(DeviceBuffer&) override {
        std::unreachable();
    }

    /**
     * @brief Copies data from host memory to device memory.
     * @param dst The destination device buffer.
     * @param src The source host memory span.
     */
    void copyHostToDevice(DeviceBuffer& dst,
                          std::span<const std::byte> src) override {
        std::unreachable();
    };

    /**
     * @brief Copies data from device memory to host memory.
     * @param dst The destination host memory span.
     * @param src The source device buffer.
     */
    void copyDeviceToHost(std::span<std::byte> dst,
                          const DeviceBuffer& src) override {
        std::unreachable();
    };

    /**
     * @brief Sets a block of memory to a specified value.
     * @param dst The destination device buffer.
     * @param value The value to set (converted to unsigned char).
     * @param count The number of bytes to set.
     */
    void memset(DeviceBuffer& dst, int value, size_t count) override {
        std::unreachable();
    };
    friend class DeviceBuffer;
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_GPU_GPU_MEMORY_H
