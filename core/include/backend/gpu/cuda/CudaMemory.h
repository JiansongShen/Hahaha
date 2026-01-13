//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//
//

#ifndef HAHAHA_GPU_MEMORY_H_8D60664C7DC2476F894F2E21EB3CEE0C
#define HAHAHA_GPU_MEMORY_H_8D60664C7DC2476F894F2E21EB3CEE0C

#include "CudaDevice.h"
#include "CudaMemoryPool.h"
#include "backend/gpu/GPUMemory.h"

namespace hahaha::backend {
class CudaMemory : public GPUMemory {
  public:
    /**
     * @brief Allocates a block of memory on the device.
     * @param size The size of the memory block in bytes.
     * @return A DeviceBuffer object representing the allocated memory.
     */
    [[nodiscard]] DeviceBuffer allocate(size_t size) override;

    /**
     * @brief Copies data from host memory to device memory.
     * @param dst The destination device buffer.
     * @param src The source host memory span.
     */
    void copyHostToDevice(DeviceBuffer& dst,
                          std::span<const std::byte> src) override;

    /**
     * @brief Copies data from device memory to host memory.
     * @param dst The destination host memory span.
     * @param src The source device buffer.
     */
    void copyDeviceToHost(std::span<std::byte> dst,
                          const DeviceBuffer& src) override;

    /**
     * @brief Copies data from one device buffer to another.
     * @param dst The destination device buffer.
     * @param src The source device buffer.
     */
    void copyDeviceToDevice(DeviceBuffer& dst,
                            const DeviceBuffer& src) override;

    /**
     * @brief Sets a block of memory to a specified value.
     * @param dst The destination device buffer.
     * @param value The value to set (converted to unsigned char).
     * @param count The number of bytes to set.
     */
    void memset(DeviceBuffer& dst, int value, size_t count) override;

  private:
    CudaMemoryPool memoryPool_;
};
} // namespace hahaha::backend

#endif // HAHAHA_GPU_MEMORY_H_8D60664C7DC2476F894F2E21EB3CEE0C
