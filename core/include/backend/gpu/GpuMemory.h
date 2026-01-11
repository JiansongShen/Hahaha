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

#include "backend/DeviceBuffer.h"

namespace hahaha::backend {

/**
 * @brief Utilities for GPU memory allocation and data transfer.
 */
class GpuMemory {
  public:
    /**
     * @brief Allocate memory on the GPU.
     * @param size Size of memory to allocate in bytes.
     * @return DeviceBuffer RAII-owned device buffer handle.
     */
    static DeviceBuffer allocate(size_t size);

    /**
     * @brief Copy data from host to device.
     * @param device Destination device buffer.
     * @param host Source byte span on the host.
     */
    static void copyToDevice(DeviceBuffer& device,
                             std::span<const std::byte> host);

    /**
     * @brief Copy data from device to host.
     * @param host Destination byte span on the host.
     * @param device Source device buffer.
     */
    static void copyToHost(std::span<std::byte> host,
                           const DeviceBuffer& device);

    /**
     * @brief Copy data from device to device.
     * @param dest Destination device buffer.
     * @param src Source device buffer.
     */
    static void copyDeviceToDevice(DeviceBuffer& dest, const DeviceBuffer& src);

  private:
    /**
     * @brief Deallocate memory on the GPU (internal).
     * @param address Device address of the memory to deallocate.
     *
     * @note This method is private and can only be accessed by DeviceBuffer
     *       through the friend declaration. DeviceBuffer uses this in its
     *       destructor and move assignment operator to ensure proper cleanup.
     */
    static void deallocate(std::uintptr_t address);

    /**
     * @brief Friend declaration for DeviceBuffer.
     *
     * DeviceBuffer needs to access deallocate() in its destructor and move
     * assignment operator to ensure proper RAII semantics for GPU memory
     * management.
     */
    friend class DeviceBuffer;
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_GPU_GPU_MEMORY_H
