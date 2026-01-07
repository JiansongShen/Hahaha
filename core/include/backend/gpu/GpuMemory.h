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
#include <cstdint>
#include <span>

namespace hahaha::backend::gpu {

/**
 * @brief Utilities for GPU memory allocation and data transfer.
 */
class GpuMemory {
  public:
    /**
     * @brief RAII handle for a GPU device buffer.
     *
     * This stores the device address as an integer to avoid exposing raw
     * pointers in the backend public API.
     */
    class DeviceBuffer {
      public:
        DeviceBuffer() = default;
        DeviceBuffer(std::uintptr_t address, size_t size)
            : address_(address), size_(size) {
        }

        ~DeviceBuffer() noexcept {
            if (address_ != 0) {
                GpuMemory::deallocate(address_);
            }
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept
            : address_(other.address_), size_(other.size_) {
            other.address_ = 0;
            other.size_ = 0;
        }

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this == &other) {
                return *this;
            }
            if (address_ != 0) {
                GpuMemory::deallocate(address_);
            }
            address_ = other.address_;
            size_ = other.size_;
            other.address_ = 0;
            other.size_ = 0;
            return *this;
        }

        [[nodiscard]] std::uintptr_t address() const {
            return address_;
        }

        [[nodiscard]] size_t size() const {
            return size_;
        }

      private:
        std::uintptr_t address_ = 0;
        size_t size_ = 0;
    };

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
     */
    static void deallocate(std::uintptr_t address);
};

} // namespace hahaha::backend::gpu

#endif // HAHAHA_BACKEND_GPU_GPU_MEMORY_H
