// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//
//

#ifndef HAHAHA_BACKEND_GPU_DEVICE_BUFFER_H
#define HAHAHA_BACKEND_GPU_DEVICE_BUFFER_H

#include <cstddef>
#include <cstdint>

namespace hahaha::backend {

// Forward declaration
class GPUMemory;

/**
 * @brief Handle for a device buffer (CPU or GPU).
 *
 * This stores the device address as an integer to avoid exposing raw
 * pointers in the backend public API. This class is not RAII: the destructor
 * does not deallocate. The caller must call Device::deallocate(buffer) when
 * the buffer is no longer needed, or memory will leak.
 */
class DeviceBuffer {
  public:
    /**
     * @brief Default constructor. Creates an empty buffer with null address.
     */
    DeviceBuffer() = default;

    /**
     * @brief Construct a DeviceBuffer with the given address and size.
     * @param address The device address as an integer.
     * @param size The size of the buffer in bytes.
     */
    DeviceBuffer(const std::uintptr_t address, const size_t size)
        : address_(address), size_(size) {
    }

    /**
     * @brief Destructor. Does not deallocate; caller must call
     *        Device::deallocate(buffer) to free the memory.
     */
    ~DeviceBuffer() noexcept;

    /**
     * @brief Copy constructor is deleted. DeviceBuffer is non-copyable.
     */
    DeviceBuffer(const DeviceBuffer&) = delete;

    /**
     * @brief Copy assignment is deleted. DeviceBuffer is non-copyable.
     */
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /**
     * @brief Move constructor. Transfers ownership of the buffer.
     * @param other The source DeviceBuffer to move from. After move, other
     *              will have a null address and zero size.
     */
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : address_(other.address_), size_(other.size_) {
        other.address_ = 0;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator. Transfers ownership (does not
     *        deallocate the previous buffer; caller is responsible for
     *        deallocation).
     * @param other The source DeviceBuffer to move from. After move, other
     *              will have a null address and zero size.
     * @return DeviceBuffer& Reference to this.
     */
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    /**
     * @brief Clear this handle to empty without deallocating. Call
     *        Device::deallocate(buffer) (or Memory::free(buffer)) before
     *        reset() to avoid leaking memory.
     */
    void reset() noexcept {
        address_ = 0;
        size_ = 0;
    }

    /**
     * @brief Get the device address of the buffer.
     * @return std::uintptr_t The device address as an integer.
     */
    [[nodiscard]] std::uintptr_t address() const {
        return address_;
    }

    /**
     * @brief Get the size of the buffer.
     * @return size_t The size of the buffer in bytes.
     */
    [[nodiscard]] size_t size() const {
        return size_;
    }

  private:
    std::uintptr_t address_ = 0;
    size_t size_ = 0;

    friend class GPUMemory;
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_GPU_DEVICE_BUFFER_H
