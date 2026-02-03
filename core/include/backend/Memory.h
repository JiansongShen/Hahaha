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

#ifndef HAHAHA_MEMORY_H_9AFD396F18A24E43AB191AAB0AC48C7B
#define HAHAHA_MEMORY_H_9AFD396F18A24E43AB191AAB0AC48C7B
#include <cstddef>
#include <span>

#include "backend/DeviceBuffer.h"

namespace hahaha::backend {

/**
 * @brief Base abstract interface class for unified heterogeneous memory
 * management
 * @details The core abstraction layer for all memory operations, implements the
 * strategy pattern for memory management. Contains all essential memory
 * operations: memory pool management, safe allocation/deallocation, byte-level
 * memory initialization, synchronous/asynchronous memory copy, device property
 * query. Pure abstract class: only pure virtual functions, no member variables,
 * no concrete implementation.
 */
class Memory {
  public:
    virtual ~Memory() = default;

    /**
     * @brief Allocates a block of memory on the device.
     * @param size The size of the memory block in bytes.
     * @return A DeviceBuffer object representing the allocated memory.
     */
    [[nodiscard]] virtual DeviceBuffer allocate(size_t size) = 0;

    /**
     * @brief Copies data from host memory to device memory.
     * @param dst The destination device buffer.
     * @param src The source host memory span.
     */
    virtual void copyHostToDevice(DeviceBuffer& dst,
                                  std::span<const std::byte> src) = 0;

    /**
     * @brief Copies data from device memory to host memory.
     * @param dst The destination host memory span.
     * @param src The source device buffer.
     */
    virtual void copyDeviceToHost(std::span<std::byte> dst,
                                  const DeviceBuffer& src) = 0;

    /**
     * @brief Sets a block of memory to a specified value.
     * @param dst The destination device buffer.
     * @param value The value to set (converted to unsigned char).
     * @param count The number of bytes to set.
     */
    virtual void memset(DeviceBuffer& dst, int value, size_t count) = 0;
};

} // namespace hahaha::backend

#endif // HAHAHA_MEMORY_H_9AFD396F18A24E43AB191AAB0AC48C7B
