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

#ifndef HAHAHA_CPUMEMORY_H_46C06FBBF7EF4ABA95410A3C5AE954CB
#define HAHAHA_CPUMEMORY_H_46C06FBBF7EF4ABA95410A3C5AE954CB

#include <cstring>
#include <span>
#include <stdexcept>

#include "backend/DeviceBuffer.h"
#include "backend/Memory.h"

namespace hahaha::backend {

/**
 * @brief CPU memory management implementation.
 * @details For CPU, "device" memory is plain heap; allocate/free and
 *          copy operations are standard host memory.
 */
class CpuMemory : public Memory {
  public:
    [[nodiscard]] DeviceBuffer allocate(size_t size) override {
        if (size == 0) {
            return {};
        }
        void* ptr = new char[size];
        return DeviceBuffer{reinterpret_cast<std::uintptr_t>(ptr), size};
    }

    void free(DeviceBuffer& buffer) override {
        if (buffer.address() == 0) {
            return;
        }
        delete[] reinterpret_cast<char*>(buffer.address());
        buffer.reset();
    }

    void copyHostToDevice(DeviceBuffer& dst,
                          std::span<const std::byte> src) override {
        if (dst.size() != src.size()) {
            throw std::runtime_error("Host to device copy size mismatch");
        }
        if (!src.empty()) {
            std::memcpy(
                reinterpret_cast<void*>(dst.address()), src.data(), src.size());
        }
    }

    void copyDeviceToHost(std::span<std::byte> dst,
                          const DeviceBuffer& src) override {
        if (dst.size() != src.size()) {
            throw std::runtime_error("Device to host copy size mismatch");
        }
        if (!dst.empty()) {
            std::memcpy(dst.data(),
                        reinterpret_cast<const void*>(src.address()),
                        src.size());
        }
    }

    void memset(DeviceBuffer& dst, int value, size_t count) override {
        if (count == 0 || dst.address() == 0) {
            return;
        }
        std::memset(reinterpret_cast<void*>(dst.address()),
                    static_cast<unsigned char>(value),
                    count);
    }
};

} // namespace hahaha::backend

#endif // HAHAHA_CPUMEMORY_H_46C06FBBF7EF4ABA95410A3C5AE954CB
