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

#ifndef HAHAHA_BACKEND_DEVICE_H
#define HAHAHA_BACKEND_DEVICE_H

#include <cstdint>
#include <string>

#include "DeviceBuffer.h"

namespace hahaha::backend {

/**
 * @brief Types of devices supported for computation.
 */
enum class DeviceType : std::uint8_t {
    CPU = 0,
    CUDA = 1, // NVIDIA GPUs
    HIP = 2,  // AMD GPUs
    MPS = 3,  // Apple Silicon
    XLA = 4,  // TPUs / OpenXLA
    COMPILE_TIME_MAX = 5
};

/**
 * @brief Represents a compute device where data resides and operations occur.
 */
class alignas(8) Device {
  public:
    virtual ~Device() = default;

    /** @brief Default constructor (CPU, ID 0). */
    Device() = default;

    /**
     * @brief Construct a Device with type and ID.
     * @param deviceType Device type.
     * @param deviceId Device ID.
     */
    explicit Device(const DeviceType deviceType,
                    const std::uint8_t deviceId = 0)
        : type_(deviceType), id_(deviceId) {
    }

    /** @brief Check if two devices are identical. */
    bool operator==(const Device& other) const {
        return type_ == other.type_ && id_ == other.id_;
    }

    /** @brief Check if two devices are different. */
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    /** @brief Get a string representation of the device. */
    [[nodiscard]] std::string toString() const {
        std::string deviceName;
        switch (type_) {
        case DeviceType::CPU:
            deviceName = "CPU";
            break;
        case DeviceType::CUDA:
            deviceName = "CUDA";
            break;
        case DeviceType::HIP:
            deviceName = "HIP";
            break;
        case DeviceType::MPS:
            deviceName = "MPS";
            break;
        case DeviceType::XLA:
            deviceName = "XLA";
            break;
        default:
            deviceName = "Unknown";
            break;
        }
        return deviceName + ":" + std::to_string(id_);
    }

    virtual DeviceBuffer allocate(size_t size) = 0;

    virtual void deallocate(DeviceBuffer buffer) = 0;

    [[nodiscard]] DeviceType getType() const {
        return type_;
    }

    [[nodiscard]] std::uint8_t getId() const {
        return id_;
    }

  protected:
    DeviceType type_ = DeviceType::CPU; /**< Type of the device. */
    std::uint8_t id_ =
        0; /**< Unique identifier for multiple devices of the same type. */
};

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_H
