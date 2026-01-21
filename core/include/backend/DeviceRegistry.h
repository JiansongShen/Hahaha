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

#ifndef HAHAHA_BACKEND_DEVICE_REGISTRY_H
#define HAHAHA_BACKEND_DEVICE_REGISTRY_H

#include <memory>
#include <mutex>
#include <unordered_map>

#include "backend/Device.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaDevice.h"
#endif
#endif

#include "backend/cpu/CPUDevice.h"

namespace hahaha::backend {

/**
 * @brief Global device registry for managing all available compute devices.
 *
 * This class provides a singleton pattern to discover and manage all available
 * devices at runtime. It automatically detects CPU and CUDA devices (if
 * available) and provides shared pointers to global device instances.
 */
class DeviceRegistry {
  public:
    /**
     * @brief Get the singleton instance of DeviceRegistry.
     * @return Reference to the singleton DeviceRegistry.
     */
    static DeviceRegistry& getInstance() {
        static DeviceRegistry instance;
        return instance;
    }

    /**
     * @brief Get the default CPU device.
     * @return Shared pointer to the default CPU device.
     */
    [[nodiscard]] std::shared_ptr<Device> getCPUDevice() const {
        return cpuDevice_;
    }

    /**
     * @brief Get a CUDA device by ID.
     * @param deviceId CUDA device ID (default: 0).
     * @return Shared pointer to the CUDA device, or nullptr if not available.
     */
    [[nodiscard]] std::shared_ptr<Device>
    getCudaDevice(std::uint8_t deviceId = 0) const {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        auto it = cudaDevices_.find(deviceId);
        if (it != cudaDevices_.end()) {
            return it->second;
        }
#endif
#endif
        return nullptr;
    }

    /**
     * @brief Get a device by type and ID.
     * @param type Device type.
     * @param deviceId Device ID (default: 0).
     * @return Shared pointer to the device, or nullptr if not available.
     */
    [[nodiscard]] std::shared_ptr<Device>
    getDevice(DeviceType type, std::uint8_t deviceId = 0) const {
        switch (type) {
        case DeviceType::CPU:
            return cpuDevice_;
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        case DeviceType::CUDA:
            return getCudaDevice(deviceId);
#endif
#endif
        default:
            return nullptr;
        }
    }

    /**
     * @brief Check if a device type is available.
     * @param type Device type.
     * @return true if the device type is available, false otherwise.
     */
    [[nodiscard]] bool isDeviceAvailable(DeviceType type) const {
        switch (type) {
        case DeviceType::CPU:
            return cpuDevice_ != nullptr;
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        case DeviceType::CUDA:
            return !cudaDevices_.empty();
#endif
#endif
        default:
            return false;
        }
    }

    /**
     * @brief Get the default device (CPU).
     * @return Shared pointer to the default device.
     */
    [[nodiscard]] std::shared_ptr<Device> getDefaultDevice() const {
        return cpuDevice_;
    }

    /**
     * @brief Get all available CUDA devices.
     * @return Map of device ID to device shared pointer.
     */
    [[nodiscard]] const std::unordered_map<std::uint8_t,
                                           std::shared_ptr<Device>>&
    getAllCudaDevices() const {
        return cudaDevices_;
    }

  private:
    /**
     * @brief Private constructor for singleton pattern.
     * Automatically discovers and initializes all available devices.
     */
    DeviceRegistry() {
        initializeDevices();
    }

    /**
     * @brief Initialize all available devices.
     * This method discovers CPU and CUDA devices at runtime.
     */
    void initializeDevices() {
        // Always initialize CPU device
        cpuDevice_ = std::make_shared<CPUDevice>();

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        // Discover CUDA devices
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess && deviceCount > 0) {
            // Store device properties in member variable to ensure lifetime
            cudaProps_.resize(deviceCount);
            for (int i = 0; i < deviceCount; ++i) {
                err = cudaGetDeviceProperties(&cudaProps_[i], i);
                if (err == cudaSuccess) {
                    cudaDevices_[static_cast<std::uint8_t>(i)] =
                        std::make_shared<CudaDevice>(&cudaProps_[i]);
                }
            }
        }
#endif
#endif
    }

    // Prevent copying and assignment
    DeviceRegistry(const DeviceRegistry&) = delete;
    DeviceRegistry& operator=(const DeviceRegistry&) = delete;

    std::shared_ptr<Device> cpuDevice_;
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
    std::unordered_map<std::uint8_t, std::shared_ptr<Device>> cudaDevices_;
    std::vector<cudaDeviceProp> cudaProps_; // Store CUDA device properties
#endif
#endif
};

/**
 * @brief Convenience function to get the default CPU device.
 * @return Shared pointer to the default CPU device.
 */
inline std::shared_ptr<Device> getCPUDevice() {
    return DeviceRegistry::getInstance().getCPUDevice();
}

/**
 * @brief Convenience function to get a CUDA device.
 * @param deviceId CUDA device ID (default: 0).
 * @return Shared pointer to the CUDA device, or nullptr if not available.
 */
inline std::shared_ptr<Device> getCudaDevice(std::uint8_t deviceId = 0) {
    return DeviceRegistry::getInstance().getCudaDevice(deviceId);
}

/**
 * @brief Convenience function to get a device by type.
 * @param type Device type.
 * @param deviceId Device ID (default: 0).
 * @return Shared pointer to the device, or nullptr if not available.
 */
inline std::shared_ptr<Device> getDevice(DeviceType type,
                                         std::uint8_t deviceId = 0) {
    return DeviceRegistry::getInstance().getDevice(type, deviceId);
}

} // namespace hahaha::backend

#endif // HAHAHA_BACKEND_DEVICE_REGISTRY_H
