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

#ifndef HAHAHA_CUDADEVICE_H_4854C266038C49758A81068C61BC2447
#define HAHAHA_CUDADEVICE_H_4854C266038C49758A81068C61BC2447

#include "CudaMemory.h"
#include "backend/Device.h"
#include "backend/gpu/GPUDevice.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <driver_types.h>
#endif
#endif

namespace hahaha::backend {

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

/**
 * @brief CUDA device implementation.
 */
class CudaDevice : public GPUDevice {
  public:
    /**
     * @brief Construct CUDA device with properties.
     * @param prop CUDA device properties.
     */
    explicit CudaDevice(cudaDeviceProp* prop) : prop_(prop) {
    }

    /**
     * @brief Construct CUDA device with type, ID and properties.
     * @param deviceType Device type.
     * @param deviceId Device ID.
     * @param prop CUDA device properties.
     */
    CudaDevice(const DeviceType deviceType,
               const std::uint8_t deviceId,
               cudaDeviceProp* prop)
        : Device(deviceType, deviceId), prop_(prop) {
    }

    /**
     * @brief Destructor.
     */
    ~CudaDevice();

    /**
     * @brief Allocate memory on device.
     * @param size Size in bytes.
     * @return DeviceBuffer.
     */
    DeviceBuffer allocate(size_t size) override {
        return memory_.allocate(size);
    }

    /**
     * @brief Deallocate memory on device.
     * @param buffer Buffer to deallocate.
     */
    void deallocate(DeviceBuffer buffer) override {
        memory_.free(buffer);
    }

  private:
    cudaDeviceProp* prop_ = nullptr;
    CudaMemory memory_;
};

#else  // !__has_include(<driver_types.h>)
// Stub implementation when CUDA headers not available
class CudaDevice : public GPUDevice {
  public:
    DeviceBuffer allocate(size_t) override {
        std::unreachable();
    }

    void deallocate(DeviceBuffer) override {
        std::unreachable();
    }
};
#endif // __has_include(<driver_types.h>)

#else  // !HAHAHA_USE_CUDA
// Stub implementation when CUDA not enabled
class CudaDevice : public GPUDevice {
  public:
    DeviceBuffer allocate(size_t) override {
        std::unreachable();
    }

    void deallocate(DeviceBuffer) override {
        std::unreachable();
    }
};
#endif // HAHAHA_USE_CUDA

} // namespace hahaha::backend

#endif // HAHAHA_CUDADEVICE_H_4854C266038C49758A81068C61BC2447
