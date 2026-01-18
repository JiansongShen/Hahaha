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

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <driver_types.h>
#endif
#endif

#include "CudaMemory.h"
#include "backend/Device.h"
#include "backend/gpu/GPUDevice.h"

namespace hahaha::backend {

#if defined(HAHAHA_USE_CUDA) && __has_include(<driver_types.h>)
class CudaDevice : public GPUDevice {
  public:
    explicit CudaDevice(cudaDeviceProp* prop) : prop_(prop) {
    }

    CudaDevice(const DeviceType deviceType,
               const std::uint8_t deviceId,
               cudaDeviceProp* prop)
        : Device(deviceType, deviceId), prop_(prop) {
    }

    ~CudaDevice();

    DeviceBuffer allocate(size_t size) {
        return;
    }

    void deallocate(DeviceBuffer buffer) {
        return {};
    }

  private:
    cudaDeviceProp* prop_ = nullptr;
    CudaMemory memory_;
};
#else
class CudaDevice : public GPUDevice {
  public:
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    DeviceBuffer allocate(size_t size) override {
        std::unreachable();
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void deallocate(DeviceBuffer buffer) override {
        std::unreachable();
    }
};
#endif

} // namespace hahaha::backend

#endif // HAHAHA_CUDADEVICE_H_4854C266038C49758A81068C61BC2447
