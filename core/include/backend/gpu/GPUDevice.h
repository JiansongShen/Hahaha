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
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef HAHAHA_GPUDEVICE_H_331F2FFB5CF742BB836F988DD6080929
#define HAHAHA_GPUDEVICE_H_331F2FFB5CF742BB836F988DD6080929
#include "GPUMemory.h"
#include "backend/Device.h"

namespace hahaha::backend {

class GPUDevice : public Device {
  public:
    explicit GPUDevice(const DeviceType type = DeviceType::CUDA) {
        this->type_ = type;
        this->id_ = 0;
    }

    GPUDevice(const DeviceType type, const std::uint8_t deviceId) {
        this->type_ = type;
        this->id_ = deviceId;
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    DeviceBuffer allocate(const size_t size) override {
        std::unreachable();
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void deallocate(DeviceBuffer buffer) override {
        std::unreachable();
    }
    void copyMemoryFrom(std::span<std::byte> src,
                        std::span<std::byte> dst,
                        const std::shared_ptr<Device>& dstDevice) override {
    }

    void copyMemoryTo(std::span<std::byte> src,
                      std::span<std::byte> dst,
                      const std::shared_ptr<Device>& srcDevice) override {
    }
};

} // namespace hahaha::backend

#endif // HAHAHA_GPUDEVICE_H_331F2FFB5CF742BB836F988DD6080929
