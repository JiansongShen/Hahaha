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

#ifndef HAHAHA_CPUDEVICE_H_4C28A153F750404B93237EDDBE6463C1
#define HAHAHA_CPUDEVICE_H_4C28A153F750404B93237EDDBE6463C1

#include <cstring>

#include "backend/Device.h"
#include "backend/cpu/CpuMemory.h"

namespace hahaha::backend {
class CPUDevice : public Device {
  public:
    CPUDevice() {
        this->type_ = DeviceType::CPU;
        this->id_ = 0;
    }

    DeviceBuffer allocate(const size_t size) override {
        return memory_.allocate(size);
    }

    void deallocate(DeviceBuffer buffer) override {
        memory_.free(buffer);
    }

    ~CPUDevice() override = default;

    void copyMemoryFromThis(std::span<std::byte> src,
                            std::span<std::byte> dst,
                            const std::shared_ptr<Device>& dstDevice) override {
        if (dstDevice->getType() == DeviceType::CPU) {
            std::memcpy(dst.data(), src.data(), src.size());
        } else if (dstDevice->getType() == DeviceType::CUDA) {
            dstDevice->copyMemoryToThis(src, dst, dstDevice);
        }
    }

    void copyMemoryToThis(std::span<std::byte> src,
                          std::span<std::byte> dst,
                          const std::shared_ptr<Device>& srcDevice) override {
        if (srcDevice->getType() == DeviceType::CPU) {
            std::memcpy(dst.data(), src.data(), src.size());
        } else if (srcDevice->getType() == DeviceType::CUDA) {
            srcDevice->copyMemoryFromThis(src, dst, srcDevice);
        }
    }

  private:
    CpuMemory memory_;
};
} // namespace hahaha::backend

#endif // HAHAHA_CPUDEVICE_H_4C28A153F750404B93237EDDBE6463C1
