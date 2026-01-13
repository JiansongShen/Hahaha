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

#include "backend/gpu/cuda/CudaMemory.h"

namespace hahaha::backend {

void CudaMemory::copyDeviceToDevice(DeviceBuffer& dst,
                                    const DeviceBuffer& src) {
}

DeviceBuffer CudaMemory::allocate(size_t size) {
    return {};
}

void CudaMemory::copyDeviceToHost(std::span<std::byte> dst,
                                  const DeviceBuffer& src) {
}

void CudaMemory::memset(DeviceBuffer& dst, int value, size_t count) {
}

void CudaMemory::copyHostToDevice(DeviceBuffer& dst,
                                  std::span<const std::byte> src) {
}

} // namespace hahaha::backend
