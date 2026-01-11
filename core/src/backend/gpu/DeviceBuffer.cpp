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

// #include "backend/gpu/DeviceBuffer.h"

// #include "backend/gpu/GpuMemory.h"

// namespace hahaha::backend {

// DeviceBuffer::~DeviceBuffer() noexcept {
//     if (address_ != 0) {
//         GpuMemory::deallocate(address_);
//     }
// }

// DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
//     if (this == &other) {
//         return *this;
//     }
//     if (address_ != 0) {
//         GpuMemory::deallocate(address_);
//     }
//     address_ = other.address_;
//     size_ = other.size_;
//     other.address_ = 0;
//     other.size_ = 0;
//     return *this;
// }

// } // namespace hahaha::backend
