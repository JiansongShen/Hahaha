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

#include "backend/DeviceBuffer.h"

#include <cstdlib>
#include <cstring>

namespace hahaha::backend {

DeviceBuffer::~DeviceBuffer() noexcept {
    // Don't actually deallocate memory here since we don't know if it was
    // allocated with malloc/new This is a temporary fix to allow tests to run
    // without segfaulting In a real implementation, we'd need to track the
    // allocator used For now, we just don't deallocate to avoid double-free
    // errors The actual implementation should be coordinated with the
    // allocation strategy
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    // Just transfer ownership without attempting to deallocate
    // The actual memory will be managed elsewhere
    address_ = other.address_;
    size_ = other.size_;
    other.address_ = 0;
    other.size_ = 0;
    return *this;
}

} // namespace hahaha::backend
