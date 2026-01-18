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

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

#include "backend/gpu/cuda/CudaMemoryPool.h"

#include <driver_types.h>
#include <expected>
#include <memory>
#include <span>

#include "backend/gpu/cuda/cuda_memory.cuh"
#include "common/errors/Error.h"

namespace hahaha::backend {

std::expected<void*, common::Error>
CudaMemoryPool::allocate(const size_t size) {
    size_t blockIdx = getBlockIndexOfSize(size);
    auto res = checkFreeBlockExist(blockIdx);
    if (res.isSuccess()) {
        return allocateOnBlock(blockIdx);
    }
    auto memAllocateRes = requireMoreMemoryBlock(getMemoryNeeded(size));
    if (!memAllocateRes.has_value()) {
        return memAllocateRes;
    }

    const auto newBlockPtr =
        static_cast<CudaMemoryBlock*>(memAllocateRes.value());
    newBlockPtr->size = size;

    return nullptr;
}

std::expected<void*, common::Error>
CudaMemoryPool::requireMoreMemoryBlock(const size_t size) {

#ifdef HAHAHA_USE_CUDA
    void* ptr;
    const cudaError_t errInfo = cudaMemoryAllocate(size, &ptr);

    if (errInfo == cudaSuccess) {
        return ptr;
    }
    switch (errInfo) {
    case cudaErrorMemoryAllocation:
        return std::unexpected(common::CudaDeviceOutOfMemoryError());
        break;

    default:
        throw std::runtime_error("Unknown Cuda Error");
    }

#endif

    return std::unexpected(common::CudaDeviceOutOfMemoryError());
}

void CudaMemoryPool::insertIntoFreeBlock(CudaMemoryBlock* block) {
    const size_t blockIdx = getBlockIndexOfSize(block->size);
    auto firstBlock = freeBlocks_[blockIdx];
    for (auto& next = firstBlock; next != nullptr; next = next->next) {
        if (next->size >= block->size) {
            block->next = next;
            firstBlock = block;
            return;
        }
    }
}

size_t CudaMemoryPool::getMemoryNeeded(size_t size) {
    return size + (sizeof(size_t) + sizeof(std::unique_ptr<CudaMemoryBlock>));
}

void* CudaMemoryPool::allocateOnBlock(size_t blockIdx) {
    const auto& block = freeBlocks_[blockIdx];
    freeBlocks_[blockIdx] = block->next;
    block->next = nullptr;

    return reinterpret_cast<char*>(block) + sizeof(size_t);
}

void CudaMemoryPool::free(void* ptr) {
}

size_t CudaMemoryPool::getBlockIndexOfSize(size_t size) {
    size_t idx = 0;
    while (size > 32) {
        size /= 2;
        ++idx;
    }
    return idx;
}

common::Error CudaMemoryPool::checkFreeBlockExist(const size_t size) const {
    if (size >= freeBlocks_.size()) {
        return common::InvalidArgumentError();
    }
    return {};
}

} // namespace hahaha::backend
#endif
#endif
