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

#include <algorithm>
#include <driver_types.h>
#include <expected>
#include <memory>

#include "backend/gpu/cuda/cuda_memory.cuh"
#include "common/errors/Error.h"

namespace hahaha::backend {

std::expected<void*, common::Error>
CudaMemoryPool::allocateSmall(const size_t size) {
    const size_t blockIdx = getBlockIndexOfSize(size);
    if (const auto res = checkFreeBlockListExist(blockIdx); !res.isSuccess()) {
        return std::unexpected(res);
    }
    return allocateOnBlock(blockIdx);
}

std::expected<void*, common::Error>
CudaMemoryPool::allocateOnBlock(const size_t blockIdx) {
    if (blockIdx >= freeSmallBlocks_.size()) {
        throw std::runtime_error("Invalid block index");
    }

    // Try to get a free block
    SmallBlockMetadata* block = freeSmallBlocks_[blockIdx];

    // If no free block, try to split from larger block
    if (!block) {
        if (auto resError = requireSplitBlock(blockIdx);
            !resError.isSuccess()) {
            if (resError.code
                == common::ErrorCode::CudaSmallObjectMemoryPoolFull) {
                return std::unexpected(resError);
            }
            return std::unexpected(resError);
        }
        block = freeSmallBlocks_[blockIdx];
    }

    if (!block) {
        return std::unexpected(common::CudaDeviceOutOfMemoryError());
    }

    // Remove from free list
    freeSmallBlocks_[blockIdx] = block->next;
    if (block->next) {
        block->next->prev = nullptr;
    }
    block->prev = nullptr;
    block->next = nullptr;
    block->isAllocated = true;

    return block->gpuPtr;
}

void CudaMemoryPool::insertIntoFreeBlock(SmallBlockMetadata* metadata) {
    if (!metadata) {
        return;
    }

    const size_t blockIdx = metadata->blockIdx;
    if (blockIdx >= MaxSmallObjectPoolListSize) {
        throw std::runtime_error("Invalid block index");
    }

    const auto isFrontBuddy =
        reinterpret_cast<uintptr_t>(metadata->gpuPtr) & metadata->size;
    std::uintptr_t buddyBlockGpuPtr;
    if (isFrontBuddy) {
        buddyBlockGpuPtr =
            reinterpret_cast<uintptr_t>(metadata->gpuPtr) | metadata->size;
    } else {
        buddyBlockGpuPtr =
            reinterpret_cast<uintptr_t>(metadata->gpuPtr) & ~metadata->size;
    }
    if (const auto pair =
            smallBlockMap_.find(reinterpret_cast<void*>(buddyBlockGpuPtr));
        pair != smallBlockMap_.end()) {
        // can be merged into a bigger one
        if (pair->second->size == metadata->size
            && !pair->second->isAllocated) {
            smallBlockMap_.erase(pair);
            smallBlockMap_[metadata->gpuPtr] = metadata;
            metadata->size = metadata->size << 1;
            metadata->isAllocated = false;
            metadata->blockIdx = blockIdx + 1;
            freeSmallBlocks_[metadata->blockIdx]->prev = metadata;
            metadata->next = freeSmallBlocks_[metadata->blockIdx];
            metadata->prev = nullptr;
            freeSmallBlocks_[metadata->blockIdx] = metadata;
            return;
        }
    }
    metadata->isAllocated = false;
    freeSmallBlocks_[metadata->blockIdx]->prev = metadata;
    metadata->next = freeSmallBlocks_[metadata->blockIdx];
    metadata->prev = nullptr;
    freeSmallBlocks_[metadata->blockIdx] = metadata;
}

void CudaMemoryPool::insertIntoBigBlock(BigBlockMetadata* metadata) {
    if (!metadata) {
        return;
    }
    metadata->cacheLiveTimes = 0;
    metadata->isAllocated = false;
    bigBlocks_.push_back(std::unique_ptr<BigBlockMetadata>(metadata));
}

std::expected<CudaMemoryPool::BigBlockMetadata*, common::Error>
CudaMemoryPool::requireNewBigBlock(const size_t size) {
    void* gpuPtr = nullptr;

    if (const cudaError_t res = cudaMemoryAllocate(size, &gpuPtr);
        res != cudaSuccess) {
        return std::unexpected(common::CudaDeviceOutOfMemoryError());
    }

    auto metadata = std::make_unique<BigBlockMetadata>();
    metadata->gpuPtr = gpuPtr;
    metadata->size = size;
    metadata->isAllocated = true;
    metadata->cacheLiveTimes = 0;

    BigBlockMetadata* rawPtr = metadata.get();
    allocatedBigBlocks_.push_back(std::move(metadata));
    bigBlockMap_[gpuPtr] = rawPtr;

    return rawPtr;
}

CudaMemoryPool::BigBlockMetadata*
CudaMemoryPool::findCachedBigBlock(const size_t size) {
    BigBlockMetadata* bestfit = nullptr;

    for (auto& block : bigBlocks_) {
        if (!block || block->isAllocated) {
            continue;
        }

        if (block->size == size) {
            block->refreshCacheLiveTime();
            block->isAllocated = true;
            return block.get();
        }

        if (block->size > size) {
            if (bestfit == nullptr || block->size < bestfit->size) {
                bestfit = block.get();
            }
            block->cacheLiveTimes++;
        }
    }

    // If not found, try to create a new one
    if (bestfit == nullptr) {
        const auto newBlock = requireNewBigBlock(size);
        if (!newBlock.has_value()) {
            return nullptr;
        }
        return newBlock.value();
    }

    bestfit->refreshCacheLiveTime();
    bestfit->isAllocated = true;
    return bestfit;
}

size_t CudaMemoryPool::getMemoryNeeded(const size_t size) {
    return size + sizeof(size_t); // Size + metadata overhead
}

void CudaMemoryPool::free(void* ptr) {
    if (!ptr) {
        return;
    }

    // Check if it's a big block
    if (const auto bigIt = bigBlockMap_.find(ptr);
        bigIt != bigBlockMap_.end()) {
        if (BigBlockMetadata* metadata = bigIt->second;
            metadata && metadata->isAllocated) {
            metadata->isAllocated = false;
            insertIntoBigBlock(metadata);
        }
        return;
    }

    // Check if it's a small block
    if (const auto smallIt = smallBlockMap_.find(ptr);
        smallIt != smallBlockMap_.end()) {
        if (SmallBlockMetadata* metadata = smallIt->second;
            metadata && metadata->isAllocated) {
            metadata->isAllocated = false;
            insertIntoFreeBlock(metadata);
        }
        return;
    }

    if (const auto block = freeSmallBlocks_[MaxSmallObjectPoolListSize - 1];
        block && !block->isAllocated) {
        const auto next = block->next;
        freeSmallBlocks_[MaxSmallObjectPoolListSize - 1] = next;
        next->prev = nullptr;
        const auto maxLevelBlock = smallBlockStorage_;
        while (maxLevelBlock) {
            if (maxLevelBlock->gpuPtr == block->gpuPtr) {
                smallBlockStorage_ = maxLevelBlock->next;
                if (smallBlockStorage_) {
                    smallBlockStorage_->prev = nullptr;
                }
                cudaMemoryFree(maxLevelBlock->gpuPtr);
                break;
            }
        }
    }

    // Not found in our maps, might be direct allocation - free directly
    cudaMemoryFree(ptr);
}

std::expected<void*, common::Error>
CudaMemoryPool::allocateBig(const size_t size) {
    if (BigBlockMetadata* bigBlock = findCachedBigBlock(size);
        bigBlock != nullptr) {
        return bigBlock->gpuPtr;
    }

    return std::unexpected(common::CudaDeviceOutOfMemoryError());
}

size_t CudaMemoryPool::getBlockIndexOfSize(const size_t size) {
    size_t idx = 0;
    size_t currentSize = BaseMemoryBlockSize;

    while (size > currentSize
           && idx
               < MaxSmallObjectPoolListSize) { // Limit to prevent infinite loop
        currentSize *= 2;
        ++idx;
    }

    return idx;
}

common::Error CudaMemoryPool::requireSplitBlock(const size_t blockIdx) {
    if (blockIdx >= MaxSmallObjectPoolListSize) { // Safety limit
        return common::CudaSmallObjectMemoryPoolFullError();
    }

    if (blockIdx == MaxSmallObjectPoolListSize - 1) {
        void* ptr = nullptr;
        const cudaError_t res =
            cudaMemoryAllocate(BaseMemoryBlockSize << blockIdx, &ptr);
        if (res != cudaSuccess) {
            return common::CudaDeviceOutOfMemoryError();
        }
        freeSmallBlocks_[blockIdx] = new SmallBlockMetadata();
        freeSmallBlocks_[blockIdx]->gpuPtr = ptr;
        freeSmallBlocks_[blockIdx]->size = BaseMemoryBlockSize << blockIdx;
        freeSmallBlocks_[blockIdx]->blockIdx = blockIdx;

        if (!smallBlockStorage_) {
            smallBlockStorage_ =
                new SmallBlockMetadata(*freeSmallBlocks_[blockIdx]);
        } else {
            smallBlockStorage_->prev =
                new SmallBlockMetadata(*freeSmallBlocks_[blockIdx]);
            smallBlockStorage_->prev->next = smallBlockStorage_;
            smallBlockStorage_ = smallBlockStorage_->prev;
        }
        return common::Error::Success();
    }

    const size_t newBlockIdx = blockIdx + 1;

    const SmallBlockMetadata* memoryBlock = freeSmallBlocks_[newBlockIdx];

    // If no block at this level, try to split from even larger block
    if (!memoryBlock) {
        if (const auto resError = requireSplitBlock(newBlockIdx);
            !resError.isSuccess()) {
            return resError;
        }
        memoryBlock = freeSmallBlocks_[newBlockIdx];
    }

    if (!memoryBlock) {
        return common::CudaSmallObjectMemoryPoolFullError();
    }

    // Remove from free list
    freeSmallBlocks_[newBlockIdx] = memoryBlock->next;
    if (memoryBlock->next) {
        memoryBlock->next->prev = nullptr;
    }

    // Calculate sizes
    const size_t blockSize = BaseMemoryBlockSize << blockIdx;
    const size_t halfSize = blockSize / 2;

    // Create two new blocks
    void* firstGpuPtr = memoryBlock->gpuPtr;
    void* secondGpuPtr = static_cast<char*>(firstGpuPtr) + halfSize;

    // Create metadata for first half
    const auto firstMeta = new SmallBlockMetadata();
    firstMeta->gpuPtr = firstGpuPtr;
    firstMeta->size = halfSize;
    firstMeta->blockIdx = blockIdx;
    firstMeta->isAllocated = false;

    // Create metadata for second half
    const auto secondMeta = new SmallBlockMetadata();
    secondMeta->gpuPtr = secondGpuPtr;
    secondMeta->size = halfSize;
    secondMeta->blockIdx = blockIdx;
    secondMeta->isAllocated = false;

    // Link them
    firstMeta->next = secondMeta;
    secondMeta->prev = firstMeta;
    // Update maps
    smallBlockMap_[firstGpuPtr] = firstMeta;
    smallBlockMap_[secondGpuPtr] = secondMeta;

    // Insert into free list
    freeSmallBlocks_[blockIdx] = firstMeta;

    return common::Error::Success();
}

common::Error CudaMemoryPool::checkFreeBlockListExist(const size_t blockIdx) {
    if (blockIdx >= MaxSmallObjectPoolListSize) { // Safety limit
        return common::InvalidArgumentError();
    }
    return common::Error::Success();
}

} // namespace hahaha::backend

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
