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

CudaMemoryPool::CudaMemoryPool() {
    freeSmallBlocks_ = {};
    freeSmallBlocks_.resize(MaxSmallObjectPoolListSize, nullptr);
}

CudaMemoryPool::~CudaMemoryPool() {
    // Free all big blocks
    for (const auto& block : freeBigBlocks_) {
        if (block && block->gpuPtr) {
            cudaMemoryFree(block->gpuPtr);
        }
    }
    freeBigBlocks_.clear();

    for (const auto& block : allocatedBigBlocks_) {
        if (block && block->gpuPtr) {
            cudaMemoryFree(block->gpuPtr);
        }
    }
    allocatedBigBlocks_.clear();
    bigBlockMap_.clear();

    // Free top-level GPU blocks and delete their metadata (smallBlockStorage_
    // holds copies of top-level block metadata only)
    const SmallBlockMetadata* current = smallBlockStorage_;
    while (current) {
        const SmallBlockMetadata* next = current->next;
        if (current->gpuPtr) {
            cudaMemoryFree(current->gpuPtr);
        }
        delete current;
        current = next;
    }
    smallBlockStorage_ = nullptr;

    // Clear free list heads before deleting map entries so we do not hold
    // dangling pointers (avoids debug heap assertion on teardown)
    for (auto& head : freeSmallBlocks_) {
        head = nullptr;
    }

    smallBlockMap_.clear();
}

std::expected<void*, common::Error>
CudaMemoryPool::allocateSmall(const size_t size) {
    if (size == 0) {
        return std::unexpected(common::InvalidArgumentError());
    }
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

    // If no free block, try to split from a larger block
    if (!block) {
        if (auto resError = requireSplitBlock(blockIdx); !resError.isSuccess()) {
            if (resError.code == common::ErrorCode::CudaSmallObjectMemoryPoolFull) {
                return std::unexpected(resError);
            }
            return std::unexpected(resError);
        }
        block = freeSmallBlocks_[blockIdx];
    }

    if (!block) {
        return std::unexpected(common::CudaDeviceOutOfMemoryError());
    }

    // Remove from the free list
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

    // Calculate buddy address: buddyPtr = basePtr XOR blockSize
    const auto basePtr = reinterpret_cast<std::uintptr_t>(metadata->gpuPtr);
    const std::uintptr_t buddyBlockGpuPtr = basePtr ^ metadata->size;

    if (metadata->blockIdx == MaxSmallObjectPoolListSize - 1) {
        metadata->isAllocated = false;
        metadata->prev = nullptr;
        metadata->next = freeSmallBlocks_[blockIdx];
        if (freeSmallBlocks_[blockIdx]) {
            freeSmallBlocks_[blockIdx]->prev = metadata;
        }
        freeSmallBlocks_[blockIdx] = metadata;
        return;
    }

    // Try to merge with buddy if it exists and is free
    if (const auto pair =
            smallBlockMap_.find(reinterpret_cast<void*>(buddyBlockGpuPtr));
        pair != smallBlockMap_.end()) {
        // Can be merged into a bigger one
        if (pair->second->size == metadata->size && !pair->second->isAllocated) {
            // Remove buddy from the free list
            SmallBlockMetadata* buddy = pair->second;
            if (buddy->prev) {
                buddy->prev->next = buddy->next;
            } else {
                // Buddy is at the head of the free list
                freeSmallBlocks_[blockIdx] = buddy->next;
            }
            if (buddy->next) {
                buddy->next->prev = buddy->prev;
            }

            // Remove buddy from map
            smallBlockMap_.erase(pair);

            // Use the lower address as the merged block
            SmallBlockMetadata* mergedBlock =
                basePtr < buddyBlockGpuPtr ? metadata : buddy;
            const SmallBlockMetadata* otherBlock =
                basePtr < buddyBlockGpuPtr ? buddy : metadata;

            // Update merged block
            mergedBlock->size = metadata->size << 1;
            mergedBlock->isAllocated = false;
            mergedBlock->blockIdx = blockIdx + 1;
            mergedBlock->prev = nullptr;
            mergedBlock->next = nullptr;

            // Update map
            smallBlockMap_[mergedBlock->gpuPtr] = mergedBlock;

            // Delete the other block metadata
            delete otherBlock;

            // Recursively insert merged block (may merge again with its buddy)
            insertIntoFreeBlock(mergedBlock);
            return;
        }
    }

    // Cannot merge, just insert into free list
    metadata->isAllocated = false;
    metadata->prev = nullptr;
    metadata->next = freeSmallBlocks_[blockIdx];
    if (freeSmallBlocks_[blockIdx]) {
        freeSmallBlocks_[blockIdx]->prev = metadata;
    }
    freeSmallBlocks_[blockIdx] = metadata;
}

void CudaMemoryPool::insertIntoFreeBigBlock(BigBlockMetadata* metadata) {
    if (!metadata) {
        return;
    }
    metadata->cacheLiveTimes = 0;
    metadata->isAllocated = false;
    // Note: metadata is already owned by allocatedBigBlocks_ (when allocated),
    // we just move it to cache (bigBlocks_)
    // Find and move from allocatedBigBlocks_ to bigBlocks_
    for (auto it = allocatedBigBlocks_.begin(); it != allocatedBigBlocks_.end();
         ++it) {
        if (it->get() == metadata) {
            freeBigBlocks_.push_back(std::move(*it));
            allocatedBigBlocks_.erase(it);
            return;
        }
    }

    // If not found in allocatedBigBlocks_, it might already be in bigBlocks_
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
    auto bestfitIt = freeBigBlocks_.end();

    for (auto it = freeBigBlocks_.begin(); it != freeBigBlocks_.end(); ++it) {
        auto& block = *it;
        if (!block || block->isAllocated) {
            continue;
        }

        if (block->size == size) {
            block->refreshCacheLiveTime();
            block->isAllocated = true;
            // Move from cache to allocated list
            allocatedBigBlocks_.push_back(std::move(block));
            BigBlockMetadata* result = allocatedBigBlocks_.back().get();
            // Remove from cache
            freeBigBlocks_.erase(it);
            return result;
        }

        if (block->size > size) {
            if (bestfit == nullptr || block->size < bestfit->size) {
                bestfit = block.get();
                bestfitIt = it;
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
    // Move from cache to allocated list
    allocatedBigBlocks_.push_back(std::move(*bestfitIt));
    freeBigBlocks_.erase(bestfitIt);
    return bestfit;
}

size_t CudaMemoryPool::getMemoryNeeded(const size_t size) {
    return size + sizeof(size_t); // Size + metadata overhead
}

void CudaMemoryPool::free(void* ptr) { // NOLINT
    if (!ptr) {
        return;
    }

    // Check if it's a big block
    if (const auto bigIt = bigBlockMap_.find(ptr); bigIt != bigBlockMap_.end()) {
        BigBlockMetadata* metadata = bigIt->second;
        if (!metadata) {
            return;
        }

        // Check cacheLiveTimes BEFORE resetting it in insertIntoFreeBigBlock
        const bool shouldFree =
            metadata->cacheLiveTimes > BigMemoryBlockMaxLiveTimes;

        if (metadata->isAllocated) {
            metadata->isAllocated = false;
            insertIntoFreeBigBlock(metadata);
        }

        // If this big block is not used for a long time, remove it from cache
        if (shouldFree) {
            void* gpuPtrToFree = metadata->gpuPtr;
            bigBlockMap_.erase(bigIt);
            // Find and remove from freeBigBlocks_
            for (auto it = freeBigBlocks_.begin(); it != freeBigBlocks_.end();
                 ++it) {
                if (it->get() == metadata) {
                    cudaMemoryFree(gpuPtrToFree);
                    freeBigBlocks_.erase(it);
                    break;
                }
            }
        }

        // Also check and free other old blocks in cache that haven't been used
        for (auto it = freeBigBlocks_.begin(); it != freeBigBlocks_.end();) {
            if ((*it) && (*it)->cacheLiveTimes > BigMemoryBlockMaxLiveTimes) {
                void* gpuPtrToFree = (*it)->gpuPtr;
                auto mapIt = bigBlockMap_.find(gpuPtrToFree);
                if (mapIt != bigBlockMap_.end()) {
                    bigBlockMap_.erase(mapIt);
                }
                cudaMemoryFree(gpuPtrToFree);
                it = freeBigBlocks_.erase(it);
            } else {
                ++it;
            }
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

    // Pointer not allocated by this pool: do not free (cudaFree(unknown ptr)
    // is undefined behavior). Caller must only free pointers returned by
    // allocateSmall/allocateBig.
    (void) ptr;
}

std::expected<void*, common::Error> CudaMemoryPool::allocateBig(const size_t size) {
    if (BigBlockMetadata* bigBlock = findCachedBigBlock(size); bigBlock != nullptr) {
        return bigBlock->gpuPtr;
    }

    return std::unexpected(common::CudaDeviceOutOfMemoryError());
}

size_t CudaMemoryPool::getBlockIndexOfSize(const size_t size) {
    size_t idx = 1;
    size_t currentSize = BaseMemoryBlockSize;

    while (size > currentSize
           && idx < MaxSmallObjectPoolListSize) { // Limit to prevent infinite loop
        currentSize *= 2;
        ++idx;
    }

    return idx - 1;
}

common::Error CudaMemoryPool::requireSplitBlock(const size_t blockIdx) {
    if (blockIdx >= MaxSmallObjectPoolListSize) { // Safety limit
        return common::CudaSmallObjectMemoryPoolFullError();
    }

    if (blockIdx == MaxSmallObjectPoolListSize - 1) {
        void* ptr = nullptr;
        const auto size = BaseMemoryBlockSize << (blockIdx);
        const cudaError_t res = cudaMemoryAllocate(size, &ptr);
        if (res != cudaSuccess) {
            return common::CudaDeviceOutOfMemoryError();
        }
        freeSmallBlocks_[blockIdx] = new SmallBlockMetadata();
        freeSmallBlocks_[blockIdx]->gpuPtr = ptr;
        freeSmallBlocks_[blockIdx]->size = size;
        freeSmallBlocks_[blockIdx]->blockIdx = blockIdx;
        smallBlockMap_[ptr] = freeSmallBlocks_[blockIdx];

        if (!smallBlockStorage_) {
            smallBlockStorage_ = new SmallBlockMetadata(*freeSmallBlocks_[blockIdx]);
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

    // If no block at this level, try to split from an even larger block
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

    // Remove from the free list
    freeSmallBlocks_[newBlockIdx] = memoryBlock->next;
    if (memoryBlock->next) {
        memoryBlock->next->prev = nullptr;
    }

    // Calculate sizes
    const size_t blockSize = memoryBlock->size;
    const size_t halfSize = blockSize / 2;

    // Create two new blocks
    void* firstGpuPtr = memoryBlock->gpuPtr;
    void* secondGpuPtr = static_cast<char*>(firstGpuPtr) + halfSize;

    // Create metadata for the first half
    const auto firstMeta = new SmallBlockMetadata();
    firstMeta->gpuPtr = firstGpuPtr;
    firstMeta->size = halfSize;
    firstMeta->blockIdx = blockIdx;
    firstMeta->isAllocated = false;

    // Create metadata for the second half
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

    // Release the original block metadata (GPU memory is now represented by
    // firstMeta/secondMeta; top-level block is still tracked in
    // smallBlockStorage_ for destructor)
    delete memoryBlock;

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
