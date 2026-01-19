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
CudaMemoryPool::allocateSmall(const size_t size) {
    const size_t blockIdx = getBlockIndexOfSize(size);
    if (const auto res = checkFreeBlockListExist(blockIdx); res.isSuccess()) {
        return allocateOnBlock(blockIdx);
    }
    throw std::runtime_error("Cannot allocate memory");
}

std::expected<void*, common::Error>
CudaMemoryPool::allocateOnBlock(const size_t blockIdx) {
    // allocate one block on the target block list
    auto block = freeBlocks_[blockIdx];
    if (!block) { // require to split a buddy block from the bigger block
        if (auto resError = requireSplitBlock(blockIdx);
            !resError.isSuccess()) {
            if (resError.code
                == common::ErrorCode::CudaSmallObjectMemoryPoolFull) {
                return std::unexpected(resError);
            }
            throw std::runtime_error("Cannot allocate memory");
        }
    }

    block = freeBlocks_[blockIdx];
    freeBlocks_[blockIdx] = block->next();
    block->setNext(nullptr);
    block->setPrev(nullptr);

    freeBlocks_[blockIdx]->setPrev(nullptr);
    return reinterpret_cast<char*>(block) + sizeof(CudaMemoryBlockHeader);
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

void CudaMemoryPool::insertIntoBigBlock(BigCudaMemoryBlock* block) {
    block->cacheLiveTimes = 0;
    bigBlocks_.push_back(block);
}

std::expected<CudaMemoryPool::BigCudaMemoryBlock*, common::Error>
CudaMemoryPool::requireNewBigBlock(const size_t size) {
    void* ptr = nullptr;
    if (const cudaError_t res =
            cudaMemoryAllocate(size + sizeof(BigCudaMemoryBlock), &ptr);
        res != cudaSuccess) {
        return std::unexpected(common::CudaDeviceOutOfMemoryError());
    }

    allocatedBigBlocks_.push_back(static_cast<BigCudaMemoryBlock*>(ptr));
    return static_cast<BigCudaMemoryBlock*>(ptr);
}

CudaMemoryPool::BigCudaMemoryBlock*
CudaMemoryPool::findCachedBigBlock(const size_t size) {
    BigCudaMemoryBlock* bestfit = nullptr;
    for (const auto& block : bigBlocks_) {
        if (block->getSize() == size) {
            block->refreshCacheLiveTime();
            return block;
        }
        if (block->getSize() > size) {
            if (bestfit == nullptr || block->getSize() < bestfit->getSize()) {
                bestfit = block;
            }
            block->cacheLiveTimes++;
        }
    }

    // if not found, try to create a new one
    if (bestfit == nullptr) {
        const auto newBlock = requireNewBigBlock(size);
        if (!newBlock.has_value()) {
            // maybe release some is better
            throw std::runtime_error("Cannot allocate memory");
        }
        return newBlock.value();
    }

    return bestfit;
}

size_t CudaMemoryPool::getMemoryNeeded(size_t size) {
    return size + (sizeof(size_t) + sizeof(std::unique_ptr<CudaMemoryBlock>));
}

void CudaMemoryPool::free(void* ptr) {
}

std::expected<void*, common::Error> CudaMemoryPool::allocateBig(size_t size) {
    if (BigCudaMemoryBlock* bigBlock = findCachedBigBlock(size);
        bigBlock != nullptr) {
        bigBlock->refreshCacheLiveTime();
        return reinterpret_cast<char*>(bigBlock) + sizeof(BigCudaMemoryBlock);
    }

    return std::unexpected(common::CudaDeviceOutOfMemoryError());
}

size_t CudaMemoryPool::getBlockIndexOfSize(size_t size) {
    size_t idx = 0;
    while (size > 32) {
        size /= 2;
        ++idx;
    }
    return idx;
}

common::Error CudaMemoryPool::requireSplitBlock(const size_t blockIdx) {
    if (blockIdx >= freeBlocks_.size()) {
        throw std::runtime_error("Invalid block index");
    }

    const auto newBlockIdx = blockIdx + 1;
    if (newBlockIdx >= freeBlocks_.size()) {
        throw std::runtime_error("Invalid block index");
    }
    auto memoryBlock = freeBlocks_[newBlockIdx];
    if (memoryBlock == nullptr) {
        if (const auto resError = requireSplitBlock(newBlockIdx);
            !resError.isSuccess()) {
            return resError;
        }
    }
    memoryBlock = freeBlocks_[newBlockIdx];
    freeBlocks_[newBlockIdx] = memoryBlock->next();
    freeBlocks_[newBlockIdx]->setPrev(nullptr);

    const auto newSize =
        (BaseMemoryBlockSize << blockIdx) - sizeof(CudaMemoryBlockHeader);
    memoryBlock->setSize(newSize);
    memoryBlock->setPrev(nullptr);
    memoryBlock->setNext(reinterpret_cast<CudaMemoryBlock*>(
        reinterpret_cast<char*>(memoryBlock) + newSize));
    memoryBlock->next()->setPrev(memoryBlock);
    memoryBlock->next()->setSize(newSize);
    memoryBlock->next()->setNext(nullptr);

    freeBlocks_[newBlockIdx] = memoryBlock;

    return common::Error::Success();
}

common::Error CudaMemoryPool::checkFreeBlockListExist(const size_t size) const {
    if (size >= freeBlocks_.size()) {
        return common::InvalidArgumentError();
    }
    return {};
}

} // namespace hahaha::backend
#endif
#endif
