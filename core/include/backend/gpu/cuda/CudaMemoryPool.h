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

#ifndef HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE
#define HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE

#include <cmath>
#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/errors/Error.h"
#include "utils/data_structure/Bitmap.h"

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <driver_types.h>
#endif
#endif

namespace hahaha::backend {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

constexpr int log2(const size_t val) {
    for (int i = 0; i < 32; ++i) {
        // ReSharper disable once CppRedundantParentheses
        if (val == (1 << i)) {
            return i;
        }
    }
    return 0;
}

/**
 * @brief CUDA memory pool with metadata stored on CPU.
 * @details All metadata (block headers, free lists, etc.) are stored in CPU
 * memory. Only the actual data buffers are allocated on GPU.
 */
class CudaMemoryPool {
  public:
    static constexpr size_t BaseMemoryBlockSize = 32;                 // 32B
    static constexpr size_t SingleSmallObjectPoolMaxSize = 512 << 20; // 512MB
    static constexpr size_t MaxSmallObjectPoolListSize =
        log2(SingleSmallObjectPoolMaxSize / BaseMemoryBlockSize) + 1;

    static constexpr size_t BigMemoryBlockMaxLiveTimes = 3;

    /**
     * @brief Constructor to initialize the memory pool.
     */
    CudaMemoryPool();

    /**
     * @brief Destructor to clean up the memory pool.
     */
    ~CudaMemoryPool();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
    CudaMemoryPool(CudaMemoryPool&&) = delete;
    CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    /**
     * @brief Check if free blocklist exists for given size.
     * @param blockIdx Block size index.
     * @return Error if invalid, Success otherwise.
     */
    [[nodiscard]] static common::Error checkFreeBlockListExist(size_t blockIdx);

    /**
     * @brief Allocate a small memory block (< threshold).
     * @param size Size in bytes.
     * @return Pointer to allocated GPU memory or error.
     */
    std::expected<void*, common::Error> allocateSmall(size_t size);

    /**
     * @brief Allocate a large memory block (>= threshold).
     * @param size Size in bytes.
     * @return Pointer to allocated GPU memory or error.
     */
    std::expected<void*, common::Error> allocateBig(size_t size);

    /**
     * @brief Free a memory block.
     * @param ptr Pointer to GPU memory to free.
     */
    void free(void* ptr);

  private:
    /**
     * @brief CPU-side metadata for a small memory block.
     */
    struct SmallBlockMetadata {
        void* gpuPtr;    // GPU memory pointer
        size_t size;     // Block size in bytes
        size_t blockIdx; // Block index in a free list
        SmallBlockMetadata* prev;
        SmallBlockMetadata* next;
        bool isAllocated; // Allocation state

        SmallBlockMetadata()
            : gpuPtr(nullptr), size(0), blockIdx(0), prev(nullptr), next(nullptr),
              isAllocated(false) {
        }
    };

    /**
     * @brief CPU-side metadata for a large memory block.
     */
    struct BigBlockMetadata {
        void* gpuPtr;          // GPU memory pointer (points to data, not header)
        size_t size;           // Block size in bytes
        size_t cacheLiveTimes; // Cache lifetime counter
        bool isAllocated;      // Allocation state

        BigBlockMetadata()
            : gpuPtr(nullptr), size(0), cacheLiveTimes(0), isAllocated(false) {
        }

        void refreshCacheLiveTime() {
            cacheLiveTimes = 0;
        }
    };

    /**
     * @brief Get block index for given size.
     * @param size Size in bytes.
     * @return Block index.
     */
    static size_t getBlockIndexOfSize(size_t size);

    /**
     * @brief Allocate memory on a specific block index.
     * @param blockIdx Block index.
     * @return Pointer to allocated GPU memory or error.
     */
    std::expected<void*, common::Error> allocateOnBlock(size_t blockIdx);

    /**
     * @brief Require splitting a block from larger size.
     * @param blockIdx Target block index.
     * @return Error if failed, Success otherwise.
     */
    common::Error requireSplitBlock(size_t blockIdx);

    /**
     * @brief Insert block into free list.
     * @param metadata Block metadata to insert.
     */
    void insertIntoFreeBlock(SmallBlockMetadata* metadata);

    /**
     * @brief Insert big block into cache.
     * @param metadata Big block metadata.
     */
    void insertIntoFreeBigBlock(BigBlockMetadata* metadata);

    /**
     * @brief Require new big block allocation.
     * @param size Size in bytes.
     * @return Big block metadata or error.
     */
    std::expected<BigBlockMetadata*, common::Error> requireNewBigBlock(size_t size);

    /**
     * @brief Find cached big block that fits size.
     * @param size Required size in bytes.
     * @return Best fit block metadata or nullptr.
     */
    BigBlockMetadata* findCachedBigBlock(size_t size);

    /**
     * @brief Calculate memory needed including overhead.
     * @param size Size in bytes.
     * @return Total memory needed.
     */
    static size_t getMemoryNeeded(size_t size);

    // Free lists for small blocks (indexed by block size)
    std::vector<SmallBlockMetadata*> freeSmallBlocks_;

    // Big block cache (CPU-side metadata)
    std::vector<std::unique_ptr<BigBlockMetadata>> freeBigBlocks_;

    // All allocated big blocks (for cleanup)
    std::vector<std::unique_ptr<BigBlockMetadata>> allocatedBigBlocks_;

    // All small block metadata (for lifetime management). it only stores the
    // copy of highest level memory block (whose size is
    // SingleSmallObjectPoolMaxSize) so that it can be found quickly.
    SmallBlockMetadata* smallBlockStorage_ = nullptr;

    // Map GPU pointer to metadata (for fast lookup)
    std::unordered_map<void*, SmallBlockMetadata*> smallBlockMap_;
    std::unordered_map<void*, BigBlockMetadata*> bigBlockMap_;
};

#else  // !__has_include(<driver_types.h>)
// Stub implementation when CUDA headers not available
class CudaMemoryPool {
  public:
    [[nodiscard]] hahaha::common::Error checkFreeBlockListExist(size_t) const {
        return hahaha::common::Error::Success();
    }

    std::expected<void*, hahaha::common::Error> allocateSmall(size_t) {
        return std::unexpected(hahaha::common::Error::Success());
    }

    std::expected<void*, hahaha::common::Error> allocateBig(size_t) {
        return std::unexpected(hahaha::common::Error::Success());
    }

    void free(void*) {
    }
};
#endif // __has_include(<driver_types.h>)

#else  // !HAHAHA_USE_CUDA
// Stub implementation when CUDA not enabled
class CudaMemoryPool {
  public:
    [[nodiscard]] hahaha::common::Error checkFreeBlockListExist(size_t) const {
        return hahaha::common::Error::Success();
    }

    std::expected<void*, hahaha::common::Error> allocateSmall(size_t) {
        return std::unexpected(hahaha::common::Error::Success());
    }

    std::expected<void*, hahaha::common::Error> allocateBig(size_t) {
        return std::unexpected(hahaha::common::Error::Success());
    }

    void free(void*) {
    }
};
#endif // HAHAHA_USE_CUDA
} // namespace hahaha::backend

#endif // HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE
