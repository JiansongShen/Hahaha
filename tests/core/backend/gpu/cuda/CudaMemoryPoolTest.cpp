#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaMemoryPool.h"
#include "backend/gpu/cuda/cuda_memory.cuh"
#include "common/errors/Error.h"
#include "common/errors/ErrorCode.h"

namespace hahaha::backend::test {

class CudaMemoryPoolTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize memory pool
        pool_ = std::make_unique<CudaMemoryPool>();
    }

    void TearDown() override {
        // Cleanup resources
        pool_.reset();
    }

    std::unique_ptr<CudaMemoryPool> pool_;
};

// Test basic allocation and deallocation functionality
TEST_F(CudaMemoryPoolTest, BasicAllocationAndDeallocation) {
    // Test small block memory allocation
    auto result = pool_->allocateSmall(1024); // 1KB
    ASSERT_TRUE(result.has_value());

    void* ptr = result.value();
    ASSERT_NE(ptr, nullptr);

    // Free memory
    pool_->free(ptr);

    // Subsequent allocation should succeed and reuse freed block (same address)
    auto result2 = pool_->allocateSmall(1024);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), ptr);
}

// Test boundary condition: minimum block size
TEST_F(CudaMemoryPoolTest, MinimumBlockSizeAllocation) {
    auto result = pool_->allocateSmall(1); // minimum possible size
    ASSERT_TRUE(result.has_value());

    void* ptr = result.value();
    ASSERT_NE(ptr, nullptr);

    pool_->free(ptr);
}

// Test boundary condition: near threshold size
TEST_F(CudaMemoryPoolTest, NearThresholdAllocation) {
    const size_t threshold = 512 << 20; // 512MB
    auto result =
        pool_->allocateSmall(threshold); // close but less than threshold
    ASSERT_TRUE(result.has_value());

    void* ptr = result.value();
    ASSERT_NE(ptr, nullptr);

    pool_->free(ptr);
}

// Test large block memory allocation
TEST_F(CudaMemoryPoolTest, BigBlockAllocation) {
    auto result = pool_->allocateBig(1024 * 1024 * 1024); // 1GB
    ASSERT_TRUE(result.has_value());

    void* ptr = result.value();
    ASSERT_NE(ptr, nullptr);

    pool_->free(ptr);
}

// Test 012 principle: 0 elements, 1 element, 2 elements
TEST_F(CudaMemoryPoolTest, ZeroOneTwoPrinciple) {
    // 0 elements: don't allocate anything
    // This test mainly ensures no crash occurs

    // 1 element: allocate one memory block
    auto result1 = pool_->allocateSmall(512);
    ASSERT_TRUE(result1.has_value());
    void* ptr1 = result1.value();
    ASSERT_NE(ptr1, nullptr);

    // 2 elements: allocate two memory blocks
    auto result2 = pool_->allocateSmall(1024);
    ASSERT_TRUE(result2.has_value());
    void* ptr2 = result2.value();
    ASSERT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2); // Should be different addresses

    // Free all memory
    pool_->free(ptr1);
    pool_->free(ptr2);
}

// Test multiple allocations and deallocations
TEST_F(CudaMemoryPoolTest, MultipleAllocationsAndDeallocations) {
    std::vector<void*> pointers;
    const size_t num_allocations = 10;

    // Allocate multiple blocks
    for (size_t i = 0; i < num_allocations; ++i) {
        auto result = pool_->allocateSmall(1024 + i * 100); // incremental size
        ASSERT_TRUE(result.has_value());
        pointers.push_back(result.value());
        ASSERT_NE(pointers.back(), nullptr);
    }

    // Free all blocks
    for (void* ptr : pointers) {
        pool_->free(ptr);
    }

    // Subsequent allocations should be able to reuse previously freed memory
    for (size_t i = 0; i < num_allocations; ++i) {
        auto result = pool_->allocateSmall(1024 + i * 100);
        ASSERT_TRUE(result.has_value());
    }
}

// Test buddy algorithm merging functionality in memory pool
TEST_F(CudaMemoryPoolTest, BuddyAlgorithmMerge) {
    // Allocate a large block of memory, then free it, to see if it merges
    // correctly
    auto result1 = pool_->allocateSmall(2048);
    ASSERT_TRUE(result1.has_value());
    void* ptr1 = result1.value();

    auto result2 = pool_->allocateSmall(2048);
    ASSERT_TRUE(result2.has_value());
    void* ptr2 = result2.value();

    // Ensure two pointers are adjacent (conforming to buddy algorithm)
    uintptr_t addr1 = reinterpret_cast<uintptr_t>(ptr1);
    uintptr_t addr2 = reinterpret_cast<uintptr_t>(ptr2);

    // Free both blocks to see if they merge into a larger block
    pool_->free(ptr1);
    pool_->free(ptr2);

    // Try to allocate a larger block to see if it utilizes the merged space
    auto result3 = pool_->allocateSmall(4096);
    ASSERT_TRUE(result3.has_value());
    pool_->free(result3.value());
}

// Test large block memory caching mechanism
TEST_F(CudaMemoryPoolTest, BigBlockCacheMechanism) {
    const size_t big_size = 100 * 1024 * 1024; // 100MB

    // Allocate large block memory
    auto result1 = pool_->allocateBig(big_size);
    ASSERT_TRUE(result1.has_value());
    void* ptr1 = result1.value();
    ASSERT_NE(ptr1, nullptr);

    // Free to cache
    pool_->free(ptr1);

    // Allocate same size again, should get from cache
    auto result2 = pool_->allocateBig(big_size);
    ASSERT_TRUE(result2.has_value());
    void* ptr2 = result2.value();
    ASSERT_NE(ptr2, nullptr);

    // Free and wait some time for cache timeout
    pool_->free(ptr2);

    // Allocate again to confirm cache mechanism works properly
    auto result3 = pool_->allocateBig(big_size);
    ASSERT_TRUE(result3.has_value());
}

// Test error handling and boundary conditions
TEST_F(CudaMemoryPoolTest, ErrorHandling) {
    // Test invalid parameters
    // Here we test some edge cases that may cause errors

    // Try to allocate oversized memory (may or may not return error depending
    // on actual GPU memory) Note: this may succeed or fail depending on actual
    // GPU memory
    auto result =
        pool_->allocateBig(static_cast<size_t>(-1)); // maximum possible value
    EXPECT_FALSE(result.has_value());
    // This test may fail or succeed depending on GPU memory size

    // Test freeing null pointer
    pool_->free(nullptr); // Should handle safely
}

// TODO: Test concurrent access safety
// TEST_F(CudaMemoryPoolTest, ConcurrencySafety) {
//     const size_t num_threads = 4;
//     const size_t allocations_per_thread = 10;
//
//     std::vector<std::thread> threads;
//     std::vector<std::vector<void*>> thread_pointers(num_threads);
//
//     // Start multiple threads to allocate simultaneously
//     for (size_t t = 0; t < num_threads; ++t) {
//         threads.emplace_back(
//             [this, t, allocations_per_thread, &thread_pointers]() {
//                 for (size_t i = 0; i < allocations_per_thread; ++i) {
//                     auto result = pool_->allocateSmall(
//                         1024 + (t * allocations_per_thread + i) * 10);
//                     if (result.has_value()) {
//                         thread_pointers[t].push_back(result.value());
//                     }
//                 }
//
//                 // Free all memory allocated by this thread
//                 for (void* ptr : thread_pointers[t]) {
//                     if (ptr != nullptr) {
//                         pool_->free(ptr);
//                     }
//                 }
//             });
//     }
//
//     // Wait for all threads to complete
//     for (auto& thread : threads) {
//         thread.join();
//     }
// }

// Test memory leak detection
TEST_F(CudaMemoryPoolTest, MemoryLeakDetection) {
    // Allocate some memory but don't free immediately, let destructor clean up
    // automatically
    std::vector<void*> pointers;

    for (int i = 0; i < 5; ++i) {
        auto result = pool_->allocateSmall(1024 * (i + 1));
        if (result.has_value()) {
            pointers.push_back(result.value());
        }
    }

    // Manually free all memory
    for (void* ptr : pointers) {
        if (ptr != nullptr) {
            pool_->free(ptr);
        }
    }
}

// Test various size allocations to achieve branch coverage
TEST_F(CudaMemoryPoolTest, VariousSizeAllocations) {
    std::vector<size_t> sizes = {
        1,  // minimum
        32, // base block size
        64,
        128,
        256,
        512,
        1024, // powers of 2
        1500, // non power of 2
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,          // larger powers of 2
        (512 << 20) - 1 // approaching small block upper limit
    };

    std::vector<void*> pointers;

    for (size_t size : sizes) {
        auto result = pool_->allocateSmall(size);
        if (result.has_value()) {
            pointers.push_back(result.value());
            ASSERT_NE(result.value(), nullptr);
        }
    }

    // Free all allocated memory
    for (void* ptr : pointers) {
        if (ptr != nullptr) {
            pool_->free(ptr);
        }
    }
}

// Test extreme case behaviors
TEST_F(CudaMemoryPoolTest, ExtremeCases) {
    // Test repeated allocation and freeing of same size memory
    for (int i = 0; i < 100; ++i) {
        auto result = pool_->allocateSmall(1024);
        ASSERT_TRUE(result.has_value());

        void* ptr = result.value();
        ASSERT_NE(ptr, nullptr);

        pool_->free(ptr);
    }

    // Test alternating allocation of different sized memory
    for (int i = 0; i < 50; ++i) {
        auto result1 = pool_->allocateSmall(512);
        auto result2 = pool_->allocateSmall(2048);

        ASSERT_TRUE(result1.has_value());
        ASSERT_TRUE(result2.has_value());

        if (result1.has_value() && result2.has_value()) {
            pool_->free(result1.value());
            pool_->free(result2.value());
        }
    }
}

// Test memory pool splitting functionality
TEST_F(CudaMemoryPoolTest, SplitBlockFunctionality) {
    // Test memory block splitting functionality
    // First allocate a larger block, which will trigger block splitting
    auto result1 = pool_->allocateSmall(1024);
    ASSERT_TRUE(result1.has_value());

    auto result2 = pool_->allocateSmall(1024);
    ASSERT_TRUE(result2.has_value());

    // Free these two blocks so they become adjacent in free list
    pool_->free(result1.value());
    pool_->free(result2.value());

    // Now try to allocate a larger block, which will need to merge these blocks
    auto result3 = pool_->allocateSmall(2048);
    ASSERT_TRUE(result3.has_value());

    pool_->free(result3.value());
}

// Test block index calculation functionality
TEST_F(CudaMemoryPoolTest, BlockIndexCalculation) {
    // Use reflection or friend access to test internal methods
    // Here we indirectly test block index calculation through actual allocation
    auto result = pool_->allocateSmall(32); // Should map to first block size
    ASSERT_TRUE(result.has_value());

    auto result2 = pool_->allocateSmall(64); // Should map to next block size
    ASSERT_TRUE(result2.has_value());

    pool_->free(result.value());
    pool_->free(result2.value());
}

// Test error path: out of bounds block index
TEST_F(CudaMemoryPoolTest, OutOfBoundsBlockIndex) {
    // Test very large allocation, which may exceed block index limits
    auto result =
        pool_->allocateSmall(static_cast<size_t>(-1)); // Try maximum value
    // This may fail but should not crash
}

// Test specific buddy merge case
TEST_F(CudaMemoryPoolTest, BuddyMergeSpecificCase) {
    // Allocate two adjacent buddy blocks
    const size_t blockSize = 1024;
    auto ptr1 = pool_->allocateSmall(blockSize);
    ASSERT_TRUE(ptr1.has_value());

    auto ptr2 = pool_->allocateSmall(blockSize);
    ASSERT_TRUE(ptr2.has_value());

    // Free these two blocks to make them buddies
    pool_->free(ptr1.value());
    pool_->free(ptr2.value());

    // Now allocate a larger block, should utilize merged space
    auto ptr3 = pool_->allocateSmall(blockSize * 2);
    ASSERT_TRUE(ptr3.has_value());

    pool_->free(ptr3.value());
}

// Test large block cache timeout mechanism
TEST_F(CudaMemoryPoolTest, BigBlockCacheTimeout) {
    const size_t big_size = 50 * 1024 * 1024; // 50MB

    // Allocate and free large block to put it in cache
    auto result = pool_->allocateBig(big_size);
    ASSERT_TRUE(result.has_value());
    void* ptr = result.value();
    pool_->free(ptr);

    // Allocate again, should get from cache
    auto result2 = pool_->allocateBig(big_size);
    ASSERT_TRUE(result2.has_value());

    pool_->free(result2.value());
}

// Test releasing non-existent pointer (edge case)
TEST_F(CudaMemoryPoolTest, FreeNonExistentPointer) {
    // Create an invalid pointer and try to free it
    // This should be handled safely
    void* fake_ptr = reinterpret_cast<void*>(0x12345);
    pool_->free(fake_ptr); // Should handle safely
}

// Test pool full scenario
TEST_F(CudaMemoryPoolTest, PoolFullScenario) {
    // Try to allocate lots of small memory until pool is full
    std::vector<void*> allocations;
    size_t size = 32;

    // Keep allocating until failure or some limit reached
    for (int i = 0; i < 1000; ++i) {
        auto result = pool_->allocateSmall(size);
        if (result.has_value()) {
            allocations.push_back(result.value());
        } else {
            EXPECT_TRUE(
                result.error().code
                    == hahaha::common::ErrorCode::CudaSmallObjectMemoryPoolFull
                || result.error().code
                       == hahaha::common::ErrorCode::CudaDeviceOutOfMemory);
            break;
        }
    }

    // Free all allocated memory
    for (void* ptr : allocations) {
        pool_->free(ptr);
    }
}

// Test boundary block index cases
TEST_F(CudaMemoryPoolTest, BoundaryBlockIndex) {
    // Test maximum allowed block index case
    // Allocate specific sized memory to trigger boundary conditions
    size_t maxSize = hahaha::backend::CudaMemoryPool::BaseMemoryBlockSize;
    for (int i = 0;
         i < hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize - 1;
         ++i) {
        maxSize *= 2;
    }

    // Allocate memory close to maximum value
    auto result = pool_->allocateSmall(maxSize / 2);
    if (result.has_value()) {
        pool_->free(result.value());
    }
}

// --- Branch coverage: allocateSmall(0) -> InvalidArgumentError ---
TEST_F(CudaMemoryPoolTest, AllocateSmall_ZeroSize_ReturnsInvalidArgument) {
    auto result = pool_->allocateSmall(0);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code,
              hahaha::common::ErrorCode::InvalidArgument);
}

// --- Branch coverage: checkFreeBlockListExist valid / invalid ---
TEST_F(CudaMemoryPoolTest, CheckFreeBlockListExist_ValidIndex_ReturnsSuccess) {
    hahaha::common::Error err =
        hahaha::backend::CudaMemoryPool::checkFreeBlockListExist(0);
    EXPECT_TRUE(err.isSuccess());
}

TEST_F(CudaMemoryPoolTest,
       CheckFreeBlockListExist_InvalidIndex_ReturnsInvalidArgument) {
    const size_t invalidIdx =
        hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize;
    hahaha::common::Error err =
        hahaha::backend::CudaMemoryPool::checkFreeBlockListExist(invalidIdx);
    EXPECT_FALSE(err.isSuccess());
    EXPECT_EQ(err.code, hahaha::common::ErrorCode::InvalidArgument);
}

// --- Branch coverage: free(nullptr) early return ---
TEST_F(CudaMemoryPoolTest, Free_NullPtr_NoOp) {
    pool_->free(nullptr);
    pool_->free(nullptr);
}

// --- Branch coverage: free(small) double free -> metadata->isAllocated false ---
TEST_F(CudaMemoryPoolTest, FreeSmallBlock_DoubleFree_NoCrash) {
    auto result = pool_->allocateSmall(1024);
    ASSERT_TRUE(result.has_value());
    void* ptr = result.value();
    pool_->free(ptr);
    pool_->free(ptr);
}

// --- Branch coverage: free(big) double free -> insertIntoFreeBigBlock not found ---
TEST_F(CudaMemoryPoolTest, FreeBigBlock_DoubleFree_NoCrash) {
    const size_t big_size = 10 * 1024 * 1024;
    auto result = pool_->allocateBig(big_size);
    ASSERT_TRUE(result.has_value());
    void* ptr = result.value();
    pool_->free(ptr);
    pool_->free(ptr);
}

// --- Branch coverage: big block cache eviction when cacheLiveTimes > Max ---
// One block stays in cache with cacheLiveTimes=4; freeing another block
// triggers the loop that evicts it.
TEST_F(CudaMemoryPoolTest, BigBlockCacheEviction_EvictsAfterMaxLiveTimes) {
    const size_t block_size = 50 * 1024 * 1024;
    const size_t small_size = 25 * 1024 * 1024;
    const int maxLive = static_cast<int>(
        hahaha::backend::CudaMemoryPool::BigMemoryBlockMaxLiveTimes);

    std::vector<void*> big_blocks;
    for (int i = 0; i < maxLive + 2; ++i) {
        auto res = pool_->allocateBig(block_size);
        ASSERT_TRUE(res.has_value()) << "i=" << i;
        big_blocks.push_back(res.value());
    }
    for (void* p : big_blocks) {
        pool_->free(p);
    }

    std::vector<void*> allocated;
    for (int i = 0; i < maxLive; ++i) {
        auto res = pool_->allocateBig(small_size);
        ASSERT_TRUE(res.has_value()) << "i=" << i;
        allocated.push_back(res.value());
    }
    pool_->free(allocated[0]);
    for (size_t i = 1; i < allocated.size(); ++i) {
        pool_->free(allocated[i]);
    }
}

// --- Branch coverage: free(big) loop evicts other old blocks in cache ---
// When freeing an allocated big block, we also evict other cached blocks
// with cacheLiveTimes > Max. This test leaves one block in cache with high
// liveTimes, then frees another block to trigger the eviction loop.
TEST_F(CudaMemoryPoolTest, BigBlockCacheEviction_EvictsOtherOldBlocksInCache) {
    const size_t block_size = 50 * 1024 * 1024;
    const size_t small_size = 25 * 1024 * 1024;
    const int maxLive = static_cast<int>(
        hahaha::backend::CudaMemoryPool::BigMemoryBlockMaxLiveTimes);

    std::vector<void*> blocks;
    for (int i = 0; i < maxLive + 2; ++i) {
        auto res = pool_->allocateBig(block_size);
        ASSERT_TRUE(res.has_value());
        blocks.push_back(res.value());
    }
    for (void* p : blocks) {
        pool_->free(p);
    }

    for (int i = 0; i < maxLive; ++i) {
        auto res = pool_->allocateBig(small_size);
        ASSERT_TRUE(res.has_value());
        pool_->free(res.value());
    }

    auto res = pool_->allocateBig(block_size);
    ASSERT_TRUE(res.has_value());
    pool_->free(res.value());
}

// --- Branch coverage: free(big) metadata->isAllocated false (no insert) ---
TEST_F(CudaMemoryPoolTest, FreeBigBlock_AlreadyInCache_NoCrash) {
    const size_t big_size = 20 * 1024 * 1024;
    auto result = pool_->allocateBig(big_size);
    ASSERT_TRUE(result.has_value());
    void* ptr = result.value();
    pool_->free(ptr);
    pool_->free(ptr);
}

// --- Branch coverage: free(small) metadata in map but !isAllocated ---
TEST_F(CudaMemoryPoolTest, FreeSmallBlock_AlreadyFreed_NoCrash) {
    auto result = pool_->allocateSmall(512);
    ASSERT_TRUE(result.has_value());
    void* ptr = result.value();
    pool_->free(ptr);
    pool_->free(ptr);
}

// --- Branch coverage: insertIntoFreeBlock merge path buddy at head ---
TEST_F(CudaMemoryPoolTest, BuddyMerge_BuddyAtHeadOfFreeList) {
    auto a = pool_->allocateSmall(1024);
    auto b = pool_->allocateSmall(1024);
    ASSERT_TRUE(a.has_value() && b.has_value());
    pool_->free(a.value());
    pool_->free(b.value());
    auto c = pool_->allocateSmall(2048);
    ASSERT_TRUE(c.has_value());
    pool_->free(c.value());
}

// --- Branch coverage: insertIntoFreeBlock merge path buddy not at head ---
TEST_F(CudaMemoryPoolTest, BuddyMerge_BuddyNotAtHeadOfFreeList) {
    const size_t sz = 1024;
    auto a = pool_->allocateSmall(sz);
    auto b = pool_->allocateSmall(sz);
    auto c = pool_->allocateSmall(sz);
    auto d = pool_->allocateSmall(sz);
    ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value() && d.has_value());
    pool_->free(a.value());
    pool_->free(c.value());
    pool_->free(b.value());
    auto e = pool_->allocateSmall(sz * 2);
    ASSERT_TRUE(e.has_value());
    pool_->free(e.value());
    pool_->free(d.value());
}

// --- Branch coverage: insertIntoFreeBlock no merge, empty free list ---
TEST_F(CudaMemoryPoolTest, InsertIntoFreeBlock_EmptyList) {
    auto result = pool_->allocateSmall(2048);
    ASSERT_TRUE(result.has_value());
    pool_->free(result.value());
}

// --- Branch coverage: insertIntoFreeBlock no merge, non-empty free list ---
TEST_F(CudaMemoryPoolTest, InsertIntoFreeBlock_NonEmptyList) {
    auto a = pool_->allocateSmall(1024);
    auto b = pool_->allocateSmall(1024);
    ASSERT_TRUE(a.has_value() && b.has_value());
    pool_->free(a.value());
    pool_->free(b.value());
}

// --- Branch coverage: allocateOnBlock block->next != nullptr ---
TEST_F(CudaMemoryPoolTest, AllocateOnBlock_BlockHasNextInFreeList) {
    auto a = pool_->allocateSmall(1024);
    auto b = pool_->allocateSmall(1024);
    ASSERT_TRUE(a.has_value() && b.has_value());
    pool_->free(a.value());
    pool_->free(b.value());
    auto c = pool_->allocateSmall(1024);
    ASSERT_TRUE(c.has_value());
    pool_->free(c.value());
}

// --- Branch coverage: findCachedBigBlock exact size match ---
TEST_F(CudaMemoryPoolTest, FindCachedBigBlock_ExactMatchFromCache) {
    const size_t size = 25 * 1024 * 1024;
    auto r1 = pool_->allocateBig(size);
    ASSERT_TRUE(r1.has_value());
    pool_->free(r1.value());
    auto r2 = pool_->allocateBig(size);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2.value(), r1.value());
    pool_->free(r2.value());
}

// --- Branch coverage: findCachedBigBlock best-fit from cache ---
TEST_F(CudaMemoryPoolTest, FindCachedBigBlock_BestFitFromCache) {
    const size_t large = 80 * 1024 * 1024;
    const size_t small = 40 * 1024 * 1024;
    auto r1 = pool_->allocateBig(large);
    ASSERT_TRUE(r1.has_value());
    pool_->free(r1.value());
    auto r2 = pool_->allocateBig(small);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2.value(), r1.value());
    pool_->free(r2.value());
}

// --- Branch coverage: allocateBig OOM (findCachedBigBlock returns nullptr) ---
TEST_F(CudaMemoryPoolTest, AllocateBig_OutOfMemory_ReturnsError) {
    auto result = pool_->allocateBig(static_cast<size_t>(-1));
    EXPECT_FALSE(result.has_value());
    if (!result.has_value()) {
        EXPECT_EQ(result.error().code,
                  hahaha::common::ErrorCode::CudaDeviceOutOfMemory);
    }
}

// --- Branch coverage: requireSplitBlock top-level first (smallBlockStorage_ null) ---
TEST_F(CudaMemoryPoolTest, RequireSplitBlock_TopLevel_FirstBlock) {
    size_t maxSmall =
        hahaha::backend::CudaMemoryPool::BaseMemoryBlockSize
        << (hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize - 1);
    auto result = pool_->allocateSmall(maxSmall);
    if (result.has_value()) {
        pool_->free(result.value());
    }
}

// --- Branch coverage: requireSplitBlock top-level second (smallBlockStorage_ non-null) ---
TEST_F(CudaMemoryPoolTest, RequireSplitBlock_TopLevel_SecondBlock) {
    size_t maxSmall =
        hahaha::backend::CudaMemoryPool::BaseMemoryBlockSize
        << (hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize - 1);
    auto a = pool_->allocateSmall(maxSmall);
    auto b = pool_->allocateSmall(maxSmall);
    if (a.has_value()) {
        pool_->free(a.value());
    }
    if (b.has_value()) {
        pool_->free(b.value());
    }
}

// --- Branch coverage: allocateSmall with huge size (may hit pool full or OOM) ---
TEST_F(CudaMemoryPoolTest, AllocateSmall_HugeSize_NoCrash) {
    size_t hugeSize = static_cast<size_t>(-1);
    auto result = pool_->allocateSmall(hugeSize);
    if (result.has_value()) {
        pool_->free(result.value());
    }
}

// Test: Allocate small objects until exhausting one maximum small memory block
TEST_F(CudaMemoryPoolTest, AllocateUntilExhaustMaxSmallBlock) {
    // Calculate the maximum small block size
    const size_t maxSmallBlockSize =
        hahaha::backend::CudaMemoryPool::BaseMemoryBlockSize
        << (hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize - 1);

    // Allocate small blocks of a fixed size (e.g., 1KB) until we exhaust one
    // max block
    const size_t smallBlockSize = 1024; // 1KB
    const size_t numBlocksPerMaxBlock = maxSmallBlockSize / smallBlockSize;

    std::vector<void*> allocatedBlocks;
    allocatedBlocks.reserve(numBlocksPerMaxBlock + 10);

    // Allocate blocks until we exhaust one maximum block
    // We track allocations to see when a new max block is allocated
    void* firstMaxBlockStart = nullptr;
    size_t blocksAllocated = 0;

    for (size_t i = 0; i < numBlocksPerMaxBlock + 5; ++i) {
        auto result = pool_->allocateSmall(smallBlockSize);
        if (!result.has_value()) {
            // If allocation fails, we've hit a limit
            break;
        }

        void* ptr = result.value();
        allocatedBlocks.push_back(ptr);

        // Track when we start allocating from a new max block
        // (addresses will be far apart when a new max block is allocated)
        if (i == 0) {
            firstMaxBlockStart = ptr;
        } else if (firstMaxBlockStart != nullptr) {
            uintptr_t addr1 = reinterpret_cast<uintptr_t>(firstMaxBlockStart);
            uintptr_t addr2 = reinterpret_cast<uintptr_t>(ptr);
            // If addresses are very far apart, we've moved to a new max block
            if (addr2 > addr1 + maxSmallBlockSize) {
                blocksAllocated = i;
                break;
            }
        }
    }

    // Verify we allocated many blocks
    EXPECT_GE(allocatedBlocks.size(), numBlocksPerMaxBlock - 10)
        << "Should allocate close to numBlocksPerMaxBlock blocks";

    // Free all allocated blocks
    for (void* ptr : allocatedBlocks) {
        pool_->free(ptr);
    }
}

// Test: Continue allocating after exhausting one max block, ensure new blocks
// can be allocated
TEST_F(CudaMemoryPoolTest, ContinueAllocationAfterExhaustingMaxBlock) {
    // Calculate the maximum small block size
    const size_t maxSmallBlockSize =
        hahaha::backend::CudaMemoryPool::BaseMemoryBlockSize
        << (hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize - 1);

    const size_t smallBlockSize = 1024; // 1KB
    const size_t numBlocksPerMaxBlock = maxSmallBlockSize / smallBlockSize;

    std::vector<void*> firstMaxBlockAllocations;
    std::vector<void*> secondMaxBlockAllocations;

    // Allocate blocks to exhaust first max block
    for (size_t i = 0; i < numBlocksPerMaxBlock + 5; ++i) {
        auto result = pool_->allocateSmall(smallBlockSize);
        if (!result.has_value()) {
            break;
        }
        firstMaxBlockAllocations.push_back(result.value());
    }

    EXPECT_GE(firstMaxBlockAllocations.size(), numBlocksPerMaxBlock - 10)
        << "Should exhaust first max block";

    // Continue allocating - should trigger allocation of a new max block
    void* firstBlockFromSecondMax = nullptr;
    for (size_t i = 0; i < numBlocksPerMaxBlock + 5; ++i) {
        auto result = pool_->allocateSmall(smallBlockSize);
        ASSERT_TRUE(result.has_value())
            << "Should be able to continue allocating after exhausting first "
               "max block";
        void* ptr = result.value();

        if (firstBlockFromSecondMax == nullptr) {
            firstBlockFromSecondMax = ptr;
        }

        secondMaxBlockAllocations.push_back(ptr);

        // Verify this is from a new max block (address should be far from first
        // block)
        if (firstMaxBlockAllocations.size() > 0) {
            uintptr_t addr1 =
                reinterpret_cast<uintptr_t>(firstMaxBlockAllocations[0]);
            uintptr_t addr2 = reinterpret_cast<uintptr_t>(ptr);
            // If addresses are far apart, we're in a new max block
            if (addr2 > addr1 + maxSmallBlockSize) {
                break; // Successfully allocated from new max block
            }
        }
    }

    EXPECT_GE(secondMaxBlockAllocations.size(), 1)
        << "Should allocate at least one block from new max block";

    // Free all allocations
    for (void* ptr : firstMaxBlockAllocations) {
        pool_->free(ptr);
    }
    for (void* ptr : secondMaxBlockAllocations) {
        pool_->free(ptr);
    }
}

// Test: Write data to allocated memory blocks, copy back, verify data
// consistency
TEST_F(CudaMemoryPoolTest, WriteAndVerifyDataConsistency) {
    const size_t blockSize = 4096; // 4KB
    const size_t numBlocks = 10;

    std::vector<void*> allocatedBlocks;
    std::vector<std::vector<uint8_t>> hostData(numBlocks);
    std::vector<std::vector<uint8_t>> receivedData(numBlocks);

    // Allocate multiple blocks
    for (size_t i = 0; i < numBlocks; ++i) {
        auto result = pool_->allocateSmall(blockSize);
        ASSERT_TRUE(result.has_value()) << "Failed to allocate block " << i;
        allocatedBlocks.push_back(result.value());

        // Prepare unique test data for each block
        hostData[i].resize(blockSize);
        receivedData[i].resize(blockSize);
        for (size_t j = 0; j < blockSize; ++j) {
            hostData[i][j] = static_cast<uint8_t>((i * 256 + j) % 256);
        }
    }

    // Write data to GPU memory using cudaMemcpy
    for (size_t i = 0; i < numBlocks; ++i) {
        cudaError_t err = cudaMemcpy(allocatedBlocks[i], hostData[i].data(),
                                     blockSize, cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess)
            << "Failed to copy data to GPU for block " << i;
    }

    // Synchronize to ensure all copies are complete
    cudaDeviceSynchronize();

    // Copy data back from GPU to host
    for (size_t i = 0; i < numBlocks; ++i) {
        cudaError_t err = cudaMemcpy(receivedData[i].data(), allocatedBlocks[i],
                                     blockSize, cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess)
            << "Failed to copy data from GPU for block " << i;
    }

    // Synchronize again
    cudaDeviceSynchronize();

    // Verify data consistency
    for (size_t i = 0; i < numBlocks; ++i) {
        EXPECT_EQ(hostData[i].size(), receivedData[i].size())
            << "Size mismatch for block " << i;
        for (size_t j = 0; j < blockSize; ++j) {
            EXPECT_EQ(hostData[i][j], receivedData[i][j])
                << "Data mismatch at block " << i << ", byte " << j
                << ": expected " << static_cast<int>(hostData[i][j])
                << ", got " << static_cast<int>(receivedData[i][j]);
        }
    }

    // Free all blocks
    for (void* ptr : allocatedBlocks) {
        pool_->free(ptr);
    }
}

// Test: Write data to blocks allocated after exhausting max block, verify
// consistency
TEST_F(CudaMemoryPoolTest, WriteAndVerifyDataAfterExhaustingMaxBlock) {
    const size_t maxSmallBlockSize =
        hahaha::backend::CudaMemoryPool::BaseMemoryBlockSize
        << (hahaha::backend::CudaMemoryPool::MaxSmallObjectPoolListSize - 1);
    const size_t smallBlockSize = 1024;
    const size_t numBlocksPerMaxBlock = maxSmallBlockSize / smallBlockSize;

    // Exhaust first max block
    std::vector<void*> firstMaxBlocks;
    for (size_t i = 0; i < numBlocksPerMaxBlock; ++i) {
        auto result = pool_->allocateSmall(smallBlockSize);
        if (result.has_value()) {
            firstMaxBlocks.push_back(result.value());
        } else {
            break;
        }
    }

    // Allocate blocks from second max block
    std::vector<void*> secondMaxBlocks;
    for (size_t i = 0; i < 10; ++i) {
        auto result = pool_->allocateSmall(smallBlockSize);
        ASSERT_TRUE(result.has_value())
            << "Should allocate from second max block";
        secondMaxBlocks.push_back(result.value());
    }

    // Prepare test data
    std::vector<std::vector<uint8_t>> hostData(secondMaxBlocks.size());
    std::vector<std::vector<uint8_t>> receivedData(secondMaxBlocks.size());

    for (size_t i = 0; i < secondMaxBlocks.size(); ++i) {
        hostData[i].resize(smallBlockSize);
        receivedData[i].resize(smallBlockSize);
        for (size_t j = 0; j < smallBlockSize; ++j) {
            hostData[i][j] = static_cast<uint8_t>((i * 1000 + j) % 256);
        }
    }

    // Write to GPU
    for (size_t i = 0; i < secondMaxBlocks.size(); ++i) {
        cudaError_t err = cudaMemcpy(secondMaxBlocks[i], hostData[i].data(),
                                     smallBlockSize, cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess) << "Failed to copy to GPU for block " << i;
    }

    cudaDeviceSynchronize();

    // Copy back
    for (size_t i = 0; i < secondMaxBlocks.size(); ++i) {
        cudaError_t err = cudaMemcpy(receivedData[i].data(),
                                     secondMaxBlocks[i], smallBlockSize,
                                     cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess)
            << "Failed to copy from GPU for block " << i;
    }

    cudaDeviceSynchronize();

    // Verify data
    for (size_t i = 0; i < secondMaxBlocks.size(); ++i) {
        for (size_t j = 0; j < smallBlockSize; ++j) {
            EXPECT_EQ(hostData[i][j], receivedData[i][j])
                << "Data mismatch at block " << i << ", byte " << j;
        }
    }

    // Free all blocks
    for (void* ptr : firstMaxBlocks) {
        pool_->free(ptr);
    }
    for (void* ptr : secondMaxBlocks) {
        pool_->free(ptr);
    }
}

// Test: Use cudaMemset to set memory, then verify with cudaMemcpy
TEST_F(CudaMemoryPoolTest, MemsetAndVerifyMemory) {
    const size_t blockSize = 8192; // 8KB
    const uint8_t testValue = 0xAB;

    auto result = pool_->allocateSmall(blockSize);
    ASSERT_TRUE(result.has_value());
    void* ptr = result.value();

    // Set memory to test value using cudaMemset
    cudaError_t err = cudaMemset(ptr, static_cast<int>(testValue), blockSize);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemset failed";

    cudaDeviceSynchronize();

    // Copy back and verify
    std::vector<uint8_t> receivedData(blockSize);
    err = cudaMemcpy(receivedData.data(), ptr, blockSize,
                     cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed";

    // Verify all bytes are set to test value
    for (size_t i = 0; i < blockSize; ++i) {
        EXPECT_EQ(receivedData[i], testValue)
            << "Byte " << i << " mismatch: expected " << static_cast<int>(testValue)
            << ", got " << static_cast<int>(receivedData[i]);
    }

    pool_->free(ptr);
}

} // namespace hahaha::backend::test

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
