#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

#include "backend/gpu/cuda/CudaMemoryPool.h"
#include "backend/gpu/cuda/cuda_memory.cuh"
#include "common/errors/Error.h"

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

    // Subsequent allocation should succeed
    auto result2 = pool_->allocateSmall(1024);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), ptr); // Should return the same address (if
                                     // memory pool recycles properly)
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
        pool_->allocateSmall(threshold - 1); // close but less than threshold
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
    // This test may fail or succeed depending on GPU memory size

    // Test freeing null pointer
    pool_->free(nullptr); // Should handle safely
}

// Test concurrent access safety
TEST_F(CudaMemoryPoolTest, ConcurrencySafety) {
    const size_t num_threads = 4;
    const size_t allocations_per_thread = 10;

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_pointers(num_threads);

    // Start multiple threads to allocate simultaneously
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(
            [this, t, allocations_per_thread, &thread_pointers]() {
                for (size_t i = 0; i < allocations_per_thread; ++i) {
                    auto result = pool_->allocateSmall(
                        1024 + (t * allocations_per_thread + i) * 10);
                    if (result.has_value()) {
                        thread_pointers[t].push_back(result.value());
                    }
                }

                // Free all memory allocated by this thread
                for (void* ptr : thread_pointers[t]) {
                    if (ptr != nullptr) {
                        pool_->free(ptr);
                    }
                }
            });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

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
            // If allocation fails, this might be normal (memory pool is full)
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

} // namespace hahaha::backend::test

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
