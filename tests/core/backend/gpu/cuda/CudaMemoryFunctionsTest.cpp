#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>) && __has_include(<cuda_runtime_api.h>)

#include "backend/gpu/cuda/cuda_memory.cuh"

namespace hahaha::backend::test {

class CudaMemoryFunctionsTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test basic functionality of cudaMemoryAllocate function
TEST_F(CudaMemoryFunctionsTest, BasicAllocate) {
    void* ptr = nullptr;
    const size_t size = 1024;

    cudaError_t result = cudaMemoryAllocate(size, &ptr);

    EXPECT_EQ(result, cudaSuccess);
    EXPECT_NE(ptr, nullptr);

    // Free memory
    cudaError_t free_result = cudaMemoryFree(ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test boundary case of cudaMemoryAllocate function: zero size
TEST_F(CudaMemoryFunctionsTest, ZeroSizeAllocate) {
    void* ptr = nullptr;
    cudaError_t result = cudaMemoryAllocate(0, &ptr);

    // According to implementation, zero-size allocation should return error
    EXPECT_EQ(result, cudaErrorInvalidValue);
    EXPECT_EQ(ptr, nullptr);
}

// Test boundary case of cudaMemoryAllocate function: null pointer
TEST_F(CudaMemoryFunctionsTest, NullPtrAllocate) {
    cudaError_t result = cudaMemoryAllocate(1024, nullptr);

    // When output pointer is null, should return error
    EXPECT_EQ(result, cudaErrorInvalidValue);
}

// Test basic functionality of cudaMemoryFree function
TEST_F(CudaMemoryFunctionsTest, BasicFree) {
    void* ptr = nullptr;
    const size_t size = 1024;

    // First allocate memory
    cudaError_t alloc_result = cudaMemoryAllocate(size, &ptr);
    EXPECT_EQ(alloc_result, cudaSuccess);
    EXPECT_NE(ptr, nullptr);

    // Then free
    cudaError_t free_result = cudaMemoryFree(ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test boundary case of cudaMemoryFree function: null pointer
TEST_F(CudaMemoryFunctionsTest, NullPtrFree) {
    // Freeing null pointer should succeed
    cudaError_t result = cudaMemoryFree(nullptr);
    EXPECT_EQ(result, cudaSuccess);
}

// Test basic functionality of cudaMemorySet function
TEST_F(CudaMemoryFunctionsTest, BasicMemset) {
    void* ptr = nullptr;
    const size_t size = 1024;

    // Allocate memory
    cudaError_t alloc_result = cudaMemoryAllocate(size, &ptr);
    EXPECT_EQ(alloc_result, cudaSuccess);
    EXPECT_NE(ptr, nullptr);

    // Set memory
    const int value = 0x42;
    cudaError_t memset_result = cudaMemorySet(ptr, value, size);
    EXPECT_EQ(memset_result, cudaSuccess);

    // Free memory
    cudaError_t free_result = cudaMemoryFree(ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test boundary case of cudaMemorySet function: zero size
TEST_F(CudaMemoryFunctionsTest, ZeroSizeMemset) {
    void* ptr = nullptr;
    const size_t size = 1024;

    // Allocate memory
    cudaError_t alloc_result = cudaMemoryAllocate(size, &ptr);
    EXPECT_EQ(alloc_result, cudaSuccess);
    EXPECT_NE(ptr, nullptr);

    // Setting memory with zero bytes should succeed
    const int value = 0x42;
    cudaError_t memset_result = cudaMemorySet(ptr, value, 0);
    EXPECT_EQ(memset_result, cudaSuccess);

    // Free memory
    cudaError_t free_result = cudaMemoryFree(ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test boundary case of cudaMemorySet function: null pointer and zero size
TEST_F(CudaMemoryFunctionsTest, NullPtrAndZeroSizeMemset) {
    // Setting memory on null pointer with zero size should succeed
    cudaError_t result = cudaMemorySet(nullptr, 0x42, 0);
    EXPECT_EQ(result, cudaSuccess);
}

// Test error case of cudaMemorySet function: null pointer
TEST_F(CudaMemoryFunctionsTest, NullPtrMemset) {
    // Setting memory on null pointer should succeed (according to
    // implementation)
    cudaError_t result = cudaMemorySet(nullptr, 0x42, 100);
    EXPECT_EQ(result, cudaSuccess);
}

// Test multiple small memory allocations to test performance and stability
TEST_F(CudaMemoryFunctionsTest, MultipleSmallAllocations) {
    const size_t num_allocations = 100;
    const size_t size_per_allocation = 256;
    std::vector<void*> pointers(num_allocations);

    // Allocate multiple small blocks
    for (size_t i = 0; i < num_allocations; ++i) {
        cudaError_t result = cudaMemoryAllocate(size_per_allocation, &pointers[i]);
        EXPECT_EQ(result, cudaSuccess);
        EXPECT_NE(pointers[i], nullptr);
    }

    // Free all memory
    for (size_t i = 0; i < num_allocations; ++i) {
        cudaError_t result = cudaMemoryFree(pointers[i]);
        EXPECT_EQ(result, cudaSuccess);
    }
}

// Test large memory allocation
TEST_F(CudaMemoryFunctionsTest, LargeAllocation) {
    void* ptr = nullptr;
    const size_t large_size = 16 * 1024 * 1024; // 16MB

    cudaError_t result = cudaMemoryAllocate(large_size, &ptr);

    if (result == cudaSuccess) {
        EXPECT_NE(ptr, nullptr);

        // Test setting large memory
        cudaError_t memset_result = cudaMemorySet(ptr, 0xAA, large_size);
        EXPECT_EQ(memset_result, cudaSuccess);

        // Free large memory
        cudaError_t free_result = cudaMemoryFree(ptr);
        EXPECT_EQ(free_result, cudaSuccess);
    }
    // If allocation fails (due to insufficient memory), this is acceptable
}

// Test 012 principle - 0 allocations
TEST_F(CudaMemoryFunctionsTest, ZeroAllocation) {
    // This test just ensures system is in clean state
    SUCCEED();
}

// Test 012 principle - 1 allocation
TEST_F(CudaMemoryFunctionsTest, SingleAllocation) {
    void* ptr = nullptr;
    const size_t size = 512;

    cudaError_t alloc_result = cudaMemoryAllocate(size, &ptr);
    EXPECT_EQ(alloc_result, cudaSuccess);
    EXPECT_NE(ptr, nullptr);

    cudaError_t free_result = cudaMemoryFree(ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test 012 principle - 2 allocations
TEST_F(CudaMemoryFunctionsTest, DoubleAllocation) {
    void* ptr1 = nullptr;
    void* ptr2 = nullptr;
    const size_t size = 512;

    cudaError_t alloc_result1 = cudaMemoryAllocate(size, &ptr1);
    cudaError_t alloc_result2 = cudaMemoryAllocate(size, &ptr2);

    EXPECT_EQ(alloc_result1, cudaSuccess);
    EXPECT_EQ(alloc_result2, cudaSuccess);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2); // Should be different addresses

    cudaError_t free_result1 = cudaMemoryFree(ptr1);
    cudaError_t free_result2 = cudaMemoryFree(ptr2);

    EXPECT_EQ(free_result1, cudaSuccess);
    EXPECT_EQ(free_result2, cudaSuccess);
}

// Test continuous allocation and deallocation to simulate memory pool usage
// pattern
TEST_F(CudaMemoryFunctionsTest, AllocationDeallocationPattern) {
    const size_t iterations = 50;

    for (size_t i = 0; i < iterations; ++i) {
        void* ptr = nullptr;
        size_t size = 1024 + (i % 10) * 100; // Varying size

        cudaError_t alloc_result = cudaMemoryAllocate(size, &ptr);
        EXPECT_EQ(alloc_result, cudaSuccess);
        EXPECT_NE(ptr, nullptr);

        // Set memory
        cudaError_t memset_result = cudaMemorySet(ptr, i % 256, size);
        EXPECT_EQ(memset_result, cudaSuccess);

        // Free immediately
        cudaError_t free_result = cudaMemoryFree(ptr);
        EXPECT_EQ(free_result, cudaSuccess);
    }
}

// Test invalid input combinations
TEST_F(CudaMemoryFunctionsTest, InvalidInputCombinations) {
    // Test null pointer and non-zero size allocation
    cudaError_t result1 = cudaMemoryAllocate(1024, nullptr);
    EXPECT_EQ(result1, cudaErrorInvalidValue);

    // Test zero-size allocation
    void* null_ptr = nullptr;
    cudaError_t result2 = cudaMemoryAllocate(0, &null_ptr);
    EXPECT_EQ(result2, cudaErrorInvalidValue);
    EXPECT_EQ(null_ptr, nullptr);
}

// Test edge case: very small allocation
TEST_F(CudaMemoryFunctionsTest, VerySmallAllocation) {
    void* ptr = nullptr;

    // Test 1-byte allocation
    cudaError_t result = cudaMemoryAllocate(1, &ptr);

    if (result == cudaSuccess) {
        EXPECT_NE(ptr, nullptr);

        cudaError_t free_result = cudaMemoryFree(ptr);
        EXPECT_EQ(free_result, cudaSuccess);
    }
    // Allocation failure is also acceptable
}

// Test edge case: single-byte memory set
TEST_F(CudaMemoryFunctionsTest, SingleByteMemset) {
    void* ptr = nullptr;

    cudaError_t alloc_result = cudaMemoryAllocate(1, &ptr);

    if (alloc_result == cudaSuccess) {
        EXPECT_NE(ptr, nullptr);

        // Set single byte
        cudaError_t memset_result = cudaMemorySet(ptr, 0xFF, 1);
        EXPECT_EQ(memset_result, cudaSuccess);

        cudaError_t free_result = cudaMemoryFree(ptr);
        EXPECT_EQ(free_result, cudaSuccess);
    }
}

// --- cudaMemoryCopy tests ---

// Test basic host-to-device copy
TEST_F(CudaMemoryFunctionsTest, BasicCopyHostToDevice) {
    const size_t size = 1024;
    std::vector<std::byte> host_src(size);
    for (size_t i = 0; i < size; ++i) {
        host_src[i] = static_cast<std::byte>(i % 256);
    }

    void* device_ptr = nullptr;
    cudaError_t alloc_result = cudaMemoryAllocate(size, &device_ptr);
    ASSERT_EQ(alloc_result, cudaSuccess);
    ASSERT_NE(device_ptr, nullptr);

    cudaError_t copy_result =
        cudaMemoryCopy(device_ptr, host_src.data(), size, cudaMemcpyHostToDevice);
    EXPECT_EQ(copy_result, cudaSuccess);

    cudaError_t free_result = cudaMemoryFree(device_ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test basic device-to-host copy (round-trip: host -> device -> host)
TEST_F(CudaMemoryFunctionsTest, BasicCopyDeviceToHost) {
    const size_t size = 1024;
    std::vector<std::byte> host_src(size);
    for (size_t i = 0; i < size; ++i) {
        host_src[i] = static_cast<std::byte>(i % 256);
    }

    void* device_ptr = nullptr;
    cudaError_t alloc_result = cudaMemoryAllocate(size, &device_ptr);
    ASSERT_EQ(alloc_result, cudaSuccess);
    ASSERT_NE(device_ptr, nullptr);

    cudaError_t to_device =
        cudaMemoryCopy(device_ptr, host_src.data(), size, cudaMemcpyHostToDevice);
    ASSERT_EQ(to_device, cudaSuccess);

    std::vector<std::byte> host_dst(size);
    cudaError_t to_host =
        cudaMemoryCopy(host_dst.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    EXPECT_EQ(to_host, cudaSuccess);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host_src[i], host_dst[i]);
    }

    cudaError_t free_result = cudaMemoryFree(device_ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test zero-count copy returns success
TEST_F(CudaMemoryFunctionsTest, CopyZeroCount) {
    void* device_ptr = nullptr;
    cudaError_t alloc_result = cudaMemoryAllocate(256, &device_ptr);
    ASSERT_EQ(alloc_result, cudaSuccess);
    ASSERT_NE(device_ptr, nullptr);

    std::vector<std::byte> host_buf(256);
    cudaError_t copy_result =
        cudaMemoryCopy(device_ptr, host_buf.data(), 0, cudaMemcpyHostToDevice);
    EXPECT_EQ(copy_result, cudaSuccess);

    copy_result =
        cudaMemoryCopy(host_buf.data(), device_ptr, 0, cudaMemcpyDeviceToHost);
    EXPECT_EQ(copy_result, cudaSuccess);

    cudaError_t free_result = cudaMemoryFree(device_ptr);
    EXPECT_EQ(free_result, cudaSuccess);
}

// Test null destination returns invalid value
TEST_F(CudaMemoryFunctionsTest, CopyNullDst) {
    std::vector<std::byte> host_src(64);
    cudaError_t result =
        cudaMemoryCopy(nullptr, host_src.data(), 64, cudaMemcpyHostToDevice);
    EXPECT_EQ(result, cudaErrorInvalidValue);
}

// Test null source returns invalid value
TEST_F(CudaMemoryFunctionsTest, CopyNullSrc) {
    void* device_ptr = nullptr;
    cudaError_t alloc_result = cudaMemoryAllocate(64, &device_ptr);
    ASSERT_EQ(alloc_result, cudaSuccess);
    ASSERT_NE(device_ptr, nullptr);

    cudaError_t result =
        cudaMemoryCopy(device_ptr, nullptr, 64, cudaMemcpyHostToDevice);
    EXPECT_EQ(result, cudaErrorInvalidValue);

    cudaMemoryFree(device_ptr);
}

} // namespace hahaha::backend::test

#endif // __has_include(<driver_types.h>) && __has_include(<cuda_runtime_api.h>)
#endif // HAHAHA_USE_CUDA
