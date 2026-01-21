// #include <cstring>
// #include <gtest/gtest.h>
// #include <vector>
//
// #ifdef HAHAHA_USE_CUDA
// #if __has_include(<driver_types.h>)
//
// #include "backend/DeviceBuffer.h"
// #include "backend/gpu/cuda/CudaMemory.h"
//
// TODO: Add Test
// namespace hahaha::backend::test {
//
// class CudaMemoryTest : public ::testing::Test {
//   protected:
//     void SetUp() override {
//         cudaMemory_ = std::make_unique<CudaMemory>();
//     }
//
//     void TearDown() override {
//         cudaMemory_.reset();
//     }
//
//     std::unique_ptr<CudaMemory> cudaMemory_;
// };
//
// // Test basic allocation functionality
// TEST_F(CudaMemoryTest, BasicAllocateAndFree) {
//     const size_t size = 1024;
//     DeviceBuffer buffer = cudaMemory_->allocate(size);
//
//     EXPECT_NE(buffer.address(), 0);
//     EXPECT_EQ(buffer.size(), size);
//
//     // Free memory
//     cudaMemory_->free(buffer);
//
//     // Verify buffer has been cleared
//     EXPECT_EQ(buffer.address(), 0);
//     EXPECT_EQ(buffer.size(), 0);
// }
//
// // Test zero-size allocation
// TEST_F(CudaMemoryTest, ZeroSizeAllocation) {
//     DeviceBuffer buffer = cudaMemory_->allocate(0);
//
//     EXPECT_EQ(buffer.address(), 0);
//     EXPECT_EQ(buffer.size(), 0);
// }
//
// // Test 012 principle - zero allocation
// TEST_F(CudaMemoryTest, ZeroAllocation) {
//     DeviceBuffer buffer = cudaMemory_->allocate(0);
//     EXPECT_EQ(buffer.address(), 0);
//     EXPECT_EQ(buffer.size(), 0);
// }
//
// // Test 012 principle - single allocation
// TEST_F(CudaMemoryTest, SingleAllocation) {
//     const size_t size = 512;
//     DeviceBuffer buffer = cudaMemory_->allocate(size);
//
//     EXPECT_NE(buffer.address(), 0);
//     EXPECT_EQ(buffer.size(), size);
//
//     cudaMemory_->free(buffer);
// }
//
// // Test 012 principle - dual allocation
// TEST_F(CudaMemoryTest, DoubleAllocation) {
//     const size_t size1 = 512;
//     const size_t size2 = 1024;
//
//     DeviceBuffer buffer1 = cudaMemory_->allocate(size1);
//     DeviceBuffer buffer2 = cudaMemory_->allocate(size2);
//
//     EXPECT_NE(buffer1.address(), 0);
//     EXPECT_NE(buffer2.address(), 0);
//     EXPECT_NE(buffer1.address(), buffer2.address()); // Different addresses
//     EXPECT_EQ(buffer1.size(), size1);
//     EXPECT_EQ(buffer2.size(), size2);
//
//     cudaMemory_->free(buffer1);
//     cudaMemory_->free(buffer2);
// }
//
// // Test host-to-device data transfer
// TEST_F(CudaMemoryTest, HostToDeviceCopy) {
//     const size_t size = 1024;
//     std::vector<std::byte> host_data(size);
//
//     // Fill test data
//     for (size_t i = 0; i < size; ++i) {
//         host_data[i] = static_cast<std::byte>(i % 256);
//     }
//
//     DeviceBuffer device_buffer = cudaMemory_->allocate(size);
//     EXPECT_NE(device_buffer.address(), 0);
//
//     // Copy data to device
//     cudaMemory_->copyHostToDevice(device_buffer, host_data);
//
//     cudaMemory_->free(device_buffer);
// }
//
// // Test device-to-host data transfer
// TEST_F(CudaMemoryTest, DeviceToHostCopy) {
//     const size_t size = 1024;
//     std::vector<std::byte> host_data(size);
//     std::vector<std::byte> received_data(size);
//
//     // Fill initial data
//     for (size_t i = 0; i < size; ++i) {
//         host_data[i] = static_cast<std::byte>(i % 256);
//     }
//
//     DeviceBuffer device_buffer = cudaMemory_->allocate(size);
//     EXPECT_NE(device_buffer.address(), 0);
//
//     // First copy data to device
//     cudaMemory_->copyHostToDevice(device_buffer, host_data);
//
//     // Copy back from device to host
//     cudaMemory_->copyDeviceToHost(received_data, device_buffer);
//
//     // Verify data integrity
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_EQ(host_data[i], received_data[i]);
//     }
//
//     cudaMemory_->free(device_buffer);
// }
//
// // Test device-to-device data transfer
// TEST_F(CudaMemoryTest, DeviceToDeviceCopy) {
//     const size_t size = 1024;
//     std::vector<std::byte> host_data(size);
//
//     // Fill test data
//     for (size_t i = 0; i < size; ++i) {
//         host_data[i] = static_cast<std::byte>(i % 256);
//     }
//
//     DeviceBuffer src_buffer = cudaMemory_->allocate(size);
//     DeviceBuffer dst_buffer = cudaMemory_->allocate(size);
//
//     EXPECT_NE(src_buffer.address(), 0);
//     EXPECT_NE(dst_buffer.address(), 0);
//     EXPECT_NE(src_buffer.address(), dst_buffer.address());
//
//     // Copy data to source device buffer
//     cudaMemory_->copyHostToDevice(src_buffer, host_data);
//
//     // Read data from destination device buffer
//     std::vector<std::byte> received_data(size);
//     cudaMemory_->copyDeviceToHost(received_data, dst_buffer);
//
//     // Verify data integrity
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_EQ(host_data[i], received_data[i]);
//     }
//
//     cudaMemory_->free(src_buffer);
//     cudaMemory_->free(dst_buffer);
// }
//
// // Test memory set functionality
// TEST_F(CudaMemoryTest, MemsetFunctionality) {
//     const size_t size = 1024;
//     DeviceBuffer buffer = cudaMemory_->allocate(size);
//
//     EXPECT_NE(buffer.address(), 0);
//
//     // Set memory to specific value
//     const int value = 0x42;
//     cudaMemory_->memset(buffer, value, size);
//
//     cudaMemory_->free(buffer);
// }
//
// // Test large/small memory allocation switching
// TEST_F(CudaMemoryTest, SmallVsBigAllocation) {
//     // Test small memory allocation (< default threshold)
//     const size_t small_size =
//         10 * 1024 * 1024; // 10MB, less than default 512MB threshold
//     DeviceBuffer small_buffer = cudaMemory_->allocate(small_size);
//     EXPECT_NE(small_buffer.address(), 0);
//     EXPECT_EQ(small_buffer.size(), small_size);
//     cudaMemory_->free(small_buffer);
//
//     // Test large memory allocation (>= threshold)
//     const size_t big_size =
//         1 * 1024 * 1024 * 1024; // 1GB, greater than threshold
//     DeviceBuffer big_buffer = cudaMemory_->allocate(big_size);
//     EXPECT_NE(big_buffer.address(), 0);
//     EXPECT_EQ(big_buffer.size(), big_size);
//     cudaMemory_->free(big_buffer);
// }
//
// // Test multiple allocations and deallocations
// TEST_F(CudaMemoryTest, MultipleAllocations) {
//     const size_t num_allocations = 10;
//     std::vector<DeviceBuffer> buffers;
//
//     // Allocate multiple buffers
//     for (size_t i = 0; i < num_allocations; ++i) {
//         const size_t size = 512 + i * 100;
//         DeviceBuffer buffer = cudaMemory_->allocate(size);
//         EXPECT_NE(buffer.address(), 0);
//         EXPECT_EQ(buffer.size(), size);
//         buffers.push_back(std::move(buffer));
//     }
//
//     // Free all buffers
//     for (auto& buffer : buffers) {
//         cudaMemory_->free(buffer);
//     }
// }
//
// // Test boundary case: maximum possible allocation
// TEST_F(CudaMemoryTest, MaximumAllocationAttempt) {
//     // Try to allocate extremely large memory (might fail but shouldn't
//     crash) DeviceBuffer buffer =
//     cudaMemory_->allocate(static_cast<size_t>(-1));
//
//     // Depending on actual GPU memory, this may succeed or fail
//     // The important thing is the program shouldn't crash
//     if (buffer.address() != 0) {
//         cudaMemory_->free(buffer);
//     }
// }
//
// // Test error handling: size mismatched copy operations
// TEST_F(CudaMemoryTest, CopySizeMismatchHandling) {
//     const size_t size1 = 1024;
//     const size_t size2 = 2048;
//
//     DeviceBuffer buffer1 = cudaMemory_->allocate(size1);
//     DeviceBuffer buffer2 = cudaMemory_->allocate(size2);
//
//     std::vector<std::byte> host_data(size1);
//
//     // Correct copy should succeed
//     cudaMemory_->copyHostToDevice(buffer1, host_data);
//
//     // Try incorrect size copy operation
//     std::vector<std::byte> wrong_size_data(size2);
//
//     // Expect exception or appropriate error handling
//     EXPECT_THROW(cudaMemory_->copyHostToDevice(
//                      buffer2, std::span<const std::byte>(host_data)),
//                  std::runtime_error);
//
//     cudaMemory_->free(buffer1);
//     cudaMemory_->free(buffer2);
// }
//
// // Test robustness with zero values and null pointers
// TEST_F(CudaMemoryTest, RobustnessWithZeroValues) {
//     // Test handling of empty device buffer
//     DeviceBuffer empty_buffer;
//     EXPECT_EQ(empty_buffer.address(), 0);
//     EXPECT_EQ(empty_buffer.size(), 0);
//
//     // Freeing empty buffer should not crash
//     cudaMemory_->free(empty_buffer);
//
//     // Test zero-length copy
//     std::vector<std::byte> empty_data(0);
//     DeviceBuffer buffer = cudaMemory_->allocate(1024);
//     EXPECT_NE(buffer.address(), 0);
//
//     // Zero-length copy should execute safely
//     cudaMemory_->copyHostToDevice(buffer,
//                                   std::span<const std::byte>(empty_data));
//
//     cudaMemory_->free(buffer);
// }
//
// // Test memory pool threshold switching
// TEST_F(CudaMemoryTest, ThresholdBoundaryTest) {
//     const size_t threshold = cudaMemory_->getSmallBlockThreshold();
//
//     // Test allocation just below threshold
//     if (threshold > 1) {
//         DeviceBuffer small_buffer = cudaMemory_->allocate(threshold - 1);
//         EXPECT_NE(small_buffer.address(), 0);
//         cudaMemory_->free(small_buffer);
//     }
//
//     // Test allocation equal to threshold
//     if (threshold > 0) {
//         DeviceBuffer threshold_buffer = cudaMemory_->allocate(threshold);
//         EXPECT_NE(threshold_buffer.address(), 0);
//         cudaMemory_->free(threshold_buffer);
//     }
//
//     // Test allocation slightly above threshold
//     DeviceBuffer big_buffer = cudaMemory_->allocate(threshold + 1);
//     EXPECT_NE(big_buffer.address(), 0);
//     cudaMemory_->free(big_buffer);
// }
//
// // // Test small memory allocation functionality
// // TEST_F(CudaMemoryTest, SmallMemoryAllocation) {
// //     DeviceBuffer buffer = cudaMemory_->allocateSmall(1024);
// //     EXPECT_NE(buffer.address(), 0);
// //     EXPECT_EQ(buffer.size(), 1024);
// //     cudaMemory_->free(buffer);
// // }
// //
// // // Test large memory allocation functionality
// // TEST_F(CudaMemoryTest, BigMemoryAllocation) {
// //     DeviceBuffer buffer = cudaMemory_->allocateBig(10 * 1024 * 1024); //
// 10MB
// //     EXPECT_NE(buffer.address(), 0);
// //     EXPECT_EQ(buffer.size(), 10 * 1024 * 1024);
// //     cudaMemory_->free(buffer);
// // }
//
// // Test memset edge cases
// TEST_F(CudaMemoryTest, MemsetEdgeCases) {
//     DeviceBuffer buffer = cudaMemory_->allocate(1024);
//     EXPECT_NE(buffer.address(), 0);
//
//     // Test 0-length memset
//     cudaMemory_->memset(buffer, 0x55, 0);
//
//     // Test full-length memset
//     cudaMemory_->memset(buffer, 0xAA, 1024);
//
//     // Test partial-length memset
//     cudaMemory_->memset(buffer, 0xBB, 512);
//
//     cudaMemory_->free(buffer);
// }
//
// // Test copy operations with empty device buffers
// TEST_F(CudaMemoryTest, EmptyBufferCopyOperations) {
//     DeviceBuffer empty_buffer;
//
//     std::vector<std::byte> host_data(1024);
//
//     // Try copy operations on empty buffer, should throw exception
//     EXPECT_THROW(cudaMemory_->copyHostToDevice(empty_buffer, host_data),
//                  std::runtime_error);
//
//     std::vector<std::byte> dest_data(1024);
//     EXPECT_THROW(cudaMemory_->copyDeviceToHost(dest_data, empty_buffer),
//                  std::runtime_error);
//     ;
// }
//
// // Test memory allocation failure scenarios
// TEST_F(CudaMemoryTest, AllocationFailureScenarios) {
//     // Try to allocate oversized memory, which may cause allocation failure
//     DeviceBuffer huge_buffer =
//         cudaMemory_->allocate(static_cast<size_t>(-1) / 2);
//     // This may return empty buffer but shouldn't crash
// }
//
// } // namespace hahaha::backend::test
//
// #endif // __has_include(<driver_types.h>)
// #endif // HAHAHA_USE_CUDA
