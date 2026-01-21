// // Copyright (c) 2025 Contributors of
// Hahaha(https://github.com/Napbad/Hahaha)
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      https://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// // Contributors:
// // Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
// TODO: CudaComputeFun
// #include <gtest/gtest.h>
//
// #ifdef HAHAHA_USE_CUDA
// #if __has_include(<driver_types.h>)
//
// #include <cuda_runtime.h>
// #include <span>
// #include <stdexcept>
// #include <vector>
//
// #include "backend/DeviceBuffer.h"
// #include "backend/gpu/cuda/CudaMemory.h"
// #include "backend/gpu/cuda/cuda_compute_fun.h"
//
// namespace hahaha::backend::test {
//
// class CudaComputeFunTest : public ::testing::Test {
//   protected:
//     void SetUp() override {
//         // Initialize CUDA if needed
//         cudaDeviceSynchronize();
//     }
//
//     void TearDown() override {
//         cudaDeviceSynchronize();
//     }
// };
//
// // Test cudaComputeAdd
// TEST_F(CudaComputeFunTest, ComputeAdd_Basic) {
//     const size_t size = 1024;
//     CudaMemory cudaMem;
//
//     // Allocate host and device memory
//     std::vector<float> input1(size, 1.0f);
//     std::vector<float> input2(size, 2.0f);
//     std::vector<float> output(size, 0.0f);
//
//     const size_t bytes = size * sizeof(float);
//     DeviceBuffer buf1 = cudaMem.allocate(bytes);
//     DeviceBuffer buf2 = cudaMem.allocate(bytes);
//     DeviceBuffer bufOut = cudaMem.allocate(bytes);
//
//     // Copy to device
//     cudaMem.copyHostToDevice(
//         buf1,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input1.data()), bytes));
//     cudaMem.copyHostToDevice(
//         buf2,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input2.data()), bytes));
//
//     // Launch kernel
//     cudaError_t err = cudaComputeAdd(reinterpret_cast<cf32*>(buf1.address()),
//                                      reinterpret_cast<cf32*>(buf2.address()),
//                                      reinterpret_cast<cf32*>(bufOut.address()),
//                                      size,
//                                      256);
//
//     EXPECT_EQ(err, cudaSuccess);
//
//     // Synchronize and copy back
//     cudaDeviceSynchronize();
//     cudaMem.copyDeviceToHost(
//         std::span<std::byte>(reinterpret_cast<std::byte*>(output.data()),
//                              bytes),
//         bufOut);
//
//     // Verify results
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_FLOAT_EQ(output[i], 3.0f);
//     }
//
//     // Cleanup
//     cudaMem.free(buf1);
//     cudaMem.free(buf2);
//     cudaMem.free(bufOut);
// }
//
// // Test cudaComputeSubtract
// TEST_F(CudaComputeFunTest, ComputeSubtract_Basic) {
//     const size_t size = 512;
//     CudaMemory cudaMem;
//
//     std::vector<float> input1(size, 5.0f);
//     std::vector<float> input2(size, 2.0f);
//     std::vector<float> output(size, 0.0f);
//
//     const size_t bytes = size * sizeof(float);
//     DeviceBuffer buf1 = cudaMem.allocate(bytes);
//     DeviceBuffer buf2 = cudaMem.allocate(bytes);
//     DeviceBuffer bufOut = cudaMem.allocate(bytes);
//
//     cudaMem.copyHostToDevice(
//         buf1,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input1.data()), bytes));
//     cudaMem.copyHostToDevice(
//         buf2,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input2.data()), bytes));
//
//     cudaError_t err =
//         cudaComputeSubtract(reinterpret_cast<cf32*>(buf1.address()),
//                             reinterpret_cast<cf32*>(buf2.address()),
//                             reinterpret_cast<cf32*>(bufOut.address()),
//                             size,
//                             256);
//
//     EXPECT_EQ(err, cudaSuccess);
//
//     cudaDeviceSynchronize();
//     cudaMem.copyDeviceToHost(
//         std::span<std::byte>(reinterpret_cast<std::byte*>(output.data()),
//                              bytes),
//         bufOut);
//
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_FLOAT_EQ(output[i], 3.0f);
//     }
//
//     cudaMem.free(buf1);
//     cudaMem.free(buf2);
//     cudaMem.free(bufOut);
// }
//
// // Test cudaComputeMultiply
// TEST_F(CudaComputeFunTest, ComputeMultiply_Basic) {
//     const size_t size = 256;
//     CudaMemory cudaMem;
//
//     std::vector<float> input1(size, 3.0f);
//     std::vector<float> input2(size, 4.0f);
//     std::vector<float> output(size, 0.0f);
//
//     const size_t bytes = size * sizeof(float);
//     DeviceBuffer buf1 = cudaMem.allocate(bytes);
//     DeviceBuffer buf2 = cudaMem.allocate(bytes);
//     DeviceBuffer bufOut = cudaMem.allocate(bytes);
//
//     cudaMem.copyHostToDevice(
//         buf1,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input1.data()), bytes));
//     cudaMem.copyHostToDevice(
//         buf2,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input2.data()), bytes));
//
//     cudaError_t err =
//         cudaComputeMultiply(reinterpret_cast<cf32*>(buf1.address()),
//                             reinterpret_cast<cf32*>(buf2.address()),
//                             reinterpret_cast<cf32*>(bufOut.address()),
//                             size,
//                             256);
//
//     EXPECT_EQ(err, cudaSuccess);
//
//     cudaDeviceSynchronize();
//     cudaMem.copyDeviceToHost(
//         std::span<std::byte>(reinterpret_cast<std::byte*>(output.data()),
//                              bytes),
//         bufOut);
//
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_FLOAT_EQ(output[i], 12.0f);
//     }
//
//     cudaMem.free(buf1);
//     cudaMem.free(buf2);
//     cudaMem.free(bufOut);
// }
//
// // Test cudaComputeDivide
// TEST_F(CudaComputeFunTest, ComputeDivide_Basic) {
//     const size_t size = 128;
//     CudaMemory cudaMem;
//
//     std::vector<float> input1(size, 12.0f);
//     std::vector<float> input2(size, 3.0f);
//     std::vector<float> output(size, 0.0f);
//
//     const size_t bytes = size * sizeof(float);
//     DeviceBuffer buf1 = cudaMem.allocate(bytes);
//     DeviceBuffer buf2 = cudaMem.allocate(bytes);
//     DeviceBuffer bufOut = cudaMem.allocate(bytes);
//
//     cudaMem.copyHostToDevice(
//         buf1,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input1.data()), bytes));
//     cudaMem.copyHostToDevice(
//         buf2,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input2.data()), bytes));
//
//     cudaError_t err =
//         cudaComputeDivide(reinterpret_cast<cf32*>(buf1.address()),
//                           reinterpret_cast<cf32*>(buf2.address()),
//                           reinterpret_cast<cf32*>(bufOut.address()),
//                           size,
//                           256);
//
//     EXPECT_EQ(err, cudaSuccess);
//
//     cudaDeviceSynchronize();
//     cudaMem.copyDeviceToHost(
//         std::span<std::byte>(reinterpret_cast<std::byte*>(output.data()),
//                              bytes),
//         bufOut);
//
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_FLOAT_EQ(output[i], 4.0f);
//     }
//
//     cudaMem.free(buf1);
//     cudaMem.free(buf2);
//     cudaMem.free(bufOut);
// }
//
// // Test cudaComputeDivide with zero divisor (should handle gracefully)
// TEST_F(CudaComputeFunTest, ComputeDivide_ZeroDivisor) {
//     const size_t size = 64;
//     CudaMemory cudaMem;
//
//     std::vector<float> input1(size, 10.0f);
//     std::vector<float> input2(size, 0.0f); // Zero divisor
//     std::vector<float> output(size, 999.0f);
//
//     const size_t bytes = size * sizeof(float);
//     DeviceBuffer buf1 = cudaMem.allocate(bytes);
//     DeviceBuffer buf2 = cudaMem.allocate(bytes);
//     DeviceBuffer bufOut = cudaMem.allocate(bytes);
//
//     cudaMem.copyHostToDevice(
//         buf1,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input1.data()), bytes));
//     cudaMem.copyHostToDevice(
//         buf2,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input2.data()), bytes));
//
//     cudaError_t err =
//         cudaComputeDivide(reinterpret_cast<cf32*>(buf1.address()),
//                           reinterpret_cast<cf32*>(buf2.address()),
//                           reinterpret_cast<cf32*>(bufOut.address()),
//                           size,
//                           256);
//
//     EXPECT_EQ(err, cudaSuccess);
//
//     cudaDeviceSynchronize();
//     cudaMem.copyDeviceToHost(
//         std::span<std::byte>(reinterpret_cast<std::byte*>(output.data()),
//                              bytes),
//         bufOut);
//
//     // Kernel should set output to 0.0f when divisor is zero
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_FLOAT_EQ(output[i], 0.0f);
//     }
//
//     cudaMem.free(buf1);
//     cudaMem.free(buf2);
//     cudaMem.free(bufOut);
// }
//
// // Test with different block sizes
// TEST_F(CudaComputeFunTest, ComputeAdd_DifferentBlockSizes) {
//     const size_t size = 1000; // Not a multiple of block size
//     CudaMemory cudaMem;
//
//     std::vector<float> input1(size);
//     std::vector<float> input2(size);
//     std::vector<float> output(size, 0.0f);
//
//     for (size_t i = 0; i < size; ++i) {
//         input1[i] = static_cast<float>(i);
//         input2[i] = static_cast<float>(i * 2);
//     }
//
//     const size_t bytes = size * sizeof(float);
//     DeviceBuffer buf1 = cudaMem.allocate(bytes);
//     DeviceBuffer buf2 = cudaMem.allocate(bytes);
//     DeviceBuffer bufOut = cudaMem.allocate(bytes);
//
//     cudaMem.copyHostToDevice(
//         buf1,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input1.data()), bytes));
//     cudaMem.copyHostToDevice(
//         buf2,
//         std::span<const std::byte>(
//             reinterpret_cast<const std::byte*>(input2.data()), bytes));
//
//     // Test with block size 128
//     cudaError_t err = cudaComputeAdd(reinterpret_cast<cf32*>(buf1.address()),
//                                      reinterpret_cast<cf32*>(buf2.address()),
//                                      reinterpret_cast<cf32*>(bufOut.address()),
//                                      size,
//                                      128);
//
//     EXPECT_EQ(err, cudaSuccess);
//
//     cudaDeviceSynchronize();
//     cudaMem.copyDeviceToHost(
//         std::span<std::byte>(reinterpret_cast<std::byte*>(output.data()),
//                              bytes),
//         bufOut);
//
//     for (size_t i = 0; i < size; ++i) {
//         EXPECT_FLOAT_EQ(output[i], static_cast<float>(i * 3));
//     }
//
//     cudaMem.free(buf1);
//     cudaMem.free(buf2);
//     cudaMem.free(bufOut);
// }
//
// // Test error handling with null pointers
// TEST_F(CudaComputeFunTest, ComputeAdd_InvalidInput) {
//     const size_t size = 100;
//     CudaMemory cudaMem;
//
//     DeviceBuffer buf = cudaMem.allocate(size * sizeof(float));
//
//     // Test with null pointers
//     cudaError_t err1 = cudaComputeAdd(nullptr,
//                                       reinterpret_cast<cf32*>(buf.address()),
//                                       reinterpret_cast<cf32*>(buf.address()),
//                                       size);
//     EXPECT_EQ(err1, cudaErrorInvalidValue);
//
//     cudaError_t err2 = cudaComputeAdd(reinterpret_cast<cf32*>(buf.address()),
//                                       nullptr,
//                                       reinterpret_cast<cf32*>(buf.address()),
//                                       size);
//     EXPECT_EQ(err2, cudaErrorInvalidValue);
//
//     cudaError_t err3 = cudaComputeAdd(reinterpret_cast<cf32*>(buf.address()),
//                                       reinterpret_cast<cf32*>(buf.address()),
//                                       nullptr,
//                                       size);
//     EXPECT_EQ(err3, cudaErrorInvalidValue);
//
//     // Test with zero size
//     cudaError_t err4 = cudaComputeAdd(reinterpret_cast<cf32*>(buf.address()),
//                                       reinterpret_cast<cf32*>(buf.address()),
//                                       reinterpret_cast<cf32*>(buf.address()),
//                                       0);
//     EXPECT_EQ(err4, cudaErrorInvalidValue);
//
//     cudaMem.free(buf);
// }
//
// } // namespace hahaha::backend::test
//
// #endif // __has_include(<driver_types.h>)
// #endif // HAHAHA_USE_CUDA
