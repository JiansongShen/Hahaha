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
#include <gtest/gtest.h>

#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)

#include <cuda_runtime.h>
#include <vector>

#include "backend/gpu/cuda/cuda_compute_fun.h"

namespace hahaha::backend::test {

class CudaComputeFunTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize CUDA if needed
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// Test cudaComputeAdd
TEST_F(CudaComputeFunTest, ComputeAdd_Basic) {
    const size_t size = 1024;

    std::vector<float> input1(size, 1.0f);
    std::vector<float> input2(size, 2.0f);
    std::vector<float> output(size, 0.0f);

    const size_t bytes = size * sizeof(float);
    cf32* d_buf1 = nullptr;
    cf32* d_buf2 = nullptr;
    cf32* d_bufOut = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf1, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf2, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bufOut, bytes), cudaSuccess);

    cudaMemcpy(d_buf1, input1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf2, input2.data(), bytes, cudaMemcpyHostToDevice);

    cudaError_t err = cudaComputeAdd(d_buf1, d_buf2, d_bufOut, size, 256);

    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_bufOut, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 3.0f);
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_bufOut);
}

// Test cudaComputeSubtract
TEST_F(CudaComputeFunTest, ComputeSubtract_Basic) {
    const size_t size = 512;

    std::vector<float> input1(size, 5.0f);
    std::vector<float> input2(size, 2.0f);
    std::vector<float> output(size, 0.0f);

    const size_t bytes = size * sizeof(float);
    cf32* d_buf1 = nullptr;
    cf32* d_buf2 = nullptr;
    cf32* d_bufOut = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf1, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf2, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bufOut, bytes), cudaSuccess);

    cudaMemcpy(d_buf1, input1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf2, input2.data(), bytes, cudaMemcpyHostToDevice);

    cudaError_t err = cudaComputeSubtract(d_buf1, d_buf2, d_bufOut, size, 256);

    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_bufOut, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 3.0f);
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_bufOut);
}

// Test cudaComputeMultiply
TEST_F(CudaComputeFunTest, ComputeMultiply_Basic) {
    const size_t size = 256;

    std::vector<float> input1(size, 3.0f);
    std::vector<float> input2(size, 4.0f);
    std::vector<float> output(size, 0.0f);

    const size_t bytes = size * sizeof(float);
    cf32* d_buf1 = nullptr;
    cf32* d_buf2 = nullptr;
    cf32* d_bufOut = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf1, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf2, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bufOut, bytes), cudaSuccess);

    cudaMemcpy(d_buf1, input1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf2, input2.data(), bytes, cudaMemcpyHostToDevice);

    cudaError_t err = cudaComputeMultiply(d_buf1, d_buf2, d_bufOut, size, 256);

    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_bufOut, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 12.0f);
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_bufOut);
}

// Test cudaComputeDivide
TEST_F(CudaComputeFunTest, ComputeDivide_Basic) {
    const size_t size = 128;

    std::vector<float> input1(size, 12.0f);
    std::vector<float> input2(size, 3.0f);
    std::vector<float> output(size, 0.0f);

    const size_t bytes = size * sizeof(float);
    cf32* d_buf1 = nullptr;
    cf32* d_buf2 = nullptr;
    cf32* d_bufOut = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf1, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf2, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bufOut, bytes), cudaSuccess);

    cudaMemcpy(d_buf1, input1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf2, input2.data(), bytes, cudaMemcpyHostToDevice);

    cudaError_t err = cudaComputeDivide(d_buf1, d_buf2, d_bufOut, size, 256);

    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_bufOut, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 4.0f);
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_bufOut);
}

// Test cudaComputeDivide with zero divisor (should handle gracefully)
TEST_F(CudaComputeFunTest, ComputeDivide_ZeroDivisor) {
    const size_t size = 64;

    std::vector<float> input1(size, 10.0f);
    std::vector<float> input2(size, 0.0f); // Zero divisor
    std::vector<float> output(size, 999.0f);

    const size_t bytes = size * sizeof(float);
    cf32* d_buf1 = nullptr;
    cf32* d_buf2 = nullptr;
    cf32* d_bufOut = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf1, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf2, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bufOut, bytes), cudaSuccess);

    cudaMemcpy(d_buf1, input1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf2, input2.data(), bytes, cudaMemcpyHostToDevice);

    cudaError_t err = cudaComputeDivide(d_buf1, d_buf2, d_bufOut, size, 256);

    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_bufOut, bytes, cudaMemcpyDeviceToHost);

    // Kernel should set output to 0.0f when divisor is zero
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_bufOut);
}

// Test with different block sizes
TEST_F(CudaComputeFunTest, ComputeAdd_DifferentBlockSizes) {
    const size_t size = 1000; // Not a multiple of block size

    std::vector<float> input1(size);
    std::vector<float> input2(size);
    std::vector<float> output(size, 0.0f);

    for (size_t i = 0; i < size; ++i) {
        input1[i] = static_cast<float>(i);
        input2[i] = static_cast<float>(i * 2);
    }

    const size_t bytes = size * sizeof(float);
    cf32* d_buf1 = nullptr;
    cf32* d_buf2 = nullptr;
    cf32* d_bufOut = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf1, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_buf2, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bufOut, bytes), cudaSuccess);

    cudaMemcpy(d_buf1, input1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf2, input2.data(), bytes, cudaMemcpyHostToDevice);

    cudaError_t err = cudaComputeAdd(d_buf1, d_buf2, d_bufOut, size, 128);

    EXPECT_EQ(err, cudaSuccess);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_bufOut, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], static_cast<float>(i * 3));
    }

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_bufOut);
}

// Test error handling with null pointers
TEST_F(CudaComputeFunTest, ComputeAdd_InvalidInput) {
    const size_t size = 100;
    const size_t bytes = size * sizeof(float);
    cf32* d_buf = nullptr;

    ASSERT_EQ(cudaMalloc(&d_buf, bytes), cudaSuccess);

    // Test with null pointers
    cudaError_t err1 = cudaComputeAdd(nullptr, d_buf, d_buf, size);
    EXPECT_EQ(err1, cudaErrorInvalidValue);

    cudaError_t err2 = cudaComputeAdd(d_buf, nullptr, d_buf, size);
    EXPECT_EQ(err2, cudaErrorInvalidValue);

    cudaError_t err3 = cudaComputeAdd(d_buf, d_buf, nullptr, size);
    EXPECT_EQ(err3, cudaErrorInvalidValue);

    // Test with zero size
    cudaError_t err4 = cudaComputeAdd(d_buf, d_buf, d_buf, 0);
    EXPECT_EQ(err4, cudaErrorInvalidValue);

    cudaFree(d_buf);
}

} // namespace hahaha::backend::test

#endif // __has_include(<driver_types.h>)
#endif // HAHAHA_USE_CUDA
