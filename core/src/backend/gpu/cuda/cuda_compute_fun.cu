#include "common/definitions.h"

// [C]uda [F]loat 32
using cf32 = hahaha::common::f32;

/**
 * @brief CUDA kernel for element-wise addition: output[i] = input1[i] +
 * input2[i]
 * @param input1 First input array
 * @param input2 Second input array
 * @param output Output array
 * @param size Number of elements
 */
__global__ void
compute_add(cf32* input1, cf32* input2, cf32* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise subtraction: output[i] = input1[i] -
 * input2[i]
 * @param input1 First input array
 * @param input2 Second input array
 * @param output Output array
 * @param size Number of elements
 */
__global__ void
compute_subtract(cf32* input1, cf32* input2, cf32* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] - input2[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise multiplication: output[i] = input1[i] *
 * input2[i]
 * @param input1 First input array
 * @param input2 Second input array
 * @param output Output array
 * @param size Number of elements
 */
__global__ void
compute_multiply(cf32* input1, cf32* input2, cf32* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] * input2[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise division: output[i] = input1[i] /
 * input2[i]
 * @param input1 First input array (dividend)
 * @param input2 Second input array (divisor)
 * @param output Output array
 * @param size Number of elements
 */
__global__ void
compute_divide(cf32* input1, cf32* input2, cf32* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Avoid division by zero
        if (input2[idx] != 0.0f) {
            output[idx] = input1[idx] / input2[idx];
        } else {
            output[idx] = 0.0f; // or could be NaN/Inf based on requirements
        }
    }
}
