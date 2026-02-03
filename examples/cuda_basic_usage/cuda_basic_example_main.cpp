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

#include <chrono>
#include <iostream>
#include <vector>

#include "backend/Memory.h"
#include "backend/cpu/CPUDevice.h"
#include "backend/gpu/cuda/CudaDevice.h"
#include "common/Config.h"
#include "common/macros.h"
#include "math/ds/TensorData.h"

using namespace hahaha;

int main() {
    std::cout << "CUDA Basic Usage Example" << std::endl;

#ifdef HAHAHA_USE_CUDA
    std::cout << "CUDA is enabled!" << std::endl;

    // Create CUDA device
    backend::CudaDevice cuda_device;

    // Create sample data
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Allocate memory on CUDA device
    auto device_buffer =
        cuda_device.memory().allocate(sizeof(float) * host_data.size());

    if (device_buffer.address() != 0) {
        std::cout << "Successfully allocated memory on CUDA device"
                  << std::endl;

        // Copy data to device
        cuda_device.memory().copyHostToDevice(
            device_buffer,
            std::span<char>(reinterpret_cast<const char*>(host_data.data()),
                            sizeof(float) * host_data.size()));

        std::cout << "Successfully copied data to CUDA device" << std::endl;

        // Test memory set operation
        cuda_device.memory().memset(
            device_buffer, 0, sizeof(float) * host_data.size());
        std::cout << "Successfully performed memset on CUDA device"
                  << std::endl;

        // Read data back to host
        std::vector<float> read_back_data(host_data.size());
        cuda_device.memory().copyDeviceToHost(
            std::span<char>(reinterpret_cast<char*>(read_back_data.data()),
                            sizeof(float) * host_data.size()),
            device_buffer);

        std::cout << "Successfully copied data back from CUDA device"
                  << std::endl;

        // Free device memory
        cuda_device.memory().free(device_buffer);
    } else {
        std::cout << "Failed to allocate memory on CUDA device" << std::endl;
    }

#else
    std::cout << "CUDA is not enabled in this build" << std::endl;
#endif

    return 0;
}
