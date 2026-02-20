// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#ifndef HAHAHA_MATH_TENSOR_WRAPPER_UTIL_H_1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D
#define HAHAHA_MATH_TENSOR_WRAPPER_UTIL_H_1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D

#include <vector>

#include "TensorWrapper.h"

namespace hahaha::math {

/**
 * @brief Concatenate multiple tensors into a single flattened 1D tensor.
 *
 * This function takes a vector of tensors and concatenates them along
 * the first dimension (flattening each tensor first if needed). All tensors
 * are reshaped to 1D and then concatenated into a single vector.
 *
 * Example:
 * @code
 *   TensorWrapper<float> t1({2, 3}, 1.0f);  // shape [2, 3]
 *   TensorWrapper<float> t2({4}, 2.0f);      // shape [4]
 *   auto result = concatenate({t1, t2});
 *   // result has shape [10] (6 + 4 elements)
 * @endcode
 *
 * @tparam T Numeric type of the tensors.
 * @param tensors Vector of tensors to concatenate.
 * @return TensorWrapper<T> A new 1D tensor containing all concatenated data.
 * @throws std::invalid_argument if tensors is empty.
 */
template <typename T>
TensorWrapper<T> concatenate(const std::vector<TensorWrapper<T>>& tensors) {
    if (tensors.empty()) {
        throw std::invalid_argument("concatenate: tensors vector cannot be empty");
    }

    // Calculate total size
    size_t totalSize = 0;
    for (const auto& tensor : tensors) {
        totalSize += tensor.getTotalSize();
    }

    // Allocate result tensor
    TensorWrapper<T> result(TensorShape({totalSize}), T(0));
    T* resultData = result.getRawData();

    // Copy each tensor's data into the result
    size_t offset = 0;
    for (const auto& tensor : tensors) {
        size_t size = tensor.getTotalSize();

        // Flatten the tensor to 1D using reshape (creates a copy)
        TensorWrapper<T> flatTensor = tensor.reshape({size});
        const T* tensorData = flatTensor.getRawData();

        // Copy data
        std::copy(tensorData, tensorData + size, resultData + offset);
        offset += size;
    }

    return result;
}

/**
 * @brief Distribute a flattened tensor back into multiple individual tensors.
 *
 * This is the inverse operation of concatenate(). It takes a flattened 1D tensor
 * and distributes its data back into the original tensor structures. The function
 * writes data directly to the target tensors' underlying storage.
 *
 * Example:
 * @code
 *   TensorWrapper<float> flat({10}, ...);  // flattened tensor with 10 elements
 *   TensorWrapper<float> t1({2, 3}, 0.0f); // shape [2, 3] - will be updated
 *   TensorWrapper<float> t2({4}, 0.0f);     // shape [4] - will be updated
 *   std::vector<TensorWrapper<float>*> tensors = {&t1, &t2};
 *   std::vector<size_t> sizes = {6, 4};      // sizes must match
 * tensor.getTotalSize() distributeTensors(flat, tensors, sizes);
 *   // t1 and t2 now contain data from flat[0:6] and flat[6:10] respectively
 * @endcode
 *
 * @tparam T Numeric type of the tensors.
 * @param flatTensor The flattened 1D tensor containing all concatenated data.
 * @param tensors Vector of pointers to TensorWrapper objects to update.
 *                These tensors will be modified in-place.
 * @param sizes Vector of sizes for each tensor (must match tensors.size()).
 *              Each size should equal the corresponding tensor's getTotalSize().
 * @throws std::invalid_argument if tensors and sizes have mismatched sizes,
 *         or if the total size doesn't match flatTensor.getTotalSize().
 */
template <typename T>
void distributeTensors(const TensorWrapper<T>& flatTensor,
                       const std::vector<TensorWrapper<T>*>& tensors,
                       const std::vector<size_t>& sizes) {
    if (tensors.size() != sizes.size()) {
        throw std::invalid_argument(
            "distributeTensors: tensors and sizes vectors must have the same size");
    }

    if (tensors.empty()) {
        return; // Nothing to distribute
    }

    // Verify total size matches
    size_t totalSize = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        size_t tensorTotalSize = tensors[i]->getTotalSize();
        if (sizes[i] != tensorTotalSize) {
            throw std::invalid_argument("distributeTensors: size["
                                        + std::to_string(i) + "] ("
                                        + std::to_string(sizes[i])
                                        + ") does not match tensor.getTotalSize() ("
                                        + std::to_string(tensorTotalSize) + ")");
        }
        totalSize += sizes[i];
    }

    if (totalSize != flatTensor.getTotalSize()) {
        throw std::invalid_argument(
            "distributeTensors: total size of tensors (" + std::to_string(totalSize)
            + ") does not match flatTensor.getTotalSize() ("
            + std::to_string(flatTensor.getTotalSize()) + ")");
    }

    const T* flatData = flatTensor.getRawData();
    size_t offset = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto& tensor = *tensors[i];
        size_t tensorTotalSize = tensor.getTotalSize();

        // Get raw data pointer directly from the original tensor
        // (reshape creates a copy, so we need to write to the original)
        T* tensorData = tensor.getRawData();

        // Copy data from flattened tensor back to individual tensor
        // The tensor may be multi-dimensional, but getRawData() gives us
        // a flat view of the underlying storage
        for (size_t j = 0; j < tensorTotalSize; ++j) {
            tensorData[j] = flatData[offset + j];
        }
        offset += sizes[i];
    }
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_UTIL_H_1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D
