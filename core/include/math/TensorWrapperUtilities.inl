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
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_UTILITIES_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_UTILITIES_INL

namespace hahaha::math {

template <typename T>
void TensorWrapper<T>::to(std::shared_ptr<backend::Device> targetDevice) {
    if (*data_.getDevice() == *targetDevice) {
        return;
    }

    // Logic for moving data between devices
    if (targetDevice->getType() == backend::DeviceType::CPU) {
        if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
            // GPU to CPU transfer: copy data back to CPU
            const size_t totalSize = getTotalSize();
            if (totalSize == 0) {
                data_.setDevice(targetDevice);
                return;
            }

            // Create new CPU data
            auto newData = std::make_shared<T[]>(totalSize);

            data_.copyOrMoveToDevice(targetDevice);
            data_.setDevice(targetDevice);
            return;
        }
    } else if (targetDevice->getType() == backend::DeviceType::CUDA
               || data_.getDevice()->getType()
                   == backend::DeviceType::CPU) {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
        // CPU to GPU transfer: copy data to GPU
        if (const size_t totalSize = getTotalSize(); totalSize == 0) {
            data_.setDevice(targetDevice);
            return;
        }

        data_.copyOrMoveToDevice(targetDevice);
        data_.setDevice(targetDevice);
        return;
#else
        throw std::runtime_error(
            "CUDA headers not available. Cannot move tensor to GPU.");
#endif
#else
        throw std::runtime_error(
            "CUDA not enabled. Cannot move tensor to GPU.");
#endif
    }

    data_.setDevice(targetDevice);
}

template <typename T>
T& TensorWrapper<T>::at(const std::initializer_list<size_t>& indices) {
    const auto& shapeDims = data_.getShape().getDims();
    if (indices.size() != shapeDims.size()) {
        throw std::out_of_range("Dimension mismatch: expected "
                                + std::to_string(shapeDims.size())
                                + " indices, got "
                                + std::to_string(indices.size()));
    }

    size_t linearIdx = 0;
    const auto* idxIt = indices.begin();
    const auto& strideDims = data_.getStride().getStrideVec();

    auto dimsSize = shapeDims.size();
    for (size_t i = 0; i < dimsSize; ++i) {
        size_t dimIdx = *idxIt;
        if (dimIdx >= shapeDims[i]) {
            throw std::out_of_range("Index out of bounds at dimension "
                                    + std::to_string(i));
        }
        linearIdx += dimIdx * strideDims[i];
        std::advance(idxIt, 1);
    }
    // Add offset for views
    return data_.getData()[data_.getOffset() + linearIdx];
}

template <typename T>
const T& TensorWrapper<T>::at(const std::initializer_list<size_t>& indices) const {
    const auto& shapeDims = data_.getShape().getDims();
    if (indices.size() != shapeDims.size()) {
        throw std::out_of_range("Dimension mismatch");
    }

    size_t linearIdx = 0;
    const auto* idxIt = indices.begin();
    const auto& strideDims = data_.getStride().getStrideVec();

    for (size_t i = 0; i < shapeDims.size(); ++i) {
        size_t dimIdx = *idxIt;
        if (dimIdx >= shapeDims[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        linearIdx += dimIdx * strideDims[i];
        std::advance(idxIt, 1);
    }
    // Add offset for views
    return data_.getData()[data_.getOffset() + linearIdx];
}

template <typename T>
void TensorWrapper<T>::axpy(T alpha, const TensorWrapper& other) {
    if (getShape() != other.getShape()) {
        throw std::invalid_argument("Shape mismatch in axpy");
    }
    checkSameDevice(other);

    if (isContiguous() && other.isContiguous()) {
        // Dispatch to backend for hardware-specific optimization
        auto res = backend::dispatchAxpy(
            data_.getDevice()->getType(), alpha, other, *this);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path for non-contiguous tensors
        const auto& shape = getShape();
        std::vector<size_t> coord(shape.size(), 0);
        const auto& strideA = data_.getStride().getStrideVec();
        const auto& strideB = other.data_.getStride().getStrideVec();
        T* ptrA = data_.getData().get();
        const T* ptrB = other.data_.getData().get();
        size_t offsetA = data_.getOffset();
        size_t offsetB = other.data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idxA = 0;
            size_t idxB = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idxA += coord[d] * strideA[d];
                idxB += coord[d] * strideB[d];
            }
            ptrA[offsetA + idxA] += alpha * ptrB[offsetB + idxB];
            
            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::ones() const {
    TensorWrapper res(TensorShape(this->getShape()), T(1), this->getDevice());
    return res;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::zeros() const {
    TensorWrapper res(TensorShape(this->getShape()), T(0), this->getDevice());
    return res;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::sameShapeWithValue(T initValue) const {
    TensorWrapper res(
        TensorShape(this->getShape()), T(initValue), this->getDevice());
    return res;
}

template <typename T>
bool TensorWrapper<T>::isContiguous() const {
    if (data_.getOffset() != 0) {
        return false;
    }

    TensorStride defaultStride(data_.getShape());
    const auto& currentStrides = data_.getStride().getStrideVec();
    const auto& defaultStrides = defaultStride.getStrideVec();

    if (currentStrides.size() != defaultStrides.size()) {
        return false;
    }

    for (size_t i = 0; i < currentStrides.size(); ++i) {
        if (currentStrides[i] != defaultStrides[i]) {
            return false;
        }
    }
    return true;
}

// slice implementation is now in TensorWrapper.h (inline)

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_UTILITIES_INL
