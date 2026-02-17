//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_ACCESSORS_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_ACCESSORS_INL

namespace hahaha::math {

template <typename T>
std::shared_ptr<T[]>& TensorWrapper<T>::getRawData() {
    return data_.getData();
}

template <typename T>
const std::shared_ptr<T[]>& TensorWrapper<T>::getRawData() const {
    return data_.getData();
}

template <typename T>
std::uintptr_t TensorWrapper<T>::getRawGpuPtr() const {
    return data_.gpuPtr;
}

template <typename T>
const std::vector<size_t>& TensorWrapper<T>::getShape() const {
    return data_.getShape().getDims();
}

template <typename T>
size_t TensorWrapper<T>::getTotalSize() const {
    if (data_.getData() == nullptr) {
        return 0;
    }
    return data_.getShape().getTotalSize();
}

template <typename T>
const TensorStride& TensorWrapper<T>::getStride() const {
    return data_.getStride();
}

template <typename T>
void TensorWrapper<T>::setStride(const TensorStride& stride) {
    data_.setStride(stride);
}

template <typename T>
std::shared_ptr<backend::Device> TensorWrapper<T>::getDevice() const {
    return data_.getDevice();
}

template <typename T>
void TensorWrapper<T>::checkSameDevice(const TensorWrapper& other) const {
    if (*getDevice() != *other.getDevice()) {

        throw std::invalid_argument(
            "Tensors must be on the same device for this operation (found "
            + getDevice()->toString() + " and "
            + other.getDevice()->toString() + ")");
    }
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_ACCESSORS_INL
