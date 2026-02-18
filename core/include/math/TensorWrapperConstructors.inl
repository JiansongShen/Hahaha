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
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_CONSTRUCTORS_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_CONSTRUCTORS_INL

namespace hahaha::math {

template <typename T>
TensorWrapper<T>::TensorWrapper(
    const TensorShape& shape,
    T initValue,
    std::shared_ptr<backend::Device> device)
    : data_(TensorData<T>(TensorShape(shape), initValue, device)) {
}

template <typename T>
TensorWrapper<T>::TensorWrapper(const TensorShape& shape,
                       std::shared_ptr<backend::Device> device)
    : data_(TensorData<T>(TensorShape(shape), T(0), device)) {
}

template <typename T>
TensorWrapper<T>::TensorWrapper(const TensorShape& shape)
    : data_(TensorData<T>(
          TensorShape(shape),
          T(0),
          backend::DeviceRegistry::getInstance().getCPUDevice())) {
}

template <typename T>
TensorWrapper<T>::TensorWrapper(const TensorWrapper& other) : data_(other.data_) {
}

template <typename T>
TensorWrapper<T>::TensorWrapper(TensorWrapper&& other) noexcept
    : data_(std::move(other.data_)) {
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator=(TensorWrapper&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
    }
    return *this;
}

template <typename T>
TensorWrapper<T>::TensorWrapper(NestedData<T>&& data) : data_(std::move(data)) {
}

template <typename T>
TensorWrapper<T>::TensorWrapper(const std::vector<T>& initVec)
    : data_(TensorData<T>(initVec)) {
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_CONSTRUCTORS_INL
