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
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_OPERATORS_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_OPERATORS_INL

namespace hahaha::math {

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator+(const TensorWrapper& other) const {
    return add(other);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator-(const TensorWrapper& other) const {
    return subtract(other);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator*(const TensorWrapper& other) const {
    return multiply(other);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator/(const TensorWrapper& other) const {
    return divide(other);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator+(T scalar) const {
    return add(scalar);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator-(T scalar) const {
    return subtract(scalar);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator*(T scalar) const {
    return multiply(scalar);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator/(T scalar) const {
    return divide(scalar);
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::operator-() const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    const size_t tensorSize = getTotalSize();
    result.data_.setData(std::shared_ptr<T[]>(new T[tensorSize]));
    result.data_.setDevice(data_.getDevice());
    for (size_t i = 0; i < tensorSize; ++i) {
        result.data_.getData()[i] = -data_.getData()[i];
    }
    return result;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator+=(const TensorWrapper& other) {
    if (other.getTotalSize() == 1) {
        return *this += other.data_.getData()[0];
    }
    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for addition");
    }

    checkSameDevice(other);

    const size_t tensorSize = getTotalSize();

    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] = data_.getData()[i] + other.data_.getData()[i];
    }

    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator-=(const TensorWrapper& other) {
    if (other.getTotalSize() == 1) {
        return *this -= other.data_.getData()[0];
    }
    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for subtraction");
    }

    checkSameDevice(other);

    const size_t tensorSize = getTotalSize();

    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] = data_.getData()[i] - other.data_.getData()[i];
    }

    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator*=(const TensorWrapper& other) {
    if (other.getTotalSize() == 1) {
        return *this *= other.data_.getData()[0];
    }
    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for multiplication");
    }

    checkSameDevice(other);

    const size_t tensorSize = getTotalSize();

    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] = data_.getData()[i] * other.data_.getData()[i];
    }

    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator/=(const TensorWrapper& other) {
    if (other.getTotalSize() == 1) {
        return *this /= other.data_.getData()[0];
    }
    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for division");
    }

    checkSameDevice(other);

    const size_t tensorSize = getTotalSize();

    for (size_t i = 0; i < tensorSize; ++i) {
        if (other.data_.getData()[i] == T(0)) {
            throw std::runtime_error("Division by zero");
        }
        data_.getData()[i] = data_.getData()[i] / other.data_.getData()[i];
    }

    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator+=(T scalar) {
    const size_t tensorSize = getTotalSize();
    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] += scalar;
    }
    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator-=(T scalar) {
    const size_t tensorSize = getTotalSize();
    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] -= scalar;
    }
    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator*=(T scalar) {
    const size_t tensorSize = getTotalSize();
    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] *= scalar;
    }
    return *this;
}

template <typename T>
TensorWrapper<T>& TensorWrapper<T>::operator/=(T scalar) {
    if (scalar == T(0)) {
        throw std::runtime_error("Division by zero");
    }
    const size_t tensorSize = getTotalSize();
    for (size_t i = 0; i < tensorSize; ++i) {
        data_.getData()[i] /= scalar;
    }
    return *this;
}

// Global operators
template <typename T>
TensorWrapper<T> operator+(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.add(scalar);
}

template <typename T>
TensorWrapper<T> operator-(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.subtractFrom(scalar);
}

template <typename T>
TensorWrapper<T> operator*(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.multiply(scalar);
}

template <typename T>
TensorWrapper<T> operator/(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.divideInto(scalar);
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_OPERATORS_INL

