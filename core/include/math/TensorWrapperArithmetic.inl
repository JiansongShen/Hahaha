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
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_ARITHMETIC_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_ARITHMETIC_INL

namespace hahaha::math {

template <typename T>
TensorWrapper<T> TensorWrapper<T>::add(const TensorWrapper& other) const {
    checkSameDevice(other);

    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.add(data_.getData()[0]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return add(other.data_.getData()[0]);
    }

    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] =
            data_.getData()[0] + other.data_.getData()[0];
        return result;
    }

    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for addition");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());
    auto res = backend::dispatchAdd<T>(
        data_.getDevice()->getType(), *this, other, result);

    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::subtract(const TensorWrapper& other) const {
    checkSameDevice(other);

    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.subtractFrom(data_.getData()[0]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return subtract(other.data_.getData()[0]);
    }
    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] =
            data_.getData()[0] - other.data_.getData()[0];
        return result;
    }

    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for subtraction");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchSub<T>(
        data_.getDevice()->getType(), *this, other, result);

    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::multiply(const TensorWrapper& other) const {
    checkSameDevice(other);
    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.multiply(data_.getData()[0]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return multiply(other.data_.getData()[0]);
    }
    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] =
            data_.getData()[0] * other.data_.getData()[0];
        return result;
    }

    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for multiplication");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchMul<T>(
        data_.getDevice()->getType(), *this, other, result);

    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::divide(const TensorWrapper& other) const {
    checkSameDevice(other);

    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.divideInto(data_.getData()[0]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return divide(other.data_.getData()[0]);
    }
    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        if (other.data_.getData()[0] == T(0)) {
            throw std::runtime_error("Division by zero");
        }
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] =
            data_.getData()[0] / other.data_.getData()[0];
        return result;
    }

    if (getShape() != other.getShape()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for division");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchDiv<T>(
        data_.getDevice()->getType(), *this, other, result);

    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::add(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchAdd(
        data_.getDevice()->getType(), *this, scalar, result);
    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::subtract(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchSub(
        data_.getDevice()->getType(), *this, scalar, result);
    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::multiply(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchMul(
        data_.getDevice()->getType(), *this, scalar, result);
    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::divide(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchDiv(
        data_.getDevice()->getType(), *this, scalar, result);
    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::subtractFrom(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchSub(
        data_.getDevice()->getType(), scalar, *this, result);
    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::divideInto(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(data_.getStride());
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    auto res = backend::dispatchDiv(
        data_.getDevice()->getType(), scalar, *this, result);
    if (!res) {
        throw std::runtime_error(res.error().message());
    }

    return result;
}

template <typename T>
void TensorWrapper<T>::squareInPlace() {
    *this *= *this;
}

template <typename T>
void TensorWrapper<T>::sqrtInPlace() {
    for (size_t i = 0; i < getTotalSize(); ++i) {
        data_.getData()[i] = std::sqrt(data_.getData()[i]);
    }
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_ARITHMETIC_INL

