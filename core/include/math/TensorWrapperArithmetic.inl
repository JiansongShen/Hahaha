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
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_ARITHMETIC_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_ARITHMETIC_INL

#include <cstddef>
namespace hahaha::math {

template <typename T>
TensorWrapper<T> TensorWrapper<T>::add(const TensorWrapper& other) const {
    checkSameDevice(other);

    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.add(data_.getData()[data_.getOffset()]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return add(other.data_.getData()[other.data_.getOffset()]);
    }

    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        TensorWrapper result;
        result.data_.setShape(data_.getShapeVecRef());
        result.data_.setStride(TensorStride(data_.getShapeVecRef()));
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] = data_.getData()[data_.getOffset()]
            + other.data_.getData()[other.data_.getOffset()];
        return result;
    }

    if (getShapeVecRef() != other.getShapeVecRef()) {
        throw std::invalid_argument("Tensors must have the same shape for addition");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    // If both are contiguous, we can use the fast path (dispatch)
    if (isContiguous() && other.isContiguous()) {
        auto res = backend::dispatchAdd<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path for non-contiguous tensors
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& strideA = data_.getStride().getStrideVec();
        const auto& strideB = other.data_.getStride().getStrideVec();
        const T* ptrA = data_.getData().get();
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
            resPtr[i] = ptrA[offsetA + idxA] + ptrB[offsetB + idxB];

            // Advance coord
            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::subtract(const TensorWrapper& other) const {
    checkSameDevice(other);

    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.subtractFrom(data_.getData()[data_.getOffset()]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return subtract(other.data_.getData()[other.data_.getOffset()]);
    }
    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        TensorWrapper result;
        result.data_.setShape(data_.getShapeVecRef());
        result.data_.setStride(TensorStride(data_.getShapeVecRef()));
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] = data_.getData()[data_.getOffset()]
            - other.data_.getData()[other.data_.getOffset()];
        return result;
    }

    if (getShapeVecRef() != other.getShapeVecRef()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for subtraction");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous() && other.isContiguous()) {
        auto res = backend::dispatchSub<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& strideA = data_.getStride().getStrideVec();
        const auto& strideB = other.data_.getStride().getStrideVec();
        const T* ptrA = data_.getData().get();
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
            resPtr[i] = ptrA[offsetA + idxA] - ptrB[offsetB + idxB];

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::multiply(const TensorWrapper& other) const {
    checkSameDevice(other);
    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.multiply(data_.getData()[data_.getOffset()]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return multiply(other.data_.getData()[other.data_.getOffset()]);
    }
    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        TensorWrapper result;
        result.data_.setShape(data_.getShapeVecRef());
        result.data_.setStride(TensorStride(data_.getShapeVecRef()));
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] = data_.getData()[data_.getOffset()]
            * other.data_.getData()[other.data_.getOffset()];
        return result;
    }

    if (getShapeVecRef() != other.getShapeVecRef()) {
        throw std::invalid_argument(
            "Tensors must have the same shape for multiplication");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous() && other.isContiguous()) {
        auto res = backend::dispatchMul<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& strideA = data_.getStride().getStrideVec();
        const auto& strideB = other.data_.getStride().getStrideVec();
        const T* ptrA = data_.getData().get();
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
            resPtr[i] = ptrA[offsetA + idxA] * ptrB[offsetB + idxB];

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::divide(const TensorWrapper& other) const {
    checkSameDevice(other);

    if (getTotalSize() == 1 && other.getTotalSize() > 1) {
        return other.divideInto(data_.getData()[data_.getOffset()]);
    }
    if (other.getTotalSize() == 1 && getTotalSize() > 1) {
        return divide(other.data_.getData()[other.data_.getOffset()]);
    }
    if (getTotalSize() == 1 && other.getTotalSize() == 1) {
        if (other.data_.getData()[other.data_.getOffset()] == T(0)) {
            throw std::runtime_error("Division by zero");
        }
        TensorWrapper result;
        result.data_.setShape(data_.getShapeVecRef());
        result.data_.setStride(TensorStride(data_.getShapeVecRef()));
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::shared_ptr<T[]>(new T[1]));
        result.data_.getData()[0] = data_.getData()[data_.getOffset()]
            / other.data_.getData()[other.data_.getOffset()];
        return result;
    }

    if (getShapeVecRef() != other.getShapeVecRef()) {
        throw std::invalid_argument("Tensors must have the same shape for division");
    }

    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous() && other.isContiguous()) {
        auto res = backend::dispatchDiv<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& strideA = data_.getStride().getStrideVec();
        const auto& strideB = other.data_.getStride().getStrideVec();
        const T* ptrA = data_.getData().get();
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
            if (ptrB[offsetB + idxB] == T(0)) {
                throw std::runtime_error("Division by zero");
            }
            resPtr[i] = ptrA[offsetA + idxA] / ptrB[offsetB + idxB];

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::add(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous()) {
        auto res = backend::dispatchAdd(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& stride = data_.getStride().getStrideVec();
        const T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            resPtr[i] = ptr[offset + idx] + scalar;

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::subtract(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous()) {
        auto res = backend::dispatchSub(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& stride = data_.getStride().getStrideVec();
        const T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            resPtr[i] = ptr[offset + idx] - scalar;

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::multiply(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous()) {
        auto res = backend::dispatchMul(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& stride = data_.getStride().getStrideVec();
        const T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            resPtr[i] = ptr[offset + idx] * scalar;

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::divide(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous()) {
        auto res = backend::dispatchDiv(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& stride = data_.getStride().getStrideVec();
        const T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            resPtr[i] = ptr[offset + idx] / scalar;

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::subtractFrom(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous()) {
        auto res = backend::dispatchSub(
            data_.getDevice()->getType(), scalar, *this, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& stride = data_.getStride().getStrideVec();
        const T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            resPtr[i] = scalar - ptr[offset + idx];

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::divideInto(T scalar) const {
    TensorWrapper result;
    result.data_.setShape(data_.getShapeVecRef());
    result.data_.setStride(TensorStride(data_.getShapeVecRef()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
        result.to(data_.getDevice());
    }
    result.data_.setDevice(data_.getDevice());

    if (isContiguous()) {
        auto res = backend::dispatchDiv(
            data_.getDevice()->getType(), scalar, *this, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        T* resPtr = result.data_.getData().get();
        const auto& stride = data_.getStride().getStrideVec();
        const T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            if (ptr[offset + idx] == T(0)) {
                throw std::runtime_error("Division by zero");
            }
            resPtr[i] = scalar / ptr[offset + idx];

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }

    return result;
}

template <typename T> void TensorWrapper<T>::squareInPlace() {
    if (isContiguous()) {
        *this *= *this;
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        const auto& stride = data_.getStride().getStrideVec();
        T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            ptr[offset + idx] *= ptr[offset + idx];

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }
}

template <typename T> void TensorWrapper<T>::sqrtInPlace() {
    if (isContiguous()) {
        for (size_t i = 0; i < getTotalSize(); ++i) {
            data_.getData()[data_.getOffset() + i] =
                std::sqrt(data_.getData()[data_.getOffset() + i]);
        }
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        const auto& stride = data_.getStride().getStrideVec();
        T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            ptr[offset + idx] = std::sqrt(ptr[offset + idx]);

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }
}

template <typename T> void TensorWrapper<T>::absInPlace() {
    if (isContiguous()) {
        for (size_t i = 0; i < getTotalSize(); ++i)
            data_.getData()[data_.getOffset() + i] =
                std::abs(data_.getData()[data_.getOffset() + i]);
    } else {
        // Slow path
        const auto& shape = getShapeVecRef();
        std::vector<size_t> coord(shape.size(), 0);
        const auto& stride = data_.getStride().getStrideVec();
        T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            ptr[offset + idx] = std::abs(ptr[offset + idx]);

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }
}

template <typename T> void TensorWrapper<T>::expInPlace() {
    if (isContiguous()) {
        for (size_t i = 0; i < getTotalSize(); ++i)
            data_.getData()[data_.getOffset() + i] =
                std::exp(data_.getData()[data_.getOffset() + i]);
    } else {
        const auto shape = getShapeVecRef();
        std::vector<std::size_t> coord(shape.size(), 0);
        const auto stride = data_.getStride().getStrideVec();
        T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            ptr[offset + idx] = std::exp(ptr[offset + idx]);

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }
}

template <typename T> void TensorWrapper<T>::logInPlace() {
    if (isContiguous()) {
        for (size_t i = 0; i < getTotalSize(); ++i)
            data_.getData()[data_.getOffset() + i] =
                std::log(data_.getData()[data_.getOffset() + i]);
    } else {
        const auto shape = getShapeVecRef();
        std::vector<std::size_t> coord(shape.size(), 0);
        const auto stride = data_.getStride().getStrideVec();
        T* ptr = data_.getData().get();
        size_t offset = data_.getOffset();

        for (size_t i = 0; i < getTotalSize(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                idx += coord[d] * stride[d];
            }
            ptr[offset + idx] = std::log(ptr[offset + idx]);

            for (long d = static_cast<long>(shape.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) {
                    break;
                }
                coord[d] = 0;
            }
        }
    }
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_ARITHMETIC_INL
