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
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_SHAPE_OPS_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_SHAPE_OPS_INL

#include <algorithm>
#include <numeric>

namespace hahaha::math {

template <typename T>
TensorWrapper<T> TensorWrapper<T>::reshape(const std::vector<size_t>& newShape) const {
    if (!isContiguous()) {
        throw std::invalid_argument("reshape is not supported for non-contiguous tensors. Call clone() first.");
    }

    const size_t totalSize = std::accumulate(
        newShape.begin(), newShape.end(), 1ULL, std::multiplies());
    if (totalSize != getTotalSize()) {
        throw std::invalid_argument("New shape total size ("
                                    + std::to_string(totalSize)
                                    + ") must match current size ("
                                    + std::to_string(getTotalSize()) + ")");
    }

    TensorWrapper result;
    result.data_.setShape(TensorShape(newShape));
    result.data_.setStride(TensorStride(result.data_.getShape()));

    size_t currentSize = getTotalSize();
    result.data_.setData(std::shared_ptr<T[]>(new T[currentSize]));
    result.data_.setDevice(data_.getDevice());
    
    std::copy(data_.getData().get() + data_.getOffset(),
              data_.getData().get() + data_.getOffset() + currentSize,
              result.data_.getData().get());

    return result;
}

template <typename T>
size_t TensorWrapper<T>::getDimensions() const {
    return data_.getShape().getDims().size();
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::broadcastTo(const TensorShape& newShape) {
    auto broadcasted =
        TensorShape::broadcastShape(this->data_.getShape(), newShape);
    if (!broadcasted.has_value() || TensorShape(*broadcasted) != newShape) {
        throw std::invalid_argument("Cannot broadcast shape "
                                    + this->data_.getShape().toString()
                                    + " to " + newShape.toString());
    }

    TensorWrapper result;
    result.data_ = this->data_.share();
    if (newShape.getDims().size() < this->getShape().size()) {
        throw std::invalid_argument(
            "invalid argument of broadcastTo(), the target shape is "
            + newShape.toString() + " but current shape is "
            + this->data_.getShape().toString());
    }

    TensorStride newStride = this->data_.getStride();
    auto shapeDiff = newShape.getDims().size() - this->getShape().size();
    newStride.getStrideVec().insert(
        newStride.getStrideVec().begin(), shapeDiff, 0);

    long newShapeIdx = static_cast<long>(newShape.getDims().size() - 1);
    long selfShapeIdx = static_cast<long>(this->getShape().size() - 1);

    while (selfShapeIdx >= 0) {
        if (newShape.getDims()[newShapeIdx] != 1
            && getShape()[selfShapeIdx] == 1) {
            newStride.getStrideVec()[newShapeIdx] = 0;
        }

        --newShapeIdx;
        --selfShapeIdx;
    }

    result.data_.setShape(newShape);
    result.data_.setStride(newStride);
    result.data_.setDevice(data_.getDevice());

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::clone() const {
    TensorWrapper result;
    result.data_.setShape(data_.getShape());
    result.data_.setStride(TensorStride(data_.getShape())); // Default contiguous stride
    result.data_.setDevice(data_.getDevice());
    result.data_.setData(std::make_shared<T[]>(getTotalSize()));
    
    if (isContiguous()) {
        // Fast path for contiguous tensors
        std::copy(data_.getData().get() + data_.getOffset(),
                  data_.getData().get() + data_.getOffset() + getTotalSize(),
                  result.data_.getData().get());
    } else {
        // Slow path for non-contiguous views
        const auto& shape = getShape();
        size_t dims = shape.size();
        std::vector<size_t> coords(dims, 0);
        
        size_t totalSize = getTotalSize();
        T* resPtr = result.data_.getData().get();
        const auto& srcStrides = data_.getStride().getStrideVec();
        
        for (size_t i = 0; i < totalSize; ++i) {
            // Reimplement access logic:
            size_t srcLinearIdx = 0;
            for(size_t d=0; d<dims; ++d) {
                srcLinearIdx += coords[d] * srcStrides[d];
            }
            resPtr[i] = data_.getData()[data_.getOffset() + srcLinearIdx];
            
            // Increment coords (odometer)
            for (long d = static_cast<long>(dims) - 1; d >= 0; --d) {
                coords[d]++;
                if (coords[d] < shape[d]) {
                    break;
                }
                coords[d] = 0;
            }
        }
    }
    return result;
}

template <typename T>
void TensorWrapper<T>::clear() {
    if (data_.getData() == nullptr) {
        return;
    }
    // If contiguous, we can memset/fill the raw buffer.
    if (isContiguous()) {
        const auto totalSize = getTotalSize();
        std::fill(data_.getData().get() + data_.getOffset(),
                  data_.getData().get() + data_.getOffset() + totalSize,
                  T());
    } else {
        // Non-contiguous: must iterate
        const auto& shape = getShape();
        size_t dims = shape.size();
        std::vector<size_t> coords(dims, 0);
        size_t totalSize = getTotalSize();
        const auto& srcStrides = data_.getStride().getStrideVec();
        
        for (size_t i = 0; i < totalSize; ++i) {
            size_t srcLinearIdx = 0;
            for(size_t d=0; d<dims; ++d) {
                srcLinearIdx += coords[d] * srcStrides[d];
            }
            data_.getData()[data_.getOffset() + srcLinearIdx] = T();
            
            for (long d = static_cast<long>(dims) - 1; d >= 0; --d) {
                coords[d]++;
                if (coords[d] < shape[d]) {
                    break;
                }
                coords[d] = 0;
            }
        }
    }
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_SHAPE_OPS_INL
