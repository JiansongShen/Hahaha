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
#ifndef HAHAHA_MATH_TENSOR_WRAPPER_MATRIX_OPS_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_MATRIX_OPS_INL

#include <algorithm>
#include <ranges>

namespace hahaha::math {

template <typename T>
TensorWrapper<T> TensorWrapper<T>::matmul(const TensorWrapper& other) const {
    if (getDimensions() != 2 || other.getDimensions() != 2) {
        throw std::invalid_argument(
            "matmul is only implemented for 2D tensors");
    }

    if (!isContiguous() || !other.isContiguous()) {
        throw std::invalid_argument("matmul is not supported for non-contiguous tensors. Call clone() first.");
    }

    checkSameDevice(other);

    const auto& thisDims = data_.getShape().getDims();
    const auto& otherDims = other.data_.getShape().getDims();

    if (thisDims[1] != otherDims[0]) {
        throw std::invalid_argument(
            "Matrix dimensions mismatch for matmul: ("
            + std::to_string(thisDims[0]) + "x"
            + std::to_string(thisDims[1]) + ") and ("
            + std::to_string(otherDims[0]) + "x"
            + std::to_string(otherDims[1]) + ")");
    }

    size_t rows = thisDims[0];
    size_t cols = otherDims[1];

    TensorWrapper result;
    result.data_.setShape(TensorShape({rows, cols}));
    result.data_.setStride(TensorStride(result.data_.getShape()));
    result.data_.setDevice(data_.getDevice());
    result.data_.setData(std::shared_ptr<T[]>(new T[rows * cols]));
    // Initialize result with zeros as matmul accumulates
    std::fill(result.data_.getData().get(), result.data_.getData().get() + rows * cols, T(0));


    auto result_val = backend::dispatchMatMul(
        data_.getDevice()->getType(), *this, other, result);
    if (!result_val) {
        throw std::runtime_error(result_val.error().message());
    }

    return result;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::transpose() const {
    if (getDimensions() != 2) {
        throw std::invalid_argument(
            "transpose is only implemented for 2D tensors for now");
    }

    if (!isContiguous()) {
        throw std::invalid_argument("transpose is not supported for non-contiguous tensors. Call clone() first.");
    }

    const auto& shapeDims = data_.getShape().getDims();
    size_t rows = shapeDims[0];
    size_t cols = shapeDims[1];

    TensorWrapper result;
    result.data_.setShape(TensorShape({cols, rows}));
    result.data_.setStride(TensorStride(result.data_.getShape()));
    result.data_.setData(std::shared_ptr<T[]>(new T[getTotalSize()]));
    result.data_.setDevice(data_.getDevice());
    
    // Since we checked isContiguous, we can use raw pointers with offset
    const T* src = data_.getData().get() + data_.getOffset();
    T* dst = result.data_.getData().get();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }

    return result;
}

template <typename T>
T TensorWrapper<T>::sum() const {
    T result = T(0);
    const auto totalSize = getTotalSize();
    
    if (isContiguous()) {
        const T* src = data_.getData().get() + data_.getOffset();
        for (size_t i = 0; i < totalSize; ++i) {
            result += src[i];
        }
    } else {
        // Slow path for non-contiguous
        const auto& shape = getShape();
        size_t dims = shape.size();
        std::vector<size_t> coords(dims, 0);
        
        for (size_t i = 0; i < totalSize; ++i) {
            size_t srcLinearIdx = 0;
            const auto& srcStrides = data_.getStride().getStrideVec();
            for(size_t d=0; d<dims; ++d) {
                srcLinearIdx += coords[d] * srcStrides[d];
            }
            result += data_.getData()[data_.getOffset() + srcLinearIdx];
            
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
TensorWrapper<T> TensorWrapper<T>::sum(std::vector<size_t> axes,
                      const bool keepDims) const {
    if (axes.empty()) {
        return this->clone();
    }
    
    if (!isContiguous()) {
        throw std::invalid_argument("sum(axes) is not supported for non-contiguous tensors. Call clone() first.");
    }

    std::ranges::sort(axes);
    axes.erase(std::ranges::unique(axes).begin(), axes.end());

    // 1. get target shape
    std::vector<bool> isReduced;
    const std::vector<size_t>& srcShape = getShape();
    std::vector<size_t> resShape;

    isReduced.resize(srcShape.size(), false);
    for (const unsigned long axe : axes) {
        if (static_cast<long>(axe)
            > static_cast<long>(isReduced.size()) - 1) {
            throw std::invalid_argument("axis is too big!");
        }
        isReduced[axe] = true;
    }

    if (axes.size() == this->getShape().size()) {
        TensorWrapper result;
        result.data_.setShape(TensorShape({}));
        result.data_.setStride(TensorStride(result.data_.getShape()));
        result.data_.setData(std::make_shared<T[]>(1));
        result.data_.setDevice(data_.getDevice());
        result.getRawData().get()[0] = this->sum();
        return result;
    }

    if (this->getShape().size() == 0) {
        return this->clone();
    }

    for (size_t i = 0; i < srcShape.size(); ++i) {
        // if sum for this dim, then remove it or set 1 in resShape
        if (isReduced[i]) {
            if (keepDims) {
                resShape.push_back(1);
            }
        } else {
            // if not then just add dim
            resShape.push_back(srcShape[i]);
        }
    }

    // 2. calculate necessary datas
    // need to calculate
    // a. how many should dstIdx reduce when it needs to reduce
    //      when coord carries at a position(current value on position is a)
    //          if this pos is not reduced, then dstIdx should minus
    //              a * correspondStride
    //          if this pos is reduced, then do nothing,
    // b. how many value should dstIdx add when it needs to add
    //      when coord increases at a position,
    //          if this pos is not reduced, then dstIdx should add a value
    //              equals to the stride
    //          if this pos is reduced, then dstIdx will add nothing
    std::vector<size_t> resStride(srcShape.size(), 0);
    TensorWrapper result((TensorShape(resShape)));
    auto resultStride = result.getStride().getStrideVec();
    size_t resultStrideIdx = 0;
    for (size_t i = 0; i < srcShape.size(); ++i) {
        if (isReduced[i]) {
            if (keepDims) {
                resultStrideIdx++;
            }
        } else {
            resStride[i] = resultStride[resultStrideIdx++];
        }
    }

    // 3. calculate data to result
    const std::shared_ptr<T[]> srcPtr = getRawData();
    std::shared_ptr<T[]> resPtr = result.getRawData();
    // Initialize result with zeros
    std::fill(resPtr.get(), resPtr.get() + result.getTotalSize(), T(0));
    
    std::vector<size_t> coord(srcShape.size(), 0);
    size_t dstIdx = 0;

    // Use offset for srcPtr
    const T* rawSrc = srcPtr.get() + data_.getOffset();

    for (size_t srcIdx = 0; srcIdx < getTotalSize(); ++srcIdx) {
        resPtr[dstIdx] += rawSrc[srcIdx];

        for (long i = static_cast<long>(coord.size() - 1); i >= 0; --i) {
            ++coord[i];
            // need to carry, next value on position will add one
            if (coord[i] == srcShape[i]) {
                coord[i] = 0;
                dstIdx -= resStride[i] * (srcShape[i] - 1);
                continue;
            }
            dstIdx += resStride[i];
            break;
        }
    }

    return result;
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_MATRIX_OPS_INL
