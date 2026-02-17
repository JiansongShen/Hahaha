#ifndef HAHAHA_MATH_TENSOR_WRAPPER_CORE_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_CORE_INL

#include <stdexcept>
#include <string>

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
    const auto& strideDims = data_.getStride().getStrides();

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
    return data_.getData()[linearIdx];
}

template <typename T>
const T& TensorWrapper<T>::at(const std::initializer_list<size_t>& indices) const {
    const auto& shapeDims = data_.getShape().getDims();
    if (indices.size() != shapeDims.size()) {
        throw std::out_of_range("Dimension mismatch");
    }

    size_t linearIdx = 0;
    const auto* idxIt = indices.begin();
    const auto& strideDims = data_.getStride().getStrides();

    for (size_t i = 0; i < shapeDims.size(); ++i) {
        size_t dimIdx = *idxIt;
        if (dimIdx >= shapeDims[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        linearIdx += dimIdx * strideDims[i];
        std::advance(idxIt, 1);
    }
    return data_.getData()[linearIdx];
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

#endif // HAHAHA_MATH_TENSOR_WRAPPER_CORE_INL
