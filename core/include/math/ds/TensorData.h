// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_MATH_DS_TENSOR_DATA_H
#define HAHAHA_MATH_DS_TENSOR_DATA_H

#include <expected>
#include <memory>
#include <stdexcept>
#include <utility>

#include "backend/DeviceRegistry.h"
#include "backend/cpu/CPUDevice.h"
#include "backend/gpu/cuda/cuda_compute_fun.h"
#include "common/errors/Error.h"
#include "math/ds/NestedData.h"
#include "math/ds/TensorShape.h"
#include "math/ds/TensorStride.h"

namespace hahaha::common {
    struct Error;
}

namespace hahaha::math {

template <typename T> class TensorWrapper;

/**
 * @brief Internal storage class for tensor data and metadata.
 *
 * TensorData manages the raw memory allocation, the shape of the tensor,
 * and the memory strides. It uses std::shared_ptr for automatic memory
 * management on the CPU. For GPU, a separate memory management strategy
 * would be needed.
 *
 * This class is designed to be wrapped by TensorWrapper, which provides
 * the high-level API.
 *
 * @tparam T The numeric type of the tensor elements.
 */
template <typename T> class TensorData {
  public:
    /**
     * @brief Default constructor for an empty storage.
     */
    TensorData() = default;

    /**
     * @brief Construct with given shape and initial value on a specific device.
     * @param shape The shape of the tensor.
     * @param initValue Initial value for all elements.
     * @param device The device where the data should reside.
     */
    TensorData(const TensorShape& shape,
               T initValue,
               const std::shared_ptr<backend::Device>& device =
                   backend::getCPUDevice())
        : shape_(shape), stride_(shape), device_(device) {
        size_t size = shape_.getTotalSize();
        if (device_->getType() == backend::DeviceType::CPU) {
            data_ = std::make_shared<T[]>(size);
            std::fill(data_.get(), data_.get() + size, initValue);
        } else {
            // TODO: Handle GPU allocation using compute::gpu::GpuMemory
            throw std::runtime_error(
                "GPU allocation not yet implemented in TensorData");
        }
    }

    /**
     * @brief Construct with given shape and initial value on a specific device.
     * @param shape The shape of the tensor.
     * @param device The device where the data should reside.
     */
    explicit TensorData(const TensorShape& shape,
                        const std::shared_ptr<backend::Device>& device =
                            backend::getCPUDevice())
        : shape_(shape), stride_(shape), device_(device) {
        const size_t size = shape_.getTotalSize();
        if (device_->getType() == backend::DeviceType::CPU) {
            data_ = std::make_shared<T[]>(size);
        } else {
            // TODO: Handle GPU allocation using compute::gpu::GpuMemory
            throw std::runtime_error(
                "GPU allocation not yet implemented in TensorData");
        }
    }
    /**
     * @brief Copy constructor. Performs a deep copy of the underlying array.
     * @param other The TensorData to copy from.
     */
    TensorData(const TensorData& other)
        : shape_(other.shape_), stride_(other.stride_), device_(other.device_) {
        size_t size = shape_.getTotalSize();
        if (device_->getType() == backend::DeviceType::CPU) {
            data_ = std::make_shared<T[]>(size);
            std::copy(other.data_.get(), other.data_.get() + size, data_.get());
        } else {
            // TODO: Handle GPU deep copy
            throw std::runtime_error(
                "GPU deep copy not yet implemented in TensorData");
        }
    }

    /**
     * @brief Move constructor. Efficiently transfers ownership of the array.
     * @param other The source TensorData to move from.
     */
    TensorData(TensorData&& other) noexcept
        : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
          stride_(std::move(other.stride_)), device_(std::move(other.device_)) {
    }

    /**
     * @brief Construct from a vector of values.
     * @param initVec Vector of values to initialize the tensor with. Creates
     *                a 1D tensor with the same size as the vector.
     */
    explicit TensorData(const std::vector<T>& initVec)
        : data_(std::make_shared<T[]>(initVec.size())),
          shape_(TensorShape(std::vector<size_t>{initVec.size()})) {
        stride_ = TensorStride(shape_);
        std::copy(initVec.begin(), initVec.end(), data_.get());
        device_ = backend::getCPUDevice();
    }

    /**
     * @brief Copy assignment is deleted. Use constructor for copying.
     */
    TensorData& operator=(const TensorData&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source TensorData to move from.
     * @return Reference to this.
     */
    TensorData& operator=(TensorData&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = std::move(other.shape_);
            stride_ = std::move(other.stride_);
            device_ = other.device_;
        }
        return *this;
    }

    /**
     * @brief Create a shared view of the tensor data.
     *
     * This creates a new TensorData instance that shares the underlying
     * data array with the current instance. Both instances will point to
     * the same memory.
     *
     * @return TensorData A new TensorData instance sharing the same data.
     */
    TensorData share() const {
        TensorData sharedData = TensorData();
        sharedData.shape_ = shape_;
        sharedData.stride_ = stride_;
        sharedData.device_ = device_;
        sharedData.data_ = data_;
        sharedData.offset_ = offset_;
        return sharedData;
    }

    /**
     * @brief Destructor. Automatically releases the shared_ptr.
     */
    ~TensorData() {
        if (device_ && device_->getType() == backend::DeviceType::CUDA) {
            if (gpuPtr) {
                device_->deallocate(backend::DeviceBuffer(
                    gpuPtr, sizeof(T) * shape_.getTotalSize()));
            }
        }
    };

    /**
     * @brief Construct from flattened nested data (e.g., from initializer
     * lists).
     * @param data The NestedData object containing flattened data and shape.
     */
    explicit TensorData(NestedData<T>&& data) : shape_(data.getShape()) {
        if (const size_t size = data.getFlatData().size(); size > 0) {
            data_ = std::make_shared<T[]>(size);
            std::copy(data.getFlatData().begin(),
                      data.getFlatData().end(),
                      data_.get());
        } else {
            data_ = nullptr; // Explicitly null for truly empty tensors
        }
        stride_ = TensorStride(shape_);
        device_ = backend::getCPUDevice();
    }

    std::expected<void, common::Error>
    copyFromCpuToCuda(const std::shared_ptr<backend::Device>& targetDevice) {
        const auto byteSize = sizeof(T) * shape_.getTotalSize();
        const auto targetBuffer = targetDevice->allocate(byteSize);
        if (targetBuffer.address() == 0) {
            return std::unexpected(common::CudaDeviceOutOfMemoryError());
        }

        targetDevice->copyMemoryToThis(
            std::span(reinterpret_cast<std::byte*>(this->data_.get()),
                      byteSize),
            std::span(reinterpret_cast<std::byte*>(targetBuffer.address()),
                      byteSize),
            device_);

        gpuPtr = targetBuffer.address();

        return {};
    }

    std::expected<void, common::Error>
    moveFromCudaToCpu(const std::shared_ptr<backend::Device>& targetDevice) {
        auto byteSize = sizeof(T) * shape_.getTotalSize();
        const auto targetBuffer = backend::DeviceBuffer(
            reinterpret_cast<uintptr_t>(this->getData().get()), byteSize);
        if (targetBuffer.address() == 0) {
            return std::unexpected(common::CudaDeviceOutOfMemoryError());
        }

        device_->copyMemoryFromThis(
            std::span(reinterpret_cast<std::byte*>(this->gpuPtr), byteSize),
            std::span(reinterpret_cast<std::byte*>(targetBuffer.address()),
                      byteSize),
            targetDevice);

        device_->deallocate(backend::DeviceBuffer(gpuPtr, byteSize));
        gpuPtr = 0;

        return {};
    }

    std::expected<void, common::Error>
    copyOrMoveToDevice(std::shared_ptr<backend::Device> targetDevice) {
        if (*this->device_ == *targetDevice) {
            return {};
        }

        if (this->device_->getType() == backend::DeviceType::CPU
            && targetDevice->getType() == backend::DeviceType::CUDA) {
            return copyFromCpuToCuda(targetDevice);
        }

        if (this->device_->getType() == backend::DeviceType::CUDA
            && targetDevice->getType() == backend::DeviceType::CPU) {
            return moveFromCudaToCpu(targetDevice);
        }

        return {};
    }

    /**
     * @brief Get the raw data pointer.
     * @return Reference to the shared_ptr holding the data.
     */
    std::shared_ptr<T[]>& getData() {
        return data_;
    }

    /**
     * @brief Const version of data pointer access.
     * @return Const reference to the shared_ptr.
     */
    const std::shared_ptr<T[]>& getData() const {
        return data_;
    }

    /**
     * @brief Replace the current data array.
     * @param data New data array as shared_ptr.
     */
    void setData(std::shared_ptr<T[]> data) {
        data_ = std::move(data);
    }

    /**
     * @brief Access element at the given flat index.
     * @param idx The linear index into the flattened data array.
     * @return T& Reference to the element at the given index.
     */
    T& operator[](size_t idx) const {
        return data_[idx];
    }

    /**
     * @brief Get the tensor shape.
     * @return Const reference to the shape.
     */
    [[nodiscard]] const TensorShape& getShape() const {
        return shape_;
    }

    /**
     * @brief Update the tensor shape.
     * @param shape New shape.
     */
    void setShape(const TensorShape& shape) {
        shape_ = shape;
    }

    /**
     * @brief Get the memory strides.
     * @return Const reference to the strides.
     */
    [[nodiscard]] const TensorStride& getStride() const {
        return stride_;
    }

    /**
     * @brief Update the memory strides.
     * @param stride New strides.
     */
    void setStride(const TensorStride& stride) {
        stride_ = stride;
    }

    /**
     * @brief Get the device where the data is stored.
     * @return Const reference to the device.
     */
    [[nodiscard]] std::shared_ptr<backend::Device> getDevice() const {
        return device_;
    }

    /**
     * @brief Set the device for this tensor data.
     * @param device New device.
     */
    void setDevice(std::shared_ptr<backend::Device> device) {
        device_ = std::move(device);
    }

    /**
     * @brief Get the offset for view operations.
     * @return size_t The current offset.
     */
    [[nodiscard]] size_t getOffset() const {
        return offset_;
    }

    /**
     * @brief Set the offset for view operations.
     * @param offset The new offset value.
     */
    void setOffset(size_t offset) {
        offset_ = offset;
    }

  private:
    std::shared_ptr<T[]> data_; /**< Raw heap-allocated data array. */
    TensorShape shape_;         /**< Dimensionality metadata. */
    TensorStride stride_;       /**< Memory skip values for indexing. */
    std::shared_ptr<backend::Device> device_ =
        backend::getCPUDevice(); /**< Device where data resides. */
    size_t offset_ = 0;

    std::uintptr_t gpuPtr = 0;

    friend class TensorWrapper<T>;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_DATA_H
