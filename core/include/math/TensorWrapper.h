#ifndef HAHAHA_MATH_TENSOR_WRAPPER_H
#define HAHAHA_MATH_TENSOR_WRAPPER_H

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "backend/Device.h"
#include "backend/DeviceComputeDispatcher.h"
#include "backend/cpu/CPUDevice.h"
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaDevice.h"
#include "backend/gpu/cuda/CudaMemory.h"
#endif
#endif
#include "common/Operator.h"
#include "math/ds/TensorData.h"
#include "math/ds/TensorShape.h"

class TensorWrapperTest;

namespace hahaha::compute {
template <typename T> class ComputeNode;
} // namespace hahaha::compute

namespace hahaha::math {
/**
 * @brief Main Tensor class providing a high-level API for numerical operations.
 *
 * This class wraps TensorData and provides a comprehensive set of operations
 * including basic arithmetic, matrix multiplication, and reshape capabilities.
 * It is designed to work with automatic differentiation and computational
 * graphs.
 *
 * @tparam T The numeric type of the tensor elements.
 */
template <typename T> class TensorWrapper {
  public:
    /**
     * @brief Default constructor for an empty tensor.
     */
    TensorWrapper() = default;

    /**
     * @brief Construct a tensor with a given shape and initial value on a
     * specific device.
     * @param shape The shape of the tensor.
     * @param initValue The initial value for all elements.
     * @param device The device where the data should reside.
     */
    explicit TensorWrapper(const TensorShape& shape,
                           T initValue,
                           std::shared_ptr<backend::Device> device =
                               std::make_shared<backend::CPUDevice>())
        : data_(TensorData<T>(TensorShape(shape), initValue, device)) {
    }

    /**
     * @brief Construct a tensor with a given shape on a specific device,
     *        with elements initialized to 0 (default).
     * @param shape The shape of the tensor.
     * @param device The device where the data should reside.
     */
    explicit TensorWrapper(const TensorShape& shape,
                           std::shared_ptr<backend::Device> device)
        : data_(TensorData<T>(TensorShape(shape), T(0), device)) {
    }

    /**
     * @brief Construct a tensor with a given shape, with elements initialized
     * to 0 (default CPU).
     * @param shape The shape of the tensor.
     */
    explicit TensorWrapper(const TensorShape& shape)
        : data_(TensorData<T>(TensorShape(shape),
                              T(0),
                              std::make_shared<backend::CPUDevice>())) {
    }

    /**
     * @brief Copy constructor. Performs a deep copy of the data.
     * @param other The tensor to copy from.
     */
    TensorWrapper(const TensorWrapper& other) : data_(other.data_) {
    }

    /**
     * @brief Move constructor. Transfers ownership of the data.
     * @param other The source tensor to move from.
     */
    TensorWrapper(TensorWrapper&& other) noexcept
        : data_(std::move(other.data_)) {
    }

    /**
     * @brief Copy assignment is deleted to encourage explicit copying.
     */
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source tensor to move from.
     * @return TensorWrapper& reference to this.
     */
    TensorWrapper& operator=(TensorWrapper&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    /**
     * @brief Construct from NestedData (e.g., nested initializer list).
     * @param data The source nested data.
     */
    explicit TensorWrapper(NestedData<T>&& data) : data_(std::move(data)) {
    }

    /**
     * @brief Construct from a vector of values.
     * @param initVec Vector of values to initialize the tensor with. Creates
     *                a 1D tensor with the same size as the vector.
     */
    explicit TensorWrapper(const std::vector<T>& initVec)
        : data_(TensorData<T>(initVec)) {
    }

    /**
     * @brief Destructor.
     */
    ~TensorWrapper() = default;

    /**
     * @brief Get a reference to the raw data pointer.
     * @return Reference to the shared_ptr holding the data array.
     */
    std::shared_ptr<T[]>& getRawData() {
        return data_.getData();
    }

    /**
     * @brief Get a const reference to the raw data pointer.
     * @return Const Reference to the shared_ptr holding the data array.
     */
    const std::shared_ptr<T[]>& getRawData() const {
        return data_.getData();
    }

    /**
     * @brief Get the tensor's shape.
     * @return const std::vector<size_t>& reference to internal shape.
     */
    [[nodiscard]] const std::vector<size_t>& getShape() const {
        return data_.getShape().getDims();
    }

    /**
     * @brief Get the tensor's shape.
     * @return size_t of totalSize.
     */
    [[nodiscard]] size_t getTotalSize() const {
        if (data_.getData() == nullptr) {
            return 0;
        }
        return data_.getShape().getTotalSize();
    }

    /**
     * @brief Get the tensor's strides.
     * @return const TensorStride& reference to internal strides.
     */
    [[nodiscard]] const TensorStride& getStride() const {
        return data_.getStride();
    }

    /**
     * @brief Set the tensor's strides.
     * @param stride The new strides.
     */
    void setStride(const TensorStride& stride) {
        data_.setStride(stride);
    }

    /**
     * @brief Get the device where the tensor resides.
     * @return std::shared_ptr<backend::Device> reference to the device.
     */
    [[nodiscard]] std::shared_ptr<backend::Device> getDevice() const {
        return data_.getDevice();
    }

    /**
     * @brief Move the tensor to a different device.
     * @param device The target device.
     * @note Currently, this only updates the device marker. Actual GPU memory
     *       storage is not yet fully implemented in TensorData. GPU operations
     *       will handle data transfer internally.
     */
    void to(std::shared_ptr<backend::Device> device) {
        if (*data_.getDevice() == *device) {
            return;
        }

        // Logic for moving data between devices
        if (device->getType() == backend::DeviceType::CPU) {
            if (data_.getDevice()->getType() == backend::DeviceType::CUDA) {
                // GPU to CPU transfer: copy data back to CPU
                const size_t totalSize = getTotalSize();
                if (totalSize == 0) {
                    data_.setDevice(device);
                    return;
                }

                // Create new CPU data
                auto newData = std::make_shared<T[]>(totalSize);
                // For now, since TensorData doesn't support GPU storage,
                // we just update the device marker
                // TODO: Implement proper GPU to CPU transfer when TensorData
                // supports GPU storage
                data_.setDevice(device);
                return;
            }
        } else if (device->getType() == backend::DeviceType::CUDA) {
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
            // CPU to GPU transfer: copy data to GPU
            const size_t totalSize = getTotalSize();
            if (totalSize == 0) {
                data_.setDevice(device);
                return;
            }

            // For now, just update the device marker
            // GPU operations will handle data transfer internally
            // TODO: Implement proper GPU storage in TensorData
            data_.setDevice(device);
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

        data_.setDevice(device);
    }

    /**
     * @brief Element access with bounds checking.
     *
     * Formula for linear index in row-major:
     * index = sum(indices[i] * strides[i])
     *
     * @param indices List of indices for each dimension.
     * @return T& reference to the element.
     */
    T& at(const std::initializer_list<size_t>& indices) {
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

    /**
     * @brief Constant element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return const T& reference to the element.
     */
    const T& at(const std::initializer_list<size_t>& indices) const {
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

    /**
     * @brief Reshape tensor to new dimensions.
     *
     * Total size must remain invariant.
     *
     * @param newShape Vector of new dimension sizes.
     * @return TensorWrapper A new tensor with reshaped dimensions.
     */
    TensorWrapper reshape(const std::vector<size_t>& newShape) const {
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
        result.data_.setData(std::make_unique<T[]>(currentSize));
        result.data_.setDevice(data_.getDevice());
        std::copy(data_.getData().get(),
                  data_.getData().get() + currentSize,
                  result.data_.getData().get());

        return result;
    }

    /**
     * @brief Number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t getDimensions() const {
        return data_.getShape().getDims().size();
    }

    /**
     * @brief Element-wise addition.
     *
     * Computes C = A + B element-wise.
     *
     * Formula:
     *     res[i] = a[i] + b[i]
     *
     * Scalar behavior:
     * - If one operand is a scalar (totalSize == 1):
     *     res[i] = tensor[i] + scalar
     *
     * Constraints:
     * - For non-scalar case, shapes must match exactly.
     * - Both tensors must be on the same device.
     *
     * @param other The tensor to add (B).
     * @return TensorWrapper Result tensor (C).
     */
    TensorWrapper add(const TensorWrapper& other) const {
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
            result.data_.setData(std::make_unique<T[]>(1));
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
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchAdd<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise subtraction.
     *
     * Computes C = A - B element-wise.
     *
     * Formula:
     *     res[i] = a[i] - b[i]
     *
     * Scalar behavior:
     * - Tensor - Scalar: res[i] = tensor[i] - scalar
     * - Scalar - Tensor: res[i] = scalar - tensor[i] (see subtractFrom)
     *
     * @param other The tensor to subtract (B).
     * @return TensorWrapper Result tensor (C).
     */
    TensorWrapper subtract(const TensorWrapper& other) const {
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
            result.data_.setData(std::make_unique<T[]>(1));
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
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchSub<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise multiplication.
     *
     * Computes C = A * B (Hadamard product).
     *
     * Formula:
     *     res[i] = a[i] * b[i]
     *
     * Scalar behavior:
     *     res[i] = tensor[i] * scalar
     *
     * @param other The tensor to multiply (B).
     * @return TensorWrapper Result tensor (C).
     */
    TensorWrapper multiply(const TensorWrapper& other) const {
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
            result.data_.setData(std::make_unique<T[]>(1));
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
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchMul<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise division.
     *
     * Computes C = A / B element-wise.
     *
     * Formula:
     *     res[i] = a[i] / b[i]   (b[i] != 0)
     *
     * Scalar behavior:
     * - Tensor / Scalar: res[i] = tensor[i] / scalar   (scalar != 0)
     * - Scalar / Tensor: res[i] = scalar / tensor[i]   (tensor[i] != 0)
     *
     * @param other The tensor to divide (B).
     * @return TensorWrapper Result tensor (C).
     */
    TensorWrapper divide(const TensorWrapper& other) const {
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
            result.data_.setData(std::make_unique<T[]>(1));
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
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchDiv<T>(
            data_.getDevice()->getType(), *this, other, result);

        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise scalar addition.
     * @param scalar The scalar value to add to each element.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper add(T scalar) const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchAdd(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise scalar subtraction.
     * @param scalar The scalar value to subtract from each element.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper subtract(T scalar) const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchSub(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise scalar multiplication.
     * @param scalar The scalar value to multiply each element by.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper multiply(T scalar) const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchMul(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise scalar division.
     * @param scalar The scalar value to divide each element by. Must not be
     * zero.
     * @return TensorWrapper Result tensor with same shape as this.
     * @throws std::runtime_error if scalar is zero.
     */
    TensorWrapper divide(T scalar) const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchDiv(
            data_.getDevice()->getType(), *this, scalar, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise subtraction from scalar (scalar - tensor).
     * @param scalar The scalar value to subtract this tensor from.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper subtractFrom(T scalar) const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchSub(
            data_.getDevice()->getType(), scalar, *this, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Element-wise division into scalar (scalar / tensor).
     * @param scalar The scalar value to divide by this tensor.
     * @return TensorWrapper Result tensor with same shape as this.
     * @throws std::runtime_error if any element of this tensor is zero.
     */
    TensorWrapper divideInto(T scalar) const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));

        auto res = backend::dispatchDiv(
            data_.getDevice()->getType(), scalar, *this, result);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }

        return result;
    }

    /**
     * @brief Matrix multiplication (for 2D tensors).
     *
     * Computes C = A * B.
     *
     * Formula:
     *     C[i, j] = sum(A[i, k] * B[k, j])
     *
     * where A is (M x K) and B is (K x N), resulting in C of shape (M x N).
     *
     * @param other The tensor to multiply with (B).
     * @return TensorWrapper Result tensor (C).
     */
    TensorWrapper matmul(const TensorWrapper& other) const {
        if (getDimensions() != 2 || other.getDimensions() != 2) {
            throw std::invalid_argument(
                "matmul is only implemented for 2D tensors");
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
        result.data_.setData(std::make_unique<T[]>(rows * cols));

        auto result_val = backend::dispatchMatMul(
            data_.getDevice()->getType(), *this, other, result);
        if (!result_val) {
            throw std::runtime_error(result_val.error().message());
        }

        return result;
    }

    /**
     * @brief Transpose operation (for 2D tensors).
     *
     * Formula: B[j, i] = A[i, j]
     *
     * @return TensorWrapper transposed tensor.
     */
    TensorWrapper transpose() const {
        if (getDimensions() != 2) {
            throw std::invalid_argument(
                "transpose is only implemented for 2D tensors for now");
        }

        const auto& shapeDims = data_.getShape().getDims();
        size_t rows = shapeDims[0];
        size_t cols = shapeDims[1];

        TensorWrapper result;
        result.data_.setShape(TensorShape({cols, rows}));
        result.data_.setStride(TensorStride(result.data_.getShape()));
        result.data_.setData(std::make_unique<T[]>(getTotalSize()));
        result.data_.setDevice(data_.getDevice());

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data_.getData()[j * rows + i] =
                    data_.getData()[i * cols + j];
            }
        }

        return result;
    }

    /**
     * @brief Sum all elements in the tensor.
     * @return T The sum of all tensor elements.
     */
    T sum() const {
        T result = T(0);
        const auto totalSize = getTotalSize();
        for (size_t i = 0; i < totalSize; ++i) {
            result += data_[i];
        }
        return result;
    }

    /**
     * @brief Create a deep copy of this tensor.
     * @return TensorWrapper A new tensor with copied data.
     */
    TensorWrapper clone() const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        result.data_.setDevice(data_.getDevice());
        result.data_.setData(std::make_shared<T[]>(getTotalSize()));
        for (size_t i = 0; i < getTotalSize(); ++i) {
            result.data_.getData()[i] = data_[i];
        }
        return result;
    }

    /**
     * @brief Sum tensor elements along specified axes.
     * @param axes The axes along which to sum. Empty axes means sum all
     * elements.
     * @param keepDims If true, the reduced dimensions are kept with size 1.
     * @return TensorWrapper Result tensor with summed values.
     * @throws std::invalid_argument if any axis is out of bounds.
     */
    TensorWrapper sum(std::vector<size_t> axes,
                      const bool keepDims = false) const {
        if (axes.empty()) {
            return this->clone();
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
        auto resultStride = result.getStride().getStrides();
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
        std::vector<size_t> coord(srcShape.size(), 0);
        size_t dstIdx = 0;

        for (size_t srcIdx = 0; srcIdx < getTotalSize(); ++srcIdx) {
            resPtr[dstIdx] += srcPtr[srcIdx];

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

    /**
     * @brief Clear all values in the tensor, setting them to default value
     * (likely 0).
     */
    void clear() {
        if (data_.getData() == nullptr) {
            return;
        }
        const auto totalSize = getTotalSize();
        for (size_t i = 0; i < totalSize; ++i) {
            data_[i] = T();
        }
    }

    /**
     * @brief Broadcast tensor to match the shape of another tensor.
     *
     * This operation does NOT allocate new storage. It returns a view.
     * Broadcasted axes are represented by setting stride[axis] = 0.
     * Reading element at coordinate `idx` uses:
     * linear = sum(idx[d] * stride[d])
     * value = base_ptr[linear]
     *
     * Example:
     * - self.shape = (1, 3), self.stride = (3, 1)
     * - newShape = (2, 3)
     * - result.stride becomes (0, 1)
     * - So result[0, j] and result[1, j] map to the same source element.
     *
     * @param newShape The target tensor shape for broadcasting.
     * @return TensorWrapper A new tensor that has the target shape but shares
     *         data with this one.
     * @throws std::invalid_argument if newShape is not broadcast-compatible,
     *         or if newShape rank is smaller than current rank.
     */
    TensorWrapper broadcastTo(const TensorShape& newShape) {
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
        newStride.getStrides().insert(
            newStride.getStrides().begin(), shapeDiff, 0);

        long newShapeIdx = static_cast<long>(newShape.getDims().size() - 1);
        long selfShapeIdx = static_cast<long>(this->getShape().size() - 1);

        while (selfShapeIdx >= 0) {
            if (newShape.getDims()[newShapeIdx] != 1
                && getShape()[selfShapeIdx] == 1) {
                newStride.getStrides()[newShapeIdx] = 0;
            }

            --newShapeIdx;
            --selfShapeIdx;
        }

        result.data_.setShape(newShape);
        result.data_.setStride(newStride);
        result.data_.setDevice(data_.getDevice());

        return result;
    }

    /**
     * @brief Addition operator (tensor + tensor).
     * @param other The tensor to add.
     * @return TensorWrapper Result of addition.
     */
    TensorWrapper operator+(const TensorWrapper& other) const {
        return add(other);
    }

    /**
     * @brief Subtraction operator (tensor - tensor).
     * @param other The tensor to subtract.
     * @return TensorWrapper Result of subtraction.
     */
    TensorWrapper operator-(const TensorWrapper& other) const {
        return subtract(other);
    }

    /**
     * @brief Multiplication operator (tensor * tensor).
     * @param other The tensor to multiply.
     * @return TensorWrapper Result of multiplication.
     */
    TensorWrapper operator*(const TensorWrapper& other) const {
        return multiply(other);
    }

    /**
     * @brief Division operator (tensor / tensor).
     * @param other The tensor to divide by.
     * @return TensorWrapper Result of division.
     */
    TensorWrapper operator/(const TensorWrapper& other) const {
        return divide(other);
    }

    /**
     * @brief Addition operator (tensor + scalar).
     * @param scalar The scalar value to add.
     * @return TensorWrapper Result of addition.
     */
    TensorWrapper operator+(T scalar) const {
        return add(scalar);
    }

    /**
     * @brief Subtraction operator (tensor - scalar).
     * @param scalar The scalar value to subtract.
     * @return TensorWrapper Result of subtraction.
     */
    TensorWrapper operator-(T scalar) const {
        return subtract(scalar);
    }

    /**
     * @brief Multiplication operator (tensor * scalar).
     * @param scalar The scalar value to multiply.
     * @return TensorWrapper Result of multiplication.
     */
    TensorWrapper operator*(T scalar) const {
        return multiply(scalar);
    }

    /**
     * @brief Division operator (tensor / scalar).
     * @param scalar The scalar value to divide by.
     * @return TensorWrapper Result of division.
     */
    TensorWrapper operator/(T scalar) const {
        return divide(scalar);
    }

    /**
     * @brief Unary negation operator (-tensor).
     * @return TensorWrapper Negated tensor.
     */
    TensorWrapper operator-() const {
        TensorWrapper result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        const size_t tensorSize = getTotalSize();
        result.data_.setData(std::make_unique<T[]>(tensorSize));
        result.data_.setDevice(data_.getDevice());
        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.getData()[i] = -data_.getData()[i];
        }
        return result;
    }

    /**
     * @brief In-place addition operator (tensor += tensor).
     * @param other The tensor to add.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator+=(const TensorWrapper& other) {
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

    /**
     * @brief In-place subtraction operator (tensor -= tensor).
     * @param other The tensor to subtract.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator-=(const TensorWrapper& other) {
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

    /**
     * @brief In-place multiplication operator (tensor *= tensor).
     * @param other The tensor to multiply.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator*=(const TensorWrapper& other) {
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

    /**
     * @brief In-place division operator (tensor /= tensor).
     * @param other The tensor to divide by.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator/=(const TensorWrapper& other) {
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

    /**
     * @brief In-place addition operator (tensor += scalar).
     * @param scalar The scalar value to add.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator+=(T scalar) {
        const size_t tensorSize = getTotalSize();
        for (size_t i = 0; i < tensorSize; ++i) {
            data_.getData()[i] += scalar;
        }
        return *this;
    }

    /**
     * @brief In-place subtraction operator (tensor -= scalar).
     * @param scalar The scalar value to subtract.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator-=(T scalar) {
        const size_t tensorSize = getTotalSize();
        for (size_t i = 0; i < tensorSize; ++i) {
            data_.getData()[i] -= scalar;
        }
        return *this;
    }

    /**
     * @brief In-place multiplication operator (tensor *= scalar).
     * @param scalar The scalar value to multiply.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator*=(T scalar) {
        const size_t tensorSize = getTotalSize();
        for (size_t i = 0; i < tensorSize; ++i) {
            data_.getData()[i] *= scalar;
        }
        return *this;
    }

    /**
     * @brief In-place division operator (tensor /= scalar).
     * @param scalar The scalar value to divide by. Must not be zero.
     * @return TensorWrapper& Reference to this.
     * @throws std::runtime_error if scalar is zero.
     */
    TensorWrapper& operator/=(T scalar) {
        if (scalar == T(0)) {
            throw std::runtime_error("Division by zero");
        }
        const size_t tensorSize = getTotalSize();
        for (size_t i = 0; i < tensorSize; ++i) {
            data_.getData()[i] /= scalar;
        }
        return *this;
    }

    /**
     * @brief In-place update: y = y + alpha * x
     *
     * Useful for optimizers (e.g., SGD: theta = theta - eta * grad).
     *
     * @param alpha Scaling factor.
     * @param other Other tensor (x).
     */
    void axpy(T alpha, const TensorWrapper& other) {
        if (getShape() != other.getShape()) {
            throw std::invalid_argument("Shape mismatch in axpy");
        }
        checkSameDevice(other);

        // Dispatch to backend for hardware-specific optimization
        auto res = backend::dispatchAxpy(
            data_.getDevice()->getType(), alpha, other, *this);
        if (!res) {
            throw std::runtime_error(res.error().message());
        }
    }

    /**
     * @brief Creates a new tensor of the same shape and device filled with
     * ones.
     * @return TensorWrapper A new tensor where every element is initialized
     * to 1.
     */
    TensorWrapper ones() const {
        TensorWrapper res(
            TensorShape(this->getShape()), T(1), this->getDevice());
        return res;
    }

    /**
     * @brief Creates a new tensor of the same shape and device filled with
     * zeros.
     * @return TensorWrapper A new tensor where every element is initialized to
     * 0.
     */
    TensorWrapper zeros() const {
        TensorWrapper res(
            TensorShape(this->getShape()), T(0), this->getDevice());
        return res;
    }

    /**
     * @brief Creates a new tensor of the same shape and device filled with a
     * specific value.
     * @param initValue The value to initialize all elements of the new tensor
     * with.
     * @return TensorWrapper A new tensor where every element is initialized to
     * @p initValue.
     */
    TensorWrapper sameShapeWithValue(T initValue) const {
        TensorWrapper res(
            TensorShape(this->getShape()), T(initValue), this->getDevice());
        return res;
    }

  private:
    TensorData<T> data_; /**< Managed tensor data and metadata. */

    /**
     * @brief Ensure that the other tensor is on the same device.
     * @param other The other tensor to check.
     */
    void checkSameDevice(const TensorWrapper& other) const {
        if (*getDevice() != *other.getDevice()) {

            throw std::invalid_argument(
                "Tensors must be on the same device for this operation (found "
                + getDevice()->toString() + " and "
                + other.getDevice()->toString() + ")");
        }
    }

    // Friend classes for internal access
    friend class TensorWrapperTest;
    friend class compute::ComputeNode<T>;
};

/**
 * @brief Addition operator (scalar + tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of addition.
 */
template <typename T>
TensorWrapper<T> operator+(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.add(scalar);
}

/**
 * @brief Subtraction operator (scalar - tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of subtraction.
 */
template <typename T>
TensorWrapper<T> operator-(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.subtractFrom(scalar);
}

/**
 * @brief Multiplication operator (scalar * tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of multiplication.
 */
template <typename T>
TensorWrapper<T> operator*(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.multiply(scalar);
}

/**
 * @brief Division operator (scalar / tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of division.
 */
template <typename T>
TensorWrapper<T> operator/(T scalar, const TensorWrapper<T>& tensor) {
    return tensor.divideInto(scalar);
}
} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_H
