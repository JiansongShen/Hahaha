#ifndef HAHAHA_MATH_TENSOR_WRAPPER_H
#define HAHAHA_MATH_TENSOR_WRAPPER_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "backend/Device.h"
#include "backend/DeviceComputeDispatcher.h"
#include "backend/DeviceRegistry.h"
#ifdef HAHAHA_USE_CUDA
#if __has_include(<driver_types.h>)
#include <cuda_runtime.h>

#include "backend/gpu/cuda/CudaDevice.h"
#include "backend/gpu/cuda/CudaMemory.h"
#endif
#endif
#include "math/ds/TensorData.h"
#include "math/ds/TensorShape.h"

class TensorWrapperTest;
struct SliceSetting;

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
    explicit TensorWrapper(
        const TensorShape& shape,
        T initValue,
        std::shared_ptr<backend::Device> device =
            backend::DeviceRegistry::getInstance().getCPUDevice());

    /**
     * @brief Construct a tensor with a given shape on a specific device,
     *        with elements initialized to 0 (default).
     * @param shape The shape of the tensor.
     * @param device The device where the data should reside.
     */
    explicit TensorWrapper(const TensorShape& shape,
                           std::shared_ptr<backend::Device> device);

    /**
     * @brief Construct a tensor with a given shape, with elements initialized
     * to 0 (default CPU).
     * @param shape The shape of the tensor.
     */
    explicit TensorWrapper(const TensorShape& shape);

    /**
     * @brief Copy constructor. Performs a deep copy of the data.
     * @param other The tensor to copy from.
     */
    TensorWrapper(const TensorWrapper& other);

    /**
     * @brief Move constructor. Transfers ownership of the data.
     * @param other The source tensor to move from.
     */
    TensorWrapper(TensorWrapper&& other) noexcept;

    /**
     * @brief Copy assignment is deleted to encourage explicit copying.
     */
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source tensor to move from.
     * @return TensorWrapper& reference to this.
     */
    TensorWrapper& operator=(TensorWrapper&& other) noexcept;

    /**
     * @brief Construct from NestedData (e.g., nested initializer list).
     * @param data The source nested data.
     */
    explicit TensorWrapper(NestedData<T>&& data);

    /**
     * @brief Construct from a vector of values.
     * @param initVec Vector of values to initialize the tensor with. Creates
     *                a 1D tensor with the same size as the vector.
     */
    explicit TensorWrapper(const std::vector<T>& initVec);

    /**
     * @brief Destructor.
     */
    ~TensorWrapper() = default;

    /**
     * @brief Get a reference to the raw data pointer.
     * @return Reference to the shared_ptr holding the data array.
     */
    std::shared_ptr<T[]>& getRawData();

    /**
     * @brief Get a const reference to the raw data pointer.
     * @return Const Reference to the shared_ptr holding the data array.
     */
    const std::shared_ptr<T[]>& getRawData() const;

    /**
     * @brief Get the GPU pointer.
     * @return std::uintptr_t of GPU pointer.
     */
    [[nodiscard]] std::uintptr_t getRawGpuPtr() const;

    /**
     * @brief Get the tensor's shape.
     * @return const std::vector<size_t>& reference to internal shape.
     */
    [[nodiscard]] const std::vector<size_t>& getShape() const;

    /**
     * @brief Get the tensor's shape.
     * @return size_t of totalSize.
     */
    [[nodiscard]] size_t getTotalSize() const;

    /**
     * @brief Get the tensor's strides.
     * @return const TensorStride& reference to internal strides.
     */
    [[nodiscard]] const TensorStride& getStride() const;

    /**
     * @brief Set the tensor's strides.
     * @param stride The new strides.
     */
    void setStride(const TensorStride& stride);

    /**
     * @brief Get the device where the tensor resides.
     * @return std::shared_ptr<backend::Device> reference to the device.
     */
    [[nodiscard]] std::shared_ptr<backend::Device> getDevice() const;

    /**
     * @brief Move the tensor to a different device.
     * @param targetDevice The target device.
     * @note Currently, this only updates the device marker. Actual GPU memory
     *       storage is not yet fully implemented in TensorData. GPU operations
     *       will handle data transfer internally.
     */
    void to(std::shared_ptr<backend::Device> targetDevice);

    /**
     * @brief Element access with bounds checking.
     *
     * Formula for linear index in row-major:
     * index = sum(indices[i] * strides[i])
     *
     * @param indices List of indices for each dimension.
     * @return T& reference to the element.
     */
    T& at(const std::initializer_list<size_t>& indices);

    /**
     * @brief Constant element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return const T& reference to the element.
     */
    const T& at(const std::initializer_list<size_t>& indices) const;

    /**
     * @brief Reshape tensor to new dimensions.
     *
     * Total size must remain invariant.
     *
     * @param newShape Vector of new dimension sizes.
     * @return TensorWrapper A new tensor with reshaped dimensions.
     */
    TensorWrapper reshape(const std::vector<size_t>& newShape) const;

    /**
     * @brief Number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t getDimensions() const;

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
    TensorWrapper add(const TensorWrapper& other) const;

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
    TensorWrapper subtract(const TensorWrapper& other) const;

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
    TensorWrapper multiply(const TensorWrapper& other) const;

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
    TensorWrapper divide(const TensorWrapper& other) const;

    /**
     * @brief Element-wise scalar addition.
     * @param scalar The scalar value to add to each element.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper add(T scalar) const;

    /**
     * @brief Element-wise scalar subtraction.
     * @param scalar The scalar value to subtract from each element.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper subtract(T scalar) const;

    /**
     * @brief Element-wise scalar multiplication.
     * @param scalar The scalar value to multiply each element by.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper multiply(T scalar) const;

    /**
     * @brief Element-wise scalar division.
     * @param scalar The scalar value to divide each element by. Must not be
     * zero.
     * @return TensorWrapper Result tensor with same shape as this.
     * @throws std::runtime_error if scalar is zero.
     */
    TensorWrapper divide(T scalar) const;

    /**
     * @brief Element-wise subtraction from scalar (scalar - tensor).
     * @param scalar The scalar value to subtract this tensor from.
     * @return TensorWrapper Result tensor with same shape as this.
     */
    TensorWrapper subtractFrom(T scalar) const;

    /**
     * @brief Element-wise division into scalar (scalar / tensor).
     * @param scalar The scalar value to divide by this tensor.
     * @return TensorWrapper Result tensor with same shape as this.
     * @throws std::runtime_error if any element of this tensor is zero.
     */
    TensorWrapper divideInto(T scalar) const;

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
    TensorWrapper matmul(const TensorWrapper& other) const;

    /**
     * @brief Transpose operation (for 2D tensors).
     *
     * Formula: B[j, i] = A[i, j]
     *
     * @return TensorWrapper transposed tensor.
     */
    TensorWrapper transpose() const;

    /**
     * @brief Sum all elements in the tensor.
     * @return T The sum of all tensor elements.
     */
    T sum() const;

    /**
     * @brief Create a deep copy of this tensor.
     * @return TensorWrapper A new tensor with copied data.
     */
    TensorWrapper clone() const;

    /**
     * @brief Sum tensor elements along specified axes.
     * @param axes The axes along which to sum. Empty axes means sum all
     * elements.
     * @param keepDims If true, the reduced dimensions are kept with size 1.
     * @return TensorWrapper Result tensor with summed values.
     * @throws std::invalid_argument if any axis is out of bounds.
     */
    TensorWrapper sum(std::vector<size_t> axes,
                      const bool keepDims = false) const;

    /**
     * @brief Clear all values in the tensor, setting them to default value
     * (likely 0).
     */
    void clear();

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
    TensorWrapper broadcastTo(const TensorShape& newShape);

    /**
     * @brief Addition operator (tensor + tensor).
     * @param other The tensor to add.
     * @return TensorWrapper Result of addition.
     */
    TensorWrapper operator+(const TensorWrapper& other) const;

    /**
     * @brief Subtraction operator (tensor - tensor).
     * @param other The tensor to subtract.
     * @return TensorWrapper Result of subtraction.
     */
    TensorWrapper operator-(const TensorWrapper& other) const;

    /**
     * @brief Multiplication operator (tensor * tensor).
     * @param other The tensor to multiply.
     * @return TensorWrapper Result of multiplication.
     */
    TensorWrapper operator*(const TensorWrapper& other) const;

    /**
     * @brief Division operator (tensor / tensor).
     * @param other The tensor to divide by.
     * @return TensorWrapper Result of division.
     */
    TensorWrapper operator/(const TensorWrapper& other) const;

    /**
     * @brief Addition operator (tensor + scalar).
     * @param scalar The scalar value to add.
     * @return TensorWrapper Result of addition.
     */
    TensorWrapper operator+(T scalar) const;

    /**
     * @brief Subtraction operator (tensor - scalar).
     * @param scalar The scalar value to subtract.
     * @return TensorWrapper Result of subtraction.
     */
    TensorWrapper operator-(T scalar) const;

    /**
     * @brief Multiplication operator (tensor * scalar).
     * @param scalar The scalar value to multiply.
     * @return TensorWrapper Result of multiplication.
     */
    TensorWrapper operator*(T scalar) const;

    /**
     * @brief Division operator (tensor / scalar).
     * @param scalar The scalar value to divide by.
     * @return TensorWrapper Result of division.
     */
    TensorWrapper operator/(T scalar) const;

    /**
     * @brief Unary negation operator (-tensor).
     * @return TensorWrapper Negated tensor.
     */
    TensorWrapper operator-() const;

    /**
     * @brief In-place addition operator (tensor += tensor).
     * @param other The tensor to add.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator+=(const TensorWrapper& other);

    /**
     * @brief In-place subtraction operator (tensor -= tensor).
     * @param other The tensor to subtract.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator-=(const TensorWrapper& other);

    /**
     * @brief In-place multiplication operator (tensor *= tensor).
     * @param other The tensor to multiply.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator*=(const TensorWrapper& other);

    /**
     * @brief In-place division operator (tensor /= tensor).
     * @param other The tensor to divide by.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator/=(const TensorWrapper& other);

    /**
     * @brief In-place addition operator (tensor += scalar).
     * @param scalar The scalar value to add.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator+=(T scalar);

    /**
     * @brief In-place subtraction operator (tensor -= scalar).
     * @param scalar The scalar value to subtract.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator-=(T scalar);

    /**
     * @brief In-place multiplication operator (tensor *= scalar).
     * @param scalar The scalar value to multiply.
     * @return TensorWrapper& Reference to this.
     */
    TensorWrapper& operator*=(T scalar);

    /**
     * @brief In-place division operator (tensor /= scalar).
     * @param scalar The scalar value to divide by. Must not be zero.
     * @return TensorWrapper& Reference to this.
     * @throws std::runtime_error if scalar is zero.
     */
    TensorWrapper& operator/=(T scalar);

    /**
     * @brief In-place update: y = y + alpha * x
     *
     * Useful for optimizers (e.g., SGD: theta = theta - eta * grad).
     *
     * @param alpha Scaling factor.
     * @param other Other tensor (x).
     */
    void axpy(T alpha, const TensorWrapper& other);

    /**
     * @brief Square all elements in place.
     */
    void squareInPlace();

    /**
     * @brief Square root all elements in place.
     */
    void sqrtInPlace();

    /**
     * @brief Creates a new tensor of the same shape and device filled with
     * ones.
     * @return TensorWrapper A new tensor where every element is initialized
     * to 1.
     */
    TensorWrapper ones() const;

    /**
     * @brief Creates a new tensor of the same shape and device filled with
     * zeros.
     * @return TensorWrapper A new tensor where every element is initialized to
     * 0.
     */
    TensorWrapper zeros() const;

    /**
     * @brief Creates a new tensor of the same shape and device filled with a
     * specific value.
     * @param initValue The value to initialize all elements of the new tensor
     * with.
     * @return TensorWrapper A new tensor where every element is initialized to
     * @p initValue.
     */
    TensorWrapper sameShapeWithValue(T initValue) const;

    /**
     * @brief Slice the tensor based on the provided settings.
     * @param requirements The slice settings.
     * @return TensorWrapper The sliced tensor.
     */
    TensorWrapper slice(SliceSetting requirements);

  private:
    TensorData<T> data_; /**< Managed tensor data and metadata. */

    /**
     * @brief Ensure that the other tensor is on the same device.
     * @param other The other tensor to check.
     */
    void checkSameDevice(const TensorWrapper& other) const;

    // Friend classes for internal access
    friend class ::TensorWrapperTest;
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
TensorWrapper<T> operator+(T scalar, const TensorWrapper<T>& tensor);

/**
 * @brief Subtraction operator (scalar - tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of subtraction.
 */
template <typename T>
TensorWrapper<T> operator-(T scalar, const TensorWrapper<T>& tensor);

/**
 * @brief Multiplication operator (scalar * tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of multiplication.
 */
template <typename T>
TensorWrapper<T> operator*(T scalar, const TensorWrapper<T>& tensor);

/**
 * @brief Division operator (scalar / tensor).
 * @tparam T The numeric type.
 * @param scalar The scalar value.
 * @param tensor The tensor.
 * @return TensorWrapper<T> Result of division.
 */
template <typename T>
TensorWrapper<T> operator/(T scalar, const TensorWrapper<T>& tensor);

} // namespace hahaha::math

// Include implementation files
#include "TensorWrapperConstructors.inl"
#include "TensorWrapperAccessors.inl"
#include "TensorWrapperArithmetic.inl"
#include "TensorWrapperMatrixOps.inl"
#include "TensorWrapperOperators.inl"
#include "TensorWrapperShapeOps.inl"
#include "TensorWrapperUtilities.inl"

#endif // HAHAHA_MATH_TENSOR_WRAPPER_H
