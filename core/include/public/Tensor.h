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
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef TENSOR_DEA87AF2_9732_4E92_8B63_8D479609FD6C
#define TENSOR_DEA87AF2_9732_4E92_8B63_8D479609FD6C

#include <memory>
#include <vector>

#include "backend/Device.h"
#include "math/TensorWrapper.h"
#include "math/ds/TensorData.h"
#include "ml/compute/graph/ComputeFun.h"
#include "ml/compute/graph/ComputeNode.h"
#include "utils/common/helper_structs.h"

namespace hahaha {

// TODO: Make sure the check of nullptr of compute node
/**
 * @brief High-level User Interface for Tensor operations and Autograd.
 *
 * This class acts as a handle to a ComputeNode in the computational graph.
 * It provides operator overloading (+, -, *, /) which automatically builds
 * the graph in the background (Dynamic Graph / Define-by-Run).
 *
 * Usage:
 *   Tensor<float> a({2, 2}, 1.0f);
 *   Tensor<float> b({2, 2}, 2.0f);
 *   auto c = a + b;
 *   c.backward(); // Propagates gradients back to a and b
 *
 * @tparam T Numeric data type.
 */
template <typename T> class Tensor {
    static_assert(utils::isLegalDataType<T>::value, "T must be a legal data type");

  public:
    /**
     * @brief Construct a Tensor from an existing TensorWrapper.
     * @param data The numerical data wrapper.
     */
    explicit Tensor(const math::TensorWrapper<T>& data)
        : computeNode_(std::make_shared<compute::ComputeNode<T>>(
              std::make_shared<math::TensorWrapper<T>>(data))) {
    }
    /**
     * @brief Default tensor constructor
     */
    Tensor() {
        computeNode_ = std::make_shared<compute::ComputeNode<T>>(
            std::make_shared<math::TensorWrapper<T>>());
    }

    /**
     * @brief Construct a Tensor from a NestedData (flattened multi-dim list).
     * @param data The source nested data.
     */
    // NOLINTNEXTLINE
    Tensor(math::NestedData<T>&& data)
        : computeNode_(std::make_shared<compute::ComputeNode<T>>(
              std::make_shared<math::TensorWrapper<T>>(std::move(data)))) {
    }

    /**
     * @brief Construct a Tensor from a shared pointer to TensorWrapper.
     * @param dataPtr pointer to the numerical data.
     */
    explicit Tensor(std::shared_ptr<math::TensorWrapper<T>> dataPtr)
        : computeNode_(std::make_shared<compute::ComputeNode<T>>(dataPtr)) {
    }

    /**
     * @brief Internal constructor to wrap a ComputeNode.
     * @param computeNode The node in the computational graph.
     */
    explicit Tensor(std::shared_ptr<compute::ComputeNode<T>> computeNode)
        : computeNode_(computeNode) {
    }

    /** @brief Build a tensor from a vector. */
    static Tensor buildFromVector(const std::vector<T>& vec) {
        auto computeNode = std::make_shared<compute::ComputeNode<T>>(
            std::make_shared<math::TensorWrapper<T>>(vec));
        return Tensor(computeNode);
    }

    static Tensor buildFromShape(std::initializer_list<size_t> shape) {
        math::TensorShape tensorShape(shape);
        auto computeNode = std::make_shared<compute::ComputeNode<T>>(
            std::make_shared<math::TensorWrapper<T>>(tensorShape));

        return Tensor(computeNode);
    }

    /** @brief Addition operator. Builds an 'Add' node. */
    Tensor operator+(const Tensor& other) const {
        return Tensor(compute::add(this->computeNode_, other.computeNode_));
    }

    /** @brief Subtraction operator. Builds a 'Sub' node. */
    Tensor operator-(const Tensor& other) const {
        return Tensor(compute::sub(this->computeNode_, other.computeNode_));
    }

    /** @brief Multiplication operator. Builds a 'Mul' node. */
    Tensor operator*(const Tensor& other) const {
        return Tensor(compute::mul(this->computeNode_, other.computeNode_));
    }

    /** @brief Division operator. Builds a 'Div' node. */
    Tensor operator/(const Tensor& other) const {
        return Tensor(compute::div(this->computeNode_, other.computeNode_));
    }

    /** @brief Scalar multiplication operator (Tensor * scalar). */
    Tensor operator*(T scalar) const {
        return Tensor(compute::mul(this->computeNode_, scalar));
    }

    /** @brief Scalar addition operator (Tensor + scalar). */
    Tensor operator+(T scalar) const {
        return Tensor(compute::add(this->computeNode_, scalar));
    }

    /** @brief Scalar subtraction operator (Tensor - scalar). */
    Tensor operator-(T scalar) const {
        return Tensor(compute::sub(this->computeNode_, scalar));
    }

    /** @brief Scalar division operator (Tensor / scalar). */
    Tensor operator/(T scalar) const {
        return Tensor(compute::div(this->computeNode_, scalar));
    }

    /** @brief Unary negation operator. */
    Tensor operator-() const {
        return Tensor(compute::neg(this->computeNode_));
    }

    /** @brief Matrix multiplication. */
    Tensor matmul(const Tensor& other) const {
        return Tensor(compute::matmul(this->computeNode_, other.computeNode_));
    }

    /**
     * @brief Reshape tensor to new dimensions.
     *
     * Total size must remain invariant.
     *
     * @param newShape Vector of new dimension sizes.
     * @return TensorWrapper<T> A new tensor with reshaped dimensions.
     */
    Tensor reshape(const std::vector<size_t>& newShape) const {
        return Tensor(compute::reshape(this->computeNode_, newShape));
    }

    /**
     * @brief Transpose operation (for 2D tensors).
     *
     * Formula: B[j, i] = A[i, j]
     *
     * @return TensorWrapper<T> transposed tensor.
     */
    Tensor transpose() const {
        return Tensor(compute::transpose(this->computeNode_));
    }

    // Friend functions for scalar-tensor operations (scalar op Tensor)
    friend Tensor operator*(T scalar, const Tensor& tensor) {
        return Tensor(compute::mul(scalar, tensor.computeNode_));
    }

    friend Tensor operator+(T scalar, const Tensor& tensor) {
        return Tensor(compute::add(scalar, tensor.computeNode_));
    }

    friend Tensor operator-(T scalar, const Tensor& tensor) {
        return Tensor(compute::sub(scalar, tensor.computeNode_));
    }

    friend Tensor operator/(T scalar, const Tensor& tensor) {
        return Tensor(compute::div(scalar, tensor.computeNode_));
    }

    /**
     * @brief Triggers backpropagation from this tensor.
     *
     * This will compute gradients for all ancestor tensors in the graph
     * that have 'requiresGrad' set to true.
     */
    void backward() {
        computeNode_->backward();
    }

    /**
     * @brief Get the managed gradient as a Tensor.
     * @return Tensor containing the accumulated gradients.
     */
    [[nodiscard]] Tensor grad() const {
        if (computeNode_->getGrad()) {
            return Tensor(computeNode_->getGrad());
        }
        return Tensor();
    }

    /**
     * @brief Check if the tensor is empty.
     * @return true if the tensor is empty.
     */
    [[nodiscard]] bool isEmpty() const {
        return getTotalSize() == 0;
    }

    /**
     * @brief Clear the tensor.
     */
    void clear() {
        computeNode_->getData()->clear();
    }

    /**
     * @brief Clean the gradient of the node.
     */
    void clearGrad() {
        computeNode_->clearGrad();
    }

    /** @brief Get the underlying data wrapper. */
    [[nodiscard]] std::shared_ptr<math::TensorWrapper<T>> data() const {
        return computeNode_->getData();
    }

    /**
     * @brief Move the tensor to a different device.
     * @param device The target device.
     */
    void to(std::shared_ptr<backend::Device> device) {
        computeNode_->getData()->to(device);
    }

    [[nodiscard]] const std::vector<size_t>& getShape() const {
        return computeNode_->getData()->getShape();
    }

    /** @brief Get the device where the tensor resides. */
    [[nodiscard]] const backend::Device& getDevice() const {
        return computeNode_->getData()->getDevice();
    }

    /** @brief Set whether this tensor requires gradients. */
    void setRequiresGrad(bool req) {
        computeNode_->setRequiresGrad(req);
    }

    /** @brief Check if gradients are required. */
    [[nodiscard]] bool getRequiresGrad() const {
        return computeNode_->getRequiresGrad();
    }

    /** @brief Access element at specified indices (for testing). */
    T& at(const std::initializer_list<size_t>& indices) {
        return computeNode_->getData()->at(indices);
    }

    /** @brief Const access to element. */
    const T& at(const std::initializer_list<size_t>& indices) const {
        return computeNode_->getData()->at(indices);
    }

    /**
     * @brief Get the underlying compute node.
     * @return shared_ptr to the node.
     */
    std::shared_ptr<compute::ComputeNode<T>> getComputeNode() const {
        return computeNode_;
    }

    /**
     * @brief Set the compute node for this tensor.
     * @param node The new node.
     */
    void setComputeNode(std::shared_ptr<compute::ComputeNode<T>> node) {
        computeNode_ = node;
    }

    /**
     * @brief Return total size of elements it holds
     * @return Total size of elements
     */
    [[nodiscard]] size_t getTotalSize() const {
        return computeNode_->getData()->getTotalSize();
    }

    /**
     * @brief sum the elements those the tensor holds
     * @return T the sum of all elements.
     */
    T sum() const {
        return computeNode_->getData()->sum();
    }

    /**
     * @brief Creates a new Tensor of the same shape, initialized with zeros.
     * * This method requests a zero-filled data container from the current
     * device and wraps it in a new @ref compute::ComputeNode.
     * * @return Tensor A new tensor instance with all elements set to 0.
     */
    Tensor zeros() const {
        auto zeroData = std::make_shared<math::TensorWrapper<T>>(
            computeNode_->getData()->zeros());
        return Tensor(zeroData);
    }

    /**
     * @brief Creates a new Tensor of the same shape, initialized with ones.
     * * This method requests a ones-filled data container from the current
     * device and wraps it in a new @ref compute::ComputeNode.
     * * @return Tensor A new tensor instance with all elements set to 1.
     */
    Tensor ones() const {
        auto oneData = std::make_shared<math::TensorWrapper<T>>(
            computeNode_->getData()->ones());
        return Tensor(oneData);
    }

    /**
     * @brief Creates a new Tensor of the same shape, initialized with a
     * specific value.
     * * @param initValue The value to fill the new tensor with.
     * @return Tensor A new tensor instance where every element is @p initValue.
     */
    Tensor sameShapeWithValue(T initValue) const {
        auto newData = std::make_shared<math::TensorWrapper<T>>(
            computeNode_->getData()->sameShapeWithValue(initValue));
        return Tensor(newData);
    }

    /**
     * @brief Create a deep copy of this tensor.
     * @return Tensor A new tensor with copied data.
     */
    Tensor clone() const {
        auto clonedData = std::make_shared<math::TensorWrapper<T>>(
            computeNode_->getData()->clone());
        return Tensor(clonedData);
    }

    /**
     * @brief Number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t getDimensions() const {
        return computeNode_->getData()->getDimensions();
    }

    /**
     * @brief Get the tensor's strides.
     * @return const TensorStride& reference to internal strides.
     */
    [[nodiscard]] const math::TensorStride& getStride() const {
        return computeNode_->getData()->getStride();
    }

    /**
     * @brief Narrow the tensor along a dimension from start with given length.
     *
     * Returns a new Tensor that is a view of the original, narrowed along
     * dimension @p dim from @p start to @p start + @p length.
     *
     * Formula:
     *     result[i0, ..., idim, ..., iN] = self[i0, ..., start+idim, ..., iN]
     *     where 0 <= idim < length
     *
     * Example:
     * @code
     *   // tensor shape (4, 5, 6)
     *   auto t2 = tensor.narrow(0, 1, 2); // shape (2, 5, 6)
     * @endcode
     *
     * @param dim    Dimension to narrow along. Must be < getDimensions().
     * @param start  Starting index along @p dim. Must be < shape[dim].
     * @param length Number of elements to keep. Must satisfy start+length <=
     * shape[dim].
     * @return Tensor A new view tensor with reduced size along @p dim.
     * @throws std::out_of_range if @p dim, @p start, or @p length are out of bounds.
     */
    Tensor narrow(size_t dim, size_t start, size_t length) const {
        auto result = computeNode_->getData()->narrow(dim, start, length);
        return Tensor(std::make_shared<math::TensorWrapper<T>>(std::move(result)));
    }

    /**
     * @brief Select a single index along a dimension, removing that dimension.
     *
     * Returns a new Tensor with dimension @p dim removed by fixing it at @p index.
     *
     * Formula:
     *     result[i0, ..., idim-1, idim+1, ..., iN] = self[i0, ..., index, ..., iN]
     *
     * Example:
     * @code
     *   // tensor shape (4, 5, 6)
     *   auto t2 = tensor.select(0, 2); // shape (5, 6)
     *   auto t3 = tensor.select(1, 1); // shape (4, 6)
     * @endcode
     *
     * @param dim   Dimension to select from. Must be < getDimensions().
     * @param index Index to select along @p dim. Must be < shape[dim].
     * @return Tensor A new view tensor with @p dim removed.
     * @throws std::out_of_range if @p dim or @p index are out of bounds.
     */
    Tensor select(size_t dim, size_t index) const {
        auto result = computeNode_->getData()->select(dim, index);
        return Tensor(std::make_shared<math::TensorWrapper<T>>(std::move(result)));
    }

    /**
     * @brief Slice along a single dimension with optional start, end, and step.
     *
     * Returns a new Tensor sliced along dimension @p dim using Python-style
     * [start:end:step] semantics. When step == 1 the result shares storage
     * with the original (zero-copy view).
     *
     * Formula:
     *     result[i0, ..., idim, ..., iN] = self[i0, ..., start+idim*step, ..., iN]
     *     where 0 <= idim < ceil((end - start) / step)
     *
     * Example:
     * @code
     *   // tensor shape (4, 5, 6)
     *   auto t2 = tensor.sliceDim(0, 0, 4, 2); // shape (2, 5, 6) — every 2nd
     *   auto t3 = tensor.sliceDim(1, 1, 4);    // shape (4, 3, 6) — [1:4]
     * @endcode
     *
     * @param dim   Dimension to slice along. Must be < getDimensions().
     * @param start Starting index (inclusive). Defaults to 0.
     * @param end   Ending index (exclusive). Defaults to shape[dim].
     * @param step  Step size (must be > 0). Defaults to 1.
     * @return Tensor A new (view) tensor with the sliced dimension.
     * @throws std::out_of_range     if @p dim is out of range.
     * @throws std::invalid_argument if step == 0 or start >= end.
     */
    Tensor sliceDim(size_t dim,
                    std::optional<size_t> start = std::nullopt,
                    std::optional<size_t> end = std::nullopt,
                    size_t step = 1) const {
        auto result = computeNode_->getData()->sliceDim(dim, start, end, step);
        return Tensor(std::make_shared<math::TensorWrapper<T>>(std::move(result)));
    }

    /**
     * @brief Slice multiple dimensions using a SliceSetting descriptor.
     *
     * Returns a new Tensor sliced according to @p setting, which specifies
     * per-dimension [start:end:step] ranges.  All-step-1 slices are zero-copy
     * views; step > 1 may produce non-contiguous views.
     *
     * Example:
     * @code
     *   // tensor shape (4, 5, 6)
     *   SliceSetting setting({
     *       {0, {1, 3, 1}},   // dim 0: [1:3]
     *       {2, {0, 6, 2}}    // dim 2: [0:6:2]
     *   });
     *   auto t2 = tensor.slice(setting); // shape (2, 5, 3)
     * @endcode
     *
     * @param setting Slice configuration (dimensions + ranges). Modified
     *                in-place to sort axes in ascending order.
     * @return Tensor A new (view) tensor with all specified dimensions sliced.
     * @throws std::invalid_argument if any dimension index or range is invalid.
     */
    Tensor slice(SliceSetting& setting) const {
        auto result = computeNode_->getData()->slice(setting);
        return Tensor(std::make_shared<math::TensorWrapper<T>>(std::move(result)));
    }

    /**
     * @brief Broadcast this tensor to a larger shape.
     *
     * Returns a new Tensor that has @p newShape but shares storage with the
     * original (zero-copy view).  Broadcasted axes use stride == 0 so that
     * reads along that axis always hit the same memory location.
     *
     * Example:
     * @code
     *   // tensor shape (1, 3)
     *   auto t2 = tensor.broadcastTo({2, 3}); // shape (2, 3), stride (0, 1)
     * @endcode
     *
     * @param newShape The target shape as a vector of dimension sizes. Must be
     *                 broadcast-compatible and have rank >= current rank.
     * @return Tensor A new view tensor with shape @p newShape.
     * @throws std::invalid_argument if shapes are not broadcast-compatible.
     */
    Tensor broadcastTo(const std::vector<size_t>& newShape) const {
        auto result =
            computeNode_->getData()->broadcastTo(math::TensorShape(newShape));
        return Tensor(std::make_shared<math::TensorWrapper<T>>(std::move(result)));
    }

    /**
     * @brief In-place addition (tensor += tensor).
     */
    Tensor& operator+=(const Tensor& other) {
        *computeNode_->getData() += *other.computeNode_->getData();
        return *this;
    }

    /**
     * @brief In-place subtraction (tensor -= tensor).
     */
    Tensor& operator-=(const Tensor& other) {
        *computeNode_->getData() -= *other.computeNode_->getData();
        return *this;
    }

    /**
     * @brief In-place multiplication (tensor *= tensor).
     */
    Tensor& operator*=(const Tensor& other) {
        *computeNode_->getData() *= *other.computeNode_->getData();
        return *this;
    }

    /**
     * @brief In-place division (tensor /= tensor).
     */
    Tensor& operator/=(const Tensor& other) {
        *computeNode_->getData() /= *other.computeNode_->getData();
        return *this;
    }

    /**
     * @brief In-place scalar addition (tensor += scalar).
     */
    Tensor& operator+=(T scalar) {
        *computeNode_->getData() += scalar;
        return *this;
    }

    /**
     * @brief In-place scalar subtraction (tensor -= scalar).
     */
    Tensor& operator-=(T scalar) {
        *computeNode_->getData() -= scalar;
        return *this;
    }

    /**
     * @brief In-place scalar multiplication (tensor *= scalar).
     */
    Tensor& operator*=(T scalar) {
        *computeNode_->getData() *= scalar;
        return *this;
    }

    /**
     * @brief In-place scalar division (tensor /= scalar).
     */
    Tensor& operator/=(T scalar) {
        *computeNode_->getData() /= scalar;
        return *this;
    }

    /**
     * @brief Square all elements.
     * @return Tensor A new tensor with squared elements.
     * @note This operation currently does not support autograd.
     */
    Tensor square() const {
        auto newData = computeNode_->getData()->clone();
        newData.squareInPlace();
        return Tensor(std::make_shared<math::TensorWrapper<T>>(std::move(newData)));
    }

    /**
     * @brief Square all elements in place.
     */
    void squareInPlace() {
        computeNode_->getData()->squareInPlace();
    }

    /**
     * @brief Square root all elements in place.
     */
    void sqrtInPlace() {
        computeNode_->getData()->sqrtInPlace();
    }

    /**
     * @brief In-place update: y = y + alpha * x
     * @param alpha Scaling factor.
     * @param other Other tensor (x).
     */
    void axpy(T alpha, const Tensor& other) {
        computeNode_->getData()->axpy(alpha, *other.computeNode_->getData());
    }

  private:
    std::shared_ptr<compute::ComputeNode<T>> computeNode_; /**< Graph link. */
};

} // namespace hahaha

#endif // TENSOR_DEA87AF2_9732_4E92_8B63_8D479609FD6C
