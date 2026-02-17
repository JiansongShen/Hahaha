#ifndef HAHAHA_MATH_TENSOR_WRAPPER_UTILITIES_INL
#define HAHAHA_MATH_TENSOR_WRAPPER_UTILITIES_INL

namespace hahaha::math {

template <typename T>
void TensorWrapper<T>::axpy(T alpha, const TensorWrapper& other) {
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

template <typename T> TensorWrapper<T> TensorWrapper<T>::ones() const {
    TensorWrapper res(TensorShape(this->getShape()), T(1), this->getDevice());
    return res;
}

template <typename T> TensorWrapper<T> TensorWrapper<T>::zeros() const {
    TensorWrapper res(TensorShape(this->getShape()), T(0), this->getDevice());
    return res;
}

template <typename T>
TensorWrapper<T> TensorWrapper<T>::sameShapeWithValue(T initValue) const {
    TensorWrapper res(
        TensorShape(this->getShape()), T(initValue), this->getDevice());
    return res;
}

template <typename T>
TensorWrapper<T>
TensorWrapper<T>::slice(SliceSetting requirements) {
    requirements.sort();

    return {};
}

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_UTILITIES_INL
