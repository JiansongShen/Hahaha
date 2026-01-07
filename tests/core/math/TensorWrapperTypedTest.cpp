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
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "backend/Device.h"
#include "common/definitions.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"
#include "math/ds/TensorShape.h"

using hahaha::backend::Device;
using hahaha::backend::DeviceType;
using hahaha::common::f32;
using hahaha::common::f64;
using hahaha::common::i16;
using hahaha::common::i32;
using hahaha::common::i64;
using hahaha::common::i8;
using hahaha::common::u16;
using hahaha::common::u32;
using hahaha::common::u64;
using hahaha::common::u8;
using hahaha::math::NestedData;
using hahaha::math::TensorShape;
using hahaha::math::TensorWrapper;

// Define all supported numeric types
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T> class TensorWrapperTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to check if type is signed integer
    static constexpr bool isSignedInteger() {
        return std::is_integral_v<T> && std::is_signed_v<T>;
    }

    // Helper to check if type is unsigned integer
    static constexpr bool isUnsignedInteger() {
        return std::is_integral_v<T> && std::is_unsigned_v<T>;
    }

    // Get maximum value for the type
    static constexpr T maxValue() {
        return std::numeric_limits<T>::max();
    }

    // Get minimum value for the type
    static constexpr T minValue() {
        return std::numeric_limits<T>::lowest();
    }

    // Get epsilon for floating point comparison
    static constexpr T epsilon() {
        if constexpr (isFloatingPoint()) {
            return std::numeric_limits<T>::epsilon();
        } else {
            return T(0);
        }
    }

    // Helper to compare values with appropriate tolerance
    void expectNear(T expected, T actual, T tolerance = T(0)) {
        if constexpr (isFloatingPoint()) {
            if (tolerance == T(0)) {
                tolerance = epsilon() * T(10); // Default tolerance
            }
            EXPECT_NEAR(expected, actual, tolerance);
        } else {
            EXPECT_EQ(expected, actual);
        }
    }

    // Helper to check if value is NaN (only for floating point)
    bool isNaN(T value) {
        if constexpr (isFloatingPoint()) {
            return std::isnan(value);
        }
        return false;
    }

    // Helper to check if value is Inf (only for floating point)
    bool isInf(T value) {
        if constexpr (isFloatingPoint()) {
            return std::isinf(value);
        }
        return false;
    }
};

TYPED_TEST_SUITE(TensorWrapperTypedTest, NumericTypes);

// ============================================================================
// Basic Construction Tests
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Constructor_Default_CreatesEmptyTensor) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor;
    EXPECT_EQ(tensor.getTotalSize(), 0);
    EXPECT_EQ(tensor.getShape().size(), 0);
}

TYPED_TEST(TensorWrapperTypedTest,
           Constructor_ShapeInitValue_CreatesCorrectTensor) {
    using T = typename TestFixture::Type;
    T initValue = T(42);
    TensorWrapper<T> tensor(TensorShape({2, 3}), initValue);
    EXPECT_EQ(tensor.getTotalSize(), 6);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 3);
    EXPECT_EQ(tensor.at({0, 0}), initValue);
    EXPECT_EQ(tensor.at({1, 2}), initValue);
}

TYPED_TEST(TensorWrapperTypedTest,
           Constructor_NestedData_2D_CorrectlyInitializes) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor.getTotalSize(), 4);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.at({0, 0}), T(1));
    EXPECT_EQ(tensor.at({1, 1}), T(4));
}

TYPED_TEST(TensorWrapperTypedTest, Constructor_Scalar_CorrectlyInitializes) {
    using T = typename TestFixture::Type;
    T scalarValue = T(99);
    TensorWrapper<T> tensor(scalarValue);
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 0); // Scalar tensor has 0 dimensions
    EXPECT_EQ(tensor.at({}), scalarValue);
}

// ============================================================================
// Arithmetic Operations: Addition
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Add_TwoTensors_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorWrapper<T> t2(NestedData<T>{{T(5), T(6)}, {T(7), T(8)}});
    auto res = t1 + t2;
    EXPECT_EQ(res.at({0, 0}), T(6));
    EXPECT_EQ(res.at({1, 1}), T(12));
}

TYPED_TEST(TensorWrapperTypedTest, Add_TensorScalar_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(10);
    auto res = tensor + scalar;
    EXPECT_EQ(res.at({0, 0}), T(11));
    EXPECT_EQ(res.at({0, 1}), T(12));
}

TYPED_TEST(TensorWrapperTypedTest, Add_ScalarTensor_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(10);
    auto res = scalar + tensor;
    EXPECT_EQ(res.at({0, 0}), T(11));
    EXPECT_EQ(res.at({0, 1}), T(12));
}

TYPED_TEST(TensorWrapperTypedTest, InPlaceAdd_TwoTensors_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorWrapper<T> t2(NestedData<T>{{T(5), T(6)}, {T(7), T(8)}});
    t1 += t2;
    EXPECT_EQ(t1.at({0, 0}), T(6));
    EXPECT_EQ(t1.at({1, 1}), T(12));
}

// ============================================================================
// Arithmetic Operations: Subtraction
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Subtract_TwoTensors_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(5), T(6)}});
    TensorWrapper<T> t2(NestedData<T>{{T(1), T(2)}});
    auto res = t1 - t2;
    EXPECT_EQ(res.at({0, 0}), T(4));
    EXPECT_EQ(res.at({0, 1}), T(4));
}

TYPED_TEST(TensorWrapperTypedTest, Subtract_TensorScalar_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(5);
    auto res = tensor - scalar;
    EXPECT_EQ(res.at({0, 0}), T(5));
    EXPECT_EQ(res.at({0, 1}), T(15));
}

TYPED_TEST(TensorWrapperTypedTest, Subtract_ScalarTensor_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(30);
    auto res = scalar - tensor;
    EXPECT_EQ(res.at({0, 0}), T(20));
    EXPECT_EQ(res.at({0, 1}), T(10));
}

// ============================================================================
// Arithmetic Operations: Multiplication
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Multiply_TwoTensors_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{{T(3), T(4)}});
    auto res = t1 * t2;
    EXPECT_EQ(res.at({0, 0}), T(3));
    EXPECT_EQ(res.at({0, 1}), T(8));
}

TYPED_TEST(TensorWrapperTypedTest, Multiply_TensorScalar_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(5);
    auto res = tensor * scalar;
    EXPECT_EQ(res.at({0, 0}), T(5));
    EXPECT_EQ(res.at({0, 1}), T(10));
}

TYPED_TEST(TensorWrapperTypedTest, Multiply_ScalarTensor_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(5);
    auto res = scalar * tensor;
    EXPECT_EQ(res.at({0, 0}), T(5));
    EXPECT_EQ(res.at({0, 1}), T(10));
}

// ============================================================================
// Arithmetic Operations: Division
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Divide_TwoTensors_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(10), T(20)}});
    TensorWrapper<T> t2(NestedData<T>{{T(2), T(4)}});
    auto res = t1 / t2;
    this->expectNear(T(5), res.at({0, 0}));
    this->expectNear(T(5), res.at({0, 1}));
}

TYPED_TEST(TensorWrapperTypedTest, Divide_TensorScalar_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(2);
    auto res = tensor / scalar;
    this->expectNear(T(5), res.at({0, 0}));
    this->expectNear(T(10), res.at({0, 1}));
}

TYPED_TEST(TensorWrapperTypedTest, Divide_ScalarTensor_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(100);
    auto res = scalar / tensor;
    this->expectNear(T(10), res.at({0, 0}));
    this->expectNear(T(5), res.at({0, 1}));
}

TYPED_TEST(TensorWrapperTypedTest,
           Divide_TwoTensors_DivisionByZero_ThrowsRuntimeError) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t_zero(NestedData<T>{{T(0), T(1)}});
    EXPECT_THROW(t1 / t_zero, std::runtime_error);
}

TYPED_TEST(TensorWrapperTypedTest,
           Divide_TensorScalar_DivisionByZero_ThrowsRuntimeError) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}});
    EXPECT_THROW(tensor / T(0), std::runtime_error);
}

TYPED_TEST(TensorWrapperTypedTest,
           Divide_ScalarTensor_DivisionByZero_ThrowsRuntimeError) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_with_zero(NestedData<T>{{T(0), T(1)}});
    EXPECT_THROW(T(10) / tensor_with_zero, std::runtime_error);
}

// ============================================================================
// Sum Operation
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Sum_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    T result = tensor.sum();
    EXPECT_EQ(result, T(10));
}

TYPED_TEST(TensorWrapperTypedTest, Sum_OnEmptyTensor_ReturnsZero) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor;
    EXPECT_EQ(tensor.sum(), T(0));
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Matmul_ValidMatrices_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> matrix_a(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorWrapper<T> matrix_b(NestedData<T>{{T(5), T(6)}, {T(7), T(8)}});
    auto matrix_c = matrix_a.matmul(matrix_b);
    // 1*5 + 2*7 = 5 + 14 = 19
    this->expectNear(T(19), matrix_c.at({0, 0}));
    // 1*6 + 2*8 = 6 + 16 = 22
    this->expectNear(T(22), matrix_c.at({0, 1}));
    // 3*5 + 4*7 = 15 + 28 = 43
    this->expectNear(T(43), matrix_c.at({1, 0}));
    // 3*6 + 4*8 = 18 + 32 = 50
    this->expectNear(T(50), matrix_c.at({1, 1}));
}

// ============================================================================
// Boundary Cases: Integer Overflow
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Add_IntegerOverflow_Behavior) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isUnsignedInteger()
                  || TestFixture::isSignedInteger()) {
        T maxVal = this->maxValue();
        TensorWrapper<T> t1(NestedData<T>{{maxVal}});
        TensorWrapper<T> t2(NestedData<T>{{T(1)}});

        auto res = t1 + t2;
        if constexpr (TestFixture::isUnsignedInteger()) {
            // Unsigned overflow wraps around: maxVal + 1 wraps to 0
            // This is well-defined behavior in C++ for unsigned types
            // Calculate expected using same type to ensure proper wrapping
            T a = maxVal;
            T b = T(1);
            T expected = a + b; // This will wrap for unsigned types
            EXPECT_EQ(res.at({0, 0}), expected)
                << "Type: " << typeid(T).name()
                << ", maxVal = " << static_cast<unsigned long long>(maxVal)
                << ", expected = " << static_cast<unsigned long long>(expected)
                << ", actual = "
                << static_cast<unsigned long long>(res.at({0}));
        } else {
            // For signed integers, overflow is undefined behavior,
            // but we just verify it doesn't crash
            EXPECT_GE(res.getTotalSize(), 1);
        }
    }
}

TYPED_TEST(TensorWrapperTypedTest, Multiply_IntegerOverflow_Behavior) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isUnsignedInteger()
                  || TestFixture::isSignedInteger()) {
        // Use values that would overflow when multiplied
        T largeVal = this->maxValue() / T(2);
        TensorWrapper<T> t1(NestedData<T>{{largeVal}});
        TensorWrapper<T> t2(NestedData<T>{{T(3)}});

        // This may overflow, but we test that it doesn't crash
        auto res = t1 * t2;
        // Just verify the operation completes without throwing
        EXPECT_GE(res.getTotalSize(), 1);
    }
}

// ============================================================================
// Boundary Cases: Floating Point Special Values
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, FloatingPoint_Infinity_Handling) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        TensorWrapper<T> t1_inf(NestedData<T>{{inf}});
        TensorWrapper<T> t2_finite(NestedData<T>{{T(1)}});

        // Inf + finite = Inf
        auto res_add = t1_inf + t2_finite;
        EXPECT_TRUE(this->isInf(res_add.at({0, 0})));

        // Inf * finite (positive) = Inf
        auto res_mul = t1_inf * t2_finite;
        EXPECT_TRUE(this->isInf(res_mul.at({0, 0})));

        // Test with negative finite value
        TensorWrapper<T> t3_neg(NestedData<T>{{T(-1)}});
        auto res_add_neg = t1_inf + t3_neg;
        // Inf + (-finite) is still Inf (not -Inf in this case, but could be
        // implementation dependent)
        EXPECT_TRUE(this->isInf(res_add_neg.at({0, 0})));
    }
}

TYPED_TEST(TensorWrapperTypedTest, FloatingPoint_NaN_Handling) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T nan = std::numeric_limits<T>::quiet_NaN();
        TensorWrapper<T> t1(NestedData<T>{{nan, T(1)}});
        TensorWrapper<T> t2(NestedData<T>{{T(1), T(2)}});

        // NaN + finite = NaN
        auto res_add = t1 + t2;
        EXPECT_TRUE(this->isNaN(res_add.at({0, 0})));
        EXPECT_FALSE(this->isNaN(res_add.at({0, 1})));

        // NaN * finite = NaN
        auto res_mul = t1 * t2;
        EXPECT_TRUE(this->isNaN(res_mul.at({0, 0})));
        EXPECT_FALSE(this->isNaN(res_mul.at({0, 1})));
    }
}

TYPED_TEST(TensorWrapperTypedTest, FloatingPoint_InfinityDivision_Handling) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        TensorWrapper<T> t1_inf(NestedData<T>{{inf}});
        TensorWrapper<T> t2_finite(NestedData<T>{{T(2)}});

        // Inf / finite = Inf
        auto res = t1_inf / t2_finite;
        EXPECT_TRUE(this->isInf(res.at({0, 0})));

        // Test division by zero separately
        TensorWrapper<T> t3_one(NestedData<T>{{T(1)}});
        TensorWrapper<T> t4_zero(NestedData<T>{{T(0)}});
        // 1 / 0 should throw runtime_error
        EXPECT_THROW(t3_one / t4_zero, std::runtime_error);
    }
}

TYPED_TEST(TensorWrapperTypedTest, FloatingPoint_ZeroDivision_Throws) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        TensorWrapper<T> t1(NestedData<T>{{T(1.0), T(2.0)}});
        TensorWrapper<T> t_zero(NestedData<T>{{T(0.0), T(1.0)}});
        EXPECT_THROW(t1 / t_zero, std::runtime_error);
    }
}

// ============================================================================
// Precision Tests: Integer Division Rounding
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, IntegerDivision_RoundingBehavior) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isSignedInteger()
                  || TestFixture::isUnsignedInteger()) {
        // Test integer division truncation
        TensorWrapper<T> t1(NestedData<T>{{T(7), T(10)}});
        TensorWrapper<T> t2(NestedData<T>{{T(3), T(3)}});
        auto res = t1 / t2;
        // 7 / 3 = 2 (truncated)
        EXPECT_EQ(res.at({0, 0}), T(2));
        // 10 / 3 = 3 (truncated)
        EXPECT_EQ(res.at({0, 1}), T(3));
    }
}

// ============================================================================
// Precision Tests: Floating Point Precision
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, FloatingPoint_Precision) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        // Test that floating point operations maintain reasonable precision
        T val1 = T(0.1);
        T val2 = T(0.2);
        TensorWrapper<T> t1(NestedData<T>{{val1}});
        TensorWrapper<T> t2(NestedData<T>{{val2}});
        auto res = t1 + t2;
        // 0.1 + 0.2 should be approximately 0.3
        this->expectNear(T(0.3), res.at({0, 0}), T(0.0001));
    }
}

// ============================================================================
// Edge Cases: Very Large Values
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, VeryLargeValues_Handling) {
    using T = typename TestFixture::Type;
    T largeVal = this->maxValue() / T(2);
    TensorWrapper<T> t1(NestedData<T>{{largeVal}});
    TensorWrapper<T> t2(NestedData<T>{{T(1)}});

    auto res = t1 + t2;
    // Should not crash and should produce a result
    EXPECT_GE(res.getTotalSize(), 1);
}

TYPED_TEST(TensorWrapperTypedTest, VerySmallValues_Handling) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T smallVal = std::numeric_limits<T>::min();
        TensorWrapper<T> t1(NestedData<T>{{smallVal}});
        TensorWrapper<T> t2(NestedData<T>{{T(1)}});

        auto res = t1 + t2;
        // Should handle very small values correctly
        this->expectNear(T(1) + smallVal, res.at({0, 0}));
    } else {
        T smallVal = this->minValue();
        TensorWrapper<T> t1(NestedData<T>{{smallVal}});
        TensorWrapper<T> t2(NestedData<T>{{T(1)}});

        auto res = t1 + t2;
        // For signed integers, min + 1 should be handled
        EXPECT_GE(res.getTotalSize(), 1);
    }
}

// ============================================================================
// Reshape Operation
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Reshape_ValidNewShape_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_orig(
        NestedData<T>{T(1), T(2), T(3), T(4), T(5), T(6)});
    auto tensor_reshaped = tensor_orig.reshape({2, 3});
    EXPECT_EQ(tensor_reshaped.getShape().size(), 2);
    EXPECT_EQ(tensor_reshaped.at({0, 0}), T(1));
    EXPECT_EQ(tensor_reshaped.at({1, 2}), T(6));
}

// ============================================================================
// Transpose Operation
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Transpose_Valid2DTensor_CorrectResult) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_orig(
        NestedData<T>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}});
    auto tensor_transposed = tensor_orig.transpose();
    EXPECT_EQ(tensor_transposed.getShape()[0], 3);
    EXPECT_EQ(tensor_transposed.getShape()[1], 2);
    EXPECT_EQ(tensor_transposed.at({0, 0}), T(1));
    EXPECT_EQ(tensor_transposed.at({0, 1}), T(4));
    EXPECT_EQ(tensor_transposed.at({2, 1}), T(6));
}

TYPED_TEST(TensorWrapperTypedTest,
           Transpose_Non2DTensor_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_1d(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_THROW(tensor_1d.transpose(), std::invalid_argument);
}

// ============================================================================
// Detailed Construction Tests
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest,
           Constructor_NestedData_3D_CorrectlyInitializes) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}},
                                          {{T(5), T(6)}, {T(7), T(8)}}});
    EXPECT_EQ(tensor.getTotalSize(), 8);
    EXPECT_EQ(tensor.getShape().size(), 3);
    EXPECT_EQ(tensor.at({0, 0, 0}), T(1));
    EXPECT_EQ(tensor.at({1, 1, 1}), T(8));
}

TYPED_TEST(TensorWrapperTypedTest,
           Constructor_NestedData_IrregularShape_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    EXPECT_THROW(TensorWrapper<T>(NestedData<T>{{T(1)}, {T(1), T(2)}}),
                 std::invalid_argument);
    if constexpr (TestFixture::isFloatingPoint()) {
        EXPECT_THROW(
            TensorWrapper<T>(NestedData<T>{{T(1.0), T(2.0)}, {T(3.0)}}),
            std::invalid_argument);
    }
}

TYPED_TEST(TensorWrapperTypedTest, Constructor_Vector_CorrectlyInitializes) {
    using T = typename TestFixture::Type;
    std::vector<T> vec = {T(1), T(2), T(3)};
    TensorWrapper<T> tensor(vec);
    EXPECT_EQ(tensor.getTotalSize(), 3);
    EXPECT_EQ(tensor.getShape().size(), 1);
    EXPECT_EQ(tensor.at({0}), T(1));
    EXPECT_EQ(tensor.at({2}), T(3));
}

TYPED_TEST(TensorWrapperTypedTest,
           Constructor_ShapeInitValueDevice_CreatesCorrectTensor) {
    using T = typename TestFixture::Type;
    T initValue = T(42);
    TensorWrapper<T> tensor(
        TensorShape({2, 3}), initValue, Device(DeviceType::CPU, 0));
    EXPECT_EQ(tensor.getTotalSize(), 6);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 3);
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
    EXPECT_EQ(tensor.at({0, 0}), initValue);
    EXPECT_EQ(tensor.at({1, 2}), initValue);
}

TYPED_TEST(TensorWrapperTypedTest,
           Constructor_ShapeDevice_CreatesCorrectTensorWithDefaultInit) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(TensorShape({2, 2}), Device(DeviceType::CPU, 0));
    EXPECT_EQ(tensor.getTotalSize(), 4);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
    EXPECT_EQ(tensor.at({0, 0}), T(0)); // Default init value
}

// ============================================================================
// Property and Accessor Tests
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, GetShape_ReturnsCorrectDimensionsAndSize) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_2d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor_2d.getShape()[0], 2);
    EXPECT_EQ(tensor_2d.getShape()[1], 2);
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);

    TensorWrapper<T> tensor_3d(NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}},
                                             {{T(5), T(6)}, {T(7), T(8)}}});
    EXPECT_EQ(tensor_3d.getShape()[0], 2);
    EXPECT_EQ(tensor_3d.getShape()[1], 2);
    EXPECT_EQ(tensor_3d.getShape()[2], 2);
    EXPECT_EQ(tensor_3d.getTotalSize(), 8);
}

TYPED_TEST(TensorWrapperTypedTest, GetStride_ReturnsCorrectValues) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_2d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor_2d.getStride().getStrides()[0], 2);
    EXPECT_EQ(tensor_2d.getStride().getStrides()[1], 1);

    TensorWrapper<T> tensor_3d(NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}},
                                             {{T(5), T(6)}, {T(7), T(8)}}});
    EXPECT_EQ(tensor_3d.getStride().getStrides()[0], 4);
    EXPECT_EQ(tensor_3d.getStride().getStrides()[1], 2);
    EXPECT_EQ(tensor_3d.getStride().getStrides()[2], 1);
}

TYPED_TEST(TensorWrapperTypedTest, GetDevice_ReturnsCorrectDefaultDevice) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(TensorShape({2, 2}));
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
    EXPECT_EQ(tensor.getDevice().id, 0);
}

TYPED_TEST(TensorWrapperTypedTest,
           ElementAccess_ReadWrite_CorrectlyModifiesAndRetrieves) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor.at({0, 0}), T(1));
    tensor.at({1, 1}) = T(10);
    EXPECT_EQ(tensor.at({1, 1}), T(10));
}

TYPED_TEST(TensorWrapperTypedTest, ElementAccess_OutOfBounds_ThrowsOutOfRange) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_THROW(tensor.at({0, 0, 0}), std::out_of_range); // Dimension mismatch
    EXPECT_THROW(tensor.at({2, 0}), std::out_of_range); // Index out of bounds
}

TYPED_TEST(TensorWrapperTypedTest,
           ElementAccess_ConstAt_WorksAndThrowsOnMismatch) {
    using T = typename TestFixture::Type;
    const TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor.at({0, 1}), T(2));
    EXPECT_THROW(tensor.at({0}), std::out_of_range);
}

TYPED_TEST(TensorWrapperTypedTest, GetDimensions_ReturnsCorrectCount) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_scalar(T(1));
    EXPECT_EQ(tensor_scalar.getDimensions(), 0);
    TensorWrapper<T> tensor_1d(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_EQ(tensor_1d.getDimensions(), 1);
    TensorWrapper<T> tensor_2d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor_2d.getDimensions(), 2);
}

TYPED_TEST(TensorWrapperTypedTest, GetTotalSize_ReturnsCorrectCount) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_scalar(NestedData<T>{T(1)});
    EXPECT_EQ(tensor_scalar.getTotalSize(), 1);
    TensorWrapper<T> tensor_1d(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_EQ(tensor_1d.getTotalSize(), 3);
    TensorWrapper<T> tensor_2d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);
}

TYPED_TEST(TensorWrapperTypedTest, GetRawData_ReturnsSharedPtrToBuffer) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(TensorShape({2, 2}), T(7));
    auto& raw = tensor.getRawData();
    EXPECT_NE(raw.get(), nullptr);
}

// ============================================================================
// Move/Copy Semantics Tests
// ============================================================================

TYPED_TEST(
    TensorWrapperTypedTest,
    MoveConstructor_TransfersOwnershipAndLeavesSourceValidButUnspecified) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_orig(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorWrapper<T> tensor_moved(std::move(tensor_orig));
    EXPECT_EQ(tensor_moved.getTotalSize(), 4);
    // tensor_orig's state is valid but unspecified, can check for some
    // properties
    EXPECT_EQ(tensor_orig.getTotalSize(),
              0); // Moved-from objects usually have empty state
}

TYPED_TEST(TensorWrapperTypedTest, MoveAssignment_TransfersOwnership) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_a(NestedData<T>{{T(1), T(1)}});
    TensorWrapper<T> tensor_b(NestedData<T>{{T(2), T(2)}});
    tensor_a = std::move(tensor_b);
    EXPECT_EQ(tensor_a.at({0, 0}), T(2));
    EXPECT_EQ(tensor_b.getTotalSize(), 0); // Moved-from object is empty
}

TYPED_TEST(TensorWrapperTypedTest, CopyConstructor_DeepCopiesData) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> original(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> copy = original; // Uses copy constructor
    original.at({0, 0}) = T(10);
    EXPECT_EQ(copy.at({0, 0}),
              T(1)); // Original modification should not affect copy
}

TYPED_TEST(TensorWrapperTypedTest, Clone_CreatesDeepCopy) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> original(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    auto copy = original.clone();
    EXPECT_EQ(copy.at({0, 0}), T(1));
    original.at({0, 0}) = T(100);
    EXPECT_EQ(copy.at({0, 0}), T(1)); // Copy should not change
}

// ============================================================================
// Error Handling Tests: Shape Mismatch
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest,
           Add_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           Subtract_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 - t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           Multiply_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 * t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           Divide_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 / t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           InPlaceAdd_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_THROW(t1 += t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           InPlaceSubtract_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_THROW(t1 -= t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           InPlaceMultiply_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_THROW(t1 *= t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           InPlaceDivide_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> t1(NestedData<T>{{T(1), T(2)}});
    TensorWrapper<T> t2(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_THROW(t1 /= t2, std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest, Reshape_SizeMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor_orig(NestedData<T>{T(1), T(2), T(3), T(4)});
    EXPECT_THROW(tensor_orig.reshape({3}),
                 std::invalid_argument); // Total size 3 != 4
    EXPECT_THROW(tensor_orig.reshape({2, 3}),
                 std::invalid_argument); // Total size 6 != 4
}

TYPED_TEST(TensorWrapperTypedTest, Matmul_Non2DTensors_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> matrix_a(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorWrapper<T> tensor_1d(NestedData<T>{T(1), T(2)});
    EXPECT_THROW(matrix_a.matmul(tensor_1d), std::invalid_argument);
    EXPECT_THROW(tensor_1d.matmul(matrix_a), std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           Matmul_InnerDimensionMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> matrix_a(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorWrapper<T> matrix_d(
        NestedData<T>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}});
    EXPECT_THROW(matrix_a.matmul(matrix_d),
                 std::invalid_argument); // (2x2) @ (3x2) -> inner dims 2 != 3
}

// ============================================================================
// Sum with Axes Tests
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Sum_WithAxes_EmptyAxes_ReturnsClone) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    auto res = tensor.sum({});
    EXPECT_EQ(res.getShape(), tensor.getShape());
    EXPECT_EQ(res.at({0, 0}), T(1));
    tensor.at({0, 0}) = T(100);
    EXPECT_EQ(res.at({0, 0}), T(1)); // clone, not view
}

TYPED_TEST(TensorWrapperTypedTest,
           Sum_WithAxes_InvalidAxis_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(
        NestedData<T>{{T(1), T(2)}, {T(3), T(4)}}); // rank 2
    EXPECT_THROW(tensor.sum({2}), std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest, Sum_WithAxes_ReducesCorrectly) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(
        NestedData<T>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}}); // 2x3
    // Sum over axis 0: {1+4, 2+5, 3+6} = {5, 7, 9}
    auto sum0 = tensor.sum({0});
    EXPECT_EQ(sum0.getShape().size(), 1);
    EXPECT_EQ(sum0.getShape()[0], 3);
    EXPECT_EQ(sum0.at({0}), T(5));
    EXPECT_EQ(sum0.at({2}), T(9));

    // Sum over axis 1: {1+2+3, 4+5+6} = {6, 15}
    auto sum1 = tensor.sum({1});
    EXPECT_EQ(sum1.getShape().size(), 1);
    EXPECT_EQ(sum1.getShape()[0], 2);
    EXPECT_EQ(sum1.at({0}), T(6));
    EXPECT_EQ(sum1.at({1}), T(15));
}

TYPED_TEST(TensorWrapperTypedTest, Sum_WithAxes_KeepDims) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(
        NestedData<T>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}}); // 2x3
    // Sum over axis 0, keep dims: shape {1, 3}
    auto sum0 = tensor.sum({0}, true);
    EXPECT_EQ(sum0.getShape().size(), 2);
    EXPECT_EQ(sum0.getShape()[0], 1);
    EXPECT_EQ(sum0.getShape()[1], 3);
    EXPECT_EQ(sum0.at({0, 0}), T(5));
}

TYPED_TEST(TensorWrapperTypedTest, Sum_WithAxes_KeepDimsFalse_ReducesRank) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}}); // 2x2
    auto sum0 = tensor.sum({0}, false); // Should be {2}
    EXPECT_EQ(sum0.getShape().size(), 1);
    EXPECT_EQ(sum0.getShape()[0], 2);
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest,
           BroadcastTo_SameShape_ReturnsViewWithSameStride) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> src(
        NestedData<T>{{T(1), T(2)}, {T(3), T(4)}}); // shape (2,2)
    auto view = src.broadcastTo(TensorShape({2, 2}));

    EXPECT_EQ(view.getShape().size(), 2);
    EXPECT_EQ(view.getShape()[0], 2);
    EXPECT_EQ(view.getShape()[1], 2);
    EXPECT_EQ(view.getStride().toString(), src.getStride().toString());
    EXPECT_EQ(view.getRawData().get(), src.getRawData().get());
}

TYPED_TEST(TensorWrapperTypedTest, BroadcastTo_PrefixDim_InsertsZeroStride) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> src(NestedData<T>{T(1), T(2), T(3)}); // shape (3)
    auto view = src.broadcastTo(TensorShape({2, 3}));      // view shape (2,3)
    EXPECT_EQ(view.getShape().size(), 2);
    EXPECT_EQ(view.getShape()[0], 2);
    EXPECT_EQ(view.getShape()[1], 3);

    // prefix dim stride is 0, last dim stride is original 1
    ASSERT_EQ(view.getStride().getStrides().size(), 2);
    EXPECT_EQ(view.getStride().getStrides()[0], 0);
    EXPECT_EQ(view.getStride().getStrides()[1], 1);
    EXPECT_EQ(view.getRawData().get(), src.getRawData().get());
}

TYPED_TEST(TensorWrapperTypedTest, BroadcastTo_DimOneGetsZeroStride) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> src(NestedData<T>{{T(1), T(2), T(3)}}); // shape (1,3)
    auto view = src.broadcastTo(TensorShape({2, 3}));        // (2,3)
    ASSERT_EQ(view.getStride().getStrides().size(), 2);
    EXPECT_EQ(view.getStride().getStrides()[0], 0);
    EXPECT_EQ(view.getStride().getStrides()[1],
              src.getStride().getStrides()[1]);
}

TYPED_TEST(TensorWrapperTypedTest,
           BroadcastTo_Incompatible_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> src(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}}); // (2,2)
    EXPECT_THROW(src.broadcastTo(TensorShape({3, 2})), std::invalid_argument);
}

TYPED_TEST(TensorWrapperTypedTest,
           BroadcastTo_TargetRankSmaller_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> src(NestedData<T>{{T(1), T(2), T(3)}}); // (1,3)
    EXPECT_THROW(src.broadcastTo(TensorShape({3})), std::invalid_argument);
}

// ============================================================================
// Device and Special Operations
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Clear_ResetsTensorDataToZero) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor({2, 2}, T(5)); // All elements are 5
    tensor.clear();
    EXPECT_EQ(tensor.at({0, 0}), T(0));
    EXPECT_EQ(tensor.at({1, 1}), T(0));
}

TYPED_TEST(TensorWrapperTypedTest, Clear_OnEmptyTensor_NoThrow) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor;
    EXPECT_NO_THROW(tensor.clear());
    EXPECT_EQ(tensor.getTotalSize(), 0);
}

TYPED_TEST(TensorWrapperTypedTest, To_SameDevice_NoChangeAndNoThrow) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor({2, 2}, T(1), Device(DeviceType::CPU, 0));
    // Should not throw, device remains the same
    EXPECT_NO_THROW(tensor.to(Device(DeviceType::CPU, 0)));
    EXPECT_EQ(tensor.getDevice().type, DeviceType::CPU);
}

TYPED_TEST(TensorWrapperTypedTest,
           To_DifferentDevice_NotImplemented_ThrowsRuntimeError) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> tensor({2, 2}, T(1), Device(DeviceType::CPU, 0));
    EXPECT_THROW(tensor.to(Device(DeviceType::GPU, 0)), std::runtime_error);
}

TYPED_TEST(TensorWrapperTypedTest, DeviceMismatch_ThrowsInvalidArgument) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> a(TensorShape({2, 2}), T(1), Device(DeviceType::CPU, 0));
    TensorWrapper<T> b(TensorShape({2, 2}), T(1), Device(DeviceType::SIMD, 0));
    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a - b, std::invalid_argument);
    EXPECT_THROW(a * b, std::invalid_argument);
    EXPECT_THROW(a / b, std::invalid_argument);
}

// ============================================================================
// Scalar Tensor Operations
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, ScalarTensorTensorOps_TwoScalars_Work) {
    using T = typename TestFixture::Type;
    TensorWrapper<T> a(T(2));
    TensorWrapper<T> b(T(3));
    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;
    auto quot = b / a;
    this->expectNear(T(5), sum.at({}));
    if constexpr (TestFixture::isSignedInteger()
                  || TestFixture::isFloatingPoint()) {
        this->expectNear(T(-1), diff.at({}));
    }
    this->expectNear(T(6), prod.at({}));
    this->expectNear(T(1) + T(1) / T(2),
                     quot.at({})); // 3/2 = 1.5 for float, 1 for int
}

// ============================================================================
// Axpy Operation
// ============================================================================

TYPED_TEST(TensorWrapperTypedTest, Axpy_UpdatesInPlace) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        // y = a * x + y
        TensorWrapper<T> x(NestedData<T>{T(1.0), T(2.0)});
        TensorWrapper<T> y(NestedData<T>{T(3.0), T(4.0)});
        T a = T(2.0);

        y.axpy(a, x); // y = 2*x + y = {2+3, 4+4} = {5, 8}

        this->expectNear(T(5.0), y.at({0}));
        this->expectNear(T(8.0), y.at({1}));
        EXPECT_EQ(x.at({0}), T(1.0)); // x unchanged
    }
}

TYPED_TEST(TensorWrapperTypedTest, Axpy_ShapeMismatch_Throws) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        TensorWrapper<T> x(NestedData<T>{T(1.0)});
        TensorWrapper<T> y(NestedData<T>{T(3.0), T(4.0)});
        EXPECT_THROW(y.axpy(T(1.0), x), std::invalid_argument);
    }
}
