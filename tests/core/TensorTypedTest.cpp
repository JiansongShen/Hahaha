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
#include <type_traits>
#include <typeinfo>

#include "backend/gpu/GPUDevice.h"
#include "common/definitions.h"
#include "math/ds/NestedData.h"
#include "public/Tensor.h"

using hahaha::Tensor;
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

// Define all supported numeric types
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T> class TensorTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(TensorTypedTest, NumericTypes);

// ============================================================================
// Basic Construction Tests
// ============================================================================

TYPED_TEST(TensorTypedTest, Constructor_FromNestedData_CreatesCorrectTensor) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 2);
    EXPECT_EQ(tensor.at({0, 0}), T(1));
    EXPECT_EQ(tensor.at({1, 1}), T(4));
}

TYPED_TEST(TensorTypedTest, Constructor_FromScalar_CreatesScalarTensor) {
    using T = TestFixture::Type;
    T scalarValue = T(99);
    Tensor<T> tensor(scalarValue);
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 0); // Scalar tensor
    EXPECT_EQ(tensor.at({}), scalarValue);
}

TYPED_TEST(TensorTypedTest, BuildFromVector_Creates1DTensor) {
    using T = TestFixture::Type;
    auto t = Tensor<T>::buildFromVector({T(1), T(2), T(3)});
    EXPECT_EQ(t.getShape().size(), 1);
    EXPECT_EQ(t.getShape()[0], 3);
    EXPECT_EQ(t.at({0}), T(1));
    EXPECT_EQ(t.at({2}), T(3));
}

// ============================================================================
// Arithmetic Operations: Addition
// ============================================================================

TYPED_TEST(TensorTypedTest, Add_TwoTensors_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    Tensor<T> t2(NestedData<T>{{T(5), T(6)}, {T(7), T(8)}});
    auto res = t1 + t2;
    EXPECT_EQ(res.at({0, 0}), T(6));
    EXPECT_EQ(res.at({1, 1}), T(12));
}

TYPED_TEST(TensorTypedTest, Add_TensorScalar_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(10);
    auto res = tensor + scalar;
    EXPECT_EQ(res.at({0, 0}), T(11));
    EXPECT_EQ(res.at({0, 1}), T(12));
}

TYPED_TEST(TensorTypedTest, Add_ScalarTensor_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(10);
    auto res = scalar + tensor;
    EXPECT_EQ(res.at({0, 0}), T(11));
    EXPECT_EQ(res.at({0, 1}), T(12));
}

// ============================================================================
// Arithmetic Operations: Subtraction
// ============================================================================

TYPED_TEST(TensorTypedTest, Subtract_TwoTensors_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(5), T(6)}});
    Tensor<T> t2(NestedData<T>{{T(1), T(2)}});
    auto res = t1 - t2;
    EXPECT_EQ(res.at({0, 0}), T(4));
    EXPECT_EQ(res.at({0, 1}), T(4));
}

TYPED_TEST(TensorTypedTest, Subtract_TensorScalar_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(5);
    auto res = tensor - scalar;
    EXPECT_EQ(res.at({0, 0}), T(5));
    EXPECT_EQ(res.at({0, 1}), T(15));
}

TYPED_TEST(TensorTypedTest, Subtract_ScalarTensor_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(30);
    auto res = scalar - tensor;
    EXPECT_EQ(res.at({0, 0}), T(20));
    EXPECT_EQ(res.at({0, 1}), T(10));
}

// ============================================================================
// Arithmetic Operations: Multiplication
// ============================================================================

TYPED_TEST(TensorTypedTest, Multiply_TwoTensors_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}});
    Tensor<T> t2(NestedData<T>{{T(3), T(4)}});
    auto res = t1 * t2;
    EXPECT_EQ(res.at({0, 0}), T(3));
    EXPECT_EQ(res.at({0, 1}), T(8));
}

TYPED_TEST(TensorTypedTest, Multiply_TensorScalar_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}});
    T scalar = T(5);
    auto res = tensor * scalar;
    EXPECT_EQ(res.at({0, 0}), T(5));
    EXPECT_EQ(res.at({0, 1}), T(10));
}

// ============================================================================
// Arithmetic Operations: Division
// ============================================================================

TYPED_TEST(TensorTypedTest, Divide_TwoTensors_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(10), T(20)}});
    Tensor<T> t2(NestedData<T>{{T(2), T(4)}});
    auto res = t1 / t2;
    this->expectNear(T(5), res.at({0, 0}));
    this->expectNear(T(5), res.at({0, 1}));
}

TYPED_TEST(TensorTypedTest, Divide_TensorScalar_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(10), T(20)}});
    T scalar = T(2);
    auto res = tensor / scalar;
    this->expectNear(T(5), res.at({0, 0}));
    this->expectNear(T(10), res.at({0, 1}));
}

TYPED_TEST(TensorTypedTest,
           Divide_TwoTensors_DivisionByZero_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}});
    Tensor<T> t_zero(NestedData<T>{{T(0), T(1)}});
    EXPECT_THROW(t1 / t_zero, std::runtime_error);
}

TYPED_TEST(TensorTypedTest,
           Divide_TensorScalar_DivisionByZero_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}});
    EXPECT_THROW(tensor / T(0), std::runtime_error);
}

// ============================================================================
// Unary Operations
// ============================================================================

TYPED_TEST(TensorTypedTest, UnaryNegation_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(-2)}});
    auto neg_tensor = -tensor;
    EXPECT_EQ(neg_tensor.at({0, 0}), T(-1));
    if constexpr (TestFixture::isSignedInteger()
                  || TestFixture::isFloatingPoint()) {
        EXPECT_EQ(neg_tensor.at({0, 1}), T(2));
    }
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

TYPED_TEST(TensorTypedTest, Matmul_ValidMatrices_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> matrix_a(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    Tensor<T> matrix_b(NestedData<T>{{T(5), T(6)}, {T(7), T(8)}});
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
// Sum Operation
// ============================================================================

TYPED_TEST(TensorTypedTest, Sum_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    T result = tensor.sum();
    EXPECT_EQ(result, T(10));
}

// ============================================================================
// Boundary Cases: Integer Overflow
// ============================================================================

TYPED_TEST(TensorTypedTest, Add_IntegerOverflow_Behavior) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isUnsignedInteger()
                  || TestFixture::isSignedInteger()) {
        T maxVal = this->maxValue();
        Tensor<T> t1(NestedData<T>{{maxVal}});
        Tensor<T> t2(NestedData<T>{{T(1)}});

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

TYPED_TEST(TensorTypedTest, Multiply_IntegerOverflow_Behavior) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isUnsignedInteger()
                  || TestFixture::isSignedInteger()) {
        // Use values that would overflow when multiplied
        T largeVal = this->maxValue() / T(2);
        Tensor<T> t1(NestedData<T>{{largeVal}});
        Tensor<T> t2(NestedData<T>{{T(3)}});

        // This may overflow, but we test that it doesn't crash
        auto res = t1 * t2;
        EXPECT_GE(res.getTotalSize(), 1);
    }
}

// ============================================================================
// Boundary Cases: Floating Point Special Values
// ============================================================================

TYPED_TEST(TensorTypedTest, FloatingPoint_Infinity_Handling) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        Tensor<T> t1_inf(NestedData<T>{{inf}});
        Tensor<T> t2_finite(NestedData<T>{{T(1)}});

        // Inf + finite = Inf
        auto res_add = t1_inf + t2_finite;
        EXPECT_TRUE(this->isInf(res_add.at({0, 0})));

        // Inf * finite (positive) = Inf
        auto res_mul = t1_inf * t2_finite;
        EXPECT_TRUE(this->isInf(res_mul.at({0, 0})));

        // Test with negative finite value
        Tensor<T> t3_neg(NestedData<T>{{T(-1)}});
        auto res_add_neg = t1_inf + t3_neg;
        // Inf + (-finite) is still Inf (not -Inf in this case, but could be
        // implementation dependent)
        EXPECT_TRUE(this->isInf(res_add_neg.at({0, 0})));
    }
}

TYPED_TEST(TensorTypedTest, FloatingPoint_NaN_Handling) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T nan = std::numeric_limits<T>::quiet_NaN();
        Tensor<T> t1(NestedData<T>{{nan, T(1)}});
        Tensor<T> t2(NestedData<T>{{T(1), T(2)}});

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

TYPED_TEST(TensorTypedTest, FloatingPoint_ZeroDivision_Throws) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        Tensor<T> t1(NestedData<T>{{T(1.0), T(2.0)}});
        Tensor<T> t_zero(NestedData<T>{{T(0.0), T(1.0)}});
        EXPECT_THROW(t1 / t_zero, std::runtime_error);
    }
}

// ============================================================================
// Precision Tests: Integer Division Rounding
// ============================================================================

TYPED_TEST(TensorTypedTest, IntegerDivision_RoundingBehavior) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isSignedInteger()
                  || TestFixture::isUnsignedInteger()) {
        // Test integer division truncation
        Tensor<T> t1(NestedData<T>{{T(7), T(10)}});
        Tensor<T> t2(NestedData<T>{{T(3), T(3)}});
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

TYPED_TEST(TensorTypedTest, FloatingPoint_Precision) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        // Test that floating point operations maintain reasonable precision
        T val1 = T(0.1);
        T val2 = T(0.2);
        Tensor<T> t1(NestedData<T>{{val1}});
        Tensor<T> t2(NestedData<T>{{val2}});
        auto res = t1 + t2;
        // 0.1 + 0.2 should be approximately 0.3
        this->expectNear(T(0.3), res.at({0, 0}), T(0.0001));
    }
}

// ============================================================================
// Reshape Operation
// ============================================================================

TYPED_TEST(TensorTypedTest, Reshape_ValidNewShape_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor_orig(NestedData<T>{T(1), T(2), T(3), T(4), T(5), T(6)});
    auto tensor_reshaped = tensor_orig.reshape({2, 3});
    EXPECT_EQ(tensor_reshaped.getShape().size(), 2);
    EXPECT_EQ(tensor_reshaped.at({0, 0}), T(1));
    EXPECT_EQ(tensor_reshaped.at({1, 2}), T(6));
}

// ============================================================================
// Transpose Operation
// ============================================================================

TYPED_TEST(TensorTypedTest, Transpose_Valid2DTensor_CorrectResult) {
    using T = TestFixture::Type;
    Tensor<T> tensor_orig(
        NestedData<T>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}});
    auto tensor_transposed = tensor_orig.transpose();
    EXPECT_EQ(tensor_transposed.getShape()[0], 3);
    EXPECT_EQ(tensor_transposed.getShape()[1], 2);
    EXPECT_EQ(tensor_transposed.at({0, 0}), T(1));
    EXPECT_EQ(tensor_transposed.at({0, 1}), T(4));
    EXPECT_EQ(tensor_transposed.at({2, 1}), T(6));
}

// ============================================================================
// Autograd Tests
// ============================================================================

TYPED_TEST(TensorTypedTest, Backward_SimpleAddition_GradientPropagation) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        Tensor<T> a(NestedData<T>{{T(1.0), T(2.0)}});
        Tensor<T> b(NestedData<T>{{T(3.0), T(4.0)}});
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        auto c = a + b;
        c.backward();

        ASSERT_FALSE(a.grad().isEmpty());
        ASSERT_FALSE(b.grad().isEmpty());
        // Gradient should be 1 for addition
        this->expectNear(T(1.0), a.grad().at({0, 0}));
        this->expectNear(T(1.0), b.grad().at({0, 0}));
    }
}

TYPED_TEST(TensorTypedTest, Backward_Multiplication_GradientPropagation) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        Tensor<T> a(NestedData<T>{{T(2.0)}});
        Tensor<T> b(NestedData<T>{{T(3.0)}});
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        auto c = a * b;
        c.backward();

        ASSERT_FALSE(a.grad().isEmpty());
        ASSERT_FALSE(b.grad().isEmpty());
        // dc/da = b = 3, dc/db = a = 2
        this->expectNear(T(3.0), a.grad().at({0, 0}));
        this->expectNear(T(2.0), b.grad().at({0, 0}));
    }
}

// ============================================================================
// Error Handling Tests: Shape Mismatch
// ============================================================================

TYPED_TEST(TensorTypedTest,
           Add_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}});
    Tensor<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TYPED_TEST(TensorTypedTest,
           Subtract_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}});
    Tensor<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 - t2, std::invalid_argument);
}

TYPED_TEST(TensorTypedTest,
           Multiply_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}});
    Tensor<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 * t2, std::invalid_argument);
}

TYPED_TEST(TensorTypedTest,
           Divide_TwoTensors_ShapeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> t1(NestedData<T>{{T(1), T(2)}});
    Tensor<T> t2(NestedData<T>{{T(1), T(2), T(3)}});
    EXPECT_THROW(t1 / t2, std::invalid_argument);
}

TYPED_TEST(TensorTypedTest, Matmul_Non2DTensors_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> matrix_a(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    Tensor<T> tensor_1d(NestedData<T>{T(1), T(2)});
    EXPECT_THROW(matrix_a.matmul(tensor_1d), std::invalid_argument);
    EXPECT_THROW(tensor_1d.matmul(matrix_a), std::invalid_argument);
}

TYPED_TEST(TensorTypedTest,
           Matmul_InnerDimensionMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> matrix_a(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    Tensor<T> matrix_d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}});
    EXPECT_THROW(matrix_a.matmul(matrix_d),
                 std::invalid_argument); // (2x2) @ (3x2) -> inner dims 2 != 3
}

TYPED_TEST(TensorTypedTest, Transpose_Non2DTensor_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> tensor_1d(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_THROW(tensor_1d.transpose(), std::invalid_argument);
}

TYPED_TEST(TensorTypedTest, Reshape_SizeMismatch_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    Tensor<T> tensor_orig(NestedData<T>{T(1), T(2), T(3), T(4)});
    EXPECT_THROW(tensor_orig.reshape({3}),
                 std::invalid_argument); // Total size 3 != 4
    EXPECT_THROW(tensor_orig.reshape({2, 3}),
                 std::invalid_argument); // Total size 6 != 4
}

// ============================================================================
// Detailed Construction Tests
// ============================================================================

TYPED_TEST(TensorTypedTest, Constructor_NestedData_3D_CorrectlyInitializes) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}},
                                   {{T(5), T(6)}, {T(7), T(8)}}});
    EXPECT_EQ(tensor.getShape().size(), 3);
    EXPECT_EQ(tensor.getTotalSize(), 8);
    EXPECT_EQ(tensor.at({0, 0, 0}), T(1));
    EXPECT_EQ(tensor.at({1, 1, 1}), T(8));
}

TYPED_TEST(TensorTypedTest,
           Constructor_NestedData_IrregularShape_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    EXPECT_THROW(Tensor<T>(NestedData<T>{{T(1)}, {T(1), T(2)}}),
                 std::invalid_argument);
    if constexpr (TestFixture::isFloatingPoint()) {
        EXPECT_THROW(Tensor<T>(NestedData<T>{{T(1.0), T(2.0)}, {T(3.0)}}),
                     std::invalid_argument);
    }
}

// ============================================================================
// Property Tests
// ============================================================================

TYPED_TEST(TensorTypedTest, GetShape_ReturnsCorrectDimensionsAndSize) {
    using T = TestFixture::Type;
    Tensor<T> tensor_2d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor_2d.getShape()[0], 2);
    EXPECT_EQ(tensor_2d.getShape()[1], 2);
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);

    Tensor<T> tensor_3d(NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}},
                                      {{T(5), T(6)}, {T(7), T(8)}}});
    EXPECT_EQ(tensor_3d.getShape()[0], 2);
    EXPECT_EQ(tensor_3d.getShape()[1], 2);
    EXPECT_EQ(tensor_3d.getShape()[2], 2);
    EXPECT_EQ(tensor_3d.getTotalSize(), 8);
}

TYPED_TEST(TensorTypedTest, GetDimensions_ReturnsCorrectCount) {
    using T = TestFixture::Type;
    Tensor<T> tensor_scalar(T(1));
    EXPECT_EQ(tensor_scalar.getShape().size(), 0); // Scalar has 0 dimensions
    Tensor<T> tensor_1d(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_EQ(tensor_1d.getShape().size(), 1);
    Tensor<T> tensor_2d(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(tensor_2d.getShape().size(), 2);
}

TYPED_TEST(TensorTypedTest, ElementAccess_OutOfBounds_ThrowsOutOfRange) {
    using T = TestFixture::Type;
    Tensor<T> tensor(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_THROW(tensor.at({0, 0, 0}), std::out_of_range); // Dimension mismatch
    EXPECT_THROW(tensor.at({2, 0}), std::out_of_range); // Index out of bounds
}

// ============================================================================
// Device Operations
// ============================================================================

TYPED_TEST(TensorTypedTest, To_Gpu_ThrowsRuntimeError) {
    using T = TestFixture::Type;
    Tensor<T> t(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    // TODO: Test success or not
}

TYPED_TEST(TensorTypedTest, Clear_ResetsUnderlyingData) {
    using T = TestFixture::Type;
    Tensor<T> t(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    t.clear();
    EXPECT_EQ(t.at({0, 0}), T(0));
    EXPECT_EQ(t.at({1, 1}), T(0));
}

// ============================================================================
// Autograd Property Tests
// ============================================================================

TYPED_TEST(TensorTypedTest, Grad_WhenNoBackward_ReturnsNullptr) {
    using T = TestFixture::Type;
    Tensor<T> t(T(3));
    EXPECT_TRUE(t.grad().isEmpty());
}

TYPED_TEST(TensorTypedTest, RequiresGrad_FlagIsStoredOnNode) {
    using T = TestFixture::Type;
    Tensor<T> t(T(1));
    EXPECT_FALSE(t.getRequiresGrad());
    t.setRequiresGrad(true);
    EXPECT_TRUE(t.getRequiresGrad());
}

TYPED_TEST(TensorTypedTest, ClearGrad_NoGrad_NoThrow) {
    using T = TestFixture::Type;
    Tensor<T> t(T(1));
    EXPECT_NO_THROW(t.clearGrad());
}
