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

#include <gtest/gtest.h>

#include "Tensor.h"
#include "common/definitions.h"

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

// Define all supported numeric types from definitions.h
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T>
class ComputeNodeErrorHandlingTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get test value
    static constexpr T testValue() {
        if constexpr (isFloatingPoint()) {
            return T(10.0);
        } else {
            return T(10);
        }
    }

    // Helper to get zero value
    static constexpr T zeroValue() {
        return T(0);
    }
};

TYPED_TEST_SUITE(ComputeNodeErrorHandlingTypedTest, NumericTypes);

class ComputeNodeErrorHandlingTest : public ::testing::Test {};

// ============================================================================
// Division By Zero Error Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(ComputeNodeErrorHandlingTypedTest,
           DivisionByZero_0D_Scalar_TensorByTensor) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T zero = TestFixture::zeroValue();
    Tensor<T> a(val);
    Tensor<T> b(zero);
    EXPECT_THROW(auto c = a / b, std::runtime_error);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest,
           DivisionByZero_0D_Scalar_TensorByScalar) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T zero = TestFixture::zeroValue();
    Tensor<T> a(val);
    EXPECT_THROW(auto d = a / zero, std::runtime_error);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest,
           DivisionByZero_0D_Scalar_ScalarByTensor) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    T zero = TestFixture::zeroValue();
    Tensor<T> b(zero);
    EXPECT_THROW(auto e = val / b, std::runtime_error);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest, DivisionByZero_1D_Vector) {
    using T = TestFixture::Type;
    T v10 = TestFixture::testValue();
    T v20 = T(20);
    T v1 = T(1);
    T zero = TestFixture::zeroValue();
    Tensor<T> a(NestedData<T>{v10, v20});
    Tensor<T> b(NestedData<T>{v1, zero});
    EXPECT_THROW(auto c = a / b, std::runtime_error);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest, DivisionByZero_2D_Matrix) {
    using T = TestFixture::Type;
    T v10 = TestFixture::testValue();
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v1 = T(1);
    T zero = TestFixture::zeroValue();
    T v2 = T(2);
    T v3 = T(3);
    Tensor<T> a(NestedData<T>{{v10, v20}, {v30, v40}});
    Tensor<T> b(NestedData<T>{{v1, zero}, {v2, v3}});
    EXPECT_THROW(auto c = a / b, std::runtime_error);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest, DivisionByZero_3D_Tensor) {
    using T = TestFixture::Type;
    T v10 = TestFixture::testValue();
    T v20 = T(20);
    T v30 = T(30);
    T v40 = T(40);
    T v50 = T(50);
    T v60 = T(60);
    T v70 = T(70);
    T v80 = T(80);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T zero = TestFixture::zeroValue();
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    Tensor<T> a(
        NestedData<T>{{{v10, v20}, {v30, v40}}, {{v50, v60}, {v70, v80}}});
    Tensor<T> b(NestedData<T>{{{v1, v2}, {v3, zero}}, {{v5, v6}, {v7, v8}}});
    EXPECT_THROW(auto c = a / b, std::runtime_error);
}

// ============================================================================
// Matrix Multiplication Error Tests - Typed Test (All Types)
// ============================================================================

TYPED_TEST(ComputeNodeErrorHandlingTypedTest, MatMul_0D_Scalar_Throws) {
    using T = TestFixture::Type;
    T val = TestFixture::isFloatingPoint() ? T(5.0) : T(5);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> s0(val);
    Tensor<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(s0.matmul(m2), std::invalid_argument);
    EXPECT_THROW(m2.matmul(s0), std::invalid_argument);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest, MatMul_1D_Vector_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> v1_tensor(NestedData<T>{v1, v2, v3});
    Tensor<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(v1_tensor.matmul(m2), std::invalid_argument);
    EXPECT_THROW(m2.matmul(v1_tensor), std::invalid_argument);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest,
           MatMul_2D_IncompatibleInnerDimensions_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    Tensor<T> m2a(NestedData<T>{{v1}, {v2}});                 // 2x2
    Tensor<T> m2b(NestedData<T>{{v1, v2, v3}, {v4, v5, v6}}); // 2x3
    EXPECT_THROW(m2a.matmul(m2b), std::invalid_argument);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest, MatMul_3D_Tensor_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    Tensor<T> t3(NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    Tensor<T> m2(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(t3.matmul(m2), std::invalid_argument);
    EXPECT_THROW(m2.matmul(t3), std::invalid_argument);
}

TYPED_TEST(ComputeNodeErrorHandlingTypedTest,
           MatMul_InvalidShape_Non2D_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    Tensor<T> a(NestedData<T>{{v1, v2}, {v3, v4}});
    Tensor<T> b(NestedData<T>{v1, v2, v3});
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}
