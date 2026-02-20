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
#include <vector>

#include "backend/Device.h"
#include "common/definitions.h"
#include "math/ds/NestedData.h"
#include "math/ds/TensorData.h"
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
using hahaha::math::TensorData;
using hahaha::math::TensorShape;

// Define all supported numeric types
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T> class TensorDataTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(TensorDataTypedTest, NumericTypes);

// ============================================================================
// Basic Construction Tests
// ============================================================================

TYPED_TEST(TensorDataTypedTest, DefaultConstructor) {
    using T = TestFixture::Type;
    TensorData<T> defaultConstructedTensor;
    EXPECT_EQ(defaultConstructedTensor.getShape().getDims().size(), 0);
    EXPECT_EQ(defaultConstructedTensor.getData().get(), nullptr);
}

TYPED_TEST(TensorDataTypedTest, ShapeOnlyConstructor_DefaultDeviceAllocates) {
    using T = TestFixture::Type;
    TensorShape shape({2, 2});
    TensorData<T> td(shape);
    EXPECT_EQ(td.getShape().getTotalSize(), 4);
    EXPECT_NE(td.getData().get(), nullptr);
    // All elements should be zero-initialized
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(td.getData()[i], T(0));
    }
}

TYPED_TEST(TensorDataTypedTest, ShapeValueConstructor) {
    using T = TestFixture::Type;
    TensorShape shape({2, 3});
    T initValue = T(7);
    TensorData<T> tensor_data(shape, initValue);
    EXPECT_EQ(tensor_data.getShape().getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(tensor_data.getData()[i], initValue);
    }
    EXPECT_EQ(tensor_data.getStride().getStrideSize(), 2);
    EXPECT_EQ(tensor_data.getStride()[0], 3);
    EXPECT_EQ(tensor_data.getStride()[1], 1);
}

TYPED_TEST(TensorDataTypedTest, InitWithNestedData) {
    using T = TestFixture::Type;
    TensorData<T> singleValueTensor(NestedData<T>{T(1)});
    EXPECT_EQ(singleValueTensor.getShape().getDims().size(), 1);
    EXPECT_EQ(singleValueTensor.getShape().getDims()[0], 1);
    EXPECT_EQ(singleValueTensor.getData()[0], T(1));

    TensorData<T> twoElementTensor(NestedData<T>{{T(1)}, {T(2)}});
    EXPECT_EQ(twoElementTensor.getShape().getDims().size(), 2);
    EXPECT_EQ(twoElementTensor.getShape().getDims()[0], 2);
    EXPECT_EQ(twoElementTensor.getShape().getDims()[1], 1);
    EXPECT_EQ(twoElementTensor.getData()[0], T(1));
    EXPECT_EQ(twoElementTensor.getData()[1], T(2));
}

TYPED_TEST(TensorDataTypedTest,
           InitWithEmptyNestedData_ProducesNullDataAndScalarShape) {
    using T = TestFixture::Type;
    TensorData<T> empty(NestedData<T>{});
    EXPECT_EQ(empty.getShape().getDims().size(), 0);
    EXPECT_EQ(empty.getData().get(), nullptr);
}

TYPED_TEST(TensorDataTypedTest, InitVecConstructor_Creates1DTensor) {
    using T = TestFixture::Type;
    std::vector<T> vec = {T(7), T(8), T(9)};
    TensorData<T> td(vec);
    EXPECT_EQ(td.getShape().getDims().size(), 1);
    EXPECT_EQ(td.getShape().getDims()[0], 3);
    EXPECT_EQ(td.getStride().getStrideSize(), 1);
    EXPECT_EQ(td.getStride()[0], 1);
    EXPECT_EQ(td.getData()[0], T(7));
    EXPECT_EQ(td.getData()[2], T(9));
}

TYPED_TEST(TensorDataTypedTest, OneDimensionalTensor) {
    using T = TestFixture::Type;
    TensorData<T> tensor_1d(NestedData<T>{T(1), T(2), T(3), T(4), T(5)});
    EXPECT_EQ(tensor_1d.getShape().getTotalSize(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(tensor_1d.getData()[i], T(i + 1));
    }
}

// ============================================================================
// Copy and Move Semantics
// ============================================================================

TYPED_TEST(TensorDataTypedTest, CopyConstructor) {
    using T = TestFixture::Type;
    TensorData<T> original(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    TensorData<T> copied(original);

    EXPECT_EQ(copied.getShape(), original.getShape());
    EXPECT_EQ(copied.getStride().getStrideSize(),
              original.getStride().getStrideSize());
    for (size_t i = 0; i < original.getShape().getTotalSize(); ++i) {
        EXPECT_EQ(copied.getData()[i], original.getData()[i]);
    }

    // Verify deep copy
    copied.getData()[0] = T(100);
    EXPECT_EQ(original.getData()[0], T(1));
    EXPECT_EQ(copied.getData()[0], T(100));
}

TYPED_TEST(TensorDataTypedTest, MoveConstructor) {
    using T = TestFixture::Type;
    TensorData<T> original(NestedData<T>{T(1), T(2), T(3)});
    void* originalPtr = original.getData().get();

    TensorData<T> moved(std::move(original));

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
    EXPECT_EQ(original.getShape().getDims().size(), 0);
}

TYPED_TEST(TensorDataTypedTest, MoveAssignment) {
    using T = TestFixture::Type;
    TensorData<T> original(NestedData<T>{T(1), T(2), T(3)});
    void* originalPtr = original.getData().get();
    TensorData<T> moved;

    moved = std::move(original);

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
}

// ============================================================================
// Share Operation
// ============================================================================

TYPED_TEST(TensorDataTypedTest, Share_SharesBufferButCopiesMetadata) {
    using T = TestFixture::Type;
    TensorData<T> original(TensorShape({2, 2}), T(3));
    auto shared = original.share();

    // Shares underlying buffer
    EXPECT_EQ(shared.getData().get(), original.getData().get());
    // Metadata is value-copied
    EXPECT_EQ(shared.getShape(), original.getShape());
    EXPECT_EQ(shared.getStride().toString(), original.getStride().toString());

    // Mutating shared data mutates original data (same buffer)
    shared.getData()[0] = T(42);
    EXPECT_EQ(original.getData()[0], T(42));

    // Mutating metadata on shared should not affect original
    shared.setShape(TensorShape({4}));
    EXPECT_NE(shared.getShape(), original.getShape());
}

TYPED_TEST(TensorDataTypedTest, Share_NullData) {
    using T = TestFixture::Type;
    TensorData<T> original;
    auto shared = original.share();
    EXPECT_EQ(shared.getData().get(), nullptr);
    EXPECT_EQ(shared.getShape().getDims().size(), 0);
}

// ============================================================================
// Device Operations
// ============================================================================

// TYPED_TEST(TensorDataTypedTest, Device_GetSet_Works) {
//     using T = TestFixture::Type;
//     TensorData<T> td(TensorShape({1}), T(1));
//     EXPECT_EQ(td.getDevice()->getType(), DeviceType::CPU);
//     td.setDevice(Device(DeviceType::SIMD, 0));
//     EXPECT_EQ(td.getDevice()->getType(), DeviceType::SIMD);
// }
//
// TYPED_TEST(TensorDataTypedTest,
//            ShapeOnlyConstructor_GpuDevice_ThrowsRuntimeError) {
//     using T = TestFixture::Type;
//     TensorShape shape({2, 2});
//     EXPECT_THROW(TensorData<T>(shape, Device(DeviceType::GPU, 0)),
//                  std::runtime_error);
// }
//
// TYPED_TEST(TensorDataTypedTest,
//            ShapeValueConstructor_GpuDevice_ThrowsRuntimeError) {
//     using T = TestFixture::Type;
//     TensorShape shape({2, 2});
//     EXPECT_THROW(TensorData<T>(shape, T(1), Device(DeviceType::GPU, 0)),
//                  std::runtime_error);
// }
//
// ============================================================================
// Operator Index
// ============================================================================

TYPED_TEST(TensorDataTypedTest, OperatorIndex_ReferencesUnderlyingData) {
    using T = TestFixture::Type;
    TensorData<T> td(TensorShape({3}), T(0));
    td[1] = T(123);
    EXPECT_EQ(td.getData()[1], T(123));
}

// ============================================================================
// Boundary Cases: Large Values
// ============================================================================

TYPED_TEST(TensorDataTypedTest, LargeValues_Handling) {
    using T = TestFixture::Type;
    T largeVal = this->maxValue() / T(2);
    TensorData<T> td(TensorShape({1}), largeVal);
    EXPECT_EQ(td.getData()[0], largeVal);
}

TYPED_TEST(TensorDataTypedTest, SmallValues_Handling) {
    using T = TestFixture::Type;
    T smallVal = this->minValue();
    TensorData<T> td(TensorShape({1}), smallVal);
    EXPECT_EQ(td.getData()[0], smallVal);
}

// ============================================================================
// Boundary Cases: Floating Point Special Values
// ============================================================================

TYPED_TEST(TensorDataTypedTest, FloatingPoint_Infinity_Storage) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        TensorData<T> td(TensorShape({1}), inf);
        EXPECT_TRUE(this->isInf(td.getData()[0]));
    }
}

TYPED_TEST(TensorDataTypedTest, FloatingPoint_NaN_Storage) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T nan = std::numeric_limits<T>::quiet_NaN();
        TensorData<T> td(TensorShape({1}), nan);
        EXPECT_TRUE(this->isNaN(td.getData()[0]));
    }
}

TYPED_TEST(TensorDataTypedTest, FloatingPoint_NestedDataWithSpecialValues) {
    using T = TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        T nan = std::numeric_limits<T>::quiet_NaN();
        TensorData<T> td(NestedData<T>{{inf, nan}, {T(1.0), T(2.0)}});
        EXPECT_TRUE(this->isInf(td.getData()[0]));
        EXPECT_TRUE(this->isNaN(td.getData()[1]));
        EXPECT_EQ(td.getData()[2], T(1.0));
        EXPECT_EQ(td.getData()[3], T(2.0));
    }
}

// ============================================================================
// Large Tensor Tests
// ============================================================================

TYPED_TEST(TensorDataTypedTest, LargeTensor_Allocation) {
    using T = TestFixture::Type;
    // Test with a reasonably large tensor (1000 elements)
    TensorShape shape({10, 10, 10});
    TensorData<T> td(shape, T(5));
    EXPECT_EQ(td.getShape().getTotalSize(), 1000);
    EXPECT_NE(td.getData().get(), nullptr);
    // Check a few elements
    EXPECT_EQ(td.getData()[0], T(5));
    EXPECT_EQ(td.getData()[500], T(5));
    EXPECT_EQ(td.getData()[999], T(5));
}

// ============================================================================
// Dimension-specific tests: 0D to 3D initialization
// ============================================================================

TYPED_TEST(TensorDataTypedTest, ZeroDimension_Scalar) {
    using T = TestFixture::Type;
    // 0D scalar: SingleValueConstruction
    T value = T(42);
    TensorData<T> td(NestedData<T>{value});
    EXPECT_EQ(td.getShape().getDims().size(),
              1); // NestedData{value} creates 1D
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], value);
}

TYPED_TEST(TensorDataTypedTest, ZeroDimension_Scalar_Direct) {
    using T = TestFixture::Type;
    // 0D scalar: using TensorShape({})
    TensorShape shape0D({});
    TensorData<T> td(shape0D, T(42));
    EXPECT_EQ(td.getShape().getDims().size(), 0);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], T(42));
}

TYPED_TEST(TensorDataTypedTest, OneDimension_Vector) {
    using T = TestFixture::Type;
    // 1D: [1, 2, 3]
    TensorData<T> td(NestedData<T>{T(1), T(2), T(3)});
    EXPECT_EQ(td.getShape().getDims().size(), 1);
    EXPECT_EQ(td.getShape().getDims()[0], 3);
    EXPECT_EQ(td.getShape().getTotalSize(), 3);
    EXPECT_EQ(td.getData()[0], T(1));
    EXPECT_EQ(td.getData()[2], T(3));
}

TYPED_TEST(TensorDataTypedTest, OneDimension_SingleElement) {
    using T = TestFixture::Type;
    // 1D with single element: [1] (different from scalar)
    TensorData<T> td(NestedData<T>{T(1)});
    EXPECT_EQ(td.getShape().getDims().size(), 1);
    EXPECT_EQ(td.getShape().getDims()[0], 1);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], T(1));
}

TYPED_TEST(TensorDataTypedTest, TwoDimension_Matrix) {
    using T = TestFixture::Type;
    // 2D: {{1, 2}, {3, 4}}
    TensorData<T> td(NestedData<T>{{T(1), T(2)}, {T(3), T(4)}});
    EXPECT_EQ(td.getShape().getDims().size(), 2);
    EXPECT_EQ(td.getShape().getDims()[0], 2);
    EXPECT_EQ(td.getShape().getDims()[1], 2);
    EXPECT_EQ(td.getShape().getTotalSize(), 4);
    EXPECT_EQ(td.getData()[0], T(1));
    EXPECT_EQ(td.getData()[3], T(4));
}

TYPED_TEST(TensorDataTypedTest, TwoDimension_SingleElement) {
    using T = TestFixture::Type;
    // 2D with single element: {{1}} (high-dimensional scalar)
    TensorData<T> td(NestedData<T>{{T(1)}});
    EXPECT_EQ(td.getShape().getDims().size(), 2);
    EXPECT_EQ(td.getShape().getDims()[0], 1);
    EXPECT_EQ(td.getShape().getDims()[1], 1);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], T(1));
}

TYPED_TEST(TensorDataTypedTest, ThreeDimension_Tensor) {
    using T = TestFixture::Type;
    // 3D: {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}
    TensorData<T> td(
        NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}}, {{T(5), T(6)}, {T(7), T(8)}}});
    EXPECT_EQ(td.getShape().getDims().size(), 3);
    EXPECT_EQ(td.getShape().getDims()[0], 2);
    EXPECT_EQ(td.getShape().getDims()[1], 2);
    EXPECT_EQ(td.getShape().getDims()[2], 2);
    EXPECT_EQ(td.getShape().getTotalSize(), 8);
    EXPECT_EQ(td.getData()[0], T(1));
    EXPECT_EQ(td.getData()[7], T(8));
}

TYPED_TEST(TensorDataTypedTest, ThreeDimension_SingleElement) {
    using T = TestFixture::Type;
    // 3D with single element: {{{1}}} (3D scalar)
    TensorData<T> td(NestedData<T>{{{T(1)}}});
    EXPECT_EQ(td.getShape().getDims().size(), 3);
    EXPECT_EQ(td.getShape().getDims()[0], 1);
    EXPECT_EQ(td.getShape().getDims()[1], 1);
    EXPECT_EQ(td.getShape().getDims()[2], 1);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], T(1));
}

// ============================================================================
// ShapeValueConstructor Tests - All Dimensions (0D to 3D)
// ============================================================================

TYPED_TEST(TensorDataTypedTest, ShapeValueConstructor_0D_Scalar) {
    using T = TestFixture::Type;
    TensorShape shape0D({});
    TensorData<T> td0D(shape0D, T(42));
    EXPECT_EQ(td0D.getShape().getDims().size(), 0);
    EXPECT_EQ(td0D.getShape().getTotalSize(), 1);
    EXPECT_EQ(td0D.getData()[0], T(42));
}

TYPED_TEST(TensorDataTypedTest, ShapeValueConstructor_1D_Vector) {
    using T = TestFixture::Type;
    TensorShape shape1D({3});
    TensorData<T> td1D(shape1D, T(7));
    EXPECT_EQ(td1D.getShape().getDims().size(), 1);
    EXPECT_EQ(td1D.getShape().getTotalSize(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(td1D.getData()[i], T(7));
    }
}

TYPED_TEST(TensorDataTypedTest, ShapeValueConstructor_2D_Matrix) {
    using T = TestFixture::Type;
    TensorShape shape2D({2, 3});
    TensorData<T> td2D(shape2D, T(5));
    EXPECT_EQ(td2D.getShape().getDims().size(), 2);
    EXPECT_EQ(td2D.getShape().getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(td2D.getData()[i], T(5));
    }
}

TYPED_TEST(TensorDataTypedTest, ShapeValueConstructor_3D_Tensor) {
    using T = TestFixture::Type;
    TensorShape shape3D({2, 2, 2});
    TensorData<T> td3D(shape3D, T(9));
    EXPECT_EQ(td3D.getShape().getDims().size(), 3);
    EXPECT_EQ(td3D.getShape().getTotalSize(), 8);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(td3D.getData()[i], T(9));
    }
}
