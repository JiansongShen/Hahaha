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
    using T = typename TestFixture::Type;
    TensorData<T> defaultConstructedTensor;
    EXPECT_EQ(defaultConstructedTensor.getShape().getDims().size(), 0);
    EXPECT_EQ(defaultConstructedTensor.getData().get(), nullptr);
}

TYPED_TEST(TensorDataTypedTest, ShapeOnlyConstructor_DefaultDeviceAllocates) {
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
    TensorData<T> empty(NestedData<T>{});
    EXPECT_EQ(empty.getShape().getDims().size(), 0);
    EXPECT_EQ(empty.getData().get(), nullptr);
}

TYPED_TEST(TensorDataTypedTest, InitVecConstructor_Creates1DTensor) {
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
    TensorData<T> original(NestedData<T>{T(1), T(2), T(3)});
    void* originalPtr = original.getData().get();

    TensorData<T> moved(std::move(original));

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
    EXPECT_EQ(original.getShape().getDims().size(), 0);
}

TYPED_TEST(TensorDataTypedTest, MoveAssignment) {
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
    TensorData<T> original;
    auto shared = original.share();
    EXPECT_EQ(shared.getData().get(), nullptr);
    EXPECT_EQ(shared.getShape().getDims().size(), 0);
}

// ============================================================================
// Device Operations
// ============================================================================

TYPED_TEST(TensorDataTypedTest, Device_GetSet_Works) {
    using T = typename TestFixture::Type;
    TensorData<T> td(TensorShape({1}), T(1));
    EXPECT_EQ(td.getDevice().type, DeviceType::CPU);
    td.setDevice(Device(DeviceType::SIMD, 0));
    EXPECT_EQ(td.getDevice().type, DeviceType::SIMD);
}

TYPED_TEST(TensorDataTypedTest,
           ShapeOnlyConstructor_GpuDevice_ThrowsRuntimeError) {
    using T = typename TestFixture::Type;
    TensorShape shape({2, 2});
    EXPECT_THROW(TensorData<T>(shape, Device(DeviceType::GPU, 0)),
                 std::runtime_error);
}

TYPED_TEST(TensorDataTypedTest,
           ShapeValueConstructor_GpuDevice_ThrowsRuntimeError) {
    using T = typename TestFixture::Type;
    TensorShape shape({2, 2});
    EXPECT_THROW(TensorData<T>(shape, T(1), Device(DeviceType::GPU, 0)),
                 std::runtime_error);
}

// ============================================================================
// Operator Index
// ============================================================================

TYPED_TEST(TensorDataTypedTest, OperatorIndex_ReferencesUnderlyingData) {
    using T = typename TestFixture::Type;
    TensorData<T> td(TensorShape({3}), T(0));
    td[1] = T(123);
    EXPECT_EQ(td.getData()[1], T(123));
}

// ============================================================================
// Boundary Cases: Large Values
// ============================================================================

TYPED_TEST(TensorDataTypedTest, LargeValues_Handling) {
    using T = typename TestFixture::Type;
    T largeVal = this->maxValue() / T(2);
    TensorData<T> td(TensorShape({1}), largeVal);
    EXPECT_EQ(td.getData()[0], largeVal);
}

TYPED_TEST(TensorDataTypedTest, SmallValues_Handling) {
    using T = typename TestFixture::Type;
    T smallVal = this->minValue();
    TensorData<T> td(TensorShape({1}), smallVal);
    EXPECT_EQ(td.getData()[0], smallVal);
}

// ============================================================================
// Boundary Cases: Floating Point Special Values
// ============================================================================

TYPED_TEST(TensorDataTypedTest, FloatingPoint_Infinity_Storage) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        TensorData<T> td(TensorShape({1}), inf);
        EXPECT_TRUE(this->isInf(td.getData()[0]));
    }
}

TYPED_TEST(TensorDataTypedTest, FloatingPoint_NaN_Storage) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T nan = std::numeric_limits<T>::quiet_NaN();
        TensorData<T> td(TensorShape({1}), nan);
        EXPECT_TRUE(this->isNaN(td.getData()[0]));
    }
}

TYPED_TEST(TensorDataTypedTest, FloatingPoint_NestedDataWithSpecialValues) {
    using T = typename TestFixture::Type;
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
    using T = typename TestFixture::Type;
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
