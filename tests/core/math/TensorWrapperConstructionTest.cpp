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

// Define all supported numeric types from definitions.h
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T>
class TensorWrapperConstructionTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    // Helper to get test value
    static constexpr T testValue() {
        if constexpr (isFloatingPoint()) {
            return T(5.0);
        } else {
            return T(5);
        }
    }

    // Helper to get scalar value for 0D tensor
    static constexpr T scalarValue() {
        if constexpr (isFloatingPoint()) {
            return T(42.0);
        } else {
            return T(42);
        }
    }

    // Helper to compare values with appropriate tolerance
    void expectNear(T expected, T actual) {
        if constexpr (std::is_same_v<T, f64>) {
            EXPECT_DOUBLE_EQ(expected, actual);
        } else if constexpr (isFloatingPoint()) {
            EXPECT_FLOAT_EQ(expected, actual);
        } else {
            EXPECT_EQ(expected, actual);
        }
    }
};

TYPED_TEST_SUITE(TensorWrapperConstructionTypedTest, NumericTypes);

class TensorWrapperConstructionTest : public ::testing::Test {};

// ============================================================================
// Default Constructor - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperConstructionTypedTest, Default_CreatesEmptyTensor) {
    using T = TestFixture::Type;
    TensorWrapper<T> tensor;
    EXPECT_EQ(tensor.getTotalSize(), 0);
    EXPECT_EQ(tensor.getShape().size(), 0);
}

// ============================================================================
// Shape-Value Constructor - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperConstructionTypedTest,
           ShapeInitValue_CreatesCorrectTensor) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> tensor(TensorShape({2, 3}), val);
    EXPECT_EQ(tensor.getTotalSize(), 6);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 3);
    this->expectNear(tensor.at({0, 0}), val);
    this->expectNear(tensor.at({1, 2}), val);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           ShapeInitValueDevice_CreatesCorrectTensor) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> tensor(TensorShape({2, 3}),
                            val,
                            std::make_shared<hahaha::backend::CPUDevice>());
    EXPECT_EQ(tensor.getTotalSize(), 6);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getDevice()->getType(), DeviceType::CPU);
    this->expectNear(tensor.at({0, 0}), val);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           ShapeDevice_CreatesCorrectTensorWithDefaultInit) {
    using T = TestFixture::Type;
    TensorWrapper<T> tensor(TensorShape({2, 2}),
                            std::make_shared<hahaha::backend::CPUDevice>());
    EXPECT_EQ(tensor.getTotalSize(), 4);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getDevice()->getType(), DeviceType::CPU);
    T expectedDefault = T(0);
    this->expectNear(tensor.at({0, 0}), expectedDefault);
}

// ============================================================================
// NestedData Constructor - All Dimensions (0D-3D) - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_0D_Scalar) {
    using T = TestFixture::Type;
    T val = TestFixture::scalarValue();
    TensorWrapper<T> tensor(val);
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 0); // Scalar tensor has 0 dimensions
    this->expectNear(tensor.at({}), val);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_1D_Vector) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> tensor(NestedData<T>{v1, v2, v3});
    EXPECT_EQ(tensor.getTotalSize(), 3);
    EXPECT_EQ(tensor.getShape().size(), 1);
    EXPECT_EQ(tensor.getShape()[0], 3);
    this->expectNear(tensor.at({0}), v1);
    this->expectNear(tensor.at({2}), v3);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_1D_SingleElement) {
    using T = TestFixture::Type;
    T val = T(1);
    TensorWrapper<T> tensor(NestedData<T>{val});
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 1);
    EXPECT_EQ(tensor.getShape()[0], 1);
    this->expectNear(tensor.at({0}), val);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_2D_Matrix) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_EQ(tensor.getTotalSize(), 4);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 2);
    this->expectNear(tensor.at({0, 0}), v1);
    this->expectNear(tensor.at({1, 1}), v4);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_2D_SingleElement) {
    using T = TestFixture::Type;
    T val = T(1);
    TensorWrapper<T> tensor(NestedData<T>{{val}});
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 1);
    EXPECT_EQ(tensor.getShape()[1], 1);
    this->expectNear(tensor.at({0, 0}), val);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_3D_Tensor) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> tensor(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_EQ(tensor.getTotalSize(), 8);
    EXPECT_EQ(tensor.getShape().size(), 3);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 2);
    EXPECT_EQ(tensor.getShape()[2], 2);
    this->expectNear(tensor.at({0, 0, 0}), v1);
    this->expectNear(tensor.at({1, 1, 1}), v8);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, NestedData_3D_SingleElement) {
    using T = TestFixture::Type;
    T val = T(1);
    TensorWrapper<T> tensor(NestedData<T>{{{val}}});
    EXPECT_EQ(tensor.getTotalSize(), 1);
    EXPECT_EQ(tensor.getShape().size(), 3);
    EXPECT_EQ(tensor.getShape()[0], 1);
    EXPECT_EQ(tensor.getShape()[1], 1);
    EXPECT_EQ(tensor.getShape()[2], 1);
    this->expectNear(tensor.at({0, 0, 0}), val);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           NestedData_IrregularShape_ThrowsInvalidArgument) {
    using T = TestFixture::Type;
    EXPECT_THROW(TensorWrapper<T>(NestedData<T>{{T(1)}, {T(1), T(2)}}),
                 std::invalid_argument);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    if constexpr (TestFixture::isFloatingPoint()) {
        EXPECT_THROW(TensorWrapper<T>(NestedData<T>{{v1, v2}, {v3}}),
                     std::invalid_argument);
    } else {
        EXPECT_THROW(TensorWrapper<T>(NestedData<T>{{v1, v2}, {v3}}),
                     std::invalid_argument);
    }
}

// ============================================================================
// Vector Constructor - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperConstructionTypedTest, Vector_CorrectlyInitializes) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    std::vector<T> vec = {v1, v2, v3};
    TensorWrapper<T> tensor(vec);
    EXPECT_EQ(tensor.getTotalSize(), 3);
    EXPECT_EQ(tensor.getShape().size(), 1);
    this->expectNear(tensor.at({0}), v1);
    this->expectNear(tensor.at({2}), v3);
}

// ============================================================================
// Copy/Move Semantics - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperConstructionTypedTest, CopyConstructor_DeepCopiesData) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v10 = T(10);
    TensorWrapper<T> original(NestedData<T>{{v1, v2}});
    TensorWrapper<T> copy = original; // Uses copy constructor
    original.at({0, 0}) = v10;
    this->expectNear(copy.at({0, 0}),
                     v1); // Original modification should not affect copy
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           MoveConstructor_TransfersOwnership) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor_orig(NestedData<T>{{v1, v2}, {v3, v4}});
    TensorWrapper<T> tensor_moved(std::move(tensor_orig));
    EXPECT_EQ(tensor_moved.getTotalSize(), 4);
    EXPECT_EQ(tensor_orig.getTotalSize(),
              0); // Moved-from objects usually have empty state
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           MoveAssignment_TransfersOwnership) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    TensorWrapper<T> tensor_a(NestedData<T>{{v1, v1}});
    TensorWrapper<T> tensor_b(NestedData<T>{{v2, v2}});
    tensor_a = std::move(tensor_b);
    this->expectNear(tensor_a.at({0, 0}), v2);
    EXPECT_EQ(tensor_b.getTotalSize(), 0); // Moved-from object is empty
}

// ============================================================================
// Properties and Accessors - All Dimensions - Typed Test
// ============================================================================

TYPED_TEST(TensorWrapperConstructionTypedTest,
           GetShape_ReturnsCorrectDimensionsAndSize) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor_2d(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_EQ(tensor_2d.getShape()[0], 2);
    EXPECT_EQ(tensor_2d.getShape()[1], 2);
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);

    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> tensor_3d(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_EQ(tensor_3d.getShape()[0], 2);
    EXPECT_EQ(tensor_3d.getShape()[1], 2);
    EXPECT_EQ(tensor_3d.getShape()[2], 2);
    EXPECT_EQ(tensor_3d.getTotalSize(), 8);
}

TYPED_TEST(TensorWrapperConstructionTypedTest, GetStride_ReturnsCorrectValues) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor_2d(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_EQ(tensor_2d.getStride().getStrideVec()[0], 2);
    EXPECT_EQ(tensor_2d.getStride().getStrideVec()[1], 1);

    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    TensorWrapper<T> tensor_3d(
        NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6}, {v7, v8}}});
    EXPECT_EQ(tensor_3d.getStride().getStrideVec()[0], 4);
    EXPECT_EQ(tensor_3d.getStride().getStrideVec()[1], 2);
    EXPECT_EQ(tensor_3d.getStride().getStrideVec()[2], 1);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           GetDevice_ReturnsCorrectDefaultDevice) {
    using T = TestFixture::Type;
    TensorWrapper<T> tensor(TensorShape({2, 2}));
    EXPECT_EQ(tensor.getDevice()->getType(), DeviceType::CPU);
    EXPECT_EQ(tensor.getDevice()->getId(), 0);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           GetRawData_ReturnsSharedPtrToBuffer) {
    using T = TestFixture::Type;
    T val = TestFixture::testValue();
    TensorWrapper<T> tensor(TensorShape({2, 2}), val);
    auto& raw = tensor.getRawData();
    EXPECT_NE(raw.get(), nullptr);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           ElementAccess_ReadWrite_CorrectlyModifiesAndRetrieves) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v10 = T(10);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}});
    this->expectNear(tensor.at({0, 0}), v1);
    tensor.at({1, 1}) = v10;
    this->expectNear(tensor.at({1, 1}), v10);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           ElementAccess_OutOfBounds_ThrowsOutOfRange) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_THROW(tensor.at({0, 0, 0}), std::out_of_range); // Dimension mismatch
    EXPECT_THROW(tensor.at({2, 0}), std::out_of_range); // Index out of bounds
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           ElementAccess_ConstAt_WorksAndThrowsOnMismatch) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    const TensorWrapper<T> tensor(NestedData<T>{{v1, v2}, {v3, v4}});
    this->expectNear(tensor.at({0, 1}), v2);
    EXPECT_THROW(tensor.at({0}), std::out_of_range);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           GetDimensions_ReturnsCorrectCount) {
    using T = TestFixture::Type;
    T val1 = T(1);
    TensorWrapper<T> tensor_scalar(val1);
    EXPECT_EQ(tensor_scalar.getDimensions(), 0);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> tensor_1d(NestedData<T>{v1, v2, v3});
    EXPECT_EQ(tensor_1d.getDimensions(), 1);
    T v4 = T(4);
    TensorWrapper<T> tensor_2d(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_EQ(tensor_2d.getDimensions(), 2);
}

TYPED_TEST(TensorWrapperConstructionTypedTest,
           GetTotalSize_ReturnsCorrectCount) {
    using T = TestFixture::Type;
    T val1 = T(1);
    TensorWrapper<T> tensor_scalar(NestedData<T>{val1});
    EXPECT_EQ(tensor_scalar.getTotalSize(), 1);
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    TensorWrapper<T> tensor_1d(NestedData<T>{v1, v2, v3});
    EXPECT_EQ(tensor_1d.getTotalSize(), 3);
    T v4 = T(4);
    TensorWrapper<T> tensor_2d(NestedData<T>{{v1, v2}, {v3, v4}});
    EXPECT_EQ(tensor_2d.getTotalSize(), 4);
}
