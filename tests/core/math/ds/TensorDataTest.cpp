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
//

#include "math/ds/TensorData.h"

#include <gtest/gtest.h>
#include <vector>

#include "backend/Device.h"
#include "backend/cpu/CPUDevice.h"
#include "backend/gpu/GPUDevice.h"

class TensorDataTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

using hahaha::math::TensorData;

TEST_F(TensorDataTest, InitWithInitializerList) {
    TensorData<int> singleValueTensor(hahaha::math::NestedData<int>{1});
    EXPECT_EQ(singleValueTensor.getShape().getDims().size(), 1);
    EXPECT_EQ(singleValueTensor.getShape().getDims()[0], 1);
    EXPECT_EQ(singleValueTensor.getData()[0], 1);

    TensorData<int> twoElementTensor(hahaha::math::NestedData<int>{{1}, {2}});
    EXPECT_EQ(twoElementTensor.getShape().getDims().size(), 2);
    EXPECT_EQ(twoElementTensor.getShape().getDims()[0], 2);
    EXPECT_EQ(twoElementTensor.getShape().getDims()[1], 1);
    EXPECT_EQ(twoElementTensor.getData()[0], 1);
    EXPECT_EQ(twoElementTensor.getData()[1], 2);
}

TEST_F(TensorDataTest, InitWithEmptyNestedData_ProducesNullDataAndScalarShape) {
    TensorData<int> empty(hahaha::math::NestedData<int>{});
    EXPECT_EQ(empty.getShape().getDims().size(), 0);
    EXPECT_EQ(empty.getData().get(), nullptr);
}

TEST_F(TensorDataTest, DefaultConstructor) {
    TensorData<float> defaultConstructedTensor;
    EXPECT_EQ(defaultConstructedTensor.getShape().getDims().size(), 0);
    EXPECT_EQ(defaultConstructedTensor.getData().get(), nullptr);
}

TEST_F(TensorDataTest, ShapeOnlyConstructor_DefaultDeviceAllocates) {
    hahaha::math::TensorShape shape({2, 2});
    TensorData<int> td(shape);
    EXPECT_EQ(td.getShape().getTotalSize(), 4);
    EXPECT_NE(td.getData().get(), nullptr);
}

TEST_F(TensorDataTest, ShapeValueConstructor_GpuDevice_ThrowsRuntimeError) {
    hahaha::math::TensorShape shape({2, 2});
    EXPECT_THROW(
        TensorData<int>(
            shape, 1, (std::make_shared<hahaha::backend::GPUDevice>())),
        std::runtime_error);
}

TEST_F(TensorDataTest, ShapeOnlyConstructor_GpuDevice_ThrowsRuntimeError) {
    hahaha::math::TensorShape shape({2, 2});
    EXPECT_THROW(TensorData<int>(
                     shape, (std::make_shared<hahaha::backend::GPUDevice>())),
                 std::runtime_error);
}

TEST_F(TensorDataTest, InitVecConstructor_Creates1DTensor) {
    std::vector<int> vec = {7, 8, 9};
    TensorData<int> td(vec);
    EXPECT_EQ(td.getShape().getDims().size(), 1);
    EXPECT_EQ(td.getShape().getDims()[0], 3);
    EXPECT_EQ(td.getStride().getStrideSize(), 1);
    EXPECT_EQ(td.getStride()[0], 1);
    EXPECT_EQ(td.getData()[0], 7);
    EXPECT_EQ(td.getData()[2], 9);
}

TEST_F(TensorDataTest, ShapeValueConstructor) {
    hahaha::math::TensorShape shape({2, 3});
    TensorData<int> tensor_data(shape, 7);
    EXPECT_EQ(tensor_data.getShape().getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(tensor_data.getData()[i], 7);
    }
    EXPECT_EQ(tensor_data.getStride().getStrideSize(), 2);
    EXPECT_EQ(tensor_data.getStride()[0], 3);
    EXPECT_EQ(tensor_data.getStride()[1], 1);
}

TEST_F(TensorDataTest, OneDimensionalTensor) {
    TensorData<int> tensor_1d(hahaha::math::NestedData<int>{1, 2, 3, 4, 5});
    EXPECT_EQ(tensor_1d.getShape().getTotalSize(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(tensor_1d.getData()[i], i + 1);
    }
}

TEST_F(TensorDataTest, CopyConstructor) {
    TensorData<int> original(hahaha::math::NestedData<int>{{1, 2}, {3, 4}});
    TensorData<int> copied(original);

    EXPECT_EQ(copied.getShape(), original.getShape());
    EXPECT_EQ(copied.getStride().getStrideSize(),
              original.getStride().getStrideSize());
    for (size_t i = 0; i < original.getShape().getTotalSize(); ++i) {
        EXPECT_EQ(copied.getData()[i], original.getData()[i]);
    }

    // Verify deep copy
    copied.getData()[0] = 100;
    EXPECT_EQ(original.getData()[0], 1);
    EXPECT_EQ(copied.getData()[0], 100);
}

TEST_F(TensorDataTest, MoveConstructor) {
    TensorData<int> original(hahaha::math::NestedData<int>{1, 2, 3});
    void* originalPtr = original.getData().get();

    TensorData<int> moved(std::move(original));

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
    EXPECT_EQ(original.getShape().getDims().size(), 0);
}

TEST_F(TensorDataTest, MoveAssignment) {
    TensorData<int> original(hahaha::math::NestedData<int>{1, 2, 3});
    void* originalPtr = original.getData().get();
    TensorData<int> moved;

    moved = std::move(original);

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
}

TEST_F(TensorDataTest, SettersAndGetters) {
    TensorData<int> td;
    auto data = std::make_unique<int[]>(4);
    data[0] = 10;

    td.setData(std::move(data));
    td.setShape(hahaha::math::TensorShape({2, 2}));
    td.setStride(hahaha::math::TensorStride(hahaha::math::TensorShape({2, 2})));

    EXPECT_EQ(td.getData()[0], 10);
    EXPECT_EQ(td.getShape().getTotalSize(), 4);
    EXPECT_EQ(td.getStride()[0], 2);
}

TEST_F(TensorDataTest, Share_SharesBufferButCopiesMetadata) {
    TensorData<int> original(hahaha::math::TensorShape({2, 2}), 3);
    auto shared = original.share();

    // Shares underlying buffer
    EXPECT_EQ(shared.getData().get(), original.getData().get());
    // Metadata is value-copied
    EXPECT_EQ(shared.getShape(), original.getShape());
    EXPECT_EQ(shared.getStride().toString(), original.getStride().toString());

    // Mutating shared data mutates original data (same buffer)
    shared.getData()[0] = 42;
    EXPECT_EQ(original.getData()[0], 42);

    // Mutating metadata on shared should not affect original
    shared.setShape(hahaha::math::TensorShape({4}));
    EXPECT_NE(shared.getShape(), original.getShape());
}

TEST_F(TensorDataTest, Device_GetSet_Works) {
    TensorData<int> td(hahaha::math::TensorShape({1}), 1);
    EXPECT_EQ(td.getDevice()->getType(), hahaha::backend::DeviceType::CPU);
    td.setDevice((std::make_shared<hahaha::backend::GPUDevice>()));
    EXPECT_EQ(td.getDevice()->getType(), hahaha::backend::DeviceType::GPU);
}

TEST_F(TensorDataTest, OperatorIndex_ReferencesUnderlyingData) {
    TensorData<int> td(hahaha::math::TensorShape({3}), 0);
    td[1] = 123;
    EXPECT_EQ(td.getData()[1], 123);
}

// TEST_F(TensorDataTest, CopyConstructor_SIMDDevice) {
//     hahaha::math::TensorShape shape({2, 2});
//     TensorData<int> original(shape, 10,
//     (std::make_shared<hahaha::backend::CPUDevice>())); TensorData<int>
//     copied(original);
//
//     EXPECT_EQ(copied.getDevice()->getType(),
//     hahaha::backend::DeviceType::SIMD); EXPECT_EQ(copied.getData()[0], 10);
// }

// TEST_F(TensorDataTest, CopyConstructor_GpuDevice_Throws) {
//     hahaha::math::TensorShape shape({1});
//     TensorData<int> original;
//     // Hack to set device to GPU since constructor throws
//     original.setDevice(std::make_shared<hahaha::backend::CPUDevice>(
//         hahaha::backend::CPUDevice()));
//
//     // We can't easily populate data for GPU yet as it's not implemented,
//     // but the copy constructor should check device->getType() first.
//     // However, original.data_ will be null. Let's see if we can trigger the
//     // throw.
//
//     // Actually, looking at TensorData.h:
//     // TensorData(const TensorData& other) : ... {
//     //    if (device_->getType() == CPU || SIMD) { ... } else { throw ... }
//     // }
//
//     EXPECT_THROW(TensorData<int> copied(original), std::runtime_error);
// }

TEST_F(TensorDataTest, Share_NullData) {
    TensorData<int> original;
    auto shared = original.share();
    EXPECT_EQ(shared.getData().get(), nullptr);
    EXPECT_EQ(shared.getShape().getDims().size(), 0);
}

TEST_F(TensorDataTest, DoubleTypeTensor) {
    TensorData<double> td(hahaha::math::TensorShape({2}), 3.14);
    EXPECT_DOUBLE_EQ(td[0], 3.14);
    EXPECT_DOUBLE_EQ(td[1], 3.14);
}

TEST_F(TensorDataTest, Share_FromNestedData) {
    TensorData<int> td(hahaha::math::NestedData<int>{{1, 2}, {3, 4}});
    auto shared = td.share();
    EXPECT_EQ(shared[0], 1);
    EXPECT_EQ(shared[3], 4);
}

// ============================================================================
// Dimension-specific tests: 0D to 3D initialization
// ============================================================================

TEST_F(TensorDataTest, ZeroDimension_Scalar) {
    // 0D scalar: SingleValueConstruction
    auto nData = hahaha::math::NestedData(42);
    TensorData td(std::move(nData));
    EXPECT_EQ(td.getShape().getDims().size(), 0);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], 42);
}

TEST_F(TensorDataTest, OneDimension_Vector) {
    // 1D: [1, 2, 3]
    TensorData<int> td(hahaha::math::NestedData<int>{1, 2, 3});
    EXPECT_EQ(td.getShape().getDims().size(), 1);
    EXPECT_EQ(td.getShape().getDims()[0], 3);
    EXPECT_EQ(td.getShape().getTotalSize(), 3);
    EXPECT_EQ(td.getData()[0], 1);
    EXPECT_EQ(td.getData()[2], 3);
}

TEST_F(TensorDataTest, OneDimension_SingleElement) {
    // 1D with single element: [1] (different from scalar)
    TensorData<int> td(hahaha::math::NestedData<int>{1});
    EXPECT_EQ(td.getShape().getDims().size(), 1);
    EXPECT_EQ(td.getShape().getDims()[0], 1);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], 1);
}

TEST_F(TensorDataTest, TwoDimension_Matrix) {
    // 2D: {{1, 2}, {3, 4}}
    TensorData<int> td(hahaha::math::NestedData<int>{{1, 2}, {3, 4}});
    EXPECT_EQ(td.getShape().getDims().size(), 2);
    EXPECT_EQ(td.getShape().getDims()[0], 2);
    EXPECT_EQ(td.getShape().getDims()[1], 2);
    EXPECT_EQ(td.getShape().getTotalSize(), 4);
    EXPECT_EQ(td.getData()[0], 1);
    EXPECT_EQ(td.getData()[3], 4);
}

TEST_F(TensorDataTest, TwoDimension_SingleElement) {
    // 2D with single element: {{1}} (high-dimensional scalar)
    TensorData<int> td(hahaha::math::NestedData<int>{{1}});
    EXPECT_EQ(td.getShape().getDims().size(), 2);
    EXPECT_EQ(td.getShape().getDims()[0], 1);
    EXPECT_EQ(td.getShape().getDims()[1], 1);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], 1);
}

TEST_F(TensorDataTest, ThreeDimension_Tensor) {
    // 3D: {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}
    TensorData<int> td(
        hahaha::math::NestedData<int>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(td.getShape().getDims().size(), 3);
    EXPECT_EQ(td.getShape().getDims()[0], 2);
    EXPECT_EQ(td.getShape().getDims()[1], 2);
    EXPECT_EQ(td.getShape().getDims()[2], 2);
    EXPECT_EQ(td.getShape().getTotalSize(), 8);
    EXPECT_EQ(td.getData()[0], 1);
    EXPECT_EQ(td.getData()[7], 8);
}

TEST_F(TensorDataTest, ThreeDimension_SingleElement) {
    // 3D with single element: {{{1}}} (3D scalar)
    TensorData<int> td(hahaha::math::NestedData<int>{{{1}}});
    EXPECT_EQ(td.getShape().getDims().size(), 3);
    EXPECT_EQ(td.getShape().getDims()[0], 1);
    EXPECT_EQ(td.getShape().getDims()[1], 1);
    EXPECT_EQ(td.getShape().getDims()[2], 1);
    EXPECT_EQ(td.getShape().getTotalSize(), 1);
    EXPECT_EQ(td.getData()[0], 1);
}

// ============================================================================
// ShapeValueConstructor Tests - All Dimensions (0D to 3D)
// ============================================================================

TEST_F(TensorDataTest, ShapeValueConstructor_0D_Scalar) {
    hahaha::math::TensorShape shape0D({});
    TensorData<int> td0D(shape0D, 42);
    EXPECT_EQ(td0D.getShape().getDims().size(), 0);
    EXPECT_EQ(td0D.getShape().getTotalSize(), 1);
    EXPECT_EQ(td0D.getData()[0], 42);
}

TEST_F(TensorDataTest, ShapeValueConstructor_1D_Vector) {
    hahaha::math::TensorShape shape1D({3});
    TensorData<int> td1D(shape1D, 7);
    EXPECT_EQ(td1D.getShape().getDims().size(), 1);
    EXPECT_EQ(td1D.getShape().getTotalSize(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(td1D.getData()[i], 7);
    }
}

TEST_F(TensorDataTest, ShapeValueConstructor_2D_Matrix) {
    hahaha::math::TensorShape shape2D({2, 3});
    TensorData<int> td2D(shape2D, 5);
    EXPECT_EQ(td2D.getShape().getDims().size(), 2);
    EXPECT_EQ(td2D.getShape().getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(td2D.getData()[i], 5);
    }
}

TEST_F(TensorDataTest, ShapeValueConstructor_3D_Tensor) {
    hahaha::math::TensorShape shape3D({2, 2, 2});
    TensorData<int> td3D(shape3D, 9);
    EXPECT_EQ(td3D.getShape().getDims().size(), 3);
    EXPECT_EQ(td3D.getShape().getTotalSize(), 8);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(td3D.getData()[i], 9);
    }
}
