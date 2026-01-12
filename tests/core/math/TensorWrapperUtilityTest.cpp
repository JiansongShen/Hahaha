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

#include <gtest/gtest.h>

#include "backend/Device.h"
#include "common/definitions.h"
#include "math/TensorWrapper.h"
#include "math/ds/NestedData.h"

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

using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T>
class TensorWrapperUtilityTypedTest : public ::testing::Test {
  protected:
    using Type = T;
    void expectNear(T expected, T actual) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(static_cast<double>(expected), static_cast<double>(actual), 1e-5);
        } else {
            EXPECT_EQ(expected, actual);
        }
    }
};

TYPED_TEST_SUITE(TensorWrapperUtilityTypedTest, NumericTypes);

// ============================================================================
// Ones Tests
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, Ones_0D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(T(5));
    auto res = t.ones();
    EXPECT_EQ(res.getShape().size(), 0);
    this->expectNear(res.at({}), T(1));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Ones_1D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({3}), T(5));
    auto res = t.ones();
    EXPECT_EQ(res.getShape().size(), 1);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(1));
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Ones_2D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({2, 2}), T(5));
    auto res = t.ones();
    EXPECT_EQ(res.getShape().size(), 2);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(1));
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Ones_3D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({2, 1, 2}), T(5));
    auto res = t.ones();
    EXPECT_EQ(res.getShape().size(), 3);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(1));
    }
}

// ============================================================================
// Zeros Tests
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, Zeros_0D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(T(5));
    auto res = t.zeros();
    EXPECT_EQ(res.getShape().size(), 0);
    this->expectNear(res.at({}), T(0));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Zeros_1D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({3}), T(5));
    auto res = t.zeros();
    EXPECT_EQ(res.getShape().size(), 1);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(0));
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Zeros_2D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({2, 2}), T(5));
    auto res = t.zeros();
    EXPECT_EQ(res.getShape().size(), 2);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(0));
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, Zeros_3D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({2, 1, 2}), T(5));
    auto res = t.zeros();
    EXPECT_EQ(res.getShape().size(), 3);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(0));
    }
}

// ============================================================================
// SameShapeWithValue Tests
// ============================================================================

TYPED_TEST(TensorWrapperUtilityTypedTest, SameShapeWithValue_0D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(T(5));
    auto res = t.sameShapeWithValue(T(42));
    EXPECT_EQ(res.getShape().size(), 0);
    this->expectNear(res.at({}), T(42));
}

TYPED_TEST(TensorWrapperUtilityTypedTest, SameShapeWithValue_1D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({3}), T(5));
    auto res = t.sameShapeWithValue(T(42));
    EXPECT_EQ(res.getShape().size(), 1);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(42));
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, SameShapeWithValue_2D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({2, 2}), T(5));
    auto res = t.sameShapeWithValue(T(42));
    EXPECT_EQ(res.getShape().size(), 2);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(42));
    }
}

TYPED_TEST(TensorWrapperUtilityTypedTest, SameShapeWithValue_3D) {
    using T = TestFixture::Type;
    TensorWrapper<T> t(TensorShape({2, 1, 2}), T(5));
    auto res = t.sameShapeWithValue(T(42));
    EXPECT_EQ(res.getShape().size(), 3);
    for (size_t i = 0; i < res.getTotalSize(); ++i) {
        this->expectNear(res.getRawData()[i], T(42));
    }
}
