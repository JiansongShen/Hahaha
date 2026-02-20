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
#include <limits>

#include "common/definitions.h"
#include "math/ds/NestedData.h"

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
class NestedDataErrorHandlingTypedTest : public ::testing::Test {
  protected:
    using Type = T;

    // Helper to check if type is floating point
    static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }
};

TYPED_TEST_SUITE(NestedDataErrorHandlingTypedTest, NumericTypes);

class NestedDataErrorHandlingTest : public ::testing::Test {};

// ============================================================================
// Error Handling Tests - Typed Test
// ============================================================================

TYPED_TEST(NestedDataErrorHandlingTypedTest, InconsistentShapes_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    EXPECT_THROW((NestedData<T>{{v1, v2, v3}, {v4, v5}}), std::invalid_argument);
}

TYPED_TEST(NestedDataErrorHandlingTypedTest, InconsistentShapesDeeper_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    T v2 = T(2);
    T v3 = T(3);
    T v4 = T(4);
    T v5 = T(5);
    T v6 = T(6);
    T v7 = T(7);
    T v8 = T(8);
    T v9 = T(9);
    T v10 = T(10);
    EXPECT_THROW(
        (NestedData<T>{{{v1, v2}, {v3, v4}}, {{v5, v6, v7}, {v8, v9, v10}}}),
        std::invalid_argument);
}

TYPED_TEST(NestedDataErrorHandlingTypedTest, NestedEmptyList_Throws) {
    using T = TestFixture::Type;
    T v1 = T(1);
    EXPECT_THROW((NestedData<T>{{}, {v1}}), std::invalid_argument);
}

TYPED_TEST(NestedDataErrorHandlingTypedTest, IntegerOverflow_HandlesGracefully) {
    using T = TestFixture::Type;
    // Test that large values don't cause overflow issues (behavior depends on
    // implementation)
    if constexpr (std::is_unsigned_v<T>) {
        // For unsigned types, max values should be accepted
        NestedData<T> nd({std::numeric_limits<T>::max()});
        ASSERT_EQ(nd.getFlatData().at(0), std::numeric_limits<T>::max());
    } else if constexpr (std::is_signed_v<T> && !TestFixture::isFloatingPoint()) {
        // For signed integer types, max and min values should be accepted
        NestedData<T> nd(
            {std::numeric_limits<T>::max(), std::numeric_limits<T>::min()});
        ASSERT_EQ(nd.getFlatData().at(0), std::numeric_limits<T>::max());
        ASSERT_EQ(nd.getFlatData().at(1), std::numeric_limits<T>::min());
    } else {
        // For floating point types, test max values
        NestedData<T> nd({std::numeric_limits<T>::max()});
        if constexpr (std::is_same_v<T, f64>) {
            EXPECT_DOUBLE_EQ(nd.getFlatData().at(0), std::numeric_limits<T>::max());
        } else {
            EXPECT_FLOAT_EQ(nd.getFlatData().at(0), std::numeric_limits<T>::max());
        }
    }
}
