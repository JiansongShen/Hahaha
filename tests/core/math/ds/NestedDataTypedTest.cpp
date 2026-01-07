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

// Define all supported numeric types
using NumericTypes =
    ::testing::Types<u8, i8, u16, i16, u32, i32, u64, i64, f32, f64>;

template <typename T> class NestedDataTypedTest : public ::testing::Test {
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

TYPED_TEST_SUITE(NestedDataTypedTest, NumericTypes);

// ============================================================================
// Basic Construction Tests
// ============================================================================

TYPED_TEST(NestedDataTypedTest, InitializeViaInitializerList) {
    using T = typename TestFixture::Type;
    NestedData<T> nd({T(1), T(2), T(3), T(4), T(5)});
    ASSERT_EQ(nd.getFlatData().size(), 5);
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape().at(0), 5);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(1), T(2));
    ASSERT_EQ(nd.getFlatData().at(2), T(3));
    ASSERT_EQ(nd.getFlatData().at(3), T(4));
    ASSERT_EQ(nd.getFlatData().at(4), T(5));
}

TYPED_TEST(NestedDataTypedTest, InitializeViaNestedInitializerList1) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {
        NestedData<T>({T(1), T(2), T(3)}),
        NestedData<T>({T(4), T(5), T(6)}),
        NestedData<T>({T(7), T(8), T(9)}),
    };
    ASSERT_EQ(nd.getFlatData().size(), 9);
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape().at(0), 3);
    ASSERT_EQ(nd.getShape().at(1), 3);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(1), T(2));
    ASSERT_EQ(nd.getFlatData().at(2), T(3));
    ASSERT_EQ(nd.getFlatData().at(3), T(4));
    ASSERT_EQ(nd.getFlatData().at(4), T(5));
    ASSERT_EQ(nd.getFlatData().at(5), T(6));
    ASSERT_EQ(nd.getFlatData().at(6), T(7));
    ASSERT_EQ(nd.getFlatData().at(7), T(8));
    ASSERT_EQ(nd.getFlatData().at(8), T(9));
}

TYPED_TEST(NestedDataTypedTest, InitializeViaNestedInitializerList2) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {
        {
            {T(1), T(2)},
            {T(3), T(4)},
        },
        {
            {T(5), T(6)},
            {T(7), T(8)},
        },
    };
    ASSERT_EQ(nd.getFlatData().size(), 8);
    ASSERT_EQ(nd.getShape().size(), 3);
    ASSERT_EQ(nd.getShape().at(0), 2);
    ASSERT_EQ(nd.getShape().at(1), 2);
    ASSERT_EQ(nd.getShape().at(2), 2);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(1), T(2));
    ASSERT_EQ(nd.getFlatData().at(2), T(3));
    ASSERT_EQ(nd.getFlatData().at(3), T(4));
    ASSERT_EQ(nd.getFlatData().at(4), T(5));
    ASSERT_EQ(nd.getFlatData().at(5), T(6));
    ASSERT_EQ(nd.getFlatData().at(6), T(7));
    ASSERT_EQ(nd.getFlatData().at(7), T(8));
}

TYPED_TEST(NestedDataTypedTest, InitializeViaNestedInitializerList3) {
    using T = typename TestFixture::Type;
    NestedData<T> nestedData({
        {
            {T(1), T(2)},
            {T(3), T(4)},
        },
        {
            {T(5), T(6)},
            {T(7), T(8)},
        },
    });

    ASSERT_EQ(nestedData.getFlatData().size(), 8);
    ASSERT_EQ(nestedData.getShape().size(), 3);
    ASSERT_EQ(nestedData.getShape().at(0), 2);
    ASSERT_EQ(nestedData.getShape().at(1), 2);
    ASSERT_EQ(nestedData.getShape().at(2), 2);
    ASSERT_EQ(nestedData.getFlatData().at(0), T(1));
    ASSERT_EQ(nestedData.getFlatData().at(1), T(2));
    ASSERT_EQ(nestedData.getFlatData().at(2), T(3));
    ASSERT_EQ(nestedData.getFlatData().at(3), T(4));
    ASSERT_EQ(nestedData.getFlatData().at(4), T(5));
    ASSERT_EQ(nestedData.getFlatData().at(5), T(6));
    ASSERT_EQ(nestedData.getFlatData().at(6), T(7));
    ASSERT_EQ(nestedData.getFlatData().at(7), T(8));
}

TYPED_TEST(NestedDataTypedTest, InitializeWithEmptyList) {
    using T = typename TestFixture::Type;
    ASSERT_NO_THROW(NestedData<T> nd({}));
    NestedData<T> nd({});
    ASSERT_EQ(nd.getFlatData().size(), 0);
    ASSERT_EQ(nd.getShape().size(), 0);
}

TYPED_TEST(NestedDataTypedTest, SingleValueConstruction) {
    using T = typename TestFixture::Type;
    T value = T(42);
    NestedData<T> nd(value);
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getFlatData()[0], value);
    ASSERT_EQ(nd.getShape().size(), 0);
}

TYPED_TEST(NestedDataTypedTest, ScalarImplicitConversion) {
    using T = typename TestFixture::Type;
    T value = T(5);
    NestedData<T> nd = value;
    EXPECT_EQ(nd.getFlatData().size(), 1);
    EXPECT_EQ(nd.getFlatData()[0], value);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TYPED_TEST(NestedDataTypedTest, InconsistentShapesThrows) {
    using T = typename TestFixture::Type;
    EXPECT_THROW((NestedData<T>{{T(1), T(2), T(3)}, {T(4), T(5)}}),
                 std::invalid_argument);
}

TYPED_TEST(NestedDataTypedTest, InconsistentShapesDeeper) {
    using T = typename TestFixture::Type;
    EXPECT_THROW((NestedData<T>{{{T(1), T(2)}, {T(3), T(4)}},
                                {{T(5), T(6), T(7)}, {T(8), T(9), T(10)}}}),
                 std::invalid_argument);
}

TYPED_TEST(NestedDataTypedTest, NestedEmptyListThrows) {
    using T = typename TestFixture::Type;
    // Current implementation: if one list is empty, its shape is empty.
    // If we have {{}, {1}}, the first has shape (), second has shape (1).
    // This should throw inconsistent shapes.
    EXPECT_THROW((NestedData<T>{{}, {T(1)}}), std::invalid_argument);
}

// ============================================================================
// Property Tests
// ============================================================================

TYPED_TEST(NestedDataTypedTest, Getters) {
    using T = typename TestFixture::Type;
    NestedData<T> nd({T(1), T(2)});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(1), T(2));
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape().at(0), 2);
}

TYPED_TEST(NestedDataTypedTest, DeepNesting) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {{{{T(1)}}}};
    ASSERT_EQ(nd.getFlatData().size(), 1);
    ASSERT_EQ(nd.getShape().size(), 4);
    ASSERT_EQ(nd.getShape().at(0), 1);
    ASSERT_EQ(nd.getShape().at(1), 1);
    ASSERT_EQ(nd.getShape().at(2), 1);
    ASSERT_EQ(nd.getShape().at(3), 1);
}

TYPED_TEST(NestedDataTypedTest, LargeList) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {
        T(1), T(2), T(3), T(4), T(5), T(6), T(7), T(8), T(9), T(10)};
    ASSERT_EQ(nd.getFlatData().size(), 10);
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape().at(0), 10);
}

TYPED_TEST(NestedDataTypedTest, SingleElementList) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {{T(1)}};
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape()[0], 1);
    ASSERT_EQ(nd.getShape()[1], 1);
}

// ============================================================================
// Boundary Cases: Large Values
// ============================================================================

TYPED_TEST(NestedDataTypedTest, LargeValues_Handling) {
    using T = typename TestFixture::Type;
    T largeVal = this->maxValue();
    NestedData<T> nd({largeVal, largeVal / T(2)});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    ASSERT_EQ(nd.getFlatData().at(0), largeVal);
    ASSERT_EQ(nd.getFlatData().at(1), largeVal / T(2));
}

TYPED_TEST(NestedDataTypedTest, SmallValues_Handling) {
    using T = typename TestFixture::Type;
    T smallVal = this->minValue();
    NestedData<T> nd({smallVal, T(0)});
    ASSERT_EQ(nd.getFlatData().size(), 2);
    ASSERT_EQ(nd.getFlatData().at(0), smallVal);
    ASSERT_EQ(nd.getFlatData().at(1), T(0));
}

// ============================================================================
// Boundary Cases: Floating Point Special Values
// ============================================================================

TYPED_TEST(NestedDataTypedTest, FloatingPoint_Infinity_Storage) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        NestedData<T> nd({inf, T(1.0)});
        ASSERT_EQ(nd.getFlatData().size(), 2);
        EXPECT_TRUE(this->isInf(nd.getFlatData().at(0)));
        EXPECT_EQ(nd.getFlatData().at(1), T(1.0));
    }
}

TYPED_TEST(NestedDataTypedTest, FloatingPoint_NaN_Storage) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T nan = std::numeric_limits<T>::quiet_NaN();
        NestedData<T> nd({nan, T(1.0)});
        ASSERT_EQ(nd.getFlatData().size(), 2);
        EXPECT_TRUE(this->isNaN(nd.getFlatData().at(0)));
        EXPECT_EQ(nd.getFlatData().at(1), T(1.0));
    }
}

TYPED_TEST(NestedDataTypedTest, FloatingPoint_NestedDataWithSpecialValues) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T inf = std::numeric_limits<T>::infinity();
        T nan = std::numeric_limits<T>::quiet_NaN();
        NestedData<T> nd = {{inf, nan}, {T(1.0), T(2.0)}};
        ASSERT_EQ(nd.getFlatData().size(), 4);
        EXPECT_TRUE(this->isInf(nd.getFlatData().at(0)));
        EXPECT_TRUE(this->isNaN(nd.getFlatData().at(1)));
        EXPECT_EQ(nd.getFlatData().at(2), T(1.0));
        EXPECT_EQ(nd.getFlatData().at(3), T(2.0));
    }
}

// ============================================================================
// Complex Nested Structures
// ============================================================================

TYPED_TEST(NestedDataTypedTest, ComplexNestedStructure_4D) {
    using T = typename TestFixture::Type;
    // Create a 4D structure: 2x2x2x2
    NestedData<T> nd = {
        {
            {{T(1), T(2)}, {T(3), T(4)}},
            {{T(5), T(6)}, {T(7), T(8)}},
        },
        {
            {{T(9), T(10)}, {T(11), T(12)}},
            {{T(13), T(14)}, {T(15), T(16)}},
        },
    };
    ASSERT_EQ(nd.getFlatData().size(), 16);
    ASSERT_EQ(nd.getShape().size(), 4);
    ASSERT_EQ(nd.getShape().at(0), 2);
    ASSERT_EQ(nd.getShape().at(1), 2);
    ASSERT_EQ(nd.getShape().at(2), 2);
    ASSERT_EQ(nd.getShape().at(3), 2);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(15), T(16));
}

TYPED_TEST(NestedDataTypedTest, RectangularNestedStructure) {
    using T = typename TestFixture::Type;
    // Create a 2x3 structure
    NestedData<T> nd = {
        {T(1), T(2), T(3)},
        {T(4), T(5), T(6)},
    };
    ASSERT_EQ(nd.getFlatData().size(), 6);
    ASSERT_EQ(nd.getShape().size(), 2);
    ASSERT_EQ(nd.getShape().at(0), 2);
    ASSERT_EQ(nd.getShape().at(1), 3);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(5), T(6));
}

TYPED_TEST(NestedDataTypedTest, VeryLargeNestedStructure) {
    using T = typename TestFixture::Type;
    // Create a large 1D structure
    NestedData<T> nd = {
        T(1),  T(2),  T(3),  T(4),  T(5),  T(6),  T(7),  T(8),  T(9),  T(10),
        T(11), T(12), T(13), T(14), T(15), T(16), T(17), T(18), T(19), T(20),
        T(21), T(22), T(23), T(24), T(25), T(26), T(27), T(28), T(29), T(30),
    };
    ASSERT_EQ(nd.getFlatData().size(), 30);
    ASSERT_EQ(nd.getShape().size(), 1);
    ASSERT_EQ(nd.getShape().at(0), 30);
    ASSERT_EQ(nd.getFlatData().at(0), T(1));
    ASSERT_EQ(nd.getFlatData().at(29), T(30));
}

// ============================================================================
// Edge Cases: Zero Values
// ============================================================================

TYPED_TEST(NestedDataTypedTest, ZeroValues_Handling) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {{T(0), T(0)}, {T(0), T(0)}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(nd.getFlatData().at(i), T(0));
    }
}

TYPED_TEST(NestedDataTypedTest, MixedZeroAndNonZero) {
    using T = typename TestFixture::Type;
    NestedData<T> nd = {{T(0), T(1)}, {T(2), T(0)}};
    ASSERT_EQ(nd.getFlatData().size(), 4);
    ASSERT_EQ(nd.getFlatData().at(0), T(0));
    ASSERT_EQ(nd.getFlatData().at(1), T(1));
    ASSERT_EQ(nd.getFlatData().at(2), T(2));
    ASSERT_EQ(nd.getFlatData().at(3), T(0));
}

// ============================================================================
// Precision Tests: Floating Point
// ============================================================================

TYPED_TEST(NestedDataTypedTest, FloatingPoint_Precision) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isFloatingPoint()) {
        T val1 = T(0.1);
        T val2 = T(0.2);
        T val3 = T(0.3);
        NestedData<T> nd = {{val1, val2}, {val3, T(0.4)}};
        ASSERT_EQ(nd.getFlatData().size(), 4);
        this->expectNear(val1, nd.getFlatData().at(0));
        this->expectNear(val2, nd.getFlatData().at(1));
        this->expectNear(val3, nd.getFlatData().at(2));
        this->expectNear(T(0.4), nd.getFlatData().at(3));
    }
}

// ============================================================================
// Integer Overflow in Nested Structures
// ============================================================================

TYPED_TEST(NestedDataTypedTest, IntegerOverflow_Values) {
    using T = typename TestFixture::Type;
    if constexpr (TestFixture::isUnsignedInteger()
                  || TestFixture::isSignedInteger()) {
        T maxVal = this->maxValue();
        T minVal = this->minValue();
        // Test that NestedData can store values at boundaries
        NestedData<T> nd = {{maxVal, minVal}, {T(0), T(1)}};
        ASSERT_EQ(nd.getFlatData().size(), 4);
        ASSERT_EQ(nd.getFlatData().at(0), maxVal);
        ASSERT_EQ(nd.getFlatData().at(1), minVal);
        ASSERT_EQ(nd.getFlatData().at(2), T(0));
        ASSERT_EQ(nd.getFlatData().at(3), T(1));
    }
}
