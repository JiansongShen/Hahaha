//  Copyright (c) 2025 - 2026 Contributors of
//  Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//
//

#include "utils/data_structure/Bitmap.h"

#include <gtest/gtest.h>

using namespace hahaha::utils;

/**
 * @brief Tests for the Bitmap class construction.
 */
TEST(BitmapTest, Construction) {
    Bitmap bmp(100);
    // Verify initialization to 0
    // Loop principle: Many iterations
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FALSE(bmp.get(i));
    }
}

/**
 * @brief Test construction with size 0 (Edge case).
 */
TEST(BitmapTest, ConstructionZeroSize) {
    Bitmap bmp(0);
    // Boundary: Accessing index 0 on size 0 should throw
    EXPECT_THROW((void) bmp.get(0), std::out_of_range);
    EXPECT_THROW(bmp.setOn(0), std::out_of_range);
}

/**
 * @brief Tests setOn() and get() methods.
 */
TEST(BitmapTest, SetOnAndGet) {
    Bitmap bmp(64);

    // Boundary: First bit
    bmp.setOn(0);
    EXPECT_TRUE(bmp.get(0));

    // Boundary: Last bit
    bmp.setOn(63);
    EXPECT_TRUE(bmp.get(63));

    // Verify other bits are not affected
    EXPECT_FALSE(bmp.get(1));
    EXPECT_FALSE(bmp.get(62));
}

/**
 * @brief Tests crossing word boundaries.
 */
TEST(BitmapTest, WordBoundary) {
    // 64-bit words implies boundary at 63/64
    Bitmap bmp(128);

    bmp.setOn(63);
    EXPECT_TRUE(bmp.get(63));
    EXPECT_FALSE(bmp.get(64));

    bmp.setOn(64);
    EXPECT_TRUE(bmp.get(64));
    EXPECT_TRUE(bmp.get(63)); // Previous still set
}

/**
 * @brief Tests setOff() method.
 */
TEST(BitmapTest, SetOff) {
    Bitmap bmp(10);
    bmp.setOn(5);
    EXPECT_TRUE(bmp.get(5));
    bmp.setOff(5);
    EXPECT_FALSE(bmp.get(5));

    // Set off already off
    bmp.setOff(5);
    EXPECT_FALSE(bmp.get(5));
}

/**
 * @brief Tests set() method with boolean values.
 */
TEST(BitmapTest, SetValue) {
    Bitmap bmp(10);
    // Branch: value = true
    bmp.set(2, true);
    EXPECT_TRUE(bmp.get(2));

    // Branch: value = false
    bmp.set(2, false);
    EXPECT_FALSE(bmp.get(2));

    // Branch: value = false with out of range (Coverage for else branch ->
    // throw)
    EXPECT_THROW(bmp.set(100, false), std::out_of_range);
}

/**
 * @brief Tests clear() method.
 */
TEST(BitmapTest, Clear) {
    Bitmap bmp(100);
    bmp.setOn(10);
    bmp.setOn(90);
    bmp.clear();
    EXPECT_FALSE(bmp.get(10));
    EXPECT_FALSE(bmp.get(90));
}

/**
 * @brief Tests clear() with 1 word (Loop 1 time).
 */
TEST(BitmapTest, ClearOneWord) {
    Bitmap bmp(1);
    bmp.setOn(0);
    bmp.clear();
    EXPECT_FALSE(bmp.get(0));
}

/**
 * @brief Tests clear() on empty bitmap (Loop 0 times logic).
 */
TEST(BitmapTest, ClearEmpty) {
    Bitmap bmp(0);
    EXPECT_NO_THROW(bmp.clear());
}

/**
 * @brief Tests out of range access handling.
 */
TEST(BitmapTest, OutOfRange) {
    Bitmap bmp(10);
    // Boundary: Exactly size
    EXPECT_THROW(bmp.setOn(10), std::out_of_range);
    EXPECT_THROW(bmp.setOff(10), std::out_of_range);
    EXPECT_THROW((void) bmp.get(10), std::out_of_range);
    EXPECT_THROW(bmp.set(10, true), std::out_of_range);

    // Boundary: Much larger
    EXPECT_THROW((void) bmp.get(100), std::out_of_range);
}

/**
 * @brief Tests expand() method logic.
 * Loop principle: Copy loop runs 1 time (size 64 = 1 word).
 */
TEST(BitmapTest, ExpandOneWord) {
    Bitmap bmp(64); // 1 word
    bmp.setOn(63);
    bmp.expand(64); // Now 128, 2 words

    // Check old values preserved
    EXPECT_TRUE(bmp.get(63));

    // Check new values are accessible
    bmp.setOn(127);
    EXPECT_TRUE(bmp.get(127));

    // Check new bounds
    EXPECT_THROW((void) bmp.get(128), std::out_of_range);
}

/**
 * @brief Tests expand() from 0 size.
 * Loop principle: Copy loop runs 0 times.
 */
TEST(BitmapTest, ExpandFromZero) {
    Bitmap bmp(0);
    bmp.expand(64);

    bmp.setOn(0);
    EXPECT_TRUE(bmp.get(0));
    EXPECT_THROW((void) bmp.get(64), std::out_of_range);
}

/**
 * @brief Tests expand() with multiple words.
 * Loop principle: Copy loop runs > 1 times (2 words).
 */
TEST(BitmapTest, ExpandMultipleWords) {
    // 130 bits -> 3 words (64, 64, 2)
    Bitmap bmp(130);
    bmp.setOn(0);
    bmp.setOn(64);
    bmp.setOn(129);

    // Expand by another 64 bits -> 194 bits -> 4 words
    bmp.expand(64);

    EXPECT_TRUE(bmp.get(0));
    EXPECT_TRUE(bmp.get(64));
    EXPECT_TRUE(bmp.get(129));

    // Test new area
    bmp.setOn(193);
    EXPECT_TRUE(bmp.get(193));
}

/**
 * @brief Tests expand() with 0 size addition.
 * Branch: expandSize == 0.
 */
TEST(BitmapTest, ExpandZeroIncrement) {
    Bitmap bmp(64);
    bmp.setOn(0);
    bmp.expand(0);
    EXPECT_TRUE(bmp.get(0));
    EXPECT_THROW((void) bmp.get(64), std::out_of_range);
}

/**
 * @brief Tests with large size bitmap.
 */
TEST(BitmapTest, LargeSize) {
    size_t size = 10000;
    Bitmap bmp(size);
    bmp.setOn(9999);
    EXPECT_TRUE(bmp.get(9999));
    bmp.setOn(0);
    EXPECT_TRUE(bmp.get(0));
}

/**
 * @brief Test clear() method with const qualifier to ensure it works properly
 */
TEST(BitmapTest, ConstClearMethod) {
    const Bitmap bmp(100);
    EXPECT_NO_THROW(bmp.clear());
}

/**
 * @brief Test constructor with very large size to ensure proper allocation
 */
TEST(BitmapTest, ConstructorLargeSize) {
    EXPECT_NO_THROW(Bitmap bmp(
        10000)); // Use a reasonable size that won't cause allocation failure
    Bitmap bmp(10000);
    EXPECT_THROW((void) bmp.get(10001),
                 std::out_of_range); // Ensure bounds checking still works
}

/**
 * @brief Test setOn method with invalid index at exact boundary
 */
TEST(BitmapTest, SetOnErrorBoundary) {
    Bitmap bmp(10);
    EXPECT_THROW(bmp.setOn(10), std::out_of_range);
    EXPECT_THROW(bmp.setOn(11), std::out_of_range);
}

/**
 * @brief Test setOff method with invalid index at exact boundary
 */
TEST(BitmapTest, SetOffErrorBoundary) {
    Bitmap bmp(10);
    EXPECT_THROW(bmp.setOff(10), std::out_of_range);
    EXPECT_THROW(bmp.setOff(11), std::out_of_range);
}

/**
 * @brief Test get method with invalid index at exact boundary
 */
TEST(BitmapTest, GetErrorBoundary) {
    Bitmap bmp(10);
    EXPECT_THROW((void) bmp.get(10), std::out_of_range);
    EXPECT_THROW((void) bmp.get(11), std::out_of_range);
}

/**
 * @brief Test set method with invalid index at exact boundary
 */
TEST(BitmapTest, SetErrorBoundary) {
    Bitmap bmp(10);
    EXPECT_THROW(bmp.set(10, true), std::out_of_range);
    EXPECT_THROW(bmp.set(10, false), std::out_of_range);
}

/**
 * @brief Test expand method with zero size to cover early return branch
 */
TEST(BitmapTest, ExpandZeroSize) {
    Bitmap bmp(100);
    bmp.setOn(50);
    EXPECT_TRUE(bmp.get(50));
    bmp.expand(0); // Should not change anything
    EXPECT_TRUE(bmp.get(50));
    EXPECT_EQ(bmp.get(49), false);
    EXPECT_EQ(bmp.get(51), false);
}

/**
 * @brief Test expand method with various sizes to ensure proper memory handling
 */
TEST(BitmapTest, ExpandVariousSizes) {
    Bitmap bmp(65); // Crosses word boundary
    bmp.setOn(0);
    bmp.setOn(64);

    bmp.expand(63); // Now should be 128 (exactly 2 words)
    EXPECT_TRUE(bmp.get(0));
    EXPECT_TRUE(bmp.get(64));

    // Test accessing the last possible bit in the new size
    EXPECT_NO_THROW(bmp.setOn(127));
    EXPECT_TRUE(bmp.get(127));

    // Ensure we can't access beyond the new size
    EXPECT_THROW((void) bmp.get(128), std::out_of_range);

    // Test expanding again
    bmp.expand(10);
    EXPECT_TRUE(bmp.get(127));
    bmp.setOn(137);
    EXPECT_TRUE(bmp.get(137));

    // Ensure we can't access beyond the new expanded size
    EXPECT_THROW((void) bmp.get(138), std::out_of_range);
}

/**
 * @brief Test edge case where size is exactly word boundary
 */
TEST(BitmapTest, WordBoundaryExact) {
    // Size exactly equal to word size (64 bits)
    Bitmap bmp(64);
    bmp.setOn(63); // Last bit in first word
    EXPECT_TRUE(bmp.get(63));

    // Ensure we can't access 64th bit (which would be in 2nd word)
    EXPECT_THROW((void) bmp.get(64), std::out_of_range);

    bmp.expand(1); // Now should be able to access 64th bit
    bmp.setOn(64);
    EXPECT_TRUE(bmp.get(64));
}

/**
 * @brief Test operations across multiple word boundaries
 */
TEST(BitmapTest, MultipleWordBoundaries) {
    Bitmap bmp(200); // More than 3 words

    // Test setting bits at word boundaries
    bmp.setOn(63);  // End of 1st word
    bmp.setOn(64);  // Beginning of 2nd word
    bmp.setOn(127); // End of 2nd word
    bmp.setOn(128); // Beginning of 3rd word
    bmp.setOn(199); // Last bit

    EXPECT_TRUE(bmp.get(63));
    EXPECT_TRUE(bmp.get(64));
    EXPECT_TRUE(bmp.get(127));
    EXPECT_TRUE(bmp.get(128));
    EXPECT_TRUE(bmp.get(199));

    // Make sure adjacent bits are not affected
    EXPECT_FALSE(bmp.get(62));
    EXPECT_FALSE(bmp.get(65));
    EXPECT_FALSE(bmp.get(126));
    EXPECT_FALSE(bmp.get(129));
    EXPECT_FALSE(bmp.get(198));
}

/**
 * @brief Test set method with true/false values specifically
 */
TEST(BitmapTest, SetTrueFalseValues) {
    Bitmap bmp(10);

    // Initially all bits are false
    for (int i = 0; i < 10; i++) {
        EXPECT_FALSE(bmp.get(i));
    }

    // Set bit to true using set method
    bmp.set(5, true);
    EXPECT_TRUE(bmp.get(5));

    // Set bit to false using set method
    bmp.set(5, false);
    EXPECT_FALSE(bmp.get(5));

    // Set to true again
    bmp.set(5, true);
    EXPECT_TRUE(bmp.get(5));
}
