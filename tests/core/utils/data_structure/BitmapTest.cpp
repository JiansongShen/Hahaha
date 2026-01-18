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
#include <stdexcept>

using namespace hahaha::utils;

class BitmapTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up if needed
    }

    void TearDown() override {
        // Clean up if needed
    }
};

TEST_F(BitmapTest, Constructor) {
    Bitmap bitmap(32);
    EXPECT_EQ(bitmap.size(), 32);
    // Initially all bits should be 0
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FALSE(bitmap.get(i));
    }
}

TEST_F(BitmapTest, ConstructorWithInitialValue) {
    Bitmap bitmap(16, true); // Initialize all bits to true
    EXPECT_EQ(bitmap.size(), 16);
    // All bits should be 1
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_TRUE(bitmap.get(i));
    }

    Bitmap bitmap2(16, false); // Initialize all bits to false
    EXPECT_EQ(bitmap2.size(), 16);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FALSE(bitmap2.get(i));
    }
}

TEST_F(BitmapTest, SetAndGetBits) {
    Bitmap bitmap(8);

    // Initially all should be false
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FALSE(bitmap.get(i));
    }

    // Set some bits to true
    bitmap.set(0, true);
    bitmap.set(2, true);
    bitmap.set(7, true);

    // Check that only those bits are true
    EXPECT_TRUE(bitmap.get(0));
    EXPECT_FALSE(bitmap.get(1));
    EXPECT_TRUE(bitmap.get(2));
    EXPECT_FALSE(bitmap.get(3));
    EXPECT_FALSE(bitmap.get(4));
    EXPECT_FALSE(bitmap.get(5));
    EXPECT_FALSE(bitmap.get(6));
    EXPECT_TRUE(bitmap.get(7));
}

TEST_F(BitmapTest, SetBitOutOfRange) {
    Bitmap bitmap(8);

    // Setting bit out of range should throw exception
    EXPECT_THROW(bitmap.set(10, true), std::out_of_range);
    EXPECT_THROW(bitmap.set(8, true), std::out_of_range);

    // Getting bit out of range should throw exception - cast to void to ignore
    // nodiscard warning
    EXPECT_THROW((void) bitmap.get(10), std::out_of_range);
    EXPECT_THROW((void) bitmap.get(8), std::out_of_range);
}

TEST_F(BitmapTest, FlipBit) {
    Bitmap bitmap(8);

    // Start with a bit at false
    EXPECT_FALSE(bitmap.get(3));

    // Flip it to true
    bitmap.flip(3);
    EXPECT_TRUE(bitmap.get(3));

    // Flip it back to false
    bitmap.flip(3);
    EXPECT_FALSE(bitmap.get(3));
}

TEST_F(BitmapTest, FlipOutOfRange) {
    Bitmap bitmap(8);

    // Flipping out of range should throw exception
    EXPECT_THROW(bitmap.flip(10), std::out_of_range);
    EXPECT_THROW(bitmap.flip(8), std::out_of_range);
}

TEST_F(BitmapTest, CountBits) {
    Bitmap bitmap(16);

    // Initially no bits set
    EXPECT_EQ(bitmap.count(), 0);

    // Set a few bits
    bitmap.set(0, true);
    bitmap.set(5, true);
    bitmap.set(10, true);
    bitmap.set(15, true);

    EXPECT_EQ(bitmap.count(), 4);

    // Set another bit
    bitmap.set(3, true);
    EXPECT_EQ(bitmap.count(), 5);

    // Unset a bit
    bitmap.set(5, false);
    EXPECT_EQ(bitmap.count(), 4);
}

TEST_F(BitmapTest, Resize) {
    Bitmap bitmap(8);

    EXPECT_EQ(bitmap.size(), 8);

    // Resize larger
    bitmap.resize(16);
    EXPECT_EQ(bitmap.size(), 16);

    // Resize smaller
    bitmap.resize(4);
    EXPECT_EQ(bitmap.size(), 4);

    // Check that original bits are preserved when expanding and shrinking
    Bitmap bitmap2(4);
    bitmap2.set(0, true);
    bitmap2.set(3, true);

    bitmap2.resize(8); // Expand
    EXPECT_TRUE(bitmap2.get(0));
    EXPECT_FALSE(bitmap2.get(1));
    EXPECT_FALSE(bitmap2.get(2));
    EXPECT_TRUE(bitmap2.get(3));

    bitmap2.resize(2); // Shrink - should preserve first 2 bits
    EXPECT_TRUE(bitmap2.get(0));
    EXPECT_FALSE(bitmap2.get(1));
}

TEST_F(BitmapTest, ResizeWithDefaultValue) {
    Bitmap bitmap(4, true);       // All bits initially true
    EXPECT_EQ(bitmap.count(), 4); // All 4 bits set

    bitmap.resize(8, false); // Expand, new bits should be false
    EXPECT_EQ(bitmap.size(), 8);
    EXPECT_EQ(bitmap.count(), 4); // Only original 4 bits should still be true

    bitmap.resize(12, true); // Expand again, new bits should be true
    EXPECT_EQ(bitmap.size(), 12);
    EXPECT_EQ(bitmap.count(), 8); // Original 4 + 4 new true bits
}

TEST_F(BitmapTest, Clear) {
    Bitmap bitmap(8, true); // All bits initially true
    EXPECT_EQ(bitmap.count(), 8);

    bitmap.clear();
    EXPECT_EQ(bitmap.count(), 0);

    // All bits should be false now
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FALSE(bitmap.get(i));
    }
}

TEST_F(BitmapTest, All) {
    Bitmap bitmap(8, false); // All bits initially false
    EXPECT_FALSE(bitmap.all());

    // Set all bits to true
    for (size_t i = 0; i < 8; ++i) {
        bitmap.set(i, true);
    }
    EXPECT_TRUE(bitmap.all());

    // Unset one bit
    bitmap.set(5, false);
    EXPECT_FALSE(bitmap.all());
}

TEST_F(BitmapTest, None) {
    Bitmap bitmap(8, false); // All bits initially false
    EXPECT_TRUE(bitmap.none());

    // Set one bit to true
    bitmap.set(3, true);
    EXPECT_FALSE(bitmap.none());

    // Unset that bit
    bitmap.set(3, false);
    EXPECT_TRUE(bitmap.none());

    // Set all bits to true and check
    for (size_t i = 0; i < 8; ++i) {
        bitmap.set(i, true);
    }
    EXPECT_FALSE(bitmap.none());
}

TEST_F(BitmapTest, Any) {
    Bitmap bitmap(8, false); // All bits initially false
    EXPECT_FALSE(bitmap.any());

    // Set one bit to true
    bitmap.set(4, true);
    EXPECT_TRUE(bitmap.any());

    // Unset that bit
    bitmap.set(4, false);
    EXPECT_FALSE(bitmap.any());

    // Set all bits to true and check
    for (size_t i = 0; i < 8; ++i) {
        bitmap.set(i, true);
    }
    EXPECT_TRUE(bitmap.any());
}

TEST_F(BitmapTest, AssignmentOperator) {
    Bitmap bitmap1(16);
    bitmap1.set(0, true);
    bitmap1.set(5, true);
    bitmap1.set(15, true);

    Bitmap bitmap2(8);

    // Assign bitmap1 to bitmap2
    bitmap2 = bitmap1;

    // bitmap2 should now have the same size and values as bitmap1
    EXPECT_EQ(bitmap2.size(), 16);
    EXPECT_EQ(bitmap2.count(), 3);
    EXPECT_TRUE(bitmap2.get(0));
    EXPECT_FALSE(bitmap2.get(1));
    EXPECT_FALSE(bitmap2.get(4));
    EXPECT_TRUE(bitmap2.get(5));
    EXPECT_FALSE(bitmap2.get(14));
    EXPECT_TRUE(bitmap2.get(15));
}

TEST_F(BitmapTest, CopyConstructor) {
    Bitmap bitmap1(12);
    bitmap1.set(2, true);
    bitmap1.set(7, true);
    bitmap1.set(11, true);

    // Copy construct bitmap2 from bitmap1
    Bitmap bitmap2(bitmap1);

    // bitmap2 should have the same values as bitmap1
    EXPECT_EQ(bitmap2.size(), 12);
    EXPECT_EQ(bitmap2.count(), 3);
    EXPECT_FALSE(bitmap2.get(0));
    EXPECT_FALSE(bitmap2.get(1));
    EXPECT_TRUE(bitmap2.get(2));
    EXPECT_FALSE(bitmap2.get(6));
    EXPECT_TRUE(bitmap2.get(7));
    EXPECT_FALSE(bitmap2.get(10));
    EXPECT_TRUE(bitmap2.get(11));
}

TEST_F(BitmapTest, MoveConstructor) {
    Bitmap bitmap1(10);
    bitmap1.set(2, true);
    bitmap1.set(8, true);

    Bitmap bitmap2(std::move(bitmap1));

    // bitmap2 should have the values that bitmap1 had
    EXPECT_EQ(bitmap2.size(), 10);
    EXPECT_EQ(bitmap2.count(), 2);
    EXPECT_FALSE(bitmap2.get(0));
    EXPECT_TRUE(bitmap2.get(2));
    EXPECT_FALSE(bitmap2.get(7));
    EXPECT_TRUE(bitmap2.get(8));

    // bitmap1 should now be empty (though this depends on implementation)
    // Since we moved the data, bitmap1 is in a valid but unspecified state
}

TEST_F(BitmapTest, MoveAssignment) {
    Bitmap bitmap1(10);
    bitmap1.set(2, true);
    bitmap1.set(8, true);

    Bitmap bitmap2(5);
    bitmap2 = std::move(bitmap1);

    // bitmap2 should have the values that bitmap1 had
    EXPECT_EQ(bitmap2.size(), 10);
    EXPECT_EQ(bitmap2.count(), 2);
    EXPECT_FALSE(bitmap2.get(0));
    EXPECT_TRUE(bitmap2.get(2));
    EXPECT_FALSE(bitmap2.get(7));
    EXPECT_TRUE(bitmap2.get(8));
}

TEST_F(BitmapTest, SetAll) {
    Bitmap bitmap(8);
    EXPECT_EQ(bitmap.count(), 0);

    bitmap.setAll();
    EXPECT_EQ(bitmap.count(), 8);

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(bitmap.get(i));
    }
}

TEST_F(BitmapTest, ZeroSizeBitmap) {
    Bitmap bitmap(0);
    EXPECT_EQ(bitmap.size(), 0);
    EXPECT_EQ(bitmap.count(), 0);
    EXPECT_TRUE(bitmap.all());  // vacuously true
    EXPECT_TRUE(bitmap.none()); // vacuously true
    EXPECT_FALSE(bitmap.any()); // no bits to be true

    // Cast to void to ignore nodiscard warning
    EXPECT_THROW((void) bitmap.get(0), std::out_of_range);
    EXPECT_THROW(bitmap.set(0, true), std::out_of_range);
    EXPECT_THROW(bitmap.setOn(0), std::out_of_range);
    EXPECT_THROW(bitmap.setOff(0), std::out_of_range);
    EXPECT_THROW(bitmap.flip(0), std::out_of_range);

    // Operations on zero-size bitmap should not crash
    bitmap.clear();
    bitmap.setAll();
    bitmap.expand(5);
    EXPECT_EQ(bitmap.size(), 5);
}

TEST_F(BitmapTest, Expand) {
    Bitmap bitmap(8);
    bitmap.set(7, true);

    bitmap.expand(8); // Now size 16
    EXPECT_EQ(bitmap.size(), 16);
    EXPECT_TRUE(bitmap.get(7));  // Original bit preserved
    EXPECT_FALSE(bitmap.get(8)); // New bits should be false

    bitmap.set(15, true);
    EXPECT_TRUE(bitmap.get(15));
    EXPECT_EQ(bitmap.count(), 2);
}

TEST_F(BitmapTest, ResizeDown) {
    Bitmap bitmap(16);
    bitmap.set(0, true);
    bitmap.set(10, true);
    bitmap.set(15, true);

    bitmap.resize(8);
    EXPECT_EQ(bitmap.size(), 8);
    EXPECT_TRUE(bitmap.get(0));   // Preserved
    EXPECT_FALSE(bitmap.get(7));  // Should be false
    EXPECT_EQ(bitmap.count(), 1); // Only bit 0 should be set
}
