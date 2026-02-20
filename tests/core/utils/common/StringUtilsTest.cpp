// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com)
//

#include "utils/common/StringUtils.h"

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

using hahaha::utils::StringUtils;

// --- StringUtils::isBlank ---------------------------------------------------

class StringUtilsIsBlankTest : public ::testing::Test {};

TEST_F(StringUtilsIsBlankTest, EmptyString) {
    // loop body never executes → returns true
    EXPECT_TRUE(StringUtils::isBlank(""));
}

TEST_F(StringUtilsIsBlankTest, AllSpaces) {
    // loop exhausted without non-space → returns true
    EXPECT_TRUE(StringUtils::isBlank("   "));
}

TEST_F(StringUtilsIsBlankTest, ContainsNonSpace) {
    // non-space character found → early-return false
    EXPECT_FALSE(StringUtils::isBlank("a"));
    EXPECT_FALSE(StringUtils::isBlank(" x "));
}

// --- StringUtils::trimSideBlank ---------------------------------------------

class StringUtilsTrimTest : public ::testing::Test {};

TEST_F(StringUtilsTrimTest, NoTrim) {
    // both while-loops never enter
    EXPECT_EQ(StringUtils::trimSideBlank("hello"), "hello");
}

TEST_F(StringUtilsTrimTest, LeadingSpacesOnly) {
    // start-while enters; end-while does not
    EXPECT_EQ(StringUtils::trimSideBlank("   hello"), "hello");
}

TEST_F(StringUtilsTrimTest, TrailingSpacesOnly) {
    // end-while enters; start-while does not
    EXPECT_EQ(StringUtils::trimSideBlank("hello   "), "hello");
}

TEST_F(StringUtilsTrimTest, BothSides) {
    // both while-loops enter
    EXPECT_EQ(StringUtils::trimSideBlank("  hello  "), "hello");
}

TEST_F(StringUtilsTrimTest, SingleChar) {
    EXPECT_EQ(StringUtils::trimSideBlank("x"), "x");
}

// --- StringUtils::split -----------------------------------------------------

class StringUtilsSplitTest : public ::testing::Test {};

// keepBlankValue = false
TEST_F(StringUtilsSplitTest, EmptyInput_DropBlanks) {
    // single blank segment → dropped → empty result
    EXPECT_TRUE(StringUtils::split("", ',', false).empty());
}

TEST_F(StringUtilsSplitTest, NoDelimiter_NonBlank) {
    auto v = StringUtils::split("abc", ',', false);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], "abc");
}

TEST_F(StringUtilsSplitTest, MultipleNonBlankSegments) {
    auto v = StringUtils::split("a,b,c", ',', false);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b");
    EXPECT_EQ(v[2], "c");
}

TEST_F(StringUtilsSplitTest, ConsecutiveDelimiters_DropBlanks) {
    // blank segment between consecutive commas is dropped
    auto v = StringUtils::split("a,,c", ',', false);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "c");
}

TEST_F(StringUtilsSplitTest, SpaceOnlySegment_DropBlanks) {
    // isBlank(" ") == true → dropped
    auto v = StringUtils::split("a, ,c", ',', false);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "c");
}

TEST_F(StringUtilsSplitTest, TrailingDelimiter_DropBlanks) {
    // last segment "" is blank → dropped
    auto v = StringUtils::split("a,b,", ',', false);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b");
}

TEST_F(StringUtilsSplitTest, LeadingDelimiter_DropBlanks) {
    // first segment "" is blank → dropped
    auto v = StringUtils::split(",a,b", ',', false);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b");
}

// keepBlankValue = true
TEST_F(StringUtilsSplitTest, EmptyInput_KeepBlanks) {
    // single blank segment is kept
    auto v = StringUtils::split("", ',', true);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], "");
}

TEST_F(StringUtilsSplitTest, ConsecutiveDelimiters_KeepBlanks) {
    auto v = StringUtils::split("a,,c", ',', true);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[1], "");
}

TEST_F(StringUtilsSplitTest, TrailingDelimiter_KeepBlanks) {
    auto v = StringUtils::split("a,b,", ',', true);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[2], "");
}

TEST_F(StringUtilsSplitTest, SingleCharSegments_KeepBlanks) {
    // regression: single-char segments must not be mistaken for blank
    auto v = StringUtils::split("a,b,c", ',', true);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b");
    EXPECT_EQ(v[2], "c");
}

TEST_F(StringUtilsSplitTest, SpaceOnlySegment_KeepBlanks) {
    // isBlank(" ") but still kept when keepBlankValue=true
    auto v = StringUtils::split("a, ,c", ',', true);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[1], " ");
}

// --- StringUtils::to<T> -----------------------------------------------------

class StringUtilsParseTest : public ::testing::Test {};

TEST_F(StringUtilsParseTest, ValidInt) {
    // parse succeeds, no leftover → has_value
    auto r = StringUtils::to<int>("42");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 42);
}

TEST_F(StringUtilsParseTest, ValidFloat) {
    auto r = StringUtils::to<float>("3.14");
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 3.14f, 1e-5f);
}

TEST_F(StringUtilsParseTest, ZeroIsNotNullopt) {
    // regression: "0" is a valid value, not a parse failure
    auto r = StringUtils::to<int>("0");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 0);
}

TEST_F(StringUtilsParseTest, NonNumericString) {
    // iss >> value fails → nullopt
    EXPECT_FALSE(StringUtils::to<int>("abc").has_value());
}

TEST_F(StringUtilsParseTest, TrailingGarbage) {
    // parse succeeds but leftover remains → nullopt
    EXPECT_FALSE(StringUtils::to<float>("1.0abc").has_value());
}

TEST_F(StringUtilsParseTest, NegativeDouble) {
    auto r = StringUtils::to<double>("-2.5");
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, -2.5, 1e-9);
}
