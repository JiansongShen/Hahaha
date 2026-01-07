// Copyright (c) 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include "utils/log/LogLevel.h"

#include <gtest/gtest.h>

using hahaha::utils::LogLevel;
using hahaha::utils::toColoredString;
using hahaha::utils::toString;

TEST(LogLevelTest, ToString_InvalidValue_ReturnsUnknown) {
    auto invalid = static_cast<LogLevel>(999);
    EXPECT_EQ(toString(invalid), "UNKNOWN");
}

TEST(LogLevelTest, ToColoredString_InvalidValue_ReturnsUnknown) {
    auto invalid = static_cast<LogLevel>(999);
    EXPECT_EQ(toColoredString(invalid), "UNKNOWN");
}
