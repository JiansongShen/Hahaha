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

#include "common/Config.h"

#include <gtest/gtest.h>

using namespace hahaha::common;

class ConfigTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up if needed
    }

    void TearDown() override {
        // Clean up if needed
    }
};

TEST_F(ConfigTest, DefaultRequiresGradValue) {
    Config config;
    EXPECT_TRUE(config.defaultRequiresGrad);
}

TEST_F(ConfigTest, ModifyRequiresGradValue) {
    Config config;
    EXPECT_TRUE(config.defaultRequiresGrad);

    config.defaultRequiresGrad = false;
    EXPECT_FALSE(config.defaultRequiresGrad);

    config.defaultRequiresGrad = true;
    EXPECT_TRUE(config.defaultRequiresGrad);
}

TEST_F(ConfigTest, SingletonGetInstance) {
    Config& config1 = getConfig();
    Config& config2 = getConfig();

    // Both should refer to the same instance
    EXPECT_EQ(&config1, &config2);

    // Modify through one reference and check the other
    config1.defaultRequiresGrad = false;
    EXPECT_FALSE(config2.defaultRequiresGrad);

    config2.defaultRequiresGrad = true;
    EXPECT_TRUE(config1.defaultRequiresGrad);
}

TEST_F(ConfigTest, SingletonDefaultState) {
    Config& config = getConfig();
    EXPECT_TRUE(config.defaultRequiresGrad);

    // Change the value
    config.defaultRequiresGrad = false;
    EXPECT_FALSE(config.defaultRequiresGrad);
}
