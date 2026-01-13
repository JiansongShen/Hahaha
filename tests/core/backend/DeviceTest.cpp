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

#include "backend/Device.h"

#include <gtest/gtest.h>

using hahaha::backend::Device;
using hahaha::backend::DeviceType;

// TEST(DeviceTest, ToString_AllTypes) {
//     EXPECT_EQ(Device(DeviceType::CPU, 0).toString(), "CPU:0");
//     EXPECT_EQ(Device(DeviceType::GPU, 1).toString(), "GPU:1");
//     EXPECT_EQ(Device(DeviceType::SIMD, 2).toString(), "SIMD:2");
// }
//
// TEST(DeviceTest, Equality) {
//     Device d1(DeviceType::CPU, 0);
//     Device d2(DeviceType::CPU, 0);
//     Device d3(DeviceType::GPU, 0);
//
//     EXPECT_TRUE(d1 == d2);
//     EXPECT_FALSE(d1 == d3);
//     EXPECT_TRUE(d1 != d3);
// }
