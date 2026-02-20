//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "backend/vectorize/arch/simd_impl.h"
#include "backend/vectorize/simd_vec.h"
#include "common/definitions.h"

using namespace hahaha::backend;
using namespace hahaha::common;

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY

// ============================================================================
// 128-bit SIMD Tests (SSE2 baseline)
// ============================================================================

TEST(SimdVecX86_128, Float32_LoadStore) {
    alignas(16) f32 input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 output[4] = {0.0f};

    auto vec = SimdVec<f32, 128>::load(input);
    vec.store(output);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_128, Float32_LoadStoreAligned) {
    alignas(16) f32 input[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    alignas(16) f32 output[4] = {0.0f};

    auto vec = SimdVec<f32, 128>::load_aligned(input);
    vec.store_aligned(output);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_128, Float32_Addition) {
    alignas(16) f32 a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    alignas(16) f32 result[4] = {0.0f};

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);
    auto vc = va + vb;
    vc.store(result);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }
}

TEST(SimdVecX86_128, Float32_Subtraction) {
    alignas(16) f32 a[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    alignas(16) f32 b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 result[4] = {0.0f};

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);
    auto vc = va - vb;
    vc.store(result);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] - b[i]);
    }
}

TEST(SimdVecX86_128, Float32_Multiplication) {
    alignas(16) f32 a[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    alignas(16) f32 b[4] = {6.0f, 7.0f, 8.0f, 9.0f};
    alignas(16) f32 result[4] = {0.0f};

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);
    auto vc = va * vb;
    vc.store(result);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] * b[i]);
    }
}

TEST(SimdVecX86_128, Float32_Division) {
    alignas(16) f32 a[4] = {12.0f, 18.0f, 24.0f, 30.0f};
    alignas(16) f32 b[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    alignas(16) f32 result[4] = {0.0f};

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);
    auto vc = va / vb;
    vc.store(result);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] / b[i]);
    }
}

TEST(SimdVecX86_128, Float32_UnaryMinus) {
    alignas(16) f32 a[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    alignas(16) f32 result[4] = {0.0f};

    auto va = SimdVec<f32, 128>::load(a);
    auto vc = -va;
    vc.store(result);

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], -a[i]);
    }
}

TEST(SimdVecX86_128, Float32_CompoundAssignment) {
    alignas(16) f32 a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) f32 b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    alignas(16) f32 result[4] = {0.0f};

    auto va = SimdVec<f32, 128>::load(a);
    auto vb = SimdVec<f32, 128>::load(b);

    va += vb;
    va.store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }

    va = SimdVec<f32, 128>::load(a);
    va -= vb;
    va.store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] - b[i]);
    }

    va = SimdVec<f32, 128>::load(a);
    va *= vb;
    va.store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] * b[i]);
    }

    va = SimdVec<f32, 128>::load(a);
    va /= vb;
    va.store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] / b[i]);
    }
}

// Float64 Tests (128-bit)
TEST(SimdVecX86_128, Float64_LoadStore) {
    alignas(16) f64 input[2] = {1.0, 2.0};
    alignas(16) f64 output[2] = {0.0};

    auto vec = SimdVec<f64, 128>::load(input);
    vec.store(output);

    for (int i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_128, Float64_Arithmetic) {
    alignas(16) f64 a[2] = {10.5, 20.5};
    alignas(16) f64 b[2] = {1.5, 2.5};
    alignas(16) f64 result[2] = {0.0};

    auto va = SimdVec<f64, 128>::load(a);
    auto vb = SimdVec<f64, 128>::load(b);

    // Addition
    (va + vb).store(result);
    EXPECT_DOUBLE_EQ(result[0], 12.0);
    EXPECT_DOUBLE_EQ(result[1], 23.0);

    // Subtraction
    (va - vb).store(result);
    EXPECT_DOUBLE_EQ(result[0], 9.0);
    EXPECT_DOUBLE_EQ(result[1], 18.0);

    // Multiplication
    (va * vb).store(result);
    EXPECT_DOUBLE_EQ(result[0], 15.75);
    EXPECT_DOUBLE_EQ(result[1], 51.25);

    // Division
    (va / vb).store(result);
    EXPECT_DOUBLE_EQ(result[0], 7.0);
    EXPECT_DOUBLE_EQ(result[1], 8.2);
}

// Int32 Tests (128-bit)
TEST(SimdVecX86_128, Int32_LoadStore) {
    alignas(16) i32 input[4] = {1, 2, 3, 4};
    alignas(16) i32 output[4] = {0};

    auto vec = SimdVec<i32, 128>::load(input);
    vec.store(output);

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_128, Int32_AdditionSubtraction) {
    alignas(16) i32 a[4] = {10, 20, 30, 40};
    alignas(16) i32 b[4] = {1, 2, 3, 4};
    alignas(16) i32 result[4] = {0};

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    // Addition
    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], a[i] + b[i]);
    }

    // Subtraction
    (va - vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], a[i] - b[i]);
    }

    // Unary minus
    (-va).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], -a[i]);
    }
}

#if defined(__SSE4_1__)
TEST(SimdVecX86_128, Int32_Multiplication_SSE41) {
    alignas(16) i32 a[4] = {2, 3, 4, 5};
    alignas(16) i32 b[4] = {6, 7, 8, 9};
    alignas(16) i32 result[4] = {0};

    auto va = SimdVec<i32, 128>::load(a);
    auto vb = SimdVec<i32, 128>::load(b);

    (va * vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], a[i] * b[i]);
    }
}
#endif

// Int64 Tests (128-bit)
TEST(SimdVecX86_128, Int64_LoadStore) {
    alignas(16) i64 input[2] = {100, 200};
    alignas(16) i64 output[2] = {0};

    auto vec = SimdVec<i64, 128>::load(input);
    vec.store(output);

    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_128, Int64_Arithmetic) {
    alignas(16) i64 a[2] = {1000, 2000};
    alignas(16) i64 b[2] = {100, 200};
    alignas(16) i64 result[2] = {0};

    auto va = SimdVec<i64, 128>::load(a);
    auto vb = SimdVec<i64, 128>::load(b);

    // Addition
    (va + vb).store(result);
    EXPECT_EQ(result[0], 1100);
    EXPECT_EQ(result[1], 2200);

    // Subtraction
    (va - vb).store(result);
    EXPECT_EQ(result[0], 900);
    EXPECT_EQ(result[1], 1800);
}

// ============================================================================
// 256-bit SIMD Tests (AVX/AVX2)
// ============================================================================

#if defined(__AVX__)

TEST(SimdVecX86_256, Float32_LoadStore) {
    alignas(32) f32 input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(32) f32 output[8] = {0.0f};

    auto vec = SimdVec<f32, 256>::load(input);
    vec.store(output);

    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_256, Float32_Arithmetic) {
    alignas(32) f32 a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(32) f32 b[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    alignas(32) f32 result[8] = {0.0f};

    auto va = SimdVec<f32, 256>::load(a);
    auto vb = SimdVec<f32, 256>::load(b);

    // Test all operations
    (va + vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }

    (va - vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] - b[i]);
    }

    (va * vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] * b[i]);
    }

    (va / vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] / b[i]);
    }
}

TEST(SimdVecX86_256, Float64_LoadStore) {
    alignas(32) f64 input[4] = {1.0, 2.0, 3.0, 4.0};
    alignas(32) f64 output[4] = {0.0};

    auto vec = SimdVec<f64, 256>::load(input);
    vec.store(output);

    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_256, Float64_Arithmetic) {
    alignas(32) f64 a[4] = {10.5, 20.5, 30.5, 40.5};
    alignas(32) f64 b[4] = {1.5, 2.5, 3.5, 4.5};
    alignas(32) f64 result[4] = {0.0};

    auto va = SimdVec<f64, 256>::load(a);
    auto vb = SimdVec<f64, 256>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(result[i], a[i] + b[i]);
    }

    (va * vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(result[i], a[i] * b[i]);
    }
}

#endif // __AVX__

#if defined(__AVX2__)

TEST(SimdVecX86_256, Int32_LoadStore_AVX2) {
    alignas(32) i32 input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) i32 output[8] = {0};

    auto vec = SimdVec<i32, 256>::load(input);
    vec.store(output);

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_256, Int32_Arithmetic_AVX2) {
    alignas(32) i32 a[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(32) i32 b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) i32 result[8] = {0};

    auto va = SimdVec<i32, 256>::load(a);
    auto vb = SimdVec<i32, 256>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i], a[i] + b[i]);
    }

    (va - vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i], a[i] - b[i]);
    }

    (va * vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i], a[i] * b[i]);
    }
}

TEST(SimdVecX86_256, Int64_LoadStore_AVX2) {
    alignas(32) i64 input[4] = {100, 200, 300, 400};
    alignas(32) i64 output[4] = {0};

    auto vec = SimdVec<i64, 256>::load(input);
    vec.store(output);

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_256, Int64_Arithmetic_AVX2) {
    alignas(32) i64 a[4] = {1000, 2000, 3000, 4000};
    alignas(32) i64 b[4] = {100, 200, 300, 400};
    alignas(32) i64 result[4] = {0};

    auto va = SimdVec<i64, 256>::load(a);
    auto vb = SimdVec<i64, 256>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], a[i] + b[i]);
    }

    (va - vb).store(result);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], a[i] - b[i]);
    }
}

#endif // __AVX2__

// ============================================================================
// 512-bit SIMD Tests (AVX-512F)
// ============================================================================

#if defined(__AVX512F__)

TEST(SimdVecX86_512, Float32_LoadStore) {
    alignas(64) f32 input[16];
    alignas(64) f32 output[16] = {0.0f};

    for (int i = 0; i < 16; ++i) {
        input[i] = static_cast<f32>(i + 1);
    }

    auto vec = SimdVec<f32, 512>::load(input);
    vec.store(output);

    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_512, Float32_Arithmetic) {
    alignas(64) f32 a[16], b[16], result[16] = {0.0f};

    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<f32>(i + 1);
        b[i] = static_cast<f32>(16 - i);
    }

    auto va = SimdVec<f32, 512>::load(a);
    auto vb = SimdVec<f32, 512>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] + b[i]);
    }

    (va * vb).store(result);
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i] * b[i]);
    }
}

TEST(SimdVecX86_512, Float64_LoadStore) {
    alignas(64) f64 input[8];
    alignas(64) f64 output[8] = {0.0};

    for (int i = 0; i < 8; ++i) {
        input[i] = static_cast<f64>(i + 1) * 10.0;
    }

    auto vec = SimdVec<f64, 512>::load(input);
    vec.store(output);

    for (int i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_512, Int32_LoadStore) {
    alignas(64) i32 input[16];
    alignas(64) i32 output[16] = {0};

    for (int i = 0; i < 16; ++i) {
        input[i] = i * 10;
    }

    auto vec = SimdVec<i32, 512>::load(input);
    vec.store(output);

    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_512, Int32_Arithmetic) {
    alignas(64) i32 a[16], b[16], result[16] = {0};

    for (int i = 0; i < 16; ++i) {
        a[i] = i * 10;
        b[i] = i + 1;
    }

    auto va = SimdVec<i32, 512>::load(a);
    auto vb = SimdVec<i32, 512>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(result[i], a[i] + b[i]);
    }

    (va * vb).store(result);
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(result[i], a[i] * b[i]);
    }
}

TEST(SimdVecX86_512, Int64_LoadStore) {
    alignas(64) i64 input[8];
    alignas(64) i64 output[8] = {0};

    for (int i = 0; i < 8; ++i) {
        input[i] = i * 1000;
    }

    auto vec = SimdVec<i64, 512>::load(input);
    vec.store(output);

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(SimdVecX86_512, Int64_Arithmetic) {
    alignas(64) i64 a[8], b[8], result[8] = {0};

    for (int i = 0; i < 8; ++i) {
        a[i] = i * 1000;
        b[i] = i + 100;
    }

    auto va = SimdVec<i64, 512>::load(a);
    auto vb = SimdVec<i64, 512>::load(b);

    (va + vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i], a[i] + b[i]);
    }

    (va - vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i], a[i] - b[i]);
    }
}

#if defined(__AVX512DQ__)
TEST(SimdVecX86_512, Int64_Multiplication_AVX512DQ) {
    alignas(64) i64 a[8], b[8], result[8] = {0};

    for (int i = 0; i < 8; ++i) {
        a[i] = i + 2;
        b[i] = i + 3;
    }

    auto va = SimdVec<i64, 512>::load(a);
    auto vb = SimdVec<i64, 512>::load(b);

    (va * vb).store(result);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result[i], a[i] * b[i]);
    }
}
#endif // __AVX512DQ__

#endif // __AVX512F__

#endif // HAHAHA_ARCH_IS_X86_FAMILY
