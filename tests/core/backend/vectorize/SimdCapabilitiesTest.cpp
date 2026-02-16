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

#include <gtest/gtest.h>
#include "backend/vectorize/arch/x86/simd_capabilities.h"

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY

using namespace hahaha::backend;

// Test compile-time capabilities
TEST(SimdCapabilitiesX86, CompileTime_SSE2) {
#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
    EXPECT_TRUE(SimdCapabilitiesCompileTime::kSSE2);
#else
    EXPECT_FALSE(SimdCapabilitiesCompileTime::kSSE2);
#endif
}

TEST(SimdCapabilitiesX86, CompileTime_SSE41) {
#if defined(__SSE4_1__)
    EXPECT_TRUE(SimdCapabilitiesCompileTime::kSSE41);
#else
    EXPECT_FALSE(SimdCapabilitiesCompileTime::kSSE41);
#endif
}

TEST(SimdCapabilitiesX86, CompileTime_AVX) {
#if defined(__AVX__)
    EXPECT_TRUE(SimdCapabilitiesCompileTime::kAVX);
#else
    EXPECT_FALSE(SimdCapabilitiesCompileTime::kAVX);
#endif
}

TEST(SimdCapabilitiesX86, CompileTime_AVX2) {
#if defined(__AVX2__)
    EXPECT_TRUE(SimdCapabilitiesCompileTime::kAVX2);
#else
    EXPECT_FALSE(SimdCapabilitiesCompileTime::kAVX2);
#endif
}

TEST(SimdCapabilitiesX86, CompileTime_AVX512F) {
#if defined(__AVX512F__)
    EXPECT_TRUE(SimdCapabilitiesCompileTime::kAVX512F);
#else
    EXPECT_FALSE(SimdCapabilitiesCompileTime::kAVX512F);
#endif
}

// Test runtime detection
TEST(SimdCapabilitiesX86, RuntimeDetection) {
    auto caps = SimdCapabilitiesRuntime::detect();
    
    // x86-64 should always support SSE2
#if defined(_M_X64) || defined(__x86_64__)
    EXPECT_TRUE(caps.sse2);
#endif

    // If compile-time flags are set, runtime should also detect them
#if defined(__SSE4_1__)
    EXPECT_TRUE(caps.sse41);
#endif

#if defined(__AVX__)
    EXPECT_TRUE(caps.avx);
#endif

#if defined(__AVX2__)
    EXPECT_TRUE(caps.avx2);
#endif

#if defined(__AVX512F__)
    EXPECT_TRUE(caps.avx512f);
#endif
}

// Test that runtime detection is consistent
TEST(SimdCapabilitiesX86, RuntimeDetection_Consistency) {
    auto caps1 = SimdCapabilitiesRuntime::detect();
    auto caps2 = SimdCapabilitiesRuntime::detect();
    
    // Multiple calls should return the same results
    EXPECT_EQ(caps1.sse2, caps2.sse2);
    EXPECT_EQ(caps1.sse41, caps2.sse41);
    EXPECT_EQ(caps1.avx, caps2.avx);
    EXPECT_EQ(caps1.avx2, caps2.avx2);
    EXPECT_EQ(caps1.avx512f, caps2.avx512f);
}

// Test instruction set hierarchy
TEST(SimdCapabilitiesX86, RuntimeDetection_Hierarchy) {
    auto caps = SimdCapabilitiesRuntime::detect();
    
    // AVX2 requires AVX
    if (caps.avx2) {
        EXPECT_TRUE(caps.avx);
    }
    
    // AVX requires SSE4.1
    if (caps.avx) {
        EXPECT_TRUE(caps.sse41);
    }
    
    // SSE4.1 requires SSE2
    if (caps.sse41) {
        EXPECT_TRUE(caps.sse2);
    }
    
    // AVX-512F requires AVX2
    if (caps.avx512f) {
        EXPECT_TRUE(caps.avx2);
    }
}

// Test compile-time and runtime consistency
TEST(SimdCapabilitiesX86, CompileTimeRuntimeConsistency) {
    auto rt = SimdCapabilitiesRuntime::detect();
    
    // If compile-time says we have it, runtime must also detect it
    if (SimdCapabilitiesCompileTime::kSSE2) {
        EXPECT_TRUE(rt.sse2) << "Compile-time SSE2 enabled but runtime detection failed";
    }
    
    if (SimdCapabilitiesCompileTime::kSSE41) {
        EXPECT_TRUE(rt.sse41) << "Compile-time SSE4.1 enabled but runtime detection failed";
    }
    
    if (SimdCapabilitiesCompileTime::kAVX) {
        EXPECT_TRUE(rt.avx) << "Compile-time AVX enabled but runtime detection failed";
    }
    
    if (SimdCapabilitiesCompileTime::kAVX2) {
        EXPECT_TRUE(rt.avx2) << "Compile-time AVX2 enabled but runtime detection failed";
    }
    
    if (SimdCapabilitiesCompileTime::kAVX512F) {
        EXPECT_TRUE(rt.avx512f) << "Compile-time AVX-512F enabled but runtime detection failed";
    }
}

#endif // HAHAHA_ARCH_IS_X86_FAMILY
