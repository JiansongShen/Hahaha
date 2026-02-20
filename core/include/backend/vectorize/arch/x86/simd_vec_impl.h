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
//  x86 SIMD implementation: SSE2 (128-bit), AVX (256-bit), optional AVX-512
//  (512-bit). Implementations are split by width to keep each file manageable.
//

#ifndef HAHAHA_SIMD_VEC_IMPL_H_92AB0F5E920442FC95C08FE4487A6DE8
#define HAHAHA_SIMD_VEC_IMPL_H_92AB0F5E920442FC95C08FE4487A6DE8

#if defined(HAHAHA_ARCH_IS_X86_FAMILY) && HAHAHA_ARCH_IS_X86_FAMILY

#include "backend/vectorize/arch/x86/simd_vec_impl_128.h"
#include "backend/vectorize/arch/x86/simd_vec_impl_256.h"
#include "backend/vectorize/arch/x86/simd_vec_impl_512.h"

#endif // HAHAHA_ARCH_IS_X86_FAMILY

#endif // HAHAHA_SIMD_VEC_IMPL_H_92AB0F5E920442FC95C08FE4487A6DE8
