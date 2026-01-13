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

#ifndef HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE
#define HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE
#include <vector>

namespace hahaha::backend {

class CudaMemoryPool {
  public:
    void* allocate(size_t size);
    void free(void* ptr);

  private:
    struct CudaMemoryBlock {
        size_t size;
        CudaMemoryBlock* next;
    };

    // records from size 32KB to 1TB
    std::vector<CudaMemoryBlock> freeBlocks_;
    std::vector<CudaMemoryBlock> allocatedBlocks_;
};
} // namespace hahaha::backend

#endif // HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE
