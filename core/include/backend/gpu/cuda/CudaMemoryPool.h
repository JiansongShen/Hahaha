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

#include <expected>
#include <vector>

#include "backend/DeviceBuffer.h"
#include "common/errors/Error.h"
#include "utils/data_structure/Bitmap.h"

namespace hahaha::backend {

class CudaMemoryPool {

    static constexpr size_t BaseMemoryBlockSize = 32 << 10;

  public:
    [[nodiscard]] common::Error checkFreeBlockListExist(size_t size) const;

    std::expected<void*, common::Error> allocateSmall(size_t size);

    void free(void* ptr);

    std::expected<void*, common::Error> allocateBig(size_t size);

  private:
    struct CudaMemoryBlockHeader {
        CudaMemoryBlockHeader* prev;
        CudaMemoryBlockHeader* next;
        size_t size;
    };

    struct CudaMemoryBlock {
        CudaMemoryBlockHeader header;

        CudaMemoryBlock* next() {
            return reinterpret_cast<CudaMemoryBlock*>(&header.next);
        }
        CudaMemoryBlock* prev() {
            return reinterpret_cast<CudaMemoryBlock*>(&header.prev);
        }

        void setNext(CudaMemoryBlock* block) {
            header.next = reinterpret_cast<CudaMemoryBlockHeader*>(block);
        }
        void setPrev(CudaMemoryBlock* block) {
            header.prev = reinterpret_cast<CudaMemoryBlockHeader*>(block);
        }

        void setSize(const size_t size) {
            header.size = size;
        };
        [[nodiscard]] size_t getSize() const {
            return header.size;
        }
    };

    struct BigCudaMemoryBlock {
        CudaMemoryBlockHeader header;
        size_t cacheLiveTimes;
        CudaMemoryBlock* next() {
            return reinterpret_cast<CudaMemoryBlock*>(&header.next);
        }
        CudaMemoryBlock* prev() {
            return reinterpret_cast<CudaMemoryBlock*>(&header.prev);
        }

        void setNext(CudaMemoryBlock* block) {
            header.next = reinterpret_cast<CudaMemoryBlockHeader*>(block);
        }
        void setPrev(CudaMemoryBlock* block) {
            header.prev = reinterpret_cast<CudaMemoryBlockHeader*>(block);
        }

        void setSize(const size_t size) {
            header.size = size;
        }

        void refreshCacheLiveTime() {
            cacheLiveTimes = 0;
        }

        [[nodiscard]] size_t getSize() const {
            return header.size;
        }
    };
    static size_t getBlockIndexOfSize(size_t size);

    common::Error requireSplitBlock(size_t blockIdx);

    std::expected<void*, common::Error> allocateOnBlock(size_t blockIdx);

    static size_t getMemoryNeeded(size_t size);

    void insertIntoFreeBlock(CudaMemoryBlock* block);

    void insertIntoBigBlock(BigCudaMemoryBlock* block);

    std::expected<BigCudaMemoryBlock*, common::Error>
    requireNewBigBlock(size_t size);

    BigCudaMemoryBlock* findCachedBigBlock(size_t size);

    // records from size 32KB to 1TB
    std::vector<CudaMemoryBlock*> freeBlocks_;
    std::vector<utils::Bitmap> blocksBitmap_;

    std::vector<BigCudaMemoryBlock*> bigBlocks_;
    std::vector<BigCudaMemoryBlock*> allocatedBigBlocks_;
};

} // namespace hahaha::backend

#endif // HAHAHA_CUDAMEMORYPOOL_H_1C230E81AAF44C518925E9CB91324ECE
