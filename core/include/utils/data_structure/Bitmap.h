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

#ifndef HAHAHA_BITMAP_H_02E01C69C9A3452CA7AA2F7B776598B6
#define HAHAHA_BITMAP_H_02E01C69C9A3452CA7AA2F7B776598B6
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace hahaha::utils {
/**
 * @brief A simple bitmap implementation for managing bits.
 * @details Uses std::uint64_t as the underlying storage word.
 */
class Bitmap {
    using wordType = std::uint64_t;
    using sizeT = std::size_t;

    static constexpr sizeT wordSize = std::numeric_limits<wordType>::digits;

  public:
    /**
     * @brief Construct a new Bitmap object.
     *
     * @param size The total number of bits to manage.
     */
    explicit Bitmap(const sizeT size) : size_(size) {
        words_ = std::make_unique<wordType[]>((size + wordSize - 1) / wordSize);
    }

    /**
     * @brief Set the bit at the specified index to 1 (on).
     *
     * @param index The index of the bit to set.
     * @throws std::out_of_range If index is >= size.
     */
    void setOn(const sizeT index) {
        if (index >= size_) {
            throw std::out_of_range("index out of range");
        }
        const sizeT wordIndex = index / wordSize;
        words_[wordIndex] |= wordType(1) << (index % wordSize);
    }

    /**
     * @brief Set the bit at the specified index to 0 (off).
     *
     * @param index The index of the bit to set.
     * @throws std::out_of_range If index is >= size.
     */
    void setOff(const sizeT index) {
        if (index >= size_) {
            throw std::out_of_range("index out of range");
        }
        const sizeT wordIndex = index / wordSize;
        words_[wordIndex] &= ~(wordType(1) << (index % wordSize));
    }

    /**
     * @brief Set the bit at the specified index to the given value.
     *
     * @param index The index of the bit to set.
     * @param value The value to set (true for 1, false for 0).
     * @throws std::out_of_range If index is >= size.
     */
    void set(const sizeT index, const bool value) {
        if (value) {
            setOn(index);
        } else {
            setOff(index);
        }
    }

    /**
     * @brief Clear all bits (set to 0).
     */
    void clear() const {
        const sizeT numWords = (size_ + wordSize - 1) / wordSize;
        std::fill_n(words_.get(), numWords, 0);
    }

    /**
     * @brief Get the value of the bit at the specified index.
     *
     * @param index The index of the bit to get.
     * @return true If the bit is 1.
     * @return false If the bit is 0.
     * @throws std::out_of_range If index is >= size.
     */
    [[nodiscard]] bool get(const sizeT index) const {
        if (index >= size_) {
            throw std::out_of_range("index out of range");
        }
        const sizeT wordIndex = index / wordSize;
        return (words_[wordIndex] & (wordType(1) << (index % wordSize))) != 0;
    }

    /**
     * @brief Expand the bitmap by a specified number of bits.
     *
     * @param expandSize The number of bits to add.
     */
    void expand(const sizeT expandSize) {
        if (expandSize == 0) {
            return;
        }
        const sizeT newSize = size_ + expandSize;
        const sizeT oldNumWords = (size_ + wordSize - 1) / wordSize;
        const sizeT newNumWords = (newSize + wordSize - 1) / wordSize;

        auto newWords = std::make_unique<wordType[]>(newNumWords);
        for (sizeT i = 0; i < oldNumWords; ++i) {
            newWords[i] = words_[i];
        }
        size_ = newSize;
        words_ = std::move(newWords);
    }

  private:
    sizeT size_;
    std::unique_ptr<wordType[]> words_;
};
} // namespace hahaha::utils

#endif // HAHAHA_BITMAP_H_02E01C69C9A3452CA7AA2F7B776598B6
