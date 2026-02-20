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
#include <bit>
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
        if (size_ > 0) {
            words_ = std::make_unique<wordType[]>((size_ + wordSize - 1) / wordSize);
            clear();
        }
    }

    /**
     * @brief Construct a new Bitmap object with initial value.
     *
     * @param size The total number of bits to manage.
     * @param initialValue The initial value for all bits.
     */
    Bitmap(const sizeT size, const bool initialValue) : size_(size) {
        if (size_ > 0) {
            words_ = std::make_unique<wordType[]>((size_ + wordSize - 1) / wordSize);
            if (initialValue) {
                setAll();
            } else {
                clear();
            }
        }
    }

    /**
     * @brief Copy constructor.
     */
    Bitmap(const Bitmap& other) : size_(other.size_) {
        if (size_ > 0) {
            const sizeT numWords = (size_ + wordSize - 1) / wordSize;
            words_ = std::make_unique<wordType[]>(numWords);
            std::copy_n(other.words_.get(), numWords, words_.get());
        }
    }

    /**
     * @brief Move constructor.
     */
    Bitmap(Bitmap&& other) noexcept
        : size_(other.size_), words_(std::move(other.words_)) {
        other.size_ = 0;
    }

    /**
     * @brief Copy assignment operator.
     */
    Bitmap& operator=(const Bitmap& other) {
        if (this != &other) {
            size_ = other.size_;
            if (size_ > 0) {
                const sizeT numWords = (size_ + wordSize - 1) / wordSize;
                words_ = std::make_unique<wordType[]>(numWords);
                std::copy_n(other.words_.get(), numWords, words_.get());
            } else {
                words_.reset();
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment operator.
     */
    Bitmap& operator=(Bitmap&& other) noexcept {
        if (this != &other) {
            size_ = other.size_;
            words_ = std::move(other.words_);
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get the size of the bitmap.
     *
     * @return sizeT The total number of bits managed.
     */
    [[nodiscard]] sizeT size() const {
        return size_;
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
        words_[wordIndex] |= static_cast<wordType>(1) << (index % wordSize);
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
        words_[wordIndex] &= ~(static_cast<wordType>(1) << (index % wordSize));
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
     * @brief Set all bits to 1.
     */
    void setAll() {
        if (size_ == 0)
            return;

        const sizeT numWords = (size_ + wordSize - 1) / wordSize;
        const sizeT lastWordBits = size_ % wordSize;

        // Set all complete words to all 1s
        std::fill_n(words_.get(),
                    numWords - (lastWordBits ? 1 : 0),
                    ~static_cast<wordType>(0));

        // Handle the last partial word if needed
        if (lastWordBits) {
            const wordType mask = (static_cast<wordType>(1) << lastWordBits) - 1;
            words_[numWords - 1] = mask;
        }
    }

    /**
     * @brief Clear all bits (set to 0).
     */
    void clear() const {
        if (size_ == 0)
            return;

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
        return (words_[wordIndex] & static_cast<wordType>(1) << (index % wordSize))
            != 0;
    }

    /**
     * @brief Count the number of set bits (population count).
     *
     * @return sizeT The number of bits set to 1.
     */
    [[nodiscard]] sizeT count() const {
        if (size_ == 0)
            return 0;

        sizeT result = 0;
        const sizeT numWords = (size_ + wordSize - 1) / wordSize;
        const sizeT lastWordBits = size_ % wordSize;

        // Count bits in all complete words (population count per word)
        for (sizeT i = 0; i < (lastWordBits ? numWords - 1 : numWords); ++i) {
            result += static_cast<sizeT>(std::popcount(words_[i]));
        }

        // Handle the last partial word if needed
        if (lastWordBits) {
            const wordType lastWord = words_[numWords - 1];
            const wordType mask = (static_cast<wordType>(1) << lastWordBits) - 1;
            result += static_cast<sizeT>(std::popcount(lastWord & mask));
        }

        return result;
    }

    /**
     * @brief Check if all bits are set to 1.
     *
     * @return true If all bits are 1, false otherwise.
     */
    [[nodiscard]] bool all() const {
        if (size_ == 0)
            return true; // Empty bitmap has all bits set (vacuous truth)

        const sizeT numWords = (size_ + wordSize - 1) / wordSize;
        const sizeT lastWordBits = size_ % wordSize;

        // Check all complete words
        for (sizeT i = 0; i < (lastWordBits ? numWords - 1 : numWords); ++i) {
            if (words_[i] != ~static_cast<wordType>(0)) {
                return false;
            }
        }

        // Check the last partial word if needed
        if (lastWordBits) {
            if (const wordType mask = (static_cast<wordType>(1) << lastWordBits) - 1;
                (words_[numWords - 1] & mask) != mask) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Check if no bits are set to 1.
     *
     * @return true If no bits are 1, false otherwise.
     */
    [[nodiscard]] bool none() const {
        if (size_ == 0)
            return true; // Empty bitmap has no bits set

        const sizeT numWords = (size_ + wordSize - 1) / wordSize;
        const sizeT lastWordBits = size_ % wordSize;

        // Check all complete words
        for (sizeT i = 0; i < (lastWordBits ? numWords - 1 : numWords); ++i) {
            if (words_[i] != 0) {
                return false;
            }
        }

        // Check the last partial word if needed
        if (lastWordBits) {
            if (const wordType mask = (static_cast<wordType>(1) << lastWordBits) - 1;
                (words_[numWords - 1] & mask) != 0) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Check if any bit is set to 1.
     *
     * @return true If at least one bit is 1, false otherwise.
     */
    [[nodiscard]] bool any() const {
        if (size_ == 0)
            return false; // Empty bitmap has no bits set

        return !none();
    }

    /**
     * @brief Flip the bit at the specified index.
     *
     * @param index The index of the bit to flip.
     * @throws std::out_of_range If index is >= size.
     */
    void flip(const sizeT index) {
        if (index >= size_) {
            throw std::out_of_range("index out of range");
        }
        const sizeT wordIndex = index / wordSize;
        words_[wordIndex] ^= static_cast<wordType>(1) << (index % wordSize);
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
        if (oldNumWords > 0) {
            std::copy_n(words_.get(), oldNumWords, newWords.get());
        }
        size_ = newSize;
        words_ = std::move(newWords);
    }

    /**
     * @brief Resize the bitmap to the specified size.
     *
     * @param newSize The new size for the bitmap.
     * @param defaultValue The value to initialize new bits with if expanding.
     */
    void resize(const sizeT newSize, const bool defaultValue = false) {
        if (newSize == size_) {
            return;
        }

        if (newSize == 0) {
            size_ = 0;
            words_.reset();
            return;
        }

        const sizeT oldSize = size_;
        const sizeT oldNumWords = (size_ + wordSize - 1) / wordSize;
        const sizeT newNumWords = (newSize + wordSize - 1) / wordSize;

        auto newWords = std::make_unique<wordType[]>(newNumWords);

        if (oldSize > 0) {
            const sizeT minWords = std::min(oldNumWords, newNumWords);
            std::copy_n(words_.get(), minWords, newWords.get());
        }

        if (newSize > oldSize && defaultValue) {
            // Initialize new bits to true
            const sizeT lastOldBit = oldSize % wordSize;
            const sizeT lastNewBit = newSize % wordSize;

            if (lastOldBit > 0) {
                // Update the last word that had old bits: set new bits in that word
                // to 1
                const sizeT lastOldWordIdx = oldSize / wordSize;
                const wordType oldMask =
                    (static_cast<wordType>(1) << lastOldBit) - 1;
                const sizeT validBitsInWord =
                    std::min(wordSize, newSize - lastOldWordIdx * wordSize);
                const wordType newMask = (validBitsInWord == wordSize)
                    ? ~static_cast<wordType>(0)
                    : (static_cast<wordType>(1) << validBitsInWord) - 1;
                newWords[lastOldWordIdx] =
                    (newWords[lastOldWordIdx] & oldMask) | (newMask & ~oldMask);
            }

            // Set remaining words to all 1s if expanding
            if (newSize > oldSize + (wordSize - lastOldBit) % wordSize) {
                const sizeT startWord = (oldSize + wordSize - 1) / wordSize;
                const sizeT endWord =
                    (newSize + wordSize - 1) / wordSize - (lastNewBit ? 1 : 0);
                for (sizeT i = startWord; i < endWord; ++i) {
                    newWords[i] = ~static_cast<wordType>(0);
                }
                if (lastNewBit) {
                    const wordType mask =
                        (static_cast<wordType>(1) << lastNewBit) - 1;
                    newWords[newNumWords - 1] = mask;
                }
            }
        } else if (newSize > oldSize) {
            // Initialize new bits to false (already done by default
            // initialization)
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
