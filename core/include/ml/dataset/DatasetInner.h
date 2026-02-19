// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//  Contributors:
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//
//

#ifndef DATASETINNER_E6338AA5_36E3_4139_84AC_2AC08F8DA122
#define DATASETINNER_E6338AA5_36E3_4139_84AC_2AC08F8DA122

#include <algorithm>
#include <cstddef>
#include <optional>
#include <random>
#include <utility>

#include "DatasetTypeUnifyStrategy.h"
#include "math/TensorWrapper.h"

namespace hahaha::ml {
class DatasetInnerLoader;

template <typename T> class DatasetInner {

  public:
    using value_type = math::TensorWrapper<T>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = math::TensorWrapper<T>&;
    using const_reference = const math::TensorWrapper<T>&;
    using pointer = math::TensorWrapper<T>*;
    using const_pointer = const math::TensorWrapper<T>*;

    // =========================================================================
    // iterator
    // =========================================================================
    class iterator {
      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename DatasetInner::value_type;
        using difference_type = typename DatasetInner::difference_type;
        using pointer = typename DatasetInner::pointer;
        using reference = typename DatasetInner::reference;

        iterator(DatasetInner* dataset, long index)
            : dataset_(dataset), index_(index) {
        }

        reference operator*() {
            current_.emplace(dataset_->samples_.select(
                0, dataset_->indices_[static_cast<size_t>(index_)]));
            return *current_;
        }

        pointer operator->() {
            return &(operator*());
        }

        iterator& operator++() noexcept {
            ++index_;
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        iterator& operator--() noexcept {
            --index_;
            return *this;
        }

        iterator operator--(int) noexcept {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }

        iterator& operator+=(difference_type n) noexcept {
            index_ += n;
            return *this;
        }

        iterator operator+(difference_type n) const noexcept {
            return iterator(dataset_, index_ + n);
        }

        iterator& operator-=(difference_type n) noexcept {
            index_ -= n;
            return *this;
        }

        iterator operator-(difference_type n) const noexcept {
            return iterator(dataset_, index_ - n);
        }

        difference_type operator-(const iterator& rhs) const noexcept {
            return static_cast<difference_type>(index_ - rhs.index_);
        }

        value_type operator[](difference_type n) const {
            return dataset_->samples_.select(
                0, dataset_->indices_[static_cast<size_t>(index_ + n)]);
        }

        bool operator==(const iterator& rhs) const noexcept {
            return dataset_ == rhs.dataset_ && index_ == rhs.index_;
        }

        bool operator!=(const iterator& rhs) const noexcept {
            return !(*this == rhs);
        }

      private:
        DatasetInner* dataset_;
        long index_;
        /// nullopt until operator*() is first called; avoids copying a null TensorWrapper.
        std::optional<math::TensorWrapper<T>> current_;
    };

    // =========================================================================
    // const_iterator
    // =========================================================================
    class const_iterator {
      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename DatasetInner::value_type;
        using difference_type = typename DatasetInner::difference_type;
        using pointer = typename DatasetInner::const_pointer;
        using reference = typename DatasetInner::const_reference;

        const_iterator(const DatasetInner* dataset, long index)
            : dataset_(dataset), index_(index) {
        }

        reference operator*() const {
            current_.emplace(dataset_->samples_.select(
                0, dataset_->indices_[static_cast<size_t>(index_)]));
            return *current_;
        }

        pointer operator->() const {
            return &(operator*());
        }

        const_iterator& operator++() noexcept {
            ++index_;
            return *this;
        }

        const_iterator operator++(int) noexcept {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        const_iterator& operator--() noexcept {
            --index_;
            return *this;
        }

        const_iterator operator--(int) noexcept {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        const_iterator& operator+=(difference_type n) noexcept {
            index_ += n;
            return *this;
        }

        const_iterator operator+(difference_type n) const noexcept {
            return const_iterator(dataset_, index_ + n);
        }

        const_iterator& operator-=(difference_type n) noexcept {
            index_ -= n;
            return *this;
        }

        const_iterator operator-(difference_type n) const noexcept {
            return const_iterator(dataset_, index_ - n);
        }

        difference_type operator-(const const_iterator& rhs) const noexcept {
            return static_cast<difference_type>(index_ - rhs.index_);
        }

        value_type operator[](difference_type n) const {
            return dataset_->samples_.select(
                0, dataset_->indices_[static_cast<size_t>(index_ + n)]);
        }

        bool operator==(const const_iterator& rhs) const noexcept {
            return dataset_ == rhs.dataset_ && index_ == rhs.index_;
        }

        bool operator!=(const const_iterator& rhs) const noexcept {
            return !(*this == rhs);
        }

      private:
        const DatasetInner* dataset_;
        long index_;
        /// nullopt until operator*() is first called; avoids copying a null TensorWrapper.
        mutable std::optional<math::TensorWrapper<T>> current_;
    };

    DatasetInner() : typeUnifyStrategy_(getDefaultDatasetTypeUnifyStrategy()) {
    }

    DatasetInner(const std::vector<std::string>& features,
                 std::string datasetName,
                 const std::vector<std::string>& labels,
                 const DatasetTypeUnifyStrategy& datasetTypeUnifyStrategy =
                     getDefaultDatasetTypeUnifyStrategy())
        : datasetName_(std::move(datasetName)), columns_(features), labels_(labels),
          typeUnifyStrategy_(datasetTypeUnifyStrategy) {
    }

    [[nodiscard]] std::string getDatasetName() const {
        return datasetName_;
    }

    [[nodiscard]] std::vector<std::string> getLabels() const {
        return labels_;
    }

    [[nodiscard]] std::vector<std::string> getFeatures() const {
        return columns_;
    }

    [[nodiscard]] DatasetTypeUnifyStrategy getDatasetTypeUnifyStrategy() const {
        return typeUnifyStrategy_;
    }


    /**
     * @brief Return the sample at logical index @p idx as a 1-D Tensor view.
     *
     * The logical index is remapped through the current permutation, so the
     * result reflects any prior call to shuffle().
     */
    math::TensorWrapper<T> getItem(size_t idx) {
        return samples_.select(0, indices_[idx]);
    }

    iterator begin() {
        return iterator(this, 0);
    }

    iterator end() {
        if (samples_.getDimensions() == 0) {
            return iterator(this, 0);
        }
        return iterator(this, static_cast<long>(samples_.getShape()[0]));
    }

    const_iterator begin() const {
        return const_iterator(this, 0);
    }

    const_iterator end() const {
        if (samples_.getDimensions() == 0) {
            return const_iterator(this, 0);
        }
        return const_iterator(this, static_cast<long>(samples_.getShape()[0]));
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }

    const_iterator cend() const noexcept {
        return end();
    }

    /**
     * @brief Return the number of samples in the dataset.
     */
    [[nodiscard]] size_type size() const {
        if (samples_.getDimensions() == 0) {
            return 0;
        }
        return samples_.getShape()[0];
    }

    /**
     * @brief Randomly permute the iteration order using a non-deterministic seed.
     *
     * Only the index mapping is shuffled — the underlying data tensor is never
     * moved or copied (O(N) time, zero extra memory for data).
     */
    void shuffle() {
        std::mt19937 rng{std::random_device{}()};
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }

    /**
     * @brief Shuffle with an explicit @p seed for reproducible runs.
     */
    void shuffle(unsigned seed) {
        std::mt19937 rng{seed};
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }

  private:
    std::string datasetName_;
    std::vector<std::string> columns_;
    std::vector<std::string> labels_;

    DatasetTypeUnifyStrategy typeUnifyStrategy_;

    math::TensorWrapper<T> samples_;
    /// Logical-to-physical row mapping. Shuffled by shuffle(); reset by the loader.
    std::vector<size_t> indices_;

    friend class DatasetInnerLoader;
};

} // namespace hahaha::ml

#endif // DATASETINNER_E6338AA5_36E3_4139_84AC_2AC08F8DA122
