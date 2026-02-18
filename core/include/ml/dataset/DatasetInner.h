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

#include <cstddef>
#include <utility>

#include "DatasetTypeUnifyStrategy.h"
#include "public/Tensor.h"

namespace hahaha::ml {
class DatasetInnerLoader;

template <typename T> class DatasetInner {

  public:
    using value_type = Tensor<T>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = Tensor<T>&;
    using const_reference = const Tensor<T>&;
    using pointer = Tensor<T>*;
    using const_pointer = const Tensor<T>*;

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
            return dataset_->samples_.select(0, index_);
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

        reference operator[](difference_type n) const {
            return *(*this + n);
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
    };

    class const_iterator {
      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename DatasetInner::value_type;
        using difference_type = typename DatasetInner::difference_type;
        using pointer = typename DatasetInner::pointer;
        using reference = typename DatasetInner::reference;

        const_iterator(const DatasetInner* dataset, long index)
            : dataset_(dataset), index_(index) {
        }

        reference operator*() const {
            return dataset_->samples_.select(0, index_);
        }

        pointer operator->() const {
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

        reference operator[](difference_type n) const {
            return *(*this + n);
        }

        bool operator==(const iterator& rhs) const noexcept {
            return dataset_ == rhs.dataset_ && index_ == rhs.index_;
        }

        bool operator!=(const iterator& rhs) const noexcept {
            return !(*this == rhs);
        }

      private:
        const DatasetInner* dataset_;
        long index_;
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

    Tensor<T> getItem(size_t idx) {
        Tensor<T> item = samples_.slice(idx);
    }

    iterator begin() {
        return iterator(*this, 0);
    }

    iterator end() {
        return iterator(*this, samples_.getShape()[0]);
    }

    const_iterator begin() const {
        return const_iterator(*this, 0);
    }

    const_iterator end() const {
        return const_iterator(*this, samples_.getShape()[0]);
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }

    const_iterator cend() const noexcept {
        return end();
    }

    void shuffle() {
    }

  private:
    std::string datasetName_;
    std::vector<std::string> columns_;
    std::vector<std::string> labels_;

    DatasetTypeUnifyStrategy typeUnifyStrategy_;

    Tensor<T> samples_;

    friend class DatasetInnerLoader;
};

} // namespace hahaha::ml

#endif // DATASETINNER_E6338AA5_36E3_4139_84AC_2AC08F8DA122
