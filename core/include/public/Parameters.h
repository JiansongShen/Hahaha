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
// Contributors:
// jiansongshen (jason.shen111@outlook.com)
//

#ifndef HAHAHA_PUBLIC_PARAMETERS_H_1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D
#define HAHAHA_PUBLIC_PARAMETERS_H_1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D

#include <memory>
#include <vector>

// Public Tensor type — the only type users touch.
#include "Tensor.h"

// Inner implementation.
#include "ml/Parameters.h"

namespace hahaha {

/**
 * @brief Public handle for a collection of trainable parameters (pimpl).
 *
 * Delegates storage to an inner @ref ml::Parameters<T> held behind a
 * @c shared_ptr.  Parameters are @ref Tensor<T> objects so gradient
 * tracking and autograd work transparently.
 *
 * @tparam T Numeric type.
 */
template <typename T>
class Parameters {
  public:
    /** @brief Construct an empty parameter collection. */
    Parameters() : impl_(std::make_shared<ml::Parameters<T>>()) {}

    /**
     * @brief Add a parameter to the collection.
     * @param param The tensor to track (should have @c requiresGrad = true).
     */
    void addParameter(Tensor<T> param) {
        impl_->addParameter(std::move(param));
    }

    /**
     * @brief Mutable access to the tracked parameters.
     * @return Reference to the internal parameter vector.
     */
    std::vector<Tensor<T>>& getParameters() {
        return impl_->getParameters();
    }

    /**
     * @brief Read-only access to the tracked parameters.
     * @return Const reference to the internal parameter vector.
     */
    [[nodiscard]] const std::vector<Tensor<T>>& getParameters() const {
        return impl_->getParameters();
    }

  private:
    std::shared_ptr<ml::Parameters<T>> impl_;
};

} // namespace hahaha

#endif // HAHAHA_PUBLIC_PARAMETERS_H_1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D
