// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include "basic_adam_optimizer_example.h"
#include "basic_sgd_optimizer_example.h"
int main() {
    std::cout << "========================SGD Optimizer======================="
              << std::endl;
    basic_linear_regression_example();
    std::cout << "========================Adam Optimizer======================="
              << std::endl;
    basic_adam_optimizer_example();
}