# Hahaha [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Hahaha is an **education-first** numerical computing and machine learning library written in **C++23**.
It aims to provide a **clear, inspectable implementation** of tensors, automatic differentiation, and training
loops, together with **visualization tooling** so learners can *see* what happens during training.

## What this project is for

- **Learning**: read the code and understand how tensors, broadcasting, and autograd work end-to-end.
- **Teaching / Demos**: run examples that visualize the training process (loss curves, parameter updates, etc.).
- **Small experiments**: simple models and datasets (e.g. MNIST) without depending on a large framework.

## Key features

- **Tensor core**: `Tensor` / `TensorWrapper` with shape (`TensorShape`), stride (`TensorStride`), broadcasting,
  reshape, transpose, and basic math.
- **Autograd + compute graph**: define-by-run graph building via `compute/graph/*` and backward propagation
  with unit tests.
- **Broadcasting as a first-class op**: explicit `Broadcast` node with correct gradient reduction.
- **Visualization + educational UX**:
  - ImGui-based visualizer and demos under `examples/` to show training dynamics.
  - Code is intentionally written to be readable and traceable, with tests as executable documentation.
- **Tooling**: CMake + vcpkg build, GoogleTest, formatting script `python3 dev/format.py`.

## Build and run

### 1. One-Click Environment Setup (Recommended)

We provide a Python script to automatically set up the complete development environment.

```bash
python3 dev/setup_dev_env.py
```

**What this script does:**
- 📦 **Installs System Dependencies**: Automatically detects your package manager (`apt` for Debian/Ubuntu, `pacman` for Arch Linux) and installs compilers, build tools, and graphics libraries.
- 🔧 **Sets up Build Tools**: Installs or configures **CMake**, **Ninja**, and **Python** tools.
- ⚓ **Configures Git Hooks**: Sets up **pre-commit** to ensure code quality before you commit.
- 📥 **Manages Dependencies**: Bootstraps **vcpkg** and installs dependencies like **GoogleTest**, **GLFW**, and **ImGui**.
- ✅ **Verifies Installation**: Checks that all required tools are present and correctly configured.

### 2. Manual Prerequisites (If not using the script)

If you prefer to set up manually or use a different OS:

- **Compiler**: C++23 compliant (GCC 13+ or Clang 16+)
- **Build System**: [CMake](https://cmake.org/) 3.20+ and [Ninja](https://ninja-build.org/)
- **Package Manager**: [vcpkg](https://vcpkg.io/) (repo-local at `vcpkg/vcpkg_root/`)
- **Libraries** (installed via vcpkg):
  - `gtest`, `glfw3`, `imgui` (for visualization)
- **Tools**: `git`, `python3`, `pre-commit`

### 3. Build (native)

```bash
# Setup vcpkg (first time only)
# If vcpkg/vcpkg_root doesn't exist, clone it:
if [ ! -d "vcpkg/vcpkg_root" ]; then
  git clone https://github.com/microsoft/vcpkg.git vcpkg/vcpkg_root
fi

# Bootstrap vcpkg and install dependencies
./vcpkg/vcpkg_root/bootstrap-vcpkg.sh  # or bootstrap-vcpkg.bat on Windows
./vcpkg/vcpkg_root/vcpkg install gtest glfw3 imgui

# Configure and build
cmake -S . -B builddir -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build builddir --parallel
ctest --test-dir builddir --output-on-failure
```

### 4. Run examples

After building, you can run:

```bash
# Basic tensor usage
./builddir/examples/hahaha_example_tensor_basic_usage

# Autograd demo
./builddir/examples/hahaha_example_autograd

# ML training demo (CLI)
./builddir/examples/hahaha_example_ml_basic_usage

# Visualization demo (requires GLFW/OpenGL but not very nice)
./builddir/examples/hahaha_example_ml_visualizer
```

## Minimal usage example

```cpp
#include "public/Tensor.h"

using hahaha::Tensor;
using hahaha::math::NestedData;

int main() {
    Tensor<float> a(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> b(NestedData<float>{{10.0f, 20.0f}, {30.0f, 40.0f}});
    a.setRequiresGrad(true);
    b.setRequiresGrad(true);

    auto c = a * b + 2.0f;
    c.backward();

    // Inspect gradients
    // a.grad(), b.grad()
    return 0;
}
```

## Where to look (recommended reading order)

- **Tensor & shape/stride**: `core/include/math/TensorWrapper.h`, `core/include/math/ds/*`
- **Compute graph & autograd**: `core/include/compute/graph/*`, especially `ComputeNode` and `compute_funs/*`
- **Visualization**: `core/src/display/*` and `examples/ml_visualizer/*`
- **Tests as documentation**: `tests/core/*` (broadcast + autograd tests show expected behavior)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

refer to [how-to-contribute](https://github.com/JiansongShen/HahahaDevDocument/blob/main/src/en/developers/how-to-contribute.md)

Please ensure your code follows the project's coding standards by running `python3 dev/format.py` before submitting.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
