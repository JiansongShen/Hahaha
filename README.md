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
- **Tooling**: Meson + Ninja build, GoogleTest, formatting script `python3 dev/format.py`.

## Build and run

### Prerequisites

- C++ compiler with C++23 support (GCC/Clang)
- Meson and Ninja
- GoogleTest (Meson can find it as a system dependency)
- Optional for visualization demos: GLFW + OpenGL
- Optional for container workflow: Docker

### Build (native)

```bash
meson setup builddir --buildtype=debug
meson compile -C builddir
meson test -C builddir -v
```

### Run examples

After building, you can run:

```bash
# Basic tensor usage
./builddir/examples/basic_usage/hahaha_example_tensor_basic_usage

# Autograd demo
./builddir/examples/autograd/hahaha_example_autograd

# ML training demo (CLI)
./builddir/examples/ml_basic_usage/hahaha_example_ml_basic_usage

# Visualization demo (requires GLFW/OpenGL but not very nice)
./builddir/examples/ml_visualizer/hahaha_example_ml_visualizer
```

## Minimal usage example

```cpp
#include "Tensor.h"

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

Please ensure your code follows the project's coding standards by running `python3 dev/format.py` before submitting.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
