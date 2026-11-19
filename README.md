# NeuralTitan

A from-scratch C++/CUDA library for building, training and running modern decoder-only LLMs (Llama-style: RMSNorm, RoPE, SwiGLU, grouped-query attention), with no third-party libraries apart from cuBLAS for matrix multiplication. It is written to be read: every operation, gradient and kernel is visible in project code and explained in a math note.

## Purpose

Frameworks hide how an LLM actually works. NeuralTitan implements every part from first principles, so you can follow a model from text to tokens, through attention and MLPs, back through the gradients, and out through sampling. The parts it covers:

- tokenizer;
- tensors and kernels;
- modules with hand-written backward passes;
- optimizer;
- training loop;
- KV-cached generation.

Correctness is demonstrated, not assumed:

- every CUDA kernel is checked against a CPU reference;
- every gradient is checked numerically;
- the library reproduces the outputs of the real SmolLM2-135M model.

Details: [requirements](docs/product/requirements.md) and the [design spec](docs/specs/2026-11-19-neuraltitan-design.md).

## Quick start

Prerequisites (see [ADR 0002](docs/adr/0002-language-and-build-toolchain.md)):

- Windows 11 with an NVIDIA GPU (developed on compute capability 8.6)
- Visual Studio with the MSVC **14.38** toolset (CUDA 12.6 does not support newer toolsets)
- CUDA Toolkit 12.6
- CMake 3.31 or newer

Build and run from the repository root in PowerShell:

```powershell
. ./scripts/dev-env.ps1; Enter-TitanDevEnv
cmake --preset release
cmake --build --preset release
./build/release/apps/titan.exe info
scripts/check.ps1
```

`Enter-TitanDevEnv` loads the MSVC 14.38 environment. `titan info` prints the version and the CUDA devices it finds. `scripts/check.ps1` is the single check command; see [CONTRIBUTING.md](CONTRIBUTING.md) for the one-time formatter setup it needs.

## Commands

| Command | What it does |
| --- | --- |
| `scripts/check.ps1 [-Preset debug\|release]` | Checks formatting, builds with warnings as errors, runs all tests including the layer-rule check, and prints `check: OK`. CI runs the same command. |
| `scripts/format.ps1 [-Fix]` | Checks C++/CUDA formatting with the pinned clang-format, or rewrites the files with `-Fix`. |
| `ctest --preset debug` | Runs the test suite of a debug build. CUDA tests skip themselves when no GPU is present. |
| `titan --version` | Prints the library version. |
| `titan info` | Prints the version and the CUDA devices found. |

## Documentation

- [Requirements](docs/product/requirements.md)
- [Architecture overview](docs/architecture/overview.md)
- [Math notes](docs/math/README.md)
- [Architecture decision records](docs/adr/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
