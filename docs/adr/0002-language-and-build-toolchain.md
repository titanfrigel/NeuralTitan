---
status: accepted
date: 2026-11-19
packages: []
---

# Language and build toolchain

## Context and problem

The library is written in C++ with CUDA kernels. It is built on Windows 11 for an RTX 3070 Laptop GPU (compute capability 8.6). On Windows, the CUDA compiler `nvcc` requires MSVC as its host compiler. Every header shared between `.cpp` and `.cu` files is compiled at the language level `nvcc` supports. Visual Studio 2026 ships two MSVC toolsets: 14.50 (the default) and 14.38.

## Decision drivers

- Officially supported combination of CUDA, host compiler and C++ dialect.
- One language level for all code, since headers are shared between host and device.
- Portable build description so a Linux build stays possible later.
- Already installed on the author's machine.

## Considered options

| Option | Verdict |
|---|---|
| C++20, CUDA 12.6, MSVC 14.38, CMake | Chosen: CUDA 12.6 officially supports MSVC 193x (VS 2022 17.x) with C++20; 14.38 is MSVC 1938. |
| C++20 with MSVC 14.50 (VS 2026 default toolset) | Rejected: not in CUDA 12.6's supported compiler table. |
| C++23 or C++26 | Rejected. `nvcc` 12.6 tops out at C++20 for device code, and MSVC 14.38 has little C++26 support. The features gained (reflection, `std::linalg`, `std::simd`) are either minor here or would replace code the project exists to write. |
| Upgrade CUDA to a newer release | Deferred: no requirement needs it now; it can be revisited through a superseding ADR. |

## Decision outcome

Chosen:
- C++20 for host and device code.
- CUDA Toolkit 12.6 (V12.6.85), targeting `sm_86`.
- MSVC host toolset 14.38.
- CMake (3.31.1 installed) as the build system.

CMake pins the MSVC toolset to 14.38 and the CUDA architecture to 86.

## Consequences

- C++20 features are available (concepts, `std::span`, designated initializers, `std::format`); nothing newer.
- The build must select toolset 14.38 explicitly; an environment that picks 14.50 fails at CUDA compilation.
- GitHub-hosted CI runners have no GPU: CI compiles the CUDA code but runs only CPU tests.
- The author has approved toolchain updates; a CUDA or MSVC upgrade is recorded by superseding this ADR.

## Links

- [CUDA 12.6 Installation Guide for Microsoft Windows, Table 1: Windows Compiler Support](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-installation-guide-microsoft-windows/index.html) (fetched 2026-11-19)
- [Design spec](../specs/2026-11-19-neuraltitan-design.md)
