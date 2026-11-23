# NeuralTitan

A from-scratch C++/CUDA library for building, training and running modern decoder-only LLMs, with no third-party libraries apart from cuBLAS for matrix multiplication. This file is the single source of project memory for Claude Code and OpenCode.

## Project setup

<!-- senior-standards:setup v1 -->
- Goal: From-scratch C++/CUDA LLM library (no third-party libraries) for deep learning-by-review of modern decoder-only transformers; success is training small models to coherent text and reproducing SmolLM2-135M's outputs exactly.
- Docs: full
- Tests: standard
- Public repository: yes
- Maintained over time: no
- Versioned releases: no
- Overrides: none
- Recorded: 2026-11-19

## Stack

- C++20 (host and device), CUDA Toolkit 12.6 targeting `sm_86`, MSVC host toolset **14.38** (not the VS 2026 default 14.50, which CUDA 12.6 does not support), CMake. See [ADR 0002](docs/adr/0002-language-and-build-toolchain.md).
- Library dependencies: the C++ standard library, the CUDA runtime and cuBLAS. cuBLAS is used for matrix multiplication only, is the default GEMM, and is linked only by `ops/cuda`. The project's own naive and tiled GEMM kernels are kept for learning and parity, selectable with `--gemm cublas|naive|tiled`. See [ADR 0007](docs/adr/0007-cublas-default-gemm.md). The test harness, JSON parser, safetensors reader and every other kernel are written in the project.
- Repository tooling (clang-format, gitleaks, markdownlint, lychee, GitHub Actions) is development-only; see [ADR 0008](docs/adr/0008-repository-tooling-baseline.md).
- Development-only tooling: Python with `torch`, `transformers` and `safetensors` in an isolated venv under `tools/`, used only to generate golden fixtures. Never part of the build. See [ADR 0004](docs/adr/0004-golden-reference-tooling.md).
- Architecture: layered, with swappable device backends in `ops`. See [ADR 0005](docs/adr/0005-layered-architecture.md).
- No data store, no hosting, no network access at runtime.

## Commands

Run from the repository root in PowerShell:

- `. ./scripts/dev-env.ps1; Enter-TitanDevEnv`: load the MSVC 14.38 environment (once per shell).
- `cmake --preset release` then `cmake --build --preset release`: configure and build (`debug` preset also exists). The program is `build/<preset>/apps/titan.exe`; its commands are `titan --version`, `titan info`, `titan tokenizer train`, `titan train` and `titan bench matmul` (below).
- `build/<preset>/apps/titan.exe tokenizer train --type char --corpus <file> --out <tok.json>`: build and save a character tokenizer.
- `build/<preset>/apps/titan.exe train --config <cfg.json> --data <corpus.txt> --out <dir> [--tokenizer <tok.json>] [--steps N] [--device cpu|cuda] [--gemm cublas|naive|tiled]`: train a model (on the GPU when one is available, otherwise on the CPU; `--gemm` picks the CUDA GEMM and needs `--device cuda` unless it is `cublas`) and write the tokenizer, token files and a sample to `<dir>`. `configs/tiny-shakespeare.json` is the M1 and M2 config.
- `build/<preset>/apps/titan.exe bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]`: time square GEMMs with each CUDA GEMM backend against the GPU's peak FP32 throughput and bandwidth (needs a GPU).
- `ctest --preset release`: run the tests of the release build (`ctest --preset debug` after building the debug preset).
- `scripts/check.ps1 [-Preset debug|release]`: the single check: format check, build with warnings as errors, all tests including the layer check. Prints `check: OK`. CI runs it.
- `scripts/format.ps1 [-Fix]`: check or fix formatting with clang-format 20.1.8. One-time setup: `python -m venv tools/.venv`, then `tools/.venv/Scripts/python -m pip install -r tools/requirements-dev.txt`.
- `git config core.hooksPath .githooks`: enable the gitleaks pre-commit hook (once per clone).
- `node <senior-standards plugin>/scripts/adr-index.mjs .` regenerates `docs/adr/README.md` after adding or superseding an ADR.

## Layout

Current:

- `src/core/`: the `core` layer (errors, device, storage, caching allocator, tensor, rng, parallel_for, `utf8.*` for UTF-8 encoding and decoding).
- `src/io/`: the `io` layer (JSON parser and writer, file helpers, token files).
- `src/tokenizer/`: the `tokenizer` layer (character-level tokenizer).
- `src/ops/`: the `ops` layer, with the CPU backend `ops/cpu` and the CUDA backend `ops/cuda` (`common.cuh` launch helpers and shared block reduction, `cublas_handle.cu`, and one `.cu` per op family: elementwise, reduce, matmul, softmax, norm, rope, layout, embedding, cross-entropy, adamw).
- `src/nn/`: the `nn` layer (modules with hand-written backward passes, and the KV cache for inference).
- `src/optim/`: the `optim` layer (AdamW, gradient clipping, learning-rate schedule).
- `src/model/`: the `model` layer (model config and the decoder-only model).
- `src/runtime/`: the `runtime` layer (data loader, trainer, checkpoints, sampler, KV-cached generator).
- `apps/titan/`: the `titan` executable (the `cli` layer): `main.cpp` and `commands.*`.
- `configs/`: model and training JSON configs (`tiny-shakespeare.json`).
- `tests/`: the test runner and tests. `tests/harness/` is the project's own test harness; `tests/layer_check/` holds fixtures for the layer include check; `tests/fixtures/cli/` holds hand-written CLI inputs (not generated golden fixtures); `tests/support/` holds shared helpers (`gradcheck.*`, and `parity.*` for CPU-vs-CUDA comparisons); `tests/ops/cuda/` holds the CPU-vs-CUDA parity tests.
- `scripts/`: `check.ps1`, `format.ps1`, `dev-env.ps1` and the layer include check.
- `tools/`: development-only Python requirements (the venv is git-ignored).
- `docs/specs/`: approved design specs.
- `docs/plans/`: implementation plans.
- `docs/product/`: requirements.
- `docs/architecture/`: architecture overview.
- `docs/math/`: math notes, one per module or op family.
- `docs/adr/`: decision records. `README.md` there is generated; do not edit it by hand.
- `.github/`: CI workflows (ci, pr-title, gitleaks, dependency-check, docs, scorecard), the pull request template and vendored checks.
- `.githooks/`: the gitleaks pre-commit hook.

Planned (created by the milestones; layer names follow the architecture overview):

- `tools/`: golden-fixture scripts, added with the first fixtures.
- `tests/fixtures/`: golden fixtures generated by `tools/`; regenerate them rather than editing them.
- `models/`, `data/`, `runs/`: downloaded models, corpora, and training outputs. All git-ignored; never commit them.

## Conventions

- Conventional Commits.
- Layer rules in `docs/architecture/overview.md` are enforced in CI; do not bypass them.
- Update docs in the same change as the code. ADRs are append-only.
- Add a dependency, service, database or hosting provider only through the decision gate and an accepted ADR.

- Errors: throw exceptions derived from `titan::Error` with context (file, tensor or parameter name, shapes).
  - Use `TITAN_CHECK(cond, msg)` for preconditions and `TITAN_CUDA_CHECK` around every CUDA call and kernel launch.
  - Validate external input only at the `io`, `tokenizer` and `cli` boundaries and in the config parsers built on `io::JsonFields` (`parse_model_config` in `model`, `parse_train_config` in `runtime`), bounds-checking every length and offset. Kernels do not validate.
- Every op has a CPU reference and a CUDA implementation behind one dispatching function in `ops`. The two device implementations never include each other.
- Every `nn` module implements `forward`, a hand-written `backward` that accumulates parameter gradients with `+=`, and `parameters`.
- Tests are written in the same change as the feature. Each one needs:
  - a CPU-vs-CUDA parity test for every op;
  - a finite-difference gradient check for every module;
  - malformed-input tests for every parser.
  
  CUDA tests skip themselves when no GPU is present.
- Readability over cleverness. Comment the math. When a kernel is optimized, keep its naive version alongside it.
- Every new module or op family ships with its math note in `docs/math/`: what it computes, forward, backward derivation, shapes, mapping to code.
- FP32 everywhere in v1. A single seed drives initialization, data order and sampling.

## Documentation

- [Requirements](docs/product/requirements.md)
- [Architecture overview](docs/architecture/overview.md)
- [Decision records](docs/adr/README.md)
- [Contributing](CONTRIBUTING.md)

## MCP servers

None.
