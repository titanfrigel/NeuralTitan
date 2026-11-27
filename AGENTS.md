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
- `cmake --preset release` then `cmake --build --preset release`: configure and build (`debug` preset also exists). The program is `build/<preset>/apps/titan.exe`; its commands are `titan --version`, `titan info`, `titan tokenizer train`, `titan tokenizer encode`, `titan train`, `titan generate` and `titan bench matmul` (below).
- `build/<preset>/apps/titan.exe tokenizer train --type char --corpus <file> --out <tok.json>`: build and save a character tokenizer.
- `build/<preset>/apps/titan.exe tokenizer train --type bpe --corpus <file> --vocab N --out <tok.json>`: train a byte-level BPE tokenizer of at most N tokens (N in [257, 262144]; the cap keeps the tokenizer inside the checkpoint's 16 MiB header; fewer when the corpus runs out of repeating pairs) and print its size, merge count and training time. `--vocab` is required with `bpe` and rejected with `char`.
- `build/<preset>/apps/titan.exe tokenizer encode --tokenizer <tok.json> (--text <s> | --file <path>) [--summary]`: encode text with a character or BPE tokenizer file (ours or a Hugging Face `tokenizer.json`) and print the ids on one line; exactly one of `--text` and `--file`. `--summary` prints `tokens N`, `crc32 0x%08x` (of the ids as little-endian int32, as in `tests/fixtures/smollm2/corpus.json`) and `bytes_per_token X.XXX` instead.
- `build/<preset>/apps/titan.exe train --config <cfg.json> --data <corpus.txt> --out <dir> [--tokenizer <tok.json>] [--steps N] [--device cpu|cuda] [--gemm cublas|naive|tiled]`: train a model (on the GPU when one is available, otherwise on the CPU; `--gemm` picks the CUDA GEMM and needs `--device cuda` unless it is `cublas`) and write the tokenizer, token files and a sample to `<dir>`. `--tokenizer` takes a character or a BPE tokenizer file; without it a character tokenizer is built from the corpus. `configs/tiny-shakespeare.json` is the M1 and M2 config (M3 adds `checkpoint_interval`). Every run saves `<dir>/latest.ckpt` (every `train.checkpoint_interval` steps, after the last step and after `--stop-at N`); `--stop-at N` ends the run after step N.
- `build/<preset>/apps/titan.exe train --resume <dir> [--stop-at N] [--device cpu|cuda] [--gemm cublas|naive|tiled]`: continue the run saved in `<dir>/latest.ckpt` on `<dir>/train.bin` and `val.bin`, bit-identical to an uninterrupted run. `--config`, `--data`, `--out`, `--tokenizer` and `--steps` are rejected with it.
- `build/<preset>/apps/titan.exe generate --checkpoint <file> [--prompt <text>] [--max-new N] [--temp T] [--top-k K] [--top-p P] [--seed S] [--device cpu|cuda] [--no-cache]`: echo the prompt (default a newline) and stream sampled text to stdout with the KV-cached generator, then print `tokens/s` to stderr. Generation stops early at the tokenizer's end-of-text token (BPE only), and bytes of a character split across tokens are held back until it is complete. Defaults: `--temp 0.8 --top-k 0 --top-p 0.9 --max-new 500`, the seed from the checkpoint. `--temp 0` is greedy; `--no-cache` recomputes the whole window per token.
- `build/<preset>/apps/titan.exe bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]`: time square GEMMs with each CUDA GEMM backend against the GPU's peak FP32 throughput and bandwidth (needs a GPU).
- `tools/.venv/Scripts/python -I tools/make_tokenizer_fixtures.py`: regenerate the SmolLM2 tokenizer fixtures in `tests/fixtures/smollm2/` (downloads `models/SmolLM2-135M/tokenizer.json` at the pinned revision if it is missing; writes `corpus.json` only when `data/tinyshakespeare/input.txt` exists) and `tests/fixtures/bpe/encode.json`, Hugging Face's ids for our `trained.json`.
- The trained BPE fixture: `tests/fixtures/bpe/corpus.txt` is the first 20000 bytes of `data/tinyshakespeare/input.txt`, cut after the last newline. Regenerate `trained.json` with `build/release/apps/titan.exe tokenizer train --type bpe --corpus tests/fixtures/bpe/corpus.txt --vocab 512 --out tests/fixtures/bpe/trained.json`, then run the fixture script for `encode.json`. `BpeTrained.FixtureIsTrainerOutput` fails when the trainer and the fixture disagree.
- `ctest --preset release`: run the tests of the release build (`ctest --preset debug` after building the debug preset).
- `scripts/check.ps1 [-Preset debug|release]`: the single check: format check, build with warnings as errors, all tests including the layer check. Prints `check: OK`. CI runs it.
- `scripts/format.ps1 [-Fix]`: check or fix formatting with clang-format 20.1.8. One-time setup: `python -m venv tools/.venv`, then `tools/.venv/Scripts/python -m pip install -r tools/requirements-dev.txt`.
- `git config core.hooksPath .githooks`: enable the gitleaks pre-commit hook (once per clone).
- `node <senior-standards plugin>/scripts/adr-index.mjs .` regenerates `docs/adr/README.md` after adding or superseding an ADR.

## Layout

Current:

- `src/core/`: the `core` layer (errors, device, storage, caching allocator, tensor, rng, parallel_for, `utf8.*` for UTF-8 encoding, decoding and `Utf8Stream`, which holds back an incomplete character when streaming).
- `src/io/`: the `io` layer (JSON parser and writer, file helpers, token files, `checksum.*` (CRC-32) and the checkpoint container `checkpoint_file.*`).
- `src/tokenizer/`: the `tokenizer` layer (character-level tokenizer, the byte-level pre-tokenizer, the BPE tokenizer and its trainer `bpe_trainer.*`, and `unicode_tables.*`, the Unicode letter, number and whitespace classes probed from the reference `tokenizers` by `tools/gen_unicode_tables.py`; do not edit them by hand).
- `src/ops/`: the `ops` layer, with the CPU backend `ops/cpu` and the CUDA backend `ops/cuda` (`common.cuh` launch helpers and shared block reduction, `cublas_handle.cu`, and one `.cu` per op family: elementwise, reduce, matmul, softmax, norm, rope, layout, embedding, cross-entropy, adamw).
- `src/nn/`: the `nn` layer (modules with hand-written backward passes, and the KV cache for inference).
- `src/optim/`: the `optim` layer (AdamW, gradient clipping, learning-rate schedule).
- `src/model/`: the `model` layer (model config and the decoder-only model).
- `src/runtime/`: the `runtime` layer (data loader, trainer, checkpoints, sampler, KV-cached generator).
- `apps/titan/`: the `titan` executable (the `cli` layer): `main.cpp`, `commands.*` and `titan.manifest`, which makes UTF-8 the process code page so that non-ASCII arguments reach `main` as UTF-8.
- `configs/`: model and training JSON configs (`tiny-shakespeare.json`).
- `tests/`: the test runner and tests. `tests/harness/` is the project's own test harness; `tests/layer_check/` holds fixtures for the layer include check; `tests/fixtures/cli/` holds hand-written CLI inputs (not generated golden fixtures); `tests/fixtures/smollm2/` holds SmolLM2-135M's `tokenizer.json` (Apache-2.0, see its `NOTICE.md`) and the reference token ids generated by `tools/make_tokenizer_fixtures.py` (regenerate them, never edit them); `tests/fixtures/bpe/` holds a corpus, the tokenizer `titan` trains on it and Hugging Face's ids for that tokenizer; the `tests/harness/expect_*.cmake` scripts drive the CLI tests (`expect_valid_utf8.cmake` checks that stdout is valid UTF-8); `tests/tools/` holds small test-only programs (`edit_token_file` writes a token file with one token replaced); `tests/support/` holds shared helpers (`gradcheck.*`, `parity.*` for CPU-vs-CUDA comparisons, and `random_text.*` for seeded random UTF-8 strings); `tests/ops/cuda/` holds the CPU-vs-CUDA parity tests.
- `scripts/`: `check.ps1`, `format.ps1`, `dev-env.ps1` and the layer include check.
- `tools/`: development-only Python requirements (the venv is git-ignored), `gen_unicode_tables.py`, which regenerates `src/tokenizer/unicode_tables.*` (`tools/.venv/Scripts/python -I tools/gen_unicode_tables.py`), and `make_tokenizer_fixtures.py`, which regenerates `tests/fixtures/smollm2/` and `tests/fixtures/bpe/encode.json`.
- `docs/specs/`: approved design specs.
- `docs/plans/`: implementation plans.
- `docs/product/`: requirements.
- `docs/architecture/`: architecture overview.
- `docs/math/`: math notes, one per module or op family.
- `docs/adr/`: decision records. `README.md` there is generated; do not edit it by hand.
- `.github/`: CI workflows (ci, pr-title, gitleaks, dependency-check, docs, scorecard), the pull request template and vendored checks.
- `.githooks/`: the gitleaks pre-commit hook.

Planned (created by the milestones; layer names follow the architecture overview):

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
