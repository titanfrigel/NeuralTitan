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
- CMake 3.28 or newer

Build and run from the repository root in PowerShell:

```powershell
. ./scripts/dev-env.ps1; Enter-TitanDevEnv
cmake --preset release
cmake --build --preset release
./build/release/apps/titan.exe info
scripts/check.ps1
```

`Enter-TitanDevEnv` loads the MSVC 14.38 environment. `titan info` prints the CPU thread count and the CUDA device name, memory and compute capability (or `CUDA: not available`); `titan --version` prints the version. `scripts/check.ps1` is the single check command and defaults to the release preset; see [CONTRIBUTING.md](CONTRIBUTING.md) for the one-time formatter setup it needs.

## Commands

| Command | What it does |
| --- | --- |
| `scripts/check.ps1 [-Preset debug\|release]` | Checks formatting, builds with warnings as errors, runs all tests including the layer-rule check, and prints `check: OK`. CI runs the same command. |
| `scripts/format.ps1 [-Fix]` | Checks C++/CUDA formatting with the pinned clang-format, or rewrites the files with `-Fix`. |
| `ctest --preset release` | Runs the test suite of the release build (`ctest --preset debug` after building the debug preset). CUDA tests skip themselves when no GPU is present. |
| `titan --version` | Prints the library version. |
| `titan info` | Prints the CPU thread count and the CUDA device name, memory and compute capability, or that CUDA is not available. |
| `titan tokenizer train --type char --corpus <file> --out <tok.json>` | Builds a character tokenizer from a corpus and saves it. |
| `titan train --config <cfg.json> --data <corpus.txt> --out <dir>` | Trains a model, on the GPU when one is available; see [Training a model](#training-a-model). |
| `titan train --resume <dir>` | Continues the run saved in `<dir>/latest.ckpt`, bit-identical to an uninterrupted run; see [Training a model](#training-a-model). |
| `titan generate --checkpoint <file> [--prompt <text>]` | Streams text sampled from a trained checkpoint with the KV cache; see [Generating text](#generating-text). |
| `titan bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]` | Times square GEMMs with the cuBLAS, naive and tiled CUDA backends and prints each against the GPU's peak FP32 throughput; see the [matmul note](docs/math/matmul.md#measured-throughput). |

## Training a model

`titan train` trains a small decoder-only model on a text file, on the GPU when one is available and otherwise on the CPU:

```powershell
./build/release/apps/titan.exe train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m1-shakespeare
```

Options: `--tokenizer <tok.json>` (use a saved tokenizer instead of building a character tokenizer from the corpus), `--steps N` (overrides `train.steps`; `train.warmup` must stay below it), `--device cpu|cuda` (`--device cuda` without a GPU fails with `no CUDA device available`) and `--gemm cublas|naive|tiled` (the CUDA GEMM, cuBLAS by default; any other value needs `--device cuda`). The command writes `tokenizer.json` (when it built one), `train.bin` and `val.bin` (the encoded splits; the last `val_fraction` of the tokens validates), and `sample.txt` (500 characters sampled at temperature 0.8) to `--out`, and prints `device` (with the GEMM backend on CUDA), `params`, `time` and `final val_loss`. `titan tokenizer train --type char --corpus <file> --out <tok.json>` builds and saves a character tokenizer on its own.

Checkpoints: every run saves `latest.ckpt` to `--out` every `train.checkpoint_interval` steps (500 in `configs/tiny-shakespeare.json`; 0 or absent means only at the end), after the last step, and after the step given by `--stop-at N`, which ends the run there. Each save replaces the previous one atomically. The file holds the configs, the tokenizer, the parameters, the optimizer moments and the data-loader positions. `titan train --resume <dir> [--stop-at N] [--device cpu|cuda] [--gemm ...]` continues that run on the run directory's `train.bin` and `val.bin` and ends exactly where the uninterrupted run would; `--config`, `--data`, `--out`, `--tokenizer` and `--steps` are rejected with it. A run stopped early skips the sample.

The TinyShakespeare corpus is not in the repository: download `input.txt` from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) to `data/tinyshakespeare/` (git-ignored).

## Generating text

`titan generate` reads a checkpoint (it needs no other file) and streams text, character by character:

```powershell
./build/release/apps/titan.exe generate --checkpoint runs/m3-shakespeare/latest.ckpt --prompt "ROMEO:" --max-new 500
```

Options: `--prompt <text>` (default a newline; every character must be in the checkpoint's vocabulary), `--max-new N` (500), `--temp T` (0.8; 0 is greedy and ignores top-k and top-p), `--top-k K` (0, off), `--top-p P` (0.9; in (0, 1]), `--seed S` (the checkpoint's `train.seed`), `--device cpu|cuda` (cuda when a GPU is available) and `--no-cache` (recompute the whole window for every token instead of using the KV cache, for comparison). The prompt is echoed, each new character is printed as soon as it is sampled, and `tokens/s` goes to stderr. Bad flags exit with code 2 before any file is read; a missing or damaged checkpoint or an unknown prompt character exits with code 1 and prints nothing to stdout. The sampling math is in the [sampling note](docs/math/sampling.md) and the cache in the [KV cache note](docs/math/kv-cache.md).

## M1 result

`configs/tiny-shakespeare.json` (4 layers, dim 128, 4 heads with 2 KV heads, SwiGLU, RMSNorm, RoPE, tied embeddings, 2000 steps of 16 x 128 tokens, seed 1337), release build, CPU only:

| Measure | Value |
| --- | --- |
| Parameters | 734,464 |
| Wall time | 879 s (14.7 min) |
| Validation loss | 2.08 at step 200, 1.56 at step 2000 (from about ln 65 = 4.17 at initialization) |

The first 10 lines of the generated sample (prompt: a newline):

```text
Poor father to my tiger, who Edward how the most,
That, my fedilent and scorn and and wife?

BENVOLIO:
I am cousin.

Messenger:
It is the day rest, lord, secret, but shouldst
Our serves blessing sing, let him but for my young.

```

The model has learned the play format (speaker names, short lines, punctuation) and English-like words, but not meaning.

## M2 result

The same config and seed on the GPU (NVIDIA GeForce RTX 3070 Laptop GPU, compute 8.6), release build, cuBLAS GEMM:

```powershell
./build/release/apps/titan.exe train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m2-shakespeare --device cuda
```

| Measure | CPU (M1) | GPU (M2) |
| --- | --- | --- |
| Wall time (training) | 879 s | 29.6 s (29.7× faster; the target was at most 176 s, 5×) |
| Throughput (4.1 M tokens / wall time) | about 4,700 tokens/s | about 138,000 tokens/s |
| Final validation loss | 1.564 | 1.564 (target: below 2.0 and within 0.1 of M1) |

Validation loss every 200 steps on the GPU: 2.0827, 1.9510, 1.8192, 1.7186, 1.7000, 1.6476, 1.6025, 1.5812, 1.5772, 1.5640. Every logged training and validation loss equals the CPU run's to the four printed decimals. The two devices sum in different orders, so the runs are not bit-identical. The samples share their first 12 lines and then diverge: the small rounding differences eventually flip one sampled token, and every later token depends on it. The first lines of the GPU sample:

```text
Poor father to my tiger, who Edward how the most,
That, my fedilent and scorn and and wife?

BENVOLIO:
I am cousin.
```

`titan bench matmul` puts cuBLAS at up to 62% of the GPU's peak FP32 throughput and the project's tiled kernel at 8%; the table and a roofline reading are in the [matmul note](docs/math/matmul.md#measured-throughput).

## M3 result

Checkpoints, resume and KV-cached generation, on the same GPU (RTX 3070 Laptop GPU), release build, cuBLAS GEMM. The run was stopped after step 1000 and resumed:

```powershell
./build/release/apps/titan.exe train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m3-shakespeare --device cuda --stop-at 1000
./build/release/apps/titan.exe train --resume runs/m3-shakespeare --device cuda
```

The resumed run ends at validation loss 1.5640 (1.56398), equal to the uninterrupted M2 run's 1.5640. Every logged training loss, learning rate and gradient norm, every validation loss, and `sample.txt` are identical to the M2 run's, so the resume is bit-exact.

Generation from the resumed checkpoint (`--prompt "ROMEO:" --max-new 500`, default sampling: temperature 0.8, top-p 0.9, seed 1337), the first lines:

```text
ROMEO:
O, my lord, to stread my lord,
That would be say the world of the day
As the boy as sun sun above it end as thou
That livest was lives than or so revenge.

Second Servingman:
We the fine eyes most silk of discover.
```

Decode speed for `--prompt "ROMEO:" --max-new 500`, median of three runs, timed from the first prefill to the last token:

| Device | KV cache | `--no-cache` |
| --- | --- | --- |
| CPU | 598 tokens/s | 25.6 tokens/s |
| GPU | 521 tokens/s | 556 tokens/s |

On the CPU the cache is 23× faster: each new token costs one position instead of up to 128. On the GPU both paths are most likely bound by per-token overhead (many small kernel launches and a copy of the logits to the host), not by arithmetic, so recomputing the 128-token window costs nothing measurable. The cache's extra write and read kernels make the cached path slightly slower there. With the same seed, the cached and uncached paths give the same first 100 tokens on both devices.

## Documentation

- [Requirements](docs/product/requirements.md)
- [Architecture overview](docs/architecture/overview.md)
- [Math notes](docs/math/README.md)
- [Architecture decision records](docs/adr/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
