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
| `titan tokenizer train --type bpe --corpus <file> --vocab N --out <tok.json>` | Trains a byte-level BPE tokenizer of at most N tokens (N from 257 to 262144, so that the tokenizer fits the checkpoint's 16 MiB header; the corpus file may be up to 4 GiB) and saves it in Hugging Face's `tokenizer.json` layout; prints the vocabulary size, the merge count and the time. |
| `titan tokenizer encode --tokenizer <tok.json> (--text <s> \| --file <path>) [--summary] [--out <tokens.bin>]` | Encodes text with a character or BPE tokenizer file (ours or a Hugging Face `tokenizer.json`), on all cores, and prints the ids; `--summary` prints the token count, the CRC-32 of the ids and the bytes per token instead; `--out` writes the ids to a token file (creating its directory) and prints that summary. A `--file` corpus may be up to 4 GiB. |
| `titan train --config <cfg.json> (--data <corpus.txt> \| --train-tokens <a.bin> --val-tokens <b.bin> --tokenizer <tok.json>) --out <dir>` | Trains a model, on the GPU when one is available; see [Training a model](#training-a-model). |
| `titan train --resume <dir>` | Continues the run saved in `<dir>/latest.ckpt`, bit-identical to an uninterrupted run; see [Training a model](#training-a-model). |
| `titan generate --checkpoint <file> [--prompt <text>]` | Streams text sampled from a trained checkpoint with the KV cache; see [Generating text](#generating-text). |
| `titan generate --hf-model <dir> [--prompt <text>]` | Streams text sampled from a Hugging Face Llama model directory, such as SmolLM2-135M; see [Generating text](#generating-text). |
| `titan bench matmul [--sizes 256,512,1024,2048,4096] [--repeat 10]` | Times square GEMMs with the cuBLAS, naive and tiled CUDA backends and prints each against the GPU's peak FP32 throughput; see the [matmul note](docs/math/matmul.md#measured-throughput). |

## Training a model

`titan train` trains a small decoder-only model on a text file, on the GPU when one is available and otherwise on the CPU:

```powershell
./build/release/apps/titan.exe train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m1-shakespeare
```

Instead of `--data`, `--train-tokens <a.bin> --val-tokens <b.bin>` train on token files written by `titan tokenizer encode --out` (the second one is the validation split, for example an official validation file); they need `--tokenizer`, the tokenizer that encoded them, and cannot be combined with `--data`. Every id is checked against the vocabulary and each file must hold at least `seq + 2` tokens, both before anything is written; `val_fraction` is ignored, and `train.bin` and `val.bin` in `--out` are copies of the two files.

Options: `--tokenizer <tok.json>` (use a saved character or BPE tokenizer instead of building a character tokenizer from the corpus; the model's vocabulary size is the tokenizer's), `--steps N` (overrides `train.steps`; `train.warmup` must stay below it), `--device cpu|cuda` (`--device cuda` without a GPU fails with `no CUDA device available`) and `--gemm cublas|naive|tiled` (the CUDA GEMM, cuBLAS by default; any other value needs `--device cuda`). The command writes `tokenizer.json` (when it built one), `train.bin` and `val.bin` (the encoded splits; with `--data`, the last `val_fraction` of the tokens validates), and `sample.txt` (500 tokens sampled at temperature 0.8, which are characters with a character tokenizer) to `--out`, and prints `device` (with the GEMM backend on CUDA), `params`, `time` and `final val_loss`. `titan tokenizer train --type char --corpus <file> --out <tok.json>` builds and saves a character tokenizer on its own.

Checkpoints: every run saves `latest.ckpt` to `--out` every `train.checkpoint_interval` steps (500 in `configs/tiny-shakespeare.json`; 0 or absent means only at the end), after the last step, and after the step given by `--stop-at N`, which ends the run there. Each save replaces the previous one atomically. The file holds the configs, the tokenizer, the parameters, the optimizer moments and the data-loader positions. `titan train --resume <dir> [--stop-at N] [--device cpu|cuda] [--gemm ...]` continues that run on the run directory's `train.bin` and `val.bin` and ends exactly where the uninterrupted run would; `--config`, `--data`, `--train-tokens`, `--val-tokens`, `--out`, `--tokenizer` and `--steps` are rejected with it. A run stopped early skips the sample.

The TinyShakespeare corpus is not in the repository: download `input.txt` from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) to `data/tinyshakespeare/` (git-ignored).

## Training on TinyStories

`configs/tinystories.json` is the M6 run: 8 layers, dim 512, 8 heads with 4 KV heads, SwiGLU, RMSNorm, RoPE, tied embeddings and a 4096-token BPE vocabulary (25,305,600 parameters), 24000 steps of 16384 tokens each (8 x 512 tokens, 4 gradient-accumulation micro-batches). The data is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) V2 (GPT-4), not in the repository; download both files at the pinned revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` to `data/tinystories/` (git-ignored):

- [`TinyStoriesV2-GPT4-train.txt`](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-train.txt), 2,227,753,162 bytes, SHA-256 `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443`.
- [`TinyStoriesV2-GPT4-valid.txt`](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt), 22,502,601 bytes, SHA-256 `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`.

Then train the tokenizer, encode both files, and train:

```powershell
./build/release/apps/titan.exe tokenizer train --type bpe --corpus data/tinystories/TinyStoriesV2-GPT4-train.txt --vocab 4096 --out data/tinystories/tokenizer.json
./build/release/apps/titan.exe tokenizer encode --tokenizer data/tinystories/tokenizer.json --file data/tinystories/TinyStoriesV2-GPT4-train.txt --out data/tinystories/train.bin
./build/release/apps/titan.exe tokenizer encode --tokenizer data/tinystories/tokenizer.json --file data/tinystories/TinyStoriesV2-GPT4-valid.txt --out data/tinystories/valid.bin
./build/release/apps/titan.exe train --config configs/tinystories.json --train-tokens data/tinystories/train.bin --val-tokens data/tinystories/valid.bin --tokenizer data/tinystories/tokenizer.json --out runs/tinystories
```

Expected on the full files: training the tokenizer takes about 166 s, peaks near 2.25 GB of RAM and gives 4096 tokens (3839 merges); encoding gives 558,703,447 training tokens in about 37 s and 5,641,102 validation tokens, both at about 3.99 bytes per token. The run checkpoints to `runs/tinystories/latest.ckpt` every 1000 steps; if it is interrupted, continue it with `titan train --resume runs/tinystories`.

## Generating text

`titan generate` reads a checkpoint (it needs no other file) and streams text, token by token:

```powershell
./build/release/apps/titan.exe generate --checkpoint runs/m3-shakespeare/latest.ckpt --prompt "ROMEO:" --max-new 500
```

Options: `--prompt <text>` (default a newline; with a character tokenizer every character must be in the checkpoint's vocabulary), `--max-new N` (500), `--temp T` (0.8; 0 is greedy and ignores top-k and top-p), `--top-k K` (0, off), `--top-p P` (0.9; in (0, 1]), `--seed S` (the checkpoint's `train.seed`), `--device cpu|cuda` (cuda when a GPU is available) and `--no-cache` (recompute the whole window for every token instead of using the KV cache, for comparison). The prompt is echoed, each new token is printed as soon as it is sampled, and `tokens/s` goes to stderr. A BPE token can hold part of a UTF-8 character, so its bytes are held back until a later token completes the character; only such an incomplete tail is held back, and ill-formed bytes the model samples are printed as they are. Non-ASCII arguments reach `titan` as UTF-8 on Windows 10 version 1903 and later (the program's manifest sets the UTF-8 code page), but a legacy console shows non-ASCII output garbled unless you run `chcp 65001` first or use a UTF-8 terminal such as Windows Terminal. With a BPE tokenizer, generation stops early when the model samples the end-of-text token `<|endoftext|>`, which is not printed. Bad flags exit with code 2 before any file is read; a missing or damaged checkpoint or an unknown prompt character exits with code 1 and prints nothing to stdout. The sampling math is in the [sampling note](docs/math/sampling.md) and the cache in the [KV cache note](docs/math/kv-cache.md).

`titan generate --hf-model <dir>` runs a Hugging Face `LlamaForCausalLM` directory instead of a checkpoint: `config.json`, `tokenizer.json` and `model.safetensors` (F32, F16 or BF16 weights, converted to FP32), as in SmolLM2-135M:

```powershell
./build/release/apps/titan.exe generate --hf-model models/SmolLM2-135M --prompt "Once upon a time" --temp 0 --max-new 64
```

It takes the same options as `--checkpoint`, except that `--seed` defaults to 0, since such a model has no training seed; exactly one of `--checkpoint` and `--hf-model` is allowed. The files are read in full and validated, `config.json` first, before anything is printed, so an unsupported or damaged file exits with code 1, prints nothing to stdout and names the file and the field or tensor. Generation stops at the model's end-of-text token. Supported configs and the mapping onto our model are in the [Hugging Face Llama note](docs/math/hf-llama.md). The weights are not in the repository: download SmolLM2-135M at the revision pinned in `tools/make_llama_fixtures.py` to `models/SmolLM2-135M/` (git-ignored), or let that script download it.

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

## M4 result

Byte-level BPE on TinyShakespeare (`data/tinyshakespeare/input.txt`, 1,115,394 bytes), release build, on the RTX 3070 Laptop GPU machine (all three commands run on the CPU only):

```powershell
./build/release/apps/titan.exe tokenizer train --type bpe --corpus data/tinyshakespeare/input.txt --vocab 4096 --out runs/m4/bpe-4096.json
./build/release/apps/titan.exe tokenizer encode --summary --tokenizer runs/m4/bpe-4096.json --file data/tinyshakespeare/input.txt
./build/release/apps/titan.exe tokenizer encode --summary --tokenizer tests/fixtures/smollm2/tokenizer.json --file data/tinyshakespeare/input.txt
```

Training the 4096-token vocabulary learns 3,839 merges (4096 minus `<|endoftext|>` and the 256 bytes) in 0.27 s; three more runs took 0.20 to 0.23 s (median 0.21 s).

| Tokenizer | Vocabulary | Tokens for TinyShakespeare | Bytes per token | Encode time |
| --- | --- | --- | --- | --- |
| Ours, trained on TinyShakespeare | 4,096 | 344,104 | 3.241 | 239 ms |
| SmolLM2-135M | 49,152 | 341,094 | 3.270 | 350 ms |

Encode time is the wall-clock time of the whole `titan tokenizer encode --summary` process, including loading the tokenizer file, as measured by PowerShell's `Measure-Command`, median of three runs. SmolLM2's 341,094 tokens and CRC-32 `0xc11ec8a4` equal the reference ids in `tests/fixtures/smollm2/corpus.json`, so our encoder reproduces Hugging Face's on the whole corpus. Our vocabulary is 12 times smaller yet only 0.9% less compact on this text, because it was trained on it; SmolLM2's vocabulary covers a broad web corpus, code and other languages. The tokenizer is explained in the [BPE note](docs/math/bpe.md).

## M5 result

SmolLM2-135M (revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`) loaded from its Hugging Face files, compared with Hugging Face transformers 5.23.0 in FP32 on the CPU (`tools/make_llama_fixtures.py`), release build, on a laptop with an Intel Core i7-11800H (16 threads) and an NVIDIA GeForce RTX 3070 Laptop GPU.

Logits (`HfGolden.SmolLM2Logits*`, every position of each prompt over 1024 sampled vocabulary ids, and the full last position): `max |Δ|` divided by the logit scale `max(1, max |reference logit|)`, against the tolerance 1e-3.

| Prompt | CPU | CUDA |
| --- | --- | --- |
| `Once upon a time` | 2.03e-6 | 2.84e-6 |
| `The capital of France is` | 4.89e-6 | 3.05e-6 |
| `def fibonacci(n):` and a newline | 2.84e-6 | 3.63e-6 |
| `Café au lait, s'il vous plaît.` and a space | 2.82e-6 | 4.70e-6 |

The worst error is 4.89e-6 on the CPU and 4.70e-6 on CUDA, about 200 times inside the tolerance. Greedy generation (`HfGolden.SmolLM2Greedy*`) matches Hugging Face's first 32 tokens on all four prompts on both devices, with no divergence and so no near-tie needed. The two tiny random Llama fixtures (tied and untied embeddings) match too: worst 9.19e-7 on the CPU and 1.71e-6 on CUDA, and 16 of 16 greedy tokens on every prompt.

The greedy continuation of `Once upon a time`, 64 tokens, identical on the CPU and CUDA:

```powershell
./build/release/apps/titan.exe generate --hf-model models/SmolLM2-135M --prompt "Once upon a time" --temp 0 --max-new 64 --device cuda
```

> Once upon a time, there was a little girl named Lily. She lived in a big house with her family, but she didn't have many toys to play with. One day, her mom told her that she could go to a special place called a "school" to learn new things and become a teacher. Lily was excited to go

Its first 32 tokens are Hugging Face's reference continuation.

| Device | tokens/s (three runs) | Median |
| --- | --- | --- |
| CPU (`--device cpu`) | 6.6, 6.7, 6.7 | 6.7 |
| CUDA (`--device cuda`) | 82.6, 86.1, 85.5 | 85.5 |

tokens/s is what `titan generate` prints: the 64 generated tokens over the generation time, with the KV cache, excluding loading. Known limitation: the CPU speed is held back by `parallel_for`, which creates its threads on every call, and decoding one token makes about 271 matmul calls (211 in the linear layers and 60 in attention). This is inferred from measurements, not profiled; a thread pool is left for later. The loading and the parity method are explained in the [Hugging Face Llama note](docs/math/hf-llama.md).

## M6 result

`configs/tinystories.json` trained from scratch on TinyStories V2 (GPT-4) with the commands in [Training on TinyStories](#training-on-tinystories), release build, CUDA with cuBLAS, on the same laptop as M5 (RTX 3070 Laptop GPU). The run read 393,216,000 tokens (24000 steps of 16384), 0.70 of an epoch of the 558,703,447 training tokens, and validated on the 5,641,102 tokens of the official validation file (200 batches per evaluation).

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 2000 | 1.5975 | 1.6675 |
| 4000 | 1.5456 | 1.4903 |
| 6000 | 1.3529 | 1.4113 |
| 8000 | 1.3584 | 1.3605 |
| 10000 | 1.3476 | 1.3440 |
| 12000 | 1.2547 | 1.3007 |
| 14000 | 1.2252 | 1.2681 |
| 16000 | 1.2288 | 1.2302 |
| 18000 | 1.2868 | 1.2172 |
| 20000 | 1.1886 | 1.1873 |
| 22000 | 1.2182 | 1.1800 |
| 24000 | 1.0967 | 1.1665 |

The train loss is the logged step's loss, so it is noisy; the validation loss falls at every evaluation but the last (1.1627 at step 23000). The final validation loss is **1.1665**, against the target of about 1.5, which it passes from step 4000 (1.4903).

| Measure | Value |
| --- | --- |
| Wall time | 19,990 s (5 h 33 min), against a budget of about 8 h |
| Throughput | 19,670 tokens/s averaged over the run; the logged median is 20,263 |
| Peak VRAM | about 3.5 GB (measured on the 100-step smoke run of this config) |
| Checkpoint | 303,797,248 bytes (`latest.ckpt`, weights and AdamW state) |

Five samples, each the prompt `Once upon a time` with the default settings (`--temp 0.8 --top-p 0.9`) and seeds 1 to 5:

```powershell
./build/release/apps/titan.exe generate --checkpoint runs/tinystories/latest.ckpt --prompt "Once upon a time" --seed 1 --device cuda
```

Seed 1:

> Once upon a time, in a big forest, there was a little bird named Billy. Billy was a nervous bird. He was scared to fly. He was scared of big animals. One day, Billy saw a big bear. The bear was sleeping under a tree. Billy wanted to be friends with the bear.
>
> Billy said, "Hello, Mr. Bear. Can we be friends?" The bear opened his eyes and looked at Billy. He said, "I am sorry, little bird. I cannot be friends with you. I am too big to play with you."
>
> Billy was sad. He wanted to play with the bear. He thought and thought. Then, he had an idea. "Maybe if I jump high, you can touch me, and I will be your friend."
>
> The bear liked the idea. He found a big leaf. Billy said, "Here, Mr. Bear. You can sit on this leaf, and I will touch you." The bear was happy. He sat on the leaf, and Billy could touch him. They became good friends.
>
> The moral of the story is: even if you are nervous, you can still make new friends if you try.

Seed 2:

> Once upon a time, there was a little girl named Lily. She loved to eat yummy food. Her favorite food was a big, red apple. Lily's mom would always tell her, "Do not eat that apple, Lily." But Lily did not care.
>
> One day, Lily saw a big, red apple on the table. She wanted to eat it. But she remembered her mom's words. She knew her mom would not let her eat the apple. So, Lily thought, "Maybe I can just have a little bite."
>
> Lily took a small bite of the apple. But it was not a good apple. It was very bad. It made her feel bad. Lily's mom found out and was very sad. Lily learned that she should always listen to her mom.

Seed 3:

> Once upon a time, there was a little girl named Sue. She was very adventurous. Sue loved to play outside and explore.
>
> One day, Sue went on a big adventure. She walked and walked until she found a big tree. She looked up and saw a bird's nest with eggs in it. Sue wanted to take the eggs home.
>
> Sue tried to reach the nest, but she was too small. She asked her friend, Tom, for help. Tom said, "I will help you." They found a long stick and used it to get the eggs. The eggs were now in the nest.
>
> Sue and Tom were happy. They took the eggs back to the tree. Sue said, "Thank you, Tom." They shared the eggs and had a great day.

Seed 4:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and have tea parties with her friends. One day, she found a big, fake pot in her room. She had an idea. She thought it would be fun to put the pot on the table and see if her toys could fit inside it.
>
> Lily asked her mom if she could put the fake pot on the table. Her mom said, "Yes, but be careful not to put too much." Lily was very happy and started to put her toys in the fake pot. She thought it was a fun game.
>
> But when she tried to fit the fake pot on the table, it did not fit. It was too big! Lily was sad. Then, her mom came into the room and saw the big mess. She said, "Don't worry, Lily. We can fix it." They worked together to fix the fake pot. In the end, Lily learned that it is not just for fun, but also for making a big mess.

Seed 5:

> Once upon a time, in a small house, there lived a little girl named Lily. She had a big sister named Emma. They loved to play together.
>
> One day, Lily and Emma were playing in the living room. Lily saw a pretty lamp on the table. She wanted to touch it, but Emma said, "No, Lily! Don't touch the lamp!"
>
> Lily did not listen to Emma. She touched the lamp and it fell on the floor. The lamp broke into pieces. Lily was very sad and said, "I'm sorry, Emma."
>
> Emma said, "It's okay, Lily. We can clean it up." They worked together to put the pieces back on the table. Lily learned to listen to her sister and to be careful with things that can break.

Every sample ends on the end-of-text token, well before `--max-new 500`. They are grammatical, keep their characters, and end with a resolution, with small lapses of logic (seed 3's eggs are "now in the nest" after being taken from it). Generation runs at about 300 tokens/s on CUDA with the KV cache.

## Documentation

- [Requirements](docs/product/requirements.md)
- [Architecture overview](docs/architecture/overview.md)
- [Math notes](docs/math/README.md)
- [Architecture decision records](docs/adr/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
