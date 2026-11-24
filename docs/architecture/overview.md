# Architecture overview

<!-- Structure follows arc42-lite. -->

## 1. Context and scope

NeuralTitan is a C++/CUDA library plus a `titan` command-line program, running on a single Windows machine with one NVIDIA GPU. Nothing runs as a service; every interaction is a local process reading and writing local files.

```mermaid
C4Context
  title System context
  Person(author, "Author", "Builds, trains and runs models; reviews the code")
  System(titan, "NeuralTitan", "LLM library and titan CLI")
  System_Ext(gpu, "NVIDIA GPU and driver", "CUDA runtime, RTX 3070 Laptop")
  System_Ext(hf, "Hugging Face Hub", "SmolLM2-135M weights and tokenizer files")
  System_Ext(corpora, "Text corpora", "TinyShakespeare, TinyStories")
  System_Ext(reftool, "Golden reference tooling", "Python, torch, transformers under tools/")
  Rel(author, titan, "Runs train / generate / tokenizer commands")
  Rel(titan, gpu, "Launches kernels, copies memory")
  Rel(titan, hf, "Reads downloaded safetensors and tokenizer.json")
  Rel(titan, corpora, "Reads downloaded text files")
  Rel(reftool, hf, "Downloads the model, runs reference inference")
  Rel(reftool, titan, "Writes fixture files used by tests")
```

- **Author.** Runs the `titan` CLI and the test suite, and reviews the code and math notes. Single user, local machine.
- **NVIDIA GPU and driver.** Reached only through the CUDA runtime API, from the `cuda` part of the `ops` layer and the device memory code in `core`.
- **Hugging Face Hub.** NeuralTitan never makes network calls. Weight (`*.safetensors`), config (`config.json`) and tokenizer (`tokenizer.json`) files are downloaded beforehand into a local `models/` directory (git-ignored), then read by the `io` and `tokenizer` layers.
- **Text corpora.** Plain text files downloaded into a local `data/` directory (git-ignored) and tokenized into binary token files by `titan`.
- **Golden reference tooling.** Development-only Python scripts under `tools/` (ADR 0004). They produce fixture files committed under `tests/fixtures/`. They are never invoked by the build or the tests. The tokenizer fixtures flow one way: `tools/make_tokenizer_fixtures.py` reads SmolLM2's `tokenizer.json` (downloaded at a pinned revision) and runs Hugging Face's `tokenizers` on fixed strings and on TinyShakespeare, writing the reference ids to `tests/fixtures/smollm2/`; it also encodes with `tests/fixtures/bpe/trained.json`, which `titan tokenizer train` produced, writing `tests/fixtures/bpe/encode.json`. The C++ tests then require our encoder to give exactly those ids. `tools/gen_unicode_tables.py` generates `src/tokenizer/unicode_tables.*` the same way. Fixtures are regenerated, never edited by hand.

## 2. Solution strategy

- **Layered architecture with swappable device backends** (ADR 0005). Calls follow the math top to bottom. Device polymorphism exists only in `ops`, behind one dispatching interface with independent CPU and CUDA implementations.
- **Module objects with hand-written backward** (design spec, approach A). Every `nn` module implements `forward`, `backward` and parameter enumeration. Composite modules call their children's backward in reverse order.
- **CPU reference first.** Every op is written first as plain CPU code, then as a CUDA kernel that is tested against it. Every module's backward is checked against finite differences.
- **C++20, CUDA 12.6, MSVC 14.38, CMake** (ADR 0002). **No third-party libraries** in the build except **cuBLAS, the default GEMM**. The project's own naive and tiled GEMM kernels are kept for learning, parity and roofline measurements, selectable at runtime (ADR 0007).
- **Independent golden reference** from Hugging Face `transformers` for the pretrained-model tests (ADR 0004).

## 3. Building blocks

Everything ships as a set of static libraries (one per layer) linked into the `titan` executable and the test runner.

```mermaid
C4Container
  title Containers
  Person(author, "Author", "Runs commands and tests")
  Container(cli, "titan CLI", "C++20", "Parses commands, wires runtime objects")
  Container(tests, "Test runner", "C++20, own harness", "Unit, parity, gradient and golden tests")
  Container(lib, "NeuralTitan library", "C++20 + CUDA 12.6, static libraries", "core, io, tokenizer, ops, nn, optim, model, runtime")
  ContainerDb(files, "Local files", "filesystem", "Configs, corpora, token files, checkpoints, HF model files, fixtures")
  System_Ext(gpu, "NVIDIA GPU", "CUDA runtime")
  Container(tools, "Reference tools", "Python, torch, transformers", "Generate golden fixtures")
  Rel(author, cli, "Runs")
  Rel(author, tests, "Runs")
  Rel(cli, lib, "Calls")
  Rel(tests, lib, "Calls")
  Rel(lib, files, "Reads and writes")
  Rel(lib, gpu, "Kernels and memory copies")
  Rel(tools, files, "Writes fixtures")
```

Library layers, bottom to top:

| Layer | Responsibility | Technology | ADR |
| --- | --- | --- | --- |
| `core` | `Tensor`, `Storage`, `Device`, CUDA caching allocator, seeded RNG, error checking | C++20, CUDA runtime | 0002, 0007 |
| `io` | JSON parser, safetensors reader, checkpoint binary format, bounds-checked binary reading | C++20 | 0007 |
| `tokenizer` | The `Tokenizer` interface and `load_tokenizer`, character tokenizer, byte-level BPE (train, encode, decode, load and save `tokenizer.json`, the end-of-text id), Unicode category tables | C++20 | 0007 |
| `ops` | Device-dispatching operations; `ops/cpu` reference implementations; `ops/cuda` kernels; GEMM through cuBLAS by default, with own naive and tiled kernels selectable | C++20, CUDA, cuBLAS | 0002, 0005, 0007 |
| `nn` | `Module`, `Parameter`, Embedding, Linear, RMSNorm, LayerNorm, Attention (GQA, RoPE, KV cache), SwiGLU, GeluMLP, TransformerBlock, LMHead, CrossEntropyLoss | C++20 | 0005 |
| `optim` | AdamW, learning-rate schedules, gradient clipping | C++20 | 0005 |
| `model` | `ModelConfig`, model assembly from config, presets, mapping Hugging Face weights to parameters, checkpoint save/load | C++20 | 0005 |
| `runtime` | DataLoader, Trainer, Generator, Sampler | C++20 | 0005 |
| `cli` | `titan` subcommands: `tokenizer train` (`--type char\|bpe`, `--vocab N`), `tokenizer encode` (`--text` or `--file`, `--summary`), `train` (`--tokenizer` with a character or BPE file; `--device cpu\|cuda`, defaulting to cuda when a GPU is available; `--gemm cublas\|naive\|tiled`; `--stop-at N`; `--resume <dir>`), `bench matmul`, `generate` (`--checkpoint`, sampling flags, `--no-cache`; stops at end of text) | C++20 | 0005 |

## 4. Runtime view

### Training step

```mermaid
sequenceDiagram
  actor Author
  participant CLI as titan train
  participant Trainer
  participant Loader as DataLoader
  participant Model
  participant Loss as CrossEntropyLoss
  participant Opt as AdamW
  participant Ops as ops (cpu or cuda)
  Author->>CLI: titan train --config c.json --data d
  CLI->>Trainer: build model, optimizer, loader
  loop each step
    Trainer->>Loader: next batch (B x T tokens, targets)
    Trainer->>Model: forward(tokens, training)
    Model->>Ops: embedding, blocks (norm, attention, MLP), head
    Trainer->>Loss: forward(logits, targets)
    Trainer->>Loss: backward()
    Trainer->>Model: backward(dlogits)
    Model->>Ops: backward ops in reverse order, accumulate grads
    Trainer->>Opt: clip grads, step(lr), zero grads
    Trainer-->>Author: log loss, tokens/s, lr
  end
  Trainer->>Trainer: periodic validation loss and checkpoint
```

### Generation

```mermaid
sequenceDiagram
  actor Author
  participant CLI as titan generate
  participant Tok as Tokenizer
  participant Gen as Generator
  participant Model
  participant Samp as Sampler
  Author->>CLI: titan generate --prompt "..."
  CLI->>Tok: encode(prompt)
  CLI->>Gen: generate(tokens, settings)
  Gen->>Model: forward(prompt tokens, inference) fills KV cache
  loop until max tokens or EOS
    Gen->>Samp: sample(last logits)
    Samp-->>Gen: next token
    Gen->>Tok: decode(token)
    Gen-->>Author: stream text
    Gen->>Model: forward(next token, inference) uses KV cache
  end
```

As implemented in M3 and M4 (`titan generate` in `apps/titan/commands.cpp`, `runtime::Generator` in `src/runtime/generator.*`):

1. The CLI validates every flag first: a bad `--temp`, `--top-k`, `--top-p`, `--max-new` or `--seed` is a usage error (exit 2) before any file is read.
2. It reads the checkpoint header (`load_checkpoint(path, nullptr)`), rebuilds the tokenizer from the checkpoint's JSON (`checkpoint_tokenizer`, which returns a `tokenizer::Tokenizer` through `load_tokenizer`, so a character or a BPE tokenizer) and encodes the prompt. A damaged file or a character outside the vocabulary fails here, with `error:` on stderr, exit 1 and nothing on stdout.
3. It builds the model from the saved config, loads the parameters, moves the model to the device, and only then constructs the `Generator`, whose KV cache is allocated on the model's device.
4. It echoes the prompt and streams each sampled token to stdout, flushing per token, then writes `tokens/s` to stderr. A BPE token may end inside a UTF-8 character, so the decoded bytes go through a `Utf8Stream` (`src/core/utf8.*`): after each token the CLI pushes `decode({id})` and prints what `push` returns, the `utf8_complete_prefix` of the pending bytes; the stream keeps the rest. When generation ends it prints `finish()`, whatever is still pending, as it is, which is an incomplete character only if the model stopped in the middle of one. Only an incomplete tail is held back: ill-formed bytes the model samples, such as a stray continuation byte token, are printed as they are, so stdout is valid UTF-8 only when the sampled tokens form valid text (see the [BPE note](../math/bpe.md#decode)). `titan.manifest` makes UTF-8 the process code page, so a non-ASCII `--prompt` reaches the tokenizer as UTF-8. It fixes the arguments, not the console: a legacy console shows non-ASCII output garbled unless its code page is UTF-8 (`chcp 65001`) or the terminal is a UTF-8 one such as Windows Terminal, and Windows ignores the manifest setting before Windows 10 version 1903.

- Generation stops at end of text: the CLI passes the tokenizer's `eos_id()` as the stop id to `Generator::generate` (and to `generate_naive` with `--no-cache`). A BPE tokenizer's `<|endoftext|>` ends generation as soon as it is sampled and is not printed. The character tokenizer has no end-of-text token, so its generation always runs `--max-new` tokens.
- When the cache is full, the Generator slides by half: it keeps the last `context_length / 2` tokens, clears the cache and prefills them again.
- `--no-cache` runs `generate_naive` with the same sampler settings instead, recomputing the whole window for every token. Before the first slide, both paths give the same tokens for the same seed.

### Failure path: malformed input file

```mermaid
sequenceDiagram
  actor Author
  participant CLI as titan generate
  participant Loader as model weight loader
  participant ST as io safetensors reader
  Author->>CLI: titan generate --hf-model models/x
  CLI->>Loader: load(models/x)
  Loader->>ST: open(model.safetensors)
  ST->>ST: check header length, JSON, dtypes, offsets within file size
  ST-->>Loader: error: tensor offset beyond end of file
  Loader-->>CLI: error with file name and tensor name
  CLI-->>Author: message on stderr, exit code 1
```

## 5. Deployment view

There is no deployed environment. NeuralTitan is built and run on the author's machine; GitHub hosts the source and runs CI.

```mermaid
flowchart LR
  Dev[Author's machine<br/>Windows 11, MSVC 14.38, CUDA 12.6, RTX 3070] -->|git push, PR| GH[GitHub repository<br/>titanfrigel/NeuralTitan]
  GH --> CI[GitHub Actions<br/>Windows runner, no GPU:<br/>build all, run CPU tests, checks]
  Dev -->|local check: build, CPU and CUDA tests| Dev
```

- **Build:** CMake configures with toolset 14.38 and CUDA architecture 86. One `check` entry point builds, runs all tests and runs the repository checks.
- **CI:** compiles everything, including CUDA code, and runs the CPU tests and repository checks. CUDA tests skip themselves when no GPU is present, and run locally before merging.
- **Releases:** none (setup record: versioned releases `no`). `main` is always the latest reviewed state.
- **Rollback:** revert the offending PR on `main`.

## 6. Cross-cutting concerns

- **Authentication and authorization:** not applicable; local single-user tool with no network access.
- **Error handling:**
  - One strategy: exceptions derived from `titan::Error`, carrying a message with context (file, tensor or parameter name, shapes).
  - `TITAN_CHECK(cond, msg)` guards preconditions such as shape compatibility and device agreement.
  - `TITAN_CUDA_CHECK` wraps every CUDA runtime call and checks kernel launches.
  - External input (JSON, safetensors, checkpoints, token files, tokenizer files, CLI arguments) is validated at the `io`/`tokenizer`/`cli` boundary and in the config parsers built on `io::JsonFields` (`parse_model_config` in `model`, `parse_train_config` in `runtime`), where every length and offset is bounds-checked.
  - The CLI catches at the top level, prints the message to stderr and exits non-zero.
  - Kernels themselves do not validate; their callers guarantee their preconditions.
- **Logging and observability:** plain text to stdout (training progress: step, loss, tokens/s, learning rate, validation loss) and stderr (errors). The trainer optionally also writes a CSV metrics file per run for plotting. No telemetry.
- **Configuration and secrets:** model and training configuration in JSON files under `configs/`, with CLI flags overriding run-level settings. No secrets exist. The Hugging Face downloads used by `tools/` need no token for SmolLM2.
- **Determinism:** one seed controls initialization, data order and sampling. Runs are reproducible on the same device, but CPU and CUDA results are not bit-identical.
- **Internationalization:** not applicable to the program. Text processing is UTF-8 throughout; the BPE tokenizer works on bytes.

### Checkpoint format

Checkpoints use a container written by `io::write_checkpoint_file` and read by `io::read_checkpoint_file` (`src/io/checkpoint_file.*`). All integers are little-endian.

```text
bytes 0..3    "TCKP"
bytes 4..7    u32 version = 1
bytes 8..15   u64 total file size
bytes 16..23  u64 header length H (1 <= H <= 16 MiB)
bytes 24..27  u32 CRC-32 of the header bytes
bytes 28..    H bytes of UTF-8 JSON:
                {"meta": {...}, "tensors": [{"name", "shape", "offset", "crc32"}, ...]}
then          zero padding to a multiple of 64, which starts the data section
data          each tensor's float32 values at its offset (relative to the data
              section, a multiple of 64), in the order listed
```

- Every tensor is padded with zeros to 64 bytes, so the file ends exactly at the end of the last tensor's padding. Offsets are packed in listed order.
- The CRC is CRC-32/ISO-HDLC (`io::crc32`, the zlib polynomial).
- The reader checks the magic, version, total size, header length and header CRC before it parses the JSON. It then bounds-checks every entry (rank 1 to 4, positive dimensions, byte size without overflow, offset and size inside the data section, unique names) before allocating, streams each tensor from the file into its host tensor, and verifies the tensor's CRC. Any failure throws a `titan::Error` naming the file and the tensor (threat T2).
- `io::read_checkpoint_meta` runs the same file-level and header checks and returns only `meta`, without reading tensor data, so a loader can read a checkpoint's configuration before building a model.
- The writer writes `<path>.tmp` and renames it over `<path>` only when the file is complete, so an existing checkpoint survives a failed write. It rejects tensors that are not on the CPU.

`runtime::save_checkpoint` and `runtime::load_checkpoint` (`src/runtime/checkpoint.*`) map a training run onto the container, so that one file is enough to resume training or to generate text:

```text
meta     {"vocab_size": V,
          "model": <model_config_to_json>, "train": <train_config_to_json>,
          "tokenizer": <the tokenizer's JSON>,
          "state": {"step", "optimizer_steps", "train_rng", "val_rng",
                    "train_tokens", "val_tokens", "train_crc32", "val_crc32"}}
tensors  every parameter under its name, in the model's parameter order,
         then "optim.m.<name>" for each, then "optim.v.<name>" for each
```

- The meta is validated like a config file: unknown or missing keys, wrong types and malformed generator states fail with the file and the field path, and the `model` and `train` sections go through `parse_model_config` and `parse_train_config`.
- `load_checkpoint(path, nullptr)` reads only the header, so a caller can build the model from the saved config first. Before returning, it bounds that config by the file (threat T1): `model::parameter_count` of the config, times three for the parameters and both moments, must equal the values the header's tensor entries hold (`io::read_checkpoint_header`). A header that claims a huge model fails there, before any model is allocated. With a model, it requires exactly the model's parameter and moment names with matching shapes, checks them all before it changes the model, and copies the parameters onto the model's current device. It also requires the saved model config to equal the model's, naming the first field that differs (for example `model.rope_theta`), since settings without a tensor of their own pass the shape checks.
- A resumed `Trainer` takes the saved `TrainerState`: it moves the model to the training device, builds AdamW there and loads the moments into it, restores both loaders' generators and continues at the step after the saved one, so the result equals the uninterrupted run bit for bit. It refuses token streams whose lengths differ from the saved counts.
- The state also carries `train_crc32` and `val_crc32`, the CRC-32 of each token stream's int32 bytes (`runtime::token_crc32`). A resumed `Trainer` refuses streams of the saved length whose checksum differs, so re-tokenized files of the same length cannot pass silently. `titan train --resume` checks both files first and names the one that changed.

The `titan train` command wires this in (M3):

- Every run saves `<out>/latest.ckpt` through `Trainer::on_checkpoint`: every `train.checkpoint_interval` steps (when above 0), after the last step, and after the `--stop-at N` step. Each save overwrites the previous one; there is no checkpoint history.
- `--stop-at N` ends a run after step N, which must lie after the last completed step and at most at `train.steps`. A stopped run skips the end-of-run sample.
- `titan train --resume <dir>` reads `<dir>/latest.ckpt` (the header first, then the model built from it), `train.bin` and `val.bin`. It checks every token id against the vocabulary and continues to `train.steps`. A checkpoint already at `train.steps` is refused ("the run already finished at step N of N") before any output. The configs and tokenizer come from the checkpoint, so `--config`, `--data`, `--out`, `--tokenizer` and `--steps` are usage errors with it. Only the device and the GEMM backend are chosen anew.
- On Windows a save can fail while another process has `latest.ckpt` open, for example `titan generate` reading a checkpoint of a run that is still training: the rename over the open file is refused. The previous checkpoint survives (the writer only replaces it once the new file is complete), and the training run stops with the error.
- The end-of-run sample still uses `generate_naive` with the M1 seed and temperature, so `sample.txt` stays reproducible across milestones.

## 7. Architecture decisions

Decisions are recorded as ADRs; see the [decision index](../adr/README.md).

- [ADR 0002: Language and build toolchain](../adr/0002-language-and-build-toolchain.md)
- [ADR 0007: Third-party libraries in the build, with cuBLAS as the default GEMM](../adr/0007-cublas-default-gemm.md) (supersedes ADRs 0003 and 0006)
- [ADR 0004: Golden reference tooling](../adr/0004-golden-reference-tooling.md)
- [ADR 0005: Architecture style](../adr/0005-layered-architecture.md)

## 8. Quality goals

| Priority | Quality goal | Scenario and target |
| --- | --- | --- |
| 1 | Correctness | Every CUDA op agrees with its CPU reference and every module passes a gradient check (100%). SmolLM2-135M logits are within 1e-3 of the reference; the first 32 greedy tokens match. |
| 2 | Understandability | A reader follows any forward or backward pass from `Model` down to a kernel through direct calls, with a math note for every module and op family (100% coverage). |
| 3 | Performance within budget | TinyShakespeare on CPU in under 30 min; TinyStories on GPU in about 8 h or less, reaching validation loss ≈ 1.5. |
| 4 | Reproducibility | Same seed, same device: identical loss curve and identical samples. |
| 5 | Robustness to bad input | A malformed or truncated model, tokenizer, checkpoint or config file produces a clear error and a non-zero exit, never a crash or out-of-bounds access. |

## 9. Threat model

### Assets

- The author's machine: memory safety of a process that parses downloaded files.
- Integrity of training runs and checkpoints (hours of GPU time).
- The public repository: no secrets, correct attribution of downloaded data and models.

### Trust boundaries

- Downloaded files (model weights, `tokenizer.json`, `config.json`, corpora) → NeuralTitan parsers.
- Checkpoint and token files written by earlier runs → loaders. These can be truncated by a crash or a full disk.
- Python packages from PyPI → `tools/` virtual environment (development only, never linked).
- Repository → GitHub Actions.

### Top threats

| Threat | Asset or boundary | Likelihood and impact |
| --- | --- | --- |
| T1. A crafted or corrupt safetensors/JSON file triggers out-of-bounds reads or huge allocations (tampering, DoS) | Downloaded files → parsers | Medium likelihood for corrupt files, low for crafted ones; impact high (memory corruption) |
| T2. A truncated checkpoint is silently loaded, corrupting a resumed run | Checkpoint files → loader | Medium likelihood; medium impact (wasted GPU hours) |
| T3. A compromised or typo-squatted PyPI package in `tools/` | PyPI → dev venv | Low likelihood; high impact on the author's machine |
| T4. A secret committed to the public repository | Repository | Low likelihood (no secrets used); high impact |
| T5. A pickle-based checkpoint executes code when loaded by the reference tooling | Hub files → `tools/` | Low likelihood; high impact |

### Mitigations

- **T1.** The `io` readers bounds-check every header length, offset and element count against the file size before allocating or reading, and cap header sizes. Tests feed malformed and truncated fixture files and expect a clean `titan::Error`.
- **T2.** The checkpoint format carries a magic number, format version, total size and per-section checksums, all verified before loading. Checkpoints are written to a temporary file and renamed into place once complete.
  - Implemented in M3: see the [checkpoint format](#checkpoint-format). `titan train --resume` and `titan generate` read only through `runtime::load_checkpoint`, which runs every check before any tensor is used. The resume path also checks the token files against the vocabulary and the saved token counts.
- **T3.** ADR 0004 names only `torch`, `transformers` and `safetensors`. Versions are pinned by the package manager inside an isolated venv, and nothing from `tools/` is linked into the build.
- **T4.** Gitleaks runs as a pre-commit hook and in CI. GitHub secret scanning with push protection is on (public repository).
- **T5.** The tooling loads weights only from `safetensors` files, never from pickle-based `.bin` files.

## 10. Layer rules

`→` means "may include headers from and link to". Every layer may use the C++ standard library. Only `core` and `ops/cuda` include CUDA runtime headers, and only `ops/cuda` may include and link cuBLAS; other layers reach the GPU only through `core` and `ops`. The public `core` headers (`tensor.hpp`, `storage.hpp`, `device.hpp`, `error.hpp`) include no CUDA header. The one `core` header that does, `core/cuda_check.hpp` with `TITAN_CUDA_CHECK`, counts as a CUDA runtime header and is likewise limited to `core` and `ops/cuda`. `titan_core` links `CUDA::cudart` privately.

| Layer or module | May depend on | Must not depend on |
| --- | --- | --- |
| `core` | — | everything else |
| `io` | `core` | `tokenizer`, `ops`, `nn`, `optim`, `model`, `runtime`, `cli` |
| `tokenizer` | `io`, `core` | `ops`, `nn`, `optim`, `model`, `runtime`, `cli` |
| `ops/cpu` | `core` | `ops/cuda`, every layer above `ops` |
| `ops/cuda` | `core` | `ops/cpu`, every layer above `ops` |
| `ops` (dispatch) | `ops/cpu`, `ops/cuda`, `core` | every layer above `ops` |
| `nn` | `ops`, `core` | `io`, `tokenizer`, `optim`, `model`, `runtime`, `cli` |
| `optim` | `nn` (`Parameter` only), `ops`, `core` | `io`, `tokenizer`, `model`, `runtime`, `cli` |
| `model` | `nn`, `io`, `ops`, `core` | `optim`, `tokenizer`, `runtime`, `cli` |
| `runtime` | `model`, `optim`, `tokenizer`, `nn`, `io`, `ops`, `core` | `cli` |
| `cli` | `runtime` and everything below | — |
| `tests` | anything | — (never linked into the library) |
| `tools/` (Python) | nothing in the build | — (never part of the build) |

Checkpoint save and load for optimizer state lives in `runtime`, which may see both `model` and `optim`. `model` only saves and loads parameters.

Enforcement:

- Each layer is its own CMake static library whose `target_link_libraries` lists only its allowed layers, so a forbidden call fails to link.
- `scripts/check_layers.cmake`, run by the CTest entry `layers_repo`, scans every project `#include "..."` under `src/` and fails on any include that points to a forbidden layer. It also applies the system-header rule: `cuda_runtime.h`, `cuda.h`, `cuda_runtime_api.h` and `core/cuda_check.hpp` only in `core` and `ops/cuda`, and `cublas_v2.h` only in `ops/cuda`.
- Both run in the single `check` step, locally and in CI.

## 11. Risks and technical debt

| Item | Kind (risk or debt) | Mitigation or plan |
| --- | --- | --- |
| M6 ~8 h budget missed | Risk | GEMM uses cuBLAS by default (ADR 0007), so the remaining cost is in own non-GEMM kernels and materialized attention. Profile at M2; if over budget, shrink the M6 model or token count. Revisit at M2. M2 measurement: the M1 config (0.73 M parameters, 4.1 M tokens) trains in 29.6 s on an RTX 3070 Laptop GPU, about 138,000 tokens/s; re-estimate the M6 budget from that throughput when its model is fixed. |
| Materialized attention scores limit context length in 8 GB | Risk | Keep context ≤ 1024 in v1 training configs. Flash-style attention is listed as a later stage. Revisit if M6 needs longer context. |
| Logits drift from the reference beyond 1e-3 (summation order) | Risk | FP32 reference, per-layer intermediate dumps from `tools/` to localize divergence, accumulation order documented. Revisit at M5. |
| CUDA code untested in CI (no GPU runners) | Debt | CUDA tests run locally before merge, and the PR template asks for it. Revisit if a GPU runner becomes available. |
| GPT-2 preset, mixed precision, autograd graph, MoE deferred | Debt (planned) | Listed as later stages in the spec. v1 interfaces (config schema, per-block types) leave room for them. |
| Unicode tables probed from a pinned `tokenizers` version (0.23.3) | Debt | The generator refuses any other version and the generated header records it; regenerate when the reference is upgraded or if tokenizer fixtures disagree. |
