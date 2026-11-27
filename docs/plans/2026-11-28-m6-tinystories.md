# M6 TinyStories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train a ~25M-parameter Llama-style model from scratch on TinyStories, with our own byte-level BPE, on the GPU overnight, to a validation loss of about 1.5 or lower and coherent short stories.

**Architecture:**

- The model, the trainer, BPE training and checkpoint resume all exist. M6 adds only what a 2.2 GB corpus needs:
  - **Tokenization once, to a token file.** Stories are encoded in parallel, chunked right after each `<|endoftext|>`, with ids identical to encoding the whole text. `titan tokenizer encode --out` writes the result.
  - **Training from token files.** `titan train --train-tokens --val-tokens` reads them, so the official validation file is the validation split and the corpus is never re-encoded.
- **The run** is configured by `configs/tinystories.json` and executed by the controller. Its results go in README "M6 result".

**Tech Stack:** C++20, CMake presets, MSVC 14.38, CUDA 12.6 (cuBLAS GEMM), the project's own test harness. No new dependencies.

**Spec:** `docs/specs/2026-11-19-neuraltitan-design.md`: §2 (goals), §3 (8 GB VRAM, FP32), §7 M6, §8 (compute budget: overnight ≤ ~8 h, validation loss ≈ 1.5 or lower). Also `docs/product/requirements.md` (NFR on the M6 run) and `docs/architecture/overview.md` §10 (layer rules).

## Global Constraints

- **M6 done when (spec §7):** a ~10–30M parameter Llama-style model trained from scratch on TinyStories with our BPE produces coherent short stories.
- **Budget (spec §8):** completes overnight (≤ ~8 hours), reaching a validation loss of about 1.5 or lower.
- **Libraries.** No library beyond the C++ standard library, the CUDA runtime and cuBLAS.
- **Input validation.** Validate external input only in `io`, `tokenizer` and `cli`. Token files and the tokenizer file given to `train` are external input:
  - check every id against the tokenizer's vocabulary before training;
  - errors name the file, the index and the value.
- **Errors.** Throw `titan::Error` with context. Usage errors exit 2 before any file is read. Bad input exits 1 and writes nothing to the output directory.
- **Layers.** `tokenizer` may use `core` and `io`; `cli` uses everything. The layer check stays green.
- **Data.** TinyStoriesV2-GPT4 from `roneneldan/TinyStories`:
  - revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, license CDLA-Sharing-1.0;
  - `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`, in git-ignored `data/tinystories/`.

  Never commit data, token files or runs.
- **Change rules:**
  - Tests go in the same change as the feature, and docs in the same change as the code.
  - Conventional Commits, with no AI attribution anywhere.
  - `scripts/check.ps1 -Preset release` must print `check: OK` before each commit.
  - `CMakeLists.txt` files are edited through the shell (a hook blocks the Edit and Write tools on them).

## Decisions (approved by the author)

1. Branch `feat/m6-tinystories` from `main`.
2. TinyStoriesV2-GPT4 train (~2.2 GB) and valid (~22 MB), downloaded by the controller at the pinned revision. Their SHA-256 is recorded in README. The official valid file is the validation split.
3. Our BPE with 4096 tokens, trained on the full train file. Fallback: the first ~500 MB, cut after a `<|endoftext|>`, recorded as a ruling.
4. A token-file pipeline: `tokenizer encode --out` (parallel, chunked at `<|endoftext|>`, ids identical to a whole-text encode) and `train --train-tokens/--val-tokens`.
5. The model: dim 512, 8 layers, 8 heads and 4 KV heads, SwiGLU hidden 1376, tied embeddings, context 512. That is 25,305,600 parameters with a 4096-token vocabulary.
6. The schedule:
   - batch 8, seq 512, grad_accum 4: 16,384 tokens per step;
   - 24,000 steps, about 393M tokens;
   - lr 1e-3 to min_lr 1e-4, cosine, warmup 1000;
   - eval and checkpoint every 1000 steps.
7. The controller runs the training in the background. It is resumable with `--resume`.
8. README "M6 result":
   - final validation loss;
   - loss-by-step table, throughput and wall time;
   - five samples at fixed seeds.

   The author judges coherence.
9. No performance work. The CPU thread pool stays deferred.

## Review Focus

1. **A corpus with CRLF line endings, or one whose last story has no trailing `<|endoftext|>`.** Chunked encoding still equals `tok.encode(text)`. Pinned in Task 1 (`EncodeChunked.MatchesWholeEncode` covers both).
2. **A token id at or above the tokenizer's vocabulary** in `--train-tokens` or `--val-tokens`. Exit 1, the message names the file, the index and the id, and no output directory is created. Pinned in Task 2 (`cli_train_tokens_out_of_vocab`).
3. **Flag combinations that mix the two data sources:**
   - `--train-tokens` without `--val-tokens`, or the reverse;
   - `--train-tokens` together with `--data`;
   - `--train-tokens` without `--tokenizer`.

   Each exits 2 with the usage text. Pinned in Task 2 (`cli_train_tokens_usage_*`).
4. **A token file shorter than seq + 2.** It gets a named error before anything is written, as with `--data`. Pinned in Task 2 (`cli_train_tokens_too_short`).
5. **Resuming a run started from token files.** `train --resume <dir>` continues it bit-identically, like a `--data` run. Pinned in Task 2 (`cli_train_tokens_resume`).

---

### Task 1: Parallel, story-chunked encoding and `tokenizer encode --out`

**Files:**

- Create: `src/tokenizer/encode_chunked.hpp`, `src/tokenizer/encode_chunked.cpp`
- Modify: `src/tokenizer/CMakeLists.txt` (shell), `apps/titan/commands.cpp` (`tokenizer_encode`), `apps/titan/main.cpp` (usage text)
- Test: `tests/tokenizer/encode_chunked_test.cpp`, CLI tests in `tests/CMakeLists.txt` (shell)
- Docs: `docs/math/bpe.md` (new section "Encoding a corpus in chunks"), README (encode command), AGENTS.md (Commands, Layout)

**Interfaces:**

- Consumes: `tokenizer::Tokenizer::encode`, `Tokenizer::eos_id()`, `titan::parallel_for` and `titan::num_threads()` (`core/parallel.hpp`), `io::write_token_file`, `runtime::token_crc32` (already used by `tokenizer_encode`).
- Produces:
  - `std::vector<std::int32_t> tokenizer::encode_chunked(const Tokenizer& tok, std::string_view text)`, in `tokenizer/encode_chunked.hpp`.
  - `titan tokenizer encode --tokenizer <tok.json> (--text <s> | --file <path>) [--summary] [--out <tokens.bin>]`.

- [ ] **Step 1: Write the failing tests** in `tests/tokenizer/encode_chunked_test.cpp`.
  - `EncodeChunked.MatchesWholeEncode`: for each tokenizer, `encode_chunked(tok, text) == tok.encode(text)`. The tokenizers are the trained fixture `tests/fixtures/bpe/trained.json` and SmolLM2's `tests/fixtures/smollm2/tokenizer.json`. The texts:
    - `""`;
    - `"<|endoftext|>"`;
    - `"a<|endoftext|>"`;
    - `"<|endoftext|><|endoftext|>b"`;
    - the BPE fixture corpus with `"\n<|endoftext|>\n"` inserted after every 7th line, with and without a trailing marker;
    - the same corpus with every `"\n"` replaced by `"\r\n"`;
    - 50 seeded `random_text` strings joined by `"<|endoftext|>"`.
  - `EncodeChunked.ThreadCountDoesNotMatter`: on the marked corpus, the result with `set_num_threads(1)` equals the result with `set_num_threads(8)`. Restore the previous count afterwards.
  - `EncodeChunked.CharTokenizerEncodesWhole`: a `CharTokenizer` (no `eos_id`) gives `tok.encode(text)`.
  - `EncodeChunked.RethrowsEncodeErrors`: a `CharTokenizer` built from `"ab"` on `"ab<|endoftext|>c"` throws `titan::Error`, the same error `encode` throws.
- [ ] **Step 2: Run them and see them fail.** Run `ctest --preset release -R unit`. Expected: they fail to build (`encode_chunked` is undefined).
- [ ] **Step 3: Implement `encode_chunked`.**
  - When `tok.eos_id()` is empty, return `tok.encode(text)`.
  - Otherwise, the chunk boundaries are the positions right after each occurrence of `"<|endoftext|>"`, plus the end of the text.
    - Group consecutive chunks into `num_threads()` ranges of roughly equal byte length.
    - Encode each range with `tok.encode` inside `parallel_for`, into one vector per range.
    - Concatenate the vectors in order.
  - Comment the reason the result is identical: `encode` matches added tokens before pre-tokenizing, so no piece spans a marker. Note the assumption that no added token contains `<|endoftext|>` past its first byte.
  - Add the file to `titan_tokenizer`.
- [ ] **Step 4: Run them and see them pass.** Same command. Expected: PASS.
- [ ] **Step 5: Add `--out <path>` to `tokenizer encode`.**
  - It encodes with `encode_chunked` and creates the parent directory.
  - It writes the ids with `io::write_token_file` and prints the three summary lines (`tokens`, `crc32`, `bytes_per_token`), never the ids.
  - Without `--out`, use `encode_chunked` too, so output is unchanged and faster.
  - `--summary` with `--out` is accepted and changes nothing.
  - Update the usage text in `main.cpp`.
- [ ] **Step 6: Add CLI tests** in `tests/CMakeLists.txt`:
  - `cli_tokenizer_encode_out`: encode the BPE fixture corpus with `trained.json` to a token file under the test's binary dir. stdout matches `tokens [0-9]+`. A following `tokenizer encode --summary` on the same corpus prints the same crc32. Use a small `expect_*.cmake` script if one command cannot check this; follow the existing scripts.
  - `cli_tokenizer_encode_out_text`: `--text "hi" --out <file>` succeeds.
- [ ] **Step 7: Docs.**
  - `docs/math/bpe.md`, section "Encoding a corpus in chunks": why chunking at an added token preserves the ids, the thread grouping, and the assumption.
  - README: the encode command gets `--out`.
  - AGENTS.md: Commands (the encode line) and Layout (`encode_chunked.*` in `src/tokenizer/`).
- [ ] **Step 8: Verify and commit.** Run `scripts/check.ps1 -Preset release`. Expected: `check: OK`. Commit `feat(tokenizer): encode corpora in parallel chunks to token files`.

### Task 2: `titan train --train-tokens --val-tokens`

**Files:**

- Modify: `apps/titan/commands.cpp` (`train_new`), `apps/titan/main.cpp` (usage)
- Test: CLI tests in `tests/CMakeLists.txt` (shell), with new fixtures under `tests/fixtures/cli/` if needed (hand-written or produced by `tests/tools/edit_token_file`)
- Docs: README (train section), AGENTS.md (Commands)

**Interfaces:**

- Consumes: `io::read_token_file`, `read_tokenizer`, the existing `read_run_tokens` pattern (ids checked against the vocabulary), `runtime::Trainer`, `run_training`, and the `edit_token_file` test tool.
- Produces: `titan train --config <cfg.json> (--data <corpus.txt> | --train-tokens <a.bin> --val-tokens <b.bin>) --out <dir> [--tokenizer <tok.json>] [--steps N] [--stop-at N] [--device cpu|cuda] [--gemm ...]`.

- [ ] **Step 1: Write the failing CLI tests** in `tests/CMakeLists.txt`. Prepare the token files at test time with `titan tokenizer encode --out`, from the BPE fixture corpus and `trained.json`, using CTest fixtures (`FIXTURES_SETUP`/`FIXTURES_REQUIRED`).
  - `cli_train_tokens_smoke`: train with `configs/tiny-shakespeare.json`, `--steps 6`, `--device cpu`, `--tokenizer trained.json`, and the two token files. It exits 0, and the output directory holds `latest.ckpt`, `train.bin`, `val.bin` and `sample.txt`. Its `train.bin` is byte-identical to the input (`cmake -E compare_files`).
  - `cli_train_tokens_resume`: the same run with `--steps 6 --stop-at 3`, then `--resume`. The final `latest.ckpt` is byte-identical to the uninterrupted run's.
  - `cli_train_tokens_out_of_vocab`: the train token file with one id set to 512 (`edit_token_file`). It exits 1, the message matches `train.*token [0-9]+ is 512, outside \[0, 512\)`, and the output directory does not exist afterwards.
  - `cli_train_tokens_too_short`: a train token file of 3 tokens. It exits 1 with the "too few" message, and nothing is written.
  - `cli_train_tokens_usage_missing_val`, `cli_train_tokens_usage_with_data` and `cli_train_tokens_usage_no_tokenizer`: each exits 2.
- [ ] **Step 2: Run them and see them fail.** Run `ctest --preset release -R cli_train_tokens`. Expected: FAIL (unknown flag, exit 2 where 0 or 1 is expected).
- [ ] **Step 3: Implement it in `train_new`.**
  - Usage:
    - exactly one of `--data` or the pair `--train-tokens`/`--val-tokens`;
    - the pair needs `--tokenizer`;
    - `--resume` still rejects all of them.
  - Read both token files and check every id against `tok->vocab_size()`, with the same message format as `read_run_tokens`. Factor the check into one helper used by both.
  - Check each stream has at least `seq + 2` tokens, with the same message style as the `--data` split.
  - Then build the model and `Trainer`, moving the token vectors into it (no copy).
  - Copy the input files to `<out>/train.bin` and `<out>/val.bin` with `std::filesystem::copy_file` (overwrite existing), so `--resume` is unchanged.
  - Keep "everything that can fail on bad input runs before anything is written". The `--data` path is unchanged.
- [ ] **Step 4: Run them and see them pass.** Same command. Expected: PASS. All existing `cli_train_*` tests still pass.
- [ ] **Step 5: Docs.**
  - README train section: the new flags, and that `val_fraction` is ignored with token files.
  - AGENTS.md: the Commands line for `titan train`.
- [ ] **Step 6: Verify and commit.** Run `scripts/check.ps1 -Preset release`. Expected: `check: OK`. Commit `feat(cli): train from token files`.

### Task 3: The TinyStories config

**Files:**

- Create: `configs/tinystories.json`
- Test: `tests/model/config_test.cpp` (or wherever `parse_model_config` tests live), a new test
- Docs: README (an "M6: TinyStories" how-to section, with the download URLs at the pinned revision and the commands), AGENTS.md (Layout: `configs/`)

**Interfaces:**

- Consumes: `model::parse_model_config(json, vocab_size)`, `runtime::parse_train_config`, `model::Model::num_parameters()`.
- Produces: `configs/tinystories.json`.

- [ ] **Step 1: Write the failing test** `Configs.TinyStoriesParses`. It parses `configs/tinystories.json` and checks:
  - `parse_model_config(cfg.at("model"), 4096)` builds a model with exactly 25,305,600 parameters;
  - `parse_train_config(cfg.at("train"))` succeeds, and `batch * seq * grad_accum == 16384`, `steps == 24000`, `warmup == 1000`, `eval_interval == 1000` and `checkpoint_interval == 1000`.

  Locate the file through the `TITAN_SOURCE_DIR` compile definition.
- [ ] **Step 2: Run it and see it fail** (the file is missing).
- [ ] **Step 3: Write `configs/tinystories.json`.**
  - `model`: dim 512, context_length 512, n_layers 8; block with n_heads 8, n_kv_heads 4, mlp swiglu, mlp_hidden 1376, norm rmsnorm, position rope; rope_theta 10000, norm_eps 1e-5, tied_embeddings true.
  - `train`: batch 8, seq 512, grad_accum 4, steps 24000, lr 1e-3, min_lr 1e-4, warmup 1000, weight_decay 0.1, beta1 0.9, beta2 0.95, eps 1e-8, grad_clip 1.0, eval_interval 1000, eval_batches 200, log_interval 100, checkpoint_interval 1000, seed 1337.
- [ ] **Step 4: Run it and see it pass.**
- [ ] **Step 5: Docs.**
  - README "Training on TinyStories": download the two files at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` into `data/tinystories/`, then three commands: `tokenizer train --type bpe --vocab 4096`, `tokenizer encode --out` for train and valid, and `train --config configs/tinystories.json --train-tokens ... --val-tokens ... --tokenizer ... --out runs/tinystories`.
  - Say that the run checkpoints every 1000 steps and continues with `--resume`.
- [ ] **Step 6: Verify and commit.** Run `scripts/check.ps1 -Preset release`. Expected: `check: OK`. Commit `feat(configs): add the tinystories training config`.

### Task 4 (controller): The run and README "M6 result"

Not dispatched to an implementer: it is long-running and operational.

- [ ] Record the SHA-256 and the sizes of the two data files.
- [ ] Train the tokenizer on the full train file and record time and peak RAM. Use the fallback from decision 3 if it exceeds ~30 min or ~20 GB.
- [ ] Encode train and valid with `--out`, and record tokens, bytes per token and time.
- [ ] Run `train` in the background on CUDA and follow the logs. If it is interrupted, resume with `--resume`. If it diverges, stop and rule.
- [ ] Generate five samples: `titan generate --checkpoint runs/tinystories/latest.ckpt --prompt "Once upon a time" --seed S` for S in 1–5, plus the default settings.
- [ ] README "M6 result":
  - the data and tokenizer facts;
  - a loss table every 2000 steps (train and val);
  - the final validation loss against the 1.5 target, wall time and tok/s;
  - the five samples.

  Update `docs/product/requirements.md` if it tracks milestone status. Run `scripts/check.ps1` and markdownlint, then commit `docs: add the m6 result`.
