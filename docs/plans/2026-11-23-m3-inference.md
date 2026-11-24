# M3 Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `titan generate` streams text from a checkpoint written by `titan train`, using a KV cache (prefill, then one-token decode steps) and a seeded sampler with temperature, top-k and top-p. Training writes checkpoints periodically and resumes from one with bit-identical results.

**Architecture:**

- `io` gains the checkpoint container: a JSON header plus named FP32 tensors, with a magic, a version, the total size and CRC-32 checksums, written atomically. It knows nothing about models.
- `runtime` builds on it: `save_checkpoint` and `load_checkpoint` map a model, its AdamW state and the trainer's progress onto the container. `Trainer` gains resume and a checkpoint callback.
- `nn` gains `KvCache`, carried in `ForwardContext`. `Attention` in inference mode writes the new keys and values into the cache and attends over the cached prefix. Two new layout ops (`kv_write`, `kv_read`) move the data, each with a CPU and a CUDA implementation.
- `runtime::Generator` runs prefill and decode, and calls a callback for each new token. `runtime::sample` implements temperature, top-k and top-p. `generate_naive` stays as the reference that the cached path is tested against.

**Tech Stack:** C++20, CUDA 12.6 for `sm_86`, cuBLAS (ADR 0007), MSVC 14.38, CMake presets, the project's own test harness. No new dependencies.

**Spec:** `docs/specs/2026-11-19-neuraltitan-design.md`: §5.3 (inference mode and the KV cache in `ForwardContext`), §5.4 (own checkpoint format, with save, load and resume including the optimizer state), §5.6 (`Trainer` periodic checkpoint, `Generator`, `Sampler`), §5.7 (`titan generate`), §7 M3. Also `docs/architecture/overview.md` §4 (generation sequence), §9 (threat T2 and its mitigation) and §10 (layer rules).

## Global Constraints

- **FP32 and seeds.** FP32 everywhere. A single seed drives initialization, data order and sampling.
- **Ops.** Every new op has a CPU reference and a CUDA implementation behind one dispatching function in `ops`. Each gets a parity test: a single element, a shape that is not a multiple of 16 or 32, and one tensor of at least 2²⁰ elements. Default tolerances: `atol = 1e-5`, `rtol = 1e-4`. Kernels do not validate, use no floating-point atomics, and follow every launch with `TITAN_CUDA_CHECK(cudaGetLastError())`.
- **Input validation.** Validate external input only at the `io`, `tokenizer` and `cli` boundaries and in the config parsers. Bounds-check every length and offset against the file size before allocating or reading (overview §9, T1). Every parser gets malformed-input tests.
- **Checkpoint integrity (overview §9, T2).** The format carries a magic number, a format version, the total size and checksums, all verified before any tensor is used. A writer writes `<path>.tmp` and renames it over `<path>` only when the file is complete.
- **Inference mode caches nothing for backward** (spec §5.3). `ForwardContext::training == false` with a KV cache is the only path that reads or writes the cache. Training with a cache throws.
- **Errors.** Throw a `titan::Error` with context: the file, the tensor or field name, the shapes.
- **Math notes.** Every new module or op family ships its note in `docs/math/`: what it computes, forward, backward (or "none, inference only"), shapes and mapping to code. Docs are additive: extend existing notes, never rewrite them.
- **Change rules.**
  - Edit `CMakeLists.txt` files through the shell, not the Edit/Write tools.
  - Conventional Commits, with no AI attribution, in English.
  - Work on branch `feat/m3-inference`, from `main`.
  - `scripts/check.ps1` prints `check: OK` after every task. CUDA tests skip without a GPU and must pass locally.
  - Prove each new test catches a deliberate break before committing.

## Review Focus

1. **A truncated or corrupt checkpoint** fails with a clean `titan::Error` naming the file and what is wrong, never a crash or a silently wrong model. This covers a cut-off file, a flipped byte, a wrong magic or version, a tensor out of bounds, and a tensor missing or of the wrong shape for the model. Pinned in Task 2 (`Checkpoint.Rejects*`) and Task 3 (`RuntimeCheckpoint.RejectsShapeMismatch`).
2. **A crash in the middle of saving** leaves the previous `latest.ckpt` intact, and no `.tmp` file is left behind by a successful save. Pinned in Task 2 (`Checkpoint.AtomicReplaceKeepsOldOnFailure`).
3. **Generating past the context length** keeps every position below `context_length` and does not crash. The tokens before the first slide equal the uncached reference. Pinned in Task 7 (`Generator.SlidesPastContext`).
4. **A prompt with a character outside the vocabulary**, or a prompt longer than the context, behaves as follows:
   - an unknown character fails before any output, naming the character;
   - a long prompt keeps its last `context_length` tokens, as `generate_naive` does.

   Pinned in Task 7 (`Generator.LongPromptKeepsTail`) and Task 8 (`cli_generate_unknown_char`).
5. **Resuming** reproduces the uninterrupted run bit for bit on the same device. Resuming with token files that changed since the save fails before training. Pinned in Task 3 (`Trainer.ResumeIsBitIdentical`, `Trainer.ResumeRejectsChangedData`).

## Decisions this plan makes (correctable by the author)

- **Self-contained checkpoints.** A checkpoint holds the model config, the train config, the tokenizer JSON, the parameters, the AdamW moments and step, the step reached, both loader RNG states, and the train and validation token counts. `titan generate --checkpoint` needs no other file.
- **Context overflow: slide by half.** When the cache is full during decode, the generator keeps the last `context_length / 2` tokens, clears the cache and prefills them again at positions `0 …`. It then continues. The cost is amortized; the trade-off is that after a slide the model sees a shorter window than `generate_naive` does.
- **Resume interface.** `titan train --resume <run dir>` reads `<run dir>/latest.ckpt`, `train.bin` and `val.bin`, and runs to the checkpoint's `train.steps`. The config and tokenizer come from the checkpoint. `--config`, `--data`, `--tokenizer` and `--steps` are rejected with `--resume`.
- **RNG seeding unchanged.** The init, train-loader, validation-loader and sample RNGs keep their M1 seeds. The M1 and M2 results stay reproducible. Checkpoints save the two loader states. Generation uses its own `--seed`.
- **`titan generate` defaults.** These are `--temp 0.8 --top-k 0 --top-p 0.9 --max-new 500`, with `--seed` defaulting to the checkpoint's `train.seed`, and `--device` defaulting to cuda when a GPU is present. `--temp 0` is greedy and ignores top-k and top-p. `--no-cache` uses `generate_naive`, for comparison.
- **Sampler order.** The sampler applies temperature, then top-k, then top-p (the smallest set of tokens whose probability reaches `p`, always at least one), then renormalizes and draws once with `rng.uniform()`. With `top_k = 0` and `top_p = 1` it consumes the RNG exactly like M1's `sample_token`, so `titan train`'s sample is unchanged.
- **Checkpoint cadence.** `train.checkpoint_interval` is an optional config field, defaulting to 0, which means a checkpoint only after the last step. The trainer overwrites `<out>/latest.ckpt` at each checkpoint. There is no checkpoint history.
- **No speed target.** The cached and uncached decode speeds are measured and reported on both devices. Correctness (cached logits equal full-recompute logits) is the gate.

## File structure

| Area | Files |
| --- | --- |
| core | `src/core/rng.hpp/.cpp` (`save_state` and `load_state`) |
| io | `src/io/checksum.hpp/.cpp` (CRC-32); `src/io/checkpoint_file.hpp/.cpp` (the container) |
| ops | `src/ops/layout.hpp/.cpp`, `src/ops/cpu/layout.*`, `src/ops/cuda/layout.*` (`kv_write`, `kv_read`) |
| nn | `src/nn/kv_cache.hpp/.cpp`; `src/nn/module.hpp` (`ForwardContext`); `src/nn/attention.*` |
| model | `src/model/config.*` (`model_config_to_json`); `src/model/model.*` (sets the layer in `ForwardContext`) |
| optim | `src/optim/adamw.*` (`AdamWState`, `state`, `load_state`) |
| runtime | `src/runtime/checkpoint.*`; `src/runtime/trainer.*` (resume, callback, `checkpoint_interval`, `train_config_to_json`); `src/runtime/sampler.*` (`SamplerConfig`, `sample`); `src/runtime/generator.*` |
| cli | `apps/titan/commands.*`, `apps/titan/main.cpp` |
| tests | `tests/io/checkpoint_file_test.cpp`; `tests/runtime/checkpoint_test.cpp`; `tests/ops/cuda/layout_parity_test.cpp`; `tests/nn/kv_cache_test.cpp`; `tests/runtime/sampler_test.cpp`; `tests/runtime/generator_test.cpp`; `tests/CMakeLists.txt` |
| docs | `docs/math/kv-cache.md` (new), `docs/math/sampling.md` (new), the overview's checkpoint-format section, README "M3 result", AGENTS.md |

---

### Task 1: Serializable state: RNG, configs and AdamW

**Files:**

- `src/core/rng.*`
- `src/model/config.*`
- `src/runtime/trainer.*` (the config writer and `checkpoint_interval` only)
- `src/optim/adamw.*`
- `tests/core/rng_test.cpp`, `tests/model/config_test.cpp`, `tests/runtime/train_config_test.cpp`, `tests/optim/adamw_test.cpp`
- `docs/math/adamw.md`: a short "State" section

**Interfaces:**

- Produces:
  - `std::string Rng::save_state() const`: the `mt19937_64` text state (`operator<<`), plus the cached normal and its flag.
  - `void Rng::load_state(std::string_view)`: throws `titan::Error` "Rng::load_state: malformed state" on any parse failure.
  - `io::JsonValue model::model_config_to_json(const ModelConfig&)`.
  - `io::JsonValue runtime::train_config_to_json(const TrainConfig&)`. It omits `device`, which is not part of the file format.
  - `TrainConfig::checkpoint_interval` (`std::int64_t`, default 0), parsed as optional and `>= 0`.
  - `struct optim::AdamWState { std::int64_t steps; std::vector<Tensor> m, v; }`. The tensors are on the CPU, in `params` order.
  - `AdamWState AdamW::state() const`, which copies to host.
  - `void AdamW::load_state(const AdamWState&)`. It copies to the parameters' device and throws on a count or shape mismatch, naming the parameter.

- [ ] **Step 1:** Write the tests:
  - `Rng.SaveLoadContinuesSameStream`: draw 3 normals (leaving one cached), save, draw 5 values, load into a fresh `Rng(0)`, draw 5 values. The two lists are equal bit for bit.
  - `Rng.LoadRejectsGarbage`: `""` and `"abc"` throw.
  - `ModelConfig.JsonRoundTrip`: `parse_model_config(model_config_to_json(c)) == c` for the TinyShakespeare config and for a heterogeneous two-block config.
  - `TrainConfig.JsonRoundTrip`.
  - `TrainConfig.CheckpointIntervalOptionalAndNonNegative`: absent gives 0, and -1 throws naming `train.checkpoint_interval`.
  - `AdamW.StateRoundTripContinuesIdentically`: 3 steps, save the state, then 2 more steps, giving parameters P. Separately, rebuild the same parameters, load the state, take 2 steps. The parameters equal P bit for bit.
  - `AdamW.LoadStateRejectsShapeMismatch`.
- [ ] **Step 2:** Run them. Expected: FAIL to compile.
- [ ] **Step 3:** Implement. `ModelConfig` and `TrainConfig` get `operator==` (defaulted) if they lack one.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Commit with `feat(runtime): serialize rng, config and optimizer state`.

### Task 2: Checkpoint container in `io`

**Files:**

- `src/io/checksum.hpp/.cpp`
- `src/io/checkpoint_file.hpp/.cpp`
- `tests/io/checkpoint_file_test.cpp`
- `docs/architecture/overview.md`: a "Checkpoint format" subsection under §6

**Interfaces:**

- Produces:
  - `std::uint32_t io::crc32(std::span<const std::byte>, std::uint32_t seed = 0)`: CRC-32/ISO-HDLC, the zlib polynomial `0xEDB88320`, reflected.
  - `struct io::NamedTensor { std::string name; Tensor value; }`, with values on the CPU.
  - `void io::write_checkpoint_file(const std::filesystem::path&, const JsonValue& meta, std::span<const NamedTensor>)`.
  - `struct io::CheckpointFile { JsonValue meta; std::vector<NamedTensor> tensors; }`.
  - `io::CheckpointFile io::read_checkpoint_file(const std::filesystem::path&)`.
- The layout, all integers little-endian:

  ```text
  bytes 0..3    "TCKP"
  bytes 4..7    u32 version = 1
  bytes 8..15   u64 total file size
  bytes 16..23  u64 header length H (1 ≤ H ≤ 16 MiB)
  bytes 24..27  u32 CRC-32 of the header bytes
  bytes 28..    H bytes of UTF-8 JSON:
                  {"meta": {...}, "tensors": [{"name", "shape": [..], "offset", "crc32"}, ...]}
  then          zero padding to a multiple of 64, which starts the data section
  data          each tensor's float32 values at its offset (relative to the data
                section, a multiple of 64), in the order listed
  ```

  The writer pads every tensor to 64 bytes. The file ends exactly at the end of the last tensor's padding.

- [ ] **Step 1:** Write the tests:
  - `Crc32.KnownVector`: `crc32("123456789") == 0xCBF43926`.
  - `Checkpoint.RoundTrip`: meta plus three tensors (shapes `{1}`, `{3,5}`, `{2,3,4,5}`) come back equal, in order.
  - `Checkpoint.AtomicReplaceKeepsOldOnFailure`: write A, then attempt to write B with a tensor on CUDA (or another induced failure). A is still readable, and no `.tmp` remains after a successful write.
  - Rejection tests, each with the message they assert:
    - `Checkpoint.RejectsTruncated`, for every cut point in a sample of 8 across the file;
    - `Checkpoint.RejectsWrongMagic`;
    - `Checkpoint.RejectsWrongVersion`;
    - `Checkpoint.RejectsSizeMismatch`;
    - `Checkpoint.RejectsHeaderTooLarge`;
    - `Checkpoint.RejectsHeaderChecksum`, for a flipped header byte;
    - `Checkpoint.RejectsTensorChecksum`, for a flipped data byte, naming the tensor;
    - `Checkpoint.RejectsTensorOutOfBounds`;
    - `Checkpoint.RejectsDuplicateName`;
    - `Checkpoint.RejectsBadShape`, for 0 or negative dimensions, a rank above 4, or an element count whose bytes overflow.
- [ ] **Step 2:** Run them. Expected: FAIL to compile.
- [ ] **Step 3:** Implement. The reader checks the magic, version, total size, header length and header CRC before parsing JSON. It bounds-checks every entry before allocating. It streams each tensor from the file into its host tensor (never the whole file into one string), and verifies each tensor's CRC.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Document the format in the overview, then commit with `feat(io): add checkpoint file format`.

### Task 3: Model checkpoints, periodic save and resume

**Files:**

- `src/runtime/checkpoint.hpp/.cpp`
- `src/runtime/trainer.hpp/.cpp`
- `tests/runtime/checkpoint_test.cpp`
- `tests/runtime/trainer_test.cpp`

**Interfaces:**

- Consumes: Task 1's state functions and Task 2's container.
- Produces:
  - `struct runtime::TrainerState { std::int64_t step; optim::AdamWState optimizer; std::string train_rng, val_rng; std::int64_t train_tokens, val_tokens; }`.
  - `struct runtime::Checkpoint { model::ModelConfig model; TrainConfig train; io::JsonValue tokenizer; TrainerState state; }`.
  - `void runtime::save_checkpoint(const std::filesystem::path&, model::Model&, const Checkpoint&)`. Tensor names: each parameter's name, then `optim.m.<name>` and `optim.v.<name>`.
  - `runtime::Checkpoint runtime::load_checkpoint(const std::filesystem::path&, model::Model* model)`. With a model, it copies every parameter onto the model's device and throws on a missing name, an extra name or a shape mismatch, naming the tensor and both shapes. With `nullptr`, it reads only the header, so the caller can build the model from `Checkpoint::model` first.
  - `Trainer` changes:
    - `TrainerState Trainer::state() const`;
    - a constructor overload taking `std::optional<TrainerState> resume`, which loads the optimizer and RNG states and starts `run()` at `resume->step + 1`. It throws "resume: train tokens changed (checkpoint N, now M)" (and the same for val) on a count mismatch;
    - `void Trainer::on_checkpoint(std::function<void(std::int64_t step)>)`, called every `checkpoint_interval` steps (when > 0) and after the last step.
  - `DataLoader::rng()` accessors: `const Rng&` and `Rng&`.

- [ ] **Step 1:** Write the tests:
  - `RuntimeCheckpoint.RoundTripRestoresModel`: logits after a load equal logits before the save, bit for bit.
  - `RuntimeCheckpoint.CudaSaveCpuLoad`: skips without a GPU. Save on CUDA, load on CPU; logits match at the parity tolerance.
  - `RuntimeCheckpoint.RejectsShapeMismatch`: a dim-64 checkpoint loaded into a dim-32 model.
  - `RuntimeCheckpoint.RejectsMissingTensor`.
  - `Trainer.ResumeIsBitIdentical`: CPU, and CUDA when available. Run A trains 12 steps. Run B trains 6 steps, checkpoints through `on_checkpoint`, loads into a fresh model and trainer, and trains to 12. Every parameter of A equals B bit for bit, and so do the final losses.
  - `Trainer.ResumeRejectsChangedData`.
  - `Trainer.CheckpointCallbackCadence`: steps 10, interval 4 gives the calls 4, 8, 10.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Commit with `feat(runtime): save, load and resume checkpoints`.

### Task 4: KV-cache layout ops

**Files:**

- `src/ops/layout.hpp/.cpp`
- `src/ops/cpu/layout.*`
- `src/ops/cuda/layout.*`
- `tests/ops/layout_test.cpp`
- `tests/ops/cuda/layout_parity_test.cpp`
- `tests/ops/cuda/rope_parity_test.cpp`
- `docs/math/attention.md`: the layout-ops subsection

**Interfaces:**

- Produces:
  - `void ops::kv_write(Tensor& cache, const Tensor& x, std::int64_t pos)`. `cache` is `(B, H, Tmax, hd)` and `x` is `(B, H, T, hd)`. It sets `cache[:, :, pos:pos+T, :] = x`. It throws unless the B, H and hd dimensions match, `pos >= 0` and `pos + T <= Tmax`.
  - `Tensor ops::kv_read(const Tensor& cache, std::int64_t len)`. It returns a new `(B, H, len, hd)` tensor holding `cache[:, :, 0:len, :]`, and throws unless `1 <= len <= Tmax`.

- [ ] **Step 1:** Write the tests:
  - `KvOps.WriteThenReadIsIdentity`;
  - `KvOps.WriteLeavesOtherRowsUntouched`;
  - `KvOps.RejectsOverflow`, for `pos + T > Tmax` and for a negative `pos`;
  - `KvOpsParity.Write` and `KvOpsParity.Read`, with the three Global Constraints shapes;
  - `RopeParity.LargePosOffset`: `pos_offset = 4096`, closing the deferred M2 finding.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. The CUDA kernels are grid-stride copies over the output elements.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Commit with `feat(ops): add kv cache write and read`.

### Task 5: KV cache in attention and the model

**Files:**

- `src/nn/kv_cache.hpp/.cpp`
- `src/nn/module.hpp`
- `src/nn/attention.*`
- `src/model/model.*`
- `tests/nn/kv_cache_test.cpp`
- `docs/math/kv-cache.md` (new)

**Interfaces:**

- Consumes: Task 4's `kv_write` and `kv_read`.
- Produces:
  - `class nn::KvCache`:
    - constructor `KvCache(std::int64_t n_layers, std::int64_t batch, std::int64_t n_kv_heads, std::int64_t max_len, std::int64_t head_dim, Device)`, with per-layer K and V tensors `(batch, n_kv_heads, max_len, head_dim)`, zero-filled;
    - `Tensor& keys(std::int64_t layer)` and `Tensor& values(std::int64_t layer)`;
    - `std::int64_t length() const`: positions filled so far;
    - `void set_length(std::int64_t)` and `void clear()`;
    - `std::int64_t max_len() const`.
  - `ForwardContext` gains `KvCache* kv_cache = nullptr;` and `std::int64_t layer = 0;`.
  - `model::Model::forward` sets `ctx.layer = i` before block `i`.
  - `nn::KvCache model::Model::make_kv_cache(std::int64_t batch) const`. It uses the model's device and `context_length`; every block must have the same `n_kv_heads` and head size, or it throws.
  - The caller advances the cache length after a forward: `ctx.pos_offset` must equal `cache.length()`, and after the forward the caller calls `set_length(pos_offset + T)`.
- Attention with a cache, in inference mode:
  1. RoPE on Q and K at `pos_offset`.
  2. `kv_write` the un-repeated K and V at `pos_offset`.
  3. `kv_read` `pos_offset + T` rows.
  4. `repeat_kv`.
  5. Scores `(T, pos_offset + T)`, then `causal_softmax(S, pos_offset)`.
  6. The rest as before.

  Without a cache, `pos_offset > 0` still throws. A cache with `training == true` throws "attention: the KV cache is inference-only".

- [ ] **Step 1:** Write the tests, using the model config from the M1 model tests, seeded:
  - `KvCache.PrefillMatchesFullForward`: the cached prefill of 10 tokens gives the same logits as the uncached forward, at `atol 1e-5`.
  - `KvCache.DecodeMatchesFullForward`: prefill 6 tokens, then decode 4 one at a time. The logits of each decode step equal the last row of an uncached forward over the whole prefix, at `atol 1e-5`.
  - `KvCache.CudaDecodeMatchesCpu`: skips without a GPU.
  - `KvCache.RejectsTrainingMode`.
  - `KvCache.RejectsOffsetMismatch`: `pos_offset != length()`.
  - `Attention.PosOffsetWithoutCacheStillThrows`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write `kv-cache.md`:
  - why causality makes the past keys and values reusable;
  - prefill against decode;
  - the cost per token, `O(T)` instead of `O(T²)` for the scores;
  - the memory, `2 · n_layers · n_kv_heads · max_len · head_dim · 4` bytes, with the TinyShakespeare value;
  - the layout and the GQA saving.

  Then commit with `feat(nn): add kv cache to attention`.

### Task 6: Samplers

**Files:**

- `src/runtime/sampler.hpp/.cpp`
- `tests/runtime/sampler_test.cpp`
- `docs/math/sampling.md` (new)

**Interfaces:**

- Produces:
  - `struct runtime::SamplerConfig { float temperature = 1.0f; std::int64_t top_k = 0; float top_p = 1.0f; }`.
  - `void runtime::validate(const SamplerConfig&)`: it throws naming the field for a temperature that is negative or non-finite, `top_k < 0`, or `top_p` outside `(0, 1]`.
  - `std::int32_t runtime::sample(std::span<const float> logits, const SamplerConfig&, Rng&)`.
  - `sample_token` keeps its signature and becomes `sample(logits, {temperature, 0, 1.0f}, rng)`.
- Algorithm:
  - `temperature == 0`: argmax, with the lowest index among ties. It draws nothing from the RNG.
  - Otherwise:
    1. Compute `p = softmax(logits / temperature)` in double.
    2. If `top_k > 0`, keep the `top_k` largest. Ties at the cutoff keep the lower index.
    3. If `top_p < 1`, sort the kept tokens by probability descending (index ascending on ties) and keep the shortest prefix whose sum is `≥ top_p`.
    4. Renormalize, then pick by inverse CDF with exactly one `rng.uniform()`, walking the kept tokens in index order.

- [ ] **Step 1:** Write the tests:
  - `Sampler.GreedyIgnoresRng`: two calls with differently seeded RNGs give the same argmax, and the RNG state is untouched.
  - `Sampler.MatchesM1SampleToken`: for 200 seeded draws on random logits, `sample({0.8, 0, 1})` equals the old algorithm's output. Capture the old outputs as a fixture before changing the code.
  - `Sampler.TopKNeverPicksOutside`: `top_k = 2` over 10,000 draws picks only the two largest.
  - `Sampler.TopPKeepsMinimalSet`: probabilities `{0.5, 0.3, 0.15, 0.05}` with `top_p = 0.75` pick only indices 0 and 1, at frequencies within 2% of `0.625` and `0.375` over 20,000 draws. `top_p = 0.01` always picks index 0.
  - `Sampler.FrequenciesMatchSoftmax`: frequencies within 1.5% of the softmax at `temperature = 1`, over 20,000 draws.
  - `Sampler.RejectsBadConfig`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Write `sampling.md`:
  - what temperature does to the distribution, with a worked 3-logit example;
  - top-k and top-p, with the same example;
  - why inverse CDF with one uniform is exact;
  - the order of operations and its RNG contract.

  Then commit with `feat(runtime): add top-k and top-p sampling`.

### Task 7: Generator

**Files:**

- `src/runtime/generator.hpp/.cpp`
- `tests/runtime/generator_test.cpp`
- `docs/math/kv-cache.md`: a "Generation loop" section

**Interfaces:**

- Consumes: `Model::make_kv_cache` (Task 5) and `sample` (Task 6).
- Produces:
  - `class runtime::Generator`:
    - constructor `Generator(model::Model&, SamplerConfig, std::uint64_t seed)`, which validates the config;
    - `std::vector<std::int32_t> generate(std::vector<std::int32_t> prompt, std::int64_t n_new, const std::function<void(std::int32_t)>& on_token = {})`. It returns the prompt followed by the new tokens and calls `on_token` once per new token, in order.
- Behavior:
  - The prompt must be non-empty, or it throws. A prompt longer than `context_length` keeps its last `context_length` tokens.
  - Prefill the kept prompt, then sample from the last position.
  - Each decode step feeds the previous token at `pos_offset = cache.length()`.
  - When the cache is full, slide by half (see Decisions).
  - Runs in inference mode, batch 1.

- [ ] **Step 1:** Write the tests:
  - `Generator.GreedyMatchesNaive`: with `temperature 0`, 40 new tokens from a 5-token prompt, CPU, and CUDA when available. The tokens equal `generate_naive` with temperature 0.
  - `Generator.SampledMatchesNaiveOnCpu`: `{0.8, 0, 1}` with the same seed gives the same tokens as `generate_naive`.
  - `Generator.StreamsEveryToken`: `on_token` sees exactly the new tokens.
  - `Generator.SlidesPastContext`: a context-16 model and 50 new tokens. The model is never called with a position ≥ 16, all 50 tokens are produced, and the tokens before the first slide equal `generate_naive`.
  - `Generator.LongPromptKeepsTail`: a 40-token prompt on a context-16 model gives the same next token as the last 16 tokens alone.
  - `Generator.RejectsEmptyPrompt`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Commit with `feat(runtime): add kv-cached generator`.

### Task 8: CLI generate and resume, M3 run

**Files:**

- `apps/titan/commands.*`
- `apps/titan/main.cpp`
- `configs/tiny-shakespeare.json`: add `"checkpoint_interval": 500`
- `tests/CMakeLists.txt`
- `tests/fixtures/cli/*`
- `README.md`
- `AGENTS.md`
- `docs/architecture/overview.md`

**Interfaces:**

- Consumes: everything above.
- Produces:
  - `titan train` writes `<out>/latest.ckpt` through `on_checkpoint`. Its end-of-run sample uses `Generator`, with output identical to before.
  - `titan train --resume <dir> [--device cpu|cuda] [--gemm …]`.
  - `titan generate --checkpoint <file> [--prompt <text>] [--max-new N] [--temp T] [--top-k K] [--top-p P] [--seed S] [--device cpu|cuda] [--no-cache]`.
    - It streams each decoded token to stdout and flushes per token. The prompt is echoed first.
    - It writes `tokens/s` to stderr at the end.
    - The prompt defaults to a newline, as in `titan train`.
    - Usage errors exit 2. Runtime errors exit 1, with `error:` and no stdout output.

- [ ] **Step 1:** Write the CTest entries:
  - `cli_train_writes_checkpoint`: a tiny config and corpus. `latest.ckpt` exists.
  - `cli_train_resume`: train to the end with `--steps 6` and interval 3; resume from a run whose checkpoint was at step 3. The final `val_loss` lines are equal.
  - `cli_train_resume_rejects_config`.
  - `cli_generate_smoke`: generate from the tiny run's checkpoint with `--max-new 20 --device cpu`. Output matches the prompt.
  - `cli_generate_greedy_cache_equals_nocache`: `--temp 0`, with and without `--no-cache`, gives the same stdout.
  - `cli_generate_unknown_char`: exits 1 and names the character.
  - `cli_generate_missing_checkpoint`.
  - `cli_generate_bad_flags`: `--top-p 0`, `--top-p 1.5`, `--top-k -1`, `--temp -1` and `--max-new 0` exit 2.
  - `cli_generate_cuda_smoke`: skips without a GPU.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Validate every flag and read the checkpoint header before any output.
- [ ] **Step 4:** Run `scripts/check.ps1`. Expected: `check: OK`.
- [ ] **Step 5:** Run the M3 demonstration:
  - `titan train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m3-shakespeare --device cuda`.
  - Kill it after the step-1000 checkpoint, resume with `--resume runs/m3-shakespeare`, and check that the final val loss equals the uninterrupted M2 run's `1.5640`.
  - `titan generate --checkpoint runs/m3-shakespeare/latest.ckpt --prompt "ROMEO:" --max-new 500`.
  - Time `--max-new 500` with and without `--no-cache` on CPU and CUDA.

  Record in the README's "M3 result":
  - the resume result;
  - an excerpt of the generation;
  - the four tokens/s numbers.
- [ ] **Step 6:** Update AGENTS.md (commands, layout) and the overview (generation runtime view, the T2 mitigation now implemented). Then commit with `feat(cli): generate text and resume training from checkpoints`.
