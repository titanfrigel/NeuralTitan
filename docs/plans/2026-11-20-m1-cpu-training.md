# M1 CPU Model and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tiny Llama-style model trains on TinyShakespeare on the CPU in under 30 minutes, its validation loss falls, and it prints Shakespeare-ish text. Every op and every module has a hand-written backward that passes a finite-difference gradient check.

**Architecture:**

- Layers stack bottom to top: `core` (RNG, parallel-for) → `ops/cpu` + `ops` dispatch → `nn` → `optim` → `io` → `tokenizer` → `model` → `runtime` → the `titan` CLI.
- Each layer is one CMake static library with only its allowed links (overview §10).
- Ops are out-of-place functions on `Tensor`, plus explicit accumulate variants used for gradients.
- Modules follow the spec §5.3 interface. They cache what `backward` needs during a training-mode `forward`.
- There is no CUDA code in M1: the `ops` dispatch rejects CUDA tensors with a clear error, and M2 fills `ops/cuda`.

**Tech Stack:** C++20, MSVC 14.38, CMake presets, the project's own test harness, `std::thread` for the CPU matmul. No new dependencies.

**Spec:** `docs/specs/2026-11-19-neuraltitan-design.md`, with `docs/product/requirements.md` (FR-3 to FR-11, FR-15, FR-16, NFR-1, NFR-3 to NFR-6), `docs/architecture/overview.md` §10 (layer rules) and ADRs 0005 and 0007.

## Global Constraints

- **FP32 and seeding.** Every parameter, activation, gradient and optimizer state is FP32. One seed drives initialization, data order and sampling (NFR-4).
- **CPU reference ops.**
  - Single-threaded and written for clarity. Any sum over more than one element accumulates in `double`, so gradient checks stay meaningful in FP32.
  - The exception is the matmul, which may use `core`'s `parallel_for`. Each output element is still computed by one thread, in a fixed order, so results are deterministic.
- **Ops dispatch.** `src/ops/<family>.hpp` is the public interface. It checks preconditions (device, rank, shapes) with `TITAN_CHECK`, with messages that name the op and print the shapes. It then calls `ops::cpu::`.
  - A CUDA tensor fails with the exact text `ops::<name>: CUDA implementation arrives in M2`.
  - `ops/cpu` functions do not validate.
- **Module interface.** Exactly as in spec §5.3:
  - `forward(const Tensor&, ForwardContext&)`;
  - `backward(const Tensor& dy)`, which returns dx and accumulates parameter gradients with `+=`;
  - `parameters(std::vector<Parameter*>&)`.
- **Weight layout.** Linear weights are stored `(out_features, in_features)`, so `y = x·Wᵀ`. This matches Hugging Face, which M5 loads.
- **RoPE layout.** RoPE uses the half-split (`rotate_half`) layout with `θ_i = theta^(-2i/head_dim)` (spec §9).
- **Gradient checks.**
  - Central differences with `eps = 1e-2`.
  - The loss is `L = Σ y ⊙ R`, where R is a fixed random tensor.
  - Pass condition per checked element: `|analytic − numeric| ≤ 1e-3 + 2e-2·|numeric|`.
  - At most 64 deterministically chosen elements are checked per tensor.
- **Library dependencies:** only the C++ standard library. No CUDA calls outside `core`.
- **Math notes.** Every op family and module ships its math note in `docs/math/`, with:
  - what it computes;
  - forward;
  - backward derivation;
  - shapes;
  - mapping to code.

  `docs/math/README.md` lists every note.
- **Change rules.**
  - `CMakeLists.txt` files are edited through the shell, not the Edit/Write tools.
  - All code, comments and docs are in English.
  - Conventional Commits, with no AI attribution.
- **Branch and PR.** Work happens on branch `feat/m1-cpu-training`. One PR into `main`, titled `feat: m1 cpu model and training on tinyshakespeare`.
- **`check`.** `scripts/check.ps1` prints `check: OK` after every task, and the whole unit test suite stays under 60 seconds in the debug preset.

## Review Focus

1. **Too-short corpus or bad split.** A corpus with fewer than `seq_len + 2` tokens, or a split that leaves it, fails with an error naming the token count and the needed count, never with an out-of-range read. Pinned in Task 15 (`DataLoader.TooShortThrows`).
2. **Invalid model configs.**
   - The config check must reject:
     - `dim % n_heads != 0`;
     - `n_heads % n_kv_heads != 0`;
     - an odd head dimension when `position` is `rope`;
     - an unknown `mlp`, `norm` or `position` string;
     - `seq_len > context_length`.
   - Each is rejected at config load with the field name in the message. Pinned in Task 13 (`ModelConfig.RejectsInvalid`).
3. **Training divergence.** A non-finite loss or gradient norm stops training with an error naming the step, instead of silently printing `nan` for thousands of steps. Pinned in Task 15 (`Trainer.StopsOnNonFiniteLoss`).
4. **`--device cuda` in M1.** It prints `error: CUDA backend arrives in M2` and exits 1, instead of crashing in an op. Pinned in Task 16 (`cli_train_cuda_rejected`).
5. **Malformed tokenizer or token files.** A truncated `tokens.bin`, a tokenizer JSON with a missing or duplicate vocabulary entry, or invalid UTF-8 in the corpus fails with a clear `titan::Error` (NFR-5). Pinned in Tasks 11, 12 and 15.

## Decisions this plan makes (correctable by the author)

- **Token ids** are `std::int32_t` in `std::span`s, not FP32 tensors. `Embedding` and the loss take ids directly, so they sit beside the `Module` interface rather than implementing it.
- **GELU** uses the tanh approximation, as GPT-2 does.
- **Initialization.** Linear and Embedding weights use N(0, 0.02). Output projections (`o_proj`, `down_proj`, `proj`) are scaled by `1/√(2·n_layers)`, as in GPT-2. Norm weights start at 1 and biases at 0.
- **Generation in M1** re-runs the full window without a KV cache (`generate_naive`), only so M1 can show samples. M3 replaces it.
- **No checkpoints in M1.** Spec §7 puts them in M3. The run writes its tokenizer, token files and sample text to `--out`.
- **The token file** is `TTOK` magic, `u32` version 1, `u64` count, then `count` little-endian `int32` ids.

## File structure

| Layer | New files |
| --- | --- |
| core | `src/core/rng.hpp/.cpp`, `src/core/parallel.hpp/.cpp` |
| ops/cpu | `src/ops/cpu/{elementwise,reduce,matmul,softmax,norm,rope,embedding,cross_entropy,adamw}.hpp/.cpp` |
| ops | `src/ops/{elementwise,reduce,matmul,softmax,norm,rope,embedding,cross_entropy,adamw}.hpp/.cpp`, plus `src/ops/detail.hpp` (device check helper) |
| nn | `src/nn/{module,init,linear,embedding,norm,mlp,attention,block,lm_head,loss}.hpp/.cpp` |
| optim | `src/optim/{adamw,clip,schedule}.hpp/.cpp` |
| io | `src/io/{json,file,token_file}.hpp/.cpp` |
| tokenizer | `src/tokenizer/{utf8,char_tokenizer}.hpp/.cpp` |
| model | `src/model/{config,model}.hpp/.cpp` |
| runtime | `src/runtime/{data_loader,trainer,sampler}.hpp/.cpp` |
| cli | `apps/titan/main.cpp` (extended), `apps/titan/commands.hpp/.cpp` |
| tests | `tests/support/gradcheck.hpp/.cpp`, `tests/<layer>/*_test.cpp` |
| configs | `configs/tiny-shakespeare.json` |
| docs | `docs/math/{elementwise,reduce,matmul,softmax,normalization,rope,embedding,cross-entropy,adamw,linear,mlp,attention,transformer-block,model}.md` |

Every new layer directory gets a `CMakeLists.txt` defining `titan_<layer>`. `ops/cpu` and `ops` are `titan_ops_cpu` and `titan_ops`. `src/CMakeLists.txt` adds the subdirectories in layer order. Every new `.cpp` test file goes into `titan_tests`' source list.

---

### Task 1: RNG and parallel-for in core

**Files:**

- Create: `src/core/rng.hpp`, `src/core/rng.cpp`, `src/core/parallel.hpp`, `src/core/parallel.cpp`
- Test: `tests/core/rng_test.cpp`, `tests/core/parallel_test.cpp`

**Interfaces:**

- **Produces `class titan::Rng`:**
  - `explicit Rng(std::uint64_t seed)`, wrapping `std::mt19937_64`. The engine's sequence is fixed by the standard; its distributions are not, so we write our own transforms.
  - `std::uint64_t next_u64()`.
  - `float uniform()`: in [0, 1), computed as `(next_u64() >> 40) * 2^-24`.
  - `float normal()`: Box–Muller in `double`, caching the second value.
  - `std::int64_t randint(std::int64_t n)`: in [0, n), by rejection sampling, so there is no modulo bias. Uses `TITAN_CHECK(n > 0)`.
  - `void fill_normal(Tensor& t, float mean, float stddev)`: CPU tensors only.
- **Produces `void titan::parallel_for(std::int64_t begin, std::int64_t end, const std::function<void(std::int64_t, std::int64_t)>& body)`:**
  - Splits `[begin, end)` into contiguous chunks, one per thread.
  - Runs inline when the range has fewer than 2 elements per thread.
  - Rethrows the first exception from any chunk.
- **Produces `int titan::num_threads()` and `void titan::set_num_threads(int n)`:**
  - The default is `std::thread::hardware_concurrency()`, or 1 if that returns 0.
  - `n < 1` throws.

- [ ] **Step 1: Write the tests.**
  - `Rng.MatchesStandardEngine`: `Rng(5489).next_u64()` equals the first output of `std::mt19937_64(5489)`.
  - `Rng.SameSeedSameSequence`: two `Rng(42)` produce identical 1000-value `uniform()` sequences.
  - `Rng.UniformInRange`: 100 000 draws all lie in [0, 1) with mean within 0.01 of 0.5.
  - `Rng.NormalMoments`: 100 000 draws have mean within 0.02 of 0 and variance within 0.03 of 1.
  - `Rng.RandintBounds`: `randint(7)` over 70 000 draws hits all of 0..6, never 7, each count within 10% of 10 000; `randint(0)` throws.
  - `Parallel.CoversRangeOnce`: `parallel_for(0, 1003, …)` increments each element of a 1003-element `std::vector<std::atomic<int>>` exactly once.
  - `Parallel.PropagatesException`: a body that throws on index 500 makes `parallel_for` throw.
  - `Parallel.SetNumThreadsRejectsZero`.
- [ ] **Step 2: Build and see them fail.** Run `ctest --preset debug -R unit`. Expected: compile errors for the missing headers.
- [ ] **Step 3: Implement**, and add the sources to `src/core/CMakeLists.txt`.
- [ ] **Step 4: Run** `scripts/check.ps1 -Preset debug`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(core): add seeded rng and parallel_for`.

### Task 2: ops layer and elementwise ops with the gradient-check helper

**Files:**

- Create:
  - `src/ops/CMakeLists.txt`, `src/ops/cpu/CMakeLists.txt`, `src/ops/detail.hpp`;
  - `src/ops/elementwise.hpp/.cpp`, `src/ops/cpu/elementwise.hpp/.cpp`;
  - `tests/support/gradcheck.hpp/.cpp`;
  - `docs/math/elementwise.md`.
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `docs/math/README.md`
- Test: `tests/ops/elementwise_test.cpp`, `tests/support/gradcheck_test.cpp`

**Interfaces:**

- **Produces, in `titan::ops`.** All inputs must be on the same device and have the same shape where two are taken:
  - `Tensor add(const Tensor& a, const Tensor& b)` and `void add_(Tensor& a, const Tensor& b)` (`a += b`);
  - `Tensor mul(const Tensor& a, const Tensor& b)` and `Tensor scale(const Tensor& x, float s)`;
  - `Tensor silu(const Tensor& x)` and `Tensor silu_backward(const Tensor& x, const Tensor& dy)`;
  - `Tensor gelu(const Tensor& x)` and `Tensor gelu_backward(const Tensor& x, const Tensor& dy)`, using the tanh approximation `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`.
- **Produces `ops::detail::require_cpu(const char* op, std::initializer_list<const Tensor*>)`:** throws the M2 message from Global Constraints.
- **Produces, in `tests/support/gradcheck.hpp`:**
  - `Tensor random_tensor(Shape s, Rng& rng, float scale = 1.0f)`.
  - `struct GradCheckResult { bool ok; std::string detail; }`. On failure, `detail` names the element index, the analytic value and the numeric value.
  - `GradCheckResult check_gradient(const std::function<Tensor()>& forward, Tensor& input, const Tensor& analytic_grad, const Tensor& r)`. It perturbs `input` in place and restores it, and applies the tolerances from Global Constraints.
- **Linking:** `titan_ops_cpu` links `titan_core`, and `titan_ops` links `titan_ops_cpu titan_core`. The test support sources go into `titan_tests`.

- [ ] **Step 1: Write the tests.**
  - `Elementwise.AddMulScale`: exact values on `{2,3}` inputs.
  - `Elementwise.AddInPlaceAccumulates`.
  - `Elementwise.ShapeMismatchThrows`: the message contains both shapes.
  - `Elementwise.SiluValues`: `silu(0)=0`, `silu(1)=0.7310586`, within 1e-6.
  - `Elementwise.GeluValues`: `gelu(1)=0.8411920`, within 1e-6.
  - `Elementwise.SiluGradCheck` and `Elementwise.GeluGradCheck`, on `{3,5}` inputs.
  - `GradCheck.DetectsWrongGradient`: a deliberately doubled analytic gradient fails.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.** Write `docs/math/elementwise.md`, including the SiLU derivative `σ(x)(1+x(1−σ(x)))` and the GELU-tanh derivative, and add it to the README index.
- [ ] **Step 4: Run** `scripts/check.ps1 -Preset debug`. Expected: `check: OK`. The layer check stays green.
- [ ] **Step 5: Commit** with `feat(ops): add elementwise ops with cpu reference and gradient checks`.

### Task 3: Reductions

**Files:**

- Create: `src/ops/reduce.hpp/.cpp`, `src/ops/cpu/reduce.hpp/.cpp`, `docs/math/reduce.md`
- Test: `tests/ops/reduce_test.cpp`

**Interfaces:**

- **Produces, in `titan::ops`:**
  - `double sum(const Tensor& x)` and `double sum_squares(const Tensor& x)`;
  - `Tensor row_max(const Tensor& x)`, with shape `x.shape()` minus the last dimension (rank-1 input gives shape `{1}`);
  - `Tensor sum_leading(const Tensor& x)`, which sums over every dimension except the last, giving shape `{x.dim(-1)}`. It is used for bias gradients;
  - `bool all_finite(const Tensor& x)`.

- [ ] **Step 1: Write the tests.**
  - `Reduce.SumAndSumSquares` and `Reduce.RowMax`: exact values on `{2,3}`.
  - `Reduce.SumLeadingShapes`: `{2,3,4}` → `{4}`, with exact values.
  - `Reduce.SumIsDoubleAccumulated`: summing 10⁷ copies of `0.1f` stays within 1e-3 relative of `1e6`.
  - `Reduce.AllFiniteDetectsNanAndInf`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement**, and write the note.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(ops): add reductions`.

### Task 4: Matmul, batched matmul and the parallel CPU kernel

**Files:**

- Create: `src/ops/matmul.hpp/.cpp`, `src/ops/cpu/matmul.hpp/.cpp`, `docs/math/matmul.md`
- Test: `tests/ops/matmul_test.cpp`

**Interfaces:**

- **Produces `Tensor ops::matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false)`:**
  - Rank 2 to 4.
  - Leading dimensions are batch dimensions and must be equal in `a` and `b`.
  - `op(a)` is `(…, M, K)` and `op(b)` is `(…, K, N)`, and the result is `(…, M, N)`.
- **Produces `void ops::matmul_acc(Tensor& out, const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b)`:** `out += op(a)·op(b)`, with the same shape rules and `out` already shaped. It is used to accumulate weight gradients.
- **CPU kernel:**
  - The loop order and blocking are the builder's choice.
  - It runs `parallel_for` over the batch × M rows.
  - Each output element is accumulated in a fixed order: per-element `double`, or `float` partial sums within a fixed K-block. Write down whichever is chosen in the note.
  - Keep the naive triple loop as `cpu::matmul_naive`, as the reference.

- [ ] **Step 1: Write the tests.**
  - `Matmul.Small2D`: a known `{2,3}×{3,2}` product.
  - `Matmul.Transposes`: all four transpose flag combinations agree with explicitly transposed inputs.
  - `Matmul.Batched3DAnd4D`: equals per-batch 2-D products.
  - `Matmul.MatchesNaiveOnAwkwardShapes`: shapes `{37,53}×{53,29}` and `{3,17,65}×{3,65,31}`, within 1e-4 relative.
  - `Matmul.DeterministicAcrossThreadCounts`: identical bits with `set_num_threads(1)` and `set_num_threads(8)`.
  - `Matmul.AccAccumulates`.
  - `Matmul.ShapeMismatchThrows`: the message has both shapes and the flags.
  - `Matmul.GradCheck`: `dA = dC·Bᵀ`, `dB = Aᵀ·dC`, checked through `check_gradient` on `{4,6}×{6,5}`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.** In `docs/math/matmul.md`, cover:
  - row-major indexing;
  - the four transpose cases;
  - the backward derivation;
  - batching;
  - the threading and determinism argument;
  - a placeholder section titled `GPU kernels and roofline (M2)`.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(ops): add matmul with transposes, batching and a parallel cpu kernel`.

### Task 5: Softmax and causal softmax

**Files:**

- Create: `src/ops/softmax.hpp/.cpp`, `src/ops/cpu/softmax.hpp/.cpp`, `docs/math/softmax.md`
- Test: `tests/ops/softmax_test.cpp`

**Interfaces:**

- **Produces `Tensor ops::softmax(const Tensor& x)`:** over the last dimension, with max subtraction for stability.
- **Produces `Tensor ops::causal_softmax(const Tensor& scores, std::int64_t pos_offset)`:**
  - `scores` is `(…, Tq, Tk)`.
  - Entries with key index `j > i + pos_offset` get probability exactly 0.
  - It requires `Tk ≥ Tq + pos_offset`. `pos_offset` is 0 in training; M3 uses it with the KV cache.
- **Produces `Tensor ops::softmax_backward(const Tensor& y, const Tensor& dy)`:**
  - Computes `dx = y ⊙ (dy − Σⱼ dyⱼyⱼ)` per row.
  - Valid for both variants, because masked `y` is 0.

- [ ] **Step 1: Write the tests.**
  - `Softmax.RowsSumToOne`.
  - `Softmax.StableForLargeInputs`: rows like `{1000, 1001, 1002}` give finite values that match `{0, 1, 2}`.
  - `Softmax.CausalMaskZeros`: `{1,4,4}` gives an upper triangle of exactly 0 and rows that sum to 1.
  - `Softmax.CausalWithOffset`: `Tq=2, Tk=5, pos_offset=3` masks only `j > i+3`.
  - `Softmax.GradCheck` and `Softmax.CausalGradCheck`, on `{2,5,5}`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement**, and write the note: the Jacobian derivation, and why the mask is applied before the max.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(ops): add row softmax, causal softmax and backward`.

### Task 6: RMSNorm and LayerNorm ops

**Files:**

- Create: `src/ops/norm.hpp/.cpp`, `src/ops/cpu/norm.hpp/.cpp`, `docs/math/normalization.md`
- Test: `tests/ops/norm_test.cpp`

**Interfaces:**

- **RMSNorm, in `titan::ops`:**
  - `Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor* inv_rms_out)` computes `yᵢ = xᵢ · r · wᵢ` with `r = 1/√(mean(x²) + eps)` per row. `weight` is `{D}`, where `D = x.dim(-1)`. When `inv_rms_out` is non-null, it receives `r` with shape `x.shape()` minus the last dimension.
  - `Tensor rms_norm_backward(const Tensor& x, const Tensor& weight, const Tensor& inv_rms, const Tensor& dy, Tensor& dweight_acc)` returns dx and adds into `dweight_acc`.
- **LayerNorm, in `titan::ops`:**
  - `Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor* mean_out, Tensor* inv_std_out)`.
  - `Tensor layer_norm_backward(const Tensor& x, const Tensor& weight, const Tensor& mean, const Tensor& inv_std, const Tensor& dy, Tensor& dweight_acc, Tensor& dbias_acc)`.

- [ ] **Step 1: Write the tests.**
  - `Norm.RmsNormValues`: the row `{1,2,3,4}` with weight 1 and eps 0 gives `x/√7.5`.
  - `Norm.LayerNormZeroMeanUnitVar`.
  - `Norm.RmsNormGradCheck`, with respect to both x and the weight on `{3,8}`.
  - `Norm.LayerNormGradCheck`, with respect to x, the weight and the bias.
  - `Norm.WeightShapeMismatchThrows`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.** The note covers both backward derivations, written out step by step (the projection term).
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(ops): add rmsnorm and layernorm with backward`.

### Task 7: RoPE

**Files:**

- Create: `src/ops/rope.hpp/.cpp`, `src/ops/cpu/rope.hpp/.cpp`, `docs/math/rope.md`
- Test: `tests/ops/rope_test.cpp`

**Interfaces:**

- **Produces `Tensor ops::rope(const Tensor& x, std::int64_t pos_offset, float theta)`:**
  - `x` is `(B, H, T, hd)` and `hd` must be even.
  - For position `p = pos_offset + t` and `i < hd/2`, with angle `a = p·theta^(-2i/hd)`:
    - `y[i] = x[i]·cos a − x[i+hd/2]·sin a`;
    - `y[i+hd/2] = x[i+hd/2]·cos a + x[i]·sin a`.
- **Produces `Tensor ops::rope_backward(const Tensor& dy, std::int64_t pos_offset, float theta)`:** the transpose of the rotation, which is a rotation by `−a`.

- [ ] **Step 1: Write the tests.**
  - `Rope.PositionZeroIsIdentity`.
  - `Rope.PreservesNorm`: per head vector, within 1e-5.
  - `Rope.KnownValue`: `hd=4`, `theta=10000`, `p=1`, `x={1,0,0,0}` gives `y={cos 1, 0, sin 1, 0}`.
  - `Rope.RelativePositionProperty`: `⟨rope(q,p), rope(k,p')⟩` depends only on `p−p'`. Shift both positions by 5 and the dot product is unchanged within 1e-4.
  - `Rope.OddHeadDimThrows`.
  - `Rope.GradCheck`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.** The note covers:
  - the 2-D rotation per pair;
  - why the half-split layout is used (Hugging Face checkpoints);
  - the relative-position proof;
  - why backward is the inverse rotation.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(ops): add rope (half-split) with backward`.

### Task 8: Embedding gather and scatter, fused cross-entropy, AdamW update

**Files:**

- Create:
  - `src/ops/embedding.hpp/.cpp`, `src/ops/cpu/embedding.hpp/.cpp`;
  - `src/ops/cross_entropy.hpp/.cpp`, `src/ops/cpu/cross_entropy.hpp/.cpp`;
  - `src/ops/adamw.hpp/.cpp`, `src/ops/cpu/adamw.hpp/.cpp`;
  - `docs/math/embedding.md`, `docs/math/cross-entropy.md`.
- Test: `tests/ops/embedding_test.cpp`, `tests/ops/cross_entropy_test.cpp`, `tests/ops/adamw_test.cpp`

**Interfaces:**

- **Embedding, in `titan::ops`:**
  - `Tensor embedding(const Tensor& table, std::span<const std::int32_t> ids, Shape out_shape)`:
    - `table` is `(V, D)`;
    - `out_shape` is the id layout, for example `{B, T}`, and the result is `out_shape + {D}`;
    - the dispatch checks every id is in `[0, V)`, with the offending id and index in the message.
  - `void embedding_backward(Tensor& dtable_acc, std::span<const std::int32_t> ids, const Tensor& dy)`: a scatter-add of dy rows into the table rows they came from.
- **Cross-entropy, in `titan::ops`:**
  - `float cross_entropy(const Tensor& logits, std::span<const std::int32_t> targets, Tensor* dlogits_out)`.
  - `logits` is `(N, V)` and `targets.size() == N`.
  - It returns the mean negative log-likelihood, computed with log-sum-exp in `double`.
  - When `dlogits_out` is non-null, it receives `(softmax − onehot)/N`.
- **AdamW:**
  - `struct ops::AdamWHyper { float lr, beta1, beta2, eps, weight_decay; std::int64_t step; }`, where `step` starts at 1.
  - `void ops::adamw_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, const AdamWHyper& h)` applies:
    - `m = β1 m + (1−β1) g`;
    - `v = β2 v + (1−β2) g²`;
    - `m̂ = m/(1−β1^step)` and `v̂ = v/(1−β2^step)`;
    - `p −= lr·(m̂/(√v̂+eps) + wd·p)`.
  - The decoupled weight-decay form is documented in Task 10's note.

- [ ] **Step 1: Write the tests.**
  - `Embedding.GatherRows` and `Embedding.ScatterAddRepeatedIds`: id 2 appearing twice accumulates twice.
  - `Embedding.OutOfRangeIdThrows`: covers -1 and V.
  - `CrossEntropy.UniformLogitsGiveLogV`: zero logits with V=65 give `ln 65` within 1e-5.
  - `CrossEntropy.StableForLargeLogits`.
  - `CrossEntropy.GradCheck`: on `{4,7}`, with the loss itself as the scalar.
  - `CrossEntropy.TargetCountMismatchThrows`.
  - `AdamW.FirstStepMatchesFormula`: one step on a hand-computed 3-element case, within 1e-6.
  - `AdamW.WeightDecayShrinksWithZeroGrad`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.** Write both notes. The cross-entropy note derives `∂L/∂z = softmax − onehot` and explains why the fused form is stable.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(ops): add embedding gather/scatter, fused cross-entropy and adamw update`.

### Task 9: nn basics: Module, Parameter, Linear, Embedding, norms, MLPs

**Files:**

- Create:
  - `src/nn/CMakeLists.txt`;
  - `src/nn/{module,init,linear,embedding,norm,mlp}.hpp/.cpp`;
  - `docs/math/linear.md`, `docs/math/mlp.md`.
- Test: `tests/nn/linear_test.cpp`, `tests/nn/norm_module_test.cpp`, `tests/nn/mlp_test.cpp`, `tests/nn/embedding_module_test.cpp`

**Interfaces:**

- **Core types.** Use `std::optional<Tensor>` for caches, since `Tensor` has no default constructor.
  - `struct nn::Parameter { std::string name; Tensor value; Tensor grad; }` and `Parameter nn::make_parameter(std::string name, Shape shape, Device device = Device::CPU)`, with zero-filled value and grad.
  - `struct nn::ForwardContext { bool training = true; std::int64_t pos_offset = 0; }`.
  - `class nn::Module`, exactly as in spec §5.3.
  - `void nn::zero_grad(const std::vector<Parameter*>&)`.
- **Init:** `void nn::init_normal(Parameter&, Rng&, float stddev)` and `void nn::init_constant(Parameter&, float value)`.
- **`nn::Linear(std::string name, std::int64_t in, std::int64_t out, bool bias, Device = CPU)`:**
  - Parameters are `<name>.weight` `(out, in)` and `<name>.bias` `(out)`.
  - `x` may have any rank whose last dimension is `in`.
  - Backward accumulates `dW += dyᵀx` over the flattened rows, and `db += sum_leading(dy)`.
  - `Parameter& weight()`.
- **`nn::Embedding(std::string name, std::int64_t vocab, std::int64_t dim)`:**
  - `Tensor forward(std::span<const std::int32_t> ids, Shape id_shape, ForwardContext&)`;
  - `void backward(const Tensor& dy)`;
  - `void parameters(std::vector<Parameter*>&)`;
  - `Parameter& weight()`.
  - It is not a `Module` subclass (see Decisions).
- **Norms:**
  - `nn::RMSNorm(std::string name, std::int64_t dim, float eps)`, with `<name>.weight` initialized to 1.
  - `nn::LayerNorm(std::string name, std::int64_t dim, float eps)`, with `<name>.weight` = 1 and `<name>.bias` = 0.
- **MLPs:**
  - `nn::SwiGLU(std::string name, std::int64_t dim, std::int64_t hidden)` has children `gate_proj`, `up_proj` and `down_proj`, all without bias. It computes `down(silu(gate x) ⊙ up x)`.
  - `nn::GeluMLP(std::string name, std::int64_t dim, std::int64_t hidden)` has `fc` and `proj`, both with bias. It computes `proj(gelu(fc x))`.
- **Parameter names** are dotted paths, for example `blocks.0.mlp.gate_proj.weight`. Each module receives its full prefix in `name`.
- **Inference mode.** A training-mode `forward` caches what `backward` needs. Calling `backward` after an inference-mode `forward` throws with `backward called without a training-mode forward`.

- [ ] **Step 1: Write the tests**, each a gradient check with respect to the input and every parameter:
  - `Linear.Forward` (known values), `Linear.GradCheckNoBias`, `Linear.GradCheckBias3D` (input `{2,3,4}`);
  - `Linear.GradsAccumulate`: two backwards give twice the gradient;
  - `RMSNormModule.GradCheck`, `LayerNormModule.GradCheck`;
  - `SwiGLU.GradCheck`, `GeluMLP.GradCheck`;
  - `EmbeddingModule.GradCheckWeight`;
  - `Module.BackwardWithoutTrainingForwardThrows`;
  - `Module.ParameterNames`: SwiGLU("mlp", …) lists exactly `mlp.gate_proj.weight`, `mlp.up_proj.weight` and `mlp.down_proj.weight`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.** Write `linear.md` and `mlp.md`. `mlp.md` covers SwiGLU and GELU-MLP: forward, the backward chain and shapes.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(nn): add module interface, linear, embedding, norms and mlps`.

### Task 10: optim: AdamW, gradient clipping, LR schedule

**Files:**

- Create: `src/optim/CMakeLists.txt`, `src/optim/{adamw,clip,schedule}.hpp/.cpp`, `docs/math/adamw.md`
- Test: `tests/optim/optim_test.cpp`

**Interfaces:**

- **AdamW:**
  - `struct optim::AdamWConfig { float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weight_decay = 0.1f; }`.
  - `class optim::AdamW`:
    - `AdamW(std::vector<nn::Parameter*> params, AdamWConfig)` creates m and v per parameter, zero-filled.
    - `void step(float lr)` uses this lr, increments its step counter, and calls `ops::adamw_update`.
    - `std::int64_t steps() const`.
  - Parameters whose shape has rank 1 (norm weights, biases) get weight decay 0, which is the common practice. This is documented in the note.
- **`double optim::clip_grad_norm(const std::vector<nn::Parameter*>&, double max_norm)`:** returns the global L2 norm before clipping, and scales every grad by `max_norm / norm` when `norm > max_norm`.
- **`float optim::lr_at(std::int64_t step, std::int64_t warmup, std::int64_t total, float max_lr, float min_lr)`:**
  - Steps start at 0.
  - Linear warmup from `max_lr/warmup` up to `max_lr` at `step = warmup − 1`.
  - Then cosine decay to `min_lr` at `step = total − 1`.

- [ ] **Step 1: Write the tests.**
  - `AdamW.MinimizesQuadratic`: one parameter `{2}` starting at `{3, −2}`, with gradient `2p`, reaches `|p| < 1e-2` within 2000 steps at lr 0.05 and wd 0.
  - `AdamW.NoDecayOnRank1`.
  - `Clip.ScalesWhenAboveMax` and `Clip.NoopWhenBelow`: the returned norm is exact.
  - `Schedule.WarmupAndCosine`: `lr_at(0,10,100,1e-3,1e-4) == 1e-4`, `lr_at(9,…) == 1e-3`, `lr_at(99,…) == 1e-4`, and the schedule never increases after warmup.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement**, and write `adamw.md`: moments, bias correction, why weight decay is decoupled, clipping and the schedule.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(optim): add adamw, gradient clipping and warmup-cosine schedule`.

### Task 11: io: JSON parser and writer, file and token-file helpers

**Files:**

- Create: `src/io/CMakeLists.txt`, `src/io/{json,file,token_file}.hpp/.cpp`
- Test: `tests/io/json_test.cpp`, `tests/io/token_file_test.cpp`

**Interfaces:**

- **`class io::JsonValue`:**
  - Kinds: null, bool, number (`double`), string (UTF-8), array, object. Objects keep key order and reject duplicate keys.
  - Accessors throw `titan::Error` naming the expected and the actual kind: `is_*()`, `as_bool()`, `as_number()`, `as_int()` (which rejects non-integers), `as_string()`, `as_array()`, `as_object()`.
  - `const JsonValue* find(std::string_view key) const` and `const JsonValue& at(std::string_view key) const`. `at` names the missing key.
- **`io::JsonValue io::parse_json(std::string_view text)`:**
  - Follows RFC 8259.
  - Escapes include `\uXXXX`, with surrogate pairs converted to UTF-8.
  - Rejects unpaired surrogates, control characters in strings, trailing content, leading zeros and nesting deeper than 64.
  - Errors name the line and column.
- **`std::string io::to_json(const JsonValue&, int indent = 2)`:** `parse_json(to_json(v))` round-trips the value.
- **File helpers:**
  - `std::string io::read_file(const std::filesystem::path&, std::size_t max_bytes = 1 << 30)` errors on a missing file or a file over `max_bytes`, naming the path.
  - `void io::write_file(const std::filesystem::path&, std::string_view)`.
- **Token files** (format in Decisions):
  - `void io::write_token_file(const std::filesystem::path&, std::span<const std::int32_t>)`.
  - `std::vector<std::int32_t> io::read_token_file(const std::filesystem::path&)` validates the magic, the version, and that the count matches the file size exactly.

- [ ] **Step 1: Write the tests.**
  - `Json.ParsesAllKinds` and `Json.RoundTrip`.
  - `Json.UnicodeEscapes`: `"é"` → `é`, and the pair `"😀"` → 😀.
  - `Json.RejectsMalformed`: each case below throws, with a line and column:
    - `{"a":1,}`;
    - `[1 2]`;
    - `"\ud83d"`;
    - `01`;
    - `{"a":1}x`;
    - `{"a":1,"a":2}`;
    - 65 nested `[`;
    - a truncated string;
    - a raw newline in a string.
  - `Json.AccessorKindErrors`: the message names both kinds.
  - `TokenFile.RoundTrip`.
  - `TokenFile.RejectsMalformed`: bad magic, wrong version, truncated payload, and extra trailing bytes.
  - `File.MissingThrowsWithPath`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement**, as a recursive-descent parser with a depth counter.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(io): add json parser and writer and token file format`.

### Task 12: tokenizer: UTF-8 and CharTokenizer

**Files:**

- Create: `src/tokenizer/CMakeLists.txt`, `src/tokenizer/{utf8,char_tokenizer}.hpp/.cpp`
- Test: `tests/tokenizer/char_tokenizer_test.cpp`

**Interfaces:**

- **UTF-8 helpers:**
  - `std::vector<char32_t> tokenizer::decode_utf8(std::string_view)` rejects overlong forms, surrogates, values over U+10FFFF and truncated sequences, with the byte offset in the message.
  - `std::string tokenizer::encode_utf8(char32_t)`.
- **`class tokenizer::CharTokenizer`:**
  - `static CharTokenizer build(std::string_view corpus)`: the vocabulary is the distinct code points, sorted ascending, and ids are their indices.
  - `std::vector<std::int32_t> encode(std::string_view text) const`: an unknown character throws with the code point (`U+XXXX`) and its offset.
  - `std::string decode(std::span<const std::int32_t> ids) const`: an out-of-range id throws.
  - `std::int64_t vocab_size() const`.
- **Saving and loading:**
  - `io::JsonValue to_json() const` writes `{"type":"char","vocab":["\n"," ","!",…]}`.
  - `static CharTokenizer from_json(const io::JsonValue&)` rejects:
    - a wrong `type`;
    - an empty vocab;
    - an entry that is not exactly one code point;
    - duplicate entries.
- **Dependencies:** `titan_tokenizer` links `titan_io titan_core`.

- [ ] **Step 1: Write the tests.**
  - `CharTokenizer.BuildSortedVocab`: `"hello"` gives the vocab `e h l o`.
  - `CharTokenizer.EncodeDecodeRoundTrip`, which includes `é` and `\n`.
  - `CharTokenizer.UnknownCharThrows`: the message contains `U+`.
  - `CharTokenizer.InvalidUtf8Throws`: covers `"\xC3"` and `"\xED\xA0\x80"`.
  - `CharTokenizer.JsonRoundTrip`.
  - `CharTokenizer.RejectsMalformedJson`: covers every rejection listed above.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(tokenizer): add utf-8 handling and character tokenizer`.

### Task 13: Attention (GQA + RoPE) and the model config

**Files:**

- Create:
  - `src/nn/attention.hpp/.cpp`;
  - `src/model/CMakeLists.txt`, `src/model/config.hpp/.cpp`;
  - `docs/math/attention.md`.
- Test: `tests/nn/attention_test.cpp`, `tests/model/config_test.cpp`

**Interfaces:**

- **Position encoding:** `enum class nn::PositionEncoding { Rope, None }`.
- **`nn::Attention`:**
  - Constructor: `Attention(std::string name, std::int64_t dim, std::int64_t n_heads, std::int64_t n_kv_heads, PositionEncoding pos, float rope_theta)`.
  - Linears without bias: `q_proj` `(dim → n_heads·hd)`, `k_proj` and `v_proj` `(dim → n_kv_heads·hd)`, and `o_proj`, with `hd = dim / n_heads`.
  - Input and output are `(B, T, dim)`.
  - Forward steps:
    1. project to Q, K and V;
    2. reshape to `(B, H, T, hd)`;
    3. apply RoPE to Q and K when `pos == Rope`;
    4. expand each KV head to its `n_heads/n_kv_heads` query heads;
    5. compute scores `Q·Kᵀ/√hd`;
    6. apply `causal_softmax` with `pos_offset` from the context;
    7. multiply by V and merge the heads;
    8. apply `o_proj`.
  - Backward reverses each step. The gradients of the expanded K and V heads are summed back over each group.
  - The constructor checks `dim % n_heads == 0`, `n_heads % n_kv_heads == 0`, and an even `hd` with RoPE.
- **Model config:**
  - `struct model::BlockConfig { std::int64_t n_heads, n_kv_heads, mlp_hidden; std::string mlp /*swiglu|gelu*/, norm /*rmsnorm|layernorm*/, position /*rope|none*/; }`.
  - `struct model::ModelConfig { std::int64_t vocab_size, dim, context_length; float rope_theta = 10000.f, norm_eps = 1e-5f; bool tied_embeddings = true; std::string final_norm = "rmsnorm"; std::vector<BlockConfig> blocks; }`.
  - `ModelConfig model::parse_model_config(const io::JsonValue& model_section, std::int64_t vocab_size)`.
    - It accepts either a `"blocks"` array or `"n_layers"` plus a `"block"` object (copied `n_layers` times).
    - `vocab_size` comes from the tokenizer, not the file.
    - Every validation from Review Focus 2 lives here, with the field path in the message (for example `blocks[2].n_kv_heads`).
  - `ModelConfig model::llama_style(std::int64_t vocab, std::int64_t dim, std::int64_t n_layers, std::int64_t n_heads, std::int64_t n_kv_heads, std::int64_t context)`: SwiGLU with hidden `round_up(8·dim/3, 8)`, RMSNorm, RoPE and tied embeddings (FR-7).
- **Dependencies:** `titan_model` links `titan_nn titan_io titan_ops titan_core`.

- [ ] **Step 1: Write the tests.**
  - `Attention.CausalityHolds`: changing input position t=3 does not change outputs at t<3.
  - `Attention.GradCheckMHA`: `n_heads = n_kv_heads = 2`, dim 8, T=5, B=2, with RoPE.
  - `Attention.GradCheckGQA`: 4 heads, 2 KV heads, with RoPE.
  - `Attention.GradCheckNoPE`.
  - `Attention.InvalidHeadsThrow`.
  - `ModelConfig.ParsesBlocksArray` and `ModelConfig.ParsesNLayersShorthand`.
  - `ModelConfig.RejectsInvalid`: one case per Review Focus 2 item, each checking the field path in the message.
  - `ModelConfig.LlamaStylePreset`: `dim=128` gives `mlp_hidden=344`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement**, and write `attention.md`:
  - the scaled dot-product derivation;
  - the causal mask;
  - GQA grouping and its gradient sum;
  - every shape through the layer;
  - why the score matrix is materialized in v1.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(nn): add grouped-query attention with rope; feat(model): add model config`. These are two commits: nn first, then model.

### Task 14: TransformerBlock, LM head, loss and Model

**Files:**

- Create:
  - `src/nn/block.hpp/.cpp`, `src/nn/lm_head.hpp/.cpp`, `src/nn/loss.hpp/.cpp`;
  - `src/model/model.hpp/.cpp`;
  - `docs/math/transformer-block.md`, `docs/math/model.md`.
- Test: `tests/nn/block_test.cpp`, `tests/model/model_test.cpp`

**Interfaces:**

- **`nn::TransformerBlock(std::string name, std::int64_t dim, const nn::BlockSpec&, float norm_eps, float rope_theta)`:**
  - `struct nn::BlockSpec` mirrors `BlockConfig` with enums: `enum class MlpType { SwiGLU, Gelu }` and `enum class NormType { RMSNorm, LayerNorm }`.
  - It computes `h = x + attn(norm1(x))`, then `y = h + mlp(norm2(h))`.
  - Backward routes the residual gradient into both branches.
- **`nn::LMHead`:**
  - Tied: `LMHead(nn::Parameter& tied_embedding_weight)`.
  - Untied: `LMHead(std::string name, std::int64_t dim, std::int64_t vocab)`.
  - It computes `logits = x·Eᵀ`, and its backward accumulates into the shared Parameter.
  - In tied mode its `parameters()` lists nothing.
- **`nn::CrossEntropyLoss`:** `float forward(const Tensor& logits, std::span<const std::int32_t> targets)` flattens `(B,T,V)` to `(B·T, V)`. `Tensor backward()` returns dlogits in the logits' shape.
- **`model::Model(const ModelConfig&)`:**
  - `void init(Rng&)` applies the Decisions init scheme.
  - `Tensor forward(std::span<const std::int32_t> ids, std::int64_t B, std::int64_t T, nn::ForwardContext&)` returns logits `(B, T, V)`.
  - `void backward(const Tensor& dlogits)`.
  - `std::vector<nn::Parameter*> parameters()` lists each parameter exactly once, tied ones included.
  - `std::int64_t num_parameters()` and `const ModelConfig& config() const`.
  - `forward` checks `T ≤ context_length`.
  - Parameter names follow Hugging Face Llama naming: `embed_tokens.weight`, `blocks.N.attn.q_proj.weight`, `blocks.N.norm1.weight`, `final_norm.weight`, `lm_head.weight` (untied only). M5 maps them.

- [ ] **Step 1: Write the tests.**
  - `Block.GradCheckLlamaStyle` and `Block.GradCheckGptStyle` (LayerNorm + GELU, NoPE).
  - `Model.ParameterCountTiny`: `llama_style(65, 16, 2, 2, 1, 32)` returns the hand-computed count, written in the test as a sum with comments.
  - `Model.TiedParameterListedOnce`.
  - `Model.GradCheckTiny`: the full model with V=11, dim 8, 2 layers, T=6, B=2. It checks the embedding weight and one block's `q_proj` and `down_proj`, against the cross-entropy loss.
  - `Model.OverfitsSingleBatch` (NFR-3):
    - `llama_style(20, 32, 2, 4, 2, 16)`, with one fixed batch B=4, T=16 and AdamW at lr 3e-3.
    - The loss falls below 0.05 within 300 steps.
    - The test runs in under 10 s in the debug preset; if not, reduce the dimensions and note it.
  - `Model.SequenceLongerThanContextThrows`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement**, and write both notes. `model.md` covers the whole forward chain with shapes, the backward order, weight tying, and the parameter-count formula.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Commit** with `feat(nn): add transformer block, lm head and loss; feat(model): add model` (two commits).

### Task 15: runtime: DataLoader, Trainer, sampler

**Files:**

- Create: `src/runtime/CMakeLists.txt`, `src/runtime/{data_loader,trainer,sampler}.hpp/.cpp`
- Test: `tests/runtime/data_loader_test.cpp`, `tests/runtime/trainer_test.cpp`, `tests/runtime/sampler_test.cpp`

**Interfaces:**

- **`runtime::DataLoader`:**
  - Constructor: `DataLoader(std::vector<std::int32_t> tokens, std::int64_t batch, std::int64_t seq, std::uint64_t seed)`.
  - `struct Batch { std::vector<std::int32_t> inputs, targets; }`, each of size B·T.
  - `Batch next()` draws B random offsets in `[0, n − T − 1]`, with `targets = inputs` shifted by one.
  - The constructor checks `n ≥ T + 2` (Review Focus 1).
- **`runtime::TrainConfig`**, parsed from the config file's `"train"` section by `TrainConfig parse_train_config(const io::JsonValue&)`. Its fields:
  - `batch` and `seq`;
  - `steps`, `grad_accum = 1`;
  - `lr`, `min_lr`, `warmup`;
  - `weight_decay`, `beta1`, `beta2`, `eps`;
  - `grad_clip = 1.0`;
  - `eval_interval`, `eval_batches`;
  - `log_interval`;
  - `seed`;
  - `val_fraction = 0.1`.
- **`runtime::Trainer`:**
  - Constructor: `Trainer(model::Model&, TrainConfig, std::vector<std::int32_t> train_tokens, std::vector<std::int32_t> val_tokens, std::ostream& log)`.
  - `struct TrainResult { float final_train_loss, final_val_loss; double seconds; std::int64_t tokens; }` and `TrainResult run()`.
  - Each step:
    1. zero the gradients;
    2. for each of `grad_accum` micro-batches, run forward, loss and backward, with the loss gradient scaled by `1/grad_accum`;
    3. `clip_grad_norm`;
    4. `AdamW::step(lr_at(...))`.
  - It throws `training diverged at step N: loss=<v> grad_norm=<v>` when the loss or the norm is not finite.
  - A log line every `log_interval` steps: `step 200/2000 loss 2.1034 lr 9.8e-04 grad_norm 0.512 tok/s 12345`.
  - An eval line every `eval_interval` steps and at the end: `eval step 200 val_loss 2.0871`.
  - Evaluation uses `eval_batches` batches from a val DataLoader seeded with `seed + 1`, in inference mode.
- **Sampling:**
  - `std::int32_t runtime::sample_token(std::span<const float> logits, float temperature, Rng&)`: `temperature == 0` means argmax, otherwise softmax(logits/T) and an inverse-CDF draw.
  - `std::vector<std::int32_t> runtime::generate_naive(model::Model&, std::vector<std::int32_t> prompt, std::int64_t n_new, float temperature, Rng&)`. Each step runs the last `context_length` tokens in inference mode and samples from the last position.
- **Dependencies:** `titan_runtime` links `titan_model titan_optim titan_tokenizer titan_nn titan_io titan_ops titan_core`.

- [ ] **Step 1: Write the tests.**
  - `DataLoader.TargetsAreShiftedInputs`, `DataLoader.SameSeedSameBatches`, `DataLoader.TooShortThrows`.
  - `TrainConfig.ParsesAndRejectsMissingField`.
  - `Trainer.LossFallsOnRepeatingPattern`: a token stream `0 1 2 3 4 …` repeating, with a tiny model and 100 steps, ends with a final val loss below 0.5 and prints the log and eval lines in the formats above.
  - `Trainer.StopsOnNonFiniteLoss`: lr 1e6 makes it throw with `diverged at step`.
  - `Trainer.GradAccumEquivalence`:
    - Run (B=4, accum=1) and (B=2, accum=2) on the same tokens and seed, with DataLoaders that yield identical windows (sequential mode via a test seam).
    - After one step, the parameters agree within 1e-5.
    - Add `DataLoader::set_sequential(bool)` for this test seam, and document it.
  - `Sampler.GreedyIsArgmax`.
  - `Sampler.TemperatureDistribution`: over 20 000 draws at T=1, the frequencies match softmax within 0.02.
  - `Sampler.SeededIsDeterministic`.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `check`. Expected: `check: OK`. The whole unit suite stays under 60 s in debug.
- [ ] **Step 5: Commit** with `feat(runtime): add data loader, trainer and naive sampler`.

### Task 16: CLI `tokenizer train` and `train`, the TinyShakespeare config, the M1 run

**Files:**

- Create: `apps/titan/commands.hpp/.cpp`, `configs/tiny-shakespeare.json`
- Modify: `apps/titan/main.cpp`, `apps/CMakeLists.txt` (link `titan_runtime`), `tests/CMakeLists.txt`, `README.md`, `AGENTS.md`, `docs/math/README.md`
- Test: CTest entries `cli_tokenizer_train`, `cli_train_smoke`, `cli_train_cuda_rejected`, `cli_train_bad_config`

**Interfaces:**

- **`titan tokenizer train --type char --corpus <file> --out <tok.json>`:** builds a CharTokenizer and writes its JSON.
- **`titan train --config <cfg.json> --data <corpus.txt> --out <dir> [--tokenizer <tok.json>] [--steps N] [--device cpu|cuda]`:**
  1. Load or build the char tokenizer. When it is built, write `<out>/tokenizer.json`.
  2. Encode the corpus once. The last `val_fraction` of the tokens is the validation split. Write `<out>/train.bin` and `<out>/val.bin` (FR-10).
  3. Build the model from the config's `"model"` section and the tokenizer's vocab size, then `init(Rng(seed))`.
  4. Train, logging to stdout.
  5. Print and write `<out>/sample.txt`: 500 new characters from the prompt `"\n"` at temperature 0.8, seed `seed`.
  6. Print `params <n>`, `time <s> s` and `final val_loss <v>`.
- **`--device`:** `cuda` exits 1 with `error: CUDA backend arrives in M2`. The default is `cpu` in M1.
- **Errors:** an unknown or missing flag prints usage and exits 2. A `titan::Error` exits 1, as in M0.
- **`configs/tiny-shakespeare.json`:**
  - `"model"`: `dim` 128, `context_length` 128, `n_layers` 4, and `block {n_heads 4, n_kv_heads 2, mlp "swiglu", mlp_hidden 344, norm "rmsnorm", position "rope"}`. Also `rope_theta` 10000, `norm_eps` 1e-5, `tied_embeddings` true.
  - `"train"`:
    - `batch` 16, `seq` 128, `steps` 2000, `grad_accum` 1;
    - `lr` 1e-3, `min_lr` 1e-4, `warmup` 100;
    - `weight_decay` 0.1, `beta1` 0.9, `beta2` 0.95, `eps` 1e-8, `grad_clip` 1.0;
    - `eval_interval` 200, `eval_batches` 20, `log_interval` 50;
    - `seed` 1337, `val_fraction` 0.1.

- [ ] **Step 1: Register the CTest entries.** All of them use `$<TARGET_FILE:titan>` with a small committed fixture corpus, `tests/fixtures/cli/tiny_corpus.txt` (about 2 KB of public-domain text that is safe to commit), and a fixture config `tests/fixtures/cli/tiny-config.json` (dim 16, 1 layer, steps 20, batch 2, seq 16).
  - `cli_train_smoke` passes on the regex `final val_loss [0-9.]+`.
  - `cli_train_cuda_rejected` uses `expect_failure.cmake` with exit 1 and `CUDA backend arrives in M2`.
  - `cli_train_bad_config` uses a config with `n_kv_heads` 3 and 4 heads, and expects exit 1 with `n_kv_heads`.
  - `cli_tokenizer_train` writes to the build dir and passes on exit 0.
- [ ] **Step 2: Build and see them fail.**
- [ ] **Step 3: Implement** the commands and the config.
- [ ] **Step 4: Run** `check`. Expected: `check: OK`.
- [ ] **Step 5: Get the corpus (ask the author first).** Downloading a file needs explicit approval. Ask before fetching `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` (about 1.1 MB, git-ignored) to `data/tinyshakespeare/input.txt`, or let the author place it there.
- [ ] **Step 6: The M1 run.**
  - Run `build/release/apps/titan.exe train --config configs/tiny-shakespeare.json --data data/tinyshakespeare/input.txt --out runs/m1-shakespeare` with the release preset.
  - Pass conditions (NFR-1, spec §7 M1):
    - it finishes in under 30 minutes;
    - the val loss falls from about `ln 65 ≈ 4.17` to below 2.0;
    - `sample.txt` shows Shakespeare-like structure: speaker names, line breaks, English-like words.
  - If time exceeds 30 minutes:
    1. first profile, and check that the matmul runs on all threads;
    2. only then lower `steps` or `dim` in the config and record why.
- [ ] **Step 7: Docs.**
  - README: a `titan train` usage section and an "M1 result" section with the measured time, the final val loss, the parameter count and a 10-line sample excerpt.
  - AGENTS.md: the Commands and Layout sections list the new layers and commands.
  - `docs/math/README.md`: every note from this milestone is listed.
- [ ] **Step 8: Commit** with `feat(cli): add tokenizer train and train commands with the tinyshakespeare config`, then `docs: record the m1 tinyshakespeare run`.
- [ ] **Step 9: Ask the author before pushing.** Then run `git push -u origin feat/m1-cpu-training` and open the PR. Its body follows the PR template and notes the measured run time and the val loss.

---

## Later milestones

- **M2:** CUDA kernels with parity tests against these CPU references, cuBLAS plus the project's own naive and tiled GEMM.
- **M3:** KV cache, samplers, checkpoints and `titan generate`.
- **M4:** BPE.
- **M5:** SmolLM2-135M.
- **M6:** TinyStories.
