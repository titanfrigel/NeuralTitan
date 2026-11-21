# Model

## What it computes

`model::Model` is the whole decoder-only language model: it turns a batch of token-id sequences into next-token logits. Position `t` of each sequence gets a score for every vocabulary entry, computed only from tokens `0..t` (the attention is causal), so one forward pass gives `T` next-token predictions per sequence. Training minimizes the mean [cross-entropy](cross-entropy.md) of those logits against the input shifted left by one.

## Forward

With `L` blocks, `V = vocab_size`, ids `(B, T)`:

1. `x₀ = embed_tokens(ids)`: a row gather from the table `E` `(V, dim)`, see [Embedding](embedding.md). `(B, T, dim)`.
2. `x_{l+1} = block_l(x_l)` for `l = 0..L−1`, each a pre-norm [transformer block](transformer-block.md). `(B, T, dim)`.
3. `z = final_norm(x_L)`, RMSNorm or LayerNorm per `final_norm` in the config. `(B, T, dim)`.
4. `logits = z Hᵀ` with `H = E` when tied, else the separate `lm_head.weight` `(V, dim)`. `(B, T, V)`.

`forward` requires `B, T ≥ 1`, `B·T` ids and `T ≤ context_length`.

## Backward derivation

Given `dlogits = ∂L/∂logits`, the backward visits the forward chain in reverse; every module returns the gradient of its input and adds (`+=`) its parameter gradients:

1. `dz = lm_head.backward(dlogits) = dlogits · H`, and `dH += dlogitsᵀ z`.
2. `dx_L = final_norm.backward(dz)`.
3. `dx_l = block_l.backward(dx_{l+1})` for `l = L−1..0`.
4. `embed_tokens.backward(dx₀)` scatter-adds the rows of `dx₀` into `dE`.

### Weight tying

With tied embeddings `H = E`: the same table embeds the input and scores the output. `E` then has two uses, so by the multivariate chain rule its gradient is the sum of the gradients of both uses:

`dE = dlogitsᵀ z + scatter(dx₀)`

The head adds the first term into `embed_tokens.weight`'s `Parameter::grad` in step 1 and the embedding adds the second in step 4; because both accumulate with `+=` into one tensor, no extra code is needed. The head lists no parameters, so `parameters()` contains the shared table exactly once, as `embed_tokens.weight`, and the optimizer updates it once per step.

### Initialization

`init(rng)` draws, in `parameters()` order:

- every rank-2 weight (embedding, projections, untied head) from `N(0, 0.02²)`;
- the projections that end a residual branch (`o_proj`, `down_proj`, `proj`) with stddev `0.02 / √(2·L)` instead. Each block adds two such branch outputs to the residual stream, `2·L` in total; scaling each by `1/√(2·L)` keeps the variance of the stream at the last block independent of depth, as in GPT-2;
- norm weights to 1 and biases to 0.

## Shapes

- ids: `B·T` integers in `[0, V)`, row-major `(B, T)`; logits and `dlogits`: `(B, T, V)`.
- Parameter count, with `hd = dim / n_heads`, `h` the MLP hidden size and `n_norm` = 1 for RMSNorm, 2 for LayerNorm (weight and bias):
  - embedding: `V·dim`;
  - per block: `2·n_norm·dim` (two norms) `+ dim·(n_heads·hd) + 2·dim·(n_kv_heads·hd) + (n_heads·hd)·dim` (q, k, v, o) `+ 3·dim·h` (SwiGLU) or `2·dim·h + h + dim` (GELU with biases);
  - final norm: `n_norm·dim`;
  - head: `V·dim` when untied, 0 when tied.
- Example, `llama_style(65, 16, 2, 2, 1, 32)`: `hd = 8`, `h = 48`, each block `32 + 256 + 256 + 256 + 2304 = 3104`, total `1040 + 2·3104 + 16 = 7264`.

## Mapping to code

- `src/model/model.hpp`, `src/model/model.cpp`: `Model` owns `embed_tokens_` (`nn::Embedding`), `blocks_` (`nn::TransformerBlock`, named `blocks.N`), `final_norm_` and `lm_head_` (`nn::LMHead`, tied or untied). The constructor turns each `BlockConfig`'s strings into an `nn::BlockSpec`. A `Model` is neither copied nor moved, since the tied head points into `embed_tokens_`.
- `forward` and `backward` are steps 1-4 of each section above; `init` follows the initialization list, recognizing the residual projections by name.
- Loss: `nn::CrossEntropyLoss` in `src/nn/loss.hpp` produces `dlogits` for `backward`. The optimizer is not part of the model layer.
- Tests: `tests/model/model_test.cpp` checks the parameter count above, that a tied table is listed once and an untied head adds `V·dim`, the init standard deviations for Llama-style and GPT-style blocks (including `proj` scaling and zero biases), a finite-difference gradient check of the loss with respect to `embed_tokens.weight` (both tied uses), `blocks.0.attn.q_proj.weight` and `blocks.0.mlp.down_proj.weight`, that AdamW overfits one fixed batch to a loss below 0.05 within 300 steps (NFR-3), the sequence-length checks and the rejection of unknown variant strings.

## Device

M2 runs the same model on the GPU. Nothing in the math changes; this section records where the parameters live and where data crosses between the host and the device.

### Initialize on the CPU, then move

`Model::init` draws every weight on the CPU (`Rng` and `nn::init` are CPU-only by design), and `Model::to(Device::CUDA)` then copies each parameter to the GPU. One seed therefore gives bit-identical initial weights on both devices, and a CPU run and a CUDA run start from the same point.

`Model::to(device)` walks `parameters()` and replaces each `Parameter`'s `value` and `grad` in place (`p->value = p->value.to(device)`, the same for `grad`). The `Parameter` objects themselves never move. `Model::device()` reports the device of the parameters, which all share one.

The optimizer creates its moments `m` and `v` on the device of the parameters it is given, so it must be constructed after the move. `runtime::Trainer` does this: its constructor moves the model to `TrainConfig::device` (set by the caller, not read from the config file) and only then builds `optim::AdamW` from the moved parameters.

### Tied weights survive the move

With tied embeddings the head holds a pointer to the `embed_tokens.weight` `Parameter`, not a copy of its tensor. `parameters()` lists that `Parameter` once, so `to` moves it once, and the head reads the moved tensors through its pointer. After the move the head's weight and `embed_tokens.weight` are still one storage, and both backward contributions (the lookup's scatter-add and the head's `dyᵀ·x`) add into the one CUDA gradient. Had the head kept its own tensor, it would stay behind on the CPU, or become a second copy that trains separately, and training would silently differ from the CPU. `Model.ToCudaKeepsTiedWeights` pins this: the head's value and gradient share storage with `embed_tokens.weight` after the move, its value still shares storage after the optimizer step, and the embedding gradient after one training step and the embedding value after the optimizer step match the CPU twin's.

### Host-device copies in a training step

Parameters, activations, gradients and optimizer state stay on the GPU for the whole run. The data loader produces token ids on the host, and the scalars the trainer prints or tests come back to the host. Per micro-batch and step that is:

- **Id uploads.** `ops::embedding` uploads the `B·T` input ids. `ops::embedding_backward` groups the ids on the host into CSR form (unique ids, offsets, positions) and uploads those three arrays. `ops::cross_entropy` uploads the `B·T` targets. The ids are range-checked on the host before any upload, because kernels do not validate.
- **The loss scalar.** `ops::cross_entropy` copies the summed loss (one `double`) back to the host and returns it as a `float`; `Trainer::evaluate` does the same for each validation batch.
- **Grad-norm scalars.** `optim::clip_grad_norm` calls `ops::sum_squares` once per parameter, and each call copies one `double` back. The host adds them, takes the square root and, when clipping, launches `scale_` on every gradient with the factor.

Each of these copies is synchronous on the default stream, so it also waits for the preceding kernels. Sampling (`runtime::generate_naive`) reads the logits with `Tensor::to_vector()`, which copies from either device.

Tests: `tests/model/model_cuda_test.cpp` checks the tie after the move, the CPU-vs-CUDA parity of the logits, the loss and every parameter gradient for a Llama-style tied model and a GPT-style untied one, and that AdamW overfits one batch on the GPU within the 300-step budget of `Model.OverfitsSingleBatch`. `tests/nn/module_parity_test.cpp` compares `y`, `dx` and every parameter gradient of each module on the two devices. `Trainer.CudaSameSeedIsBitReproducible` in `tests/runtime/trainer_test.cpp` runs the trainer twice on the GPU and expects bit-identical results.
