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
