# KV cache

## What it computes

Generation produces one token at a time: run the model on the sequence so far, sample the next token from the logits of the last position, append it and repeat. Run naively, step `n` recomputes the keys and values of all `n` earlier positions, although they have not changed since the previous step. The KV cache stores, for every attention layer, the keys and values of the positions already processed, so each new token is run through the model once and attends over the stored keys and values.

`nn::KvCache` holds the stored tensors and the number of filled positions; `nn::Attention` reads and writes it when `ForwardContext::kv_cache` is set; `model::Model::make_kv_cache` builds a cache that fits the model.

### Why causality makes the past reusable

In a causal decoder, the hidden state of position `t` in every layer depends only on tokens `0 … t`: each layer mixes positions only through attention, and the causal mask hides every key `j > t`; everything else (norms, MLP, residuals, projections) acts on each position on its own. By induction over the layers, appending a token at position `n` changes nothing at positions `0 … n − 1`. Their keys `K = h W_kᵀ` and values `V = h W_vᵀ`, computed from those hidden states, are therefore final the first time they are computed, and can be kept and reused by every later query.

RoPE does not break this: the key of position `j` is rotated by the absolute angle of `j`, which does not depend on the query that reads it, so the cache stores keys after the rotation.

## Forward

Notation as in [Attention](attention.md): `x` is `(B, T, dim)`, `H = n_heads`, `Hkv = n_kv_heads`, `hd = dim / H`, `p₀ = ctx.pos_offset`. With a cache, `p₀` must equal the cache length `L`, the number of positions already stored.

1. Project and split heads as without a cache: `Q` is `(B, H, T, hd)`, `K` and `V` are `(B, Hkv, T, hd)`.
2. RoPE at positions `p₀ … p₀ + T − 1`: `Q ← rope(Q, p₀)`, `K ← rope(K, p₀)`.
3. Store: `kv_write(cache_K, K, p₀)` and `kv_write(cache_V, V, p₀)` set rows `p₀ … p₀ + T − 1` of the layer's cache.
4. Load: `K ← kv_read(cache_K, p₀ + T)`, `V ← kv_read(cache_V, p₀ + T)`, now `(B, Hkv, p₀ + T, hd)`: the stored past followed by the new rows.
5. `repeat_kv`, then the scores `S = s · Q K_rᵀ`, `(B, H, T, p₀ + T)`.
6. `P = causal_softmax(S, p₀)`: query `t` (absolute position `p₀ + t`) sees key `j` when `j ≤ t + p₀`.
7. `O = P V_r` and the output projection, as without a cache.

After the forward, the caller sets the cache length to `p₀ + T`. The model forward writes every layer at the same `p₀`, so the length is one number for the whole cache and only the caller advances it, once per forward.

### Prefill against decode

- **Prefill** runs the whole prompt of `T` tokens at `p₀ = 0`. The scores are `(T, T)` and the mask is the ordinary causal one, exactly the uncached forward; on top of it, the keys and values of all `T` positions are stored. Only the logits of the last row are needed to sample the first new token.
- **Decode** runs one token, `T = 1`, at `p₀ = L`. The single query sees all `L + 1` keys, so the mask hides nothing and the scores are one row `(1, L + 1)` per head. The logits equal the last row of an uncached forward over the whole sequence, which `tests/nn/kv_cache_test.cpp` checks.

### Cost per token

Without the cache, generating the token at position `n` reruns the full forward over `n + 1` positions: the projections and the MLP cost `O(n)` and the scores `O(n²)` per layer and head (an `(n + 1) × (n + 1)` matrix). With the cache, a decode step runs the projections and the MLP for one position, `O(1)`, and computes one score row against `n + 1` keys, `O(n)`. Generating `N` tokens thus costs `O(N²)` for the scores instead of `O(N³)`.

`kv_read` copies the filled rows into a new tensor every step, which is also `O(n)` per layer and so does not change the order. A view over the first rows would avoid the copy, but the cache rows of one head are not contiguous across heads once `L < Tmax`, and the matmul expects packed operands.

### Memory

Each layer stores K and V of shape `(B, Hkv, Tmax, hd)` in FP32, so the cache takes

`2 · n_layers · B · n_kv_heads · max_len · head_dim · 4` bytes,

with `max_len = context_length`. For the TinyShakespeare config (`configs/tiny-shakespeare.json`: 4 layers, `n_kv_heads = 2`, `context_length = 128`, `hd = 128 / 4 = 32`) and one sequence this is `2 · 4 · 2 · 128 · 32 · 4 = 262 144` bytes, 256 KiB. The cache is allocated once, at its full size, and zero-filled, so generation allocates nothing per step for it.

### Layout and the GQA saving

Each layer's cache is `(B, Hkv, Tmax, hd)`, the head-major layout attention computes in, so `kv_write` and `kv_read` move whole rows of `hd` values and step 4 needs no transpose. The cache stores the un-repeated keys and values, one per KV head, and `repeat_kv` expands them after the read. With grouped-query attention the cache is therefore `n_heads / n_kv_heads` times smaller than with one KV head per query head: the TinyShakespeare model, with 4 query heads over 2 KV heads, needs 256 KiB instead of 512 KiB. Keeping the cache small is the main reason modern models use GQA, since in long generations the cache, not the weights, dominates memory traffic.

Blocks may have different `n_kv_heads` and head sizes, so the cache keeps a shape per layer (`nn::KvLayerShape`); `make_kv_cache` reads them from each block's config.

## Generation loop

`runtime::Generator` drives the cache to extend a prompt by `N` tokens, batch 1, in inference mode. It owns the cache, built once by `make_kv_cache(1)` on the model's device, and one `Rng` seeded at construction. Write `C = context_length` and `L` for the cache length.

### Prefill then decode

1. **Prefill.** Clear the cache (`L = 0`) and run the last `min(len, C)` prompt tokens in one forward at `p₀ = 0`, then set `L` to that count. A prompt longer than `C` is fed by its tail, because no position may reach `C`; the returned sequence still starts with the whole prompt. Sample the first new token from the last row of the logits.
2. **Decode.** Append the sampled token, call `on_token` with it, and, unless it was the last one, run it alone (`T = 1`) at `p₀ = L`, set `L ← L + 1` and sample the next token from the single row of logits.

Each forward goes through one helper that sets `pos_offset = L`, runs the model, advances the length to `L + T` and samples from row `T − 1`. The logits are copied to the host for sampling, since the sampler runs on host floats.

Until the cache fills, every sampled distribution comes from the same logits as an uncached forward over the whole sequence (see [Prefill against decode](#prefill-against-decode)), so the Generator and `runtime::generate_naive` produce the same tokens. On the CPU the cached and uncached logits agree exactly, and `tests/runtime/generator_test.cpp` checks that sampled tokens with the same seed are equal, not only greedy ones.

### Sliding by half

Decode needs a free row: a step at `L = C` would write position `C`, past the cache and past the positions the model was built for (`kv_write` throws). When the cache is full, the Generator slides: it keeps the last `K = max(1, ⌊C / 2⌋)` tokens of the sequence, which end with the token just sampled, clears the cache and prefills those `K` tokens at positions `0 … K − 1`. Decoding then continues at `L = K` and the next slide comes after `C − K` more tokens.

- **Cost.** A slide is one prefill of `K` tokens, about the cost of `K` decode steps. It happens once every `C − K ≈ C / 2` tokens, so it adds about one decode step's worth of work per generated token: the amortized cost per token stays `O(C)` for the scores, the same order as an ordinary decode step. Sliding by one token instead would need a full re-prefill of `C − 1` tokens every step, since every cached key carries the RoPE rotation of its old position and none of them can be reused at a shifted position.
- **Trade-off.** Right after a slide, the model sees only the last `K` tokens instead of the last `C`, and the window then grows back to `C` before the next slide. `generate_naive` always sees the last `min(len, C)` tokens, so the two agree only until the first slide; after it, they condition on different windows and their tokens differ. Halving keeps the visible window between `C / 2` and `C` tokens while keeping slides rare.

With the prompt of length `P ≤ C`, the first slide comes before new token `C − P + 1` (counting from 0): new tokens `0 … C − P` are produced at lengths `P … C`, as `tests/runtime/generator_test.cpp` checks through `Generator::cache()` from the `on_token` callback.

### The RNG contract

A single seed drives sampling: the Generator seeds its `Rng` once, at construction, and every `generate` call continues the same stream. Each new token is chosen by exactly one call to `runtime::sample`, which draws exactly one `uniform()` when the temperature is above 0 and none for greedy decoding. Prefill, decode and slide steps all draw the same way, and the order of draws is the order of the tokens, the same as in `generate_naive`. So for one seed and one model, the Generator and `generate_naive` draw the same numbers and, while their logits agree, pick the same tokens; a slide changes which logits the draws are applied to, never the number or order of the draws.

## Backward derivation

None, inference only. A forward with a cache requires `training == false` and throws "attention: the KV cache is inference-only" otherwise. That forward stores nothing for backward: attention resets its saved `Q`, `K_r`, `V_r` and `P`, so a later `backward` throws. `kv_write` and `kv_read` are copies with no gradient (see [Attention](attention.md#the-layout-ops)).

## Shapes

| Tensor | Shape |
| --- | --- |
| `keys(i)`, `values(i)` | `(B, Hkv_i, max_len, hd_i)` |
| `x`, `y` of a cached forward | `(B, T, dim)`, with `p₀ + T ≤ max_len` |
| `K`, `V` after the read | `(B, Hkv, p₀ + T, hd)` |
| `S`, `P` | `(B, H, T, p₀ + T)` |
| logits of a decode step | `(B, 1, vocab)` |

The checks: `p₀ = length()`, the cache has one layer per block, layer `i` matches the block's `n_kv_heads` and head size, the input batch equals the cache batch, and `p₀ + T ≤ context_length`. Without a cache, `p₀ > 0` throws.

## Mapping to code

- `nn::KvCache`, `nn::KvLayerShape`: `src/nn/kv_cache.hpp`, `src/nn/kv_cache.cpp`. The constructor allocates the zero-filled per-layer tensors; `length`, `set_length` and `clear` keep the fill count.
- `ForwardContext::kv_cache` and `ForwardContext::layer`: `src/nn/module.hpp`.
- The cached path of `nn::Attention::forward`: `src/nn/attention.cpp`, steps 3-4 above between the RoPE and the `repeat_kv` of the uncached path.
- `model::Model::make_kv_cache` and the per-block `ctx.layer`: `src/model/model.cpp`.
- The copies: `ops::kv_write`, `ops::kv_read` in `src/ops/layout.hpp`.
- The generation loop: `runtime::Generator` in `src/runtime/generator.hpp`, `src/runtime/generator.cpp`; `feed` is the helper that runs one forward at `pos_offset = length()` and samples.
- Tests: `tests/nn/kv_cache_test.cpp` compares a cached prefill and every decode step with the uncached forward, for the M1 model config and for two blocks with different KV shapes, compares cached decoding on CUDA with the CPU, and checks the training-mode, offset, cache-shape and context rejections and that a cached forward leaves nothing for backward. `tests/runtime/generator_test.cpp` compares the Generator with `generate_naive` (greedy on the CPU and CUDA, sampled on the CPU), and checks streaming, the slide, the long-prompt tail, the input checks and that `nn::KvCache` cannot be copied.
- The command line: `titan generate` in `apps/titan/commands.cpp` moves the model to its device before it constructs the Generator, and runs `generate_naive` with the same `SamplerConfig` for `--no-cache`. The measured decode speeds of both paths on both devices are in the README's "M3 result".
