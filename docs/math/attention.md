# Attention

## What it computes

Attention lets every position of a sequence read from the earlier positions. Each position emits a query `q`, a key `k` and a value `v`; a query scores every visible key by a dot product, the softmax of the scores weights the values, and the weighted sum is the position's output. Several heads do this in parallel on slices of the model dimension, each free to attend to different things, and an output projection mixes the heads back together.

`nn::Attention` is the causal, grouped-query (GQA) variant used by Llama-style models:

- **Causal**: position `t` sees only positions `j ≤ t`, so a language model cannot read the token it must predict.
- **Grouped-query**: there are `n_heads` query heads but only `n_kv_heads` key/value heads; each KV head serves a group of `n_rep = n_heads / n_kv_heads` query heads. With `n_kv_heads = n_heads` this is ordinary multi-head attention (MHA). Fewer KV heads shrink the K/V projections and, in M3, the KV cache.
- **RoPE** (optional): queries and keys are rotated by their absolute positions, so `q·k` depends only on the relative position. See [RoPE](rope.md). With `PositionEncoding::None` no rotation is applied.

Two head layout ops in `ops` move tensors between the token-major layout of the projections and the head-major layout the score computation needs:

- `swap_dims_1_2` swaps dimensions 1 and 2 of a rank-4 tensor: `(B, T, H, hd) ↔ (B, H, T, hd)`.
- `repeat_kv` gives each query head its KV head.

## Forward

Notation: `x` is `(B, T, dim)`, `H = n_heads`, `Hkv = n_kv_heads`, `hd = dim / H`, `n_rep = H / Hkv`, `s = 1/√hd`, `p₀ = ctx.pos_offset`.

1. Project: `Q = x W_qᵀ`, `K = x W_kᵀ`, `V = x W_vᵀ` (each a [Linear](linear.md) without bias).
2. Split heads: reshape `Q` to `(B, T, H, hd)` and `swap_dims_1_2` it to `(B, H, T, hd)`; likewise `K` and `V` with `Hkv` heads.
3. RoPE (when enabled): `Q ← rope(Q, p₀)`, `K ← rope(K, p₀)`; token `t` sits at position `p₀ + t`.
4. Expand KV heads: `K_r = repeat_kv(K, n_rep)`, `V_r = repeat_kv(V, n_rep)`, so query head `h` uses KV head `⌊h / n_rep⌋`.
5. Scores, per batch element and head: `S = s · Q K_rᵀ`, shape `(T, T)`, `S_tj = s Σ_i Q_ti K_ji`.
6. Causal softmax: `P = causal_softmax(S, p₀)`; row `t` puts probability 0 on keys `j > t + p₀`.
7. Values: `O = P V_r`, `(B, H, T, hd)`; `swap_dims_1_2` and reshape merge the heads back to `(B, T, dim)`.
8. Output projection: `y = O W_oᵀ`.

### Why scale by 1/√hd

If the entries of `q` and `k` are independent with mean 0 and variance 1, `q·k` is a sum of `hd` such products and has variance `hd`. Without scaling, larger heads push the scores into the saturated region of the softmax, where one key takes almost all the weight and the gradient vanishes. Multiplying by `1/√hd` brings the variance back to 1 whatever the head size.

### The causal mask

Within one sequence, position `t` must not depend on later tokens. The mask is applied inside `causal_softmax` before the row maximum, by skipping the masked keys rather than adding `−∞` (see [Softmax](softmax.md) for why the order matters). In M1 the full sequence is processed at once, `p₀ = 0`, and the visible keys of row `t` are `0 … t`. A nonzero `p₀` only makes sense with a KV cache (M3), where the `T` queries are the last of a longer key sequence; without one, `causal_softmax` rejects it because `Tk = T < T + p₀`.

### The layout ops

`swap_dims_1_2`: `y[a, c, b, d] = x[a, b, c, d]`. Only the order of elements changes; each innermost row of `D` values moves as one block.

`repeat_kv`: `y[b, h, t, i] = x[b, ⌊h / n_rep⌋, t, i]`. Output heads `g·n_rep … g·n_rep + n_rep − 1` are copies of KV head `g`.

`kv_write` and `kv_read` lay out the KV cache used during generation. A cache is a `(B, H, Tmax, hd)` tensor whose first rows hold the keys or values of the positions already processed. `kv_write(cache, x, pos)` sets `cache[b, h, pos + t, i] = x[b, h, t, i]` for `t = 0 … T − 1`, in place; every other row keeps its value. `kv_read(cache, len)` returns a new tensor with `y[b, h, t, i] = cache[b, h, t, i]` for `t < len`. Both only copy, so they have no backward pass: they are inference only, and training never uses a cache.

## Backward derivation

Given `dy = ∂L/∂y`, each step is reversed in turn. For a product `C = A B` the gradients are `dA = dC Bᵀ` and `dB = Aᵀ dC` (see [Matmul](matmul.md)).

- **Step 8, output projection**: `o_proj.backward(dy)` adds `dW_o` and returns `dO` `(B, T, dim)`.
- **Step 7, values**: the merge is a reshape and a swap, both permutations, so `dO` is reshaped to `(B, T, H, hd)` and swapped to `(B, H, T, hd)`. Then from `O = P V_r`: `dP = dO V_rᵀ`, `(B, H, T, T)`, and `dV_r = Pᵀ dO`, `(B, H, T, hd)`.
- **Step 6, softmax**: per row, `dS_t = P_t ⊙ (dP_t − Σ_j dP_tj P_tj)` (`softmax_backward`). Masked entries have `P_tj = 0`, so they receive no gradient and the mask needs no separate backward.
- **Step 5, scores**: `S = s Q K_rᵀ` gives `dQ = s · dS K_r` and `dK_r = s · dSᵀ Q`. The code scales `dS` by `s` once and uses it for both.
- **Step 4, KV expansion**: `K[b, g]` is used by the `n_rep` query heads `g·n_rep + r`, each with derivative 1, so its gradient is the sum over the group, `dK[b, g, t, i] = Σ_r dK_r[b, g·n_rep + r, t, i]`, and likewise for `dV`. This is `repeat_kv_backward`, summed in `double`.
- **Step 3, RoPE**: rotating by angle `a` is an orthogonal map `R`, so `d(input) = Rᵀ d(output)`, the rotation by `−a`: `dQ ← rope_backward(dQ, p₀)`, `dK ← rope_backward(dK, p₀)`.
- **Step 2, split heads**: swap back to `(B, T, H, hd)` and reshape to `(B, T, H·hd)`, the inverse permutation.
- **Step 1, projections**: `x` feeds all three projections, so `dx = q_proj.backward(dQ) + k_proj.backward(dK) + v_proj.backward(dV)`; each call also adds its weight gradient.

`swap_dims_1_2` is a permutation `P` of the elements, so `y = P x` and `∂L/∂x = Pᵀ ∂L/∂y`. Swapping the same two dimensions twice restores the original order, so `P` is its own inverse, and the inverse of a permutation is its transpose: `Pᵀ = P`. Its backward is `swap_dims_1_2(dy)`.

## Shapes

Through the layer, with `B` batch, `T` sequence length:

| Tensor | Shape |
| --- | --- |
| `x`, `y` | `(B, T, dim)` |
| `W_q` | `(H·hd, dim)` = `(dim, dim)` |
| `W_k`, `W_v` | `(Hkv·hd, dim)` |
| `W_o` | `(dim, H·hd)` = `(dim, dim)` |
| `Q` after split and RoPE | `(B, H, T, hd)` |
| `K`, `V` after split and RoPE | `(B, Hkv, T, hd)` |
| `K_r`, `V_r` | `(B, H, T, hd)` |
| `S`, `P` | `(B, H, T, T)` |
| `O` before merge | `(B, H, T, hd)` |
| `O` after merge | `(B, T, dim)` |

Layout ops:

- `swap_dims_1_2(x)`: `x` is `(A, B, C, D)`, the result is `(A, C, B, D)`.
- `repeat_kv(x, n_rep)`: `x` is `(B, Hkv, T, hd)`, the result is `(B, Hkv·n_rep, T, hd)`. Requires `n_rep ≥ 1`.
- `repeat_kv_backward(dy, n_rep)`: `dy` is `(B, H, T, hd)` with `n_rep` dividing `H`; the result is `(B, H / n_rep, T, hd)`.
- `kv_write(cache, x, pos)`: `cache` is `(B, H, Tmax, hd)` and `x` is `(B, H, T, hd)`; `B`, `H` and `hd` must match, `pos ≥ 0` and `pos + T ≤ Tmax`. Both tensors must be on the same device. Nothing is returned.
- `kv_read(cache, len)`: `cache` is `(B, H, Tmax, hd)`, the result is `(B, H, len, hd)`. Requires `1 ≤ len ≤ Tmax`.

The constructor requires `H ≥ 1`, `Hkv ≥ 1`, `dim % H = 0`, `H % Hkv = 0` and, with RoPE, an even `hd` (RoPE rotates pairs `(i, i + hd/2)`).

### Why the score matrix is materialized in v1

The layer stores `P`, a full `(B, H, T, T)` tensor, for the backward pass. That costs `O(T²)` memory per head, which fused kernels such as FlashAttention avoid by computing the softmax in tiles and recomputing it in the backward pass. M1 keeps the plain form on purpose: every step is an existing op with a CPU reference and its own tests, the backward is a direct transcription of the chain rule above, and at the M1 sizes (`T ≤ 128`) the matrix is small. A fused kernel would be an optimization of this reference, kept beside it, not a replacement.

## Mapping to code

- `nn::Attention`, `nn::PositionEncoding`: `src/nn/attention.hpp`, `src/nn/attention.cpp`. `forward` follows steps 1-8 with the step numbers in comments and caches `Q`, `K_r` (after RoPE), `V_r`, `P` and `p₀`; `backward` reverses them. The helpers `split_heads` and `merge_heads` do the reshape and swap of steps 2 and 7.
- Layout ops, public interface and checks: `src/ops/layout.hpp`, `src/ops/layout.cpp`. The dispatch routes by the tensor's device to the CPU or the CUDA implementation. Reference implementation: `src/ops/cpu/layout.cpp`; CUDA kernels: `src/ops/cuda/layout.cu`. `kv_write` and `kv_read` live in the same files.
- Scores and values use `ops::matmul` with its transpose flags, the scale `ops::scale_`, the mask and normalization `ops::causal_softmax` and `ops::softmax_backward`, and the rotation `ops::rope` and `ops::rope_backward`.
- Tests: `tests/ops/cuda/layout_parity_test.cpp` compares the three layout ops with the CPU for `n_rep` 1 and 3. `tests/ops/layout_test.cpp` covers the layout ops (values, the inverse property of the swap, group sums, gradient checks, error cases) and the cache ops (write then read, untouched rows, overflow and shape errors); the parity file also compares `kv_write` and `kv_read` with the CPU and checks that a CPU `x` with a CUDA cache is rejected. `tests/nn/attention_test.cpp` checks causality, the output of a one-head case with identity projections against an explicit `softmax(q kᵀ/√2) v`, finite-difference gradients of the input and all four weights for MHA with RoPE, GQA with RoPE and GQA without position encoding, and the head-configuration errors.

## CUDA implementation

The layout kernels are in `src/ops/cuda/layout.cu`. None uses an atomic, and the only sum has a fixed order, so results are deterministic. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary. Every kernel is one thread per output element in a grid-stride loop; the thread decodes its output index with `/` and `%` and reads the input element or elements it maps to.

- `swap_dims_1_2`: output index `o` is decoded as `(a, c, b, d)` in row-major order of the `(A, C, B, D)` output, and the thread copies `x[a, b, c, d]`.
- `repeat_kv`: `o` is split into a flat `(batch, output head)` index and an offset inside the `(T, hd)` head. Output head `h` reads KV head `⌊h / n_rep⌋` of the same batch entry.
- `repeat_kv_backward`: one thread per element of `dx`. It adds the `n_rep` copies `dy[b, g·n_rep + r]` for `r = 0, 1, …, n_rep − 1`, in that order, so no atomics are needed. The sum is kept in `double` and rounded once, as on the CPU, so the CPU and the GPU agree to the last `float` bit for any `n_rep`.
- `kv_write`: one thread per element of `x`. `o` is decoded as `(bh, t, i)`, with `bh` the flat `(batch, head)` index, and the thread stores `x[o]` at `cache[(bh·Tmax + pos + t)·hd + i]`. Distinct elements of `x` map to distinct cache elements, so threads never write the same location.
- `kv_read`: one thread per element of the result; `o` is decoded as `(bh, t, i)` and the thread copies `cache[(bh·Tmax + t)·hd + i]`.

Differences from the CPU: the copies are exact and the backward sum has the CPU's order and precision, so the results are identical. Only the memory access pattern differs, which affects speed, not values.
