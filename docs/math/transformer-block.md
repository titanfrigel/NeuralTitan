# Transformer block

## What it computes

One layer of a decoder-only transformer: a causal [attention](attention.md) sublayer that lets each position read from the earlier ones, then a position-wise [feed-forward](mlp.md) sublayer, each wrapped in a pre-norm residual connection. The block keeps the model dimension: it maps `(B, T, dim)` to `(B, T, dim)`, so blocks stack.

Pre-norm means each sublayer sees a normalized copy of the residual stream `x` and adds its output back to the unnormalized `x`. The stream itself is never normalized inside a block, which keeps a direct identity path from the last layer to the embedding and makes deep pre-norm stacks easier to train than post-norm ones. A final norm after the last block (see [Model](model.md)) normalizes the stream once before the head.

`nn::BlockSpec` picks the variant:

- Llama style (SmolLM2): `NormType::RMSNorm`, `MlpType::SwiGLU`, `PositionEncoding::Rope`.
- GPT-2 style: `NormType::LayerNorm`, `MlpType::Gelu`, `PositionEncoding::None` (positions would come from a learned table, not part of M1).

## Forward

With `norm1`, `norm2` the two norms ([Normalization](normalization.md)), `attn` the attention and `mlp` the feed-forward block:

1. `a = norm1(x)`
2. `h = x + attn(a)`
3. `m = norm2(h)`
4. `y = h + mlp(m)`

## Backward derivation

Given `dy = ∂L/∂y`, each residual add `u = v + f(g(v))` sends the upstream gradient to `v` along two paths: directly, with Jacobian `I`, and through the branch, by the chain rule through `f` then `g`. The two contributions add:

`∂L/∂v = ∂L/∂u + g'ᵀ f'ᵀ ∂L/∂u`

Applied to step 4 and then step 2:

- `dh = dy + norm2.backward(mlp.backward(dy))`
- `dx = dh + norm1.backward(attn.backward(dh))`

Every child module accumulates its own parameter gradients during its `backward` call. The block owns no parameters of its own.

## Shapes

- `x`, `a`, `h`, `m`, `y` and their gradients: `(B, T, dim)`.
- Children and their parameters, with `<name>` the block name, e.g. `blocks.0`:
  - `<name>.norm1.weight` `(dim)` and, for LayerNorm, `<name>.norm1.bias` `(dim)`; likewise `norm2`.
  - `<name>.attn.q_proj.weight` `(n_heads·hd, dim)`, `k_proj` and `v_proj` `(n_kv_heads·hd, dim)`, `o_proj` `(dim, n_heads·hd)`.
  - SwiGLU: `<name>.mlp.gate_proj.weight` and `up_proj` `(hidden, dim)`, `down_proj` `(dim, hidden)`. GELU: `<name>.mlp.fc` `(hidden, dim)` plus bias, `proj` `(dim, hidden)` plus bias.

## Mapping to code

- `src/nn/block.hpp`, `src/nn/block.cpp`: `nn::TransformerBlock`, `nn::BlockSpec`, `nn::MlpType`, `nn::NormType`. The norms and the MLP are held as `std::unique_ptr<Module>` built from the spec; the attention is a direct member.
- The residual adds use `ops::add`, out of place: the children cache references to their inputs `x` and `h` until backward, so the block never adds into them in place.
- `backward` is the two lines of the derivation above.
- Tests: `tests/nn/block_test.cpp` gradient-checks a Llama-style block (RMSNorm, GQA with RoPE, SwiGLU) and a GPT-style block (LayerNorm, MHA without position encoding, GELU) with respect to the input and every parameter, and checks the parameter names.
