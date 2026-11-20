# Attention

## What it computes

Attention lets every position of a sequence read from the earlier positions. This note covers the head layout ops that move tensors between the token-major layout of the projections and the head-major layout the score computation needs.

- `swap_dims_1_2` swaps dimensions 1 and 2 of a rank-4 tensor. Attention uses it to go from `(B, T, H, hd)` to `(B, H, T, hd)` and back.
- `repeat_kv` gives each query head its key/value head in grouped-query attention, where `n_kv_heads` divides `n_heads` and each KV head serves `n_rep = n_heads / n_kv_heads` query heads.

## Forward

`swap_dims_1_2`: `y[a, c, b, d] = x[a, b, c, d]`. Only the order of elements changes; each innermost row of `D` values moves as one block.

`repeat_kv`: `y[b, h, t, i] = x[b, ⌊h / n_rep⌋, t, i]`. Output heads `g·n_rep … g·n_rep + n_rep − 1` are copies of KV head `g`.

## Backward derivation

`swap_dims_1_2` is a permutation `P` of the elements, so `y = P x`, `∂L/∂x = Pᵀ ∂L/∂y`. Swapping the same two dimensions twice restores the original order, so `P` is its own inverse, and the inverse of a permutation is its transpose: `Pᵀ = P`. The backward is `swap_dims_1_2(dy)`.

`repeat_kv` uses each input element `x[b, g, t, i]` in exactly `n_rep` outputs, `y[b, g·n_rep + r, t, i]` for `r = 0 … n_rep − 1`, each with derivative 1. By the chain rule its gradient is the sum over those outputs:

`dx[b, g, t, i] = Σ_r dy[b, g·n_rep + r, t, i]`

`repeat_kv_backward` computes this sum in `double`.

## Shapes

- `swap_dims_1_2(x)`: `x` is `(A, B, C, D)`, the result is `(A, C, B, D)`.
- `repeat_kv(x, n_rep)`: `x` is `(B, Hkv, T, hd)`, the result is `(B, Hkv·n_rep, T, hd)`. Requires `n_rep ≥ 1`.
- `repeat_kv_backward(dy, n_rep)`: `dy` is `(B, H, T, hd)` with `n_rep` dividing `H`; the result is `(B, H / n_rep, T, hd)`.

## Mapping to code

- Public interface and precondition checks: `src/ops/layout.hpp`, `src/ops/layout.cpp`. A CUDA tensor fails with `ops::swap_dims_1_2: CUDA implementation arrives in M2` (likewise for the other two).
- Reference implementation: `src/ops/cpu/layout.cpp`.
- Tests: `tests/ops/layout_test.cpp` covers values, the inverse property of the swap, the group sums of the backward, finite-difference gradient checks of both ops and the error cases.
