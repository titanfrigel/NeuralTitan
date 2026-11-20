# Softmax

## What it computes

Turns a row of arbitrary scores into a probability distribution: positive entries that sum to 1. Attention uses it to turn query-key scores into weights, and the language-model loss uses it on the logits. The causal variant additionally gives probability exactly 0 to the keys a query must not see, the later positions of the sequence.

## Forward

For a row `x` of length `n`:

`y_j = exp(x_j − m) / Σ_k exp(x_k − m)`, with `m = max_k x_k`

Subtracting `m` does not change the result, because the factor `exp(−m)` cancels between numerator and denominator, but it keeps every exponent at or below 0, so `exp` cannot overflow. The largest term is exactly 1, so the sum is at least 1 and never underflows to 0.

Causal softmax takes scores of shape `(…, Tq, Tk)`. Query row `i` may see key `j` when `j ≤ i + pos_offset`. For the visible keys the formula above applies, with `m` and the sum taken over the visible keys only; for the others `y_j = 0` exactly. The mask must be applied before the maximum: a masked score may be larger than every visible one, and if it set `m`, all visible exponents `exp(x_j − m)` could underflow to 0 and give 0/0. Skipping masked entries entirely also means their values, even infinities or NaN, cannot leak into the result. `pos_offset` is 0 in training; with a KV cache the queries are the last `Tq` of `Tk` positions, and `pos_offset = Tk − Tq` makes the last query see every key.

Row sums accumulate in `double`.

## Backward derivation

Let `y = softmax(x)` for one row. Then

`∂y_j/∂x_i = y_j (δ_ij − y_i)`

Derivation: `∂/∂x_i` of `exp(x_j)/S`, with `S = Σ_k exp(x_k)`, is `δ_ij exp(x_j)/S − exp(x_j) exp(x_i)/S² = y_j δ_ij − y_j y_i`.

With `dy = ∂L/∂y`:

`dx_i = Σ_j dy_j ∂y_j/∂x_i = Σ_j dy_j y_j (δ_ij − y_i) = y_i dy_i − y_i Σ_j dy_j y_j = y_i (dy_i − Σ_j dy_j y_j)`

So each row needs one dot product `Σ_j dy_j y_j`, then one elementwise pass. The same formula is valid for causal softmax: a masked entry has `y_j = 0`, so it contributes nothing to the dot product and its own `dx_j = 0`. The max subtraction has no gradient term because the result does not depend on `m`.

## Shapes

- `softmax(x)`: any rank 1 to 4, softmax over the last dimension, output shaped like `x`.
- `causal_softmax(scores, pos_offset)`: rank at least 2, shape `(…, Tq, Tk)`, requires `pos_offset ≥ 0` and `Tk ≥ Tq + pos_offset`; output shaped like `scores`.
- `softmax_backward(y, dy)`: `y` and `dy` have equal shapes; the result has the same shape.

## Mapping to code

- Public interface and precondition checks: `src/ops/softmax.hpp`, `src/ops/softmax.cpp`. A CUDA tensor fails with `ops::softmax: CUDA implementation arrives in M2` (likewise for the other two).
- Reference implementation: `src/ops/cpu/softmax.cpp`; one helper handles a row with a given number of visible entries, used with all entries visible for `softmax` and `i + pos_offset + 1` entries for `causal_softmax`.
- Tests: `tests/ops/softmax_test.cpp` covers normalization, stability, the mask with and without offset, masked inputs not leaking, error cases and finite-difference gradient checks of both variants.
