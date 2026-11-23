# Reductions

## What it computes

Operations that collapse one or more dimensions of a tensor into a smaller result: total sums, a sum of squares, the maximum along the last dimension, a sum over all leading dimensions, and a finiteness check. They are the building blocks of losses, gradient norms, softmax stabilisation and bias gradients.

## Forward

- `sum`: `s = Σᵢ xᵢ` over all elements.
- `sum_squares`: `q = Σᵢ xᵢ²` over all elements.
- `row_max`: for `x` viewed as `R` rows of length `C` (the last dimension), `mᵣ = maxⱼ x[r, j]`. Softmax subtracts this value before exponentiating so that `exp` cannot overflow.
- `sum_leading`: for `x` viewed as `R × C`, `bⱼ = Σᵣ x[r, j]`.
- `all_finite`: true when every element is neither NaN nor ±∞.

Every sum accumulates in `double` and is rounded to `float` only when stored in a tensor. A `float` accumulator loses low-order bits once the running total is large compared to the addends: summing 10⁷ copies of `0.1f` in `float` drifts visibly, while the `double` accumulator stays within rounding error of the exact result.

## Backward derivation

These ops have no backward function of their own; the modules that use them derive their gradients. The one gradient they implement is a consequence of broadcasting: a bias `b` of shape `(C)` is added to every row of `y = x + b`, so `∂L/∂bⱼ = Σᵣ dy[r, j]`, which is exactly `sum_leading(dy)`. `row_max` is only used to stabilise softmax, where the shift cancels out of the result and needs no gradient.

## Shapes

- `sum`, `sum_squares`: any shape to a scalar `double`.
- `row_max`: shape `(d₁, …, d_k)` gives `(d₁, …, d_{k−1})`; a rank-1 input `(C)` gives `(1)`.
- `sum_leading`: shape `(d₁, …, d_k)` gives `(d_k)`.
- `all_finite`: any shape to `bool`.

## Mapping to code

- Public interface and precondition checks: `src/ops/reduce.hpp`, `src/ops/reduce.cpp`. A CUDA tensor fails with `ops::<name>: CUDA implementation arrives in M2`.
- CPU reference loops: `src/ops/cpu/reduce.cpp`, one function per formula above.
- Tests: `tests/ops/reduce_test.cpp` checks exact values, output shapes, the double accumulation and the finiteness check.
