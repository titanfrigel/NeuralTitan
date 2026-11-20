# Linear

## What it computes

A learned affine map applied to the last dimension: every row of the input is multiplied by a weight matrix and, optionally, shifted by a bias. Projections in attention and in the feed-forward blocks, and the output head, are all linear layers.

## Forward

For a row `x` of length `in`, a weight `W` of shape `(out, in)` and a bias `b` of length `out`:

`y = W x + b`, that is `y_j = Σ_k W[j][k] x_k + b_j`

The input may have any rank whose last dimension is `in`. The leading dimensions are flattened into `N` rows, so `X` is `(N, in)` and

`Y = X Wᵀ + 1 bᵀ`

with `1` a column of `N` ones. The result is reshaped back to the input's shape with the last dimension replaced by `out`. The weight is stored `(out, in)`, the layout of Hugging Face checkpoints, so the product uses the transposed-B flag of `matmul`.

## Backward derivation

With `L` the loss and `dY = ∂L/∂Y` of shape `(N, out)`:

- Input: `Y_{nj} = Σ_k X_{nk} W_{jk}`, so `∂L/∂X_{nk} = Σ_j dY_{nj} W_{jk}`, that is `dX = dY W`.
- Weight: `∂L/∂W_{jk} = Σ_n dY_{nj} X_{nk}`, that is `dW = dYᵀ X`.
- Bias: `b_j` is added to every row, so `∂L/∂b_j = Σ_n dY_{nj}`, that is `db` is the sum of `dY` over the rows.

The parameter gradients are added (`+=`) to `Parameter::grad`, so calling `backward` twice without `zero_grad` gives twice the gradient; this is what gradient accumulation across micro-batches relies on. `dX` is reshaped to the input's shape.

## Shapes

- Input `x`: `(..., in)`, rank 1 to 4. Output: the same with `out` as last dimension.
- `<name>.weight`: `(out, in)`. `<name>.bias`: `(out)`, present only with bias.
- `dy` has the output's shape; `dx` has the input's.

## Mapping to code

- `src/nn/linear.hpp`, `src/nn/linear.cpp`: `Linear::forward` calls `ops::matmul(X, W, false, true)` and adds the bias row by row; `Linear::backward` calls `ops::matmul_acc(dW, dY, X, true, false)`, `ops::sum_leading` and `ops::matmul(dY, W)`.
- The input is cached by a training-mode forward. Calling `backward` after an inference-mode forward throws `backward called without a training-mode forward`.
- Tests: `tests/nn/linear_test.cpp` covers known values, gradient checks (no bias, bias with a rank-3 input) and accumulation.
