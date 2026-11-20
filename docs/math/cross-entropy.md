# Cross-entropy

## What it computes

The training loss of a language model: how surprised the model is by the true next token. Given the logits for each position, it is the average negative log-probability that the softmax of the logits gives to the correct token. The op fuses the softmax and the loss, and produces the gradient with respect to the logits in the same pass.

## Forward

For logits `z` of shape `(N, V)` and targets `t_i ∈ [0, V)`:

`p_ij = exp(z_ij) / Σ_k exp(z_ik)`

`L = −(1/N) Σ_i log p_{i,t_i}`

Computed directly, `log(softmax(z))` can overflow in `exp` or underflow to `log 0`. The fused form avoids both. With `m_i = max_k z_ik` and the log-sum-exp `lse_i = m_i + log Σ_k exp(z_ik − m_i)`:

`log p_{i,t_i} = z_{i,t_i} − lse_i`

so `L = (1/N) Σ_i (lse_i − z_{i,t_i})`. Every exponent is at most 0 and the largest term of the sum is exactly 1, so the sum is at least 1 and its logarithm is finite, for any logits. The row sums, the logarithm and the mean over rows are computed in `double`; only the result is rounded to `float`. Zero logits with `V` classes give exactly `log V`, the loss of a model that guesses uniformly.

## Backward derivation

Fix one row and drop the index `i`. With `S = Σ_k exp(z_k)` and target `t`:

`ℓ = −log p_t = log S − z_t`

`∂ℓ/∂z_j = exp(z_j)/S − δ_jt = p_j − δ_jt`

since `∂ log S/∂z_j = exp(z_j)/S`. In vector form `∂ℓ/∂z = softmax(z) − onehot(t)`. The loss averages over `N` rows, and row `i` only depends on its own logits, so

`∂L/∂z_ij = (p_ij − δ_{j,t_i}) / N`

This is the whole backward pass: no separate softmax Jacobian is needed, because the Jacobian of the softmax and the derivative of the logarithm cancel into this one subtraction (compare the softmax note). Each gradient row sums to 0, because `p` sums to 1 and the one-hot vector sums to 1. The probabilities are computed as `exp(z_ij − lse_i)`, which is at most 1 and stable for the same reason as the forward pass.

## Shapes

- `logits`: `(N, V)`; `targets`: `N` integers, each in `[0, V)`; the error names the offending target and its index.
- Returns the scalar mean loss.
- `dlogits_out`, when non-null, is assigned a new `(N, V)` tensor holding `(softmax − onehot)/N`; pass `nullptr` for evaluation.

## Mapping to code

- Public interface and precondition checks: `src/ops/cross_entropy.hpp`, `src/ops/cross_entropy.cpp`. A CUDA tensor fails with `ops::cross_entropy: CUDA implementation arrives in M2`.
- Reference implementation: `src/ops/cpu/cross_entropy.cpp`; one pass per row computes `m`, `lse` and the loss term, then the gradient row.
- `src/nn/loss.hpp`, `src/nn/loss.cpp`: `nn::CrossEntropyLoss` flattens logits `(B, T, V)` to `(B·T, V)`, calls `ops::cross_entropy` with a gradient output, and returns that gradient from `backward()` reshaped to `(B, T, V)`. Its tests are in `tests/nn/block_test.cpp`.
- Tests: `tests/ops/cross_entropy_test.cpp` covers the uniform loss `log 65`, large logits, gradient rows summing to 0, a finite-difference gradient check with the loss itself as the scalar, and error cases.
