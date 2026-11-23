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

- Public interface and precondition checks: `src/ops/cross_entropy.hpp`, `src/ops/cross_entropy.cpp`. The dispatch routes by the logits' device to the CPU or the CUDA implementation. The targets live on the host and are checked there, on both devices, before anything is uploaded.
- Reference implementation: `src/ops/cpu/cross_entropy.cpp`; one pass per row computes `m`, `lse` and the loss term, then the gradient row.
- `src/nn/loss.hpp`, `src/nn/loss.cpp`: `nn::CrossEntropyLoss` flattens logits `(B, T, V)` to `(B·T, V)`, calls `ops::cross_entropy` with a gradient output, and returns that gradient from `backward()` reshaped to `(B, T, V)`. Its tests are in `tests/nn/block_test.cpp`.
- Tests: `tests/ops/cross_entropy_test.cpp` covers the uniform loss `log 65`, large logits, gradient rows summing to 0, a finite-difference gradient check with the loss itself as the scalar, and error cases.
- CUDA kernels: `src/ops/cuda/cross_entropy.cu`, declared in `src/ops/cuda/cross_entropy.hpp`. `tests/ops/cuda/cross_entropy_parity_test.cpp` compares the loss and the gradient with the CPU for `V = 65` and `V = 1000`, for 70000 rows and for logits of ±1000, checks repeatability, and checks that an out-of-range target on CUDA fails with the CPU's message.

## CUDA implementation

The kernels are in `src/ops/cuda/cross_entropy.cu`. None uses an atomic, and every reduction order is fixed by the block size alone, so a repeated run on the same GPU gives bit-identical results. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary. The targets arrive as a host array; the dispatch checks them on the host first, because the kernels do not validate, then they are copied with one `cudaMemcpy` into a device buffer from the caching allocator. When `dlogits_out` is non-null it is assigned a new CUDA tensor, replacing whatever it held (`nn::CrossEntropyLoss` passes a 1-element CPU placeholder).

- Row kernel: one block of 256 threads per row, with the rows on `gridDim.x` (`gridDim.y` would stop at 65535 rows). Thread `t` takes logits `t, t + 256, …` of the row. A shared-memory tree, the `block_reduce` of `common.cuh`, gives the row maximum `m`. A second strided loop and tree give `S = Σ_j exp(z_j − m)` in `float`. Then `lse = m + log S`, and thread 0 writes the row loss `lse − z_t` as a `double` into a scratch array of `N` entries. When a gradient is requested, a third strided loop writes `(exp(z_j − lse) − δ_jt) · (1/N)`.
- Mean: a second kernel of one block sums the `N` row losses in `double`. Thread `t` adds rows `t, t + 256, …` in order, and the same fixed tree combines the 256 partial sums. The host copies that one `double` back, which waits for both kernels, and returns `float(total · (1/N))`.

Differences from the CPU:

- The CPU computes `m`, `S`, `lse` and each probability in `double`; the GPU computes them in `float`, as the other row reductions do. The two agree to about 10⁻⁷ relative, well inside the parity tolerance (`atol = 1e-5`, `rtol = 1e-4`) for the loss and the gradient.
- Both devices sum the row losses in `double`, but in a different order: the CPU in row order, the GPU in the strided-then-tree order above. Each order is fixed, so each device is repeatable.
- The GPU multiplies the gradient by `float(1/N)` in `float`; the CPU multiplies by `1/N` in `double` before rounding. The difference is one `float` rounding.
