# Normalization (RMSNorm and LayerNorm)

## What it computes

Rescales each row of activations so its magnitude is under control, then applies a learned per-feature scale. Transformers put a normalization in front of every attention and feed-forward block; it keeps activations in a range where training is stable. RMSNorm (used by SmolLM2 and most modern decoders) only rescales by the root mean square of the row. LayerNorm also subtracts the row mean and adds a learned bias.

Both act independently on each row, that is on the last dimension of size `D`.

## Forward

RMSNorm, for a row `x` of length `D` and a learned `w`:

`r = 1 / √(mean(x²) + ε)`, with `mean(x²) = (1/D) Σ_k x_k²`

`y_i = x_i · r · w_i`

LayerNorm, for a row `x`, learned `w` and `b`:

`μ = (1/D) Σ_k x_k`

`σ² = (1/D) Σ_k (x_k − μ)²` (the biased variance, divided by `D`)

`s = 1 / √(σ² + ε)`

`x̂_i = (x_i − μ) · s`

`y_i = x̂_i · w_i + b_i`

With `ε = 0`, `x̂` has mean 0 and variance 1. `ε` only guards against division by zero for constant rows. The row statistics (`r`, or `μ` and `s`) accumulate in `double` and are saved by the forward pass for the backward pass.

## Backward derivation

Let `dy = ∂L/∂y` and `g_i = dy_i · w_i`, the gradient with respect to the normalized value.

### Parameters

`y_i = x̂_i · w_i + b_i` (LayerNorm) gives directly

`∂L/∂w_i = Σ_rows dy_i · x̂_i`, `∂L/∂b_i = Σ_rows dy_i`

The sum runs over all rows, because every row shares the same `w` and `b`. The code sums the rows in `double` and applies one float `+=` per parameter at the end. For RMSNorm `x̂_i = x_i · r`, so `∂L/∂w_i = Σ_rows dy_i · x_i · r`.

### RMSNorm input

The output is `y_i = w_i · x_i · r`, and `r` depends on every `x_k`. With `u = mean(x²) + ε`, `r = u^(−1/2)`:

`∂r/∂x_k = −½ u^(−3/2) · ∂u/∂x_k = −½ r³ · (2 x_k / D) = −r³ x_k / D`

Then, collecting the direct path through `x_i` and the path through `r`:

`dx_k = Σ_i g_i ∂(x_i r)/∂x_k = g_k r + Σ_i g_i x_i · ∂r/∂x_k = r g_k − (r³ x_k / D) Σ_i g_i x_i`

So one dot product `Σ_i g_i x_i` per row, then an elementwise pass. The second term is the part of `g` along the direction of `x`, scaled; it is the "projection term". Using `x̂ = x r`, it reads `dx = r (g − x̂ · mean(g ⊙ x̂))`.

### LayerNorm input

Here `x̂` depends on `x_k` both directly and through `μ` and `s`.

1. Through `μ`: `∂μ/∂x_k = 1/D`, so `∂(x_i − μ)/∂x_k = δ_ik − 1/D`.
2. Through `σ²`: `∂σ²/∂x_k = (2/D)(x_k − μ)`; the term from `μ` vanishes because `Σ_i (x_i − μ) = 0`. With `s = (σ² + ε)^(−1/2)`: `∂s/∂x_k = −½ s³ · (2/D)(x_k − μ) = −s³ (x_k − μ) / D = −s² x̂_k / D`.
3. Combine: `∂x̂_i/∂x_k = s (δ_ik − 1/D) + (x_i − μ) ∂s/∂x_k = s (δ_ik − 1/D − x̂_i x̂_k / D)`.
4. Contract with `g`: `dx_k = Σ_i g_i ∂x̂_i/∂x_k = s (g_k − (1/D) Σ_i g_i − (x̂_k / D) Σ_i g_i x̂_i)`.

That is `dx = s · (g − mean(g) − x̂ · mean(g ⊙ x̂))`: the gradient with its mean removed and its component along `x̂` removed, scaled by `s`. This matches the RMSNorm result with the extra mean-removal term, as expected since LayerNorm is mean-centering followed by an RMS-style rescale. Two row sums (`Σ g`, `Σ g x̂`) are needed, accumulated in `double`.

## Shapes

- `x`: rank 1 to 4, normalized over the last dimension `D`; output and `dx` have the shape of `x`.
- `weight`, `bias`, `dweight_acc`, `dbias_acc`: `{D}`.
- Saved statistics (`inv_rms`, `mean`, `inv_std`): the shape of `x` without its last dimension, or `{1}` for a rank-1 `x`; one value per row.
- `dy` has the shape of `x`.

## Mapping to code

- Public interface and precondition checks (device, parameter shapes, statistic sizes): `src/ops/norm.hpp`, `src/ops/norm.cpp`. The dispatch routes by the tensor's device to the CPU or the CUDA implementation; mixed devices throw.
- Reference implementation: `src/ops/cpu/norm.cpp`. `rms_norm` and `layer_norm` allocate the saved statistics when the output pointer is non-null; the backward functions implement the formulas above, with `dot` for RMSNorm and `sum_g`, `sum_gx` for LayerNorm.
- Tests: `tests/ops/norm_test.cpp` covers known values, statistic shapes, zero mean and unit variance, accumulation into the parameter gradients, shape errors, and finite-difference gradient checks for `x`, `w` and `b`.
- CUDA kernels: `src/ops/cuda/norm.cu`, declared in `src/ops/cuda/norm.hpp`. `tests/ops/cuda/norm_parity_test.cpp` compares forward outputs, saved statistics, `dx` and the accumulated `dweight`/`dbias` (2049 rows, non-zero accumulators) with the CPU.
- Modules: `src/nn/norm.hpp`, `src/nn/norm.cpp` wrap these ops as `RMSNorm` (parameter `<name>.weight`, initialized to 1) and `LayerNorm` (`<name>.weight` = 1, `<name>.bias` = 0). A training-mode forward saves `x` and the statistics; `backward` accumulates into the parameter gradients. Tests: `tests/nn/norm_module_test.cpp`.

## CUDA implementation

The kernels are in `src/ops/cuda/norm.cu`. None uses an atomic, and every reduction order is fixed by the block size alone, so a repeated run on the same GPU gives bit-identical results. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary.

- Forward and `dx`: one block of 256 threads per row, with the rows on `gridDim.x`. Thread `t` accumulates the row sums over `i = t, t + 256, …` in `float`, and the shared block tree (the one used by `row_max` and softmax) combines the 256 partials. RMSNorm needs one sum (`Σ x²` forward, `dot` backward). LayerNorm forward needs two in sequence (the mean, then the variance of the centered values, as on the CPU), and LayerNorm `dx` needs two (`Σ g`, `Σ g x̂`). A final strided loop writes the outputs. The saved statistics are written by thread 0 when the output pointer is non-null, with the same shape as on the CPU (`{1}` for a rank-1 `x`).
- `dweight`, `dbias`: a separate kernel with one thread per column `i`. It walks the rows `0 … R − 1` in order, accumulates `Σ dy x̂` (and `Σ dy`) in `double`, and then adds the sum once to the accumulator with `+=`. Neighbouring threads read neighbouring columns, so the loads coalesce. No two threads touch the same output, which is why no atomic is needed; the price is that only `D` threads work, which is acceptable for the small `D` of this project's models.
- `eps` is added to the mean square (or variance) before the square root, exactly as on the CPU.

Differences from the CPU:

- The CPU accumulates the row sums in `double`; the GPU accumulates them in `float` with the block tree, and computes `x̂`, `inv` and `dx` in `float`. The results agree to about 10⁻⁶ relative, inside the parity tolerance (`atol = 1e-5`, `rtol = 1e-4`).
- The column sums for `dweight` and `dbias` are in `double` on both devices, in the same row order, so they differ only by the `float` inputs' rounding in `x̂` (`mean` and `inv_std` are the saved `float` statistics on both).
