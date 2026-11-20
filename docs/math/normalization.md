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

- Public interface and precondition checks (device, parameter shapes, statistic sizes): `src/ops/norm.hpp`, `src/ops/norm.cpp`. A CUDA tensor fails with `ops::<name>: CUDA implementation arrives in M2`.
- Reference implementation: `src/ops/cpu/norm.cpp`. `rms_norm` and `layer_norm` allocate the saved statistics when the output pointer is non-null; the backward functions implement the formulas above, with `dot` for RMSNorm and `sum_g`, `sum_gx` for LayerNorm.
- Tests: `tests/ops/norm_test.cpp` covers known values, statistic shapes, zero mean and unit variance, accumulation into the parameter gradients, shape errors, and finite-difference gradient checks for `x`, `w` and `b`.
