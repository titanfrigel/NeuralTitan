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

- Public interface and precondition checks: `src/ops/softmax.hpp`, `src/ops/softmax.cpp`. The dispatch routes by the tensor's device to the CPU or the CUDA implementation.
- Reference implementation: `src/ops/cpu/softmax.cpp`; one helper handles a row with a given number of visible entries, used with all entries visible for `softmax` and `i + pos_offset + 1` entries for `causal_softmax`.
- Tests: `tests/ops/softmax_test.cpp` covers normalization, stability, the mask with and without offset, masked inputs not leaking, error cases and finite-difference gradient checks of both variants.
- CUDA kernels: `src/ops/cuda/softmax.cu`, declared in `src/ops/cuda/softmax.hpp`. `tests/ops/cuda/softmax_parity_test.cpp` compares all three ops with the CPU, including 70000 rows, row sums and a repeatability check.

## CUDA implementation

The kernels are in `src/ops/cuda/softmax.cu`. None uses an atomic, and every reduction order is fixed by the block size alone, so a repeated run on the same GPU gives bit-identical results. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary.

- `softmax`, `causal_softmax`: one block of 256 threads per row, with the rows on `gridDim.x` (the grid equals the row count; `gridDim.x` allows up to 2³¹ − 1 blocks, whereas `gridDim.y` stops at 65535). The row is processed in three passes over global memory, each a strided loop (thread `t` takes entries `t, t + 256, …`) followed by a shared-memory tree: the maximum, the sum of `exp(x − m)`, and the normalization `y = e · (1 / S)`. The tree is the one used by `row_max`: at each step thread `i < offset` combines its value with that of thread `i + offset`, with `offset` going 128, 64, …, 1. Every thread of the block reaches every `__syncthreads`. The entries from `valid` on are written as exactly 0.
- Causal mask: row `r` of the flattened `(…, Tq, Tk)` tensor is query `i = r mod Tq` and keeps `valid = i + pos_offset + 1` entries. Because `pos_offset ≥ 0`, column 0 is always kept, so no row is fully masked and `S ≥ 1`. The maximum and the sum run over the visible entries only, as on the CPU.
- `softmax_backward`: one block per row. Thread `t` accumulates `dy_j · y_j` over `j = t, t + 256, …` in `float`, the tree adds the 256 partial dots, and a second strided loop writes `dx_j = y_j (dy_j − dot)`.

Differences from the CPU:

- The CPU accumulates the row sum and the backward dot product in `double`; the GPU accumulates in `float` with the block tree. The two agree to about 10⁻⁷ relative, well inside the parity tolerance (`atol = 1e-5`, `rtol = 1e-4`).
- The CPU multiplies by `float(1 / S)` with `S` in `double`; the GPU divides in `float`. The difference is at the level of one `float` rounding.
- `expf` on the GPU differs from `std::exp` by at most a couple of ulps.
