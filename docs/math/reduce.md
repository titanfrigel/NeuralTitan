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

- Public interface and precondition checks: `src/ops/reduce.hpp`, `src/ops/reduce.cpp`. The dispatch routes by the tensor's device to the CPU or the CUDA implementation.
- CPU reference loops: `src/ops/cpu/reduce.cpp`, one function per formula above.
- CUDA kernels: `src/ops/cuda/reduce.cu`, declared in `src/ops/cuda/reduce.hpp`.
- Tests: `tests/ops/reduce_test.cpp` checks exact values, output shapes, the double accumulation and the finiteness check. `tests/ops/cuda/reduce_parity_test.cpp` compares each CUDA op with the CPU one.

## CUDA implementation

The kernels are in `src/ops/cuda/reduce.cu`. None uses a floating-point `atomicAdd`, and every reduction order is fixed by the launch shape alone, so a repeated run on the same GPU gives bit-identical results. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary.

- `sum`, `sum_squares`: two passes. In pass 1 the grid has `min(ceil(n / 256), 65535)` blocks of 256 threads. Thread `t` accumulates elements `t, t + stride, t + 2·stride, …` in a `double`, in index order (a grid-stride loop, `stride` being the total number of threads). Each block then adds its 256 accumulators in shared memory with a fixed tree: at each step thread `i < offset` adds the value of thread `i + offset`, with `offset` going 128, 64, …, 1. The block writes one `double` partial. In pass 2 a single block sums the partials: thread `t` adds partials `t, t + 256, …` in order, then the same tree combines the threads. The result is copied to the host as a `double`, which also waits for the GPU.
- `row_max`: one block of 256 threads per row, with the rows on `gridDim.x`, so a row count above 65535 (the limit of `gridDim.y`) needs no special case. Thread `t` takes the maximum of the row's elements `t, t + 256, …`, then a tree takes the maximum over the threads. A maximum is exact, so the order cannot change the value.
- `sum_leading`: one thread per column (a grid-stride loop over columns). It adds the column's `R` rows in order into a `double`, exactly like the CPU loop, and rounds to `float` at the end. Neighbouring threads read neighbouring addresses of a row, so the loads are coalesced. A column is summed by one thread, so with few columns and many rows the launch has little parallelism; that is acceptable for a bias gradient.
- `all_finite`: each thread scans its grid-stride elements, and any thread that sees a NaN or an infinity does `atomicOr` of 1 on an integer flag. All writers store the same value and integer atomics are exact, so the outcome does not depend on thread timing.

Differences from the CPU:

- `sum` and `sum_squares` add the same numbers in a different order (partial sums, then a tree) than the CPU's single left-to-right loop. Both accumulate in `double`, so the two results agree to about 10⁻¹⁶ relative, and the parity test allows 10⁻⁹.
- `row_max` on a row containing NaN is unspecified on both devices: the CPU starts from the row's first element and the GPU from negative infinity, and every comparison with a NaN is false. Softmax inputs are finite.
- `sum_leading` adds in the same order as the CPU, so the two agree up to the final rounding to `float`.
