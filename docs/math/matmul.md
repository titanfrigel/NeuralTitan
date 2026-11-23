# Matmul

## What it computes

The matrix product, the operation that dominates the cost of a transformer: every linear layer and both attention products are matmuls. `ops::matmul` also handles batches of independent products and the transposed forms that the backward pass needs.

## Forward

For `op(a)` of shape `(M, K)` and `op(b)` of shape `(K, N)`:

`C[i][j] = Σ_k op(a)[i][k] · op(b)[k][j]`

Tensors are row-major, so element `(r, c)` of a stored `(R, C)` matrix is at offset `r·C + c`. The transpose flags change only how the stored matrix is read:

| `transpose_a` | `transpose_b` | stored `a` | stored `b` | `op(a)[i][k]` | `op(b)[k][j]` |
| --- | --- | --- | --- | --- | --- |
| false | false | `(M, K)` | `(K, N)` | `a[i·K + k]` | `b[k·N + j]` |
| true | false | `(K, M)` | `(K, N)` | `a[k·M + i]` | `b[k·N + j]` |
| false | true | `(M, K)` | `(N, K)` | `a[i·K + k]` | `b[j·K + k]` |
| true | true | `(K, M)` | `(N, K)` | `a[k·M + i]` | `b[j·K + k]` |

Batching: for rank 3 or 4 the leading dimensions are batch dimensions, equal in `a` and `b`. With `B` the product of the batch dimensions, the tensors are viewed as `B` consecutive matrices and product `t` multiplies matrix `t` of `a` by matrix `t` of `b`. The result has the same batch dimensions.

`matmul_acc` computes `out += op(a)·op(b)` into an `out` that is already shaped `(…, M, N)`. It accumulates weight gradients across uses of the same weight.

## Backward derivation

Let `C = A·B` (no transposes) and `dC = ∂L/∂C`. Each `A[i][k]` affects only row `i` of `C`:

`∂L/∂A[i][k] = Σ_j dC[i][j] · ∂C[i][j]/∂A[i][k] = Σ_j dC[i][j] · B[k][j]`

which is `(dC·Bᵀ)[i][k]`. Likewise each `B[k][j]` affects only column `j`:

`∂L/∂B[k][j] = Σ_i dC[i][j] · A[i][k] = (Aᵀ·dC)[k][j]`

So `dA = dC·Bᵀ` (`transpose_b = true`) and `dB = Aᵀ·dC` (`transpose_a = true`), with no explicit transposed copy needed at the call site. The transposed cases of the other forms follow by applying these to `op`: for `Y = A·Bᵀ` (the linear layer `y = x·Wᵀ`), `dA = dY·B` and `dB = dYᵀ·A`. Batched products differentiate per batch, independently.

## Shapes

- Inputs `a`, `b`: equal rank 2 to 4, equal leading dimensions.
- `op(a)`: `(…, M, K)`; `op(b)`: `(…, K, N)`; result `(…, M, N)`.
- Mismatches throw `titan::Error` naming both shapes and both flags.

## Threading, determinism and accumulation

Loop order (`cpu::matmul_acc`): for each output row `(t, i)` (batch index and row of `op(a)`) the kernel runs `k` in the middle and `j` innermost, the i-k-j order. The innermost loop adds `op(a)[i][k] · op(b)[k][·]`, a whole row of `op(b)`, to a row of partial sums, so it reads and writes contiguous memory and the compiler can vectorize it. If `b` is transposed, `op(b)` is first packed into a contiguous `(K, N)` copy, so the inner loop is the same for all four cases. `op(a)` is read through index arithmetic, one scalar per `(i, k)`.

Accumulation order: `K` is split into blocks of 64. Within a block, terms are added in float in ascending `k`, starting from zero. Each block sum is then added to a per-element `double` total, blocks in ascending order, and the final total is rounded to float and added to `out`. This is a function of the shapes only.

Threading: `parallel_for` splits the `B·M` output rows into contiguous chunks, one per thread. A row is computed completely by one thread, from read-only inputs, into its own slice of `out`, with its own scratch buffers. No two threads touch the same output element and no sums are combined across threads, so the result is bit-identical for any thread count.

`cpu::matmul_naive` is the reference: a plain triple loop with a single `double` accumulator per element. The fast kernel agrees with it up to float rounding; the tests require 1e-4 relative.

## Mapping to code

- Public interface and precondition checks: `src/ops/matmul.hpp`, `src/ops/matmul.cpp`. A CUDA tensor fails with `ops::matmul: CUDA implementation arrives in M2`.
- Fast kernel and reference: `src/ops/cpu/matmul.cpp` (`matmul_acc`, `matmul_naive`).
- Tests: `tests/ops/matmul_test.cpp` covers values, all four transpose cases, batching, the naive reference on awkward shapes, thread-count determinism, accumulation, error messages and the gradients above through `tests/support/gradcheck.hpp`.

## GPU kernels and roofline (M2)

Placeholder. M2 adds the naive, tiled and cuBLAS GEMMs with a roofline analysis of each.
