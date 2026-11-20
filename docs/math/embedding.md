# Embedding

## What it computes

Turns integer token ids into vectors. The model owns a table with one learned row per vocabulary entry; looking up an id means copying its row. The backward pass sends the gradient of each looked-up vector back to the row it came from.

## Forward

For a table `E` of shape `(V, D)` and ids `i_0 … i_{M−1}`, each in `[0, V)`:

`y_k = E[i_k]`, for `k = 0 … M−1`

This is a gather: no arithmetic, only row copies. It equals the matrix product `onehot(i_k) · E`, without building the one-hot matrix. The ids are laid out as `out_shape` (for example `{B, T}`), so the result has shape `out_shape + {D}`, with ids and rows in the same row-major order.

## Backward derivation

The lookup is linear in `E`: `y_k = Σ_v onehot(i_k)_v E[v]`. With `dy_k = ∂L/∂y_k`:

`∂L/∂E[v] = Σ_k dy_k ∂y_k/∂E[v] = Σ_{k : i_k = v} dy_k`

So each row of `dy` is added to the table row it was gathered from, and a row that was gathered several times receives the sum of all its `dy` rows (a scatter-add). Rows never looked up get a zero contribution. The ids have no gradient: they are discrete.

`embedding_backward` accumulates into `dtable_acc` with `+=`, like every gradient in the library. The sum over the repeated ids of one row is collected in a `double` scratch the size of the table, in id order, and added to the accumulator once at the end. That keeps the result accurate when a frequent token appears thousands of times in a batch.

## Shapes

- `embedding(table, ids, out_shape)`: `table` is `(V, D)`; the product of `out_shape` equals `ids.size()`; `out_shape` has rank at most 3 so that the result, of rank `rank(out_shape) + 1`, is at most 4. Every id must be in `[0, V)`; the error names the offending id and its index.
- `embedding_backward(dtable_acc, ids, dy)`: `dtable_acc` has the table's shape `(V, D)`; `dy` has `ids.size() · D` elements and last dimension `D`.

## Mapping to code

- Public interface and precondition checks, including the id range check: `src/ops/embedding.hpp`, `src/ops/embedding.cpp`. A CUDA tensor fails with `ops::embedding: CUDA implementation arrives in M2` (likewise for `embedding_backward`).
- Reference implementation: `src/ops/cpu/embedding.cpp`.
- Tests: `tests/ops/embedding_test.cpp` covers the gather, scatter-add with repeated ids and accumulation, out-of-range ids, shape errors and the CUDA message.
