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

- Public interface and precondition checks, including the id range check: `src/ops/embedding.hpp`, `src/ops/embedding.cpp`. The dispatch routes by the tensors' device to the CPU or the CUDA implementation. The ids live on the host and are checked there, on both devices, before anything is uploaded.
- Reference implementation: `src/ops/cpu/embedding.cpp`.
- Tests: `tests/ops/embedding_test.cpp` covers the gather, scatter-add with repeated ids and accumulation, out-of-range ids and shape errors.
- CUDA kernels: `src/ops/cuda/embedding.cu`, declared in `src/ops/cuda/embedding.hpp`. `tests/ops/cuda/embedding_parity_test.cpp` compares the forward and the accumulating backward with the CPU bit for bit, with repeated ids and a non-zero accumulator, and checks that an out-of-range id on CUDA fails with the CPU's message.

## CUDA implementation

The kernels are in `src/ops/cuda/embedding.cu`. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary. The ids arrive as a host array. The dispatch checks every one of them against `[0, V)` on the host first, because the kernels do not validate; an id out of range would otherwise read or write outside the table. Then they are copied with one `cudaMemcpy` into a device buffer from the caching allocator.

- `embedding`: one thread per output element, in a grid-stride loop over `N · D`. Element `i` is lookup `k = ⌊i / D⌋`, column `j = i mod D`, and copies `E[i_k, j]`. Neighbouring threads copy neighbouring columns of one row, so the reads and the writes are coalesced.
- `embedding_backward`: a scatter-add without atomics, in three steps.
  1. On the host, a counting sort groups the lookup positions by id into a compressed sparse row (CSR) layout: `unique_ids[U]` lists the `U` ids that occur, in increasing order; `offsets[U + 1]` delimits each one's slice of `positions[N]`; and a slice holds that id's positions `k` in ascending order (the positions are scanned in increasing `k`). This is `O(V + N)` work.
  2. The three arrays are uploaded.
  3. One thread per (occurring id `u`, column `j`), in a grid-stride loop over `U · D`, adds `dy[positions[p], j]` for `p` from `offsets[u]` to `offsets[u + 1] − 1`, in that order, into a `double`, then does `dE[unique_ids[u], j] += float(sum)`.

  No two threads write the same element, so no atomics are needed. The total work is `O(N · D)`, as on the CPU.

Why not atomics: the obvious kernel gives one thread to each element of `dy` and does `atomicAdd(&dE[i_k, j], dy[k, j])`. It needs no host work and no upload. But the order in which the additions land depends on thread timing, and floating-point addition is not associative, so a frequent id's row would change in its last bits from run to run. The library requires bit-identical reruns (overview section 6), and `float` atomics cannot accumulate in `double`. The CSR form trades a host pass over the ids and a small upload of `2U + 1 + N` integers for a fixed order and a `double` sum. Its cost is load balance: the thread of a token that appears thousands of times in a batch loops thousands of times, while a rare token's thread finishes at once. At the batch sizes of this library, that imbalance costs far less than the rest of a training step.

Differences from the CPU: none in value. The CPU sums each element in a `double` scratch in increasing `k` and adds the rounded sum once; the GPU does the same sums in the same order, so both devices agree to the last bit (the parity test uses tolerance 0). The only other difference is that the CPU also adds `0.0f` to the rows of ids that do not occur, which the GPU skips; that is the identity on every value except `−0.0`, which the CPU turns into `+0.0`.
