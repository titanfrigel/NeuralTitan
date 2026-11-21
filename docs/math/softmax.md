
- CUDA kernels: `src/ops/cuda/softmax.cu`, declared in `src/ops/cuda/softmax.hpp`. `tests/ops/cuda/softmax_parity_test.cpp` compares all three ops with the CPU, including 70000 rows and a repeatability check.

## CUDA implementation

The kernels are in `src/ops/cuda/softmax.cu`. None uses an atomic, and every reduction order is fixed by the block size alone, so a repeated run on the same GPU gives bit-identical results. See [CUDA execution](cuda-execution.md) for the grid and block vocabulary.

- `softmax`, `causal_softmax`: one block of 256 threads per row, with the rows on `gridDim.x` (the grid equals the row count; `gridDim.x` allows up to 2³¹ − 1 blocks, whereas `gridDim.y` stops at 65535). The row is processed in three passes over global memory, each a strided loop (thread `t` takes entries `t, t + 256, …`) followed by a shared-memory tree: the maximum, the sum of `exp(x − m)`, and the normalization `y = e · (1 / S)`. The tree is the one used by `row_max`: at each step thread `i < offset` combines its value with that of thread `i + offset`, with `offset` going 128, 64, …, 1. Every thread of the block reaches every `__syncthreads`. The entries from `valid` on are written as exactly 0.
- Causal mask: row `r` of the flattened `(…, Tq, Tk)` tensor is query `i = r mod Tq` and keeps `valid = i + pos_offset + 1` entries. Because `pos_offset ≥ 0`, column 0 is always kept, so no row is fully masked and `S ≥ 1`. The maximum and the sum run over the visible entries only, as on the CPU.
- `softmax_backward`: one block per row. Thread `t` accumulates `dy_j · y_j` over `j = t, t + 256, …` in `float`, the tree adds the 256 partial dots, and a second strided loop writes `dx_j = y_j (dy_j − dot)`.

Differences from the CPU:

- The CPU accumulates the row sum and the backward dot product in `double`; the GPU accumulates in `float` with the block tree. The two agree to about 10⁻⁷ relative, well inside the parity tolerance (`atol = 1e-5`, `rtol = 1e-4`).
- The CPU multiplies by `float(1 / S)` with `S` in `double`; the GPU divides in `float`. The difference is at the level of one `float` rounding.
- `expf` on the GPU differs from `std::exp` by at most a couple of ulps.
