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

Loop order (`cpu::matmul_acc`): the work is split into items, each one output row `(t, i)` (batch index and row of `op(a)`) restricted to one tile of columns `[j0, j0 + w)`. Within an item the kernel runs `k` in the middle and `j` innermost, the i-k-j order. The innermost loop adds `op(a)[i][k] · op(b)[k][j0…j0+w)`, a piece of a row of `op(b)`, to a row of partial sums, so it reads and writes contiguous memory and the compiler can vectorize it. If `b` is transposed and `M ≥ 4`, `op(b)` is first packed into a contiguous `(K, N)` copy, the packing split over the threads by column, so the inner loop is the same for all four cases. With `M < 4` (decoding has `M = 1`) packing would cost as much as the products themselves, so a transposed `b` is read in place: row `j` of `b` is column `j` of `op(b)` and is contiguous in `k`, and each element is a dot product along it. `op(a)` is read through index arithmetic, one scalar per `(i, k)`.

Tile width: with `R = B·M` output rows and `T` threads, the kernel wants `4T` items, so each row gets `⌈4T / R⌉` tiles, at most `⌈N / 16⌉` (a tile keeps at least 16 columns so that the inner loop still vectorizes), and the tile width is `w = ⌈N / tiles⌉`. With many rows a row is one tile, which is the row-parallel kernel of earlier milestones; with `M = 1` the columns are what is split, so decoding uses every thread instead of one.

Accumulation order: `K` is split into blocks of 64. Within a block, terms are added in float in ascending `k`, starting from zero. Each block sum is then added to a per-element `double` total, blocks in ascending order, and the final total is rounded to float and added to `out`. This is a function of the shapes only.

Threading: `parallel_for` splits the items into contiguous chunks, one per thread. An item is computed completely by one thread, from read-only inputs, into its own slice of `out`, with its own scratch buffers. No two threads touch the same output element and no sums are combined across threads, and every element follows the accumulation order above whatever its tile or path (packed, in place), so the result is bit-identical for any thread count and to `cpu::matmul_acc_reference`, a serial loop per element with the same order, which `MatmulCpu.TiledMatchesReference` compares bit for bit.

`cpu::matmul_naive` is the reference: a plain triple loop with a single `double` accumulator per element. The fast kernel agrees with it up to float rounding; the tests require 1e-4 relative.

## Mapping to code

- Public interface and precondition checks: `src/ops/matmul.hpp`, `src/ops/matmul.cpp`. The dispatch routes by the tensors' device to the CPU or the CUDA implementation; on CUDA it passes the selected `GemmBackend` along. `ops::matmul` creates a zeroed `out` on the operands' device and calls the accumulating kernel, on both devices.
- CPU fast kernel and references: `src/ops/cpu/matmul.cpp` (`matmul_acc`, `matmul_acc_reference`, `matmul_naive`).
- CUDA kernels: `src/ops/cuda/matmul.cu` (`matmul_acc`, `matmul_naive`, `matmul_tiled` and the cuBLAS call), declared in `src/ops/cuda/matmul.hpp`. The cuBLAS handle and `TITAN_CUBLAS_CHECK`: `src/ops/cuda/cublas_handle.cuh` and `src/ops/cuda/cublas_handle.cu`.
- Tests: `tests/ops/matmul_test.cpp` covers values, all four transpose cases, batching, the naive reference on awkward shapes, thread-count determinism, accumulation, error messages and the gradients above through `tests/support/gradcheck.hpp`. `tests/ops/cuda/matmul_parity_test.cpp` compares each GEMM backend with the CPU and cuBLAS with the tiled kernel, and checks the backend switch.

## CUDA implementation

Three GEMMs compute the same `out += op(a)·op(b)`. `ops::set_gemm_backend(GemmBackend)` picks one for every later CUDA matmul and `ops::gemm_backend()` reads it back. The default is `cublas` (ADR 0007); CPU matmuls ignore the setting. The value is process-wide state in `ops/matmul.cpp`, which is fine because training is single-threaded. `ops/cuda` may include only `core`, so it declares its own `cuda::GemmBackend` with the same three values, and `ops/matmul.cpp` maps one to the other. See [CUDA execution](cuda-execution.md) for grids, blocks and shared memory.

### cuBLAS and the column-major mapping

cuBLAS follows the Fortran convention: matrices are column-major, and element `(r, c)` of a matrix with leading dimension `ld` is at `r + c·ld`. Our tensors are row-major. The key fact: a row-major `(R, C)` matrix `X`, read column-major with `ld = C`, is the `C × R` matrix `Xᵀ`, because `X[r][c]` at `r·C + c` is element `(c, r)` of the column-major view. No data moves.

So for row-major `C (M, N) = op(A)·op(B)`:

- cuBLAS sees our `C` as `Cᵀ`, an `N × M` matrix with `ldc = N`.
- `Cᵀ = (op(A)·op(B))ᵀ = op(B)ᵀ·op(A)ᵀ`, so cuBLAS computes a product of size `m = N`, `n = M`, `k = K`, with `B` as its first operand and `A` as its second.
- First operand. If `transpose_b` is false, stored `b` is `(K, N)`, which column-major is the `N × K` matrix `op(B)ᵀ`, exactly what is needed: `CUBLAS_OP_N`, `ldb = N`. If `transpose_b` is true, stored `b` is `(N, K)`, which column-major is the `K × N` matrix `op(B)`, so cuBLAS must transpose it: `CUBLAS_OP_T`, `ldb = K`.
- Second operand. If `transpose_a` is false, stored `a` is `(M, K)`, column-major the `K × M` matrix `op(A)ᵀ`: `CUBLAS_OP_N`, `lda = K`. If `transpose_a` is true, stored `a` is `(K, M)`, column-major `op(A)`: `CUBLAS_OP_T`, `lda = M`.

| `transpose_a` | `transpose_b` | cuBLAS `transa` (on `b`) | `ldb` | cuBLAS `transb` (on `a`) | `lda` |
| --- | --- | --- | --- | --- | --- |
| false | false | `CUBLAS_OP_N` | `N` | `CUBLAS_OP_N` | `K` |
| true | false | `CUBLAS_OP_N` | `N` | `CUBLAS_OP_T` | `M` |
| false | true | `CUBLAS_OP_T` | `K` | `CUBLAS_OP_N` | `K` |
| true | true | `CUBLAS_OP_T` | `K` | `CUBLAS_OP_T` | `M` |

In the transpose columns `CUBLAS_OP_N` means "no transpose" and `CUBLAS_OP_T` means "transpose"; in the `ldb` and `lda` columns `N`, `M` and `K` are the dimensions of the product.

In every case the cuBLAS transpose of an operand equals our flag for it, and its leading dimension is the last dimension of the stored tensor.

The call is `cublasSgemmStridedBatched(handle, tb, ta, N, M, K, 1, b, ldb, K·N, a, lda, M·K, 1, out, N, M·N, B)`, with `B` the product of the batch dimensions. The strides are the element counts of one matrix, since the batch is contiguous. `beta = 1` makes it accumulate, which is what `matmul_acc` means; `ops::matmul` passes a zeroed `out`. cuBLAS takes `int` sizes, so a dimension or batch count above 2³¹ − 1 throws.

The handle is created on first use and never destroyed, like the caching allocator. It runs with `CUBLAS_PEDANTIC_MATH`, so FP32 GEMMs never use TF32 tensor cores (which round the inputs to 10 mantissa bits) and stay comparable with the CPU. Every cuBLAS call goes through `TITAN_CUBLAS_CHECK`, which throws `titan::Error` with the call, the status name and number, and `file:line`. cuBLAS chooses its own order of summation over `k`, fixed for a given GPU, shape and library version.

### Naive kernel

One thread per output element, in a grid-stride loop over the `B·M·N` outputs with `j` the fastest index, so neighbouring threads write neighbouring outputs and, when `b` is not transposed, read neighbouring elements of `op(b)`. Each thread reads `op(a)` and `op(b)` through the index table at the top of this note, adds the `K` products in increasing `k` into one `float` accumulator, then adds that sum to `out`.

### Tiled kernel

Output tiles of 16 × 16, one per 16 × 16 thread block; thread `(ty, tx)` owns output `(row0 + ty, col0 + tx)`. For each step of 16 along `k`:

1. Every thread loads one element of `op(a)[row0 … row0 + 15][k0 … k0 + 15]` and one of `op(b)[k0 … k0 + 15][col0 … col0 + 15]` into two 16 × 16 shared-memory tiles.
2. The block synchronizes.
3. Each thread adds its 16 products from shared memory.
4. The block synchronizes again before the tiles are overwritten.

Every load is guarded: outside the matrix it stores 0, so edges need no special case. A padded `k` contributes `0 · 0`, so the accumulation is the naive kernel's: one `float`, in increasing `k`. Transposes use the same index arithmetic as the naive kernel, at the cost of uncoalesced loads for a transposed operand.

Batching folds into `gridDim.x`. The `B · ⌈M/16⌉ · ⌈N/16⌉` (batch, tile row, tile column) triples are numbered by one flat index, the grid has `min(that count, 65535)` blocks, and each block loops over tile indices with the grid size as its stride. This keeps every grid dimension within its limit for any batch or row count (`gridDim.y` and `gridDim.z` would cap at 65535), and is the tile-level form of the grid-stride loop. The loop bound depends only on `blockIdx`, so all threads of a block reach the same `__syncthreads()`.

### Arithmetic intensity and the roofline

A GEMM performs `2·M·N·K` floating-point operations. Its arithmetic intensity is the number of operations per byte moved from global memory. A kernel whose intensity is below the GPU's ridge point (peak FLOP/s divided by peak bandwidth, a few tens of FLOP per byte on an sm_86 card) is bandwidth-bound; above it, compute-bound.

- Naive: each output reads `2K` floats (`8K` bytes) for `2K` operations, an intensity of `0.25` FLOP/byte before caching. Neighbouring threads share rows of `op(a)` and columns of `op(b)` through the L1 and L2 caches, which helps, but the kernel stays far below the ridge: bandwidth-bound.
- Tiled, with tile size `T = 16`: per step a block loads `2·T²` floats (`8T²` bytes) and performs `2·T³` operations, an intensity of `T/4 = 4` FLOP/byte, sixteen times the naive one. Still below the ridge: each element in shared memory is reused only 16 times.
- cuBLAS: register tiling on top of shared-memory tiling (each thread keeps a block of outputs in registers, each thread block covers a tile 64 or 128 wide), with vectorized and double-buffered loads, takes the intensity above the ridge for large products. The bound for any GEMM is reading each input and writing the output once: `2MNK / (4(MK + KN + MN))` FLOP/byte, about 85 for `512³`.

The measured throughput of each kernel next to the GPU's peak compute and bandwidth, from `titan bench matmul`, is in "Measured throughput" below.

Differences from the CPU:

- The CPU sums blocks of 64 `k` in `float` and the blocks in `double`; the naive and tiled kernels sum all of `k` in one `float`, and cuBLAS in its own order. The results agree up to float rounding, which grows with `K`, so the parity tests use `atol = 1e-4, rtol = 1e-4` instead of the default `1e-5` absolute.
- The naive and tiled kernels add in the same order and are deterministic. cuBLAS is deterministic for a given GPU and library version, not across them.

### Measured throughput

`titan bench matmul` times `out += a·b` for square `n × n` matrices with each GEMM backend. Per size and backend it makes one untimed warm-up call, then times `--repeat` calls (default 10) with `std::chrono::steady_clock` started and stopped on `cuda_synchronize()`, because kernels run asynchronously on the default stream. It reports milliseconds per call, `2n³ / time` in GFLOP/s, and that as a percentage of the GPU's peak.

The two peaks come from `cudaDeviceProp` through `core::cuda_device_info()`:

- Peak FP32 throughput `= SMs × 128 × 2 × clock`. Each sm_86 SM has 128 FP32 lanes (four partitions of 32), and each lane completes one fused multiply-add, 2 FLOP, per clock. The 128 is an assumption for sm_86 only; other architectures have other lane counts. The clock is the device's reported peak (boost) clock, which a laptop GPU sustains only within its power limit, so the peak is an upper bound.
- Peak memory bandwidth `= 2 × memory clock × bus width / 8`. GDDR memory transfers on both clock edges, and each transfer moves `bus width / 8` bytes.

The ridge point is their ratio: an operation with fewer FLOP per byte than that is bandwidth-bound at best.

Measured on an NVIDIA GeForce RTX 3070 Laptop GPU (compute 8.6, 40 SMs at 1560 MHz, 256-bit bus), release build, default sizes and `--repeat 10`. Peak FP32 15,974 GFLOP/s, peak bandwidth 448 GB/s, ridge point 35.7 FLOP/byte.

| n | cuBLAS ms | cuBLAS GFLOP/s | cuBLAS % | naive ms | naive GFLOP/s | naive % | tiled ms | tiled GFLOP/s | tiled % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.013 | 2,493 | 15.6 | 0.049 | 686 | 4.3 | 0.036 | 944 | 5.9 |
| 512 | 0.050 | 5,374 | 33.6 | 0.330 | 814 | 5.1 | 0.220 | 1,219 | 7.6 |
| 1024 | 0.247 | 8,686 | 54.4 | 2.881 | 746 | 4.7 | 1.680 | 1,278 | 8.0 |
| 2048 | 1.739 | 9,879 | 61.8 | 33.457 | 514 | 3.2 | 13.936 | 1,233 | 7.7 |
| 4096 | 16.283 | 8,441 | 52.8 | 596.492 | 230 | 1.4 | 107.747 | 1,276 | 8.0 |

Reading it against the roofline:

- **Naive: bound by memory traffic.** At its nominal 0.25 FLOP/byte, DRAM alone would cap it at `0.25 × 448 ≈ 112` GFLOP/s. It does better up to `n = 1024` (about 690 to 810 GFLOP/s) because neighbouring threads read the same row of `a`, so L1 and L2 serve most loads and the effective intensity is higher than the nominal one. As `n` grows, a column strip of `b` no longer fits in L2 and the kernel falls towards the DRAM bound: 230 GFLOP/s at `n = 4096`.
- **Tiled: bound by memory traffic, not compute.** Its 4 FLOP/byte from global memory gives a DRAM roof of `4 × 448 ≈ 1,792` GFLOP/s, and it sustains about 1,250, 70% of that roof but only 8% of compute peak. It is flat from `n = 512` on, because every 16 × 16 tile does the same work per byte at any size. Each shared-memory value feeds one multiply-add per load, so shared-memory bandwidth and the two `__syncthreads()` per step also limit it before the FP32 lanes do.
- **cuBLAS: bound by compute at large sizes.** A large GEMM's ideal intensity (about 341 FLOP/byte at `n = 2048`) is far above the ridge, and register tiling lets cuBLAS approach it: 62% of the nominal peak at `n = 2048`. Small sizes are bound by launch overhead and too few thread blocks to fill 40 SMs (`n = 256` has 16% of peak). The dip at `n = 4096` is probably clock throttling under the sustained load, which the nominal peak does not model; the benchmark does not read the clock, so this is not measured.

cuBLAS is about 7 to 8 times faster than the tiled kernel from `n = 1024` on (6.8× at 1024, 8.0× at 2048, 6.6× at 4096), which is why it is the default (ADR 0007). The training run's GEMMs are small (at most 2048 × 344 × 128 for the M1 config), where the gap is smaller but launch overhead also matters.
