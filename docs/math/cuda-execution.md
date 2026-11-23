# CUDA execution

## What it computes

Nothing numeric, so the forward, backward and shapes sections of the [template](README.md) do not apply. This note explains how the CUDA backend in `src/ops/cuda` runs work on the GPU: how a kernel launch maps threads to elements, what warps and shared memory are, why every kernel runs on the default stream, and how device memory is reused. Every CUDA section of the other notes builds on these rules.

## Grids, blocks and threads

A kernel is a function that many GPU threads run at once, each on its own data. A launch `kernel<<<blocks, threads>>>(args)` starts a **grid** of `blocks` **blocks**, each of `threads` threads. Inside the kernel every thread reads its position from built-in variables:

- `threadIdx.x`: the thread's index in its block, `0 … blockDim.x − 1`;
- `blockIdx.x`: the block's index in the grid, `0 … gridDim.x − 1`;
- `blockDim.x` and `gridDim.x`: the block size and the grid size.

Its global index in a one-dimensional grid is

```text
i = blockIdx.x · blockDim.x + threadIdx.x
```

computed in 64 bits (`global_thread_index()`), so offsets past 2³¹ do not overflow. The hardware limits the dimensions: a block has at most 1024 threads, `gridDim.x` is at most 2³¹ − 1, and `gridDim.y` and `gridDim.z` at most 65535. Because of the last limit, row kernels put one block per row on `gridDim.x` and never use `gridDim.y` for rows: a tensor with more than 65535 rows would otherwise fail to launch.

## The grid-stride loop

Element-parallel kernels, where every output element is computed independently, use a **grid-stride loop**:

```text
for (i = global_thread_index(); i < n; i += gridDim.x · blockDim.x)
    out[i] = f(in[i])
```

The grid has `grid_for(n, kBlock) = min(⌈n / kBlock⌉, 65535)` blocks of `kBlock = 256` threads. When `n ≤ 65535 · 256 ≈ 16.8 million`, every thread handles at most one element, exactly as with one thread per element; the threads of the last block that fall past `n` fail the loop test and do nothing. When `n` is larger, each thread walks on by the total thread count, so any `n` is covered with a grid that stays within its limits. The loop also keeps neighbouring threads on neighbouring addresses in every pass, which is what the memory system needs (below).

A block of 256 threads is a multiple of the warp size, and is small enough that several blocks fit on one streaming multiprocessor (SM) at once, so the SM can switch to another block while one waits for memory.

## Warps

The GPU does not schedule threads one by one. It runs them in **warps** of 32 consecutive threads of a block, which execute the same instruction at the same time. Two consequences shape the kernels:

- **Divergence.** When threads of one warp take different branches, the warp runs both paths one after the other, with the threads of the other path idle. Bounds checks only diverge in the last warp, so they cost nothing measurable.
- **Coalescing.** When the 32 threads of a warp read 32 consecutive floats, the hardware serves them with a few wide memory transactions. Scattered addresses need one transaction each. Thread `i` handling element `i` gives the consecutive pattern.

Elementwise kernels do a few arithmetic operations per 8 to 12 bytes moved, so their speed is set by memory bandwidth, and coalescing is what matters.

## Shared memory

Each block has a small on-chip **shared memory** (up to 100 KB per SM on sm_86), visible to all threads of the block and far faster than device memory. Threads write to it, wait at a barrier `__syncthreads()` until every thread of the block has arrived, and then read what the others wrote. Reductions and the tiled matrix product use it; the elementwise kernels do not, because no thread needs another thread's data.

## Deterministic reductions

Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)` can differ in the last bit. An `atomicAdd` of floats adds in whatever order the threads happen to arrive, so two runs on the same GPU can differ. The kernels of this project never use a floating-point `atomicAdd`. Every reduction combines partial sums in a fixed tree order, so a repeated run on the same GPU gives bit-identical results, as the determinism rule of the [architecture overview](../architecture/overview.md) (section 6) requires. The order differs from the CPU's left-to-right sum, which is why parity tests compare with a tolerance instead of exactly.

## The default stream

A **stream** is a queue of GPU work that runs in order. Every kernel and every `cudaMemcpy` in this project goes to the default stream, so they execute in the order the host issued them: a kernel always sees the results of the kernels launched before it, with no explicit synchronization.

A launch returns to the host immediately, before the kernel has run. The host only waits where it must: a `cudaMemcpy` to the host (`Tensor::to_vector`, `Tensor::to(Device::CPU)`) blocks until all earlier work on the stream is done, so the values it copies are final.

A launch also returns no status. A bad configuration, such as a block larger than 1024 threads, is reported by the next call to `cudaGetLastError()`. Every launch is therefore followed by `check_launch()`, which throws a `titan::Error` naming the launching file and line. An error that happens while the kernel runs, such as an out-of-bounds access, surfaces at a later synchronizing call.

## The caching allocator and the stream

Device tensors get their memory from the caching allocator described in [Tensor memory](tensor-memory.md). A tensor can be freed on the host while a kernel that reads it is still queued, and its block can be handed to a new tensor at once. This is safe because there is one stream: any kernel that writes the reused block was launched later, so it runs after every kernel that read the old contents. A second stream would break this argument, which is one reason everything stays on the default stream.

## Mapping to code

| Concept | Where |
| --- | --- |
| Block size, grid size, launch check | `kBlock`, `kMaxBlocks`, `grid_for` and `check_launch` in `src/ops/cuda/common.cuh`. |
| Global index and grid stride | `global_thread_index()` and `grid_stride()` in `src/ops/cuda/common.cuh`. |
| Device routing | `detail::route` in `src/ops/detail.hpp` returns `Backend::cpu` or `Backend::cuda` and rejects tensors on different devices before any kernel is launched. |
| Plain C++ interface of the backend | `src/ops/cuda/*.hpp`, included by the dispatch in `src/ops/*.cpp`; only the `.cu` files include CUDA headers. |
| Error checks | `TITAN_CUDA_CHECK` and `throw_cuda_error` in `src/core/cuda_check.hpp`. |
| Parity tests | `TITAN_REQUIRE_CUDA` and `TITAN_EXPECT_PARITY` in `tests/support/parity.hpp`; one parity test file per op family in `tests/ops/cuda/`. |
