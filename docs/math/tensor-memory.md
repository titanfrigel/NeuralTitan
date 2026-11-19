# Tensor memory

## What it computes

Nothing numeric. This note explains how a tensor's elements are laid out in memory, which operations copy them and which only reinterpret them, and how device memory is allocated. Every later op relies on these rules.

## Row-major layout

A tensor with shape `(d₀, d₁, …, dₙ₋₁)` stores its elements in one flat array. Row-major means the last index varies fastest: element `(i₀, i₁, …, iₙ₋₁)` sits at

```text
address = offset + Σₖ iₖ · strideₖ,    strideₖ = Π_{j>k} dⱼ
```

where `offset` is the position of the first element of the tensor in its buffer, and the empty product (for the last dimension) is 1. For shape `(2, 3, 4)` the strides are `(12, 4, 1)`, so element `(1, 2, 3)` is at `offset + 12 + 8 + 3 = offset + 23`.

NeuralTitan tensors are always contiguous row-major, so they do not store strides. The strides are fully determined by the shape through the formula above, and a tensor is just `(storage, offset, shape)`. This is a deliberate simplification: every kernel can index with the formula and never has to handle arbitrary strides.

## Views that share storage

Some operations return a new `Tensor` that points at the same buffer instead of copying it:

- **`reshape`** keeps `offset` and changes only the shape. Because the elements are contiguous, any shape with the same element count describes the same flat array, so `(2, 3, 4)` can become `(6, 4)` or `(24)` at no cost.
- **`slice_rows(begin, end)`** takes rows `begin` to `end - 1` of dimension 0. Each row of dimension 0 occupies `stride₀ = d₁ · … · dₙ₋₁` consecutive elements, so the selected rows are one contiguous block starting at `offset + begin · stride₀`. The view has shape `(end − begin, d₁, …)` and a larger `offset`.

A slice along an inner dimension cannot work this way. Taking columns `1..2` of a `(2, 4)` tensor selects elements `1, 2` of row 0 and `5, 6` of row 1, with a gap in between. The result is not a contiguous block, and describing it needs a stride that differs from the shape-derived one. Rather than carry strides everywhere, such a slice would be an explicit copy.

A view shares its buffer, so writing through one view is visible in the others. `clone()` is the deep copy for when that is not wanted. Copying a `Tensor` object only copies the handle: the buffer is reference counted and freed when the last view goes away.

## Host and device memory

A tensor lives on one device: the CPU (host RAM) or the GPU (device memory). A GPU kernel can only read device memory, and the CPU can only read host memory, so data has to be copied between them. `to(Device)` does that with `cudaMemcpy` and returns the tensor itself when it is already on the requested device.

The copy crosses the PCIe bus, which is far slower than either memory. A PCIe 4.0 x16 link moves tens of GB/s, while GPU memory moves hundreds of GB/s. The rule that follows: move data to the GPU once, keep it there for the whole computation, and copy only small results (a loss value, a sampled token) back.

## The caching allocator

Training allocates and frees tensors of the same few sizes on every step. `cudaMalloc` and `cudaFree` are slow, and `cudaFree` also waits for all work on the device to finish. Calling them per tensor would dominate the step time.

`CudaCachingAllocator` avoids this by keeping freed blocks instead of returning them to the driver:

1. A request is rounded up to a multiple of 512 bytes, so near-equal sizes share a bucket and every block is suitably aligned.
2. If a freed block of exactly that rounded size is cached, it is reused.
3. Otherwise `cudaMalloc` provides a new block. If it fails with out-of-memory, the allocator releases the whole cache and retries once before throwing.

Reuse is by exact size only, with no splitting or merging of blocks. That wastes some memory when sizes vary but keeps the bookkeeping short enough to read in a minute.

## Mapping to code

| Concept | Where |
| --- | --- |
| `(storage, offset, shape)` and the element address | `Tensor` in `src/core/tensor.hpp`: `storage_`, `offset_`, `shape_`; `data()` returns the storage pointer plus `offset_`. |
| Views | `Tensor::reshape` and `Tensor::slice_rows` in `src/core/tensor.cpp` share `storage_`; `shares_storage_with` tests it. |
| Deep copy and device transfer | `Tensor::clone` and `Tensor::to` in `src/core/tensor.cpp`, using `cudaMemcpy`. |
| The buffer | `Storage` in `src/core/storage.hpp`: raw bytes on one device, held by `std::shared_ptr`. CUDA storage gets its memory from the allocator. |
| Caching | `CudaCachingAllocator` in `src/core/cuda_allocator.hpp` and `.cpp`: 512-byte rounding, free lists keyed by size, release and retry on out-of-memory. |
