#pragma once

// Launch helpers shared by every kernel of ops/cuda. CUDA-only: included by
// .cu files, never by the plain C++ headers of this directory.

#include <cuda_runtime.h>

#include <cstdint>
#include <source_location>

#include "core/cuda_check.hpp"

namespace titan::ops::cuda {

// Threads per block of the element-parallel kernels: a multiple of the
// 32-thread warp, small enough that several blocks fit on one SM at once.
inline constexpr int kBlock = 256;

// Upper bound on the blocks of an element-parallel launch. It is far below
// gridDim.x's limit (2^31 - 1) and still fills every SM many times over; the
// grid-stride loop covers any element count beyond kMaxBlocks * block.
inline constexpr std::int64_t kMaxBlocks = 65535;

// Blocks for `n` elements at `block` threads per block: ceil(n / block),
// clamped to [1, kMaxBlocks].
inline unsigned int grid_for(std::int64_t n, int block) {
    const std::int64_t blocks = (n + block - 1) / block;
    if (blocks < 1) return 1;
    if (blocks > kMaxBlocks) return static_cast<unsigned int>(kMaxBlocks);
    return static_cast<unsigned int>(blocks);
}

// Called right after every kernel launch. A launch returns no status; a bad
// configuration is reported by the next cudaGetLastError(). The default
// argument records the caller's file and line for the error message.
inline void check_launch(
    std::source_location where = std::source_location::current()) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw_cuda_error(where.file_name(), static_cast<int>(where.line()),
                         "cudaGetLastError()", status);
    }
}

// Global index of the calling thread in a 1-D grid of 1-D blocks, in 64 bits
// so that offsets past 2^31 do not overflow.
__device__ inline std::int64_t global_thread_index() {
    return static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// Total threads in the grid: the step of a grid-stride loop.
__device__ inline std::int64_t grid_stride() {
    return static_cast<std::int64_t>(gridDim.x) * blockDim.x;
}

}  // namespace titan::ops::cuda
