// Compiles one real CUDA translation unit, so the check build exercises nvcc,
// sm_86 code generation and the CUDA warnings-as-errors flags.

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "core/cuda_check.hpp"
#include "core/device.hpp"
#include "core/tensor.hpp"
#include "harness/cuda.hpp"
#include "harness/test.hpp"

namespace {

// Writes out[i] = 2 * i for every i in [0, n).
//
// The launch uses a 1-D grid of 1-D blocks. Each thread handles one element,
// whose global index is its block's offset plus its position in the block:
// i = blockIdx.x * blockDim.x + threadIdx.x. The grid is rounded up to whole
// blocks, so when n is not a multiple of the block size the last block has
// threads past the end; the bounds check makes them do nothing.
__global__ void double_index_kernel(float* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = static_cast<float>(i) * 2.0f;
    }
}

}  // namespace

TITAN_TEST(CudaToolchain, KernelRuns) {
    TITAN_SKIP_IF_NO_CUDA();
    // Not a multiple of the block size, so the bounds check is exercised.
    constexpr int kCount = 1000;
    constexpr int kBlockSize = 256;
    constexpr int kBlocks = (kCount + kBlockSize - 1) / kBlockSize;

    titan::Tensor out = titan::Tensor::empty({kCount}, titan::Device::CUDA);
    double_index_kernel<<<kBlocks, kBlockSize>>>(out.data(), kCount);
    TITAN_CUDA_CHECK(cudaGetLastError());

    // to_vector copies with a blocking cudaMemcpy, which also waits for the
    // kernel to finish.
    const std::vector<float> host = out.to_vector();
    TITAN_EXPECT_EQ(host.size(), static_cast<std::size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        TITAN_EXPECT_EQ(host[static_cast<std::size_t>(i)],
                        static_cast<float>(i) * 2.0f);
    }
}
