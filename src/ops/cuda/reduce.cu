#include "ops/cuda/reduce.hpp"

#include <cstddef>
#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// Owns a device allocation for the scratch of a reduction (the partial sums,
// the flag) and frees it on every path, including a thrown error.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::int64_t count) {
        TITAN_CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&ptr_),
                       static_cast<std::size_t>(count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

// Sums shared[0..kBlock) in a fixed tree: at each step thread t adds the value
// of thread t + offset, halving offset from kBlock / 2 down to 1. The order
// depends only on kBlock, so the result is the same on every run. Every thread
// of the block must call it; the total is valid in every thread afterwards.
__device__ double block_sum(double* shared) {
    for (int offset = kBlock / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    return shared[0];
}

// Pass 1 of sum / sum_squares. Thread t of the grid accumulates elements
// t, t + stride, ... in double, in index order; each block then reduces its
// 256 accumulators in the fixed tree and writes one partial.
template <bool Square>
__global__ void partial_sum_kernel(const float* x, std::int64_t n,
                                   double* partials) {
    __shared__ double shared[kBlock];
    double acc = 0.0;
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        const double v = x[i];
        acc += Square ? v * v : v;
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    const double total = block_sum(shared);
    if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

// Pass 2: one block sums the `count` partials. Thread t takes partials
// t, t + 256, ... in order, then the same fixed tree combines the threads.
__global__ void final_sum_kernel(const double* partials, std::int64_t count,
                                 double* result) {
    __shared__ double shared[kBlock];
    double acc = 0.0;
    for (std::int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        acc += partials[i];
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    const double total = block_sum(shared);
    if (threadIdx.x == 0) *result = total;
}

template <bool Square>
double sum_impl(const Tensor& x) {
    const std::int64_t n = x.numel();
    const unsigned int blocks = grid_for(n, kBlock);
    DeviceBuffer<double> partials(blocks);
    DeviceBuffer<double> result(1);
    partial_sum_kernel<Square><<<blocks, kBlock>>>(x.data(), n, partials.get());
    check_launch();
    final_sum_kernel<<<1, kBlock>>>(partials.get(), blocks, result.get());
    check_launch();
    double host = 0.0;
    // The copy on the default stream waits for both kernels.
    TITAN_CUDA_CHECK(cudaMemcpy(&host, result.get(), sizeof(double),
                                cudaMemcpyDeviceToHost));
    return host;
}

// One block per row, on gridDim.x (the number of rows can exceed the 65535
// limit of gridDim.y). Thread t takes elements t, t + 256, ... of the row,
// then a fixed tree takes the maximum. Max is exact, so the order cannot
// change the value; it is fixed anyway.
__global__ void row_max_kernel(const float* x, std::int64_t cols, float* out) {
    __shared__ float shared[kBlock];
    const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
    float m = __int_as_float(0xff800000);  // -infinity
    for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        if (row[c] > m) m = row[c];
    }
    shared[threadIdx.x] = m;
    __syncthreads();
    for (int offset = kBlock / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset &&
            shared[threadIdx.x + offset] > shared[threadIdx.x]) {
            shared[threadIdx.x] = shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = shared[0];
}

// One thread per column (grid-stride over columns). It adds the column's rows
// in order into a double, as the CPU does, so the result is the same on every
// run. Neighbouring threads read neighbouring addresses of a row (coalesced).
__global__ void sum_leading_kernel(const float* x, std::int64_t rows,
                                   std::int64_t cols, float* out) {
    for (std::int64_t c = global_thread_index(); c < cols; c += grid_stride()) {
        double acc = 0.0;
        for (std::int64_t r = 0; r < rows; ++r) acc += x[r * cols + c];
        out[c] = static_cast<float>(acc);
    }
}

// Raises the flag when any element is NaN or infinite. Every writer stores the
// same value, so the outcome does not depend on thread timing; an integer
// atomic is exact and deterministic, unlike a floating-point one.
__global__ void all_finite_kernel(const float* x, std::int64_t n, int* flag) {
    int bad = 0;
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        if (!isfinite(x[i])) bad = 1;
    }
    if (bad) atomicOr(flag, 1);
}

}  // namespace

double sum(const Tensor& x) { return sum_impl<false>(x); }

double sum_squares(const Tensor& x) { return sum_impl<true>(x); }

Tensor row_max(const Tensor& x) {
    const std::int64_t cols = x.dim(-1);
    const std::int64_t rows = x.numel() / cols;
    Shape out_shape(x.shape().begin(), x.shape().end() - 1);
    if (out_shape.empty()) out_shape.push_back(1);
    Tensor out = Tensor::empty(out_shape, x.device());
    row_max_kernel<<<static_cast<unsigned int>(rows), kBlock>>>(x.data(), cols,
                                                                out.data());
    check_launch();
    return out;
}

Tensor sum_leading(const Tensor& x) {
    const std::int64_t cols = x.dim(-1);
    const std::int64_t rows = x.numel() / cols;
    Tensor out = Tensor::empty({cols}, x.device());
    sum_leading_kernel<<<grid_for(cols, kBlock), kBlock>>>(x.data(), rows, cols,
                                                           out.data());
    check_launch();
    return out;
}

bool all_finite(const Tensor& x) {
    const std::int64_t n = x.numel();
    DeviceBuffer<int> flag(1);
    TITAN_CUDA_CHECK(cudaMemset(flag.get(), 0, sizeof(int)));
    all_finite_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), n, flag.get());
    check_launch();
    int host = 0;
    TITAN_CUDA_CHECK(
        cudaMemcpy(&host, flag.get(), sizeof(int), cudaMemcpyDeviceToHost));
    return host == 0;
}

}  // namespace titan::ops::cuda
