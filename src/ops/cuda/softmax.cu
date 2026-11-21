#include "ops/cuda/softmax.hpp"

#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// Reduces shared[0..kBlock) with a fixed tree: at each step thread t combines
// its value with that of thread t + offset, halving offset from kBlock / 2 to
// 1. The order depends only on kBlock, so the result is the same on every run.
// Every thread of the block must call it; the result is valid in every thread
// afterwards.
template <typename Op>
__device__ float block_reduce(float* shared, Op op) {
    for (int offset = kBlock / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] =
                op(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    const float result = shared[0];
    // Keep the next use of `shared` from overwriting shared[0] early.
    __syncthreads();
    return result;
}

struct MaxOp {
    __device__ float operator()(float a, float b) const {
        return a > b ? a : b;
    }
};

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// One block per row, rows on gridDim.x (a row count above 65535 would not fit
// gridDim.y). The first `valid` entries of the row are softmaxed; the rest are
// written as exactly 0. For plain softmax valid = cols. For causal softmax row
// r is query i = r % tq and sees keys 0..i + pos_offset, so valid =
// i + pos_offset + 1 >= 1 (pos_offset >= 0), and no row is fully masked.
// Three passes, each a strided loop (thread t takes t, t + 256, ...) followed
// by a block tree: the maximum, the sum of exp(x - max), then y = exp / sum.
template <bool Causal>
__global__ void softmax_kernel(const float* x, float* y, std::int64_t cols,
                               std::int64_t tq, std::int64_t pos_offset) {
    __shared__ float shared[kBlock];
    const std::int64_t r = blockIdx.x;
    const float* in = x + r * cols;
    float* out = y + r * cols;
    const std::int64_t valid = Causal ? (r % tq) + pos_offset + 1 : cols;

    float m = __int_as_float(0xff800000);  // -infinity
    for (std::int64_t j = threadIdx.x; j < valid; j += blockDim.x) {
        if (in[j] > m) m = in[j];
    }
    shared[threadIdx.x] = m;
    __syncthreads();
    m = block_reduce(shared, MaxOp());

    float s = 0.0f;
    for (std::int64_t j = threadIdx.x; j < valid; j += blockDim.x) {
        const float e = expf(in[j] - m);
        out[j] = e;
        s += e;
    }
    shared[threadIdx.x] = s;
    __syncthreads();
    s = block_reduce(shared, SumOp());

    const float inv = 1.0f / s;
    for (std::int64_t j = threadIdx.x; j < valid; j += blockDim.x) {
        out[j] *= inv;
    }
    for (std::int64_t j = valid + threadIdx.x; j < cols; j += blockDim.x) {
        out[j] = 0.0f;
    }
}

// dx = y * (dy - sum_j dy_j y_j), one block per row. Thread t accumulates
// t, t + 256, ... of the dot product in float, then a block tree combines.
__global__ void softmax_backward_kernel(const float* y, const float* dy,
                                        float* dx, std::int64_t cols) {
    __shared__ float shared[kBlock];
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * cols;
    float dot = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        dot += dy[base + j] * y[base + j];
    }
    shared[threadIdx.x] = dot;
    __syncthreads();
    dot = block_reduce(shared, SumOp());
    for (std::int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        dx[base + j] = y[base + j] * (dy[base + j] - dot);
    }
}

}  // namespace

Tensor softmax(const Tensor& x) {
    const std::int64_t cols = x.dim(-1);
    const std::int64_t rows = x.numel() / cols;
    Tensor out = Tensor::empty(x.shape(), x.device());
    softmax_kernel<false><<<static_cast<unsigned int>(rows), kBlock>>>(
        x.data(), out.data(), cols, 1, 0);
    check_launch();
    return out;
}

Tensor causal_softmax(const Tensor& scores, std::int64_t pos_offset) {
    const std::int64_t tk = scores.dim(-1);
    const std::int64_t tq = scores.dim(-2);
    const std::int64_t rows = scores.numel() / tk;
    Tensor out = Tensor::empty(scores.shape(), scores.device());
    softmax_kernel<true><<<static_cast<unsigned int>(rows), kBlock>>>(
        scores.data(), out.data(), tk, tq, pos_offset);
    check_launch();
    return out;
}

Tensor softmax_backward(const Tensor& y, const Tensor& dy) {
    const std::int64_t cols = y.dim(-1);
    const std::int64_t rows = y.numel() / cols;
    Tensor dx = Tensor::empty(y.shape(), y.device());
    softmax_backward_kernel<<<static_cast<unsigned int>(rows), kBlock>>>(
        y.data(), dy.data(), dx.data(), cols);
    check_launch();
    return dx;
}

}  // namespace titan::ops::cuda
