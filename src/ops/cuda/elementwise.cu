#include "ops/cuda/elementwise.hpp"

#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// sqrt(2 / pi) and the cubic coefficient of the GELU tanh approximation; the
// same float constants as ops/cpu.
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCubic = 0.044715f;

// Every kernel below is element-parallel: thread t of the grid handles
// elements t, t + stride, t + 2 * stride, ... (a grid-stride loop), so a grid
// of grid_for(n, kBlock) blocks covers any n.

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void add_kernel(const float* a, const float* b, float* out,
                           std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        out[i] = a[i] + b[i];
    }
}

__global__ void add_inplace_kernel(float* a, const float* b, std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        a[i] += b[i];
    }
}

// y is n elements in rows of `cols`; element i is in column i mod cols.
__global__ void add_rowwise_kernel(float* y, const float* b, std::int64_t n,
                                   std::int64_t cols) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        y[i] += b[i % cols];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out,
                           std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        out[i] = a[i] * b[i];
    }
}

// out may equal x: each element is read before it is written, by one thread.
__global__ void scale_kernel(const float* x, float s, float* out,
                             std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        out[i] = s * x[i];
    }
}

// y = x * sigmoid(x)
__global__ void silu_kernel(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        out[i] = x[i] * sigmoid(x[i]);
    }
}

// dx = dy * s * (1 + x * (1 - s)) with s = sigmoid(x).
__global__ void silu_backward_kernel(const float* x, const float* dy,
                                     float* out, std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        const float s = sigmoid(x[i]);
        out[i] = dy[i] * s * (1.0f + x[i] * (1.0f - s));
    }
}

// y = 0.5 x (1 + tanh(u)), u = sqrt(2/pi) (x + 0.044715 x^3)
__global__ void gelu_kernel(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        const float v = x[i];
        const float u = kSqrt2OverPi * (v + kGeluCubic * v * v * v);
        out[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}

// dx = dy * (0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2)),
// with t = tanh(u).
__global__ void gelu_backward_kernel(const float* x, const float* dy,
                                     float* out, std::int64_t n) {
    for (std::int64_t i = global_thread_index(); i < n; i += grid_stride()) {
        const float v = x[i];
        const float u = kSqrt2OverPi * (v + kGeluCubic * v * v * v);
        const float t = tanhf(u);
        const float du = kSqrt2OverPi * (1.0f + 3.0f * kGeluCubic * v * v);
        out[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du);
    }
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::int64_t n = a.numel();
    add_kernel<<<grid_for(n, kBlock), kBlock>>>(a.data(), b.data(), out.data(),
                                                n);
    check_launch();
    return out;
}

void add_(Tensor& a, const Tensor& b) {
    const std::int64_t n = a.numel();
    add_inplace_kernel<<<grid_for(n, kBlock), kBlock>>>(a.data(), b.data(), n);
    check_launch();
}

void add_rowwise_(Tensor& y, const Tensor& b) {
    const std::int64_t n = y.numel();
    add_rowwise_kernel<<<grid_for(n, kBlock), kBlock>>>(y.data(), b.data(), n,
                                                        b.numel());
    check_launch();
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::empty(a.shape(), a.device());
    const std::int64_t n = a.numel();
    mul_kernel<<<grid_for(n, kBlock), kBlock>>>(a.data(), b.data(), out.data(),
                                                n);
    check_launch();
    return out;
}

Tensor scale(const Tensor& x, float s) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const std::int64_t n = x.numel();
    scale_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), s, out.data(), n);
    check_launch();
    return out;
}

void scale_(Tensor& x, float s) {
    const std::int64_t n = x.numel();
    scale_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), s, x.data(), n);
    check_launch();
}

Tensor silu(const Tensor& x) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const std::int64_t n = x.numel();
    silu_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), out.data(), n);
    check_launch();
    return out;
}

Tensor silu_backward(const Tensor& x, const Tensor& dy) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const std::int64_t n = x.numel();
    silu_backward_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), dy.data(),
                                                          out.data(), n);
    check_launch();
    return out;
}

Tensor gelu(const Tensor& x) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const std::int64_t n = x.numel();
    gelu_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), out.data(), n);
    check_launch();
    return out;
}

Tensor gelu_backward(const Tensor& x, const Tensor& dy) {
    Tensor out = Tensor::empty(x.shape(), x.device());
    const std::int64_t n = x.numel();
    gelu_backward_kernel<<<grid_for(n, kBlock), kBlock>>>(x.data(), dy.data(),
                                                          out.data(), n);
    check_launch();
    return out;
}

}  // namespace titan::ops::cuda
