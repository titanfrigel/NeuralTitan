#include "ops/cuda/layout.hpp"

#include <cstdint>

#include "ops/cuda/common.cuh"

namespace titan::ops::cuda {

namespace {

// Every kernel is element-parallel over its output: thread t handles output
// elements t, t + stride, ... (a grid-stride loop) and reads the one input
// element (or, for the backward pass, the n_rep input elements) it maps to.

// y[a, c, b, d] = x[a, b, c, d]; x is (n_a, n_b, n_c, n_d). The output index
// o is decoded as (a, c, b, d) in row-major order.
__global__ void swap_dims_1_2_kernel(const float* x, float* y, std::int64_t n,
                                     std::int64_t n_b, std::int64_t n_c,
                                     std::int64_t n_d) {
    for (std::int64_t o = global_thread_index(); o < n; o += grid_stride()) {
        const std::int64_t d = o % n_d;
        const std::int64_t b = (o / n_d) % n_b;
        const std::int64_t c = (o / (n_d * n_b)) % n_c;
        const std::int64_t a = o / (n_d * n_b * n_c);
        y[o] = x[((a * n_b + b) * n_c + c) * n_d + d];
    }
}

// y[b, g * n_rep + r] = x[b, g]. `block` is the size of one (T, hd) head and
// heads_out = kv_heads * n_rep. The output index o is decoded into a flat
// (batch, output head) index and an offset inside the head.
__global__ void repeat_kv_kernel(const float* x, float* y, std::int64_t n,
                                 std::int64_t block, std::int64_t heads_out,
                                 std::int64_t n_rep) {
    const std::int64_t kv_heads = heads_out / n_rep;
    for (std::int64_t o = global_thread_index(); o < n; o += grid_stride()) {
        const std::int64_t within = o % block;
        const std::int64_t out_head = o / block;
        const std::int64_t b = out_head / heads_out;
        const std::int64_t h = out_head % heads_out;
        y[o] = x[(b * kv_heads + h / n_rep) * block + within];
    }
}

// dx[b, g] = sum over r of dy[b, g * n_rep + r]. The n_rep copies are added
// in increasing r, in double, then rounded: the order and precision of the
// CPU loop, so no atomics are needed.
__global__ void repeat_kv_backward_kernel(const float* dy, float* dx,
                                          std::int64_t n, std::int64_t block,
                                          std::int64_t n_rep) {
    for (std::int64_t o = global_thread_index(); o < n; o += grid_stride()) {
        const std::int64_t within = o % block;
        const std::int64_t bg = o / block;  // flat (batch, kv head)
        double acc = 0.0;
        for (std::int64_t r = 0; r < n_rep; ++r) {
            acc += dy[(bg * n_rep + r) * block + within];
        }
        dx[o] = static_cast<float>(acc);
    }
}

// cache[b, h, pos + t, d] = x[b, h, t, d]. One thread per element of x; the
// index o is decoded as (bh, t, d) with bh the flat (batch, head) index.
__global__ void kv_write_kernel(const float* x, float* cache, std::int64_t n,
                                std::int64_t t, std::int64_t t_max,
                                std::int64_t hd, std::int64_t pos) {
    for (std::int64_t o = global_thread_index(); o < n; o += grid_stride()) {
        const std::int64_t d = o % hd;
        const std::int64_t row = (o / hd) % t;
        const std::int64_t bh = o / (hd * t);
        cache[(bh * t_max + pos + row) * hd + d] = x[o];
    }
}

// y[b, h, t, d] = cache[b, h, t, d] for t < len. One thread per element of y.
__global__ void kv_read_kernel(const float* cache, float* y, std::int64_t n,
                               std::int64_t len, std::int64_t t_max,
                               std::int64_t hd) {
    for (std::int64_t o = global_thread_index(); o < n; o += grid_stride()) {
        const std::int64_t d = o % hd;
        const std::int64_t row = (o / hd) % len;
        const std::int64_t bh = o / (hd * len);
        y[o] = cache[(bh * t_max + row) * hd + d];
    }
}

}  // namespace

Tensor swap_dims_1_2(const Tensor& x) {
    Tensor y =
        Tensor::empty({x.dim(0), x.dim(2), x.dim(1), x.dim(3)}, x.device());
    const std::int64_t n = x.numel();
    swap_dims_1_2_kernel<<<grid_for(n, kBlock), kBlock>>>(
        x.data(), y.data(), n, x.dim(1), x.dim(2), x.dim(3));
    check_launch();
    return y;
}

Tensor repeat_kv(const Tensor& x, std::int64_t n_rep) {
    const std::int64_t heads_out = x.dim(1) * n_rep;
    Tensor y =
        Tensor::empty({x.dim(0), heads_out, x.dim(2), x.dim(3)}, x.device());
    const std::int64_t n = y.numel();
    repeat_kv_kernel<<<grid_for(n, kBlock), kBlock>>>(
        x.data(), y.data(), n, x.dim(2) * x.dim(3), heads_out, n_rep);
    check_launch();
    return y;
}

Tensor repeat_kv_backward(const Tensor& dy, std::int64_t n_rep) {
    Tensor dx = Tensor::empty(
        {dy.dim(0), dy.dim(1) / n_rep, dy.dim(2), dy.dim(3)}, dy.device());
    const std::int64_t n = dx.numel();
    repeat_kv_backward_kernel<<<grid_for(n, kBlock), kBlock>>>(
        dy.data(), dx.data(), n, dy.dim(2) * dy.dim(3), n_rep);
    check_launch();
    return dx;
}

void kv_write(Tensor& cache, const Tensor& x, std::int64_t pos) {
    const std::int64_t n = x.numel();
    kv_write_kernel<<<grid_for(n, kBlock), kBlock>>>(
        x.data(), cache.data(), n, x.dim(2), cache.dim(2), x.dim(3), pos);
    check_launch();
}

Tensor kv_read(const Tensor& cache, std::int64_t len) {
    Tensor y = Tensor::empty({cache.dim(0), cache.dim(1), len, cache.dim(3)},
                             cache.device());
    const std::int64_t n = y.numel();
    kv_read_kernel<<<grid_for(n, kBlock), kBlock>>>(
        cache.data(), y.data(), n, len, cache.dim(2), cache.dim(3));
    check_launch();
    return y;
}

}  // namespace titan::ops::cuda
